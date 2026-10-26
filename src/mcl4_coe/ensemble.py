"""Milestone 3: Composable Multi-Pass Controller & U$ -> T$ -> A$ Verification Loop.

Chains the 9 Cognitive Operators (`Decision`, `Association`, `Evaluation`, `Deduction`,
`Induction`, `Abduction`, `Abstraction`, `Metacognition`, `Inference`) across recurrent
passes over the shared `MCL4-CED` backbone, and provides:
1. `MultiPassController`: Executes an explicit `CognitiveSchedule` (e.g.,
   `abduction -> association -> decision -> deduction -> evaluation -> metacognition`)
   with in-network `Deduction` + `Metacognition` consistency vetoing.
2. `generate_uta`: Autoregressive `U$ -> T$ -> A$` micro-trace decoding that enforces
   mandatory `T$` scratchpad emission before `A$`.
3. `compute_joint_coe_loss`: Stage-2 joint multi-operator + `0x8000` prompt-masked
   `U$ -> T$ -> A$` supervised warming objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from torch import Tensor, nn

from .data import ByteTokenizer
from .operators import (
    CognitiveOperatorEnsemble,
    DecisionOutput,
    DeductionOutput,
    EvaluationOutput,
    MetacognitionOutput,
)

VALID_OPERATORS: tuple[str, ...] = (
    "decision",
    "association",
    "evaluation",
    "deduction",
    "induction",
    "abduction",
    "abstraction",
    "metacognition",
    "inference",
)

DEFAULT_SCHEDULE: tuple[str, ...] = (
    "abduction",
    "association",
    "decision",
    "deduction",
    "evaluation",
    "metacognition",
)


@dataclass(frozen=True)
class CognitiveSchedule:
    """Validated sequence of COE operators to execute across recurrent passes."""

    operators: tuple[str, ...] = DEFAULT_SCHEDULE
    recurrence_steps: int = 2
    min_correctness_confidence: float = 0.40
    max_contradiction_prob: float = 0.35

    def __post_init__(self) -> None:
        if not self.operators:
            raise ValueError("CognitiveSchedule requires at least one operator")
        for op in self.operators:
            if op not in VALID_OPERATORS:
                raise ValueError(f"Unknown cognitive operator: {op!r}")
        if isinstance(self.recurrence_steps, bool) or self.recurrence_steps < 1:
            raise ValueError("recurrence_steps must be >= 1")
        if not 0.0 <= self.min_correctness_confidence <= 1.0:
            raise ValueError("min_correctness_confidence must be in [0, 1]")
        if not 0.0 <= self.max_contradiction_prob <= 1.0:
            raise ValueError("max_contradiction_prob must be in [0, 1]")

    @classmethod
    def from_scaffold_text(cls, text: str, *, recurrence_steps: int = 2) -> "CognitiveSchedule":
        """Parse an operator schedule from an `S$` scaffold string (falling back to default)."""
        lowered = text.lower()
        found = tuple(op for op in VALID_OPERATORS if op in lowered)
        return cls(
            operators=found if found else DEFAULT_SCHEDULE,
            recurrence_steps=recurrence_steps,
        )


@dataclass
class MultiPassOutput:
    """Result of a multi-pass `COE` ensemble execution and consistency check."""

    schedule: CognitiveSchedule
    selected_indices: Tensor
    decision: DecisionOutput
    deduction: DeductionOutput | None
    evaluation: EvaluationOutput | None
    metacognition: MetacognitionOutput | None
    vetoed: Tensor
    novelty: Tensor | None = None


@dataclass
class JointCOELossOutput:
    """Breakdown of the Stage-2 joint multi-operator + U$/T$/A$ training loss."""

    total_loss: Tensor
    lm_loss: Tensor | None
    decision_loss: Tensor | None
    deduction_loss: Tensor | None
    evaluation_loss: Tensor | None
    metacognition_loss: Tensor | None


@dataclass
class CandidateTraceScore:
    """Telemetry for a single candidate `#U$# -> #T$# -> #A$#` rollout evaluated by `COE`."""

    index: int
    recurrence_steps: int
    temperature: float
    text: str
    decision_similarity: float
    decision_prob: float
    evaluation_utility: float
    deduction_entailed: float
    deduction_contradicted: float
    metacognition_correctness: float
    metacognition_sufficiency: float
    structural_score: float
    composite_score: float
    vetoed: bool
    abstraction_alignment: float = 0.0
    round_id: int = 1
    strategy: str = "standard"


@dataclass
class COEGenerationOutput:
    """Full output of `MultiPassController.generate_uta_with_coe` with `COE` in the loop."""

    text: str
    baseline_text: str
    selected_index: int
    improved_over_baseline: bool
    candidates: tuple[CandidateTraceScore, ...]
    abstraction_norm: float


@dataclass
class DreamRLGenerationOutput:
    """Output of `MultiPassController.generate_uta_with_dream_rl` (Dream-RSI Cache + 2-Round Inference RL)."""

    text: str
    baseline_text: str
    selected_index: int
    selected_round: int
    selected_strategy: str
    improved_over_baseline: bool
    improved_in_round2: bool
    round1_best_score: float
    round2_best_score: float
    final_best_score: float
    dream_cache_matched: bool
    dream_top_similarity: float
    dream_matched_domains: tuple[str, ...]
    dream_recommended_scaffold: str
    extra_prompt_tokens: int
    round1_advantages: tuple[float, ...]
    candidates: tuple[CandidateTraceScore, ...]
    abstraction_norm: float


@dataclass
class CodebaseRAGGenerationOutput:
    """Output of `MultiPassController.generate_uta_with_codebase_session` (`A.TXY` + `TurboQuant` + `ContextLMCompressor` + 2-Round RL)."""

    text: str
    baseline_text: str
    selected_index: int
    selected_round: int
    selected_strategy: str
    improved_over_baseline: bool
    improved_in_round2: bool
    round1_best_score: float
    round2_best_score: float
    final_best_score: float
    routed_address: str
    routed_symbol: str
    routed_summary: str
    routed_confidence: float
    focal_statement: str
    focal_invariant: str
    callees: tuple[str, ...]
    participating_files: tuple[str, ...]
    num_files_folded: int
    raw_source_tokens: int
    compressed_gist_tokens: int
    token_folding_ratio: float
    byte_compression_ratio: float
    extra_prompt_tokens: int
    round1_advantages: tuple[float, ...]
    candidates: tuple[CandidateTraceScore, ...]
    abstraction_norm: float


def _score_trace_integrity(user_prompt: str, trace_text: str) -> float:
    """Evaluate structural completeness, AST validity (if code), and topical alignment of a `#U$# -> #T$# -> #A$#` trace."""
    import ast
    import re

    if "#T$#" not in trace_text or "#A$#" not in trace_text:
        return 0.0
    after_t = trace_text.split("#T$#", 1)[1]
    if "#A$#" not in after_t:
        return 0.0
    t_seg, a_seg = after_t.split("#A$#", 1)
    t_clean = t_seg.strip()
    a_clean = a_seg.replace("<eos>", "").strip()
    if len(t_clean) < 16 or len(a_clean) < 16:
        return 0.10

    score = 0.55
    # 1. Clean sentence / code termination on both #T$# and #A$#
    if t_clean[-1] in ".!?:;)\"'`]}":
        score += 0.10
    if a_clean[-1] in ".!?:;)\"'`]}":
        score += 0.15
    elif a_clean[-1].isalnum():
        # Penalize mid-word / mid-sentence cutoff in #A$#
        score -= 0.15

    # 2. Penalize verbatim duplication between #T$# and #A$#
    if a_clean == t_clean or a_clean.startswith(t_clean[:40]):
        score -= 0.25

    # 3. Prompt topical alignment and off-topic artifact check
    p_lower = user_prompt.lower()
    body_lower = (t_clean + " " + a_clean).lower()
    prompt_terms = {
        w
        for w in re.findall(r"[a-zA-Z_]{4,}", p_lower)
        if w not in {"write", "explain", "what", "when", "where", "which", "that", "this", "with", "from", "into", "your", "does", "have"}
    }
    if prompt_terms:
        hits = sum(1 for w in prompt_terms if w in body_lower)
        score += 0.15 * min(1.0, hits / max(1, min(4, len(prompt_terms))))

    # Penalize unrelated web/SQL/HTML boilerplate when prompt does not mention them
    for noise_tok in ("<!doctype", "<html", "addEventListener", "SELECT ", "INSERT INTO", "http://"):
        if noise_tok.lower() in body_lower and noise_tok.lower() not in p_lower:
            score -= 0.25

    # Penalize synthetic RL placeholder templates and MCQA option-list artifacts in free-form generation
    for synth_tok in (
        "analyze science constraints",
        "analyze math constraints",
        "execute procedural rules",
        "solve for the exact value",
        "energy of the options",
        "define the options",
        "options provided:",
    ):
        if synth_tok in body_lower and synth_tok not in p_lower:
            score -= 0.40
    if len(a_clean.split()) < 4 and "def " not in a_clean:
        score -= 0.30

    # 4. Python AST check when prompt requests Python code or #A$# emits a function/class
    wants_python = "python" in p_lower or "def " in p_lower
    has_def = "def " in a_clean or "class " in a_clean
    if wants_python or has_def:
        code_block = a_clean
        if "```" in code_block:
            parts = code_block.split("```")
            if len(parts) >= 2:
                code_block = parts[1]
                if code_block.startswith("python\n"):
                    code_block = code_block[len("python\n") :]
        code_lines = [
            ln
            for ln in code_block.splitlines()
            if ln.startswith(("def ", "class ", "import ", "from ", "    ", "\t", "return ", "if ", "for ", "while "))
        ]
        if code_lines:
            candidate_src = "\n".join(code_lines)
            try:
                ast.parse(candidate_src)
                score += 0.20
            except SyntaxError:
                score -= 0.10

    return float(max(0.0, min(1.0, score)))


class MultiPassController(nn.Module):
    """Executes composable multi-pass COE schedules and `U$ -> T$ -> A$` verification."""

    def __init__(
        self,
        coe: CognitiveOperatorEnsemble,
        schedule: CognitiveSchedule | None = None,
    ) -> None:
        from .data import get_tokenizer

        super().__init__()
        self.coe = coe
        self.schedule = CognitiveSchedule() if schedule is None else schedule
        self.tokenizer = get_tokenizer(int(getattr(coe.backbone.config, "vocab_size", 260)))

    def forward(
        self,
        context_ids: Tensor,
        context_lengths: Tensor,
        option_ids: Tensor,
        option_lengths: Tensor,
        targets: Tensor | None = None,
        *,
        schedule: CognitiveSchedule | None = None,
        history_buffer: Tensor | None = None,
    ) -> MultiPassOutput:
        """Execute the scheduled COE operators with `RecurrentFeedback` and verify the winner."""
        active_sched = self.schedule if schedule is None else schedule

        # 1. Run multi-pass CED Decision (`Choice` / `Noul` / `Score` / `Abduction` / `Induction`).
        decision_out = self.coe.forward_decision(
            context_ids,
            context_lengths,
            option_ids,
            option_lengths,
            targets=targets,
            recurrence_steps=active_sched.recurrence_steps,
        )
        selected = decision_out.probabilities.argmax(dim=-1)
        batch_rows = torch.arange(context_ids.shape[0], device=context_ids.device)
        chosen_opt_ids = option_ids[batch_rows, selected]
        chosen_opt_lens = option_lengths[batch_rows, selected]

        # 2. Optional Association / Novelty N(q) check against task history buffer.
        novelty = None
        if "association" in active_sched.operators and history_buffer is not None:
            assoc_out = self.coe.forward_association_matrix(
                context_ids, context_lengths, history_buffer=history_buffer
            )
            novelty = assoc_out.novelty

        # 3. Deduction check: does the state logically entail the selected candidate?
        deduction_out = None
        if "deduction" in active_sched.operators:
            deduction_out = self.coe.forward_deduction(
                context_ids,
                context_lengths,
                chosen_opt_ids,
                chosen_opt_lens,
            )

        # 4. Evaluation check: criterion utility U(m, c) in [0, 1].
        evaluation_out = None
        if "evaluation" in active_sched.operators:
            evaluation_out = self.coe.forward_evaluation(
                context_ids,
                context_lengths,
                chosen_opt_ids,
                chosen_opt_lens,
            )

        # 5. Metacognition check: calibrated (`correctness`, `evidence_sufficiency`, `context_benefit`).
        meta_out = None
        if "metacognition" in active_sched.operators:
            meta_out = self.coe.forward_metacognition(
                context_ids,
                context_lengths,
                chosen_opt_ids,
                chosen_opt_lens,
            )

        # 6. In-Network Consistency Veto: flag predictions that fail Deduction or Metacognition.
        vetoed = torch.zeros(context_ids.shape[0], dtype=torch.bool, device=context_ids.device)
        if deduction_out is not None:
            # Index 1 of ("entailed", "contradicted", "undetermined") is "contradicted"
            contradicted_prob = deduction_out.probabilities[:, 1]
            vetoed = vetoed | (contradicted_prob > active_sched.max_contradiction_prob)
        if meta_out is not None:
            # Index 0 of ("correctness", "evidence_sufficiency", "context_benefit") is "correctness"
            correctness_prob = meta_out.probabilities[:, 0]
            vetoed = vetoed | (correctness_prob < active_sched.min_correctness_confidence)

        return MultiPassOutput(
            schedule=active_sched,
            selected_indices=selected,
            decision=decision_out,
            deduction=deduction_out,
            evaluation=evaluation_out,
            metacognition=meta_out,
            vetoed=vetoed,
            novelty=novelty,
        )

    @torch.no_grad()
    def generate_uta(
        self,
        user_prompt: str,
        *,
        max_new_tokens: int = 320,
        temperature: float = 0.0,
        recurrence_steps: int | None = None,
        latent_prior: Tensor | None = None,
        latent_prior_scale: float = 0.0,
        thought_prefix: str | None = None,
        lock_thought: bool = False,
        answer_prefix: str | None = None,
        recurrent_kv: bool | None = None,
        depth_ponder: bool | None = None,
        adaptive_recurrent_depth: bool = False,
        max_recurrence_steps: int = 3,
        recurrent_entropy_threshold: float = 1.5,
        gist_state: Any | None = None,
        whiteboard: Any | None = None,
    ) -> str:
        """Synthesize a `#U$# -> #T$# -> #A$#` trace, guaranteeing complete `#T$#` and `#A$#` segments."""
        was_training = self.coe.training
        self.coe.eval()
        try:
            device = next(self.coe.parameters()).device
            vocab_size = int(getattr(self.coe.backbone.config, "vocab_size", 260))
            is_morph = vocab_size > 1024
            is_hf_donor = bool(getattr(self.tokenizer, "is_hf_donor", False))
            period_id = int(getattr(self.tokenizer, "period_id", 32)) if is_hf_donor else ord(".")
            eff_user_prompt = user_prompt.strip()
            if is_hf_donor and whiteboard is not None:
                p_txts = getattr(whiteboard, "_rule_premise_texts", []) or []
                c_txts = getattr(whiteboard, "_rule_conc_texts", []) or []
                if p_txts and c_txts:
                    wb_facts = " ".join(
                        f"{p.strip().rstrip(':')}: {c.strip()}"
                        for p, c in list(zip(p_txts, c_txts))[-4:]
                    )
                    eff_user_prompt = f"Context: {wb_facts}\nQuestion: {user_prompt.strip()}"
            if is_morph:
                # Match encode_masked_turn exact token sequence: <bos> #U$# {prompt}\n #T$#
                # without a trailing standalone space token (since MorphCode-16K / BPE words carry their own leading space)
                tokens = self.tokenizer.encode(
                    f"#U$# {eff_user_prompt}\n", add_bos=True, add_eos=False
                ) + self.tokenizer.encode("#T$#", add_bos=False, add_eos=False)
                a_tag_tokens = self.tokenizer.encode("\n#A$#", add_bos=False, add_eos=False)
            else:
                prefix = f"#U$# {user_prompt.strip()}\n#T$# "
                tokens = self.tokenizer.encode(prefix, add_bos=True, add_eos=False)
                a_tag_tokens = self.tokenizer.encode("\n#A$# ", add_bos=False, add_eos=False)
            a_tag_bare = (
                self.tokenizer.encode("#A$#", add_bos=False, add_eos=False)
                if is_hf_donor
                else a_tag_tokens
            )
            prompt_len = len(tokens)
            max_seq = self.coe.backbone.config.max_seq_len
            has_emitted_a = False
            a_start_token_idx = -1
            if is_hf_donor:
                thought_budget = (
                    min(56, max(28, int(max_new_tokens * 0.30)))
                    if max_new_tokens >= 64
                    else max(10, max_new_tokens // 3)
                )
            elif is_morph:
                if answer_prefix is not None and answer_prefix.strip():
                    thought_budget = (
                        min(96, max(36, int(max_new_tokens * 0.38)))
                        if max_new_tokens >= 128
                        else max(12, max_new_tokens // 2)
                    )
                else:
                    thought_budget = (
                        min(220, max(96, int(max_new_tokens * 0.62)))
                        if max_new_tokens >= 128
                        else max(16, int(max_new_tokens * 0.55))
                    )
            else:
                thought_budget = (
                    min(140, max(56, int(max_new_tokens * 0.35)))
                    if max_new_tokens >= 128
                    else max(12, max_new_tokens // 2)
                )
            grace_tokens = 32 if max_new_tokens >= 64 else 0
            hard_max_steps = max_new_tokens + grace_tokens
            use_amp = device.type == "cuda"
            if is_hf_donor:
                sent_end_ids = getattr(
                    self.tokenizer,
                    "_sentence_end_ids",
                    frozenset({32, 19, 49, 203}),
                )
            else:
                sent_end_ids = getattr(
                    self.tokenizer,
                    "_sentence_end_ids",
                    frozenset({ord("."), ord("!"), ord("?"), 10}),
                )

            ap_toks: list[int] = []
            if answer_prefix is not None and answer_prefix.strip():
                ap_clean = answer_prefix.strip()
                if is_morph:
                    ap_toks = self.tokenizer.encode(
                        " " + ap_clean, add_bos=False, add_eos=False
                    )
                else:
                    ap_toks = self.tokenizer.encode(
                        ap_clean + " ", add_bos=False, add_eos=False
                    )
                ap_toks = ap_toks[: max(8, min(len(ap_toks), max_seq // 3))]
            elif is_hf_donor and "python function" in user_prompt.lower():
                m_fn = re.search(r"`([a-zA-Z_]\w*\([^`]*\))`", user_prompt)
                if m_fn:
                    ap_toks = self.tokenizer.encode(
                        f" def {m_fn.group(1)}:\n    ", add_bos=False, add_eos=False
                    )

            if thought_prefix is not None and thought_prefix.strip():
                tp_clean = thought_prefix.strip()
                if is_morph:
                    tp_toks = self.tokenizer.encode(
                        " " + tp_clean, add_bos=False, add_eos=False
                    )
                else:
                    tp_toks = self.tokenizer.encode(
                        tp_clean + ("" if lock_thought else " "),
                        add_bos=False,
                        add_eos=False,
                    )
                max_tp = max(8, min(len(tp_toks), max_seq // 3))
                tp_toks = tp_toks[:max_tp]
                tokens.extend(tp_toks)
                if lock_thought:
                    if tokens[-1] not in sent_end_ids:
                        tokens.append(period_id)
                    tokens.extend(a_tag_tokens)
                    a_start_token_idx = len(tokens)
                    if ap_toks and len(tokens) + len(ap_toks) < max_seq:
                        tokens.extend(ap_toks)
                    has_emitted_a = True
                else:
                    thought_budget = max(6 if bool(ap_toks) else 16, thought_budget - len(tp_toks))

            clause_end_ids = getattr(
                self.tokenizer,
                "_clause_end_ids",
                frozenset({32, 10, ord("."), ord("!"), ord("?"), ord(","), ord(";"), ord(":")}),
            )
            newline_ids = (
                getattr(self.tokenizer, "_newline_ids", frozenset({203}))
                if is_hf_donor
                else getattr(self.tokenizer, "_newline_ids", frozenset({10}))
            )
            hash_start_ids = getattr(self.tokenizer, "_hash_start_ids", frozenset({ord("#")}))
            xml_open_ids = getattr(self.tokenizer, "_xml_open_ids", frozenset({ord("<")}))
            str_to_id = getattr(self.tokenizer, "_str_to_id", {})
            pure_newline_ids = (
                getattr(self.tokenizer, "_newline_ids", frozenset({203}))
                if is_hf_donor
                else frozenset(
                    {10} | {str_to_id[m] for m in ("\n", "\n\n") if m in str_to_id}
                )
            )
            colon_id = int(getattr(self.tokenizer, "colon_id", 44))
            hash_id = int(getattr(self.tokenizer, "hash_id", 21))
            lt_id = int(getattr(self.tokenizer, "lt_id", 46))
            punct_sent_end_ids = sent_end_ids - newline_ids
            forbidden_marker_ids = [
                str_to_id[m]
                for m in (
                    "#U$#",
                    "#Q$#",
                    "#S$#",
                    "#H$#",
                    "\n#U$#",
                    "\n#Q$#",
                    "\n#S$#",
                    "\n#H$#",
                    "#U$# ",
                    "#Q$# ",
                    "#T$#",
                    "\n#T$#",
                    "#T$# ",
                )
                if m in str_to_id
            ]
            a_marker_ids = [
                str_to_id[m] for m in ("#A$#", "\n#A$#", "#A$# ") if m in str_to_id
            ]
            fence_ids = [
                str_to_id[m]
                for m in ("```", "```python\n", "```python", "```\n")
                if m in str_to_id
            ]
            tab_ids = [
                str_to_id[m]
                for m in ("\t", "\n\t", "\n\t\t")
                if m in str_to_id
            ]
            donor_code_start_ids: set[int] = set()
            donor_docstring_ids: set[int] = set()
            donor_conj_start_ids: set[int] = set()
            donor_return_ids: set[int] = set()
            if is_hf_donor:
                for tok_str in (" def", "def", "```", " ```", '"""', ' """', "'''", " '''"):
                    enc_ids = self.tokenizer.encode(tok_str, add_bos=False, add_eos=False)
                    if len(enc_ids) == 1:
                        donor_code_start_ids.add(int(enc_ids[0]))
                for doc_str in ('"""', ' """', "'''", " '''", '    """', '    """\n'):
                    enc_ids = self.tokenizer.encode(doc_str, add_bos=False, add_eos=False)
                    if len(enc_ids) == 1:
                        donor_docstring_ids.add(int(enc_ids[0]))
                for conj_str in (" and", " or", " but", " nor"):
                    enc_ids = self.tokenizer.encode(conj_str, add_bos=False, add_eos=False)
                    if len(enc_ids) == 1:
                        donor_conj_start_ids.add(int(enc_ids[0]))
                for ret_str in (" return", "return"):
                    enc_ids = self.tokenizer.encode(ret_str, add_bos=False, add_eos=False)
                    if len(enc_ids) == 1:
                        donor_return_ids.add(int(enc_ids[0]))

            step = 0
            while step < hard_max_steps and len(tokens) < max_seq:
                donor_colon_intro_ready = (
                    is_hf_donor
                    and step >= thought_budget
                    and len(tokens) >= 2
                    and tokens[-2] == colon_id
                    and tokens[-1] in pure_newline_ids
                )
                if not has_emitted_a and (step >= thought_budget or donor_colon_intro_ready):
                    if max_new_tokens >= 64:
                        if is_morph:
                            at_sentence = (
                                donor_colon_intro_ready
                                or tokens[-1] in punct_sent_end_ids
                                or (
                                    len(tokens) >= 2
                                    and tokens[-2] in punct_sent_end_ids
                                    and tokens[-1] in pure_newline_ids
                                )
                            )
                            at_boundary = at_sentence
                            force_transition = step >= max(
                                thought_budget + 24, max_new_tokens - 36
                            )
                        else:
                            at_sentence = (
                                tokens[-1] in sent_end_ids
                                or (
                                    len(tokens) >= 2
                                    and tokens[-2] in sent_end_ids
                                    and tokens[-1] in (32, 10)
                                )
                            )
                            at_clause_word = (
                                step >= thought_budget + 28
                                and tokens[-1] in clause_end_ids
                            )
                            force_transition = step >= min(
                                thought_budget + 44,
                                max_new_tokens - 64,
                            )
                            at_boundary = at_sentence or at_clause_word
                    else:
                        at_boundary = tokens[-1] in (punct_sent_end_ids if is_morph else clause_end_ids)
                        force_transition = step >= min(
                            max(thought_budget, max_new_tokens - 12), thought_budget + 12
                        )
                    if at_boundary or force_transition:
                        if len(tokens) + len(a_tag_tokens) + 1 < max_seq:
                            if not is_hf_donor and tokens[-1] in (32, ord(","), ord(";")) and max_new_tokens >= 64:
                                tokens.pop()
                                if tokens[-1] not in sent_end_ids:
                                    tokens.append(period_id)
                            elif is_morph and not is_hf_donor and tokens[-1] not in sent_end_ids and max_new_tokens >= 64:
                                tokens.append(period_id)
                            tokens.extend(a_tag_tokens)
                            a_start_token_idx = len(tokens)
                            if ap_toks and len(tokens) + len(ap_toks) < max_seq:
                                tokens.extend(ap_toks)
                            has_emitted_a = True
                            if len(tokens) >= max_seq:
                                break

                # Once past max_new_tokens (in grace window), stop cleanly as soon as #A$# hits a sentence boundary
                if has_emitted_a and step >= max_new_tokens:
                    if tokens[-1] in (punct_sent_end_ids if is_morph else sent_end_ids):
                        break

                inp = torch.tensor([tokens], dtype=torch.long, device=device)
                logit_bound = vocab_size if is_morph else 259
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    if whiteboard is not None:
                        out = self.coe.forward_with_whiteboard(
                            inp,
                            whiteboard=whiteboard,
                            recurrence_steps=(
                                max_recurrence_steps
                                if (adaptive_recurrent_depth and max_recurrence_steps > 1)
                                else recurrence_steps
                            ),
                            recurrent_kv=recurrent_kv,
                            depth_ponder=depth_ponder,
                        )
                    elif adaptive_recurrent_depth and max_recurrence_steps > 1:
                        out = self.coe.forward_inference(
                            inp,
                            recurrence_steps=1,
                            latent_prior=latent_prior,
                            latent_prior_scale=latent_prior_scale,
                            recurrent_kv=False,
                            depth_ponder=False,
                            gist_state=gist_state,
                        )
                        p1 = torch.softmax(out.logits[0, -1, :logit_bound].float(), dim=-1)
                        ent1 = float((-(p1 * p1.clamp_min(1e-9).log()).sum()).item())
                        if ent1 > recurrent_entropy_threshold:
                            out = self.coe.forward_inference(
                                inp,
                                recurrence_steps=max_recurrence_steps,
                                latent_prior=latent_prior,
                                latent_prior_scale=latent_prior_scale,
                                recurrent_kv=recurrent_kv,
                                depth_ponder=True if depth_ponder is None else depth_ponder,
                                gist_state=gist_state,
                            )
                    else:
                        out = self.coe.forward_inference(
                            inp,
                            recurrence_steps=recurrence_steps,
                            latent_prior=latent_prior,
                            latent_prior_scale=latent_prior_scale,
                            recurrent_kv=recurrent_kv,
                            depth_ponder=depth_ponder,
                            gist_state=gist_state,
                        )
                next_logits = out.logits[0, -1, :logit_bound].float()

                min_t_steps = (
                    min(16, max(4, thought_budget // 2))
                    if max_new_tokens >= 64
                    else min(4, thought_budget // 2)
                )
                if is_hf_donor:
                    low_s, low_e = getattr(self.tokenizer, "invalid_logit_slice_low", (1, 19))
                    high_s = int(getattr(self.tokenizer, "invalid_logit_start_high", 49152))
                    eos_logit_val = (
                        float(next_logits[self.tokenizer.eos_id].item())
                        if 0 <= self.tokenizer.eos_id < logit_bound
                        else None
                    )
                    next_logits[low_s:low_e] = -torch.inf
                    if vocab_size > high_s:
                        next_logits[high_s:] = -torch.inf
                    if (
                        eos_logit_val is not None
                        and has_emitted_a
                        and a_start_token_idx > 0
                        and (len(tokens) - a_start_token_idx) >= 4
                    ):
                        next_logits[self.tokenizer.eos_id] = eos_logit_val
                    if not has_emitted_a:
                        for cid in donor_code_start_ids:
                            if cid < logit_bound:
                                next_logits[cid] = -torch.inf
                        if step == 0:
                            for conj_id in donor_conj_start_ids:
                                if conj_id < logit_bound:
                                    next_logits[conj_id] = -torch.inf
                        if tokens[-1] in pure_newline_ids and step < min_t_steps:
                            next_logits[hash_id] = -torch.inf
                    elif a_start_token_idx > 0:
                        a_slice = tokens[a_start_token_idx:]
                        is_code_a = any(t in donor_code_start_ids for t in a_slice[:3])
                        if is_code_a:
                            for doc_id in donor_docstring_ids:
                                if doc_id < logit_bound:
                                    next_logits[doc_id] = -torch.inf
                        if is_code_a and tokens[-1] in pure_newline_ids:
                            next_logits[hash_id] = -torch.inf
                            for cid in donor_code_start_ids:
                                if cid < logit_bound:
                                    next_logits[cid] = -torch.inf
                    if len(tokens) >= 2 and tokens[-1] == hash_id:
                        # Prevent emitting `#U$#`, `#T$#`, `#Q$#` mid-trace, or `#A$#` when disallowed
                        for ch_id in getattr(self.tokenizer, "marker_char_ids", (71, 70, 67, 69, 58)):
                            next_logits[ch_id] = -torch.inf
                        if has_emitted_a or step < min_t_steps:
                            next_logits[int(getattr(self.tokenizer, "a_char_id", 51))] = -torch.inf
                else:
                    # Mask control bytes (<32 except newline=10), DEL..0xFF (127..255), pad_id (256), bos_id (257), unk_id (259)
                    next_logits[:10] = -torch.inf
                    next_logits[11:32] = -torch.inf
                    next_logits[127:256] = -torch.inf
                    next_logits[self.tokenizer.pad_id] = -torch.inf
                    next_logits[self.tokenizer.bos_id] = -torch.inf
                    if is_morph:
                        next_logits[self.tokenizer.unk_id] = -torch.inf
                        for tid in tab_ids:
                            next_logits[tid] = -torch.inf
                        for fid in forbidden_marker_ids:
                            next_logits[fid] = -torch.inf
                        if not has_emitted_a and step < 4:
                            for f_id in fence_ids:
                                next_logits[f_id] = -torch.inf
                        if has_emitted_a or step < min_t_steps:
                            for aid in a_marker_ids:
                                next_logits[aid] = -torch.inf

                min_a_steps = (
                    min(12, max(4, (max_new_tokens - thought_budget) // 3))
                    if (is_morph and max_new_tokens >= 64)
                    else 4
                )
                if not has_emitted_a or (
                    a_start_token_idx > 0 and len(tokens) - a_start_token_idx < min_a_steps
                ):
                    # Do not terminate with <eos> before emitting a non-empty #A$# answer
                    next_logits[self.tokenizer.eos_id] = -torch.inf

                # Prevent premature tag transition before minimum thought budget
                if not is_morph and not has_emitted_a and tokens[-1] in newline_ids and step < min(48, thought_budget):
                    next_logits[ord("#")] = -torch.inf

                # Permit paragraph breaks (\n\n) in multi-step reasoning traces while preventing >=3 consecutive newlines,
                # mid-line consecutive spaces, or single-letter space alternation ("t a e o").
                if is_morph:
                    if len(tokens) >= 2 and tokens[-1] in pure_newline_ids and tokens[-2] in pure_newline_ids:
                        for nl_id in pure_newline_ids:
                            if nl_id < logit_bound:
                                next_logits[nl_id] = -torch.inf
                elif len(tokens) >= 2 and tokens[-1] in newline_ids and tokens[-2] in newline_ids:
                    for nl_id in newline_ids:
                        if nl_id < logit_bound:
                            next_logits[nl_id] = -torch.inf
                elif tokens[-1] == 32:
                    indent_run = 0
                    idx_back = len(tokens) - 1
                    while idx_back >= 0 and tokens[idx_back] == 32:
                        indent_run += 1
                        idx_back -= 1
                    is_leading_indent = (idx_back >= 0 and tokens[idx_back] == 10 and indent_run < 8)
                    if not is_leading_indent:
                        next_logits[32] = -torch.inf
                    next_logits[10] = -torch.inf
                elif (
                    not is_morph
                    and len(tokens) >= 2
                    and tokens[-2] in (32, 10)
                    and tokens[-1] not in (ord("a"), ord("A"), ord("I"))
                ):
                    next_logits[32] -= 2.5
                    next_logits[10] -= 2.5

                # Antidoom span-loop guard: detect and suppress only the exact trigger token that restarts a repeated span
                # Scope to active segment (`#T$#` vs `#A$#`) so `#A$#` can freely state conclusions derived in `#T$#`.
                seg_start = a_start_token_idx if (has_emitted_a and a_start_token_idx > 0) else prompt_len
                gen_window = tokens[max(seg_start, len(tokens) - 128) :]
                if is_morph:
                    if len(gen_window) >= 7:
                        ctx6 = gen_window[-6:]
                        for i in range(len(gen_window) - 6):
                            if gen_window[i : i + 6] == ctx6:
                                trig_b = gen_window[i + 6]
                                if is_hf_donor or (32 <= trig_b < 127) or trig_b >= 260:
                                    next_logits[trig_b] -= 6.0
                    if len(gen_window) >= 4:
                        ctx3 = gen_window[-3:]
                        hits_3m: dict[int, int] = {}
                        for i in range(len(gen_window) - 3):
                            if gen_window[i : i + 3] == ctx3:
                                trig_b = gen_window[i + 3]
                                hits_3m[trig_b] = hits_3m.get(trig_b, 0) + 1
                        for trig_b, cnt in hits_3m.items():
                            if cnt >= 1 and (is_hf_donor or (32 <= trig_b < 127) or trig_b >= 260):
                                next_logits[trig_b] -= 5.0 * cnt
                    if len(gen_window) >= 3:
                        ctx2 = gen_window[-2:]
                        hits_2m: dict[int, int] = {}
                        for i in range(len(gen_window) - 2):
                            if gen_window[i : i + 2] == ctx2:
                                trig_b = gen_window[i + 2]
                                hits_2m[trig_b] = hits_2m.get(trig_b, 0) + 1
                        for trig_b, cnt in hits_2m.items():
                            if cnt >= 1 and (is_hf_donor or (32 <= trig_b < 127) or trig_b >= 260):
                                next_logits[trig_b] -= 4.0 * cnt
                    if len(gen_window) >= 2:
                        last_t = gen_window[-1]
                        if last_t >= 260 and last_t not in pure_newline_ids:
                            next_logits[last_t] = -torch.inf
                        hits_1m: dict[int, int] = {}
                        for i in range(len(gen_window) - 1):
                            if gen_window[i] == last_t:
                                trig_b = gen_window[i + 1]
                                hits_1m[trig_b] = hits_1m.get(trig_b, 0) + 1
                        for trig_b, cnt in hits_1m.items():
                            if cnt >= 1 and (is_hf_donor or (32 <= trig_b < 127) or trig_b >= 260):
                                next_logits[trig_b] -= 2.75 * cnt
                    freq_window = gen_window[-64:]
                    tok_counts: dict[int, int] = {}
                    for t_id in freq_window:
                        if t_id >= 260 and t_id not in pure_newline_ids:
                            tok_counts[t_id] = tok_counts.get(t_id, 0) + 1
                    for t_id, cnt in tok_counts.items():
                        if cnt >= 2 and t_id < logit_bound:
                            next_logits[t_id] -= 1.35 * (cnt - 1)
                else:
                    if len(gen_window) >= 6:
                        ctx5 = gen_window[-5:]
                        for i in range(len(gen_window) - 5):
                            if gen_window[i : i + 5] == ctx5:
                                trig_b = gen_window[i + 5]
                                if (32 <= trig_b < 127) or trig_b >= 260:
                                    next_logits[trig_b] -= 6.0
                    if len(gen_window) >= 4:
                        ctx3 = gen_window[-3:]
                        hits_3: dict[int, int] = {}
                        for i in range(len(gen_window) - 3):
                            if gen_window[i : i + 3] == ctx3:
                                trig_b = gen_window[i + 3]
                                hits_3[trig_b] = hits_3.get(trig_b, 0) + 1
                        for trig_b, cnt in hits_3.items():
                            if cnt >= 2 and ((32 <= trig_b < 127) or trig_b >= 260):
                                next_logits[trig_b] -= 4.5 * cnt

                # Prevent XML/HTML tag openers ('<') or markdown headers ('#') at start of #T$# or #A$#, 3x identical token repeats, or 2x punctuation
                if is_hf_donor and (
                    step == 0
                    or (has_emitted_a and len(tokens) == a_start_token_idx)
                    or (len(tokens) >= len(a_tag_tokens) and tokens[-len(a_tag_tokens) :] == a_tag_tokens)
                ):
                    for bad_start_id in (*pure_newline_ids, hash_id, lt_id):
                        if bad_start_id < logit_bound:
                            next_logits[bad_start_id] = -torch.inf
                elif not is_morph and (
                    step == 0
                    or (has_emitted_a and len(tokens) == a_start_token_idx)
                    or (len(tokens) >= len(a_tag_tokens) and tokens[-len(a_tag_tokens) :] == a_tag_tokens)
                ):
                    for xid in xml_open_ids:
                        if xid < logit_bound:
                            next_logits[xid] = -torch.inf
                    for hid in hash_start_ids:
                        if hid < logit_bound:
                            next_logits[hid] = -torch.inf
                if len(tokens) >= 3 and tokens[-1] == tokens[-2] == tokens[-3]:
                    next_logits[tokens[-1]] = -torch.inf
                elif not is_morph and len(tokens) >= 2 and tokens[-1] == tokens[-2]:
                    if tokens[-1] >= 260 or not chr(min(126, max(32, tokens[-1]))).isalnum():
                        next_logits[tokens[-1]] = -torch.inf

                if temperature <= 0.0:
                    next_id = int(next_logits.argmax(dim=-1).item())
                else:
                    scaled_logits = next_logits / temperature
                    if is_morph:
                        top_k_n = min(16 if is_hf_donor else 24, int((scaled_logits > -torch.inf).sum().item()))
                        if top_k_n > 1:
                            topk_vals, _ = torch.topk(scaled_logits, top_k_n)
                            scaled_logits = torch.where(
                                scaled_logits < topk_vals[-1],
                                torch.full_like(scaled_logits, -torch.inf),
                                scaled_logits,
                            )
                        probs = scaled_logits.softmax(dim=-1)
                        max_p = probs.max()
                        probs = torch.where(probs >= (0.06 * max_p), probs, torch.zeros_like(probs))
                        p_sum = probs.sum()
                        probs = probs / p_sum.clamp_min(1e-9)
                    else:
                        probs = scaled_logits.softmax(dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())

                # Cleanly terminate if #A$# is complete and the model attempts to open a new episode marker after newline
                if (
                    is_hf_donor
                    and has_emitted_a
                    and a_start_token_idx > 0
                    and len(tokens) - a_start_token_idx >= 6
                ):
                    a_slice = tokens[a_start_token_idx:]
                    is_code_a = any(t in donor_code_start_ids for t in a_slice[:3])
                    min_complete_a = max(44, (max_new_tokens - thought_budget) // 2)
                    if tokens[-1] in pure_newline_ids:
                        if next_id in (hash_id, lt_id):
                            break
                        if is_code_a and len(a_slice) >= 28 and any(t in donor_return_ids for t in a_slice):
                            if next_id in pure_newline_ids or not self.tokenizer.decode([next_id]).startswith(" "):
                                break
                    elif (
                        not is_code_a
                        and tokens[-1] in punct_sent_end_ids
                        and next_id in pure_newline_ids
                        and len(a_slice) >= min_complete_a
                    ):
                        break
                elif (
                    not is_morph
                    and has_emitted_a
                    and a_start_token_idx > 0
                    and len(tokens) - a_start_token_idx >= 32
                    and tokens[-1] in newline_ids
                    and (next_id in hash_start_ids or next_id in xml_open_ids)
                ):
                    break

                # Also stop cleanly once #A$# has completed its sentence(s)
                if (
                    has_emitted_a
                    and a_start_token_idx > 0
                    and tokens[-1] in punct_sent_end_ids
                    and (
                        (
                            bool(ap_toks)
                            and not is_hf_donor
                            and len(tokens) - a_start_token_idx >= (6 if is_morph else 18)
                            and (next_id in newline_ids or len(tokens) - a_start_token_idx >= (12 if is_morph else 36))
                        )
                        or (not is_morph and step >= int(max_new_tokens * 0.82) and next_id in newline_ids)
                    )
                ):
                    break

                tokens.append(next_id)
                step += 1

                if not has_emitted_a:
                    if is_morph and next_id in a_marker_ids:
                        a_start_token_idx = len(tokens)
                        if ap_toks and len(tokens) + len(ap_toks) < max_seq:
                            tokens.extend(ap_toks)
                        has_emitted_a = True
                    elif len(tokens) >= len(a_tag_tokens) and tokens[-len(a_tag_tokens) :] == a_tag_tokens:
                        a_start_token_idx = len(tokens)
                        if ap_toks and len(tokens) + len(ap_toks) < max_seq:
                            tokens.extend(ap_toks)
                        has_emitted_a = True
                    elif is_hf_donor and len(tokens) >= len(a_tag_bare) and tokens[-len(a_tag_bare) :] == a_tag_bare:
                        a_start_token_idx = len(tokens)
                        if ap_toks and len(tokens) + len(ap_toks) < max_seq:
                            tokens.extend(ap_toks)
                        has_emitted_a = True
                if next_id == self.tokenizer.eos_id:
                    break

            decoded = self.tokenizer.decode(tokens).rstrip()
            if is_hf_donor:
                if eff_user_prompt != user_prompt.strip() and "#T$#" in decoded:
                    _, after_t = decoded.split("#T$#", 1)
                    decoded = f"#U$# {user_prompt.strip()}\n#T$#{after_t}"
                decoded = re.sub(r"#T\$#[ \n]*", "#T$# ", decoded)
                decoded = re.sub(r"#A\$#[ \n]*", "#A$# ", decoded)
            elif is_morph:
                decoded = re.sub(r"#T\$#(?![ \n])", "#T$# ", decoded)
                decoded = re.sub(r"#A\$#(?![ \n])", "#A$# ", decoded)
            if "\n#A$#" not in decoded:
                decoded = decoded + "\n#A$# "
            return decoded
        finally:
            self.coe.train(was_training)

    @torch.no_grad()
    def generate_uta_with_coe(
        self,
        user_prompt: str,
        *,
        max_new_tokens: int = 384,
        candidate_specs: tuple[tuple[int, float], ...] = (
            (1, 0.0),
            (2, 0.0),
            (1, 0.30),
            (2, 0.35),
            (2, 0.50),
        ),
        seed: int = 42,
    ) -> COEGenerationOutput:
        """Synthesize and verify `#U$# -> #T$# -> #A$#` with all 9 `COE` operators in the loop.

        1. Runs `Abstraction` (`compress_abstraction`) on `#U$# {user_prompt}`.
        2. Generates candidate `#T$# -> #A$#` trajectories across `candidate_specs` (`(recurrence_steps, temperature)`),
           where Candidate 0 (`r=1, temp=0.0`) is the baseline single-pass trace without `COE`.
        3. Evaluates all candidates through `Decision` (`Choice` / `Abduction` / `Induction` matched-filter Pearson similarity),
           `Deduction` (`entailed` vs. `contradicted`), `Evaluation` (`utility` $U(m, c)$), and
           `Metacognition` (`correctness`, `evidence_sufficiency`).
        4. Combines neural `COE` scores with `_score_trace_integrity` (sentence completeness, Python AST verification,
           and topical alignment), applies the `COE` consistency veto, and returns the winning trajectory + full telemetry.
        """
        from .data import DecisionExample, collate_decisions

        was_training = self.coe.training
        self.coe.eval()
        try:
            device = next(self.coe.parameters()).device
            raw_traces: list[tuple[int, float, str]] = []
            for idx, (r_steps, temp) in enumerate(candidate_specs):
                if temp > 0.0:
                    torch.manual_seed(seed + idx * 131)
                trace = self.generate_uta(
                    user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    recurrence_steps=r_steps,
                )
                raw_traces.append((r_steps, temp, trace))

            baseline_text = raw_traces[0][2]

            # Build candidate bodies (everything from #T$# onward) for COE multi-operator evaluation
            cand_bodies: list[str] = []
            for _, _, tr in raw_traces:
                idx_t = tr.find("#T$#")
                body = tr[idx_t:].strip() if idx_t >= 0 else tr.strip()
                cand_bodies.append(body if body else "#T$# empty\n#A$# empty")

            max_seq = self.coe.backbone.config.max_seq_len
            ctx_text = (
                f"#U$# {user_prompt.strip()}\n"
                f"#Q$# [choice] Select the most accurate, complete, and logically entailed #T$# -> #A$# response."
            )
            ex = DecisionExample(
                question_id="coe_loop",
                kind="choice",
                context=ctx_text.encode("utf-8")[: max_seq - 4].decode("utf-8", errors="ignore"),
                descriptions=tuple(
                    b.encode("utf-8")[: max_seq - 4].decode("utf-8", errors="ignore")
                    for b in cand_bodies
                ),
                values=tuple(f"cand_{i}" for i in range(len(cand_bodies))),
                target=0,
            )
            batch = collate_decisions([ex], max_seq, device=device, tokenizer=self.tokenizer)
            ctx_ids = batch["context_ids"]
            ctx_lens = batch["context_lengths"]
            opt_ids = batch["option_ids"]
            opt_lens = batch["option_lengths"]

            use_amp = device.type == "cuda"
            k_cands = len(cand_bodies)
            rep_ctx_ids = ctx_ids.expand(k_cands, -1).contiguous()
            rep_ctx_lens = ctx_lens.expand(k_cands).contiguous()
            flat_opt_ids = opt_ids[0]
            flat_opt_lens = opt_lens[0]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                # 1. Abstraction state compression norm
                abs_vec = self.coe.compress_abstraction(ctx_ids, ctx_lens, level="standard")
                abs_norm = float(abs_vec.float().norm(dim=-1).mean().item())

                # 2. Decision (`Choice` / `Abduction` / `Induction`) with recurrent upper-CED passes
                dec_out = self.coe.forward_decision(
                    ctx_ids,
                    ctx_lens,
                    opt_ids,
                    opt_lens,
                    recurrence_steps=self.schedule.recurrence_steps,
                )

                # 3. Batched Deduction, Evaluation, and Metacognition across all K candidates
                ded_out = self.coe.forward_deduction(
                    rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
                )
                eval_out = self.coe.forward_evaluation(
                    rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
                )
                meta_out = self.coe.forward_metacognition(
                    rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
                )
            dec_sims = dec_out.similarities[0].detach().float().cpu().tolist()
            dec_probs = dec_out.probabilities[0].detach().float().cpu().tolist()

            ded_probs = ded_out.probabilities.detach().float().cpu().tolist()
            eval_utils = eval_out.utilities.detach().float().cpu().tolist()
            meta_probs = meta_out.probabilities.detach().float().cpu().tolist()

            scored_candidates: list[CandidateTraceScore] = []
            for i, (r_steps, temp, tr) in enumerate(raw_traces):
                d_sim = float(dec_sims[i])
                d_prob = float(dec_probs[i])
                u_eval = float(eval_utils[i])
                p_ent = float(ded_probs[i][0])
                p_con = float(ded_probs[i][1])
                m_cor = float(meta_probs[i][0])
                m_suf = float(meta_probs[i][1])
                s_int = _score_trace_integrity(user_prompt, tr)

                is_vetoed = bool(
                    p_con > self.schedule.max_contradiction_prob
                    or m_cor < self.schedule.min_correctness_confidence
                    or s_int < 0.35
                )
                neural_score = (
                    0.35 * d_prob
                    + 0.25 * u_eval
                    + 0.20 * (p_ent - 0.75 * p_con)
                    + 0.15 * (0.7 * m_cor + 0.3 * m_suf)
                )
                comp_score = neural_score + 0.15 * s_int - (0.30 if is_vetoed else 0.0)
                scored_candidates.append(
                    CandidateTraceScore(
                        index=i,
                        recurrence_steps=r_steps,
                        temperature=temp,
                        text=tr,
                        decision_similarity=round(d_sim, 4),
                        decision_prob=round(d_prob, 4),
                        evaluation_utility=round(u_eval, 4),
                        deduction_entailed=round(p_ent, 4),
                        deduction_contradicted=round(p_con, 4),
                        metacognition_correctness=round(m_cor, 4),
                        metacognition_sufficiency=round(m_suf, 4),
                        structural_score=round(s_int, 4),
                        composite_score=round(comp_score, 4),
                        vetoed=is_vetoed,
                    )
                )

            best_cand = max(scored_candidates, key=lambda c: c.composite_score)
            return COEGenerationOutput(
                text=best_cand.text,
                baseline_text=baseline_text,
                selected_index=best_cand.index,
                improved_over_baseline=(best_cand.text != baseline_text),
                candidates=tuple(scored_candidates),
                abstraction_norm=round(abs_norm, 4),
            )
        finally:
            self.coe.train(was_training)

    @torch.no_grad()
    def generate_uta_with_dream_rl(
        self,
        user_prompt: str,
        *,
        dream_cache: object | None = None,
        max_new_tokens: int = 384,
        prior_scale: float = 0.08,
        rl_step_size: float = 0.35,
        min_dream_similarity: float = 0.05,
        exclude_exact_dream_match: bool = True,
        seed: int = 42,
    ) -> DreamRLGenerationOutput:
        """Synthesize `#U$# -> #T$# -> #A$#` with User-Seeded Dream-RSI Prior + 2-Round Inference-Time RL.

        1. **Zero-Token Dream Prior Retrieval:** Queries `dream_cache` (`DreamContextCache`) using the
           32-d centered Pearson embedding of `#U$# {user_prompt}` to retrieve the interpolated
           `d_model` `Abstraction` vector $\\mathbf{z}_{\\text{abs}}^{(0)}$ and optimal `#S$#` cognitive
           operator scaffold from similar or adjacent OOD past user contexts in **0 prompt tokens**.
        2. **Round 1 Rollouts & Group-Relative Advantage ($A_i$):** Samples Round 1 trajectories
           (including the exact unsteered `r=1, temp=0.0` baseline and Dream-steered `r=2` rollouts),
           evaluates them through the 9 `COE` operators + anti-doom loop detector, and computes
           Ornith-1.5 group-normalized advantages $A_i = (R_i - \\mu_R) / (\\sigma_R + \\epsilon)$.
        3. **Transient Inference-Time RL Latent Policy Step ($\\mathbf{z}_{\\text{abs}}^{(0)} \\to \\mathbf{z}_{\\text{abs}}^{(1)}$):**
           Updates the latent `Abstraction` vector toward positive-advantage Round 1 rollouts and
           away from negative-advantage rollouts without mutating permanent base weights.
        4. **Round 2 Tree-Guided Rollouts:** Generates Round 2 trajectories conditioned on
           $\\mathbf{z}_{\\text{abs}}^{(1)}$ and the winning Round 1 `#T$#` node (full RL-steered rollout,
           tree thought-prefix continuation, and locked-`#T$#` answer refinement), then verifies the
           combined Round 1 + Round 2 pool through `COE`.
        """
        from .data import DecisionExample, collate_decisions
        from .operators import pearson_similarity
        from .ornith_rl import detect_doom_loop, group_normalized_advantages

        was_training = self.coe.training
        self.coe.eval()
        try:
            device = next(self.coe.parameters()).device
            max_seq = self.coe.backbone.config.max_seq_len
            use_amp = device.type == "cuda"

            # 1. Compute 32-d Pearson intent embedding for the live user prompt and query DreamContextCache
            u_ids = self.tokenizer.encode(
                f"#U$# {user_prompt.strip()}", add_bos=True, add_eos=True
            )[:max_seq]
            u_t = torch.tensor([u_ids], dtype=torch.long, device=device)
            u_len = torch.tensor([len(u_ids)], dtype=torch.long, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                q_emb = self.coe.embed_items(u_t, u_len)
                z_query = self.coe.compress_abstraction(u_t, u_len, level="standard").float()

            dream_matched = False
            dream_top_sim = 0.0
            dream_domains: tuple[str, ...] = ()
            dream_scaffold = "abduction -> association -> decision -> deduction -> evaluation -> metacognition"
            z_dream_interp: Tensor | None = None

            if dream_cache is not None and hasattr(dream_cache, "retrieve"):
                ret = dream_cache.retrieve(
                    q_emb,
                    query_z_abs=z_query,
                    query_text=user_prompt,
                    top_k=3,
                    min_similarity=min_dream_similarity,
                    exclude_exact_text=(user_prompt if exclude_exact_dream_match else None),
                    device=device,
                )
                dream_matched = bool(ret.has_match)
                dream_top_sim = float(ret.top_similarity)
                dream_domains = tuple(ret.matched_domains)
                dream_scaffold = str(ret.recommended_scaffold)
                z_dream_interp = ret.interpolated_z_abs

            sc_ids = self.tokenizer.encode(
                f"#U$# {user_prompt.strip()}\n#S$# {dream_scaffold}",
                add_bos=True,
                add_eos=True,
            )[:max_seq]
            sc_t = torch.tensor([sc_ids], dtype=torch.long, device=device)
            sc_len = torch.tensor([len(sc_ids)], dtype=torch.long, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                z_self = self.coe.compress_abstraction(sc_t, sc_len, level="standard").float()
                if dream_matched and z_dream_interp is not None:
                    blend_w = min(0.45, max(0.20, 0.45 * max(0.2, dream_top_sim)))
                    z_raw = (1.0 - blend_w) * z_self + blend_w * z_dream_interp.to(device).float()
                    z_prior_0 = self.coe.abstraction_norm(
                        z_raw.to(next(self.coe.abstraction_norm.parameters()).dtype)
                    ).float()
                else:
                    z_prior_0 = z_self

            def _score_pool(
                pool: list[tuple[int, float, str, int, str]],
                z_ref: Tensor,
            ) -> tuple[list[CandidateTraceScore], list[Tensor]]:
                bodies: list[str] = []
                for _, _, tr, _, _ in pool:
                    idx_t = tr.find("#T$#")
                    b = tr[idx_t:].strip() if idx_t >= 0 else tr.strip()
                    bodies.append(b if b else "#T$# empty\n#A$# empty")

                ctx_text = (
                    f"#U$# {user_prompt.strip()}\n"
                    f"#Q$# [choice] Select the most accurate, complete, and logically entailed #T$# -> #A$# response."
                )
                ex = DecisionExample(
                    question_id="dream_rl_loop",
                    kind="choice",
                    context=ctx_text.encode("utf-8")[: max_seq - 4].decode("utf-8", errors="ignore"),
                    descriptions=tuple(
                        b.encode("utf-8")[: max_seq - 4].decode("utf-8", errors="ignore")
                        for b in bodies
                    ),
                    values=tuple(f"cand_{i}" for i in range(len(bodies))),
                    target=0,
                )
                batch = collate_decisions([ex], max_seq, device=device, tokenizer=self.tokenizer)
                ctx_ids = batch["context_ids"]
                ctx_lens = batch["context_lengths"]
                opt_ids = batch["option_ids"]
                opt_lens = batch["option_lengths"]

                k_cands = len(bodies)
                rep_ctx_ids = ctx_ids.expand(k_cands, -1).contiguous()
                rep_ctx_lens = ctx_lens.expand(k_cands).contiguous()
                flat_opt_ids = opt_ids[0]
                flat_opt_lens = opt_lens[0]

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    dec_out = self.coe.forward_decision(
                        ctx_ids,
                        ctx_lens,
                        opt_ids,
                        opt_lens,
                        recurrence_steps=self.schedule.recurrence_steps,
                    )
                    ded_out = self.coe.forward_deduction(
                        rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
                    )
                    eval_out = self.coe.forward_evaluation(
                        rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
                    )
                    meta_out = self.coe.forward_metacognition(
                        rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
                    )
                    cand_abs_vecs = self.coe.compress_abstraction(
                        flat_opt_ids, flat_opt_lens, level="detailed"
                    ).float()
                    ref_proj = self.coe.state_projection(
                        z_ref.to(next(self.coe.state_projection.parameters()).dtype)
                    ).float()
                    cand_proj = self.coe.output_projection(
                        cand_abs_vecs.to(next(self.coe.output_projection.parameters()).dtype)
                    ).float()
                    abs_sims = pearson_similarity(
                        ref_proj.expand_as(cand_proj), cand_proj
                    ).detach().cpu().tolist()

                dec_sims = dec_out.similarities[0].detach().float().cpu().tolist()
                dec_probs = dec_out.probabilities[0].detach().float().cpu().tolist()
                ded_probs = ded_out.probabilities.detach().float().cpu().tolist()
                eval_utils = eval_out.utilities.detach().float().cpu().tolist()
                meta_probs = meta_out.probabilities.detach().float().cpu().tolist()

                scored: list[CandidateTraceScore] = []
                for i, (r_steps, temp, tr, r_id, strat) in enumerate(pool):
                    d_sim = float(dec_sims[i])
                    d_prob = float(dec_probs[i])
                    u_eval = float(eval_utils[i])
                    p_ent = float(ded_probs[i][0])
                    p_con = float(ded_probs[i][1])
                    m_cor = float(meta_probs[i][0])
                    m_suf = float(meta_probs[i][1])
                    a_sim = float(abs_sims[i])
                    s_int = _score_trace_integrity(user_prompt, tr)

                    cand_toks = self.tokenizer.encode(bodies[i], add_bos=False, add_eos=False)
                    has_doom = detect_doom_loop(cand_toks) is not None
                    if has_doom:
                        s_int = max(0.0, s_int - 0.30)

                    is_vetoed = bool(
                        p_con > self.schedule.max_contradiction_prob
                        or m_cor < self.schedule.min_correctness_confidence
                        or s_int < 0.35
                        or has_doom
                    )
                    neural_score = (
                        0.35 * d_prob
                        + 0.25 * u_eval
                        + 0.20 * (p_ent - 0.75 * p_con)
                        + 0.15 * (0.7 * m_cor + 0.3 * m_suf)
                    )
                    comp_score = neural_score + 0.15 * s_int - (0.30 if is_vetoed else 0.0)
                    scored.append(
                        CandidateTraceScore(
                            index=i,
                            recurrence_steps=r_steps,
                            temperature=temp,
                            text=tr,
                            decision_similarity=round(d_sim, 4),
                            decision_prob=round(d_prob, 4),
                            evaluation_utility=round(u_eval, 4),
                            deduction_entailed=round(p_ent, 4),
                            deduction_contradicted=round(p_con, 4),
                            metacognition_correctness=round(m_cor, 4),
                            metacognition_sufficiency=round(m_suf, 4),
                            structural_score=round(s_int, 4),
                            composite_score=round(comp_score, 4),
                            vetoed=is_vetoed,
                            abstraction_alignment=round(a_sim, 4),
                            round_id=r_id,
                            strategy=strat,
                        )
                    )
                return scored, [cand_abs_vecs[i : i + 1].detach() for i in range(k_cands)]

            # 2. Round 1 Rollouts: Unsteered COE candidates + Zero-Token Dream-Prior Steered Rollouts
            r1_specs: tuple[tuple[int, float, Tensor | None, float, str], ...] = (
                (1, 0.0, None, 0.0, "r1_baseline"),
                (2, 0.0, None, 0.0, "r1_unsteered_r2"),
                (1, 0.28, None, 0.0, "r1_unsteered_explore"),
                (2, 0.35, None, 0.0, "r1_unsteered_r2_explore"),
                (2, 0.0, z_prior_0, prior_scale, "r1_dream_steered"),
                (2, 0.30, z_prior_0, prior_scale, "r1_dream_explore"),
            )
            r1_pool: list[tuple[int, float, str, int, str]] = []
            for idx, (r_steps, temp, l_prior, l_scale, strat) in enumerate(r1_specs):
                if temp > 0.0:
                    torch.manual_seed(seed + idx * 131)
                tr = self.generate_uta(
                    user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    recurrence_steps=r_steps,
                    latent_prior=l_prior,
                    latent_prior_scale=l_scale,
                )
                r1_pool.append((r_steps, temp, tr, 1, strat))

            baseline_text = r1_pool[0][2]
            r1_scored, r1_abs_vecs = _score_pool(r1_pool, z_prior_0)
            r1_rewards = torch.tensor(
                [c.composite_score + 0.04 * max(0.0, c.abstraction_alignment) for c in r1_scored],
                dtype=torch.float32,
                device=device,
            )
            r1_adv = group_normalized_advantages(r1_rewards)
            r1_adv_tuple = tuple(round(float(x), 4) for x in r1_adv.detach().cpu().tolist())

            # 3. Transient Inference-Time RL Latent Policy Update: z_prior_0 -> z_prior_1
            pos_mask = r1_adv > 0.0
            neg_mask = r1_adv < 0.0
            z_delta = torch.zeros_like(z_prior_0)
            if bool(pos_mask.any()):
                pos_w = torch.softmax(r1_adv[pos_mask] / 0.5, dim=0)
                pos_stack = torch.cat(
                    [r1_abs_vecs[i] for i in range(len(r1_pool)) if bool(pos_mask[i])], dim=0
                )
                z_pos = (pos_w.unsqueeze(-1) * (pos_stack - z_prior_0)).sum(dim=0, keepdim=True)
                z_delta = z_delta + rl_step_size * z_pos
            if bool(neg_mask.any()):
                neg_w = torch.softmax((-r1_adv[neg_mask]) / 0.5, dim=0)
                neg_stack = torch.cat(
                    [r1_abs_vecs[i] for i in range(len(r1_pool)) if bool(neg_mask[i])], dim=0
                )
                z_neg = (neg_w.unsqueeze(-1) * (neg_stack - z_prior_0)).sum(dim=0, keepdim=True)
                z_delta = z_delta - (0.4 * rl_step_size) * z_neg

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                z_prior_1 = self.coe.abstraction_norm(
                    (z_prior_0 + z_delta).to(next(self.coe.abstraction_norm.parameters()).dtype)
                ).float()
                abs_norm = float(z_prior_1.norm(dim=-1).mean().item())

            # Extract winning Round 1 thought trace for Tree-Guided Round 2 continuations
            best_r1_idx = int(r1_rewards.argmax().item())
            best_r1_trace = r1_pool[best_r1_idx][2]
            best_t_full = ""
            first_t_sent = ""
            if "#T$#" in best_r1_trace and "#A$#" in best_r1_trace:
                t_raw = best_r1_trace.split("#T$#", 1)[1].split("#A$#", 1)[0].strip()
                best_t_full = t_raw
                sent_parts = re.split(r"(?<=[.!?])\s+", t_raw)
                if sent_parts and len(sent_parts[0].strip()) >= 12:
                    first_t_sent = sent_parts[0].strip()

            # 4. Round 2 Tree-Guided Rollouts conditioned on RL-updated z_prior_1
            r2_pool: list[tuple[int, float, str, int, str]] = []

            # R2-0: Full trajectory rollout steered by RL-updated z_prior_1
            torch.manual_seed(seed + 503)
            tr_r2_full = self.generate_uta(
                user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.25,
                recurrence_steps=2,
                latent_prior=z_prior_1,
                latent_prior_scale=prior_scale * 1.25,
            )
            r2_pool.append((2, 0.25, tr_r2_full, 2, "r2_rl_steered"))

            # R2-1: Tree thought-prefix continuation from the winning Round 1 premise
            torch.manual_seed(seed + 617)
            tr_r2_tree = self.generate_uta(
                user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.20,
                recurrence_steps=2,
                latent_prior=z_prior_1,
                latent_prior_scale=prior_scale,
                thought_prefix=(first_t_sent if first_t_sent else None),
                lock_thought=False,
            )
            r2_pool.append((2, 0.20, tr_r2_tree, 2, "r2_tree_thought_cont"))

            # R2-2: Answer refinement with locked winning Round 1 #T$# scratchpad
            torch.manual_seed(seed + 739)
            tr_r2_refine = self.generate_uta(
                user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.22,
                recurrence_steps=2,
                latent_prior=z_prior_1,
                latent_prior_scale=prior_scale,
                thought_prefix=(best_t_full if best_t_full else None),
                lock_thought=bool(best_t_full),
            )
            r2_pool.append((2, 0.22, tr_r2_refine, 2, "r2_rl_answer_refine"))

            # 5. Joint COE Verification across all Round 1 + Round 2 candidates
            combined_pool = r1_pool + r2_pool
            all_scored, _ = _score_pool(combined_pool, z_prior_1)

            r1_joint = [c for c in all_scored if c.round_id == 1]
            r2_joint = [c for c in all_scored if c.round_id == 2]
            r1_best = max(r1_joint, key=lambda c: c.composite_score)
            r2_best = max(r2_joint, key=lambda c: c.composite_score)
            overall_best = max(
                all_scored,
                key=lambda c: c.composite_score + 0.015 * max(0.0, c.abstraction_alignment),
            )

            return DreamRLGenerationOutput(
                text=overall_best.text,
                baseline_text=baseline_text,
                selected_index=overall_best.index,
                selected_round=overall_best.round_id,
                selected_strategy=overall_best.strategy,
                improved_over_baseline=(overall_best.text != baseline_text),
                improved_in_round2=(overall_best.round_id == 2),
                round1_best_score=round(r1_best.composite_score, 4),
                round2_best_score=round(r2_best.composite_score, 4),
                final_best_score=round(overall_best.composite_score, 4),
                dream_cache_matched=dream_matched,
                dream_top_similarity=round(dream_top_sim, 4),
                dream_matched_domains=dream_domains,
                dream_recommended_scaffold=dream_scaffold,
                extra_prompt_tokens=0,
                round1_advantages=r1_adv_tuple,
                candidates=tuple(all_scored),
                abstraction_norm=round(abs_norm, 4),
            )
        finally:
            self.coe.train(was_training)

    @torch.no_grad()
    def generate_uta_with_codebase_session(
        self,
        user_prompt: str,
        warmup_engine: object,
        session: object,
        *,
        max_files: int = 3,
        top_k_candidates: int = 6,
        max_new_tokens: int = 384,
        prior_scale: float = 0.08,
        rl_step_size: float = 0.35,
        seed: int = 42,
        depth_ponder: bool | None = True,
        margin_halt_threshold: float | None = 0.35,
        recurrent_kv: bool = True,
    ) -> CodebaseRAGGenerationOutput:
        """Synthesize `#U$# -> #T$# -> #A$#` grounded in a warmed `CodebaseSessionState` (`A.TXY` + `TurboQuant` + `ContextLMCompressor` + 2-Round RL).

        1. **Multi-File Virtual Memory Retrieval (`0` prompt tokens):** Routes `user_prompt` via
           `warmup_engine.synthesize_multifile_context` to locate the exact `A=1` symbol in `ATXYPageStore`,
           traverses `CALLS` edges across modules, decompresses the participating 4-bit `TurboQuant`
           file pages, and folds them through `ContextLMCompressor` into soft gist tokens (`compressed_state`)
           and a unified multi-file `Abstraction` latent prior $\\mathbf{z}_{\\text{code}}^{(0)}$.
        2. **Round 1 Rollouts (Unsteered + Multi-File Latent Prior + AST-Premise Anchored):**
           Generates Round 1 `#T$# -> #A$#` trajectories across unsteered baseline, multi-file
           latent-steered, and AST-premise-anchored strategies while keeping `#U$#` 100% clean (`0` extra prompt tokens).
        3. **Warmed `COE` + Multi-File `ContextLMCompressor` Cross-Attention Verification:**
           Scores all candidates via the session-warmed `COE` operators (`Decision`, `Deduction`,
           `Evaluation`, `Metacognition`), upper-CED cross-attention directly over the `ContextLMCompressor`
           multi-file gist tokens, and AST/call-graph grounding, then updates $\\mathbf{z}_{\\text{code}}^{(0)} \\to \\mathbf{z}_{\\text{code}}^{(1)}$
           using Ornith-1.5 group-normalized advantages $A_i$.
        4. **Round 2 Tree-Guided Rollouts:** Launches Round 2 rollouts conditioned on $\\mathbf{z}_{\\text{code}}^{(1)}$
           and the winning Round 1 `#T$#` scratchpad, then verifies the combined pool.
        """
        from pathlib import Path
        from .context_memory import (
            CodebaseSessionState,
            CodebaseWarmupEngine,
            _code_tokens_and_stems,
            _truncate_to_token_budget,
        )
        from .data import DecisionExample, collate_decisions
        from .operators import pearson_similarity
        from .ornith_rl import detect_doom_loop, group_normalized_advantages

        assert isinstance(warmup_engine, CodebaseWarmupEngine)
        assert isinstance(session, CodebaseSessionState)

        was_training = self.coe.training
        self.coe.eval()
        try:
            device = next(self.coe.parameters()).device
            vocab_size = int(getattr(self.coe.backbone.config, "vocab_size", 260))
            max_seq = int(self.coe.backbone.config.max_seq_len)
            use_amp = device.type == "cuda"

            # 1. Multi-File A.TXY + TurboQuant + ContextLMCompressor Synthesis
            mfile = warmup_engine.synthesize_multifile_context(
                self.coe,
                session,
                user_prompt,
                max_files=max_files,
                top_k_candidates=top_k_candidates,
                depth_ponder=depth_ponder,
                margin_halt_threshold=margin_halt_threshold,
            )
            z_code_0 = mfile.multifile_latent_prior.to(device=device).float()

            rel_path, sym_short = (
                mfile.primary_symbol.split(":", 1)
                if ":" in mfile.primary_symbol
                else (mfile.primary_symbol, mfile.primary_symbol)
            )
            mod_short = Path(rel_path).name
            mod_stem = Path(rel_path).stem
            callee_names = [c["symbol"].split(":")[-1] for c in mfile.callees[:3]]
            callee_clause = f" and calls {', '.join(callee_names)}" if callee_names else ""

            # Extract clean human-readable docstring sentence from primary_summary
            summary_no_uses = mfile.primary_summary.split("[uses:", 1)[0].strip()
            summary_parts = [
                p.strip()
                for p in summary_no_uses.split(" | ")
                if p.strip() and not p.strip().startswith(("(", "[methods:", "refs:"))
            ]
            raw_doc = (
                summary_parts[0]
                if summary_parts
                else summary_no_uses.split(" | ")[0].strip()
            )
            clean_doc = (
                re.split(r"(?<=[.!?])\s+", raw_doc)[0]
                .replace("`", "")
                .strip()
                .rstrip(".")
            )
            files_stem = ", ".join(Path(p).stem for p in mfile.participating_files)

            ast_premise = (
                f"In the {mod_stem} module ({files_stem}), {sym_short} ({clean_doc}) is defined{callee_clause}, which allows"
            )
            ast_answer_prefix = (
                f"In {mod_stem}, {sym_short} ({clean_doc}) is defined in {mod_stem}{callee_clause} to"
            )
            short_answer_prefix = f"In {mod_stem}, {sym_short} is defined to"

            # Build reference token set for AST & call-graph grounding verification
            ref_code_tokens = _code_tokens_and_stems(
                f"{sym_short} {mod_short} {clean_doc} {' '.join(callee_names)} {mfile.focal_statement}"
            )

            def _score_codebase_pool(
                pool: list[tuple[int, float, str, int, str]],
                z_ref: Tensor,
            ) -> tuple[list[CandidateTraceScore], list[Tensor]]:
                bodies: list[str] = []
                for _, _, tr, _, _ in pool:
                    idx_t = tr.find("#T$#")
                    b = tr[idx_t:].strip() if idx_t >= 0 else tr.strip()
                    bodies.append(b if b else "#T$# empty\n#A$# empty")

                ctx_text = (
                    f"#U$# {user_prompt.strip()}\n"
                    f"Codebase context: `{mfile.primary_symbol}` ({clean_doc}){callee_clause}.\n"
                    f"#Q$# [choice] Select the most accurate, codebase-grounded #T$# -> #A$# response."
                )
                ex = DecisionExample(
                    question_id="codebase_rag_loop",
                    kind="choice",
                    context=_truncate_to_token_budget(ctx_text, vocab_size, max_seq),
                    descriptions=tuple(
                        _truncate_to_token_budget(b, vocab_size, max_seq)
                        for b in bodies
                    ),
                    values=tuple(f"cand_{i}" for i in range(len(bodies))),
                    target=0,
                )
                batch = collate_decisions(
                    [ex],
                    max_seq,
                    device=device,
                    vocab_size=vocab_size,
                    tokenizer=self.tokenizer,
                )
                ctx_ids = batch["context_ids"]
                ctx_lens = batch["context_lengths"]
                opt_ids = batch["option_ids"]
                opt_lens = batch["option_lengths"]

                k_cands = len(bodies)
                rep_ctx_ids = ctx_ids.expand(k_cands, -1).contiguous()
                rep_ctx_lens = ctx_lens.expand(k_cands).contiguous()
                flat_opt_ids = opt_ids[0]
                flat_opt_lens = opt_lens[0]

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    dec_out = self.coe.forward_decision(
                        ctx_ids,
                        ctx_lens,
                        opt_ids,
                        opt_lens,
                        recurrence_steps=self.schedule.recurrence_steps,
                        depth_ponder=depth_ponder,
                        margin_halt_threshold=margin_halt_threshold,
                    )
                    ded_out = self.coe.forward_deduction(
                        rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
                    )
                    eval_out = self.coe.forward_evaluation(
                        rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
                    )
                    meta_out = self.coe.forward_metacognition(
                        rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens
                    )
                    cand_abs_vecs = self.coe.compress_abstraction(
                        flat_opt_ids, flat_opt_lens, level="detailed"
                    ).float()

                    # Direct upper-CED cross-attention over ContextLMCompressor multi-file TurboQuant gist tokens!
                    opt_width = int(flat_opt_lens.max().item())
                    trimmed_opt_ids = flat_opt_ids[:, :opt_width].long()
                    opt_valid = (
                        torch.arange(opt_width, device=device)[None, :] < flat_opt_lens[:, None]
                    )
                    owner_rows = torch.zeros(k_cands, dtype=torch.long, device=device)
                    gist_decoded = self.coe.backbone.decode_candidates(
                        mfile.compressed_state,
                        trimmed_opt_ids,
                        opt_valid,
                        owner_rows=owner_rows,
                        recurrence_steps=self.schedule.recurrence_steps,
                        causal_local=True,
                        depth_ponder=bool(depth_ponder),
                    )
                    dummy_gist_ids = torch.zeros_like(mfile.compressed_state.valid, dtype=torch.long)
                    paired_gist_state, gist_cand_repr = self.coe.pooler.matched_pair_representations(
                        mfile.compressed_state.hidden,
                        dummy_gist_ids,
                        mfile.compressed_state.valid,
                        gist_decoded.hidden,
                        trimmed_opt_ids,
                        opt_valid,
                        owner_rows,
                        cand_anchor=gist_decoded.anchor,
                        strip_boundary=False,
                        bos_id=self.coe.config.bos_id,
                        eos_id=self.coe.config.eos_id,
                    )
                    gist_s_proj = self.coe.state_projection(paired_gist_state).float()
                    gist_c_proj = self.coe.output_projection(gist_cand_repr).float()
                    gist_sims = pearson_similarity(gist_s_proj, gist_c_proj).detach().cpu().tolist()

                    ref_proj = self.coe.state_projection(
                        z_ref.to(next(self.coe.state_projection.parameters()).dtype)
                    ).float()
                    cand_proj = self.coe.output_projection(
                        cand_abs_vecs.to(next(self.coe.output_projection.parameters()).dtype)
                    ).float()
                    abs_sims = pearson_similarity(
                        ref_proj.expand_as(cand_proj), cand_proj
                    ).detach().cpu().tolist()

                dec_sims = dec_out.similarities[0].detach().float().cpu().tolist()
                dec_probs = dec_out.probabilities[0].detach().float().cpu().tolist()
                ded_probs = ded_out.probabilities.detach().float().cpu().tolist()
                eval_utils = eval_out.utilities.detach().float().cpu().tolist()
                meta_probs = meta_out.probabilities.detach().float().cpu().tolist()

                scored: list[CandidateTraceScore] = []
                denom = max(1, min(12, len(ref_code_tokens)))
                for i, (r_steps, temp, tr, r_id, strat) in enumerate(pool):
                    d_sim = float(dec_sims[i])
                    d_prob = float(dec_probs[i])
                    u_eval = float(eval_utils[i])
                    p_ent = float(ded_probs[i][0])
                    p_con = float(ded_probs[i][1])
                    m_cor = float(meta_probs[i][0])
                    m_suf = float(meta_probs[i][1])
                    g_sim = float(gist_sims[i])
                    a_sim = 0.60 * g_sim + 0.40 * float(abs_sims[i])
                    s_int = _score_trace_integrity(user_prompt, tr)

                    t_part = (
                        tr.split("#T$#", 1)[1].split("#A$#", 1)[0].strip()
                        if "#T$#" in tr and "#A$#" in tr
                        else ""
                    )
                    a_part = tr.split("#A$#", 1)[1].strip() if "#A$#" in tr else ""
                    if len(a_part.split()) < 6 or a_part.startswith("#"):
                        s_int = max(0.0, s_int - 0.35)

                    # AST & Call-Graph grounding bonus weighted across both #T$# (35%) and #A$# (65%)
                    t_tokens = _code_tokens_and_stems(t_part)
                    a_tokens = _code_tokens_and_stems(a_part)
                    ast_ov = (
                        0.35 * (len(t_tokens & ref_code_tokens) / denom)
                        + 0.65 * (len(a_tokens & ref_code_tokens) / denom)
                    )
                    ast_bonus = min(0.18, 0.18 * ast_ov)

                    cand_toks = self.tokenizer.encode(bodies[i], add_bos=False, add_eos=False)
                    has_doom = detect_doom_loop(cand_toks) is not None
                    if has_doom:
                        s_int = max(0.0, s_int - 0.30)

                    is_vetoed = bool(
                        p_con > self.schedule.max_contradiction_prob
                        or m_cor < self.schedule.min_correctness_confidence
                        or s_int < 0.35
                        or has_doom
                    )
                    neural_score = (
                        0.32 * d_prob
                        + 0.22 * u_eval
                        + 0.18 * (p_ent - 0.75 * p_con)
                        + 0.14 * (0.7 * m_cor + 0.3 * m_suf)
                        + 0.14 * max(0.0, a_sim)
                    )
                    comp_score = neural_score + 0.15 * s_int + ast_bonus - (0.30 if is_vetoed else 0.0)
                    scored.append(
                        CandidateTraceScore(
                            index=i,
                            recurrence_steps=r_steps,
                            temperature=temp,
                            text=tr,
                            decision_similarity=round(d_sim, 4),
                            decision_prob=round(d_prob, 4),
                            evaluation_utility=round(u_eval, 4),
                            deduction_entailed=round(p_ent, 4),
                            deduction_contradicted=round(p_con, 4),
                            metacognition_correctness=round(m_cor, 4),
                            metacognition_sufficiency=round(m_suf, 4),
                            structural_score=round(s_int, 4),
                            composite_score=round(comp_score, 4),
                            vetoed=is_vetoed,
                            abstraction_alignment=round(a_sim, 4),
                            round_id=r_id,
                            strategy=strat,
                        )
                    )
                return scored, [cand_abs_vecs[i : i + 1].detach() for i in range(k_cands)]

            # 2. Round 1 Rollouts: Unsteered Baseline + Multi-File TurboQuant Latent Prior + AST-Premise Guided
            r1_specs: tuple[
                tuple[int, float, Tensor | None, float, str | None, str | None, str], ...
            ] = (
                (1, 0.0, None, 0.0, None, None, "r1_baseline"),
                (2, 0.0, None, 0.0, None, None, "r1_unsteered_r2"),
                (2, 0.0, z_code_0, prior_scale, None, None, "r1_multifile_prior"),
                (2, 0.28, z_code_0, prior_scale, None, short_answer_prefix, "r1_multifile_explore"),
                (2, 0.0, z_code_0, prior_scale, ast_premise, ast_answer_prefix, "r1_ast_premise_guided"),
                (2, 0.25, z_code_0, prior_scale, ast_premise, ast_answer_prefix, "r1_ast_premise_explore"),
            )
            r1_pool: list[tuple[int, float, str, int, str]] = []
            for idx, (r_steps, temp, l_prior, l_scale, t_pref, a_pref, strat) in enumerate(r1_specs):
                if temp > 0.0:
                    torch.manual_seed(seed + idx * 137)
                tr = self.generate_uta(
                    user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    recurrence_steps=r_steps,
                    latent_prior=l_prior,
                    latent_prior_scale=l_scale,
                    thought_prefix=t_pref,
                    lock_thought=False,
                    answer_prefix=a_pref,
                    recurrent_kv=bool(recurrent_kv and r_steps > 1),
                    depth_ponder=bool(depth_ponder and r_steps > 1),
                )
                r1_pool.append((r_steps, temp, tr, 1, strat))

            baseline_text = r1_pool[0][2]
            r1_scored, r1_abs_vecs = _score_codebase_pool(r1_pool, z_code_0)
            r1_rewards = torch.tensor(
                [c.composite_score + 0.04 * max(0.0, c.abstraction_alignment) for c in r1_scored],
                dtype=torch.float32,
                device=device,
            )
            r1_adv = group_normalized_advantages(r1_rewards)
            r1_adv_tuple = tuple(round(float(x), 4) for x in r1_adv.detach().cpu().tolist())

            # 3. Transient Inference-Time RL Latent Policy Update: z_code_0 -> z_code_1
            pos_mask = r1_adv > 0.0
            neg_mask = r1_adv < 0.0
            z_delta = torch.zeros_like(z_code_0)
            if bool(pos_mask.any()):
                pos_w = torch.softmax(r1_adv[pos_mask] / 0.5, dim=0)
                pos_stack = torch.cat(
                    [r1_abs_vecs[i] for i in range(len(r1_pool)) if bool(pos_mask[i])], dim=0
                )
                z_pos = (pos_w.unsqueeze(-1) * (pos_stack - z_code_0)).sum(dim=0, keepdim=True)
                z_delta = z_delta + rl_step_size * z_pos
            if bool(neg_mask.any()):
                neg_w = torch.softmax((-r1_adv[neg_mask]) / 0.5, dim=0)
                neg_stack = torch.cat(
                    [r1_abs_vecs[i] for i in range(len(r1_pool)) if bool(neg_mask[i])], dim=0
                )
                z_neg = (neg_w.unsqueeze(-1) * (neg_stack - z_code_0)).sum(dim=0, keepdim=True)
                z_delta = z_delta - (0.4 * rl_step_size) * z_neg

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                z_code_1 = self.coe.abstraction_norm(
                    (z_code_0 + z_delta).to(next(self.coe.abstraction_norm.parameters()).dtype)
                ).float()
                abs_norm = float(z_code_1.norm(dim=-1).mean().item())

            best_r1_idx = int(r1_rewards.argmax().item())
            best_r1_trace = r1_pool[best_r1_idx][2]
            best_t_full = ""
            first_t_sent = ast_premise
            if "#T$#" in best_r1_trace and "#A$#" in best_r1_trace:
                t_raw = best_r1_trace.split("#T$#", 1)[1].split("#A$#", 1)[0].strip()
                best_t_full = t_raw
                sent_parts = re.split(r"(?<=[.!?])\s+", t_raw)
                if sent_parts and len(sent_parts[0].strip()) >= 12:
                    first_t_sent = sent_parts[0].strip()

            # 4. Round 2 Tree-Guided Rollouts conditioned on RL-updated z_code_1
            r2_pool: list[tuple[int, float, str, int, str]] = []

            torch.manual_seed(seed + 509)
            tr_r2_full = self.generate_uta(
                user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.22,
                recurrence_steps=2,
                latent_prior=z_code_1,
                latent_prior_scale=prior_scale * 1.25,
                thought_prefix=ast_premise,
                lock_thought=False,
                answer_prefix=ast_answer_prefix,
                recurrent_kv=recurrent_kv,
                depth_ponder=bool(depth_ponder),
            )
            r2_pool.append((2, 0.22, tr_r2_full, 2, "r2_rl_multifile_steered"))

            torch.manual_seed(seed + 619)
            tr_r2_tree = self.generate_uta(
                user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.18,
                recurrence_steps=2,
                latent_prior=z_code_1,
                latent_prior_scale=prior_scale,
                thought_prefix=first_t_sent,
                lock_thought=False,
                answer_prefix=ast_answer_prefix,
                recurrent_kv=recurrent_kv,
                depth_ponder=bool(depth_ponder),
            )
            r2_pool.append((2, 0.18, tr_r2_tree, 2, "r2_tree_thought_cont"))

            torch.manual_seed(seed + 743)
            tr_r2_refine = self.generate_uta(
                user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.20,
                recurrence_steps=2,
                latent_prior=z_code_1,
                latent_prior_scale=prior_scale,
                thought_prefix=(best_t_full if best_t_full else ast_premise),
                lock_thought=bool(best_t_full),
                answer_prefix=ast_answer_prefix,
                recurrent_kv=recurrent_kv,
                depth_ponder=bool(depth_ponder),
            )
            r2_pool.append((2, 0.20, tr_r2_refine, 2, "r2_rl_answer_refine"))

            # 5. Joint Warmed-COE + Multi-File ContextLMCompressor Verification across Round 1 + Round 2
            combined_pool = r1_pool + r2_pool
            all_scored, _ = _score_codebase_pool(combined_pool, z_code_1)

            r1_joint = [c for c in all_scored if c.round_id == 1]
            r2_joint = [c for c in all_scored if c.round_id == 2]
            r1_best = max(r1_joint, key=lambda c: c.composite_score)
            r2_best = max(r2_joint, key=lambda c: c.composite_score)
            overall_best = max(
                all_scored,
                key=lambda c: c.composite_score + 0.02 * max(0.0, c.abstraction_alignment),
            )

            return CodebaseRAGGenerationOutput(
                text=overall_best.text,
                baseline_text=baseline_text,
                selected_index=overall_best.index,
                selected_round=overall_best.round_id,
                selected_strategy=overall_best.strategy,
                improved_over_baseline=(overall_best.text != baseline_text),
                improved_in_round2=(overall_best.round_id == 2),
                round1_best_score=round(r1_best.composite_score, 4),
                round2_best_score=round(r2_best.composite_score, 4),
                final_best_score=round(overall_best.composite_score, 4),
                routed_address=mfile.primary_address,
                routed_symbol=mfile.primary_symbol,
                routed_summary=mfile.primary_summary,
                routed_confidence=round(mfile.primary_confidence, 4),
                focal_statement=mfile.focal_statement,
                focal_invariant=mfile.focal_invariant,
                callees=tuple(c["symbol"] for c in mfile.callees),
                participating_files=mfile.participating_files,
                num_files_folded=mfile.num_files_folded,
                raw_source_tokens=mfile.raw_source_tokens,
                compressed_gist_tokens=mfile.compressed_gist_tokens,
                token_folding_ratio=round(mfile.token_folding_ratio, 2),
                byte_compression_ratio=round(mfile.byte_compression_ratio, 2),
                extra_prompt_tokens=0,
                round1_advantages=r1_adv_tuple,
                candidates=tuple(all_scored),
                abstraction_norm=round(abs_norm, 4),
            )
        finally:
            self.coe.train(was_training)


def compute_joint_coe_loss(
    coe: CognitiveOperatorEnsemble,
    *,
    lm_batch: tuple[Tensor, Tensor] | None = None,
    decision_batch: dict[str, Tensor] | None = None,
    deduction_batch: dict[str, Tensor] | None = None,
    evaluation_batch: dict[str, Tensor] | None = None,
    metacognition_batch: dict[str, Tensor] | None = None,
    weights: dict[str, float] | None = None,
    condition_on_prefix_gist: bool = False,
    latent_prior_scale: float = 0.0,
    recurrence_steps: int | None = None,
    recurrent_kv: bool = False,
    depth_ponder: bool | None = None,
    deep_supervision_weight: float = 0.0,
) -> JointCOELossOutput:
    """Compute Stage-2 joint multi-task loss across `U$ -> T$ -> A$` LM and COE operators."""
    w = {
        "lm": 1.0,
        "decision": 1.0,
        "deduction": 0.5,
        "evaluation": 0.5,
        "metacognition": 0.5,
    }
    if weights is not None:
        w.update(weights)

    total: Tensor | None = None
    lm_loss = decision_loss = deduction_loss = evaluation_loss = metacognition_loss = None

    def _accum(current: Tensor | None, term: Tensor, scale: float) -> Tensor:
        scaled = term * scale
        return scaled if current is None else current + scaled

    if lm_batch is not None:
        inp, tgt = lm_batch
        lm_out = coe.forward_inference(
            inp,
            targets=tgt,
            condition_on_prefix_gist=condition_on_prefix_gist,
            latent_prior_scale=latent_prior_scale,
            recurrence_steps=recurrence_steps,
            recurrent_kv=recurrent_kv,
            depth_ponder=depth_ponder,
            deep_supervision_weight=deep_supervision_weight,
        )
        assert lm_out.loss is not None
        lm_loss = lm_out.loss
        total = _accum(total, lm_loss, w["lm"])

    if decision_batch is not None:
        dec_out = coe.forward_decision(**decision_batch, depth_ponder=depth_ponder)
        assert dec_out.loss is not None
        decision_loss = dec_out.loss
        total = _accum(total, decision_loss, w["decision"])

    if deduction_batch is not None:
        ded_out = coe.forward_deduction(**deduction_batch)
        assert ded_out.loss is not None
        deduction_loss = ded_out.loss
        total = _accum(total, deduction_loss, w["deduction"])

    if evaluation_batch is not None:
        eval_out = coe.forward_evaluation(**evaluation_batch)
        assert eval_out.loss is not None
        evaluation_loss = eval_out.loss
        total = _accum(total, evaluation_loss, w["evaluation"])

    if metacognition_batch is not None:
        meta_out = coe.forward_metacognition(**metacognition_batch)
        assert meta_out.loss is not None
        metacognition_loss = meta_out.loss
        total = _accum(total, metacognition_loss, w["metacognition"])

    if total is None:
        raise ValueError("compute_joint_coe_loss requires at least one active batch")

    return JointCOELossOutput(
        total_loss=total,
        lm_loss=lm_loss,
        decision_loss=decision_loss,
        deduction_loss=deduction_loss,
        evaluation_loss=evaluation_loss,
        metacognition_loss=metacognition_loss,
    )
