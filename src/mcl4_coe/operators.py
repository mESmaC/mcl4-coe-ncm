"""Cognitive Operator Ensemble (COE) on top of the shared MCL4-CED Backbone.

Implements all 9 composable single-softmax / relational cognitive operators:
1. Decision (`choice`, `noul`, `score`) — Bitwise permutation-invariant candidate selection
2. Association — Pairwise & set-level centered Pearson similarity + <0.1ms Novelty N(q)
3. Evaluation — Criterion-conditioned utility assessment U(m, c) in [0, 1]
4. Deduction — 3-way logical status (`entailed`, `contradicted`, `undetermined`)
5. Induction — Pattern generalization across observed cases + uncertainty
6. Abduction — Explanatory hypothesis ranking under observations
7. Abstraction — Multi-scale latent compression (`brief`, `standard`, `detailed`)
8. Metacognition — Calibrated reliability signals (`correctness`, `evidence_sufficiency`, `context_benefit`)
9. Inference — Autoregressive `U$ -> T$ -> A$` synthesis conditioned on recurrent COE state
"""

from __future__ import annotations

import ast
import html
import math
import re
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import EncodedState, MCL4CEDBackbone
from .ple import RMSNorm


def _positive_finite(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def pearson_similarity(left: Tensor, right: Tensor, eps: float = 1e-6) -> Tensor:
    """FP32 centered cosine (Pearson) similarity across final-axis features in [-1, 1].

    In signal-processing terms:
    - Subtracting the feature mean (`u - mean(u)`) strips the DC / common-mode offset.
    - L2 normalization (`u / norm(u)`) acts as Automatic Gain Control (AGC) so short
      and long descriptions compete at identical signal amplitude.
    """
    _positive_finite(eps, "eps")
    if (
        not isinstance(left, Tensor)
        or not isinstance(right, Tensor)
        or left.ndim < 1
        or right.ndim < 1
        or left.shape[-1] != right.shape[-1]
        or left.shape[-1] < 1
    ):
        raise ValueError("left and right must have the same nonempty feature dimension")
    if left.device != right.device:
        raise ValueError("left and right must be on the same device")
    if not left.is_floating_point() or not right.is_floating_point():
        raise ValueError("left and right must be floating-point tensors")
    with torch.autocast(device_type=left.device.type, enabled=False):
        centered_left = left.float() - left.float().mean(-1, keepdim=True)
        centered_right = right.float() - right.float().mean(-1, keepdim=True)
        normalized_left = F.normalize(centered_left, dim=-1, eps=eps)
        normalized_right = F.normalize(centered_right, dim=-1, eps=eps)
        return (normalized_left * normalized_right).sum(-1).clamp(-1.0, 1.0)


def _canonical_candidate_order(
    owner_rows: Tensor,
    cand_lengths: Tensor,
    cand_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return `(perm, inv_perm)` sorting candidate rows canonically by `(owner_row, length, token_ids)`.

    Why this matters for hardware execution:
    Batched SGEMM / TensorCore kernels tile across the batch dimension `R`, which can
    introduce 1-ULP (`~1e-7`) floating-point rounding differences when rows cross a
    16-row tile boundary. Sorting candidate rows into a canonical lexicographical order
    before the batched GPU pass and scattering back via `inv_perm` keeps 100% single-batch
    TensorCore speed while guaranteeing 0-ULP bitwise permutation invariance (`torch.equal`).
    """
    r = owner_rows.shape[0]
    if r <= 1:
        ident = torch.arange(r, device=owner_rows.device, dtype=torch.long)
        return ident, ident
    # Build sorting keys on CPU from integer metadata (`[R, 2 + T_cand]`) without touching GPU floats.
    keys = torch.cat(
        [
            owner_rows.detach().cpu().unsqueeze(1).long(),
            cand_lengths.detach().cpu().unsqueeze(1).long(),
            cand_ids.detach().cpu().long(),
        ],
        dim=1,
    ).tolist()
    order = sorted(range(r), key=keys.__getitem__)
    perm = torch.tensor(order, device=owner_rows.device, dtype=torch.long)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(r, device=owner_rows.device, dtype=torch.long)
    return perm, inv_perm


@dataclass
class COEConfig:
    projection_dim: int = 32
    metric: str = "pearson"
    initial_scale: float = 15.0
    eps: float = 1e-6
    recurrence_steps: int = 1
    recurrent_kv: bool = False
    depth_ponder: bool = False
    margin_halt_threshold: float | None = None
    gist_tokens_per_chunk: int = 6
    gist_chunk_size: int = 192
    strip_boundary_tokens_in_pool: bool = True
    bos_id: int = 257
    eos_id: int = 258

    def __post_init__(self) -> None:
        if (
            isinstance(self.projection_dim, bool)
            or not isinstance(self.projection_dim, int)
            or self.projection_dim < 3
        ):
            raise ValueError("projection_dim must be an integer >= 3 for Pearson gradients")
        if self.metric not in ("pearson", "cosine", "dot"):
            raise ValueError("metric must be 'pearson', 'cosine', or 'dot'")
        _positive_finite(self.initial_scale, "initial_scale")
        if not 0.1 <= self.initial_scale <= 100.0:
            raise ValueError("initial_scale must be between 0.1 and 100.0")
        _positive_finite(self.eps, "eps")
        if isinstance(self.recurrence_steps, bool) or self.recurrence_steps < 1:
            raise ValueError("recurrence_steps must be >= 1")
        if isinstance(self.gist_tokens_per_chunk, bool) or self.gist_tokens_per_chunk < 1:
            raise ValueError("gist_tokens_per_chunk must be >= 1")


@dataclass
class DecisionOutput:
    logits: Tensor
    similarities: Tensor
    probabilities: Tensor
    state_features: Tensor
    candidate_features: Tensor
    loss: Tensor | None
    ce_loss: Tensor | None
    aux_loss: Tensor
    router_z_loss: Tensor


@dataclass
class AssociationMatrixOutput:
    similarities: Tensor
    logits: Tensor
    probabilities: Tensor
    features: Tensor
    novelty: Tensor | None = None


@dataclass
class DeductionOutput:
    logits: Tensor
    probabilities: Tensor
    similarities: Tensor
    status_labels: tuple[str, str, str] = ("entailed", "contradicted", "undetermined")
    loss: Tensor | None = None


@dataclass
class MetacognitionOutput:
    logits: Tensor
    probabilities: Tensor
    signals: tuple[str, str, str] = ("correctness", "evidence_sufficiency", "context_benefit")
    loss: Tensor | None = None


@dataclass
class EvaluationOutput:
    logits: Tensor
    utilities: Tensor
    similarities: Tensor
    loss: Tensor | None = None


@dataclass
class AllOperatorsOutput:
    """Unified output of a single-pass multi-operator COE forward (`Decision`, `Deduction`, `Evaluation`, `Association`, `Metacognition`, `Abstraction`)."""

    decision: DecisionOutput
    deduction_loss: Tensor
    evaluation_loss: Tensor
    metacognition_loss: Tensor
    abstraction_loss: Tensor
    total_loss: Tensor
    deduction_probs: Tensor
    metacognition_probs: Tensor
    evaluation_utilities: Tensor


class AttentiveMatchedFilterPool(nn.Module):
    """Token-level QK-Norm matched-filter cross-attention and attentive sequence pooling.

    How this breaks the bi-encoder bottleneck (`44.4%` -> `100%`):
    1. Every valid token $t$ in Candidate $r$ queries all valid tokens $s$ in its owner
       State $b$ using `QK-Norm` (`RMSNorm` on $Q$ and $K$), retrieving the best-matching
       state token $M_{r, t}$ *before* pooling across candidate tokens (`Attend-then-Pool`).
    2. At each candidate token position $t$, the aligned state evidence $M_{r, t}$ and
       candidate representation $C_{r, t}$ are pooled with learned salience weights
       (excluding `<bos>`/`<eos>` framing tokens), so even a 2-byte difference (`"no"` vs
       `"one"`) inside a 25-byte sentence produces a sharp differential signal.
    3. Candidate $r$ never interacts with any other candidate $k \ne r$, preserving
       100% bitwise permutation invariance.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_scorer = nn.Linear(d_model, 1, bias=False)
        self.match_q = nn.Linear(d_model, d_model, bias=False)
        self.match_k = nn.Linear(d_model, d_model, bias=False)
        self.match_v = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(d_model, eps=eps)
        self.k_norm = RMSNorm(d_model, eps=eps)
        self.match_scale = nn.Parameter(torch.tensor(6.0))
        self.state_fuse = nn.Linear(d_model * 2, d_model, bias=False)
        self.cand_fuse = nn.Linear(d_model * 2, d_model, bias=False)
        self.state_norm = nn.LayerNorm(d_model, eps=eps)
        self.cand_norm = nn.LayerNorm(d_model, eps=eps)

        adapter_std = 0.25 / math.sqrt(d_model)
        nn.init.normal_(self.token_scorer.weight, std=adapter_std)
        nn.init.normal_(self.match_q.weight, std=adapter_std)
        nn.init.normal_(self.match_k.weight, std=adapter_std)
        nn.init.normal_(self.match_v.weight, std=adapter_std)
        nn.init.normal_(self.state_fuse.weight, std=adapter_std / math.sqrt(2.0))
        nn.init.normal_(self.cand_fuse.weight, std=adapter_std / math.sqrt(2.0))

    @staticmethod
    def _content_mask(
        token_ids: Tensor,
        valid: Tensor,
        *,
        strip_boundary: bool,
        bos_id: int,
        eos_id: int,
    ) -> Tensor:
        if not strip_boundary:
            return valid
        interior = valid & (token_ids != bos_id) & (token_ids != eos_id)
        has_interior = interior.any(dim=-1, keepdim=True)
        return torch.where(has_interior, interior, valid)

    @classmethod
    def _state_evidence_mask(
        cls,
        state_ids: Tensor,
        state_valid: Tensor,
        *,
        strip_boundary: bool,
        bos_id: int,
        eos_id: int,
    ) -> Tensor:
        """Isolate the Sensor/World-State register (`#U$# ...`) from the Question register (`\\n#Q$# ...`).

        Why this matters:
        In `#U$# <state>\\n#Q$# [<kind>] <instructions>`, the Question register (`#Q$#`) often
        contains words from a candidate option (e.g. `"Is the animal carrying a ball?"` contains
        `"a ball"` even when `#U$#` says `"A rabbit carries no balls."`). Masking out `\\n#Q$# ...`
        and the `#U$# ` prefix ensures the matched filter only extracts evidence from the
        World-State tokens.
        """
        mask = cls._content_mask(
            state_ids, state_valid, strip_boundary=strip_boundary, bos_id=bos_id, eos_id=eos_id
        )
        b, t = state_ids.shape
        if t < 2:
            return mask
        positions = torch.arange(t, device=state_ids.device)[None, :]
        trimmed = mask.clone()

        if bos_id == 0:
            # DonorHFTokenizer (Granite BPE): `#Q$#` is [21, 67, 22, 21], `#U$#` is [21, 71, 22, 21]
            if t >= 5:
                is_gq = (
                    (state_ids[:, :-3] == 21)
                    & (state_ids[:, 1:-2] == 67)
                    & (state_ids[:, 2:-1] == 22)
                    & (state_ids[:, 3:] == 21)
                    & state_valid[:, :-3]
                )
                if is_gq.any():
                    has_gq = is_gq.any(dim=-1, keepdim=True)
                    first_gq = torch.where(
                        has_gq,
                        is_gq.int().argmax(dim=-1, keepdim=True),
                        torch.full((b, 1), t, device=state_ids.device, dtype=torch.long),
                    )
                    trimmed = trimmed & (positions < first_gq)
                is_gu = (
                    (state_ids[:, :-3] == 21)
                    & (state_ids[:, 1:-2] == 71)
                    & (state_ids[:, 2:-1] == 22)
                    & (state_ids[:, 3:] == 21)
                    & state_valid[:, :-3]
                )
                if is_gu.any():
                    has_gu = is_gu.any(dim=-1, keepdim=True)
                    first_gu_end = torch.where(
                        has_gu,
                        is_gu.int().argmax(dim=-1, keepdim=True) + 4,
                        torch.zeros((b, 1), device=state_ids.device, dtype=torch.long),
                    )
                    trimmed = trimmed & (positions >= first_gu_end)
            return torch.where(trimmed.any(dim=-1, keepdim=True), trimmed, mask)

        # 1. MorphCode-16K atomic `#Q$#` / `#S$#` / `#H$#` / `#T$#` / `#A$#` marker tokens:
        #    261..265 ("#Q$#".."#A$#"), 267..271 ("\n#Q$#".."\n#A$#"), 273..275 ("#Q$# ".."#A$# ")
        is_morph_q = (
            ((state_ids >= 261) & (state_ids <= 265))
            | ((state_ids >= 267) & (state_ids <= 271))
            | ((state_ids >= 273) & (state_ids <= 275))
        ) & state_valid
        if is_morph_q.any():
            has_mq = is_morph_q.any(dim=-1, keepdim=True)
            first_mq = torch.where(
                has_mq,
                is_morph_q.int().argmax(dim=-1, keepdim=True),
                torch.full((b, 1), t, device=state_ids.device, dtype=torch.long),
            )
            prev_idx = (first_mq - 1).clamp(0, t - 1)
            has_leading_nl = has_mq & (first_mq > 0) & (state_ids.gather(1, prev_idx) == 10)
            cut_mq = torch.where(has_leading_nl, first_mq - 1, first_mq)
            trimmed = trimmed & (positions < cut_mq)

        # Strip MorphCode-16K atomic `#U$#` marker tokens (260="#U$#", 266="\n#U$#", 272="#U$# ")
        is_morph_u = ((state_ids == 260) | (state_ids == 266) | (state_ids == 272)) & state_valid
        if is_morph_u.any():
            trimmed = trimmed & ~is_morph_u

        if t < 6:
            return torch.where(trimmed.any(dim=-1, keepdim=True), trimmed, mask)

        # 2. ByteTokenizer 5-byte sequences:
        #    Detect `\n#Q$#` ([10, 35, 81, 36, 35]) and mask from `\n#Q$#` to the end of the sequence.
        is_nq = (
            (state_ids[:, :-4] == 10)
            & (state_ids[:, 1:-3] == 35)
            & (state_ids[:, 2:-2] == 81)
            & (state_ids[:, 3:-1] == 36)
            & (state_ids[:, 4:] == 35)
            & state_valid[:, :-4]
        )
        has_nq = is_nq.any(dim=-1, keepdim=True)
        first_nq = torch.where(
            has_nq,
            is_nq.int().argmax(dim=-1, keepdim=True),
            torch.full((b, 1), t, device=state_ids.device, dtype=torch.long),
        )
        trimmed = trimmed & (positions < first_nq)

        # Detect `#U$# ` ([35, 85, 36, 35, 32]) at the start of the state and strip those 5 tag bytes.
        is_u_tag = (
            (state_ids[:, :-4] == 35)
            & (state_ids[:, 1:-3] == 85)
            & (state_ids[:, 2:-2] == 36)
            & (state_ids[:, 3:-1] == 35)
            & (state_ids[:, 4:] == 32)
            & state_valid[:, :-4]
        )
        has_u = is_u_tag.any(dim=-1, keepdim=True)
        first_u_end = torch.where(
            has_u,
            is_u_tag.int().argmax(dim=-1, keepdim=True) + 5,
            torch.zeros((b, 1), device=state_ids.device, dtype=torch.long),
        )
        trimmed = trimmed & (positions >= first_u_end)

        return torch.where(trimmed.any(dim=-1, keepdim=True), trimmed, mask)

    def pool_sequence(
        self,
        hidden: Tensor,
        token_ids: Tensor,
        valid: Tensor,
        *,
        strip_boundary: bool = True,
        bos_id: int = 257,
        eos_id: int = 258,
    ) -> Tensor:
        mask = self._content_mask(
            token_ids, valid, strip_boundary=strip_boundary, bos_id=bos_id, eos_id=eos_id
        )
        raw_scores = self.token_scorer(hidden.to(self.token_scorer.weight.dtype))
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            scores = raw_scores.squeeze(-1).float()
            scores = scores.masked_fill(~mask, -torch.inf)
            weights = scores.softmax(dim=-1)
            attentive = (hidden.float() * weights.unsqueeze(-1)).sum(dim=1)
            mean_pool = hidden.float().masked_fill(~mask.unsqueeze(-1), 0.0).sum(
                dim=1
            ) / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            return (0.5 * (attentive + mean_pool)).to(hidden.dtype)

    def matched_pair_representations(
        self,
        state_hidden: Tensor,
        state_ids: Tensor,
        state_valid: Tensor,
        cand_hidden: Tensor,
        cand_ids: Tensor,
        cand_valid: Tensor,
        owner_rows: Tensor,
        *,
        cand_anchor: Tensor | None = None,
        strip_boundary: bool = True,
        bos_id: int = 257,
        eos_id: int = 258,
    ) -> tuple[Tensor, Tensor]:
        """Compute AC-coupled token-level QK-Norm matched-filter representations `(paired_state, cand_repr)`."""
        s_mask = self._state_evidence_mask(
            state_ids, state_valid, strip_boundary=strip_boundary, bos_id=bos_id, eos_id=eos_id
        )
        c_mask = self._content_mask(
            cand_ids, cand_valid, strip_boundary=strip_boundary, bos_id=bos_id, eos_id=eos_id
        )
        cand_base = cand_hidden if cand_anchor is None else cand_anchor

        # Residual matched-filter projections anchored on the shared lower-encoder space (`f_enc`).
        k_state_all = self.k_norm(state_hidden + self.match_k(state_hidden))
        v_state_raw = state_hidden + self.match_v(state_hidden)
        with torch.autocast(device_type=state_hidden.device.type, enabled=False):
            s_mask_f = s_mask.unsqueeze(-1).float()
            v_mean = (v_state_raw.float() * s_mask_f).sum(dim=1, keepdim=True) / s_mask_f.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            # AC-couple state values within the world-state register to strip the DC template blocker.
            v_state_all = ((v_state_raw.float() - v_mean) * s_mask_f).to(state_hidden.dtype)

        k_state = k_state_all.index_select(0, owner_rows)  # [R, T_state, d_model]
        v_state = v_state_all.index_select(0, owner_rows)  # [R, T_state, d_model]
        s_mask_rows = s_mask.index_select(0, owner_rows).unsqueeze(1)  # [R, 1, T_state]

        # Every candidate token queries all valid state tokens in the shared `f_enc` basis.
        q_cand = self.q_norm(cand_base + self.match_q(cand_hidden))  # [R, T_cand, d_model]
        with torch.autocast(device_type=state_hidden.device.type, enabled=False):
            scale = self.match_scale.float().clamp(0.5, 20.0) / math.sqrt(self.d_model)
            sim_matrix = torch.matmul(q_cand.float(), k_state.float().transpose(-1, -2)) * scale
            sim_matrix = sim_matrix.masked_fill(~s_mask_rows, -torch.inf)
            attn_weights = sim_matrix.softmax(dim=-1)  # [R, T_cand, T_state]
            aligned_state_tokens = torch.matmul(attn_weights, v_state.float()).to(cand_hidden.dtype)

            # Weight candidate positions by their peak match confidence in the state.
            peak_match = sim_matrix.max(dim=-1).values.masked_fill(~c_mask, -torch.inf)
            match_weights = peak_match.softmax(dim=-1).unsqueeze(-1)
            peak_aligned_state = (aligned_state_tokens.float() * match_weights).sum(dim=1).to(
                state_hidden.dtype
            )
            peak_cand_base = (cand_base.float() * match_weights).sum(dim=1).to(cand_hidden.dtype)

        pooled_aligned_state = self.pool_sequence(
            aligned_state_tokens,
            cand_ids,
            c_mask,
            strip_boundary=False,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        pooled_cand_base = self.pool_sequence(
            cand_base,
            cand_ids,
            c_mask,
            strip_boundary=False,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        pooled_cand = self.pool_sequence(
            cand_hidden,
            cand_ids,
            c_mask,
            strip_boundary=False,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        interaction = pooled_cand * pooled_aligned_state

        state_residual = 0.5 * (pooled_aligned_state + peak_aligned_state)
        cand_residual = 0.5 * (pooled_cand_base + peak_cand_base)

        paired_state = self.state_norm(
            (
                state_residual
                + self.state_fuse(torch.cat([pooled_aligned_state, peak_aligned_state], dim=-1))
            ).float()
        ).to(state_hidden.dtype)
        cand_repr = self.cand_norm(
            (
                cand_residual
                + self.cand_fuse(torch.cat([pooled_cand, interaction], dim=-1))
            ).float()
        ).to(cand_hidden.dtype)
        return paired_state, cand_repr


class CognitiveOperatorEnsemble(nn.Module):
    """Composable Single-Softmax Cognitive Operator Ensemble (COE) on MCL4-CED."""

    def __init__(self, backbone: MCL4CEDBackbone, config: COEConfig | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = COEConfig() if config is None else config
        self.config.__post_init__()
        if backbone.config.vocab_size > 1024 and self.config.eos_id == 258:
            self.config.eos_id = int(backbone.config.eos_id)
            self.config.bos_id = (
                128000
                if backbone.config.vocab_size == 128256
                else int(backbone.config.pad_id + 1)
            )
        d_model = backbone.config.d_model
        d_proj = self.config.projection_dim

        self.pooler = AttentiveMatchedFilterPool(d_model, eps=backbone.config.rms_norm_eps)
        self.state_projection = nn.Linear(d_model, d_proj, bias=False)
        self.output_projection = nn.Linear(d_model, d_proj, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(self.config.initial_scale)))

        # Association independent link calibration: logit = alpha * sim + beta
        self.assoc_alpha = nn.Parameter(torch.tensor(6.0))
        self.assoc_beta = nn.Parameter(torch.tensor(0.0))

        # Deduction 3-way logical status prototypes in Pearson space:
        # (entailed, contradicted, undetermined)
        self.deduction_query_proj = nn.Linear(d_model * 2, d_proj, bias=False)
        self.deduction_prototypes = nn.Parameter(torch.randn(3, d_proj) * 0.1)

        # Metacognition 3 calibrated binary signals:
        # (correctness, evidence_sufficiency, context_benefit)
        self.meta_query_proj = nn.Linear(d_model * 2, d_proj, bias=False)
        self.meta_prototypes = nn.Parameter(torch.randn(3, d_proj) * 0.1)
        self.meta_bias = nn.Parameter(torch.zeros(3))

        # Abstraction multi-scale compression adapters (`brief`, `standard`, `detailed`)
        self.abstraction_gates = nn.ParameterDict(
            {
                "brief": nn.Parameter(torch.tensor(-0.5)),
                "standard": nn.Parameter(torch.tensor(0.0)),
                "detailed": nn.Parameter(torch.tensor(0.5)),
            }
        )
        self.abstraction_proj = nn.Linear(d_model, d_model, bias=False)
        self.abstraction_norm = RMSNorm(d_model, eps=backbone.config.rms_norm_eps)

        proj_std = 1.0 / math.sqrt(d_model)
        nn.init.normal_(self.state_projection.weight, std=proj_std)
        with torch.no_grad():
            self.output_projection.weight.copy_(self.state_projection.weight)
        nn.init.normal_(self.deduction_query_proj.weight, std=proj_std)
        nn.init.normal_(self.meta_query_proj.weight, std=proj_std)
        nn.init.normal_(self.abstraction_proj.weight, std=proj_std)

        # Persistent ContextLMCompressor soft gist queries & LayerNorm trained end-to-end
        self.gist_norm = nn.LayerNorm(d_model)
        self.gist_queries = nn.Parameter(
            torch.randn(self.config.gist_tokens_per_chunk, d_model) * 0.02
        )

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for opt_key, param in (
            ("gist_queries", self.gist_queries),
            ("gist_norm.weight", self.gist_norm.weight),
            ("gist_norm.bias", self.gist_norm.bias),
        ):
            full_key = prefix + opt_key
            if full_key not in state_dict:
                state_dict[full_key] = param.detach().clone()
            elif opt_key == "gist_queries":
                loaded_q = state_dict[full_key]
                is_corrupt = (
                    not bool(torch.isfinite(loaded_q).all().item())
                    or float(loaded_q.float().abs().max().item()) > 10.0
                )
                if loaded_q.shape != param.shape:
                    self.gist_queries = nn.Parameter(
                        (torch.randn_like(loaded_q, device=param.device, dtype=torch.float32) * 0.02).to(
                            param.dtype
                        )
                    )
                if is_corrupt:
                    state_dict[full_key] = self.gist_queries.detach().clone().to(
                        device=loaded_q.device, dtype=loaded_q.dtype
                    )
                    for k_sd, v_sd in list(state_dict.items()):
                        if isinstance(v_sd, torch.Tensor) and (
                            ".global_out_proj." in k_sd or ".ple.gate_scale" in k_sd
                        ):
                            state_dict[k_sd] = (v_sd.float() * 0.75).to(v_sd.dtype)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _similarity(self, left: Tensor, right: Tensor) -> Tensor:
        if self.config.metric == "pearson":
            return pearson_similarity(left, right, self.config.eps)
        with torch.autocast(device_type=left.device.type, enabled=False):
            if self.config.metric == "cosine":
                u = F.normalize(left.float(), dim=-1, eps=self.config.eps)
                v = F.normalize(right.float(), dim=-1, eps=self.config.eps)
                return (u * v).sum(-1).clamp(-1.0, 1.0)
            return (left.float() * right.float()).sum(-1) / math.sqrt(self.config.projection_dim)

    def _scale(self) -> Tensor:
        return self.log_scale.float().clamp(math.log(0.5), math.log(4.5)).exp()

    def _get_active_ncm_bridge(self) -> Any | None:
        ncm = getattr(self, "ncm_system", None)
        if ncm is None or not getattr(ncm.config, "enable_coe_head_bridge", True):
            return None
        return getattr(ncm, "coe_bridge", None)

    def _compute_candidate_lm_continuation_score(
        self,
        ctx_last_hidden: Tensor | None,
        cand_hidden: Tensor,
        canon_opt_ids: Tensor,
        canon_valid: Tensor,
        canon_owners: Tensor,
        batch_size: int,
    ) -> Tensor | None:
        """Compute standardized 16-layer causal continuation log-likelihood across options."""
        if (
            not self.backbone.config.preserve_residual_across_stacks
            or ctx_last_hidden is None
            or canon_opt_ids.shape[1] < 2
        ):
            return None
        target_tokens = canon_opt_ids[:, 1:].long()
        eff_eos_id = int(self.backbone.config.eos_id)
        tok_mask = (
            canon_valid[:, 1:]
            & (target_tokens != eff_eos_id)
            & (target_tokens != self.config.eos_id)
        )
        has_any = tok_mask.any(dim=-1, keepdim=True)
        tok_mask = torch.where(has_any, tok_mask, canon_valid[:, 1:])

        pred_h = torch.cat([ctx_last_hidden.unsqueeze(1), cand_hidden[:, 1:-1, :]], dim=1)
        lm_logits = self.backbone.lm_head(pred_h)
        with torch.autocast(device_type=pred_h.device.type, enabled=False):
            logits_f = lm_logits.float()
            if self.backbone.config.vocab_size == 128256 and logits_f.shape[-1] > 128000:
                logits_f = logits_f.clone()
                logits_f[:, :, 128000:] = -1e4
            token_nll = F.cross_entropy(
                logits_f.reshape(-1, logits_f.shape[-1]),
                target_tokens.reshape(-1),
                reduction="none",
            ).view_as(target_tokens)
            tok_mask_f = tok_mask.float()
            tok_counts = tok_mask_f.sum(dim=-1).clamp_min(1.0)
            seq_lp = -(token_nll * tok_mask_f).sum(dim=-1) / tok_counts

            owner_counts = (
                torch.zeros(batch_size, device=seq_lp.device, dtype=torch.float32)
                .scatter_add_(0, canon_owners, torch.ones_like(seq_lp))
                .clamp_min(1.0)
            )
            owner_mean = (
                torch.zeros(batch_size, device=seq_lp.device, dtype=torch.float32).scatter_add_(
                    0, canon_owners, seq_lp
                )
                / owner_counts
            )
            centered_lp = seq_lp - owner_mean.index_select(0, canon_owners)
            owner_var = (
                torch.zeros(batch_size, device=seq_lp.device, dtype=torch.float32).scatter_add_(
                    0, canon_owners, centered_lp.square()
                )
                / owner_counts
            )
            owner_std = owner_var.sqrt().clamp_min(0.25).index_select(0, canon_owners)
            lm_rel_score = torch.tanh((centered_lp / owner_std) * 0.65)
        return lm_rel_score.to(cand_hidden.dtype)

    @torch.no_grad()
    def _compute_joint_comparative_coe_score(
        self,
        trimmed_ctx_ids: Tensor,
        context_lengths: Tensor,
        option_ids: Tensor,
        option_lengths: Tensor,
        valid_options: Tensor,
        *,
        recurrence_steps: int = 2,
        depth_ponder: bool = True,
    ) -> Tensor | None:
        """Evaluate joint cross-candidate multi-option `COE` scratchpad scores (`A..H`) for BPE donors."""
        if (
            self.training
            or not self.backbone.config.preserve_residual_across_stacks
            or int(self.backbone.config.vocab_size) != 128256
        ):
            return None
        batch_size, num_opts, _ = option_ids.shape
        if num_opts < 2 or num_opts > 8:
            return None
        if num_opts == 2 and int(option_lengths.max().item()) <= 4:
            return None
        from .data import get_tokenizer

        bpe_tok = get_tokenizer(128256)
        bos_tok = int(bpe_tok.bos_id)
        letters = ("A", "B", "C", "D", "E", "F", "G", "H")
        letter_ids = [
            bpe_tok.encode(letters[k], add_bos=False, add_eos=False)[0] for k in range(num_opts)
        ]
        sp_letter_ids = [
            bpe_tok.encode(" " + letters[k], add_bos=False, add_eos=False)[0]
            for k in range(num_opts)
        ]
        p_rows: list[list[int]] = []
        p_last: list[int] = []
        row_tok_pairs: list[list[tuple[int, int]]] = []
        row_scale: list[float] = []
        for b in range(batch_size):
            c_len = int(context_lengths[b].item())
            raw_q = bpe_tok.decode(trimmed_ctx_ids[b, :c_len]).strip()
            if "#S$#" in raw_q:
                return None
            is_q_task = ("?" in raw_q) or ("Question:" in raw_q) or ("Subject:" in raw_q)
            q_str = raw_q
            if q_str.endswith("\nAnswer:"):
                q_str = q_str[: -len("\nAnswer:")].strip()
            elif q_str.endswith("Answer:"):
                q_str = q_str[: -len("Answer:")].strip()
            if q_str.startswith("Question:"):
                q_str = q_str[len("Question:") :].strip()

            opt_lines: list[str] = []
            tot_opt_toks = 0
            for k in range(num_opts):
                if not bool(valid_options[b, k].item()):
                    continue
                o_len = int(option_lengths[b, k].item())
                tot_opt_toks += o_len
                o_str = bpe_tok.decode(option_ids[b, k, :o_len]).strip()
                opt_lines.append(f"{letters[k]}. {o_str}")
            opts_block = "\n".join(opt_lines)
            mcqa_prompt = f"Question: {q_str}\n{opts_block}\nAnswer:"
            row_tok_pairs.append([(letter_ids[k], sp_letter_ids[k]) for k in range(num_opts)])
            avg_opt_len = tot_opt_toks / max(1, len(opt_lines))
            row_scale.append(0.15 if (not is_q_task and avg_opt_len > 15.0) else 1.0)

            p_ids = bpe_tok.encode(mcqa_prompt, add_bos=True, add_eos=False)
            if len(p_ids) > 384:
                p_ids = [bos_tok] + p_ids[-383:]
            p_rows.append(p_ids)
            p_last.append(len(p_ids) - 1)

        max_p = max(len(r) for r in p_rows)
        inp_t = torch.tensor(
            [r + [bpe_tok.pad_id] * (max_p - len(r)) for r in p_rows],
            dtype=torch.long,
            device=trimmed_ctx_ids.device,
        )
        inf_out = self.forward_inference(
            inp_t,
            recurrence_steps=max(1, int(recurrence_steps)),
            depth_ponder=bool(depth_ponder),
            condition_on_prefix_gist=bool(depth_ponder or recurrence_steps > 1),
            latent_prior_scale=0.05 if (depth_ponder or recurrence_steps > 1) else 0.0,
        )
        out_z = torch.zeros(
            (batch_size, num_opts), dtype=torch.float32, device=trimmed_ctx_ids.device
        )
        for b in range(batch_size):
            l_row = inf_out.logits[b, p_last[b]].float()
            k_v = int(valid_options[b].sum().item())
            if k_v < 2:
                continue
            t_pairs = row_tok_pairs[b]
            scores = torch.tensor(
                [
                    max(float(l_row[t_pairs[k][0]].item()), float(l_row[t_pairs[k][1]].item()))
                    for k in range(k_v)
                ],
                device=trimmed_ctx_ids.device,
                dtype=torch.float32,
            )
            s_z = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-4)
            out_z[b, :k_v] = s_z * row_scale[b]
        return out_z

    def forward_decision(
        self,
        context_ids: Tensor,
        context_lengths: Tensor,
        option_ids: Tensor,
        option_lengths: Tensor,
        targets: Tensor | None = None,
        *,
        recurrence_steps: int | None = None,
        depth_ponder: bool | None = None,
        margin_halt_threshold: float | None = None,
        whiteboard: QuarantinedWhiteboardState | None = None,
    ) -> DecisionOutput:
        """Evaluate `Decision` (`Choice` / `Noul` / `Score`) with CED cross-attention.

        - Encodes `context_ids` once per batch row in the lower CED encoder.
        - Canonically orders valid options before batched upper-CED decoding so
          output scores are 100% 0-ULP bitwise permutation-invariant (`torch.equal`).
        """
        if context_ids.ndim != 2 or min(context_ids.shape) < 1:
            raise ValueError("context_ids must have nonempty shape [batch, tokens]")
        batch = context_ids.shape[0]
        if context_lengths.shape != (batch,):
            raise ValueError("context_lengths must have shape [batch]")
        if option_ids.ndim != 3 or option_ids.shape[0] != batch or min(option_ids.shape[1:]) < 1:
            raise ValueError("option_ids must have nonempty shape [batch, options, tokens]")
        if option_lengths.shape != option_ids.shape[:2]:
            raise ValueError("option_lengths must have shape [batch, options]")
        valid_options = option_lengths > 0
        if (valid_options.sum(-1) < 2).any():
            raise ValueError("Each decision row requires at least two valid options")
        if targets is not None:
            if targets.shape != (batch,):
                raise ValueError("targets must have shape [batch]")
            if ((targets < 0) | (targets >= option_ids.shape[1])).any():
                raise ValueError("targets out of range")
            if not valid_options.gather(-1, targets.long().unsqueeze(-1)).all():
                raise ValueError("targets cannot select a masked option")

        use_ponder = self.config.depth_ponder if depth_ponder is None else depth_ponder
        halt_thresh = (
            self.config.margin_halt_threshold
            if margin_halt_threshold is None
            else margin_halt_threshold
        )
        if recurrence_steps is None:
            steps = (
                max(2, self.config.recurrence_steps)
                if (use_ponder or halt_thresh is not None)
                else self.config.recurrence_steps
            )
        else:
            steps = recurrence_steps
        ctx_width = int(context_lengths.max().item())
        trimmed_ctx_ids = context_ids[:, :ctx_width].long()
        ctx_valid = (
            torch.arange(ctx_width, device=context_ids.device)[None, :] < context_lengths[:, None]
        )

        # 1. Encode shared state/question once in the lower CED encoder layers.
        encoded_state: EncodedState = self.backbone.encode_state(
            trimmed_ctx_ids, ctx_valid, causal=True
        )
        state_hidden_for_pool = encoded_state.hidden
        state_for_decode = encoded_state
        s_global_rule_delta: Tensor | None = None
        if whiteboard is not None:
            mod_s = self.backbone.encoder_norm(
                whiteboard.apply_adapter(encoded_state.hidden, ctx_valid)
            ).masked_fill(~ctx_valid.unsqueeze(-1), 0)
            state_hidden_for_pool = 0.85 * encoded_state.hidden + 0.15 * mod_s
            state_for_decode = EncodedState(
                hidden=state_hidden_for_pool,
                valid=ctx_valid,
                aux_loss=encoded_state.aux_loss,
                router_z_loss=encoded_state.router_z_loss,
                expert_load=encoded_state.expert_load,
                gist_prefix_len=0,
                raw_hidden=encoded_state.raw_hidden,
                state_ids=encoded_state.state_ids,
            )
            s_ev_mask = self.pooler._state_evidence_mask(
                trimmed_ctx_ids,
                ctx_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            s_glob = self.pooler.state_norm(
                self.pooler.pool_sequence(
                    encoded_state.hidden, trimmed_ctx_ids, s_ev_mask, strip_boundary=False
                ).float()
            ).to(encoded_state.hidden.dtype)
            s_global_rule_delta = whiteboard.apply_rule_readout(
                s_glob,
                token_ids=trimmed_ctx_ids,
                valid_mask=s_ev_mask,
                token_embedding=self.backbone.token_embedding,
            )

        # 2. Extract valid options and sort into canonical order per question row
        #    so batched GEMM tiling is 100% invariant to option permutation.
        valid_lengths = option_lengths[valid_options]
        opt_width = int(valid_lengths.max().item())
        flat_opt_ids = option_ids[valid_options][:, :opt_width].long()
        owner_rows = valid_options.nonzero(as_tuple=False)[:, 0]
        perm, inv_perm = _canonical_candidate_order(owner_rows, valid_lengths, flat_opt_ids)

        canon_opt_ids = flat_opt_ids.index_select(0, perm)
        canon_lengths = valid_lengths.index_select(0, perm)
        canon_owners = owner_rows.index_select(0, perm)
        canon_valid = (
            torch.arange(opt_width, device=option_ids.device)[None, :] < canon_lengths[:, None]
        )

        decoded_options = self.backbone.decode_candidates(
            state_for_decode,
            canon_opt_ids,
            canon_valid,
            owner_rows=canon_owners,
            recurrence_steps=steps,
            causal_local=True,
            depth_ponder=use_ponder,
        )
        has_active_rule = (
            whiteboard is not None
            and float(torch.linalg.matrix_norm(whiteboard.rule_val_B.float(), ord="fro").item()) > 1e-6
        )
        dec_opt_hidden = (
            whiteboard.apply_adapter(decoded_options.hidden, canon_valid)
            if (whiteboard is not None and not has_active_rule)
            else decoded_options.hidden
        )
        cand_anchor_for_pool = (
            whiteboard.apply_adapter(decoded_options.anchor, canon_valid)
            if (whiteboard is not None and decoded_options.anchor is not None and not has_active_rule)
            else decoded_options.anchor
        )

        # 3. Token-level QK-Norm matched-filter pooling + centered Pearson projection.
        paired_state_repr, cand_repr = self.pooler.matched_pair_representations(
            state_hidden_for_pool,
            trimmed_ctx_ids,
            ctx_valid,
            dec_opt_hidden,
            canon_opt_ids,
            canon_valid,
            canon_owners,
            cand_anchor=cand_anchor_for_pool,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        ctx_l15_norm: Tensor | None = None
        if decoded_options.ctx_last_hidden is not None:
            ctx_l15_norm = self.pooler.state_norm(
                decoded_options.ctx_last_hidden.float()
            ).to(paired_state_repr.dtype)
            paired_state_repr = 0.55 * paired_state_repr + 0.45 * ctx_l15_norm
        lm_cont_score = self._compute_candidate_lm_continuation_score(
            decoded_options.ctx_last_hidden,
            dec_opt_hidden,
            canon_opt_ids,
            canon_valid,
            canon_owners,
            batch,
        )
        rule_state_proj_rows: Tensor | None = None
        rule_cand_proj: Tensor | None = None
        r_scaled: Tensor | None = None
        w_rule = 0.0
        if whiteboard is not None:
            paired_state_repr = whiteboard.apply_adapter(paired_state_repr)
            if s_global_rule_delta is not None and has_active_rule:
                w_rule = 0.88
                r_cent = s_global_rule_delta.float() - s_global_rule_delta.float().mean(
                    dim=-1, keepdim=True
                )
                r_unit = F.normalize(r_cent, dim=-1, eps=1e-6)
                r_scaled = (
                    r_unit * math.sqrt(float(self.backbone.config.d_model))
                ).to(paired_state_repr.dtype)
                rule_state_proj_rows = self.state_projection(r_scaled).index_select(
                    0, canon_owners
                )
                paired_state_repr = (
                    0.25 * paired_state_repr
                    + 0.75 * r_scaled.index_select(0, canon_owners)
                )
                anc_src = (
                    decoded_options.anchor
                    if decoded_options.anchor is not None
                    else dec_opt_hidden
                )
                c_anc_norm = self.pooler.cand_norm(
                    self.pooler.pool_sequence(
                        anc_src,
                        canon_opt_ids,
                        canon_valid,
                        strip_boundary=self.config.strip_boundary_tokens_in_pool,
                        bos_id=self.config.bos_id,
                        eos_id=self.config.eos_id,
                    ).float()
                ).to(cand_repr.dtype)
                rule_cand_proj = self.output_projection(c_anc_norm)
            elif not has_active_rule:
                cand_repr = whiteboard.apply_adapter(cand_repr)
        canon_state_proj = self.state_projection(paired_state_repr)
        canon_cand_proj = self.output_projection(cand_repr)
        canon_scores = self._similarity(canon_state_proj, canon_cand_proj)
        rule_sim: Tensor | None = None
        if (
            whiteboard is not None
            and s_global_rule_delta is not None
            and rule_state_proj_rows is not None
            and rule_cand_proj is not None
            and w_rule > 0.0
        ):
            r_sim_ctx = self._similarity(rule_state_proj_rows, rule_cand_proj)
            cand_ev_mask = self.pooler._content_mask(
                canon_opt_ids,
                canon_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            r_sim_holo = whiteboard.score_holo_candidates(
                s_glob,
                trimmed_ctx_ids,
                s_ev_mask,
                canon_opt_ids,
                cand_ev_mask,
                canon_owners,
                self.backbone.token_embedding,
            )
            if r_sim_holo is not None:
                rule_sim = torch.tanh(
                    (0.55 * r_sim_ctx + 0.45 * r_sim_holo.to(r_sim_ctx.dtype)) * 3.5
                )
            else:
                rule_sim = torch.tanh(r_sim_ctx * 3.5)
            canon_scores = (1.0 - w_rule) * canon_scores + w_rule * rule_sim

        similarities_step1: Tensor | None = None
        if (use_ponder or halt_thresh is not None) and len(decoded_options.step_states) > 1:
            ps_1, cr_1 = self.pooler.matched_pair_representations(
                state_hidden_for_pool,
                trimmed_ctx_ids,
                ctx_valid,
                decoded_options.step_states[0],
                canon_opt_ids,
                canon_valid,
                canon_owners,
                cand_anchor=cand_anchor_for_pool,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            if whiteboard is not None:
                ps_1 = whiteboard.apply_adapter(ps_1)
                if r_scaled is not None and w_rule > 0.0:
                    ps_1 = 0.25 * ps_1 + 0.75 * r_scaled.index_select(0, canon_owners)
                elif not has_active_rule:
                    cr_1 = whiteboard.apply_adapter(cr_1)
            cs_1 = self._similarity(
                self.state_projection(ps_1),
                self.output_projection(cr_1),
            )
            if rule_sim is not None and w_rule > 0.0:
                cs_1 = (1.0 - w_rule) * cs_1 + w_rule * rule_sim
            v_scores_1 = cs_1.index_select(0, inv_perm)
            similarities_step1 = v_scores_1.new_zeros(valid_options.shape).masked_scatter(
                valid_options, v_scores_1
            )
            if use_ponder and halt_thresh is None:
                canon_scores = 0.70 * cs_1 + 0.30 * canon_scores
            paired_state_repr = 0.70 * ps_1 + 0.30 * paired_state_repr
            cand_repr = 0.70 * cr_1 + 0.30 * cand_repr
            canon_state_proj = self.state_projection(paired_state_repr)
            canon_cand_proj = self.output_projection(cand_repr)

        ncm_bridge = self._get_active_ncm_bridge()
        if ncm_bridge is not None:
            s_ncm_mask = self.pooler._state_evidence_mask(
                trimmed_ctx_ids,
                ctx_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            c_ncm_mask = self.pooler._content_mask(
                canon_opt_ids,
                canon_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            canon_scores, canon_state_proj, canon_cand_proj, _, _ = (
                ncm_bridge.enhance_decision_and_eval(
                    trimmed_ctx_ids,
                    s_ncm_mask,
                    canon_opt_ids,
                    c_ncm_mask,
                    canon_owners,
                    canon_state_proj,
                    canon_cand_proj,
                    canon_scores,
                    self._similarity,
                )
            )
            prior_w = float(getattr(self.ncm_system.config, "coe_decision_prior_weight", 0.0))
            if prior_w > 0.0:
                hier_lp, _ = self.ncm_system.compute_hierarchy_log_probs(
                    state_for_decode,
                    canon_opt_ids,
                    owner_rows=canon_owners,
                    full_length=int(canon_opt_ids.shape[1]),
                    normalize_by_tokens=True,
                )
                prior_score = torch.tanh(hier_lp / 10.0).to(canon_scores.dtype)
                canon_scores = (1.0 - prior_w) * canon_scores + prior_w * prior_score

        if lm_cont_score is not None:
            w_lm = 0.15 if (whiteboard is not None and w_rule > 0.0) else 0.78
            canon_scores = (1.0 - w_lm) * canon_scores + w_lm * lm_cont_score
            if similarities_step1 is not None:
                lm_step1 = lm_cont_score.index_select(0, inv_perm)
                lm_grid = lm_step1.new_zeros(valid_options.shape).masked_scatter(
                    valid_options, lm_step1
                )
                similarities_step1 = (1.0 - w_lm) * similarities_step1 + w_lm * lm_grid

        # Restore original caller option order via inv_perm
        valid_scores = canon_scores.index_select(0, inv_perm)
        state_proj = canon_state_proj.index_select(0, inv_perm)
        cand_proj = canon_cand_proj.index_select(0, inv_perm)

        similarities = valid_scores.new_zeros(valid_options.shape).masked_scatter(
            valid_options, valid_scores
        )
        joint_mcqa_z = self._compute_joint_comparative_coe_score(
            trimmed_ctx_ids,
            context_lengths,
            option_ids,
            option_lengths,
            valid_options,
            recurrence_steps=steps,
            depth_ponder=use_ponder,
        )
        if joint_mcqa_z is not None:
            with torch.autocast(device_type=context_ids.device.type, enabled=False):
                sims_f = similarities.float().masked_fill(~valid_options, 0.0)
                v_cnt = valid_options.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
                sims_m = sims_f.sum(dim=-1, keepdim=True) / v_cnt
                sims_var = (
                    ((sims_f - sims_m).masked_fill(~valid_options, 0.0) ** 2).sum(
                        dim=-1, keepdim=True
                    )
                    / v_cnt
                ).clamp_min(1e-4)
                sims_z = (sims_f - sims_m) / sims_var.sqrt()
                similarities = (
                    torch.tanh((0.70 * joint_mcqa_z + 0.30 * sims_z) * 0.65)
                    .masked_fill(~valid_options, -1.0)
                    .to(similarities.dtype)
                )

        with torch.autocast(device_type=context_ids.device.type, enabled=False):
            scale = self._scale()
            if halt_thresh is not None and similarities_step1 is not None:
                logits_1 = (similarities_step1.float() * scale).masked_fill(
                    ~valid_options, -torch.inf
                )
                sorted_logits_1, _ = logits_1.sort(dim=-1, descending=True)
                sorted_probs_1 = sorted_logits_1.softmax(dim=-1)
                margin_1 = (sorted_probs_1[:, 0] - sorted_probs_1[:, 1]).unsqueeze(-1)
                blended_sims = 0.70 * similarities_step1 + 0.30 * similarities
                similarities = torch.where(
                    margin_1 >= float(halt_thresh),
                    similarities_step1,
                    blended_sims,
                )
            logits = (similarities.float() * scale).masked_fill(~valid_options, -torch.inf)
            probabilities = logits.softmax(dim=-1)
            aux_loss = 0.5 * (encoded_state.aux_loss.float() + decoded_options.aux_loss.float())
            router_z_loss = 0.5 * (
                encoded_state.router_z_loss.float() + decoded_options.router_z_loss.float()
            )
            ce_loss, loss = None, None
            if targets is not None:
                ce_loss = F.cross_entropy(logits, targets.long())
                loss = (
                    ce_loss
                    + self.backbone.config.aux_loss_coef * aux_loss
                    + self.backbone.config.router_z_loss_coef * router_z_loss
                )

        return DecisionOutput(
            logits=logits,
            similarities=similarities,
            probabilities=probabilities,
            state_features=state_proj,
            candidate_features=cand_proj,
            loss=loss,
            ce_loss=ce_loss,
            aux_loss=aux_loss,
            router_z_loss=router_z_loss,
        )

    forward = forward_decision

    def forward_all_operators(
        self,
        context_ids: Tensor,
        context_lengths: Tensor,
        option_ids: Tensor,
        option_lengths: Tensor,
        targets: Tensor,
        *,
        recurrence_steps: int | None = None,
        depth_ponder: bool | None = None,
        margin_halt_threshold: float | None = None,
        candidate_encoded: EncodedState | None = None,
    ) -> AllOperatorsOutput:
        """Train all COE operators in a single shared `encode_state` + `decode_candidates` pass.

        Reuses the canonicalized matched-filter representations `(paired_state_repr, cand_repr)`
        from `Decision` to simultaneously supervise:
        1. `Decision` / `Abduction` / `Induction` (option cross-entropy over Pearson similarities)
        2. `Deduction` (3-way `entailed=0`, `contradicted=1`, `undetermined=2` over positive,
           hardest-negative, second-negative, and cross-row unrelated state-candidate pairs)
        3. `Evaluation` & `Association` (`assoc_alpha`, `assoc_beta` utility calibration in `[0, 1]`)
        4. `Metacognition` (`correctness`, `evidence_sufficiency`, `context_benefit` across
           entailed `[1, 1, 1]`, contradicted `[0, 1, 0.5]`, and undetermined `[0, 0, 0]` pairs)
        5. `Abstraction` (`brief` / `standard` / `detailed` multi-scale state compression alignment)
        """
        batch = context_ids.shape[0]
        valid_options = option_lengths > 0
        use_ponder = self.config.depth_ponder if depth_ponder is None else depth_ponder
        halt_thresh = (
            self.config.margin_halt_threshold
            if margin_halt_threshold is None
            else margin_halt_threshold
        )
        if recurrence_steps is None:
            steps = (
                max(2, self.config.recurrence_steps)
                if (use_ponder or halt_thresh is not None)
                else self.config.recurrence_steps
            )
        else:
            steps = recurrence_steps
        ctx_width = int(context_lengths.max().item())
        trimmed_ctx_ids = context_ids[:, :ctx_width].long()
        ctx_valid = (
            torch.arange(ctx_width, device=context_ids.device)[None, :] < context_lengths[:, None]
        )

        # 1. Shared lower-CED state encoding + canonicalized upper-CED candidate decoding
        encoded_state: EncodedState = self.backbone.encode_state(
            trimmed_ctx_ids, ctx_valid, causal=True
        )
        valid_lengths = option_lengths[valid_options]
        opt_width = int(valid_lengths.max().item())
        flat_opt_ids = option_ids[valid_options][:, :opt_width].long()
        owner_rows = valid_options.nonzero(as_tuple=False)[:, 0]
        perm, inv_perm = _canonical_candidate_order(owner_rows, valid_lengths, flat_opt_ids)

        canon_opt_ids = flat_opt_ids.index_select(0, perm)
        canon_lengths = valid_lengths.index_select(0, perm)
        canon_owners = owner_rows.index_select(0, perm)
        canon_valid = (
            torch.arange(opt_width, device=option_ids.device)[None, :] < canon_lengths[:, None]
        )

        decoded_options = self.backbone.decode_candidates(
            encoded_state,
            canon_opt_ids,
            canon_valid,
            owner_rows=canon_owners,
            recurrence_steps=steps,
            causal_local=True,
            candidate_encoded=candidate_encoded,
            depth_ponder=use_ponder,
        )
        paired_state_repr, cand_repr = self.pooler.matched_pair_representations(
            encoded_state.hidden,
            trimmed_ctx_ids,
            ctx_valid,
            decoded_options.hidden,
            canon_opt_ids,
            canon_valid,
            canon_owners,
            cand_anchor=decoded_options.anchor,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        if decoded_options.ctx_last_hidden is not None:
            ctx_l15_norm = self.pooler.state_norm(
                decoded_options.ctx_last_hidden.float()
            ).to(paired_state_repr.dtype)
            paired_state_repr = 0.55 * paired_state_repr + 0.45 * ctx_l15_norm
        lm_cont_score = self._compute_candidate_lm_continuation_score(
            decoded_options.ctx_last_hidden,
            decoded_options.hidden,
            canon_opt_ids,
            canon_valid,
            canon_owners,
            batch,
        )

        state_in = F.dropout(paired_state_repr, p=0.10, training=self.training)
        cand_in = F.dropout(cand_repr, p=0.10, training=self.training)
        canon_state_proj = self.state_projection(state_in)
        canon_cand_proj = self.output_projection(cand_in)
        canon_scores = self._similarity(canon_state_proj, canon_cand_proj)

        if (use_ponder or halt_thresh is not None) and len(decoded_options.step_states) > 1:
            ps_1, cr_1 = self.pooler.matched_pair_representations(
                encoded_state.hidden,
                trimmed_ctx_ids,
                ctx_valid,
                decoded_options.step_states[0],
                canon_opt_ids,
                canon_valid,
                canon_owners,
                cand_anchor=decoded_options.anchor,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            cs_1 = self._similarity(
                self.state_projection(F.dropout(ps_1, p=0.10, training=self.training)),
                self.output_projection(F.dropout(cr_1, p=0.10, training=self.training)),
            )
            if halt_thresh is not None:
                v_s1 = cs_1.index_select(0, inv_perm)
                sims_1 = v_s1.new_zeros(valid_options.shape).masked_scatter(valid_options, v_s1)
                with torch.autocast(device_type=context_ids.device.type, enabled=False):
                    sc = self._scale()
                    sorted_l1, _ = (
                        (sims_1.float() * sc).masked_fill(~valid_options, -torch.inf)
                    ).sort(dim=-1, descending=True)
                    sorted_p1 = sorted_l1.softmax(dim=-1)
                    m1 = (sorted_p1[:, 0] - sorted_p1[:, 1]).index_select(0, canon_owners)
                canon_scores = torch.where(
                    m1 >= float(halt_thresh),
                    cs_1,
                    0.70 * cs_1 + 0.30 * canon_scores,
                )
            elif use_ponder:
                canon_scores = 0.70 * cs_1 + 0.30 * canon_scores
            paired_state_repr = 0.70 * ps_1 + 0.30 * paired_state_repr
            cand_repr = 0.70 * cr_1 + 0.30 * cand_repr
            canon_state_proj = self.state_projection(
                F.dropout(paired_state_repr, p=0.10, training=self.training)
            )
            canon_cand_proj = self.output_projection(
                F.dropout(cand_repr, p=0.10, training=self.training)
            )

        ncm_bridge = self._get_active_ncm_bridge()
        state_ms = None
        cand_ms = None
        if ncm_bridge is not None:
            s_ncm_mask = self.pooler._state_evidence_mask(
                trimmed_ctx_ids,
                ctx_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            c_ncm_mask = self.pooler._content_mask(
                canon_opt_ids,
                canon_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            canon_scores, canon_state_proj, canon_cand_proj, state_ms, cand_ms = (
                ncm_bridge.enhance_decision_and_eval(
                    trimmed_ctx_ids,
                    s_ncm_mask,
                    canon_opt_ids,
                    c_ncm_mask,
                    canon_owners,
                    canon_state_proj,
                    canon_cand_proj,
                    canon_scores,
                    self._similarity,
                )
            )

        if lm_cont_score is not None:
            canon_scores = 0.22 * canon_scores + 0.78 * lm_cont_score

        valid_scores = canon_scores.index_select(0, inv_perm)
        state_proj = canon_state_proj.index_select(0, inv_perm)
        cand_proj = canon_cand_proj.index_select(0, inv_perm)
        state_repr = paired_state_repr.index_select(0, inv_perm)
        opt_repr = cand_repr.index_select(0, inv_perm)

        similarities = valid_scores.new_zeros(valid_options.shape).masked_scatter(
            valid_options, valid_scores
        )
        joint_mcqa_z = self._compute_joint_comparative_coe_score(
            trimmed_ctx_ids,
            context_lengths,
            option_ids,
            option_lengths,
            valid_options,
            recurrence_steps=steps,
            depth_ponder=use_ponder,
        )
        if joint_mcqa_z is not None:
            with torch.autocast(device_type=context_ids.device.type, enabled=False):
                sims_f = similarities.float().masked_fill(~valid_options, 0.0)
                v_cnt = valid_options.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
                sims_m = sims_f.sum(dim=-1, keepdim=True) / v_cnt
                sims_var = (
                    ((sims_f - sims_m).masked_fill(~valid_options, 0.0) ** 2).sum(
                        dim=-1, keepdim=True
                    )
                    / v_cnt
                ).clamp_min(1e-4)
                sims_z = (sims_f - sims_m) / sims_var.sqrt()
                similarities = (
                    torch.tanh((0.70 * joint_mcqa_z + 0.30 * sims_z) * 0.65)
                    .masked_fill(~valid_options, -1.0)
                    .to(similarities.dtype)
                )

        # Scatter [R, d_model] representations into [B, M, d_model] grids for multi-operator readout
        d_m = state_repr.shape[-1]
        num_opts = option_ids.shape[1]
        mask_3d = valid_options.unsqueeze(-1).expand(batch, num_opts, d_m)
        state_grid = state_repr.new_zeros((batch, num_opts, d_m)).masked_scatter(mask_3d, state_repr)
        opt_grid = opt_repr.new_zeros((batch, num_opts, d_m)).masked_scatter(mask_3d, opt_repr)

        b_rows = torch.arange(batch, device=context_ids.device)
        t_idx = targets.long()
        s_pos = state_grid[b_rows, t_idx]
        c_pos = opt_grid[b_rows, t_idx]

        distractor_mask = valid_options.clone()
        distractor_mask[b_rows, t_idx] = False
        neg_scores = similarities.float().masked_fill(~distractor_mask, -torch.inf)
        d_idx = neg_scores.argmax(dim=-1)
        s_neg = state_grid[b_rows, d_idx]
        c_neg = opt_grid[b_rows, d_idx]

        # Also select the second-highest (or alternate) valid distractor so Deduction and
        # Metacognition learn contradiction across all distractors rather than a single pair
        distractor_mask_2 = distractor_mask.clone()
        distractor_mask_2[b_rows, d_idx] = False
        has_second = distractor_mask_2.any(dim=-1)
        neg_scores_2 = similarities.float().masked_fill(~distractor_mask_2, -torch.inf)
        d_idx2 = torch.where(has_second, neg_scores_2.argmax(dim=-1), d_idx)
        s_neg2 = state_grid[b_rows, d_idx2]
        c_neg2 = opt_grid[b_rows, d_idx2]

        # Cross-row mismatched pair represents `undetermined` (unrelated state and candidate)
        s_und = s_pos
        c_und = torch.roll(c_pos, shifts=1, dims=0) if batch > 1 else (0.5 * (c_pos + c_neg))

        s_all4 = torch.cat([s_pos, s_neg, s_neg2, s_und], dim=0)
        c_all4 = torch.cat([c_pos, c_neg, c_neg2, c_und], dim=0)
        joint_pair = F.dropout(torch.cat([s_all4, c_all4], dim=-1), p=0.10, training=self.training)

        ded_joint = self.deduction_query_proj(joint_pair)
        meta_joint = self.meta_query_proj(joint_pair)

        if ncm_bridge is not None and state_ms is not None and cand_ms is not None:
            def _scatter_cand_feat(feat_canon: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                f_caller = feat_canon.index_select(0, inv_perm)
                dim_f = f_caller.shape[-1]
                m_f = valid_options.unsqueeze(-1).expand(batch, num_opts, dim_f)
                grid_f = f_caller.new_zeros((batch, num_opts, dim_f)).masked_scatter(m_f, f_caller)
                f_pos = grid_f[b_rows, t_idx]
                f_neg = grid_f[b_rows, d_idx]
                f_neg2 = grid_f[b_rows, d_idx2]
                f_und = torch.roll(f_pos, shifts=1, dims=0) if batch > 1 else (0.5 * (f_pos + f_neg))
                return f_pos, f_neg, f_neg2, f_und

            c_zc_4 = torch.cat(_scatter_cand_feat(cand_ms.z_coarse_pool), dim=0)
            c_zr_4 = torch.cat(_scatter_cand_feat(cand_ms.z_res_fine_pool), dim=0)
            c_ent_4 = torch.cat(_scatter_cand_feat(cand_ms.scale_entropy), dim=0)
            c_cons_4 = torch.cat(_scatter_cand_feat(cand_ms.scale_consistency), dim=0)

            s_zc_4 = state_ms.z_coarse_pool.repeat(4, 1)
            s_zr_4 = state_ms.z_res_fine_pool.repeat(4, 1)
            s_ent_4 = state_ms.scale_entropy.repeat(4, 1)
            s_cons_4 = state_ms.scale_consistency.repeat(4, 1)

            ded_joint = ncm_bridge.enhance_deduction_query(
                ded_joint, s_zc_4, c_zc_4, s_zr_4, c_zr_4
            )
            meta_joint = ncm_bridge.enhance_metacognition_query(
                meta_joint, s_zc_4, c_zc_4, c_zr_4, s_ent_4, c_ent_4, s_cons_4, c_cons_4
            )

        pooled_s = self.pooler.pool_sequence(
            encoded_state.hidden,
            trimmed_ctx_ids,
            ctx_valid,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        pos_cand_proj = self.output_projection(c_pos)
        if ncm_bridge is not None:
            abs_sims = []
            for lvl_name in ("brief", "standard", "detailed"):
                abs_vec_l = self.compress_abstraction(
                    trimmed_ctx_ids,
                    context_lengths,
                    level=lvl_name,
                    encoded_hidden=encoded_state.hidden,
                )
                abs_sims.append(self._similarity(self.state_projection(abs_vec_l), pos_cand_proj))
            abs_sim = torch.stack(abs_sims, dim=0).mean(dim=0)
        else:
            gate_mean = torch.stack([g.sigmoid() for g in self.abstraction_gates.values()]).mean()
            abs_state = self.abstraction_norm(pooled_s + gate_mean * self.abstraction_proj(pooled_s))
            abs_proj = self.state_projection(abs_state)
            abs_sim = self._similarity(abs_proj, pos_cand_proj)

        with torch.autocast(device_type=context_ids.device.type, enabled=False):
            scale = self._scale()
            logits = (similarities.float() * scale).masked_fill(~valid_options, -torch.inf)
            probabilities = logits.softmax(dim=-1)
            aux_loss = 0.5 * (encoded_state.aux_loss.float() + decoded_options.aux_loss.float())
            router_z_loss = 0.5 * (
                encoded_state.router_z_loss.float() + decoded_options.router_z_loss.float()
            )
            log_probs = logits.log_softmax(dim=-1)
            nll = -log_probs.gather(-1, t_idx.unsqueeze(-1)).squeeze(-1)
            if self.training:
                safe_lp = log_probs.masked_fill(~valid_options, 0.0)
                smooth_nll = -safe_lp.sum(dim=-1) / valid_options.float().sum(dim=-1).clamp_min(1.0)
                ce_loss = ((1.0 - 0.05) * nll + 0.05 * smooth_nll).mean()
            else:
                ce_loss = nll.mean()
            dec_loss = (
                ce_loss
                + self.backbone.config.aux_loss_coef * aux_loss
                + self.backbone.config.router_z_loss_coef * router_z_loss
            )

            # 2. Deduction 3-way loss (`entailed=0`, `contradicted=1`, `undetermined=2`)
            ded_sims = pearson_similarity(
                ded_joint.unsqueeze(1), self.deduction_prototypes.unsqueeze(0), eps=self.config.eps
            )
            ded_logits = ded_sims.float() * scale
            ded_probs = ded_logits.softmax(dim=-1)
            ded_targets = torch.cat(
                [
                    torch.zeros(batch, device=context_ids.device, dtype=torch.long),
                    torch.ones(batch, device=context_ids.device, dtype=torch.long),
                    torch.ones(batch, device=context_ids.device, dtype=torch.long),
                    torch.full((batch,), 2, device=context_ids.device, dtype=torch.long),
                ],
                dim=0,
            )
            deduction_loss = F.cross_entropy(
                ded_logits, ded_targets, label_smoothing=(0.04 if self.training else 0.0)
            )

            # 3. Evaluation & Association utility loss in [0, 1]
            eval_logits = valid_scores.float() * self.assoc_alpha.float() + self.assoc_beta.float()
            eval_utilities = eval_logits.sigmoid()
            target_grid = F.one_hot(t_idx, num_classes=num_opts).float()
            eval_targets = target_grid[valid_options]
            if self.training:
                eval_targets = eval_targets * 0.92 + 0.04
            evaluation_loss = F.binary_cross_entropy_with_logits(eval_logits, eval_targets)

            # 4. Metacognition 3-signal loss (`correctness`, `evidence_sufficiency`, `context_benefit`)
            meta_sims = pearson_similarity(
                meta_joint.unsqueeze(1), self.meta_prototypes.unsqueeze(0), eps=self.config.eps
            )
            meta_logits = meta_sims.float() * scale + self.meta_bias.float().unsqueeze(0)
            meta_probs = meta_logits.sigmoid()
            t_pos_meta = torch.tensor(
                [1.0, 1.0, 1.0], device=context_ids.device, dtype=torch.float32
            ).expand(batch, 3)
            t_neg_meta = torch.tensor(
                [0.0, 1.0, 0.5], device=context_ids.device, dtype=torch.float32
            ).expand(batch, 3)
            t_und_meta = torch.tensor(
                [0.0, 0.0, 0.0], device=context_ids.device, dtype=torch.float32
            ).expand(batch, 3)
            meta_targets = torch.cat([t_pos_meta, t_neg_meta, t_neg_meta, t_und_meta], dim=0)
            if self.training:
                meta_targets = meta_targets * 0.92 + 0.04
            metacognition_loss = F.binary_cross_entropy_with_logits(meta_logits, meta_targets)

            # 5. Abstraction alignment loss
            abstraction_loss = (1.0 - abs_sim.float()).mean()

            total_op_loss = (
                dec_loss
                + 0.30 * deduction_loss
                + 0.25 * evaluation_loss
                + 0.25 * metacognition_loss
                + 0.10 * abstraction_loss
            )

        decision_out = DecisionOutput(
            logits=logits,
            similarities=similarities,
            probabilities=probabilities,
            state_features=state_proj,
            candidate_features=cand_proj,
            loss=dec_loss,
            ce_loss=ce_loss,
            aux_loss=aux_loss,
            router_z_loss=router_z_loss,
        )
        return AllOperatorsOutput(
            decision=decision_out,
            deduction_loss=deduction_loss,
            evaluation_loss=evaluation_loss,
            metacognition_loss=metacognition_loss,
            abstraction_loss=abstraction_loss,
            total_loss=total_op_loss,
            deduction_probs=ded_probs[:batch],
            metacognition_probs=meta_probs[:batch],
            evaluation_utilities=eval_utilities,
        )

    def embed_items(self, item_ids: Tensor, item_lengths: Tensor) -> Tensor:
        """Project a batch of items `[N, T]` into 32-dim centered Pearson vectors."""
        width = int(item_lengths.max().item())
        trimmed = item_ids[:, :width].long()
        valid = torch.arange(width, device=item_ids.device)[None, :] < item_lengths[:, None]
        encoded = self.backbone.encode_state(trimmed, valid, causal=True)
        pooled = self.pooler.pool_sequence(
            encoded.hidden,
            trimmed,
            valid,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        normed = self.pooler.cand_norm(pooled.float()).to(pooled.dtype)
        proj = self.output_projection(normed).float()
        ncm_bridge = self._get_active_ncm_bridge()
        if ncm_bridge is not None:
            c_mask = self.pooler._content_mask(
                trimmed,
                valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            proj = ncm_bridge.enhance_item_embedding(trimmed, c_mask, proj)
        centered = proj - proj.mean(dim=-1, keepdim=True)
        return F.normalize(centered, dim=-1, eps=self.config.eps)

    def forward_association_matrix(
        self,
        item_ids: Tensor,
        item_lengths: Tensor,
        *,
        history_buffer: Tensor | None = None,
    ) -> AssociationMatrixOutput:
        """Compute pairwise Association matrix and optional <0.1ms Ornith-1.5 Novelty N(q)."""
        features = self.embed_items(item_ids, item_lengths)
        with torch.autocast(device_type=features.device.type, enabled=False):
            similarities = (features @ features.transpose(-1, -2)).clamp(-1.0, 1.0)
            logits = self.assoc_alpha.float() * similarities + self.assoc_beta.float()
            probabilities = logits.sigmoid()
            novelty = None
            if history_buffer is not None and history_buffer.numel() > 0:
                buf = F.normalize(
                    history_buffer.float() - history_buffer.float().mean(-1, keepdim=True),
                    dim=-1,
                    eps=self.config.eps,
                )
                max_sim = (features @ buf.transpose(-1, -2)).clamp(-1.0, 1.0).max(dim=-1).values
                novelty = (1.0 - max_sim).clamp(0.0, 2.0)
        return AssociationMatrixOutput(
            similarities=similarities,
            logits=logits,
            probabilities=probabilities,
            features=features,
            novelty=novelty,
        )

    def forward_deduction(
        self,
        premise_ids: Tensor,
        premise_lengths: Tensor,
        conclusion_ids: Tensor,
        conclusion_lengths: Tensor,
        targets: Tensor | None = None,
    ) -> DeductionOutput:
        """Predict 3-way logical status (`entailed=0`, `contradicted=1`, `undetermined=2`)."""
        batch = premise_ids.shape[0]
        p_width = int(premise_lengths.max().item())
        c_width = int(conclusion_lengths.max().item())
        p_trim = premise_ids[:, :p_width].long()
        c_trim = conclusion_ids[:, :c_width].long()
        p_valid = torch.arange(p_width, device=premise_ids.device)[None, :] < premise_lengths[:, None]
        c_valid = (
            torch.arange(c_width, device=conclusion_ids.device)[None, :]
            < conclusion_lengths[:, None]
        )

        encoded_p = self.backbone.encode_state(p_trim, p_valid, causal=True)
        owner_rows = torch.arange(batch, device=premise_ids.device)
        decoded_c = self.backbone.decode_candidates(
            encoded_p,
            c_trim,
            c_valid,
            owner_rows=owner_rows,
            recurrence_steps=self.config.recurrence_steps,
        )
        p_repr, c_repr = self.pooler.matched_pair_representations(
            encoded_p.hidden,
            p_trim,
            p_valid,
            decoded_c.hidden,
            c_trim,
            c_valid,
            owner_rows,
            cand_anchor=decoded_c.anchor,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        joint = self.deduction_query_proj(torch.cat([p_repr, c_repr], dim=-1))
        ncm_bridge = self._get_active_ncm_bridge()
        if ncm_bridge is not None:
            p_mask = self.pooler._state_evidence_mask(
                p_trim,
                p_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            c_mask = self.pooler._content_mask(
                c_trim,
                c_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            p_ms = ncm_bridge.extract_features(p_trim, p_mask)
            c_ms = ncm_bridge.extract_features(c_trim, c_mask)
            joint = ncm_bridge.enhance_deduction_query(
                joint,
                p_ms.z_coarse_pool,
                c_ms.z_coarse_pool,
                p_ms.z_res_fine_pool,
                c_ms.z_res_fine_pool,
            )
        sims = pearson_similarity(
            joint.unsqueeze(1), self.deduction_prototypes.unsqueeze(0), eps=self.config.eps
        )
        logits = sims * self._scale()
        probs = logits.softmax(dim=-1)
        loss = F.cross_entropy(logits, targets.long()) if targets is not None else None
        return DeductionOutput(logits=logits, probabilities=probs, similarities=sims, loss=loss)

    def forward_metacognition(
        self,
        state_ids: Tensor,
        state_lengths: Tensor,
        candidate_ids: Tensor,
        candidate_lengths: Tensor,
        targets: Tensor | None = None,
    ) -> MetacognitionOutput:
        """Evaluate calibrated reliability signals (`correctness`, `evidence_sufficiency`, `context_benefit`)."""
        batch = state_ids.shape[0]
        s_width = int(state_lengths.max().item())
        c_width = int(candidate_lengths.max().item())
        s_trim = state_ids[:, :s_width].long()
        c_trim = candidate_ids[:, :c_width].long()
        s_valid = torch.arange(s_width, device=state_ids.device)[None, :] < state_lengths[:, None]
        c_valid = (
            torch.arange(c_width, device=candidate_ids.device)[None, :]
            < candidate_lengths[:, None]
        )

        encoded_s = self.backbone.encode_state(s_trim, s_valid, causal=True)
        owner_rows = torch.arange(batch, device=state_ids.device)
        decoded_c = self.backbone.decode_candidates(
            encoded_s,
            c_trim,
            c_valid,
            owner_rows=owner_rows,
            recurrence_steps=self.config.recurrence_steps,
        )
        s_repr, c_repr = self.pooler.matched_pair_representations(
            encoded_s.hidden,
            s_trim,
            s_valid,
            decoded_c.hidden,
            c_trim,
            c_valid,
            owner_rows,
            cand_anchor=decoded_c.anchor,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        joint = self.meta_query_proj(torch.cat([s_repr, c_repr], dim=-1))
        ncm_bridge = self._get_active_ncm_bridge()
        if ncm_bridge is not None:
            s_mask = self.pooler._state_evidence_mask(
                s_trim,
                s_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            c_mask = self.pooler._content_mask(
                c_trim,
                c_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            s_ms = ncm_bridge.extract_features(s_trim, s_mask)
            c_ms = ncm_bridge.extract_features(c_trim, c_mask)
            joint = ncm_bridge.enhance_metacognition_query(
                joint,
                s_ms.z_coarse_pool,
                c_ms.z_coarse_pool,
                c_ms.z_res_fine_pool,
                s_ms.scale_entropy,
                c_ms.scale_entropy,
                s_ms.scale_consistency,
                c_ms.scale_consistency,
            )
        sims = pearson_similarity(
            joint.unsqueeze(1), self.meta_prototypes.unsqueeze(0), eps=self.config.eps
        )
        logits = sims * self._scale() + self.meta_bias.float().unsqueeze(0)
        probs = logits.sigmoid()
        loss = (
            F.binary_cross_entropy_with_logits(logits, targets.float())
            if targets is not None
            else None
        )
        return MetacognitionOutput(logits=logits, probabilities=probs, loss=loss)

    def compress_abstraction(
        self,
        item_ids: Tensor,
        item_lengths: Tensor,
        *,
        level: str = "standard",
        encoded_hidden: Tensor | None = None,
    ) -> Tensor:
        """Multi-scale latent abstraction (`brief`, `standard`, `detailed`) for recurrent injection."""
        if level not in self.abstraction_gates:
            raise ValueError("level must be 'brief', 'standard', or 'detailed'")
        if encoded_hidden is not None:
            width = int(encoded_hidden.shape[1])
            trimmed = item_ids[:, :width].long()
            valid = torch.arange(width, device=item_ids.device)[None, :] < item_lengths[:, None]
            hid = encoded_hidden
        else:
            width = int(item_lengths.max().item())
            trimmed = item_ids[:, :width].long()
            valid = torch.arange(width, device=item_ids.device)[None, :] < item_lengths[:, None]
            encoded = self.backbone.encode_state(trimmed, valid, causal=True)
            hid = encoded.hidden
        pooled = self.pooler.pool_sequence(
            hid,
            trimmed,
            valid,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        ncm_bridge = self._get_active_ncm_bridge()
        if ncm_bridge is not None:
            pooled = ncm_bridge.enhance_abstraction(
                trimmed,
                valid,
                pooled,
                level=level,
                pooler=self.pooler,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
        gate = self.abstraction_gates[level].sigmoid()
        return self.abstraction_norm(pooled + gate * self.abstraction_proj(pooled))

    def forward_evaluation(
        self,
        criterion_ids: Tensor,
        criterion_lengths: Tensor,
        candidate_ids: Tensor,
        candidate_lengths: Tensor,
        targets: Tensor | None = None,
    ) -> EvaluationOutput:
        """Operator 3 (`Evaluation`): Criterion-conditioned utility $U(m, c) \\in [0, 1]$."""
        batch = criterion_ids.shape[0]
        k_width = int(criterion_lengths.max().item())
        c_width = int(candidate_lengths.max().item())
        k_trim = criterion_ids[:, :k_width].long()
        c_trim = candidate_ids[:, :c_width].long()
        k_valid = (
            torch.arange(k_width, device=criterion_ids.device)[None, :]
            < criterion_lengths[:, None]
        )
        c_valid = (
            torch.arange(c_width, device=candidate_ids.device)[None, :]
            < candidate_lengths[:, None]
        )

        encoded_k = self.backbone.encode_state(k_trim, k_valid, causal=True)
        owner_rows = torch.arange(batch, device=criterion_ids.device)
        decoded_c = self.backbone.decode_candidates(
            encoded_k,
            c_trim,
            c_valid,
            owner_rows=owner_rows,
            recurrence_steps=self.config.recurrence_steps,
        )
        k_repr, c_repr = self.pooler.matched_pair_representations(
            encoded_k.hidden,
            k_trim,
            k_valid,
            decoded_c.hidden,
            c_trim,
            c_valid,
            owner_rows,
            cand_anchor=decoded_c.anchor,
            strip_boundary=self.config.strip_boundary_tokens_in_pool,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
        )
        k_proj = self.state_projection(k_repr)
        c_proj = self.output_projection(c_repr)
        sims = self._similarity(k_proj, c_proj)
        ncm_bridge = self._get_active_ncm_bridge()
        if ncm_bridge is not None:
            k_mask = self.pooler._state_evidence_mask(
                k_trim,
                k_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            c_mask = self.pooler._content_mask(
                c_trim,
                c_valid,
                strip_boundary=self.config.strip_boundary_tokens_in_pool,
                bos_id=self.config.bos_id,
                eos_id=self.config.eos_id,
            )
            sims, k_proj, c_proj, _, _ = ncm_bridge.enhance_decision_and_eval(
                k_trim,
                k_mask,
                c_trim,
                c_mask,
                owner_rows,
                k_proj,
                c_proj,
                sims,
                self._similarity,
            )
        with torch.autocast(device_type=criterion_ids.device.type, enabled=False):
            logits = sims.float() * self.assoc_alpha.float() + self.assoc_beta.float()
            utilities = logits.sigmoid()
            loss = (
                F.binary_cross_entropy_with_logits(logits, targets.float())
                if targets is not None
                else None
            )
        return EvaluationOutput(
            logits=logits,
            utilities=utilities,
            similarities=sims,
            loss=loss,
        )

    def forward_induction(
        self,
        episodes_ids: Tensor,
        episodes_lengths: Tensor,
        rule_option_ids: Tensor,
        rule_option_lengths: Tensor,
        targets: Tensor | None = None,
    ) -> DecisionOutput:
        """Operator 5 (`Induction`): Generalize latent rule candidates across observed episodes."""
        return self.forward_decision(
            episodes_ids,
            episodes_lengths,
            rule_option_ids,
            rule_option_lengths,
            targets=targets,
        )

    def forward_abduction(
        self,
        observation_ids: Tensor,
        observation_lengths: Tensor,
        hypothesis_option_ids: Tensor,
        hypothesis_option_lengths: Tensor,
        targets: Tensor | None = None,
    ) -> DecisionOutput:
        """Operator 6 (`Abduction`): Rank explanatory hypotheses under observed outcomes."""
        return self.forward_decision(
            observation_ids,
            observation_lengths,
            hypothesis_option_ids,
            hypothesis_option_lengths,
            targets=targets,
        )

    def compress_gist_context(
        self,
        token_ids: Tensor,
        valid_mask: Tensor | None = None,
        *,
        chunk_size: int | None = None,
        gist_prefix_len: int = 0,
    ) -> EncodedState:
        """Fold a context sequence `[B, L]` into recurrent soft gist tokens `[B, N_gist, d_model]`."""
        batch_size, total_len = token_ids.shape
        if valid_mask is None:
            valid_mask = token_ids != self.backbone.config.pad_id
        c_sz = max(4, int(chunk_size or self.config.gist_chunk_size))
        d_model = self.backbone.config.d_model
        n_gist_per_chunk = self.gist_queries.shape[0]

        gist_chunks: list[Tensor] = []
        gist_valids: list[Tensor] = []
        enc_chunks: list[Tensor] = []
        running_gist: Tensor | None = None
        total_aux = token_ids.new_zeros((), dtype=torch.float32)
        total_z = token_ids.new_zeros((), dtype=torch.float32)
        loads: list[Tensor] = []

        for start in range(0, total_len, c_sz):
            end = min(total_len, start + c_sz)
            c_ids = token_ids[:, start:end]
            c_val = valid_mask[:, start:end]
            if not c_val.any():
                continue

            row_has_valid = c_val.any(dim=-1)
            c_val_safe = c_val.clone()
            c_val_safe[~row_has_valid, 0] = True

            enc = self.backbone.encode_state(c_ids, c_val_safe, causal=False)
            enc_chunks.append(enc.hidden)
            total_aux = total_aux + enc.aux_loss
            total_z = total_z + enc.router_z_loss
            loads.append(enc.expert_load)

            q = self.gist_queries.unsqueeze(0).expand(batch_size, -1, -1).to(enc.hidden.dtype)
            scores = torch.matmul(q.float(), enc.hidden.float().transpose(-1, -2)) / math.sqrt(
                d_model
            )
            scores = scores.masked_fill(~c_val_safe.unsqueeze(1), -1e9)
            attn = F.softmax(scores, dim=-1).to(enc.hidden.dtype)
            pooled = torch.matmul(attn, enc.hidden).float()
            chunk_gist = (
                0.85 * pooled + 0.15 * self.gist_norm(pooled.to(self.gist_norm.weight.dtype)).float()
            ).to(enc.hidden.dtype)

            chunk_has_valid = row_has_valid.unsqueeze(-1).expand(batch_size, n_gist_per_chunk)
            chunk_gist = chunk_gist.masked_fill(~chunk_has_valid.unsqueeze(-1), 0)
            if running_gist is not None:
                safe_valid = chunk_has_valid.clone()
                safe_valid[~row_has_valid, 0] = True
                updated_gist = self.backbone.feedback(chunk_gist, running_gist, valid=safe_valid)
                chunk_gist = torch.where(
                    chunk_has_valid.unsqueeze(-1),
                    0.75 * chunk_gist + 0.25 * updated_gist,
                    running_gist,
                )
            running_gist = chunk_gist
            gist_chunks.append(chunk_gist)
            gist_valids.append(chunk_has_valid)

        if not gist_chunks:
            self._last_gist_encoded_hidden = None
            fallback_valid = valid_mask[:, :1].clone()
            fallback_valid[:, 0] = True
            return self.backbone.encode_state(token_ids[:, :1], fallback_valid, causal=False)

        self._last_gist_encoded_hidden = torch.cat(enc_chunks, dim=1) if enc_chunks else None
        compressed_hidden = torch.cat(gist_chunks, dim=1)
        compressed_valid = torch.cat(gist_valids, dim=1)
        avg_load = torch.stack(loads, dim=0).mean(dim=0)
        return EncodedState(
            hidden=compressed_hidden,
            valid=compressed_valid,
            aux_loss=total_aux / max(1, len(loads)),
            router_z_loss=total_z / max(1, len(loads)),
            expert_load=avg_load,
            gist_prefix_len=int(gist_prefix_len),
        )

    def forward_inference(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        recurrence_steps: int | None = None,
        latent_prior: Tensor | None = None,
        latent_prior_scale: float = 0.0,
        recurrent_kv: bool | None = None,
        depth_ponder: bool | None = None,
        gist_state: EncodedState | None = None,
        prior_start_pos: int = 0,
        deep_supervision_weight: float = 0.0,
        condition_on_prefix_gist: bool = False,
    ):
        """Operator 9 (`Inference`): Autoregressive `U$ -> T$ -> A$` synthesis on the shared backbone."""
        use_ponder = self.config.depth_ponder if depth_ponder is None else depth_ponder
        use_rec_kv = self.config.recurrent_kv if recurrent_kv is None else recurrent_kv
        if recurrence_steps is None:
            steps = (
                max(2, self.config.recurrence_steps)
                if use_ponder
                else self.config.recurrence_steps
            )
        else:
            steps = recurrence_steps

        eff_gist = gist_state
        eff_prior = latent_prior
        eff_scale = latent_prior_scale
        eff_start = prior_start_pos
        if condition_on_prefix_gist and input_ids.shape[1] >= 12 and eff_gist is None:
            eos_hits = input_ids == self.backbone.config.eos_id
            if eos_hits.shape[1] > 0:
                eos_hits = eos_hits.clone()
                eos_hits[:, 0] = False
            first_ep_mask = eos_hits.cumsum(dim=1) == 0
            min_ep_len = int(first_ep_mask.sum(dim=1).min().item())
            first_unmasked = int(prior_start_pos) if prior_start_pos >= 12 else 0
            if first_unmasked == 0 and targets is not None and targets.shape == input_ids.shape:
                unmasked = targets != -100
                if unmasked.any(dim=1).all():
                    first_unmasked = int(unmasked.int().argmax(dim=1).min().item())
            if first_unmasked >= 12:
                pref_len = min(first_unmasked, int(input_ids.shape[1] * 0.85), 384)
            else:
                pref_len = max(
                    4,
                    min(
                        max(12, int(min_ep_len * 0.75)),
                        int(input_ids.shape[1] * 0.75),
                        384,
                    ),
                )
            pref_ids = input_ids[:, :pref_len]
            pref_valid = pref_ids != self.backbone.config.pad_id
            if not pref_valid.any(dim=-1).all():
                pref_valid = pref_valid.clone()
                pref_valid[:, 0] = True
            eff_chunk_sz = max(16, (pref_len + 1) // 2) if pref_len >= 32 else max(4, pref_len // 2)
            eff_gist = self.compress_gist_context(
                pref_ids,
                pref_valid,
                chunk_size=eff_chunk_sz,
                gist_prefix_len=pref_len,
            )
            if eff_prior is None:
                pref_lengths = pref_valid.sum(dim=-1).clamp_min(1)
                cached_enc = getattr(self, "_last_gist_encoded_hidden", None)
                self._last_gist_encoded_hidden = None
                enc_arg = (
                    cached_enc
                    if (cached_enc is not None and cached_enc.shape[1] == pref_len)
                    else None
                )
                abs_detailed = self.compress_abstraction(
                    pref_ids,
                    pref_lengths,
                    level="detailed",
                    encoded_hidden=enc_arg,
                )
                abs_brief = self.compress_abstraction(
                    pref_ids,
                    pref_lengths,
                    level="brief",
                    encoded_hidden=enc_arg,
                )
                joint_pair = torch.cat([abs_detailed, abs_brief], dim=-1)
                ded_q = self.deduction_query_proj(joint_pair)
                ded_sims = pearson_similarity(
                    ded_q.unsqueeze(1), self.deduction_prototypes.unsqueeze(0), eps=self.config.eps
                )
                ded_probs = (ded_sims * self._scale()).softmax(dim=-1)
                meta_q = self.meta_query_proj(joint_pair)
                meta_sims = pearson_similarity(
                    meta_q.unsqueeze(1), self.meta_prototypes.unsqueeze(0), eps=self.config.eps
                )
                meta_probs = (
                    meta_sims * self._scale() + self.meta_bias.float().unsqueeze(0)
                ).sigmoid()
                ded_entail_gain = (ded_probs[:, 0:1] - 0.5 * ded_probs[:, 1:2]).clamp(0.0, 1.0)
                meta_ctx_gain = 0.5 * (meta_probs[:, 1:2] + meta_probs[:, 2:3])
                coe_gate = (0.5 * ded_entail_gain + 0.5 * meta_ctx_gain).to(abs_detailed.dtype)
                eff_prior = abs_detailed + 0.30 * coe_gate * (abs_brief - abs_detailed)
                if enc_arg is not None and eff_gist is not None:
                    with torch.autocast(device_type=input_ids.device.type, enabled=False):
                        enc_f = F.normalize(enc_arg.float(), dim=-1, eps=1e-6)
                        q_hop1 = F.normalize(abs_detailed.float(), dim=-1, eps=1e-6).unsqueeze(1)
                        attn1_logits = (q_hop1 @ enc_f.transpose(1, 2)).squeeze(1) * 8.0
                        attn1_logits = attn1_logits.masked_fill(~pref_valid, -1e9)
                        attn1_w = attn1_logits.softmax(dim=-1)
                        z_hop1 = (attn1_w.unsqueeze(1) @ enc_arg.float()).squeeze(1)

                        q_hop2 = F.normalize(
                            (z_hop1 + 0.5 * eff_prior.float()), dim=-1, eps=1e-6
                        ).unsqueeze(1)
                        attn2_logits = (
                            (q_hop2 @ enc_f.transpose(1, 2)).squeeze(1) * 8.0
                            - 2.5 * attn1_w
                        )
                        attn2_logits = attn2_logits.masked_fill(~pref_valid, -1e9)
                        attn2_w = attn2_logits.softmax(dim=-1)
                        z_hop2 = (attn2_w.unsqueeze(1) @ enc_arg.float()).squeeze(1)
                    z_hop1_t = z_hop1.to(eff_prior.dtype)
                    z_hop2_t = z_hop2.to(eff_prior.dtype)
                    eff_prior = 0.80 * eff_prior + 0.20 * (0.60 * z_hop1_t + 0.40 * z_hop2_t)
                    scratch_slots = torch.stack(
                        [eff_prior, z_hop1_t, z_hop2_t], dim=1
                    ).to(eff_gist.hidden.dtype)
                    scratch_valid = torch.ones(
                        (eff_gist.valid.shape[0], 3),
                        dtype=torch.bool,
                        device=eff_gist.valid.device,
                    )
                    eff_gist = EncodedState(
                        hidden=torch.cat([eff_gist.hidden, scratch_slots], dim=1),
                        valid=torch.cat([eff_gist.valid, scratch_valid], dim=1),
                        aux_loss=eff_gist.aux_loss,
                        router_z_loss=eff_gist.router_z_loss,
                        expert_load=eff_gist.expert_load,
                        gist_prefix_len=eff_gist.gist_prefix_len,
                    )
                del cached_enc
                if eff_scale == 0.0:
                    eff_scale = 0.15
            eff_start = pref_len

        return self.backbone.forward(
            input_ids,
            targets=targets,
            recurrence_steps=steps,
            latent_prior=eff_prior,
            latent_prior_scale=eff_scale,
            recurrent_kv=use_rec_kv,
            depth_ponder=use_ponder,
            gist_state=eff_gist,
            prior_start_pos=eff_start,
            deep_supervision_weight=deep_supervision_weight,
        )

    def create_whiteboard_controller(
        self,
        *,
        whiteboard_slots: int = 8,
        lora_rank: int = 8,
        lora_alpha: float = 4.0,
        lr: float = 3e-2,
        inner_steps: int = 2,
        max_prefix_norm: float = 0.50,
        max_adapter_frobenius_norm: float = 0.25,
        min_novelty: float = 0.02,
        min_entropy_floor: float = 0.40,
        max_anchor_drift: float = 0.08,
    ) -> COEPlasticityController:
        """Create a COE-gated quarantined fast-weight plasticity controller for test-time learning."""
        return COEPlasticityController(
            self,
            whiteboard_slots=whiteboard_slots,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lr=lr,
            inner_steps=inner_steps,
            max_prefix_norm=max_prefix_norm,
            max_adapter_frobenius_norm=max_adapter_frobenius_norm,
            min_novelty=min_novelty,
            min_entropy_floor=min_entropy_floor,
            max_anchor_drift=max_anchor_drift,
        )

    def create_ncm_system(self, config: Any | None = None) -> Any:
        """Create a Next-Concept Modeling (`NCMSystem`) anchored to this `CognitiveOperatorEnsemble`.

        Shares `self.backbone` (`token_embedding`, `lm_head`, lower encoder stack, upper CED
        decoder stack, and `SparseMoE` routers/experts) and attaches `self` for mid-hierarchy
        9-operator `COE` verification and adaptive halting.
        """
        from .ncm import NCMConfig, NCMSystem

        ncm_cfg = config if isinstance(config, NCMConfig) else NCMConfig(
            d_model=int(self.backbone.config.d_model),
            vocab_size=int(self.backbone.config.vocab_size),
            pad_id=int(self.backbone.config.pad_id),
            rms_norm_eps=float(self.backbone.config.rms_norm_eps),
        )
        system = NCMSystem(ncm_cfg, self.backbone, coe_model=self)
        p0 = next(self.parameters())
        system = system.to(device=p0.device, dtype=p0.dtype)
        self.ncm_system = system
        return system

    def forward_with_whiteboard(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        whiteboard: QuarantinedWhiteboardState,
        recurrence_steps: int | None = None,
        recurrent_kv: bool | None = None,
        depth_ponder: bool | None = None,
        include_generative_readout: bool = True,
    ):
        """Run inference conditioned on the quarantined fast-weight whiteboard state."""
        from .backbone import CausalLMOutput

        use_ponder = self.config.depth_ponder if depth_ponder is None else depth_ponder
        use_rec_kv = self.config.recurrent_kv if recurrent_kv is None else recurrent_kv
        steps = (
            (max(2, self.config.recurrence_steps) if use_ponder else self.config.recurrence_steps)
            if recurrence_steps is None
            else recurrence_steps
        )
        valid = input_ids != self.backbone.config.pad_id
        if not valid.any(dim=-1).all():
            valid = valid.clone()
            valid[:, 0] = True
        is_dec_only_donor = (
            self.backbone.config.n_encoder_layers == 0
            and not getattr(self.backbone.config, "tie_word_embeddings", True)
        )
        wb_gist = (
            None
            if is_dec_only_donor
            else whiteboard.build_gist_state(self.backbone, batch_size=input_ids.shape[0])
        )
        enc_lower = self.backbone.encode_state(input_ids, valid, causal=True)
        mod_lower = whiteboard.apply_adapter(enc_lower.hidden, valid)
        prior_vec = mod_lower.mean(dim=1)

        encoded = self.backbone.encode(
            input_ids,
            recurrence_steps=steps,
            latent_prior=prior_vec,
            latent_prior_scale=0.02 if is_dec_only_donor else 0.15,
            recurrent_kv=use_rec_kv,
            depth_ponder=use_ponder,
            gist_state=wb_gist,
            prior_start_pos=0,
        )
        has_gen = include_generative_readout and (
            float(torch.linalg.matrix_norm(whiteboard.gen_val_B.float(), ord="fro").item())
            > 1e-6
        )
        if has_gen:
            mod_dec = whiteboard.apply_generative_readout(encoded.hidden, valid)
        else:
            mod_dec = self.backbone.final_norm(
                whiteboard.apply_adapter(encoded.hidden, valid)
            ).masked_fill(~valid.unsqueeze(-1), 0)
        logits = self.backbone.lm_head(mod_dec)
        if self.backbone.config.logits_scaling != 1.0:
            logits = logits / float(self.backbone.config.logits_scaling)

        ce_loss, loss = None, None
        if targets is not None:
            valid_mask = targets.ne(-100)
            valid_tok_count = valid_mask.sum().clamp_min(1)
            ce_sum = F.cross_entropy(
                logits.float().reshape(-1, self.backbone.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            ce_loss = ce_sum / valid_tok_count
            loss = (
                ce_loss
                + self.backbone.config.aux_loss_coef * encoded.aux_loss
                + self.backbone.config.router_z_loss_coef * encoded.router_z_loss
            )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            aux_loss=encoded.aux_loss,
            router_z_loss=encoded.router_z_loss,
            expert_load=encoded.expert_load,
            step_states=encoded.step_states,
        )


@dataclass
class PlasticityTransactionReport:
    """Telemetry receipt for a single COE-gated test-time fast-weight transaction."""

    committed: bool
    status: str
    whiteboard_loss_before: float
    whiteboard_loss_after: float
    coalition_agreement_before: float
    coalition_agreement_after: float
    cycle_consistency_before: float
    cycle_consistency_after: float
    coe_score_before: float
    coe_score_after: float
    entropy_after: float
    anchor_drift: float
    novelty: float
    adapter_frobenius_norm: float
    prefix_delta_norm: float


class QuarantinedWhiteboardState(nn.Module):
    """Quarantined fast-weight workspace ($\\Delta W_{\\text{fast}}$ + latent prefix) for test-time plasticity.

    Core slow weights ($\\theta_{\\text{slow}}$) in `MCL4CEDBackbone` and `CognitiveOperatorEnsemble`
    are never mutated. All test-time adaptation is strictly confined to:
    1. `whiteboard_prefix_delta`: `[1, K_wb, d_model]` position-invariant latent prefix delta.
    2. `fast_lora_A` & `fast_lora_B`: Low-rank residual state/decoder adapter $\\frac{\\alpha}{r}(h A) B$.
    3. `rule_key_A` & `rule_val_B`: Biorthogonal Hopfield-Kanerva premise->conclusion rule readout.
    4. `gen_key_A` & `gen_val_B`: Surprise-weighted natural-gradient generative token adapter.
    Includes hard Frobenius / $L_2$ trust-region projection and $O(1)$ transactional snapshot/rollback.
    """

    def __init__(
        self,
        d_model: int,
        *,
        whiteboard_slots: int = 8,
        lora_rank: int = 8,
        lora_alpha: float = 4.0,
        max_prefix_norm: float = 0.50,
        max_adapter_frobenius_norm: float = 0.25,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if whiteboard_slots < 1:
            raise ValueError("whiteboard_slots must be >= 1")
        if lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        self.d_model = int(d_model)
        self.whiteboard_slots = int(whiteboard_slots)
        self.lora_rank = int(lora_rank)
        self.rule_rank = max(32, self.lora_rank)
        self.gen_rank = max(32, self.lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.max_prefix_norm = float(max_prefix_norm)
        self.max_adapter_frobenius_norm = float(max_adapter_frobenius_norm)

        self.register_buffer(
            "base_prefix",
            torch.zeros(1, self.whiteboard_slots, self.d_model, device=device, dtype=dtype),
        )
        self.whiteboard_prefix_delta = nn.Parameter(
            torch.zeros(1, self.whiteboard_slots, self.d_model, device=device, dtype=dtype)
        )
        self.fast_lora_A = nn.Parameter(
            torch.randn(self.d_model, self.lora_rank, device=device, dtype=dtype)
            * (0.02 / math.sqrt(self.d_model))
        )
        self.fast_lora_B = nn.Parameter(
            torch.zeros(self.lora_rank, self.d_model, device=device, dtype=dtype)
        )
        self.rule_key_A = nn.Parameter(
            torch.randn(self.d_model, self.rule_rank, device=device, dtype=dtype)
            * (0.04 / math.sqrt(self.d_model))
        )
        self.rule_val_B = nn.Parameter(
            torch.zeros(self.rule_rank, self.d_model, device=device, dtype=dtype)
        )
        self.gen_key_A = nn.Parameter(
            torch.randn(self.d_model, self.gen_rank, device=device, dtype=dtype)
            * (0.02 / math.sqrt(self.d_model))
        )
        self.gen_val_B = nn.Parameter(
            torch.zeros(self.gen_rank, self.d_model, device=device, dtype=dtype)
        )
        self._rule_key_bank: Tensor | None = None
        self._rule_cand_bank: Tensor | None = None
        self._rule_premise_holo_bank: Tensor | None = None
        self._rule_conc_holo_bank: Tensor | None = None
        self._rule_dual_key_bank: Tensor | None = None
        self._rule_closed_conc_holo_bank: Tensor | None = None
        self._rule_z_row_bank: Tensor | None = None
        self._rule_closure_op: Tensor | None = None
        self._rule_premise_texts: list[str] = []
        self._rule_conc_texts: list[str] = []
        self.revision_history: list[dict[str, Any]] = []
        self._snapshot: tuple[Any, ...] | None = None

    _TOKENIZER_CACHE: dict[int, object] = {}
    _STOPWORDS: frozenset[str] = frozenset(
        {
            "the", "and", "for", "with", "that", "this", "from", "into", "when",
            "whenever", "where", "which", "what", "how", "does", "select", "valid",
            "implication", "choice", "suppose", "during", "examine", "inspect",
            "according", "under", "within", "across", "between", "after", "before",
            "below", "above", "has", "have", "are", "was", "were", "will", "would",
            "should", "could", "can", "may", "might", "must", "its", "their", "our",
            "your", "his", "her", "then", "than", "also", "only", "such", "each",
            "every", "any", "all", "some", "more", "most", "other", "another",
            "system", "specification", "condition", "happens", "action", "performed",
            "performs", "perform", "indicate", "indicates", "indicated", "return",
            "returns", "returned", "default", "defaults", "assigned", "configured",
            "function", "class", "method", "attribute", "module", "constant",
            "portal", "table", "section", "definition", "list", "theorem", "lemma",
            "in", "on", "at", "to", "of", "by", "as", "is", "be", "or", "an", "if",
            "calling", "otherwise", "specifies", "downstream", "guarantee", "holds",
            "respond", "status", "string", "hardware", "storage", "controller",
            "measured", "arrives", "incoming", "shard", "experts",
        }
    )

    @property
    def scale(self) -> float:
        return self.lora_alpha / float(self.lora_rank)

    @property
    def episodic_rule_count(self) -> int:
        return 0 if self._rule_key_bank is None else int(self._rule_key_bank.shape[0])

    @classmethod
    def _fnv1a_63(cls, text: str) -> int:
        h = 1469598103934665603
        for b in text.encode("utf-8"):
            h = ((h ^ b) * 1099511628211) & 0x7FFFFFFFFFFFFFFF
        return max(1, h)

    @classmethod
    def decode_clean_texts(
        cls,
        token_ids: Tensor,
        valid_mask: Tensor,
        vocab_size: int,
    ) -> tuple[list[str], list[bool], list[list[int]]]:
        """Decode token IDs into normalized lowercase strings and printable-text flags."""
        tok = cls._TOKENIZER_CACHE.get(vocab_size)
        if tok is None and vocab_size > 260:
            try:
                from .data import get_tokenizer

                tok = get_tokenizer(vocab_size)
                cls._TOKENIZER_CACHE[vocab_size] = tok
            except Exception:
                tok = None

        ids_cpu = token_ids.detach().cpu()
        mask_cpu = valid_mask.detach().cpu()
        bsz = int(token_ids.shape[0])
        clean_texts: list[str] = []
        printable_flags: list[bool] = []
        v_ids_all: list[list[int]] = []

        for b in range(bsz):
            v_ids = ids_cpu[b][mask_cpu[b]].tolist()
            v_ids_all.append(v_ids)
            txt = ""
            if tok is not None:
                try:
                    txt = tok.decode(v_ids)
                except Exception:
                    txt = ""
            if not txt:
                raw_bytes = bytes(
                    int(x) & 0xFF for x in v_ids if 32 <= int(x) <= 126 or int(x) == 10
                )
                txt = raw_bytes.decode("utf-8", errors="ignore")

            txt_clean = re.sub(r"\n?#Q\$#[\s\S]*$", "", txt)
            txt_clean = re.sub(r"#[UQTSHA]\$#|\b[UQTSHA]\$", " ", txt_clean)
            txt_lower = re.sub(r"\s+", " ", txt_clean.lower()).strip()
            is_printable = (tok is not None and bool(txt)) or (
                all(
                    (
                        32 <= int(x) <= 126
                        or int(x) in (0, 1, 2, 3, 9, 10, 13, 256, 257, 258, 259)
                    )
                    for x in v_ids
                )
                and (" " in txt_lower)
            )
            clean_texts.append(txt_lower)
            printable_flags.append(is_printable)
        return clean_texts, printable_flags, v_ids_all

    @classmethod
    def extract_symbolic_predicates(
        cls, txt_lower: str
    ) -> tuple[list[tuple[str, str, float]], list[tuple[str, bool]]]:
        """Extract `(var_key, op, value)` numeric predicates and `(var_key, bool_val)` boolean guards."""
        # Mask hyphenated/versioned identifiers (e.g. v4.2, krylov-7, cxl-3.0, e-904, lane_4, merkle_256) without masking scientific notation (1e-6)
        masked = re.sub(
            r"\b(?:v\d+(?:\.\d+)?|[a-z][a-z0-9]*[-_]\d+(?:\.\d+)?)\b", " ", txt_lower
        )

        num_preds: list[tuple[str, str, float]] = []
        bool_preds: list[tuple[str, bool]] = []

        def _norm_var(lhs_str: str) -> str:
            snakes = re.findall(r"[a-z][a-z0-9]*_[a-z0-9_]+", lhs_str)
            if snakes:
                return snakes[-1]
            words = [
                w
                for w in re.findall(r"[a-z]{3,}", lhs_str)
                if w not in cls._STOPWORDS
                and w not in ("remains", "strictly", "bounded", "stays", "value", "level", "rate")
            ]
            return words[-1] if words else ""

        # 1. Symbolic numeric comparisons (>=, <=, ==, >, <, =)
        for m in re.finditer(
            r"([a-z_][a-z0-9_.\s]{1,45}?)\s*(>=|<=|==|>|<|=)\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\b",
            masked,
        ):
            v_key = _norm_var(m.group(1))
            op_raw = m.group(2)
            op = "==" if op_raw == "=" else op_raw
            try:
                val = float(m.group(3))
            except ValueError:
                continue
            if v_key:
                num_preds.append((v_key, op, val))

        # 2. Natural-language comparative numeric phrases
        nl_pat = re.compile(
            r"([a-z_][a-z0-9_.\s]{1,50}?)\s+"
            r"(exceeds|is\s+greater\s+than|rises\s+above|stays\s+above|drops\s+below|falls\s+below|stays\s+below|"
            r"remains\s+strictly\s+bounded\s+below|is\s+bounded\s+below|bounded\s+below|"
            r"is\s+less\s+than|at\s+least|at\s+most|equals|is\s+equal\s+to|"
            r"is\s+measured\s+at|measured\s+at|arrives\s+with)\s+"
            r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\b"
        )
        for m in nl_pat.finditer(masked):
            v_key = _norm_var(m.group(1))
            phrase = re.sub(r"\s+", " ", m.group(2).strip())
            if phrase in ("exceeds", "is greater than", "rises above", "stays above"):
                op = ">"
            elif phrase == "at least":
                op = ">="
            elif phrase in (
                "drops below",
                "falls below",
                "stays below",
                "remains strictly bounded below",
                "is bounded below",
                "bounded below",
                "is less than",
            ):
                op = "<"
            elif phrase == "at most":
                op = "<="
            else:
                op = "=="
            try:
                val = float(m.group(3))
            except ValueError:
                continue
            if v_key and (v_key, op, val) not in num_preds:
                num_preds.append((v_key, op, val))

        # 3. Explicit boolean assignments (e.g. `crc_mismatch is false`, `crc_mismatch = true`)
        for bm in re.finditer(
            r"\b([a-z_][a-z0-9_]*)\s+(?:is\s+|=\s*|==\s*)(true|false)\b", txt_lower
        ):
            b_var = bm.group(1)
            b_val = bm.group(2) == "true"
            bool_preds.append((b_var, b_val))

        # 4. Implicit boolean guard in extracted code premise (`when crc_mismatch` or `when not crc_mismatch`)
        if not bool_preds:
            gm = re.search(r"\bwhen\s+(not\s+)?([a-z_][a-z0-9_]*)\s*[,.]?$", txt_lower)
            if gm and gm.group(2) not in ("nominal", "true", "false"):
                bool_preds.append((gm.group(2), not bool(gm.group(1))))

        return num_preds, bool_preds

    def score_symbolic_predicates(
        self,
        query_texts: list[str],
        device: torch.device,
    ) -> Tensor | None:
        """Evaluate concrete numerical & boolean test inputs in `query_texts` against stored rule inequalities/guards."""
        if (
            self._rule_key_bank is None
            or not self._rule_premise_texts
            or len(self._rule_premise_texts) != int(self._rule_key_bank.shape[0])
        ):
            return None

        n_rules = len(self._rule_premise_texts)
        bsz = len(query_texts)
        sym_adj = torch.zeros((bsz, n_rules), dtype=torch.float32, device=device)

        rule_parsed = [
            self.extract_symbolic_predicates(p_txt) for p_txt in self._rule_premise_texts
        ]
        rule_scopes: list[str] = []
        rule_is_fallback: list[bool] = []
        for p_txt in self._rule_premise_texts:
            is_fb = "otherwise when nominal" in p_txt
            rule_is_fallback.append(is_fb)
            if p_txt.startswith("in ") and "," in p_txt:
                rule_scopes.append(p_txt.split(",", 1)[0].strip())
            else:
                rule_scopes.append("")

        any_active = False
        for b, q_txt in enumerate(query_texts):
            q_num, q_bool = self.extract_symbolic_predicates(q_txt)
            q_num_assigns = [(v, val) for (v, op, val) in q_num if op == "=="]
            q_bool_map = dict(q_bool)
            if not q_num_assigns and not q_bool_map:
                continue

            scope_sat: dict[str, int] = {}
            scope_falsified: dict[str, int] = {}

            for r in range(n_rules):
                r_num, r_bool = rule_parsed[r]
                sc = rule_scopes[r]

                for r_var, r_op, r_val in r_num:
                    if r_op not in (">", ">=", "<", "<="):
                        continue
                    for q_var, q_val in q_num_assigns:
                        if q_var == r_var:
                            any_active = True
                            sat = (
                                (q_val > r_val)
                                if r_op == ">"
                                else (q_val >= r_val)
                                if r_op == ">="
                                else (q_val < r_val)
                                if r_op == "<"
                                else (q_val <= r_val)
                            )
                            if sat:
                                sym_adj[b, r] += 0.85
                                if sc:
                                    scope_sat[sc] = scope_sat.get(sc, 0) + 1
                            else:
                                sym_adj[b, r] -= 0.85
                                if sc:
                                    scope_falsified[sc] = scope_falsified.get(sc, 0) + 1

                for r_bvar, r_bval in r_bool:
                    if r_bvar in q_bool_map:
                        any_active = True
                        if q_bool_map[r_bvar] == r_bval:
                            sym_adj[b, r] += 0.45
                            if sc:
                                scope_sat[sc] = scope_sat.get(sc, 0) + 1
                        else:
                            sym_adj[b, r] -= 0.85
                            if sc:
                                scope_falsified[sc] = scope_falsified.get(sc, 0) + 1

            for r in range(n_rules):
                sc = rule_scopes[r]
                if (
                    rule_is_fallback[r]
                    and sc
                    and scope_falsified.get(sc, 0) >= 1
                    and scope_sat.get(sc, 0) == 0
                ):
                    sym_adj[b, r] += 0.95

        return sym_adj if any_active else None

    @classmethod
    def compute_holographic_sketch(
        cls,
        token_ids: Tensor,
        valid_mask: Tensor,
        token_embedding: nn.Embedding,
    ) -> Tensor:
        """Compute an isotropic, tokenizer-invariant Rademacher Content-Word / Identifier / Bigram sketch in `d_model`."""
        device = token_ids.device
        bsz = int(token_ids.shape[0])
        d_model = int(token_embedding.embedding_dim)
        vocab_size = int(token_embedding.num_embeddings)

        clean_texts, printable_flags, v_ids_all = cls.decode_clean_texts(
            token_ids, valid_mask, vocab_size
        )
        row_hashes: list[list[int]] = []

        for b in range(bsz):
            txt_lower = clean_texts[b]
            is_printable_text = printable_flags[b]
            v_ids = v_ids_all[b]

            feats: list[str] = []
            if is_printable_text:
                # 1. High-specificity compound identifiers (snake_case, URL paths, hyphenated codes)
                compounds = re.findall(
                    r"[a-z0-9]+(?:[_/\-][a-z0-9]+)+", txt_lower
                )
                for comp in compounds:
                    if len(comp) >= 4:
                        feats.append(f"id:{comp}")
                        feats.append(f"id:{comp}")

                # 2. Content words & numbers (excluding generic framing stopwords)
                words_raw = re.findall(r"[a-z0-9]+", txt_lower)
                content_words = [
                    w
                    for w in words_raw
                    if (len(w) >= 3 or any(ch.isdigit() for ch in w))
                    and w not in cls._STOPWORDS
                ]
                for w in content_words:
                    feats.append(f"w:{w}")
                    if len(w) >= 6:
                        feats.append(f"st:{w[:5]}")
                for k in range(len(content_words) - 1):
                    feats.append(f"bg:{content_words[k]}_{content_words[k + 1]}")

            if len(feats) < 2:
                # Fallback for synthetic integer token sequences in unit tests
                for idx_t, tid in enumerate(v_ids):
                    feats.append(f"t1:{tid}")
                    if idx_t + 1 < len(v_ids):
                        feats.append(f"t2:{tid}_{v_ids[idx_t + 1]}")
                    if idx_t + 2 < len(v_ids):
                        feats.append(f"t3:{tid}_{v_ids[idx_t + 1]}_{v_ids[idx_t + 2]}")

            if not feats:
                feats = ["empty:0"]
            row_hashes.append([cls._fnv1a_63(f) for f in feats[:96]])

        max_m = max(len(r) for r in row_hashes)
        h_mat = torch.zeros((bsz, max_m), dtype=torch.int64, device=device)
        m_mat = torch.zeros((bsz, max_m), dtype=torch.float32, device=device)
        for b, r_list in enumerate(row_hashes):
            h_mat[b, : len(r_list)] = torch.tensor(r_list, dtype=torch.int64, device=device)
            m_mat[b, : len(r_list)] = 1.0

        d_idx = torch.arange(d_model, device=device, dtype=torch.int64).view(1, 1, d_model)
        mixed = (
            h_mat.unsqueeze(-1) * 2862933555777941757
            + d_idx * 3037000493
            + 1442695040888963407
        )
        mixed = mixed ^ (mixed >> 29)
        mixed = mixed * 2685821657736338717
        mixed = mixed ^ (mixed >> 27)
        signs = torch.where((mixed & 1) == 0, 1.0, -1.0)
        sk = (signs * m_mat.unsqueeze(-1)).sum(dim=1)
        sk = sk - sk.mean(dim=-1, keepdim=True)
        return F.normalize(sk, dim=-1, eps=1e-6)

    def score_holo_candidates(
        self,
        global_state: Tensor,
        state_ids: Tensor,
        state_mask: Tensor,
        cand_ids: Tensor,
        cand_mask: Tensor,
        owner_rows: Tensor,
        token_embedding: nn.Embedding,
    ) -> Tensor | None:
        """Score candidate options against the transitive-closed Rademacher holographic conclusion bank."""
        if (
            self._rule_dual_key_bank is None
            or self._rule_closed_conc_holo_bank is None
            or self._rule_premise_holo_bank is None
        ):
            return None
        s_key = self.compute_rule_key(
            global_state,
            token_ids=state_ids,
            valid_mask=state_mask,
            token_embedding=token_embedding,
        ).float()
        s_holo = self.compute_holographic_sketch(
            state_ids, state_mask, token_embedding
        ).float()
        d_keys = self._rule_dual_key_bank.to(device=s_key.device, dtype=torch.float32)
        ph_bank = self._rule_premise_holo_bank.to(device=s_key.device, dtype=torch.float32)
        h_closed = self._rule_closed_conc_holo_bank.to(
            device=s_key.device, dtype=torch.float32
        )

        w_dual = s_key @ d_keys.T
        w_holo = s_holo @ ph_bank.T
        w_comb = 0.50 * w_dual + 0.50 * w_holo

        q_texts, _, _ = self.decode_clean_texts(
            state_ids, state_mask, int(token_embedding.num_embeddings)
        )
        sym_adj = self.score_symbolic_predicates(q_texts, s_key.device)
        if sym_adj is not None:
            w_comb = w_comb + sym_adj

        w_max = w_comb.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        w_sparse = torch.where(
            w_comb >= 0.45 * w_max, w_comb.clamp_min(0.0), torch.zeros_like(w_comb)
        )
        pred_holo = F.normalize(w_sparse @ h_closed, dim=-1, eps=1e-6)
        pred_rows = pred_holo.index_select(0, owner_rows)
        c_holo = self.compute_holographic_sketch(
            cand_ids, cand_mask, token_embedding
        ).float()
        return (pred_rows * c_holo).sum(dim=-1).clamp(-1.0, 1.0)

    def compute_rule_key(
        self,
        global_state: Tensor,
        token_ids: Tensor | None = None,
        valid_mask: Tensor | None = None,
        token_embedding: nn.Embedding | None = None,
    ) -> Tensor:
        """Compute hybrid contextual + position-invariant Rademacher N-gram rule key."""
        s_cent = global_state.float() - global_state.float().mean(dim=-1, keepdim=True)
        s_ctx = F.normalize(s_cent, dim=-1, eps=1e-6)
        if token_ids is not None and valid_mask is not None and token_embedding is not None:
            s_holo = self.compute_holographic_sketch(token_ids, valid_mask, token_embedding)
            s_comb = 0.35 * s_ctx + 0.65 * s_holo
            s_comb = s_comb - s_comb.mean(dim=-1, keepdim=True)
            return F.normalize(s_comb, dim=-1, eps=1e-6)
        return s_ctx

    @torch.no_grad()
    def append_episodic_rules(
        self,
        s_f_new: Tensor,
        c_unit_new: Tensor,
        s_holo_new: Tensor | None = None,
        c_holo_new: Tensor | None = None,
        *,
        premise_texts: list[str] | None = None,
        conclusion_texts: list[str] | None = None,
        allow_belief_revision: bool = True,
        max_rules: int = 64,
    ) -> tuple[Tensor, Tensor]:
        """Accumulate premise/conclusion pairs with Non-Monotonic AGM Belief Revision on premise collision."""
        s_new = s_f_new.detach().float()
        c_new = c_unit_new.detach().float()
        sh_new = (
            s_holo_new.detach().float()
            if s_holo_new is not None
            else s_new.clone()
        )
        ch_new = (
            c_holo_new.detach().float()
            if c_holo_new is not None
            else torch.zeros_like(s_new)
        )
        n_inc = int(s_new.shape[0])
        p_txts_in = (
            list(premise_texts)
            if premise_texts is not None and len(premise_texts) == n_inc
            else [""] * n_inc
        )
        c_txts_in = (
            list(conclusion_texts)
            if conclusion_texts is not None and len(conclusion_texts) == n_inc
            else [""] * n_inc
        )

        if self._rule_key_bank is None or self._rule_cand_bank is None:
            self._rule_key_bank = s_new[-max_rules:].clone()
            self._rule_cand_bank = c_new[-max_rules:].clone()
            self._rule_premise_holo_bank = sh_new[-max_rules:].clone()
            self._rule_conc_holo_bank = ch_new[-max_rules:].clone()
            self._rule_premise_texts = p_txts_in[-max_rules:]
            self._rule_conc_texts = c_txts_in[-max_rules:]
            return self._rule_key_bank, self._rule_cand_bank

        k_bank = self._rule_key_bank.to(device=s_new.device, dtype=torch.float32)
        c_bank = self._rule_cand_bank.to(device=c_new.device, dtype=torch.float32)
        ph_bank = (
            self._rule_premise_holo_bank.to(device=s_new.device, dtype=torch.float32)
            if self._rule_premise_holo_bank is not None
            else k_bank.clone()
        )
        ch_bank = (
            self._rule_conc_holo_bank.to(device=s_new.device, dtype=torch.float32)
            if self._rule_conc_holo_bank is not None
            else torch.zeros_like(k_bank)
        )
        p_bank_txts = (
            list(self._rule_premise_texts)
            if len(self._rule_premise_texts) == int(k_bank.shape[0])
            else [""] * int(k_bank.shape[0])
        )
        c_bank_txts = (
            list(self._rule_conc_texts)
            if len(self._rule_conc_texts) == int(k_bank.shape[0])
            else [""] * int(k_bank.shape[0])
        )

        for i in range(n_inc):
            s_row = s_new[i : i + 1]
            c_row = c_new[i : i + 1]
            sh_row = sh_new[i : i + 1]
            ch_row = ch_new[i : i + 1]
            p_txt = p_txts_in[i]
            c_txt = c_txts_in[i]

            sims = (k_bank @ s_row.T).squeeze(-1)
            max_sim, max_idx = sims.max(dim=0)
            idx_k = int(max_idx.item())
            if float(max_sim.item()) > 0.995:
                if allow_belief_revision:
                    c_sim_val = float((ch_bank[idx_k] * ch_row[0]).sum().item())
                    if c_sim_val < 0.78:
                        self.revision_history.append(
                            {
                                "rule_index": idx_k,
                                "old_premise": p_bank_txts[idx_k],
                                "old_conclusion": c_bank_txts[idx_k],
                                "new_premise": p_txt,
                                "new_conclusion": c_txt,
                                "premise_similarity": float(max_sim.item()),
                                "conclusion_similarity": c_sim_val,
                            }
                        )
                k_bank[idx_k] = s_row[0]
                c_bank[idx_k] = c_row[0]
                ph_bank[idx_k] = sh_row[0]
                ch_bank[idx_k] = ch_row[0]
                p_bank_txts[idx_k] = p_txt
                c_bank_txts[idx_k] = c_txt
                continue

            # Non-Monotonic AGM Belief Revision: check premise equivalence with conclusion divergence
            if allow_belief_revision and ph_bank.shape[0] > 0:
                p_sims = (ph_bank @ sh_row.T).squeeze(-1)
                best_p_sim, best_p_idx = p_sims.max(dim=0)
                idx_p = int(best_p_idx.item())
                p_sim_val = float(best_p_sim.item())
                old_p_txt = p_bank_txts[idx_p]
                old_num, old_bool = self.extract_symbolic_predicates(old_p_txt)
                new_num, new_bool = self.extract_symbolic_predicates(p_txt)
                same_pred_guard = (old_num == new_num) and (old_bool == new_bool)
                if p_sim_val >= 0.80 and same_pred_guard:
                    c_sim_val = float((ch_bank[idx_p] * ch_row[0]).sum().item())
                    if c_sim_val < 0.78:
                        self.revision_history.append(
                            {
                                "rule_index": idx_p,
                                "old_premise": old_p_txt,
                                "old_conclusion": c_bank_txts[idx_p],
                                "new_premise": p_txt,
                                "new_conclusion": c_txt,
                                "premise_similarity": p_sim_val,
                                "conclusion_similarity": c_sim_val,
                            }
                        )
                    k_bank[idx_p] = s_row[0]
                    c_bank[idx_p] = c_row[0]
                    ph_bank[idx_p] = sh_row[0]
                    ch_bank[idx_p] = ch_row[0]
                    p_bank_txts[idx_p] = p_txt
                    c_bank_txts[idx_p] = c_txt
                    continue

            k_bank = torch.cat([k_bank, s_row], dim=0)
            c_bank = torch.cat([c_bank, c_row], dim=0)
            ph_bank = torch.cat([ph_bank, sh_row], dim=0)
            ch_bank = torch.cat([ch_bank, ch_row], dim=0)
            p_bank_txts.append(p_txt)
            c_bank_txts.append(c_txt)

        if k_bank.shape[0] > max_rules:
            k_bank = k_bank[-max_rules:]
            c_bank = c_bank[-max_rules:]
            ph_bank = ph_bank[-max_rules:]
            ch_bank = ch_bank[-max_rules:]
            p_bank_txts = p_bank_txts[-max_rules:]
            c_bank_txts = c_bank_txts[-max_rules:]
        self._rule_key_bank = k_bank.clone()
        self._rule_cand_bank = c_bank.clone()
        self._rule_premise_holo_bank = ph_bank.clone()
        self._rule_conc_holo_bank = ch_bank.clone()
        self._rule_premise_texts = p_bank_txts
        self._rule_conc_texts = c_bank_txts
        return self._rule_key_bank, self._rule_cand_bank

    @torch.no_grad()
    def initialize_base_prefix(self, prefix_hidden: Tensor) -> None:
        """Seed the base whiteboard prefix from an encoded state or `WorkingStateLedger`."""
        p = prefix_hidden.detach().to(
            device=self.base_prefix.device, dtype=self.base_prefix.dtype
        )
        if p.ndim == 2:
            p = p.unsqueeze(0)
        if p.shape[0] > 1:
            p = p.mean(dim=0, keepdim=True)
        if p.shape[1] >= self.whiteboard_slots:
            pooled = p[:, : self.whiteboard_slots, :]
        else:
            reps = (self.whiteboard_slots + p.shape[1] - 1) // p.shape[1]
            pooled = p.repeat(1, reps, 1)[:, : self.whiteboard_slots, :]
        self.base_prefix.copy_(pooled)

    @torch.no_grad()
    def snapshot(self) -> None:
        """Save an $O(1)$ restore point prior to a provisional plasticity step."""
        self._snapshot = (
            self.whiteboard_prefix_delta.detach().clone(),
            self.fast_lora_A.detach().clone(),
            self.fast_lora_B.detach().clone(),
            self.rule_key_A.detach().clone(),
            self.rule_val_B.detach().clone(),
            self.gen_key_A.detach().clone(),
            self.gen_val_B.detach().clone(),
            self._rule_key_bank.detach().clone() if self._rule_key_bank is not None else None,
            self._rule_cand_bank.detach().clone() if self._rule_cand_bank is not None else None,
            self._rule_premise_holo_bank.detach().clone()
            if self._rule_premise_holo_bank is not None
            else None,
            self._rule_conc_holo_bank.detach().clone()
            if self._rule_conc_holo_bank is not None
            else None,
            self._rule_dual_key_bank.detach().clone()
            if self._rule_dual_key_bank is not None
            else None,
            self._rule_closed_conc_holo_bank.detach().clone()
            if self._rule_closed_conc_holo_bank is not None
            else None,
            self._rule_z_row_bank.detach().clone()
            if self._rule_z_row_bank is not None
            else None,
            self._rule_closure_op.detach().clone()
            if self._rule_closure_op is not None
            else None,
            list(self._rule_premise_texts),
            list(self._rule_conc_texts),
            list(self.revision_history),
        )

    @torch.no_grad()
    def rollback(self) -> None:
        """Instant $O(1)$ rollback to the last committed snapshot."""
        if self._snapshot is not None:
            (
                d,
                a,
                b,
                ka,
                vb,
                ga,
                gb,
                rkb,
                rcb,
                phb,
                chb,
                dkb,
                cchb,
                zrb,
                cop,
                ptxts,
                ctxts,
                rev_h,
            ) = self._snapshot
            self.whiteboard_prefix_delta.copy_(d)
            self.fast_lora_A.copy_(a)
            self.fast_lora_B.copy_(b)
            self.rule_key_A.copy_(ka)
            self.rule_val_B.copy_(vb)
            self.gen_key_A.copy_(ga)
            self.gen_val_B.copy_(gb)
            self._rule_key_bank = rkb.clone() if rkb is not None else None
            self._rule_cand_bank = rcb.clone() if rcb is not None else None
            self._rule_premise_holo_bank = phb.clone() if phb is not None else None
            self._rule_conc_holo_bank = chb.clone() if chb is not None else None
            self._rule_dual_key_bank = dkb.clone() if dkb is not None else None
            self._rule_closed_conc_holo_bank = cchb.clone() if cchb is not None else None
            self._rule_z_row_bank = zrb.clone() if zrb is not None else None
            self._rule_closure_op = cop.clone() if cop is not None else None
            self._rule_premise_texts = list(ptxts)
            self._rule_conc_texts = list(ctxts)
            self.revision_history = list(rev_h)

    @torch.no_grad()
    def reset_fast_weights(self) -> None:
        """Zero out all plastic fast-weight deltas, returning to the exact base model."""
        self.whiteboard_prefix_delta.zero_()
        nn.init.normal_(self.fast_lora_A, std=0.02 / math.sqrt(self.d_model))
        self.fast_lora_B.zero_()
        nn.init.normal_(self.rule_key_A, std=0.04 / math.sqrt(self.d_model))
        self.rule_val_B.zero_()
        nn.init.normal_(self.gen_key_A, std=0.02 / math.sqrt(self.d_model))
        self.gen_val_B.zero_()
        self._rule_key_bank = None
        self._rule_cand_bank = None
        self._rule_premise_holo_bank = None
        self._rule_conc_holo_bank = None
        self._rule_dual_key_bank = None
        self._rule_closed_conc_holo_bank = None
        self._rule_z_row_bank = None
        self._rule_closure_op = None
        self._rule_premise_texts = []
        self._rule_conc_texts = []
        self.revision_history = []
        self._snapshot = None

    @torch.no_grad()
    def project_to_trust_region(self) -> None:
        """Enforce hard $L_2$ and Frobenius norm bounds so fast weights can never diverge."""
        slot_norms = self.whiteboard_prefix_delta.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
        clamp_scale = torch.clamp(self.max_prefix_norm / slot_norms, max=1.0).to(
            self.whiteboard_prefix_delta.dtype
        )
        self.whiteboard_prefix_delta.mul_(clamp_scale)

        eff_mat = (self.fast_lora_A.float() @ self.fast_lora_B.float()) * self.scale
        frob = float(eff_mat.norm().item())
        if frob > self.max_adapter_frobenius_norm and frob > 0.0:
            ratio = math.sqrt(self.max_adapter_frobenius_norm / frob)
            self.fast_lora_A.mul_(ratio)
            self.fast_lora_B.mul_(ratio)

        rule_mat = (self.rule_key_A.float() @ self.rule_val_B.float()) * self.scale
        rule_frob = float(rule_mat.norm().item())
        max_rule_frob = self.max_adapter_frobenius_norm * math.sqrt(float(self.rule_rank))
        if rule_frob > max_rule_frob and rule_frob > 0.0:
            r_ratio = math.sqrt(max_rule_frob / rule_frob)
            self.rule_key_A.mul_(r_ratio)
            self.rule_val_B.mul_(r_ratio)

        gen_mat = (self.gen_key_A.float() @ self.gen_val_B.float()) * self.scale
        gen_frob = float(gen_mat.norm().item())
        max_gen_frob = self.max_adapter_frobenius_norm * math.sqrt(float(self.gen_rank)) * 1.5
        if gen_frob > max_gen_frob and gen_frob > 0.0:
            g_ratio = math.sqrt(max_gen_frob / gen_frob)
            self.gen_key_A.mul_(g_ratio)
            self.gen_val_B.mul_(g_ratio)

    def adapter_frobenius_norm(self) -> float:
        with torch.no_grad():
            eff_mat = (self.fast_lora_A.float() @ self.fast_lora_B.float()) * self.scale
            return float(eff_mat.norm().item())

    def prefix_delta_norm(self) -> float:
        with torch.no_grad():
            return float(self.whiteboard_prefix_delta.float().norm(dim=-1).max().item())

    def apply_adapter(self, hidden: Tensor, valid: Tensor | None = None) -> Tensor:
        """Apply the quarantined low-rank fast adapter $h + \\frac{\\alpha}{r}(h A) B$."""
        h_f = hidden.to(self.fast_lora_A.dtype)
        delta = ((h_f @ self.fast_lora_A) @ self.fast_lora_B) * self.scale
        out = hidden + delta.to(hidden.dtype)
        if valid is not None:
            out = out.masked_fill(~valid.unsqueeze(-1), 0)
        return out

    def apply_rule_readout(
        self,
        global_state: Tensor,
        token_ids: Tensor | None = None,
        valid_mask: Tensor | None = None,
        token_embedding: nn.Embedding | None = None,
    ) -> Tensor:
        """Read out directed premise->conclusion rule vector $\\frac{\\alpha}{r}(\\tilde{s}_{\\text{key}} K_{\\text{rule}}) V_{\\text{rule}}$."""
        s_f = self.compute_rule_key(
            global_state,
            token_ids=token_ids,
            valid_mask=valid_mask,
            token_embedding=token_embedding,
        ).to(self.rule_key_A.dtype)
        if (
            token_ids is not None
            and valid_mask is not None
            and token_embedding is not None
            and self._rule_dual_key_bank is not None
            and self._rule_z_row_bank is not None
        ):
            q_texts, _, _ = self.decode_clean_texts(
                token_ids, valid_mask, int(token_embedding.num_embeddings)
            )
            sym_adj = self.score_symbolic_predicates(q_texts, global_state.device)
            if sym_adj is not None and float(sym_adj.abs().max().item()) > 1e-4:
                d_keys = self._rule_dual_key_bank.to(
                    device=global_state.device, dtype=torch.float32
                )
                z_bank = self._rule_z_row_bank.to(
                    device=global_state.device, dtype=torch.float32
                )
                w_dual = s_f.float() @ d_keys.T
                if self._rule_premise_holo_bank is not None:
                    s_holo = self.compute_holographic_sketch(
                        token_ids, valid_mask, token_embedding
                    ).float()
                    ph_b = self._rule_premise_holo_bank.to(
                        device=global_state.device, dtype=torch.float32
                    )
                    w_comb = 0.50 * w_dual + 0.50 * (s_holo @ ph_b.T) + sym_adj
                else:
                    w_comb = w_dual + sym_adj
                w_max = w_comb.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
                w_sparse = torch.where(
                    w_comb >= 0.45 * w_max,
                    w_comb.clamp_min(0.0),
                    torch.zeros_like(w_comb),
                )
                return F.normalize(w_sparse @ z_bank, dim=-1, eps=1e-6).to(
                    global_state.dtype
                )
        delta = ((s_f @ self.rule_key_A) @ self.rule_val_B) * self.scale
        return delta.to(global_state.dtype)

    def apply_generative_readout(self, hidden: Tensor, valid: Tensor | None = None) -> Tensor:
        """Apply the natural-gradient token generative adapter $h + \\frac{\\alpha}{r}(h A_{\\text{gen}}) B_{\\text{gen}}$."""
        h_f = hidden.to(self.gen_key_A.dtype)
        delta = ((h_f @ self.gen_key_A) @ self.gen_val_B) * self.scale
        out = hidden + delta.to(hidden.dtype)
        if valid is not None:
            out = out.masked_fill(~valid.unsqueeze(-1), 0)
        return out

    def active_prefix_hidden(self, batch_size: int = 1, dtype: torch.dtype | None = None) -> Tensor:
        """Return the fast-modulated `[B, K_wb, d_model]` whiteboard prefix tensor."""
        raw = self.base_prefix + self.whiteboard_prefix_delta
        mod = self.apply_adapter(raw)
        if batch_size > 1:
            mod = mod.expand(batch_size, -1, -1)
        if dtype is not None:
            mod = mod.to(dtype=dtype)
        return mod

    def build_gist_state(self, backbone: MCL4CEDBackbone, batch_size: int = 1) -> EncodedState:
        """Wrap the active whiteboard prefix as a position-invariant `EncodedState` (`n_prefix`)."""
        device = backbone.token_embedding.weight.device
        dtype = backbone.token_embedding.weight.dtype
        wb_hid = self.active_prefix_hidden(batch_size=batch_size, dtype=dtype).to(device=device)
        wb_normed = backbone.encoder_norm(wb_hid)
        wb_val = torch.ones(
            (batch_size, self.whiteboard_slots), device=device, dtype=torch.bool
        )
        zero = torch.zeros((), device=device, dtype=torch.float32)
        load = torch.full(
            (backbone.config.n_encoder_layers, backbone.config.num_experts),
            1.0 / max(1, backbone.config.num_experts),
            device=device,
            dtype=torch.float32,
        )
        return EncodedState(
            hidden=wb_normed,
            valid=wb_val,
            aux_loss=zero,
            router_z_loss=zero,
            expert_load=load,
            gist_prefix_len=0,
        )


class COEPlasticityController:
    """COE-Gated Transactional Fast-Weight Plasticity ('Einstein on a Whiteboard' Engine).

    Solves the two classic failure modes of test-time plasticity:
    1. **Volatility / Catastrophic Forgetting:** All core weights (`backbone` and `coe`) remain
       100% frozen. Gradients only update a `QuarantinedWhiteboardState` bounded by a hard
       Frobenius trust region with instant $O(1)$ rollback.
    2. **Closed-Loop Collapse / Wireheading:** Instead of training on ungrounded self-sampled
       token cross-entropy, updates are driven by three structural conservation laws:
       - **Orthogonal Expert Coalition Path-Invariance:** Disjoint expert coalitions (`Coalition A`
         vs `Coalition B`) must commute to the same latent conclusion in `COE` Pearson space.
       - **Forward-Inverse Abductive Cycle Consistency:** The conclusion + whiteboard state must
         reconstruct the original premise state via `AttentiveMatchedFilterPool` in reverse.
       - **System-2 $\\to$ System-1 Latent Crystallization:** Pass-1 fast intuition is distilled
         toward the multi-pass recurrent fixed point $\\text{sg}(z^{(R)})$.
       Every update is gated before and after by `COE` (`Association`, `Deduction`, `Evaluation`,
       `Metacognition`) plus the **Correspondence Principle** on pinned anchor invariants.
    """

    def __init__(
        self,
        coe: CognitiveOperatorEnsemble,
        *,
        whiteboard_slots: int = 8,
        lora_rank: int = 8,
        lora_alpha: float = 4.0,
        lr: float = 3e-2,
        inner_steps: int = 2,
        max_prefix_norm: float = 0.50,
        max_adapter_frobenius_norm: float = 0.25,
        min_novelty: float = 0.02,
        min_entropy_floor: float = 0.40,
        max_anchor_drift: float = 0.08,
        coe_score_tolerance: float = 0.01,
    ) -> None:
        self.coe = coe
        self.backbone = coe.backbone
        device = self.backbone.token_embedding.weight.device
        self.whiteboard = QuarantinedWhiteboardState(
            self.backbone.config.d_model,
            whiteboard_slots=whiteboard_slots,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            max_prefix_norm=max_prefix_norm,
            max_adapter_frobenius_norm=max_adapter_frobenius_norm,
            device=device,
            dtype=torch.float32,
        )
        self.lr = float(lr)
        self.inner_steps = max(1, int(inner_steps))
        self.min_novelty = float(min_novelty)
        self.min_entropy_floor = float(min_entropy_floor)
        self.max_anchor_drift = float(max_anchor_drift)
        self.coe_score_tolerance = float(coe_score_tolerance)
        self.history_features: Tensor | None = None
        self.anchor_features_ref: Tensor | None = None
        self.anchor_ids: Tensor | None = None
        self.anchor_lengths: Tensor | None = None
        self.committed_steps: int = 0
        self.rolled_back_steps: int = 0

    @torch.no_grad()
    def register_anchor_invariants(
        self,
        anchor_ids: Tensor,
        anchor_lengths: Tensor | None = None,
    ) -> Tensor:
        """Pin verified anchor statements (Correspondence Principle) that plastic updates must never degrade."""
        device = self.backbone.token_embedding.weight.device
        a_ids = anchor_ids.to(device=device, dtype=torch.long)
        if a_ids.ndim == 1:
            a_ids = a_ids.unsqueeze(0)
        if anchor_lengths is None:
            a_lens = (a_ids != self.backbone.config.pad_id).sum(dim=-1).clamp_min(1)
        else:
            a_lens = anchor_lengths.to(device=device, dtype=torch.long)
        self.anchor_ids = a_ids
        self.anchor_lengths = a_lens
        self.anchor_features_ref = self._compute_anchor_features(a_ids, a_lens).detach().clone()
        return self.anchor_features_ref

    def _compute_anchor_features(self, a_ids: Tensor, a_lens: Tensor) -> Tensor:
        width = int(a_lens.max().item())
        trimmed = a_ids[:, :width].long()
        valid = torch.arange(width, device=a_ids.device)[None, :] < a_lens[:, None]
        enc = self.backbone.encode_state(trimmed, valid, causal=True)
        mod_hid = self.whiteboard.apply_adapter(enc.hidden, valid)
        pooled = self.coe.pooler.pool_sequence(
            mod_hid,
            trimmed,
            valid,
            strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
            bos_id=self.coe.config.bos_id,
            eos_id=self.coe.config.eos_id,
        )
        normed = self.coe.pooler.cand_norm(pooled.float()).to(pooled.dtype)
        proj = self.coe.output_projection(normed).float()
        centered = proj - proj.mean(dim=-1, keepdim=True)
        return F.normalize(centered, dim=-1, eps=self.coe.config.eps)

    @torch.no_grad()
    def compute_anchor_drift(self) -> float:
        """Return $1 - \\text{Pearson}(\\text{anchor}_{\\text{now}}, \\text{anchor}_{\\text{ref}})$ on pinned invariants."""
        if (
            self.anchor_ids is None
            or self.anchor_lengths is None
            or self.anchor_features_ref is None
        ):
            return 0.0
        cur_feat = self._compute_anchor_features(self.anchor_ids, self.anchor_lengths)
        sim = pearson_similarity(cur_feat, self.anchor_features_ref, eps=self.coe.config.eps)
        return float((1.0 - sim.mean()).clamp_min(0.0).item())

    def _evaluate_whiteboard_objectives(
        self,
        s_trim: Tensor,
        s_valid: Tensor,
        base_encoded_s: EncodedState,
        base_state_target_proj: Tensor,
        h_trim: Tensor,
        h_valid: Tensor,
        *,
        recurrence_steps: int = 2,
        include_rule_readout: bool = True,
    ) -> dict[str, Tensor]:
        """Compute the 4 whiteboard conservation laws + COE epistemic score with gradients into `self.whiteboard`."""
        batch = s_trim.shape[0]
        owner_rows = torch.arange(batch, device=s_trim.device)
        wb_prefix = self.whiteboard.active_prefix_hidden(
            batch_size=batch, dtype=base_encoded_s.hidden.dtype
        )
        wb_normed = self.backbone.encoder_norm(wb_prefix)
        wb_valid = torch.ones(
            (batch, self.whiteboard.whiteboard_slots),
            device=s_trim.device,
            dtype=torch.bool,
        )
        mod_s_hid = self.whiteboard.apply_adapter(base_encoded_s.hidden, s_valid)
        mod_s_normed = self.backbone.encoder_norm(mod_s_hid).masked_fill(
            ~s_valid.unsqueeze(-1), 0
        )

        # State for upper-CED cross-attention: [Quarantined Whiteboard Prefix || Fast-Modulated State]
        state_for_kv = EncodedState(
            hidden=torch.cat([wb_normed, mod_s_normed], dim=1),
            valid=torch.cat([wb_valid, s_valid], dim=1),
            aux_loss=base_encoded_s.aux_loss,
            router_z_loss=base_encoded_s.router_z_loss,
            expert_load=base_encoded_s.expert_load,
            gist_prefix_len=0,
        )

        num_exp = self.backbone.config.num_experts
        coalition_a = [i for i in range(num_exp) if i % 2 == 0]
        coalition_b = [i for i in range(num_exp) if i % 2 == 1] or coalition_a

        saved_dense_indices = self.backbone.config.dense_expert_indices
        saved_dense_k = self.backbone.config.dense_subset_k
        saved_combine = self.backbone.config.combine_dense_and_sparse
        try:
            # 1. Orthogonal Expert Coalition A vs Coalition B Path-Invariance
            self.backbone.set_dense_expert_subset(
                coalition_a, dense_subset_k=0, combine_dense_and_sparse=False
            )
            dec_a = self.backbone.decode_candidates(
                state_for_kv,
                h_trim,
                h_valid,
                owner_rows=owner_rows,
                recurrence_steps=max(2, recurrence_steps),
                causal_local=True,
            )
            self.backbone.set_dense_expert_subset(
                coalition_b, dense_subset_k=0, combine_dense_and_sparse=False
            )
            dec_b = self.backbone.decode_candidates(
                state_for_kv,
                h_trim,
                h_valid,
                owner_rows=owner_rows,
                recurrence_steps=max(2, recurrence_steps),
                causal_local=True,
            )
        finally:
            self.backbone.set_dense_expert_subset(
                saved_dense_indices,
                dense_subset_k=saved_dense_k,
                combine_dense_and_sparse=saved_combine,
            )

        mod_h_a = self.whiteboard.apply_adapter(dec_a.hidden, h_valid)
        mod_h_b = self.whiteboard.apply_adapter(dec_b.hidden, h_valid)
        mod_anc_a = (
            self.whiteboard.apply_adapter(dec_a.anchor, h_valid)
            if dec_a.anchor is not None
            else None
        )
        mod_anc_b = (
            self.whiteboard.apply_adapter(dec_b.anchor, h_valid)
            if dec_b.anchor is not None
            else None
        )
        mod_h_mean = 0.5 * (mod_h_a + mod_h_b)

        s_ev_mask = self.coe.pooler._state_evidence_mask(
            s_trim,
            s_valid,
            strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
            bos_id=self.coe.config.bos_id,
            eos_id=self.coe.config.eos_id,
        )
        s_glob = self.coe.pooler.state_norm(
            self.coe.pooler.pool_sequence(
                base_encoded_s.hidden, s_trim, s_ev_mask, strip_boundary=False
            ).float()
        ).to(base_encoded_s.hidden.dtype)
        rule_delta = self.whiteboard.apply_rule_readout(
            s_glob,
            token_ids=s_trim,
            valid_mask=s_ev_mask,
            token_embedding=self.backbone.token_embedding,
        )
        has_rule = include_rule_readout and (
            float(torch.linalg.matrix_norm(self.whiteboard.rule_val_B.float(), ord="fro").item())
            > 1e-6
        )

        ps_a, cr_a = self.coe.pooler.matched_pair_representations(
            mod_s_normed,
            s_trim,
            s_valid,
            mod_h_a,
            h_trim,
            h_valid,
            owner_rows,
            cand_anchor=mod_anc_a,
            strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
            bos_id=self.coe.config.bos_id,
            eos_id=self.coe.config.eos_id,
        )
        ps_b, cr_b = self.coe.pooler.matched_pair_representations(
            mod_s_normed,
            s_trim,
            s_valid,
            mod_h_b,
            h_trim,
            h_valid,
            owner_rows,
            cand_anchor=mod_anc_b,
            strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
            bos_id=self.coe.config.bos_id,
            eos_id=self.coe.config.eos_id,
        )
        ps_a = self.whiteboard.apply_adapter(ps_a)
        ps_b = self.whiteboard.apply_adapter(ps_b)
        cr_a = self.whiteboard.apply_adapter(cr_a)
        cr_b = self.whiteboard.apply_adapter(cr_b)
        if has_rule:
            r_cent_eval = rule_delta.float() - rule_delta.float().mean(dim=-1, keepdim=True)
            r_unit_eval = F.normalize(r_cent_eval, dim=-1, eps=1e-6)
            r_scaled_eval = (
                r_unit_eval * math.sqrt(float(self.backbone.config.d_model))
            ).to(ps_a.dtype)
        else:
            r_scaled_eval = None

        proj_a = self.coe.output_projection(cr_a)
        proj_b = self.coe.output_projection(cr_b)
        coalition_sim = pearson_similarity(proj_a, proj_b, eps=self.coe.config.eps).mean()
        coalition_loss = (1.0 - coalition_sim) + 0.25 * F.mse_loss(mod_h_a.float(), mod_h_b.float())

        # 2. Forward-Inverse Abductive Cycle Consistency (Hypothesis + Whiteboard -> Reconstruct Premise State)
        _, inv_reconstructed_s = self.coe.pooler.matched_pair_representations(
            mod_h_mean,
            h_trim,
            h_valid,
            mod_s_normed,
            s_trim,
            s_valid,
            owner_rows,
            strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
            bos_id=self.coe.config.bos_id,
            eos_id=self.coe.config.eos_id,
        )
        inv_reconstructed_s = self.whiteboard.apply_adapter(inv_reconstructed_s)
        inv_s_proj = self.coe.state_projection(inv_reconstructed_s)
        cycle_sim = pearson_similarity(
            inv_s_proj, base_state_target_proj.detach(), eps=self.coe.config.eps
        ).mean()
        cycle_loss = 1.0 - cycle_sim

        # 3. System-2 -> System-1 Latent Crystallization (Pass 1 fast state -> sg(Pass R deep state))
        if len(dec_a.step_states) > 1:
            s1_fast = self.whiteboard.apply_adapter(dec_a.step_states[0], h_valid)
            s_deep_target = 0.5 * (dec_a.step_states[-1].detach() + dec_b.step_states[-1].detach())
            crystallize_loss = F.smooth_l1_loss(s1_fast.float(), s_deep_target.float())
        else:
            crystallize_loss = coalition_loss.new_zeros(())

        # 4. COE Epistemic Operators
        ps_mean = 0.5 * (ps_a + ps_b)
        cr_mean = 0.5 * (cr_a + cr_b)
        s_proj = self.coe.state_projection(ps_mean)
        c_proj = self.coe.output_projection(cr_mean)
        if has_rule and r_scaled_eval is not None:
            rule_only_proj = self.coe.state_projection(r_scaled_eval)
            anc_eval = (
                dec_a.anchor
                if dec_a.anchor is not None
                else 0.5 * (dec_a.hidden + dec_b.hidden)
            )
            c_anc_eval = self.coe.pooler.cand_norm(
                self.coe.pooler.pool_sequence(
                    anc_eval,
                    h_trim,
                    h_valid,
                    strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                    bos_id=self.coe.config.bos_id,
                    eos_id=self.coe.config.eos_id,
                ).float()
            ).to(cr_mean.dtype)
            c_anc_proj = self.coe.output_projection(c_anc_eval)
            r_sim_ctx = self.coe._similarity(rule_only_proj, c_anc_proj)
            h_ev_mask_eval = self.coe.pooler._content_mask(
                h_trim,
                h_valid,
                strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                bos_id=self.coe.config.bos_id,
                eos_id=self.coe.config.eos_id,
            )
            r_sim_holo = self.whiteboard.score_holo_candidates(
                s_glob,
                s_trim,
                s_ev_mask,
                h_trim,
                h_ev_mask_eval,
                owner_rows,
                self.backbone.token_embedding,
            )
            if r_sim_holo is not None:
                r_sim_eval = (
                    (0.55 * r_sim_ctx + 0.45 * r_sim_holo.to(r_sim_ctx.dtype)) * 4.5
                ).clamp(-1.0, 1.0)
            else:
                r_sim_eval = (r_sim_ctx * 4.5).clamp(-1.0, 1.0)
            pair_sim = 0.20 * self.coe._similarity(s_proj, c_proj) + 0.80 * r_sim_eval
        else:
            pair_sim = self.coe._similarity(s_proj, c_proj)
        eval_logits = pair_sim.float() * self.coe.assoc_alpha.float() + self.coe.assoc_beta.float()
        eval_util = eval_logits.sigmoid().mean()

        scale = self.coe._scale()
        joint = torch.cat([ps_mean, cr_mean], dim=-1)
        ded_q = self.coe.deduction_query_proj(joint)
        ded_sims = pearson_similarity(
            ded_q.unsqueeze(1),
            self.coe.deduction_prototypes.unsqueeze(0),
            eps=self.coe.config.eps,
        )
        ded_probs = (ded_sims.float() * scale).softmax(dim=-1)
        p_entailed = ded_probs[:, 0].mean()
        p_contradicted = ded_probs[:, 1].mean()

        meta_q = self.coe.meta_query_proj(joint)
        meta_sims = pearson_similarity(
            meta_q.unsqueeze(1),
            self.coe.meta_prototypes.unsqueeze(0),
            eps=self.coe.config.eps,
        )
        meta_probs = (
            meta_sims.float() * scale + self.coe.meta_bias.float().unsqueeze(0)
        ).sigmoid()
        p_correct = meta_probs[:, 0].mean()

        coe_score = 0.40 * eval_util + 0.35 * (p_entailed - 0.5 * p_contradicted) + 0.25 * p_correct
        epistemic_loss = (1.0 - coe_score).clamp_min(0.0)

        # 5. Predictive Token Entropy Barrier (prevents mode collapse)
        logits = self.backbone.lm_head(mod_h_mean).float()
        if self.backbone.config.logits_scaling != 1.0:
            logits = logits / float(self.backbone.config.logits_scaling)
        probs = logits.softmax(dim=-1)
        tok_ent = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
        valid_ent = (tok_ent * h_valid.float()).sum() / h_valid.float().sum().clamp_min(1.0)
        entropy_barrier = F.relu(self.min_entropy_floor - valid_ent)

        total_whiteboard_loss = (
            0.30 * coalition_loss
            + 0.25 * cycle_loss
            + 0.15 * crystallize_loss
            + 0.30 * epistemic_loss
            + 0.50 * entropy_barrier
        )
        return {
            "total_loss": total_whiteboard_loss,
            "coalition_agreement": coalition_sim,
            "cycle_consistency": cycle_sim,
            "crystallize_loss": crystallize_loss,
            "coe_score": coe_score,
            "entropy": valid_ent,
        }

    def step(
        self,
        state_ids: Tensor,
        hypothesis_ids: Tensor,
        *,
        state_lengths: Tensor | None = None,
        hypothesis_lengths: Tensor | None = None,
        recurrence_steps: int = 2,
    ) -> PlasticityTransactionReport:
        """Execute a single COE-gated transactional plasticity update on the quarantined whiteboard."""
        device = self.backbone.token_embedding.weight.device
        s_ids = state_ids.to(device=device, dtype=torch.long)
        h_ids = hypothesis_ids.to(device=device, dtype=torch.long)
        if s_ids.ndim == 1:
            s_ids = s_ids.unsqueeze(0)
        if h_ids.ndim == 1:
            h_ids = h_ids.unsqueeze(0)
        if s_ids.shape[0] != h_ids.shape[0]:
            raise ValueError("state_ids and hypothesis_ids must have matching batch size")

        if state_lengths is None:
            s_lens = (s_ids != self.backbone.config.pad_id).sum(dim=-1).clamp_min(1)
        else:
            s_lens = state_lengths.to(device=device, dtype=torch.long)
        if hypothesis_lengths is None:
            h_lens = (h_ids != self.backbone.config.pad_id).sum(dim=-1).clamp_min(1)
        else:
            h_lens = hypothesis_lengths.to(device=device, dtype=torch.long)

        s_width = int(s_lens.max().item())
        h_width = int(h_lens.max().item())
        s_trim = s_ids[:, :s_width].long()
        h_trim = h_ids[:, :h_width].long()
        s_valid = torch.arange(s_width, device=device)[None, :] < s_lens[:, None]
        h_valid = torch.arange(h_width, device=device)[None, :] < h_lens[:, None]

        # Gate 1: COE Association Novelty Input Filter
        with torch.no_grad():
            assoc_out = self.coe.forward_association_matrix(
                h_trim, h_lens, history_buffer=self.history_features
            )
            novelty = (
                float(assoc_out.novelty.mean().item())
                if assoc_out.novelty is not None
                else 1.0
            )
            base_encoded_s = self.backbone.encode_state(s_trim, s_valid, causal=True)
            if float(self.whiteboard.base_prefix.abs().sum().item()) == 0.0:
                self.whiteboard.initialize_base_prefix(base_encoded_s.hidden)
            base_s_pooled = self.coe.pooler.pool_sequence(
                base_encoded_s.hidden,
                s_trim,
                s_valid,
                strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                bos_id=self.coe.config.bos_id,
                eos_id=self.coe.config.eos_id,
            )
            base_state_target_proj = self.coe.state_projection(
                self.coe.pooler.state_norm(base_s_pooled.float()).to(base_s_pooled.dtype)
            ).detach()

        if novelty < self.min_novelty:
            with torch.no_grad():
                m0 = self._evaluate_whiteboard_objectives(
                    s_trim,
                    s_valid,
                    base_encoded_s,
                    base_state_target_proj,
                    h_trim,
                    h_valid,
                    recurrence_steps=recurrence_steps,
                )
            return PlasticityTransactionReport(
                committed=False,
                status="skipped_low_novelty",
                whiteboard_loss_before=float(m0["total_loss"].item()),
                whiteboard_loss_after=float(m0["total_loss"].item()),
                coalition_agreement_before=float(m0["coalition_agreement"].item()),
                coalition_agreement_after=float(m0["coalition_agreement"].item()),
                cycle_consistency_before=float(m0["cycle_consistency"].item()),
                cycle_consistency_after=float(m0["cycle_consistency"].item()),
                coe_score_before=float(m0["coe_score"].item()),
                coe_score_after=float(m0["coe_score"].item()),
                entropy_after=float(m0["entropy"].item()),
                anchor_drift=self.compute_anchor_drift(),
                novelty=novelty,
                adapter_frobenius_norm=self.whiteboard.adapter_frobenius_norm(),
                prefix_delta_norm=self.whiteboard.prefix_delta_norm(),
            )

        # Freeze 100% of slow weights (`coe` + `backbone`) and enable gradients ONLY on `self.whiteboard`
        slow_grad_flags = [(p, p.requires_grad) for p in self.coe.parameters()]
        was_training = self.coe.training
        self.whiteboard.snapshot()
        try:
            self.coe.eval()
            for p, _ in slow_grad_flags:
                p.requires_grad_(False)
            for name, p in self.whiteboard.named_parameters():
                p.requires_grad_(not (name.startswith("rule_") or name.startswith("gen_")))

            with torch.no_grad():
                before = self._evaluate_whiteboard_objectives(
                    s_trim,
                    s_valid,
                    base_encoded_s,
                    base_state_target_proj,
                    h_trim,
                    h_valid,
                    recurrence_steps=recurrence_steps,
                )

            trainable_wb_params = [p for p in self.whiteboard.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable_wb_params, lr=self.lr, weight_decay=1e-4)
            with torch.enable_grad():
                for _ in range(self.inner_steps):
                    opt.zero_grad(set_to_none=True)
                    step_metrics = self._evaluate_whiteboard_objectives(
                        s_trim,
                        s_valid,
                        base_encoded_s,
                        base_state_target_proj,
                        h_trim,
                        h_valid,
                        recurrence_steps=recurrence_steps,
                    )
                    loss = step_metrics["total_loss"]
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.whiteboard.parameters(), max_norm=1.0)
                    opt.step()
                    self.whiteboard.project_to_trust_region()

            with torch.no_grad():
                after = self._evaluate_whiteboard_objectives(
                    s_trim,
                    s_valid,
                    base_encoded_s,
                    base_state_target_proj,
                    h_trim,
                    h_valid,
                    recurrence_steps=recurrence_steps,
                )
                drift = self.compute_anchor_drift()
        finally:
            for p, flag in slow_grad_flags:
                p.requires_grad_(flag)
            self.coe.train(was_training)

        loss_before = float(before["total_loss"].item())
        loss_after = float(after["total_loss"].item())
        coe_before = float(before["coe_score"].item())
        coe_after = float(after["coe_score"].item())
        ent_after = float(after["entropy"].item())

        # Gate 3: COE Synaptic Commit / Rollback Verification
        rollback_reason: str | None = None
        if not math.isfinite(loss_after) or loss_after >= loss_before:
            rollback_reason = "rolled_back_non_decreasing_loss"
        elif coe_after < coe_before - self.coe_score_tolerance:
            rollback_reason = "rolled_back_coe_score_regression"
        elif ent_after < self.min_entropy_floor:
            rollback_reason = "rolled_back_entropy_collapse"
        elif drift > self.max_anchor_drift:
            rollback_reason = "rolled_back_anchor_invariant_drift"

        if rollback_reason is not None:
            self.whiteboard.rollback()
            self.rolled_back_steps += 1
            return PlasticityTransactionReport(
                committed=False,
                status=rollback_reason,
                whiteboard_loss_before=loss_before,
                whiteboard_loss_after=loss_after,
                coalition_agreement_before=float(before["coalition_agreement"].item()),
                coalition_agreement_after=float(after["coalition_agreement"].item()),
                cycle_consistency_before=float(before["cycle_consistency"].item()),
                cycle_consistency_after=float(after["cycle_consistency"].item()),
                coe_score_before=coe_before,
                coe_score_after=coe_after,
                entropy_after=ent_after,
                anchor_drift=drift,
                novelty=novelty,
                adapter_frobenius_norm=self.whiteboard.adapter_frobenius_norm(),
                prefix_delta_norm=self.whiteboard.prefix_delta_norm(),
            )

        # Commit succeeded: update history buffer for novelty tracking
        self.committed_steps += 1
        with torch.no_grad():
            new_feats = assoc_out.features.detach()
            if self.history_features is None:
                self.history_features = new_feats
            else:
                self.history_features = torch.cat([self.history_features, new_feats], dim=0)[-128:]

        return PlasticityTransactionReport(
            committed=True,
            status="committed",
            whiteboard_loss_before=loss_before,
            whiteboard_loss_after=loss_after,
            coalition_agreement_before=float(before["coalition_agreement"].item()),
            coalition_agreement_after=float(after["coalition_agreement"].item()),
            cycle_consistency_before=float(before["cycle_consistency"].item()),
            cycle_consistency_after=float(after["cycle_consistency"].item()),
            coe_score_before=coe_before,
            coe_score_after=coe_after,
            entropy_after=ent_after,
            anchor_drift=drift,
            novelty=novelty,
            adapter_frobenius_norm=self.whiteboard.adapter_frobenius_norm(),
            prefix_delta_norm=self.whiteboard.prefix_delta_norm(),
        )

    @staticmethod
    def _split_clauses_left_aligned(
        token_ids: Tensor,
        valid_mask: Tensor,
        pad_id: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split each row's valid tokens at its midpoint into left-aligned `(u_ids, u_lens, h_ids, h_lens)`."""
        b = token_ids.shape[0]
        device = token_ids.device
        u_rows: list[Tensor] = []
        h_rows: list[Tensor] = []
        for i in range(b):
            toks = token_ids[i][valid_mask[i]]
            if toks.numel() < 4:
                toks = token_ids[i, : max(4, min(token_ids.shape[1], 8))]
            mid = max(2, int(toks.numel()) // 2)
            u_rows.append(toks[:mid])
            h_rows.append(toks[mid:])
        u_max = max(int(r.numel()) for r in u_rows)
        h_max = max(int(r.numel()) for r in h_rows)
        fill_val = max(0, int(pad_id))
        u_ids = torch.full((b, u_max), fill_val, device=device, dtype=torch.long)
        h_ids = torch.full((b, h_max), fill_val, device=device, dtype=torch.long)
        u_lens = torch.zeros(b, device=device, dtype=torch.long)
        h_lens = torch.zeros(b, device=device, dtype=torch.long)
        for i, (ur, hr) in enumerate(zip(u_rows, h_rows)):
            u_ids[i, : ur.numel()] = ur
            h_ids[i, : hr.numel()] = hr
            u_lens[i] = int(ur.numel())
            h_lens[i] = int(hr.numel())
        return u_ids, u_lens, h_ids, h_lens

    def absorb_context(
        self,
        context_ids: Tensor,
        context_lengths: Tensor | None = None,
        *,
        premise_ids: Tensor | None = None,
        premise_lengths: Tensor | None = None,
        conclusion_ids: Tensor | None = None,
        conclusion_lengths: Tensor | None = None,
        recurrence_steps: int = 2,
    ) -> PlasticityTransactionReport:
        """Learn on the fly from a novel domain specification, rule set, or context document.

        Combines:
        1. **Grounded Predictive Surprise ($\\mathcal{L}_{\\text{LM}}$):** Next-token prediction error
           on the new specification via `forward_with_whiteboard`.
        2. **In-Document Contrastive `COE` Clause Alignment:** Aligns each specification clause's
           premise with its true continuation/conclusion in `COE` space over other clauses.
        3. **4 Whiteboard Conservation Laws + Correspondence Principle Anchor Gate:** Ensures
           orthogonal expert coalition consensus, inverse abductive cycle consistency, entropy
           floor, and bounded anchor drift with $O(1)$ rollback.
        """
        device = self.backbone.token_embedding.weight.device
        pad_id = self.backbone.config.pad_id
        c_ids = context_ids.to(device=device, dtype=torch.long)
        if c_ids.ndim == 1:
            c_ids = c_ids.unsqueeze(0)
        if context_lengths is None:
            c_lens = (c_ids != pad_id).sum(dim=-1).clamp_min(2)
        else:
            c_lens = context_lengths.to(device=device, dtype=torch.long).clamp_min(2)

        c_width = int(c_lens.max().item())
        c_trim = c_ids[:, :c_width].long()
        c_valid = torch.arange(c_width, device=device)[None, :] < c_lens[:, None]

        if premise_ids is not None and conclusion_ids is not None:
            s_trim = premise_ids.to(device=device, dtype=torch.long)
            h_trim = conclusion_ids.to(device=device, dtype=torch.long)
            if s_trim.ndim == 1:
                s_trim = s_trim.unsqueeze(0)
            if h_trim.ndim == 1:
                h_trim = h_trim.unsqueeze(0)
            s_lens = (
                (s_trim != pad_id).sum(dim=-1).clamp_min(1)
                if premise_lengths is None
                else premise_lengths.to(device=device, dtype=torch.long)
            )
            h_lens = (
                (h_trim != pad_id).sum(dim=-1).clamp_min(1)
                if conclusion_lengths is None
                else conclusion_lengths.to(device=device, dtype=torch.long)
            )
            s_trim = s_trim[:, : int(s_lens.max().item())]
            h_trim = h_trim[:, : int(h_lens.max().item())]
        else:
            # If single long document `[1, T]`, partition into contiguous clauses `[C, T_c]`
            if c_trim.shape[0] == 1 and c_width >= 16:
                toks = c_trim[0][c_valid[0]]
                n_clauses = min(6, max(2, int(toks.numel()) // 12))
                chunk_sz = max(4, int(toks.numel()) // n_clauses)
                clause_list = [
                    toks[i * chunk_sz : (i + 1) * chunk_sz]
                    for i in range(n_clauses)
                    if toks[i * chunk_sz : (i + 1) * chunk_sz].numel() >= 4
                ]
                max_cl = max(int(x.numel()) for x in clause_list)
                fill_val = max(0, int(pad_id))
                cl_mat = torch.full((len(clause_list), max_cl), fill_val, device=device, dtype=torch.long)
                cl_val = torch.zeros((len(clause_list), max_cl), device=device, dtype=torch.bool)
                for idx, cl in enumerate(clause_list):
                    cl_mat[idx, : cl.numel()] = cl
                    cl_val[idx, : cl.numel()] = True
                s_trim, s_lens, h_trim, h_lens = self._split_clauses_left_aligned(
                    cl_mat, cl_val, pad_id
                )
            else:
                s_ev_mask = self.coe.pooler._state_evidence_mask(
                    c_trim,
                    c_valid,
                    strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                    bos_id=self.coe.config.bos_id,
                    eos_id=self.coe.config.eos_id,
                )
                s_trim, s_lens, h_trim, h_lens = self._split_clauses_left_aligned(
                    c_trim, s_ev_mask, pad_id
                )

        s_valid = torch.arange(s_trim.shape[1], device=device)[None, :] < s_lens[:, None]
        h_valid = torch.arange(h_trim.shape[1], device=device)[None, :] < h_lens[:, None]

        with torch.no_grad():
            assoc_out = self.coe.forward_association_matrix(
                h_trim, h_lens, history_buffer=self.history_features
            )
            novelty = (
                float(assoc_out.novelty.mean().item())
                if assoc_out.novelty is not None
                else 1.0
            )
            base_encoded_s = self.backbone.encode_state(s_trim, s_valid, causal=True)
            if float(self.whiteboard.base_prefix.abs().sum().item()) == 0.0:
                self.whiteboard.initialize_base_prefix(base_encoded_s.hidden)
            base_s_pooled = self.coe.pooler.pool_sequence(
                base_encoded_s.hidden,
                s_trim,
                s_valid,
                strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                bos_id=self.coe.config.bos_id,
                eos_id=self.coe.config.eos_id,
            )
            base_state_target_proj = self.coe.state_projection(
                self.coe.pooler.state_norm(base_s_pooled.float()).to(base_s_pooled.dtype)
            ).detach()

        lm_in = c_trim[:, :-1]
        lm_tgt = c_trim[:, 1:].masked_fill(~c_valid[:, 1:], -100)

        def _joint_eval(include_rule_readout: bool = True) -> dict[str, Tensor]:
            wb_m = self._evaluate_whiteboard_objectives(
                s_trim,
                s_valid,
                base_encoded_s,
                base_state_target_proj,
                h_trim,
                h_valid,
                recurrence_steps=recurrence_steps,
                include_rule_readout=include_rule_readout,
            )
            if lm_in.shape[1] >= 1 and (lm_tgt != -100).any():
                lm_out = self.coe.forward_with_whiteboard(
                    lm_in,
                    targets=lm_tgt,
                    whiteboard=self.whiteboard,
                    recurrence_steps=recurrence_steps,
                    include_generative_readout=True,
                )
                lm_loss = lm_out.ce_loss if lm_out.ce_loss is not None else wb_m["total_loss"].new_zeros(())
            else:
                lm_loss = wb_m["total_loss"].new_zeros(())
            wb_m["total_loss"] = wb_m["total_loss"] + 0.50 * lm_loss
            return wb_m

        slow_grad_flags = [(p, p.requires_grad) for p in self.coe.parameters()]
        was_training = self.coe.training
        self.whiteboard.snapshot()
        try:
            self.coe.eval()
            for p, _ in slow_grad_flags:
                p.requires_grad_(False)
            for name, p in self.whiteboard.named_parameters():
                p.requires_grad_(not (name.startswith("rule_") or name.startswith("gen_")))

            with torch.no_grad():
                before = _joint_eval(include_rule_readout=False)

                # 1. Analytically factor Biorthogonal Hopfield-Kanerva rule readout (S_f -> Q_dual -> Z_target)
                saved_d_idx = self.backbone.config.dense_expert_indices
                saved_d_k = self.backbone.config.dense_subset_k
                saved_d_comb = self.backbone.config.combine_dense_and_sparse
                try:
                    self.backbone.set_dense_expert_subset(
                        [], dense_subset_k=0, combine_dense_and_sparse=False
                    )
                    s_ev_mask_abs = self.coe.pooler._state_evidence_mask(
                        s_trim,
                        s_valid,
                        strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                        bos_id=self.coe.config.bos_id,
                        eos_id=self.coe.config.eos_id,
                    )
                    enc_s_canon = self.backbone.encode_state(s_trim, s_valid, causal=True)
                    s_glob_abs = self.coe.pooler.state_norm(
                        self.coe.pooler.pool_sequence(
                            enc_s_canon.hidden, s_trim, s_ev_mask_abs, strip_boundary=False
                        ).float()
                    )
                    s_f_new = self.whiteboard.compute_rule_key(
                        s_glob_abs,
                        token_ids=s_trim,
                        valid_mask=s_ev_mask_abs,
                        token_embedding=self.backbone.token_embedding,
                    )
                    s_holo_new = self.whiteboard.compute_holographic_sketch(
                        s_trim, s_ev_mask_abs, self.backbone.token_embedding
                    )

                    enc_h_abs = self.backbone.encode_state(h_trim, h_valid, causal=True)
                    anc_h_abs = (
                        enc_h_abs.raw_hidden
                        if (
                            self.backbone.config.preserve_residual_across_stacks
                            and enc_h_abs.raw_hidden is not None
                        )
                        else enc_h_abs.hidden
                    )
                    c_anc_abs = self.coe.pooler.cand_norm(
                        self.coe.pooler.pool_sequence(
                            anc_h_abs,
                            h_trim,
                            h_valid,
                            strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                            bos_id=self.coe.config.bos_id,
                            eos_id=self.coe.config.eos_id,
                        ).float()
                    ).to(enc_h_abs.hidden.dtype)
                finally:
                    self.backbone.set_dense_expert_subset(
                        saved_d_idx,
                        dense_subset_k=saved_d_k,
                        combine_dense_and_sparse=saved_d_comb,
                    )

                h_ev_mask_abs = self.coe.pooler._content_mask(
                    h_trim,
                    h_valid,
                    strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                    bos_id=self.coe.config.bos_id,
                    eos_id=self.coe.config.eos_id,
                )
                c_holo_new = self.whiteboard.compute_holographic_sketch(
                    h_trim, h_ev_mask_abs, self.backbone.token_embedding
                )

                v_sz = int(self.backbone.token_embedding.num_embeddings)
                p_txts_abs, _, _ = self.whiteboard.decode_clean_texts(
                    s_trim, s_ev_mask_abs, v_sz
                )
                c_txts_abs, _, _ = self.whiteboard.decode_clean_texts(
                    h_trim, h_ev_mask_abs, v_sz
                )

                c_proj_abs = self.coe.output_projection(c_anc_abs).float()
                c_unit_new = F.normalize(
                    c_proj_abs - c_proj_abs.mean(dim=-1, keepdim=True), dim=-1, eps=1e-6
                )
                self.whiteboard.append_episodic_rules(
                    s_f_new,
                    c_unit_new,
                    s_holo_new=s_holo_new,
                    c_holo_new=c_holo_new,
                    premise_texts=p_txts_abs,
                    conclusion_texts=c_txts_abs,
                )
                self._solve_closed_form_rule_banks(self.whiteboard)

            trainable_wb = [p for p in self.whiteboard.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable_wb, lr=self.lr, weight_decay=1e-4)
            with torch.enable_grad():
                for _ in range(self.inner_steps):
                    opt.zero_grad(set_to_none=True)
                    step_metrics = _joint_eval(include_rule_readout=True)
                    loss = step_metrics["total_loss"]
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_wb, max_norm=1.0)
                    opt.step()
                    self.whiteboard.project_to_trust_region()

            with torch.no_grad():

                # 2. Analytically factor Surprise-Weighted Natural-Gradient Generative Readout (H_v -> Delta_LM)
                if lm_in.shape[1] >= 1 and (lm_tgt != -100).any():
                    valid_lm_mask = lm_tgt != -100
                    in_valid = lm_in != pad_id
                    if not in_valid.any(dim=-1).all():
                        in_valid = in_valid.clone()
                        in_valid[:, 0] = True
                    is_dec_only_donor = (
                        self.backbone.config.n_encoder_layers == 0
                        and not getattr(self.backbone.config, "tie_word_embeddings", True)
                    )
                    wb_gist_gen = (
                        None
                        if is_dec_only_donor
                        else self.whiteboard.build_gist_state(
                            self.backbone, batch_size=lm_in.shape[0]
                        )
                    )
                    enc_low_gen = self.backbone.encode_state(lm_in, in_valid, causal=True)
                    mod_low_gen = self.whiteboard.apply_adapter(enc_low_gen.hidden, in_valid)
                    prior_gen = mod_low_gen.mean(dim=1)
                    use_ponder_gen = self.coe.config.depth_ponder
                    steps_gen = (
                        max(2, self.coe.config.recurrence_steps)
                        if use_ponder_gen
                        else self.coe.config.recurrence_steps
                    ) if recurrence_steps is None else recurrence_steps
                    enc_gen = self.backbone.encode(
                        lm_in,
                        recurrence_steps=steps_gen,
                        latent_prior=prior_gen,
                        latent_prior_scale=0.02 if is_dec_only_donor else 0.15,
                        recurrent_kv=self.coe.config.recurrent_kv,
                        depth_ponder=use_ponder_gen,
                        gist_state=wb_gist_gen,
                        prior_start_pos=0,
                    )
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        h_gen = enc_gen.hidden.float()
                        w_lm = self.backbone.lm_head.weight.float()
                        lm_scale = float(self.backbone.config.logits_scaling)
                        base_lm_logits = F.linear(h_gen, w_lm) / lm_scale
                        h_v = h_gen[valid_lm_mask]
                        y_v = lm_tgt[valid_lm_mask]
                        probs_v = F.softmax(base_lm_logits[valid_lm_mask], dim=-1)
                        p_true = probs_v.gather(-1, y_v.unsqueeze(-1))
                        g_v = F.embedding(y_v, w_lm) - (probs_v @ w_lm)
                        g_unit = F.normalize(g_v, dim=-1, eps=1e-6) * (1.0 - p_true).sqrt()
                        h_rms = h_v.norm(dim=-1, keepdim=True).mean().clamp_min(1e-4)
                        eff_delta_scale = (
                            min(float(0.70 * h_rms.item()), 1.25 * lm_scale)
                            if not getattr(self.backbone.config, "tie_word_embeddings", True)
                            else (0.70 * h_rms)
                        )
                        delta_target = g_unit * eff_delta_scale

                        old_gen_w = 0.50 * (
                            self.whiteboard.gen_key_A.float() @ self.whiteboard.gen_val_B.float()
                        ) * self.whiteboard.scale
                        res_target = delta_target - (h_v @ old_gen_w)
                        gram_h = h_v.T @ h_v + (8e-3 * float(h_v.shape[0])) * torch.eye(
                            h_v.shape[1], device=device, dtype=torch.float32
                        )
                        w_gen_full = old_gen_w + torch.linalg.solve(gram_h, h_v.T @ res_target)
                        u_g, s_g, vh_g = torch.linalg.svd(
                            w_gen_full / max(1e-6, self.whiteboard.scale), full_matrices=False
                        )
                        r_gen = min(int(s_g.numel()), self.whiteboard.gen_rank)
                        sqrt_sg = s_g[:r_gen].clamp_min(0.0).sqrt()
                        g_a = u_g[:, :r_gen] * sqrt_sg.unsqueeze(0)
                        g_b = sqrt_sg.unsqueeze(1) * vh_g[:r_gen, :]
                        self.whiteboard.gen_key_A.zero_()
                        self.whiteboard.gen_val_B.zero_()
                        self.whiteboard.gen_key_A[:, :r_gen].copy_(
                            g_a.to(self.whiteboard.gen_key_A.dtype)
                        )
                        self.whiteboard.gen_val_B[:r_gen, :].copy_(
                            g_b.to(self.whiteboard.gen_val_B.dtype)
                        )

                self.whiteboard.project_to_trust_region()
                after = _joint_eval()
                drift = self.compute_anchor_drift()
        finally:
            for p, flag in slow_grad_flags:
                p.requires_grad_(flag)
            self.coe.train(was_training)

        loss_before = float(before["total_loss"].item())
        loss_after = float(after["total_loss"].item())
        coe_before = float(before["coe_score"].item())
        coe_after = float(after["coe_score"].item())
        ent_after = float(after["entropy"].item())

        rollback_reason: str | None = None
        if not math.isfinite(loss_after) or loss_after >= loss_before:
            rollback_reason = "rolled_back_non_decreasing_loss"
        elif coe_after < coe_before - self.coe_score_tolerance:
            rollback_reason = "rolled_back_coe_score_regression"
        elif ent_after < self.min_entropy_floor:
            rollback_reason = "rolled_back_entropy_collapse"
        elif drift > self.max_anchor_drift:
            rollback_reason = "rolled_back_anchor_invariant_drift"

        if rollback_reason is not None:
            self.whiteboard.rollback()
            self.rolled_back_steps += 1
            return PlasticityTransactionReport(
                committed=False,
                status=rollback_reason,
                whiteboard_loss_before=loss_before,
                whiteboard_loss_after=loss_after,
                coalition_agreement_before=float(before["coalition_agreement"].item()),
                coalition_agreement_after=float(after["coalition_agreement"].item()),
                cycle_consistency_before=float(before["cycle_consistency"].item()),
                cycle_consistency_after=float(after["cycle_consistency"].item()),
                coe_score_before=coe_before,
                coe_score_after=coe_after,
                entropy_after=ent_after,
                anchor_drift=drift,
                novelty=novelty,
                adapter_frobenius_norm=self.whiteboard.adapter_frobenius_norm(),
                prefix_delta_norm=self.whiteboard.prefix_delta_norm(),
            )

        self.committed_steps += 1
        return PlasticityTransactionReport(
            committed=True,
            status="committed",
            whiteboard_loss_before=loss_before,
            whiteboard_loss_after=loss_after,
            coalition_agreement_before=float(before["coalition_agreement"].item()),
            coalition_agreement_after=float(after["coalition_agreement"].item()),
            cycle_consistency_before=float(before["cycle_consistency"].item()),
            cycle_consistency_after=float(after["cycle_consistency"].item()),
            coe_score_before=coe_before,
            coe_score_after=coe_after,
            entropy_after=ent_after,
            anchor_drift=drift,
            novelty=novelty,
            adapter_frobenius_norm=self.whiteboard.adapter_frobenius_norm(),
            prefix_delta_norm=self.whiteboard.prefix_delta_norm(),
        )

    @torch.no_grad()
    def consolidate_whiteboard_to_mitotic_children(
        self,
        *,
        consolidation_rate: float = 0.25,
        reset_whiteboard_after: bool = False,
    ) -> dict[str, Any]:
        """Consolidate ('sleep-fold') verified whiteboard fast weights into mitotic child experts.

        Guarantees:
        1. **Parent Expert Bitwise Immutability (`0.0` delta):** Parent experts (`p_idx`) in every
           layer are 100% frozen and untouched (`parent_expert_max_delta == 0.0`).
        2. **Gram-Schmidt Orthogonal Child Specialization:** Folds the top-$r$ singular directions
           of the whiteboard adapter into the specialist channels of mitotic child experts (`c_idx`),
           orthogonalized against their parent expert.
        3. **Correspondence Principle Anchor Verification:** Verifies `compute_anchor_drift() <= max_anchor_drift`
           and rolls back child experts in $O(1)$ if anchor invariants are exceeded.
        """
        device = self.backbone.token_embedding.weight.device
        eff_wb = (
            (self.whiteboard.fast_lora_A.float() @ self.whiteboard.fast_lora_B.float())
            + 0.35 * (self.whiteboard.rule_key_A.float() @ self.whiteboard.rule_val_B.float())
        ) * self.whiteboard.scale
        wb_frob = float(eff_wb.norm().item())
        if wb_frob <= 1e-8:
            return {
                "consolidated": False,
                "status": "skipped_empty_whiteboard",
                "num_child_experts_updated": 0,
                "parent_expert_max_delta": 0.0,
                "child_expert_frobenius_delta": 0.0,
                "anchor_drift_after": self.compute_anchor_drift(),
            }

        u_w, s_w, vh_w = torch.linalg.svd(eff_wb, full_matrices=False)
        r_fold = min(int(s_w.numel()), self.whiteboard.lora_rank)
        sqrt_sw = (s_w[:r_fold].clamp_min(0.0) * float(consolidation_rate)).sqrt()
        in_dirs = (u_w[:, :r_fold] * sqrt_sw.unsqueeze(0)).T  # [r_fold, d_model]
        out_dirs = vh_w[:r_fold, :].T * sqrt_sw.unsqueeze(0)  # [d_model, r_fold]

        all_layers = list(self.backbone.encoder_layers) + list(self.backbone.decoder_layers)
        parent_snapshots: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        child_snapshots: list[tuple[nn.Parameter, Tensor, nn.Parameter, Tensor]] = []
        layer_pairs: list[tuple[Any, list[tuple[int, int]]]] = []

        for layer in all_layers:
            moe = layer.moe
            if moe.mitotic_pairs:
                pairs = [(int(p), int(c)) for p, c in moe.mitotic_pairs]
            else:
                pairs = [
                    (2 * k, 2 * k + 1)
                    for k in range(moe.num_experts // 2)
                ]
            layer_pairs.append((moe, pairs))
            for p_idx, c_idx in pairs:
                ep = moe.experts[p_idx]
                ec = moe.experts[c_idx]
                parent_snapshots.append(
                    (
                        ep.up_proj.weight,
                        ep.up_proj.weight.detach().clone(),
                        ep.down_proj.weight,
                        ep.down_proj.weight.detach().clone(),
                    )
                )
                child_snapshots.append(
                    (
                        ec.up_proj.weight,
                        ec.up_proj.weight.detach().clone(),
                        ec.down_proj.weight,
                        ec.down_proj.weight.detach().clone(),
                    )
                )

        child_frob_accum = 0.0
        num_children = 0
        for moe, pairs in layer_pairs:
            for p_idx, c_idx in pairs:
                ep = moe.experts[p_idx]
                ec = moe.experts[c_idx]
                d_ffn = ec.down_proj.weight.shape[1]
                r_ch = min(r_fold, d_ffn)
                if r_ch < 1:
                    continue
                p_val = ep.up_proj.weight[2 * d_ffn - r_ch :, :].float()
                p_dn = ep.down_proj.weight[:, d_ffn - r_ch :].float()

                # Gram-Schmidt project in_dirs / out_dirs orthogonal to parent expert's channels
                u_in = in_dirs[:r_ch, :].to(device=p_val.device, dtype=torch.float32)
                p_val_unit = F.normalize(p_val, dim=-1, eps=1e-6)
                u_in_ortho = u_in - (u_in * p_val_unit).sum(dim=-1, keepdim=True) * p_val_unit
                scale_in = p_val.norm(dim=-1, keepdim=True).mean().clamp_min(1e-4) * 0.15
                delta_up = F.normalize(u_in_ortho, dim=-1, eps=1e-6) * scale_in * sqrt_sw[:r_ch].unsqueeze(-1)

                v_out = out_dirs[:, :r_ch].to(device=p_dn.device, dtype=torch.float32)
                p_dn_unit = F.normalize(p_dn, dim=0, eps=1e-6)
                v_out_ortho = v_out - (v_out * p_dn_unit).sum(dim=0, keepdim=True) * p_dn_unit
                scale_out = p_dn.norm(dim=0, keepdim=True).mean().clamp_min(1e-4) * 0.15
                delta_dn = F.normalize(v_out_ortho, dim=0, eps=1e-6) * scale_out * sqrt_sw[:r_ch].unsqueeze(0)

                ec.up_proj.weight.data[2 * d_ffn - r_ch :, :].add_(
                    delta_up.to(ec.up_proj.weight.dtype)
                )
                ec.down_proj.weight.data[:, d_ffn - r_ch :].add_(
                    delta_dn.to(ec.down_proj.weight.dtype)
                )
                moe._grouped_weight_cache = None
                child_frob_accum += float(
                    math.hypot(float(delta_up.norm().item()), float(delta_dn.norm().item()))
                )
                num_children += 1

        drift_after = self.compute_anchor_drift()
        if drift_after > self.max_anchor_drift:
            for up_p, up_snap, dn_p, dn_snap in child_snapshots:
                up_p.data.copy_(up_snap)
                dn_p.data.copy_(dn_snap)
            for moe, _ in layer_pairs:
                moe._grouped_weight_cache = None
            return {
                "consolidated": False,
                "status": "rolled_back_anchor_invariant_drift",
                "num_child_experts_updated": 0,
                "parent_expert_max_delta": 0.0,
                "child_expert_frobenius_delta": 0.0,
                "anchor_drift_after": drift_after,
            }

        parent_max_delta = 0.0
        for up_p, up_ref, dn_p, dn_ref in parent_snapshots:
            parent_max_delta = max(
                parent_max_delta,
                float((up_p.detach() - up_ref).abs().max().item()),
                float((dn_p.detach() - dn_ref).abs().max().item()),
            )

        if reset_whiteboard_after:
            nn.init.normal_(self.whiteboard.fast_lora_A, std=0.02 / math.sqrt(self.whiteboard.d_model))
            self.whiteboard.fast_lora_B.zero_()

        return {
            "consolidated": True,
            "status": "consolidated_to_mitotic_children",
            "num_child_experts_updated": num_children,
            "parent_expert_max_delta": parent_max_delta,
            "child_expert_frobenius_delta": child_frob_accum / max(1, num_children),
            "anchor_drift_after": drift_after,
        }

    def solve_decision_on_whiteboard(
        self,
        context_ids: Tensor,
        context_lengths: Tensor,
        option_ids: Tensor,
        option_lengths: Tensor,
        targets: Tensor | None = None,
        *,
        recurrence_steps: int = 2,
        reset_per_batch: bool = False,
        adapt_on_context: bool = True,
    ) -> tuple[DecisionOutput, PlasticityTransactionReport]:
        """Solve a multi-option decision problem via premise-grounded whiteboard plasticity & counterfactual candidate falsification.

        Why this avoids confirmation bias:
        1. **Left-Aligned Premise-Only Plasticity:** Splits each row's actual valid `#U$#` world-state
           tokens into `(premise_first_half, premise_second_half)` without right-padding contamination
           or option-guess bias.
        2. **Orthogonal Coalition Consensus + Disagreement Penalty:** Evaluates every candidate option
           across disjoint expert coalitions (`Coalition A` vs `Coalition B`), penalizing candidates
           whose support is pathway-fragile.
        3. **Bidirectional Forward + Abductive Inverse Verification:** Combines forward matched-filter
           evidence (`Candidate -> State`) with inverse abductive coverage (`State Evidence -> Candidate`)
           and `COE` `Deduction` entailment margin.
        """
        if reset_per_batch:
            self.whiteboard.reset_fast_weights()

        batch = context_ids.shape[0]
        valid_options = option_lengths > 0
        ctx_width = int(context_lengths.max().item())
        trimmed_ctx_ids = context_ids[:, :ctx_width].long()
        ctx_valid = (
            torch.arange(ctx_width, device=context_ids.device)[None, :]
            < context_lengths[:, None]
        )

        # 1. Isolate `#U$#` World-State tokens and split each row's valid tokens cleanly at midpoint
        s_ev_mask = self.coe.pooler._state_evidence_mask(
            trimmed_ctx_ids,
            ctx_valid,
            strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
            bos_id=self.coe.config.bos_id,
            eos_id=self.coe.config.eos_id,
        )
        u_ids, u_lens, q_ids, q_lens = self._split_clauses_left_aligned(
            trimmed_ctx_ids, s_ev_mask, self.backbone.config.pad_id
        )

        if adapt_on_context:
            report = self.step(
                u_ids,
                q_ids,
                state_lengths=u_lens,
                hypothesis_lengths=q_lens,
                recurrence_steps=recurrence_steps,
            )
        else:
            report = PlasticityTransactionReport(
                committed=self.committed_steps > 0,
                status="retained_existing_whiteboard",
                whiteboard_loss_before=0.0,
                whiteboard_loss_after=0.0,
                coalition_agreement_before=1.0,
                coalition_agreement_after=1.0,
                cycle_consistency_before=1.0,
                cycle_consistency_after=1.0,
                coe_score_before=1.0,
                coe_score_after=1.0,
                entropy_after=1.0,
                anchor_drift=self.compute_anchor_drift(),
                novelty=0.0,
                adapter_frobenius_norm=self.whiteboard.adapter_frobenius_norm(),
                prefix_delta_norm=self.whiteboard.prefix_delta_norm(),
            )

        r_frob = float(
            torch.linalg.matrix_norm(self.whiteboard.rule_val_B.float(), ord="fro").item()
        )
        has_active_rule = r_frob > 1e-6
        use_wb = (
            report.committed
            or (self.whiteboard.adapter_frobenius_norm() > 1e-6)
            or has_active_rule
        )
        has_mitosis = any(bool(layer.moe.mitotic_pairs) for layer in self.backbone.encoder_layers)
        with torch.no_grad():
            encoded_state = self.backbone.encode_state(trimmed_ctx_ids, ctx_valid, causal=True)
            base_s = encoded_state.hidden
            if use_wb and has_active_rule and (
                self.backbone.config.dense_subset_k > 0
                or bool(self.backbone.config.dense_expert_indices)
            ):
                s_idx_r = self.backbone.config.dense_expert_indices
                s_k_r = self.backbone.config.dense_subset_k
                s_comb_r = self.backbone.config.combine_dense_and_sparse
                try:
                    self.backbone.set_dense_expert_subset(
                        [], dense_subset_k=0, combine_dense_and_sparse=False
                    )
                    canon_s_enc = self.backbone.encode_state(
                        trimmed_ctx_ids, ctx_valid, causal=True
                    )
                finally:
                    self.backbone.set_dense_expert_subset(
                        s_idx_r, dense_subset_k=s_k_r, combine_dense_and_sparse=s_comb_r
                    )
                s_for_rule = canon_s_enc.hidden
            else:
                s_for_rule = base_s

            s_glob = self.coe.pooler.state_norm(
                self.coe.pooler.pool_sequence(
                    s_for_rule, trimmed_ctx_ids, s_ev_mask, strip_boundary=False
                ).float()
            ).to(base_s.dtype)
            s_global_rule_delta = (
                self.whiteboard.apply_rule_readout(
                    s_glob,
                    token_ids=trimmed_ctx_ids,
                    valid_mask=s_ev_mask,
                    token_embedding=self.backbone.token_embedding,
                )
                if use_wb
                else None
            )
            if use_wb and (has_active_rule or has_mitosis):
                mod_s = self.backbone.encoder_norm(
                    self.whiteboard.apply_adapter(base_s, ctx_valid)
                ).masked_fill(~ctx_valid.unsqueeze(-1), 0)
                s_blend = 0.85 * base_s + 0.15 * mod_s
            else:
                s_blend = base_s

            state_for_decode = EncodedState(
                hidden=s_blend,
                valid=ctx_valid,
                aux_loss=encoded_state.aux_loss,
                router_z_loss=encoded_state.router_z_loss,
                expert_load=encoded_state.expert_load,
                gist_prefix_len=0,
            )

            valid_lengths = option_lengths[valid_options]
            opt_width = int(valid_lengths.max().item())
            flat_opt_ids = option_ids[valid_options][:, :opt_width].long()
            owner_rows = valid_options.nonzero(as_tuple=False)[:, 0]
            perm, inv_perm = _canonical_candidate_order(owner_rows, valid_lengths, flat_opt_ids)

            canon_opt_ids = flat_opt_ids.index_select(0, perm)
            canon_lengths = valid_lengths.index_select(0, perm)
            canon_owners = owner_rows.index_select(0, perm)
            canon_valid = (
                torch.arange(opt_width, device=option_ids.device)[None, :]
                < canon_lengths[:, None]
            )

            rule_state_proj_rows = None
            rule_cand_proj = None
            s_rule_rows = None
            w_rule = 0.0
            if use_wb and s_global_rule_delta is not None and has_active_rule:
                w_rule = 0.88
                r_cent = s_global_rule_delta.float() - s_global_rule_delta.float().mean(
                    dim=-1, keepdim=True
                )
                r_unit = F.normalize(r_cent, dim=-1, eps=1e-6)
                s_rule_state = (
                    r_unit * math.sqrt(float(self.backbone.config.d_model))
                ).to(s_glob.dtype)
                s_rule_rows = s_rule_state.index_select(0, canon_owners)
                rule_state_proj_rows = self.coe.state_projection(s_rule_state).index_select(
                    0, canon_owners
                )

            # Decode candidates via standard calibrated routing + Pass-1 / Pass-R ponder
            steps = max(2, recurrence_steps)
            decoded_std = self.backbone.decode_candidates(
                state_for_decode,
                canon_opt_ids,
                canon_valid,
                owner_rows=canon_owners,
                recurrence_steps=steps,
                causal_local=True,
            )
            adapt_cands = use_wb and (not has_active_rule) and has_mitosis
            dec_hid = (
                self.whiteboard.apply_adapter(decoded_std.hidden, canon_valid)
                if adapt_cands
                else decoded_std.hidden
            )
            dec_anc = (
                self.whiteboard.apply_adapter(decoded_std.anchor, canon_valid)
                if (adapt_cands and decoded_std.anchor is not None)
                else decoded_std.anchor
            )

            paired_state_repr, cand_repr = self.coe.pooler.matched_pair_representations(
                s_blend,
                trimmed_ctx_ids,
                ctx_valid,
                dec_hid,
                canon_opt_ids,
                canon_valid,
                canon_owners,
                cand_anchor=dec_anc,
                strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                bos_id=self.coe.config.bos_id,
                eos_id=self.coe.config.eos_id,
            )
            if use_wb:
                if s_rule_rows is not None:
                    paired_state_repr = (
                        0.25 * self.whiteboard.apply_adapter(paired_state_repr)
                        + 0.75 * s_rule_rows
                    )
                    if (
                        self.backbone.config.dense_subset_k > 0
                        or bool(self.backbone.config.dense_expert_indices)
                    ):
                        s_idx_c = self.backbone.config.dense_expert_indices
                        s_k_c = self.backbone.config.dense_subset_k
                        s_comb_c = self.backbone.config.combine_dense_and_sparse
                        try:
                            self.backbone.set_dense_expert_subset(
                                [], dense_subset_k=0, combine_dense_and_sparse=False
                            )
                            enc_opt_canon = self.backbone.encode_state(
                                canon_opt_ids, canon_valid, causal=True
                            )
                        finally:
                            self.backbone.set_dense_expert_subset(
                                s_idx_c,
                                dense_subset_k=s_k_c,
                                combine_dense_and_sparse=s_comb_c,
                            )
                        anc_src = (
                            enc_opt_canon.raw_hidden
                            if (
                                self.backbone.config.preserve_residual_across_stacks
                                and enc_opt_canon.raw_hidden is not None
                            )
                            else enc_opt_canon.hidden
                        )
                    else:
                        anc_src = (
                            decoded_std.anchor if decoded_std.anchor is not None else dec_hid
                        )
                    c_anc_norm = self.coe.pooler.cand_norm(
                        self.coe.pooler.pool_sequence(
                            anc_src,
                            canon_opt_ids,
                            canon_valid,
                            strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                            bos_id=self.coe.config.bos_id,
                            eos_id=self.coe.config.eos_id,
                        ).float()
                    ).to(cand_repr.dtype)
                    rule_cand_proj = self.coe.output_projection(c_anc_norm)
                elif has_mitosis:
                    paired_state_repr = self.whiteboard.apply_adapter(paired_state_repr)
                    cand_repr = self.whiteboard.apply_adapter(cand_repr)
            canon_state_proj = self.coe.state_projection(paired_state_repr)
            canon_cand_proj = self.coe.output_projection(cand_repr)
            fwd_scores = self.coe._similarity(canon_state_proj, canon_cand_proj)
            rule_sim: Tensor | None = None
            if (
                s_global_rule_delta is not None
                and rule_state_proj_rows is not None
                and rule_cand_proj is not None
                and w_rule > 0.0
            ):
                r_sim_ctx = self.coe._similarity(rule_state_proj_rows, rule_cand_proj)
                cand_ev_mask = self.coe.pooler._content_mask(
                    canon_opt_ids,
                    canon_valid,
                    strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                    bos_id=self.coe.config.bos_id,
                    eos_id=self.coe.config.eos_id,
                )
                r_sim_holo = self.whiteboard.score_holo_candidates(
                    s_glob,
                    trimmed_ctx_ids,
                    s_ev_mask,
                    canon_opt_ids,
                    cand_ev_mask,
                    canon_owners,
                    self.backbone.token_embedding,
                )
                if r_sim_holo is not None:
                    rule_sim = torch.tanh(
                        (0.55 * r_sim_ctx + 0.45 * r_sim_holo.to(r_sim_ctx.dtype)) * 3.5
                    )
                else:
                    rule_sim = torch.tanh(r_sim_ctx * 3.5)
                fwd_scores = (1.0 - w_rule) * fwd_scores + w_rule * rule_sim

            # Pass-1 intuition anchor blend (System-1 + System-2 agreement)
            step0_hid = (
                self.whiteboard.apply_adapter(decoded_std.step_states[0], canon_valid)
                if (adapt_cands and len(decoded_std.step_states) > 0)
                else (decoded_std.step_states[0] if len(decoded_std.step_states) > 0 else dec_hid)
            )
            if len(decoded_std.step_states) > 1:
                ps_1, cr_1 = self.coe.pooler.matched_pair_representations(
                    s_blend,
                    trimmed_ctx_ids,
                    ctx_valid,
                    step0_hid,
                    canon_opt_ids,
                    canon_valid,
                    canon_owners,
                    cand_anchor=dec_anc,
                    strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                    bos_id=self.coe.config.bos_id,
                    eos_id=self.coe.config.eos_id,
                )
                if use_wb:
                    if s_rule_rows is not None:
                        ps_1 = (
                            0.25 * self.whiteboard.apply_adapter(ps_1) + 0.75 * s_rule_rows
                        )
                    elif has_mitosis:
                        ps_1 = self.whiteboard.apply_adapter(ps_1)
                        cr_1 = self.whiteboard.apply_adapter(cr_1)
                cs_1 = self.coe._similarity(
                    self.coe.state_projection(ps_1),
                    self.coe.output_projection(cr_1),
                )
                if rule_sim is not None and w_rule > 0.0:
                    cs_1 = (1.0 - w_rule) * cs_1 + w_rule * rule_sim
                if w_rule > 0.0:
                    if self.coe.config.depth_ponder:
                        fwd_scores = 0.70 * cs_1 + 0.30 * fwd_scores
                        paired_state_repr = 0.70 * ps_1 + 0.30 * paired_state_repr
                        cand_repr = 0.70 * cr_1 + 0.30 * cand_repr
                elif has_mitosis:
                    pass_disagreement = (cs_1 - fwd_scores).abs()
                    fwd_scores = 0.85 * cs_1 + 0.15 * fwd_scores - 0.03 * pass_disagreement
                    paired_state_repr = 0.85 * ps_1 + 0.15 * paired_state_repr
                    cand_repr = 0.85 * cr_1 + 0.15 * cand_repr
                else:
                    fwd_scores = cs_1
                    paired_state_repr = ps_1
                    cand_repr = cr_1
                canon_state_proj = self.coe.state_projection(paired_state_repr)
                canon_cand_proj = self.coe.output_projection(cand_repr)

            # Abductive Inverse Matched-Filter Reconstruction (`State Evidence -> Candidate`)
            id_owners = torch.arange(canon_owners.shape[0], device=option_ids.device)
            _, inv_s_repr = self.coe.pooler.matched_pair_representations(
                step0_hid,
                canon_opt_ids,
                canon_valid,
                s_blend.index_select(0, canon_owners),
                trimmed_ctx_ids.index_select(0, canon_owners),
                ctx_valid.index_select(0, canon_owners),
                id_owners,
                strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                bos_id=self.coe.config.bos_id,
                eos_id=self.coe.config.eos_id,
            )
            if use_wb:
                inv_s_repr = self.whiteboard.apply_adapter(inv_s_repr)
            s_glob_target_proj = self.coe.state_projection(s_glob).index_select(0, canon_owners)
            inv_scores = self.coe._similarity(
                self.coe.state_projection(inv_s_repr),
                s_glob_target_proj,
            )

            # Orthogonal Expert Coalition Falsification Check (Coalition A vs Coalition B)
            num_exp = self.backbone.config.num_experts
            coal_a = [i for i in range(num_exp) if i % 2 == 0]
            coal_b = [i for i in range(num_exp) if i % 2 == 1] or coal_a
            saved_idx = self.backbone.config.dense_expert_indices
            saved_k = self.backbone.config.dense_subset_k
            saved_comb = self.backbone.config.combine_dense_and_sparse
            try:
                self.backbone.set_dense_expert_subset(coal_a, dense_subset_k=0, combine_dense_and_sparse=False)
                dec_a = self.backbone.decode_candidates(
                    state_for_decode, canon_opt_ids, canon_valid, owner_rows=canon_owners, recurrence_steps=1, causal_local=True
                )
                self.backbone.set_dense_expert_subset(coal_b, dense_subset_k=0, combine_dense_and_sparse=False)
                dec_b = self.backbone.decode_candidates(
                    state_for_decode, canon_opt_ids, canon_valid, owner_rows=canon_owners, recurrence_steps=1, causal_local=True
                )
            finally:
                self.backbone.set_dense_expert_subset(saved_idx, dense_subset_k=saved_k, combine_dense_and_sparse=saved_comb)

            ps_a, cr_a = self.coe.pooler.matched_pair_representations(
                s_blend, trimmed_ctx_ids, ctx_valid,
                self.whiteboard.apply_adapter(dec_a.hidden, canon_valid) if adapt_cands else dec_a.hidden,
                canon_opt_ids, canon_valid, canon_owners,
                cand_anchor=dec_anc, strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                bos_id=self.coe.config.bos_id, eos_id=self.coe.config.eos_id,
            )
            ps_b, cr_b = self.coe.pooler.matched_pair_representations(
                s_blend, trimmed_ctx_ids, ctx_valid,
                self.whiteboard.apply_adapter(dec_b.hidden, canon_valid) if adapt_cands else dec_b.hidden,
                canon_opt_ids, canon_valid, canon_owners,
                cand_anchor=dec_anc, strip_boundary=self.coe.config.strip_boundary_tokens_in_pool,
                bos_id=self.coe.config.bos_id, eos_id=self.coe.config.eos_id,
            )
            if use_wb:
                if s_rule_rows is not None:
                    ps_a = 0.25 * self.whiteboard.apply_adapter(ps_a) + 0.75 * s_rule_rows
                    ps_b = 0.25 * self.whiteboard.apply_adapter(ps_b) + 0.75 * s_rule_rows
                elif has_mitosis:
                    ps_a = self.whiteboard.apply_adapter(ps_a)
                    cr_a = self.whiteboard.apply_adapter(cr_a)
                    ps_b = self.whiteboard.apply_adapter(ps_b)
                    cr_b = self.whiteboard.apply_adapter(cr_b)
            sim_a = self.coe._similarity(self.coe.state_projection(ps_a), self.coe.output_projection(cr_a))
            sim_b = self.coe._similarity(self.coe.state_projection(ps_b), self.coe.output_projection(cr_b))
            if rule_sim is not None and w_rule > 0.0:
                sim_a = (1.0 - w_rule) * sim_a + w_rule * rule_sim
                sim_b = (1.0 - w_rule) * sim_b + w_rule * rule_sim
            coal_consensus = 0.5 * (sim_a + sim_b) - 0.08 * (sim_a - sim_b).abs()

            # COE Metacognition Correctness & Deduction Margin
            joint = torch.cat([paired_state_repr, cand_repr], dim=-1)
            meta_q = self.coe.meta_query_proj(joint)
            meta_sims = pearson_similarity(
                meta_q.unsqueeze(1),
                self.coe.meta_prototypes.unsqueeze(0),
                eps=self.coe.config.eps,
            )
            meta_correct = meta_sims[:, 0]

            if w_rule > 0.0:
                canon_scores = (
                    0.92 * fwd_scores
                    + 0.02 * inv_scores.to(fwd_scores.dtype)
                    + 0.02 * meta_correct.to(fwd_scores.dtype)
                    + 0.04 * coal_consensus.to(fwd_scores.dtype)
                )
            elif has_mitosis:
                canon_scores = (
                    0.76 * fwd_scores
                    + 0.10 * inv_scores.to(fwd_scores.dtype)
                    + 0.08 * meta_correct.to(fwd_scores.dtype)
                    + 0.06 * coal_consensus.to(fwd_scores.dtype)
                )
            else:
                canon_scores = (
                    fwd_scores
                    + 0.12 * inv_scores.to(fwd_scores.dtype)
                    + 0.10 * meta_correct.to(fwd_scores.dtype)
                    + 0.05 * coal_consensus.to(fwd_scores.dtype)
                )
            valid_scores = canon_scores.index_select(0, inv_perm)
            state_proj = canon_state_proj.index_select(0, inv_perm)
            cand_proj = canon_cand_proj.index_select(0, inv_perm)
            similarities = valid_scores.new_zeros(valid_options.shape).masked_scatter(
                valid_options, valid_scores
            )
            scale = self.coe._scale()
            logits = (similarities.float() * scale).masked_fill(~valid_options, -torch.inf)
            probabilities = logits.softmax(dim=-1)
            aux_loss = 0.5 * (encoded_state.aux_loss.float() + decoded_std.aux_loss.float())
            router_z_loss = 0.5 * (
                encoded_state.router_z_loss.float() + decoded_std.router_z_loss.float()
            )
            ce_loss, loss = None, None
            if targets is not None:
                ce_loss = F.cross_entropy(logits, targets.long())
                loss = (
                    ce_loss
                    + self.backbone.config.aux_loss_coef * aux_loss
                    + self.backbone.config.router_z_loss_coef * router_z_loss
                )
            dec_out = DecisionOutput(
                logits=logits,
                similarities=similarities,
                probabilities=probabilities,
                state_features=state_proj,
                candidate_features=cand_proj,
                loss=loss,
                ce_loss=ce_loss,
                aux_loss=aux_loss,
                router_z_loss=router_z_loss,
            )
        return dec_out, report

    @classmethod
    def _extract_prose_propositions(cls, text: str) -> list[tuple[str, str, str]]:
        """Extract `(premise, conclusion, full_clause)` propositions from raw books, papers, and technical prose."""
        props: list[tuple[str, str, str]] = []
        section_ctx = ""
        raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        paragraphs: list[tuple[str, str]] = []
        buf: list[str] = []
        for ln in raw_lines:
            hdr_match = re.match(r"^(?:#{1,6}\s+|Section\s+\d+[:.\s]+)(.+)$", ln, flags=re.I)
            if hdr_match:
                if buf:
                    paragraphs.append((section_ctx, " ".join(buf)))
                    buf = []
                section_ctx = hdr_match.group(1).strip().rstrip(":.")
                continue
            bullet_match = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", ln)
            if bullet_match:
                if buf:
                    paragraphs.append((section_ctx, " ".join(buf)))
                    buf = []
                paragraphs.append((section_ctx, bullet_match.group(1).strip()))
            else:
                buf.append(ln)
        if buf:
            paragraphs.append((section_ctx, " ".join(buf)))

        cond_re = re.compile(
            r"^(?:When|Whenever|If|Where|Because|Since|Under|Given|In\s+cases\s+where)\s+(.+?),\s*(?:then\s+)?(.+)$",
            flags=re.I,
        )
        colon_re = re.compile(
            r"^((?:Theorem|Lemma|Definition|Proposition|Invariant|Rule|Axiom|Law|Property)\s*[^:]{0,60}|[A-Z][A-Za-z0-9_\-\s()/]{2,55}):\s+(.+)$"
        )
        verb_re = re.compile(
            r"^(.{4,95}?)\s+(is\s+defined\s+as|is\s+governed\s+by|is\s+bounded\s+by|routes|enforces|guarantees|uses|computes|bounds|scales|maps|requires|produces|returns|triggers|maintains|executes|prevents|reduces|stores|projects|assigns|halts|operates\s+at)\s+(.+)$",
            flags=re.I,
        )

        for sec, para in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`(])", para)
            for sent in sentences:
                s_clean = re.sub(r"\s+", " ", sent).strip()
                if len(s_clean) < 12:
                    continue
                m_cond = cond_re.match(s_clean)
                if m_cond:
                    prem = m_cond.group(1).strip().rstrip(",;.")
                    conc = m_cond.group(2).strip()
                    if not conc.endswith("."):
                        conc += "."
                    props.append((prem, conc, s_clean))
                    continue

                m_colon = colon_re.match(s_clean)
                if m_colon:
                    rhs = m_colon.group(2).strip()
                    m_rhs_cond = cond_re.match(rhs)
                    if m_rhs_cond:
                        prem = m_rhs_cond.group(1).strip().rstrip(",;.")
                        conc = m_rhs_cond.group(2).strip()
                    else:
                        prem = m_colon.group(1).strip()
                        conc = rhs
                    if not conc.endswith("."):
                        conc += "."
                    props.append((prem, conc, s_clean))
                    continue

                m_verb = verb_re.match(s_clean)
                if m_verb:
                    subj = m_verb.group(1).strip()
                    verb = m_verb.group(2).strip()
                    rest = m_verb.group(3).strip()
                    prem = f"{subj} {verb}" if len(subj.split()) <= 5 else subj
                    conc = f"{verb} {rest}" if prem == subj else f"{subj} {verb} {rest}"
                    if not conc.endswith("."):
                        conc += "."
                    props.append((prem, conc, s_clean))
                    continue

                if "," in s_clean:
                    parts = [p.strip() for p in s_clean.split(",") if p.strip()]
                    if len(parts) >= 2 and len(parts[0]) >= 8:
                        prem = parts[0]
                        conc = ", ".join(parts[1:])
                        if not conc.endswith("."):
                            conc += "."
                        props.append((prem, conc, s_clean))
                        continue

                words = s_clean.split()
                if len(words) >= 6:
                    mid = max(3, len(words) // 2)
                    prem = f"{sec}: {' '.join(words[:mid])}" if sec else " ".join(words[:mid])
                    conc = " ".join(words[mid:])
                    if not conc.endswith("."):
                        conc += "."
                    props.append((prem, conc, s_clean))
        return props

    @classmethod
    def _extract_code_propositions(cls, code_text: str) -> list[tuple[str, str, str]]:
        """Extract `(premise, conclusion, full_clause)` propositions from source code via AST + multi-language patterns."""
        import textwrap

        props: list[tuple[str, str, str]] = []
        raw_unwrapped = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$", "", code_text.strip("\r\n"))
        clean_code = textwrap.dedent(raw_unwrapped).strip()
        try:
            tree = ast.parse(clean_code)
            for node in tree.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    if isinstance(node, ast.Assign) and len(node.targets) == 1:
                        t_str = ast.unparse(node.targets[0]).strip()
                        v_str = ast.unparse(node.value).strip() if node.value is not None else ""
                    elif isinstance(node, ast.AnnAssign):
                        t_str = ast.unparse(node.target).strip()
                        v_str = ast.unparse(node.value).strip() if node.value is not None else ""
                    else:
                        continue
                    if t_str and v_str:
                        prem = f"Module constant {t_str}"
                        conc = f"{t_str} is set to {v_str}."
                        props.append((prem, conc, f"{t_str} = {v_str}"))

                elif isinstance(node, ast.ClassDef):
                    cls_name = node.name
                    bases = ", ".join(ast.unparse(b) for b in node.bases)
                    doc = ast.get_docstring(node)
                    if doc:
                        first_doc = re.split(r"(?<=[.!?])\s+", doc.strip())[0].strip()
                        if not first_doc.endswith("."):
                            first_doc += "."
                        prem = f"Class {cls_name}" + (f"({bases})" if bases else "")
                        props.append((prem, first_doc, f"class {cls_name}: {first_doc}"))
                    for item in node.body:
                        if isinstance(item, (ast.Assign, ast.AnnAssign)):
                            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                                f_name = ast.unparse(item.targets[0]).strip()
                                f_val = ast.unparse(item.value).strip() if item.value else ""
                            elif isinstance(item, ast.AnnAssign):
                                f_name = ast.unparse(item.target).strip()
                                f_val = ast.unparse(item.value).strip() if item.value else ""
                            else:
                                continue
                            if f_name and f_val:
                                prem = f"In class {cls_name}, attribute {f_name}"
                                conc = f"{cls_name}.{f_name} defaults to {f_val}."
                                props.append((prem, conc, f"{cls_name}.{f_name} = {f_val}"))
                        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            props.extend(
                                cls._extract_py_func_propositions(item, scope_prefix=cls_name)
                            )

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    props.extend(cls._extract_py_func_propositions(node, scope_prefix=""))
            if props:
                return props
        except SyntaxError:
            pass

        # Regex fallback for Rust / TypeScript / C++ / Go / partial snippets
        fn_re = re.compile(
            r"(?:pub\s+)?(?:async\s+)?(?:def|fn|function|func)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)[^{:]*[:{]([\s\S]*?)(?=\n(?:pub\s+|async\s+|def\s+|fn\s+|function\s+|func\s+|class\s+|struct\s+)|\Z)"
        )
        for m in fn_re.finditer(clean_code):
            fn_name, fn_args, fn_body = m.group(1).strip(), m.group(2).strip(), m.group(3)
            for if_m in re.finditer(
                r"if\s+([^{:\n]+)[:{]?\s*(?:return|raise|throw)\s+([^;\n}]+)", fn_body
            ):
                cond_s = if_m.group(1).strip().strip("()")
                act_s = if_m.group(2).strip()
                prem = f"In {fn_name}, when {cond_s}"
                conc = f"{fn_name} returns {act_s}."
                props.append((prem, conc, f"In {fn_name}: if {cond_s} -> {act_s}"))
            ret_matches = list(re.finditer(r"return\s+([^;\n}]+)", fn_body))
            if ret_matches:
                last_ret = ret_matches[-1].group(1).strip()
                prem = f"Function {fn_name}({fn_args})"
                conc = f"{fn_name} returns {last_ret}."
                props.append((prem, conc, f"{fn_name}({fn_args}) returns {last_ret}"))
        return props

    @classmethod
    def _extract_py_func_propositions(
        cls,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        scope_prefix: str = "",
    ) -> list[tuple[str, str, str]]:
        props: list[tuple[str, str, str]] = []
        qual_name = f"{scope_prefix}.{func_node.name}" if scope_prefix else func_node.name
        arg_names = [a.arg for a in func_node.args.args if a.arg != "self" and a.arg != "cls"]
        args_sig = ", ".join(arg_names)
        doc = ast.get_docstring(func_node)
        if doc:
            first_doc = re.split(r"(?<=[.!?])\s+", doc.strip())[0].strip()
            if not first_doc.endswith("."):
                first_doc += "."
            prem = f"Function {qual_name}({args_sig})"
            props.append((prem, first_doc, f"def {qual_name}({args_sig}): {first_doc}"))

        local_env: dict[str, str] = {}
        has_branch_return = False
        for stmt in func_node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                var_name = stmt.targets[0].id
                expr_str = ast.unparse(stmt.value).strip()
                local_env[var_name] = expr_str
            elif isinstance(stmt, ast.If):
                cond_str = ast.unparse(stmt.test).strip()
                for sub in stmt.body:
                    if isinstance(sub, ast.Return) and sub.value is not None:
                        has_branch_return = True
                        ret_s = ast.unparse(sub.value).strip()
                        ret_exp = local_env.get(ret_s, ret_s)
                        prem = f"In {qual_name}, when {cond_str}"
                        conc = f"{qual_name} returns {ret_exp}."
                        props.append(
                            (prem, conc, f"In {qual_name}: when {cond_str}, returns {ret_exp}.")
                        )
                    elif isinstance(sub, ast.Raise) and sub.exc is not None:
                        has_branch_return = True
                        exc_s = ast.unparse(sub.exc).strip()
                        prem = f"In {qual_name}, when {cond_str}"
                        conc = f"{qual_name} raises {exc_s}."
                        props.append(
                            (prem, conc, f"In {qual_name}: when {cond_str}, raises {exc_s}.")
                        )
                    elif isinstance(sub, ast.Assign) and len(sub.targets) == 1:
                        t_s = ast.unparse(sub.targets[0]).strip()
                        v_s = ast.unparse(sub.value).strip()
                        prem = f"In {qual_name}, when {cond_str}"
                        conc = f"{t_s} is set to {v_s}."
                        props.append(
                            (prem, conc, f"In {qual_name}: when {cond_str}, {t_s} = {v_s}.")
                        )
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                ret_s = ast.unparse(stmt.value).strip()
                ret_exp = local_env.get(ret_s, ret_s)
                if ret_exp != ret_s:
                    conc = f"{qual_name} computes {ret_s} = {ret_exp} and returns {ret_exp}."
                else:
                    conc = f"{qual_name} returns {ret_exp}."
                prem = (
                    f"In {qual_name}, otherwise when nominal"
                    if has_branch_return
                    else f"Calling {qual_name}({args_sig})"
                )
                props.append((prem, conc, f"def {qual_name}({args_sig}): {conc}"))
        return props

    @classmethod
    def _extract_web_propositions(cls, html_text: str) -> list[tuple[str, str, str]]:
        """Extract `(premise, conclusion, full_clause)` propositions from HTML DOM, tables, cards, and web pages."""
        props: list[tuple[str, str, str]] = []
        cleaned = re.sub(r"<(script|style)\b[^>]*>[\s\S]*?</\1>", " ", html_text, flags=re.I)

        def _strip_tags(fragment: str) -> str:
            txt = re.sub(r"<[^>]+>", " ", fragment)
            return re.sub(r"\s+", " ", html.unescape(txt)).strip()

        title_m = re.search(r"<title\b[^>]*>([\s\S]*?)</title>", cleaned, flags=re.I)
        page_title = _strip_tags(title_m.group(1)) if title_m else ""

        # 1. HTML Tables (<tr> / <th> / <td>)
        for tbl_m in re.finditer(r"<table\b[^>]*>([\s\S]*?)</table>", cleaned, flags=re.I):
            tbl_html = tbl_m.group(1)
            headers: list[str] = []
            for tr_m in re.finditer(r"<tr\b[^>]*>([\s\S]*?)</tr>", tbl_html, flags=re.I):
                row_html = tr_m.group(1)
                ths = [
                    _strip_tags(m.group(1))
                    for m in re.finditer(r"<th\b[^>]*>([\s\S]*?)</th>", row_html, flags=re.I)
                ]
                if ths and not headers:
                    headers = ths
                    continue
                tds = [
                    _strip_tags(m.group(1))
                    for m in re.finditer(r"<td\b[^>]*>([\s\S]*?)</td>", row_html, flags=re.I)
                ]
                if len(tds) >= 2 and tds[0]:
                    entity = tds[0]
                    col_pairs: list[str] = []
                    for c_idx in range(1, len(tds)):
                        val = tds[c_idx]
                        if not val:
                            continue
                        hdr = headers[c_idx] if c_idx < len(headers) and headers[c_idx] else ""
                        if hdr:
                            col_pairs.append(f"{hdr}: {val}")
                            prem_col = f"{entity} {hdr}"
                            conc_col = f"{entity} has {hdr} {val}."
                            props.append((prem_col, conc_col, f"{entity} — {hdr}: {val}."))
                        else:
                            col_pairs.append(val)
                    if col_pairs and not headers:
                        prem_row = entity
                        conc_row = f"{entity} specifies {', '.join(col_pairs)}."
                        props.append((prem_row, conc_row, f"{prem_row}: {conc_row}"))

        # 2. Definition Lists (<dt> / <dd>)
        for dl_m in re.finditer(
            r"<dt\b[^>]*>([\s\S]*?)</dt>\s*<dd\b[^>]*>([\s\S]*?)</dd>", cleaned, flags=re.I
        ):
            dt_txt = _strip_tags(dl_m.group(1)).rstrip(":")
            dd_txt = _strip_tags(dl_m.group(2))
            if dt_txt and dd_txt:
                if not dd_txt.endswith("."):
                    dd_txt += "."
                props.append((dt_txt, dd_txt, f"{dt_txt}: {dd_txt}"))

        # 3. Remaining DOM leaf blocks (<h1>..<h6>, <p>, <li> inside <article>/<section>/<body>)
        without_tables = re.sub(r"<(table|dl)\b[^>]*>[\s\S]*?</\1>", " ", cleaned, flags=re.I)
        block_lines: list[str] = []
        if page_title:
            block_lines.append(f"# {page_title}")
        for blk_m in re.finditer(
            r"<(h[1-6]|p|li)\b[^>]*>([\s\S]*?)</\1>",
            without_tables,
            flags=re.I,
        ):
            tag = blk_m.group(1).lower()
            inner = blk_m.group(2)
            txt = _strip_tags(inner)
            if not txt:
                continue
            if tag.startswith("h"):
                block_lines.append(f"## {txt}")
            elif tag == "li":
                block_lines.append(f"- {txt}")
            else:
                block_lines.append(txt)
        if block_lines:
            props.extend(cls._extract_prose_propositions("\n".join(block_lines)))
        return props

    @classmethod
    def extract_universal_propositions(
        cls,
        raw_text: str,
        *,
        modality: str = "auto",
    ) -> list[tuple[str, str, str]]:
        """Parse raw prose (books/papers), source code, or web pages (HTML/DOM) into `(premise, conclusion, full_clause)` triples."""
        text = raw_text.strip()
        if not text:
            return []
        mod = modality.lower().strip()
        if mod == "code":
            return cls._extract_code_propositions(text)
        if mod == "web" or mod == "html":
            return cls._extract_web_propositions(text)
        if mod == "prose" or mod == "paper" or mod == "book":
            return cls._extract_prose_propositions(text)

        # Auto-detect mixed or single-modality artifacts
        props: list[tuple[str, str, str]] = []
        remaining = text

        # Extract fenced code blocks first if present in mixed markdown/papers
        fenced_re = re.compile(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```")
        for f_m in fenced_re.finditer(text):
            props.extend(cls._extract_code_propositions(f_m.group(1)))
        remaining = fenced_re.sub(" ", remaining).strip()

        if re.search(r"<(html|head|body|table|tr|td|article|section|div|dl|dt|dd)\b", remaining, flags=re.I):
            props.extend(cls._extract_web_propositions(remaining))
        elif re.search(
            r"^\s*(?:def\s+|class\s+|import\s+|from\s+\S+\s+import|pub\s+fn\s+|fn\s+|function\s+)",
            remaining,
            flags=re.M,
        ):
            props.extend(cls._extract_code_propositions(remaining))
        elif remaining:
            props.extend(cls._extract_prose_propositions(remaining))

        # Deduplicate while preserving stream order
        deduped: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for prem, conc, full_c in props:
            key = (prem.strip().lower(), conc.strip().lower())
            if key not in seen and prem.strip() and conc.strip():
                seen.add(key)
                deduped.append((prem.strip(), conc.strip(), full_c.strip()))
        return deduped

    def ingest_artifact(
        self,
        raw_text: str,
        *,
        tokenizer: Any,
        modality: str = "auto",
        pager: Any | None = None,
        recurrence_steps: int = 2,
        reset_whiteboard: bool = False,
    ) -> tuple[PlasticityTransactionReport, list[tuple[str, str, str]]]:
        """Read a raw book/paper excerpt, source code file, or HTML web page into the Quarantined Whiteboard (and optional `InfiniteContextPager`)."""
        if reset_whiteboard:
            self.whiteboard.reset_fast_weights()

        propositions = self.extract_universal_propositions(raw_text, modality=modality)
        device = self.backbone.token_embedding.weight.device
        pad_id = self.backbone.config.pad_id

        if pager is not None:
            raw_ids = torch.tensor(
                tokenizer.encode(raw_text, add_bos=False, add_eos=False),
                dtype=torch.long,
                device=device,
            )
            if raw_ids.numel() > 0:
                pager.ingest_tokens(
                    self.backbone,
                    raw_ids,
                    source_tag=modality,
                )

        if not propositions:
            empty_report = PlasticityTransactionReport(
                committed=False,
                status="no_propositions_extracted",
                whiteboard_loss_before=0.0,
                whiteboard_loss_after=0.0,
                coalition_agreement_before=1.0,
                coalition_agreement_after=1.0,
                cycle_consistency_before=1.0,
                cycle_consistency_after=1.0,
                coe_score_before=1.0,
                coe_score_after=1.0,
                entropy_after=1.0,
                anchor_drift=self.compute_anchor_drift(),
                novelty=0.0,
                adapter_frobenius_norm=self.whiteboard.adapter_frobenius_norm(),
                prefix_delta_norm=self.whiteboard.prefix_delta_norm(),
            )
            return empty_report, []

        max_len = int(self.backbone.config.max_seq_len)
        fill_id = max(0, int(pad_id))

        def _pad_batch(texts: list[str]) -> tuple[Tensor, Tensor]:
            enc = [tokenizer.encode(t, add_bos=True, add_eos=True)[:max_len] for t in texts]
            lens = torch.tensor([len(x) for x in enc], dtype=torch.long, device=device)
            w = int(lens.max().item())
            mat = torch.full((len(texts), w), fill_id, dtype=torch.long, device=device)
            for i, seq in enumerate(enc):
                mat[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            return mat, lens

        s_texts = [
            f"#U$# {prem}." if not prem.endswith(".") else f"#U$# {prem}"
            for prem, _, _ in propositions
        ]
        h_texts = [conc for _, conc, _ in propositions]
        lm_texts = [
            full_c if full_c.startswith("#U$#") else f"#U$# {full_c}"
            for _, _, full_c in propositions
        ]

        s_ids, s_lens = _pad_batch(s_texts)
        h_ids, h_lens = _pad_batch(h_texts)
        lm_ids, lm_lens = _pad_batch(lm_texts)

        report = self.absorb_context(
            lm_ids,
            lm_lens,
            premise_ids=s_ids,
            premise_lengths=s_lens,
            conclusion_ids=h_ids,
            conclusion_lengths=h_lens,
            recurrence_steps=recurrence_steps,
        )
        return report, propositions

    @torch.no_grad()
    def _solve_closed_form_rule_banks(
        self, whiteboard: QuarantinedWhiteboardState | None = None
    ) -> None:
        """Solve the directed multi-hop transitive closure and biorthogonal Hopfield-Kanerva banks on `whiteboard`."""
        wb = whiteboard if whiteboard is not None else self.whiteboard
        if wb._rule_key_bank is None or wb._rule_cand_bank is None:
            return
        device = wb.rule_key_A.device
        s_f_all = wb._rule_key_bank.to(device=device, dtype=torch.float32)
        c_unit_all = wb._rule_cand_bank.to(device=device, dtype=torch.float32)
        n_rules = int(c_unit_all.shape[0])
        if n_rules == 0:
            return

        ph_all = (
            wb._rule_premise_holo_bank.to(device=device, dtype=torch.float32)
            if wb._rule_premise_holo_bank is not None
            else s_f_all
        )
        ch_all = (
            wb._rule_conc_holo_bank.to(device=device, dtype=torch.float32)
            if wb._rule_conc_holo_bank is not None
            else s_f_all
        )

        with torch.amp.autocast(device_type=device.type, enabled=False):
            # Directed Multi-Hop Transitive Closure (A => B and B => C |- A => C)
            closure_op = torch.eye(n_rules, device=device, dtype=torch.float32)
            if n_rules >= 2:
                self_overlap = (ch_all * ph_all).sum(dim=-1, keepdim=True).clamp_min(0.0)
                ch_novel = F.normalize(
                    ch_all - 0.70 * self_overlap * ph_all, dim=-1, eps=1e-6
                )
                raw_link = ch_novel @ ph_all.T
                prem_sim = (ph_all @ ph_all.T).clamp_min(0.0)
                link_sim = raw_link - 0.55 * raw_link.T.clamp_min(0.0) - 0.35 * prem_sim
                link_sim.fill_diagonal_(-1.0)
                row_max, row_argmax = link_sim.max(dim=-1)
                if n_rules >= 3:
                    top2_vals = torch.topk(link_sim, k=2, dim=-1).values
                    link_margin = top2_vals[:, 0] - top2_vals[:, 1]
                else:
                    link_margin = row_max
                t_adj = torch.zeros((n_rules, n_rules), device=device, dtype=torch.float32)
                for r_idx in range(n_rules):
                    tgt_idx = int(row_argmax[r_idx].item())
                    if (
                        float(row_max[r_idx].item()) > 0.22
                        and float(link_margin[r_idx].item()) > 0.06
                    ):
                        t_adj[r_idx, tgt_idx] = 1.0
                t_adj_2 = (t_adj @ t_adj).clamp(0.0, 1.0)
                t_adj_2.fill_diagonal_(0.0)
                closure_op = closure_op + 0.78 * t_adj + 0.55 * t_adj_2
            wb._rule_closure_op = closure_op.detach().clone()

            gram_c = c_unit_all @ c_unit_all.T + 1e-2 * torch.eye(
                n_rules, device=device, dtype=torch.float32
            )
            b_dual_c = torch.linalg.solve(gram_c, c_unit_all)
            q_dual = F.normalize(closure_op @ b_dual_c, dim=-1, eps=1e-6)

            w_s = self.coe.state_projection.weight.detach().to(
                device=device, dtype=torch.float32
            )
            w_s = w_s - w_s.mean(dim=0, keepdim=True)
            w_s = w_s - w_s.mean(dim=1, keepdim=True)
            gram_w = w_s @ w_s.T + 1e-3 * torch.eye(
                w_s.shape[0], device=device, dtype=torch.float32
            )
            z_row = F.normalize(
                torch.linalg.solve(gram_w, q_dual.T).T @ w_s, dim=-1, eps=1e-6
            )
            wb._rule_z_row_bank = z_row.detach().clone()
            z_target = z_row

            gram_h = ch_all @ ch_all.T + 1e-2 * torch.eye(
                n_rules, device=device, dtype=torch.float32
            )
            b_dual_h = torch.linalg.solve(gram_h, ch_all)
            wb._rule_closed_conc_holo_bank = (
                F.normalize(closure_op @ b_dual_h, dim=-1, eps=1e-6).detach().clone()
            )

            gram_s_reg = s_f_all @ s_f_all.T + 1e-3 * torch.eye(
                n_rules, device=device, dtype=torch.float32
            )
            dual_keys = torch.linalg.solve(gram_s_reg, s_f_all)
            wb._rule_dual_key_bank = dual_keys.detach().clone()
            q_k, r_k = torch.linalg.qr(dual_keys.T, mode="reduced")
            q_v, r_v = torch.linalg.qr(z_target.T, mode="reduced")
            core = (r_k @ r_v.T) / max(1e-6, wb.scale)
            u_c, s_c, vh_c = torch.linalg.svd(core, full_matrices=False)
            r_keep = min(int(s_c.numel()), wb.rule_rank)
            sqrt_s = s_c[:r_keep].clamp_min(0.0).sqrt()
            k_a = (q_k @ u_c[:, :r_keep]) * sqrt_s.unsqueeze(0)
            v_b = sqrt_s.unsqueeze(1) * (vh_c[:r_keep, :] @ q_v.T)
            wb.rule_key_A.zero_()
            wb.rule_val_B.zero_()
            wb.rule_key_A[:, :r_keep].copy_(k_a.to(wb.rule_key_A.dtype))
            wb.rule_val_B[:r_keep, :].copy_(v_b.to(wb.rule_val_B.dtype))

    @torch.no_grad()
    def merge_whiteboards(
        self,
        cartridges: list[QuarantinedWhiteboardState],
        *,
        allow_belief_revision: bool = True,
        max_rules: int = 64,
    ) -> QuarantinedWhiteboardState:
        """Algebraically fuse multiple independent `QuarantinedWhiteboardState` cartridges into `self.whiteboard` without re-encoding tokens."""
        self.whiteboard.reset_fast_weights()
        if not cartridges:
            return self.whiteboard

        device = self.whiteboard.fast_lora_A.device
        dtype = self.whiteboard.fast_lora_A.dtype
        d_model = self.whiteboard.d_model

        lora_sum = torch.zeros((d_model, d_model), device=device, dtype=torch.float32)
        gen_sum = torch.zeros((d_model, d_model), device=device, dtype=torch.float32)
        prefix_sum = torch.zeros_like(
            self.whiteboard.whiteboard_prefix_delta, dtype=torch.float32
        )
        n_valid_carts = 0

        for cart in cartridges:
            if cart._rule_key_bank is not None and cart._rule_cand_bank is not None:
                self.whiteboard.append_episodic_rules(
                    cart._rule_key_bank.to(device=device),
                    cart._rule_cand_bank.to(device=device),
                    s_holo_new=cart._rule_premise_holo_bank.to(device=device)
                    if cart._rule_premise_holo_bank is not None
                    else None,
                    c_holo_new=cart._rule_conc_holo_bank.to(device=device)
                    if cart._rule_conc_holo_bank is not None
                    else None,
                    premise_texts=cart._rule_premise_texts,
                    conclusion_texts=cart._rule_conc_texts,
                    allow_belief_revision=allow_belief_revision,
                    max_rules=max_rules,
                )
            lora_sum = lora_sum + (
                cart.fast_lora_A.to(device=device, dtype=torch.float32)
                @ cart.fast_lora_B.to(device=device, dtype=torch.float32)
            ) * cart.scale
            gen_sum = gen_sum + (
                cart.gen_key_A.to(device=device, dtype=torch.float32)
                @ cart.gen_val_B.to(device=device, dtype=torch.float32)
            ) * cart.scale
            if cart.whiteboard_prefix_delta.shape == prefix_sum.shape:
                prefix_sum = prefix_sum + cart.whiteboard_prefix_delta.to(
                    device=device, dtype=torch.float32
                )
                n_valid_carts += 1

        if n_valid_carts > 0:
            self.whiteboard.whiteboard_prefix_delta.copy_(
                (prefix_sum / float(n_valid_carts)).to(dtype=dtype)
            )

        scale_wb = max(1e-6, self.whiteboard.scale)
        if float(lora_sum.norm().item()) > 1e-7:
            u_l, s_l, vh_l = torch.linalg.svd(
                (lora_sum / float(len(cartridges))) / scale_wb, full_matrices=False
            )
            rk_l = min(int(s_l.numel()), self.whiteboard.lora_rank)
            sq_l = s_l[:rk_l].clamp_min(0.0).sqrt()
            self.whiteboard.fast_lora_A.zero_()
            self.whiteboard.fast_lora_B.zero_()
            self.whiteboard.fast_lora_A[:, :rk_l].copy_(
                (u_l[:, :rk_l] * sq_l.unsqueeze(0)).to(dtype=dtype)
            )
            self.whiteboard.fast_lora_B[:rk_l, :].copy_(
                (sq_l.unsqueeze(1) * vh_l[:rk_l, :]).to(dtype=dtype)
            )

        if float(gen_sum.norm().item()) > 1e-7:
            u_g, s_g, vh_g = torch.linalg.svd(
                (gen_sum / float(len(cartridges))) / scale_wb, full_matrices=False
            )
            rk_g = min(int(s_g.numel()), self.whiteboard.gen_rank)
            sq_g = s_g[:rk_g].clamp_min(0.0).sqrt()
            self.whiteboard.gen_key_A.zero_()
            self.whiteboard.gen_val_B.zero_()
            self.whiteboard.gen_key_A[:, :rk_g].copy_(
                (u_g[:, :rk_g] * sq_g.unsqueeze(0)).to(dtype=dtype)
            )
            self.whiteboard.gen_val_B[:rk_g, :].copy_(
                (sq_g.unsqueeze(1) * vh_g[:rk_g, :]).to(dtype=dtype)
            )

        self._solve_closed_form_rule_banks(self.whiteboard)
        self.whiteboard.project_to_trust_region()
        return self.whiteboard

    @torch.no_grad()
    def audit_contradictions(
        self,
        artifact_a: str | QuarantinedWhiteboardState,
        artifact_b: str | QuarantinedWhiteboardState,
        *,
        tokenizer: Any | None = None,
        modality_a: str = "auto",
        modality_b: str = "auto",
    ) -> list[dict[str, Any]]:
        """Audit two artifacts or whiteboards for direct and 2-hop cross-modal contradictions and bound violations."""
        if tokenizer is None:
            from .data import get_tokenizer

            tokenizer = get_tokenizer(int(self.backbone.token_embedding.num_embeddings))

        def _get_props(
            art: str | QuarantinedWhiteboardState, mod: str
        ) -> list[tuple[str, str]]:
            if isinstance(art, QuarantinedWhiteboardState):
                return list(zip(art._rule_premise_texts, art._rule_conc_texts))
            return [
                (p, c)
                for p, c, _ in self.extract_universal_propositions(str(art), modality=mod)
            ]

        props_a = _get_props(artifact_a, modality_a)
        props_b = _get_props(artifact_b, modality_b)
        if not props_a or not props_b:
            return []

        device = self.backbone.token_embedding.weight.device
        pad_id = max(0, int(self.backbone.config.pad_id))
        max_len = int(self.backbone.config.max_seq_len)

        def _sketch_list(texts: list[str]) -> Tensor:
            enc = [
                tokenizer.encode(t, add_bos=False, add_eos=False)[:max_len] or [pad_id]
                for t in texts
            ]
            w = max(len(x) for x in enc)
            mat = torch.full((len(texts), w), pad_id, dtype=torch.long, device=device)
            msk = torch.zeros((len(texts), w), dtype=torch.bool, device=device)
            for idx, seq in enumerate(enc):
                mat[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                msk[idx, : len(seq)] = True
            return self.whiteboard.compute_holographic_sketch(
                mat, msk, self.backbone.token_embedding
            )

        p_txts_a = [p.lower() for p, _ in props_a]
        c_txts_a = [c.lower() for _, c in props_a]
        p_txts_b = [p.lower() for p, _ in props_b]
        c_txts_b = [c.lower() for _, c in props_b]

        ph_a = _sketch_list(p_txts_a)
        ch_a = _sketch_list(c_txts_a)
        ph_b = _sketch_list(p_txts_b)
        ch_b = _sketch_list(c_txts_b)

        # Identify 2-hop transitive implications inside artifact_a (i -> k)
        hop2_target_a: dict[int, int] = {}
        n_a = len(props_a)
        if n_a >= 2:
            raw_l = ch_a @ ph_a.T
            prem_s = (ph_a @ ph_a.T).clamp_min(0.0)
            ls_a = raw_l - 0.55 * raw_l.T.clamp_min(0.0) - 0.35 * prem_s
            ls_a.fill_diagonal_(-1.0)
            r_max, r_arg = ls_a.max(dim=-1)
            for i in range(n_a):
                if float(r_max[i].item()) > 0.22:
                    hop2_target_a[i] = int(r_arg[i].item())

        conflicts: list[dict[str, Any]] = []
        p_sim_mat = ph_a @ ph_b.T
        c_sim_mat = ch_a @ ch_b.T

        for i in range(len(props_a)):
            num_pa, bool_pa = self.whiteboard.extract_symbolic_predicates(p_txts_a[i])
            for j in range(len(props_b)):
                p_sim = float(p_sim_mat[i, j].item())
                if p_sim < 0.65:
                    continue
                num_pb, bool_pb = self.whiteboard.extract_symbolic_predicates(p_txts_b[j])
                if (num_pa != num_pb) or (bool_pa != bool_pb):
                    continue

                # 1. Check direct 1-hop conclusion conflict or numeric bound violation
                c_sim = float(c_sim_mat[i, j].item())
                num_ca, _ = self.whiteboard.extract_symbolic_predicates(c_txts_a[i])
                num_cb, _ = self.whiteboard.extract_symbolic_predicates(c_txts_b[j])

                bound_conflict = False
                for va, opa, vala in num_ca:
                    for vb, opb, valb in num_cb:
                        if va == vb and (opa != opb or abs(vala - valb) > 1e-12):
                            bound_conflict = True

                # If artifact_a has a 2-hop target k, check if artifact_b is asserting a 2-hop conflict
                k = hop2_target_a.get(i)
                if k is not None:
                    num_ck, _ = self.whiteboard.extract_symbolic_predicates(c_txts_a[k])
                    c_sim_k = float((ch_a[k] * ch_b[j]).sum().item())
                    hop2_bound_conflict = False
                    for vk, opk, valk in num_ck:
                        for vb, opb, valb in num_cb:
                            if vk == vb and (opk != opb or abs(valk - valb) > 1e-12):
                                hop2_bound_conflict = True
                    if hop2_bound_conflict or (c_sim < 0.55 and c_sim_k < 0.55):
                        conflicts.append(
                            {
                                "type": "transitive_2hop_bound_violation"
                                if hop2_bound_conflict
                                else "transitive_2hop_contradiction",
                                "premise_a": props_a[i][0],
                                "conclusion_a": f"{props_a[i][1]} => {props_a[k][1]}",
                                "premise_b": props_b[j][0],
                                "conclusion_b": props_b[j][1],
                                "premise_similarity": p_sim,
                                "conclusion_similarity": c_sim_k,
                            }
                        )
                        continue

                if bound_conflict or c_sim < 0.60:
                    conflicts.append(
                        {
                            "type": "numeric_bound_violation"
                            if bound_conflict
                            else "direct_conclusion_contradiction",
                            "premise_a": props_a[i][0],
                            "conclusion_a": props_a[i][1],
                            "premise_b": props_b[j][0],
                            "conclusion_b": props_b[j][1],
                            "premise_similarity": p_sim,
                            "conclusion_similarity": c_sim,
                        }
                    )

        return conflicts








