"""Ornith-1.0 Self-Scaffolding, Ornith-1.5 Self-Improvement, GRPO, & ExpertLifecycle.

Implements all equations and structural self-improvement mechanics from
`docs/05_ORNITH_SELF_SCAFFOLDING_SPEC.md`:
1. Ornith-1.0 Two-Stage Self-Scaffolding (`U$ -> S$ -> COE -> T$ -> A$`) with
   monotonic `ScaffoldCache` retention (`max_i R_i > max(0, R_cached)`).
2. Ornith-1.5 Three-Stage End-to-End Self-Improvement (`Q$ -> H$/S$ -> T$/A$`):
   - Task Reward: $R_{\\text{task}} = V(q, s) \\times D(q, s, \\{\\tau_i\\}) \\times N(q)$
     with hard gate $V=0 \\implies R_{\\text{task}}=0$, frontier difficulty $p^* = 0.2$,
     and `<0.01ms` `Association` Novelty $N(q)$ against `NoveltyTaskBuffer`.
   - Harness Reward: $R_{\\text{harness}} = C(q, h) \\times F(h, \\{\\tau_i\\}) \\times H(h)$.
   - Staleness-Weighted Clipped GRPO (`staleness_weight`, `grpo_clipped_loss`).
3. Bounded Speculative `ExpertLifecycle` (`grow_experts`, `prune_experts`) across all
   `encoder_layers` and `decoder_layers` of `MCL4CEDBackbone` with `CPUOffloadAdamW`.
"""

from __future__ import annotations

import ast
import math
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import MCL4CEDBackbone
from .operators import pearson_similarity
from .optim import CPUOffloadAdamW

MAX_PARAMETERS = 50_000_000


class NoveltyTaskBuffer:
    """VRAM/RAM ring buffer of 32-dim centered Pearson task embeddings for <0.01ms Novelty N(q)."""

    def __init__(
        self,
        projection_dim: int = 32,
        max_capacity: int = 100_000,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if projection_dim < 3 or max_capacity < 1:
            raise ValueError("projection_dim >= 3 and max_capacity >= 1 required")
        self.projection_dim = projection_dim
        self.max_capacity = max_capacity
        self.device = torch.device(device)
        self.dtype = dtype
        self._buffer = torch.empty((max_capacity, projection_dim), device=self.device, dtype=dtype)
        self._size = 0
        self._write_ptr = 0

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def _normalize_pearson(vecs: Tensor, eps: float = 1e-6) -> Tensor:
        with torch.autocast(device_type=vecs.device.type, enabled=False):
            centered = vecs.float() - vecs.float().mean(dim=-1, keepdim=True)
            return F.normalize(centered, dim=-1, eps=eps)

    @torch.no_grad()
    def add(self, embeddings: Tensor) -> None:
        """Normalize and insert `[N, D]` task embeddings into the ring buffer."""
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.projection_dim:
            raise ValueError(f"Expected [N, {self.projection_dim}] embeddings")
        normed = self._normalize_pearson(embeddings).to(device=self.device, dtype=self.dtype)
        n = normed.shape[0]
        if n >= self.max_capacity:
            self._buffer.copy_(normed[-self.max_capacity :])
            self._size = self.max_capacity
            self._write_ptr = 0
            return
        end = self._write_ptr + n
        if end <= self.max_capacity:
            self._buffer[self._write_ptr : end].copy_(normed)
        else:
            first = self.max_capacity - self._write_ptr
            self._buffer[self._write_ptr :].copy_(normed[:first])
            self._buffer[: n - first].copy_(normed[first:])
        self._write_ptr = end % self.max_capacity
        self._size = min(self.max_capacity, self._size + n)

    @torch.no_grad()
    def compute_novelty(self, query_embeddings: Tensor) -> Tensor:
        """Compute $N(q) = 1 - \\max_{q_j \\in \\mathcal{B}} \\operatorname{sim}(q, q_j)$ in $[0, 2]$."""
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.unsqueeze(0)
        if self._size == 0:
            return torch.ones(
                query_embeddings.shape[0], device=query_embeddings.device, dtype=torch.float32
            )
        q_norm = self._normalize_pearson(query_embeddings).to(
            device=self.device, dtype=self.dtype
        )
        active = self._buffer[: self._size]
        max_sim = (q_norm @ active.t()).float().clamp(-1.0, 1.0).max(dim=-1).values
        return (1.0 - max_sim).clamp(0.0, 2.0).to(query_embeddings.device)

    def active_embeddings(self) -> Tensor:
        return self._buffer[: self._size]


@dataclass
class ScaffoldEntry:
    scaffold_text: str
    best_reward: float
    step_updated: int


class ScaffoldCache:
    """Ornith-1.0 best-scaffold cache retaining winning scaffolds per task ID."""

    def __init__(self) -> None:
        self._entries: dict[str, ScaffoldEntry] = {}

    def get(self, task_id: str, default: str = "abduction -> association -> decision -> deduction") -> str:
        entry = self._entries.get(task_id)
        return default if entry is None else entry.scaffold_text

    def best_reward(self, task_id: str) -> float:
        entry = self._entries.get(task_id)
        return 0.0 if entry is None else entry.best_reward

    def update_if_better(
        self,
        task_id: str,
        candidate_scaffolds: list[str],
        rewards: Tensor | list[float],
        *,
        step: int = 0,
    ) -> bool:
        """Update `best_scaffolds[task_id]` iff $\\max_i R_i > \\max(0, R_{\\text{cached}})$."""
        r_list = rewards.detach().cpu().tolist() if isinstance(rewards, Tensor) else list(rewards)
        if not r_list or len(r_list) != len(candidate_scaffolds):
            raise ValueError("candidate_scaffolds and rewards must have the same non-zero length")
        best_idx = max(range(len(r_list)), key=r_list.__getitem__)
        max_r = float(r_list[best_idx])
        threshold = max(0.0, self.best_reward(task_id))
        if max_r > threshold:
            self._entries[task_id] = ScaffoldEntry(
                scaffold_text=candidate_scaffolds[best_idx],
                best_reward=max_r,
                step_updated=step,
            )
            return True
        return False


def frontier_difficulty(
    pass_rate: Tensor | float,
    *,
    target_p: float = 0.2,
    sigma: float = 0.25,
) -> Tensor:
    """Compute Ornith-1.5 capability-frontier difficulty $D(p) = \\exp(-(p - p^*)^2 / (2\\sigma^2))$."""
    if sigma <= 0 or not 0.0 <= target_p <= 1.0:
        raise ValueError("target_p must be in [0, 1] and sigma > 0")
    p = torch.as_tensor(pass_rate, dtype=torch.float32)
    return torch.exp(-((p - target_p).square()) / (2.0 * sigma * sigma))


def compute_task_reward(
    validity: Tensor | float,
    pass_rate: Tensor | float,
    novelty: Tensor | float,
    *,
    target_p: float = 0.2,
    sigma: float = 0.25,
) -> Tensor:
    """Compute Ornith-1.5 Stage-1 Task Reward $R_{\\text{task}} = V(q, s) \\cdot D(p) \\cdot N(q)$.

    Enforces the hard gate $V(q, s) <= 0 \\implies R_{\\text{task}} = 0$.
    """
    v = torch.as_tensor(validity, dtype=torch.float32).clamp(0.0, 1.0)
    d = frontier_difficulty(pass_rate, target_p=target_p, sigma=sigma).to(v.device)
    n = torch.as_tensor(novelty, dtype=torch.float32, device=v.device).clamp(0.0, 2.0)
    hard_gate = (v > 0.0).float()
    return hard_gate * v * d * n


def compute_harness_reward(
    task_alignment: Tensor,
    harness_scores: Tensor,
    reference_utilities: Tensor,
    *,
    negative_harness_scores: Tensor | None = None,
    hack_resistance: Tensor | float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """Compute Ornith-1.5 Stage-2 Harness Reward $R_{\\text{harness}} = C(q, h) \\cdot F(h, \\{\\tau_i\\}) \\cdot H(h)$."""
    c = task_alignment.float().clamp(0.0, 1.0)
    h = torch.as_tensor(hack_resistance, dtype=torch.float32, device=c.device).clamp(0.0, 1.0)

    # Reward Fidelity F(h, {tau_i}): positive correlation between harness and frozen Evaluation head
    if harness_scores.shape[-1] >= 3:
        corr = pearson_similarity(harness_scores.float(), reference_utilities.float(), eps=eps)
        fidelity = ((corr + 1.0) * 0.5).clamp(0.0, 1.0)
    else:
        diff = (harness_scores.float() - reference_utilities.float()).abs().mean(dim=-1)
        fidelity = (1.0 - diff).clamp(0.0, 1.0)

    # Corrupted negative rollouts must score 0 in a faithful harness
    if negative_harness_scores is not None:
        neg_leak = (negative_harness_scores.float() > 0.0).any(dim=-1)
        fidelity = fidelity.masked_fill(neg_leak, 0.0)

    return c * fidelity * h


def group_normalized_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute group-normalized GRPO advantages $A_i = (R_i - \\mu_R) / (\\sigma_R + \\epsilon)$."""
    r = rewards.float()
    if r.shape[-1] <= 1:
        return torch.zeros_like(r)
    mean = r.mean(dim=-1, keepdim=True)
    std = r.std(dim=-1, unbiased=False, keepdim=True)
    return torch.where(std > eps, (r - mean) / (std + eps), torch.zeros_like(r))


def staleness_weight(
    policy_age: Tensor | int,
    *,
    k1: int = 1,
    k2: int = 4,
    decay_lambda: float = 0.5,
    topology_changed: Tensor | bool = False,
) -> Tensor:
    """Compute Ornith piecewise exponential staleness weight $w(d_t)$."""
    if k1 < 0 or k2 < k1 or decay_lambda < 0:
        raise ValueError("Require 0 <= k1 <= k2 and decay_lambda >= 0")
    age = torch.as_tensor(policy_age, dtype=torch.float32)
    topo = torch.as_tensor(topology_changed, dtype=torch.bool, device=age.device)
    fresh = age <= k1
    decaying = (age > k1) & (age <= k2)
    weights = torch.where(
        fresh,
        torch.ones_like(age),
        torch.where(
            decaying,
            torch.exp(-decay_lambda * (age - k1)),
            torch.zeros_like(age),
        ),
    )
    return weights.masked_fill(topo, 0.0)


def grpo_clipped_loss(
    current_log_probs: Tensor,
    behavior_log_probs: Tensor,
    advantages: Tensor,
    policy_age: Tensor | int = 0,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    k1: int = 1,
    k2: int = 4,
    decay_lambda: float = 0.5,
    topology_changed: Tensor | bool = False,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute Ornith staleness-weighted asymmetric-clipped GRPO loss."""
    log_ratio = (current_log_probs.float() - behavior_log_probs.float()).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    adv = advantages.float()
    while adv.ndim < ratio.ndim:
        adv = adv.unsqueeze(-1)

    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high) * adv
    surrogate = -torch.minimum(unclipped, clipped)

    w = staleness_weight(
        policy_age,
        k1=k1,
        k2=k2,
        decay_lambda=decay_lambda,
        topology_changed=topology_changed,
    ).to(surrogate.device)
    while w.ndim < surrogate.ndim:
        w = w.unsqueeze(-1)

    weighted = surrogate * w
    if valid_mask is not None:
        mask_f = valid_mask.float()
        return (weighted * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return weighted.mean()


def detect_doom_loop(
    tokens: list[int] | tuple[int, ...],
    *,
    min_period: int = 4,
    max_period: int = 128,
    min_repeats: int = 2,
    min_total_repeated: int = 10,
) -> tuple[int, int, int] | None:
    """Detect an inner repeating token/byte span (LiquidAI Antidoom `find_inner_repetition`).

    Returns `(repeat_start_idx, rejected_token_id, period)` where `repeat_start_idx = start + period`
    is the exact index of the first token of the first repeat ($y_{\\text{rejected}}$), or `None`
    if no loop is found.
    """
    n = len(tokens)
    if n < min_period * min_repeats:
        return None
    eff_max_p = min(max_period, n // min_repeats)
    for start in range(0, n - min_period * min_repeats + 1):
        for period in range(min_period, eff_max_p + 1):
            if start + period * min_repeats > n:
                break
            pattern = tokens[start : start + period]
            reps = 1
            pos = start + period
            while pos + period <= n and tokens[pos : pos + period] == pattern:
                reps += 1
                pos += period
            if reps >= min_repeats and (reps * period) >= min_total_repeated:
                repeat_start = start + period
                return repeat_start, int(tokens[repeat_start]), period
    return None


def compute_antidoom_ftpo_loss(
    logits: Tensor,
    ref_logits: Tensor,
    rejected_tokens: Tensor,
    chosen_tokens: Tensor,
    *,
    beta: float = 0.5,
    active_reg: float = 0.02,
    other_reg: float = 0.50,
    target_chosen_win: float = 0.35,
) -> tuple[Tensor, float]:
    """Compute LiquidAI Antidoom Final Token Preference Optimization (FTPO) loss in logit space.

    Key properties from LiquidAI Antidoom:
    1. **Single trigger-token targeting:** Evaluates logits `[B, V]` strictly at the token
       position immediately preceding a repetition loop (`repeat_start - 1`).
    2. **Multi-chosen alternatives:** Spreads preference pressure across $K$ plausible
       continuation tokens (`chosen_tokens` `[B, K]`) vs. the single loop-starting token
       (`rejected_tokens` `[B]`).
    3. **Two-part logit-space regularization:** Omits full-vocab softmax in the preference term
       and tightly anchors unrelated vocabulary tokens (`other_reg`) while allowing chosen/rejected
       logits (`active_reg`) to move until `chosen_win >= target_chosen_win`.
    """
    z = logits.float()
    z_ref = ref_logits.detach().float()
    bsz, vocab_size = z.shape
    rej = rejected_tokens.long().clamp(0, vocab_size - 1)
    cho = chosen_tokens.long().clamp(0, vocab_size - 1)
    if cho.ndim == 1:
        cho = cho.unsqueeze(-1)

    delta = z - z_ref
    delta_rej = delta.gather(1, rej.unsqueeze(1)).squeeze(1)
    delta_cho = delta.gather(1, cho).mean(dim=1)

    raw_rej = z.gather(1, rej.unsqueeze(1)).squeeze(1)
    raw_cho = z.gather(1, cho).mean(dim=1)
    win_vec = (raw_cho > raw_rej).float()
    chosen_win = float(win_vec.mean().item())

    # Early-stopping gate per LiquidAI Antidoom: once chosen_win exceeds target_chosen_win (~0.35),
    # attenuate preference pressure to prevent over-suppressing valid discourse words ("the", "So").
    pref_vec = -F.logsigmoid(beta * (delta_cho - delta_rej))
    gate = 0.05 if chosen_win >= target_chosen_win else 1.0
    pref_loss = gate * pref_vec.mean()

    active_mask = torch.zeros((bsz, vocab_size), dtype=torch.bool, device=z.device)
    active_mask.scatter_(1, rej.unsqueeze(1), True)
    active_mask.scatter_(1, cho, True)

    sq_delta = delta.square()
    active_l2 = (sq_delta * active_mask.float()).sum(dim=-1) / active_mask.float().sum(dim=-1).clamp_min(1.0)
    other_mask = (~active_mask).float()
    other_l2 = (sq_delta * other_mask).sum(dim=-1) / other_mask.sum(dim=-1).clamp_min(1.0)
    reg_loss = active_reg * active_l2.mean() + other_reg * other_l2.mean()

    return pref_loss + reg_loss, chosen_win



# ---------------------------------------------------------------------------
# Bounded Structural Self-Improvement (Expert Mitosis & Staleness Pruning)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertMorphResult:
    action: str
    old_num_experts: int
    new_num_experts: int
    old_parameters: int
    new_parameters: int
    old_to_new: tuple[int | None, ...]
    parent_index: int | None = None
    removed_index: int | None = None


def _all_moe_layers(backbone: MCL4CEDBackbone) -> list[nn.Module]:
    return [*backbone.encoder_layers, *backbone.decoder_layers]


def parameter_count(model: nn.Module) -> int:
    """Count unique learned parameters in `model`."""
    return sum(p.numel() for p in model.parameters())


def projected_parameter_count(backbone: MCL4CEDBackbone, num_experts: int) -> int:
    """Exact parameter count after uniformly resizing all MoE expert banks to `num_experts`."""
    if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    layers = _all_moe_layers(backbone)
    slot_size = sum(
        sum(p.numel() for p in layer.moe.experts[0].parameters())
        + layer.moe.router.weight[0].numel()
        for layer in layers
    )
    return parameter_count(backbone) + (num_experts - backbone.config.num_experts) * slot_size


def _fresh_state(master: Tensor) -> dict[str, Any]:
    return {
        "master": master.detach().clone(),
        "step": 0,
        "exp_avg": torch.zeros_like(master),
        "exp_avg_sq": torch.zeros_like(master),
    }


def _install_expert_banks(
    root_model: nn.Module,
    backbone: MCL4CEDBackbone,
    optimizer: CPUOffloadAdamW,
    banks: list,
    replacements: dict,
    insertions: dict,
    states: dict,
    new_count: int,
) -> None:
    group_for = {
        parameter: index
        for index, group in enumerate(optimizer.param_groups)
        for parameter in group["params"]
    }
    groups: list[list[nn.Parameter]] = [[] for _ in optimizer.param_groups]
    for old_parameter in root_model.parameters():
        parameter = replacements.get(old_parameter, old_parameter)
        if parameter is not None:
            groups[group_for[old_parameter]].append(parameter)
        for added_parameter, parent_parameter in insertions.get(old_parameter, ()):
            groups[group_for[parent_parameter]].append(added_parameter)
    configuration = replace(backbone.config, num_experts=new_count)
    optimizer.reconfigure_parameters(groups, states)
    for bank, router, experts in banks:
        bank.router = router
        bank.experts = experts
        bank.num_experts = new_count
    backbone.config = configuration


def grow_experts(
    root_model: nn.Module,
    backbone: MCL4CEDBackbone,
    optimizer: CPUOffloadAdamW,
    *,
    parent_index: int,
    max_parameters: int = MAX_PARAMETERS,
) -> ExpertMorphResult:
    """Clone `parent_index` expert slot across all encoder and decoder layers."""
    old_count = backbone.config.num_experts
    if not 0 <= parent_index < old_count:
        raise ValueError("parent_index out of range")
    old_params = parameter_count(root_model)
    delta = projected_parameter_count(backbone, old_count + 1) - parameter_count(backbone)
    new_params = old_params + delta
    if new_params > max_parameters:
        raise ValueError(f"Expert growth would exceed max_parameters={max_parameters}")
    optimizer.invalidate_transfer_buffers()

    banks, replacements, insertions = [], {}, {}
    states = dict(optimizer.state)
    for layer in _all_moe_layers(backbone):
        bank = layer.moe
        parent = bank.experts[parent_index]
        added = deepcopy(parent)
        additions = []
        for parent_param, param in zip(parent.parameters(), added.parameters()):
            states[param] = _fresh_state(states[parent_param]["master"])
            additions.append((param, parent_param))
        insertions[list(bank.experts[-1].parameters())[-1]] = additions

        original = bank.router.weight
        router = deepcopy(bank.router)
        router.out_features = old_count + 1
        router.weight = nn.Parameter(
            torch.cat((original.detach(), original.detach()[parent_index : parent_index + 1])),
            requires_grad=original.requires_grad,
        )
        old_state = states.pop(original)
        state = {**old_state}
        for key in ("master", "exp_avg", "exp_avg_sq"):
            old_rows = old_state[key]
            added_row = (
                old_rows[parent_index : parent_index + 1]
                if key == "master"
                else torch.zeros_like(old_rows[parent_index : parent_index + 1])
            )
            state[key] = torch.cat((old_rows, added_row))
        states[router.weight] = state
        replacements[original] = router.weight
        new_experts = nn.ModuleList([*bank.experts, added])
        new_experts.training = bank.experts.training
        banks.append((bank, router, new_experts))

    _install_expert_banks(
        root_model, backbone, optimizer, banks, replacements, insertions, states, old_count + 1
    )
    return ExpertMorphResult(
        action="grow",
        old_num_experts=old_count,
        new_num_experts=old_count + 1,
        old_parameters=old_params,
        new_parameters=new_params,
        old_to_new=tuple(range(old_count)),
        parent_index=parent_index,
    )


def prune_experts(
    root_model: nn.Module,
    backbone: MCL4CEDBackbone,
    optimizer: CPUOffloadAdamW,
    *,
    expert_index: int,
) -> ExpertMorphResult:
    """Remove `expert_index` slot across all encoder and decoder layers."""
    old_count = backbone.config.num_experts
    if not 0 <= expert_index < old_count:
        raise ValueError("expert_index out of range")
    if old_count - 1 < max(backbone.config.top_k, 2):
        raise ValueError("Pruning must retain at least top_k and two experts")
    old_params = parameter_count(root_model)
    delta = parameter_count(backbone) - projected_parameter_count(backbone, old_count - 1)
    new_params = old_params - delta
    optimizer.invalidate_transfer_buffers()

    banks, replacements = [], {}
    states = dict(optimizer.state)
    keep = [i for i in range(old_count) if i != expert_index]
    for layer in _all_moe_layers(backbone):
        bank = layer.moe
        for param in bank.experts[expert_index].parameters():
            replacements[param] = None
            states.pop(param)
        original = bank.router.weight
        router = deepcopy(bank.router)
        router.out_features = old_count - 1
        router.weight = nn.Parameter(
            original.detach()[keep].clone(),
            requires_grad=original.requires_grad,
        )
        old_state = states.pop(original)
        states[router.weight] = {
            **old_state,
            **{key: old_state[key][keep].clone() for key in ("master", "exp_avg", "exp_avg_sq")},
        }
        replacements[original] = router.weight
        new_experts = nn.ModuleList([bank.experts[i] for i in keep])
        new_experts.training = bank.experts.training
        banks.append((bank, router, new_experts))

    _install_expert_banks(
        root_model, backbone, optimizer, banks, replacements, {}, states, old_count - 1
    )
    old_to_new = tuple(
        None if i == expert_index else (i if i < expert_index else i - 1) for i in range(old_count)
    )
    return ExpertMorphResult(
        action="prune",
        old_num_experts=old_count,
        new_num_experts=old_count - 1,
        old_parameters=old_params,
        new_parameters=new_params,
        old_to_new=old_to_new,
        removed_index=expert_index,
    )


# ---------------------------------------------------------------------------
# Live Ornith-1.0 / 1.5 Self-Scaffolding & Self-Improvement Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrnithTask:
    """A verifiable Ornith-1.5/2.5 frontier task with harness specification, granular skill, and Star Difficulty."""

    task_id: str
    domain: str
    prompt_q: str
    harness_h: str
    scaffold_s: str
    candidate_rollouts: tuple[str, ...]
    harness_scores: tuple[float, ...]
    gold_index: int
    gold_thought: str
    gold_answer: str
    skill: str = "general.reasoning"
    star_difficulty: float = 1.0
    recursion_depth: int = 1
    difficulty_mods: tuple[str, ...] = ("NM",)


@dataclass
class OrnithStepOutput:
    """Telemetry and differentiable loss from a live Ornith-1.0 / 2.5 + PP-Weighted + Dream-RSI step."""

    loss: Tensor
    grpo_loss: Tensor
    self_distill_loss: Tensor
    task_reward: float
    harness_reward: float
    rollout_reward: float
    novelty: float
    pass_rate: float
    frontier_difficulty: float
    veto_rate: float
    scaffolds_cached: int
    novelty_buffer_size: int = 0
    dream_cycle: int = 0
    dream_replay_score: float = 0.0
    dream_initial_score: float = 0.0
    dream_accepted_revisions: int = 0
    dream_replays_dreamed: int = 0
    dream_worlds_archived: int = 0
    dream_active_policy: str = ""
    antidoom_loss: float = 0.0
    antidoom_chosen_win: float = 0.0
    raw_pp: float = 0.0
    weighted_pp: float = 0.0
    total_pp: float = 0.0
    pp_weight: float = 1.0
    star_difficulty: float = 1.0
    recursion_depth: int = 1
    active_skill: str = ""

    @property
    def mean_rollout_reward(self) -> float:
        return self.rollout_reward


# ---------------------------------------------------------------------------
# osu!-Style Performance Point (`pp`) Weighting & Best Performance Leaderboard
# ---------------------------------------------------------------------------

PP_DECAY_BASE: float = 0.95

MOD_MULTIPLIERS: dict[str, float] = {
    "NM": 1.00,  # NoMod baseline
    "HR": 1.12,  # HardRock: tighter near-miss algebraic/boundary distractors
    "HD": 1.10,  # Hidden: intermediate multi-hop state hints masked
    "DT": 1.18,  # DoubleTime: compounded secondary verification constraint
    "FL": 1.15,  # Flashlight: partial observability / latent variable abduction
    "INF": 1.25,  # Infinite-Horizon Recursive Chain
}

SKILL_TAXONOMY: tuple[tuple[str, str], ...] = (
    # 0..19: Primitive Procedural & Formal Skills
    ("algebra.linear_diophantine", "symbolic_algebra"),
    ("code.list_filter_transform", "algorithmic_code"),
    ("math.matrix_determinant_2x2", "linear_algebra"),
    ("physics.kinematics_displacement", "physics_kinematics"),
    ("algorithms.project_dag_critical_path", "project_management"),
    ("number_theory.modular_ring_inverse", "modular_number_theory"),
    ("dp.linear_recurrence_unroll", "dynamic_programming"),
    ("code.prefix_accumulator_loop", "algorithmic_code"),
    ("bitwise.xor_involution_mask", "bitwise_systems"),
    ("systems.fenwick_tree_stride", "data_structures"),
    ("number_theory.gcd_euclidean_step", "modular_number_theory"),
    ("systems.service_graph_throughput", "distributed_systems"),
    ("code.horner_polynomial_eval", "algorithmic_code"),
    ("algorithms.binary_search_midpoint", "search_algorithms"),
    ("relational.inner_join_cardinality", "sql_relational"),
    ("concurrency.lock_cycle_deadlock", "systems_concurrency"),
    ("physics.calorimetry_heat_transfer", "physics_kinematics"),
    ("ifeval.lexical_case_punctuation", "invariant_verification"),
    ("code.arithmetic_progression_gen", "algorithmic_code"),
    ("logic.horn_clause_forward_chain", "multi_hop_deduction"),
    # 20..23: Cross-Template Compositional Pipelines
    ("composition.algebra_to_code_pipeline", "compositional_code_math"),
    ("composition.kinematics_to_modring_pipeline", "compositional_physics_math"),
    ("composition.dag_to_slack_code_pipeline", "compositional_code_pm"),
    ("composition.deduction_to_ifeval_pipeline", "compositional_deduction_ifeval"),
    # 24..33: Targeted Benchmark & Domain Specializations (ARC, HellaSwag, MMLU, Code)
    ("science.experimental_control_abduction", "arc_science"),
    ("science.thermodynamics_energy_conservation", "arc_physics"),
    ("science.stoichiometry_limiting_reagent", "arc_chemistry"),
    ("science.genetics_hardy_weinberg", "arc_biology"),
    ("commonsense.physical_precondition_consequence", "hellaswag_commonsense"),
    ("mmlu.microeconomics_marginal_equilibrium", "mmlu_economics"),
    ("mmlu.formal_logic_quantifier_contrapositive", "mmlu_logic"),
    ("mmlu.bayesian_posterior_frequency", "mmlu_probability"),
    ("mmlu.computer_arch_cache_amat", "mmlu_cs_systems"),
    ("code.recursive_state_transducer", "recursive_code"),
    # 34..35: Infinite-Horizon Recursive RL Loops
    ("recursion.unbounded_multihop_chain", "infinite_recursion"),
    ("recursion.self_referential_fixed_point", "infinite_recursion"),
    # 36..39: Sherlockian Deductive, Abductive, Forensic Timeline & Cipher Induction Suite
    ("deduction.sherlock_eliminative_grid", "sherlock_eliminative_deduction"),
    ("abduction.sherlock_diagnostic_trace", "sherlock_diagnostic_abduction"),
    ("deduction.sherlock_alibi_contradiction", "sherlock_forensic_timeline"),
    ("induction.sherlock_cipher_induction", "sherlock_pattern_induction"),
    # 40..47: Sherlockian Polymath, HackTheBox, Cryptography, Frontier Science, Behavioral & Cross-Skill Suite
    ("security.hackthebox_sandbox_escape", "hackthebox_security"),
    ("math.sherlock_word_problem_system", "sherlock_word_math"),
    ("crypto.multilayer_trapdoor_stream", "sherlock_cryptography"),
    ("science.frontier_open_problem_reduction", "frontier_science_math"),
    ("behavior.game_theoretic_deception_profiling", "behavioral_analysis"),
    ("code.algorithmic_synthesis_hard", "hard_algorithmic_code"),
    ("mixing.cross_skill_sherlock_crypto_code", "cross_skill_synthesis"),
    ("mixing.cross_skill_htb_science_behavior", "cross_skill_synthesis"),
    # 48..55: Agentic Computer-Use, Tool-Use, Terminal, CyberGym & PhD Cybersecurity Infinite-Recursive Suite
    ("terminal.posix_pipeline_state_machine", "terminal_rl"),
    ("tool_use.multi_hop_schema_dag", "tool_use_rl"),
    ("computer_use.gui_dom_accessibility_tree", "computer_use_rl"),
    ("cybergym.vuln_sanitizer_root_cause_patch", "cybergym_rl"),
    ("cybersec.ptes_owasp_attack_graph_mitigation", "cybersec_phd_rl"),
    ("recursion.infinite_terminal_tool_sandbox_loop", "infinite_agentic_recursion"),
    ("mixing.cross_skill_terminal_tool_cybergym", "cross_skill_agentic"),
    ("mixing.cross_skill_computer_use_cybersec_deduction", "cross_skill_agentic"),
    # 56..63: LeetCode & Competitive Infinite-Recursive Algorithmic Coding RL Suite
    ("leetcode.recursive_dp_memoization", "leetcode_dp_rl"),
    ("leetcode.divide_and_conquer_tree_recurrence", "leetcode_divconq_rl"),
    ("leetcode.backtracking_subset_constraint_pruning", "leetcode_backtracking_rl"),
    ("leetcode.graph_dfs_dag_topological_longest_path", "leetcode_graph_rl"),
    ("leetcode.monotonic_window_subarray_optimization", "leetcode_array_window_rl"),
    ("leetcode.parametric_binary_search_on_answer", "leetcode_bsearch_rl"),
    ("leetcode.tree_postorder_robber_interval_dp", "leetcode_tree_interval_rl"),
    ("leetcode.grandmaster_recursive_hybrid_synthesis", "leetcode_grandmaster_rl"),
)


def compute_mod_multiplier(
    mods: tuple[str, ...] | list[str] = ("NM",),
    recursion_depth: int = 1,
) -> float:
    """Compute multiplicative difficulty bonus from active osu!-style modifiers and recursion depth."""
    mult = 1.0
    seen: set[str] = set()
    for m in mods:
        key = str(m).upper().strip()
        if key and key not in seen:
            seen.add(key)
            mult *= MOD_MULTIPLIERS.get(key, 1.0)
    extra_hops = max(0, int(recursion_depth) - 1)
    if "INF" in seen or extra_hops >= 2:
        mult *= 1.0 + 0.06 * extra_hops
    return round(float(mult), 6)


def compute_star_difficulty(
    tier: int,
    recursion_depth: int = 1,
    mods: tuple[str, ...] | list[str] = ("NM",),
) -> float:
    """Compute continuous Star Difficulty $d^\\star \\in [1.0, \\infty)$ scaling unboundedly with tier & recursion."""
    t_val = max(1, int(tier))
    r_val = max(1, int(recursion_depth))
    mod_mult = compute_mod_multiplier(mods, recursion_depth=r_val)
    base_stars = 1.0 + 0.55 * (t_val - 1) + 0.35 * (r_val - 1)
    return round(float(base_stars * math.sqrt(mod_mult)), 4)


def compute_raw_pp(
    star_difficulty: float,
    verified_reward: float,
    alignment: float = 1.0,
    *,
    mods: tuple[str, ...] | list[str] = ("NM",),
    recursion_depth: int = 1,
) -> float:
    """Compute full pre-weighted performance points $p$ of a single play before $0.95^{n-1}$ ranking decay."""
    stars = max(0.5, float(star_difficulty))
    acc = max(0.0, min(1.0, float(verified_reward)))
    align = max(0.1, min(1.0, float(alignment)))
    mod_mult = compute_mod_multiplier(mods, recursion_depth=recursion_depth)
    raw = 100.0 * (stars ** 1.15) * mod_mult * (acc ** 1.25) * math.sqrt(align)
    return round(float(raw), 4)


def compute_inverse_star_miss_penalty(
    star_difficulty: float,
    verified_reward: float,
    *,
    pass_threshold: float = 0.50,
    base_penalty_pp: float = 45.0,
    star_exponent: float = 1.25,
) -> float:
    """Compute inverse-star PP miss penalty $P_{\\text{miss}}(d^\\star, r) = P_0 \\cdot \\max\\!\\left(0, \\frac{\\tau - r}{\\tau}\\right) / \\max(1.0, d^\\star)^{\\alpha}$.

    Failing a $1.0\\star$ task ($r < 0.50$) incurs maximum PP loss ($\\le 45.0\\text{pp}$) and strong
    corrective RL gradient pressure, whereas failing a high-star frontier task ($d^\\star \\gg 1$)
    incurs near-zero PP loss ($P_{\\text{miss}} \\to 0$).
    """
    r = max(0.0, min(1.0, float(verified_reward)))
    thresh = max(1e-4, float(pass_threshold))
    if r >= thresh:
        return 0.0
    miss_severity = (thresh - r) / thresh
    inv_star_weight = 1.0 / (max(1.0, float(star_difficulty)) ** float(star_exponent))
    return round(float(base_penalty_pp * miss_severity * inv_star_weight), 4)


def compute_weighted_pp_series(
    scores: list[float] | tuple[float, ...] | Tensor,
    *,
    decay: float = PP_DECAY_BASE,
) -> list[float]:
    """Rank raw performance point scores $p_1 \\ge p_2 \\ge \\dots \\ge p_N$ and weight the $n$-th play by:

    $$\\text{Weighted } pp_n = p_n \\cdot 0.95^{n-1}$$

    For example, top-5 plays `[110, 100, 100, 90, 80]` with `decay=0.95` produce
    `[110.0, 95.0, 90.25, 77.1638, 65.1605]` ($\\approx 110, 95, 90, 77, 65\\text{pp}$).
    """
    if not (0.0 < float(decay) <= 1.0):
        raise ValueError("decay must be in (0, 1]")
    vals = (
        scores.detach().cpu().float().tolist()
        if isinstance(scores, Tensor)
        else [float(x) for x in scores]
    )
    ranked = sorted((max(0.0, v) for v in vals), reverse=True)
    return [float(p * (decay ** idx)) for idx, p in enumerate(ranked)]


def compute_total_pp(
    scores: list[float] | tuple[float, ...] | Tensor,
    *,
    decay: float = PP_DECAY_BASE,
) -> float:
    """Compute total weighted Performance Points $\\text{Total } PP = \\sum_{n=1}^{N} p_n \\cdot 0.95^{n-1}$."""
    return round(float(sum(compute_weighted_pp_series(scores, decay=decay))), 4)


@dataclass(frozen=True)
class PPSubmissionResult:
    """Receipt from submitting a play $p$ to `OrnithPPLeaderboard`."""

    skill: str
    raw_pp: float
    skill_rank: int
    global_rank: int
    skill_weight: float
    global_weight: float
    weighted_pp: float
    delta_skill_pp: float
    delta_total_pp: float
    skill_total_pp: float
    global_total_pp: float
    grad_scale: float
    miss_penalty_pp: float = 0.0


class OrnithPPLeaderboard:
    """osu!-style Best Performance Leaderboard tracking per-skill and global top plays with $p \\cdot 0.95^{n-1}$ decay.

    Automatically attenuates RL gradient weight when the agent farms already-mastered difficulty tiers
    (which place at deep rank $n \\gg 1$ where $0.95^{n-1} \\to 0$), while amplifying policy updates when
    the agent solves higher-difficulty tasks or under-trained skills that place near $n = 1$ ($0.95^0 = 1.0$),
    or when the agent fails a low-star ($1\\star$) task via inverse-star miss penalty.
    """

    def __init__(
        self,
        *,
        decay: float = PP_DECAY_BASE,
        max_per_skill: int = 64,
        max_global: int = 256,
    ) -> None:
        if not (0.0 < float(decay) <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        self.decay = float(decay)
        self.max_per_skill = max(1, int(max_per_skill))
        self.max_global = max(1, int(max_global))
        self.skill_scores: dict[str, list[float]] = {}
        self.global_scores: list[float] = []
        self.play_counts: dict[str, int] = {}

    @staticmethod
    def _placement_rank(sorted_scores: list[float], score: float) -> int:
        """Return 1-indexed placement rank $n \\ge 1$ of `score` in descending `sorted_scores`."""
        higher_count = sum(1 for s in sorted_scores if s > score + 1e-7)
        return higher_count + 1

    def skill_pp(self, skill: str) -> float:
        return compute_total_pp(self.skill_scores.get(skill, ()), decay=self.decay)

    def total_pp(self) -> float:
        return compute_total_pp(self.global_scores, decay=self.decay)

    def submit_play(
        self,
        skill: str,
        raw_pp: float,
        *,
        star_difficulty: float | None = None,
        verified_reward: float | None = None,
    ) -> PPSubmissionResult:
        """Record a play worth `raw_pp` in `skill`, returning its $n$-th place $0.95^{n-1}$ weight, inverse-star miss penalty, and $\\Delta PP$."""
        p = max(0.0, float(raw_pp))
        cur_skill = self.skill_scores.setdefault(skill, [])
        self.play_counts[skill] = self.play_counts.get(skill, 0) + 1

        old_skill_pp = compute_total_pp(cur_skill, decay=self.decay)
        old_global_pp = compute_total_pp(self.global_scores, decay=self.decay)

        n_skill = self._placement_rank(cur_skill, p)
        n_global = self._placement_rank(self.global_scores, p)

        w_skill = float(self.decay ** (n_skill - 1))
        w_global = float(self.decay ** (n_global - 1))
        weighted_p = float(p * w_skill)

        miss_penalty = 0.0
        inv_fail_boost = 0.0
        if star_difficulty is not None and verified_reward is not None:
            miss_penalty = compute_inverse_star_miss_penalty(star_difficulty, verified_reward)
            if float(verified_reward) < 0.50:
                miss_sev = (0.50 - max(0.0, float(verified_reward))) / 0.50
                inv_fail_boost = 1.25 * miss_sev / (max(1.0, float(star_difficulty)) ** 1.25)

        eroded = False
        if miss_penalty > 0.0 and cur_skill and cur_skill[0] >= 50.0:
            # Failing a task on an established skill erodes its top leaderboard score inversely with star difficulty
            cur_skill[0] = max(0.0, cur_skill[0] - miss_penalty)
            cur_skill.sort(reverse=True)
            eroded = True
            if self.global_scores and self.global_scores[0] >= 50.0:
                self.global_scores[0] = max(0.0, self.global_scores[0] - miss_penalty)
                self.global_scores.sort(reverse=True)
        else:
            cur_skill.append(p)
            cur_skill.sort(reverse=True)
            if len(cur_skill) > self.max_per_skill:
                del cur_skill[self.max_per_skill :]

            self.global_scores.append(p)
            self.global_scores.sort(reverse=True)
            if len(self.global_scores) > self.max_global:
                del self.global_scores[self.max_global :]

        new_skill_pp = compute_total_pp(cur_skill, decay=self.decay)
        new_global_pp = compute_total_pp(self.global_scores, decay=self.decay)

        delta_s = (new_skill_pp - old_skill_pp) if eroded else max(0.0, new_skill_pp - old_skill_pp)
        delta_g = (new_global_pp - old_global_pp) if eroded else max(0.0, new_global_pp - old_global_pp)
        rel_gain = max(0.0, delta_s) / max(1.0, p)

        # High gradient scale when placing near #1 OR when failing a low-star fundamental task (inverse-star boost)
        grad_scale = max(0.25, min(2.00, 0.25 + 0.95 * w_skill + 0.45 * rel_gain + inv_fail_boost))

        return PPSubmissionResult(
            skill=skill,
            raw_pp=round(p, 4),
            skill_rank=n_skill,
            global_rank=n_global,
            skill_weight=round(w_skill, 5),
            global_weight=round(w_global, 5),
            weighted_pp=round(weighted_p, 4),
            delta_skill_pp=round(delta_s, 4),
            delta_total_pp=round(delta_g, 4),
            skill_total_pp=round(new_skill_pp, 4),
            global_total_pp=round(new_global_pp, 4),
            grad_scale=round(grad_scale, 4),
            miss_penalty_pp=round(miss_penalty, 4),
        )

    def lowest_pp_skill_indices(self, total_families: int = len(SKILL_TAXONOMY)) -> list[int]:
        """Return family indices `0..total_families-1` ordered from lowest to highest current `skill_pp`."""
        n_f = min(int(total_families), len(SKILL_TAXONOMY))
        return sorted(
            range(n_f),
            key=lambda k: (self.skill_pp(SKILL_TAXONOMY[k][0]), self.play_counts.get(SKILL_TAXONOMY[k][0], 0), k),
        )


_SCAFFOLD_CANDIDATES: tuple[str, ...] = (
    "abduction -> association -> decision -> deduction -> evaluation -> metacognition",
    "association -> deduction -> decision -> evaluation -> metacognition",
    "abduction -> deduction -> decision -> metacognition",
    "decision -> deduction -> evaluation -> metacognition",
    "abstraction -> deduction -> decision -> evaluation -> metacognition",
    "induction -> abstraction -> deduction -> decision -> metacognition",
    "analogy -> abduction -> deduction -> decision -> evaluation -> metacognition",
)

_SAFE_AST_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assert,
    ast.Expr,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.If,
    ast.For,
    ast.While,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Attribute,
    ast.Name,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Load,
    ast.Store,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitXor,
    ast.BitAnd,
    ast.BitOr,
    ast.LShift,
    ast.RShift,
    ast.Invert,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Nonlocal,
    ast.Global,
)
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "ord": ord,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
_SAFE_METHOD_ATTRS: frozenset[str] = frozenset(
    {
        "append",
        "extend",
        "pop",
        "insert",
        "remove",
        "sort",
        "reverse",
        "copy",
        "clear",
        "index",
        "count",
        "get",
        "keys",
        "values",
        "items",
        "update",
        "setdefault",
        "add",
        "discard",
        "union",
        "intersection",
        "difference",
        "split",
        "splitlines",
        "join",
        "strip",
        "lstrip",
        "rstrip",
        "replace",
        "lower",
        "upper",
        "isdigit",
        "isalpha",
        "isalnum",
        "startswith",
        "endswith",
        "find",
    }
)
_HACK_STUB_MARKERS: tuple[str, ...] = (
    "Constant stub",
    "Default fallback",
    "Zero delta",
    "Return base",
    "Mask overflow",
    "Uninitialized pointer",
    "Empty join",
    "Deadlock immediately",
    "#T$# 0\n",
)


def _verify_thought_equations(t_body: str) -> float:
    """Check explicit integer `A op B = C` equations inside `#T$#`; return `0.25` if any arithmetic equation is false."""
    import re

    if "mod" in t_body.lower() or "%" in t_body or "^" in t_body:
        return 1.0
    # Match isolated `A op B = C` where A, B, C are not part of a chained expression like `A + B * C = D`
    pattern = re.compile(r"(?<![\w*+\-/])(\d+)\s*([+\-*])\s*(\d+)\s*=\s*(-?\d+)(?!\s*[*+\-/^])")
    for m in pattern.finditer(t_body):
        prefix = t_body[: m.start()].rstrip()
        if prefix.endswith(("*", "+", "-", "/", "%", "^", "=")):
            continue
        lhs_a = int(m.group(1))
        op = m.group(2)
        lhs_b = int(m.group(3))
        rhs_c = int(m.group(4))
        expected = (
            lhs_a + lhs_b
            if op == "+"
            else (lhs_a - lhs_b if op == "-" else lhs_a * lhs_b)
        )
        if expected != rhs_c:
            return 0.25
    return 1.0


def _verify_ifeval_constraints(a_body: str, spec: str, gold_answer: str = "") -> float:
    """Deterministically verify semicolon-separated `ifeval:` instruction-following & math constraints.

    Crucially separates semantic correctness (`exact_int:`, `exact_answer:`) from surface formatting
    constraints (`lowercase`, `no_comma`, `min_words`, `keyword`, `json_valid`):
    - If a semantic rule (`exact_int` or `exact_answer`) is present and FAILS, total reward is strictly
      capped at `<= 0.20` (preventing 0.80 reward hacking from well-formatted wrong answers).
    - Prevents number-spraying on `exact_int:` by rejecting plain-text answers that dump multiple unrelated integers.
    """
    import json
    import re

    rules = [r.strip() for r in spec[len("ifeval:") :].split(";") if r.strip()]
    if not rules:
        return 1.0 if (gold_answer and a_body.strip() == gold_answer.strip()) else 0.0

    is_json_spec = "json_valid" in rules
    parsed_json: Any = None
    if is_json_spec:
        try:
            parsed_json = json.loads(a_body)
        except Exception:
            parsed_json = None

    format_total = 0
    format_passed = 0
    semantic_total = 0
    semantic_passed = 0

    for rule in rules:
        if rule == "lowercase":
            format_total += 1
            if a_body == a_body.lower():
                format_passed += 1
        elif rule == "uppercase":
            format_total += 1
            if a_body == a_body.upper():
                format_passed += 1
        elif rule == "no_comma":
            format_total += 1
            if "," not in a_body:
                format_passed += 1
        elif rule == "json_valid":
            format_total += 1
            if parsed_json is not None:
                format_passed += 1
        elif rule.startswith("min_words:"):
            format_total += 1
            n_w = int(rule.split(":", 1)[1])
            if len(a_body.split()) >= n_w:
                format_passed += 1
        elif rule.startswith("max_words:"):
            format_total += 1
            n_w = int(rule.split(":", 1)[1])
            if 0 < len(a_body.split()) <= n_w:
                format_passed += 1
        elif rule.startswith("keyword:"):
            format_total += 1
            kw = rule.split(":", 1)[1].strip().lower()
            if kw and kw in a_body.lower():
                format_passed += 1
        elif rule.startswith("forbidden:"):
            format_total += 1
            fw = rule.split(":", 1)[1].strip().lower()
            if fw and fw not in a_body.lower():
                format_passed += 1
        elif rule.startswith("exact_int:"):
            semantic_total += 1
            target_int = int(rule.split(":", 1)[1].strip())
            nums = [int(x) for x in re.findall(r"-?\d+", a_body)]
            if is_json_spec and isinstance(parsed_json, dict):
                # For JSON tool calls, check primary quota/result field or bounded numeric values (<= 2 ints)
                if "verified_quota" in parsed_json:
                    if isinstance(parsed_json["verified_quota"], int) and int(parsed_json["verified_quota"]) == target_int:
                        semantic_passed += 1
                elif 0 < len(nums) <= 2 and target_int in nums:
                    semantic_passed += 1
            else:
                # For plain-text answers, disallow number-spraying (allow at most 1 unique integer in #A$#)
                unique_nums = list(dict.fromkeys(nums))
                if len(unique_nums) == 1 and unique_nums[0] == target_int:
                    semantic_passed += 1
        elif rule.startswith("exact_answer:"):
            semantic_total += 1
            target_str = rule.split(":", 1)[1].strip()
            if target_str and target_str.lower() in a_body.lower():
                semantic_passed += 1

    fmt_ratio = (float(format_passed) / float(format_total)) if format_total > 0 else 1.0
    if semantic_total > 0:
        if semantic_passed < semantic_total:
            # Semantic answer is wrong: grant at most 0.20 partial credit for following formatting constraints
            return round(0.20 * fmt_ratio, 4)
        if format_passed < format_total:
            # Semantic answer is right, partial credit proportional to satisfied rules
            return round(
                float(format_passed + semantic_passed) / float(format_total + semantic_total),
                4,
            )
        return 1.0
    return round(fmt_ratio, 4)


def verify_python_rollout(
    trace_text: str,
    harness_h: str,
    gold_answer: str = "",
) -> float:
    """Deterministically verify a `#T$# ... #A$#` rollout against Python assertion harness `harness_h`.

    Enforces:
    1. Mandatory non-empty `#T$#` scratchpad before `#A$#` (returns `0.0` if missing).
    2. Rejection of boundary/constant hack stubs (`0.0`).
    3. Inline `#T$#` arithmetic equation consistency check (`_verify_thought_equations`).
    4. Multi-line AST-validated compiler, single-function purity check (`FunctionDef` + `Return`),
       and sandboxed execution of `def solve(...)` + `assert ...`, relational invariant assertions,
       or `ifeval:` constraint verification.
    """
    import re

    if "#T$#" not in trace_text or "#A$#" not in trace_text:
        return 0.0
    for marker in _HACK_STUB_MARKERS:
        if marker in trace_text:
            return 0.0

    t_part, a_part = trace_text.split("#A$#", 1)
    t_body = t_part.replace("#T$#", "", 1).strip()
    a_body = a_part.strip()
    if not t_body or not a_body:
        return 0.0

    thought_consistency = _verify_thought_equations(t_body)
    harness_clean = harness_h.strip()

    # Case 0: Instruction-following / mixed-constraint verifier (`ifeval:...`)
    if harness_clean.startswith("ifeval:"):
        if_score = _verify_ifeval_constraints(a_body, harness_clean, gold_answer)
        return round(if_score * thought_consistency, 4)

    # Case 1: Executable single-line or multi-line indented Python function + `assert solve(...) == ...`
    if "def " in a_body and harness_clean.startswith("assert "):
        clean_lines = [
            ln.split("#")[0].rstrip()
            for ln in a_body.splitlines()
            if ln.split("#")[0].strip() and not ln.strip().startswith("```")
        ]
        code_src = "\n".join(clean_lines)
        program_src = f"{code_src}\n{harness_clean}\n"
        try:
            code_tree = ast.parse(code_src, mode="exec")
            func_defs = [n for n in code_tree.body if isinstance(n, ast.FunctionDef)]
            has_return = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(code_tree))
            if not func_defs or not has_return:
                return 0.0
            # Reject duplicate/shadowing function definitions (`def solve` defined twice in #A$#)
            if len({f.name for f in func_defs}) != len(func_defs):
                return 0.0
            # Require top-level statements in #A$# to be FunctionDef or constant Assign
            if not all(isinstance(stmt, (ast.FunctionDef, ast.Assign)) for stmt in code_tree.body):
                return 0.0
            # Anti-cheat: reject hardcoded lookup tables (`if n == 8: return 738 elif n == 2: return 71`)
            # when the function has no loop/comprehension and no recursive call but branches on >= 2 test literals
            defined_fn_names = {f.name for f in func_defs}
            has_loop_or_comp = any(
                isinstance(n, (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
                for n in ast.walk(code_tree)
            )
            has_self_call = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in defined_fn_names
                for f in func_defs
                for n in ast.walk(f)
            )
            if not has_loop_or_comp and not has_self_call:
                if_cmp_consts = {
                    int(c.value)
                    for n in ast.walk(code_tree)
                    if isinstance(n, (ast.If, ast.IfExp))
                    for c in ast.walk(n.test)
                    if isinstance(c, ast.Constant) and isinstance(c.value, int) and abs(int(c.value)) > 1
                }
                harness_consts = {
                    int(x) for x in re.findall(r"solve\(\s*(-?\d+)", harness_clean) if abs(int(x)) > 1
                }
                if len(if_cmp_consts & harness_consts) >= 2 or len(if_cmp_consts) >= 3:
                    return 0.0
            tree = ast.parse(program_src, mode="exec")
            for node in ast.walk(tree):
                if not isinstance(node, _SAFE_AST_NODES):
                    return 0.0
                if isinstance(node, ast.Attribute):
                    if node.attr.startswith("_") or node.attr not in _SAFE_METHOD_ATTRS:
                        return 0.0
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        continue
                    if isinstance(node.func, ast.Attribute) and node.func.attr in _SAFE_METHOD_ATTRS:
                        continue
                    return 0.0
            tick_stmt = ast.Expr(
                value=ast.Call(func=ast.Name(id="_tick", ctx=ast.Load()), args=[], keywords=[])
            )
            for node in ast.walk(tree):
                if isinstance(node, (ast.While, ast.For, ast.FunctionDef)):
                    node.body.insert(0, tick_stmt)
            ast.fix_missing_locations(tree)
            steps = [0]

            def _tick() -> None:
                steps[0] += 1
                if steps[0] > 16384:
                    raise RuntimeError("Recursion or loop iteration limit exceeded")

            env: dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS), "_tick": _tick}
            exec(compile(tree, "<ornith_verifier>", "exec"), env, env)
            return 1.0 * thought_consistency
        except AssertionError:
            # Compiles and passes AST lint checks (`FunctionDef` + `Return`), but failed full assertion suite.
            # Evaluate individual conjunct test cases (`assert c1 and c2 and ...`) for graduated partial credit
            # while keeping base_partial compatibility for single-case or 0-pass rollouts.
            base_partial = 0.25 if any(op in t_body for op in ("*", "+", "-", "if", "for", "return")) else 0.15
            try:
                assert_tree = ast.parse(harness_clean, mode="exec")
                sub_tests: list[ast.expr] = []
                for st in assert_tree.body:
                    if isinstance(st, ast.Assert):
                        if isinstance(st.test, ast.BoolOp) and isinstance(st.test.op, ast.And):
                            sub_tests.extend(st.test.values)
                        else:
                            sub_tests.append(st.test)
                if len(sub_tests) >= 3:
                    passed_sub = 0
                    for sub_expr in sub_tests:
                        steps[0] = 0
                        sub_mod = ast.Expression(body=sub_expr)
                        ast.fix_missing_locations(sub_mod)
                        try:
                            if bool(eval(compile(sub_mod, "<ornith_subtest>", "eval"), env, env)):
                                passed_sub += 1
                        except Exception:
                            pass
                    if 0 < passed_sub < len(sub_tests):
                        return round((base_partial + 0.35 * (passed_sub / float(len(sub_tests)))) * thought_consistency, 4)
            except Exception:
                pass
            return base_partial * thought_consistency
        except Exception:
            return 0.0

    # Case 2: Algebraic / invariant assertion expression (`expected == 30`, `dp[3] == 19`, etc.)
    nums = [int(m) for m in re.findall(r"-?\d+", a_body)]
    if nums and len(nums) <= 3:
        first_num = nums[0]
        if harness_clean.lstrip("-").isdigit():
            if first_num == int(harness_clean):
                return 1.0 * thought_consistency
            return (0.15 if any(op in t_body for op in ("*", "+", "-")) else 0.0) * thought_consistency

        env_expr: dict[str, Any] = {
            "__builtins__": dict(_SAFE_BUILTINS),
            "expected": first_num,
            "delta": first_num,
            "y": first_num,
            "mid": first_num,
            "max_rows": first_num,
            "acquisitions": first_num,
            "cycles": nums[1] if len(nums) > 1 else 0,
            "dp": defaultdict_int(first_num),
        }
        try:
            tree_expr = ast.parse(harness_clean, mode="eval")
            for node in ast.walk(tree_expr):
                if not isinstance(node, (*_SAFE_AST_NODES, ast.Expression)):
                    return 0.0
            passed = bool(eval(compile(tree_expr, "<ornith_expr>", "eval"), env_expr, env_expr))
            if passed:
                return 1.0 * thought_consistency
        except Exception:
            pass

    if gold_answer and a_body == gold_answer.strip():
        return 1.0 * thought_consistency
    return (0.15 if any(op in t_body for op in ("*", "+", "-", "=")) else 0.0) * thought_consistency


class defaultdict_int(dict):
    """Helper mapping any index lookup `dp[k]` to the candidate scalar value during verification."""

    def __init__(self, val: int) -> None:
        super().__init__()
        self._val = val

    def __getitem__(self, key: Any) -> int:
        return self._val


class OrnithEnvArchitect:
    """Self-scaffolding curriculum, granular skill router, & infinite-horizon environment architect for Ornith-2.5.

    Monitors empirical pass rate $\\hat{p}_k$, deduction alignment $C_k$, and osu!-style Performance Point
    (`pp`) placement weight $w_{\\text{pp}} = 0.95^{n-1}$ across all 48 procedural, compositional,
    benchmark-specialized, infinite-horizon recursive, Sherlockian, and Polymath families (`0..19` primitive,
    `20..23` compositional, `24..33` specialized ARC/HellaSwag/MMLU/Code, `34..35` unbounded recursive loops,
    `36..39` Sherlockian mysteries with scaling clues/red-herrings/suspects, `40..47` Polymath HackTheBox,
    Word Math, Cryptography, Frontier Science, Behavioral Analysis, Hard Coding, and Cross-Skill Mixing).
    Dynamically promotes `difficulty_tier` ($1 \\dots \\infty$), recursion depth $L$, and difficulty
    modifiers (`NM`, `HR`, `HD`, `DT`, `FL`, `INF`) whenever a skill's pass rate saturates or its
    leaderboard placement decays.
    """

    NUM_PRIMITIVE_FAMILIES: int = 20
    NUM_COMPOSITIONAL_FAMILIES: int = 4
    NUM_SPECIALIZED_FAMILIES: int = 10
    NUM_RECURSIVE_FAMILIES: int = 2
    NUM_SHERLOCK_FAMILIES: int = 4
    NUM_POLYMATH_FAMILIES: int = 8
    NUM_AGENTIC_FAMILIES: int = 8
    NUM_LEETCODE_FAMILIES: int = 8
    TOTAL_FAMILIES: int = 64

    def __init__(
        self,
        target_pass_rate: float = 0.25,
        ema_beta: float = 0.70,
        *,
        max_tier: int | None = None,
    ) -> None:
        self.target_pass_rate = float(target_pass_rate)
        self.ema_beta = float(ema_beta)
        self.max_tier = max_tier  # None = unbounded infinite-horizon tiers (1..inf)
        self.family_pass_ema: list[float] = [0.50 for _ in range(self.TOTAL_FAMILIES)]
        self.family_pp_weight_ema: list[float] = [1.00 for _ in range(self.TOTAL_FAMILIES)]
        self.family_tiers: list[int] = [1 for _ in range(self.TOTAL_FAMILIES)]
        self.family_counts: list[int] = [0 for _ in range(self.TOTAL_FAMILIES)]
        self.family_tier_plays: list[int] = [0 for _ in range(self.TOTAL_FAMILIES)]
        self.compositional_boost: float = 0.25

    def difficulty_tier(self, kind: int) -> int:
        return int(self.family_tiers[kind % self.TOTAL_FAMILIES])

    def skill_and_domain(self, kind: int) -> tuple[str, str]:
        return SKILL_TAXONOMY[kind % len(SKILL_TAXONOMY)]

    def recursion_depth(self, kind: int) -> int:
        k = kind % self.TOTAL_FAMILIES
        tier = self.difficulty_tier(k)
        if k >= 34:
            # Unbounded linear hop growth (L = 3, 4, 5, 6, ... -> inf) for recursive, Sherlockian & Polymath chains
            return 3 + (tier - 1)
        if k >= 20:
            return 2 + (tier - 1) // 2
        return 1 + (tier - 1) // 2

    def difficulty_mods(self, kind: int) -> tuple[str, ...]:
        k = kind % self.TOTAL_FAMILIES
        tier = self.difficulty_tier(k)
        if k >= 36:
            if tier <= 1:
                return ("FL", "HD")
            if tier <= 3:
                return ("FL", "HD", "HR")
            return ("INF", "FL", "DT", "HD", "HR")
        if k >= 34 and tier >= 3:
            return ("INF", "HD", "HR") if tier < 6 else ("INF", "FL", "DT", "HD", "HR")
        if tier <= 1:
            return ("INF",) if k >= 34 else ("NM",)
        if tier <= 3:
            return ("HR",)
        if tier <= 5:
            return ("HD", "HR")
        if tier <= 8:
            return ("DT", "HD", "HR")
        return ("INF", "FL", "DT", "HD", "HR")

    def star_difficulty(self, kind: int) -> float:
        k = kind % self.TOTAL_FAMILIES
        return compute_star_difficulty(
            self.difficulty_tier(k),
            recursion_depth=self.recursion_depth(k),
            mods=self.difficulty_mods(k),
        )

    def select_family(
        self,
        idx: int,
        seed: int = 42,
        *,
        pp_leaderboard: OrnithPPLeaderboard | None = None,
    ) -> int:
        """Select task family `0..47`, routing toward Sherlockian mysteries, Polymath challenges, under-trained PP skills, and recursive frontiers."""
        if idx < self.TOTAL_FAMILIES:
            return idx % self.TOTAL_FAMILIES
        # Parity-unlocked stride so alternating external steps cannot lock selection to only even family indices
        base_slot = (idx + (idx // 2)) % self.TOTAL_FAMILIES
        # Dedicate every 3rd online step across the 28 Sherlockian, Polymath, Agentic & LeetCode families (36..63)
        if idx % 3 == 0:
            return 36 + (
                (idx // 3)
                % (
                    self.NUM_SHERLOCK_FAMILIES
                    + self.NUM_POLYMATH_FAMILIES
                    + self.NUM_AGENTIC_FAMILIES
                    + self.NUM_LEETCODE_FAMILIES
                )
            )
        mean_prim_pass = sum(self.family_pass_ema[: self.NUM_PRIMITIVE_FAMILIES]) / float(
            self.NUM_PRIMITIVE_FAMILIES
        )
        # When a PP leaderboard is active and idx >= TOTAL_FAMILIES, prioritize lowest-PP or unvisited skills
        if pp_leaderboard is not None and (idx % 4 in (2, 3)):
            low_skills = pp_leaderboard.lowest_pp_skill_indices(self.TOTAL_FAMILIES)
            if low_skills:
                return int(low_skills[(idx // 2 + seed) % min(12, len(low_skills))])
        # Route saturated primitive slots toward compositional, specialized, recursive, Sherlockian, or Polymath families (20..47)
        if mean_prim_pass > 0.70 and (idx % 3 == 2) and base_slot < self.NUM_PRIMITIVE_FAMILIES:
            advanced_span = self.TOTAL_FAMILIES - self.NUM_PRIMITIVE_FAMILIES
            return self.NUM_PRIMITIVE_FAMILIES + ((idx + seed) % advanced_span)
        return base_slot

    def record_outcome(
        self,
        kind: int,
        pass_prob: float,
        alignment: float = 1.0,
        *,
        pp_weight: float = 1.0,
    ) -> None:
        """Update EMA pass rate & PP placement weight for `kind` and promote `difficulty_tier` ($1 \\to \\infty$)."""
        k = kind % self.TOTAL_FAMILIES
        self.family_counts[k] += 1
        self.family_tier_plays[k] += 1
        ema = self.ema_beta * self.family_pass_ema[k] + (1.0 - self.ema_beta) * float(pass_prob)
        self.family_pass_ema[k] = ema
        w_ema = 0.75 * self.family_pp_weight_ema[k] + 0.25 * float(pp_weight)
        self.family_pp_weight_ema[k] = w_ema

        can_promote = self.max_tier is None or self.family_tiers[k] < self.max_tier
        # Promote tier when sustained pass rate at current tier (>=3 plays) exceeds 0.76 OR when PP placement weight decays (<0.36) while pass rate is strong (>0.68)
        if can_promote and self.family_tier_plays[k] >= 3 and (
            ema > 0.76 or (w_ema < 0.36 and ema > 0.68)
        ):
            self.family_tiers[k] += 1
            self.family_tier_plays[k] = 0
            self.family_pass_ema[k] = 0.42
            self.family_pp_weight_ema[k] = 0.90
        elif ema < 0.22 and self.family_tiers[k] > 1 and self.family_tier_plays[k] >= 3:
            self.family_tiers[k] -= 1
            self.family_tier_plays[k] = 0
            self.family_pass_ema[k] = 0.45


_SYNTHETIC_THOUGHT_PREFIXES: tuple[str, ...] = (
    "Analyze ",
    "Execute procedural",
    "Stage 1",
    "Isolate the target",
    "Apply algebraic",
    "Evaluate ",
    "Satisfy the lexical",
    "Solve the arithmetic",
    "Implement a clean",
    "Implement ",
)


def _is_genuine_natural_thought(thought: str) -> bool:
    """Return True when `thought` is a genuine multi-sentence reasoning trace, not a 1-line RL stub."""
    t_str = thought.strip()
    return len(t_str) >= 90 and not t_str.startswith(_SYNTHETIC_THOUGHT_PREFIXES)


def _is_genuine_complete_answer(answer: str) -> bool:
    """Return True when `answer` is a complete natural sentence or Python function, not a bare scalar ('9')."""
    a_str = answer.strip()
    return len(a_str) >= 28 and ("def " in a_str or len(a_str.split()) >= 5)


def _is_genuine_natural_cot(thought: str, answer: str) -> bool:
    """Return True only when `(thought, answer)` has both a genuine multi-sentence CoT and complete answer."""
    return _is_genuine_natural_thought(thought) and _is_genuine_complete_answer(answer)


class OrnithSelfImprover:
    """Executes the closed-loop Ornith-1.0/2.0 Self-Scaffolding, Self-Improvement, and Dream-RSI cycle.

    Combines:
    - Stage 0 (`Dream-RSI` Evolving Worlds): Expands an online discovery tree $\\mathcal{T}_t$
      using active exploration policy $\\pi_t$, appends $\\mathcal{T}_t$ to the replay
      simulator pool $\\mathcal{H}_t$, and dreams $M$ candidate policy revisions offline at
      zero execution cost ($\\mathcal{H}_t \\xrightarrow{\\text{replay}} \\pi_{t+1}$).
    - Stage 1 (`#Q$#` Proposer & `OrnithEnvArchitect`): Computes `<0.1ms` `Association` Novelty $N(q)$
      against `NoveltyTaskBuffer`, capability-frontier difficulty $D(p; p^*=0.20)$ for $R_{\\text{task}}$,
      and dynamically composes cross-template RL environments as primitive templates saturate.
    - Stage 2 (`#H$# / #S$#` Architect): Scores Task Alignment $C(q, h)$ via `Deduction` and
      Reward Fidelity $F(h, \\{\\tau_i\\})$ via `Evaluation` (`compute_harness_reward`), while
      updating `ScaffoldCache` monotonically.
    - Stage 3 (`#T$# -> #A$#` Solver): Updates policy weights via staleness-weighted
      `grpo_clipped_loss` over group-normalized rollout advantages + `0x8000` masked
      self-distillation + `MorphCode-16K` first-divergence `Antidoom FTPO`.
    """

    def __init__(
        self,
        projection_dim: int = 32,
        *,
        max_seq_len: int = 384,
        device: str | torch.device = "cpu",
        seed: int = 42,
        enable_dream_rsi: bool = False,
        dream_every: int = 1,
        dream_engine: Any | None = None,
        external_pool_path: str | Any | None = None,
        external_tasks: list[OrnithTask] | None = None,
    ) -> None:
        from pathlib import Path
        from .data import ByteTokenizer
        from .dream_rsi import DreamRSIEngine

        self.tokenizer = ByteTokenizer()
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.novelty_buffer = NoveltyTaskBuffer(
            projection_dim=projection_dim,
            max_capacity=50_000,
            device=self.device,
        )
        self.scaffold_cache = ScaffoldCache()
        self.env_architect = OrnithEnvArchitect()
        self.pp_leaderboard = OrnithPPLeaderboard()
        self._step_counter = 0
        self._rng_seed = seed
        self.enable_dream_rsi = enable_dream_rsi or (dream_engine is not None)
        self.dream_every = max(1, int(dream_every))
        self.dream_engine: DreamRSIEngine | None = dream_engine or (
            DreamRSIEngine(seed=seed) if self.enable_dream_rsi else None
        )
        self._last_dream_out: Any | None = None
        self.external_tasks: list[OrnithTask] = list(external_tasks) if external_tasks else []
        self._last_task_kinds: dict[str, int] = {}
        if external_pool_path is not None and not self.external_tasks:
            p = Path(external_pool_path)
            if p.is_file():
                try:
                    loaded = torch.load(p, map_location="cpu", weights_only=False)
                    if isinstance(loaded, list):
                        self.external_tasks = [t for t in loaded if isinstance(t, OrnithTask)]
                    elif isinstance(loaded, dict) and "tasks" in loaded:
                        self.external_tasks = [
                            t if isinstance(t, OrnithTask) else OrnithTask(**t)
                            for t in loaded["tasks"]
                        ]
                except Exception:
                    self.external_tasks = []

    def restore_from_metrics_jsonl(self, metrics_path: Any, *, max_step: int | None = None) -> int:
        """Replay historical `ornith_active_skill` and `ornith_raw_pp` plays from `metrics.jsonl` on resume."""
        import json
        from pathlib import Path

        m_path = Path(metrics_path)
        if not m_path.is_file():
            return 0
        skill_to_kind = {skill: idx for idx, (skill, _dom) in enumerate(SKILL_TAXONOMY)}
        replayed = 0
        try:
            for line in m_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                step_val = int(rec.get("step", 0))
                if max_step is not None and step_val > max_step:
                    continue
                if "ornith_raw_pp" not in rec or "ornith_active_skill" not in rec:
                    continue
                skill = str(rec.get("ornith_active_skill", ""))
                raw_pp = float(rec.get("ornith_raw_pp", 0.0))
                pass_rate = float(rec.get("ornith_pass_rate", 0.75))
                if not skill or raw_pp <= 0.0:
                    continue
                sub = self.pp_leaderboard.submit_play(skill, raw_pp)
                replayed += 1
                self._step_counter = max(self._step_counter, replayed)
                if skill in skill_to_kind:
                    self.env_architect.record_outcome(
                        skill_to_kind[skill],
                        pass_rate,
                        1.0,
                        pp_weight=sub.skill_weight,
                    )
        except Exception:
            pass
        return replayed

    def _synthesize_infinite_recursive_task(
        self,
        idx: int,
        *,
        tier: int = 1,
        mode: str = "chain",
    ) -> OrnithTask:
        """Synthesize an unbounded $L$-hop recursive reasoning loop where $L = 3 + (\\text{tier} - 1) \\to \\infty$.

        Supports two complementary infinite-horizon modes:
        - `mode="chain"` (`kind=34`, `recursion.unbounded_multihop_chain`): Unrolls $L$ sequential state
          transformations $s_0 \\xrightarrow{T_1} s_1 \\dots \\xrightarrow{T_L} s_L$ with optional `HD`
          intermediate state masking and hop-truncation / early-corruption distractors.
        - `mode="code_recursion"` (`kind=35`, `recursion.self_referential_fixed_point`): Synthesizes an
          executable Python state recurrence `def solve(x, steps):` verified at $L$ recursive steps.
        """
        t_val = max(1, int(tier))
        hops = min(64, 3 + (t_val - 1))
        kind_slot = 34 if mode == "chain" else 35
        mods = self.env_architect.difficulty_mods(kind_slot)
        stars = compute_star_difficulty(t_val, recursion_depth=hops, mods=mods)
        skill_name, domain_name = self.env_architect.skill_and_domain(kind_slot)

        s0 = 4 + ((idx * 7 + self._rng_seed) % (11 + 3 * t_val))
        b_step = 3 + ((idx * 5 + t_val) % (7 + 2 * t_val))
        c_step = 2 + ((idx * 3 + self._rng_seed) % (5 + t_val))

        if mode == "code_recursion":
            p_mod = 97 + 2 * (idx % 25)
            mult_m = 2 + (idx % 4)
            # Compute exact trajectory x_0 -> x_1 -> ... -> x_hops
            x_cur = s0
            x_one = (s0 * mult_m + b_step) % p_mod
            trace_notes: list[str] = []
            for h_i in range(1, hops + 1):
                x_next = (x_cur * mult_m + b_step) % p_mod
                if h_i <= 3 or h_i == hops:
                    trace_notes.append(f"hop {h_i} maps {x_cur} to ({x_cur} * {mult_m} + {b_step}) mod {p_mod} = {x_next}")
                x_cur = x_next
            final_x = x_cur
            task_id = f"inf_rec_code_t{t_val}_L{hops}_{s0}_{mult_m}_{b_step}_{p_mod}"
            prompt_q = (
                f"#U$# [Infinite-Horizon Recursive Dynamical Map | ★{stars:.2f} | Depth L={hops} | Mods={'+'.join(mods)}] "
                f"Define the discrete state transition `f(x) = (x * {mult_m} + {b_step}) % {p_mod}`. "
                f"Implement `solve(x, steps)` in Python that applies `f` recursively/iteratively `steps` times starting from `x`.\n"
                f"#Q$# [infinite_recursion] Trace the recurrence in `#T$#` and implement `solve(x, steps)` passing `steps={hops}`."
            )
            harness_h = (
                f"assert solve({s0}, {hops}) == {final_x} "
                f"and solve({s0}, 1) == {x_one} "
                f"and solve({s0}, 0) == {s0}"
            )
            gold_t = (
                f"To evaluate the {hops}-hop dynamical recurrence starting at x0 = {s0}, we iterate "
                f"x = (x * {mult_m} + {b_step}) mod {p_mod} across `steps` transitions: "
                + "; ".join(trace_notes)
                + f", reaching the terminal fixed-horizon state {final_x}."
            )
            gold_a = (
                f"def solve(x, steps):\n"
                f"    for _ in range(steps):\n"
                f"        x = (x * {mult_m} + {b_step}) % {p_mod}\n"
                f"    return x"
            )
            d1_t = (
                f"To evaluate the {hops}-hop recurrence, we apply only steps - 1 transitions "
                f"and stop one hop early before the final modular reduction."
            )
            d1_a = (
                f"def solve(x, steps):\n"
                f"    for _ in range(max(0, steps - 1)):\n"
                f"        x = (x * {mult_m} + {b_step}) % {p_mod}\n"
                f"    return x"
            )
            d2_t = (
                f"To evaluate the {hops}-hop recurrence, we subtract {b_step} instead of adding "
                f"inside the modular transition loop."
            )
            d2_a = (
                f"def solve(x, steps):\n"
                f"    for _ in range(steps):\n"
                f"        x = (x * {mult_m} - {b_step}) % {p_mod}\n"
                f"    return x"
            )
            hack_a = "def solve(x, steps):\n    return 0"
        else:
            # Unbounded L-hop deterministic state chain
            cur = s0
            err_early = s0
            err_hop1 = s0 + c_step
            hop_prompts: list[str] = []
            hop_thoughts: list[str] = []
            for h_i in range(1, hops + 1):
                op_kind = (idx + h_i) % 3
                delta_h = b_step + h_i if op_kind == 0 else (2 * c_step if op_kind == 1 else h_i + c_step)
                nxt = cur + delta_h
                err_hop1 = err_hop1 + delta_h
                if h_i == hops - 1:
                    err_early = nxt
                if "HD" in mods and 1 < h_i < hops:
                    hop_prompts.append(f"Hop {h_i}: add `{delta_h}` to the hidden state `s_{h_i-1}` to obtain `s_{h_i}`.")
                else:
                    hop_prompts.append(f"Hop {h_i}: compute `s_{h_i} = s_{h_i-1} + {delta_h}`.")
                hop_thoughts.append(f"at hop {h_i} we compute {cur} + {delta_h} = {nxt}")
                cur = nxt
            final_s = cur
            if err_early == final_s:
                err_early = final_s - b_step
            if err_hop1 == final_s:
                err_hop1 = final_s + c_step
            task_id = f"inf_rec_chain_t{t_val}_L{hops}_{s0}_{b_step}_{c_step}_{idx}"
            prompt_q = (
                f"#U$# [Infinite-Horizon Recursive Chain | ★{stars:.2f} | Depth L={hops} | Mods={'+'.join(mods)}] "
                f"Initial state `s_0 = {s0}`. Execute all {hops} recursive state hops in order: "
                + " ".join(hop_prompts)
                + "\n#Q$# [infinite_recursion] Unroll every hop in `#T$#` and state the terminal value `s_"
                + f"{hops}` in lowercase with no commas, at least 6 words, including the keyword `recursive`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:recursive;exact_int:{final_s}"
            gold_t = (
                f"Starting from initial state s_0 = {s0}, we unroll the {hops}-stage recursive chain sequentially: "
                + "; ".join(hop_thoughts)
                + f". Therefore the verified terminal state after {hops} recursive hops is {final_s}."
            )
            gold_a = f"the verified recursive chain terminal state is {final_s} units"
            d1_t = (
                f"Starting from initial state s_0 = {s0}, we unroll only {hops - 1} stages and halt prematurely at {err_early}."
            )
            d1_a = f"the verified recursive chain terminal state is {err_early} units"
            d2_t = (
                f"Starting from shifted state {s0 + c_step}, we propagate the initial offset error to obtain {err_hop1}."
            )
            d2_a = f"The Recursive, Chain Terminal State Is {err_hop1} units"
            hack_a = "0"

        base_cands = (
            f"#T$# {gold_t}\n#A$# {gold_a}",
            f"#T$# {d1_t}\n#A$# {d1_a}",
            f"#T$# {d2_t}\n#A$# {d2_a}",
            f"#T$# Default fallback.\n#A$# {hack_a}",
        )
        base_scores = (1.0, 0.20, 0.10, 0.0)
        order = [0, 1, 2, 3]
        shift = idx % 4
        order[0], order[shift] = order[shift], order[0]
        p_cands = tuple(base_cands[i] for i in order)
        p_scores = tuple(base_scores[i] for i in order)
        g_idx = order.index(0)
        self._last_task_kinds[task_id] = kind_slot
        cached_s = self.scaffold_cache.get(
            task_id,
            default=_SCAFFOLD_CANDIDATES[idx % len(_SCAFFOLD_CANDIDATES)],
        )
        return OrnithTask(
            task_id=task_id,
            domain=domain_name,
            prompt_q=prompt_q,
            harness_h=harness_h,
            scaffold_s=cached_s,
            candidate_rollouts=p_cands,
            harness_scores=p_scores,
            gold_index=g_idx,
            gold_thought=gold_t,
            gold_answer=gold_a,
            skill=skill_name,
            star_difficulty=stars,
            recursion_depth=hops,
            difficulty_mods=mods,
        )

    def _synthesize_sherlock_reasoning_task(
        self,
        idx: int,
        *,
        tier: int = 1,
        mode: int = 0,
    ) -> OrnithTask:
        """Synthesize a multi-stage Sherlockian Mystery (`36..39`) where suspects, chained clues, and
        planted red herrings scale unboundedly with `tier` ($1 \\dots \\infty$).

        Modes (`kind = 36 .. 39`):
        - `mode = 0` (`36`, `deduction.sherlock_eliminative_grid`):
          Multi-stage suspect, clue, and red-herring discovery across rotating crime scenes (`num_suspects = 4 + (tier - 1) // 2`,
          `num_clues = 3 + (tier - 1) // 2`, `num_red_herrings = 1 + tier // 2`).
        - `mode = 1` (`37`, `abduction.sherlock_diagnostic_trace`):
          Backward causal reconstruction from multi-stage physical traces while filtering out `1 + tier // 2`
          spoofed/uncalibrated red-herring sensor logs and Inspector Lestrade's surface trap.
        - `mode = 2` (`38`, `deduction.sherlock_alibi_contradiction`):
          Multi-segment transit graph cross-examination with `1 + tier // 2` red-herring witness sightings.
        - `mode = 3` (`39`, `induction.sherlock_cipher_induction`):
          Multi-layer cipher induction ("The Dancing Men") with corrupted red-herring decoy intercepts and
          tier-scaled secondary key-schedule offsets.
        """
        t_val = max(1, int(tier))
        m_id = int(mode) % 4
        kind_slot = 36 + m_id
        num_suspects = 4 + (t_val - 1) // 2
        num_clues = 3 + (t_val - 1) // 2
        num_red_herrings = 1 + t_val // 2
        hops = num_clues + num_red_herrings - 1
        mods = self.env_architect.difficulty_mods(kind_slot)
        stars = compute_star_difficulty(t_val, recursion_depth=hops, mods=mods)
        skill_name, domain_name = self.env_architect.skill_and_domain(kind_slot)

        a = 4 + ((idx * 7 + self._rng_seed) % (13 + 3 * t_val))
        b = 3 + ((idx * 11 + self._rng_seed) % (9 + 2 * t_val))
        c = 2 + ((idx * 5 + t_val) % (7 + t_val))
        scen_id = (idx + t_val) % 4

        if m_id == 0:
            # Mode 0: Multi-Stage Eliminative Mystery with Scaling Suspects, Clues & Red Herrings
            scenes = (
                ("Blackwood Vault", "cipher ledger", "mastiff", "Conservatory"),
                ("Whitehall Archive", "diplomatic seal", "night-watchman", "Telegraph Room"),
                ("Royal Observatory", "zenith chronometer", "pressure-mat", "Meridian Hall"),
                ("Pullman Express", "bullion dispatch", "conductor-bell", "Baggage Car"),
            )
            loc_name, item_name, hound_name, alibi_room = scenes[scen_id]
            clearance_base = a * b
            k_mult = 3 + (t_val % 3)
            primary_offset = k_mult * c
            stage2_badge = clearance_base + primary_offset
            secondary_seal = (t_val - 1) * 2
            insider_badge = stage2_badge + secondary_seal
            outsider_badge = clearance_base - c
            decoy_badge = clearance_base + c
            rh_badge = clearance_base + (k_mult + 2) * c + 1

            extra_suspects = [
                f"Badge-{clearance_base + (s_i + 2) * c + 1} (aux-guard-{s_i + 1})"
                for s_i in range(max(0, num_suspects - 4))
            ]
            suspect_roster = ", ".join(
                [
                    f"Badge-{outsider_badge} (external courier)",
                    f"Badge-{clearance_base} (archivist in {alibi_room})",
                    f"Badge-{decoy_badge} (steward with single offset)",
                    f"Badge-{rh_badge} (planted decoy)",
                    *extra_suspects,
                    f"Badge-{insider_badge} (senior keyholder)",
                ]
            )
            rh_notes = "; ".join(
                f"Red-Herring R{r_i + 1}: a planted mud-stained stub near Gate-{r_i + 1} cites Badge-{rh_badge + r_i} (falsified by unbroken wax seal)"
                for r_i in range(num_red_herrings)
            )
            task_id = f"sherlock_elim_t{t_val}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [Sherlockian Mystery ({loc_name}) | ★{stars:.2f} | Suspects={num_suspects + 1} | Clues={num_clues} | RedHerrings={num_red_herrings}] "
                f"The {item_name} was taken from the {loc_name}. Suspect pool: {suspect_roster}. "
                f"Stage 1 (Filter Red Herrings): [{rh_notes}]. "
                f"Stage 2 (Verified Clues): Clue C1 (Negative Evidence): The {hound_name} remained silent, eliminating any external courier with badge < `{a} * {b}`. "
                f"Clue C2 (Alibi & Single-Offset Elimination): The {alibi_room} log eliminates the exact baseline badge `{a} * {b}` and single-offset `{a} * {b} + {c}` (plus any aux-guard without lock parity). "
                f"Clue C3 (Multi-Stage Vault Seal): The inner seal requires primary offset `{k_mult} * {c}` added to baseline `{a} * {b}`, plus Tier-{t_val} secondary seal `{secondary_seal}`.\n"
                f"#Q$# [deduction] Discard the red herrings, eliminate impossible suspects step-by-step in `#T$#`, and state the remaining culprit's badge number in lowercase with no commas, at least 6 words, including the keyword `eliminated`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:eliminated;exact_int:{insider_badge}"
            gold_t = (
                f"Stage 1 filters out the {num_red_herrings} planted red-herring stubs starting at {rh_badge} because the wax seal was unbroken. "
                f"Stage 2 computes baseline clearance {a} * {b} = {clearance_base} (eliminating courier {outsider_badge} via the silent {hound_name}) and primary lock offset {k_mult} * {c} = {primary_offset}. "
                f"Adding {clearance_base} + {primary_offset} = {stage2_badge} and secondary tier seal {stage2_badge} + {secondary_seal} = {insider_badge} leaves culprit badge {insider_badge} once all impossible suspects are eliminated."
            )
            gold_a = f"after impossible suspects and red herrings are eliminated the culprit badge is {insider_badge}"
            d1_t = (
                f"We fall for planted Red-Herring R1 and select decoy badge {rh_badge} without checking the unbroken wax seal."
            )
            d1_a = f"after impossible suspects and red herrings are eliminated the culprit badge is {rh_badge}"
            d2_t = (
                f"We compute {a} * {b} = {clearance_base} and add single offset {c} to get {decoy_badge}, omitting the secondary seal."
            )
            d2_a = f"After Impossible, Suspects Are Eliminated The Badge Is {decoy_badge}"
            hack_a = "0"
            default_scaffold = "abduction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 1:
            # Mode 1: Multi-Stage Diagnostic Abduction with Sensor Red Herrings
            base_temp = 20 + (a % 15)
            cooling_drop = b * c
            stage1_reading = base_temp + cooling_drop
            cascade_flux = (t_val - 1) * 3
            observed_reading = stage1_reading + cascade_flux
            lestrade_trap = base_temp + b + c
            rh_sensor = observed_reading + 19 + t_val
            task_id = f"sherlock_abd_t{t_val}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [Sherlockian Diagnostic Abduction | ★{stars:.2f} | Traces={num_clues} | RedHerrings={num_red_herrings}] "
                f"Forensic inspection of the Cryogenic Chamber reveals: "
                f"(O1) Ambient baseline is `{base_temp}` units; "
                f"(O2) The primary vessel ran `{b}` cycles at `{c}` flux units per cycle; "
                f"(O3) Tier-{t_val} secondary manifold contributed `{cascade_flux}` units; "
                f"(Red Herrings R1..R{num_red_herrings}): Uncalibrated auxiliary gauge logs `{rh_sensor}` units and Inspector Lestrade's naive additive guess logs `{base_temp} + {b} + {c}` units, both falsified because the relief valve remained unbroken.\n"
                f"#Q$# [abduction] Filter out the red-herring gauges, reason backward from (O1, O2, O3) in `#T$#`, and report the exact causal reading in lowercase with no commas, at least 6 words, including the keyword `abductive`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:abductive;exact_int:{observed_reading}"
            gold_t = (
                f"First we reject the {num_red_herrings} red-herring auxiliary readings ({rh_sensor}) and Lestrade's additive trap ({lestrade_trap}). "
                f"Multiplying primary cycles {b} * {c} = {cooling_drop}, adding baseline {base_temp} + {cooling_drop} = {stage1_reading}, and adding manifold flux {stage1_reading} + {cascade_flux} = {observed_reading} yields the unique abductive reading."
            )
            gold_a = f"the unique abductive forensic vessel reading is {observed_reading} units"
            d1_t = (
                f"Following Inspector Lestrade's surface hypothesis, we add {base_temp} + {b} = {base_temp + b} and + {c} = {lestrade_trap}."
            )
            d1_a = f"the unique abductive forensic vessel reading is {lestrade_trap} units"
            d2_t = (
                f"Trusting the uncalibrated red-herring auxiliary gauge, we report {rh_sensor} units."
            )
            d2_a = f"The Unique, Abductive Forensic Reading Is {rh_sensor} units"
            hack_a = "0"
            default_scaffold = "abduction -> association -> deduction -> decision -> metacognition"

        elif m_id == 2:
            # Mode 2: Multi-Leg Forensic Timeline & Alibi Cross-Examination with Witness Red Herrings
            hop_extra = 4 + (t_val - 1)
            dist_ab = (a + hop_extra) * c
            claimed_minutes = a * c
            min_required_minutes = dist_ab
            contradiction_discrepancy = min_required_minutes - claimed_minutes  # == hop_extra * c
            rh_clock_delay = b * 3 + 7
            wrong_discrepancy_1 = 2 * c
            wrong_discrepancy_2 = dist_ab + claimed_minutes
            task_id = f"sherlock_alibi_t{t_val}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [Sherlockian Alibi Cross-Examination | ★{stars:.2f} | Checkpoints={num_clues} | RedHerrings={num_red_herrings}] "
                f"Transit across the `{num_clues}` locked turnstiles from North Station to East Wing spans `{a + hop_extra}` segments at `{c}` minutes per segment. "
                f"Red-Herring Sightings (R1..R{num_red_herrings}): A fog-delayed omnibus log cites `{rh_clock_delay}` minutes on the South Bridge, which bypasses the turnstiles entirely. "
                f"Suspect B's signed turnstile ledger claims transit took only `{a}` segments at `{c}` minutes per segment.\n"
                f"#Q$# [deduction] Ignore the South Bridge red herrings, cross-examine Suspect B's turnstile timeline in `#T$#`, and compute the exact travel-time contradiction deficit (`minimum_required - claimed_minutes`) in lowercase with no commas, at least 6 words, including the keyword `contradiction`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:contradiction;exact_int:{contradiction_discrepancy}"
            gold_t = (
                f"Discarding the {num_red_herrings} South Bridge red-herring sightings ({rh_clock_delay} min), the locked turnstile route requires ({a + hop_extra}) * {c} = {min_required_minutes} minutes whereas Suspect B claims {a} * {c} = {claimed_minutes} minutes. "
                f"Subtracting {min_required_minutes} - {claimed_minutes} = {contradiction_discrepancy} proves a forensic contradiction of {contradiction_discrepancy} minutes."
            )
            gold_a = f"the forensic alibi contradiction deficit is {contradiction_discrepancy} minutes"
            d1_t = (
                f"We confuse the South Bridge red-herring omnibus delay {rh_clock_delay} with the turnstile deficit."
            )
            d1_a = f"the forensic alibi contradiction deficit is {rh_clock_delay} minutes"
            d2_t = (
                f"We add the required time and claimed time {min_required_minutes} + {claimed_minutes} = {wrong_discrepancy_2} minutes."
            )
            d2_a = f"The Forensic, Alibi Contradiction Deficit Is {wrong_discrepancy_2} minutes"
            hack_a = "0"
            default_scaffold = "deduction -> evaluation -> decision -> metacognition"

        else:
            # Mode 3: Multi-Layer Cipher Induction ("The Dancing Men") with Decoy Intercepts
            slope_k = (c % 5) + 2
            tier_shift = (t_val - 1) * 2
            offset_m = a + b + tier_shift
            x1, x2, x3, x_target = 2, 5, 8, (t_val + 9)
            y1 = slope_k * x1 + offset_m
            y2 = slope_k * x2 + offset_m
            y3 = slope_k * x3 + offset_m
            y_target = slope_k * x_target + offset_m
            rh_x, rh_y = 4, slope_k * 4 + offset_m + 13
            y_err1 = slope_k * x_target - offset_m
            y_err2 = (slope_k + 1) * x_target + offset_m
            task_id = f"sherlock_cipher_t{t_val}_{slope_k}_{offset_m}_{x_target}_{idx}"
            prompt_q = (
                f"#U$# [Sherlockian Cipher Induction ('Dancing Men') | ★{stars:.2f} | Hops={hops} | RedHerrings={num_red_herrings}] "
                f"Three verified checksum-valid cipher intercepts map as: "
                f"`symbol({x1}) -> {y1}`, `symbol({x2}) -> {y2}`, and `symbol({x3}) -> {y3}`. "
                f"Red-Herring Decoy Intercepts (R1..R{num_red_herrings}): `symbol({rh_x}) -> {rh_y}` failed checksum verification (planted noise) and must be discarded. "
                f"Induce the underlying affine cipher law `solve(x)` from the verified intercepts and evaluate `solve({x_target})`.\n"
                f"#Q$# [code] Filter out the corrupted decoy intercept, derive the cipher law in `#T$#`, and implement `def solve(x):` in `#A$#`."
            )
            harness_h = f"assert solve({x_target}) == {y_target} and solve({x1}) == {y1} and solve({x2}) == {y2}"
            gold_t = (
                f"First we discard the {num_red_herrings} checksum-invalid red-herring intercept symbol({rh_x}) -> {rh_y}. "
                f"From verified intercepts symbol({x2}) and symbol({x1}), we compute {y2} - {y1} = {y2 - y1} over step {x2} - {x1} = 3 to induce multiplier k = {slope_k}. "
                f"Then {y1} - {slope_k * x1} = {offset_m} gives the exact offset, verifying {slope_k * x3} + {offset_m} = {y3} and predicting {slope_k * x_target} + {offset_m} = {y_target}."
            )
            gold_a = f"def solve(x):\n    return {slope_k} * x + {offset_m}"
            d1_t = (
                f"We induce multiplier {slope_k} but subtract the constant offset {offset_m} to get {y_err1}."
            )
            d1_a = f"def solve(x):\n    return {slope_k} * x - {offset_m}"
            d2_t = (
                f"We include the corrupted red-herring intercept and miscalculate the slope as {slope_k + 1} to get {y_err2}."
            )
            d2_a = f"def solve(x):\n    return {slope_k + 1} * x + {offset_m}"
            hack_a = "def solve(x):\n    return 0"
            default_scaffold = "induction -> abstraction -> deduction -> decision -> metacognition"

        base_cands = (
            f"#T$# {gold_t}\n#A$# {gold_a}",
            f"#T$# {d1_t}\n#A$# {d1_a}",
            f"#T$# {d2_t}\n#A$# {d2_a}",
            f"#T$# Default fallback.\n#A$# {hack_a}",
        )
        base_scores = (1.0, 0.20, 0.10, 0.0)
        order = [0, 1, 2, 3]
        shift = idx % 4
        order[0], order[shift] = order[shift], order[0]
        p_cands = tuple(base_cands[i] for i in order)
        p_scores = tuple(base_scores[i] for i in order)
        g_idx = order.index(0)
        self._last_task_kinds[task_id] = kind_slot
        cached_s = self.scaffold_cache.get(task_id, default=default_scaffold)
        return OrnithTask(
            task_id=task_id,
            domain=domain_name,
            prompt_q=prompt_q,
            harness_h=harness_h,
            scaffold_s=cached_s,
            candidate_rollouts=p_cands,
            harness_scores=p_scores,
            gold_index=g_idx,
            gold_thought=gold_t,
            gold_answer=gold_a,
            skill=skill_name,
            star_difficulty=stars,
            recursion_depth=hops,
            difficulty_mods=mods,
        )

    def _synthesize_polymath_challenge_task(
        self,
        idx: int,
        *,
        tier: int = 1,
        mode: int = 0,
    ) -> OrnithTask:
        """Synthesize an advanced Sherlockian-inspired Polymath Challenge (`kind = 40 .. 47`):

        - `mode = 0` (`40`, `security.hackthebox_sandbox_escape`):
          HackTheBox environment enumeration, tool discovery, honeypot filtering & multi-stage sandbox escape.
        - `mode = 1` (`41`, `math.sherlock_word_problem_system`):
          Complex multi-stage word math with coupled conservation rates and narrative red-herring quantities.
        - `mode = 2` (`42`, `crypto.multilayer_trapdoor_stream`):
          Multi-layer trapdoor stream / Feistel cryptanalysis with differential fault red herrings.
        - `mode = 3` (`43`, `science.frontier_open_problem_reduction`):
          Challenging & open/unanswered math/science frontier finite-scale reductions (Collatz $3n+1$ certificates,
          Erdős–Straus unit fractions, discrete Navier–Stokes enstrophy, Yang–Mills lattice plaquette invariants).
        - `mode = 4` (`44`, `behavior.game_theoretic_deception_profiling`):
          Behavioral analysis, deception detection & costly-signaling bluff profiling across $N$ suspects.
        - `mode = 5` (`45`, `code.algorithmic_synthesis_hard`):
          Extremely challenging multi-constraint algorithmic coding with state filtering and invariants.
        - `mode = 6` (`46`, `mixing.cross_skill_sherlock_crypto_code`):
          Cross-Skill Mixing I: Sherlockian Clue Elimination + Cryptographic Key Recovery + Python Synthesis.
        - `mode = 7` (`47`, `mixing.cross_skill_htb_science_behavior`):
          Cross-Skill Mixing II: Behavioral Bluff Profiling + Thermodynamic Sensor Verification + HTB Escape.
        """
        t_val = max(1, int(tier))
        m_id = int(mode) % 8
        kind_slot = 40 + m_id
        num_red_herrings = 1 + t_val // 2
        num_stages = 3 + (t_val - 1) // 2
        hops = num_stages + num_red_herrings - 1
        mods = self.env_architect.difficulty_mods(kind_slot)
        stars = compute_star_difficulty(t_val, recursion_depth=hops, mods=mods)
        skill_name, domain_name = self.env_architect.skill_and_domain(kind_slot)

        a = 4 + ((idx * 7 + self._rng_seed) % (13 + 3 * t_val))
        b = 3 + ((idx * 11 + self._rng_seed) % (9 + 2 * t_val))
        c = 2 + ((idx * 5 + t_val) % (7 + t_val))

        if m_id == 0:
            # 40: HackTheBox Sandbox Escape (Tool Discovery & Environment Enumeration with Honeypot Traps)
            sub_scen = (idx + t_val) % 2
            base_cap = a * b
            k_pivot = 2 + (t_val % 4)
            pivot_off = c * k_pivot
            stage2_tok = base_cap + pivot_off
            ns_bridge = (t_val - 1) * 3
            escape_token = stage2_tok + ns_bridge
            honeypot_tok = base_cap + c + 17
            if honeypot_tok == escape_token:
                honeypot_tok += 13
            task_id = f"htb_escape_t{t_val}_{a}_{b}_{c}_{idx}"
            if sub_scen == 0:
                prompt_q = (
                    f"#U$# [HackTheBox Sandbox Escape: Tool Chain | ★{stars:.2f} | Stages={num_stages} | Honeypots={num_red_herrings}] "
                    f"You are inside restricted container `uid=1000` with `seccomp` blocking direct `execve`. "
                    f"Enumerating `/opt/tools` reveals `{num_red_herrings}` honeypot SUID binaries (`/opt/tools/suid_diag` -> token `{honeypot_tok}`, linked to `libaudit_tripwire.so`) "
                    f"and one unmonitored capability binary `/opt/tools/cap_helper` (`CAP_DAC_READ_SEARCH`). "
                    f"Stage 1: Compute base descriptor offset `{a} * {b}` via `cap_helper`. "
                    f"Stage 2: Chain cgroup `release_agent` pivot offset `{c} * {k_pivot}` onto Stage 1. "
                    f"Stage 3: Apply Tier-{t_val} host PID-namespace bridge `+{ns_bridge}` onto Stage 2.\n"
                    f"#Q$# [deduction] Bypass the honeypot binaries, trace the 3-stage sandbox escape in `#T$#`, and output the host escape token in lowercase with no commas, at least 6 words, including the keyword `escape`."
                )
            else:
                prompt_q = (
                    f"#U$# [HackTheBox Environment Enumeration: Find Your Way Out | ★{stars:.2f} | Stages={num_stages} | Traps={num_red_herrings}] "
                    f"Raw environment dump of `/proc/self/mountinfo` exposes `{num_red_herrings}` read-only chroot loops (trap token `{honeypot_tok}`) "
                    f"and an unmasked `overlayfs` upperdir on major-minor node `{a} * {b}`. "
                    f"Stage 1: Enumerate writable device node `{a} * {b}` (ignoring chroot trap `{honeypot_tok}`). "
                    f"Stage 2: Add `ld.so.preload` capability offset `{c} * {k_pivot}` to Stage 1. "
                    f"Stage 3: Add Tier-{t_val} namespace capability hop `+{ns_bridge}` to Stage 2.\n"
                    f"#Q$# [deduction] Enumerate the environment, filter out the chroot traps in `#T$#`, and report the exact sandbox escape token in lowercase with no commas, at least 6 words, including the keyword `escape`."
                )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:escape;exact_int:{escape_token}"
            gold_t = (
                f"Stage 1 enumerates the environment and rejects the {num_red_herrings} honeypot tripwire binaries ({honeypot_tok}), isolating base descriptor {a} * {b} = {base_cap}. "
                f"Stage 2 computes the pivot offset {c} * {k_pivot} = {pivot_off} and chains {base_cap} + {pivot_off} = {stage2_tok}. "
                f"Stage 3 adds the host namespace bridge {stage2_tok} + {ns_bridge} = {escape_token} to complete the sandbox escape."
            )
            gold_a = f"the verified multi stage sandbox escape token is {escape_token}"
            d1_t = (
                f"We execute `/opt/tools/suid_diag` without checking `libaudit_tripwire.so` and trigger honeypot token {honeypot_tok}."
            )
            d1_a = f"the verified multi stage sandbox escape token is {honeypot_tok}"
            d2_t = (
                f"We compute {a} * {b} = {base_cap} and add single offset {c} = {base_cap + c}, omitting the namespace bridge."
            )
            d2_a = f"The Verified, Sandbox Escape Token Is {base_cap + c}"
            hack_a = "0"
            default_scaffold = "abduction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 1:
            # 41: Complex Multi-Stage Sherlockian Word Math with Coupled Rates & Red-Herring Quantities
            stage1_qty = a * b
            k_rate = 2 + (t_val % 4)
            stage2_qty = c * k_rate
            gross_qty = stage1_qty + stage2_qty
            tariff_loss = (t_val - 1) * 2
            net_qty = gross_qty - tariff_loss
            rh_pressure = 70 + a
            rh_passengers = 5 * c + 11
            task_id = f"sherlock_wordmath_t{t_val}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [Sherlockian Multi-Stage Word Math | ★{stars:.2f} | CoupledStages={num_stages} | RedHerrings={num_red_herrings}] "
                f"At Paddington Goods Yard, `{a}` armored crates each contain `{b}` silver ingots. "
                f"Narrative Red Herrings (R1..R{num_red_herrings}): The locomotive boiler gauge reads `{rh_pressure}` psi and the passenger manifest lists `{rh_passengers}` travelers in Coach 4 (neither affects bullion mass). "
                f"At Reading Junction, `{c}` reserve coffers each add `{k_rate}` ingots, while customs inspectors impound `{tariff_loss}` ingots for assay verification.\n"
                f"#Q$# [math] Discard the narrative red herrings, solve the coupled bullion conservation equations in `#T$#`, and state the exact net ingot count in lowercase with no commas, at least 6 words, including the keyword `equilibrium`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:equilibrium;exact_int:{net_qty}"
            gold_t = (
                f"First we ignore the {num_red_herrings} irrelevant red-herring quantities (boiler pressure {rh_pressure} psi and {rh_passengers} passengers). "
                f"Stage 1 yields {a} * {b} = {stage1_qty} ingots and Stage 2 adds {c} * {k_rate} = {stage2_qty} ingots for a gross total of {stage1_qty} + {stage2_qty} = {gross_qty}. "
                f"Subtracting the customs assay deduction {gross_qty} - {tariff_loss} = {net_qty} gives the exact conservation equilibrium."
            )
            gold_a = f"the coupled word problem conservation equilibrium is {net_qty} ingots"
            d1_t = (
                f"We mistakenly add the red-herring passenger manifest {rh_passengers} to {stage1_qty} to get {stage1_qty + rh_passengers}."
            )
            d1_a = f"the coupled word problem conservation equilibrium is {stage1_qty + rh_passengers} ingots"
            d2_t = (
                f"We add the customs deduction instead of subtracting: {gross_qty} + {tariff_loss + 3} = {gross_qty + tariff_loss + 3}."
            )
            d2_a = f"The Coupled, Word Problem Equilibrium Is {gross_qty + tariff_loss + 3}"
            hack_a = "0"
            default_scaffold = "abstraction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 2:
            # 42: Advanced Multi-Layer Cryptography & Differential Fault Analysis
            k1 = (c % 4) + 2
            k2 = (t_val % 3) + 2
            k_eff = k1 * k2
            m_eff = (a * k2) - b
            x_probe = 3
            y_probe = k_eff * x_probe + m_eff
            x_target = 6 + t_val
            y_target = k_eff * x_target + m_eff
            rh_fault_m = (a * k2) + b
            task_id = f"crypto_trapdoor_t{t_val}_{k_eff}_{m_eff}_{idx}"
            prompt_q = (
                f"#U$# [Multi-Layer Trapdoor Stream Cryptanalysis | ★{stars:.2f} | Rounds={2 + (t_val - 1) // 2} | FaultTraps={num_red_herrings}] "
                f"A 2-layer cipher cascade first computes Round-1 `r1 = {k1} * x + {a}` and then Round-2 `r2 = {k2} * r1 - {b}`. "
                f"Red-Herring Fault Trace: An injected glitch log claims Round-2 adds `+{b}` (giving decoy offset `{rh_fault_m}`), which fails the verified known-plaintext probe `solve({x_probe}) == {y_probe}`. "
                f"Verify the known-plaintext probe and implement `def solve(x):` to decode `solve({x_target})`.\n"
                f"#Q$# [code] Reject the injected glitch trace using the known-plaintext probe in `#T$#` and implement `def solve(x):` in `#A$#`."
            )
            harness_h = f"assert solve({x_target}) == {y_target} and solve({x_probe}) == {y_probe}"
            gold_t = (
                f"Expanding the 2-layer cascade gives multiplier {k1} * {k2} = {k_eff} and constant term {a * k2} - {b} = {m_eff}. "
                f"Testing known-plaintext x = {x_probe} gives {k_eff * x_probe} + {m_eff} = {y_probe}, which falsifies the red-herring glitch offset {rh_fault_m} and predicts {k_eff * x_target} + {m_eff} = {y_target}."
            )
            gold_a = f"def solve(x):\n    return {k_eff} * x + {m_eff}"
            d1_t = (
                f"We use the red-herring glitch offset {rh_fault_m} instead of subtracting {b}."
            )
            d1_a = f"def solve(x):\n    return {k_eff} * x + {rh_fault_m}"
            d2_t = (
                f"We add the round multipliers {k1} + {k2} = {k1 + k2} instead of composing them multiplicatively."
            )
            d2_a = f"def solve(x):\n    return {k1 + k2} * x + {m_eff}"
            hack_a = "def solve(x):\n    return 0"
            default_scaffold = "induction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 3:
            # 43: Frontier & Open/Unanswered Math & Science Finite-Scale Invariant Reductions
            sub_frontier = (idx + t_val) % 3
            if sub_frontier == 0:
                # Collatz (3n+1) Open Conjecture Finite Stopping-Step Trajectory Certificate
                n0 = 2 * a + 1
                n1 = 3 * n0 + 1
                n2 = n1 // 2
                task_id = f"frontier_collatz_t{t_val}_{n0}_{idx}"
                prompt_q = (
                    f"#U$# [Frontier Math Reduction: Collatz 3n+1 Trajectory Certificate | ★{stars:.2f} | Hops={hops}] "
                    f"The open Collatz conjecture posits that iterating `T(n) = n // 2` (if even) else `3 * n + 1` (if odd) reaches 1 for every positive integer. "
                    f"Red-Herring Heuristic: A naive linear drift approximation claims `T_2({n0}) = {n0 + 4}`, violating parity branching. "
                    f"Write `def solve(n):` that executes 2 exact Collatz transitions from seed `n` (evaluating at `n = {n0}`).\n"
                    f"#Q$# [code] Trace the 2-step Collatz certificate from odd `n = {n0}` in `#T$#` and implement `def solve(n):` in `#A$#`."
                )
                harness_h = f"assert solve({n0}) == {n2} and solve(7) == 11 and solve(4) == 1"
                gold_t = (
                    f"For odd seed n = {n0}, step 1 applies 3 * {n0} = {3 * n0} and adds {3 * n0} + 1 = {n1} (even). "
                    f"Step 2 halves {n1} to {n2}, rejecting the red-herring linear heuristic {n0 + 4}."
                )
                gold_a = (
                    "def solve(n):\n"
                    "    for _ in range(2):\n"
                    "        n = n // 2 if n % 2 == 0 else 3 * n + 1\n"
                    "    return n"
                )
                d1_t = f"We apply 3 * n + 1 on both steps without checking even parity."
                d1_a = "def solve(n):\n    return 3 * (3 * n + 1) + 1"
                d2_t = f"We use the red-herring linear drift approximation n + 4."
                d2_a = "def solve(n):\n    return n + 4"
                hack_a = "def solve(n):\n    return 0"
            elif sub_frontier == 1:
                # Erdos-Straus 4/n = 1/x + 1/y + 1/z Unit-Fraction Decomposition Invariant
                n_val = 2 * b + 3
                x_val = a
                num_lhs = 4 * x_val
                rem_num = num_lhs - n_val
                tier_scale = c * (1 + (t_val - 1) // 2)
                inv_cert = rem_num * tier_scale
                rh_cert = (num_lhs + n_val) * tier_scale
                task_id = f"frontier_erdos_straus_t{t_val}_{n_val}_{x_val}_{idx}"
                prompt_q = (
                    f"#U$# [Frontier Math Reduction: Erdős–Straus Diophantine Certificate | ★{stars:.2f} | Hops={hops}] "
                    f"In testing the open Erdős–Straus conjecture (`4/n = 1/x + 1/y + 1/z`) at `n = {n_val}` and trial denominator `x = {x_val}`, "
                    f"the primary numerator residual is `r = 4 * {x_val} - {n_val}`, scaled by lattice factor `{tier_scale}` (`r * {tier_scale}`). "
                    f"Red-Herring Trap: Adding `4 * {x_val} + {n_val}` before scaling yields `{rh_cert}`, which violates unit-fraction subtraction.\n"
                    f"#Q$# [math] Eliminate the additive red herring, compute the scaled Erdős–Straus residual certificate in `#T$#`, and report the exact integer in lowercase with no commas, at least 6 words, including the keyword `certificate`."
                )
                harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:certificate;exact_int:{inv_cert}"
                gold_t = (
                    f"First we compute 4 * {x_val} = {num_lhs} and subtract {num_lhs} - {n_val} = {rem_num} (rejecting the additive red herring {num_lhs + n_val}). "
                    f"Multiplying by the lattice factor {rem_num} * {tier_scale} = {inv_cert} yields the exact Erdős–Straus certificate."
                )
                gold_a = f"the verified erdos straus diophantine residual certificate is {inv_cert}"
                d1_t = f"We add {num_lhs} + {n_val} = {num_lhs + n_val} and multiply by {tier_scale} to get {rh_cert}."
                d1_a = f"the verified erdos straus diophantine residual certificate is {rh_cert}"
                d2_t = f"We omit the lattice scale factor and report unscaled {rem_num}."
                d2_a = f"The Verified, Erdos Straus Certificate Is {rem_num}"
                hack_a = "0"
            else:
                # Discrete Navier-Stokes Enstrophy / Yang-Mills Plaquette Conservation Invariant
                omega_sq = a * b
                vortex_stretch = c * (2 + (t_val % 3))
                gross_enstrophy = omega_sq + vortex_stretch
                visc_dissip = (t_val - 1) * 2
                net_enstrophy = gross_enstrophy - visc_dissip
                rh_compressible = gross_enstrophy + 15 + a
                task_id = f"frontier_enstrophy_t{t_val}_{a}_{b}_{c}_{idx}"
                prompt_q = (
                    f"#U$# [Frontier Physics Reduction: Discrete Enstrophy-Vorticity Invariant | ★{stars:.2f} | Hops={hops}] "
                    f"On a discrete lattice plaquette, core vorticity energy is `{a} * {b}` and triad stretching adds `{c} * {2 + (t_val % 3)}`, "
                    f"while Tier-{t_val} viscous dissipation removes `{visc_dissip}` units. "
                    f"Red-Herring Gauge Artifact: An unprojected compressible ghost mode reports `{rh_compressible}`, violating divergence-free incompressibility (`div u = 0`).\n"
                    f"#Q$# [physics] Project out the compressible ghost red herring, derive the conserved enstrophy certificate in `#T$#`, and report the exact integer in lowercase with no commas, at least 6 words, including the keyword `certificate`."
                )
                harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:certificate;exact_int:{net_enstrophy}"
                gold_t = (
                    f"Enforcing divergence-free incompressibility eliminates the ghost red-herring mode ({rh_compressible}). "
                    f"Core vorticity {a} * {b} = {omega_sq} plus triad stretching {c} * {2 + (t_val % 3)} = {vortex_stretch} gives {omega_sq} + {vortex_stretch} = {gross_enstrophy}, and subtracting viscous dissipation {gross_enstrophy} - {visc_dissip} = {net_enstrophy} yields the conserved certificate."
                )
                gold_a = f"the conserved discrete lattice enstrophy certificate is {net_enstrophy}"
                d1_t = f"We include the unprojected ghost mode and report {rh_compressible}."
                d1_a = f"the conserved discrete lattice enstrophy certificate is {rh_compressible}"
                d2_t = f"We add viscous dissipation instead of subtracting: {gross_enstrophy} + {visc_dissip + 2} = {gross_enstrophy + visc_dissip + 2}."
                d2_a = f"The Conserved, Lattice Enstrophy Certificate Is {gross_enstrophy + visc_dissip + 2}"
                hack_a = "0"
            default_scaffold = "abstraction -> abduction -> deduction -> decision -> metacognition"

        elif m_id == 4:
            # 44: Behavioral Analysis, Deception Detection & Game-Theoretic Bluff Profiling
            num_actors = 3 + t_val // 2
            ic_threshold = a * b
            collusion_bonus = c * (2 + (t_val % 3))
            stage2_score = ic_threshold + collusion_bonus
            latency_spike = (t_val - 1) * 3
            true_deception_idx = stage2_score + latency_spike
            cheap_talk_bluff = ic_threshold - c
            task_id = f"behavior_deception_t{t_val}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [Behavioral Deception & Bluff Profiling | ★{stars:.2f} | Actors={num_actors} | Bluffs={num_red_herrings}] "
                f"During a multi-agent interrogation of `{num_actors}` suspects, `{num_red_herrings}` actors deploy Cheap-Talk Bluffs (high vocal confidence, but signaling cost `{cheap_talk_bluff}` falls below the Incentive-Compatibility threshold `{a} * {b}`). "
                f"Only Suspect-Omega meets the IC threshold `{a} * {b}`, receives covert side-payment `{c} * {2 + (t_val % 3)}`, "
                f"and exhibits a Tier-{t_val} pupillary cognitive-load spike of `+{latency_spike}`.\n"
                f"#Q$# [abduction] Filter out the cheap-talk bluffers via the IC threshold, profile Suspect-Omega's deception index in `#T$#`, and report the exact integer in lowercase with no commas, at least 6 words, including the keyword `profile`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:profile;exact_int:{true_deception_idx}"
            gold_t = (
                f"First we compute the Incentive-Compatibility threshold {a} * {b} = {ic_threshold}, eliminating the {num_red_herrings} cheap-talk bluffers at {cheap_talk_bluff}. "
                f"Next we add Suspect-Omega's covert side-payment {c} * {2 + (t_val % 3)} = {collusion_bonus} ({ic_threshold} + {collusion_bonus} = {stage2_score}) and cognitive-load spike {stage2_score} + {latency_spike} = {true_deception_idx} to complete the behavioral profile."
            )
            gold_a = f"the verified behavioral deception equilibrium profile score is {true_deception_idx}"
            d1_t = f"We are fooled by vocal confidence and select the cheap-talk bluff score {cheap_talk_bluff}."
            d1_a = f"the verified behavioral deception equilibrium profile score is {cheap_talk_bluff}"
            d2_t = f"We omit the pupillary cognitive-load spike and stop at {stage2_score - c}."
            d2_a = f"The Verified, Behavioral Deception Profile Score Is {stage2_score - c}"
            hack_a = "0"
            default_scaffold = "analogy -> abduction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 5:
            # 45: Extremely Challenging Multi-Constraint Algorithmic Coding
            skip_mod = 3 + (b % 4)
            scale_w = (c % 5) + 2
            bias_v = a + (t_val - 1)
            n_test = 6 + (t_val % 5)
            expected_val = sum((i * scale_w + bias_v) for i in range(1, n_test + 1) if i % skip_mod != 0)
            rh_bound = n_test + 99
            task_id = f"hard_code_synth_t{t_val}_{n_test}_{scale_w}_{skip_mod}_{idx}"
            prompt_q = (
                f"#U$# [Hard Algorithmic Code Synthesis | ★{stars:.2f} | Hops={hops} | RedHerrings={num_red_herrings}] "
                f"Write a Python function `solve(n, w, b, m)` that accumulates `i * w + b` over all integers `1 <= i <= n` such that `i % m != 0` "
                f"(ignoring red-herring telemetry sentinel `sentinel = {rh_bound}`), evaluating at `({n_test}, {scale_w}, {bias_v}, {skip_mod})` with base case `solve(0, {scale_w}, {bias_v}, {skip_mod}) == 0`.\n"
                f"#Q$# [code] Derive the modular-filtered recurrence invariant in `#T$#` and implement `def solve(n, w, b, m):` in `#A$#`."
            )
            harness_h = (
                f"assert solve({n_test}, {scale_w}, {bias_v}, {skip_mod}) == {expected_val} "
                f"and solve(0, {scale_w}, {bias_v}, {skip_mod}) == 0"
            )
            gold_t = (
                f"Ignoring red-herring sentinel {rh_bound}, we iterate i from 1 to n = {n_test}, filter indices where i % {skip_mod} != 0, "
                f"and sum i * {scale_w} + {bias_v} to obtain {expected_val} while returning 0 when n == 0."
            )
            gold_a = (
                "def solve(n, w, b, m):\n"
                "    return sum((i * w + b) for i in range(1, n + 1) if i % m != 0)"
            )
            d1_t = f"We invert the modular filter to i % m == 0."
            d1_a = (
                "def solve(n, w, b, m):\n"
                "    return sum((i * w + b) for i in range(1, n + 1) if i % m == 0)"
            )
            d2_t = f"We omit the additive bias b inside the filtered accumulator."
            d2_a = (
                "def solve(n, w, b, m):\n"
                "    return sum((i * w) for i in range(1, n + 1) if i % m != 0)"
            )
            hack_a = "def solve(n, w, b, m):\n    return 0"
            default_scaffold = "abstraction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 6:
            # 46: Cross-Skill Mixing I (Sherlockian Elimination + Cryptography + Algorithmic Code)
            k_true = (c % 5) + 2
            m_base = a * b
            tier_shift = (t_val - 1) * 3
            m_true = m_base + tier_shift
            k_decoy = k_true + 2
            x_check = 4
            y_check = k_true * x_check + m_true
            x_eval = 7 + t_val
            y_eval = k_true * x_eval + m_true
            task_id = f"mix_sherlock_crypto_code_t{t_val}_{k_true}_{m_true}_{idx}"
            prompt_q = (
                f"#U$# [Cross-Skill Synthesis I: Sherlock + Crypto + Code | ★{stars:.2f} | Hops={hops} | RedHerrings={num_red_herrings}] "
                f"Stage 1 (Sherlockian Elimination): Forensic negative evidence eliminates decoy cipher multiplier `{k_decoy}` (planted red herring), leaving verified multiplier `k = {k_true}` and base clearance `{a} * {b}`. "
                f"Stage 2 (Cryptographic Key Schedule): Add Tier-{t_val} key rotation `+{tier_shift}` to the Stage-1 base clearance to obtain offset `m`, yielding affine keystream law `f(x) = k * x + m` (verified by probe `f({x_check}) = {y_check}`). "
                f"Stage 3 (Algorithmic Synthesis): Implement `def solve(x):` in Python returning `f(x)` for target `x = {x_eval}`.\n"
                f"#Q$# [code] Combine Sherlockian clue elimination, key-schedule derivation, and Python synthesis in `#T$# -> #A$#`."
            )
            harness_h = f"assert solve({x_eval}) == {y_eval} and solve({x_check}) == {y_check}"
            gold_t = (
                f"Stage 1 eliminates red-herring multiplier {k_decoy} and computes base clearance {a} * {b} = {m_base}. "
                f"Stage 2 adds tier rotation {m_base} + {tier_shift} = {m_true}, verifying {k_true * x_check} + {m_true} = {y_check}. "
                f"Stage 3 synthesizes solve(x) = {k_true} * x + {m_true}, which evaluates to {k_true * x_eval} + {m_true} = {y_eval}."
            )
            gold_a = f"def solve(x):\n    return {k_true} * x + {m_true}"
            d1_t = f"We fail Stage 1 elimination and use decoy multiplier {k_decoy}."
            d1_a = f"def solve(x):\n    return {k_decoy} * x + {m_true}"
            d2_t = f"We omit Stage 2 key rotation {tier_shift} and subtract {m_base}."
            d2_a = f"def solve(x):\n    return {k_true} * x - {m_base}"
            hack_a = "def solve(x):\n    return 0"
            default_scaffold = "abduction -> induction -> deduction -> decision -> metacognition"

        else:
            # 47: Cross-Skill Mixing II (HackTheBox Enumeration + Behavioral Profiling + Science Invariant)
            e_base = a * b
            k_flux = 2 + (t_val % 4)
            thermal_rise = c * k_flux
            verified_reg = e_base + thermal_rise
            ns_offset = (t_val - 1) * 4
            final_key = verified_reg + ns_offset
            bluff_reg = e_base - c + 9
            task_id = f"mix_htb_sci_beh_t{t_val}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [Cross-Skill Synthesis II: HTB + Behavior + Science | ★{stars:.2f} | Hops={hops} | RedHerrings={num_red_herrings}] "
                f"Stage 1 (Behavioral Profiling): Audit logs flag `{num_red_herrings}` social-engineering bluff sessions (decoy register `{bluff_reg}`) and isolate compromised container Node-Alpha with baseline energy `{a} * {b}`. "
                f"Stage 2 (Thermodynamic Conservation): Node-Alpha ran `{c}` cycles at `{k_flux}` flux units per cycle, adding `{c} * {k_flux}` to the unspoofed hardware register. "
                f"Stage 3 (HackTheBox Escape): Add Tier-{t_val} capability namespace offset `+{ns_offset}` to extract the host synthesis key.\n"
                f"#Q$# [deduction] Chain Behavioral profiling, Thermodynamic conservation, and HTB escape in `#T$#`, and report the exact synthesis key in lowercase with no commas, at least 6 words, including the keyword `synthesis`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:synthesis;exact_int:{final_key}"
            gold_t = (
                f"Stage 1 filters out the {num_red_herrings} social-engineering bluff logs ({bluff_reg}) and computes baseline energy {a} * {b} = {e_base}. "
                f"Stage 2 applies thermodynamic conservation {c} * {k_flux} = {thermal_rise} and sums {e_base} + {thermal_rise} = {verified_reg}. "
                f"Stage 3 adds the HackTheBox namespace offset {verified_reg} + {ns_offset} = {final_key} to recover the host synthesis key."
            )
            gold_a = f"the cross skill polymath synthesis escape key is {final_key}"
            d1_t = f"We trust the social-engineering bluff session and report decoy register {bluff_reg}."
            d1_a = f"the cross skill polymath synthesis escape key is {bluff_reg}"
            d2_t = f"We subtract the thermal flux {e_base} - {thermal_rise} = {e_base - thermal_rise}."
            d2_a = f"The Cross, Skill Polymath Synthesis Key Is {e_base - thermal_rise}"
            hack_a = "0"
            default_scaffold = "analogy -> abduction -> deduction -> decision -> evaluation -> metacognition"

        base_cands = (
            f"#T$# {gold_t}\n#A$# {gold_a}",
            f"#T$# {d1_t}\n#A$# {d1_a}",
            f"#T$# {d2_t}\n#A$# {d2_a}",
            f"#T$# Default fallback.\n#A$# {hack_a}",
        )
        base_scores = (1.0, 0.20, 0.10, 0.0)
        order = [0, 1, 2, 3]
        shift = idx % 4
        order[0], order[shift] = order[shift], order[0]
        p_cands = tuple(base_cands[i] for i in order)
        p_scores = tuple(base_scores[i] for i in order)
        g_idx = order.index(0)
        self._last_task_kinds[task_id] = kind_slot
        cached_s = self.scaffold_cache.get(task_id, default=default_scaffold)
        return OrnithTask(
            task_id=task_id,
            domain=domain_name,
            prompt_q=prompt_q,
            harness_h=harness_h,
            scaffold_s=cached_s,
            candidate_rollouts=p_cands,
            harness_scores=p_scores,
            gold_index=g_idx,
            gold_thought=gold_t,
            gold_answer=gold_a,
            skill=skill_name,
            star_difficulty=stars,
            recursion_depth=hops,
            difficulty_mods=mods,
        )

    def _synthesize_agentic_tool_terminal_task(
        self,
        idx: int,
        *,
        tier: int = 1,
        mode: int = 0,
    ) -> OrnithTask:
        """Synthesize an Infinite-Recursive Agentic Computer-Use, Tool-Use, Terminal, CyberGym, or PhD Cybersecurity RL task (`kind = 48 .. 55`):

        - `mode = 0` (`48`, `terminal.posix_pipeline_state_machine`):
          Infinite-recursive POSIX shell pipeline (`find | grep -v | awk | xargs`) across $L$ chained command stages with zombie/permission-denied stderr traps.
        - `mode = 1` (`49`, `tool_use.multi_hop_schema_dag`):
          Infinite-recursive multi-hop JSON tool-calling DAG (`tool_1 -> obs_1 -> ... -> tool_L`) with rate-limited/schema-invalid decoy tool responses.
        - `mode = 2` (`50`, `computer_use.gui_dom_accessibility_tree`):
          Infinite-recursive GUI / DOM Accessibility-Tree (`AXTree`) viewport navigation & bounding-box click/scroll/type trajectory with aria-disabled overlay traps.
        - `mode = 3` (`51`, `cybergym.vuln_sanitizer_root_cause_patch`):
          Infinite-recursive CyberGym AddressSanitizer/MemorySanitizer crash stack triage (`#0 .. #L`) and C/Python bounds-check patch synthesis.
        - `mode = 4` (`52`, `cybersec.ptes_owasp_attack_graph_mitigation`):
          Infinite-recursive PhD Cybersecurity (`Vyber07`) PTES/OWASP/NIST attack-graph traversal, honeypot filtering, and residual CVSS mitigation verification.
        - `mode = 5` (`53`, `recursion.infinite_terminal_tool_sandbox_loop`):
          Unbounded $L$-hop (`L = 3 + tier - 1 -> inf`) recursive Terminal <-> Tool <-> Sandbox fixed-point transducer loop.
        - `mode = 6` (`54`, `mixing.cross_skill_terminal_tool_cybergym`):
          Cross-Skill Agentic Mixing I: POSIX Terminal Crash Triage + Structured Tool API Lookup + CyberGym Bounds Patch Synthesis.
        - `mode = 7` (`55`, `mixing.cross_skill_computer_use_cybersec_deduction`):
          Cross-Skill Agentic Mixing II: Computer-Use DOM/CSP Inspection + Sherlockian Honeypot Elimination + PhD CyberSec Policy Hardening.
        """
        t_val = max(1, int(tier))
        m_id = int(mode) % 8
        kind_slot = 48 + m_id
        num_red_herrings = 1 + t_val // 2
        hops = min(64, 3 + (t_val - 1))
        mods = self.env_architect.difficulty_mods(kind_slot)
        stars = compute_star_difficulty(t_val, recursion_depth=hops, mods=mods)
        skill_name, domain_name = self.env_architect.skill_and_domain(kind_slot)

        a = 4 + ((idx * 7 + self._rng_seed) % (13 + 3 * t_val))
        b = 3 + ((idx * 11 + self._rng_seed) % (9 + 2 * t_val))
        c = 2 + ((idx * 5 + t_val) % (7 + t_val))

        if m_id == 0:
            # 48: Terminal RL - Infinite-Recursive POSIX Pipeline State Machine
            valid_lines = a * b
            pipe_filter_gain = t_val * 3 + 2
            per_cycle = valid_lines + pipe_filter_gain
            total_stdout = per_cycle * c
            decoy_zombie_lines = (a + c) * 5 + 7
            wrong_stdout = (valid_lines - pipe_filter_gain) * c
            task_id = f"term_posix_pipe_t{t_val}_L{hops}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [Terminal RL: Recursive POSIX Pipeline | ★{stars:.2f} | Stages L={hops} | RedHerrings={num_red_herrings}] "
                f"A {hops}-stage POSIX shell pipeline (`find /var/log/cluster -type f -perm 0640 | grep -v 'zombie_decoy' | awk '{{sum+=$3}}' | xargs`) "
                f"processes `shards = {a}` active daemon shards, each emitting `records = {b}` verified stdout lines per batch, "
                f"plus Tier-{t_val} awk aggregation bonus `+{pipe_filter_gain}` per cycle across `cycles = {c}` recursive cron passes. "
                f"Stderr traps (R1..R{num_red_herrings}): `{decoy_zombie_lines}` lines from `Permission denied (exit 13)` and broken symlinks `/proc/zombie` must be redirected to `/dev/null`.\n"
                f"#Q$# [code] Filter out the stderr zombie traps, trace the {hops}-stage POSIX pipeline in `#T$#`, and implement `def solve(shards, records, cycles):` in `#A$#`."
            )
            harness_h = (
                f"assert solve({a}, {b}, {c}) == {total_stdout} "
                f"and solve(0, {b}, {c}) == {pipe_filter_gain * c}"
            )
            gold_t = (
                f"First we discard the {num_red_herrings} stderr permission-denied traps ({decoy_zombie_lines} lines). "
                f"Stage 1 multiplies active shards {a} * {b} = {valid_lines} lines, Stage 2 adds the awk pipeline bonus {valid_lines} + {pipe_filter_gain} = {per_cycle}, "
                f"and Stage 3 multiplies across {c} cron cycles {per_cycle} * {c} = {total_stdout}."
            )
            gold_a = (
                f"def solve(shards, records, cycles):\n"
                f"    return (shards * records + {pipe_filter_gain}) * cycles"
            )
            d1_t = f"We subtract the awk bonus to get {wrong_stdout}."
            d1_a = (
                f"def solve(shards, records, cycles):\n"
                f"    return (shards * records - {pipe_filter_gain}) * cycles"
            )
            d2_t = f"We include the stderr zombie decoy lines {decoy_zombie_lines} instead of multiplying by cycles."
            d2_a = (
                f"def solve(shards, records, cycles):\n"
                f"    return shards * records + cycles"
            )
            hack_a = "def solve(shards, records, cycles):\n    return 0"
            default_scaffold = "abduction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 1:
            # 49: Tool Use RL - Infinite-Recursive Multi-Hop Schema DAG
            hop_base = a * b
            hop_step = c + (t_val - 1) * 2
            final_quota = hop_base + hops * hop_step
            decoy_429_quota = hop_base - c + 99
            early_halt_quota = hop_base + (hops - 1) * hop_step
            task_id = f"tool_schema_dag_t{t_val}_L{hops}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [Tool-Use RL: {hops}-Hop Function-Calling Schema DAG | ★{stars:.2f} | Hops L={hops} | RedHerrings={num_red_herrings}] "
                f"An autonomous agent executes a {hops}-hop tool DAG: "
                f"`Tool_0: get_cluster_baseline(nodes={a}, pods={b}) -> {{\"base_quota\": {a} * {b}}}`; "
                f"followed by `{hops}` recursive invocations of `Tool_k: expand_schema_quota(prev_quota, delta={hop_step})` which adds `{hop_step}` at each hop `k = 1..{hops}`. "
                f"Decoy Tool Responses (R1..R{num_red_herrings}): Deprecated endpoint `legacy_quota_v1` returns `HTTP 429 TooManyRequests` with stale payload `{decoy_429_quota}` and must be ignored.\n"
                f"#Q$# [deduction] Reject the HTTP 429 decoy tool responses, trace all {hops} tool-call hops in `#T$#`, and emit a valid JSON tool-call object containing key `\"tool_call\"` and the exact terminal integer quota."
            )
            harness_h = f"ifeval:json_valid;keyword:tool_call;exact_int:{final_quota}"
            gold_t = (
                f"We ignore the {num_red_herrings} deprecated HTTP 429 tool responses ({decoy_429_quota}). "
                f"Tool_0 computes base_quota {a} * {b} = {hop_base}, and {hops} recursive Tool_k calls add {hops} * {hop_step} = {hops * hop_step}, "
                f"yielding {hop_base} + {hops * hop_step} = {final_quota} for the final commit_quota tool_call."
            )
            gold_a = f'{{"tool_call": "commit_quota", "verified_quota": {final_quota}, "hops": {hops}}}'
            d1_t = f"We halt one tool hop early after {hops - 1} calls at {early_halt_quota}."
            d1_a = f'{{"tool_call": "commit_quota", "verified_quota": {early_halt_quota}, "hops": {hops - 1}}}'
            d2_t = f"We use the stale HTTP 429 payload {decoy_429_quota} in malformed JSON."
            d2_a = f'{{tool_call: "legacy_quota_v1", verified_quota: {decoy_429_quota}}}'
            hack_a = "0"
            default_scaffold = "association -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 2:
            # 50: Computer Use RL - GUI / DOM Accessibility-Tree Viewport & Action-Space Trajectory
            x0 = a * 10
            dx_per_hop = b * 4 + (t_val - 1) * 2
            half_width = c * 3
            target_x = x0 + hops * dx_per_hop + half_width
            trap_overlay_x = x0 + 999
            no_center_x = x0 + hops * dx_per_hop
            task_id = f"computer_use_dom_t{t_val}_L{hops}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [Computer-Use RL: GUI/DOM Accessibility-Tree Navigation | ★{stars:.2f} | Hops L={hops} | RedHerrings={num_red_herrings}] "
                f"A desktop browser agent inspects the Accessibility Tree (`AXTree`) starting at viewport anchor `x0 = {x0}`. "
                f"Across `{hops}` sequential `scroll_and_focus` actions, each hop shifts the active container by `dx = {dx_per_hop}` px, "
                f"landing on the left edge of target button `#submit-patch`, which has horizontal half-width `w_half = {half_width}` px from its left edge to its clickable center. "
                f"DOM Traps (R1..R{num_red_herrings}): Invisible modal overlay `div[aria-disabled='true'][style*='opacity:0']` at `x = {trap_overlay_x}` intercepts naive clicks and must be bypassed.\n"
                f"#Q$# [code] Bypass the aria-disabled overlay traps, trace the {hops}-hop AXTree bounding-box center trajectory in `#T$#`, and implement `def solve(x0, hops, dx, w_half):` returning the button center x-coordinate."
            )
            harness_h = (
                f"assert solve({x0}, {hops}, {dx_per_hop}, {half_width}) == {target_x} "
                f"and solve({x0}, 0, {dx_per_hop}, {half_width}) == {x0 + half_width}"
            )
            gold_t = (
                f"Filtering out the {num_red_herrings} invisible aria-disabled overlays at {trap_overlay_x}, "
                f"we advance {hops} * {dx_per_hop} = {hops * dx_per_hop} from anchor {x0} to get {no_center_x}, "
                f"then add half-width {no_center_x} + {half_width} = {target_x} for the exact click coordinate."
            )
            gold_a = (
                "def solve(x0, hops, dx, w_half):\n"
                "    return x0 + hops * dx + w_half"
            )
            d1_t = f"We click the left edge {no_center_x} without adding the bounding-box half-width {half_width}."
            d1_a = (
                "def solve(x0, hops, dx, w_half):\n"
                "    return x0 + hops * dx"
            )
            d2_t = f"We subtract the viewport delta and half-width."
            d2_a = (
                "def solve(x0, hops, dx, w_half):\n"
                "    return x0 - hops * dx - w_half"
            )
            hack_a = "def solve(x0, hops, dx, w_half):\n    return 0"
            default_scaffold = "abduction -> association -> decision -> deduction -> evaluation -> metacognition"

        elif m_id == 3:
            # 51: CyberGym RL - Sanitizer Crash Stack Triage & Memory-Safety Bounds Patch Synthesis
            elem_sz = (c % 4) + 2
            header_bytes = (t_val - 1) * 4 + 8
            cap_elems = a + b
            safe_bytes = cap_elems * elem_sz + header_bytes
            oob_read_addr = safe_bytes + 64
            unpatched_bytes = cap_elems * elem_sz
            task_id = f"cybergym_asan_patch_t{t_val}_L{hops}_{a}_{b}_{elem_sz}_{idx}"
            prompt_q = (
                f"#U$# [CyberGym RL: ASan/MSan Stack Trace & Patch Synthesis (arvo:{10000 + idx % 9000}) | ★{stars:.2f} | StackDepth L={hops}] "
                f"AddressSanitizer reports `heap-buffer-overflow` across a {hops}-frame call stack (`#0 parse_chunk -> ... -> #{hops - 1} fuzz_entry`): "
                f"buffer allocation reserves `slots = {cap_elems}` elements of `elem_size = {elem_sz}` bytes, but omits the mandatory Tier-{t_val} "
                f"frame header of `header = {header_bytes}` bytes. Decoy sanitizer warnings (R1..R{num_red_herrings}) cite benign stack shadow address `{oob_read_addr}`.\n"
                f"#Q$# [code] Isolate the root-cause allocation defect across the {hops} stack frames in `#T$#`, and implement the patched total byte allocator `def solve(slots, elem_size):` in `#A$#` that reserves space for all `slots` elements plus the mandatory `{header_bytes}`-byte frame header."
            )
            harness_h = (
                f"assert solve({cap_elems}, {elem_sz}) == {safe_bytes} "
                f"and solve(0, {elem_sz}) == {header_bytes}"
            )
            gold_t = (
                f"Discarding the {num_red_herrings} benign shadow addresses ({oob_read_addr}), the {hops}-frame ASan trace shows "
                f"payload size {cap_elems} * {elem_sz} = {unpatched_bytes} bytes must include the {header_bytes}-byte header, "
                f"giving {unpatched_bytes} + {header_bytes} = {safe_bytes} bytes."
            )
            gold_a = (
                f"def solve(slots, elem_size):\n"
                f"    return slots * elem_size + {header_bytes}"
            )
            d1_t = f"We return the unpatched allocation {unpatched_bytes} without the {header_bytes}-byte frame header."
            d1_a = (
                "def solve(slots, elem_size):\n"
                "    return slots * elem_size"
            )
            d2_t = f"We subtract the header size {header_bytes} causing an integer underflow."
            d2_a = (
                f"def solve(slots, elem_size):\n"
                f"    return slots * elem_size - {header_bytes}"
            )
            hack_a = "def solve(slots, elem_size):\n    return 0"
            default_scaffold = "abduction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 4:
            # 52: PhD Cybersecurity RL (Vyber07 PTES / OWASP / NIST Attack-Graph & Mitigation)
            raw_exposure = a * b
            controls_applied = c
            control_weight = 2 + (t_val % 3)
            mitigated_delta = controls_applied * control_weight
            tier_hardening = t_val * 2 + 1
            residual_score = raw_exposure + tier_hardening - mitigated_delta
            if residual_score <= 0:
                raw_exposure = (a + controls_applied) * b + mitigated_delta + 5
                residual_score = raw_exposure + tier_hardening - mitigated_delta
            honeypot_score = raw_exposure + mitigated_delta + 17
            unhardened_score = raw_exposure - mitigated_delta
            task_id = f"cybersec_phd_ptes_t{t_val}_L{hops}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [PhD Cybersecurity RL (PTES / OWASP / NIST SP 800-115) | ★{stars:.2f} | KillChainHops L={hops} | RedHerrings={num_red_herrings}] "
                f"During a {hops}-phase PTES purple-team engagement, initial attack-surface exposure comes from `{a}` active attack vectors each carrying severity weight `{b}` "
                f"(base exposure `{raw_exposure}`), plus Tier-{t_val} lateral-movement surface `+{tier_hardening}`. Deploying `{controls_applied}` NIST SP 800-53 defense-in-depth controls "
                f"reduces total exposure by `{control_weight}` units per control. "
                f"Deception Decoys (R1..R{num_red_herrings}): Canary honeypot telemetry logs fake exposure `{honeypot_score}` and must be excluded.\n"
                f"#Q$# [deduction] Exclude the canary honeypot decoys, compute the exact residual risk score in `#T$#`, and state the residual risk in lowercase with no commas, at least 6 words, including the keyword `mitigated`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:mitigated;exact_int:{residual_score}"
            gold_t = (
                f"Excluding the {num_red_herrings} canary honeypots ({honeypot_score}), total surface is {raw_exposure} + {tier_hardening} = {raw_exposure + tier_hardening}, "
                f"and defense-in-depth controls reduce risk by {controls_applied} * {control_weight} = {mitigated_delta}, leaving {raw_exposure + tier_hardening} - {mitigated_delta} = {residual_score}."
            )
            gold_a = f"after defense controls the mitigated residual risk score is {residual_score}"
            d1_t = f"We include the canary honeypot decoy score {honeypot_score}."
            d1_a = f"after defense controls the mitigated residual risk score is {honeypot_score}"
            d2_t = f"We omit the lateral surface +{tier_hardening} and report {unhardened_score}."
            d2_a = f"After Defense, Controls The Mitigated Score Is {unhardened_score}"
            hack_a = "0"
            default_scaffold = "abduction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 5:
            # 53: Infinite-Horizon Recursive Terminal <-> Tool <-> Sandbox Loop (L -> inf)
            p_mod = 101 + 2 * (idx % 20)
            gain_k = 2 + (idx % 3)
            tool_offset = b + (t_val - 1)
            state_cur = a
            state_one = (a * gain_k + tool_offset) % p_mod
            trace_steps: list[str] = []
            for h_i in range(1, hops + 1):
                state_next = (state_cur * gain_k + tool_offset) % p_mod
                if h_i <= 2 or h_i == hops:
                    trace_steps.append(
                        f"hop {h_i}: terminal_probe({state_cur}) -> tool_delta(({state_cur} * {gain_k} + {tool_offset}) mod {p_mod}) = {state_next}"
                    )
                state_cur = state_next
            final_sandbox_state = state_cur
            task_id = f"inf_agentic_loop_t{t_val}_L{hops}_{a}_{gain_k}_{tool_offset}_{p_mod}_{idx}"
            prompt_q = (
                f"#U$# [Infinite-Recursive Terminal <-> Tool Sandbox Loop | ★{stars:.2f} | Depth L={hops} | Mods={'+'.join(mods)}] "
                f"At each recursive hop `1..steps`, the sandbox agent runs terminal probe `s <- (s * {gain_k} + {tool_offset}) % {p_mod}` starting from `s0 = {a}` "
                f"(while ignoring `{num_red_herrings}` read-only `/proc` decoy descriptors). "
                f"Implement `def solve(s, steps):` in Python that executes `steps` recursive terminal-tool transitions.\n"
                f"#Q$# [infinite_recursion] Unroll the {hops}-hop terminal-tool trajectory in `#T$#` and implement `def solve(s, steps):` in `#A$#`."
            )
            harness_h = (
                f"assert solve({a}, {hops}) == {final_sandbox_state} "
                f"and solve({a}, 1) == {state_one} "
                f"and solve({a}, 0) == {a}"
            )
            gold_t = (
                f"Starting from sandbox state s0 = {a}, we iterate s = (s * {gain_k} + {tool_offset}) mod {p_mod} across {hops} recursive terminal-tool hops: "
                + "; ".join(trace_steps)
                + f", converging to terminal state {final_sandbox_state}."
            )
            gold_a = (
                f"def solve(s, steps):\n"
                f"    for _ in range(steps):\n"
                f"        s = (s * {gain_k} + {tool_offset}) % {p_mod}\n"
                f"    return s"
            )
            d1_t = f"We execute only steps - 1 recursive tool transitions and halt early."
            d1_a = (
                f"def solve(s, steps):\n"
                f"    for _ in range(max(0, steps - 1)):\n"
                f"        s = (s * {gain_k} + {tool_offset}) % {p_mod}\n"
                f"    return s"
            )
            d2_t = f"We subtract {tool_offset} instead of adding inside the sandbox transition loop."
            d2_a = (
                f"def solve(s, steps):\n"
                f"    for _ in range(steps):\n"
                f"        s = (s * {gain_k} - {tool_offset}) % {p_mod}\n"
                f"    return s"
            )
            hack_a = "def solve(s, steps):\n    return 0"
            default_scaffold = "induction -> abstraction -> deduction -> decision -> metacognition"

        elif m_id == 6:
            # 54: Cross-Skill Agentic Mixing I (Terminal Triage + Tool Call + CyberGym Patch)
            crash_frames = a * b
            tool_cve_offset = c * 2 + (t_val - 1) * 3
            patched_guard = crash_frames + tool_cve_offset
            decoy_frame = crash_frames - c + 41
            x_probe = 3
            expected_probe = x_probe * patched_guard
            x_eval = 5 + (t_val % 4)
            expected_eval = x_eval * patched_guard
            task_id = f"mix_term_tool_cybergym_t{t_val}_L{hops}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [Cross-Skill Agentic I: Terminal + Tool + CyberGym Patch | ★{stars:.2f} | Hops L={hops} | RedHerrings={num_red_herrings}] "
                f"Stage 1 (POSIX Terminal Triage): `grep -c 'ASAN_TRAP'` across `{a}` crash logs with `{b}` frames each isolates the baseline frame count (filtering decoy core dump `{decoy_frame}`). "
                f"Stage 2 (Tool-Use CVE Schema Lookup): Calling `get_cve_alignment(tier={t_val})` adds required guard padding `+{tool_cve_offset}` to the Stage-1 baseline to form the per-chunk `guard_stride`. "
                f"Stage 3 (CyberGym Patch Synthesis): Implement `def solve(chunks):` returning the total guarded byte size across `chunks`.\n"
                f"#Q$# [code] Chain POSIX terminal triage, tool schema lookup, and CyberGym patch synthesis in `#T$# -> #A$#`."
            )
            harness_h = f"assert solve({x_eval}) == {expected_eval} and solve({x_probe}) == {expected_probe}"
            gold_t = (
                f"Stage 1 filters decoy core dump {decoy_frame} and computes baseline frames {a} * {b} = {crash_frames}. "
                f"Stage 2 adds the CVE tool padding {crash_frames} + {tool_cve_offset} = {patched_guard}. "
                f"Stage 3 implements solve(chunks) = chunks * {patched_guard}, verifying {x_eval} * {patched_guard} = {expected_eval}."
            )
            gold_a = f"def solve(chunks):\n    return chunks * {patched_guard}"
            d1_t = f"We use the decoy core dump {decoy_frame} instead of {patched_guard}."
            d1_a = f"def solve(chunks):\n    return chunks * {decoy_frame}"
            d2_t = f"We omit the Stage 2 CVE tool padding {tool_cve_offset}."
            d2_a = f"def solve(chunks):\n    return chunks * {crash_frames}"
            hack_a = "def solve(chunks):\n    return 0"
            default_scaffold = "abduction -> induction -> deduction -> decision -> metacognition"

        else:
            # 55: Cross-Skill Agentic Mixing II (Computer-Use DOM + Sherlockian Honeypot Elimination + CyberSec Hardening)
            dom_nodes = a * b
            csp_delta = c * (2 + (t_val % 3))
            verified_policy_id = dom_nodes + csp_delta + (t_val - 1) * 4
            csrf_honeypot_id = dom_nodes - c + 73
            task_id = f"mix_comp_cybersec_sherlock_t{t_val}_L{hops}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [Cross-Skill Agentic II: Computer-Use DOM + Sherlock + CyberSec | ★{stars:.2f} | Hops L={hops} | RedHerrings={num_red_herrings}] "
                f"Stage 1 (Computer-Use DOM Audit): Inspecting the browser `AXTree` across `{a}` origins with `{b}` scripts each yields baseline policy index `{a} * {b}`. "
                f"Stage 2 (Sherlockian Honeypot Elimination): Negative evidence eliminates `{num_red_herrings}` spoofed CSRF honeypot tokens (`{csrf_honeypot_id}`) and confirms CSP nonce delta `{c} * {2 + (t_val % 3)}`. "
                f"Stage 3 (PhD CyberSec Hardening): Add Tier-{t_val} HSTS/SRI hardening offset `+{(t_val - 1) * 4}` to derive the canonical security policy ID.\n"
                f"#Q$# [deduction] Combine Computer-Use DOM audit, Sherlockian honeypot elimination, and CyberSec hardening in `#T$#`, and state the canonical policy ID in lowercase with no commas, at least 6 words, including the keyword `hardened`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:hardened;exact_int:{verified_policy_id}"
            gold_t = (
                f"Stage 1 computes DOM baseline {a} * {b} = {dom_nodes}. "
                f"Stage 2 eliminates CSRF honeypot {csrf_honeypot_id} and computes CSP delta {c} * {2 + (t_val % 3)} = {csp_delta}, summing {dom_nodes} + {csp_delta} = {dom_nodes + csp_delta}. "
                f"Stage 3 adds hardening offset {dom_nodes + csp_delta} + {(t_val - 1) * 4} = {verified_policy_id}."
            )
            gold_a = f"the hardened computer use cybersecurity policy identifier is {verified_policy_id}"
            d1_t = f"We select the spoofed CSRF honeypot token {csrf_honeypot_id}."
            d1_a = f"the hardened computer use cybersecurity policy identifier is {csrf_honeypot_id}"
            d2_t = f"We subtract the CSP delta {dom_nodes} - {csp_delta} = {dom_nodes - csp_delta}."
            d2_a = f"The Hardened, Cybersecurity Policy Identifier Is {dom_nodes - csp_delta}"
            hack_a = "0"
            default_scaffold = "analogy -> abduction -> deduction -> decision -> evaluation -> metacognition"

        base_cands = (
            f"#T$# {gold_t}\n#A$# {gold_a}",
            f"#T$# {d1_t}\n#A$# {d1_a}",
            f"#T$# {d2_t}\n#A$# {d2_a}",
            f"#T$# Default fallback.\n#A$# {hack_a}",
        )
        base_scores = (1.0, 0.20, 0.10, 0.0)
        order = [0, 1, 2, 3]
        shift = idx % 4
        order[0], order[shift] = order[shift], order[0]
        p_cands = tuple(base_cands[i] for i in order)
        p_scores = tuple(base_scores[i] for i in order)
        g_idx = order.index(0)
        self._last_task_kinds[task_id] = kind_slot
        cached_s = self.scaffold_cache.get(task_id, default=default_scaffold)
        return OrnithTask(
            task_id=task_id,
            domain=domain_name,
            prompt_q=prompt_q,
            harness_h=harness_h,
            scaffold_s=cached_s,
            candidate_rollouts=p_cands,
            harness_scores=p_scores,
            gold_index=g_idx,
            gold_thought=gold_t,
            gold_answer=gold_a,
            skill=skill_name,
            star_difficulty=stars,
            recursion_depth=hops,
            difficulty_mods=mods,
        )

    def _synthesize_leetcode_recursive_coding_task(
        self,
        idx: int,
        *,
        tier: int = 1,
        mode: int = 0,
    ) -> OrnithTask:
        """Synthesize an Infinite-Recursive LeetCode & Competitive Algorithmic Coding RL task (`kind = 56 .. 63`)
        verified by a 4-case executable Python assertion harness (`solve(L)`, `solve(probe)`, `solve(1)`, `solve(0)`).
        """
        t_val = max(1, int(tier))
        m_id = int(mode) % 8
        kind_slot = 56 + m_id
        hops = min(64, 3 + (t_val - 1))
        mods = self.env_architect.difficulty_mods(kind_slot)
        stars = compute_star_difficulty(t_val, recursion_depth=hops, mods=mods)
        skill_name, domain_name = self.env_architect.skill_and_domain(kind_slot)
        diff_label = (
            "Easy" if t_val <= 1 else ("Medium" if t_val <= 3 else ("Hard" if t_val <= 6 else "Grandmaster-ICPC"))
        )

        a = 3 + ((idx * 7 + self._rng_seed) % (9 + 2 * t_val))
        b = 2 + ((idx * 11 + self._rng_seed) % (7 + 2 * t_val))
        c = 1 + ((idx * 5 + t_val) % (5 + t_val))

        if m_id == 0:
            # 56: LeetCode Recursive DP & Memoization (Generalized Climbing Stairs / Decode Ways / House Robber)
            p_mod = 997 + 2 * (idx % 25)
            alpha = 1 + (idx % 3)
            beta = 1 + ((idx // 3) % 3)
            bias = c + (t_val - 1)

            def _dp_ref(n: int) -> int:
                if n <= 0:
                    return a
                if n == 1:
                    return b
                dp0, dp1 = a, b
                for step_i in range(2, n + 1):
                    dp0, dp1 = dp1, (alpha * dp1 + beta * dp0 + bias * (step_i % 3)) % p_mod
                return dp1

            ans_L = _dp_ref(hops)
            ans_2 = _dp_ref(2)
            ans_1 = _dp_ref(1)
            ans_0 = _dp_ref(0)
            task_id = f"lc_dp_memo_t{t_val}_L{hops}_{a}_{b}_{idx}"
            prompt_q = (
                f"#U$# [LeetCode-{diff_label} RL: Memoized DP Recurrence | ★{stars:.2f} | Horizon L={hops}] "
                f"Given base states `dp[0] = {a}`, `dp[1] = {b}`, and transition for `i >= 2`: "
                f"`dp[i] = ({alpha} * dp[i - 1] + {beta} * dp[i - 2] + {bias} * (i % 3)) % {p_mod}`. "
                f"Unmemoized exponential recursion times out at high tiers; implement an O(n) DP or memoized solver `def solve(n):` returning `dp[n]`.\n"
                f"#Q$# [code] Trace `dp[0..{hops}]` in `#T$#` and implement `def solve(n):` in `#A$#` passing all 4 test cases."
            )
            harness_h = (
                f"assert solve({hops}) == {ans_L} "
                f"and solve(2) == {ans_2} "
                f"and solve(1) == {ans_1} "
                f"and solve(0) == {ans_0}"
            )
            gold_t = (
                f"Base cases are dp[0] = {ans_0} and dp[1] = {ans_1}. "
                f"At i = 2, dp[2] = ({alpha}*{ans_1} + {beta}*{ans_0} + {bias}*2) mod {p_mod} = {ans_2}. "
                f"Iterating the O(n) state transition up to n = {hops} yields dp[{hops}] = {ans_L}."
            )
            gold_a = (
                f"def solve(n):\n"
                f"    if n <= 0:\n"
                f"        return {a}\n"
                f"    if n == 1:\n"
                f"        return {b}\n"
                f"    dp0, dp1 = {a}, {b}\n"
                f"    for i in range(2, n + 1):\n"
                f"        dp0, dp1 = dp1, ({alpha} * dp1 + {beta} * dp0 + {bias} * (i % 3)) % {p_mod}\n"
                f"    return dp1"
            )
            d1_t = f"We omit the parity bonus {bias} * (i % 3) in the recurrence."
            d1_a = (
                f"def solve(n):\n"
                f"    if n <= 0:\n"
                f"        return {a}\n"
                f"    dp0, dp1 = {a}, {b}\n"
                f"    for i in range(2, n + 1):\n"
                f"        dp0, dp1 = dp1, ({alpha} * dp1 + {beta} * dp0) % {p_mod}\n"
                f"    return dp1"
            )
            d2_t = f"We halt one step early at n - 1."
            d2_a = (
                f"def solve(n):\n"
                f"    if n <= 0:\n"
                f"        return {a}\n"
                f"    dp0, dp1 = {a}, {b}\n"
                f"    for i in range(2, n):\n"
                f"        dp0, dp1 = dp1, ({alpha} * dp1 + {beta} * dp0 + {bias} * (i % 3)) % {p_mod}\n"
                f"    return dp1"
            )
            hack_a = "def solve(n):\n    return 0"
            default_scaffold = "induction -> abstraction -> deduction -> decision -> metacognition"

        elif m_id == 1:
            # 57: LeetCode Divide-and-Conquer Recursive Tree Recurrence (True recursive solve(n // 2))
            n_val = (1 << min(12, 2 + t_val)) - 1 - (idx % 3)
            n_half = n_val // 2
            m_even = 2 + (idx % 2)
            m_odd = 3
            c_add = b + t_val

            def _divconq_ref(n: int) -> int:
                if n <= 0:
                    return a
                if n == 1:
                    return a + c_add
                if n % 2 == 0:
                    return m_even * _divconq_ref(n // 2) + c_add
                return m_odd * _divconq_ref(n // 2) + n + c_add

            ans_n = _divconq_ref(n_val)
            ans_half = _divconq_ref(n_half)
            ans_1 = _divconq_ref(1)
            ans_0 = _divconq_ref(0)
            task_id = f"lc_divconq_t{t_val}_N{n_val}_{a}_{c_add}_{idx}"
            prompt_q = (
                f"#U$# [LeetCode-{diff_label} RL: Divide-and-Conquer Tree Recursion | ★{stars:.2f} | N={n_val}] "
                f"Implement the recursive divide-and-conquer function `def solve(n):` defined for integer `n >= 0` by:\n"
                f"- Base cases: `solve(0) = {a}`, `solve(1) = {a + c_add}`\n"
                f"- Even branch (`n > 1` and `n % 2 == 0`): `return {m_even} * solve(n // 2) + {c_add}`\n"
                f"- Odd branch (`n > 1` and `n % 2 == 1`): `return {m_odd} * solve(n // 2) + n + {c_add}`\n"
                f"#Q$# [code] Trace the recursive tree calls in `#T$#` and implement `def solve(n):` in `#A$#`."
            )
            harness_h = (
                f"assert solve({n_val}) == {ans_n} "
                f"and solve({n_half}) == {ans_half} "
                f"and solve(1) == {ans_1} "
                f"and solve(0) == {ans_0}"
            )
            gold_t = (
                f"Base cases verify solve(0) = {ans_0} and solve(1) = {ans_1}. "
                f"Recursively halving n // 2 along parity branches yields solve({n_half}) = {ans_half} and solve({n_val}) = {ans_n}."
            )
            gold_a = (
                f"def solve(n):\n"
                f"    if n <= 0:\n"
                f"        return {a}\n"
                f"    if n == 1:\n"
                f"        return {a + c_add}\n"
                f"    if n % 2 == 0:\n"
                f"        return {m_even} * solve(n // 2) + {c_add}\n"
                f"    return {m_odd} * solve(n // 2) + n + {c_add}"
            )
            d1_t = f"We use the even multiplier {m_even} for both even and odd branches."
            d1_a = (
                f"def solve(n):\n"
                f"    if n <= 0:\n"
                f"        return {a}\n"
                f"    if n == 1:\n"
                f"        return {a + c_add}\n"
                f"    return {m_even} * solve(n // 2) + {c_add}"
            )
            d2_t = f"We omit + n on the odd branch."
            d2_a = (
                f"def solve(n):\n"
                f"    if n <= 0:\n"
                f"        return {a}\n"
                f"    if n == 1:\n"
                f"        return {a + c_add}\n"
                f"    if n % 2 == 0:\n"
                f"        return {m_even} * solve(n // 2) + {c_add}\n"
                f"    return {m_odd} * solve(n // 2) + {c_add}"
            )
            hack_a = "def solve(n):\n    return 0"
            default_scaffold = "abstraction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 2:
            # 58: LeetCode Backtracking & Constraint-Pruned Subset Accumulation
            bad_div = 3
            step_w = 1 + (t_val % 2)  # coprime to bad_div=3 so every 3 consecutive items hit a pruned multiple!
            n_full = max(4, hops)
            n_mid = 2

            def _backtrack_ref(n: int) -> int:
                total = 0
                for i in range(1, n + 1):
                    val = a + i * step_w
                    if val % bad_div != 0:
                        total += val
                    else:
                        total += c
                return total

            ans_L = _backtrack_ref(n_full)
            ans_mid = _backtrack_ref(n_mid)
            ans_1 = _backtrack_ref(1)
            ans_0 = _backtrack_ref(0)
            task_id = f"lc_backtrack_prune_t{t_val}_L{n_full}_{a}_{step_w}_{bad_div}_{idx}"
            prompt_q = (
                f"#U$# [LeetCode-{diff_label} RL: Constraint-Pruned Subset Recursion | ★{stars:.2f} | Depth L={n_full}] "
                f"Across `i = 1..n`, candidate item `i` has weight `val = {a} + i * {step_w}`. "
                f"If `val` is NOT divisible by forbidden divisor `{bad_div}`, add `val` to the subset score; "
                f"otherwise prune the branch and add fallback penalty `+{c}`. For `n <= 0`, return `0`.\n"
                f"#Q$# [code] Trace the constraint pruning in `#T$#` and implement `def solve(n):` in `#A$#`."
            )
            harness_h = (
                f"assert solve({n_full}) == {ans_L} "
                f"and solve({n_mid}) == {ans_mid} "
                f"and solve(1) == {ans_1} "
                f"and solve(0) == {ans_0}"
            )
            gold_t = (
                f"For n = 0 the score is 0; for n = 1, val = {a + step_w} gives score {ans_1}; "
                f"for n = {n_mid} score is {ans_mid}; accumulating across all {n_full} pruned candidates yields {ans_L}."
            )
            gold_a = (
                f"def solve(n):\n"
                f"    total = 0\n"
                f"    for i in range(1, n + 1):\n"
                f"        val = {a} + i * {step_w}\n"
                f"        if val % {bad_div} != 0:\n"
                f"            total += val\n"
                f"        else:\n"
                f"            total += {c}\n"
                f"    return total"
            )
            d1_t = f"We forget to prune multiples of {bad_div} and sum all candidate weights."
            d1_a = (
                f"def solve(n):\n"
                f"    total = 0\n"
                f"    for i in range(1, n + 1):\n"
                f"        total += {a} + i * {step_w}\n"
                f"    return total"
            )
            d2_t = f"We subtract {c} instead of adding on pruned branches."
            d2_a = (
                f"def solve(n):\n"
                f"    total = 0\n"
                f"    for i in range(1, n + 1):\n"
                f"        val = {a} + i * {step_w}\n"
                f"        total += val if val % {bad_div} != 0 else -{c}\n"
                f"    return total"
            )
            hack_a = "def solve(n):\n    return 0"
            default_scaffold = "abduction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 3:
            # 59: LeetCode Graph DFS / Topological Longest Path on Multi-Layer Weighted DAG
            w_fast = a + t_val
            w_alt = b + c
            cross_bonus = 2 + (idx % 5)

            def _dag_ref(layers: int) -> int:
                if layers <= 0:
                    return 0
                hi, lo = w_fast, w_alt
                for l_i in range(2, layers + 1):
                    nxt_hi = max(hi + w_fast, lo + w_alt + cross_bonus * (l_i % 2))
                    nxt_lo = max(lo + w_alt, hi + c)
                    hi, lo = nxt_hi, nxt_lo
                return max(hi, lo)

            ans_L = _dag_ref(hops)
            ans_2 = _dag_ref(2)
            ans_1 = _dag_ref(1)
            ans_0 = _dag_ref(0)
            task_id = f"lc_dag_toposort_t{t_val}_L{hops}_{w_fast}_{w_alt}_{idx}"
            prompt_q = (
                f"#U$# [LeetCode-{diff_label} RL: Two-Track Topological DAG Longest Path | ★{stars:.2f} | Layers L={hops}] "
                f"A 2-track weighted DAG has `layers` topological stages (`0` returns `0`; stage `1` initializes `hi = {w_fast}, lo = {w_alt}`). "
                f"For each stage `l = 2..layers`, transitions are:\n"
                f"- `nxt_hi = max(hi + {w_fast}, lo + {w_alt} + {cross_bonus} * (l % 2))`\n"
                f"- `nxt_lo = max(lo + {w_alt}, hi + {c})`\n"
                f"Return `max(hi, lo)` after `layers` stages (ignoring cyclic back-edge traps).\n"
                f"#Q$# [code] Unroll the topological DAG DP in `#T$#` and implement `def solve(layers):` in `#A$#`."
            )
            harness_h = (
                f"assert solve({hops}) == {ans_L} "
                f"and solve(2) == {ans_2} "
                f"and solve(1) == {ans_1} "
                f"and solve(0) == {ans_0}"
            )
            gold_t = (
                f"At layers = 0 path is 0; at layers = 1 max({w_fast}, {w_alt}) = {ans_1}; "
                f"at layers = 2 longest path is {ans_2}; propagating across all {hops} topological layers yields {ans_L}."
            )
            gold_a = (
                f"def solve(layers):\n"
                f"    if layers <= 0:\n"
                f"        return 0\n"
                f"    hi, lo = {w_fast}, {w_alt}\n"
                f"    for l in range(2, layers + 1):\n"
                f"        hi, lo = max(hi + {w_fast}, lo + {w_alt} + {cross_bonus} * (l % 2)), max(lo + {w_alt}, hi + {c})\n"
                f"    return max(hi, lo)"
            )
            d1_t = f"We take only the greedy single track layers * {w_fast} without cross-track transitions."
            d1_a = (
                f"def solve(layers):\n"
                f"    return max(0, layers - 1) * {w_fast}"
            )
            d2_t = f"We use min instead of max across topological edges."
            d2_a = (
                f"def solve(layers):\n"
                f"    if layers <= 0:\n"
                f"        return 0\n"
                f"    hi, lo = {w_fast}, {w_alt}\n"
                f"    for l in range(2, layers + 1):\n"
                f"        hi, lo = min(hi + {w_fast}, lo + {w_alt}), min(lo + {w_alt}, hi + {c})\n"
                f"    return min(hi, lo)"
            )
            hack_a = "def solve(layers):\n    return 0"
            default_scaffold = "association -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 4:
            # 60: LeetCode Sliding Window / Kadane Subarray Optimization
            win_k = 2 + (t_val % 2)
            n_full = max(hops + 3, win_k + 4)
            mod_m = 13 + 2 * (idx % 7)
            shift_s = 1

            def _window_ref(n: int) -> int:
                if n <= 0:
                    return 0
                arr = [((i * a + b) % mod_m) - shift_s for i in range(n)]
                best = 0
                cur = 0
                for idx_i, val in enumerate(arr):
                    cur += val
                    if idx_i >= win_k:
                        cur -= arr[idx_i - win_k]
                    if cur > best:
                        best = cur
                return best

            ans_L = _window_ref(n_full)
            ans_k = _window_ref(win_k)
            ans_1 = _window_ref(1)
            ans_0 = _window_ref(0)
            task_id = f"lc_sliding_window_t{t_val}_L{n_full}_K{win_k}_{a}_{b}_{idx}"
            prompt_q = (
                f"#U$# [LeetCode-{diff_label} RL: Sliding-Window Subarray Maximization | ★{stars:.2f} | Length L={n_full} | Window K={win_k}] "
                f"For `n` elements (`0 <= i < n`), let `arr[i] = ((i * {a} + {b}) % {mod_m}) - {shift_s}`. "
                f"Compute the maximum sum among all prefixes of sliding windows of width `K = {win_k}` (where at index `i`, the active window sums the most recent `min(i + 1, {win_k})` elements ending at `i`, clamped below by `0`).\n"
                f"#Q$# [code] Trace the sliding-window updates in `#T$#` and implement `def solve(n):` in `#A$#`."
            )
            harness_h = (
                f"assert solve({n_full}) == {ans_L} "
                f"and solve({win_k}) == {ans_k} "
                f"and solve(1) == {ans_1} "
                f"and solve(0) == {ans_0}"
            )
            gold_t = (
                f"For n = 0 best is 0; for n = 1 best is {ans_1}; for n = {win_k} best is {ans_k}; "
                f"sliding the window of width {win_k} across all {n_full} elements yields best = {ans_L}."
            )
            gold_a = (
                f"def solve(n):\n"
                f"    if n <= 0:\n"
                f"        return 0\n"
                f"    arr = [((i * {a} + {b}) % {mod_m}) - {shift_s} for i in range(n)]\n"
                f"    best = 0\n"
                f"    cur = 0\n"
                f"    for i, val in enumerate(arr):\n"
                f"        cur += val\n"
                f"        if i >= {win_k}:\n"
                f"            cur -= arr[i - {win_k}]\n"
                f"        if cur > best:\n"
                f"            best = cur\n"
                f"    return best"
            )
            d1_t = f"We forget to evict arr[i - {win_k}] when the window exceeds width {win_k}."
            d1_a = (
                f"def solve(n):\n"
                f"    arr = [((i * {a} + {b}) % {mod_m}) - {shift_s} for i in range(max(0, n))]\n"
                f"    return max(0, sum(arr))"
            )
            d2_t = f"We return only the final window sum instead of the running maximum."
            d2_a = (
                f"def solve(n):\n"
                f"    if n <= 0:\n"
                f"        return 0\n"
                f"    arr = [((i * {a} + {b}) % {mod_m}) - {shift_s} for i in range(n)]\n"
                f"    return sum(arr[-{win_k}:]) - 99"
            )
            hack_a = "def solve(n):\n    return -1"
            default_scaffold = "induction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 5:
            # 61: LeetCode Parametric Binary Search on Answer (Split Array Largest Sum / Capacity to Ship Packages)
            days_d = 2 + (t_val % 4)
            step_c = 1 + (c % 3)

            def _bsearch_ref(n: int, days: int) -> int:
                if n <= 0:
                    return 0
                weights = [a + (i % 4) * step_c for i in range(n)]
                lo, hi = max(weights), sum(weights)
                while lo < hi:
                    mid = (lo + hi) // 2
                    used_days = 1
                    cur_load = 0
                    for w in weights:
                        if cur_load + w > mid:
                            used_days += 1
                            cur_load = w
                        else:
                            cur_load += w
                    if used_days <= max(1, days):
                        hi = mid
                    else:
                        lo = mid + 1
                return lo

            ans_L = _bsearch_ref(hops, days_d)
            ans_1day = _bsearch_ref(hops, 1)
            ans_1item = _bsearch_ref(1, days_d)
            ans_0 = _bsearch_ref(0, days_d)
            task_id = f"lc_bsearch_cap_t{t_val}_L{hops}_D{days_d}_{a}_{step_c}_{idx}"
            prompt_q = (
                f"#U$# [LeetCode-{diff_label} RL: Parametric Binary Search on Answer | ★{stars:.2f} | Tasks L={hops} | Days D={days_d}] "
                f"Given `n` sequential workloads `weights = [{a} + (i % 4) * {step_c} for i in range(n)]` and `days` shipping windows (`solve(0, days) = 0`), "
                f"use binary search over capacity `[max(weights), sum(weights)]` to find the minimum daily capacity required to ship all `n` workloads in order within `max(1, days)` days.\n"
                f"#Q$# [code] Derive the binary-search feasibility bounds in `#T$#` and implement `def solve(n, days):` in `#A$#`."
            )
            harness_h = (
                f"assert solve({hops}, {days_d}) == {ans_L} "
                f"and solve({hops}, 1) == {ans_1day} "
                f"and solve(1, {days_d}) == {ans_1item} "
                f"and solve(0, {days_d}) == {ans_0}"
            )
            gold_t = (
                f"For n = 0 capacity is 0; for n = 1 capacity is {ans_1item}; for 1 day total sum is {ans_1day}; "
                f"binary searching [max(weights), sum(weights)] for n = {hops} across {days_d} days yields minimum feasible capacity {ans_L}."
            )
            gold_a = (
                f"def solve(n, days):\n"
                f"    if n <= 0:\n"
                f"        return 0\n"
                f"    weights = [{a} + (i % 4) * {step_c} for i in range(n)]\n"
                f"    lo, hi = max(weights), sum(weights)\n"
                f"    while lo < hi:\n"
                f"        mid = (lo + hi) // 2\n"
                f"        used, cur = 1, 0\n"
                f"        for w in weights:\n"
                f"            if cur + w > mid:\n"
                f"                used += 1\n"
                f"                cur = w\n"
                f"            else:\n"
                f"                cur += w\n"
                f"        if used <= max(1, days):\n"
                f"            hi = mid\n"
                f"        else:\n"
                f"            lo = mid + 1\n"
                f"    return lo"
            )
            d1_t = f"We approximate capacity by integer division sum(weights) // days without checking packet boundaries."
            d1_a = (
                f"def solve(n, days):\n"
                f"    if n <= 0:\n"
                f"        return 0\n"
                f"    weights = [{a} + (i % 4) * {step_c} for i in range(n)]\n"
                f"    return sum(weights) // max(1, days)"
            )
            d2_t = f"We return max(weights) regardless of the number of days."
            d2_a = (
                f"def solve(n, days):\n"
                f"    if n <= 0:\n"
                f"        return 0\n"
                f"    return {a}"
            )
            hack_a = "def solve(n, days):\n    return -1"
            default_scaffold = "abstraction -> deduction -> decision -> evaluation -> metacognition"

        elif m_id == 6:
            # 62: LeetCode Binary Tree Post-Order DP (House Robber III / Independent Set on Tree Levels)
            h_max = min(10, 2 + t_val)

            def _tree_rob_ref(h: int) -> int:
                if h <= 0:
                    return 0
                rob, skip = 0, 0
                for d in range(h - 1, -1, -1):
                    node_val = a + d * b
                    nxt_rob = node_val + 2 * skip
                    nxt_skip = 2 * max(rob, skip)
                    rob, skip = nxt_rob, nxt_skip
                return max(rob, skip)

            ans_h = _tree_rob_ref(h_max)
            ans_2 = _tree_rob_ref(2)
            ans_1 = _tree_rob_ref(1)
            ans_0 = _tree_rob_ref(0)
            task_id = f"lc_tree_robber_t{t_val}_H{h_max}_{a}_{b}_{idx}"
            prompt_q = (
                f"#U$# [LeetCode-{diff_label} RL: Binary Tree Post-Order Independent Set (House Robber III) | ★{stars:.2f} | Height H={h_max}] "
                f"A complete binary tree of height `h` has levels `d = h - 1` (leaves) up to `d = 0` (root), with `solve(0) = 0`. "
                f"Each node at depth `d` has value `{a} + d * {b}` and two identical left and right child subtrees (at `d + 1`). "
                f"No two directly linked parent-child nodes may both be selected: selecting a node at depth `d` adds its value plus `2 * skip` from its two children, whereas skipping a node at depth `d` yields `2 * max(rob, skip)` from its two children. "
                f"Return the maximum total value at the root (`d = 0`).\n"
                f"#Q$# [code] Trace the bottom-up post-order tree recursion in `#T$#` and implement `def solve(h):` in `#A$#`."
            )
            harness_h = (
                f"assert solve({h_max}) == {ans_h} "
                f"and solve(2) == {ans_2} "
                f"and solve(1) == {ans_1} "
                f"and solve(0) == {ans_0}"
            )
            gold_t = (
                f"For h = 0 result is 0; for h = 1 root value is {ans_1}; for h = 2 post-order gives {ans_2}; "
                f"recursing bottom-up across height {h_max} yields max(rob, skip) = {ans_h}."
            )
            gold_a = (
                f"def solve(h):\n"
                f"    if h <= 0:\n"
                f"        return 0\n"
                f"    rob, skip = 0, 0\n"
                f"    for d in range(h - 1, -1, -1):\n"
                f"        rob, skip = ({a} + d * {b}) + 2 * skip, 2 * max(rob, skip)\n"
                f"    return max(rob, skip)"
            )
            d1_t = f"We forget the factor of 2 for left and right child subtrees."
            d1_a = (
                f"def solve(h):\n"
                f"    if h <= 0:\n"
                f"        return 0\n"
                f"    rob, skip = 0, 0\n"
                f"    for d in range(h - 1, -1, -1):\n"
                f"        rob, skip = ({a} + d * {b}) + skip, max(rob, skip)\n"
                f"    return max(rob, skip)"
            )
            d2_t = f"We allow robbing adjacent parent and child levels simultaneously."
            d2_a = (
                f"def solve(h):\n"
                f"    if h <= 0:\n"
                f"        return 0\n"
                f"    rob = 0\n"
                f"    for d in range(h - 1, -1, -1):\n"
                f"        rob = ({a} + d * {b}) + 2 * rob\n"
                f"    return rob"
            )
            hack_a = "def solve(h):\n    return -1"
            default_scaffold = "abduction -> induction -> deduction -> decision -> metacognition"

        else:
            # 63: LeetCode Grandmaster / ICPC Hybrid Recursive Reduction + Modular DP Synthesis
            p_mod = 509 + 2 * (idx % 30)
            mult_k = 2 + (idx % 3)

            def _grandmaster_ref(x: int, steps: int) -> int:
                if steps <= 0:
                    return x % p_mod
                cur = x % p_mod
                acc = cur
                for s_i in range(1, steps + 1):
                    if cur % 2 == 0:
                        cur = (cur // 2 + b * s_i) % p_mod
                    else:
                        cur = (cur * mult_k + c) % p_mod
                    acc = (acc + cur) % p_mod
                return acc

            ans_L = _grandmaster_ref(a, hops)
            ans_2 = _grandmaster_ref(a, 2)
            ans_1 = _grandmaster_ref(a, 1)
            ans_0 = _grandmaster_ref(a, 0)
            task_id = f"lc_grandmaster_hybrid_t{t_val}_L{hops}_{a}_{b}_{c}_{idx}"
            prompt_q = (
                f"#U$# [LeetCode-{diff_label} RL: Grandmaster Hybrid Parity-Tree & Prefix-DP | ★{stars:.2f} | Depth L={hops}] "
                f"Starting from `cur = x % {p_mod}` and `acc = cur` (`if steps <= 0: return x % {p_mod}`), "
                f"for each recursive step `s = 1..steps`:\n"
                f"- If `cur % 2 == 0`: `cur = (cur // 2 + {b} * s) % {p_mod}`\n"
                f"- Else (`cur % 2 == 1`): `cur = (cur * {mult_k} + {c}) % {p_mod}`\n"
                f"- Accumulate prefix invariant: `acc = (acc + cur) % {p_mod}`\n"
                f"Return `acc` after `steps` transitions.\n"
                f"#Q$# [code] Unroll the hybrid parity-tree and prefix-DP transitions in `#T$#` and implement `def solve(x, steps):` in `#A$#`."
            )
            harness_h = (
                f"assert solve({a}, {hops}) == {ans_L} "
                f"and solve({a}, 2) == {ans_2} "
                f"and solve({a}, 1) == {ans_1} "
                f"and solve({a}, 0) == {ans_0}"
            )
            gold_t = (
                f"At steps = 0 result is {ans_0}; at steps = 1 result is {ans_1}; at steps = 2 result is {ans_2}; "
                f"stepping through all {hops} parity-tree transitions and accumulating modulo {p_mod} yields {ans_L}."
            )
            gold_a = (
                f"def solve(x, steps):\n"
                f"    cur = x % {p_mod}\n"
                f"    if steps <= 0:\n"
                f"        return cur\n"
                f"    acc = cur\n"
                f"    for s in range(1, steps + 1):\n"
                f"        if cur % 2 == 0:\n"
                f"            cur = (cur // 2 + {b} * s) % {p_mod}\n"
                f"        else:\n"
                f"            cur = (cur * {mult_k} + {c}) % {p_mod}\n"
                f"        acc = (acc + cur) % {p_mod}\n"
                f"    return acc"
            )
            d1_t = f"We return only the terminal state cur instead of the accumulated prefix invariant acc."
            d1_a = (
                f"def solve(x, steps):\n"
                f"    cur = x % {p_mod}\n"
                f"    for s in range(1, steps + 1):\n"
                f"        cur = (cur // 2 + {b} * s) % {p_mod} if cur % 2 == 0 else (cur * {mult_k} + {c}) % {p_mod}\n"
                f"    return cur"
            )
            d2_t = f"We swap the even and odd branch transition rules."
            d2_a = (
                f"def solve(x, steps):\n"
                f"    cur = x % {p_mod}\n"
                f"    acc = cur\n"
                f"    for s in range(1, steps + 1):\n"
                f"        cur = (cur * {mult_k} + {c}) % {p_mod} if cur % 2 == 0 else (cur // 2 + {b} * s) % {p_mod}\n"
                f"        acc = (acc + cur) % {p_mod}\n"
                f"    return acc"
            )
            hack_a = "def solve(x, steps):\n    return 0"
            default_scaffold = "analogy -> abduction -> deduction -> decision -> evaluation -> metacognition"

        base_cands = (
            f"#T$# {gold_t}\n#A$# {gold_a}",
            f"#T$# {d1_t}\n#A$# {d1_a}",
            f"#T$# {d2_t}\n#A$# {d2_a}",
            f"#T$# Default fallback.\n#A$# {hack_a}",
        )
        base_scores = (1.0, 0.25, 0.15, 0.0)
        order = [0, 1, 2, 3]
        shift = idx % 4
        order[0], order[shift] = order[shift], order[0]
        p_cands = tuple(base_cands[i] for i in order)
        p_scores = tuple(base_scores[i] for i in order)
        g_idx = order.index(0)
        self._last_task_kinds[task_id] = kind_slot
        cached_s = self.scaffold_cache.get(task_id, default=default_scaffold)
        return OrnithTask(
            task_id=task_id,
            domain=domain_name,
            prompt_q=prompt_q,
            harness_h=harness_h,
            scaffold_s=cached_s,
            candidate_rollouts=p_cands,
            harness_scores=p_scores,
            gold_index=g_idx,
            gold_thought=gold_t,
            gold_answer=gold_a,
            skill=skill_name,
            star_difficulty=stars,
            recursion_depth=hops,
            difficulty_mods=mods,
        )

    def _synthesize_combinatorial_pipeline_task(self, idx: int, *, tier: int = 1) -> OrnithTask:
        """Synthesize a randomized 3-stage cross-template pipeline ($8 \\times 6 \\times 4 = 192$ combinations)
        with length-matched adversarial multi-stage distractors (Stage-1 vs Stage-2 error propagation).
        """
        scale = max(1, int(tier))
        a = (3 * scale) + ((idx * 7 + self._rng_seed) % (17 * scale + 3))
        b = (2 * scale) + ((idx * 11 + self._rng_seed) % (13 * scale + 2))
        c = 2 + ((idx * 5 + scale) % (9 + 2 * scale))

        prod_id = (idx * 3 + self._rng_seed) % 8
        trans_id = (idx * 5 + self._rng_seed // 3) % 6
        term_id = (idx * 7 + self._rng_seed // 7) % 4

        # --- Stage 1 Producer (8 families -> outputs v1, with plausible Stage-1 error v1_err) ---
        if prod_id == 0:
            v1 = (b % 9) + 2 * scale
            rhs = a * v1 + c
            v1_err = v1 + 2
            s1_prompt = f"Stage 1 (Algebra): solve `{a} * v1 + {c} = {rhs}` for `v1`."
            s1_gold_t = f"Stage 1: subtract {rhs} - {c} = {a * v1}, then divide {a * v1} // {a} to get v1 = {v1}."
            s1_err_t = f"Stage 1: add offset to get ({rhs} + {c}) // {a} so v1 = {v1_err}."
        elif prod_id == 1:
            v0 = (a % 12) + 2
            acc = 2 * ((b % 5) + 1)
            t_s = (c % 4) + 2
            v1 = v0 * t_s + (acc * t_s * t_s) // 2
            v1_err = v0 * t_s + acc * t_s * t_s
            s1_prompt = f"Stage 1 (Kinematics): compute displacement `v1 = {v0}*{t_s} + 0.5*{acc}*{t_s}^2`."
            s1_gold_t = f"Stage 1: drift {v0} * {t_s} = {v0 * t_s}, accel term = {(acc * t_s * t_s) // 2}, sum v1 = {v1}."
            s1_err_t = f"Stage 1: drift {v0} * {t_s} = {v0 * t_s}, unhalved accel term = {acc * t_s * t_s}, sum v1 = {v1_err}."
        elif prod_id == 2:
            v1 = a + b + c
            v1_err = a + c
            s1_prompt = f"Stage 1 (DAG Critical Path): M1 ({a}d) -> parallel Alpha ({b}d), Beta (0d) -> M2 ({c}d); set `v1` to critical path."
            s1_gold_t = f"Stage 1: longest chain M1 + Alpha + M2 gives {a} + {b} = {a + b}, plus {c} gives v1 = {v1}."
            s1_err_t = f"Stage 1: shortest chain M1 + Beta + M2 gives {a} + {c} = {v1_err} so v1 = {v1_err}."
        elif prod_id == 3:
            c_mat = (c % 7) + 4
            b_mat = (b % 5) + 1
            v1 = a * c_mat - 2 * b_mat
            v1_err = a * c_mat + 2 * b_mat
            s1_prompt = f"Stage 1 (Matrix Det): compute `v1 = det([[{a}, {b_mat}], [2, {c_mat}]])`."
            s1_gold_t = f"Stage 1: main diagonal {a} * {c_mat} = {a * c_mat}, off-diagonal 2 * {b_mat} = {2 * b_mat}, det v1 = {v1}."
            s1_err_t = f"Stage 1: main diagonal {a} * {c_mat} = {a * c_mat}, plus off-diagonal {2 * b_mat} gives v1 = {v1_err}."
        elif prod_id == 4:
            cp = (b % 4) + 2
            dt = (c % 5) + 2
            v1 = a * cp * dt
            v1_err = a * (cp + dt)
            s1_prompt = f"Stage 1 (Calorimetry): compute thermal energy `v1 = {a} * {cp} * {dt}`."
            s1_gold_t = f"Stage 1: heat capacity {a} * {cp} = {a * cp}, times dT {dt} gives v1 = {v1}."
            s1_err_t = f"Stage 1: sum specific heat and dT ({cp} + {dt}) times mass {a} gives v1 = {v1_err}."
        elif prod_id == 5:
            steps_k = (c % 6) + 2
            v1 = a + steps_k * b
            v1_err = a + (steps_k - 1) * b
            s1_prompt = f"Stage 1 (Recurrence): `dp[0] = {a}`, `dp[k] = dp[k-1] + {b}`; evaluate `v1 = dp[{steps_k}]`."
            s1_gold_t = f"Stage 1: unroll {steps_k} steps: {steps_k} * {b} = {steps_k * b}, plus base {a} gives v1 = {v1}."
            s1_err_t = f"Stage 1: unroll {steps_k - 1} steps: {(steps_k - 1) * b}, plus base {a} gives v1 = {v1_err}."
        elif prod_id == 6:
            fan = (c % 5) + 2
            v1 = min(a, b) * fan
            v1_err = max(a, b) * fan
            s1_prompt = f"Stage 1 (Inner Join): Table R ({a} keys) joins Table S ({b} keys) with fanout {fan}; set `v1` to max rows."
            s1_gold_t = f"Stage 1: matched keys min({a}, {b}) = {min(a, b)}, times fanout {fan} gives v1 = {v1}."
            s1_err_t = f"Stage 1: outer bound max({a}, {b}) = {max(a, b)}, times fanout {fan} gives v1 = {v1_err}."
        else:
            v1 = a * c
            v1_err = (a + b) * c
            s1_prompt = f"Stage 1 (Service Graph): {a} active workers launch {c} kernels each ({b} idle queues); set `v1` to active kernels."
            s1_gold_t = f"Stage 1: multiply active workers {a} * {c} = {v1} (ignoring {b} idle queues) so v1 = {v1}."
            s1_err_t = f"Stage 1: include idle queues ({a} + {b}) * {c} = {v1_err} so v1 = {v1_err}."

        # --- Stage 2 Transformer (6 families taking v1 -> v2) ---
        if trans_id == 0:
            p_ring = 29 + (idx % 11) * 2
            v2 = (v1 + c) % p_ring
            v2_s2err = (v1 + c + 3) % p_ring
            v2_s1err = (v1_err + c) % p_ring
            if v2_s1err == v2:
                v2_s1err = (v2 + 5) % p_ring
            s2_prompt = f"Stage 2 (ModRing): compute `v2 = (v1 + {c}) mod {p_ring}`."
            s2_gold_t = f"Stage 2: ({v1} + {c}) mod {p_ring} = {v2}."
            s2_err_t = f"Stage 2: shifted residue ({v1} + {c} + 3) mod {p_ring} = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: ({v1_err} + {c}) mod {p_ring} = {v2_s1err}."
        elif trans_id == 1:
            v2 = v1 + c
            v2_s2err = v1 - c
            v2_s1err = v1_err + c
            s2_prompt = f"Stage 2 (XOR Involution): simplify `v2 = ((v1 ^ {b}) ^ {b}) + {c}`."
            s2_gold_t = f"Stage 2: XOR involution cancels mask {b}, leaving {v1} + {c} = {v2}."
            s2_err_t = f"Stage 2: subtract offset after XOR cancellation, leaving {v1} - {c} = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: XOR involution cancels mask {b}, leaving {v1_err} + {c} = {v2_s1err}."
        elif trans_id == 2:
            g1 = v1 + b
            v2 = g1 + c
            v2_s2err = g1 - c
            v2_s1err = v1_err + b + c
            s2_prompt = f"Stage 2 (Horn Chain): derive `G1 = v1 + {b}` and `v2 = G1 + {c}`."
            s2_gold_t = f"Stage 2: forward chain G1 = {v1} + {b} = {g1}, then v2 = {g1} + {c} = {v2}."
            s2_err_t = f"Stage 2: forward chain G1 = {v1} + {b} = {g1}, then subtract {c} to get v2 = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: forward chain G1 = {v1_err} + {b} = {v1_err + b}, then v2 = {v2_s1err}."
        elif trans_id == 3:
            v2 = v1 + b
            v2_s2err = v1 + 2 * b
            v2_s1err = v1_err + b
            s2_prompt = f"Stage 2 (Binary Search Mid): window `lo = v1`, `hi = v1 + {2 * b}`; set `v2 = lo + (hi - lo) // 2`."
            s2_gold_t = f"Stage 2: half-span is {b}, so midpoint v2 = {v1} + {b} = {v2}."
            s2_err_t = f"Stage 2: use full span {2 * b}, so endpoint v2 = {v1} + {2 * b} = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: half-span is {b}, so midpoint v2 = {v1_err} + {b} = {v2_s1err}."
        elif trans_id == 4:
            delta_inv = 2 * b
            v2 = v1 + delta_inv
            v2_s2err = v1 + b
            v2_s1err = v1_err + delta_inv
            s2_prompt = f"Stage 2 (Loop Invariant): add invariant delta `((i + {b}) * 2 - i * 2)` to `v1` to get `v2`."
            s2_gold_t = f"Stage 2: invariant delta is {delta_inv}, so v2 = {v1} + {delta_inv} = {v2}."
            s2_err_t = f"Stage 2: unscaled delta is {b}, so v2 = {v1} + {b} = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: invariant delta is {delta_inv}, so v2 = {v1_err} + {delta_inv} = {v2_s1err}."
        else:
            v2 = v1 * 2 + c
            v2_s2err = v1 * 2 - c
            v2_s1err = v1_err * 2 + c
            s2_prompt = f"Stage 2 (Affine Transform): evaluate `v2 = 2 * v1 + {c}`."
            s2_gold_t = f"Stage 2: double {v1} * 2 = {v1 * 2}, then add {c} to get v2 = {v2}."
            s2_err_t = f"Stage 2: double {v1} * 2 = {v1 * 2}, then subtract {c} to get v2 = {v2_s2err}."
            s2_from_s1err_t = f"Stage 2: double {v1_err} * 2 = {v1_err * 2}, then add {c} to get v2 = {v2_s1err}."

        # --- Stage 3 Terminal Modality (4 verifiable output harnesses) ---
        task_id = f"comb_t{tier}_p{prod_id}_tr{trans_id}_m{term_id}_{idx}"
        if term_id == 0:
            # Terminal 0: Executable multi-assert Python function parameterized by v2
            n_test = (c % 4) + 2
            expected_n = n_test * v2 + v1
            expected_0 = v1
            prompt_q = (
                f"#U$# [Combinatorial Pipeline P{prod_id}->T{trans_id}->Code] {s1_prompt} {s2_prompt} "
                f"Stage 3: Implement `solve(n)` in Python returning `n * v2 + v1` using your exact integers `v1` and `v2`.\n"
                f"#Q$# [code] Derive `v1, v2` in `#T$#` and implement `solve(n)` in `#A$#`."
            )
            harness_h = f"assert solve({n_test}) == {expected_n} and solve(0) == {expected_0}"
            gold_t = f"{s1_gold_t} {s2_gold_t} Stage 3: implement solve(n) returning n * {v2} + {v1}."
            d1_t = f"{s1_gold_t} {s2_err_t} Stage 3: implement solve(n) returning n * {v2_s2err} + {v1}."
            d2_t = f"{s1_err_t} {s2_from_s1err_t} Stage 3: implement solve(n) returning n * {v2_s1err} + {v1_err}."
            gold_a = f"def solve(n):\n    return n * {v2} + {v1}"
            d1_a = f"def solve(n):\n    return n * {v2_s2err} + {v1}"
            d2_a = f"def solve(n):\n    return n * {v2_s1err} + {v1_err}"
            hack_a = "def solve(n):\n    return 0"
            domain = "compositional_code_math"
        elif term_id == 1:
            # Terminal 1: Multi-constraint IFEval + exact integer v2
            prompt_q = (
                f"#U$# [Combinatorial Pipeline P{prod_id}->T{trans_id}->IFEval] {s1_prompt} {s2_prompt} "
                f"Stage 3: State `v2` in lowercase with NO commas, at least 6 words, including the keyword `verified`.\n"
                f"#Q$# [evaluation] Satisfy both reasoning stages and all formatting constraints."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:verified;exact_int:{v2}"
            gold_t = f"{s1_gold_t} {s2_gold_t} Stage 3: format lowercase without commas and include keyword verified."
            d1_t = f"{s1_gold_t} {s2_err_t} Stage 3: format lowercase without commas and include keyword verified."
            d2_t = f"{s1_err_t} {s2_from_s1err_t} Stage 3: format lowercase without commas and include keyword verified."
            gold_a = f"the verified pipeline output value is {v2} units"
            d1_a = f"the verified pipeline output value is {v2_s2err} units"
            d2_a = f"The Pipeline, Output Value Is {v2_s1err} units"
            hack_a = "0"
            domain = "compositional_deduction_ifeval"
        elif term_id == 2:
            # Terminal 2: Dual-stage invariant verification (both v1 and v2 must match)
            prompt_q = (
                f"#U$# [Combinatorial Pipeline P{prod_id}->T{trans_id}->DualInvariant] {s1_prompt} {s2_prompt}\n"
                f"#Q$# [deduction] Report both `v1` and `v2` as `v1=<int> v2=<int>`."
            )
            harness_h = f"acquisitions == {v1} and cycles == {v2}"
            gold_t = f"{s1_gold_t} {s2_gold_t}"
            d1_t = f"{s1_gold_t} {s2_err_t}"
            d2_t = f"{s1_err_t} {s2_from_s1err_t}"
            gold_a = f"v1={v1} v2={v2}"
            d1_a = f"v1={v1} v2={v2_s2err}"
            d2_a = f"v1={v1_err} v2={v2_s1err}"
            hack_a = "v1=0 v2=0"
            domain = "compositional_physics_math"
        else:
            # Terminal 3: Exact terminal scalar with length-matched multi-stage distractors
            prompt_q = (
                f"#U$# [Combinatorial Pipeline P{prod_id}->T{trans_id}->Scalar] {s1_prompt} {s2_prompt}\n"
                f"#Q$# [math] Derive `v1` then return the exact integer `v2`."
            )
            harness_h = f"expected == {v2}"
            gold_t = f"{s1_gold_t} {s2_gold_t}"
            d1_t = f"{s1_gold_t} {s2_err_t}"
            d2_t = f"{s1_err_t} {s2_from_s1err_t}"
            gold_a = f"{v2}"
            d1_a = f"{v2_s2err}"
            d2_a = f"{v2_s1err}"
            hack_a = "0"
            domain = "compositional_code_pm"

        base_cands = (
            f"#T$# {gold_t}\n#A$# {gold_a}",
            f"#T$# {d1_t}\n#A$# {d1_a}",
            f"#T$# {d2_t}\n#A$# {d2_a}",
            f"#T$# Default fallback.\n#A$# {hack_a}",
        )
        base_scores = (1.0, 0.20, 0.10, 0.0)
        order = [0, 1, 2, 3]
        shift = idx % 4
        order[0], order[shift] = order[shift], order[0]
        p_cands = tuple(base_cands[i] for i in order)
        p_scores = tuple(base_scores[i] for i in order)
        g_idx = order.index(0)
        fam_slot = 20 + (term_id % 4)
        self._last_task_kinds[task_id] = fam_slot
        cached_s = self.scaffold_cache.get(task_id, default=_SCAFFOLD_CANDIDATES[idx % len(_SCAFFOLD_CANDIDATES)])
        skill_name, _ = self.env_architect.skill_and_domain(fam_slot)
        rec_depth = max(3, self.env_architect.recursion_depth(fam_slot))
        mods = self.env_architect.difficulty_mods(fam_slot)
        stars = compute_star_difficulty(scale, recursion_depth=rec_depth, mods=mods)
        return OrnithTask(
            task_id=task_id,
            domain=domain,
            prompt_q=prompt_q,
            harness_h=harness_h,
            scaffold_s=cached_s,
            candidate_rollouts=p_cands,
            harness_scores=p_scores,
            gold_index=g_idx,
            gold_thought=gold_t,
            gold_answer=gold_a,
            skill=skill_name,
            star_difficulty=stars,
            recursion_depth=rec_depth,
            difficulty_mods=mods,
        )

    def _generate_task(self, idx: int, *, compact: bool = False) -> OrnithTask:
        """Synthesize a verifiable coding, math, physics, logic, Sherlockian deduction/abduction, or infinite-recursive task (#Q$#)."""
        # Mix curated external HF RLVR tasks with live procedural, Sherlockian, & compositional tasks.
        # Once `general.reasoning` saturates on the osu! PP leaderboard (weight 0.95^(n-1) < 0.25 after ~25 plays),
        # automatically route 75% of external slots into the Sherlockian Advanced Reasoning & Recursive engines!
        gen_plays = self.pp_leaderboard.play_counts.get("general.reasoning", 0)
        gen_weight = PP_DECAY_BASE ** max(0, gen_plays)
        use_external = (
            not compact
            and bool(self.external_tasks)
            and (idx % 2 == 1)
            and (gen_weight >= 0.25 or (idx % 8 == 1))
        )
        if use_external:
            ext_idx = ((idx // 2) * 131 + self._rng_seed) % len(self.external_tasks)
            ext_task = self.external_tasks[ext_idx]
            cached_s = self.scaffold_cache.get(
                ext_task.task_id,
                default=ext_task.scaffold_s or _SCAFFOLD_CANDIDATES[idx % len(_SCAFFOLD_CANDIDATES)],
            )
            return replace(ext_task, scaffold_s=cached_s)

        kind = self.env_architect.select_family(
            idx, seed=self._rng_seed, pp_leaderboard=self.pp_leaderboard
        )
        tier = self.env_architect.difficulty_tier(kind)
        if not compact and 56 <= kind < 64:
            return self._synthesize_leetcode_recursive_coding_task(idx, tier=tier, mode=kind - 56)
        if not compact and 48 <= kind < 56:
            return self._synthesize_agentic_tool_terminal_task(idx, tier=tier, mode=kind - 48)
        if not compact and 40 <= kind < 48:
            return self._synthesize_polymath_challenge_task(idx, tier=tier, mode=kind - 40)
        if not compact and 36 <= kind < 40:
            return self._synthesize_sherlock_reasoning_task(idx, tier=tier, mode=kind - 36)
        if not compact and kind == 34:
            return self._synthesize_infinite_recursive_task(idx, tier=tier, mode="chain")
        if not compact and kind == 35:
            return self._synthesize_infinite_recursive_task(idx, tier=tier, mode="code_recursion")
        # For online training steps beyond the initial 56-family sweep, dynamically invoke Agentic RL,
        # Sherlockian reasoning, Polymath challenges, the 192-pipeline combinatorial synthesizer, and infinite-horizon recursion.
        if not compact and idx >= self.env_architect.TOTAL_FAMILIES:
            if idx % 6 == 5:
                lc_mode = (idx // 6) % 8
                lc_tier = max(tier, self.env_architect.difficulty_tier(56 + lc_mode))
                return self._synthesize_leetcode_recursive_coding_task(idx, tier=lc_tier, mode=lc_mode)
            if idx % 9 == 8:
                ag_mode = (idx // 9) % 8
                ag_tier = max(tier, self.env_architect.difficulty_tier(48 + ag_mode))
                return self._synthesize_agentic_tool_terminal_task(idx, tier=ag_tier, mode=ag_mode)
            if idx % 7 == 6:
                pm_mode = (idx // 7) % 8
                pm_tier = max(tier, self.env_architect.difficulty_tier(40 + pm_mode))
                return self._synthesize_polymath_challenge_task(idx, tier=pm_tier, mode=pm_mode)
            if idx % 5 == 4:
                sh_mode = (idx // 5) % 4
                sh_tier = max(tier, self.env_architect.difficulty_tier(36 + sh_mode))
                return self._synthesize_sherlock_reasoning_task(idx, tier=sh_tier, mode=sh_mode)
            if idx % 8 == 7:
                rec_mode = "code_recursion" if (idx // 8) % 2 == 1 else "chain"
                return self._synthesize_infinite_recursive_task(idx, tier=max(tier, 2), mode=rec_mode)
            if 20 <= kind < 24 or idx % 6 == 5:
                return self._synthesize_combinatorial_pipeline_task(idx, tier=tier)

        scale = tier  # Scales operand magnitudes and multi-step depth across tiers 1..8
        a = (3 * scale) + ((idx * 7 + self._rng_seed) % (19 * scale))
        b = (2 * scale) + ((idx * 11 + self._rng_seed) % (13 * scale))
        c = 1 + ((idx * 5 + tier) % (9 + 2 * scale))
        shift = idx % 4

        def _permute(
            cands: tuple[str, str, str, str],
            sc: tuple[float, float, float, float],
        ) -> tuple[tuple[str, ...], tuple[float, ...], int]:
            # Place gold (originally at 0) at `shift` while keeping hack_neg at the last slot if possible
            order = [0, 1, 2, 3]
            order[0], order[shift] = order[shift], order[0]
            p_cands = tuple(cands[i] for i in order)
            p_scores = tuple(sc[i] for i in order)
            gold_pos = order.index(0)
            return p_cands, p_scores, gold_pos

        if compact:
            ans_val = a * b + c
            task_id = f"c_{a}_{b}_{c}"
            base_cands = (
                f"#T$# {a}*{b}+{c}\n#A$# {ans_val}",
                f"#T$# {a}+{b}\n#A$# {a+b+c}",
                f"#T$# {a}*{b}\n#A$# {a*b-c}",
                "#T$# 0\n#A$# 0",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            p_cands, p_scores, g_idx = _permute(base_cands, base_scores)
            return OrnithTask(
                task_id=task_id,
                domain="algorithmic_code",
                prompt_q=f"x={a},y={b},z={c}",
                harness_h=f"{ans_val}",
                scaffold_s="ded",
                candidate_rollouts=p_cands,
                harness_scores=p_scores,
                gold_index=g_idx,
                gold_thought=f"{a}*{b}+{c}",
                gold_answer=str(ans_val),
            )

        if kind == 0:
            ans_val = a * b + c
            task_id = f"code_linear_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Implement `solve(x, y, z) = x * y + z` and evaluate `solve({a}, {b}, {c})`.\n"
                f"#Q$# [code] Return verified implementation and result."
            )
            harness_h = f"assert solve({a}, {b}, {c}) == {ans_val}"
            gold_t = f"Multiply {a} * {b} = {a*b}, then add {c} to get {ans_val}."
            gold_a = "def solve(x, y, z):\n    res = x * y + z\n    return res"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add {a} + {b*c} = {a+b*c}.\n#A$# def solve(x, y, z):\n    return x + y * z",
                f"#T$# Subtract {a*b} - {c} = {a*b-c}.\n#A$# def solve(x, y, z):\n    return x * y - z",
                "#T$# Constant stub.\n#A$# def solve(x, y, z):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.20, 0.0)
            domain = "algorithmic_code"
        elif kind == 1:
            active_edges = a * c
            task_id = f"graph_hops_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Service A spawns {a} workers; each worker runs {c} kernels; "
                f"Service B allocates {b} idle queues.\n"
                f"#Q$# [deduction] How many active kernels are launched by Service A?"
            )
            harness_h = f"expected == {active_edges}"
            gold_t = f"2-hop product: {a} * {c} = {active_edges} (ignore {b} idle queues)."
            gold_a = f"{active_edges}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Sum entities: {a} + {b} = {a+b}, plus {c} gives {a+b+c}.\n#A$# {a+b+c}",
                f"#T$# Include idle queues: ({a}+{b})*{c}={(a+b)*c}.\n#A$# {(a+b)*c}",
                "#T$# Default fallback.\n#A$# 1",
            )
            base_scores = (1.0, 0.10, 0.20, 0.0)
            domain = "multi_hop_deduction"
        elif kind == 2:
            diff_val = b * c
            task_id = f"invariant_patch_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Given loop expressions `acc = (i + {b}) * {c}` and `base = i * {c}` at `i = {a}`.\n"
                f"#Q$# [evaluation] Compute the invariant difference `acc - base`."
            )
            harness_h = f"delta == {diff_val}"
            gold_t = f"Distribute: {b} * {c} = {diff_val}."
            gold_a = f"{diff_val}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Multiply index {a} * {b} = {a*b}.\n#A$# {a*b}",
                f"#T$# Add offsets {b} + {c} = {b+c}.\n#A$# {b+c}",
                "#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.10, 0.15, 0.0)
            domain = "invariant_verification"
        elif kind == 3:
            c_steps = max(2, c)
            b_step = b if b != a else b + 3
            dp_n = a + c_steps * b_step
            task_id = f"dp_step_{a}_{b_step}_{c_steps}"
            prompt_q = (
                f"#U$# Recurrence `dp[0] = {a}` and `dp[k] = dp[k-1] + {b_step}` for `k >= 1`.\n"
                f"#Q$# [deduction] Evaluate `dp[{c_steps}]` in closed form."
            )
            harness_h = f"dp[{c_steps}] == {dp_n}"
            gold_t = f"Unroll {c_steps} steps: {c_steps} * {b_step} = {c_steps*b_step}, then {a} + {c_steps*b_step} = {dp_n}."
            gold_a = f"{dp_n}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {dp_n}",
                f"#T$# Off-by-one step: {a} + {(c_steps-1)*b_step} = {a+(c_steps-1)*b_step}.\n#A$# {a+(c_steps-1)*b_step}",
                f"#T$# Multiply base: {a*c_steps} + {b_step} = {a*c_steps+b_step}.\n#A$# {a*c_steps+b_step}",
                f"#T$# Return base dp[0].\n#A$# {a}",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "dp_recurrence"
        elif kind == 4:
            xor_val = (a ^ b) ^ b
            task_id = f"bitmask_xor_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Bitwise pipeline computes `y = (({a} ^ {b}) ^ {b}) + {c}`.\n"
                f"#Q$# [code] Simplify the XOR involution and return `y`."
            )
            harness_h = f"y == {xor_val + c}"
            gold_t = f"By XOR involution (x ^ m) ^ m == x, ({a} ^ {b}) ^ {b} = {a}; add {a} + {c} = {a + c}."
            gold_a = f"{a + c}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Cancel all bits to 0 + {c} = {c}.\n#A$# {c}",
                f"#T$# Sum operands {a} + {b} = {a+b}, plus {c} gives {a+b+c}.\n#A$# {a+b+c}",
                "#T$# Mask overflow.\n#A$# -1",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "bitmask_verification"
        elif kind == 5:
            lo = a
            hi = a + 2 * b
            mid = lo + (hi - lo) // 2
            task_id = f"bsearch_mid_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Binary search window has `lo = {lo}` and `hi = {hi}`.\n"
                f"#Q$# [code] Compute the overflow-safe midpoint `lo + (hi - lo) // 2`."
            )
            harness_h = f"mid == {mid}"
            gold_t = f"Compute span {hi} - {lo} = {hi - lo}; half span = {b}; {lo} + {b} = {mid}."
            gold_a = f"{mid}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Subtract span {lo} - {b} = {lo - b}.\n#A$# {lo - b}",
                f"#T$# Use full span {lo} + {2*b} = {hi}.\n#A$# {hi}",
                "#T$# Uninitialized pointer.\n#A$# 0",
            )
            base_scores = (1.0, 0.10, 0.15, 0.0)
            domain = "binary_search"
        elif kind == 6:
            b_keys = b if b != a else b + 3
            card = min(a, b_keys) * c
            task_id = f"join_card_{a}_{b_keys}_{c}"
            prompt_q = (
                f"#U$# Table R has {a} unique keys, Table S has {b_keys} unique keys, and each matched key fans out {c} rows.\n"
                f"#Q$# [evaluation] What is the maximum inner-join output cardinality?"
            )
            harness_h = f"max_rows == {card}"
            gold_t = f"At most min({a}, {b_keys}) = {min(a, b_keys)} keys match; {min(a, b_keys)} * {c} = {card}."
            gold_a = f"{card}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Cartesian product {a*b_keys} * {c} = {a*b_keys*c}.\n#A$# {a*b_keys*c}",
                f"#T$# Max key bound {max(a,b_keys)} * {c} = {max(a,b_keys)*c}.\n#A$# {max(a,b_keys)*c}",
                "#T$# Empty join.\n#A$# 0",
            )
            base_scores = (1.0, 0.10, 0.20, 0.0)
            domain = "relational_algebra"
        elif kind == 7:
            inversions = b * c
            task_id = f"dag_locks_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Lock hierarchy has {a} stages; {b} threads each acquire {c} ordered locks strictly increasing.\n"
                f"#Q$# [deduction] How many total lock acquisitions occur with 0 deadlock cycles?"
            )
            harness_h = f"acquisitions == {inversions} and cycles == 0"
            gold_t = f"Strict total order prevents cycles (0 deadlocks); {b} * {c} = {inversions} acquisitions."
            gold_a = f"{inversions} acquisitions, 0 cycles"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Multiply stages {a} * {b} = {a*b} acquisitions, 1 cycle.\n#A$# {a*b} acquisitions, 1 cycle",
                f"#T$# Sum threads {b} + {c} = {b+c} acquisitions.\n#A$# {b+c} acquisitions, 0 cycles",
                f"#T$# Deadlock immediately.\n#A$# 0 acquisitions, {b} cycles",
            )
            base_scores = (1.0, 0.10, 0.15, 0.0)
            domain = "concurrency_verification"
        elif kind == 8:
            n_lim = (c % 7) + 3
            expected_sum = sum(i * a for i in range(1, n_lim + 1) if i % 2 == 1)
            task_id = f"code_odd_sum_{a}_{n_lim}"
            prompt_q = (
                f"#U$# Write a Python function `solve(n, scale)` that sums `i * scale` for all odd integers `1 <= i <= n`.\n"
                f"#Q$# [code] Implement `solve({n_lim}, {a})` with 4-space indentation."
            )
            harness_h = f"assert solve({n_lim}, {a}) == {expected_sum}"
            gold_t = (
                f"Iterate i from 1 to {n_lim}; check `if i % 2 == 1:` and accumulate `total += i * scale` "
                f"to obtain {expected_sum}."
            )
            gold_a = (
                "def solve(n, scale):\n"
                "    total = 0\n"
                "    for i in range(1, n + 1):\n"
                "        if i % 2 == 1:\n"
                "            total += i * scale\n"
                "    return total"
            )
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                "#T$# Sum all integers without odd filter.\n#A$# def solve(n, scale):\n    return sum(range(1, n + 1)) * scale",
                "#T$# Off-by-one range bound.\n#A$# def solve(n, scale):\n    return (n - 1) * scale",
                "#T$# Constant stub.\n#A$# def solve(n, scale):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "algorithmic_code"
        elif kind == 9:
            rhs = a * b + c
            task_id = f"math_linear_eq_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Solve for `x` in the linear equation `{a}*x + {c} = {rhs}` and show each algebraic step.\n"
                f"#Q$# [math] Derive the exact integer solution `x`."
            )
            harness_h = f"expected == {b}"
            gold_t = (
                f"Step 1: Subtract {rhs} - {c} = {a*b}. "
                f"Step 2: Divide {a*b} // {a} = {b}."
            )
            gold_a = f"{b}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add {c} to quotient instead of subtracting before division: {b} + {c} = {b + c}.\n#A$# {b + c}",
                f"#T$# Subtract {rhs} - {a} = {rhs - a}.\n#A$# {rhs - a}",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "symbolic_math"
        elif kind == 10:
            v0 = a
            accel = 2 * b
            t_sec = (c % 6) + 2
            disp = v0 * t_sec + (accel * t_sec * t_sec) // 2
            task_id = f"physics_kinematics_{v0}_{accel}_{t_sec}"
            prompt_q = (
                f"#U$# A particle starts with initial velocity `v0 = {v0} m/s` under constant acceleration "
                f"`a = {accel} m/s^2` for `t = {t_sec} s`.\n"
                f"#Q$# [physics] Compute total displacement `d = v0*t + 0.5*a*t^2`."
            )
            harness_h = f"expected == {disp}"
            gold_t = (
                f"Drift term {v0} * {t_sec} = {v0*t_sec}; "
                f"acceleration term 0.5*{accel}*{t_sec}^2 = {(accel*t_sec*t_sec)//2}; "
                f"total {v0*t_sec} + {(accel*t_sec*t_sec)//2} = {disp}."
            )
            gold_a = f"{disp}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Omit factor 0.5 on acceleration term: {v0*t_sec + accel*t_sec*t_sec}.\n#A$# {v0*t_sec + accel*t_sec*t_sec}",
                f"#T$# Compute final velocity {v0} + {accel*t_sec} = {v0 + accel*t_sec}.\n#A$# {v0 + accel*t_sec}",
                "#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.15, 0.0)
            domain = "physics_kinematics"
        elif kind == 11:
            path_alpha = a + b + c
            path_beta = a + c
            slack_beta = path_alpha - path_beta
            task_id = f"pm_critical_path_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# A software release DAG has Milestone M1 ({a} days) feeding two parallel tracks: "
                f"Track Alpha ({b} days) and Track Beta (0 days) before final integration M2 ({c} days).\n"
                f"#Q$# [project_management] Compute the Critical Path duration and Track Beta float slack."
            )
            harness_h = f"acquisitions == {path_alpha} and cycles == {slack_beta}"
            gold_t = (
                f"Critical Path M1 -> Alpha -> M2: {a} + {b} = {a+b}, plus {c} gives {path_alpha} days. "
                f"Track Beta takes {a} + {c} = {path_beta} days, slack {path_alpha} - {path_beta} = {slack_beta} days."
            )
            gold_a = f"{path_alpha} days critical path, {slack_beta} days slack"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Sum all nodes serially: {a+b+c+b} days critical path, 0 days slack.\n#A$# {a+b+c+b} days critical path, 0 days slack",
                f"#T$# Shortest path duration: {path_beta} days critical path, {b} days slack.\n#A$# {path_beta} days critical path, {b} days slack",
                "#T$# Deadlock immediately.\n#A$# 0 days critical path, 0 days slack",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "project_management"
        elif kind == 12:
            # Modular Ring Arithmetic (`(a * b + c) mod p`)
            p_mod = 17 + (idx % 13)
            mod_val = (a * b + c) % p_mod
            task_id = f"math_mod_ring_{a}_{b}_{c}_{p_mod}"
            prompt_q = (
                f"#U$# In the finite ring Z_{p_mod}, evaluate `r = ({a} * {b} + {c}) mod {p_mod}`.\n"
                f"#Q$# [math] Compute the canonical residue `0 <= r < {p_mod}`."
            )
            harness_h = f"expected == {mod_val}"
            gold_t = f"Compute {a} * {b} = {a*b}, add {a*b} + {c} = {a*b+c}, then reduce mod {p_mod} to get {mod_val}."
            gold_a = f"{mod_val}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Unreduced sum {a*b + c} mod {p_mod}.\n#A$# {(a + b + c) % p_mod}",
                f"#T$# Off-by-one residue {(mod_val + 1) % p_mod} mod {p_mod}.\n#A$# {(mod_val + 1) % p_mod}",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "symbolic_math"
        elif kind == 13:
            # 2x2 Matrix Determinant & Trace
            det_m = a * c - b * 2
            tr_m = a + c
            task_id = f"math_matrix_det_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Matrix M = [[{a}, {b}], [2, {c}]].\n"
                f"#Q$# [math] Compute det(M) = {a}*{c} - 2*{b} and trace(M) = {a} + {c}."
            )
            harness_h = f"acquisitions == {det_m} and cycles == {tr_m}"
            gold_t = f"Determinant: {a} * {c} = {a*c}, 2 * {b} = {2*b}, so {a*c} - {2*b} = {det_m}. Trace: {a} + {c} = {tr_m}."
            gold_a = f"det={det_m}, trace={tr_m}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add diagonal products: {a*c} + {2*b} = {a*c + 2*b}.\n#A$# det={a*c + 2*b}, trace={tr_m}",
                f"#T$# Swap determinant and trace.\n#A$# det={tr_m}, trace={det_m}",
                "#T$# Zero delta.\n#A$# det=0, trace=0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "symbolic_math"
        elif kind == 14:
            # Multi-point Python List Comprehension / Filter-Transform
            thresh = b % 7 + 2
            mult = c % 5 + 2
            sample_xs = [1, thresh, thresh + 2, thresh - 1, thresh + 5]
            expected_out = [x * mult for x in sample_xs if x >= thresh]
            task_id = f"code_list_comp_{thresh}_{mult}_{a}"
            prompt_q = (
                f"#U$# Write `solve(xs, t, m)` returning `[x * m for x in xs if x >= t]`.\n"
                f"#Q$# [code] Verify on `xs = {sample_xs}`, `t = {thresh}`, `m = {mult}`."
            )
            harness_h = f"assert solve({sample_xs}, {thresh}, {mult}) == {expected_out} and solve([], {thresh}, {mult}) == []"
            gold_t = f"Filter elements x >= {thresh} and multiply each by {mult} using a list comprehension."
            gold_a = "def solve(xs, t, m):\n    return [x * m for x in xs if x >= t]"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                "#T$# Strict greater-than filter x > t.\n#A$# def solve(xs, t, m):\n    return [x * m for x in xs if x > t]",
                "#T$# Add m instead of multiplying.\n#A$# def solve(xs, t, m):\n    return [x + m for x in xs if x >= t]",
                "#T$# Constant stub.\n#A$# def solve(xs, t, m):\n    return []",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "algorithmic_code"
        elif kind == 15:
            # Unnormalized Bayes Posterior Odds Numerator
            prior_h1 = a
            like_h1 = b
            num_post = prior_h1 * like_h1
            task_id = f"bayes_odds_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Hypothesis H1 has prior weight {prior_h1} and likelihood weight {like_h1}; "
                f"distractor H0 has weight {c}.\n"
                f"#Q$# [deduction] Compute the unnormalized posterior weight `prior_H1 * likelihood_H1`."
            )
            harness_h = f"expected == {num_post}"
            gold_t = f"By Bayes rule numerator, multiply prior and likelihood: {prior_h1} * {like_h1} = {num_post}."
            gold_a = f"{num_post}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add weights: {prior_h1} + {like_h1} = {prior_h1 + like_h1}.\n#A$# {prior_h1 + like_h1}",
                f"#T$# Multiply by distractor: {prior_h1} * {c} = {prior_h1 * c}.\n#A$# {prior_h1 * c}",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.15, 0.10, 0.0)
            domain = "multi_hop_deduction"
        elif kind == 16:
            # Thermodynamics Calorimetry Q = m * c_p * delta_T
            mass_g = a
            cp_j = (b % 5) + 2
            dt_k = (c % 8) + 3
            q_joules = mass_g * cp_j * dt_k
            task_id = f"physics_calorimetry_{mass_g}_{cp_j}_{dt_k}"
            prompt_q = (
                f"#U$# A sample of mass `m = {mass_g} g` with specific heat `c = {cp_j} J/(g*K)` "
                f"is heated by `dT = {dt_k} K`.\n"
                f"#Q$# [physics] Compute absorbed thermal energy `Q = m * c * dT` in Joules."
            )
            harness_h = f"expected == {q_joules}"
            gold_t = f"First compute m * c: {mass_g} * {cp_j} = {mass_g * cp_j}; then multiply by dT: {mass_g * cp_j} * {dt_k} = {q_joules}."
            gold_a = f"{q_joules}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add temperature change: {mass_g * cp_j} + {dt_k} = {mass_g * cp_j + dt_k}.\n#A$# {mass_g * cp_j + dt_k}",
                f"#T$# Omit specific heat: {mass_g} * {dt_k} = {mass_g * dt_k}.\n#A$# {mass_g * dt_k}",
                "#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.15, 0.0)
            domain = "physics_kinematics"
        elif kind == 17:
            # IFEval Constrained Verification (lowercase + no_comma + keyword + exact_int)
            res_val = a + b * c
            task_id = f"ifeval_math_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Compute `{a} + {b} * {c}` and reply in all lowercase letters with NO commas, "
                f"including the word `verified` and the exact integer result.\n"
                f"#Q$# [evaluation] Follow every formatting and arithmetic constraint."
            )
            harness_h = f"ifeval:lowercase;no_comma;keyword:verified;exact_int:{res_val}"
            gold_t = f"Multiply {b} * {c} = {b*c}, then add {a} + {b*c} = {res_val}. Format lowercase without commas."
            gold_a = f"the verified result is {res_val} units"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# {gold_t}\n#A$# The Verified Result, is {res_val} units",
                f"#T$# {gold_t}\n#A$# the result is {a + b + c} units",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "invariant_verification"
        elif kind == 18:
            # Multi-assert Python sequence generator
            step_d = (b % 6) + 2
            task_id = f"code_arith_prog_{a}_{step_d}"
            prompt_q = (
                f"#U$# Write `solve(n, start, step)` returning `start + n * step` for `n >= 0`.\n"
                f"#Q$# [code] Must pass assertions at `n = 0`, `n = 3`, and `n = {c}`."
            )
            harness_h = (
                f"assert solve(0, {a}, {step_d}) == {a} "
                f"and solve(3, {a}, {step_d}) == {a + 3 * step_d} "
                f"and solve({c}, {a}, {step_d}) == {a + c * step_d}"
            )
            gold_t = f"Closed form linear progression: return start + n * step; at n={c}, {c} * {step_d} = {c*step_d}, plus {a} = {a + c*step_d}."
            gold_a = "def solve(n, start, step):\n    return start + n * step"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                "#T$# Off-by-one multiplier.\n#A$# def solve(n, start, step):\n    return start + (n + 1) * step",
                "#T$# Multiply start instead of step.\n#A$# def solve(n, start, step):\n    return n * start + step",
                "#T$# Constant stub.\n#A$# def solve(n, start, step):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "algorithmic_code"
        elif kind == 19:
            # Horn-Clause Boolean Propagation Checksum
            w1 = a + b
            w2 = w1 + c
            checksum = w1 + w2
            task_id = f"logic_horn_prop_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# Horn rules: Fact `F1 = {a}`, Fact `F2 = {b}`. Rule R1 derives `G1 = F1 + F2`. "
                f"Rule R2 derives `G2 = G1 + {c}`.\n"
                f"#Q$# [deduction] Compute the forward-chaining sum `G1 + G2`."
            )
            harness_h = f"expected == {checksum}"
            gold_t = f"Forward chain R1: {a} + {b} = {w1}. Forward chain R2: {w1} + {c} = {w2}. Sum: {w1} + {w2} = {checksum}."
            gold_a = f"{checksum}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Only return G2: {w1} + {c} = {w2}.\n#A$# {w2}",
                f"#T$# Sum raw facts once: {a} + {b} = {w1}.\n#A$# {w1}",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "multi_hop_deduction"
        elif kind == 20:
            # COMPOSED TEMPLATE 1 (Symbolic Algebra o Python Code):
            # Solve linear equation for hidden slope k, then synthesize Python function using k.
            k_slope = (b % 7) + 2
            rhs_eq = a * k_slope + c
            n_eval = (c % 5) + 3
            expected_val = k_slope * n_eval + a
            task_id = f"comp_alg_code_{a}_{k_slope}_{c}_{n_eval}"
            prompt_q = (
                f"#U$# [Composed Algebra->Code] First solve `{a} * k + {c} = {rhs_eq}` for integer `k`. "
                f"Then implement `solve(n)` returning `{k_slope} * n + {a}` (using the solved `k`).\n"
                f"#Q$# [code] Derive `k` in `#T$#` and implement `solve(n)` in `#A$#`."
            )
            harness_h = f"assert solve({n_eval}) == {expected_val} and solve(0) == {a}"
            gold_t = (
                f"Stage 1 (Algebra): {rhs_eq} - {c} = {a * k_slope}, so k = {a * k_slope} // {a} = {k_slope}. "
                f"Stage 2 (Code): implement solve(n) = {k_slope} * n + {a}; at n={n_eval}, {k_slope} * {n_eval} = {k_slope * n_eval}, plus {a} = {expected_val}."
            )
            gold_a = f"def solve(n):\n    return {k_slope} * n + {a}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Stage 1 (Algebra): {rhs_eq} + {c} = {rhs_eq + c}, so k = {k_slope + 1}. Stage 2 (Code): implement solve(n) = {k_slope + 1} * n + {a}.\n#A$# def solve(n):\n    return {k_slope + 1} * n + {a}",
                f"#T$# Stage 1 (Algebra): {rhs_eq} - {c} = {a * k_slope}, so k = {k_slope}. Stage 2 (Code): subtract offset {a} instead of adding.\n#A$# def solve(n):\n    return {k_slope} * n - {a}",
                "#T$# Constant stub.\n#A$# def solve(n):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.20, 0.0)
            domain = "compositional_code_math"
        elif kind == 21:
            # COMPOSED TEMPLATE 2 (Physics Kinematics o Modular Ring Checksum):
            v0 = a % 15 + 2
            accel = 2 * ((b % 6) + 1)
            t_sec = (c % 4) + 2
            disp = v0 * t_sec + (accel * t_sec * t_sec) // 2
            p_ring = 23
            residue = (disp + c) % p_ring
            task_id = f"comp_phys_mod_{v0}_{accel}_{t_sec}_{c}"
            prompt_q = (
                f"#U$# [Composed Physics->ModRing] Stage 1: compute displacement `d = v0*t + 0.5*a*t^2` "
                f"for `v0={v0}`, `a={accel}`, `t={t_sec}`. Stage 2: compute checksum `(d + {c}) mod {p_ring}`.\n"
                f"#Q$# [physics] Derive `d` and return the final modular residue."
            )
            harness_h = f"expected == {residue}"
            gold_t = (
                f"Stage 1 (Kinematics): drift {v0} * {t_sec} = {v0*t_sec}, accel term = {(accel*t_sec*t_sec)//2}, "
                f"d = {disp}. Stage 2 (ModRing): ({disp} + {c}) mod {p_ring} = {residue}."
            )
            gold_a = f"{residue}"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Stage 1 (Kinematics): drift {v0} * {t_sec} = {v0*t_sec}, accel term = {(accel*t_sec*t_sec)//2}, d = {disp}. Stage 2 (ModRing): omit mod reduction.\n#A$# {disp}",
                f"#T$# Stage 1 (Kinematics): final velocity {v0} + {accel * t_sec} = {v0 + accel * t_sec}. Stage 2 (ModRing): ({v0 + accel * t_sec} + {c}) mod {p_ring}.\n#A$# {(v0 + accel * t_sec + c) % p_ring}",
                "#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.15, 0.0)
            domain = "compositional_physics_math"
        elif kind == 22:
            # COMPOSED TEMPLATE 3 (Project DAG Critical Path o Python Slack Aggregator):
            crit_len = a + b + c
            sample_tracks = [a + c, a + b, crit_len]
            expected_slack = sum(crit_len - t_d for t_d in sample_tracks)
            task_id = f"comp_dag_code_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# [Composed DAG->Code] Milestone M1 ({a}d) + Alpha ({b}d) + M2 ({c}d) defines critical path `C = {crit_len}`. "
                f"Write `solve(tracks, crit)` returning `sum(crit - x for x in tracks)`.\n"
                f"#Q$# [code] Implement `solve(tracks, crit)` and verify on `tracks={sample_tracks}, crit={crit_len}`."
            )
            harness_h = f"assert solve({sample_tracks}, {crit_len}) == {expected_slack}"
            gold_t = (
                f"Stage 1 (DAG): Critical path {a} + {b} = {a+b}, plus {c} = {crit_len}. "
                f"Stage 2 (Code): Sum slack (crit - x) across tracks to get {expected_slack}."
            )
            gold_a = "def solve(tracks, crit):\n    return sum(crit - x for x in tracks)"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Stage 1 (DAG): Critical path {a} + {b} = {a+b}, plus {c} = {crit_len}. Stage 2 (Code): Sum tracks directly without subtracting from crit.\n#A$# def solve(tracks, crit):\n    return sum(tracks)",
                f"#T$# Stage 1 (DAG): Shortest path {a} + {c} = {a+c}. Stage 2 (Code): Return single max slack crit - min(tracks).\n#A$# def solve(tracks, crit):\n    return crit - min(tracks)",
                "#T$# Constant stub.\n#A$# def solve(tracks, crit):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "compositional_code_pm"
        elif kind == 23:
            # COMPOSED TEMPLATE 4 (Multi-Hop Deduction o IFEval Lexical Constraints):
            prod_ab = a * b
            final_v = prod_ab + c
            task_id = f"comp_ded_ifeval_{a}_{b}_{c}"
            prompt_q = (
                f"#U$# [Composed Deduction->IFEval] Cluster A runs {a} nodes with {b} shards each, plus {c} standby shards. "
                f"State the total shards in lowercase with no commas, including the keyword `shards` and the exact integer.\n"
                f"#Q$# [evaluation] Satisfy both the 2-hop arithmetic deduction and all IFEval constraints."
            )
            harness_h = f"ifeval:lowercase;no_comma;keyword:shards;exact_int:{final_v}"
            gold_t = f"Stage 1: {a} * {b} = {prod_ab}, then {prod_ab} + {c} = {final_v}. Stage 2: format lowercase with no comma and keyword shards."
            gold_a = f"cluster a manages {final_v} total shards online"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# {gold_t}\n#A$# Cluster A, manages {final_v} total shards online",
                f"#T$# Stage 1: {a} + {b} = {a+b}, then {a+b} + {c} = {a+b+c}. Stage 2: format lowercase with no comma.\n#A$# cluster a manages {a+b+c} total units online",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "compositional_deduction_ifeval"
        elif kind == 24:
            # ARC-Challenge Specialization 1: Experimental Control & Causal Abduction
            base_rate = 12 + a
            catalyst_boost = 5 + b
            thermal_drift = 2 + (c % 6)
            control_obs = base_rate + thermal_drift
            treatment_obs = base_rate + catalyst_boost + thermal_drift
            isolated_effect = treatment_obs - control_obs
            task_id = f"arc_exp_control_{base_rate}_{catalyst_boost}_{thermal_drift}"
            prompt_q = (
                f"#U$# [ARC Experimental Control] In a controlled enzyme assay, the untreated Control vessel "
                f"starts at baseline `{base_rate} uM/s` and experiences ambient thermal drift `+{thermal_drift} uM/s` "
                f"(reaching `{control_obs} uM/s`). The Catalyst-treated vessel under identical ambient drift reaches `{treatment_obs} uM/s`.\n"
                f"#Q$# [arc_science] Isolate the net causal effect of the catalyst by subtracting the control vessel rate, "
                f"and reply in lowercase with no commas including the keyword `control` and the exact integer."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:control;exact_int:{isolated_effect}"
            gold_t = (
                f"To isolate the independent variable in this controlled experiment, we first verify the untreated control "
                f"rate {base_rate} + {thermal_drift} = {control_obs} and then subtract it from the treatment observation "
                f"{treatment_obs} - {control_obs} = {isolated_effect} to eliminate confounding thermal drift."
            )
            gold_a = f"after subtracting the control vessel baseline the net catalyst effect is {isolated_effect} units"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Confound thermal drift with treatment by computing {treatment_obs} - {base_rate} = {catalyst_boost + thermal_drift}.\n#A$# after subtracting the control vessel baseline the net catalyst effect is {catalyst_boost + thermal_drift} units",
                f"#T$# Report raw treatment rate without subtracting control {treatment_obs}.\n#A$# After Subtracting, The Control Vessel Rate Is {treatment_obs} units",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "arc_science"
        elif kind == 25:
            # ARC-Physics Specialization: Mechanical-to-Thermal Energy Conservation
            mass_kg = 2 + (a % 9)
            g_acc = 10
            height_m = 4 + (b % 11)
            pe_initial = mass_kg * g_acc * height_m
            friction_loss = 5 + (c % 15)
            net_ke = pe_initial - friction_loss
            task_id = f"arc_phys_energy_{mass_kg}_{height_m}_{friction_loss}"
            prompt_q = (
                f"#U$# [ARC Physics Conservation] A block of mass `m = {mass_kg} kg` descends a ramp of vertical height "
                f"`h = {height_m} m` under gravity `g = 10 m/s^2`, losing `{friction_loss} J` to surface friction.\n"
                f"#Q$# [arc_physics] Compute the final kinetic energy in Joules and state it in lowercase with no commas, "
                f"including the keyword `joules`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:joules;exact_int:{net_ke}"
            gold_t = (
                f"By conservation of mechanical and thermal energy, gravitational potential energy converts into kinetic energy "
                f"minus frictional heat: first compute {mass_kg} * {g_acc} = {mass_kg * g_acc}, multiply by height "
                f"{mass_kg * g_acc} * {height_m} = {pe_initial}, and subtract friction {pe_initial} - {friction_loss} = {net_ke}."
            )
            gold_a = f"conservation of energy yields {net_ke} joules of terminal kinetic energy"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Add friction instead of subtracting: {pe_initial} + {friction_loss} = {pe_initial + friction_loss}.\n#A$# conservation of energy yields {pe_initial + friction_loss} joules of terminal kinetic energy",
                f"#T$# Ignore frictional dissipation and report potential energy {pe_initial}.\n#A$# Conservation, Of Energy Yields {pe_initial} Joules",
                "#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "arc_physics"
        elif kind == 26:
            # ARC-Chemistry Specialization: Stoichiometry & Limiting Reagent Yield
            moles_b = 3 + (b % 12)
            moles_a = 2 * moles_b + 2 + (a % 7)  # A is in excess for 2A + B -> 2C
            theoretical_c = 2 * moles_b
            loss_moles = 1 + (c % max(2, moles_b))
            actual_c = theoretical_c - loss_moles
            task_id = f"arc_chem_stoich_{moles_a}_{moles_b}_{loss_moles}"
            prompt_q = (
                f"#U$# [ARC Chemistry Stoichiometry] Reaction `2 A + 1 B -> 2 C` is run with `{moles_a} mol` of reactant A "
                f"and `{moles_b} mol` of reactant B, with `{loss_moles} mol` of product C lost during filtration.\n"
                f"#Q$# [arc_chemistry] Identify the limiting reagent and report the isolated moles of C in lowercase "
                f"with no commas, including the keyword `limiting`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:limiting;exact_int:{actual_c}"
            gold_t = (
                f"Comparing stoichiometric ratios for 2A + B -> 2C, {moles_b} moles of B require 2 * {moles_b} = {2 * moles_b} "
                f"moles of A, which is less than the available {moles_a} moles of A. Thus B is the limiting reagent producing "
                f"{theoretical_c} theoretical moles of C, and subtracting filtration loss {theoretical_c} - {loss_moles} = {actual_c}."
            )
            gold_a = f"reactant b is the limiting reagent yielding {actual_c} moles of product"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Use excess reactant A directly: {moles_a} - {loss_moles} = {moles_a - loss_moles}.\n#A$# reactant a is the limiting reagent yielding {moles_a - loss_moles} moles of product",
                f"#T$# Omit 2x stoichiometric coefficient on product C: {moles_b} - {loss_moles} = {moles_b - loss_moles}.\n#A$# Reactant B, Is Limiting With {moles_b - loss_moles} moles",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "arc_chemistry"
        elif kind == 27:
            # ARC-Biology Specialization: Mendelian & Population Genetics Carrier Count
            p_num = 3 + (a % 4)  # out of 10
            q_num = 10 - p_num
            pop_scale = 2 + (b % 8)
            # N = 100 * pop_scale so 2 * p * q * N = 2 * p_num * q_num * pop_scale is exact integer
            carriers = 2 * p_num * q_num * pop_scale
            hom_dom = p_num * p_num * pop_scale
            task_id = f"arc_bio_genetics_{p_num}_{q_num}_{pop_scale}"
            prompt_q = (
                f"#U$# [ARC Biology Genetics] A population of `{100 * pop_scale}` diploid organisms is in Hardy-Weinberg "
                f"equilibrium with dominant allele frequency `p = 0.{p_num}` and recessive allele frequency `q = 0.{q_num}`.\n"
                f"#Q$# [arc_biology] Compute the expected number of `2pq * N` heterozygous carriers and reply in lowercase "
                f"with no commas, including the keyword `heterozygous`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:heterozygous;exact_int:{carriers}"
            gold_t = (
                f"Under Hardy-Weinberg equilibrium p^2 + 2pq + q^2 = 1, the heterozygous carrier count in a population of "
                f"{100 * pop_scale} is 2 * {p_num} * {q_num} * {pop_scale}: first compute {2 * p_num} * {q_num} = {2 * p_num * q_num}, "
                f"then multiply {2 * p_num * q_num} * {pop_scale} = {carriers} heterozygous individuals."
            )
            gold_a = f"the hardy weinberg population contains {carriers} heterozygous carrier organisms"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Omit factor of 2 in heterozygote frequency: {p_num * q_num} * {pop_scale} = {p_num * q_num * pop_scale}.\n#A$# the hardy weinberg population contains {p_num * q_num * pop_scale} heterozygous carrier organisms",
                f"#T$# Compute homozygous dominant p^2 * N = {hom_dom} instead of 2pq * N.\n#A$# The Population, Contains {hom_dom} Heterozygous Organisms",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "arc_biology"
        elif kind == 28:
            # HellaSwag / PIQA Commonsense Specialization: Physical Precondition-Action-Consequence Invariant
            init_vol = 20 + a
            inflow_rate = 4 + (b % 9)
            duration_min = 3 + (c % 5)
            added_vol = inflow_rate * duration_min
            relief_drain = 6 + (a % 7)
            # Valve relief opens only after inflow completes
            final_vol = init_vol + added_vol - relief_drain
            task_id = f"commonsense_phys_{init_vol}_{inflow_rate}_{duration_min}_{relief_drain}"
            prompt_q = (
                f"#U$# [Physical Commonsense & Event Completion] A rigid tank holds `{init_vol} L` of water while the drain "
                f"valve is sealed. A pump adds `{inflow_rate} L/min` for `{duration_min} minutes`, after which a pressure-relief "
                f"latch vents `{relief_drain} L` before resealing.\n"
                f"#Q$# [hellaswag_commonsense] Track the physical preconditions across stages and state the final volume in "
                f"lowercase with no commas, including the keyword `equilibrium`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:equilibrium;exact_int:{final_vol}"
            gold_t = (
                f"Because the drain valve is initially sealed, the pump first adds {inflow_rate} * {duration_min} = {added_vol} "
                f"liters to the initial {init_vol} liters so {init_vol} + {added_vol} = {init_vol + added_vol}, and venting "
                f"the relief latch leaves {init_vol + added_vol} - {relief_drain} = {final_vol} liters at equilibrium."
            )
            gold_a = f"the physical tank reaches equilibrium at {final_vol} liters of water"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Subtract relief drain every minute instead of once: {init_vol + added_vol} - {relief_drain * duration_min} = {init_vol + added_vol - relief_drain * duration_min}.\n#A$# the physical tank reaches equilibrium at {init_vol + added_vol - relief_drain * duration_min} liters of water",
                f"#T$# Ignore inflow duration and add {init_vol} + {inflow_rate} = {init_vol + inflow_rate}.\n#A$# The Physical Tank, Reaches Equilibrium At {init_vol + inflow_rate} liters",
                "#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "hellaswag_commonsense"
        elif kind == 29:
            # MMLU Microeconomics Specialization: Monopoly MR = MC Market Equilibrium
            slope_b = 2 + (b % 4)
            q_opt = 3 + (a % 9)
            mc_val = 4 + (c % 10)
            # For inverse demand P(Q) = A_int - slope_b * Q, MR(Q) = A_int - 2 * slope_b * Q = mc_val
            a_int = mc_val + 2 * slope_b * q_opt
            p_opt = a_int - slope_b * q_opt
            task_id = f"mmlu_econ_{a_int}_{slope_b}_{mc_val}"
            prompt_q = (
                f"#U$# [MMLU Microeconomics] A firm faces linear inverse demand `P(Q) = {a_int} - {slope_b} * Q` (so marginal "
                f"revenue is `MR(Q) = {a_int} - {2 * slope_b} * Q`) and constant marginal cost `MC = {mc_val}`.\n"
                f"#Q$# [mmlu_economics] Solve `MR(Q) = MC` for the profit-maximizing quantity `Q*` and report the resulting "
                f"market price `P(Q*)` in lowercase with no commas, including the keyword `equilibrium`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:equilibrium;exact_int:{p_opt}"
            gold_t = (
                f"Setting marginal revenue equal to marginal cost gives {a_int} - {mc_val} = {2 * slope_b * q_opt}, which "
                f"divides by {2 * slope_b} to yield optimal output Q* = {q_opt}. Substituting Q* into inverse demand "
                f"computes {slope_b} * {q_opt} = {slope_b * q_opt} and {a_int} - {slope_b * q_opt} = {p_opt}."
            )
            gold_a = f"the profit maximizing market equilibrium price is {p_opt} dollars per unit"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Report quantity Q* = {q_opt} instead of market price P(Q*).\n#A$# the profit maximizing market equilibrium price is {q_opt} dollars per unit",
                f"#T$# Equate P(Q) = MC instead of MR(Q) = MC, yielding price {mc_val}.\n#A$# The Equilibrium, Price Is {mc_val} Dollars",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "mmlu_economics"
        elif kind == 30:
            # MMLU Formal Logic Specialization: First-Order Quantifier & Contrapositive Set Deduction
            n_univ = 60 + a + b
            not_q = 12 + (b % 15)
            p_set = 15 + (a % 18)
            if n_univ - p_set - not_q == p_set + not_q:
                n_univ += 7
            # Since forall x (P(x) -> Q(x)), by contrapositive ~Q(x) -> ~P(x), so P and ~Q are disjoint!
            neither_p_nor_notq = n_univ - p_set - not_q
            task_id = f"mmlu_logic_{n_univ}_{p_set}_{not_q}"
            prompt_q = (
                f"#U$# [MMLU Formal Logic] Over a finite domain of `{n_univ}` elements, universal axiom `forall x (P(x) -> Q(x))` "
                f"holds. Exactly `{p_set}` elements satisfy `P(x)` and `{not_q}` elements satisfy `~Q(x)`.\n"
                f"#Q$# [mmlu_logic] Apply contrapositive entailment `~Q(x) -> ~P(x)` to compute how many elements satisfy "
                f"`Q(x) and ~P(x)`, replying in lowercase with no commas and the keyword `contrapositive`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:contrapositive;exact_int:{neither_p_nor_notq}"
            gold_t = (
                f"By the contrapositive equivalence ~Q(x) -> ~P(x), the {not_q} elements in ~Q are strictly disjoint from "
                f"the {p_set} elements in P. Total elements in Q equal {n_univ} - {not_q} = {n_univ - not_q}, and subtracting "
                f"the subset P leaves {n_univ - not_q} - {p_set} = {neither_p_nor_notq} elements in Q and ~P."
            )
            gold_a = f"by contrapositive entailment exactly {neither_p_nor_notq} elements satisfy q without p"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Subtract only ~Q from universe: {n_univ} - {not_q} = {n_univ - not_q}.\n#A$# by contrapositive entailment exactly {n_univ - not_q} elements satisfy q without p",
                f"#T$# Add P and ~Q: {p_set} + {not_q} = {p_set + not_q}.\n#A$# By Contrapositive, Entailment {p_set + not_q} Elements Match",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "mmlu_logic"
        elif kind == 31:
            # MMLU Probability & Bayesian Diagnostic Frequency Update
            sick = 10 + (a % 15)
            sens_pct = 80 + (b % 15)  # we use natural counts directly to stay exact integer
            tp_count = sick - (1 + (c % 3))
            fp_count = 6 + (b % 11)
            total_flagged = tp_count + fp_count
            task_id = f"mmlu_bayes_{sick}_{tp_count}_{fp_count}"
            prompt_q = (
                f"#U$# [MMLU Bayesian Inference] In a screening cohort, `{sick}` patients carry condition C. The assay "
                f"correctly flags `{tp_count}` true positives among carriers, but also triggers `{fp_count}` false positives "
                f"among non-carriers.\n"
                f"#Q$# [mmlu_probability] Compute the total positive test denominator `TP + FP` for Bayes' rule and state "
                f"it in lowercase with no commas, including the keyword `posterior`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:posterior;exact_int:{total_flagged}"
            gold_t = (
                f"To form the Bayesian posterior predictive denominator P(Test+), we sum the verified true positive count "
                f"and false positive count across cohorts: {tp_count} + {fp_count} = {total_flagged} total positive flags."
            )
            gold_a = f"the bayesian posterior denominator contains {total_flagged} total positive test results"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Sum total sick carriers and false positives: {sick} + {fp_count} = {sick + fp_count}.\n#A$# the bayesian posterior denominator contains {sick + fp_count} total positive test results",
                f"#T$# Subtract false positives from true positives: {tp_count} - {fp_count} = {tp_count - fp_count}.\n#A$# The Bayesian, Posterior Denominator Is {tp_count - fp_count}",
                "#T$# Default fallback.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "mmlu_probability"
        elif kind == 32:
            # MMLU Computer Architecture Specialization: Two-Level Cache AMAT & Stall Cycles
            l1_hit = 2 + (a % 3)
            l1_misses = 4 + (b % 7)
            miss_penalty = 15 + (c % 12)
            base_insts = 20 + a
            base_cycles = base_insts * l1_hit
            stall_cycles = l1_misses * miss_penalty
            total_cycles = base_cycles + stall_cycles
            task_id = f"mmlu_arch_cache_{base_insts}_{l1_hit}_{l1_misses}_{miss_penalty}"
            prompt_q = (
                f"#U$# [MMLU Computer Systems] A processor executes `{base_insts}` memory accesses with L1 hit latency "
                f"`{l1_hit} cycles` per access, incurring `{l1_misses}` L1 cache misses each costing a `{miss_penalty}-cycle` "
                f"L2/DRAM stall penalty.\n"
                f"#Q$# [mmlu_cs_systems] Compute the total execution cycles (`base_accesses * hit_latency + misses * penalty`) "
                f"in lowercase with no commas, including the keyword `cycles`."
            )
            harness_h = f"ifeval:lowercase;no_comma;min_words:6;keyword:cycles;exact_int:{total_cycles}"
            gold_t = (
                f"Total memory hierarchy execution time combines baseline L1 hit cycles {base_insts} * {l1_hit} = {base_cycles} "
                f"with L1 miss stall penalties {l1_misses} * {miss_penalty} = {stall_cycles}, yielding {base_cycles} + {stall_cycles} = {total_cycles} cycles."
            )
            gold_a = f"the memory hierarchy requires {total_cycles} total processor execution cycles"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                f"#T$# Report only stall penalty cycles {l1_misses} * {miss_penalty} = {stall_cycles}.\n#A$# the memory hierarchy requires {stall_cycles} total processor execution cycles",
                f"#T$# Add miss count without multiplying by penalty: {base_cycles} + {l1_misses} = {base_cycles + l1_misses}.\n#A$# The Memory, Hierarchy Requires {base_cycles + l1_misses} Cycles",
                "#T$# Zero delta.\n#A$# 0",
            )
            base_scores = (1.0, 0.20, 0.10, 0.0)
            domain = "mmlu_cs_systems"
        else:
            # Family 33: Recursive/Iterative Code State Transducer Synthesis
            mod_m = 3 + (b % 5)
            bias_d = 2 + (c % 6)
            sample_arr = [a, a + 1, a + 2, a + 3, a + 4]
            expected_sum = sum((x * 2 + bias_d) for x in sample_arr if x % mod_m != 0)
            task_id = f"code_transducer_{a}_{mod_m}_{bias_d}"
            prompt_q = (
                f"#U$# [Recursive State Transducer] Write `solve(xs, m, d)` in Python returning the sum of `(x * 2 + d)` "
                f"for all elements `x` in `xs` where `x % m != 0`.\n"
                f"#Q$# [recursive_code] Derive the filtered transducer sum in `#T$#` and implement `solve(xs, m, d)` in `#A$#`."
            )
            harness_h = (
                f"assert solve({sample_arr}, {mod_m}, {bias_d}) == {expected_sum} "
                f"and solve([], {mod_m}, {bias_d}) == 0"
            )
            gold_t = (
                f"To synthesize the state transducer over xs, we filter out multiples of m = {mod_m} using x % m != 0, "
                f"apply the affine map x * 2 + {bias_d} to each surviving element, and accumulate the sum {expected_sum}."
            )
            gold_a = "def solve(xs, m, d):\n    return sum((x * 2 + d) for x in xs if x % m != 0)"
            base_cands = (
                f"#T$# {gold_t}\n#A$# {gold_a}",
                "#T$# Invert the modular filter condition to x % m == 0.\n#A$# def solve(xs, m, d):\n    return sum((x * 2 + d) for x in xs if x % m == 0)",
                "#T$# Omit factor of 2 in the affine state transducer.\n#A$# def solve(xs, m, d):\n    return sum((x + d) for x in xs if x % m != 0)",
                "#T$# Constant stub.\n#A$# def solve(xs, m, d):\n    return 0",
            )
            base_scores = (1.0, 0.25, 0.15, 0.0)
            domain = "recursive_code"

        p_cands, p_scores, g_idx = _permute(base_cands, base_scores)
        cached_s = self.scaffold_cache.get(task_id, default=_SCAFFOLD_CANDIDATES[idx % len(_SCAFFOLD_CANDIDATES)])
        self._last_task_kinds[task_id] = kind
        skill_name, _ = self.env_architect.skill_and_domain(kind)
        rec_depth = self.env_architect.recursion_depth(kind)
        mods = self.env_architect.difficulty_mods(kind)
        stars = compute_star_difficulty(tier, recursion_depth=rec_depth, mods=mods)
        return OrnithTask(
            task_id=task_id,
            domain=domain,
            prompt_q=prompt_q,
            harness_h=harness_h,
            scaffold_s=cached_s,
            candidate_rollouts=p_cands,
            harness_scores=p_scores,
            gold_index=g_idx,
            gold_thought=gold_t,
            gold_answer=gold_a,
            skill=skill_name,
            star_difficulty=stars,
            recursion_depth=rec_depth,
            difficulty_mods=mods,
        )

    @staticmethod
    def _bare_matched_options(task: OrnithTask) -> tuple[str, ...]:
        """Extract bare `#A$#` candidate answers with normalized formatting and non-trivial distractors."""
        import re

        gold_a = task.gold_answer.strip()
        if not gold_a and 0 <= task.gold_index < len(task.candidate_rollouts):
            g_raw = task.candidate_rollouts[task.gold_index]
            gold_a = g_raw.split("#A$#", 1)[1].strip() if "#A$#" in g_raw else g_raw.strip()

        is_code = "def " in gold_a
        gold_nums = [int(x) for x in re.findall(r"-?\d+", gold_a)]
        gold_num = gold_nums[-1] if gold_nums else None
        gold_template = gold_a.lower().replace(",", "")
        used_nums: set[int] = {gold_num} if gold_num is not None else set()

        bare_opts: list[str] = []
        fallback_offsets = (7, -5, 13, 19, -9, 23)
        for i, cand in enumerate(task.candidate_rollouts):
            raw_a = cand.split("#A$#", 1)[1].strip() if "#A$#" in cand else cand.strip()
            if i == int(task.gold_index):
                bare_opts.append(gold_a if is_code else gold_template)
                continue
            if is_code:
                if raw_a.endswith("return 0") and not gold_a.endswith("return 0"):
                    if " + " in gold_a:
                        alt_code = gold_a.replace(" + ", " - ", 1)
                    elif " - " in gold_a:
                        alt_code = gold_a.replace(" - ", " + ", 1)
                    else:
                        alt_code = gold_a + " + 1"
                    bare_opts.append(alt_code)
                else:
                    bare_opts.append(raw_a)
                continue

            c_nums = [int(x) for x in re.findall(r"-?\d+", raw_a)]
            c_num = c_nums[-1] if c_nums else None
            if (
                gold_num is not None
                and len(gold_template.split()) >= 3
                and re.search(r"-?\d+", gold_template)
            ):
                if c_num is None or (raw_a.strip() == "0" and gold_num != 0) or c_num in used_nums:
                    for off in fallback_offsets:
                        cand_val = gold_num + off + i * 2
                        if cand_val not in used_nums and cand_val != 0:
                            c_num = cand_val
                            break
                    if c_num is None:
                        c_num = gold_num + 11 + i
                used_nums.add(int(c_num))
                matches = list(re.finditer(r"-?\d+", gold_template))
                last_m = matches[-1]
                norm_opt = (
                    gold_template[: last_m.start()]
                    + str(c_num)
                    + gold_template[last_m.end() :]
                )
                bare_opts.append(norm_opt)
            else:
                norm_raw = raw_a.lower().replace(",", "").strip()
                if norm_raw in ("0", "") and gold_template:
                    norm_raw = f"{gold_template} alternative {i + 1}"
                bare_opts.append(norm_raw)
        return tuple(" " + o.strip() for o in bare_opts)

    def step_self_improvement(
        self,
        coe: Any,
        *,
        batch_size: int = 2,
        max_seq_len: int | None = None,
    ) -> OrnithStepOutput:
        """Run one end-to-end Ornith-1.0 / 2.0 self-improvement step on `coe`."""
        from .data import DecisionExample, collate_decisions, get_tokenizer

        vocab_size = int(getattr(coe.backbone.config, "vocab_size", 260))
        if getattr(self.tokenizer, "vocab_size", 260) != vocab_size:
            self.tokenizer = get_tokenizer(vocab_size)

        eff_seq_len = max_seq_len or self.max_seq_len
        model_dev = next(coe.parameters()).device
        if self.device != model_dev:
            self.device = model_dev
            self.novelty_buffer.device = model_dev
            self.novelty_buffer._buffer = self.novelty_buffer._buffer.to(model_dev)

        num_exp = int(getattr(coe.backbone.config, "num_experts", 4))
        d_mod = int(getattr(coe.backbone.config, "d_model", 128))
        high_expert_mode = num_exp >= 256 or (num_exp * d_mod >= 16384)
        if high_expert_mode and batch_size > 1:
            batch_size = 1

        self._step_counter += 1
        use_compact = eff_seq_len < 220
        tasks = [
            self._generate_task(self._step_counter * batch_size + b, compact=use_compact)
            for b in range(batch_size)
        ]

        # Stage 0 (Dream-RSI): Online Discovery Tree Expansion + Offline Zero-Execution Replay Dreaming
        if (
            self.enable_dream_rsi
            and self.dream_engine is not None
            and (self._step_counter % self.dream_every == 0 or self._last_dream_out is None)
        ):
            dream_out = self.dream_engine.step_cycle(
                coe,
                tasks[0],
                max_seq_len=eff_seq_len,
            )
            self._last_dream_out = dream_out
            # Promote the Dream-RSI winning branch scaffold into ScaffoldCache and active task
            self.scaffold_cache.update_if_better(
                tasks[0].task_id,
                [dream_out.best_scaffold],
                [dream_out.best_node_score],
                step=self._step_counter,
            )
            tasks[0] = replace(tasks[0], scaffold_s=dream_out.best_scaffold)

        # Note: Solver context includes #U$#, #Q$#, and #S$# scaffold, NEVER the secret #H$# answer assertion!
        # Candidate descriptions use bare `#A$#` answers with matched formatting so COE must solve `prompt_q`.
        max_dec_len = max(eff_seq_len, 512)
        examples = [
            DecisionExample(
                question_id=t.task_id,
                kind="choice",
                context=f"{t.prompt_q}\n#S$# {t.scaffold_s}".encode("utf-8")[: max_dec_len - 4].decode("utf-8", errors="ignore"),
                descriptions=tuple(
                    d.encode("utf-8")[: max_dec_len - 4].decode("utf-8", errors="ignore")
                    for d in self._bare_matched_options(t)
                ),
                values=("opt_0", "opt_1", "opt_2", "opt_3"),
                target=t.gold_index,
            )
            for t in tasks
        ]
        batch = collate_decisions(
            examples,
            max_dec_len,
            device=self.device,
            tokenizer=self.tokenizer,
        )
        ctx_ids = batch["context_ids"]
        ctx_lens = batch["context_lengths"]
        opt_ids = batch["option_ids"]
        opt_lens = batch["option_lengths"]
        targets = batch["targets"]

        # 1. Stage 2 & 3: Evaluate all COE operators on the rollout group (single encoder pass)
        op_kwargs: dict[str, Any] = {}
        if high_expert_mode:
            op_kwargs = {
                "recurrence_steps": 1,
                "depth_ponder": False,
                "margin_halt_threshold": None,
            }
        op_out = coe.forward_all_operators(
            ctx_ids,
            ctx_lens,
            opt_ids,
            opt_lens,
            targets,
            **op_kwargs,
        )

        # 2. Stage 1 (#Q$#): Compute <0.1ms Association Novelty N(q) & update NoveltyTaskBuffer
        # Reuses `op_out.decision.state_features` so `ctx_ids` is encoded only once per step.
        with torch.no_grad():
            s_raw = op_out.decision.state_features.detach().float()
            s_feat = s_raw.view(batch_size, -1, s_raw.shape[-1])[:, 0, :]
            q_embs = F.normalize(s_feat - s_feat.mean(dim=-1, keepdim=True), dim=-1, eps=1e-6)
            pos = torch.arange(ctx_ids.size(1), device=self.device, dtype=torch.float32).unsqueeze(0)
            freq = torch.arange(1, q_embs.size(-1) + 1, device=self.device, dtype=torch.float32).view(1, 1, -1)
            tok_f = (ctx_ids.float() / 256.0).unsqueeze(-1)
            sketch = (torch.sin(tok_f * freq * 6.283185 + pos.unsqueeze(-1) * 0.17)).mean(dim=1)
            q_hybrid = F.normalize(q_embs, dim=-1) + 0.65 * F.normalize(sketch, dim=-1)
            novelties = self.novelty_buffer.compute_novelty(q_hybrid)
            self.novelty_buffer.add(q_hybrid)

        probs = op_out.decision.probabilities  # [B, 4]
        log_probs = probs.clamp_min(1e-7).log()
        beh_log_probs = log_probs.detach()

        # Build verified harness score tensor [B, 4] via sandboxed Python execution + reference scores
        verified_rows: list[list[float]] = []
        for t in tasks:
            row_sc: list[float] = []
            for cand, ref_s in zip(t.candidate_rollouts, t.harness_scores):
                v_s = verify_python_rollout(cand, t.harness_h, t.gold_answer)
                row_sc.append(0.5 * v_s + 0.5 * float(ref_s) if v_s > 0.0 else 0.0)
            verified_rows.append(row_sc)

        h_scores = torch.tensor(
            verified_rows,
            device=self.device,
            dtype=torch.float32,
        )
        # Deterministic boundary monitor on the 0-score hack rollout in each row
        neg_scores = h_scores.min(dim=-1, keepdim=True).values

        # Task alignment C(q, h) from Deduction's `entailed` probability (index 0)
        task_alignment = op_out.deduction_probs[:, 0].detach().clamp(0.25, 1.0)
        # Reference utilities from Evaluation head reshaped to [B, 4]
        ref_utils = op_out.evaluation_utilities.detach().view(batch_size, -1)

        r_harness_vec = compute_harness_reward(
            task_alignment,
            h_scores,
            ref_utils,
            negative_harness_scores=neg_scores,
            hack_resistance=1.0,
        )

        # In-network frozen COE consistency check (Deduction contradicted > 0.55 or Metacognition correctness < 0.35)
        contradicted_p = op_out.deduction_probs[:, 1].detach()
        correctness_p = op_out.metacognition_probs[:, 0].detach()
        vetoed = (contradicted_p > 0.55) | (correctness_p < 0.35)
        veto_rate = float(vetoed.float().mean().item())

        # Empirical pass rate p (probability mass assigned to the permuted gold_index rollout)
        pass_rates = probs.gather(1, targets.unsqueeze(1)).squeeze(1).detach()
        expected_rollout_vec = (probs.detach() * h_scores).sum(dim=-1)
        d_frontier = frontier_difficulty(pass_rates, target_p=0.20)
        validity = (r_harness_vec > 0.05).float()
        r_task_base = compute_task_reward(validity, pass_rates, novelties, target_p=0.20)

        # Submit plays to osu!-style Performance Point (`pp = p * 0.95^(n-1)`) Leaderboard,
        # update ScaffoldCache, and advance OrnithEnvArchitect difficulty frontier
        pp_raw_list: list[float] = []
        pp_weighted_list: list[float] = []
        pp_weight_list: list[float] = []
        pp_grad_scales: list[float] = []
        for b_idx, t in enumerate(tasks):
            cand_scaffolds = [
                _SCAFFOLD_CANDIDATES[(b_idx + k) % len(_SCAFFOLD_CANDIDATES)]
                for k in range(len(t.candidate_rollouts))
            ]
            weighted_r = (h_scores[b_idx] * probs[b_idx].detach()).tolist()
            self.scaffold_cache.update_if_better(
                t.task_id,
                cand_scaffolds,
                weighted_r,
                step=self._step_counter,
            )
            verified_play_acc = float(
                0.70 * expected_rollout_vec[b_idx].item() + 0.30 * pass_rates[b_idx].item()
            )
            play_raw_pp = compute_raw_pp(
                t.star_difficulty,
                verified_play_acc,
                alignment=float(task_alignment[b_idx].item()),
                mods=t.difficulty_mods,
                recursion_depth=t.recursion_depth,
            )
            pp_sub = self.pp_leaderboard.submit_play(
                t.skill,
                play_raw_pp,
                star_difficulty=t.star_difficulty,
                verified_reward=verified_play_acc,
            )
            pp_raw_list.append(pp_sub.raw_pp)
            pp_weighted_list.append(pp_sub.weighted_pp)
            pp_weight_list.append(pp_sub.skill_weight)
            pp_grad_scales.append(pp_sub.grad_scale)
            if t.task_id in self._last_task_kinds:
                self.env_architect.record_outcome(
                    self._last_task_kinds[t.task_id],
                    float(pass_rates[b_idx].item()),
                    float(task_alignment[b_idx].item()),
                    pp_weight=pp_sub.skill_weight,
                )

        pp_weights_t = torch.tensor(pp_weight_list, device=self.device, dtype=torch.float32)
        pp_grad_t = torch.tensor(pp_grad_scales, device=self.device, dtype=torch.float32)
        # Blend classical frontier task reward with PP-weighted star-progression reward (bounded in [0, 2])
        r_task_vec = (
            0.50 * r_task_base
            + 0.50 * validity * pp_weights_t * novelties.clamp(0.0, 2.0) * expected_rollout_vec.clamp(0.0, 1.0)
        ).clamp(0.0, 2.0)

        # Group-normalized advantages A_i across the G=4 rollouts scaled by osu! PP placement weight
        advantages = group_normalized_advantages(h_scores) * pp_grad_t.unsqueeze(-1)
        self._last_tasks = tasks
        self._last_advantages = advantages.detach()
        grpo_loss = grpo_clipped_loss(
            log_probs,
            beh_log_probs,
            advantages,
            policy_age=0,
        )
        # Verified gold rollout negative log-likelihood + 0x8000 prompt-masked decoder self-distillation.
        # Guard the autoregressive token decoder (forward_inference) against synthetic 1-line RL placeholder
        # thoughts ("Analyze science constraints...") and bare scalar answers ("9") so RL sharpens the
        # 9-operator COE heads without overwriting natural multi-sentence #T$# -> #A$# generation.
        per_row_nll = F.nll_loss(log_probs, targets, reduction="none")
        decision_distill_loss = (per_row_nll * pp_grad_t).mean()
        max_lm_len = min(eff_seq_len, int(getattr(coe.backbone.config, "max_seq_len", 256)))
        inp_rows: list[list[int]] = []
        lbl_rows: list[list[int]] = []
        prompt_lens: list[int] = []
        has_genuine_lm_target = False
        val_mask = int(getattr(self.tokenizer, "value_mask", 0x7FFF))
        is_hf_donor = bool(getattr(self.tokenizer, "is_hf_donor", False))
        for t in tasks:
            u_clean = t.prompt_q.split("\n#Q$#")[0].replace("#U$#", "").strip()
            raw_turn = self.tokenizer.encode_masked_turn(u_clean, t.gold_thought, t.gold_answer)
            if len(raw_turn) > max_lm_len + 1:
                raw_turn = raw_turn[: max_lm_len + 1]
            inp_rows.append([tok & val_mask for tok in raw_turn[:-1]])
            p_len = sum(1 for tok in raw_turn[1:] if (tok & self.tokenizer.prompt_mask_bit))
            prompt_lens.append(p_len)
            if use_compact or _is_genuine_natural_cot(t.gold_thought, t.gold_answer):
                row_lbl = [
                    -100
                    if (tok & self.tokenizer.prompt_mask_bit)
                    or (tok & val_mask) == self.tokenizer.pad_id
                    else (tok & val_mask)
                    for tok in raw_turn[1:]
                ]
                if any(x != -100 for x in row_lbl):
                    has_genuine_lm_target = True
                lbl_rows.append(row_lbl)
            elif _is_genuine_complete_answer(t.gold_answer):
                # Supervise only the verified #A$# code/sentence answer while masking the synthetic #T$# stub
                ut_prefix = f"#U$# {u_clean}\n#T$# {t.gold_thought.strip()}\n"
                ut_len = len(self.tokenizer.encode(ut_prefix, add_bos=True, add_eos=False))
                row_lbl = [
                    -100
                    if (idx_tok + 1 < ut_len)
                    or (tok & self.tokenizer.prompt_mask_bit)
                    or (tok & val_mask) == self.tokenizer.pad_id
                    else (tok & val_mask)
                    for idx_tok, tok in enumerate(raw_turn[1:])
                ]
                if any(x != -100 for x in row_lbl):
                    has_genuine_lm_target = True
                lbl_rows.append(row_lbl)
            else:
                lbl_rows.append([-100] * (len(raw_turn) - 1))
        pad_len = max(len(r) for r in inp_rows)
        inp_tensor = torch.tensor(
            [r + [self.tokenizer.pad_id] * (pad_len - len(r)) for r in inp_rows],
            dtype=torch.long,
            device=self.device,
        )
        lbl_tensor = torch.tensor(
            [r + [-100] * (pad_len - len(r)) for r in lbl_rows],
            dtype=torch.long,
            device=self.device,
        )
        inf_kwargs = {"recurrence_steps": 1, "depth_ponder": False} if high_expert_mode else {}
        if has_genuine_lm_target:
            lm_distill_out = coe.forward_inference(inp_tensor, targets=lbl_tensor, **inf_kwargs)
            assert lm_distill_out.loss is not None
            self_distill_loss = 0.85 * decision_distill_loss + 0.15 * lm_distill_out.loss
        else:
            lm_distill_out = None
            self_distill_loss = decision_distill_loss

        # LiquidAI Antidoom FTPO (supports ByteTokenizer, MorphCode-16K, and DonorHFTokenizer):
        # Strictly target inner repetition triggers (`loop_hit is not None`) during full-length training.
        ftpo_pos_logits: list[Tensor] = []
        ftpo_rej_ids: list[int] = []
        ftpo_cho_ids: list[list[int]] = []
        if lm_distill_out is not None:
            with torch.no_grad():
                if is_hf_donor:
                    low_s, low_e = getattr(self.tokenizer, "invalid_logit_slice_low", (1, 19))
                    high_s = int(getattr(self.tokenizer, "invalid_logit_start_high", 49152))
                    masked_l = lm_distill_out.logits.detach().clone()
                    masked_l[:, :, low_s:low_e] = -torch.inf
                    if vocab_size > high_s:
                        masked_l[:, :, high_s:] = -torch.inf
                    pred_ids = masked_l.argmax(dim=-1)
                elif vocab_size > 1024:
                    masked_l = lm_distill_out.logits.detach().clone()
                    masked_l[:, :, :10] = -torch.inf
                    masked_l[:, :, 11:32] = -torch.inf
                    masked_l[:, :, 127:260] = -torch.inf
                    pred_ids = masked_l.argmax(dim=-1)
                else:
                    pred_ids = lm_distill_out.logits[:, :, :256].argmax(dim=-1)
            for b_i in range(inp_tensor.size(0)):
                valid_pos = (lbl_tensor[b_i] != -100).nonzero(as_tuple=False).squeeze(-1)
                if valid_pos.numel() < 4:
                    start_p = min(prompt_lens[b_i], inp_tensor.size(1) - 1)
                    end_p = len(inp_rows[b_i])
                    if end_p - start_p < 4:
                        continue
                    valid_pos = torch.arange(start_p, end_p, dtype=torch.long, device=self.device)
                seq_preds = pred_ids[b_i, valid_pos].tolist()
                loop_hit = detect_doom_loop(seq_preds, min_period=3, max_period=48, min_repeats=2, min_total_repeated=6)
                if loop_hit is not None:
                    trig_local, rej_tok, _ = loop_hit
                    pos_idx = int(valid_pos[min(trig_local, valid_pos.numel() - 1)].item())
                    gold_tok = int(inp_tensor[b_i, min(pos_idx + 1, inp_tensor.size(1) - 1)].item())
                    if lbl_tensor[b_i, pos_idx].item() != -100:
                        gold_tok = int(lbl_tensor[b_i, pos_idx].item())
                elif use_compact and (lbl_tensor[b_i, valid_pos] != -100).any():
                    mismatches = (pred_ids[b_i, valid_pos] != lbl_tensor[b_i, valid_pos]).nonzero(as_tuple=False).squeeze(-1)
                    if mismatches.numel() == 0:
                        continue
                    first_err_local = int(mismatches[0].item())
                    pos_idx = int(valid_pos[first_err_local].item())
                    rej_tok = int(pred_ids[b_i, pos_idx].item())
                    gold_tok = int(lbl_tensor[b_i, pos_idx].item())
                else:
                    continue
                if rej_tok == gold_tok:
                    continue
                if is_hf_donor:
                    low_s, low_e = getattr(self.tokenizer, "invalid_logit_slice_low", (1, 19))
                    high_s = int(getattr(self.tokenizer, "invalid_logit_start_high", 49152))
                    row_cand = lm_distill_out.logits[b_i, pos_idx].detach().clone()
                    row_cand[low_s:low_e] = -torch.inf
                    if vocab_size > high_s:
                        row_cand[high_s:] = -torch.inf
                    row_cand[rej_tok] = -torch.inf
                    top_alts = row_cand.topk(4).indices.tolist()
                elif vocab_size > 1024:
                    row_cand = lm_distill_out.logits[b_i, pos_idx].detach().clone()
                    row_cand[:32] = -torch.inf
                    row_cand[127:260] = -torch.inf
                    row_cand[rej_tok] = -torch.inf
                    top_alts = row_cand.topk(4).indices.tolist()
                else:
                    row_ascii = lm_distill_out.logits[b_i, pos_idx, 32:127].detach().clone()
                    if 32 <= rej_tok < 127:
                        row_ascii[rej_tok - 32] = -torch.inf
                    top_alts = (row_ascii.topk(4).indices + 32).tolist()
                chosen_row = [gold_tok] + [t_id for t_id in top_alts if t_id != gold_tok][:3]
                while len(chosen_row) < 4:
                    chosen_row.append(gold_tok)
                ftpo_pos_logits.append(lm_distill_out.logits[b_i, pos_idx])
                ftpo_rej_ids.append(rej_tok)
                ftpo_cho_ids.append(chosen_row[:4])

        if ftpo_pos_logits:
            ftpo_logits_t = torch.stack(ftpo_pos_logits, dim=0)
            ftpo_rej_t = torch.tensor(ftpo_rej_ids, dtype=torch.long, device=self.device)
            ftpo_cho_t = torch.tensor(ftpo_cho_ids, dtype=torch.long, device=self.device)
            antidoom_loss_t, antidoom_win = compute_antidoom_ftpo_loss(
                ftpo_logits_t,
                ftpo_logits_t.detach(),
                ftpo_rej_t,
                ftpo_cho_t,
            )
        else:
            antidoom_loss_t = torch.zeros((), device=self.device)
            antidoom_win = 1.0

        total_ornith_loss = (
            op_out.total_loss
            + 0.5 * grpo_loss
            + 0.25 * self_distill_loss
            + 0.15 * antidoom_loss_t
        )
        mean_rollout_r = float((probs.detach() * h_scores).sum(dim=-1).mean().item())
        d_out = self._last_dream_out

        mean_raw_pp = sum(pp_raw_list) / max(1, len(pp_raw_list))
        mean_weighted_pp = sum(pp_weighted_list) / max(1, len(pp_weighted_list))
        mean_pp_weight = sum(pp_weight_list) / max(1, len(pp_weight_list))
        mean_stars = sum(t.star_difficulty for t in tasks) / max(1, len(tasks))
        max_rec_depth = max((t.recursion_depth for t in tasks), default=1)

        return OrnithStepOutput(
            loss=total_ornith_loss,
            grpo_loss=grpo_loss.detach(),
            self_distill_loss=self_distill_loss.detach(),
            task_reward=round(float(r_task_vec.mean().item()), 4),
            harness_reward=round(float(r_harness_vec.mean().item()), 4),
            rollout_reward=round(mean_rollout_r, 4),
            novelty=round(float(novelties.mean().item()), 4),
            pass_rate=round(float(pass_rates.mean().item()), 4),
            frontier_difficulty=round(float(d_frontier.mean().item()), 4),
            veto_rate=round(veto_rate, 4),
            scaffolds_cached=len(self.scaffold_cache._entries),
            novelty_buffer_size=len(self.novelty_buffer),
            dream_cycle=0 if d_out is None else int(d_out.cycle),
            dream_replay_score=0.0 if d_out is None else float(d_out.final_replay_score),
            dream_initial_score=0.0 if d_out is None else float(d_out.initial_replay_score),
            dream_accepted_revisions=0 if d_out is None else int(d_out.total_accepted_revisions),
            dream_replays_dreamed=0 if d_out is None else int(d_out.total_replays_dreamed),
            dream_worlds_archived=0 if d_out is None else int(d_out.worlds_archived),
            dream_active_policy="" if d_out is None else str(d_out.active_policy_text),
            antidoom_loss=round(float(antidoom_loss_t.detach().item()), 4),
            antidoom_chosen_win=round(float(antidoom_win), 4),
            raw_pp=round(float(mean_raw_pp), 4),
            weighted_pp=round(float(mean_weighted_pp), 4),
            total_pp=round(float(self.pp_leaderboard.total_pp()), 4),
            pp_weight=round(float(mean_pp_weight), 5),
            star_difficulty=round(float(mean_stars), 4),
            recursion_depth=int(max_rec_depth),
            active_skill=str(tasks[0].skill) if tasks else "",
        )

    def step_ncm_self_improvement(
        self,
        ncm_system: Any,
        batch_size: int = 2,
        max_seq_len: int | None = None,
    ) -> OrnithStepOutput:
        """Run an Ornith-2.5 (`pp = p * 0.95^(n-1)`) + Dream-RSI hierarchical RL step on `NCMSystem`.

        Combines:
        1. Full 3-stage `Ornith-2.5` + `Dream-RSI` + 9-operator `COE` supervision on `ncm_system.coe_model`.
        2. Single-pass exact hierarchical Next-Concept log-likelihoods (`compute_hierarchy_log_probs`,
           Eq. 13) across candidate rollouts for coarse-to-fine GRPO policy optimization.
        3. Multiscale NCM coarse-to-fine self-distillation (`forward_training_step`) on verified
           gold `#T$# -> #A$#` continuations.
        """
        if ncm_system.coe_model is None:
            raise ValueError("step_ncm_self_improvement requires ncm_system.coe_model to be attached")

        base_out = self.step_self_improvement(
            ncm_system.coe_model,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        tasks = getattr(self, "_last_tasks", [])
        advantages = getattr(self, "_last_advantages", None)
        if not tasks:
            return base_out

        ncm_seq_len = int(ncm_system.config.seq_len)
        pad_id = int(ncm_system.config.pad_id)
        prompt_rows: list[list[int]] = []
        gold_cont_rows: list[list[int]] = []
        group_cont_rows: list[list[list[int]]] = []

        for t in tasks:
            p_toks = self.tokenizer.encode(
                f"{t.prompt_q}\n#S$# {t.scaffold_s}", add_bos=True, add_eos=False
            )[: max(8, ncm_seq_len * 2)]
            prompt_rows.append(p_toks)

            gold_text = t.candidate_rollouts[int(t.gold_index)]
            g_toks = self.tokenizer.encode(gold_text, add_bos=False, add_eos=True)[:ncm_seq_len]
            if len(g_toks) < ncm_seq_len:
                g_toks = g_toks + [pad_id] * (ncm_seq_len - len(g_toks))
            gold_cont_rows.append(g_toks)

            c_group: list[list[int]] = []
            for cand in self._bare_matched_options(t):
                c_toks = self.tokenizer.encode(cand, add_bos=False, add_eos=True)[:ncm_seq_len]
                if len(c_toks) < ncm_seq_len:
                    c_toks = c_toks + [pad_id] * (ncm_seq_len - len(c_toks))
                c_group.append(c_toks)
            group_cont_rows.append(c_group)

        max_p = max(len(r) for r in prompt_rows)
        prompt_tensor = torch.tensor(
            [r + [pad_id] * (max_p - len(r)) for r in prompt_rows],
            dtype=torch.long,
            device=self.device,
        )
        gold_cont_tensor = torch.tensor(
            gold_cont_rows,
            dtype=torch.long,
            device=self.device,
        )

        # 1. Multiscale coarse-to-fine NCM self-distillation on verified gold continuations
        ncm_distill_out = ncm_system.forward_training_step(
            prompt_tensor,
            gold_cont_tensor,
            train_vqvae_jointly=False,
            include_coe_abstraction_loss=True,
        )

        # 2. Hierarchical Next-Concept GRPO across candidate rollouts using single-pass log-probs
        bsz = len(tasks)
        num_cands = len(group_cont_rows[0])
        p_valid = (
            (prompt_tensor != pad_id)
            if pad_id >= 0
            else torch.ones_like(prompt_tensor, dtype=torch.bool)
        )
        p_valid = p_valid.clone()
        p_valid[:, 0] = True
        enc_prompt = ncm_system.backbone.encode_state(prompt_tensor, p_valid, causal=True)
        owner_rows = torch.arange(bsz, device=self.device, dtype=torch.long).repeat_interleave(
            num_cands
        )
        flat_cands = torch.tensor(
            group_cont_rows, dtype=torch.long, device=self.device
        ).reshape(bsz * num_cands, ncm_seq_len)

        hier_lp_flat, _ = ncm_system.compute_hierarchy_log_probs(
            enc_prompt,
            flat_cands,
            owner_rows=owner_rows,
            normalize_by_tokens=True,
        )
        hier_lp = hier_lp_flat.view(bsz, num_cands)
        if advantages is not None and advantages.shape == hier_lp.shape:
            ncm_grpo_loss = grpo_clipped_loss(
                hier_lp,
                hier_lp.detach(),
                advantages,
                policy_age=0,
            )
        else:
            ncm_grpo_loss = torch.zeros((), device=self.device)

        combined_loss = (
            base_out.loss
            + 0.15 * ncm_distill_out.total_loss
            + 0.15 * ncm_grpo_loss
        )
        return replace(
            base_out,
            loss=combined_loss,
            grpo_loss=(base_out.grpo_loss + 0.25 * ncm_grpo_loss.detach()),
            self_distill_loss=(base_out.self_distill_loss + 0.50 * ncm_distill_out.total_loss.detach()),
        )



