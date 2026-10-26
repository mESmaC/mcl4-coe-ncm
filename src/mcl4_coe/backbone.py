"""Shared MCL4-CED MoE Backbone with Per-Layer Embeddings and Recurrent Feedback.

Combines:
1. Lower Causal Encoder Layers (`n_encoder_layers`):
   Process the shared conditioning state/prefix once using MCL4 sparse graph
   attention, Per-Layer Embeddings (PLE), and dropless top-k SwiGLU MoE.
2. Upper Causal Encoder-Decoder (CED) Layers (`n_decoder_layers`):
   Process candidate options, operator queries, or `T$ -> A$` tokens using:
   - Local MCL4 attention (restarting RoPE at position 0 per candidate so candidate
     evaluation is 100% bitwise permutation-equivariant), and
   - Global Cross-Attention into the lower encoder's cached K/V projections
     (computed once per state row and broadcast across candidates).
3. RecurrentFeedback (`anchor + sigmoid(gate) * LayerNorm(state)`):
   Enables multi-pass latent refinement across Cognitive Operators without
   leaking future tokens or breaking candidate permutation invariance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .markov_local4 import MarkovLocal4Router
from .nvfp4 import NVFP4Linear, has_triton_nvfp4_grouped, nvfp4_grouped_linear
from .ple import PerLayerEmbedding, RMSNorm


@dataclass
class MCL4CEDConfig:
    vocab_size: int = 260
    pad_id: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    ple_dim: int = 64
    expert_hidden_dim: int = 512
    num_experts: int = 4
    top_k: int = 2
    max_seq_len: int = 512
    dropout: float = 0.0
    aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    attention_type: str = "markov_local4"
    markov_router_dim: int = 16
    markov_graph_degree: int = 4
    markov_num_walkers: int = 4
    markov_steps: int = 3
    markov_attend_tokens: int = 24
    markov_local_tokens: int = 8
    markov_beam_size: int = 8
    markov_restart: float = 0.1
    markov_query_chunk_size: int = 1024
    eos_token_id: int | None = None
    use_qk_norm: bool = True
    qk_norm_mode: str = "head"
    norm_topk_prob: bool = True
    tie_word_embeddings: bool | None = None
    attention_multiplier: float | None = None
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0
    preserve_residual_across_stacks: bool = False
    recurrent_coverage_penalty: float = 0.0
    mitotic_ortho_coef: float = 0.0
    sequence_diversity_coef: float = 0.0
    dense_expert_indices: tuple[int, ...] = ()
    dense_subset_k: int = 0
    combine_dense_and_sparse: bool = False
    routed_scaling_factor: float = 1.0
    rope_scaling: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.dense_expert_indices, list):
            self.dense_expert_indices = tuple(int(i) for i in self.dense_expert_indices)
        else:
            self.dense_expert_indices = tuple(int(i) for i in (self.dense_expert_indices or ()))
        for name in (
            "vocab_size",
            "d_model",
            "n_heads",
            "n_encoder_layers",
            "n_decoder_layers",
            "expert_hidden_dim",
            "num_experts",
            "top_k",
            "max_seq_len",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.dense_subset_k < 0 or self.dense_subset_k > self.num_experts:
            raise ValueError("dense_subset_k must be in [0, num_experts]")
        for idx in self.dense_expert_indices:
            if idx < 0 or idx >= self.num_experts:
                raise ValueError(f"dense_expert_indices entry {idx} out of range [0, {self.num_experts})")
        if self.attention_type not in ("markov_local4", "standard"):
            raise ValueError("attention_type must be 'markov_local4' or 'standard'")
        if self.qk_norm_mode not in ("head", "full"):
            raise ValueError("qk_norm_mode must be 'head' or 'full'")

    @property
    def eos_id(self) -> int:
        if self.eos_token_id is not None:
            return int(self.eos_token_id)
        return int(self.pad_id) + 2


@dataclass
class EncodedState:
    """Cached lower-encoder state and MoE routing statistics."""

    hidden: Tensor
    valid: Tensor
    aux_loss: Tensor
    router_z_loss: Tensor
    expert_load: Tensor
    gist_prefix_len: int = 0
    raw_hidden: Tensor | None = None
    state_ids: Tensor | None = None


@dataclass
class BackboneOutput:
    """Output of a full or candidate-decoded pass through MCL4CEDBackbone."""

    hidden: Tensor
    valid: Tensor
    aux_loss: Tensor
    router_z_loss: Tensor
    expert_load: Tensor
    anchor: Tensor | None = None
    step_states: tuple[Tensor, ...] = ()
    ctx_last_hidden: Tensor | None = None


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None
    ce_loss: Tensor | None
    aux_loss: Tensor
    router_z_loss: Tensor
    expert_load: Tensor
    step_states: tuple[Tensor, ...] = ()


class RecurrentFeedback(nn.Module):
    """Token-local recurrent state refinement adapter (`anchor + gate * LN(state)`)."""

    def __init__(self, dim: int, *, zero_init_tanh: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(()))
        self.zero_init_tanh = bool(zero_init_tanh)

    def forward(self, anchor: Tensor, state: Tensor, valid: Tensor | None = None) -> Tensor:
        if anchor.shape != state.shape or anchor.device != state.device:
            raise ValueError("anchor and state must match in shape and device")
        normed = self.norm(state.to(self.norm.weight.dtype)).to(anchor.dtype)
        g = self.gate.to(anchor.dtype)
        gate_val = torch.tanh(g) if self.zero_init_tanh else g.sigmoid()
        updated = anchor + gate_val * normed
        if valid is not None:
            updated = updated.masked_fill(~valid.unsqueeze(-1), 0)
        return updated


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * value)


class SparseMoE(nn.Module):
    """Dropless top-k SwiGLU MoE with FP32 routing and load-balance penalties.

    Validates token masks so right-padding tokens never skew expert load statistics
    or corrupt router z-loss.
    """

    def __init__(self, config: MCL4CEDConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        self.recurrent_coverage_penalty = float(
            getattr(config, "recurrent_coverage_penalty", 0.0)
        )
        self.mitotic_ortho_coef = float(getattr(config, "mitotic_ortho_coef", 0.0))
        self.sequence_diversity_coef = float(
            getattr(config, "sequence_diversity_coef", 0.0)
        )
        self.dense_expert_indices: tuple[int, ...] = tuple(
            int(i) for i in getattr(config, "dense_expert_indices", ())
        )
        self.dense_subset_k: int = int(getattr(config, "dense_subset_k", 0))
        self.combine_dense_and_sparse: bool = bool(
            getattr(config, "combine_dense_and_sparse", False)
        )
        self.routed_scaling_factor: float = float(
            getattr(config, "routed_scaling_factor", 1.0)
        )
        self.router = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLUExpert(config.d_model, config.expert_hidden_dim)
            for _ in range(config.num_experts)
        )
        self.mitotic_pairs: list[tuple[int, int]] = []
        self._mitotic_idx_cache: tuple[Any, Tensor, Tensor] | None = None
        self._grouped_weight_cache: tuple[Any, Tensor, Tensor] | None = None
        self.last_selected_mask: Tensor | None = None
        self.lineage_num_roots: int = 0
        self.lineage_top_families: int = 0
        self.lineage_child_share: float = 0.25
        self.lineage_root_ids: Tensor | None = None
        self.lineage_logit_prior: Tensor | None = None

    @torch.no_grad()
    def configure_lineage_family_routing(
        self,
        num_roots: int = 64,
        *,
        top_families: int = 8,
        child_share: float = 0.25,
    ) -> None:
        """Configure parent-anchored lineage-calibrated family routing for multi-generation mitotic MoE layers."""
        if num_roots <= 0 or top_families <= 0 or num_roots >= self.num_experts:
            self.lineage_num_roots = 0
            self.lineage_top_families = 0
            self.lineage_root_ids = None
            self.lineage_logit_prior = None
            return
        dev = self.router.weight.device
        ups = torch.stack(
            [e.up_proj.weight.detach().float().reshape(-1) for e in self.experts], dim=0
        )
        ups = F.normalize(ups, p=2.0, dim=-1, eps=1e-8)
        sim_roots = ups[num_roots:] @ ups[:num_roots].T
        child_roots = sim_roots.argmax(dim=-1)
        root_ids = torch.cat(
            [torch.arange(num_roots, dtype=torch.long, device=dev), child_roots.to(dev)],
            dim=0,
        )
        fam_counts = torch.bincount(root_ids, minlength=num_roots)
        cs = float(max(0.0, min(0.95, child_share)))
        prior_w = torch.ones(self.num_experts, dtype=torch.float32, device=dev)
        for e_idx in range(self.num_experts):
            r_idx = int(root_ids[e_idx].item())
            m_r = int(fam_counts[r_idx].item())
            if m_r > 1:
                if e_idx < num_roots:
                    prior_w[e_idx] = max(1e-6, 1.0 - cs)
                else:
                    prior_w[e_idx] = max(1e-6, cs / float(m_r - 1))
        self.lineage_num_roots = int(num_roots)
        self.lineage_top_families = int(min(top_families, num_roots))
        self.lineage_child_share = cs
        self.lineage_root_ids = root_ids
        self.lineage_logit_prior = torch.log(prior_w)

    def _route_lineage_families(
        self,
        router_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute calibrated probabilities, selected expert indices, and routing weights across root families."""
        assert self.lineage_root_ids is not None and self.lineage_logit_prior is not None
        dev = router_logits.device
        if self.lineage_root_ids.device != dev:
            self.lineage_root_ids = self.lineage_root_ids.to(dev)
        if self.lineage_logit_prior.device != dev:
            self.lineage_logit_prior = self.lineage_logit_prior.to(dev)
        root_ids = self.lineage_root_ids
        adj_logits = router_logits + self.lineage_logit_prior.unsqueeze(0)
        probabilities = F.softmax(adj_logits, dim=-1)
        n_tok = router_logits.shape[0]
        fam_probs = router_logits.new_zeros((n_tok, self.lineage_num_roots))
        fam_probs.scatter_add_(
            1, root_ids.unsqueeze(0).expand(n_tok, -1), probabilities
        )
        _, top_fams = fam_probs.topk(self.lineage_top_families, dim=-1)
        fam_mask = router_logits.new_zeros((n_tok, self.lineage_num_roots))
        fam_mask.scatter_(1, top_fams, 1.0)
        exp_mask = fam_mask.index_select(1, root_ids)
        masked_probs = probabilities * exp_mask
        max_active = (
            int(exp_mask.sum(dim=-1).max().item()) if n_tok > 0 else self.top_k
        )
        k_sel = max(self.top_k, min(self.num_experts, max_active))
        weights, selected = masked_probs.topk(k_sel, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return probabilities, weights, selected

    def forward_dense_subset(
        self,
        x: Tensor,
        expert_indices: Sequence[int] | Tensor,
        *,
        expert_weights: Tensor | None = None,
        valid: Tensor | None = None,
    ) -> Tensor:
        """Execute a selected subset of `M` experts as a single fused wide-dense SwiGLU layer of width `M * d_ffn`.

        Because SwiGLU is linear in `down_proj`, concatenating `up_proj` (`[M * 2 * d_ffn, d_model]`)
        and `down_proj` (`[d_model, M * d_ffn]`) across the `M` selected experts is mathematically
        identical to a wide dense FFN of hidden dimension `M * d_ffn` executed in just 2 contiguous GEMMs.
        """
        if isinstance(expert_indices, torch.Tensor):
            idx_list = [int(i) for i in expert_indices.detach().cpu().reshape(-1).tolist()]
        else:
            idx_list = [int(i) for i in expert_indices]
        if not idx_list:
            return torch.zeros_like(x)
        m_sub = len(idx_list)
        flat = x.reshape(-1, x.shape[-1])
        n_tok = flat.shape[0]
        if n_tok == 0:
            return torch.zeros_like(x)

        can_fuse = all(
            type(self.experts[i].up_proj) is nn.Linear
            and type(self.experts[i].down_proj) is nn.Linear
            for i in idx_list
        )
        if can_fuse:
            up_cat = torch.cat(
                [self.experts[i].up_proj.weight for i in idx_list], dim=0
            ).to(flat.dtype)
            down_cat = torch.cat(
                [self.experts[i].down_proj.weight for i in idx_list], dim=1
            ).to(flat.dtype)
            up_out = F.linear(flat, up_cat).view(n_tok, m_sub, 2, -1)
            act = F.silu(up_out[:, :, 0, :]) * up_out[:, :, 1, :]
            if expert_weights is not None:
                w = expert_weights.to(act.dtype)
                if w.ndim == 1:
                    w = w.view(1, m_sub, 1)
                elif w.ndim == 2:
                    w = w.reshape(-1, m_sub, 1)
                act = act * w
            out = F.linear(act.reshape(n_tok, -1), down_cat).view_as(x)
        elif (
            self._can_use_grouped_dispatch(flat)
            and isinstance(self.experts[0].up_proj, NVFP4Linear)
            and len(set(idx_list)) == m_sub
        ):
            sorted_pairs = sorted(enumerate(idx_list), key=lambda pair: pair[1])
            orig_order = [pos for pos, _ in sorted_pairs]
            counts_list = [0] * self.num_experts
            for _, e_idx in sorted_pairs:
                counts_list[e_idx] = n_tok
            offs_sub = torch.tensor(
                counts_list, dtype=torch.int32, device=flat.device
            ).cumsum(0, dtype=torch.int32)
            flat_rep = flat.repeat(m_sub, 1)
            out_rep = (
                self._run_grouped_experts(flat_rep, offs_sub)
                .view(m_sub, n_tok, -1)
                .transpose(0, 1)
                .to(flat.dtype)
            )
            if expert_weights is not None:
                w = expert_weights.to(out_rep.dtype)
                if w.ndim == 1:
                    w = w[orig_order].view(1, m_sub, 1)
                else:
                    w = w.reshape(-1, m_sub)[:, orig_order].unsqueeze(-1)
                out_rep = out_rep * w
            out = out_rep.sum(dim=1).view_as(x)
        else:
            acc = torch.zeros_like(flat)
            for m_pos, e_idx in enumerate(idx_list):
                e_out = self.experts[e_idx](flat)
                if expert_weights is not None:
                    w = expert_weights.to(e_out.dtype)
                    if w.ndim == 1:
                        e_out = e_out * w[m_pos]
                    else:
                        e_out = e_out * w.reshape(-1, m_sub)[:, m_pos : m_pos + 1]
                acc = acc + e_out
            out = acc.view_as(x)

        if valid is not None:
            out = out.masked_fill(~valid.unsqueeze(-1), 0)
        return out

    def compute_mitotic_ortho_loss(self, device: torch.device) -> Tensor:
        """Compute chordal orthogonality loss `max(0, sqrt(2) - ||u_p - u_c||_2) / sqrt(2)` across mitotic pairs.

        Unlike `cos^2(theta)` whose gradient vanishes as `O(theta)` at birth (`theta ~ 1e-3`),
        the chordal distance to orthogonality (`||u_p - u_c||_2 = sqrt(2)` when `u_p ⊥ u_c`)
        maintains unit-strength repulsive gradients along the Gram-Schmidt split direction right from birth.
        """
        if not self.mitotic_pairs:
            return torch.zeros((), dtype=torch.float32, device=device)
        valid_pairs = [
            (p_idx, c_idx)
            for p_idx, c_idx in self.mitotic_pairs
            if 0 <= p_idx < self.num_experts and 0 <= c_idx < self.num_experts and p_idx != c_idx
        ]
        if torch.is_grad_enabled():
            grad_pairs = [
                (p_idx, c_idx)
                for p_idx, c_idx in valid_pairs
                if self.experts[p_idx].up_proj.weight.requires_grad
                or self.experts[c_idx].up_proj.weight.requires_grad
            ]
            if grad_pairs:
                valid_pairs = grad_pairs
        if not valid_pairs:
            return torch.zeros((), dtype=torch.float32, device=device)
        max_pairs = 2 if (self.num_experts * self.router.in_features >= 65536) else 16
        if len(valid_pairs) > max_pairs:
            offset = int(getattr(self, "_ortho_pair_cursor", 0)) % len(valid_pairs)
            self._ortho_pair_cursor = offset + max_pairs
            doubled = valid_pairs + valid_pairs
            valid_pairs = doubled[offset : offset + max_pairs]
        sqrt2 = 1.4142135623730951
        up_p = F.normalize(
            torch.stack([self.experts[p].up_proj.weight for p, _ in valid_pairs], dim=0).float(),
            p=2.0,
            dim=-1,
            eps=1e-8,
        )
        up_c = F.normalize(
            torch.stack([self.experts[c].up_proj.weight for _, c in valid_pairs], dim=0).float(),
            p=2.0,
            dim=-1,
            eps=1e-8,
        )
        loss_up = ((sqrt2 - (up_p - up_c).norm(p=2.0, dim=-1)) / sqrt2).clamp_min(0.0).mean()

        dn_p = F.normalize(
            torch.stack([self.experts[p].down_proj.weight for p, _ in valid_pairs], dim=0).float(),
            p=2.0,
            dim=1,
            eps=1e-8,
        )
        dn_c = F.normalize(
            torch.stack([self.experts[c].down_proj.weight for _, c in valid_pairs], dim=0).float(),
            p=2.0,
            dim=1,
            eps=1e-8,
        )
        loss_dn = ((sqrt2 - (dn_p - dn_c).norm(p=2.0, dim=1)) / sqrt2).clamp_min(0.0).mean()
        return 0.5 * (loss_up + loss_dn)

    def _compute_sequence_diversity_loss(
        self,
        probabilities_bt: Tensor,
        valid: Tensor | None,
    ) -> Tensor:
        """Penalize within-sequence expert collapse so tokens across a sequence recruit diverse experts."""
        if probabilities_bt.ndim != 3 or probabilities_bt.shape[1] <= 1:
            return probabilities_bt.new_zeros(())
        if valid is not None:
            v_bt = valid.to(probabilities_bt.dtype).unsqueeze(-1)
            denom = v_bt.sum(dim=1).clamp_min(1.0)
            seq_probs = (probabilities_bt * v_bt).sum(dim=1) / denom
            row_valid = (valid.any(dim=1)).to(probabilities_bt.dtype)
            n_rows = row_valid.sum().clamp_min(1.0)
            collision = (self.num_experts * seq_probs.square().sum(dim=-1) - 1.0).clamp_min(0.0)
            return (collision * row_valid).sum() / n_rows
        seq_probs = probabilities_bt.mean(dim=1)
        return (self.num_experts * seq_probs.square().sum(dim=-1).mean() - 1.0).clamp_min(0.0)

    def _get_mitotic_pair_tensors(self, device: torch.device) -> tuple[Tensor, Tensor] | None:
        if not self.mitotic_pairs:
            return None
        key = (tuple(self.mitotic_pairs), self.num_experts, device)
        cached = self._mitotic_idx_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        valid_pairs = [
            (p_idx, c_idx)
            for p_idx, c_idx in self.mitotic_pairs
            if 0 <= p_idx < self.num_experts and 0 <= c_idx < self.num_experts and p_idx != c_idx
        ]
        if not valid_pairs:
            return None
        p_t = torch.tensor([p for p, _ in valid_pairs], dtype=torch.long, device=device)
        c_t = torch.tensor([c for _, c in valid_pairs], dtype=torch.long, device=device)
        self._mitotic_idx_cache = (key, p_t, c_t)
        return p_t, c_t

    def _get_grouped_expert_weights(self, device: torch.device) -> tuple[Tensor, Tensor]:
        if torch.is_grad_enabled():
            self._grouped_weight_cache = None
            w_up = (
                torch.stack([e.up_proj.weight for e in self.experts], dim=0)
                .to(torch.bfloat16)
                .transpose(-1, -2)
            )
            w_down = (
                torch.stack([e.down_proj.weight for e in self.experts], dim=0)
                .to(torch.bfloat16)
                .transpose(-1, -2)
            )
            return w_up, w_down
        key = (
            len(self.experts),
            device,
            tuple(e.up_proj.weight._version for e in self.experts),
            tuple(e.down_proj.weight._version for e in self.experts),
            self.experts[0].up_proj.weight.data_ptr(),
            self.experts[-1].up_proj.weight.data_ptr(),
        )
        cached = self._grouped_weight_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        w_up = (
            torch.stack([e.up_proj.weight.to(torch.bfloat16) for e in self.experts], dim=0)
            .transpose(-1, -2)
            .contiguous()
        )
        w_down = (
            torch.stack([e.down_proj.weight.to(torch.bfloat16) for e in self.experts], dim=0)
            .transpose(-1, -2)
            .contiguous()
        )
        self._grouped_weight_cache = (key, w_up, w_down)
        return w_up, w_down

    def _can_use_nvfp4_grouped(self, device: torch.device) -> bool:
        if not self.experts or not has_triton_nvfp4_grouped(device):
            return False
        e0_up = self.experts[0].up_proj
        e0_dn = self.experts[0].down_proj
        return (
            isinstance(e0_up, NVFP4Linear)
            and isinstance(e0_dn, NVFP4Linear)
            and e0_up.in_features % 128 == 0
            and e0_up.out_features % 128 == 0
            and e0_dn.in_features % 128 == 0
            and e0_dn.out_features % 128 == 0
            and all(
                isinstance(e.up_proj, NVFP4Linear)
                and isinstance(e.down_proj, NVFP4Linear)
                and e.up_proj.bias is None
                and e.down_proj.bias is None
                for e in self.experts
            )
        )

    def _can_use_grouped_dispatch(self, flat: Tensor) -> bool:
        if (
            not flat.is_cuda
            or flat.shape[0] == 0
            or (self.num_experts * flat.shape[-1] >= 65536)
        ):
            return False
        if not (flat.dtype == torch.bfloat16 or torch.is_autocast_enabled()):
            return False
        if self._can_use_nvfp4_grouped(flat.device):
            return True
        return (
            hasattr(torch, "_grouped_mm")
            and all(
                type(e.up_proj) is nn.Linear and type(e.down_proj) is nn.Linear
                for e in self.experts
            )
            and len({e.up_proj.weight.requires_grad for e in self.experts}) <= 1
        )

    def _run_grouped_experts(self, gx: Tensor, offs: Tensor) -> Tensor:
        gx_bf = gx.to(torch.bfloat16)
        if isinstance(self.experts[0].up_proj, NVFP4Linear):
            up_out = nvfp4_grouped_linear(gx_bf, self.experts, "up_proj", offs)
            gate, val = up_out.chunk(2, dim=-1)
            return nvfp4_grouped_linear(F.silu(gate) * val, self.experts, "down_proj", offs)
        w_up, w_down = self._get_grouped_expert_weights(gx.device)
        gate, val = torch._grouped_mm(gx_bf, w_up, offs=offs).chunk(2, dim=-1)
        return torch._grouped_mm(F.silu(gate) * val, w_down, offs=offs)

    def _save_to_state_dict(self, destination: dict, prefix: str, keep_vars: bool) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.mitotic_pairs:
            destination[prefix + "mitotic_pairs_tensor"] = torch.tensor(
                self.mitotic_pairs, dtype=torch.long
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
        self._mitotic_idx_cache = None
        self._grouped_weight_cache = None
        setattr(self.experts, "_nvfp4_grouped_cache", None)
        mp_key = prefix + "mitotic_pairs_tensor"
        if mp_key in state_dict:
            mp_val = state_dict.pop(mp_key)
            if isinstance(mp_val, torch.Tensor) and mp_val.ndim == 2 and mp_val.shape[1] == 2:
                self.mitotic_pairs = [
                    (int(r[0]), int(r[1])) for r in mp_val.detach().cpu().tolist()
                ]
            else:
                self.mitotic_pairs = []
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: Tensor,
        valid: Tensor | None = None,
        *,
        prior_expert_mask: Tensor | None = None,
        dense_expert_indices: Sequence[int] | Tensor | None = None,
        dense_subset_k: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        eff_dense_indices: list[int]
        if dense_expert_indices is not None:
            if isinstance(dense_expert_indices, torch.Tensor):
                eff_dense_indices = [
                    int(i) for i in dense_expert_indices.detach().cpu().reshape(-1).tolist()
                ]
            else:
                eff_dense_indices = [int(i) for i in dense_expert_indices]
        else:
            eff_dense_indices = list(self.dense_expert_indices)
        eff_dense_k = int(self.dense_subset_k if dense_subset_k is None else dense_subset_k)

        if eff_dense_indices or eff_dense_k > 0:
            flat_valid = (
                torch.ones(flat.shape[0], dtype=torch.bool, device=x.device)
                if valid is None
                else valid.reshape(-1)
            )
            active_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                zero = x.new_zeros((), dtype=torch.float32)
                loads = x.new_zeros(self.num_experts, dtype=torch.float32)
                self.last_selected_mask = x.new_zeros(
                    (*x.shape[:-1], self.num_experts), dtype=torch.float32
                )
                return torch.zeros_like(x), zero, zero, loads

            active_x = flat.index_select(0, active_indices)
            with torch.autocast(device_type=x.device.type, enabled=False):
                router_logits = F.linear(active_x.float(), self.router.weight.float())
                probabilities = F.softmax(router_logits, dim=-1)
                routed_probs = probabilities
                pair_tensors = self._get_mitotic_pair_tensors(x.device)
                if pair_tensors is not None:
                    p_t, c_t = pair_tensors
                    routed_probs = probabilities.clone()
                    p_wins = router_logits.index_select(1, p_t) >= router_logits.index_select(
                        1, c_t
                    )
                    routed_probs[:, c_t] = torch.where(
                        p_wins,
                        torch.zeros_like(p_wins, dtype=routed_probs.dtype),
                        probabilities.index_select(1, c_t),
                    )
                    routed_probs[:, p_t] = torch.where(
                        p_wins,
                        probabilities.index_select(1, p_t),
                        torch.zeros_like(p_wins, dtype=routed_probs.dtype),
                    )
                selection_probs = routed_probs
                if prior_expert_mask is not None and self.recurrent_coverage_penalty > 0.0:
                    flat_prior = (
                        prior_expert_mask.reshape(-1, self.num_experts)
                        .index_select(0, active_indices)
                        .to(routed_probs.dtype)
                    )
                    selection_probs = routed_probs * (
                        1.0 - self.recurrent_coverage_penalty * flat_prior
                    ).clamp_min(0.05)

                if eff_dense_indices:
                    chosen_list = [i for i in eff_dense_indices if 0 <= i < self.num_experts]
                    chosen_t = torch.tensor(chosen_list, dtype=torch.long, device=x.device)
                else:
                    k_coal = min(max(1, eff_dense_k), self.num_experts)
                    pooled_scores = selection_probs.mean(dim=0)
                    _, chosen_t = pooled_scores.topk(k_coal, dim=-1)
                    chosen_list = [int(i) for i in chosen_t.detach().cpu().tolist()]

                dense_weights = probabilities.index_select(1, chosen_t)
                dense_weights = dense_weights / dense_weights.sum(dim=-1, keepdim=True).clamp_min(
                    1e-9
                )
                if self.routed_scaling_factor != 1.0:
                    dense_weights = dense_weights * self.routed_scaling_factor
                sel_mask_active = x.new_zeros(
                    (active_x.shape[0], self.num_experts), dtype=torch.float32
                )
                sel_mask_active[:, chosen_t] = 1.0
                full_sel_mask = x.new_zeros((flat.shape[0], self.num_experts), dtype=torch.float32)
                full_sel_mask.index_copy_(0, active_indices, sel_mask_active)
                self.last_selected_mask = full_sel_mask.view(*x.shape[:-1], self.num_experts)

                counts = sel_mask_active.sum(dim=0) / max(1, active_x.shape[0] * len(chosen_list))
                aux_loss = self.num_experts * (counts * probabilities.mean(dim=0)).sum()
                router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
                if self.sequence_diversity_coef > 0.0 and x.ndim == 3 and x.shape[1] > 1:
                    full_probs = probabilities.new_zeros((flat.shape[0], self.num_experts))
                    full_probs.index_copy_(0, active_indices, probabilities)
                    probs_bt = full_probs.view(x.shape[0], x.shape[1], self.num_experts)
                    aux_loss = aux_loss + self.sequence_diversity_coef * (
                        self._compute_sequence_diversity_loss(probs_bt, valid)
                    )
                if self.mitotic_ortho_coef > 0.0 and self.mitotic_pairs:
                    aux_loss = aux_loss + self.mitotic_ortho_coef * (
                        self.compute_mitotic_ortho_loss(x.device)
                    )

            active_out = self.forward_dense_subset(
                active_x, chosen_list, expert_weights=dense_weights
            )
            if self.combine_dense_and_sparse and len(chosen_list) < self.num_experts:
                rem_k = min(self.top_k, self.num_experts - len(chosen_list))
                if rem_k > 0:
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        dense_mass = (
                            probabilities.index_select(1, chosen_t)
                            .sum(dim=-1, keepdim=True)
                            .clamp_min(1e-9)
                        )
                        rem_probs = selection_probs.clone()
                        rem_probs[:, chosen_t] = -1.0
                        _, rem_sel = rem_probs.topk(rem_k, dim=-1)
                        rem_raw_w = routed_probs.gather(-1, rem_sel)
                        rem_mass = rem_raw_w.sum(dim=-1, keepdim=True).clamp_min(0.0)
                        rem_w = rem_raw_w / rem_mass.clamp_min(1e-9)
                        total_mass = (dense_mass + rem_mass).clamp_min(1e-9)
                        alpha_dense = (dense_mass / total_mass).to(active_x.dtype)
                        alpha_rem = (rem_mass / total_mass).to(active_x.dtype)
                        rem_oh = F.one_hot(rem_sel, num_classes=self.num_experts)
                        rem_counts_int = rem_oh.sum(dim=(0, 1))
                    flat_rem = rem_sel.reshape(-1)
                    perm_r = flat_rem.argsort(stable=True)
                    sorted_tok_r = torch.div(perm_r, rem_k, rounding_mode="floor")
                    sorted_w_r = rem_w.reshape(-1).index_select(0, perm_r).unsqueeze(-1)
                    gathered_r = active_x.index_select(0, sorted_tok_r)
                    if self._can_use_grouped_dispatch(gathered_r):
                        offs_r = rem_counts_int.to(torch.int32).cumsum(0, dtype=torch.int32)
                        all_r = self._run_grouped_experts(gathered_r, offs_r)
                        weighted_r = (all_r * sorted_w_r.to(all_r.dtype)).to(active_x.dtype)
                        unpermed_r = torch.empty_like(weighted_r)
                        unpermed_r.index_copy_(0, perm_r, weighted_r)
                        sparse_add = (
                            unpermed_r.view(active_x.shape[0], rem_k, active_x.shape[-1])
                            .sum(dim=1)
                            .to(active_x.dtype)
                        )
                        active_out = alpha_dense * active_out + alpha_rem * sparse_add
                    else:
                        chunks_r = gathered_r.split(rem_counts_int.tolist(), dim=0)
                        outs_r = [
                            expert(chunk)
                            for expert, chunk in zip(self.experts, chunks_r)
                            if chunk.shape[0] > 0
                        ]
                        if outs_r:
                            all_r = torch.cat(outs_r, dim=0)
                            weighted_r = (all_r * sorted_w_r.to(all_r.dtype)).to(active_x.dtype)
                            unpermed_r = torch.empty_like(weighted_r)
                            unpermed_r.index_copy_(0, perm_r, weighted_r)
                            sparse_add = (
                                unpermed_r.view(active_x.shape[0], rem_k, active_x.shape[-1])
                                .sum(dim=1)
                                .to(active_x.dtype)
                            )
                            active_out = alpha_dense * active_out + alpha_rem * sparse_add

            output = torch.zeros_like(flat)
            output.index_copy_(0, active_indices, active_out)
            return output.view_as(x), aux_loss, router_z_loss, counts.detach()

        use_grouped = self._can_use_grouped_dispatch(flat)
        if use_grouped:
            with torch.autocast(device_type=x.device.type, enabled=False):
                router_logits = F.linear(flat.float(), self.router.weight.float())
                pair_tensors = self._get_mitotic_pair_tensors(x.device)
                if pair_tensors is not None and not self.norm_topk_prob:
                    p_t, c_t = pair_tensors
                    ln2 = 0.6931471805599453
                    adj_logits = router_logits.clone()
                    adj_logits[:, p_t] = adj_logits[:, p_t] - ln2
                    adj_logits[:, c_t] = adj_logits[:, c_t] - ln2
                    probabilities = F.softmax(adj_logits, dim=-1)
                else:
                    probabilities = F.softmax(router_logits, dim=-1)
                routed_probs = probabilities
                if pair_tensors is not None:
                    p_t, c_t = pair_tensors
                    routed_probs = probabilities.clone()
                    p_wins = router_logits.index_select(1, p_t) >= router_logits.index_select(
                        1, c_t
                    )
                    if not self.norm_topk_prob:
                        pair_mass = probabilities.index_select(1, p_t) + probabilities.index_select(
                            1, c_t
                        )
                        routed_probs[:, c_t] = torch.where(
                            p_wins,
                            torch.zeros_like(p_wins, dtype=routed_probs.dtype),
                            pair_mass,
                        )
                        routed_probs[:, p_t] = torch.where(
                            p_wins,
                            pair_mass,
                            torch.zeros_like(p_wins, dtype=routed_probs.dtype),
                        )
                    else:
                        routed_probs[:, c_t] = torch.where(
                            p_wins,
                            torch.zeros_like(p_wins, dtype=routed_probs.dtype),
                            probabilities.index_select(1, c_t),
                        )
                        routed_probs[:, p_t] = torch.where(
                            p_wins,
                            probabilities.index_select(1, p_t),
                            torch.zeros_like(p_wins, dtype=routed_probs.dtype),
                        )
                if (
                    self.lineage_top_families > 0
                    and self.lineage_root_ids is not None
                    and self.lineage_logit_prior is not None
                    and pair_tensors is None
                ):
                    probabilities, weights, selected = self._route_lineage_families(
                        router_logits
                    )
                else:
                    if prior_expert_mask is not None and self.recurrent_coverage_penalty > 0.0:
                        flat_prior = prior_expert_mask.reshape(-1, self.num_experts).to(
                            routed_probs.dtype
                        )
                        selection_probs = routed_probs * (
                            1.0 - self.recurrent_coverage_penalty * flat_prior
                        ).clamp_min(0.05)
                        _, selected = selection_probs.topk(self.top_k, dim=-1)
                        weights = routed_probs.gather(-1, selected)
                    else:
                        weights, selected = routed_probs.topk(self.top_k, dim=-1)
                    if self.norm_topk_prob:
                        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                if self.routed_scaling_factor != 1.0:
                    weights = weights * self.routed_scaling_factor
                sel_mask_2d = x.new_zeros((flat.shape[0], self.num_experts), dtype=torch.float32)
                sel_mask_2d.scatter_add_(
                    1,
                    selected,
                    torch.ones_like(selected, dtype=torch.float32),
                )
                self.last_selected_mask = sel_mask_2d.detach().view(*x.shape[:-1], self.num_experts)
                counts_int_all = torch.bincount(
                    selected.reshape(-1), minlength=self.num_experts
                )
                if valid is not None:
                    flat_valid = valid.reshape(-1)
                    f_mask = flat_valid.unsqueeze(-1)
                    f_mask_f = f_mask.float()
                    n_valid_raw = flat_valid.sum()
                    has_valid = (n_valid_raw > 0).float()
                    n_valid = n_valid_raw.clamp_min(1).float()
                    weights = weights * f_mask_f
                    counts_valid = (sel_mask_2d * f_mask_f).sum(dim=0) / (n_valid * self.top_k)
                    mean_probs = (probabilities * f_mask_f).sum(dim=0) / n_valid
                    aux_loss = self.num_experts * (counts_valid * mean_probs).sum() * has_valid
                    z_sq = torch.logsumexp(router_logits, dim=-1).square()
                    router_z_loss = ((z_sq * flat_valid.float()).sum() / n_valid) * has_valid
                    counts = counts_valid * has_valid
                else:
                    counts = counts_int_all.float() / max(1, selected.numel())
                    aux_loss = self.num_experts * (counts * probabilities.mean(dim=0)).sum()
                    router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()

                if self.sequence_diversity_coef > 0.0 and x.ndim == 3 and x.shape[1] > 1:
                    probs_bt = probabilities.view(x.shape[0], x.shape[1], self.num_experts)
                    aux_loss = aux_loss + self.sequence_diversity_coef * (
                        self._compute_sequence_diversity_loss(probs_bt, valid)
                    )
                if self.mitotic_ortho_coef > 0.0 and self.mitotic_pairs:
                    aux_loss = aux_loss + self.mitotic_ortho_coef * (
                        self.compute_mitotic_ortho_loss(x.device)
                    )

            k_eff = selected.shape[1]
            flat_selected = selected.reshape(-1)
            perm = flat_selected.argsort(stable=True)
            sorted_tokens = torch.div(perm, k_eff, rounding_mode="floor")
            sorted_weights = weights.reshape(-1).index_select(0, perm).unsqueeze(-1)
            gathered_x = flat.index_select(0, sorted_tokens)
            offs = counts_int_all.to(torch.int32).cumsum(0, dtype=torch.int32)
            all_expert_out = self._run_grouped_experts(gathered_x, offs)
            weighted = (all_expert_out * sorted_weights.to(all_expert_out.dtype)).to(flat.dtype)
            unpermed = torch.empty_like(weighted)
            unpermed.index_copy_(0, perm, weighted)
            out = (
                unpermed.view(flat.shape[0], k_eff, flat.shape[-1])
                .sum(dim=1)
                .to(flat.dtype)
            )
            return out.view_as(x), aux_loss, router_z_loss, counts.detach()

        flat_valid = (
            torch.ones(flat.shape[0], dtype=torch.bool, device=x.device)
            if valid is None
            else valid.reshape(-1)
        )
        active_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
        if active_indices.numel() == 0:
            zero = x.new_zeros((), dtype=torch.float32)
            loads = x.new_zeros(self.num_experts, dtype=torch.float32)
            self.last_selected_mask = x.new_zeros(
                (*x.shape[:-1], self.num_experts), dtype=torch.float32
            )
            return torch.zeros_like(x), zero, zero, loads

        active_x = flat.index_select(0, active_indices)
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = F.linear(active_x.float(), self.router.weight.float())
            pair_tensors = self._get_mitotic_pair_tensors(x.device)
            if pair_tensors is not None and not self.norm_topk_prob:
                p_t, c_t = pair_tensors
                ln2 = 0.6931471805599453
                adj_logits = router_logits.clone()
                adj_logits[:, p_t] = adj_logits[:, p_t] - ln2
                adj_logits[:, c_t] = adj_logits[:, c_t] - ln2
                probabilities = F.softmax(adj_logits, dim=-1)
            else:
                probabilities = F.softmax(router_logits, dim=-1)
            routed_probs = probabilities
            if pair_tensors is not None:
                p_t, c_t = pair_tensors
                routed_probs = probabilities.clone()
                p_wins = router_logits.index_select(1, p_t) >= router_logits.index_select(1, c_t)
                if not self.norm_topk_prob:
                    pair_mass = probabilities.index_select(1, p_t) + probabilities.index_select(
                        1, c_t
                    )
                    routed_probs[:, c_t] = torch.where(
                        p_wins,
                        torch.zeros_like(p_wins, dtype=routed_probs.dtype),
                        pair_mass,
                    )
                    routed_probs[:, p_t] = torch.where(
                        p_wins,
                        pair_mass,
                        torch.zeros_like(p_wins, dtype=routed_probs.dtype),
                    )
                else:
                    routed_probs[:, c_t] = torch.where(
                        p_wins,
                        torch.zeros_like(p_wins, dtype=routed_probs.dtype),
                        probabilities.index_select(1, c_t),
                    )
                    routed_probs[:, p_t] = torch.where(
                        p_wins,
                        probabilities.index_select(1, p_t),
                        torch.zeros_like(p_wins, dtype=routed_probs.dtype),
                    )
            if (
                self.lineage_top_families > 0
                and self.lineage_root_ids is not None
                and self.lineage_logit_prior is not None
                and pair_tensors is None
            ):
                probabilities, weights, selected = self._route_lineage_families(
                    router_logits
                )
            else:
                if prior_expert_mask is not None and self.recurrent_coverage_penalty > 0.0:
                    flat_prior = (
                        prior_expert_mask.reshape(-1, self.num_experts)
                        .index_select(0, active_indices)
                        .to(routed_probs.dtype)
                    )
                    selection_probs = routed_probs * (
                        1.0 - self.recurrent_coverage_penalty * flat_prior
                    ).clamp_min(0.05)
                    _, selected = selection_probs.topk(self.top_k, dim=-1)
                    weights = routed_probs.gather(-1, selected)
                else:
                    weights, selected = routed_probs.topk(self.top_k, dim=-1)
                if self.norm_topk_prob:
                    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            if self.routed_scaling_factor != 1.0:
                weights = weights * self.routed_scaling_factor
            one_hot_sel = F.one_hot(selected, num_classes=self.num_experts)
            full_sel_mask = x.new_zeros((flat.shape[0], self.num_experts), dtype=torch.float32)
            full_sel_mask.index_copy_(
                0, active_indices, one_hot_sel.sum(dim=1).detach().to(torch.float32)
            )
            self.last_selected_mask = full_sel_mask.view(*x.shape[:-1], self.num_experts)
            counts_int = one_hot_sel.sum(dim=(0, 1))
            counts = counts_int.float() / max(1, selected.numel())
            aux_loss = self.num_experts * (counts * probabilities.mean(dim=0)).sum()
            router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
            if self.sequence_diversity_coef > 0.0 and x.ndim == 3 and x.shape[1] > 1:
                full_probs = probabilities.new_zeros((flat.shape[0], self.num_experts))
                full_probs.index_copy_(0, active_indices, probabilities)
                probs_bt = full_probs.view(x.shape[0], x.shape[1], self.num_experts)
                aux_loss = aux_loss + self.sequence_diversity_coef * (
                    self._compute_sequence_diversity_loss(probs_bt, valid)
                )
            if self.mitotic_ortho_coef > 0.0 and self.mitotic_pairs:
                aux_loss = aux_loss + self.mitotic_ortho_coef * (
                    self.compute_mitotic_ortho_loss(x.device)
                )

        k_eff = selected.shape[1]
        flat_selected = selected.reshape(-1)
        perm = flat_selected.argsort(stable=True)
        sorted_tokens = torch.div(perm, k_eff, rounding_mode="floor")
        sorted_weights = weights.reshape(-1).index_select(0, perm).unsqueeze(-1)
        gathered_x = active_x.index_select(0, sorted_tokens)
        expert_chunks = gathered_x.split(counts_int.tolist(), dim=0)

        expert_outs = [
            expert(chunk)
            for expert, chunk in zip(self.experts, expert_chunks)
            if chunk.shape[0] > 0
        ]
        if expert_outs:
            all_expert_out = torch.cat(expert_outs, dim=0)
            weighted = (all_expert_out * sorted_weights.to(all_expert_out.dtype)).to(
                active_x.dtype
            )
            unpermed = torch.empty_like(weighted)
            unpermed.index_copy_(0, perm, weighted)
            active_out = (
                unpermed.view(active_x.shape[0], k_eff, active_x.shape[-1])
                .sum(dim=1)
                .to(active_x.dtype)
            )
        else:
            active_out = torch.zeros_like(active_x)

        output = torch.zeros_like(flat)
        output.index_copy_(0, active_indices, active_out)
        return output.view_as(x), aux_loss, router_z_loss, counts.detach()


def _is_all_valid(valid: Tensor) -> bool:
    cached = getattr(valid, "_all_valid", None)
    if cached is not None:
        return bool(cached)
    res = bool(valid.all())
    try:
        setattr(valid, "_all_valid", res)
    except Exception:
        pass
    return res


class LocalAttentionBlock(nn.Module):
    """Causal or bidirectional self-attention supporting both MCL4 and standard SDPA."""

    def __init__(self, config: MCL4CEDConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.use_qk_norm = bool(config.use_qk_norm)
        self.qk_norm_mode = str(getattr(config, "qk_norm_mode", "head"))
        self.attention_multiplier = config.attention_multiplier
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        qk_norm_dim = config.d_model if self.qk_norm_mode == "full" else self.head_dim
        self.q_norm = (
            RMSNorm(qk_norm_dim, eps=config.rms_norm_eps)
            if self.use_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(qk_norm_dim, eps=config.rms_norm_eps)
            if self.use_qk_norm
            else nn.Identity()
        )
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        inv_freq = self._compute_inv_freq(torch.device("cpu"))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.markov_router: MarkovLocal4Router | None = None
        self.markov_gate: nn.Parameter | None = None
        self._rope_cache: tuple[Any, Tensor, Tensor] | None = None
        self._markov_gate_cache: tuple[Any, bool] | None = None
        if config.attention_type == "markov_local4":
            self.markov_router = MarkovLocal4Router(
                config.d_model,
                config.n_heads,
                router_dim=config.markov_router_dim,
                graph_degree=config.markov_graph_degree,
                num_walkers=config.markov_num_walkers,
                steps=config.markov_steps,
                attend_tokens=config.markov_attend_tokens,
                local_tokens=config.markov_local_tokens,
                beam_size=config.markov_beam_size,
                restart=config.markov_restart,
                query_chunk_size=config.markov_query_chunk_size,
            )
            if config.preserve_residual_across_stacks:
                self.markov_gate = nn.Parameter(torch.zeros(()))

    def _compute_inv_freq(self, device: torch.device) -> Tensor:
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim
            )
        )
        rope_scaling = getattr(self.config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and str(
            rope_scaling.get("rope_type", rope_scaling.get("type", ""))
        ).lower() == "llama3":
            factor = float(rope_scaling.get("factor", 8.0))
            low_freq_factor = float(rope_scaling.get("low_freq_factor", 1.0))
            high_freq_factor = float(rope_scaling.get("high_freq_factor", 4.0))
            old_context_len = float(
                rope_scaling.get("original_max_position_embeddings", 8192)
            )
            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            wavelen = 2.0 * math.pi / inv_freq
            inv_freq_llama = torch.where(
                wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
            )
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smoothed_inv_freq = (
                1.0 - smooth_factor
            ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(
                wavelen > low_freq_wavelen
            )
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return inv_freq

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def _get_inv_freq(self, device: torch.device) -> Tensor:
        if self.inv_freq.dtype == torch.float32 and self.inv_freq.device == device:
            return self.inv_freq
        inv_freq = self._compute_inv_freq(device)
        self.inv_freq = inv_freq
        return inv_freq

    def _get_rope_cos_sin(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        rope_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if rope_positions is not None:
            positions = rope_positions.to(device=device, dtype=torch.float32)
            inv_freq = self._get_inv_freq(device)
            angles = positions[:, None] * inv_freq[None, :]
            return angles.cos().to(dtype), angles.sin().to(dtype)
        key = (n, device, dtype, self.inv_freq.dtype, self.inv_freq.data_ptr())
        cached = self._rope_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        positions = torch.arange(n, device=device, dtype=torch.float32)
        inv_freq = self._get_inv_freq(device)
        angles = positions[:, None] * inv_freq[None, :]
        cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
        self._rope_cache = (key, cos, sin)
        return cos, sin

    def _is_markov_gate_active(self) -> bool:
        if self.markov_gate is None:
            return False
        key = (self.markov_gate._version, self.markov_gate.data_ptr())
        cached = self._markov_gate_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        active = float(self.markov_gate.detach().abs().item()) > 1e-6
        self._markov_gate_cache = (key, active)
        return active

    def _sdpa_attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        valid: Tensor,
        *,
        causal: bool,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        b, _, n, _ = q.shape
        if attn_mask is None and causal and _is_all_valid(valid):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
                scale=self.attention_multiplier,
            )
        allowed = valid[:, None, None, :].expand(b, 1, n, n)
        if causal:
            causal_mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
            allowed = allowed & causal_mask[None, None]
        if attn_mask is not None:
            allowed = allowed & attn_mask
        fallback = torch.eye(n, dtype=torch.bool, device=q.device)[None, None]
        allowed = allowed | (~allowed.any(-1, keepdim=True) & fallback)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed,
            dropout_p=self.config.dropout if self.training else 0.0,
            scale=self.attention_multiplier,
        )

    def forward(
        self,
        x_norm: Tensor,
        valid: Tensor,
        *,
        causal: bool = True,
        local_attn_mask: Tensor | None = None,
        rope_positions: Tensor | None = None,
    ) -> Tensor:
        b, n, width = x_norm.shape
        if self.use_qk_norm and self.qk_norm_mode == "full":
            q_flat, k_flat, v_flat = self.qkv(x_norm).chunk(3, dim=-1)
            q_flat, k_flat = self.q_norm(q_flat), self.k_norm(k_flat)
            q = q_flat.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
            k = k_flat.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
            v = v_flat.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            qkv = self.qkv(x_norm).view(b, n, 3, self.n_heads, self.head_dim)
            q, k, v = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
            if self.use_qk_norm:
                q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = self._get_rope_cos_sin(
            n, x_norm.device, q.dtype, rope_positions=rope_positions
        )
        q, k = self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)

        if self.markov_router is not None and self.markov_gate is not None:
            sdpa_attended = self._sdpa_attend(
                q, k, v, valid, causal=causal, attn_mask=local_attn_mask
            )
            if (
                self.training
                or torch.is_grad_enabled()
                or self._is_markov_gate_active()
            ):
                markov_attended = self.markov_router(
                    q,
                    k,
                    v,
                    x_norm,
                    valid,
                    causal=causal,
                    dropout_p=self.config.dropout,
                    attn_scale=self.attention_multiplier,
                    attn_mask=local_attn_mask,
                )
                attended = (
                    sdpa_attended
                    + torch.tanh(self.markov_gate.to(sdpa_attended.dtype)) * markov_attended
                )
            else:
                attended = sdpa_attended
        elif self.markov_router is not None:
            attended = self.markov_router(
                q,
                k,
                v,
                x_norm,
                valid,
                causal=causal,
                dropout_p=self.config.dropout,
                attn_scale=self.attention_multiplier,
                attn_mask=local_attn_mask,
            )
        else:
            attended = self._sdpa_attend(
                q, k, v, valid, causal=causal, attn_mask=local_attn_mask
            )
        out = self.out_proj(attended.transpose(1, 2).contiguous().view(b, n, width))
        return out.masked_fill(~valid.unsqueeze(-1), 0)


class CEDEncoderLayer(nn.Module):
    """Lower Causal Encoder Layer: PLE + MCL4 Self-Attention + SparseMoE."""

    def __init__(self, config: MCL4CEDConfig) -> None:
        super().__init__()
        self.residual_multiplier = float(config.residual_multiplier)
        self.ple = (
            PerLayerEmbedding(
                config.d_model,
                config.ple_dim,
                vocab_size=config.vocab_size,
                pad_id=config.pad_id,
                rms_norm_eps=config.rms_norm_eps,
            )
            if config.ple_dim > 0
            else None
        )
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attention = LocalAttentionBlock(config)
        self.moe_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = SparseMoE(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        original_inputs: Tensor,
        token_ids: Tensor | None,
        valid: Tensor,
        *,
        causal: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.ple is not None:
            x = self.ple(x, original_inputs, token_ids=token_ids, valid=valid)
        res_scale = self.residual_multiplier
        x = x + res_scale * self.dropout(self.attention(self.attn_norm(x), valid, causal=causal))
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        moe_out, aux_loss, z_loss, load = self.moe(self.moe_norm(x), valid=valid)
        x = (x + res_scale * self.dropout(moe_out)).masked_fill(~valid.unsqueeze(-1), 0)
        return x, aux_loss, z_loss, load


class CEDDecoderLayer(nn.Module):
    """Upper Causal Encoder-Decoder Layer: Local MCL4 + Global State Cross-Attention + PLE + MoE.

    Global K/V projections are computed once per unique state row (`[B_state, T_state, d_model]`)
    and indexed via `owner_rows` when scoring multiple candidates per state.
    """

    def __init__(self, config: MCL4CEDConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.residual_multiplier = float(config.residual_multiplier)
        self.ple = (
            PerLayerEmbedding(
                config.d_model,
                config.ple_dim,
                vocab_size=config.vocab_size,
                pad_id=config.pad_id,
                rms_norm_eps=config.rms_norm_eps,
            )
            if config.ple_dim > 0
            else None
        )
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.local_attention = LocalAttentionBlock(config)

        # Global cross-attention into the lower encoder's cached representation with QK-Norm + RoPE.
        self.global_q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.global_kv_proj = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.global_q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.global_k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.global_context_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.global_out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._rope_cache: dict[Any, tuple[Tensor, Tensor]] = {}

        self.moe_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = SparseMoE(config)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def _get_inv_freq(self, device: torch.device) -> Tensor:
        if self.inv_freq.dtype == torch.float32:
            return self.inv_freq
        return 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim
            )
        )

    def _get_rope_cos_sin(
        self,
        n: int,
        n_prefix: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        rope_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if rope_positions is not None:
            positions = rope_positions.to(device=device, dtype=torch.float32)
            inv_freq = self._get_inv_freq(device)
            angles = positions[:, None] * inv_freq[None, :]
            return angles.cos().to(dtype), angles.sin().to(dtype)
        key = (n, int(n_prefix), device, dtype, self.inv_freq.dtype, self.inv_freq.data_ptr())
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        if n_prefix > 0:
            positions = (
                torch.arange(n, device=device, dtype=torch.float32) - float(max(0, int(n_prefix)))
            ).clamp_min(0.0)
        else:
            positions = torch.arange(n, device=device, dtype=torch.float32)
        inv_freq = self._get_inv_freq(device)
        angles = positions[:, None] * inv_freq[None, :]
        cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
        if len(self._rope_cache) > 16:
            self._rope_cache.clear()
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def project_global_kv(
        self,
        encoded_state: Tensor,
        *,
        n_prefix: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Project state `[B_state, T_state, d_model]` into `(K_global, V_global)` once with RoPE."""
        b_state, t_state, _ = encoded_state.shape
        kv = self.global_kv_proj(encoded_state).view(
            b_state, t_state, 2, self.n_heads, self.head_dim
        )
        k_global, v_global = (part.transpose(1, 2).contiguous() for part in kv.unbind(dim=2))
        k_normed = self.global_k_norm(k_global)
        cos, sin = self._get_rope_cos_sin(t_state, int(n_prefix), encoded_state.device, k_normed.dtype)
        return self._apply_rope(k_normed, cos, sin), v_global

    def _cross_attend(
        self,
        q_norm: Tensor,
        k_global: Tensor,
        v_global: Tensor,
        state_valid: Tensor,
        cand_valid: Tensor,
        owner_rows: Tensor | None = None,
        causal_cross_mask: Tensor | None = None,
        causal_cross: bool = False,
        rope_positions: Tensor | None = None,
    ) -> Tensor:
        r, t_cand, width = q_norm.shape
        q = self.global_q_proj(q_norm).view(r, t_cand, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.global_q_norm(q)
        cos, sin = self._get_rope_cos_sin(
            t_cand, 0, q_norm.device, q.dtype, rope_positions=rope_positions
        )
        q = self._apply_rope(q, cos, sin)
        if owner_rows is not None:
            k = k_global.index_select(0, owner_rows)
            v = v_global.index_select(0, owner_rows)
            s_valid = state_valid.index_select(0, owner_rows)
        else:
            k, v, s_valid = k_global, v_global, state_valid

        t_state = k.shape[2]
        if causal_cross and causal_cross_mask is None and t_cand == t_state and _is_all_valid(s_valid):
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            allowed = s_valid[:, None, None, :].expand(r, 1, t_cand, t_state)
            if causal_cross and causal_cross_mask is None:
                diag = max(0, t_state - t_cand)
                causal_cross_mask = torch.ones(
                    1, 1, t_cand, t_state, dtype=torch.bool, device=q_norm.device
                ).tril(diagonal=diag)
            if causal_cross_mask is not None:
                allowed = allowed & causal_cross_mask
            # Prevent all-False rows if a state row were ever completely masked.
            fallback = torch.zeros_like(allowed)
            fallback[..., 0] = True
            allowed = allowed | (~allowed.any(-1, keepdim=True) & fallback)

            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed,
                dropout_p=self.config.dropout if self.training else 0.0,
            )
        merged = self.global_context_norm(
            context.transpose(1, 2).contiguous().view(r, t_cand, width)
        )
        out = self.global_out_proj(merged)
        return out.masked_fill(~cand_valid.unsqueeze(-1), 0)

    def forward(
        self,
        x: Tensor,
        original_inputs: Tensor,
        token_ids: Tensor | None,
        cand_valid: Tensor,
        k_global: Tensor,
        v_global: Tensor,
        state_valid: Tensor,
        *,
        owner_rows: Tensor | None = None,
        causal_local: bool = True,
        local_attn_mask: Tensor | None = None,
        rope_positions: Tensor | None = None,
        causal_cross_mask: Tensor | None = None,
        causal_cross: bool = False,
        prior_expert_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.ple is not None:
            x = self.ple(x, original_inputs, token_ids=token_ids, valid=cand_valid)
        normed = self.attn_norm(x)
        local_out = self.local_attention(
            normed,
            cand_valid,
            causal=causal_local,
            local_attn_mask=local_attn_mask,
            rope_positions=rope_positions,
        )
        global_out = self._cross_attend(
            normed,
            k_global,
            v_global,
            state_valid,
            cand_valid,
            owner_rows=owner_rows,
            causal_cross_mask=causal_cross_mask,
            causal_cross=causal_cross,
            rope_positions=rope_positions,
        )
        res_scale = self.residual_multiplier
        x = (
            x
            + res_scale * self.dropout(local_out)
            + res_scale * self.dropout(global_out)
        ).masked_fill(~cand_valid.unsqueeze(-1), 0)
        moe_out, aux_loss, z_loss, load = self.moe(
            self.moe_norm(x), valid=cand_valid, prior_expert_mask=prior_expert_mask
        )
        x = (x + res_scale * self.dropout(moe_out)).masked_fill(~cand_valid.unsqueeze(-1), 0)
        return x, aux_loss, z_loss, load


class MCL4CEDBackbone(nn.Module):
    """Unified MCL4 + CED + PLE + SparseMoE + RecurrentFeedback Backbone."""

    def __init__(self, config: MCL4CEDConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.encoder_layers = nn.ModuleList(
            CEDEncoderLayer(config) for _ in range(config.n_encoder_layers)
        )
        self.encoder_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.decoder_layers = nn.ModuleList(
            CEDDecoderLayer(config) for _ in range(config.n_decoder_layers)
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.feedback = RecurrentFeedback(
            config.d_model, zero_init_tanh=config.preserve_residual_across_stacks
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._initialize)
        total_layers = config.n_encoder_layers + config.n_decoder_layers
        residual_scale = 1.0 / math.sqrt(2.0 * total_layers)
        for layer in self.encoder_layers:
            nn.init.normal_(
                layer.attention.out_proj.weight,
                std=residual_scale / math.sqrt(config.d_model),
            )
            for expert in layer.moe.experts:
                nn.init.normal_(
                    expert.down_proj.weight,
                    std=residual_scale / math.sqrt(config.expert_hidden_dim),
                )
        for layer in self.decoder_layers:
            nn.init.normal_(
                layer.local_attention.out_proj.weight,
                std=residual_scale / math.sqrt(config.d_model),
            )
            nn.init.normal_(
                layer.global_out_proj.weight,
                std=residual_scale / math.sqrt(config.d_model),
            )
            for expert in layer.moe.experts:
                nn.init.normal_(
                    expert.down_proj.weight,
                    std=residual_scale / math.sqrt(config.expert_hidden_dim),
                )
        if 0 <= config.pad_id < config.vocab_size:
            with torch.no_grad():
                self.token_embedding.weight[config.pad_id].zero_()
        should_tie = (
            bool(config.tie_word_embeddings)
            if config.tie_word_embeddings is not None
            else (config.vocab_size > 1024)
        )
        if should_tie:
            self.lm_head.weight = self.token_embedding.weight
        elif config.vocab_size <= 1024:
            with torch.no_grad():
                self.lm_head.weight.copy_(self.token_embedding.weight)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.in_features))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.embedding_dim))

    def set_dense_expert_subset(
        self,
        expert_indices: Sequence[int] | None = None,
        *,
        dense_subset_k: int | None = None,
        combine_dense_and_sparse: bool | None = None,
    ) -> None:
        """Configure all MoE layers to execute an explicit or dynamically routed subset of experts densely."""
        if expert_indices is not None:
            clean_idx = tuple(int(i) for i in expert_indices)
            self.config.dense_expert_indices = clean_idx
        if dense_subset_k is not None:
            self.config.dense_subset_k = int(dense_subset_k)
        if combine_dense_and_sparse is not None:
            self.config.combine_dense_and_sparse = bool(combine_dense_and_sparse)
        for layer in list(self.encoder_layers) + list(self.decoder_layers):
            if expert_indices is not None:
                layer.moe.dense_expert_indices = self.config.dense_expert_indices
            if dense_subset_k is not None:
                layer.moe.dense_subset_k = self.config.dense_subset_k
            if combine_dense_and_sparse is not None:
                layer.moe.combine_dense_and_sparse = self.config.combine_dense_and_sparse

    def set_lineage_family_routing(
        self,
        num_roots: int = 64,
        *,
        top_families: int = 8,
        child_share: float = 0.25,
    ) -> None:
        """Configure parent-anchored lineage-calibrated family routing across all encoder and decoder MoE layers."""
        for layer in list(self.encoder_layers) + list(self.decoder_layers):
            layer.moe.configure_lineage_family_routing(
                num_roots=num_roots,
                top_families=top_families,
                child_share=child_share,
            )

    def encode_state(
        self,
        state_ids: Tensor,
        state_valid: Tensor | None = None,
        *,
        causal: bool = True,
    ) -> EncodedState:
        """Encode shared state/context once through the lower CED encoder stack."""
        if state_ids.ndim != 2 or min(state_ids.shape) < 1:
            raise ValueError("state_ids must have nonempty shape [batch, seq_len]")
        if state_valid is None:
            state_valid = state_ids != self.config.pad_id
            if self.config.pad_id < 0:
                try:
                    setattr(state_valid, "_all_valid", True)
                except Exception:
                    pass
        if getattr(state_valid, "_all_valid", None) is not True:
            if not state_valid.any(dim=-1).all():
                raise ValueError("Every state row must have at least one valid token")
        x = self.token_embedding(state_ids.long())
        if self.config.embedding_multiplier != 1.0:
            x = x * float(self.config.embedding_multiplier)
        x = x.masked_fill(~state_valid.unsqueeze(-1), 0)
        original_inputs = x
        aux_losses, z_losses, loads = [], [], []
        for layer in self.encoder_layers:
            x, aux, z, load = layer(x, original_inputs, state_ids, state_valid, causal=causal)
            layer.moe.last_selected_mask = None
            aux_losses.append(aux)
            z_losses.append(z)
            loads.append(load)
        hidden = self.encoder_norm(x).masked_fill(~state_valid.unsqueeze(-1), 0)
        return EncodedState(
            hidden=hidden,
            valid=state_valid,
            aux_loss=torch.stack(aux_losses).mean(),
            router_z_loss=torch.stack(z_losses).mean(),
            expert_load=torch.stack(loads),
            raw_hidden=x,
            state_ids=state_ids,
        )

    def decode_candidates(
        self,
        encoded_state: EncodedState,
        candidate_ids: Tensor,
        candidate_valid: Tensor | None = None,
        *,
        owner_rows: Tensor | None = None,
        recurrence_steps: int = 1,
        causal_local: bool = True,
        causal_cross_mask: Tensor | None = None,
        causal_cross: bool = False,
        candidate_encoded: EncodedState | None = None,
        recurrent_kv: bool = False,
        depth_ponder: bool = False,
    ) -> BackboneOutput:
        """Run upper CED decoder layers over candidates cross-attending to `encoded_state`."""
        if recurrence_steps < 1:
            raise ValueError("recurrence_steps must be at least 1")
        if candidate_ids.ndim != 2 or min(candidate_ids.shape) < 1:
            raise ValueError("candidate_ids must have nonempty shape [rows, seq_len]")
        if candidate_valid is None:
            candidate_valid = candidate_ids != self.config.pad_id
            if self.config.pad_id < 0:
                try:
                    setattr(candidate_valid, "_all_valid", True)
                except Exception:
                    pass
        if getattr(candidate_valid, "_all_valid", None) is not True:
            if not candidate_valid.any(dim=-1).all():
                raise ValueError("Every decoded candidate row must have at least one valid token")

        # For donor-grafted models (`preserve_residual_across_stacks=True`) decoding candidate options
        # against a context (`owner_rows is not None`), prepend the right-aligned valid question tail
        # before running `encoder_layers[0..7]` and `decoder_layers[0..7]` so all 16 layers share
        # consistent RoPE positions and native causal self-attention across `[question -> candidate]`.
        k_ctx_prefix = 0
        strip_cand_bos = False
        q_row_lens: Tensor | None = None
        cand_gather_idx: Tensor | None = None
        work_ids = candidate_ids
        work_valid = candidate_valid
        if (
            candidate_encoded is None
            and owner_rows is not None
            and causal_local
            and self.config.preserve_residual_across_stacks
            and encoded_state.state_ids is not None
        ):
            s_ids = encoded_state.state_ids
            s_val = encoded_state.valid
            s_content_val = s_val & (s_ids != self.config.eos_id)
            s_content_val = torch.where(
                s_content_val.any(dim=-1, keepdim=True), s_content_val, s_val
            )
            max_ctx_len = int(s_content_val.sum(dim=-1).max().item())
            if max_ctx_len >= 1:
                k_ctx_prefix = min(384, max_ctx_len)
                _, right_perm = torch.sort(s_content_val.to(torch.int32), dim=-1, stable=True)
                tail_perm = right_perm[:, -k_ctx_prefix:]
                tail_ids_b = s_ids.gather(1, tail_perm)
                tail_val_b = s_content_val.gather(1, tail_perm)
                tail_ids = tail_ids_b.index_select(0, owner_rows)
                tail_val = tail_val_b.index_select(0, owner_rows)
                q_row_lens = tail_val.sum(dim=-1).clamp_min(1)
                if (
                    candidate_ids.shape[1] >= 2
                    and bool(candidate_valid[:, 1:].any(dim=-1).all())
                    and (
                        (self.config.vocab_size == 128256 and bool((candidate_ids[:, 0] == 128000).all()))
                        or bool((candidate_ids[:, 0] == self.config.pad_id + 1).all())
                    )
                ):
                    strip_cand_bos = True
                    cat_ids = torch.cat([tail_ids, candidate_ids[:, 1:]], dim=1)
                    cat_val = torch.cat([tail_val, candidate_valid[:, 1:]], dim=1)
                else:
                    cat_ids = torch.cat([tail_ids, candidate_ids], dim=1)
                    cat_val = torch.cat([tail_val, candidate_valid], dim=1)
                _, left_perm = torch.sort((~cat_val).to(torch.int32), dim=-1, stable=True)
                work_ids = cat_ids.gather(1, left_perm)
                work_valid = cat_val.gather(1, left_perm)
                start_pos = (q_row_lens - 1) if strip_cand_bos else q_row_lens
                t_cand = int(candidate_ids.shape[1])
                cand_gather_idx = (
                    start_pos.unsqueeze(1)
                    + torch.arange(t_cand, device=work_ids.device).unsqueeze(0)
                ).clamp(0, work_ids.shape[1] - 1)

        work_orig = self.token_embedding(work_ids.long())
        if self.config.embedding_multiplier != 1.0:
            work_orig = work_orig * float(self.config.embedding_multiplier)
        work_orig = work_orig.masked_fill(~work_valid.unsqueeze(-1), 0)
        if candidate_encoded is None:
            candidate_encoded = self.encode_state(
                work_ids, work_valid, causal=causal_local
            )
        work_anchor = (
            candidate_encoded.raw_hidden
            if (
                self.config.preserve_residual_across_stacks
                and candidate_encoded.raw_hidden is not None
            )
            else candidate_encoded.hidden
        )
        if k_ctx_prefix > 0 and cand_gather_idx is not None:
            gather_3d = cand_gather_idx.unsqueeze(-1).expand(-1, -1, work_anchor.shape[-1])
            anchor = work_anchor.gather(1, gather_3d).masked_fill(
                ~candidate_valid.unsqueeze(-1), 0
            )
            eff_kv_state = candidate_encoded
            eff_owner_rows: Tensor | None = None
            eff_causal_cross = True
            eff_cross_mask: Tensor | None = None
        else:
            anchor = work_anchor
            eff_kv_state = encoded_state
            eff_owner_rows = owner_rows
            eff_causal_cross = causal_cross
            eff_cross_mask = causal_cross_mask

        n_prefix_kv = (
            max(0, int(eff_kv_state.hidden.shape[1] - work_ids.shape[1]))
            if (eff_causal_cross and eff_owner_rows is None)
            else 0
        )
        # Precompute K_global, V_global per decoder layer once for the state rows.
        global_kv_cache = [
            layer.project_global_kv(eff_kv_state.hidden, n_prefix=n_prefix_kv)
            for layer in self.decoder_layers
        ]

        state = work_anchor
        step_states: list[Tensor] = []
        raw_step_states: list[Tensor] = []
        aux_losses = [candidate_encoded.aux_loss]
        z_losses = [candidate_encoded.router_z_loss]
        loads = [candidate_encoded.expert_load]
        visited_expert_masks: list[Tensor | None] = [None] * len(self.decoder_layers)
        for step in range(recurrence_steps):
            x = work_anchor if step == 0 else self.feedback(work_anchor, state, valid=work_valid)
            step_kv_cache = global_kv_cache
            if (
                recurrent_kv
                and step > 0
                and eff_owner_rows is None
                and eff_kv_state.hidden.shape[0] == x.shape[0]
                and eff_kv_state.hidden.shape[1] >= x.shape[1]
            ):
                t_seq = x.shape[1]
                t_tot = eff_kv_state.hidden.shape[1]
                if t_tot == t_seq:
                    kv_state = self.encoder_norm(0.70 * eff_kv_state.hidden + 0.30 * x).masked_fill(
                        ~eff_kv_state.valid.unsqueeze(-1), 0
                    )
                else:
                    n_prefix = t_tot - t_seq
                    seq_part = self.encoder_norm(
                        0.70 * eff_kv_state.hidden[:, n_prefix:] + 0.30 * x
                    )
                    kv_state = torch.cat(
                        (eff_kv_state.hidden[:, :n_prefix], seq_part), dim=1
                    ).masked_fill(~eff_kv_state.valid.unsqueeze(-1), 0)
                step_kv_cache = [
                    layer.project_global_kv(kv_state, n_prefix=n_prefix_kv)
                    for layer in self.decoder_layers
                ]
            for l_idx, (layer, (k_global, v_global)) in enumerate(
                zip(self.decoder_layers, step_kv_cache)
            ):
                x, aux, z, load = layer(
                    x,
                    work_orig,
                    work_ids,
                    work_valid,
                    k_global,
                    v_global,
                    eff_kv_state.valid,
                    owner_rows=eff_owner_rows,
                    causal_local=causal_local,
                    causal_cross_mask=eff_cross_mask,
                    causal_cross=eff_causal_cross,
                    prior_expert_mask=visited_expert_masks[l_idx],
                )
                if recurrence_steps > 1 and layer.moe.last_selected_mask is not None:
                    cur_mask = layer.moe.last_selected_mask
                    prev_mask = visited_expert_masks[l_idx]
                    visited_expert_masks[l_idx] = (
                        cur_mask if prev_mask is None else (prev_mask + cur_mask).clamp_max(1.0)
                    )
                layer.moe.last_selected_mask = None
                aux_losses.append(aux)
                z_losses.append(z)
                loads.append(load.unsqueeze(0))
            raw_step_states.append(x)
            state = self.final_norm(x).masked_fill(~work_valid.unsqueeze(-1), 0)
            step_states.append(state)

        if depth_ponder and len(step_states) > 1:
            s1 = step_states[0]
            if len(step_states) == 2:
                s_deep = step_states[1]
            else:
                deep_stack = torch.stack(step_states[1:], dim=0)
                w_decay = torch.pow(
                    0.65,
                    torch.arange(
                        len(step_states) - 1,
                        device=s1.device,
                        dtype=s1.dtype,
                    ),
                )
                w_decay = (w_decay / w_decay.sum()).view(-1, 1, 1, 1)
                s_deep = (deep_stack * w_decay).sum(dim=0)

            if self.config.preserve_residual_across_stacks:
                r1 = raw_step_states[0]
                if len(raw_step_states) == 2:
                    r_deep = raw_step_states[1]
                else:
                    r_stack = torch.stack(raw_step_states[1:], dim=0)
                    w_decay = torch.pow(
                        0.65,
                        torch.arange(
                            len(raw_step_states) - 1,
                            device=r1.device,
                            dtype=r1.dtype,
                        ),
                    )
                    w_decay = (w_decay / w_decay.sum()).view(-1, 1, 1, 1)
                    r_deep = (r_stack * w_decay).sum(dim=0)
                if causal_cross:
                    with torch.no_grad():
                        ent_chunks: list[Tensor] = []
                        for r_i in range(0, s1.shape[0], 4):
                            l_c = self.lm_head(s1[r_i : r_i + 4]).float()
                            if self.config.logits_scaling != 1.0:
                                l_c = l_c / float(self.config.logits_scaling)
                            p_c = F.softmax(l_c, dim=-1)
                            del l_c
                            ent_chunks.append(torch.special.entr(p_c).sum(dim=-1, keepdim=True))
                            del p_c
                        ent1 = torch.cat(ent_chunks, dim=0)
                        gate_ent = (0.50 + 0.35 * torch.sigmoid((ent1 - 1.5) / 0.6)).to(r1.dtype)
                    blended = self.final_norm(r1 + gate_ent * (r_deep - r1))
                else:
                    blended = self.final_norm(0.40 * r1 + 0.60 * r_deep)
                state = blended.masked_fill(~work_valid.unsqueeze(-1), 0)
            elif causal_cross:
                with torch.no_grad():
                    ent_chunks = []
                    for r_i in range(0, s1.shape[0], 4):
                        l_c = self.lm_head(s1[r_i : r_i + 4]).float()
                        if self.config.logits_scaling != 1.0:
                            l_c = l_c / float(self.config.logits_scaling)
                        p_c = F.softmax(l_c, dim=-1)
                        del l_c
                        ent_chunks.append(torch.special.entr(p_c).sum(dim=-1, keepdim=True))
                        del p_c
                    ent1 = torch.cat(ent_chunks, dim=0)
                    gate_ent = (0.50 + 0.35 * torch.sigmoid((ent1 - 1.5) / 0.6)).to(s1.dtype)
                blended = self.final_norm(s1 + gate_ent * (s_deep - s1))
                state = blended.masked_fill(~work_valid.unsqueeze(-1), 0)
            else:
                blended = self.final_norm(0.40 * s1 + 0.60 * s_deep)
                state = blended.masked_fill(~work_valid.unsqueeze(-1), 0)

        ctx_last_hidden: Tensor | None = None
        if k_ctx_prefix > 0 and q_row_lens is not None and cand_gather_idx is not None:
            s_for_cand = (0.85 * step_states[0] + 0.15 * state) if len(step_states) > 1 else state
            q_last_idx = (
                (q_row_lens - 1)
                .clamp(0, s_for_cand.shape[1] - 1)
                .view(-1, 1, 1)
                .expand(-1, 1, s_for_cand.shape[-1])
            )
            ctx_last_hidden = s_for_cand.gather(1, q_last_idx).squeeze(1)
            gather_3d_s = cand_gather_idx.unsqueeze(-1).expand(-1, -1, s_for_cand.shape[-1])
            state = s_for_cand.gather(1, gather_3d_s).masked_fill(
                ~candidate_valid.unsqueeze(-1), 0
            )
            step_states = [
                s.gather(1, gather_3d_s).masked_fill(~candidate_valid.unsqueeze(-1), 0)
                for s in step_states
            ]

        return BackboneOutput(
            hidden=state,
            valid=candidate_valid,
            aux_loss=torch.stack(aux_losses).mean(),
            router_z_loss=torch.stack(z_losses).mean(),
            expert_load=torch.cat(loads, dim=0),
            anchor=anchor,
            step_states=tuple(step_states),
            ctx_last_hidden=ctx_last_hidden,
        )

    def encode(
        self,
        input_ids: Tensor,
        *,
        recurrence_steps: int = 1,
        latent_prior: Tensor | None = None,
        latent_prior_scale: float = 0.0,
        recurrent_kv: bool = False,
        depth_ponder: bool = False,
        gist_state: EncodedState | None = None,
        prior_start_pos: int = 0,
    ) -> BackboneOutput:
        """Causal full-sequence pass through lower encoder and upper CED decoder."""
        valid = input_ids != self.config.pad_id
        if self.config.pad_id < 0:
            try:
                setattr(valid, "_all_valid", True)
            except Exception:
                pass
        elif not valid.any(dim=-1).all():
            # Fallback if an entire block is pad_id: treat index 0 as valid for numeric safety.
            valid = valid.clone()
            valid[:, 0] = True
        encoded = self.encode_state(input_ids, valid, causal=True)
        state_for_kv = encoded
        eos_hits = input_ids == self.config.eos_id
        if eos_hits.shape[1] > 0:
            eos_hits = eos_hits.clone()
            eos_hits[:, 0] = False
        first_episode_mask = eos_hits.cumsum(dim=1) == 0
        if latent_prior is not None and latent_prior_scale != 0.0:
            prior = latent_prior.to(device=encoded.hidden.device, dtype=encoded.hidden.dtype)
            if prior.ndim == 1:
                prior = prior.unsqueeze(0)
            if prior.ndim == 2:
                prior = prior.unsqueeze(1)
            prior_normed = self.encoder_norm(prior)
            if prior_start_pos > 0 and prior_start_pos < encoded.hidden.shape[1]:
                pos_mask = (
                    (
                        torch.arange(encoded.hidden.shape[1], device=encoded.hidden.device).unsqueeze(0)
                        >= int(prior_start_pos)
                    )
                    & first_episode_mask
                ).unsqueeze(-1)
                steered_raw = torch.where(
                    pos_mask,
                    encoded.hidden + float(latent_prior_scale) * prior_normed,
                    encoded.hidden,
                )
            else:
                steered_raw = encoded.hidden + float(latent_prior_scale) * prior_normed
            steered_hidden = self.encoder_norm(steered_raw).masked_fill(~valid.unsqueeze(-1), 0)
            state_for_kv = EncodedState(
                hidden=steered_hidden,
                valid=encoded.valid,
                aux_loss=encoded.aux_loss,
                router_z_loss=encoded.router_z_loss,
                expert_load=encoded.expert_load,
                gist_prefix_len=int(prior_start_pos),
                raw_hidden=encoded.raw_hidden,
            )

        causal_cross_mask: Tensor | None = None
        if gist_state is not None and gist_state.hidden.shape[1] > 0:
            bsz, t_cand, _ = state_for_kv.hidden.shape
            g_hid = gist_state.hidden.to(
                device=state_for_kv.hidden.device, dtype=state_for_kv.hidden.dtype
            )
            g_val = gist_state.valid.to(device=state_for_kv.valid.device, dtype=torch.bool)
            if g_hid.shape[0] == 1 and bsz > 1:
                g_hid = g_hid.expand(bsz, -1, -1)
                g_val = g_val.expand(bsz, -1)
            g_normed = self.encoder_norm(g_hid).masked_fill(~g_val.unsqueeze(-1), 0)
            n_gist = g_normed.shape[1]
            eff_prefix_len = max(0, int(prior_start_pos or gist_state.gist_prefix_len))
            q_positions = torch.arange(t_cand, device=input_ids.device).unsqueeze(0)
            if eff_prefix_len > 0:
                gist_allowed = (
                    (q_positions >= eff_prefix_len) & first_episode_mask
                ).unsqueeze(-1).expand(bsz, t_cand, n_gist)
            else:
                gist_allowed = (q_positions >= 0).unsqueeze(-1).expand(bsz, t_cand, n_gist)
            seq_allowed = (
                torch.ones(t_cand, t_cand, dtype=torch.bool, device=input_ids.device)
                .tril()
                .unsqueeze(0)
                .expand(bsz, t_cand, t_cand)
            )
            causal_cross_mask = torch.cat((gist_allowed, seq_allowed), dim=-1).unsqueeze(1)
            state_for_kv = EncodedState(
                hidden=torch.cat((g_normed, state_for_kv.hidden), dim=1),
                valid=torch.cat((g_val, state_for_kv.valid), dim=1),
                aux_loss=0.5 * (
                    state_for_kv.aux_loss + gist_state.aux_loss.to(state_for_kv.aux_loss.device)
                ),
                router_z_loss=0.5 * (
                    state_for_kv.router_z_loss
                    + gist_state.router_z_loss.to(state_for_kv.router_z_loss.device)
                ),
                expert_load=state_for_kv.expert_load,
                gist_prefix_len=eff_prefix_len,
                raw_hidden=state_for_kv.raw_hidden,
            )

        decoded = self.decode_candidates(
            state_for_kv,
            input_ids,
            valid,
            recurrence_steps=recurrence_steps,
            causal_local=True,
            causal_cross_mask=causal_cross_mask,
            causal_cross=True,
            candidate_encoded=encoded,
            recurrent_kv=recurrent_kv,
            depth_ponder=depth_ponder,
        )
        return BackboneOutput(
            hidden=decoded.hidden,
            valid=valid,
            aux_loss=decoded.aux_loss,
            router_z_loss=decoded.router_z_loss,
            expert_load=decoded.expert_load,
            anchor=decoded.anchor,
            step_states=decoded.step_states,
        )

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        recurrence_steps: int = 1,
        latent_prior: Tensor | None = None,
        latent_prior_scale: float = 0.0,
        recurrent_kv: bool = False,
        depth_ponder: bool = False,
        gist_state: EncodedState | None = None,
        prior_start_pos: int = 0,
        deep_supervision_weight: float = 0.0,
    ) -> CausalLMOutput:
        """Autoregressive `U$ -> T$ -> A$` language modeling pass."""
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must match input_ids shape [batch, seq_len]")
        encoded = self.encode(
            input_ids,
            recurrence_steps=recurrence_steps,
            latent_prior=latent_prior,
            latent_prior_scale=latent_prior_scale,
            recurrent_kv=recurrent_kv,
            depth_ponder=depth_ponder,
            gist_state=gist_state,
            prior_start_pos=prior_start_pos,
        )
        logits = self.lm_head(encoded.hidden)
        if self.config.logits_scaling != 1.0:
            logits = logits / float(self.config.logits_scaling)
        ce_loss, loss = None, None
        if targets is not None:
            valid_mask = targets.ne(-100)
            valid_tok_count = valid_mask.sum().clamp_min(1)
            ce_sum = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            ce_loss = ce_sum / valid_tok_count
            loss = (
                ce_loss
                + self.config.aux_loss_coef * encoded.aux_loss
                + self.config.router_z_loss_coef * encoded.router_z_loss
            )
            if deep_supervision_weight > 0.0 and len(encoded.step_states) > 1:
                valid_targets = targets[valid_mask]
                if valid_targets.numel() > 0:
                    step_ces: list[Tensor] = []
                    for s_r in encoded.step_states:
                        l_r = self.lm_head(s_r[valid_mask])
                        if self.config.logits_scaling != 1.0:
                            l_r = l_r / float(self.config.logits_scaling)
                        s_sum = F.cross_entropy(
                            l_r.float(),
                            valid_targets,
                            reduction="sum",
                        )
                        step_ces.append(s_sum / valid_tok_count)
                    loss = loss + float(deep_supervision_weight) * torch.stack(step_ces).mean()
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            aux_loss=encoded.aux_loss,
            router_z_loss=encoded.router_z_loss,
            expert_load=encoded.expert_load,
            step_states=encoded.step_states,
        )

    def forward_windowless(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        pager: Any = None,
        active_window_len: int | None = None,
        page_size: int = 512,
        top_k_pages: int = 4,
        working_state_ids: Tensor | None = None,
        recurrence_steps: int = 1,
        latent_prior: Tensor | None = None,
        latent_prior_scale: float = 0.0,
        recurrent_kv: bool = False,
        depth_ponder: bool = False,
        deep_supervision_weight: float = 0.0,
    ) -> CausalLMOutput:
        """Windowless demand-paged forward pass for arbitrary-length ($1\\text{M}\\to\\infty$) streams.

        Automatically pages out prefix tokens exceeding ``active_window_len`` into an
        ``InfiniteContextPager`` (compressed 4.2x via TurboQuant with local RoPE ``[0..page_size]``),
        retrieves the pinned ``WorkingStateLedger`` + top-$B$ relevant pages into a position-invariant
        ``gist_state`` prefix, and runs the active tail within bounded ``[0..active_window_len]`` RoPE.
        """
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must match input_ids shape [batch, seq_len]")
        eff_window = (
            int(active_window_len)
            if active_window_len is not None and int(active_window_len) > 0
            else min(int(self.config.max_seq_len), max(int(page_size) * 2, 1024))
        )
        if pager is None:
            from .context_memory import InfiniteContextPager

            pager = InfiniteContextPager(
                page_size=int(page_size),
                top_k_pages=int(top_k_pages),
            )
        if working_state_ids is not None:
            pager.update_working_state(self, working_state_ids)

        total_len = int(input_ids.shape[1])
        if total_len > eff_window:
            prefix_len = total_len - eff_window
            prefix_ids = input_ids[:, :prefix_len]
            active_ids = input_ids[:, prefix_len:]
            active_targets = targets[:, prefix_len:] if targets is not None else None
            pager.ingest_tokens(self, prefix_ids)
        else:
            active_ids = input_ids
            active_targets = targets

        gist_state = pager.build_gist_state(self, active_ids, top_k=int(top_k_pages))
        return self.forward(
            active_ids,
            active_targets,
            recurrence_steps=recurrence_steps,
            latent_prior=latent_prior,
            latent_prior_scale=latent_prior_scale,
            recurrent_kv=recurrent_kv,
            depth_ponder=depth_ponder,
            gist_state=gist_state,
            prior_start_pos=0,
            deep_supervision_weight=deep_supervision_weight,
        )

