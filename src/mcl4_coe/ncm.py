"""Next-Concept Modeling (NCM) integrated with MCL4-COE, CED, SparseMoE, Ornith-2.5, & Dream-RSI.

Implements the complete Next-Concept Modeling framework (Kumbong, Khare, Arora, Mirhoseini,
Ré et al., Stanford Hazy Research & Together AI, Oct 2026, arXiv:2610.14162) unified with
`MCL4-COE`:
1. Stage 1 — Multiscale Residual VQ-VAE (`MultiscaleResidualQuantizer`, `DonorAnchoredMultiscaleVQVAE`):
   - Encodes a length-$L$ token sequence $x_{1:L}$ into $K$ coarse-to-fine discrete concept code
     levels $r_{1:K} = (r_1, \\dots, r_K)$ with lengths $\\ell_1 < \\ell_2 < \\dots < \\ell_K = L$.
   - Learned cascaded stride-$S$ down/upsamplers at level 1 (`learned_l1_sampling` / `learned_l1_cascade`),
     area downsampling (`Down_k`) and linear upsampling (`Up_k`) at levels $k > 1$, per-level
     residual 1D convolutions (`PhiResidualConv`), Laplace-smoothed EMA codebook updates with
     dead-code restart, and nested dropout (`fine_drop`) so every partial cumulative reconstruction
     $\\hat{z}^{(k)} = \\sum_{j=1}^k \\Phi_j(\\mathrm{Up}_j(e_j(r_j)))$ is decodable at any scale $k$.
   - Supports both standalone VQ-VAE training and zero-drift Donor-Anchored mode (`Granite 3.1 MoE` /
     `OLMoE` / `MCL4CEDBackbone` frozen embeddings and `lm_head`).
2. Stage 2 — Next-Concept Generator on `MCL4-CED` (`NCMGeneratorAdapter`, `TiedCodebookHead`, `NCMSystem`):
   - Uses lower `CEDEncoderLayer` stack to encode `#U$#` (and `#S$#` scaffold) once into `(K_global, V_global)`.
   - Uses upper `CEDDecoderLayer` stack with block-diagonal intra-level attention (`attn_pattern="block_diagonal"`),
     per-level reset RoPE `[0 .. \\ell_k - 1]`, multiscale concept `PLE` injection, and distance-tied
     codebook heads (`TiedCodebookHead`: $(2 \\langle z, E \\rangle - \\|E\\|^2) / \\overline{\\|E\\|^2}$).
   - Scale-Adaptive `SparseMoE` Active Expert Density (`top_k(k)`): Activates higher expert density
     (`coarse_top_k`) on the short coarse planning levels ($\\ell_k \\le 16$, $<2\\%$ FLOP overhead)
     and tapers to `fine_top_k` on long fine levels, while tracking per-scale expert loads for
     scale-specialized mitotic splitting.
3. Mid-Hierarchy 9-Operator `COE` Verification, Adaptive Halting, & Exact-Likelihood `Ornith-2.5` GRPO:
   - Evaluates intermediate cumulative reconstructions $\\hat{z}^{(k)}$ with `Decision`, `Deduction`,
     `Evaluation`, `Abstraction`, and `Metacognition` for latent coarse-scale candidate selection,
     mid-level contradiction veto (`RecurrentFeedback` lateral refinement), and adaptive early-exit halting.
   - Computes exact single-pass trajectory log-likelihoods $\\sum_{k=1}^K w_k \\sum_{i=1}^{\\ell_k} \\log p_\\theta(r_k^i \\mid \\hat{z}^{(k-1)})$
     for `Ornith-2.5` (`pp = p * 0.95^(n-1)`) GRPO and `Dream-RSI` scale-tree discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import EncodedState, MCL4CEDBackbone
from .ple import RMSNorm


# ---------------------------------------------------------------------------
# 1. Coarse-to-Fine Level Schedules (Appendix F.1.2 of arXiv:2610.14162)
# ---------------------------------------------------------------------------


def compute_ncm_levels(
    target_length: int,
    num_levels: int = 8,
    *,
    schedule: str = "quadratic",
    start_length: int = 1,
    manual_levels: Sequence[int] = (),
) -> tuple[int, ...]:
    """Compute strictly increasing coarse-to-fine level lengths $\\ell_1 < \\dots < \\ell_K = L$."""
    if manual_levels:
        cleaned = tuple(int(x) for x in manual_levels)
        if not cleaned or cleaned[-1] != int(target_length):
            raise ValueError(
                f"manual_levels must end at target_length={target_length}; got {cleaned}"
            )
        if any(cleaned[i] <= 0 for i in range(len(cleaned))):
            raise ValueError("All level lengths must be positive integers")
        return cleaned

    L = int(target_length)
    K = int(num_levels)
    s0 = max(1, int(start_length))
    if L < 1 or K < 1:
        raise ValueError("target_length and num_levels must be >= 1")
    if K == 1 or L == 1:
        return (L,)
    if K > L:
        K = L

    t = np.linspace(0.0, 1.0, K)
    sched = str(schedule).lower().strip()
    if sched == "linear":
        frac = t
    elif sched == "quadratic":
        frac = t**2
    elif sched in ("sqrt", "square_root"):
        frac = np.sqrt(t)
    elif sched in ("log", "logarithmic"):
        frac = np.log1p(t * (math.e - 1.0))
    elif sched in ("exp", "exponential"):
        frac = (np.exp(t) - 1.0) / (math.e - 1.0)
    elif sched == "cosine":
        frac = 0.5 * (1.0 - np.cos(math.pi * t))
    else:
        raise ValueError(
            f"Unknown NCM level schedule {schedule!r}; choose from "
            "('quadratic', 'linear', 'sqrt', 'log', 'exp', 'cosine', 'manual')"
        )

    raw = np.round(s0 + (L - s0) * frac).astype(int).tolist()
    raw[0] = s0
    raw[-1] = L
    # Ensure strict monotonicity when K <= L
    for i in range(1, K):
        if raw[i] <= raw[i - 1]:
            raw[i] = min(L - (K - 1 - i), raw[i - 1] + 1)
    for i in range(K - 2, -1, -1):
        if raw[i] >= raw[i + 1]:
            raw[i] = max(1, raw[i + 1] - 1)
    raw[-1] = L
    return tuple(int(x) for x in raw)


def compute_ncm_level_weights(
    levels: Sequence[int],
    *,
    alpha: float = 0.5,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Compute NCM Eq. 10 per-level loss weights $w_k = \\ell_k^{-\\alpha} / \\sum_j \\ell_j^{1-\\alpha}$.

    When multiplied by the sum over $\\ell_k$ positions in level $k$:
    - $\\alpha = 0$: every code position across all levels has equal weight ($1 / \\sum_j \\ell_j$).
    - $\\alpha = 1$: every level $k$ has equal total weight ($1 / (K \\cdot \\ell_k)$ per position).
    """
    ells = torch.tensor([float(l) for l in levels], dtype=torch.float32, device=device)
    denom = (ells ** (1.0 - float(alpha))).sum().clamp_min(1e-8)
    return (ells ** (-float(alpha))) / denom


# ---------------------------------------------------------------------------
# 2. Residual Quantization Convolution (`Phi`) & Cascaded Stride Down/Upsamplers
# ---------------------------------------------------------------------------


class PhiResidualConv(nn.Conv1d):
    """Per-level residual 1D convolution after upsampling: $(1 - r) h + r \\, \\mathrm{Conv1d}_3(h)$."""

    def __init__(
        self,
        embed_dim: int,
        quant_resi: float = 0.5,
        *,
        kernel_size: int = 3,
        learnable_ratio: bool = False,
    ) -> None:
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        init_ratio = min(max(abs(float(quant_resi)), 1e-4), 1.0 - 1e-4)
        self._learnable_ratio = bool(learnable_ratio)
        if self._learnable_ratio:
            init_logit = math.log(init_ratio / (1.0 - init_ratio))
            self.ratio_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
            self._ratio_const = init_ratio
        else:
            self.register_parameter("ratio_logit", None)
            self._ratio_const = float(abs(quant_resi))

    @property
    def resi_ratio(self) -> Tensor | float:
        if self._learnable_ratio and self.ratio_logit is not None:
            return torch.sigmoid(self.ratio_logit)
        return self._ratio_const

    def forward(self, h_bcl: Tensor) -> Tensor:
        r = self.resi_ratio
        conv_out = F.conv1d(
            h_bcl,
            self.weight.to(h_bcl.dtype),
            self.bias.to(h_bcl.dtype) if self.bias is not None else None,
            stride=self.stride,
            padding=self.padding,
        )
        if isinstance(r, Tensor):
            r_t = r.to(h_bcl.dtype)
            return h_bcl * (1.0 - r_t) + conv_out * r_t
        return h_bcl * (1.0 - r) + conv_out * r


def _plan_cascade_strides(total_stride: int, base: int = 4) -> list[int]:
    s = max(1, int(total_stride))
    b = max(2, int(base))
    strides: list[int] = []
    while s > 1:
        f = min(b, s)
        while f > 1 and s % f != 0:
            f -= 1
        if f <= 1:
            strides.append(s)
            s = 1
        else:
            strides.append(f)
            s //= f
    return strides or [1]


def _make_learned_down_l1(d_vq: int, stride: int = 4) -> nn.Conv1d:
    conv = nn.Conv1d(d_vq, d_vq, kernel_size=stride, stride=stride, bias=True)
    with torch.no_grad():
        conv.weight.zero_()
        eye_idx = torch.arange(d_vq)
        conv.weight[eye_idx, eye_idx, :] = 1.0 / float(max(1, stride))
        if conv.bias is not None:
            conv.bias.zero_()
    return conv


def _make_learned_up_l1(d_vq: int, stride: int = 4) -> nn.ConvTranspose1d:
    conv_t = nn.ConvTranspose1d(d_vq, d_vq, kernel_size=stride, stride=stride, bias=True)
    with torch.no_grad():
        conv_t.weight.zero_()
        eye_idx = torch.arange(d_vq)
        conv_t.weight[eye_idx, eye_idx, :] = 1.0
        if conv_t.bias is not None:
            conv_t.bias.zero_()
    return conv_t


def _make_cascaded_down_l1(d_vq: int, total_stride: int, base: int = 4) -> nn.Sequential:
    strides = _plan_cascade_strides(total_stride, base=base)
    return nn.Sequential(*[_make_learned_down_l1(d_vq, stride=s) for s in strides])


def _make_cascaded_up_l1(d_vq: int, total_stride: int, base: int = 4) -> nn.Sequential:
    strides = list(reversed(_plan_cascade_strides(total_stride, base=base)))
    return nn.Sequential(*[_make_learned_up_l1(d_vq, stride=s) for s in strides])


# ---------------------------------------------------------------------------
# 3. Stage 1: Multiscale Residual Vector Quantizer (`MultiscaleResidualQuantizer`)
# ---------------------------------------------------------------------------


@dataclass
class NCMConfig:
    """Configuration for Next-Concept Modeling (Stage-1 VQ-VAE + Stage-2 MCL4-COE Generator)."""

    target_length: int = 64
    num_levels: int = 6
    level_schedule: str = "quadratic"
    manual_levels: tuple[int, ...] = ()
    codebook_size: int = 1024
    vq_embed_dim: int = 128
    d_model: int = 128
    vocab_size: int = 260
    pad_id: int = 256
    rms_norm_eps: float = 1e-5
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    restart_threshold: float = 0.05
    quant_resi_ratio: float = 0.5
    learnable_quant_resi: bool = False
    learned_l1_sampling: bool = True
    learned_l1_cascade: bool = True
    learned_l1_cascade_base: int = 4
    fine_drop_prob: float = 0.5
    fine_drop_min_keep: int = 1
    pre_quant_groups: int = 8
    vq_decoder_layers: int = 1
    vq_decoder_heads: int = 4
    tie_heads_to_codebook: bool = True
    tie_heads_learn_scale: bool = True
    tie_heads_bias: bool = True
    post_upsample_conv: bool = True
    post_upsample_kernel: int = 3
    loss_alpha: float = 0.5
    conditioning_corruption_rate: float = 0.25
    # MCL4-COE Specific Enhancements:
    scale_adaptive_top_k: bool = True
    coarse_top_k: int = 16
    mid_top_k: int = 8
    fine_top_k: int = 4
    coarse_level_cutoff: int = 2
    mid_level_cutoff: int = 4
    enable_mid_hierarchy_coe: bool = True
    enable_coe_head_bridge: bool = True
    coe_decision_prior_weight: float = 0.0
    coarse_branch_candidates: int = 2
    deduction_veto_threshold: float = 0.60
    metacognition_halt_threshold: float = 0.88
    min_early_exit_level: int = 3
    # Convenience aliases for external configs:
    seq_len: int | None = None
    codebook_dim: int | None = None
    scale_adaptive_moe: bool | None = None

    def __post_init__(self) -> None:
        if self.seq_len is not None:
            self.target_length = int(self.seq_len)
        else:
            self.seq_len = int(self.target_length)
        if self.codebook_dim is not None:
            self.vq_embed_dim = int(self.codebook_dim)
        else:
            self.codebook_dim = int(self.vq_embed_dim)
        if self.scale_adaptive_moe is not None:
            self.scale_adaptive_top_k = bool(self.scale_adaptive_moe)
        else:
            self.scale_adaptive_moe = bool(self.scale_adaptive_top_k)

    @property
    def levels(self) -> tuple[int, ...]:
        return compute_ncm_levels(
            self.target_length,
            self.num_levels,
            schedule=self.level_schedule,
            manual_levels=self.manual_levels,
        )


@dataclass
class QuantizerOutput:
    """Output of `MultiscaleResidualQuantizer.forward`."""

    z_hat: Tensor
    vq_loss: Tensor
    code_indices: list[Tensor]
    concat_indices: Tensor
    z_hats_all: list[Tensor]
    z_levels_all: list[Tensor]
    per_level_utilization: list[float]
    global_utilization: float
    active_cutoff_level: int


class MultiscaleResidualQuantizer(nn.Module):
    """NCM Stage-1 Multiscale Residual Vector Quantizer with EMA codebooks & nested dropout."""

    def __init__(self, config: NCMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = int(config.vq_embed_dim)
        self.levels: tuple[int, ...] = config.levels
        self.num_levels = len(self.levels)
        self.target_length = int(self.levels[-1])
        self.codebook_size = int(config.codebook_size)
        self.commitment_cost = float(config.commitment_cost)
        self.ema_decay = float(config.ema_decay)
        self.ema_eps = float(config.ema_eps)
        self.restart_threshold = float(config.restart_threshold)
        self.fine_drop_prob = float(config.fine_drop_prob)
        self.fine_drop_min_keep = max(1, int(config.fine_drop_min_keep))

        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.codebook_size, self.embed_dim) for _ in range(self.num_levels)]
        )
        for i, emb in enumerate(self.embeddings):
            nn.init.normal_(emb.weight, mean=0.0, std=1.0 / math.sqrt(self.embed_dim))
            emb.weight.requires_grad_(False)
            self.register_buffer(f"ema_cluster_size_{i}", torch.ones(self.codebook_size))
            self.register_buffer(f"ema_w_{i}", emb.weight.data.clone())
            self.register_buffer(
                f"codebook_hit_{i}", torch.zeros(self.codebook_size, dtype=torch.long)
            )

        self.quant_resi = nn.ModuleList(
            [
                PhiResidualConv(
                    self.embed_dim,
                    config.quant_resi_ratio,
                    learnable_ratio=config.learnable_quant_resi,
                )
                if abs(config.quant_resi_ratio) > 1e-6
                else nn.Identity()
                for _ in range(self.num_levels)
            ]
        )

        ell_1 = int(self.levels[0])
        self.learned_l1_sampling = bool(
            config.learned_l1_sampling and self.target_length % ell_1 == 0 and self.target_length > ell_1
        )
        if self.learned_l1_sampling:
            stride = self.target_length // ell_1
            if config.learned_l1_cascade:
                self.learned_down_l1: nn.Module | None = _make_cascaded_down_l1(
                    self.embed_dim, stride, base=config.learned_l1_cascade_base
                )
                self.learned_up_l1: nn.Module | None = _make_cascaded_up_l1(
                    self.embed_dim, stride, base=config.learned_l1_cascade_base
                )
            else:
                self.learned_down_l1 = _make_learned_down_l1(self.embed_dim, stride=stride)
                self.learned_up_l1 = _make_learned_up_l1(self.embed_dim, stride=stride)
        else:
            self.learned_down_l1 = None
            self.learned_up_l1 = None

        self._stoch_encode = False
        self._stoch_temp = 0.5
        self._stoch_top_k = 10
        self._stoch_top_p = 0.95

    @staticmethod
    def _distances(flat_input: Tensor, emb_w: Tensor) -> Tensor:
        f32_in = flat_input.float()
        f32_w = emb_w.float()
        return (
            f32_in.square().sum(dim=1, keepdim=True)
            + f32_w.square().sum(dim=1)
            - 2.0 * torch.matmul(f32_in, f32_w.t())
        )

    def _down_for_quant(self, z_blc: Tensor, level_idx: int, target_len: int) -> Tensor:
        if z_blc.shape[1] == target_len:
            return z_blc
        z_bcl = z_blc.transpose(-1, -2).contiguous()
        if (
            level_idx == 0
            and self.learned_l1_sampling
            and self.learned_down_l1 is not None
            and z_blc.shape[1] == self.target_length
        ):
            return self.learned_down_l1(z_bcl).transpose(-1, -2).contiguous()
        return F.interpolate(z_bcl, size=(target_len,), mode="area").transpose(-1, -2).contiguous()

    def _up_to_full(self, q_blc: Tensor, level_idx: int, full_len: int) -> Tensor:
        if q_blc.shape[1] == full_len:
            return q_blc
        q_bcl = q_blc.transpose(-1, -2).contiguous()
        if (
            level_idx == 0
            and self.learned_l1_sampling
            and self.learned_up_l1 is not None
            and full_len == self.target_length
            and q_blc.shape[1] == self.levels[0]
        ):
            return self.learned_up_l1(q_bcl).transpose(-1, -2).contiguous()
        return F.interpolate(q_bcl, size=(full_len,), mode="linear", align_corners=False).transpose(-1, -2).contiguous()

    def _apply_quant_resi(self, h_blc: Tensor, level_idx: int) -> Tensor:
        mod = self.quant_resi[min(level_idx, len(self.quant_resi) - 1)]
        if isinstance(mod, nn.Identity):
            return h_blc
        return mod(h_blc.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()

    @torch.no_grad()
    def _restart_dead_codes(
        self, flat_input: Tensor, ema_cs: Tensor, ema_w: Tensor, emb_i: nn.Embedding
    ) -> None:
        if self.restart_threshold <= 0.0 or flat_input.shape[0] == 0:
            return
        mean_cs = ema_cs.mean().clamp_min(1e-6)
        dead = (ema_cs / mean_cs) < self.restart_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        src = torch.randint(0, flat_input.shape[0], (n_dead,), device=flat_input.device)
        new_vecs = flat_input[src].to(emb_i.weight.dtype)
        emb_i.weight.data[dead] = new_vecs
        ema_cs[dead] = mean_cs
        ema_w[dead] = (new_vecs.float() * mean_cs).to(ema_w.dtype)

    @torch.no_grad()
    def kmeans_init_level(self, level_idx: int, samples: Tensor, n_iters: int = 15) -> None:
        """Initialize level `level_idx` codebook via k-means over `[N, d_vq]` residual vectors."""
        v = self.codebook_size
        dev = samples.device
        xs_all = samples.detach().float()
        n = xs_all.shape[0]
        if n == 0:
            return
        if n >= v:
            perm = torch.randperm(n, device=dev)[:v]
            centers = xs_all[perm].clone()
        else:
            idx = torch.randint(0, n, (v,), device=dev)
            centers = xs_all[idx].clone()
        for _ in range(max(1, n_iters)):
            d2 = self._distances(xs_all, centers)
            assigns = d2.argmin(dim=1)
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(v, device=dev, dtype=torch.float32)
            new_centers.index_add_(0, assigns, xs_all)
            counts.index_add_(0, assigns, torch.ones_like(assigns, dtype=torch.float32))
            mask = counts > 0
            new_centers[mask] /= counts[mask].unsqueeze(1)
            empty = (~mask).nonzero(as_tuple=False).flatten()
            if empty.numel() > 0:
                r_idx = torch.randint(0, n, (empty.numel(),), device=dev)
                new_centers[empty] = xs_all[r_idx]
            centers = new_centers
        emb = self.embeddings[level_idx]
        emb.weight.data.copy_(centers.to(emb.weight.dtype))
        getattr(self, f"ema_cluster_size_{level_idx}").fill_(1.0)
        getattr(self, f"ema_w_{level_idx}").copy_(centers.to(emb.weight.dtype))

    @torch.no_grad()
    def init_codebooks_from_z(self, z: Tensor, n_iters: int = 15) -> None:
        """Initialize all $K$ coarse-to-fine codebooks sequentially via residual k-means on `z` (`[B, L, d_vq]`)."""
        z_rest = z.detach().float().clone()
        b, full_len, c = z_rest.shape
        for i, level_len in enumerate(self.levels):
            eff_len = min(full_len, int(level_len)) if i < self.num_levels - 1 else full_len
            flat_blc = (
                self._down_for_quant(z_rest, i, eff_len) if i < self.num_levels - 1 else z_rest
            )
            self.kmeans_init_level(i, flat_blc.reshape(-1, c), n_iters=n_iters)
            emb_i = self.embeddings[i]
            d2 = self._distances(flat_blc.reshape(-1, c), emb_i.weight)
            idx = d2.argmin(dim=1).view(b, eff_len)
            q_blc = emb_i(idx).float()
            quantized = (
                self._up_to_full(q_blc, i, full_len) if i < self.num_levels - 1 else q_blc
            )
            quantized = self._apply_quant_resi(quantized, i)
            z_rest = z_rest - quantized

    def forward(self, z: Tensor, *, force_cutoff_level: int | None = None) -> QuantizerOutput:
        """Quantize `z` (`[B, L, d_vq]`) across all $K$ coarse-to-fine residual levels."""
        orig_dtype = z.dtype
        z_f = z.float()
        if self.training and int(getattr(self, "codebook_hit_0").sum().item()) == 0:
            self.init_codebooks_from_z(z_f, n_iters=10)
        b, full_len, c = z_f.shape
        z_no_grad = z_f.detach()
        z_rest = z_no_grad.clone()
        z_hat = torch.zeros_like(z_rest)

        z_hats_all: list[Tensor] = []
        z_levels_all: list[Tensor] = []
        code_indices: list[Tensor] = []
        vq_loss = torch.zeros((), device=z.device, dtype=torch.float32)

        for i, level_len in enumerate(self.levels):
            emb_i = self.embeddings[i]
            eff_len = min(full_len, int(level_len)) if i < self.num_levels - 1 else full_len
            flat_input_blc = (
                self._down_for_quant(z_rest, i, eff_len) if i < self.num_levels - 1 else z_rest
            )
            flat_input = flat_input_blc.reshape(-1, c)
            distances = self._distances(flat_input, emb_i.weight)
            indices = torch.argmin(distances, dim=1).view(b, eff_len)
            code_indices.append(indices)

            if self.training:
                with torch.no_grad():
                    encodings = torch.zeros(
                        b * eff_len, self.codebook_size, device=z.device, dtype=torch.float32
                    )
                    encodings.scatter_(1, indices.view(-1, 1), 1.0)
                    ema_cs: Tensor = getattr(self, f"ema_cluster_size_{i}")
                    ema_w: Tensor = getattr(self, f"ema_w_{i}")
                    if ema_cs.dtype != torch.float32:
                        ema_cs = ema_cs.float()
                        setattr(self, f"ema_cluster_size_{i}", ema_cs)
                    if ema_w.dtype != torch.float32:
                        ema_w = ema_w.float()
                        setattr(self, f"ema_w_{i}", ema_w)
                    enc_sum = encodings.sum(0)
                    dw = torch.matmul(encodings.t(), flat_input.detach())
                    ema_cs.mul_(self.ema_decay).add_(enc_sum, alpha=1.0 - self.ema_decay)
                    n_tot = ema_cs.sum().clamp_min(1e-6)
                    smoothed = (
                        (ema_cs + self.ema_eps)
                        / (n_tot + self.codebook_size * self.ema_eps)
                        * n_tot
                    )
                    ema_w.mul_(self.ema_decay).add_(dw.to(ema_w.dtype), alpha=1.0 - self.ema_decay)
                    emb_i.weight.data.copy_((ema_w.float() / smoothed.unsqueeze(1)).to(emb_i.weight.dtype))
                    self._restart_dead_codes(flat_input.detach(), ema_cs, ema_w, emb_i)

            q_blc = emb_i(indices).float()
            # Straight-through estimator on learned level-0 downsampler (matches HazyResearch/ncm)
            if i == 0 and self.learned_l1_sampling:
                q_blc = flat_input_blc + (q_blc - flat_input_blc).detach()

            quantized = (
                self._up_to_full(q_blc, i, full_len) if i < self.num_levels - 1 else q_blc
            )
            quantized = self._apply_quant_resi(quantized, i)

            z_hat = z_hat + quantized
            z_rest = z_rest - quantized
            z_hats_all.append(z_hat.to(orig_dtype))
            z_levels_all.append(
                self._down_for_quant(z_hat, i, eff_len).to(orig_dtype)
                if i < self.num_levels - 1
                else z_hat.to(orig_dtype)
            )

            vq_loss = (
                vq_loss
                + self.commitment_cost * F.mse_loss(z_hat.detach(), z_f)
                + F.mse_loss(z_hat, z_no_grad)
            )
            with torch.no_grad():
                getattr(self, f"codebook_hit_{i}")[indices.reshape(-1)] = 1

        vq_loss = vq_loss / float(self.num_levels)

        # Nested dropout (`fine_drop`): randomly truncate at cutoff level `k_cut` during training
        active_cutoff = self.num_levels - 1
        if force_cutoff_level is not None:
            active_cutoff = max(0, min(self.num_levels - 1, int(force_cutoff_level)))
            z_hat = z_hats_all[active_cutoff].float()
        elif (
            self.training
            and self.fine_drop_prob > 0.0
            and self.num_levels >= 2
            and float(torch.rand(()).item()) < self.fine_drop_prob
        ):
            kmin = max(0, min(self.fine_drop_min_keep - 1, self.num_levels - 2))
            kmax = self.num_levels - 2
            if kmax >= kmin:
                active_cutoff = int(torch.randint(kmin, kmax + 1, ()).item())
                z_hat = z_hats_all[active_cutoff].float()

        # Straight-through reconstruction gradient back to encoder
        z_hat_st = (z_f + (z_hat - z_f).detach()).to(orig_dtype)

        per_level_util = [
            float(torch.sum(getattr(self, f"codebook_hit_{i}")).item()) / float(self.codebook_size)
            for i in range(self.num_levels)
        ]
        global_util = sum(per_level_util) / float(max(1, self.num_levels))

        return QuantizerOutput(
            z_hat=z_hat_st,
            vq_loss=vq_loss,
            code_indices=code_indices,
            concat_indices=torch.cat(code_indices, dim=1),
            z_hats_all=z_hats_all,
            z_levels_all=z_levels_all,
            per_level_utilization=per_level_util,
            global_utilization=global_util,
            active_cutoff_level=active_cutoff,
        )

    def quantize_for_coe(
        self, z: Tensor, valid: Tensor | None = None
    ) -> tuple[list[Tensor], list[Tensor], Tensor]:
        """Deterministically quantize `z` (`[N, T, d_vq]`) across all $K$ scales for COE operator heads.

        - Does NOT mutate EMA cluster counters or draw random `fine_drop` cutoffs, ensuring
          100% row-independent determinism (preserving 0-ULP bitwise permutation invariance in `Decision`).
        - Preserves straight-through gradients into both the encoder latents `z` and the per-level
          codebooks `embeddings[k]`.
        - Returns `(z_hats_all, code_indices, scale_entropy)` where `scale_entropy` has shape `[N, K]`.
        """
        orig_dtype = z.dtype
        z_f = z.float()
        if valid is not None:
            z_f = z_f.masked_fill(~valid.unsqueeze(-1), 0.0)
        b, full_len, c = z_f.shape
        z_rest = z_f.detach().clone()
        z_hat = torch.zeros_like(z_f)

        z_hats_all: list[Tensor] = []
        code_indices: list[Tensor] = []
        scale_entropies: list[Tensor] = []
        inv_temp = 1.0 / max(1.0, math.sqrt(float(c)))
        norm_log_v = math.log(max(2, self.codebook_size))

        for i, level_len in enumerate(self.levels):
            emb_i = self.embeddings[i]
            eff_len = min(full_len, int(level_len)) if i < self.num_levels - 1 else full_len
            flat_input_blc = (
                self._down_for_quant(z_rest, i, eff_len) if i < self.num_levels - 1 else z_rest
            )
            flat_input = flat_input_blc.reshape(-1, c)
            distances = self._distances(flat_input, emb_i.weight)
            indices = torch.argmin(distances, dim=1).view(b, eff_len)
            code_indices.append(indices)

            with torch.no_grad():
                probs_i = F.softmax(-distances.float() * inv_temp, dim=-1)
                ent_pos = (
                    -(probs_i * probs_i.clamp_min(1e-9).log()).sum(dim=-1) / norm_log_v
                ).view(b, eff_len)
                scale_entropies.append(ent_pos.mean(dim=-1))

            q_raw = emb_i(indices).float()
            # Dual straight-through estimator: passes gradients to both codebook E_i and input z
            q_blc = q_raw + (flat_input_blc - flat_input_blc.detach())
            quantized = (
                self._up_to_full(q_blc, i, full_len) if i < self.num_levels - 1 else q_blc
            )
            quantized = self._apply_quant_resi(quantized, i)
            if valid is not None:
                quantized = quantized.masked_fill(~valid.unsqueeze(-1), 0.0)

            z_hat = z_hat + quantized
            z_rest = z_rest - quantized.detach()
            z_hat_st_i = (z_f + (z_hat - z_f.detach())).to(orig_dtype)
            if valid is not None:
                z_hat_st_i = z_hat_st_i.masked_fill(~valid.unsqueeze(-1), 0.0)
            z_hats_all.append(z_hat_st_i)

        scale_entropy_tensor = torch.stack(scale_entropies, dim=-1)  # [N, K]
        return z_hats_all, code_indices, scale_entropy_tensor

    @torch.no_grad()
    def z_to_idx(
        self, z: Tensor, *, return_zhats: bool = False, return_zlevels: bool = False
    ) -> Tensor | tuple[Any, ...]:
        """Deterministically (or stochastically) quantize `z` (`[B, L, d_vq]`) into concatenated code indices."""
        out = self.forward(z)
        if not (return_zhats or return_zlevels):
            return out.concat_indices
        res: tuple[Any, ...] = (out.concat_indices,)
        if return_zhats:
            res = res + (out.z_hats_all,)
        if return_zlevels:
            res = res + (out.z_levels_all,)
        return res

    @torch.no_grad()
    def idx_to_zhats(
        self, concat_idx: Tensor, *, full_length: int | None = None
    ) -> list[Tensor]:
        """Reconstruct cumulative latent states `[z_hat^(1), ..., z_hat^(K)]` from concatenated indices."""
        b = concat_idx.shape[0]
        full_l = int(full_length or self.target_length)
        z_hat = torch.zeros(
            b, full_l, self.embed_dim, device=concat_idx.device, dtype=self.embeddings[0].weight.dtype
        )
        z_hats: list[Tensor] = []
        offset = 0
        for i, level_len in enumerate(self.levels):
            eff_len = min(full_l, int(level_len)) if i < self.num_levels - 1 else full_l
            idx_i = concat_idx[:, offset : offset + eff_len]
            offset += eff_len
            emb_at_level = self.embeddings[i](idx_i)
            z_q = (
                self._up_to_full(emb_at_level, i, full_l)
                if i < self.num_levels - 1
                else emb_at_level
            )
            z_q = self._apply_quant_resi(z_q, i)
            z_hat = z_hat + z_q
            z_hats.append(z_hat.clone())
        return z_hats

    @torch.no_grad()
    def build_corrupted_ncm_inputs(
        self,
        code_indices: Sequence[Tensor],
        *,
        full_length: int | None = None,
        corruption_rate: float = 0.0,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Construct Stage-2 NCM teacher-forced inputs $\\mathrm{Down}_k(\\tilde{z}^{(k-1)})$ with Conditioning Corruption.

        Follows Section 3.4 & Appendix F.3 of arXiv:2610.14162:
        - Replaces a linearly decreasing fraction of conditioning codes $\\tilde{r}_k$ with uniform draws
          from $\\mathcal{V}_k$, keeping ground-truth labels $r_k^*$ uncorrupted.
        - Returns `(level_inputs, parent_code_ids, cumulative_z_hats)` where:
          * `level_inputs[k]` has shape `[B, ell_k, d_vq]` for $k = 0 \\dots K-1$
          * `parent_code_ids[k]` has shape `[B, ell_k]` (nearest-upsampled parent codes for multiscale `PLE`)
          * `cumulative_z_hats[k]` has shape `[B, L, d_vq]` ($\\tilde{z}^{(k)}$ after level $k$)
        """
        b = code_indices[0].shape[0]
        dev = code_indices[0].device
        dtype = self.embeddings[0].weight.dtype
        full_l = int(full_length or self.target_length)

        z_hat = torch.zeros(b, full_l, self.embed_dim, device=dev, dtype=dtype)
        level_inputs: list[Tensor] = []
        parent_code_ids: list[Tensor] = []
        cumulative_z_hats: list[Tensor] = []

        prev_codes: Tensor | None = None
        for k in range(self.num_levels):
            eff_len = (
                min(full_l, int(self.levels[k])) if k < self.num_levels - 1 else full_l
            )
            # Input to level k is Down_k(z_hat^(k-1)) (for k=0, z_hat^(0) = 0)
            if k == 0:
                inp_k = torch.zeros(b, eff_len, self.embed_dim, device=dev, dtype=dtype)
                parent_k = torch.zeros(b, eff_len, device=dev, dtype=torch.long)
            else:
                inp_k = (
                    F.interpolate(
                        z_hat.transpose(-1, -2), size=(eff_len,), mode="area"
                    ).transpose(-1, -2).contiguous()
                    if eff_len != full_l
                    else z_hat.clone()
                )
                assert prev_codes is not None
                if prev_codes.shape[1] == eff_len:
                    parent_k = prev_codes
                else:
                    parent_k = (
                        F.interpolate(
                            prev_codes.float().unsqueeze(1), size=(eff_len,), mode="nearest"
                        )
                        .squeeze(1)
                        .long()
                    )
            level_inputs.append(inp_k)
            parent_code_ids.append(parent_k)

            # Apply linearly decreasing conditioning corruption across levels 0 .. K-2
            r_k = code_indices[k]
            if corruption_rate > 0.0 and self.num_levels > 1:
                decay_factor = max(0.1, 1.0 - float(k) / float(self.num_levels))
                p_k = min(0.9, float(corruption_rate) * decay_factor)
                mask = torch.rand(r_k.shape, device=dev) < p_k
                rand_codes = torch.randint(0, self.codebook_size, r_k.shape, device=dev)
                r_tilde = torch.where(mask, rand_codes, r_k)
            else:
                r_tilde = r_k
            prev_codes = r_tilde

            emb_k = self.embeddings[k](r_tilde)
            up_k = (
                self._up_to_full(emb_k, k, full_l) if k < self.num_levels - 1 else emb_k
            )
            up_k = self._apply_quant_resi(up_k, k)
            z_hat = z_hat + up_k
            cumulative_z_hats.append(z_hat.clone())

        return level_inputs, parent_code_ids, cumulative_z_hats

    @torch.no_grad()
    def step_next_level_reconstruction(
        self,
        level_idx: int,
        z_hat: Tensor,
        sampled_codes: Tensor,
        *,
        full_length: int | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Update cumulative reconstruction $\\hat{z}^{(k)}$ from `sampled_codes` $r_k$ and prepare input for level $k+1$."""
        full_l = int(full_length or z_hat.shape[1])
        emb_k = self.embeddings[level_idx](sampled_codes).to(z_hat.dtype)
        up_k = (
            self._up_to_full(emb_k, level_idx, full_l)
            if level_idx < self.num_levels - 1
            else emb_k
        )
        up_k = self._apply_quant_resi(up_k, level_idx)
        z_hat_next = z_hat + up_k

        if level_idx + 1 < self.num_levels:
            next_len = (
                min(full_l, int(self.levels[level_idx + 1]))
                if (level_idx + 1) < self.num_levels - 1
                else full_l
            )
            next_in = (
                F.interpolate(
                    z_hat_next.transpose(-1, -2), size=(next_len,), mode="area"
                )
                .transpose(-1, -2)
                .contiguous()
                if next_len != full_l
                else z_hat_next.clone()
            )
            if sampled_codes.shape[1] == next_len:
                parent_next = sampled_codes
            else:
                parent_next = (
                    F.interpolate(
                        sampled_codes.float().unsqueeze(1), size=(next_len,), mode="nearest"
                    )
                    .squeeze(1)
                    .long()
                )
            return z_hat_next, next_in, parent_next
        return z_hat_next, None, None


# ---------------------------------------------------------------------------
# 4. Stage 1 VQ-VAE Tokenizer (Donor-Anchored or Standalone)
# ---------------------------------------------------------------------------


class BidirectionalVQDecoderBlock(nn.Module):
    """Lightweight bidirectional Transformer block for the Stage-1 VQ-VAE decoder ($D_\\psi$)."""

    def __init__(self, embed_dim: int, n_heads: int = 4, eps: float = 1e-6) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = max(1, min(embed_dim, n_heads))
        while embed_dim % self.n_heads != 0 and self.n_heads > 1:
            self.n_heads -= 1
        self.head_dim = embed_dim // self.n_heads
        self.ln_1 = RMSNorm(embed_dim, eps=eps)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln_2 = RMSNorm(embed_dim, eps=eps)
        self.up_proj = nn.Linear(embed_dim, 4 * embed_dim, bias=False)
        self.down_proj = nn.Linear(2 * embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, valid: Tensor | None = None) -> Tensor:
        b, n, c = x.shape
        h = self.ln_1(x)
        qkv = self.qkv(h).view(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        q, k = self.q_norm(q), self.k_norm(k)
        if valid is not None:
            allowed = valid[:, None, None, :].expand(b, 1, n, n)
            fallback = torch.eye(n, dtype=torch.bool, device=x.device)[None, None]
            allowed = allowed | (~allowed.any(-1, keepdim=True) & fallback)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out_proj(attn_out.transpose(1, 2).contiguous().view(b, n, c))
        gate, val = self.up_proj(self.ln_2(x)).chunk(2, dim=-1)
        x = x + self.down_proj(F.silu(gate) * val)
        if valid is not None:
            x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        return x


@dataclass
class VQVAETokenizerOutput:
    """Output of `DonorAnchoredMultiscaleVQVAE.forward`."""

    logits: Tensor
    recon_loss: Tensor
    vq_loss: Tensor
    total_loss: Tensor
    recon_accuracy: float
    quantizer_out: QuantizerOutput


class DonorAnchoredMultiscaleVQVAE(nn.Module):
    """Multiscale Residual VQ-VAE Tokenizer anchored to `MCL4CEDBackbone` embeddings and `lm_head`.

    Why Donor-Anchored Design Matters (`docs/07_PROPER_DONOR_RL_MITOSIS_PROTOCOL.md` & `docs/10_NCM_MCL4_COE_ARCHITECTURE_MAPPING.md`):
    - In `HazyResearch/ncm`, `enc_layers = 0` (`wte + wpe + GroupNorm(8, affine=False)`) and `dec_layers = 1`.
    - By projecting directly from the frozen/anchored `MCL4CEDBackbone.token_embedding` into `vq_embed_dim`
      and decoding through `BidirectionalVQDecoderBlock` back into the frozen `MCL4CEDBackbone.lm_head`,
      the multiscale concept codebooks align directly with the donor's multi-trillion-token lexical
      manifold with **zero drift to the donor's weights**.
    """

    def __init__(
        self,
        config: NCMConfig,
        *,
        backbone: MCL4CEDBackbone | None = None,
        vocab_size: int = 260,
        d_model: int = 128,
        pad_id: int = 256,
        logits_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config
        # Store backbone in a 1-element tuple so PyTorch does not duplicate its parameters in state_dict
        self._backbone_ref = (backbone,) if backbone is not None else ()
        if backbone is not None:
            vocab_size = int(backbone.config.vocab_size)
            d_model = int(backbone.config.d_model)
            pad_id = int(backbone.config.pad_id)
            logits_scaling = float(getattr(backbone.config, "logits_scaling", 1.0))

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.logits_scaling = logits_scaling
        self.vq_dim = int(config.vq_embed_dim)

        if backbone is None:
            self.standalone_wte: nn.Embedding | None = nn.Embedding(
                vocab_size, d_model, padding_idx=pad_id if 0 <= pad_id < vocab_size else None
            )
            self.standalone_lm_head: nn.Linear | None = nn.Linear(d_model, vocab_size, bias=False)
            self.standalone_lm_head.weight = self.standalone_wte.weight
        else:
            self.standalone_wte = None
            self.standalone_lm_head = None

        self.wpe = nn.Embedding(max(512, config.target_length * 2), self.vq_dim)
        nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)

        self.enc_proj = nn.Linear(d_model, self.vq_dim, bias=False)
        self.enc_conv = nn.Conv1d(
            self.vq_dim, self.vq_dim, kernel_size=3, padding=1, groups=1, bias=True
        )
        with torch.no_grad():
            self.enc_conv.weight.zero_()
            if self.enc_conv.bias is not None:
                self.enc_conv.bias.zero_()

        # affine=False as in HazyResearch/ncm so ||z - e||^2 cannot inflate without bound
        n_groups = max(1, min(self.vq_dim, int(config.pre_quant_groups)))
        while self.vq_dim % n_groups != 0 and n_groups > 1:
            n_groups -= 1
        self.pre_quant_norm = nn.GroupNorm(n_groups, self.vq_dim, affine=False)

        self.vq = MultiscaleResidualQuantizer(config)

        self.dec_blocks = nn.ModuleList(
            [
                BidirectionalVQDecoderBlock(self.vq_dim, n_heads=config.vq_decoder_heads)
                for _ in range(max(1, int(config.vq_decoder_layers)))
            ]
        )
        self.dec_norm = RMSNorm(self.vq_dim)
        self.dec_out_proj = nn.Linear(self.vq_dim, d_model, bias=False)

    @property
    def backbone(self) -> MCL4CEDBackbone | None:
        return self._backbone_ref[0] if self._backbone_ref else None

    def _lookup_token_embeddings(self, token_ids: Tensor) -> Tensor:
        safe_ids = token_ids.clamp(0, self.vocab_size - 1).long()
        if self.backbone is not None:
            with torch.no_grad():
                emb = self.backbone.token_embedding(safe_ids)
                if self.backbone.config.embedding_multiplier != 1.0:
                    emb = emb * float(self.backbone.config.embedding_multiplier)
            return emb
        assert self.standalone_wte is not None
        return self.standalone_wte(safe_ids)

    def _project_to_vocab_logits(self, hidden_d_model: Tensor) -> Tensor:
        if self.backbone is not None:
            fn = self.backbone.final_norm
            x_f = hidden_d_model.float()
            rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + fn.eps)
            normed = (x_f * rms).to(hidden_d_model.dtype) * fn.weight.detach().to(hidden_d_model.dtype)
            logits = F.linear(normed, self.backbone.lm_head.weight.detach().to(normed.dtype))
            if self.logits_scaling != 1.0:
                logits = logits / float(self.logits_scaling)
            return logits
        assert self.standalone_lm_head is not None
        return self.standalone_lm_head(hidden_d_model)

    def encode_pre_quant(self, token_ids: Tensor, valid: Tensor | None = None) -> Tensor:
        """Map `token_ids` (`[B, L]`) into scale-invariant pre-quantized latents `z` (`[B, L, d_vq]`)."""
        b, seq_len = token_ids.shape
        if valid is None:
            valid = (token_ids != self.pad_id) if self.pad_id >= 0 else torch.ones_like(token_ids, dtype=torch.bool)
        tok_emb = self._lookup_token_embeddings(token_ids)
        z = self.enc_proj(tok_emb.to(self.enc_proj.weight.dtype))
        pos = torch.arange(seq_len, device=token_ids.device, dtype=torch.long).clamp(
            max=self.wpe.num_embeddings - 1
        )
        z = z + self.wpe(pos).unsqueeze(0).to(z.dtype)
        z_conv = self.enc_conv(z.transpose(1, 2)).transpose(1, 2)
        z = z + z_conv
        z = self.pre_quant_norm(z.float().transpose(1, 2)).transpose(1, 2).to(tok_emb.dtype)
        return z.masked_fill(~valid.unsqueeze(-1), 0.0)

    def decode_hidden_from_z_hat(
        self, z_hat: Tensor, valid: Tensor | None = None
    ) -> Tensor:
        """Decode cumulative concept reconstruction `z_hat` (`[B, L, d_vq]`) to `d_model_hidden` (`[B, L, d_model]`)."""
        h = z_hat.to(self.dec_out_proj.weight.dtype)
        for blk in self.dec_blocks:
            h = blk(h, valid=valid)
        h = self.dec_norm(h)
        d_model_hidden = self.dec_out_proj(h)
        if valid is not None:
            d_model_hidden = d_model_hidden.masked_fill(~valid.unsqueeze(-1), 0.0)
        return d_model_hidden

    def decode_from_z_hat(
        self, z_hat: Tensor, valid: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Decode cumulative concept reconstruction `z_hat` (`[B, L, d_vq]`) to `(logits, d_model_hidden)`."""
        d_model_hidden = self.decode_hidden_from_z_hat(z_hat, valid=valid)
        logits = self._project_to_vocab_logits(d_model_hidden)
        return logits, d_model_hidden

    @torch.no_grad()
    def encode(self, token_ids: Tensor, valid: Tensor | None = None) -> list[Tensor]:
        """Encode `token_ids` (`[B, L]`) into per-level discrete concept code tensors `[r_1, ..., r_K]`."""
        z = self.encode_pre_quant(token_ids, valid=valid)
        q_out = self.vq(z)
        return q_out.code_indices

    @torch.no_grad()
    def reconstruct_multiscale_tokens(
        self, token_ids: Tensor, valid: Tensor | None = None
    ) -> list[Tensor]:
        """Decode token predictions at every cumulative scale $k = 1 \\dots K$ (`reconstruct_multiscales`)."""
        z = self.encode_pre_quant(token_ids, valid=valid)
        q_out = self.vq(z)
        level_preds: list[Tensor] = []
        for z_hat_k in q_out.z_hats_all:
            logits_k, _ = self.decode_from_z_hat(z_hat_k, valid=valid)
            level_preds.append(logits_k.argmax(dim=-1))
        return level_preds

    def forward(
        self,
        token_ids: Tensor,
        valid: Tensor | None = None,
        *,
        force_cutoff_level: int | None = None,
    ) -> VQVAETokenizerOutput:
        """Execute Stage-1 Multiscale Residual VQ-VAE forward & compute NCM Eq. 9 reconstruction + VQ loss."""
        if valid is None:
            valid = (
                (token_ids != self.pad_id)
                if self.pad_id >= 0
                else torch.ones_like(token_ids, dtype=torch.bool)
            )
        z = self.encode_pre_quant(token_ids, valid=valid)
        q_out = self.vq(z, force_cutoff_level=force_cutoff_level)
        logits, _ = self.decode_from_z_hat(q_out.z_hat, valid=valid)

        targets = token_ids.long().masked_fill(~valid, -100)
        recon_loss = F.cross_entropy(
            logits.float().view(-1, self.vocab_size),
            targets.view(-1),
            ignore_index=-100,
        )
        total_loss = recon_loss + q_out.vq_loss

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            n_valid = int(valid.sum().item())
            if n_valid > 0:
                recon_acc = float(((preds == token_ids) & valid).float().sum().item() / n_valid)
            else:
                recon_acc = 1.0

        return VQVAETokenizerOutput(
            logits=logits,
            recon_loss=recon_loss,
            vq_loss=q_out.vq_loss,
            total_loss=total_loss,
            recon_accuracy=recon_acc,
            quantizer_out=q_out,
        )


# ---------------------------------------------------------------------------
# 5. Distance-Tied Codebook Output Head (`TiedCodebookHead`)
# ---------------------------------------------------------------------------


class TiedCodebookHead(nn.Module):
    """Distance-form NCM output head tied directly to level $k$'s VQ codebook $E_k$.

    Inverts `argmin_j ||u - E_j||^2`: since $||u||^2$ is constant across $j$ inside the softmax,
    $-||u - E_j||^2 \\equiv 2 \\langle u, E_j \\rangle - ||E_j||^2$, normalized by $\\overline{||E||^2}$
    and scaled by learned inverse temperature $\\exp(\\text{log\\_inv\\_T})$.
    """

    def __init__(
        self,
        d_model: int,
        codebook_embedding: nn.Embedding,
        *,
        learn_scale: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        vocab_k, d_vq = codebook_embedding.weight.shape
        self.vocab_size = vocab_k
        self.d_vq = d_vq
        self._codebook_ref = (codebook_embedding,)
        self.proj = nn.Linear(d_model, d_vq, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_k)) if use_bias else None
        self.log_inv_T = (
            nn.Parameter(torch.tensor(math.log(5.0), dtype=torch.float32))
            if learn_scale
            else None
        )

    @property
    def codebook_weight(self) -> Tensor:
        return self._codebook_ref[0].weight

    def forward(self, hidden: Tensor) -> Tensor:
        u_raw = self.proj(hidden).float()
        e_w = self.codebook_weight.detach().float()
        u_norm = F.normalize(u_raw, p=2.0, dim=-1, eps=1e-6)
        e_norm = F.normalize(e_w, p=2.0, dim=-1, eps=1e-6)
        sim = torch.matmul(u_norm, e_norm.t())
        inv_t = (
            self.log_inv_T.clamp(math.log(0.5), math.log(50.0)).exp()
            if self.log_inv_T is not None
            else 5.0
        )
        logits = sim * inv_t
        if self.bias is not None:
            logits = logits + self.bias.float()
        return logits


# ---------------------------------------------------------------------------
# 6. Stage 2: Next-Concept Generator on `MCL4-CED` + 9-Operator `COE`
# ---------------------------------------------------------------------------


@dataclass
class NCMTrainingOutput:
    """Output of a single-pass all-level NCM training step on `MCL4CEDBackbone`."""

    total_loss: Tensor
    ncm_ce_loss: Tensor
    vq_recon_loss: Tensor
    vq_commit_loss: Tensor
    abstraction_align_loss: Tensor
    aux_loss: Tensor
    router_z_loss: Tensor
    per_level_ce: list[float]
    per_level_acc: list[float]
    codebook_utilization: float
    recon_accuracy: float


@dataclass
class NCMGenerationOutput:
    """Output of coarse-to-fine parallel NCM generation with mid-hierarchy COE verification."""

    token_ids: Tensor
    text: str
    code_indices: list[Tensor]
    final_z_hat: Tensor
    levels_executed: int
    total_levels: int
    halted_early: bool
    lateral_refinements: int
    per_level_active_top_k: list[int]
    per_level_metacognition: list[dict[str, float]]
    per_level_deduction: list[dict[str, float]]
    level_decoded_texts: list[str] = field(default_factory=list)


class NCMGeneratorAdapter(nn.Module):
    """Adapts `MCL4CEDBackbone` (`CEDEncoderLayer` + `CEDDecoderLayer`) into a full Next-Concept Generator."""

    def __init__(
        self,
        config: NCMConfig,
        vqvae: DonorAnchoredMultiscaleVQVAE,
        backbone: MCL4CEDBackbone,
    ) -> None:
        super().__init__()
        self.config = config
        self._vqvae_ref = (vqvae,)
        self._backbone_ref = (backbone,)
        self.levels = vqvae.vq.levels
        self.num_levels = len(self.levels)
        self.d_model = int(backbone.config.d_model)
        self.d_vq = int(config.vq_embed_dim)

        # Level-0 learned query vectors (`l0_queries` in HazyResearch/ncm)
        self.l0_queries = nn.Parameter(
            torch.zeros(1, int(self.levels[0]), self.d_vq)
        )
        nn.init.normal_(self.l0_queries, mean=0.0, std=1.0 / math.sqrt(self.d_vq))

        # Per-level post-upsample zero-init residual Conv1d (`post_upsample_convs` in HazyResearch/ncm)
        ks = int(config.post_upsample_kernel)
        if ks % 2 == 0:
            ks += 1
        self.use_post_upsample_conv = bool(config.post_upsample_conv)
        if self.use_post_upsample_conv:
            self.post_upsample_convs = nn.ModuleList(
                [
                    nn.Conv1d(self.d_vq, self.d_vq, kernel_size=ks, padding=ks // 2)
                    for _ in range(self.num_levels)
                ]
            )
            for conv in self.post_upsample_convs:
                nn.init.zeros_(conv.weight)
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
        else:
            self.post_upsample_convs = nn.ModuleList()

        # Projects resized cumulative concept reconstruction [B, ell_k, d_vq] -> [B, ell_k, d_model]
        self.ncm_in_proj = nn.Linear(self.d_vq, self.d_model, bias=True)
        self.ncm_in_norm = RMSNorm(self.d_model, eps=backbone.config.rms_norm_eps)
        self.level_embedding = nn.Embedding(self.num_levels, self.d_model)
        nn.init.normal_(self.level_embedding.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        self.ncm_pos_emb = nn.Embedding(max(512, int(config.target_length) * 2), self.d_model)
        nn.init.normal_(self.ncm_pos_emb.weight, mean=0.0, std=0.02)

        # Dedicated NCM prompt cross-attention + post-decoder refinement block (trainable even when donor is frozen)
        self.n_attn_heads = max(1, min(self.d_model, int(backbone.config.n_heads)))
        while self.d_model % self.n_attn_heads != 0 and self.n_attn_heads > 1:
            self.n_attn_heads -= 1
        self.attn_head_dim = self.d_model // self.n_attn_heads
        self.prompt_q_norm = RMSNorm(self.d_model, eps=backbone.config.rms_norm_eps)
        self.prompt_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.prompt_kv = nn.Linear(self.d_model, 2 * self.d_model, bias=False)
        self.prompt_out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.post_norm = RMSNorm(self.d_model, eps=backbone.config.rms_norm_eps)
        self.post_ffn_up = nn.Linear(self.d_model, 2 * self.d_model, bias=False)
        self.post_ffn_down = nn.Linear(self.d_model, self.d_model, bias=False)

        # Distance-tied codebook output heads (one per scale k = 0 .. K-1)
        if config.tie_heads_to_codebook:
            self.level_heads = nn.ModuleList(
                [
                    TiedCodebookHead(
                        self.d_model,
                        vqvae.vq.embeddings[k],
                        learn_scale=config.tie_heads_learn_scale,
                        use_bias=config.tie_heads_bias,
                    )
                    for k in range(self.num_levels)
                ]
            )
        else:
            self.level_heads = nn.ModuleList(
                [
                    nn.Linear(self.d_model, config.codebook_size, bias=config.tie_heads_bias)
                    for _ in range(self.num_levels)
                ]
            )

        # Track per-scale expert routing loads `[num_levels, num_experts]` for scale-aware mitosis
        self.register_buffer(
            "scale_expert_loads",
            torch.full(
                (self.num_levels, int(backbone.config.num_experts)),
                1.0 / max(1, int(backbone.config.num_experts)),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @property
    def vqvae(self) -> DonorAnchoredMultiscaleVQVAE:
        return self._vqvae_ref[0]

    @property
    def backbone(self) -> MCL4CEDBackbone:
        return self._backbone_ref[0]

    def get_scale_top_k(self, level_idx: int) -> int:
        """Return scale-adaptive active MoE expert count `top_k(k)` for level `level_idx`.

        Coarse concept levels ($k \\le k_{\\text{coarse}}$, short sequence length $\\ell_k \\le 16$)
        activate higher expert density (`coarse_top_k`) for global planning at $<2\\%$ FLOP cost,
        tapering to `fine_top_k` on long fine levels.
        """
        num_e = int(self.backbone.config.num_experts)
        if not self.config.scale_adaptive_top_k:
            return min(num_e, max(1, int(self.backbone.config.top_k)))
        if level_idx < self.config.coarse_level_cutoff:
            return min(num_e, max(1, int(self.config.coarse_top_k)))
        if level_idx < self.config.mid_level_cutoff:
            return min(num_e, max(1, int(self.config.mid_top_k)))
        return min(num_e, max(1, int(self.config.fine_top_k)))

    @staticmethod
    def build_block_diagonal_mask_and_rope(
        level_lengths: Sequence[int],
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build NCM's `[1, 1, T_total, T_total]` block-diagonal mask, level IDs, and reset RoPE positions."""
        lvl_ids = torch.cat(
            [
                torch.full((int(l),), k, device=device, dtype=torch.long)
                for k, l in enumerate(level_lengths)
            ],
            dim=0,
        )
        rope_pos = torch.cat(
            [
                torch.arange(int(l), device=device, dtype=torch.float32)
                for l in level_lengths
            ],
            dim=0,
        )
        # Block-diagonal: position i attends to position j iff lvl_ids[i] == lvl_ids[j]
        block_mask = (lvl_ids[:, None] == lvl_ids[None, :])[None, None, :, :]
        return block_mask, lvl_ids, rope_pos

    def _embed_level_input(
        self, raw_vq_input: Tensor, level_idx: int
    ) -> Tensor:
        """Apply post-upsample residual conv, linear projection, norm, scale embedding, and fractional position embedding."""
        x = raw_vq_input.to(self.ncm_in_proj.weight.dtype)
        eff_len = x.shape[1]
        if self.use_post_upsample_conv and level_idx < len(self.post_upsample_convs):
            delta = self.post_upsample_convs[level_idx](x.transpose(-1, -2)).transpose(-1, -2)
            x = x + delta
        # Inject learned l0_queries at level 0 when raw_vq_input is zero-initialized
        if level_idx == 0 and eff_len == self.l0_queries.shape[1]:
            x = x + self.l0_queries.to(x.dtype)
        h = self.ncm_in_norm(self.ncm_in_proj(x))
        lvl_emb = self.level_embedding.weight[level_idx].view(1, 1, -1).to(h.dtype)
        base_pos = self.ncm_pos_emb.weight[: max(1, int(self.config.target_length))].float()
        if eff_len == base_pos.shape[0]:
            pos_emb = base_pos.unsqueeze(0).to(h.dtype)
        elif eff_len == 1:
            pos_emb = base_pos.mean(dim=0, keepdim=True).unsqueeze(0).to(h.dtype)
        else:
            pos_emb = (
                F.interpolate(
                    base_pos.t().unsqueeze(0),
                    size=(eff_len,),
                    mode="linear",
                    align_corners=False,
                )
                .transpose(-1, -2)
                .contiguous()
                .to(h.dtype)
            )
        return h + lvl_emb + pos_emb

    def _apply_prompt_refinement(
        self,
        h: Tensor,
        prompt_hidden: Tensor | None,
        state_valid: Tensor,
        valid: Tensor,
        *,
        owner_rows: Tensor | None = None,
    ) -> Tensor:
        """Point-wise prompt cross-attention + SwiGLU FFN refinement (trainable even when donor is frozen)."""
        if prompt_hidden is not None:
            r, t_cand, width = h.shape
            q = (
                self.prompt_q(self.prompt_q_norm(h))
                .view(r, t_cand, self.n_attn_heads, self.attn_head_dim)
                .transpose(1, 2)
            )
            p_src = prompt_hidden.detach().to(self.prompt_kv.weight.dtype)
            kv = self.prompt_kv(p_src).view(
                p_src.shape[0], p_src.shape[1], 2, self.n_attn_heads, self.attn_head_dim
            )
            k, v = (part.transpose(1, 2).contiguous() for part in kv.unbind(dim=2))
            if owner_rows is not None:
                k = k.index_select(0, owner_rows)
                v = v.index_select(0, owner_rows)
                s_valid = state_valid.index_select(0, owner_rows)
            else:
                s_valid = state_valid
            allowed = s_valid[:, None, None, :].expand(r, 1, t_cand, k.shape[2])
            fallback = torch.zeros_like(allowed)
            fallback[..., 0] = True
            allowed = allowed | (~allowed.any(-1, keepdim=True) & fallback)
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            h = h + self.prompt_out(ctx.transpose(1, 2).contiguous().view(r, t_cand, width))
        gate, val = self.post_ffn_up(self.post_norm(h)).chunk(2, dim=-1)
        h = h + self.post_ffn_down(F.silu(gate) * val)
        return h.masked_fill(~valid.unsqueeze(-1), 0.0)

    def _run_decoder_stack(
        self,
        h_in: Tensor,
        parent_code_ids: Tensor | None,
        valid: Tensor,
        global_kv_cache: list[tuple[Tensor, Tensor]],
        state_valid: Tensor,
        *,
        prompt_hidden: Tensor | None = None,
        owner_rows: Tensor | None = None,
        local_attn_mask: Tensor | None = None,
        rope_positions: Tensor | None = None,
        level_idx_for_density: int | None = None,
        recurrence_steps: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the upper `CEDDecoderLayer` stack over single-level or concatenated multi-level NCM states."""
        backbone = self.backbone
        saved_top_ks = [layer.moe.top_k for layer in backbone.decoder_layers]
        if level_idx_for_density is not None:
            active_k = self.get_scale_top_k(level_idx_for_density)
            for layer in backbone.decoder_layers:
                layer.moe.top_k = active_k

        try:
            anchor = h_in
            state = anchor
            aux_losses: list[Tensor] = []
            z_losses: list[Tensor] = []
            loads: list[Tensor] = []
            visited_masks: list[Tensor | None] = [None] * len(backbone.decoder_layers)

            for step in range(max(1, recurrence_steps)):
                x = anchor if step == 0 else backbone.feedback(anchor, state, valid=valid)
                for l_idx, layer in enumerate(backbone.decoder_layers):
                    k_glob, v_glob = global_kv_cache[l_idx]
                    x, aux, z, load = layer(
                        x,
                        anchor,
                        parent_code_ids,
                        valid,
                        k_glob,
                        v_glob,
                        state_valid,
                        owner_rows=owner_rows,
                        causal_local=False,
                        causal_cross=False,
                        prior_expert_mask=visited_masks[l_idx],
                        local_attn_mask=local_attn_mask,
                        rope_positions=rope_positions,
                    )
                    if layer.moe.last_selected_mask is not None:
                        cur_m = layer.moe.last_selected_mask
                        visited_masks[l_idx] = (
                            cur_m
                            if visited_masks[l_idx] is None
                            else (visited_masks[l_idx] + cur_m).clamp_max(1.0)
                        )
                    layer.moe.last_selected_mask = None
                    aux_losses.append(aux)
                    z_losses.append(z)
                    loads.append(load)
                state = x

            fn = backbone.final_norm
            s_f = state.float()
            rms = torch.rsqrt(s_f.pow(2).mean(dim=-1, keepdim=True) + fn.eps)
            normed = ((s_f * rms).to(state.dtype) * fn.weight.detach().to(state.dtype)).masked_fill(
                ~valid.unsqueeze(-1), 0.0
            )
            normed = self._apply_prompt_refinement(
                normed,
                prompt_hidden,
                state_valid,
                valid,
                owner_rows=owner_rows,
            )
            mean_load = torch.stack(loads).mean(dim=0)
            if (
                level_idx_for_density is not None
                and 0 <= level_idx_for_density < self.num_levels
                and mean_load.numel() == self.scale_expert_loads.shape[1]
            ):
                with torch.no_grad():
                    self.scale_expert_loads[level_idx_for_density].mul_(0.95).add_(
                        mean_load.detach().to(self.scale_expert_loads.device).float(), alpha=0.05
                    )
            return (
                normed,
                torch.stack(aux_losses).mean(),
                torch.stack(z_losses).mean(),
                mean_load,
            )
        finally:
            for layer, orig_k in zip(backbone.decoder_layers, saved_top_ks):
                layer.moe.top_k = orig_k

    def forward_all_levels_parallel(
        self,
        encoded_state: EncodedState,
        level_inputs: Sequence[Tensor],
        parent_code_ids: Sequence[Tensor],
        *,
        owner_rows: Tensor | None = None,
    ) -> tuple[list[Tensor], Tensor, Tensor, Tensor]:
        """Predict code logits for all $K$ levels in a **single block-diagonal forward pass** (Algorithm 1)."""
        b = level_inputs[0].shape[0]
        dev = level_inputs[0].device
        level_lengths = [int(t.shape[1]) for t in level_inputs]

        embedded_chunks = [
            self._embed_level_input(level_inputs[k], k) for k in range(len(level_inputs))
        ]
        h_concat = torch.cat(embedded_chunks, dim=1)
        p_concat = torch.cat([p.to(dev) for p in parent_code_ids], dim=1)
        valid_concat = torch.ones((b, h_concat.shape[1]), dtype=torch.bool, device=dev)

        block_mask, _, rope_pos = self.build_block_diagonal_mask_and_rope(level_lengths, dev)
        global_kv_cache = [
            layer.project_global_kv(encoded_state.hidden, n_prefix=0)
            for layer in self.backbone.decoder_layers
        ]
        hidden_concat, aux_loss, z_loss, mean_load = self._run_decoder_stack(
            h_concat,
            p_concat,
            valid_concat,
            global_kv_cache,
            encoded_state.valid,
            prompt_hidden=encoded_state.hidden,
            owner_rows=owner_rows,
            local_attn_mask=block_mask,
            rope_positions=rope_pos,
            level_idx_for_density=None,
        )

        logits_per_level: list[Tensor] = []
        offset = 0
        for k, ell_k in enumerate(level_lengths):
            chunk_h = hidden_concat[:, offset : offset + ell_k, :]
            logits_per_level.append(self.level_heads[k](chunk_h))
            offset += ell_k

        return logits_per_level, aux_loss, z_loss, mean_load


# ---------------------------------------------------------------------------
# 6b. NCM -> COE Multiscale Head Bridge (`NCMCOEMultiscaleFeatures`, `NCMCOEHeadBridge`)
# ---------------------------------------------------------------------------


@dataclass
class NCMCOEMultiscaleFeatures:
    """Multiscale NCM concept representations extracted for the 9 COE operator heads."""

    z_coarse_seq: Tensor
    z_mid_seq: Tensor
    z_fine_seq: Tensor
    z_res_fine_seq: Tensor
    z_coarse_pool: Tensor
    z_mid_pool: Tensor
    z_fine_pool: Tensor
    z_res_fine_pool: Tensor
    scale_entropy: Tensor
    scale_consistency: Tensor
    code_indices: list[Tensor]


class NCMCOEHeadBridge(nn.Module):
    """Bridges Stage-1/Stage-2 NCM multiscale concept hierarchies directly into the 9 COE heads.

    Provides multiscale concept advantages to:
    1. `Decision`, `Evaluation`, `Induction`, `Abduction`:
       Fuses coarse global concept Pearson matching ($\\hat{z}_{\\text{coarse}}$) and multiscale
       concept matching ($[\\hat{z}_{\\text{coarse}}, \\hat{z}_{\\text{mid}}, \\hat{z}_{\\text{fine}}, \\Delta z_{\\text{fine}}]$)
       into the token-level matched-filter scores while preserving 0-ULP permutation invariance.
    2. `Deduction` (`entailed`, `contradicted`, `undetermined`):
       Explicitly decomposes premise/conclusion representations into **coarse concept alignment**
       ($\\hat{z}_{\\text{coarse}}$) vs. **fine residual differences** ($\\Delta z_{\\text{fine}} = \\hat{z}_{\\text{fine}} - \\hat{z}_{\\text{coarse}}$),
       isolating local negations (`"not"`, `"no"`) and numeric bound flips from topic overlap.
    3. `Metacognition` (`correctness`, `evidence_sufficiency`, `context_benefit`):
       Injects per-scale VQ assignment entropy $H(r_1 \\dots r_K)$, coarse-to-fine consistency
       $\\cos(\\hat{z}_{\\text{coarse}}, \\hat{z}_{\\text{fine}})$, and coarse-vs-fine similarity gaps.
    4. `Abstraction` (`brief`, `standard`, `detailed`):
       Routes `brief` $\\to \\hat{z}_{\\text{coarse}}$ ($k=0$), `standard` $\\to \\hat{z}_{\\text{mid}}$ ($k=K//2$),
       and `detailed` $\\to \\hat{z}_{\\text{fine}}$ ($k=K-1$) so each abstraction level is genuinely
       information-bottlenecked at its corresponding NCM concept scale.
    5. `Association` & Multiscale Novelty:
       Enriches 32-dim centered Pearson item embeddings with coarse/mid/fine concept vectors.
    """

    def __init__(
        self,
        config: NCMConfig,
        vqvae: DonorAnchoredMultiscaleVQVAE,
        *,
        d_model: int,
        d_proj: int = 32,
    ) -> None:
        super().__init__()
        self.config = config
        self._vqvae_ref = (vqvae,)
        self.d_vq = int(config.vq_embed_dim)
        self.d_model = int(d_model)
        self.d_proj = int(d_proj)
        self.num_levels = len(vqvae.vq.levels)

        self.vq_token_scorer = nn.Linear(self.d_vq, 1, bias=False)
        self.vq_res_scorer = nn.Linear(self.d_vq, 1, bias=False)

        # 1. Decision / Evaluation / Induction / Abduction projections
        self.state_coarse_proj = nn.Linear(self.d_vq, self.d_proj, bias=False)
        self.cand_coarse_proj = nn.Linear(self.d_vq, self.d_proj, bias=False)
        self.state_multi_proj = nn.Linear(self.d_vq * 3, self.d_proj, bias=False)
        self.cand_multi_proj = nn.Linear(self.d_vq * 4, self.d_proj, bias=False)
        self.decision_gate = nn.Parameter(torch.tensor(-1.0))

        # 2. Deduction coarse-topic vs. fine-residual SwiGLU adapter (7 * d_vq -> d_proj)
        ded_in_dim = self.d_vq * 7
        self.deduction_norm = RMSNorm(ded_in_dim, eps=config.rms_norm_eps)
        self.deduction_up = nn.Linear(ded_in_dim, self.d_vq * 2, bias=False)
        self.deduction_down = nn.Linear(self.d_vq, self.d_proj, bias=False)
        self.deduction_gate = nn.Parameter(torch.tensor(0.35))

        # 3. Metacognition hierarchical entropy & cross-scale consistency adapter
        meta_in_dim = self.d_vq * 3 + 2 * self.num_levels + 5
        self.meta_norm = RMSNorm(meta_in_dim, eps=config.rms_norm_eps)
        self.meta_up = nn.Linear(meta_in_dim, self.d_vq * 2, bias=False)
        self.meta_down = nn.Linear(self.d_vq, self.d_proj, bias=False)
        self.meta_gate = nn.Parameter(torch.tensor(0.35))

        # 4. Abstraction scale-specific VQ-to-d_model projections (`brief`, `standard`, `detailed`)
        self.abstraction_vq_proj = nn.ModuleDict(
            {
                "brief": nn.Linear(self.d_vq, self.d_model, bias=False),
                "standard": nn.Linear(self.d_vq, self.d_model, bias=False),
                "detailed": nn.Linear(self.d_vq, self.d_model, bias=False),
            }
        )
        self.abstraction_blend_gate = nn.Parameter(torch.tensor(-0.5))

        # 5. Association multiscale item embedding projection
        self.assoc_item_proj = nn.Linear(self.d_vq * 3, self.d_proj, bias=False)
        self.assoc_gate = nn.Parameter(torch.tensor(-1.0))

        std_vq = 0.25 / math.sqrt(max(1, self.d_vq))
        nn.init.normal_(self.vq_token_scorer.weight, std=std_vq)
        nn.init.normal_(self.vq_res_scorer.weight, std=std_vq)
        nn.init.normal_(self.state_coarse_proj.weight, std=std_vq)
        with torch.no_grad():
            self.cand_coarse_proj.weight.copy_(self.state_coarse_proj.weight)
        nn.init.normal_(self.state_multi_proj.weight, std=std_vq / math.sqrt(3.0))
        nn.init.normal_(self.cand_multi_proj.weight, std=std_vq / math.sqrt(4.0))
        nn.init.normal_(self.deduction_up.weight, std=std_vq / math.sqrt(7.0))
        nn.init.normal_(self.deduction_down.weight, std=std_vq)
        nn.init.normal_(self.meta_up.weight, std=std_vq / math.sqrt(4.0))
        nn.init.normal_(self.meta_down.weight, std=std_vq)
        for mod in self.abstraction_vq_proj.values():
            nn.init.normal_(mod.weight, std=std_vq)
        nn.init.normal_(self.assoc_item_proj.weight, std=std_vq / math.sqrt(3.0))

    @property
    def vqvae(self) -> DonorAnchoredMultiscaleVQVAE:
        return self._vqvae_ref[0]

    def _pool_vq_seq(
        self, seq: Tensor, valid: Tensor, *, use_residual_scorer: bool = False
    ) -> Tensor:
        """Salience-weighted + mean pooling over valid positions `[N, T, d_vq] -> [N, d_vq]`."""
        scorer = self.vq_res_scorer if use_residual_scorer else self.vq_token_scorer
        safe_valid = valid.clone()
        has_any = safe_valid.any(dim=-1)
        if not has_any.all():
            safe_valid[~has_any, 0] = True
        raw_scores = scorer(seq.to(scorer.weight.dtype)).squeeze(-1)
        with torch.autocast(device_type=seq.device.type, enabled=False):
            scores = raw_scores.float().masked_fill(~safe_valid, -torch.inf)
            weights = scores.softmax(dim=-1)
            attn_pool = (seq.float() * weights.unsqueeze(-1)).sum(dim=1)
            mask_f = safe_valid.unsqueeze(-1).float()
            mean_pool = (seq.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            return (0.5 * (attn_pool + mean_pool)).to(seq.dtype)

    def extract_features(
        self, token_ids: Tensor, content_mask: Tensor
    ) -> NCMCOEMultiscaleFeatures:
        """Extract deterministic coarse, mid, fine, and fine-residual NCM features for COE heads."""
        safe_mask = content_mask.clone()
        has_any = safe_mask.any(dim=-1)
        if not has_any.all():
            safe_mask[~has_any, 0] = True

        z = self.vqvae.encode_pre_quant(token_ids, valid=safe_mask)
        z_hats_all, code_indices, scale_entropy = self.vqvae.vq.quantize_for_coe(
            z, valid=safe_mask
        )
        n_lvls = len(z_hats_all)
        k_coarse = 0
        k_mid = max(0, (n_lvls - 1) // 2)
        k_fine = n_lvls - 1

        z_coarse_seq = z_hats_all[k_coarse]
        z_mid_seq = z_hats_all[k_mid]
        z_fine_seq = z_hats_all[k_fine]
        z_res_fine_seq = z_fine_seq - z_coarse_seq

        z_coarse_pool = self._pool_vq_seq(z_coarse_seq, safe_mask, use_residual_scorer=False)
        z_mid_pool = self._pool_vq_seq(z_mid_seq, safe_mask, use_residual_scorer=False)
        z_fine_pool = self._pool_vq_seq(z_fine_seq, safe_mask, use_residual_scorer=False)
        z_res_fine_pool = self._pool_vq_seq(z_res_fine_seq, safe_mask, use_residual_scorer=True)

        with torch.autocast(device_type=token_ids.device.type, enabled=False):
            scale_consistency = F.cosine_similarity(
                z_coarse_pool.float(), z_fine_pool.float(), dim=-1, eps=1e-6
            ).unsqueeze(-1)

        return NCMCOEMultiscaleFeatures(
            z_coarse_seq=z_coarse_seq,
            z_mid_seq=z_mid_seq,
            z_fine_seq=z_fine_seq,
            z_res_fine_seq=z_res_fine_seq,
            z_coarse_pool=z_coarse_pool,
            z_mid_pool=z_mid_pool,
            z_fine_pool=z_fine_pool,
            z_res_fine_pool=z_res_fine_pool,
            scale_entropy=scale_entropy,
            scale_consistency=scale_consistency,
            code_indices=code_indices,
        )

    def enhance_decision_and_eval(
        self,
        state_ids: Tensor,
        state_mask: Tensor,
        cand_ids: Tensor,
        cand_mask: Tensor,
        owner_rows: Tensor,
        base_state_proj: Tensor,
        base_cand_proj: Tensor,
        base_scores: Tensor,
        similarity_fn: Any,
    ) -> tuple[Tensor, Tensor, Tensor, NCMCOEMultiscaleFeatures, NCMCOEMultiscaleFeatures]:
        """Fuse coarse + multiscale NCM concept similarity into `Decision` / `Evaluation` / `Induction` / `Abduction`."""
        state_ms = self.extract_features(state_ids, state_mask)
        cand_ms = self.extract_features(cand_ids, cand_mask)

        w_dt = self.state_coarse_proj.weight.dtype
        s_coarse_all = self.state_coarse_proj(state_ms.z_coarse_pool.to(w_dt))
        s_coarse = s_coarse_all.index_select(0, owner_rows)
        c_coarse = self.cand_coarse_proj(cand_ms.z_coarse_pool.to(w_dt))

        s_multi_in = torch.cat(
            [state_ms.z_coarse_pool, state_ms.z_mid_pool, state_ms.z_fine_pool], dim=-1
        ).to(w_dt)
        c_multi_in = torch.cat(
            [
                cand_ms.z_coarse_pool,
                cand_ms.z_mid_pool,
                cand_ms.z_fine_pool,
                cand_ms.z_res_fine_pool,
            ],
            dim=-1,
        ).to(w_dt)
        s_multi = self.state_multi_proj(s_multi_in).index_select(0, owner_rows)
        c_multi = self.cand_multi_proj(c_multi_in)

        sim_coarse = similarity_fn(s_coarse, c_coarse)
        sim_multi = similarity_fn(s_multi, c_multi)
        ncm_sim = 0.45 * sim_coarse + 0.55 * sim_multi

        w_dec = 0.30 * torch.sigmoid(self.decision_gate.float()).to(base_scores.dtype)
        enhanced_scores = (1.0 - w_dec) * base_scores + w_dec * ncm_sim.to(base_scores.dtype)
        enhanced_state_proj = base_state_proj + w_dec.to(base_state_proj.dtype) * s_multi.to(
            base_state_proj.dtype
        )
        enhanced_cand_proj = base_cand_proj + w_dec.to(base_cand_proj.dtype) * c_multi.to(
            base_cand_proj.dtype
        )
        return enhanced_scores, enhanced_state_proj, enhanced_cand_proj, state_ms, cand_ms

    def enhance_deduction_query(
        self,
        base_ded_joint: Tensor,
        z_coarse_s: Tensor,
        z_coarse_c: Tensor,
        z_res_fine_s: Tensor,
        z_res_fine_c: Tensor,
    ) -> Tensor:
        """Inject coarse-concept alignment vs. fine-residual contradiction features into `Deduction`."""
        w_dt = self.deduction_up.weight.dtype
        zc_s = z_coarse_s.to(w_dt)
        zc_c = z_coarse_c.to(w_dt)
        zr_s = z_res_fine_s.to(w_dt)
        zr_c = z_res_fine_c.to(w_dt)

        ded_in = torch.cat(
            [
                zc_s,
                zc_c,
                zc_s * zc_c,
                zr_s,
                zr_c,
                (zr_s - zr_c).abs(),
                zr_s * zr_c,
            ],
            dim=-1,
        )
        gate, val = self.deduction_up(self.deduction_norm(ded_in)).chunk(2, dim=-1)
        delta = self.deduction_down(F.silu(gate) * val)
        scale_g = 0.35 * torch.tanh(self.deduction_gate.float()).to(base_ded_joint.dtype)
        return base_ded_joint + scale_g * delta.to(base_ded_joint.dtype)

    def enhance_metacognition_query(
        self,
        base_meta_joint: Tensor,
        z_coarse_s: Tensor,
        z_coarse_c: Tensor,
        z_res_fine_c: Tensor,
        s_entropy: Tensor,
        c_entropy: Tensor,
        s_consistency: Tensor,
        c_consistency: Tensor,
    ) -> Tensor:
        """Inject per-scale VQ entropy and coarse-to-fine consistency into `Metacognition`."""
        w_dt = self.meta_up.weight.dtype
        with torch.autocast(device_type=base_meta_joint.device.type, enabled=False):
            coarse_sim = F.cosine_similarity(
                z_coarse_s.float(), z_coarse_c.float(), dim=-1, eps=1e-6
            ).unsqueeze(-1)
            fine_res_norm = (
                z_res_fine_c.float().norm(dim=-1, keepdim=True)
                / math.sqrt(max(1.0, float(self.d_vq)))
            ).clamp(0.0, 5.0)
            sim_gap = coarse_sim - 0.5 * (s_consistency.float() + c_consistency.float())

        meta_in = torch.cat(
            [
                z_coarse_s.to(w_dt),
                z_coarse_c.to(w_dt),
                z_res_fine_c.to(w_dt),
                s_entropy.to(w_dt),
                c_entropy.to(w_dt),
                s_consistency.to(w_dt),
                c_consistency.to(w_dt),
                coarse_sim.to(w_dt),
                fine_res_norm.to(w_dt),
                sim_gap.to(w_dt),
            ],
            dim=-1,
        )
        gate, val = self.meta_up(self.meta_norm(meta_in)).chunk(2, dim=-1)
        delta = self.meta_down(F.silu(gate) * val)
        scale_g = 0.35 * torch.tanh(self.meta_gate.float()).to(base_meta_joint.dtype)
        return base_meta_joint + scale_g * delta.to(base_meta_joint.dtype)

    def enhance_abstraction(
        self,
        item_ids: Tensor,
        valid: Tensor,
        base_pooled: Tensor,
        *,
        level: str,
        pooler: Any,
        strip_boundary: bool,
        bos_id: int,
        eos_id: int,
    ) -> Tensor:
        """Route `Abstraction` (`brief`, `standard`, `detailed`) through its matching NCM concept scale."""
        safe_valid = valid.clone()
        has_any = safe_valid.any(dim=-1)
        if not has_any.all():
            safe_valid[~has_any, 0] = True

        ms = self.extract_features(item_ids, safe_valid)
        if level == "brief":
            z_seq, z_pool = ms.z_coarse_seq, ms.z_coarse_pool
        elif level == "standard":
            z_seq, z_pool = ms.z_mid_seq, ms.z_mid_pool
        else:
            z_seq, z_pool = ms.z_fine_seq, ms.z_fine_pool

        h_scale = self.vqvae.decode_hidden_from_z_hat(z_seq, valid=safe_valid)
        pooled_dec = pooler.pool_sequence(
            h_scale,
            item_ids,
            safe_valid,
            strip_boundary=strip_boundary,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        vq_proj_mod = self.abstraction_vq_proj[level]
        vq_vec = vq_proj_mod(z_pool.to(vq_proj_mod.weight.dtype))
        ncm_abs = 0.5 * (pooled_dec.to(base_pooled.dtype) + vq_vec.to(base_pooled.dtype))
        w_abs = 0.35 * torch.sigmoid(self.abstraction_blend_gate.float()).to(base_pooled.dtype)
        return (1.0 - w_abs) * base_pooled + w_abs * ncm_abs

    def enhance_item_embedding(
        self,
        item_ids: Tensor,
        content_mask: Tensor,
        base_proj: Tensor,
    ) -> Tensor:
        """Enrich `Association` item embeddings (`embed_items`) with multiscale NCM concept vectors."""
        ms = self.extract_features(item_ids, content_mask)
        w_dt = self.assoc_item_proj.weight.dtype
        multi_in = torch.cat([ms.z_coarse_pool, ms.z_mid_pool, ms.z_fine_pool], dim=-1).to(w_dt)
        ncm_proj = self.assoc_item_proj(multi_in).to(base_proj.dtype)
        w_assoc = 0.25 * torch.sigmoid(self.assoc_gate.float()).to(base_proj.dtype)
        return (1.0 - w_assoc) * base_proj + w_assoc * ncm_proj


# ---------------------------------------------------------------------------
# 7. Unified `NCMSystem`: End-to-End Training, Generation, COE Verification & RL
# ---------------------------------------------------------------------------


class NCMSystem(nn.Module):
    """Complete Next-Concept Modeling + `MCL4-COE` System.

    Wraps `CognitiveOperatorEnsemble`, `DonorAnchoredMultiscaleVQVAE`, and `NCMGeneratorAdapter`
    into a single interface supporting:
    - Stage-1 VQ-VAE tokenizer training (`forward_tokenizer_step`)
    - Stage-2 single-pass block-diagonal NCM generator training + `COE` multiscale `Abstraction`
      alignment (`forward_training_step`)
    - Exact single-pass hierarchical trajectory log-probabilities $\\log \\pi_\\theta(r_{1:K} \\mid q)$
      for `Ornith-2.5` (`pp = p * 0.95^(n-1)`) GRPO (`compute_hierarchy_log_probs`)
    - Coarse-to-fine parallel generation with mid-hierarchy `COE` verification, `Deduction` veto
      with lateral `RecurrentFeedback`, `Metacognition` early-exit, Classifier-Free Guidance,
      and Latent MBR (`generate`).
    """

    def __init__(
        self,
        coe_or_config: Any,
        config_or_backbone: Any | None = None,
        *,
        coe_model: Any | None = None,
    ) -> None:
        super().__init__()
        if isinstance(coe_or_config, NCMConfig):
            self.config = coe_or_config
            backbone = config_or_backbone if isinstance(config_or_backbone, MCL4CEDBackbone) else getattr(coe_model, "backbone", None)
            coe = coe_model
        else:
            coe = coe_or_config
            self.config = config_or_backbone if isinstance(config_or_backbone, NCMConfig) else NCMConfig()
            backbone = getattr(coe, "backbone", None)

        if backbone is None:
            raise ValueError("NCMSystem requires an MCL4CEDBackbone or CognitiveOperatorEnsemble")

        # Store `coe` and `backbone` in 1-tuples so `coe.ncm_system = system` creates no module cycle
        self._coe_ref = (coe,) if coe is not None else ()
        self._backbone_ref = (backbone,)
        self.vqvae = DonorAnchoredMultiscaleVQVAE(
            self.config,
            backbone=backbone,
        )
        self.generator = NCMGeneratorAdapter(
            self.config,
            self.vqvae,
            backbone,
        )
        coe_proj_dim = (
            int(coe.config.projection_dim)
            if (coe is not None and hasattr(coe, "config") and hasattr(coe.config, "projection_dim"))
            else 32
        )
        self.coe_bridge = NCMCOEHeadBridge(
            self.config,
            self.vqvae,
            d_model=int(backbone.config.d_model),
            d_proj=coe_proj_dim,
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
        for name, param in self.coe_bridge.named_parameters():
            full_key = f"{prefix}coe_bridge.{name}"
            if full_key not in state_dict:
                state_dict[full_key] = param.detach().clone()
        for name, buf in self.coe_bridge.named_buffers():
            full_key = f"{prefix}coe_bridge.{name}"
            if full_key not in state_dict:
                state_dict[full_key] = buf.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def coe(self) -> Any:
        return self._coe_ref[0] if self._coe_ref else None

    @property
    def coe_model(self) -> Any:
        return self._coe_ref[0] if self._coe_ref else None

    @property
    def backbone(self) -> MCL4CEDBackbone:
        return self._backbone_ref[0]

    @property
    def levels(self) -> tuple[int, ...]:
        return self.vqvae.vq.levels

    @property
    def num_levels(self) -> int:
        return len(self.vqvae.vq.levels)

    def forward_tokenizer_step(
        self,
        token_ids: Tensor,
        valid: Tensor | None = None,
        *,
        force_cutoff_level: int | None = None,
    ) -> VQVAETokenizerOutput:
        """Execute Stage-1 Multiscale Residual VQ-VAE step (`DonorAnchoredMultiscaleVQVAE`)."""
        return self.vqvae(token_ids, valid=valid, force_cutoff_level=force_cutoff_level)

    def forward_training_step(
        self,
        prompt_ids: Tensor,
        target_ids: Tensor,
        *,
        prompt_valid: Tensor | None = None,
        target_valid: Tensor | None = None,
        train_vqvae_jointly: bool = True,
        include_coe_abstraction_loss: bool = True,
        corruption_rate: float | None = None,
    ) -> NCMTrainingOutput:
        """Run one end-to-end NCM + `MCL4-COE` training step across all $K$ levels in parallel."""
        if prompt_valid is None:
            prompt_valid = (
                (prompt_ids != self.backbone.config.pad_id)
                if self.backbone.config.pad_id >= 0
                else torch.ones_like(prompt_ids, dtype=torch.bool)
            )
        if target_valid is None:
            target_valid = (
                (target_ids != self.backbone.config.pad_id)
                if self.backbone.config.pad_id >= 0
                else torch.ones_like(target_ids, dtype=torch.bool)
            )
        # Ensure at least one valid token per row
        prompt_valid = prompt_valid.clone()
        prompt_valid[:, 0] = True
        target_valid = target_valid.clone()
        target_valid[:, 0] = True

        # 1. Stage 1: Multiscale Residual VQ-VAE forward (encoding target_ids -> r_{1:K}^*)
        if train_vqvae_jointly:
            vq_out = self.vqvae(target_ids, valid=target_valid)
        else:
            with torch.no_grad():
                vq_out = self.vqvae(target_ids, valid=target_valid)

        code_targets = vq_out.quantizer_out.code_indices
        full_l = int(target_ids.shape[1])
        corr_rate = (
            float(self.config.conditioning_corruption_rate)
            if corruption_rate is None
            else float(corruption_rate)
        )

        # 2. Build teacher-forced corrupted cumulative reconstructions \tilde{z}^{(0..K-1)}
        level_inputs, parent_codes, cum_z_hats = self.vqvae.vq.build_corrupted_ncm_inputs(
            code_targets,
            full_length=full_l,
            corruption_rate=corr_rate if self.training else 0.0,
        )

        # 3. Encode conditioning prompt #U$# once through the lower CEDEncoderLayer stack
        encoded_prompt = self.backbone.encode_state(prompt_ids, prompt_valid, causal=True)

        # 4. Predict all K concept levels in a single block-diagonal forward pass
        logits_per_level, dec_aux, dec_z, _ = self.generator.forward_all_levels_parallel(
            encoded_prompt,
            level_inputs,
            parent_codes,
        )

        # 5. Compute NCM Eq. 10 level-weighted cross-entropy loss across all K levels
        level_lengths = [int(t.shape[1]) for t in code_targets]
        w_k = compute_ncm_level_weights(
            level_lengths, alpha=self.config.loss_alpha, device=target_ids.device
        )
        eff_weighted_tokens = sum(
            float(w_k[k].item()) * float(max(1, tgt_k.shape[1]))
            for k, tgt_k in enumerate(code_targets)
        )
        ncm_ce_loss = torch.zeros((), device=target_ids.device, dtype=torch.float32)
        per_level_ce: list[float] = []
        per_level_acc: list[float] = []

        for k, (logits_k, tgt_k) in enumerate(zip(logits_per_level, code_targets)):
            sum_ce_k = F.cross_entropy(
                logits_k.float().reshape(-1, logits_k.shape[-1]),
                tgt_k.reshape(-1),
                reduction="sum",
            ) / float(max(1, target_ids.shape[0]))
            ncm_ce_loss = ncm_ce_loss + w_k[k] * sum_ce_k
            mean_ce_k = float(sum_ce_k.detach().item()) / float(max(1, tgt_k.shape[1]))
            per_level_ce.append(round(mean_ce_k, 4))
            with torch.no_grad():
                acc_k = float((logits_k.argmax(dim=-1) == tgt_k).float().mean().item())
                per_level_acc.append(round(acc_k, 4))
        ncm_ce_loss = ncm_ce_loss / max(1.0, eff_weighted_tokens)

        # 6. Multiscale COE `Abstraction` Alignment (`brief` <-> coarse, `standard` <-> mid, `detailed` <-> fine)
        abs_loss = torch.zeros((), device=target_ids.device, dtype=torch.float32)
        coe = self.coe
        if (
            include_coe_abstraction_loss
            and coe is not None
            and hasattr(coe, "abstraction_gates")
            and len(cum_z_hats) >= 2
        ):
            k_brief = 0
            k_std = len(cum_z_hats) // 2
            k_det = len(cum_z_hats) - 1
            h_brief_seq = self.vqvae.decode_hidden_from_z_hat(cum_z_hats[k_brief], valid=target_valid)
            h_std_seq = self.vqvae.decode_hidden_from_z_hat(cum_z_hats[k_std], valid=target_valid)
            h_det_seq = self.vqvae.decode_hidden_from_z_hat(cum_z_hats[k_det], valid=target_valid)

            def _abs_vec(h_seq: Tensor, z_seq: Tensor, lvl_name: str) -> Tensor:
                pooled = coe.pooler.pool_sequence(
                    h_seq, target_ids, target_valid, strip_boundary=False
                )
                if self.config.enable_coe_head_bridge and hasattr(self, "coe_bridge"):
                    z_pool = self.coe_bridge._pool_vq_seq(z_seq, target_valid)
                    vq_proj = self.coe_bridge.abstraction_vq_proj[lvl_name]
                    vq_vec = vq_proj(z_pool.to(vq_proj.weight.dtype)).to(pooled.dtype)
                    w_abs = (
                        0.35
                        * torch.sigmoid(self.coe_bridge.abstraction_blend_gate.float()).to(
                            pooled.dtype
                        )
                    )
                    pooled = (1.0 - w_abs) * pooled + w_abs * vq_vec
                gate = coe.abstraction_gates[lvl_name].sigmoid()
                return coe.abstraction_norm(pooled + gate * coe.abstraction_proj(pooled))

            v_brief = _abs_vec(h_brief_seq, cum_z_hats[k_brief], "brief")
            v_std = _abs_vec(h_std_seq, cum_z_hats[k_std], "standard")
            v_det = _abs_vec(h_det_seq, cum_z_hats[k_det], "detailed")
            sim_bs = F.cosine_similarity(v_brief.float(), v_det.detach().float(), dim=-1)
            sim_sd = F.cosine_similarity(v_std.float(), v_det.detach().float(), dim=-1)
            abs_loss = 0.5 * ((1.0 - sim_bs).mean() + (1.0 - sim_sd).mean())

        aux_loss = 0.5 * (encoded_prompt.aux_loss + dec_aux)
        router_z_loss = 0.5 * (encoded_prompt.router_z_loss + dec_z)

        total_loss = (
            ncm_ce_loss
            + (vq_out.total_loss if train_vqvae_jointly else 0.0)
            + 0.10 * abs_loss
            + self.backbone.config.aux_loss_coef * aux_loss
            + self.backbone.config.router_z_loss_coef * router_z_loss
        )

        return NCMTrainingOutput(
            total_loss=total_loss,
            ncm_ce_loss=ncm_ce_loss.detach(),
            vq_recon_loss=vq_out.recon_loss.detach(),
            vq_commit_loss=vq_out.vq_loss.detach(),
            abstraction_align_loss=abs_loss.detach(),
            aux_loss=aux_loss.detach(),
            router_z_loss=router_z_loss.detach(),
            per_level_ce=per_level_ce,
            per_level_acc=per_level_acc,
            codebook_utilization=vq_out.quantizer_out.global_utilization,
            recon_accuracy=vq_out.recon_accuracy,
        )

    def compute_hierarchy_log_probs(
        self,
        prompt_or_encoded: EncodedState | Tensor,
        codes_or_target_ids: Sequence[Tensor] | Tensor,
        *,
        owner_rows: Tensor | None = None,
        full_length: int | None = None,
        normalize_by_tokens: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        """Compute exact normalized log-likelihood $\\log \\pi_\\theta(r_{1:K} \\mid q)$ in **1 forward pass** (Eq. 13).

        Accepts either:
        - `(encoded_prompt: EncodedState, code_indices: Sequence[Tensor])`, or
        - `(prompt_ids: Tensor, target_ids: Tensor)`.

        Returns `(trajectory_log_prob [R], per_level_log_probs [K x [R]])` for `Ornith-2.5` GRPO.
        """
        if isinstance(prompt_or_encoded, EncodedState):
            encoded_prompt = prompt_or_encoded
        else:
            p_ids = prompt_or_encoded
            p_valid = (
                (p_ids != self.backbone.config.pad_id)
                if self.backbone.config.pad_id >= 0
                else torch.ones_like(p_ids, dtype=torch.bool)
            )
            p_valid = p_valid.clone()
            p_valid[:, 0] = True
            encoded_prompt = self.backbone.encode_state(p_ids, p_valid, causal=True)

        if isinstance(codes_or_target_ids, Tensor):
            t_ids = codes_or_target_ids
            t_valid = (
                (t_ids != self.backbone.config.pad_id)
                if self.backbone.config.pad_id >= 0
                else torch.ones_like(t_ids, dtype=torch.bool)
            )
            t_valid = t_valid.clone()
            t_valid[:, 0] = True
            code_indices: Sequence[Tensor] = self.vqvae.encode(t_ids, valid=t_valid)
            full_l = int(t_ids.shape[1])
        else:
            code_indices = codes_or_target_ids
            full_l = int(full_length or self.config.target_length)

        level_inputs, parent_codes, _ = self.vqvae.vq.build_corrupted_ncm_inputs(
            code_indices,
            full_length=full_l,
            corruption_rate=0.0,
        )
        logits_per_level, _, _, _ = self.generator.forward_all_levels_parallel(
            encoded_prompt,
            level_inputs,
            parent_codes,
            owner_rows=owner_rows,
        )
        level_lengths = [int(t.shape[1]) for t in code_indices]
        w_k = compute_ncm_level_weights(
            level_lengths, alpha=self.config.loss_alpha, device=code_indices[0].device
        )

        per_level_lps: list[Tensor] = []
        total_lp = torch.zeros(
            code_indices[0].shape[0], device=code_indices[0].device, dtype=torch.float32
        )
        for k, (logits_k, tgt_k) in enumerate(zip(logits_per_level, code_indices)):
            log_p_k = F.log_softmax(logits_k.float(), dim=-1)
            chosen_lp = log_p_k.gather(-1, tgt_k.unsqueeze(-1)).squeeze(-1)  # [R, ell_k]
            level_mean_lp = chosen_lp.mean(dim=-1)
            per_level_lps.append(level_mean_lp)
            if normalize_by_tokens:
                total_lp = total_lp + w_k[k] * chosen_lp.sum(dim=-1)
            else:
                total_lp = total_lp + w_k[k] * chosen_lp.sum(dim=-1)
        return total_lp, per_level_lps

    @staticmethod
    def _sample_codes_from_logits(
        logits: Tensor,
        *,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Tensor:
        if temperature <= 1e-6:
            return logits.argmax(dim=-1)
        scaled = logits.float() / float(temperature)
        if top_k > 0 and top_k < scaled.shape[-1]:
            kth = torch.topk(scaled, top_k, dim=-1).values[..., -1, None]
            scaled = scaled.masked_fill(scaled < kth, -torch.inf)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            remove = (probs.cumsum(dim=-1) - probs) > float(top_p)
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
            scaled = torch.full_like(scaled, -torch.inf).scatter(-1, sorted_idx, sorted_logits)
        probs_final = F.softmax(scaled.reshape(-1, scaled.shape[-1]), dim=-1)
        sampled = torch.multinomial(probs_final, num_samples=1).view(logits.shape[:-1])
        return sampled

    @torch.no_grad()
    def _evaluate_mid_hierarchy_coe(
        self,
        encoded_prompt: EncodedState,
        z_hat_candidates: Tensor,
        *,
        owner_rows: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run in-network `COE` (`Decision`, `Evaluation`, `Deduction`, `Metacognition`) on $\\hat{z}^{(k)}$."""
        from .operators import pearson_similarity

        r_batch, full_l, _ = z_hat_candidates.shape
        dev = z_hat_candidates.device
        cand_valid = torch.ones((r_batch, full_l), dtype=torch.bool, device=dev)
        cand_logits, cand_hidden = self.vqvae.decode_from_z_hat(z_hat_candidates, valid=cand_valid)
        cand_ids = cand_logits.argmax(dim=-1)

        if owner_rows is None:
            owner_rows = torch.zeros(r_batch, dtype=torch.long, device=dev)

        coe = self.coe
        if coe is None:
            zeros = torch.zeros(r_batch, device=dev, dtype=torch.float32)
            ones = torch.ones(r_batch, device=dev, dtype=torch.float32)
            return {
                "composite_score": zeros,
                "decision_sim": zeros,
                "utility": ones * 0.5,
                "entailed_prob": ones * 0.5,
                "contradicted_prob": zeros,
                "correctness_prob": ones * 0.5,
                "sufficiency_prob": ones * 0.5,
            }

        state_ids = torch.zeros(
            encoded_prompt.hidden.shape[:2], dtype=torch.long, device=dev
        )
        paired_state_repr, cand_repr = coe.pooler.matched_pair_representations(
            encoded_prompt.hidden,
            state_ids,
            encoded_prompt.valid,
            cand_hidden,
            cand_ids,
            cand_valid,
            owner_rows,
            cand_anchor=None,
            strip_boundary=False,
            bos_id=coe.config.bos_id,
            eos_id=coe.config.eos_id,
        )
        s_proj = coe.state_projection(paired_state_repr)
        c_proj = coe.output_projection(cand_repr)
        decision_sim = coe._similarity(s_proj, c_proj)
        scale = coe._scale()

        eval_logits = decision_sim.float() * coe.assoc_alpha.float() + coe.assoc_beta.float()
        utility = eval_logits.sigmoid()

        joint_pair = torch.cat([paired_state_repr, cand_repr], dim=-1)
        ded_joint = coe.deduction_query_proj(joint_pair)
        ded_sims = pearson_similarity(
            ded_joint.unsqueeze(1), coe.deduction_prototypes.unsqueeze(0), eps=coe.config.eps
        )
        ded_probs = (ded_sims.float() * scale).softmax(dim=-1)

        meta_joint = coe.meta_query_proj(joint_pair)
        meta_sims = pearson_similarity(
            meta_joint.unsqueeze(1), coe.meta_prototypes.unsqueeze(0), eps=coe.config.eps
        )
        meta_probs = (meta_sims.float() * scale + coe.meta_bias.float().unsqueeze(0)).sigmoid()

        entailed_p = ded_probs[:, 0]
        contradicted_p = ded_probs[:, 1]
        correctness_p = meta_probs[:, 0]
        sufficiency_p = meta_probs[:, 1]
        composite_score = (
            0.35 * decision_sim
            + 0.25 * utility
            + 0.25 * (entailed_p - contradicted_p)
            + 0.15 * correctness_p
        )
        return {
            "composite_score": composite_score,
            "decision_sim": decision_sim,
            "utility": utility,
            "entailed_prob": entailed_p,
            "contradicted_prob": contradicted_p,
            "correctness_prob": correctness_p,
            "sufficiency_prob": sufficiency_p,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        *,
        tokenizer: Any | None = None,
        target_length: int | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        cfg_scale: float = 0.0,
        enable_mid_hierarchy_coe: bool | None = None,
        enable_early_exit: bool = False,
        coarse_branch_candidates: int | None = None,
        decode_all_levels: bool = False,
    ) -> NCMGenerationOutput:
        """Generate text from coarse concepts to fine details in $K \\ll L$ parallel steps (Algorithm 2 + COE)."""
        dev = prompt_ids.device
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        b = prompt_ids.shape[0]
        full_l = int(target_length or self.config.target_length)
        use_coe = (
            self.config.enable_mid_hierarchy_coe
            if enable_mid_hierarchy_coe is None
            else bool(enable_mid_hierarchy_coe)
        )
        n_branch = max(
            1,
            int(
                self.config.coarse_branch_candidates
                if coarse_branch_candidates is None
                else coarse_branch_candidates
            ),
        )

        prompt_valid = (
            (prompt_ids != self.backbone.config.pad_id)
            if self.backbone.config.pad_id >= 0
            else torch.ones_like(prompt_ids, dtype=torch.bool)
        )
        prompt_valid = prompt_valid.clone()
        prompt_valid[:, 0] = True
        encoded_prompt = self.backbone.encode_state(prompt_ids, prompt_valid, causal=True)
        global_kv_cache = [
            layer.project_global_kv(encoded_prompt.hidden, n_prefix=0)
            for layer in self.backbone.decoder_layers
        ]

        # Optional Classifier-Free Guidance (CFG) null prompt state
        null_kv_cache: list[tuple[Tensor, Tensor]] | None = None
        null_valid: Tensor | None = None
        if cfg_scale > 0.0:
            null_ids = torch.full_like(
                prompt_ids,
                max(0, min(self.backbone.config.vocab_size - 1, int(self.backbone.config.eos_id))),
            )
            null_valid = torch.ones_like(prompt_valid)
            null_enc = self.backbone.encode_state(null_ids, null_valid, causal=True)
            null_kv_cache = [
                layer.project_global_kv(null_enc.hidden, n_prefix=0)
                for layer in self.backbone.decoder_layers
            ]

        vq_dtype = self.vqvae.vq.embeddings[0].weight.dtype
        z_hat = torch.zeros((b, full_l, self.config.vq_embed_dim), device=dev, dtype=vq_dtype)
        ell_0 = min(full_l, int(self.levels[0])) if self.num_levels > 1 else full_l
        cur_vq_in = torch.zeros((b, ell_0, self.config.vq_embed_dim), device=dev, dtype=vq_dtype)
        cur_parent_codes = torch.zeros((b, ell_0), device=dev, dtype=torch.long)

        code_indices: list[Tensor] = []
        per_level_active_top_k: list[int] = []
        per_level_meta: list[dict[str, float]] = []
        per_level_ded: list[dict[str, float]] = []
        level_decoded_texts: list[str] = []
        lateral_refinements = 0
        halted_early = False
        levels_executed = 0

        for k in range(self.num_levels):
            levels_executed = k + 1
            eff_len = cur_vq_in.shape[1]
            active_k = self.generator.get_scale_top_k(k)
            per_level_active_top_k.append(active_k)

            h_in = self.generator._embed_level_input(cur_vq_in, k)
            valid_k = torch.ones((b, eff_len), dtype=torch.bool, device=dev)
            rope_pos = torch.arange(eff_len, device=dev, dtype=torch.float32)

            hidden_k, _, _, _ = self.generator._run_decoder_stack(
                h_in,
                cur_parent_codes,
                valid_k,
                global_kv_cache,
                encoded_prompt.valid,
                prompt_hidden=encoded_prompt.hidden,
                local_attn_mask=None,
                rope_positions=rope_pos,
                level_idx_for_density=k,
                recurrence_steps=1,
            )
            logits_k = self.generator.level_heads[k](hidden_k)

            # Classifier-Free Guidance (VAR/NCM scale-ramped weight t_k = cfg_scale * k / (K - 1))
            if cfg_scale > 0.0 and null_kv_cache is not None and null_valid is not None:
                null_h, _, _, _ = self.generator._run_decoder_stack(
                    h_in,
                    cur_parent_codes,
                    valid_k,
                    null_kv_cache,
                    null_valid,
                    prompt_hidden=null_enc.hidden,
                    local_attn_mask=None,
                    rope_positions=rope_pos,
                    level_idx_for_density=k,
                    recurrence_steps=1,
                )
                null_logits_k = self.generator.level_heads[k](null_h)
                t_cfg = float(cfg_scale) * (float(k) / float(max(1, self.num_levels - 1)))
                logits_k = (1.0 + t_cfg) * logits_k - t_cfg * null_logits_k

            # Coarse-Scale Latent Candidate Branching + In-Network COE Selection (b == 1)
            if (
                use_coe
                and b == 1
                and n_branch > 1
                and k < self.config.coarse_level_cutoff
            ):
                cand_codes_list: list[Tensor] = [
                    self._sample_codes_from_logits(logits_k, temperature=0.0)
                ]
                for g_i in range(1, n_branch):
                    cand_codes_list.append(
                        self._sample_codes_from_logits(
                            logits_k,
                            temperature=max(0.35, float(temperature) or 0.55),
                            top_k=max(4, top_k),
                            top_p=top_p,
                        )
                    )
                cand_codes_batch = torch.cat(cand_codes_list, dim=0)  # [G, ell_k]
                z_hat_rep = z_hat.expand(n_branch, -1, -1).clone()
                z_hat_cands, _, _ = self.vqvae.vq.step_next_level_reconstruction(
                    k, z_hat_rep, cand_codes_batch, full_length=full_l
                )
                coe_scores = self._evaluate_mid_hierarchy_coe(
                    encoded_prompt,
                    z_hat_cands,
                    owner_rows=torch.zeros(n_branch, dtype=torch.long, device=dev),
                )
                best_g = int(coe_scores["composite_score"].argmax().item())
                sampled_k = cand_codes_batch[best_g : best_g + 1]
            else:
                sampled_k = self._sample_codes_from_logits(
                    logits_k, temperature=temperature, top_k=top_k, top_p=top_p
                )

            z_hat_cand, next_in, next_parent = self.vqvae.vq.step_next_level_reconstruction(
                k, z_hat, sampled_k, full_length=full_l
            )

            # Evaluate mid-hierarchy COE signals on z_hat_cand
            if use_coe:
                owner = torch.arange(b, dtype=torch.long, device=dev)
                coe_eval = self._evaluate_mid_hierarchy_coe(
                    encoded_prompt, z_hat_cand, owner_rows=owner
                )
                contradicted_p = float(coe_eval["contradicted_prob"].mean().item())
                correctness_p = float(coe_eval["correctness_prob"].mean().item())
                sufficiency_p = float(coe_eval["sufficiency_prob"].mean().item())
                entailed_p = float(coe_eval["entailed_prob"].mean().item())

                # Mid-level Deduction Veto -> Lateral RecurrentFeedback refinement at scale k
                if contradicted_p > self.config.deduction_veto_threshold:
                    lateral_refinements += 1
                    refined_h, _, _, _ = self.generator._run_decoder_stack(
                        h_in,
                        cur_parent_codes,
                        valid_k,
                        global_kv_cache,
                        encoded_prompt.valid,
                        prompt_hidden=encoded_prompt.hidden,
                        local_attn_mask=None,
                        rope_positions=rope_pos,
                        level_idx_for_density=k,
                        recurrence_steps=2,
                    )
                    logits_ref = self.generator.level_heads[k](refined_h)
                    sampled_k = self._sample_codes_from_logits(
                        logits_ref, temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    z_hat_cand, next_in, next_parent = (
                        self.vqvae.vq.step_next_level_reconstruction(
                            k, z_hat, sampled_k, full_length=full_l
                        )
                    )
                    coe_eval = self._evaluate_mid_hierarchy_coe(
                        encoded_prompt, z_hat_cand, owner_rows=owner
                    )
                    contradicted_p = float(coe_eval["contradicted_prob"].mean().item())
                    correctness_p = float(coe_eval["correctness_prob"].mean().item())
                    sufficiency_p = float(coe_eval["sufficiency_prob"].mean().item())
                    entailed_p = float(coe_eval["entailed_prob"].mean().item())

                per_level_meta.append(
                    {
                        "level": k + 1,
                        "ell_k": eff_len,
                        "correctness": round(correctness_p, 4),
                        "evidence_sufficiency": round(sufficiency_p, 4),
                    }
                )
                per_level_ded.append(
                    {
                        "level": k + 1,
                        "ell_k": eff_len,
                        "entailed": round(entailed_p, 4),
                        "contradicted": round(contradicted_p, 4),
                    }
                )

            z_hat = z_hat_cand
            code_indices.append(sampled_k)

            if decode_all_levels and tokenizer is not None:
                lvl_logits, _ = self.vqvae.decode_from_z_hat(z_hat)
                lvl_toks = lvl_logits[0].argmax(dim=-1).tolist()
                level_decoded_texts.append(tokenizer.decode(lvl_toks))

            # Adaptive Metacognition Early-Exit Halting
            if (
                enable_early_exit
                and use_coe
                and (k + 1) >= self.config.min_early_exit_level
                and k < self.num_levels - 1
                and per_level_meta
                and per_level_meta[-1]["correctness"] >= self.config.metacognition_halt_threshold
                and per_level_meta[-1]["evidence_sufficiency"]
                >= self.config.metacognition_halt_threshold
            ):
                halted_early = True
                break

            if next_in is not None and next_parent is not None:
                cur_vq_in = next_in
                cur_parent_codes = next_parent

        # Decode final cumulative reconstruction z_hat^(K) into tokens in 1 parallel pass
        final_logits, _ = self.vqvae.decode_from_z_hat(z_hat)
        out_token_ids = final_logits.argmax(dim=-1)
        out_text = (
            tokenizer.decode(out_token_ids[0].tolist())
            if tokenizer is not None
            else ""
        )

        return NCMGenerationOutput(
            token_ids=out_token_ids,
            text=out_text,
            code_indices=code_indices,
            final_z_hat=z_hat,
            levels_executed=levels_executed,
            total_levels=self.num_levels,
            halted_early=halted_early,
            lateral_refinements=lateral_refinements,
            per_level_active_top_k=per_level_active_top_k,
            per_level_metacognition=per_level_meta,
            per_level_deduction=per_level_ded,
            level_decoded_texts=level_decoded_texts,
        )



def load_pretrained_mcl4_coe_ncm(
    repo_id_or_path: str,
    *,
    device: str | torch.device = "auto",
    dtype: str | torch.dtype = "bfloat16",
    token: str | None = None,
) -> CognitiveOperatorEnsemble:
    """Load an `MCL4-COE + NCM` model from a Hugging Face repo ID or local directory (`model.safetensors` + `config.json`)."""
    from dataclasses import fields
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    from .backbone import MCL4CEDBackbone, MCL4CEDConfig
    from .operators import COEConfig, CognitiveOperatorEnsemble

    if str(device) == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map.get(str(dtype).lower(), torch.bfloat16) if isinstance(dtype, str) else dtype

    local_dir = Path(repo_id_or_path)
    if local_dir.is_dir() and (local_dir / "config.json").is_file():
        cfg_path = local_dir / "config.json"
        st_path = local_dir / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        cfg_path = Path(hf_hub_download(repo_id=repo_id_or_path, filename="config.json", token=token))
        st_path = Path(hf_hub_download(repo_id=repo_id_or_path, filename="model.safetensors", token=token))

    meta = json.loads(cfg_path.read_text(encoding="utf-8"))
    valid_b = {f.name for f in fields(MCL4CEDConfig)}
    valid_c = {f.name for f in fields(COEConfig)}
    valid_n = {f.name for f in fields(NCMConfig)}

    b_dict = {k: v for k, v in (meta.get("backbone_config") or {}).items() if k in valid_b and v is not None}
    c_dict = {k: v for k, v in (meta.get("coe_config") or {}).items() if k in valid_c and v is not None}
    n_dict = {k: v for k, v in (meta.get("ncm_config") or {}).items() if k in valid_n and v is not None}

    if "dense_expert_indices" in b_dict and isinstance(b_dict["dense_expert_indices"], list):
        b_dict["dense_expert_indices"] = tuple(b_dict["dense_expert_indices"])
    if "manual_levels" in n_dict and isinstance(n_dict["manual_levels"], list):
        n_dict["manual_levels"] = tuple(n_dict["manual_levels"])

    b_cfg = MCL4CEDConfig(**b_dict)
    c_cfg = COEConfig(**c_dict)
    ncm_cfg = NCMConfig(**n_dict) if n_dict else None

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        model = CognitiveOperatorEnsemble(MCL4CEDBackbone(b_cfg), c_cfg)
        if ncm_cfg is not None:
            model.create_ncm_system(ncm_cfg)
    finally:
        torch.set_default_dtype(prev_dtype)

    state_dict = load_file(str(st_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)

    mp_list = [tuple(int(x) for x in pair) for pair in meta.get("mitotic_pairs", []) if len(pair) == 2]
    rel_set = {tuple(int(x) for x in pair) for pair in meta.get("released_mitotic_pairs", []) if len(pair) == 2}
    active_pairs = [p for p in mp_list if p not in rel_set]
    for blk in list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers):
        blk.moe.mitotic_pairs = list(active_pairs)

    model.to(device=target_device, dtype=target_dtype)
    for mod in model.modules():
        if isinstance(mod, torch.nn.LayerNorm):
            mod.to(device=target_device, dtype=torch.float32)
    model.eval()
    return model
