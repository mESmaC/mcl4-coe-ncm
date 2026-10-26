"""Native Blackwell (sm_120) NVFP4 training & rollout kernels with 2:4 Structured Sparsity.

Adapted from oracle/src/oracle/nvfp4.py for MCL4-COE on the RTX 5070:
- Master weights and optimizer states remain FP32 (`torch.optim.AdamW` or `CPUOffloadAdamW`).
- Matrix operands use packed 4-bit E2M1 (`torch.float4_e2m1fn_x2`), swizzled FP8 E4M3
  block scales (`SWIZZLE_32_4_4`) per 16 values, and a global FP32 decoding scale.
- Causal Row-Wise Activation Scaling (`causal_activation_scale=True`):
  Activations use `global_scale = 1.0` with strictly per-row `1x16` `FP8 E4M3` block scales
  (dynamic range `[-2688, 2688]` per row) so future tokens never leak into earlier tokens'
  quantization scales across causal sequences.
- 2:4 Transposable Structured Weight Sparsity (`2to4_transposable`) + SR-STE:
  Enforces exact 50% 2:4 structured sparsity across every 4x4 weight tile (2 non-zeros per
  4 elements in both `W` for `forward` and `W.T` for `dgrad` via the 90 doubly-2:4 patterns),
  paired with Sparse-Refined Straight-Through Estimator (`sr_ste_decay`) in `wgrad`.
- Backward quantizes gradients with stochastic rounding and applies a shared fixed-sign
  orthonormal 16-point Hadamard transform (`_hadamard16`) to both reduction operands of `wgrad`.
"""

from __future__ import annotations

import random
from collections import Counter
from functools import lru_cache

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_GEMM_COUNTS: Counter = Counter(forward=0, dgrad=0, wgrad=0, other=0, sparse_2to4=0)

SPARSITY_MODES = ("none", "2to4_row", "2to4_transposable")


def get_gemm_counts() -> dict[str, int]:
    """Return successful native NVFP4 GEMM calls, grouped by training phase."""
    return dict(_GEMM_COUNTS)


_NVFP4_OPT_STEP: int = 0


def bump_nvfp4_optimizer_step() -> None:
    global _NVFP4_OPT_STEP
    _NVFP4_OPT_STEP += 1


def reset_gemm_counts() -> None:
    """Zero all NVFP4 GEMM phase counters."""
    for phase in _GEMM_COUNTS:
        _GEMM_COUNTS[phase] = 0


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@lru_cache(maxsize=8)
def _transposable_2to4_catalog(device_str: str) -> Tensor:
    """Return the `[90, 16]` catalog of 4x4 binary masks that are 2:4 in both rows and columns."""
    row_patterns = [
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ]
    valid_tiles: list[list[float]] = []
    for r0 in row_patterns:
        for r1 in row_patterns:
            for r2 in row_patterns:
                for r3 in row_patterns:
                    if all(r0[c] + r1[c] + r2[c] + r3[c] == 2 for c in range(4)):
                        valid_tiles.append([float(x) for row in (r0, r1, r2, r3) for x in row])
    return torch.tensor(valid_tiles, dtype=torch.float32, device=torch.device(device_str))


@lru_cache(maxsize=8)
def _transposable_2to4_triton_tables(device_str: str) -> tuple[Tensor, Tensor]:
    """Return `(cat_16x128_bf16, cat_90x16_bool)` cached on `device_str` for Triton TensorCore 2:4 masking."""
    cat_90x16 = _transposable_2to4_catalog(device_str)
    cat_16x128 = torch.full(
        (16, 128), -100.0, dtype=torch.bfloat16, device=torch.device(device_str)
    )
    cat_16x128[:, :90] = cat_90x16.t().to(torch.bfloat16)
    return cat_16x128.contiguous(), cat_90x16.bool().contiguous()


@lru_cache(maxsize=8)
def _unit_scale_f32(device_str: str) -> Tensor:
    return torch.ones(1, dtype=torch.float32, device=torch.device(device_str))


def _compute_global_scale_core(values: Tensor, one_f32: Tensor) -> Tensor:
    amax = values.abs().amax().float()
    return (
        torch.where(amax > 0, amax * (1.0 / 2688.0), one_f32)
        .clamp_min(1.17549435e-38)
        .reshape(1)
    )


try:
    _compute_global_scale_compiled = torch.compile(_compute_global_scale_core, dynamic=True)
except Exception:
    _compute_global_scale_compiled = _compute_global_scale_core


@torch.no_grad()
def compute_2to4_mask(weight: Tensor, mode: str = "2to4_transposable") -> Tensor:
    """Compute a 50%-sparse 2:4 boolean mask for a 2D weight matrix `[out_features, in_features]`.

    - `'2to4_row'`: keeps top-2 magnitudes out of every 4 contiguous elements along `in_features`.
    - `'2to4_transposable'`: selects the maximum-L1 doubly-2:4 pattern on every 4x4 tile so
       BOTH `W` (`forward`) and `W.T` (`dgrad`) satisfy exact 2:4 structured sparsity.
    """
    if mode not in SPARSITY_MODES:
        raise ValueError(f"sparsity mode must be one of {SPARSITY_MODES}, got {mode!r}")
    if mode == "none":
        return torch.ones_like(weight, dtype=torch.bool)
    if weight.ndim != 2:
        raise ValueError("compute_2to4_mask expects a 2D weight matrix")

    rows, cols = weight.shape
    if mode == "2to4_row" or rows % 4 != 0 or cols % 4 != 0:
        pad_c = _round_up(cols, 4) - cols
        w_abs = F.pad(weight.detach().abs().float(), (0, pad_c)) if pad_c else weight.detach().abs().float()
        grouped = w_abs.view(rows, -1, 4)
        top2 = grouped.topk(2, dim=-1).indices
        mask = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, top2, True).view(rows, -1)
        return mask[:, :cols]

    if (
        _HAS_TRITON_DOT_SCALED
        and weight.device.type == "cuda"
        and rows >= 64
        and cols >= 64
        and (cols // 4) % 4 == 0
    ):
        cat_16x128, cat_bool = _transposable_2to4_triton_tables(str(weight.device))
        w_c = weight.detach().contiguous()
        mask_out = torch.empty((rows, cols), dtype=torch.bool, device=weight.device)
        n_tile4 = rows // 4
        k_tile4 = cols // 4
        grid_2to4 = (triton.cdiv(n_tile4, 32), triton.cdiv(k_tile4, 4))
        _triton_2to4_direct_2d_kernel[grid_2to4](
            w_c,
            cat_16x128,
            cat_bool,
            w_c,
            mask_out,
            w_c,
            0.0,
            n_tile4,
            K_TILE4=k_tile4,
            K=cols,
            BLOCK_R=128,
            WRITE_MASKED_W=False,
            WRITE_SR_STE=False,
            num_warps=4,
        )
        return mask_out

    catalog = _transposable_2to4_catalog(str(weight.device))
    tiles = (
        weight.detach()
        .abs()
        .float()
        .view(rows // 4, 4, cols // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 16)
    )
    best_idx = torch.matmul(tiles, catalog.t()).argmax(dim=-1)
    chosen = catalog.index_select(0, best_idx).bool()
    return (
        chosen.view(rows // 4, cols // 4, 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(rows, cols)
        .contiguous()
    )


def has_native_nvfp4(device: torch.device | str = "cuda") -> bool:
    """Return True if `device` and the active PyTorch build support native Blackwell sm_100+ NVFP4."""
    if not torch.cuda.is_available():
        return False
    dev = torch.device(device)
    if dev.type != "cuda":
        return False
    if not (
        hasattr(torch, "float4_e2m1fn_x2")
        and hasattr(F, "scaled_mm")
        and hasattr(F, "ScalingType")
        and hasattr(F, "SwizzleType")
    ):
        return False
    major, _ = torch.cuda.get_device_capability(dev)
    return major >= 10


def _swizzle_scales(scales: Tensor) -> Tensor:
    """Pack block scales into cuBLASLt's 128-row by 4-scale tiles with inner 32x4x4 layout."""
    rows, columns = scales.shape
    padded_rows, padded_columns = _round_up(rows, 128), _round_up(columns, 4)
    s_u8 = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    if padded_rows != rows or padded_columns != columns:
        s_u8 = F.pad(s_u8, (0, padded_columns - columns, 0, padded_rows - rows))
    return (
        s_u8.view(padded_rows // 128, 4, 32, padded_columns // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(padded_rows, padded_columns)
        .view(torch.float8_e4m3fn)
    )


def _unswizzle_scales(swizzled: Tensor, rows: int, columns: int) -> Tensor:
    """Invert `SWIZZLE_32_4_4` block scales back to `[rows, columns]` FP32 scales."""
    padded_rows, padded_columns = swizzled.shape
    return (
        swizzled.float()
        .view(padded_rows // 128, padded_columns // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(padded_rows, padded_columns)[:rows, :columns]
    )


@lru_cache(maxsize=8)
def _e2m1_levels(device_str: str) -> Tensor:
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=torch.device(device_str),
    )


@lru_cache(maxsize=8)
def _e2m1_midpoints(device_str: str) -> Tensor:
    return torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=torch.device(device_str),
    )


@lru_cache(maxsize=8)
def _hadamard16_signs(device_str: str) -> Tensor:
    return torch.tensor(
        [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
        dtype=torch.float32,
        device=torch.device(device_str),
    )


def _e2m1_codes(values: Tensor, stochastic: bool) -> Tensor:
    """Round normalized values to E2M1, returning one unsigned nibble per scalar."""
    m = values.abs().clamp(max=6.0)
    u = torch.where(m <= 2.0, m * 2.0, torch.where(m <= 4.0, m + 2.0, m * 0.5 + 4.0))
    if stochastic:
        code = (u + torch.rand_like(u)).to(torch.uint8).clamp(max=7)
    else:
        code = torch.round(u).to(torch.uint8)
    return code | (torch.signbit(values).to(torch.uint8) << 3)


def _quantize_nvfp4_core(
    values: Tensor,
    two_dimensional: bool,
    stochastic: bool,
    causal_row_scale: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    rows, columns = values.shape
    if causal_row_scale and not two_dimensional:
        global_scale = values.new_ones(1)
    else:
        amax = values.abs().amax()
        global_scale = torch.where(amax > 0, amax / 2688.0, torch.ones_like(amax))
        global_scale = global_scale.clamp_min(1.17549435e-38).reshape(1)
    if two_dimensional:
        block_amax = values.abs().view(rows // 16, 16, columns // 16, 16).amax(dim=(1, 3))
        block_amax = block_amax.repeat_interleave(16, dim=0)
    else:
        block_amax = values.abs().view(rows, columns // 16, 16).amax(dim=-1)
    block_scales = (block_amax / (6.0 * global_scale)).clamp(max=448.0).to(torch.float8_e4m3fn)
    decoding = block_scales.float() * global_scale
    inverse = torch.where(
        decoding > 0,
        decoding.reciprocal().clamp(max=3.402823466e38),
        torch.zeros_like(decoding),
    )
    normalized = (values.view(rows, columns // 16, 16) * inverse.unsqueeze(-1)).reshape(rows, columns)
    codes = _e2m1_codes(normalized, stochastic)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, block_scales, global_scale


try:
    _quantize_nvfp4_compiled = torch.compile(_quantize_nvfp4_core, dynamic=True)
except Exception:
    _quantize_nvfp4_compiled = _quantize_nvfp4_core


@torch.no_grad()
def quantize_nvfp4(
    matrix: Tensor,
    *,
    two_dimensional: bool = False,
    stochastic: bool = False,
    causal_row_scale: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode a CUDA matrix as packed FP4 (`float4_e2m1fn_x2`), swizzled FP8 scales, and FP32 scale."""
    if matrix.ndim != 2 or not matrix.is_floating_point() or matrix.numel() == 0:
        raise ValueError("NVFP4 quantization requires a nonempty floating-point matrix")
    if matrix.device.type != "cuda":
        raise ValueError("NVFP4 quantization requires a CUDA matrix")
    rows, columns = matrix.shape
    if columns % 16 or (two_dimensional and rows % 16):
        raise ValueError("NVFP4 blocks require aligned dimensions divisible by 16")
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    if _HAS_TRITON_DOT_SCALED and rows * columns >= 4096:
        mat_c = matrix.detach().contiguous()
        one_f32 = _unit_scale_f32(str(matrix.device))
        scale_cols = columns // 16
        padded_rows = _round_up(rows, 128)
        padded_cols = _round_up(scale_cols, 4)
        packed = torch.empty((rows, columns // 2), dtype=torch.uint8, device=matrix.device)
        if padded_rows == rows and padded_cols == scale_cols:
            swizzled = torch.empty(
                (padded_rows, padded_cols), dtype=torch.float8_e4m3fn, device=matrix.device
            )
        else:
            swizzled = torch.zeros(
                (padded_rows, padded_cols), dtype=torch.float8_e4m3fn, device=matrix.device
            )
        if two_dimensional and not stochastic:
            global_scale = _compute_global_scale_compiled(mat_c, one_f32)
            _nvfp4_quantize_2d_swizzled_kernel[(rows // 16, scale_cols)](
                mat_c,
                global_scale,
                packed,
                swizzled,
                rows,
                K=columns,
                PADDED_SCALE_COLS=padded_cols,
                num_warps=1,
            )
            return packed.view(fp4_dtype), swizzled, global_scale
        if not two_dimensional and columns % 128 == 0:
            unit_g = bool(causal_row_scale)
            global_scale = one_f32 if unit_g else _compute_global_scale_compiled(mat_c, one_f32)
            seed = random.getrandbits(31) if stochastic else 0
            grid_1d = (triton.cdiv(rows, 64), columns // 128)
            _nvfp4_quantize_1d_swizzled_kernel[grid_1d](
                mat_c,
                global_scale,
                packed,
                swizzled,
                rows,
                K=columns,
                PADDED_SCALE_COLS=padded_cols,
                seed=seed,
                BLOCK_M=64,
                BLOCK_K=128,
                STOCHASTIC=stochastic,
                UNIT_GLOBAL_SCALE=unit_g,
                num_warps=4,
            )
            return packed.view(fp4_dtype), swizzled, global_scale

    values = matrix.detach().float().contiguous()
    packed, block_scales, global_scale = _quantize_nvfp4_compiled(
        values, two_dimensional, stochastic, causal_row_scale
    )
    return (
        packed.view(fp4_dtype),
        _swizzle_scales(block_scales),
        global_scale,
    )


@torch.no_grad()
def _dequantize_nvfp4(
    packed: Tensor,
    swizzled_scales: Tensor,
    global_scale: Tensor,
    rows: int,
    columns: int,
) -> Tensor:
    """Decode packed E2M1 + swizzled FP8 E4M3 block scales back to FP32 on pre-Blackwell GPUs."""
    raw_u8 = packed.view(torch.uint8)
    c0 = raw_u8 & 0x0F
    c1 = (raw_u8 >> 4) & 0x0F
    codes = torch.stack((c0, c1), dim=-1).reshape(rows, columns)
    sign = torch.where((codes & 0x08) != 0, -1.0, 1.0)
    mag_idx = (codes & 0x07).long()
    levels = _e2m1_levels(str(packed.device))
    norm_vals = sign * levels[mag_idx]
    block_scales = _unswizzle_scales(swizzled_scales, rows, columns // 16)
    decoding = block_scales * global_scale.float()
    return (norm_vals.view(rows, columns // 16, 16) * decoding.unsqueeze(-1)).reshape(rows, columns)


@torch.no_grad()
def nvfp4_mm(
    a: Tensor,
    b: Tensor,
    *,
    weight_b: bool = False,
    stochastic_a: bool = False,
    stochastic_b: bool = False,
    causal_a: bool = False,
    prequantized_b: tuple[Tensor, Tensor, Tensor] | None = None,
    _phase: str = "other",
) -> Tensor:
    """Return `a @ b.T` using native Blackwell NVFP4 (`F.scaled_mm`) or exact E2M1 emulation on pre-Blackwell CUDA."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("NVFP4 matmul expects a[M,K] and b[N,K]")
    if a.device != b.device or a.device.type != "cuda":
        raise ValueError("NVFP4 matmul requires both operands on the same CUDA device")
    m, k = a.shape
    n = b.shape[0]
    if min(m, n, k) == 0:
        raise ValueError("NVFP4 matmul requires nonempty operands")
    kp = _round_up(k, 32)
    mp = _round_up(m, 32)
    np_ = _round_up(n, 32)
    ap = F.pad(a, (0, kp - k, 0, mp - m)) if (kp != k or mp != m) else a
    qa, sa, ga = quantize_nvfp4(ap, stochastic=stochastic_a, causal_row_scale=causal_a)
    if prequantized_b is not None:
        qb, sb, gb = prequantized_b
    else:
        bp = F.pad(b, (0, kp - k, 0, np_ - n)) if (kp != k or np_ != n) else b
        qb, sb, gb = quantize_nvfp4(bp, two_dimensional=weight_b, stochastic=stochastic_b)
    if has_native_nvfp4(a.device):
        recipe = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
        swizzle = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
        result = F.scaled_mm(
            qa,
            qb.t(),
            scale_a=[sa, ga],
            scale_recipe_a=recipe,
            scale_b=[sb, gb],
            scale_recipe_b=recipe,
            swizzle_a=swizzle,
            swizzle_b=swizzle,
            output_dtype=torch.float32,
        )
    else:
        deq_a = _dequantize_nvfp4(qa, sa, ga, mp, kp)
        deq_b = _dequantize_nvfp4(qb, sb, gb, np_, kp)
        result = torch.matmul(deq_a, deq_b.t())
    _GEMM_COUNTS[_phase] += 1
    return result[:m, :n]


@lru_cache(maxsize=8)
def _hadamard16_matrix(device_str: str) -> Tensor:
    """Return the precomputed 16x16 signed Walsh-Hadamard orthonormal matrix on `device_str`."""
    values = torch.eye(16, dtype=torch.float32, device=torch.device(device_str)).reshape(16, 1, 16)
    values = values * _hadamard16_signs(device_str)
    for width in (1, 2, 4, 8):
        pairs = values.reshape(16, 1, 16 // (2 * width), 2, width)
        left, right = pairs[..., 0, :], pairs[..., 1, :]
        values = torch.stack((left + right, left - right), dim=-2).reshape(16, 1, 16)
    return (values * 0.25).reshape(16, 16).contiguous()


def _hadamard16(matrix: Tensor) -> Tensor:
    """Shared fixed-sign orthonormal transform over groups of 16 columns."""
    rows, columns = matrix.shape
    cp = _round_up(columns, 16)
    values = F.pad(matrix.float(), (0, cp - columns)) if cp != columns else matrix.float()
    h16 = _hadamard16_matrix(str(values.device))
    return torch.matmul(values.reshape(-1, 16), h16).reshape(rows, cp)


@torch.no_grad()
def _get_or_build_weight_nvfp4_cache(
    weight: Tensor,
    sparsity_mode: str,
    *,
    need_bwd: bool = False,
) -> list:
    """Cache `[key, mask, w_eff, fwd_q, bwd_q]` on `weight` keyed by `weight._version`.

    Within a training step (before `optimizer.step()` increments `weight._version`),
    `weight` is immutable across micro-batches, recurrence steps, `forward_all_operators`,
    `Ornith`, and `Dream-RSI`, eliminating ~85% of redundant 2:4 mask and 2D weight
    quantization passes with zero mathematical difference.
    """
    key = (weight._version, weight.data_ptr(), weight.shape, str(weight.device), sparsity_mode)
    cache = getattr(weight, "_nvfp4_weight_cache", None)
    if cache is None or cache[0] != key:
        mask: Tensor | None = None
        w_eff = weight.detach()
        if sparsity_mode != "none":
            mask = compute_2to4_mask(w_eff, mode=sparsity_mode)
            w_eff = w_eff * mask.to(w_eff.dtype)
        n, k = w_eff.shape
        np_, kp = _round_up(n, 32), _round_up(k, 32)
        bp = F.pad(w_eff, (0, kp - k, 0, np_ - n)) if (kp != k or np_ != n) else w_eff
        fwd_q = quantize_nvfp4(bp, two_dimensional=True, stochastic=False)
        cache = [key, mask, w_eff, fwd_q, None]
        try:
            setattr(weight, "_nvfp4_weight_cache", cache)
        except Exception:
            pass
    if need_bwd and cache[4] is None:
        w_t = cache[2].t().contiguous()
        kt, nt = w_t.shape
        ktp, ntp = _round_up(kt, 32), _round_up(nt, 32)
        bpt = F.pad(w_t, (0, ntp - nt, 0, ktp - kt)) if (ntp != nt or ktp != kt) else w_t
        cache[4] = quantize_nvfp4(bpt, two_dimensional=True, stochastic=False)
    return cache


class _NVFP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        sparsity_mode: str = "none",
        sr_ste_decay: float = 0.0,
        causal_activation_scale: bool = True,
    ) -> Tensor:
        cache = _get_or_build_weight_nvfp4_cache(weight, sparsity_mode, need_bwd=False)
        mask: Tensor | None = cache[1]
        w_eff: Tensor = cache[2]
        fwd_q: tuple[Tensor, Tensor, Tensor] = cache[3]
        if sparsity_mode != "none":
            _GEMM_COUNTS["sparse_2to4"] += 1
        ctx.save_for_backward(inputs, weight, mask)
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.sparsity_mode = sparsity_mode
        ctx.sr_ste_decay = float(sr_ste_decay)
        output = nvfp4_mm(
            inputs,
            w_eff,
            weight_b=True,
            causal_a=causal_activation_scale,
            prequantized_b=fwd_q,
            _phase="forward",
        )
        if bias is not None:
            output = output + bias.float()
        return output.to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        inputs, weight, mask = ctx.saved_tensors
        grad_inputs = grad_weight = grad_bias = None
        gradient = grad_output.reshape(-1, weight.shape[0])
        if ctx.needs_input_grad[0]:
            cache = _get_or_build_weight_nvfp4_cache(weight, ctx.sparsity_mode, need_bwd=True)
            w_bwd_t = cache[2].t()
            bwd_q = cache[4]
            if mask is not None:
                _GEMM_COUNTS["sparse_2to4"] += 1
            grad_inputs = nvfp4_mm(
                gradient,
                w_bwd_t,
                weight_b=True,
                stochastic_a=True,
                prequantized_b=bwd_q,
                _phase="dgrad",
            ).to(inputs.dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = nvfp4_mm(
                _hadamard16(gradient.t()),
                _hadamard16(inputs.t()),
                stochastic_a=True,
                _phase="wgrad",
            ).to(weight.dtype)
            if mask is not None and ctx.sr_ste_decay > 0.0:
                # SR-STE (Zhou et al., ICLR 2021): penalize masked-out master weights
                # so they smoothly decay toward 0 without destabilizing mask transitions.
                grad_weight = grad_weight + (
                    ctx.sr_ste_decay * (~mask).to(weight.dtype) * weight
                )
        if ctx.bias_dtype is not None and ctx.needs_input_grad[2]:
            grad_bias = gradient.float().sum(dim=0).to(ctx.bias_dtype)
        return grad_inputs, grad_weight, grad_bias, None, None, None


class NVFP4Linear(nn.Linear):
    """Projection executing native Blackwell NVFP4 GEMMs with optional 2:4 structured sparsity."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        sparsity_mode: str = "2to4_transposable",
        sr_ste_decay: float = 1e-4,
        causal_activation_scale: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if sparsity_mode not in SPARSITY_MODES:
            raise ValueError(f"sparsity_mode must be one of {SPARSITY_MODES}")
        self.sparsity_mode = sparsity_mode
        self.sr_ste_decay = float(sr_ste_decay)
        self.causal_activation_scale = bool(causal_activation_scale)

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        sparsity_mode: str = "2to4_transposable",
        sr_ste_decay: float = 1e-4,
        causal_activation_scale: bool = True,
    ) -> "NVFP4Linear":
        converted = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
            sparsity_mode=sparsity_mode,
            sr_ste_decay=sr_ste_decay,
            causal_activation_scale=causal_activation_scale,
        )
        converted.load_state_dict(layer.state_dict())
        converted.weight.requires_grad_(layer.weight.requires_grad)
        if converted.bias is not None:
            converted.bias.requires_grad_(layer.bias.requires_grad)
        converted.train(layer.training)
        return converted

    @torch.no_grad()
    def active_2to4_mask(self) -> Tensor:
        """Return the current 2:4 boolean weight mask `[out_features, in_features]`."""
        return compute_2to4_mask(self.weight, mode=self.sparsity_mode)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError("NVFP4Linear input must end in in_features")
        if inputs.device.type != "cuda":
            w = (
                self.weight * compute_2to4_mask(self.weight, self.sparsity_mode).to(self.weight.dtype)
                if self.sparsity_mode != "none"
                else self.weight
            )
            return F.linear(inputs, w, self.bias)
        flat = inputs.reshape(-1, self.in_features)
        if flat.shape[0] == 0:
            return inputs.new_empty(*inputs.shape[:-1], self.out_features)
        result = _NVFP4Linear.apply(
            flat,
            self.weight,
            self.bias,
            self.sparsity_mode,
            self.sr_ste_decay,
            self.causal_activation_scale,
        )
        return result.reshape(*inputs.shape[:-1], self.out_features)


class NVFP4Conv2d(nn.Conv2d):
    """Convolution expressed as unfold plus native NVFP4 training GEMMs."""

    @classmethod
    def from_conv(cls, conv: nn.Conv2d) -> "NVFP4Conv2d":
        if conv.groups != 1 or conv.padding_mode != "zeros" or isinstance(conv.padding, str):
            raise ValueError("NVFP4Conv2d currently supports groups=1 and numeric zero padding")
        converted = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        converted.load_state_dict(conv.state_dict())
        converted.weight.requires_grad_(conv.weight.requires_grad)
        if converted.bias is not None:
            converted.bias.requires_grad_(conv.bias.requires_grad)
        converted.train(conv.training)
        return converted

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("NVFP4Conv2d expects [batch, channels, height, width]")
        if inputs.device.type != "cuda":
            return self._conv_forward(inputs, self.weight, self.bias)
        batch, _, height, width = inputs.shape
        patches = F.unfold(
            inputs,
            self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        patches = patches.transpose(1, 2).reshape(-1, patches.shape[1])
        result = _NVFP4Linear.apply(patches, self.weight.flatten(1), self.bias, "none", 0.0, False)
        out_height = (
            height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        out_width = (
            width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1
        ) // self.stride[1] + 1
        return (
            result.reshape(batch, out_height * out_width, self.out_channels)
            .transpose(1, 2)
            .reshape(batch, self.out_channels, out_height, out_width)
        )


def _convert_expert_ffns(
    module: nn.Module,
    *,
    min_features: int,
    sparsity_mode: str,
    sr_ste_decay: float,
    causal_activation_scale: bool,
) -> int:
    """Convert `SwiGLUExpert` (`up_proj`, `down_proj`) FFN layers to `NVFP4Linear`."""
    converted = 0
    for child in module.modules():
        if child.__class__.__name__ == "SwiGLUExpert":
            for attr in ("up_proj", "down_proj"):
                layer = getattr(child, attr, None)
                if (
                    isinstance(layer, nn.Linear)
                    and not isinstance(layer, NVFP4Linear)
                    and layer.in_features >= min_features
                    and layer.out_features >= min_features
                ):
                    setattr(
                        child,
                        attr,
                        NVFP4Linear.from_linear(
                            layer,
                            sparsity_mode=sparsity_mode,
                            sr_ste_decay=sr_ste_decay,
                            causal_activation_scale=causal_activation_scale,
                        ),
                    )
                    converted += 1
    return converted


def convert_linear_to_nvfp4(
    module: nn.Module,
    *,
    min_features: int = 32,
    experts_only: bool = True,
    sparsity_mode: str = "2to4_transposable",
    sr_ste_decay: float = 1e-4,
    causal_activation_scale: bool = True,
    exclude_names: tuple[str, ...] = ("router", "token_scorer", "ple", "route_", "start_query"),
) -> int:
    """Convert eligible `nn.Linear` submodules to `NVFP4Linear` with 2:4 sparsity.

    Following `oracle/src/oracle/multimodal.py:_convert_feedforwards`, when `module`
    contains `SwiGLUExpert` blocks and `experts_only=True`, only the expert FFN
    projections (`up_proj`, `down_proj`) are converted to `NVFP4Linear`, keeping
    attention, `PerLayerEmbedding`, `MCL4` routers, MoE routers, and `COE` 32-dim
    single-softmax readout heads in BF16/FP32.
    """
    if experts_only:
        expert_count = _convert_expert_ffns(
            module,
            min_features=min_features,
            sparsity_mode=sparsity_mode,
            sr_ste_decay=sr_ste_decay,
            causal_activation_scale=causal_activation_scale,
        )
        if expert_count > 0:
            return expert_count

    converted_count = 0
    for child_name, child in list(module.named_children()):
        if any(ex in child_name for ex in exclude_names):
            continue
        if (
            isinstance(child, nn.Linear)
            and not isinstance(child, NVFP4Linear)
            and child.in_features >= min_features
            and child.out_features >= min_features
        ):
            setattr(
                module,
                child_name,
                NVFP4Linear.from_linear(
                    child,
                    sparsity_mode=sparsity_mode,
                    sr_ste_decay=sr_ste_decay,
                    causal_activation_scale=causal_activation_scale,
                ),
            )
            converted_count += 1
        else:
            converted_count += convert_linear_to_nvfp4(
                child,
                min_features=min_features,
                experts_only=False,
                sparsity_mode=sparsity_mode,
                sr_ste_decay=sr_ste_decay,
                causal_activation_scale=causal_activation_scale,
                exclude_names=exclude_names,
            )
    return converted_count


try:
    import triton
    import triton.language as tl

    _HAS_TRITON_DOT_SCALED = hasattr(tl, "dot_scaled") and hasattr(tl, "float8e4nv")
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON_DOT_SCALED = False


def has_triton_nvfp4_grouped(device: torch.device | str = "cuda") -> bool:
    """Return True if Triton `tl.dot_scaled` native Blackwell NVFP4 grouped kernels are supported."""
    return bool(_HAS_TRITON_DOT_SCALED and has_native_nvfp4(device))


if _HAS_TRITON_DOT_SCALED:

    @triton.jit
    def _e2m1_nibble_triton(x, noise, STOCHASTIC: tl.constexpr):
        m = tl.minimum(tl.abs(x), 6.0)
        u = tl.where(m <= 2.0, m * 2.0, tl.where(m <= 4.0, m + 2.0, m * 0.5 + 4.0))
        if STOCHASTIC:
            code = tl.minimum((u + noise).to(tl.uint8), 7)
        else:
            code = tl.extra.cuda.libdevice.rint(u).to(tl.uint8)
        sign = tl.where(x < 0.0, 8, 0).to(tl.uint8)
        return code | sign

    @triton.jit
    def _nvfp4_quantize_1d_swizzled_kernel(
        X_ptr,
        GlobalScale_ptr,
        Packed_ptr,
        SwizzledScale_ptr,
        M,
        K: tl.constexpr,
        PADDED_SCALE_COLS: tl.constexpr,
        seed,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        STOCHASTIC: tl.constexpr,
        UNIT_GLOBAL_SCALE: tl.constexpr,
    ):
        """DeepGEMM-inspired single-pass 1D NVFP4 quantizer + in-register `SWIZZLE_32_4_4` scale layout."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        N_BLOCKS: tl.constexpr = BLOCK_K // 16
        k_block_start = pid_k * N_BLOCKS
        offs_b = k_block_start + tl.arange(0, N_BLOCKS)
        mask_b = offs_b < (K // 16)
        offs_pair = tl.arange(0, 8)

        col_even = offs_b[None, :, None] * 16 + offs_pair[None, None, :] * 2
        col_odd = col_even + 1
        row_base = offs_m[:, None, None] * K
        mask_load = mask_m[:, None, None] & mask_b[None, :, None]

        v0 = tl.load(X_ptr + row_base + col_even, mask=mask_load, other=0.0).to(tl.float32)
        v1 = tl.load(X_ptr + row_base + col_odd, mask=mask_load, other=0.0).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(v0), axis=2), tl.max(tl.abs(v1), axis=2))
        if UNIT_GLOBAL_SCALE:
            scale_f8 = tl.minimum(amax / 6.0, 448.0).to(tl.float8e4nv)
            decoding = scale_f8.to(tl.float32)
        else:
            g_a = tl.load(GlobalScale_ptr).to(tl.float32)
            scale_f8 = tl.minimum(amax / (6.0 * g_a), 448.0).to(tl.float8e4nv)
            decoding = scale_f8.to(tl.float32) * g_a
        inv = tl.where(decoding > 0.0, 1.0 / decoding, 0.0)[:, :, None]

        if STOCHASTIC:
            n0 = tl.rand(seed, row_base + col_even)
            n1 = tl.rand(seed, row_base + col_odd)
        else:
            n0 = 0.0
            n1 = 0.0

        c0 = _e2m1_nibble_triton(v0 * inv, n0, STOCHASTIC)
        c1 = _e2m1_nibble_triton(v1 * inv, n1, STOCHASTIC)
        packed = (c0 | (c1 << 4)).reshape(BLOCK_M, BLOCK_K // 2)

        offs_kp = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
        mask_kp = offs_kp < (K // 2)
        tl.store(
            Packed_ptr + offs_m[:, None] * (K // 2) + offs_kp[None, :],
            packed,
            mask=mask_m[:, None] & mask_kp[None, :],
        )

        r = offs_m[:, None]
        c = offs_b[None, :]
        i0 = r // 128
        i1 = (r // 32) % 4
        i2 = r % 32
        i3 = c // 4
        i4 = c % 4
        swizzled_idx = (i0 * (PADDED_SCALE_COLS // 4) + i3) * 512 + i2 * 16 + i1 * 4 + i4
        tl.store(
            SwizzledScale_ptr + swizzled_idx,
            scale_f8,
            mask=mask_m[:, None] & mask_b[None, :],
        )

    @triton.jit
    def _nvfp4_quantize_2d_swizzled_kernel(
        W_ptr,
        GlobalScale_ptr,
        Packed_ptr,
        SwizzledScale_ptr,
        M,
        K: tl.constexpr,
        PADDED_SCALE_COLS: tl.constexpr,
    ):
        """DeepGEMM-inspired single-pass 2D weight quantizer + in-register `SWIZZLE_32_4_4` scale layout."""
        bn = tl.program_id(0)
        bk = tl.program_id(1)

        offs_m = bn * 16 + tl.arange(0, 16)
        mask_m = offs_m < M
        offs_pair = tl.arange(0, 8)
        offs_k_even = bk * 16 + offs_pair * 2
        offs_k_odd = offs_k_even + 1

        v_even = tl.load(
            W_ptr + offs_m[:, None] * K + offs_k_even[None, :], mask=mask_m[:, None], other=0.0
        ).to(tl.float32)
        v_odd = tl.load(
            W_ptr + offs_m[:, None] * K + offs_k_odd[None, :], mask=mask_m[:, None], other=0.0
        ).to(tl.float32)

        g_b = tl.load(GlobalScale_ptr).to(tl.float32)
        block_amax = tl.maximum(tl.max(tl.abs(v_even)), tl.max(tl.abs(v_odd)))
        scale_f8 = tl.minimum(block_amax / (6.0 * g_b), 448.0).to(tl.float8e4nv)
        dec = scale_f8.to(tl.float32) * g_b
        inv = tl.where(dec > 0.0, 1.0 / dec, 0.0)

        c_even = _e2m1_nibble_triton(v_even * inv, 0.0, False)
        c_odd = _e2m1_nibble_triton(v_odd * inv, 0.0, False)
        packed = c_even | (c_odd << 4)

        offs_kp = bk * 8 + offs_pair
        tl.store(
            Packed_ptr + offs_m[:, None] * (K // 2) + offs_kp[None, :],
            packed,
            mask=mask_m[:, None],
        )

        r = offs_m
        c = bk
        i0 = r // 128
        i1 = (r // 32) % 4
        i2 = r % 32
        i3 = c // 4
        i4 = c % 4
        swizzled_idx = (i0 * (PADDED_SCALE_COLS // 4) + i3) * 512 + i2 * 16 + i1 * 4 + i4
        tl.store(SwizzledScale_ptr + swizzled_idx, scale_f8, mask=mask_m)

    @triton.jit
    def _triton_2to4_direct_2d_kernel(
        W_ptr,
        Cat_ptr,
        CatBool_ptr,
        WMasked_ptr,
        Mask_ptr,
        SR_ptr,
        sr_decay,
        N_TILE4,
        K_TILE4: tl.constexpr,
        K: tl.constexpr,
        BLOCK_R: tl.constexpr,
        WRITE_MASKED_W: tl.constexpr,
        WRITE_SR_STE: tl.constexpr,
    ):
        """In-register BF16 TensorCore doubly-2:4 (`2to4_transposable`) mask & SR-STE kernel."""
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_t = tl.arange(0, BLOCK_R)
        r4 = pid_r * (BLOCK_R // 4) + offs_t // 4
        c4 = pid_c * 4 + (offs_t % 4)
        mask_t = (r4 < N_TILE4) & (c4 < K_TILE4)

        offs_16 = tl.arange(0, 16)
        dr = offs_16 // 4
        dc = offs_16 % 4
        addr = (r4[:, None] * 4 + dr[None, :]) * K + (c4[:, None] * 4 + dc[None, :])

        w_val = tl.load(W_ptr + addr, mask=mask_t[:, None], other=0.0)
        w_abs = tl.abs(w_val.to(tl.float32)).to(tl.bfloat16)

        offs_128 = tl.arange(0, 128)
        cat = tl.load(Cat_ptr + offs_16[:, None] * 128 + offs_128[None, :])
        scores = tl.dot(w_abs, cat)
        best_idx = tl.argmax(scores, axis=1)

        m_bool = tl.load(
            CatBool_ptr + best_idx[:, None] * 16 + offs_16[None, :],
            mask=mask_t[:, None],
            other=0,
        ).to(tl.int1)
        tl.store(Mask_ptr + addr, m_bool, mask=mask_t[:, None])
        if WRITE_MASKED_W:
            tl.store(WMasked_ptr + addr, tl.where(m_bool, w_val, 0.0), mask=mask_t[:, None])
        if WRITE_SR_STE:
            sr_val = tl.where(m_bool, 0.0, sr_decay * w_val.to(tl.float32)).to(
                SR_ptr.dtype.element_ty
            )
            tl.store(SR_ptr + addr, sr_val, mask=mask_t[:, None])

    @triton.jit
    def _triton_quantize_2d_grouped_weights_kernel(
        W_ptr,
        Bg_ptr,
        Bfwd_ptr,
        Bsfwd_ptr,
        Bbwd_ptr,
        Bsbwd_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
    ):
        """Fused single-pass 2D forward + transposed backward NVFP4 grouped weight quantizer."""
        d = tl.program_id(0)
        bn = tl.program_id(1)
        bk = tl.program_id(2)

        offs_n = bn * 16 + tl.arange(0, 16)
        offs_k_pair = tl.arange(0, 8)
        offs_k_even = bk * 16 + offs_k_pair * 2
        offs_k_odd = offs_k_even + 1

        w_base = W_ptr + d * N * K
        v_even = tl.load(w_base + offs_n[:, None] * K + offs_k_even[None, :]).to(tl.float32)
        v_odd = tl.load(w_base + offs_n[:, None] * K + offs_k_odd[None, :]).to(tl.float32)

        bg = tl.load(Bg_ptr + d).to(tl.float32)
        block_amax = tl.maximum(tl.max(tl.abs(v_even)), tl.max(tl.abs(v_odd)))
        bs_f8 = tl.minimum(block_amax / (6.0 * bg), 448.0).to(tl.float8e4nv)
        dec = bs_f8.to(tl.float32) * bg
        inv = tl.where(dec > 0.0, 1.0 / dec, 0.0)

        c_even = _e2m1_nibble_triton(v_even * inv, 0.0, False)
        c_odd = _e2m1_nibble_triton(v_odd * inv, 0.0, False)

        packed_fwd = tl.trans(c_even | (c_odd << 4))
        offs_kp = bk * 8 + offs_k_pair
        bfwd_base = Bfwd_ptr + d * (K // 2) * N
        tl.store(bfwd_base + offs_kp[:, None] * N + offs_n[None, :], packed_fwd)

        bsfwd_base = Bsfwd_ptr + d * N * (K // 16)
        tl.store(bsfwd_base + offs_n * (K // 16) + bk, bs_f8)

        offs_k = bk * 16 + tl.arange(0, 16)
        offs_n_pair = tl.arange(0, 8)
        offs_n_even = bn * 16 + offs_n_pair * 2
        offs_n_odd = offs_n_even + 1
        vn_even = tl.load(w_base + offs_n_even[:, None] * K + offs_k[None, :]).to(tl.float32)
        vn_odd = tl.load(w_base + offs_n_odd[:, None] * K + offs_k[None, :]).to(tl.float32)
        cn_even = _e2m1_nibble_triton(vn_even * inv, 0.0, False)
        cn_odd = _e2m1_nibble_triton(vn_odd * inv, 0.0, False)
        packed_bwd = cn_even | (cn_odd << 4)
        offs_np = bn * 8 + offs_n_pair
        bbwd_base = Bbwd_ptr + d * (N // 2) * K
        tl.store(bbwd_base + offs_np[:, None] * K + offs_k[None, :], packed_bwd)

        bsbwd_base = Bsbwd_ptr + d * K * (N // 16)
        tl.store(bsbwd_base + offs_k * (N // 16) + bn, bs_f8)

    @triton.jit
    def _nvfp4_quantize_1d_kernel(
        X_ptr,
        GlobalScale_ptr,
        Packed_ptr,
        Scale_ptr,
        M,
        K: tl.constexpr,
        seed,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        STOCHASTIC: tl.constexpr,
        UNIT_GLOBAL_SCALE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        N_BLOCKS: tl.constexpr = BLOCK_K // 16
        k_block_start = pid_k * N_BLOCKS
        offs_b = k_block_start + tl.arange(0, N_BLOCKS)
        offs_pair = tl.arange(0, 8)

        col_even = offs_b[None, :, None] * 16 + offs_pair[None, None, :] * 2
        col_odd = col_even + 1
        row_base = offs_m[:, None, None] * K

        v0 = tl.load(X_ptr + row_base + col_even, mask=mask_m[:, None, None], other=0.0).to(
            tl.float32
        )
        v1 = tl.load(X_ptr + row_base + col_odd, mask=mask_m[:, None, None], other=0.0).to(
            tl.float32
        )

        amax = tl.maximum(tl.max(tl.abs(v0), axis=2), tl.max(tl.abs(v1), axis=2))
        if UNIT_GLOBAL_SCALE:
            scale_f8 = tl.minimum(amax / 6.0, 448.0).to(tl.float8e4nv)
            decoding = scale_f8.to(tl.float32)
        else:
            g_a = tl.load(GlobalScale_ptr).to(tl.float32)
            scale_f8 = tl.minimum(amax / (6.0 * g_a), 448.0).to(tl.float8e4nv)
            decoding = scale_f8.to(tl.float32) * g_a
        inv = tl.where(decoding > 0.0, 1.0 / decoding, 0.0)[:, :, None]

        if STOCHASTIC:
            n0 = tl.rand(seed, row_base + col_even)
            n1 = tl.rand(seed, row_base + col_odd)
        else:
            n0 = 0.0
            n1 = 0.0

        c0 = _e2m1_nibble_triton(v0 * inv, n0, STOCHASTIC)
        c1 = _e2m1_nibble_triton(v1 * inv, n1, STOCHASTIC)
        packed = (c0 | (c1 << 4)).reshape(BLOCK_M, BLOCK_K // 2)

        offs_kp = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
        offs_kp = tl.max_contiguous(tl.multiple_of(offs_kp, BLOCK_K // 2), BLOCK_K // 2)
        offs_b = tl.max_contiguous(tl.multiple_of(offs_b, N_BLOCKS), N_BLOCKS)
        tl.store(
            Packed_ptr + offs_m[:, None] * (K // 2) + offs_kp[None, :],
            packed,
            mask=mask_m[:, None],
        )
        tl.store(
            Scale_ptr + offs_m[:, None] * (K // 16) + offs_b[None, :],
            scale_f8,
            mask=mask_m[:, None],
        )

    @triton.jit
    def _nvfp4_grouped_mm_kernel(
        A_ptr,
        As_ptr,
        Ag_ptr,
        B_ptr,
        Bs_ptr,
        Bg_ptr,
        Offs_ptr,
        Out_ptr,
        M_total,
        N: tl.constexpr,
        K: tl.constexpr,
        E: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        UNIT_A_SCALE: tl.constexpr,
    ):
        """DeepGEMM-inspired L2-swizzled multi-stage grouped NVFP4 TensorCore GEMM kernel."""
        tile_id = tl.program_id(0)
        pid_n = tl.program_id(1)
        num_pid_m = tl.num_programs(0)
        num_pid_n = tl.num_programs(1)
        tile_id, pid_n = tl.swizzle2d(tile_id, pid_n, num_pid_m, num_pid_n, GROUP_M)

        cum_tiles = 0
        chosen_e = -1
        chosen_m_start = 0
        chosen_m_end = 0
        prev_off = 0
        for e in tl.static_range(E):
            cur_off = tl.load(Offs_ptr + e)
            cnt = cur_off - prev_off
            n_tiles = (cnt + BLOCK_M - 1) // BLOCK_M
            is_match = (tile_id >= cum_tiles) & (tile_id < cum_tiles + n_tiles)
            chosen_e = tl.where(is_match, e, chosen_e)
            chosen_m_start = tl.where(
                is_match, prev_off + (tile_id - cum_tiles) * BLOCK_M, chosen_m_start
            )
            chosen_m_end = tl.where(is_match, cur_off, chosen_m_end)
            cum_tiles += n_tiles
            prev_off = cur_off

        if chosen_e < 0:
            return

        offs_m = chosen_m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < chosen_m_end
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

        b_expert_base = B_ptr + chosen_e * (K // 2) * N
        bs_expert_base = Bs_ptr + chosen_e * N * (K // 16)
        g_b = tl.load(Bg_ptr + chosen_e).to(tl.float32)
        if UNIT_A_SCALE:
            g_scale = g_b
        else:
            g_a = tl.load(Ag_ptr).to(tl.float32)
            g_scale = g_a * g_b

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_kp = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
            offs_ks = (k0 // 16) + tl.arange(0, BLOCK_K // 16)
            offs_kp = tl.max_contiguous(tl.multiple_of(offs_kp, BLOCK_K // 2), BLOCK_K // 2)
            offs_ks = tl.max_contiguous(tl.multiple_of(offs_ks, BLOCK_K // 16), BLOCK_K // 16)

            a = tl.load(
                A_ptr + offs_m[:, None] * (K // 2) + offs_kp[None, :],
                mask=mask_m[:, None],
                other=0,
            )
            a_scale = tl.load(
                As_ptr + offs_m[:, None] * (K // 16) + offs_ks[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            b = tl.load(b_expert_base + offs_kp[:, None] * N + offs_n[None, :])
            b_scale = tl.load(bs_expert_base + offs_n[:, None] * (K // 16) + offs_ks[None, :])
            acc = tl.dot_scaled(
                a, a_scale, "e2m1", b, b_scale, "e2m1", acc=acc, rhs_k_pack=True
            )

        acc = acc * g_scale
        tl.store(
            Out_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(Out_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )

    @triton.jit
    def _grouped_wgrad_kernel(
        Grad_ptr,
        X_ptr,
        SR_ptr,
        Offs_ptr,
        MaskReq_ptr,
        dW_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_N: tl.constexpr,
        HAS_SR_STE: tl.constexpr,
    ):
        """DeepGEMM-inspired coalesced SMEM-transposed L2-swizzled grouped `wgrad` + fused SR-STE kernel."""
        e = tl.program_id(0)
        req = tl.load(MaskReq_ptr + e)
        if req == 0:
            return
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        num_pid_n = tl.num_programs(1)
        num_pid_k = tl.num_programs(2)
        pid_n, pid_k = tl.swizzle2d(pid_n, pid_k, num_pid_n, num_pid_k, GROUP_N)

        start_m = 0 if e == 0 else tl.load(Offs_ptr + e - 1)
        end_m = tl.load(Offs_ptr + e)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
        offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for m0 in range(start_m, end_m, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < end_m
            g = tl.load(
                Grad_ptr + offs_m[:, None] * N + offs_n[None, :],
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            x_sub = tl.load(
                X_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            acc = tl.dot(tl.trans(g), x_sub, acc=acc)

        ep_offset = (e * N + offs_n[:, None]) * K + offs_k[None, :]
        if HAS_SR_STE:
            sr_val = tl.load(SR_ptr + ep_offset).to(tl.float32)
            acc = acc + sr_val

        tl.store(dW_ptr + ep_offset, acc.to(dW_ptr.dtype.element_ty))


def _quantize_grouped_weights_core(
    w_batch: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    d, n, k = w_batch.shape
    w_f = w_batch.float()
    w_abs = w_f.abs()
    amax = w_abs.amax(dim=(1, 2))
    bg = torch.where(amax > 0, amax / 2688.0, torch.ones_like(amax)).clamp_min(1.17549435e-38)
    block_amax = w_abs.view(d, n // 16, 16, k // 16, 16).amax(dim=(2, 4))
    bs_tiles = (block_amax / (6.0 * bg.view(d, 1, 1))).clamp(max=448.0).to(torch.float8_e4m3fn)

    dec_tiles = bs_tiles.float() * bg.view(d, 1, 1)
    inv_tiles = torch.where(
        dec_tiles > 0, dec_tiles.reciprocal().clamp(max=3.402823466e38), torch.zeros_like(dec_tiles)
    )
    norm_5d = w_f.view(d, n // 16, 16, k // 16, 16) * inv_tiles[:, :, None, :, None]
    codes_5d = _e2m1_codes(norm_5d, False)

    codes_fwd = codes_5d.reshape(d, n, k)
    b_fwd = (codes_fwd[:, :, 0::2] | (codes_fwd[:, :, 1::2] << 4)).transpose(1, 2).contiguous()
    bs_fwd = bs_tiles.repeat_interleave(16, dim=1).contiguous()

    codes_bwd = codes_5d.permute(0, 3, 4, 1, 2).reshape(d, k, n)
    b_bwd = (codes_bwd[:, :, 0::2] | (codes_bwd[:, :, 1::2] << 4)).transpose(1, 2).contiguous()
    bs_bwd = bs_tiles.transpose(1, 2).repeat_interleave(16, dim=1).contiguous()

    return b_fwd, bs_fwd, bg, b_bwd, bs_bwd


try:
    _quantize_grouped_weights_compiled = torch.compile(_quantize_grouped_weights_core, dynamic=True)
except Exception:
    _quantize_grouped_weights_compiled = _quantize_grouped_weights_core


@torch.no_grad()
def _get_or_build_grouped_nvfp4_cache(experts: nn.ModuleList, proj_name: str) -> dict:
    """Return cached 3D packed NVFP4 expert weight stacks across `experts` for `proj_name`.

    Only re-quantizes experts whose `weight._version` has changed since the last step,
    so frozen donor experts are quantized once and plastic experts are quantized once per
    `optimizer.step()` in a single batched call.
    """
    layers: list[NVFP4Linear] = [getattr(e, proj_name) for e in experts]
    weights = [layer.weight for layer in layers]
    e_count = len(weights)
    n, k = weights[0].shape
    device = weights[0].device
    w_dtype = weights[0].dtype
    sparsity_mode = layers[0].sparsity_mode
    sr_ste_decay = layers[0].sr_ste_decay

    cur_versions = [
        (w._version, _NVFP4_OPT_STEP if w.requires_grad else 0, w.data_ptr(), layer.sparsity_mode)
        for w, layer in zip(weights, layers)
    ]
    cache_dict = getattr(experts, "_nvfp4_grouped_cache", None)
    if cache_dict is None:
        cache_dict = {}
        try:
            setattr(experts, "_nvfp4_grouped_cache", cache_dict)
        except Exception:
            pass

    entry = cache_dict.get(proj_name)
    shape_key = (e_count, n, k, str(device))
    if entry is None or entry.get("shape_key") != shape_key:
        entry = {
            "shape_key": shape_key,
            "versions": [None] * e_count,
            "B_fwd": torch.empty((e_count, k // 2, n), dtype=torch.uint8, device=device),
            "Bs_fwd": torch.empty((e_count, n, k // 16), dtype=torch.float8_e4m3fn, device=device),
            "Bg": torch.empty((e_count,), dtype=torch.float32, device=device),
            "B_bwd": torch.empty((e_count, n // 2, k), dtype=torch.uint8, device=device),
            "Bs_bwd": torch.empty((e_count, k, n // 16), dtype=torch.float8_e4m3fn, device=device),
            "masks_stack": torch.ones((e_count, n, k), dtype=torch.bool, device=device),
            "sr_ste_stack": torch.zeros((e_count, n, k), dtype=w_dtype, device=device),
            "w_masked_scratch": torch.empty((e_count, n, k), dtype=w_dtype, device=device),
            "masks": [None] * e_count,
            "sparsity_mode": sparsity_mode,
            "sr_ste_decay": sr_ste_decay,
            "one_f32": _unit_scale_f32(str(device)),
            "req_cache": {},
        }
        cache_dict[proj_name] = entry

    if entry["versions"] == cur_versions:
        return entry

    dirty = [
        i
        for i, (old_v, new_v) in enumerate(zip(entry["versions"], cur_versions))
        if old_v != new_v
    ]
    if dirty:
        is_contiguous_range = (dirty[-1] - dirty[0] + 1) == len(dirty)
        can_use_triton_2d = (
            _HAS_TRITON_DOT_SCALED
            and device.type == "cuda"
            and n >= 64
            and k >= 64
            and n % 16 == 0
            and k % 16 == 0
            and (k // 4) % 4 == 0
            and is_contiguous_range
            and sparsity_mode in ("2to4_transposable", "none")
        )
        if can_use_triton_2d:
            d_cnt = len(dirty)
            sl = slice(dirty[0], dirty[-1] + 1)
            w_batch_in = torch.stack([weights[i].detach() for i in dirty], dim=0).contiguous()
            if sparsity_mode == "2to4_transposable":
                cat_16x128, cat_bool = _transposable_2to4_triton_tables(str(device))
                w_masked = entry["w_masked_scratch"][:d_cnt]
                m_out = entry["masks_stack"][sl]
                sr_out = entry["sr_ste_stack"][sl]
                n_tile4 = d_cnt * (n // 4)
                k_tile4 = k // 4
                grid_2to4 = (triton.cdiv(n_tile4, 32), triton.cdiv(k_tile4, 4))
                _triton_2to4_direct_2d_kernel[grid_2to4](
                    w_batch_in,
                    cat_16x128,
                    cat_bool,
                    w_masked,
                    m_out,
                    sr_out,
                    float(sr_ste_decay),
                    n_tile4,
                    K_TILE4=k_tile4,
                    K=k,
                    BLOCK_R=128,
                    WRITE_MASKED_W=True,
                    WRITE_SR_STE=bool(sr_ste_decay > 0.0),
                    num_warps=4,
                )
                for idx in dirty:
                    entry["masks"][idx] = entry["masks_stack"][idx]
            else:
                w_masked = w_batch_in
                for idx in dirty:
                    entry["masks"][idx] = None

            amax = w_masked.abs().amax(dim=(1, 2)).float()
            bg = torch.where(amax > 0, amax * (1.0 / 2688.0), entry["one_f32"]).clamp_min(
                1.17549435e-38
            )
            entry["Bg"][sl].copy_(bg)
            grid_q2d = (d_cnt, n // 16, k // 16)
            _triton_quantize_2d_grouped_weights_kernel[grid_q2d](
                w_masked,
                entry["Bg"][sl],
                entry["B_fwd"][sl],
                entry["Bs_fwd"][sl],
                entry["B_bwd"][sl],
                entry["Bs_bwd"][sl],
                N=n,
                K=k,
                num_warps=1,
            )
        else:
            dirty_w_list: list[Tensor] = []
            for idx in dirty:
                w_det = weights[idx].detach().float()
                smode = layers[idx].sparsity_mode
                if smode != "none":
                    m = compute_2to4_mask(w_det, mode=smode)
                    entry["masks"][idx] = m
                    entry["masks_stack"][idx].copy_(m)
                    if sr_ste_decay > 0.0:
                        entry["sr_ste_stack"][idx].copy_(
                            (sr_ste_decay * (~m).float() * w_det).to(w_dtype)
                        )
                    w_det = w_det * m.float()
                else:
                    entry["masks"][idx] = None
                dirty_w_list.append(w_det)
            w_batch = torch.stack(dirty_w_list, dim=0)
            b_fwd, bs_fwd, bg, b_bwd, bs_bwd = _quantize_grouped_weights_compiled(w_batch)
            if is_contiguous_range:
                sl = slice(dirty[0], dirty[-1] + 1)
                entry["B_fwd"][sl].copy_(b_fwd)
                entry["Bs_fwd"][sl].copy_(bs_fwd)
                entry["Bg"][sl].copy_(bg)
                entry["B_bwd"][sl].copy_(b_bwd)
                entry["Bs_bwd"][sl].copy_(bs_bwd)
            else:
                idx_t = torch.tensor(dirty, dtype=torch.long, device=device)
                entry["B_fwd"].index_copy_(0, idx_t, b_fwd)
                entry["Bs_fwd"].view(torch.uint8).index_copy_(0, idx_t, bs_fwd.view(torch.uint8))
                entry["Bg"].index_copy_(0, idx_t, bg)
                entry["B_bwd"].index_copy_(0, idx_t, b_bwd)
                entry["Bs_bwd"].view(torch.uint8).index_copy_(0, idx_t, bs_bwd.view(torch.uint8))
        entry["versions"] = cur_versions
        entry["sparsity_mode"] = sparsity_mode
        entry["sr_ste_decay"] = sr_ste_decay

    return entry


class _NVFP4GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: Tensor,
        offs: Tensor,
        experts: nn.ModuleList,
        proj_name: str,
        *weights: Tensor,
    ) -> Tensor:
        m_total, k = inputs.shape
        e_count = len(weights)
        n = weights[0].shape[0]
        ctx.save_for_backward(inputs, offs, *weights)
        ctx.experts = experts
        ctx.proj_name = proj_name
        if m_total == 0:
            return inputs.new_empty((0, n))

        entry = _get_or_build_grouped_nvfp4_cache(experts, proj_name)
        x_c = inputs.contiguous()
        qa = torch.empty((m_total, k // 2), dtype=torch.uint8, device=inputs.device)
        sa = torch.empty((m_total, k // 16), dtype=torch.float8_e4m3fn, device=inputs.device)
        grid_q = (triton.cdiv(m_total, 64), k // 128)
        _nvfp4_quantize_1d_kernel[grid_q](
            x_c,
            entry["one_f32"],
            qa,
            sa,
            m_total,
            K=k,
            seed=0,
            BLOCK_M=64,
            BLOCK_K=128,
            STOCHASTIC=False,
            UNIT_GLOBAL_SCALE=True,
            num_warps=4,
        )

        out = torch.empty((m_total, n), dtype=inputs.dtype, device=inputs.device)
        grid_mm = ((m_total + 127) // 128 + e_count, n // 128)
        _nvfp4_grouped_mm_kernel[grid_mm](
            qa,
            sa,
            entry["one_f32"],
            entry["B_fwd"],
            entry["Bs_fwd"],
            entry["Bg"],
            offs,
            out,
            m_total,
            N=n,
            K=k,
            E=e_count,
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=128,
            GROUP_M=8,
            UNIT_A_SCALE=True,
            num_warps=4,
            num_stages=3,
        )
        _GEMM_COUNTS["forward"] += e_count
        if entry["sparsity_mode"] != "none":
            _GEMM_COUNTS["sparse_2to4"] += e_count
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        inputs, offs, *weights = ctx.saved_tensors
        experts: nn.ModuleList = ctx.experts
        proj_name: str = ctx.proj_name
        e_count = len(weights)
        m_total, k = inputs.shape
        n = weights[0].shape[0]
        grad_out_c = grad_output.reshape(m_total, n).contiguous()
        entry = _get_or_build_grouped_nvfp4_cache(experts, proj_name)

        grad_inputs: Tensor | None = None
        if ctx.needs_input_grad[0]:
            if m_total == 0:
                grad_inputs = torch.zeros_like(inputs)
            else:
                ga_bwd = _compute_global_scale_compiled(grad_out_c, entry["one_f32"])
                qg = torch.empty((m_total, n // 2), dtype=torch.uint8, device=inputs.device)
                sg = torch.empty(
                    (m_total, n // 16), dtype=torch.float8_e4m3fn, device=inputs.device
                )
                seed = random.getrandbits(31)
                grid_qg = (triton.cdiv(m_total, 64), n // 128)
                _nvfp4_quantize_1d_kernel[grid_qg](
                    grad_out_c,
                    ga_bwd,
                    qg,
                    sg,
                    m_total,
                    K=n,
                    seed=seed,
                    BLOCK_M=64,
                    BLOCK_K=128,
                    STOCHASTIC=True,
                    UNIT_GLOBAL_SCALE=False,
                    num_warps=4,
                )
                grad_inputs = torch.empty((m_total, k), dtype=inputs.dtype, device=inputs.device)
                grid_dgrad = ((m_total + 127) // 128 + e_count, k // 128)
                _nvfp4_grouped_mm_kernel[grid_dgrad](
                    qg,
                    sg,
                    ga_bwd,
                    entry["B_bwd"],
                    entry["Bs_bwd"],
                    entry["Bg"],
                    offs,
                    grad_inputs,
                    m_total,
                    N=k,
                    K=n,
                    E=e_count,
                    BLOCK_M=128,
                    BLOCK_N=128,
                    BLOCK_K=128,
                    GROUP_M=8,
                    UNIT_A_SCALE=False,
                    num_warps=4,
                    num_stages=3,
                )
                _GEMM_COUNTS["dgrad"] += e_count
                if entry["sparsity_mode"] != "none":
                    _GEMM_COUNTS["sparse_2to4"] += e_count

        need_w_flags = tuple(bool(ctx.needs_input_grad[4 + i]) for i in range(e_count))
        grad_weights: list[Tensor | None] = [None] * e_count
        if any(need_w_flags):
            if m_total == 0:
                for i, flag in enumerate(need_w_flags):
                    if flag:
                        grad_weights[i] = torch.zeros_like(weights[i])
            else:
                req_mask = entry["req_cache"].get(need_w_flags)
                if req_mask is None:
                    req_mask = torch.tensor(
                        [1 if f else 0 for f in need_w_flags],
                        dtype=torch.int32,
                        device=inputs.device,
                    )
                    entry["req_cache"][need_w_flags] = req_mask
                w_dtype = weights[0].dtype
                has_sr = bool(
                    entry["sparsity_mode"] != "none" and float(entry["sr_ste_decay"]) > 0.0
                )
                dw_all = torch.empty((e_count, n, k), dtype=w_dtype, device=inputs.device)
                grid_wgrad = (e_count, n // 128, k // 128)
                _grouped_wgrad_kernel[grid_wgrad](
                    grad_out_c,
                    inputs.contiguous(),
                    entry["sr_ste_stack"],
                    offs,
                    req_mask,
                    dw_all,
                    N=n,
                    K=k,
                    BLOCK_M=64,
                    BLOCK_N=128,
                    BLOCK_K=128,
                    GROUP_N=8,
                    HAS_SR_STE=has_sr,
                    num_warps=4,
                    num_stages=3,
                )
                _GEMM_COUNTS["wgrad"] += sum(need_w_flags)
                for i, flag in enumerate(need_w_flags):
                    if flag:
                        grad_weights[i] = dw_all[i]

        return (grad_inputs, None, None, None, *grad_weights)


def nvfp4_grouped_linear(
    inputs: Tensor,
    experts: nn.ModuleList,
    proj_name: str,
    offs: Tensor,
) -> Tensor:
    """Execute fused multi-expert native Blackwell NVFP4 grouped linear across `experts`."""
    weights = [getattr(e, proj_name).weight for e in experts]
    return _NVFP4GroupedLinear.apply(inputs, offs, experts, proj_name, *weights)
