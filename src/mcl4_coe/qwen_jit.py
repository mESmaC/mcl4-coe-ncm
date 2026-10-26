"""
MCL4-JIT: Just-In-Time NVFP4 Expert Streaming & Mitotic Binary Compiler
for Frontier MoE Models (nvidia/Qwen3.8-Flash-Next-NVFP4, 180B Total / 6B Active)
on NVIDIA Blackwell (SM120, native F.scaled_mm float4_e2m1fn_x2).
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

_E2M1_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def swizzle_nvfp4_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert [M, K//16] float8_e4m3fn block scales to cuBLASLt SWIZZLE_32_4_4 layout."""
    M, K_blocks = scales.shape
    s = scales.view(M // 128, 4, 32, K_blocks // 4, 4)
    return s.permute(0, 3, 2, 1, 4).contiguous()


def pack_fp4_e2m1_x2(x_scaled: torch.Tensor) -> torch.Tensor:
    """Pack float32 tensor in [-6, 6] into float4_e2m1fn_x2 (2 values per byte)."""
    vals = _E2M1_VALS.to(x_scaled.device)
    sign = (x_scaled < 0).to(torch.uint8) << 3
    abs_x = x_scaled.abs().unsqueeze(-1)
    idx = torch.argmin((abs_x - vals).abs(), dim=-1).to(torch.uint8)
    codes = sign | idx
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed = (hi << 4) | lo
    return packed.view(torch.float4_e2m1fn_x2)


def dequantize_nvfp4_weight(
    w_u8: torch.Tensor, s_f8: torch.Tensor, s2_f32: torch.Tensor
) -> torch.Tensor:
    """Dequantize a packed NVFP4 checkpoint tensor into BF16 [Out, In] (used once when spawning a Mitotic Clone)."""
    lut = _E2M1_LUT.to(w_u8.device)
    lo = lut[(w_u8 & 0x0F).long()]
    hi = lut[((w_u8 >> 4) & 0x0F).long()]
    w_fp32 = torch.stack([lo, hi], dim=-1).reshape(w_u8.shape[0], w_u8.shape[1] * 2)
    scales = s_f8.float() * s2_f32.float()
    w_deq = w_fp32.view(w_u8.shape[0], -1, 16) * scales.unsqueeze(-1)
    return w_deq.reshape(w_u8.shape[0], -1).to(torch.bfloat16)


def compile_bf16_to_nvfp4_tensors(
    w_bf16: torch.Tensor, input_scale: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """
    JIT Binary Compiler: Quantize a learned BF16 linear matrix [Out, In] into
    exact nvidia/Qwen3.8-Flash-Next-NVFP4 binary shard tensors.
    """
    w_f32 = w_bf16.detach().float()
    out_features, in_features = w_f32.shape
    w_blocks = w_f32.reshape(out_features, in_features // 16, 16)
    block_amax = w_blocks.abs().amax(dim=-1).clamp(min=1e-12)
    global_amax = block_amax.max()
    weight_scale_2 = (global_amax / (6.0 * 448.0)).clamp(min=1e-12).to(torch.float32)
    raw_block_scales = (block_amax / (6.0 * weight_scale_2)).clamp(max=448.0)
    weight_scale_f8 = raw_block_scales.to(torch.float8_e4m3fn)
    eff_scales = (weight_scale_f8.float() * weight_scale_2).unsqueeze(-1).clamp(min=1e-12)
    w_scaled = (w_blocks / eff_scales).reshape(out_features, in_features)
    weight_u8 = pack_fp4_e2m1_x2(w_scaled).view(torch.uint8)
    if input_scale is None:
        input_scale = torch.tensor(0.01, dtype=torch.float32)
    else:
        input_scale = input_scale.detach().cpu().to(torch.float32).reshape(())
    return {
        "weight": weight_u8.cpu().contiguous(),
        "weight_scale": weight_scale_f8.cpu().contiguous(),
        "weight_scale_2": weight_scale_2.cpu().reshape(()),
        "input_scale": input_scale,
    }


def blackwell_nvfp4_linear(
    x_bf16: torch.Tensor,
    w_u8: torch.Tensor,
    w_scale_swizzled: torch.Tensor,
    w_scale_2: torch.Tensor,
) -> torch.Tensor:
    """
    Hardware NVFP4 matrix multiplication on Blackwell 5th-Gen Tensor Cores via F.scaled_mm.
    """
    orig_shape = x_bf16.shape
    in_features = orig_shape[-1]
    out_features = w_u8.shape[0]
    x_2d = x_bf16.reshape(-1, in_features)
    M = x_2d.shape[0]
    M_pad = max(128, ((M + 127) // 128) * 128)
    if M_pad > M:
        x_2d_pad = F.pad(x_2d, (0, 0, 0, M_pad - M))
    else:
        x_2d_pad = x_2d

    x_f32 = x_2d_pad.float()
    x_blocks = x_f32.reshape(M_pad, in_features // 16, 16)
    x_block_amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-12)
    x_global_amax = x_block_amax.max()
    x_s2 = (x_global_amax / (6.0 * 448.0)).clamp(min=1e-12)
    x_scales_f8 = (x_block_amax / (6.0 * x_s2)).clamp(max=448.0).to(torch.float8_e4m3fn)
    x_eff_scales = (x_scales_f8.float() * x_s2).unsqueeze(-1).clamp(min=1e-12)
    x_packed = pack_fp4_e2m1_x2((x_blocks / x_eff_scales).reshape(M_pad, in_features))
    x_swizzled = swizzle_nvfp4_scales(x_scales_f8)

    w_fp4 = w_u8.view(torch.float4_e2m1fn_x2)
    alpha = (x_s2 * w_scale_2).to(torch.float32)

    out_pad = F.scaled_mm(
        x_packed,
        w_fp4.t(),
        scale_a=x_swizzled,
        scale_recipe_a=F.ScalingType.BlockWise1x16,
        scale_b=w_scale_swizzled,
        scale_recipe_b=F.ScalingType.BlockWise1x16,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )
    out_pad = (out_pad.float() * alpha).to(torch.bfloat16)
    return out_pad[:M].reshape(*orig_shape[:-1], out_features)


@dataclass
class MountedNVFP4Expert:
    """A 2.64 MiB NVFP4 expert mounted in the GPU Working-Set Cache."""
    expert_id: int
    gate_w: torch.Tensor
    gate_swiz: torch.Tensor
    gate_s2: torch.Tensor
    up_w: torch.Tensor
    up_swiz: torch.Tensor
    up_s2: torch.Tensor
    down_w: torch.Tensor
    down_swiz: torch.Tensor
    down_s2: torch.Tensor

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.gate_w, self.gate_swiz, self.gate_s2,
                self.up_w, self.up_swiz, self.up_s2,
                self.down_w, self.down_swiz, self.down_s2,
            )
        )


class PlasticMitoticExpertClone(nn.Module):
    """
    A trainable BF16 Mitotic Expert Clone spawned from a high-traffic NVFP4 expert
    during RL/RSI, before being JIT-compiled back into a static 2.64 MiB .nvfp4 binary shard.
    """
    def __init__(
        self,
        parent_expert_id: int,
        gate_bf16: torch.Tensor,
        up_bf16: torch.Tensor,
        down_bf16: torch.Tensor,
        noise_std: float = 1e-3,
    ):
        super().__init__()
        self.parent_expert_id = parent_expert_id
        with torch.no_grad():
            g_init = gate_bf16.clone() + torch.randn_like(gate_bf16) * noise_std
            u_init = up_bf16.clone() + torch.randn_like(up_bf16) * noise_std
            d_init = down_bf16.clone()
        self.gate_proj = nn.Parameter(g_init)
        self.up_proj = nn.Parameter(u_init)
        self.down_proj = nn.Parameter(d_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, self.gate_proj)) * F.linear(x, self.up_proj), self.down_proj)


class NVFP4JITExpertBank(nn.Module):
    """
    Unloaded NVFP4 Expert Store + GPU Working-Set LRU Cache + Mitotic Binary Compiler.
    - Baseline 512 experts remain unloaded on disk (mmap via safetensors).
    - Active experts are dynamically JIT-mounted into a bounded GPU LRU cache and
      executed via native Blackwell F.scaled_mm (float4_e2m1fn_x2).
    - Supports spawning plastic BF16 Mitotic Clones for RL/RSI and compiling them
      out into standalone .nvfp4 binary shards on disk.
    """
    def __init__(
        self,
        layer_idx: int,
        shard_path: str,
        num_base_experts: int = 512,
        max_cached_experts: int = 64,
        device: torch.device = torch.device("cuda"),
        prefix: str | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.shard_path = shard_path
        self.num_base_experts = num_base_experts
        self.max_cached_experts = max_cached_experts
        self.exec_device = device
        self.prefix = prefix or f"model.language_model.layers.{layer_idx}.mlp.experts"

        # Persistent mmap handle on CPU/NVMe (consumes 0 GPU VRAM)
        self._mmap_handle = safe_open(shard_path, framework="pt", device="cpu")
        self.compiled_shard_paths: dict[int, str] = {}
        self.plastic_clones = nn.ModuleDict()
        self.next_expert_id = num_base_experts

        # Bounded GPU Working-Set LRU Cache
        self.gpu_cache: OrderedDict[int, MountedNVFP4Expert] = OrderedDict()

        # Telemetry counters
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.total_mount_ms = 0.0
        self.total_exec_ms = 0.0

    def reset_telemetry(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.total_mount_ms = 0.0
        self.total_exec_ms = 0.0

    @property
    def cached_vram_bytes(self) -> int:
        return sum(exp.nbytes for exp in self.gpu_cache.values())

    def _read_raw_nvfp4_from_disk(self, expert_id: int) -> dict[str, torch.Tensor]:
        if expert_id < self.num_base_experts:
            f = self._mmap_handle
            base = f"{self.prefix}.{expert_id}"
            out = {}
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                    out[f"{proj}.{suffix}"] = f.get_tensor(f"{base}.{proj}.{suffix}")
            return out
        elif expert_id in self.compiled_shard_paths:
            with safe_open(self.compiled_shard_paths[expert_id], framework="pt", device="cpu") as f:
                return {k: f.get_tensor(k) for k in f.keys()}
        else:
            raise KeyError(f"Expert {expert_id} not found in base shard or compiled mitotic shards.")

    def mount_expert(self, expert_id: int) -> MountedNVFP4Expert:
        """JIT-mount a 2.64 MiB NVFP4 expert from mmap/NVMe into the GPU Working-Set LRU Cache."""
        if expert_id in self.gpu_cache:
            self.cache_hits += 1
            self.gpu_cache.move_to_end(expert_id)
            return self.gpu_cache[expert_id]

        t0 = time.perf_counter()
        self.cache_misses += 1
        raw = self._read_raw_nvfp4_from_disk(expert_id)
        dev = self.exec_device

        g_w = raw["gate_proj.weight"].to(dev, non_blocking=True)
        g_swiz = swizzle_nvfp4_scales(raw["gate_proj.weight_scale"].to(dev, non_blocking=True))
        g_s2 = raw["gate_proj.weight_scale_2"].to(dev, non_blocking=True)

        u_w = raw["up_proj.weight"].to(dev, non_blocking=True)
        u_swiz = swizzle_nvfp4_scales(raw["up_proj.weight_scale"].to(dev, non_blocking=True))
        u_s2 = raw["up_proj.weight_scale_2"].to(dev, non_blocking=True)

        d_w = raw["down_proj.weight"].to(dev, non_blocking=True)
        d_swiz = swizzle_nvfp4_scales(raw["down_proj.weight_scale"].to(dev, non_blocking=True))
        d_s2 = raw["down_proj.weight_scale_2"].to(dev, non_blocking=True)

        mounted = MountedNVFP4Expert(
            expert_id=expert_id,
            gate_w=g_w, gate_swiz=g_swiz, gate_s2=g_s2,
            up_w=u_w, up_swiz=u_swiz, up_s2=u_s2,
            down_w=d_w, down_swiz=d_swiz, down_s2=d_s2,
        )
        while len(self.gpu_cache) >= self.max_cached_experts:
            self.gpu_cache.popitem(last=False)
            self.evictions += 1
        self.gpu_cache[expert_id] = mounted
        self.total_mount_ms += (time.perf_counter() - t0) * 1000.0
        return mounted

    def exec_mounted_nvfp4_expert(self, x: torch.Tensor, mounted: MountedNVFP4Expert) -> torch.Tensor:
        gate = blackwell_nvfp4_linear(x, mounted.gate_w, mounted.gate_swiz, mounted.gate_s2)
        up = blackwell_nvfp4_linear(x, mounted.up_w, mounted.up_swiz, mounted.up_s2)
        hidden = F.silu(gate) * up
        return blackwell_nvfp4_linear(hidden, mounted.down_w, mounted.down_swiz, mounted.down_s2)

    def spawn_mitotic_clone(self, parent_expert_id: int, noise_std: float = 1e-3) -> int:
        """
        Spawn a plastic BF16 Mitotic Clone in GPU VRAM from a frozen NVFP4 expert.
        Returns the new expert_id (>= 512).
        """
        raw = self._read_raw_nvfp4_from_disk(parent_expert_id)
        dev = self.exec_device
        g_bf16 = dequantize_nvfp4_weight(
            raw["gate_proj.weight"].to(dev),
            raw["gate_proj.weight_scale"].to(dev),
            raw["gate_proj.weight_scale_2"].to(dev),
        )
        u_bf16 = dequantize_nvfp4_weight(
            raw["up_proj.weight"].to(dev),
            raw["up_proj.weight_scale"].to(dev),
            raw["up_proj.weight_scale_2"].to(dev),
        )
        d_bf16 = dequantize_nvfp4_weight(
            raw["down_proj.weight"].to(dev),
            raw["down_proj.weight_scale"].to(dev),
            raw["down_proj.weight_scale_2"].to(dev),
        )
        new_id = self.next_expert_id
        self.next_expert_id += 1
        self.plastic_clones[str(new_id)] = PlasticMitoticExpertClone(
            parent_expert_id=parent_expert_id,
            gate_bf16=g_bf16,
            up_bf16=u_bf16,
            down_bf16=d_bf16,
            noise_std=noise_std,
        )
        return new_id

    def compile_mitotic_clone_to_disk(self, expert_id: int, output_dir: str) -> str:
        """
        JIT Binary Compiler:
        Compiles a plastic BF16 Mitotic Clone into a standalone 2.64 MiB .nvfp4.safetensors
        binary shard on disk, unloads the BF16 parameters from VRAM, and registers the
        binary shard for future JIT streaming via F.scaled_mm.
        """
        key = str(expert_id)
        if key not in self.plastic_clones:
            raise KeyError(f"Plastic mitotic clone {expert_id} not found.")
        clone: PlasticMitoticExpertClone = self.plastic_clones[key]

        os.makedirs(output_dir, exist_ok=True)
        shard_file = os.path.join(output_dir, f"layer_{self.layer_idx:02d}_expert_{expert_id:04d}_mitotic.nvfp4.safetensors")

        tensors: dict[str, torch.Tensor] = {}
        for proj_name, param in (
            ("gate_proj", clone.gate_proj),
            ("up_proj", clone.up_proj),
            ("down_proj", clone.down_proj),
        ):
            compiled = compile_bf16_to_nvfp4_tensors(param)
            for k, v in compiled.items():
                tensors[f"{proj_name}.{k}"] = v

        save_file(tensors, shard_file)
        self.compiled_shard_paths[expert_id] = shard_file

        # Evict plastic BF16 clone from GPU VRAM!
        del self.plastic_clones[key]
        return shard_file

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        t_exec0 = time.perf_counter()
        final_hidden = torch.zeros_like(hidden_states)
        unique_experts = [int(x) for x in torch.unique(selected_experts).tolist()]
        # Cache-Resident-First (CRF) schedule: execute experts already mounted in GPU VRAM first
        # so ascending-index scans never evict resident experts before they are consumed.
        plastic_ids = [e for e in unique_experts if str(e) in self.plastic_clones]
        cached_ids = [e for e in unique_experts if (str(e) not in self.plastic_clones and e in self.gpu_cache)]
        unmounted_ids = [e for e in unique_experts if (str(e) not in self.plastic_clones and e not in self.gpu_cache)]
        ordered_experts = plastic_ids + cached_ids + unmounted_ids

        for exp_id_int in ordered_experts:
            mask = (selected_experts == exp_id_int)
            token_idx, topk_pos = torch.where(mask)
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states[token_idx]
            weights = routing_weights[token_idx, topk_pos].unsqueeze(-1)

            if str(exp_id_int) in self.plastic_clones:
                clone_out = self.plastic_clones[str(exp_id_int)](current_state)
                weighted_out = (clone_out * weights).to(final_hidden.dtype)
                final_hidden = final_hidden.index_add(0, token_idx, weighted_out)
            else:
                with torch.no_grad():
                    mounted = self.mount_expert(exp_id_int)
                    nvfp4_out = self.exec_mounted_nvfp4_expert(current_state.detach(), mounted)
                weighted_out = (nvfp4_out.to(final_hidden.dtype) * weights)
                final_hidden = final_hidden.index_add(0, token_idx, weighted_out)

        self.total_exec_ms += (time.perf_counter() - t_exec0) * 1000.0
        return final_hidden


class Qwen4ExpRMSNorm(nn.Module):
    """Exact Qwen4ExpTextRMSNorm: output * (1.0 + weight.float())."""
    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
        self.group_size = group_size

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen4ExpRMSNormGated(nn.Module):
    """Exact Qwen4ExpTextRMSNormGated for GatedDeltaNet output normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h_f32 = hidden_states.to(torch.float32)
        variance = h_f32.pow(2).mean(-1, keepdim=True)
        h_norm = h_f32 * torch.rsqrt(variance + self.variance_epsilon)
        h_scaled = self.weight * h_norm.to(input_dtype)
        h_gated = h_scaled * F.silu(gate.to(torch.float32))
        return h_gated.to(input_dtype)


class Qwen4ExpGatedResidual(nn.Module):
    """Exact Qwen4ExpTextGatedResidual (4-stream Hyper-Connection mixer + block injector)."""
    def __init__(
        self,
        hidden_size: int = 2560,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        eps: float = 1e-6,
        use_combine: bool = True,
    ):
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        hc_hidden_size = hc_count * hidden_size
        self.hc_norm = Qwen4ExpRMSNorm(hc_hidden_size, group_size=hidden_size, eps=eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, hc_lowrank, bias=False, dtype=torch.bfloat16)
        self.input_mix_weight_up = nn.Linear(hc_lowrank, hc_hidden_size, bias=False, dtype=torch.bfloat16)
        self.block_inject_weight = (
            nn.Linear(hc_hidden_size, hc_count, bias=False, dtype=torch.bfloat16) if use_combine else None
        )

    def forward(self, hyper_input: torch.Tensor):
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (
            input_mix_weight * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2.0 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
        return mixed_input, hyper_input, injection_weights


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Exact Qwen4Exp torch_chunk_gated_delta_rule in FP32 with QK L2-norm."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
    decay = g

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    query = _l2norm(query, dim=-1, eps=1e-6) * (query.shape[-1] ** -0.5)
    key = _l2norm(key, dim=-1, eps=1e-6)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size > 0:
        query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
        beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise_decay = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(strictly_upper_mask, float("-inf")).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)

    last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    core_attn_out = torch.zeros_like(new_values)

    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    core_attn_out = core_attn_out.reshape(padded_output_shape)[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)


class Qwen4ExpGatedDeltaNet(nn.Module):
    """Exact Qwen4ExpTextGatedDeltaNet linear attention block (used in 36 of 48 layers)."""
    def __init__(self, layer_idx: int, hidden_size: int = 2560, eps: float = 1e-6):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_v_heads = 48
        self.num_k_heads = 16
        self.head_k_dim = 128
        self.head_v_dim = 128
        self.key_dim = 2048
        self.value_dim = 6144
        self.conv_dim = 10240
        self.conv_kernel_size = 4

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            dtype=torch.bfloat16,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=torch.bfloat16))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        self.norm = Qwen4ExpRMSNormGated(self.head_v_dim, eps=eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False, dtype=torch.bfloat16)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False, dtype=torch.bfloat16)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False, dtype=torch.bfloat16)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False, dtype=torch.bfloat16)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        repeats = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)

        core_attn_out = chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
):
    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos_u.shape[-1]
    q_rope, q_nope = q[..., :rotary_dim], q[..., rotary_dim:]
    q_rope = (q_rope * cos_u) + (_rotate_half(q_rope) * sin_u)
    q_rotated = torch.cat([q_rope, q_nope], dim=-1)
    if k is not None:
        k_rope, k_nope = k[..., :rotary_dim], k[..., rotary_dim:]
        k_rope = (k_rope * cos_u) + (_rotate_half(k_rope) * sin_u)
        k_rotated = torch.cat([k_rope, k_nope], dim=-1)
        return q_rotated, k_rotated
    return q_rotated


class Qwen4ExpRotaryEmbedding(nn.Module):
    """Exact Qwen4ExpTextRotaryEmbedding with partial_rotary_factor=0.25 and mrope_section=[11, 11, 10]."""
    def __init__(self, head_dim: int = 256, rope_theta: float = 10_000_000.0, partial_rotary_factor: float = 0.25):
        super().__init__()
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = [11, 11, 10]

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32).view(1, 1, -1).expand(3, batch_size, -1)
        freqs = pos[..., None] * self.inv_freq.to(x.device)
        cos = self._recompose(freqs.cos()).to(x.dtype)
        sin = self._recompose(freqs.sin()).to(x.dtype)
        return cos, sin

    def _recompose(self, freq: torch.Tensor) -> torch.Tensor:
        freqs_thw = freq[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)


class Qwen4ExpQSAIndexer(nn.Module):
    """Exact Qwen4ExpTextQSAIndexer for Qwen Sparse Attention (QSA)."""
    def __init__(self, layer_idx: int, hidden_size: int = 2560, eps: float = 1e-6):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = 4
        self.index_kv_heads = 1
        self.index_head_dim = 128
        self.token_budget = 2048
        self.compress_ratio = 16
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.q_layernorm = Qwen4ExpRMSNorm(self.index_head_dim, eps=eps)
        self.k_layernorm = Qwen4ExpRMSNorm(self.index_head_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        causal_bool_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
        full_cos, full_sin = position_embeddings

        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )
        q = self.q_layernorm(q.reshape(*hidden_shape))
        raw_keys = token_k.reshape(*hidden_shape).squeeze(2)
        q = apply_rotary_pos_emb(q, cos=full_cos, sin=full_sin, unsqueeze_dim=2)

        selected_token_indices = torch.full(
            (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
            -1,
            dtype=torch.int64,
            device=hidden_states.device,
        )
        for batch_idx in range(batch_size):
            for query_idx in range(seq_length):
                local_visible = torch.nonzero(causal_bool_mask[batch_idx, 0, query_idx], as_tuple=False).flatten()
                num_blocks = local_visible.shape[-1] // self.compress_ratio
                if num_blocks > 0:
                    block_tok = local_visible[: num_blocks * self.compress_ratio].view(num_blocks, self.compress_ratio)
                    key_groups = raw_keys[batch_idx].index_select(0, block_tok.flatten()).view(
                        num_blocks, self.compress_ratio, self.index_head_dim
                    )
                    pooled_keys = self.k_layernorm(key_groups.float().mean(dim=1).to(raw_keys.dtype))
                    group_starts = block_tok[:, 0]
                    block_keys = apply_rotary_pos_emb(
                        pooled_keys.unsqueeze(1),
                        cos=full_cos[batch_idx].index_select(0, group_starts),
                        sin=full_sin[batch_idx].index_select(0, group_starts),
                    ).squeeze(1)
                    scores = torch.matmul(q[batch_idx, query_idx].float(), block_keys.float().transpose(-1, -2)).transpose(-1, -2)
                    scores = F.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)
                    sel_blocks = scores.topk(min(self.block_topk, num_blocks), dim=0).indices
                    sel_tokens = block_tok.index_select(0, sel_blocks).flatten()
                else:
                    sel_tokens = torch.tensor([], dtype=torch.int64, device=hidden_states.device)
                tail = local_visible[num_blocks * self.compress_ratio :]
                sel_tokens = torch.cat([sel_tokens, tail])
                selected_token_indices[batch_idx, query_idx, : sel_tokens.numel()] = sel_tokens

        selected_mask = torch.zeros(
            (batch_size, seq_length, seq_length + 1), device=hidden_states.device, dtype=torch.bool
        )
        scatter_idx = torch.where(selected_token_indices >= 0, selected_token_indices, seq_length)
        selected_mask = selected_mask.scatter(-1, scatter_idx, True)[..., :seq_length].unsqueeze(1)
        return selected_mask


class Qwen4ExpAttention(nn.Module):
    """Exact Qwen4ExpTextAttention (QSA full/indexed attention used in 12 of 48 layers, including Layer 47)."""
    def __init__(self, layer_idx: int, hidden_size: int = 2560, eps: float = 1e-6):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = 24
        self.num_kv_heads = 2
        self.head_dim = 256
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim * 2, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False, dtype=torch.bfloat16)
        self.q_norm = Qwen4ExpRMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen4ExpRMSNorm(self.head_dim, eps=eps)
        self.indexer = Qwen4ExpQSAIndexer(layer_idx, hidden_size=hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        causal_bool = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device)).view(
            1, 1, seq_len, seq_len
        ).expand(batch_size, 1, seq_len, seq_len)

        selected_mask = self.indexer(hidden_states, position_embeddings, causal_bool)
        combined_bool = causal_bool & selected_mask
        float_mask = torch.where(
            combined_bool,
            torch.zeros((), dtype=hidden_states.dtype, device=hidden_states.device),
            torch.tensor(torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype, device=hidden_states.device),
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling + float_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1) * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen4ExpJITMoEBlock(nn.Module):
    """
    Hybrid Static/Plastic MoE Block for MCL4-JIT:
    - Shared Expert (BF16) + Shared Gate (BF16) pinned in GPU Core VRAM.
    - TopK Router (BF16, top_k=10 of 512 + N_mitotic) pinned in GPU Core VRAM,
      with extensible plastic router rows for Mitotic Clones.
    - NVFP4JITExpertBank: 512 baseline NVFP4 experts unloaded on disk (mmap) +
      GPU Working-Set LRU Cache + Native Blackwell F.scaled_mm execution.
    """
    def __init__(
        self,
        layer_idx: int,
        shard_path: str,
        hidden_size: int = 2560,
        moe_intermediate_size: int = 640,
        num_base_experts: int = 512,
        top_k: int = 10,
        max_cached_experts: int = 64,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_base_experts = num_base_experts
        self.top_k = top_k

        self.gate_weight = nn.Parameter(
            torch.zeros(num_base_experts, hidden_size, dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )
        self.plastic_router_weight: nn.Parameter | None = None

        self.shared_gate_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False, dtype=torch.bfloat16, device=device)
        self.shared_up_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False, dtype=torch.bfloat16, device=device)
        self.shared_down_proj = nn.Linear(moe_intermediate_size, hidden_size, bias=False, dtype=torch.bfloat16, device=device)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False, dtype=torch.bfloat16, device=device)

        self.expert_bank = NVFP4JITExpertBank(
            layer_idx=layer_idx,
            shard_path=shard_path,
            num_base_experts=num_base_experts,
            max_cached_experts=max_cached_experts,
            device=device,
        )

    def spawn_mitotic_expert(self, parent_expert_id: int, noise_std: float = 1e-3, router_boost: float = 0.05) -> int:
        new_id = self.expert_bank.spawn_mitotic_clone(parent_expert_id=parent_expert_id, noise_std=noise_std)
        with torch.no_grad():
            parent_row = self.gate_weight[parent_expert_id : parent_expert_id + 1].clone()
            new_row = parent_row * (1.0 + router_boost) + torch.randn_like(parent_row) * (noise_std * 0.1)
            if self.plastic_router_weight is None:
                self.plastic_router_weight = nn.Parameter(new_row)
            else:
                combined = torch.cat([self.plastic_router_weight.detach(), new_row], dim=0)
                self.plastic_router_weight = nn.Parameter(combined)
        return new_id

    def compile_all_mitotic_clones(self, output_dir: str) -> list[tuple[int, str]]:
        compiled_records = []
        active_ids = sorted(int(k) for k in self.expert_bank.plastic_clones.keys())
        for exp_id in active_ids:
            shard_file = self.expert_bank.compile_mitotic_clone_to_disk(exp_id, output_dir=output_dir)
            compiled_records.append((exp_id, shard_file))

        if self.plastic_router_weight is not None:
            with torch.no_grad():
                merged_gate = torch.cat([self.gate_weight.data, self.plastic_router_weight.detach()], dim=0)
                self.gate_weight = nn.Parameter(merged_gate, requires_grad=False)
            self.plastic_router_weight = None
        return compiled_records

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x_flat = hidden_states.view(-1, hidden_dim)

        shared_out = self.shared_down_proj(F.silu(self.shared_gate_proj(x_flat)) * self.shared_up_proj(x_flat))
        shared_out = torch.sigmoid(self.shared_expert_gate(x_flat)) * shared_out

        if self.plastic_router_weight is not None:
            full_gate_weight = torch.cat([self.gate_weight, self.plastic_router_weight], dim=0)
        else:
            full_gate_weight = self.gate_weight

        router_logits = F.linear(x_flat, full_gate_weight)
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        top_vals, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        top_vals = (top_vals / top_vals.sum(dim=-1, keepdim=True).clamp(min=1e-12)).to(hidden_states.dtype)

        routed_out = self.expert_bank(x_flat, selected_experts, top_vals)

        total_out = (routed_out + shared_out).view(batch_size, seq_len, hidden_dim)
        return total_out, router_logits


class Qwen4ExpJITDecoderLayer(nn.Module):
    """Complete Qwen4Exp Decoder Layer with Static Core + JIT NVFP4 Expert Bank."""
    def __init__(
        self,
        layer_idx: int,
        layer_type: str,
        shard_path: str,
        hidden_size: int = 2560,
        max_cached_experts: int = 64,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.attn_hyper_connection = Qwen4ExpGatedResidual(hidden_size=hidden_size, use_combine=True).to(device)
        if layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpGatedDeltaNet(layer_idx=layer_idx, hidden_size=hidden_size).to(device)
            self.self_attn = None
        else:
            self.self_attn = Qwen4ExpAttention(layer_idx=layer_idx, hidden_size=hidden_size).to(device)
            self.linear_attn = None
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(hidden_size=hidden_size, use_combine=True).to(device)
        self.mlp = Qwen4ExpJITMoEBlock(
            layer_idx=layer_idx,
            shard_path=shard_path,
            hidden_size=hidden_size,
            max_cached_experts=max_cached_experts,
            device=device,
        )

    def load_core_weights_from_safetensors(self, shard_path: str) -> int:
        prefix = f"model.language_model.layers.{self.layer_idx}."
        loaded_bytes = 0
        with safe_open(shard_path, framework="pt", device="cuda") as f:
            state_map = {
                "attn_hyper_connection.hc_norm.weight": self.attn_hyper_connection.hc_norm.weight,
                "attn_hyper_connection.input_mix_weight_down.weight": self.attn_hyper_connection.input_mix_weight_down.weight,
                "attn_hyper_connection.input_mix_weight_up.weight": self.attn_hyper_connection.input_mix_weight_up.weight,
                "attn_hyper_connection.block_inject_weight.weight": self.attn_hyper_connection.block_inject_weight.weight,
                "mlp_hyper_connection.hc_norm.weight": self.mlp_hyper_connection.hc_norm.weight,
                "mlp_hyper_connection.input_mix_weight_down.weight": self.mlp_hyper_connection.input_mix_weight_down.weight,
                "mlp_hyper_connection.input_mix_weight_up.weight": self.mlp_hyper_connection.input_mix_weight_up.weight,
                "mlp_hyper_connection.block_inject_weight.weight": self.mlp_hyper_connection.block_inject_weight.weight,
                "mlp.gate.weight": self.mlp.gate_weight,
                "mlp.shared_expert.gate_proj.weight": self.mlp.shared_gate_proj.weight,
                "mlp.shared_expert.up_proj.weight": self.mlp.shared_up_proj.weight,
                "mlp.shared_expert.down_proj.weight": self.mlp.shared_down_proj.weight,
                "mlp.shared_expert_gate.weight": self.mlp.shared_expert_gate.weight,
            }
            if self.layer_type == "linear_attention":
                state_map.update({
                    "linear_attn.A_log": self.linear_attn.A_log,
                    "linear_attn.conv1d.weight": self.linear_attn.conv1d.weight,
                    "linear_attn.dt_bias": self.linear_attn.dt_bias,
                    "linear_attn.in_proj_a.weight": self.linear_attn.in_proj_a.weight,
                    "linear_attn.in_proj_b.weight": self.linear_attn.in_proj_b.weight,
                    "linear_attn.in_proj_qkv.weight": self.linear_attn.in_proj_qkv.weight,
                    "linear_attn.in_proj_z.weight": self.linear_attn.in_proj_z.weight,
                    "linear_attn.norm.weight": self.linear_attn.norm.weight,
                    "linear_attn.out_proj.weight": self.linear_attn.out_proj.weight,
                })
            else:
                state_map.update({
                    "self_attn.q_proj.weight": self.self_attn.q_proj.weight,
                    "self_attn.k_proj.weight": self.self_attn.k_proj.weight,
                    "self_attn.v_proj.weight": self.self_attn.v_proj.weight,
                    "self_attn.o_proj.weight": self.self_attn.o_proj.weight,
                    "self_attn.q_norm.weight": self.self_attn.q_norm.weight,
                    "self_attn.k_norm.weight": self.self_attn.k_norm.weight,
                    "self_attn.indexer.index_qk_proj.weight": self.self_attn.indexer.index_qk_proj.weight,
                    "self_attn.indexer.q_layernorm.weight": self.self_attn.indexer.q_layernorm.weight,
                    "self_attn.indexer.k_layernorm.weight": self.self_attn.indexer.k_layernorm.weight,
                })
            with torch.no_grad():
                for rel_key, param in state_map.items():
                    t = f.get_tensor(prefix + rel_key)
                    param.copy_(t.to(param.dtype))
                    loaded_bytes += param.numel() * param.element_size()
        return loaded_bytes

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mixed_attn_in, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(mixed_attn_in)
        else:
            attn_out = self.self_attn(mixed_attn_in, position_embeddings)

        injection = attn_out.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)

        mixed_mlp_in, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        mlp_out, router_logits = self.mlp(mixed_mlp_in)

        injection = mlp_out.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)
        return hidden_states, router_logits


class NeuralCodebookMemory2560(nn.Module):
    """
    Factorized Neural Codebook Memory (NCM) scaled to Qwen3.8-Flash-Next (d_model=2560):
    4 subspaces x 64 codes (each code dim = 640), providing a compact differentiable
    semantic state cache across recurrent CoE ticks.
    """
    def __init__(self, hidden_size: int = 2560, num_subspaces: int = 4, num_codes: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_subspaces = num_subspaces
        self.num_codes = num_codes
        self.sub_dim = hidden_size // num_subspaces

        self.codebooks = nn.Parameter(
            torch.randn(num_subspaces, num_codes, self.sub_dim, dtype=torch.bfloat16) * 0.02
        )
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.read_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q = self.query_proj(x).view(B, T, self.num_subspaces, self.sub_dim)
        q_norm = F.normalize(q.float(), dim=-1)
        cb_norm = F.normalize(self.codebooks.float(), dim=-1)

        sim = torch.einsum("btsd,skd->btsk", q_norm, cb_norm) * 10.0
        probs = F.softmax(sim, dim=-1)
        entropy = -(probs * (probs.clamp(min=1e-9)).log()).sum(dim=-1).mean()

        retrieved = torch.einsum("btsk,skd->btsd", probs.to(x.dtype), self.codebooks).reshape(B, T, D)
        gated = x + torch.sigmoid(self.read_gate) * self.out_proj(retrieved)
        return gated, entropy


class MCL4CoENCMReasoningGraft(nn.Module):
    """
    Plastic MCL4-COE + NCM Reasoning Module grafted onto the top Qwen4Exp representation (d_model=2560):
    - 5 Typed Cognitive Operators: Deduction, Induction, Evaluation, Metacognition, Decision
    - Recurrent Chain-of-Experts loop (R=4 ticks) with NCM semantic memory read/write
    - Residual anchor blending to preserve base frontier representations while enabling RL/RSI plasticity
    """
    OPERATOR_NAMES = ("deduction", "induction", "evaluation", "metacognition", "decision")

    def __init__(self, hidden_size: int = 2560, rank: int = 320, num_ticks: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_ticks = num_ticks
        self.num_ops = len(self.OPERATOR_NAMES)

        self.norm = Qwen4ExpRMSNorm(hidden_size)
        self.ncm = NeuralCodebookMemory2560(hidden_size=hidden_size)
        self.op_router = nn.Linear(hidden_size, self.num_ops, bias=False, dtype=torch.bfloat16)

        self.op_down = nn.Parameter(torch.randn(self.num_ops, rank, hidden_size, dtype=torch.bfloat16) * 0.02)
        self.op_up = nn.Parameter(torch.zeros(self.num_ops, hidden_size, rank, dtype=torch.bfloat16))
        self.meta_halt_head = nn.Linear(hidden_size, 1, bias=True, dtype=torch.bfloat16)
        self.anchor_alpha = nn.Parameter(torch.tensor(0.15, dtype=torch.bfloat16))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        anchor = hidden_states
        h = hidden_states
        op_probs_per_tick = []
        ncm_entropies = []

        for _ in range(self.num_ticks):
            h_norm = self.norm(h)
            h_mem, ncm_ent = self.ncm(h_norm)
            ncm_entropies.append(ncm_ent)

            op_logits = self.op_router(h_mem)
            op_probs = F.softmax(op_logits, dtype=torch.float32, dim=-1).to(h.dtype)
            op_probs_per_tick.append(op_probs)

            low = torch.einsum("btd,ord->btor", h_mem, self.op_down)
            act = F.silu(low)
            op_outs = torch.einsum("btor,odr->btod", act, self.op_up)
            combined_op = (op_outs * op_probs.unsqueeze(-1)).sum(dim=-2)

            halt_conf = torch.sigmoid(self.meta_halt_head(h_mem))
            h = h + (1.0 - 0.5 * halt_conf) * combined_op

        alpha = torch.clamp(self.anchor_alpha, 0.01, 0.50)
        final_h = (1.0 - alpha) * anchor + alpha * h
        mean_op_probs = torch.stack(op_probs_per_tick, dim=0).mean(dim=0)
        telemetry = {
            "mean_op_probs": mean_op_probs,
            "ncm_entropy": torch.stack(ncm_entropies).mean(),
        }
        return final_h, telemetry


class Qwen4ExpMCL4JITEngine(nn.Module):
    """
    Full MCL4-JIT Engine for nvidia/Qwen3.8-Flash-Next-NVFP4:
    1. Static BF16 Core pinned in GPU VRAM (embed_tokens, Hyper-Connections, GatedDeltaNet,
       QwenSparseAttention, SharedExperts, Routers, hyper_connection_mixer, lm_head).
    2. Unloaded NVFP4 Routed Experts on disk (mmap) + GPU Working-Set LRU Cache +
       Blackwell F.scaled_mm (float4_e2m1fn_x2) execution.
    3. Static Stem Execution (torch.no_grad() on lower layers) + Plastic Top Layer
       (Mitotic Expert Clones + MCL4-COE + NCM) for fast RL/RSI updates and
       JIT compilation of graduated Mitotic Clones into .nvfp4 binary shards on disk.
    """
    def __init__(
        self,
        layer_configs: list[tuple[int, str, str]],
        s9_shard_path: str,
        hidden_size: int = 2560,
        vocab_size: int = 248320,
        max_cached_experts_per_layer: int = 64,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.exec_device = device

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16, device=device)
        self.embed_tokens.weight.requires_grad_(False)

        self.rotary_emb = Qwen4ExpRotaryEmbedding(head_dim=256).to(device)
        self.layers = nn.ModuleList()
        for layer_idx, layer_type, shard_path in layer_configs:
            layer = Qwen4ExpJITDecoderLayer(
                layer_idx=layer_idx,
                layer_type=layer_type,
                shard_path=shard_path,
                hidden_size=hidden_size,
                max_cached_experts=max_cached_experts_per_layer,
                device=device,
            )
            layer.load_core_weights_from_safetensors(shard_path)
            for p in layer.parameters():
                p.requires_grad_(False)
            self.layers.append(layer)

        self.hyper_connection_mixer = Qwen4ExpGatedResidual(hidden_size=hidden_size, use_combine=False).to(device)
        for p in self.hyper_connection_mixer.parameters():
            p.requires_grad_(False)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.bfloat16, device=device)
        self.lm_head.weight.requires_grad_(False)

        self._load_top_level_weights(s9_shard_path)

        self.coe_graft = MCL4CoENCMReasoningGraft(hidden_size=hidden_size, rank=320, num_ticks=4).to(device)

    def _load_top_level_weights(self, s9_shard_path: str) -> None:
        with safe_open(s9_shard_path, framework="pt", device="cuda") as f:
            with torch.no_grad():
                self.embed_tokens.weight.copy_(f.get_tensor("model.language_model.embed_tokens.weight"))
                self.lm_head.weight.copy_(f.get_tensor("lm_head.weight"))
                self.hyper_connection_mixer.hc_norm.weight.copy_(
                    f.get_tensor("model.language_model.hyper_connection_mixer.hc_norm.weight")
                )
                self.hyper_connection_mixer.input_mix_weight_down.weight.copy_(
                    f.get_tensor("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight")
                )
                self.hyper_connection_mixer.input_mix_weight_up.weight.copy_(
                    f.get_tensor("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight")
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        plastic_top_layers: int = 1,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T = input_ids.shape
        num_layers = len(self.layers)
        stem_cutoff = max(0, num_layers - plastic_top_layers)

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
            pos_emb = self.rotary_emb(inputs_embeds, T)
            hidden_states = inputs_embeds.repeat(1, 1, 4)

            for idx in range(stem_cutoff):
                hidden_states, _ = self.layers[idx](hidden_states, pos_emb)

        hidden_states = hidden_states.detach().requires_grad_(True)
        top_router_logits = []
        for idx in range(stem_cutoff, num_layers):
            hidden_states, r_logits = self.layers[idx](hidden_states, pos_emb)
            top_router_logits.append(r_logits)

        mixed_final = self.hyper_connection_mixer(hidden_states)
        coe_hidden, coe_telemetry = self.coe_graft(mixed_final)
        logits = self.lm_head(coe_hidden)

        coe_telemetry["top_router_logits"] = top_router_logits
        return logits, coe_telemetry
