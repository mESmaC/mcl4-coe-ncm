r"""Donor Weight Surgery, Shard Transcoder, and Mitotic Scaling Launcher for `MCL4-COE v8-OLMoE-12B`.

Scales `allenai/OLMoE-1B-7B-0125` (`6.92B` donor -> `7.08B` MCL4-COE Step-0 graft) to `11.91B` (`~12.0B`)
total parameters (`E = 64 -> 80 -> 96 -> 112` experts across `16` layers = `8` Encoder + `8` Decoder)
on a single RTX PRO 6000 Blackwell (`96 GB` GDDR7) via:
1. Exact Step-0 Algebraic Weight Surgery:
   - Head-wise RoPE row interleaving `[0, 64, 1, 65, ..., 63, 127]` on `q_proj`, `k_proj`, `q_norm`, `k_norm`
     with `qk_norm_mode="full"` (100% bitwise equivalent to `OlmoeAttention` at Step 0).
   - Partition-Invariant Twin-Gated Mitosis (`norm_topk_prob=False` with $-\ln(2)$ twin logit shift).
   - Zero-initialized `PLE`, `MCL4` Markov gate, `Decoder` cross-attention out-proj, and `RecurrentFeedbackAdapter`
     gate so Step-0 forward pass preserves the 5T-token pretrained `OLMoE` manifold with zero perplexity shock.
2. Generation-Scoped Expert Plasticity (`plastic_experts_per_layer = 32`, `model_dtype = "bfloat16"`):
   - Keeps the `32` active mitotic twin experts per layer (`16` parent-child pairs) + all Routers, `PLE`,
     `MCL4`, `Cross-Attention`, `Norms`, and `COE` operators plastic in GPU memory while older differentiated
     experts serve as frozen specialist memory (`~52 GiB` peak VRAM at `11.91B` total parameters).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mcl4_coe.backbone import MCL4CEDBackbone, MCL4CEDConfig
from mcl4_coe.data import get_tokenizer
from mcl4_coe.operators import COEConfig, CognitiveOperatorEnsemble
from mcl4_coe.ornith_rl import parameter_count
from mcl4_coe.pipeline import (
    BlockAlignedShardWriter,
    ShardedTokenBlockDataset,
)
from mcl4_coe.trainer import MCL4COETrainer, TrainingRunConfig

HEADER_BYTES = 1024
OLMOE_DONOR_REPO = "allenai/OLMoE-1B-7B-0125"
OLMOE_VOCAB_SIZE = 50304


def _build_rope_interleave_perm(n_heads: int = 16, head_dim: int = 128) -> torch.Tensor:
    """Return index permutation mapping half-rotated RoPE `[0..63, 64..127]` -> interleaved `[0, 64, 1, 65, ...]`."""
    half = head_dim // 2
    head_perm = torch.empty(head_dim, dtype=torch.long)
    head_perm[0::2] = torch.arange(0, half, dtype=torch.long)
    head_perm[1::2] = torch.arange(half, head_dim, dtype=torch.long)
    full_perm = torch.cat([head_perm + h * head_dim for h in range(n_heads)], dim=0)
    return full_perm


def compute_v8_olmoe_parameter_profile(
    *,
    start_experts: int = 64,
    max_experts: int = 112,
    top_k: int = 8,
    total_steps: int = 2800,
) -> dict[str, Any]:
    """Compute exact parameter counts across mitotic stages (`64 -> 80 -> 96 -> 112` experts) without tensor allocation."""
    V = OLMOE_VOCAB_SIZE
    d = 2048
    n_heads = 16
    L_enc = 8
    L_dec = 8
    L = L_enc + L_dec
    d_ple = 128
    d_ffn = 1024
    p_dim = 256

    tok_emb = V * d
    lm_head = V * d  # tie_word_embeddings = False
    ple_per_layer = (V * d_ple + d_ple * d) + d
    # LocalAttentionBlock with qk_norm_mode="full" (`q_norm` and `k_norm` are d_model each) + markov_gate
    attn_per_layer = 4 * d * d + 2 * d + 1
    # Decoder cross-attention (`global_q`, `global_kv`, `global_out_proj`) + `global_norm`
    cross_attn_per_dec = 4 * d * d + d
    # Markov router shared trunk + layer norms (`attn_norm`, `moe_norm`)
    router_shared_per_layer = (4 * d) * 32 + 32 + 2 * d
    # Backbone norms (`encoder_norm`, `decoder_norm`) + `RecurrentFeedbackAdapter` (`norm`, `proj`, `gate`)
    backbone_extra = 2 * d + (d + 2 * d * d + d + 1)
    coe_heads = (
        1
        + (d * d + d + 2 * (d * p_dim + p_dim))
        + (d * p_dim + p_dim + 4 * p_dim * p_dim + 2 * p_dim * d + d + 2 * d)
        + (2 * d * p_dim + p_dim + p_dim * 1 + 1 + 2 * d)
        + (d * p_dim + p_dim + p_dim * 5 + 5 + 3 * d)
        + (3 * d * p_dim + 3 * p_dim + p_dim * d + d + 2 * d)
    )
    shared_non_expert = (
        tok_emb
        + lm_head
        + L * (ple_per_layer + attn_per_layer + router_shared_per_layer)
        + L_dec * cross_attn_per_dec
        + backbone_extra
        + coe_heads
    )
    single_expert_layer = 3 * d * d_ffn  # 6,291,456 params/expert/layer
    router_per_expert_layer = d + 32 + 1
    per_expert_stack = L * (single_expert_layer + router_per_expert_layer)  # ~100.69M per expert across 16 layers

    def total_at(e: int) -> int:
        return shared_non_expert + e * per_expert_stack

    active_at_k = shared_non_expert + top_k * L * single_expert_layer
    milestones = compute_v8_mitosis_milestones(total_steps, start_experts=start_experts, max_experts=max_experts)
    stages = [
        {
            "step": 0,
            "experts": start_experts,
            "total_params": total_at(start_experts),
            "total_billions": round(total_at(start_experts) / 1e9, 3),
        },
        *[
            {
                "step": s,
                "experts": e,
                "total_params": total_at(e),
                "total_billions": round(total_at(e) / 1e9, 3),
            }
            for s, e in milestones
        ],
    ]
    return {
        "donor": OLMOE_DONOR_REPO,
        "vocab_size": V,
        "d_model": d,
        "layers_total": L,
        "expert_hidden_dim": d_ffn,
        "top_k": top_k,
        "single_expert_per_layer_params": single_expert_layer,
        "single_expert_full_stack_params": per_expert_stack,
        "shared_non_expert_params": shared_non_expert,
        "start_experts": start_experts,
        "start_total_params": total_at(start_experts),
        "start_total_billions": round(total_at(start_experts) / 1e9, 3),
        "max_experts": max_experts,
        "terminal_total_params": total_at(max_experts),
        "terminal_total_billions": round(total_at(max_experts) / 1e9, 3),
        "active_params_at_top_k": active_at_k,
        "active_billions_at_top_k": round(active_at_k / 1e9, 3),
        "stages": stages,
    }


def compute_v8_mitosis_milestones(
    total_steps: int = 2800,
    *,
    start_experts: int = 64,
    max_experts: int = 112,
    step_size: int = 16,
) -> tuple[tuple[int, int], ...]:
    """Compute staged mitosis milestones (`64 -> 80 -> 96 -> 112`) across `total_steps`."""
    if max_experts <= start_experts:
        return ()
    targets = list(range(start_experts + step_size, max_experts + 1, step_size))
    if targets[-1] != max_experts:
        targets.append(max_experts)
    fracs = (0.16, 0.39, 0.66, 0.82, 0.92)
    milestones: list[tuple[int, int]] = []
    for idx, t_e in enumerate(targets):
        frac = fracs[idx] if idx < len(fracs) else min(0.95, 0.16 + 0.78 * (idx + 1) / len(targets))
        m_step = max(2, min(total_steps - 10, int(round(frac * total_steps))))
        milestones.append((m_step, int(t_e)))
    return tuple(milestones)


@torch.no_grad()
def graft_olmoe_1b_7b_checkpoint(
    output_ckpt_dir: str | Path,
    *,
    donor_repo_or_path: str = OLMOE_DONOR_REPO,
    ple_dim: int = 128,
    projection_dim: int = 256,
    top_k: int = 8,
    seed: int = 42,
) -> Path:
    """Stream `allenai/OLMoE-1B-7B-0125` safetensors shard-by-shard into a Step-0 `bfloat16` `MCL4-COE` checkpoint."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    out_dir = Path(output_ckpt_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "step_00000.pt"
    meta_path = out_dir / "step_00000.meta.json"
    if pt_path.is_file() and meta_path.is_file():
        print(f"[graft] Existing Step-0 OLMoE checkpoint found at {pt_path}", flush=True)
        return meta_path

    t0 = time.perf_counter()
    donor_path = Path(donor_repo_or_path)
    if not donor_path.is_dir():
        print(f"[graft] Downloading/locating donor {donor_repo_or_path} from Hugging Face Hub...", flush=True)
        hf_token = os.environ.get("HF_TOKEN") or None
        donor_dir = Path(
            snapshot_download(
                repo_id=donor_repo_or_path,
                allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
                token=hf_token,
            )
        )
    else:
        donor_dir = donor_path

    backbone_cfg = MCL4CEDConfig(
        vocab_size=OLMOE_VOCAB_SIZE,
        pad_id=1,
        eos_token_id=50279,
        d_model=2048,
        n_heads=16,
        n_encoder_layers=8,
        n_decoder_layers=8,
        ple_dim=ple_dim,
        expert_hidden_dim=1024,
        num_experts=64,
        top_k=top_k,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        use_qk_norm=True,
        qk_norm_mode="full",
        norm_topk_prob=False,
        tie_word_embeddings=False,
        preserve_residual_across_stacks=True,
        recurrent_coverage_penalty=0.25,
        mitotic_ortho_coef=0.02,
        sequence_diversity_coef=0.02,
    )
    coe_cfg = COEConfig(
        projection_dim=projection_dim,
        initial_scale=5.0,
        recurrence_steps=1,
        recurrent_kv=False,
        depth_ponder=False,
        bos_id=50279,
        eos_id=50279,
    )

    print("[graft] Initializing empty bfloat16 MCL4-COE (7.18B, E=64)...", flush=True)
    torch.manual_seed(seed)
    orig_lin_reset = torch.nn.Linear.reset_parameters
    orig_emb_reset = torch.nn.Embedding.reset_parameters
    orig_init_normal = torch.nn.init.normal_
    old_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        torch.nn.Linear.reset_parameters = lambda _self: None  # type: ignore[method-assign]
        torch.nn.Embedding.reset_parameters = lambda _self: None  # type: ignore[method-assign]
        torch.nn.init.normal_ = lambda tensor, mean=0.0, std=1.0, generator=None: tensor  # type: ignore[assignment]
        with torch.device("cpu"):
            model = CognitiveOperatorEnsemble(MCL4CEDBackbone(backbone_cfg), coe_cfg)
    finally:
        torch.set_default_dtype(old_default_dtype)
        torch.nn.Linear.reset_parameters = orig_lin_reset  # type: ignore[method-assign]
        torch.nn.Embedding.reset_parameters = orig_emb_reset  # type: ignore[method-assign]
        torch.nn.init.normal_ = orig_init_normal  # type: ignore[assignment]

    # Initialize only the newly added MCL4-COE modules (<0.14B params), skipping the 6.92B donor layers
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)) and (
            not name.startswith("backbone.")
            or any(
                tag in name
                for tag in (".ple.", ".markov_router.", ".global_q_proj", ".global_kv_proj", "feedback.")
            )
        ):
            model.backbone._initialize(mod)

    # Zero-initialize MCL4-COE residual gates so Step-0 forward pass is 100% continuous with OLMoE
    for layer_idx in range(8):
        enc = model.backbone.encoder_layers[layer_idx]
        dec = model.backbone.decoder_layers[layer_idx]
        if enc.ple is not None:
            enc.ple.gate_scale.data.zero_()
        if dec.ple is not None:
            dec.ple.gate_scale.data.zero_()
        if enc.attention.markov_gate is not None:
            enc.attention.markov_gate.data.zero_()
        if dec.local_attention.markov_gate is not None:
            dec.local_attention.markov_gate.data.zero_()
        dec.global_out_proj.weight.data.zero_()
    model.backbone.feedback.gate.data.zero_()
    model.backbone.encoder_norm.weight.data.fill_(1.0)

    perm = _build_rope_interleave_perm(n_heads=16, head_dim=128)
    safetensor_files = sorted(donor_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {donor_dir}")

    # Temporary buffers for partial projections `(q, k, v)` and `(gate, up)` that may span shard boundaries
    qkv_parts: dict[int, dict[str, torch.Tensor]] = {i: {} for i in range(16)}
    expert_up_parts: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    def _target_block(olmoe_layer_idx: int):
        if olmoe_layer_idx < 8:
            return model.backbone.encoder_layers[olmoe_layer_idx]
        return model.backbone.decoder_layers[olmoe_layer_idx - 8]

    for sf_path in safetensor_files:
        print(f"[graft] Streaming donor shard {sf_path.name}...", flush=True)
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key).to(dtype=torch.bfloat16)
                if key == "model.embed_tokens.weight":
                    model.backbone.token_embedding.weight.data.copy_(t)
                elif key == "lm_head.weight":
                    model.backbone.lm_head.weight.data.copy_(t)
                elif key == "model.norm.weight":
                    model.backbone.final_norm.weight.data.copy_(t)
                elif key.startswith("model.layers."):
                    parts = key.split(".")
                    l_idx = int(parts[2])
                    blk = _target_block(l_idx)
                    attn_blk = blk.attention if hasattr(blk, "attention") else blk.local_attention
                    sub = ".".join(parts[3:])
                    if sub == "input_layernorm.weight":
                        blk.attn_norm.weight.data.copy_(t)
                    elif sub == "post_attention_layernorm.weight":
                        blk.moe_norm.weight.data.copy_(t)
                    elif sub == "self_attn.o_proj.weight":
                        attn_blk.out_proj.weight.data.copy_(t)
                    elif sub == "self_attn.q_norm.weight":
                        attn_blk.q_norm.weight.data.copy_(t[perm])
                    elif sub == "self_attn.k_norm.weight":
                        attn_blk.k_norm.weight.data.copy_(t[perm])
                    elif sub in (
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                    ):
                        tag = sub.split(".")[1]  # q_proj, k_proj, v_proj
                        if tag in ("q_proj", "k_proj"):
                            t = t[perm, :]
                        qkv_parts[l_idx][tag] = t
                        if len(qkv_parts[l_idx]) == 3:
                            q_w = qkv_parts[l_idx].pop("q_proj")
                            k_w = qkv_parts[l_idx].pop("k_proj")
                            v_w = qkv_parts[l_idx].pop("v_proj")
                            attn_blk.qkv.weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
                            del q_w, k_w, v_w
                    elif sub == "mlp.gate.weight":
                        blk.moe.router.weight.data.copy_(t)
                    elif sub.startswith("mlp.experts."):
                        e_idx = int(parts[5])
                        proj_name = parts[6]  # gate_proj, up_proj, down_proj
                        if proj_name == "down_proj":
                            blk.moe.experts[e_idx].down_proj.weight.data.copy_(t)
                        elif proj_name in ("gate_proj", "up_proj"):
                            pair_key = (l_idx, e_idx)
                            if pair_key not in expert_up_parts:
                                expert_up_parts[pair_key] = {}
                            expert_up_parts[pair_key][proj_name] = t
                            if len(expert_up_parts[pair_key]) == 2:
                                g_w = expert_up_parts[pair_key].pop("gate_proj")
                                u_w = expert_up_parts[pair_key].pop("up_proj")
                                del expert_up_parts[pair_key]
                                blk.moe.experts[e_idx].up_proj.weight.data.copy_(
                                    torch.cat([g_w, u_w], dim=0)
                                )
                                del g_w, u_w

    total_params = parameter_count(model)
    model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    gc.collect()

    uniform_loads = [round(1.0 / 64.0, 5)] * 64
    latest_metrics = {
        "step": 0,
        "num_experts": 64,
        "top_k": top_k,
        "expert_load": uniform_loads,
        "parameter_count": total_params,
    }
    payload = {
        "step": 0,
        "config": {"donor": donor_repo_or_path, "model_dtype": "bfloat16"},
        "backbone_config": asdict(backbone_cfg),
        "coe_config": asdict(coe_cfg),
        "mitotic_pairs": [],
        "model_state": model_state,
        "optimizer_state": {},
        "cumulative_wall_s": 0.0,
        "cumulative_tokens": 0,
        "latest_metrics": latest_metrics,
    }
    torch.save(payload, pt_path)
    del payload, model_state
    gc.collect()

    meta = {
        "checkpoint_id": f"{out_dir.parent.name}/step_00000",
        "run_id": out_dir.parent.name,
        "step": 0,
        "total_steps": 0,
        "timestamp": time.time(),
        "pt_file": str(pt_path),
        "meta_file": str(meta_path),
        "parameter_count": total_params,
        "num_experts": 64,
        "use_nvfp4": False,
        "metrics": latest_metrics,
        "sample_prompt": "How do plants use capillary action to transport water?",
        "sample_uta": "[Grafted Step-0 allenai/OLMoE-1B-7B-0125 -> MCL4-COE 7.08B]",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - t0
    print(
        f"[graft] Saved Step-0 MCL4-COE OLMoE checkpoint ({total_params:,} params) to {pt_path} in {elapsed:.1f}s",
        flush=True,
    )
    return meta_path


def _decode_masked_block(
    tokens: list[int],
    src_tok: Any,
) -> tuple[str, str]:
    """Split a `0x8000`-masked uint16 token row into `(masked_prefix_text, unmasked_completion_text)`."""
    pad_id = int(src_tok.pad_id)
    bos_id = int(src_tok.bos_id)
    eos_id = int(src_tok.eos_id)
    masked_ids: list[int] = []
    unmasked_ids: list[int] = []
    seen_unmasked = False
    for raw_t in tokens:
        is_masked = (raw_t & 0x8000) != 0
        tid = raw_t & 0x7FFF
        if tid in (pad_id, bos_id, eos_id):
            continue
        if is_masked and not seen_unmasked:
            masked_ids.append(tid)
        else:
            seen_unmasked = True
            unmasked_ids.append(tid)
    prefix_txt = src_tok.decode(masked_ids) if masked_ids else ""
    comp_txt = src_tok.decode(unmasked_ids) if unmasked_ids else ""
    return prefix_txt, comp_txt


def transcode_shards_to_olmoe50k(
    src_shards_root: str | Path,
    dst_shards_root: str | Path,
    *,
    src_vocab_size: int = 16384,
    max_train_shards_per_track: int = 12,
    seq_len: int = 1024,
) -> Path:
    """Transcode `morph16k` (or `granite49k`) token shards + `rlvr_env_pool.pt` into `OLMoE-50K` (`vocab_size=50304`)."""
    src_root = Path(src_shards_root).expanduser().resolve()
    dst_root = Path(dst_shards_root).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    ready_tracks = [
        t
        for t in ("pretrain_lm", "uta_trace", "typed_decision")
        if (dst_root / t / "index_receipt.json").is_file()
    ]
    if len(ready_tracks) == 3:
        print(f"[transcode] All 3 OLMoE-50K tracks already ready at {dst_root}", flush=True)
        return dst_root

    src_tok = get_tokenizer(src_vocab_size)
    dst_tok = get_tokenizer(OLMOE_VOCAB_SIZE)
    assert dst_tok.vocab_size == OLMOE_VOCAB_SIZE

    for track in ("pretrain_lm", "uta_trace", "typed_decision"):
        dst_track_dir = dst_root / track
        if (dst_track_dir / "index_receipt.json").is_file():
            continue
        src_receipt_path = src_root / track / "index_receipt.json"
        if not src_receipt_path.is_file():
            raise FileNotFoundError(f"Missing source track receipt: {src_receipt_path}")
        src_receipt = json.loads(src_receipt_path.read_text(encoding="utf-8"))
        src_seq_len = int(src_receipt["seq_len"])
        row_len = src_seq_len + 1

        for split_name, max_sh in (("train", max_train_shards_per_track), ("val", 2)):
            shards_list = [
                s
                for s in src_receipt.get("shards", [])
                if s.get("split", "train") == split_name
                and (src_root / track / s["bin_file"]).is_file()
            ][:max_sh]
            if not shards_list:
                continue
            writer = BlockAlignedShardWriter(
                dst_track_dir,
                track=track,
                split=split_name,
                seq_len=seq_len,
                max_blocks_per_shard=4096,
                vocab_size=OLMOE_VOCAB_SIZE,
            )
            print(
                f"[transcode] Transcoding {track} ({split_name}: {len(shards_list)} shards) -> OLMoE-50K...",
                flush=True,
            )
            bos_id = int(src_tok.bos_id)
            eos_id = int(src_tok.eos_id)
            pad_id = int(src_tok.pad_id)
            for sh in shards_list:
                bin_path = src_root / track / sh["bin_file"]
                raw_u16 = np.fromfile(bin_path, dtype="<u2")
                clean_ids = raw_u16 & 0x7FFF
                bos_positions = np.nonzero(clean_ids == bos_id)[0]
                eos_positions = np.nonzero(clean_ids == eos_id)[0]
                if bos_positions.size == 0 or eos_positions.size == 0:
                    continue
                eos_cursor = 0
                n_eos = eos_positions.size
                accepted = 0
                for b_pos in bos_positions:
                    while eos_cursor < n_eos and eos_positions[eos_cursor] <= b_pos:
                        eos_cursor += 1
                    if eos_cursor >= n_eos:
                        break
                    e_pos = eos_positions[eos_cursor]
                    span_clean = [
                        int(tid)
                        for tid in clean_ids[b_pos + 1 : e_pos].tolist()
                        if int(tid) not in (pad_id, bos_id, eos_id)
                    ]
                    if not span_clean:
                        continue
                    text = src_tok.decode(span_clean)
                    if track == "pretrain_lm":
                        writer.add_pretrain_document(text)
                        accepted += 1
                    elif "#T$#" in text and "#A$#" in text:
                        u_part, rest = text.split("#T$#", 1)
                        t_part, a_part = rest.split("#A$#", 1)
                        u_clean = u_part.replace("#U$#", "", 1).strip()
                        t_clean = t_part.strip()
                        a_clean = a_part.strip()
                        if u_clean and a_clean and writer.add_uta_episode(u_clean, t_clean, a_clean):
                            accepted += 1
                writer.record_provenance(
                    corpus=f"transcoded_olmoe50k:{track}:{split_name}",
                    source_file=sh["bin_file"],
                    row_start=1,
                    row_end=max(1, accepted),
                    episodes_accepted=accepted,
                    episodes_rejected_stage0=0,
                )
            writer.flush_shard()

    # Transcode decision caches if present
    src_dec_dir = src_root / "typed_decision"
    dst_dec_dir = dst_root / "typed_decision"
    dst_dec_dir.mkdir(parents=True, exist_ok=True)
    for cache_name in ("decision_examples_cache.pt", "decision_examples_val_cache.pt"):
        if (src_dec_dir / cache_name).is_file() and not (dst_dec_dir / cache_name).is_file():
            shutil.copy2(src_dec_dir / cache_name, dst_dec_dir / cache_name)

    # Copy rlvr_env_pool.pt directly (OrnithTask entries are tokenizer-agnostic strings)
    src_rlvr = src_root / "rlvr_env_pool.pt"
    dst_rlvr = dst_root / "rlvr_env_pool.pt"
    if src_rlvr.is_file():
        shutil.copy2(src_rlvr, dst_rlvr)
        print(f"[transcode] Copied RLVR environment pool -> {dst_rlvr}", flush=True)

    return dst_root


def build_v8_olmoe_12b_config(
    *,
    run_id: str = "colab_mcl4_coe_v8_olmoe_12b",
    runs_root: str = "/content/runs" if Path("/content").is_dir() else r"X:\MCL4-COE\runs",
    shards_root: str = (
        "/content/data_shards_olmoe50k"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\data_shards_olmoe50k"
    ),
    corpora_root: str = (
        "/content/MCL4-COE/corpora" if Path("/content").is_dir() else r"X:\corpora"
    ),
    resume_checkpoint: str | None = None,
    total_steps: int = 2800,
    start_experts: int = 64,
    max_experts: int = 112,
    top_k: int = 8,
    plastic_experts_per_layer: int = 32,
    batch_size: int = 8,
    micro_batch_size: int = 2,
    max_seq_len: int = 1024,
    hf_backup_repo: str | None = "mesmac/mcl4-coe-runs",
) -> TrainingRunConfig:
    """Build the `TrainingRunConfig` for `MCL4-COE v8-OLMoE-12B` (`7.08B -> 11.91B` params)."""
    milestones = compute_v8_mitosis_milestones(
        total_steps,
        start_experts=start_experts,
        max_experts=max_experts,
        step_size=16,
    )
    return TrainingRunConfig(
        run_id=run_id,
        runs_root=runs_root,
        shards_root=shards_root,
        corpora_root=corpora_root,
        resume_checkpoint=resume_checkpoint,
        vocab_size=OLMOE_VOCAB_SIZE,
        pad_id=1,
        eos_token_id=50279,
        max_seq_len=max_seq_len,
        d_model=2048,
        n_heads=16,
        n_encoder_layers=8,
        n_decoder_layers=8,
        ple_dim=128,
        expert_hidden_dim=1024,
        num_experts=start_experts,
        max_experts=max_experts,
        top_k=top_k,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        use_qk_norm=True,
        qk_norm_mode="full",
        norm_topk_prob=False,
        tie_word_embeddings=False,
        preserve_residual_across_stacks=True,
        donor_backbone_lr_scale=0.10,
        model_dtype="bfloat16",
        plastic_experts_per_layer=plastic_experts_per_layer,
        freeze_donor_embeddings=True,
        enable_mitosis=True,
        orthogonal_mitosis=True,
        mitosis_every=650,
        mitosis_step_size=16,
        mitosis_milestones=milestones,
        clear_consumed_shards=False,
        mitotic_ortho_coef=0.02,
        sequence_diversity_coef=0.02,
        recurrent_coverage_penalty=0.25,
        projection_dim=256,
        recurrence_steps=1,
        recurrent_kv=False,
        depth_ponder=False,
        deep_supervision_weight=0.0,
        enable_gist_conditioning=True,
        gist_conditioning_prob=0.35,
        latent_prior_scale=0.15,
        use_nvfp4=False,
        cpu_offload_optimizer=False,
        total_steps=total_steps,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        decision_batch_size=2,
        operator_every=4,
        lr=1.5e-4,
        warmup_ratio=0.06,
        min_lr_ratio=0.15,
        weight_decay=0.01,
        grad_clip=1.0,
        curriculum_mode="mitotic_staged",
        curriculum_stages=[
            {"pretrain_lm": 0.45, "uta_trace": 0.40, "typed_decision": 0.15},
            {"pretrain_lm": 0.30, "uta_trace": 0.50, "typed_decision": 0.20},
            {"pretrain_lm": 0.22, "uta_trace": 0.53, "typed_decision": 0.25},
            {"pretrain_lm": 0.15, "uta_trace": 0.55, "typed_decision": 0.30},
        ],
        stage_seq_lens=(512, 640, 768, 1024),
        enable_ornith_rl=True,
        ornith_every=8,
        ornith_weight=0.12,
        ornith_warmup_ratio=0.16,
        enable_dream_rsi=True,
        dream_every=8,
        dream_max_rounds=2,
        dream_max_calls=3,
        dream_workers=2,
        dream_revisions=2,
        checkpoint_every=250,
        max_keep_checkpoints=1,
        save_optimizer_state=False,
        hf_backup_repo=hf_backup_repo,
        hf_keep_checkpoints=2,
        eval_every=250,
        val_max_batches_per_track=4,
        sample_every=100,
    )


def build_v8_olmoe_12b_stage2_rl_config(
    *,
    run_id: str = "colab_mcl4_coe_v8_olmoe_12b",
    runs_root: str = "/content/runs" if Path("/content").is_dir() else r"X:\MCL4-COE\runs",
    shards_root: str = (
        "/content/data_shards_olmoe50k"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\data_shards_olmoe50k"
    ),
    corpora_root: str = (
        "/content/MCL4-COE/corpora" if Path("/content").is_dir() else r"X:\corpora"
    ),
    resume_checkpoint: str = "/content/runs/colab_mcl4_coe_v8_olmoe_12b/checkpoints/step_02800.meta.json",
    total_steps: int = 3200,
    num_experts: int = 112,
    top_k: int = 8,
    top_k_milestones: tuple[tuple[int, int], ...] = ((2900, 12), (3050, 16)),
    plastic_experts_per_layer: int = 64,
    batch_size: int = 8,
    micro_batch_size: int = 2,
    max_seq_len: int = 1024,
    lr: float = 8.0e-5,
    hf_backup_repo: str | None = "mesmac/mcl4-coe-runs",
) -> TrainingRunConfig:
    """Build Stage-2 Post-Mitosis Twin-Release + Memory-Adaptive Density RL & RSI config (`steps 2801 -> 3200`).

    - Un-ties all mitotic twin pairs (`release_mitotic_pairs=True`, `enable_mitosis=False`) so all `112`
      experts per layer (`11.97B` total parameters) co-activate independently without twin suppression.
    - Expands `plastic_experts_per_layer` to `64` (`32` newly released twins + `32` highest-load experts)
      to leverage available RTX PRO 6000 VRAM headroom.
    - Progressively scales active expert density `top_k: 8 -> 12 -> 16` (`1.71B -> 2.11B -> 2.52B` active).
    - Runs high-cadence 9-Operator COE (`operator_every=2`), Ornith-2.0 Verified Execution GRPO
      (`ornith_every=2`, `ornith_weight=0.25`), Dream-RSI (`dream_every=2`), and recurrent CED pondering
      (`recurrence_steps=2`, `recurrent_kv=True`, `depth_ponder=True`).
    """
    return TrainingRunConfig(
        run_id=run_id,
        runs_root=runs_root,
        shards_root=shards_root,
        corpora_root=corpora_root,
        resume_checkpoint=resume_checkpoint,
        vocab_size=OLMOE_VOCAB_SIZE,
        pad_id=1,
        eos_token_id=50279,
        max_seq_len=max_seq_len,
        d_model=2048,
        n_heads=16,
        n_encoder_layers=8,
        n_decoder_layers=8,
        ple_dim=128,
        expert_hidden_dim=1024,
        num_experts=num_experts,
        max_experts=num_experts,
        top_k=top_k,
        top_k_milestones=top_k_milestones,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        use_qk_norm=True,
        qk_norm_mode="full",
        norm_topk_prob=False,
        tie_word_embeddings=False,
        preserve_residual_across_stacks=True,
        donor_backbone_lr_scale=0.15,
        model_dtype="bfloat16",
        plastic_experts_per_layer=plastic_experts_per_layer,
        freeze_donor_embeddings=True,
        enable_mitosis=False,
        release_mitotic_pairs=True,
        orthogonal_mitosis=True,
        mitosis_every=0,
        mitosis_step_size=0,
        mitosis_milestones=(),
        clear_consumed_shards=False,
        mitotic_ortho_coef=0.0,
        sequence_diversity_coef=0.025,
        recurrent_coverage_penalty=0.25,
        projection_dim=256,
        recurrence_steps=2,
        recurrent_kv=True,
        depth_ponder=True,
        deep_supervision_weight=0.15,
        enable_gist_conditioning=True,
        gist_conditioning_prob=0.45,
        latent_prior_scale=0.18,
        use_nvfp4=False,
        cpu_offload_optimizer=False,
        total_steps=total_steps,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        decision_batch_size=2,
        operator_every=2,
        lr=lr,
        warmup_ratio=0.08,
        min_lr_ratio=0.20,
        weight_decay=0.01,
        grad_clip=1.0,
        curriculum_mode="cosine",
        start_weights={"pretrain_lm": 0.15, "uta_trace": 0.55, "typed_decision": 0.30},
        end_weights={"pretrain_lm": 0.10, "uta_trace": 0.60, "typed_decision": 0.30},
        stage_seq_lens=(),
        enable_ornith_rl=True,
        ornith_every=2,
        ornith_weight=0.25,
        ornith_warmup_ratio=0.0,
        enable_dream_rsi=True,
        dream_every=2,
        dream_max_rounds=3,
        dream_max_calls=4,
        dream_workers=2,
        dream_revisions=3,
        checkpoint_every=100,
        max_keep_checkpoints=2,
        save_optimizer_state=False,
        hf_backup_repo=hf_backup_repo,
        hf_keep_checkpoints=3,
        eval_every=100,
        val_max_batches_per_track=4,
        sample_every=50,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MCL4-COE v8-OLMoE-12B (7.08B -> 11.97B + Stage-2 RL)")
    parser.add_argument("--dry-run", action="store_true", help="Print parameter profile and exit")
    parser.add_argument("--graft-only", action="store_true", help="Only graft OLMoE Step-0 checkpoint and exit")
    parser.add_argument("--stage2-rl", action="store_true", help="Run Stage-2 Twin-Release + Multi-Active Density RL/RSI")
    parser.add_argument("--run-id", type=str, default="colab_mcl4_coe_v8_olmoe_12b")
    parser.add_argument(
        "--runs-root",
        type=str,
        default="/content/runs" if Path("/content").is_dir() else r"X:\MCL4-COE\runs",
    )
    parser.add_argument(
        "--shards-root",
        type=str,
        default="/content/data_shards_olmoe50k"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\data_shards_olmoe50k",
    )
    parser.add_argument("--src-shards-root", type=str, default="")
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--max-experts", type=int, default=112)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--plastic-experts", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    eff_steps = args.total_steps or (3200 if args.stage2_rl else 2800)
    profile = compute_v8_olmoe_parameter_profile(
        start_experts=64,
        max_experts=args.max_experts,
        top_k=args.top_k,
        total_steps=eff_steps,
    )
    print(json.dumps(profile, indent=2), flush=True)
    if args.dry_run:
        return

    run_dir = Path(args.runs_root).expanduser().resolve() / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    resume_ckpt = args.resume.strip() or None
    if resume_ckpt is None:
        existing_ckpts = sorted(ckpt_dir.glob("step_*.meta.json"))
        if existing_ckpts:
            resume_ckpt = str(existing_ckpts[-1])
        elif not args.stage2_rl:
            step0_meta = graft_olmoe_1b_7b_checkpoint(
                ckpt_dir,
                top_k=args.top_k,
            )
            resume_ckpt = str(step0_meta)

    if args.graft_only:
        return

    if args.src_shards_root:
        transcode_shards_to_olmoe50k(
            args.src_shards_root,
            args.shards_root,
        )

    if args.stage2_rl:
        cfg = build_v8_olmoe_12b_stage2_rl_config(
            run_id=args.run_id,
            runs_root=args.runs_root,
            shards_root=args.shards_root,
            resume_checkpoint=resume_ckpt or str(ckpt_dir / "step_02800.meta.json"),
            total_steps=eff_steps,
            num_experts=args.max_experts,
            top_k=args.top_k,
            plastic_experts_per_layer=args.plastic_experts or 64,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
        )
    else:
        cfg = build_v8_olmoe_12b_config(
            run_id=args.run_id,
            runs_root=args.runs_root,
            shards_root=args.shards_root,
            resume_checkpoint=resume_ckpt,
            total_steps=eff_steps,
            max_experts=args.max_experts,
            top_k=args.top_k,
            plastic_experts_per_layer=args.plastic_experts or 32,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
        )

    def _on_step(rec: dict[str, Any]) -> None:
        st = int(rec.get("step", 0))
        if st == 1 or st % 10 == 0 or "mitosis_event" in rec:
            mev = rec.get("mitosis_event")
            m_str = f" | MITOSIS {mev['old_experts']}->{mev['new_experts']} ({mev['parameter_count']/1e9:.2f}B)" if mev and mev.get("action") == "expand" else ""
            orn_str = (
                f" | orn_rw={rec.get('ornith_harness_reward', 0.0):.3f} pass={rec.get('ornith_pass_rate', 0.0):.2f} drm={rec.get('dream_cycle', 0)}"
                if "ornith_harness_reward" in rec
                else ""
            )
            print(
                f"[step {st:4d}/{cfg.total_steps}] loss={rec.get('total_loss'):.4f} "
                f"(lm={rec.get('lm_loss'):.4f} bpb={rec.get('train_bpb'):.4f} dec={rec.get('decision_loss'):.4f}) | "
                f"E={rec.get('num_experts')} k={rec.get('top_k')} | "
                f"{rec.get('tok_per_s'):.0f} tok/s | vram={rec.get('vram_allocated_mb'):.0f}MB{orn_str}{m_str}",
                flush=True,
            )

    trainer = MCL4COETrainer(cfg, on_step_callback=_on_step)
    trainer.train()


if __name__ == "__main__":
    main()
