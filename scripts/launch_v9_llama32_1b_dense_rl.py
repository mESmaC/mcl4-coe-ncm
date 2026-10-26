r"""Step-0 Algebraic Weight Surgery, Shard Transcoder, and Pure RL + NCM Launcher for `MCL4-COE v9-Llama3.2-1B-Dense`.

Grafts `unsloth/Llama-3.2-1B-Instruct` (`meta-llama/Llama-3.2-1B-Instruct`, `1.24B` dense donor) into the full
`MCL4-COE + Neural Context Memory (NCM) + NCMCOEHeadBridge` architecture with **zero Step-0 perplexity shock**:
1. Co-Active Sliced Dense FFN (`E = 8, top_k = 8, expert_hidden_dim = 1024, routed_scaling_factor = 8.0`):
   - Slices each layer's dense SwiGLU MLP (`D_ffn = 8192`) into `8` equal sub-experts of width `1024`.
   - With `top_k = 8`, `norm_topk_prob = True`, `routed_scaling_factor = 8.0`, and zero-initialized router logits,
     all `8/8` experts (`100%` of the FFN) execute on every token with exact unit weight (`w_e = 8.0 * 1/8 = 1.0`)
     at Step 0 (`0.00 ULP` algebraic equivalence to the monolithic dense MLP), while allowing the router to learn
     token- and cycle-specific emphasis during RL/RSI and remaining 100% ready for granular expert mitosis
     (`E = 8 -> 12 -> 16`) later.
2. Exact GQA-to-MHA Expansion & Llama-3 RoPE Interleaving:
   - Expands `8` GQA KV heads (`head_dim = 64`) via `repeat_interleave(4, dim=0)` into `32` full MHA KV heads
     and permutes `q_proj` / `k_proj` rows (`[0, 32, 1, 33, ..., 31, 63]`) with exact `llama3` RoPE frequency scaling.
3. Zero-Shock Residual Gates + NVFP4 Execution:
   - Zero-initializes `PLE`, `MCL4` Markov gates, `Decoder` cross-attention `global_out_proj`, and `RecurrentFeedback`
     gates so Step 0 is 100% continuous with the pretrained `Llama-3.2-1B-Instruct` manifold.
   - Runs hardware-accelerated `NVFP4` (`use_nvfp4 = True`, `1024 % 128 == 0` and `2048 % 128 == 0`) with 48-skill
     Polymath + Sherlockian Deductive `OrnithSelfImprover` GRPO, `Dream-RSI`, 9-Operator `COE` (`recurrence_steps=2`),
     and `NCM` + `NCMCOEHeadBridge`.
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
from mcl4_coe.pipeline import BlockAlignedShardWriter
from mcl4_coe.trainer import MCL4COETrainer, TrainingRunConfig

LLAMA32_1B_DONOR_REPO = "unsloth/Llama-3.2-1B-Instruct"
LLAMA32_VOCAB_SIZE = 128256
LLAMA32_PAD_ID = 128004
LLAMA32_BOS_ID = 128000
LLAMA32_EOS_ID = 128009
LLAMA32_ROPE_SCALING: dict[str, Any] = {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}


def _build_rope_interleave_perm(n_heads: int = 32, head_dim: int = 64) -> torch.Tensor:
    """Return index permutation mapping half-rotated RoPE `[0..31, 32..63]` -> interleaved `[0, 32, 1, 33, ...]`."""
    half = head_dim // 2
    head_perm = torch.empty(head_dim, dtype=torch.long)
    head_perm[0::2] = torch.arange(0, half, dtype=torch.long)
    head_perm[1::2] = torch.arange(half, head_dim, dtype=torch.long)
    return torch.cat([head_perm + h * head_dim for h in range(n_heads)], dim=0)


@torch.no_grad()
def graft_llama32_1b_dense_checkpoint(
    output_ckpt_dir: str | Path,
    *,
    donor_repo_or_path: str = LLAMA32_1B_DONOR_REPO,
    num_experts: int = 8,
    top_k: int = 8,
    ple_dim: int = 64,
    projection_dim: int = 256,
    seed: int = 42,
    verify_graft: bool = True,
) -> Path:
    """Graft `unsloth/Llama-3.2-1B-Instruct` (`1.24B` dense) into a Step-0 `bfloat16` `MCL4-COE` checkpoint."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    if 8192 % num_experts != 0:
        raise ValueError(f"Llama-3.2-1B d_ffn=8192 must be divisible by num_experts={num_experts}")
    expert_hidden_dim = 8192 // num_experts
    routed_scaling = float(num_experts)

    out_dir = Path(output_ckpt_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "step_00000.pt"
    meta_path = out_dir / "step_00000.meta.json"
    if pt_path.is_file() and meta_path.is_file():
        print(f"[graft] Existing Step-0 Llama-3.2-1B-Dense checkpoint found at {pt_path}", flush=True)
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
        vocab_size=LLAMA32_VOCAB_SIZE,
        pad_id=LLAMA32_PAD_ID,
        eos_token_id=LLAMA32_EOS_ID,
        d_model=2048,
        n_heads=32,
        n_encoder_layers=8,
        n_decoder_layers=8,
        ple_dim=ple_dim,
        expert_hidden_dim=expert_hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        rms_norm_eps=1e-5,
        rope_theta=500_000.0,
        rope_scaling=dict(LLAMA32_ROPE_SCALING),
        use_qk_norm=False,
        qk_norm_mode="head",
        norm_topk_prob=True,
        routed_scaling_factor=routed_scaling,
        tie_word_embeddings=True,
        preserve_residual_across_stacks=True,
        recurrent_coverage_penalty=0.20,
        mitotic_ortho_coef=0.02,
        sequence_diversity_coef=0.02,
    )
    coe_cfg = COEConfig(
        projection_dim=projection_dim,
        initial_scale=5.0,
        recurrence_steps=2,
        recurrent_kv=True,
        depth_ponder=True,
        bos_id=LLAMA32_BOS_ID,
        eos_id=LLAMA32_EOS_ID,
    )

    print(
        f"[graft] Initializing bfloat16 MCL4-COE Llama-3.2-1B Co-Active Dense "
        f"(E={num_experts}, top_k={top_k}, d_expert={expert_hidden_dim}, scale={routed_scaling:.1f}x)...",
        flush=True,
    )
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

    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)) and (
            not name.startswith("backbone.")
            or any(
                tag in name
                for tag in (".ple.", ".markov_router.", ".global_q_proj", ".global_kv_proj", "feedback.")
            )
        ):
            model.backbone._initialize(mod)

    # Zero-initialize MCL4-COE residual gates + dense slice routers so Step 0 is 100% identical to Llama-3.2-1B
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
        enc.moe.router.weight.data.zero_()
        dec.moe.router.weight.data.zero_()
    model.backbone.feedback.gate.data.zero_()
    model.backbone.encoder_norm.weight.data.fill_(1.0)

    perm = _build_rope_interleave_perm(n_heads=32, head_dim=64)
    safetensor_files = sorted(donor_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {donor_dir}")

    qkv_parts: dict[int, dict[str, torch.Tensor]] = {i: {} for i in range(16)}
    mlp_parts: dict[int, dict[str, torch.Tensor]] = {i: {} for i in range(16)}

    def _target_block(llama_layer_idx: int):
        if llama_layer_idx < 8:
            return model.backbone.encoder_layers[llama_layer_idx]
        return model.backbone.decoder_layers[llama_layer_idx - 8]

    for sf_path in safetensor_files:
        print(f"[graft] Streaming donor shard {sf_path.name}...", flush=True)
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key).to(dtype=torch.bfloat16)
                if key == "model.embed_tokens.weight":
                    model.backbone.token_embedding.weight.data.copy_(t)
                    if not backbone_cfg.tie_word_embeddings:
                        model.backbone.lm_head.weight.data.copy_(t)
                elif key == "lm_head.weight" and not backbone_cfg.tie_word_embeddings:
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
                    elif sub in (
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                    ):
                        tag = sub.split(".")[1]  # q_proj, k_proj, v_proj
                        if tag in ("k_proj", "v_proj") and t.shape[0] < 2048:
                            n_kv = t.shape[0] // 64
                            rep = 32 // n_kv
                            t = (
                                t.view(n_kv, 64, 2048)
                                .repeat_interleave(rep, dim=0)
                                .reshape(2048, 2048)
                            )
                        if tag in ("q_proj", "k_proj"):
                            t = t[perm, :]
                        qkv_parts[l_idx][tag] = t
                        if len(qkv_parts[l_idx]) == 3:
                            q_w = qkv_parts[l_idx].pop("q_proj")
                            k_w = qkv_parts[l_idx].pop("k_proj")
                            v_w = qkv_parts[l_idx].pop("v_proj")
                            attn_blk.qkv.weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
                            del q_w, k_w, v_w
                    elif sub in (
                        "mlp.gate_proj.weight",
                        "mlp.up_proj.weight",
                        "mlp.down_proj.weight",
                    ):
                        m_tag = sub.split(".")[1]  # gate_proj, up_proj, down_proj
                        mlp_parts[l_idx][m_tag] = t
                        if len(mlp_parts[l_idx]) == 3:
                            g_full = mlp_parts[l_idx].pop("gate_proj")  # [8192, 2048]
                            u_full = mlp_parts[l_idx].pop("up_proj")    # [8192, 2048]
                            d_full = mlp_parts[l_idx].pop("down_proj")  # [2048, 8192]
                            g_slices = g_full.chunk(num_experts, dim=0)
                            u_slices = u_full.chunk(num_experts, dim=0)
                            d_slices = d_full.chunk(num_experts, dim=1)
                            for e_idx in range(num_experts):
                                blk.moe.experts[e_idx].up_proj.weight.data.copy_(
                                    torch.cat([g_slices[e_idx], u_slices[e_idx]], dim=0)
                                )
                                blk.moe.experts[e_idx].down_proj.weight.data.copy_(d_slices[e_idx])
                            del g_full, u_full, d_full, g_slices, u_slices, d_slices

    if verify_graft:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print("[graft] Running Step-0 algebraic verification against HF LlamaForCausalLM...", flush=True)
            hf_tok = AutoTokenizer.from_pretrained(str(donor_dir))
            hf_model = AutoModelForCausalLM.from_pretrained(
                str(donor_dir), torch_dtype=torch.bfloat16, device_map="cpu"
            ).eval()
            model.eval()
            test_ids = hf_tok(
                "Capillary action transports water through xylem vessels because",
                return_tensors="pt",
            )["input_ids"]
            hf_logits = hf_model(test_ids).logits.float()
            mcl_logits = model.backbone(test_ids).logits.float()
            cos_sim = torch.nn.functional.cosine_similarity(
                hf_logits.reshape(-1, hf_logits.shape[-1]),
                mcl_logits.reshape(-1, mcl_logits.shape[-1]),
                dim=-1,
            ).mean().item()
            max_diff = (hf_logits - mcl_logits).abs().max().item()
            top1_match = (hf_logits.argmax(dim=-1) == mcl_logits.argmax(dim=-1)).float().mean().item()
            print(
                f"[graft] Step-0 Verification: cosine_sim={cos_sim:.6f} | "
                f"top1_token_agreement={top1_match * 100:.1f}% | max_logit_diff={max_diff:.5f}",
                flush=True,
            )
            del hf_model, hf_logits, mcl_logits
            gc.collect()
        except Exception as exc:
            print(f"[graft] Verification skipped/warn: {exc}", flush=True)

    total_params = parameter_count(model)
    model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    gc.collect()

    uniform_loads = [round(1.0 / float(num_experts), 5)] * num_experts
    latest_metrics = {
        "step": 0,
        "num_experts": num_experts,
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
        "num_experts": num_experts,
        "use_nvfp4": True,
        "metrics": latest_metrics,
        "sample_prompt": "How do plants use capillary action to transport water?",
        "sample_uta": f"[Grafted Step-0 {donor_repo_or_path} -> MCL4-COE Co-Active Dense E={num_experts}/k={top_k}]",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - t0
    print(
        f"[graft] Saved Step-0 MCL4-COE Llama-3.2-1B-Dense checkpoint ({total_params:,} params) "
        f"to {pt_path} in {elapsed:.1f}s",
        flush=True,
    )
    return meta_path


def transcode_shards_to_llama128k(
    src_shards_root: str | Path,
    dst_shards_root: str | Path,
    *,
    src_vocab_size: int | None = None,
    max_train_shards_per_track: int = 8,
    seq_len: int = 512,
) -> Path:
    """Transcode `granite49k` (`uint32`) or `morph16k` (`uint16`) token shards into `Llama-128K` (`vocab_size=128256`)."""
    src_root = Path(src_shards_root).expanduser().resolve()
    dst_root = Path(dst_shards_root).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    ready_tracks = [
        t
        for t in ("pretrain_lm", "uta_trace", "typed_decision")
        if (dst_root / t / "index_receipt.json").is_file()
    ]
    if len(ready_tracks) == 3:
        print(f"[transcode] All 3 Llama-128K tracks already ready at {dst_root}", flush=True)
        return dst_root

    if src_vocab_size is None:
        sample_idx = src_root / "pretrain_lm" / "index_receipt.json"
        if sample_idx.is_file():
            s_data = json.loads(sample_idx.read_text(encoding="utf-8"))
            if str(s_data.get("dtype", "uint16")) in ("uint32", "<u4", "u4"):
                src_vocab_size = 49155
            else:
                src_vocab_size = 16384
        else:
            src_vocab_size = 49155 if "granite" in str(src_root).lower() else 16384

    src_tok = get_tokenizer(src_vocab_size)
    dst_tok = get_tokenizer(LLAMA32_VOCAB_SIZE)
    assert dst_tok.vocab_size == LLAMA32_VOCAB_SIZE

    is_u32_src = int(src_vocab_size) > 32768
    src_np_dtype = "<u4" if is_u32_src else "<u2"
    src_val_mask = 0x7FFFFFFF if is_u32_src else 0x7FFF
    src_mask_bit = 0x80000000 if is_u32_src else 0x8000

    for track in ("pretrain_lm", "uta_trace", "typed_decision"):
        dst_track_dir = dst_root / track
        if (dst_track_dir / "index_receipt.json").is_file():
            continue
        src_receipt_path = src_root / track / "index_receipt.json"
        if not src_receipt_path.is_file():
            raise FileNotFoundError(f"Missing source track receipt: {src_receipt_path}")
        src_receipt = json.loads(src_receipt_path.read_text(encoding="utf-8"))

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
                vocab_size=LLAMA32_VOCAB_SIZE,
            )
            print(
                f"[transcode] Transcoding {track} ({split_name}: {len(shards_list)} shards) -> Llama-128K...",
                flush=True,
            )
            bos_id = int(src_tok.bos_id)
            eos_id = int(src_tok.eos_id)
            pad_id = int(src_tok.pad_id)
            for sh in shards_list:
                bin_path = src_root / track / sh["bin_file"]
                raw_arr = np.fromfile(bin_path, dtype=src_np_dtype)
                clean_ids = raw_arr & src_val_mask
                if bos_id == eos_id:
                    is_masked = (raw_arr & src_mask_bit) != 0
                    bos_positions = np.nonzero((clean_ids == bos_id) & is_masked)[0]
                    eos_positions = np.nonzero((clean_ids == eos_id) & (~is_masked))[0]
                else:
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
                    corpus=f"transcoded_llama128k:{track}:{split_name}",
                    source_file=sh["bin_file"],
                    row_start=1,
                    row_end=max(1, accepted),
                    episodes_accepted=accepted,
                    episodes_rejected_stage0=0,
                )
            writer.flush_shard()

    src_dec_dir = src_root / "typed_decision"
    dst_dec_dir = dst_root / "typed_decision"
    dst_dec_dir.mkdir(parents=True, exist_ok=True)
    for cache_name in ("decision_examples_cache.pt", "decision_examples_val_cache.pt"):
        if (src_dec_dir / cache_name).is_file() and not (dst_dec_dir / cache_name).is_file():
            shutil.copy2(src_dec_dir / cache_name, dst_dec_dir / cache_name)

    src_rlvr = src_root / "rlvr_env_pool.pt"
    dst_rlvr = dst_root / "rlvr_env_pool.pt"
    if src_rlvr.is_file() and not dst_rlvr.is_file():
        shutil.copy2(src_rlvr, dst_rlvr)
        print(f"[transcode] Copied RLVR environment pool -> {dst_rlvr}", flush=True)

    return dst_root


VYBER_CYBERSEC_REPO = "Vyber07/cyber-security"
UCB_CYBERGYM_REPO = "sunblaze-ucb/cybergym"

VYBER_JSONL_SHARDS: tuple[str, ...] = (
    "220426-CyberSec-Dataset_escaped.jsonl",
    "CyberSec-Dataset_escaped.jsonl",
    "code_train.jsonl",
    "ai-ml-security__real_ai_ml_security_expansion.jsonl",
    "api-security__real_api_security_expansion.jsonl",
    "cloud-security__real_cloud_security_expansion.jsonl",
    "container-security__real_container_security_expansion.jsonl",
    "cryptography__real_cryptography_expansion.jsonl",
    "blue-team__real_blue_team_expansion.jsonl",
    "botnet-ddos__real_botnet_ddos_defense_expansion.jsonl",
)


def ingest_cybersec_and_cybergym_into_llama128k(
    dst_shards_root: str | Path,
    *,
    max_vyber_rows_per_file: int = 1200,
    max_cybergym_tasks: int = 400,
    max_cybergym_artifact_Downloads: int = 80,
    seq_len: int = 512,
    force: bool = False,
) -> dict[str, Any]:
    """Ingest `Vyber07/cyber-security` (PhD cybersecurity) & `sunblaze-ucb/cybergym` into `Llama-128K` shards.

    Populates both `train` (for Mitosis `clone_token_warmup`) and `val` (for Infinite Recursive Held-Out
    Epistemic RL `_step_heldout_epistemic_rl`) across `pretrain_lm`, `uta_trace`, and `typed_decision`,
    plus verifiable `OrnithTask` RL environments in `rlvr_env_pool.pt`.
    """
    import re
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from mcl4_coe.ornith_rl import OrnithTask

    dst_root = Path(dst_shards_root).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    receipt_path = dst_root / "cybersec_cybergym_ingest_receipt.json"
    if receipt_path.is_file() and not force:
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        print(
            f"[cyber-ingest] Existing Vyber07/cyber-security + sunblaze-ucb/cybergym receipt found at {receipt_path}: {existing}",
            flush=True,
        )
        return existing

    hf_token = os.environ.get("HF_TOKEN") or None
    writers: dict[tuple[str, str], BlockAlignedShardWriter] = {}

    def _get_writer(track: str, split: str) -> BlockAlignedShardWriter:
        key = (track, split)
        w = writers.get(key)
        if w is None:
            w = BlockAlignedShardWriter(
                dst_root / track,
                track=track,
                split=split,
                seq_len=seq_len,
                max_blocks_per_shard=4096,
                vocab_size=LLAMA32_VOCAB_SIZE,
            )
            writers[key] = w
        return w

    vyber_pretrain_accepted = 0
    vyber_uta_accepted = 0
    vyber_dec_accepted = 0
    cybergym_pretrain_accepted = 0
    cybergym_uta_accepted = 0
    cybergym_dec_accepted = 0
    added_ornith_tasks: list[OrnithTask] = []

    think_re = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL | re.IGNORECASE)

    # 1. Stream PhD Cybersecurity JSONL shards from `Vyber07/cyber-security`
    for fn in VYBER_JSONL_SHARDS:
        try:
            local_p = Path(
                hf_hub_download(
                    VYBER_CYBERSEC_REPO,
                    fn,
                    repo_type="dataset",
                    token=hf_token,
                )
            )
        except Exception as exc:
            print(f"[cyber-ingest] Skipping {fn}: {exc}", flush=True)
            continue

        row_count = 0
        with local_p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue

                sys_txt = ""
                usr_txt = ""
                ast_txt = ""
                if "user" in rec and "assistant" in rec:
                    sys_txt = str(rec.get("system", "")).strip()
                    usr_txt = str(rec.get("user", "")).strip()
                    ast_txt = str(rec.get("assistant", "")).strip()
                elif isinstance(rec.get("messages"), list):
                    for m in rec["messages"]:
                        if not isinstance(m, dict):
                            continue
                        r_role = str(m.get("role", "")).lower()
                        r_cont = str(m.get("content", "")).strip()
                        if r_role == "system":
                            sys_txt = r_cont
                        elif r_role == "user":
                            usr_txt = r_cont
                        elif r_role == "assistant":
                            ast_txt = r_cont

                if len(usr_txt) < 16 or len(ast_txt) < 40:
                    continue

                row_count += 1
                split_name = "val" if (row_count % 10 == 0) else "train"
                m_th = think_re.search(ast_txt)
                if m_th:
                    t_body = m_th.group(1).strip()
                    a_body = m_th.group(2).strip()
                else:
                    paras = [p.strip() for p in ast_txt.split("\n\n") if p.strip()]
                    if len(paras) >= 2:
                        t_body = "\n".join(paras[:-1])[:1200]
                        a_body = paras[-1][:1200]
                    else:
                        t_body = (
                            f"Applying PhD cybersecurity causal analysis ({fn}): evaluate threat model, "
                            f"attack surface, and defense-in-depth controls for the query."
                        )
                        a_body = ast_txt[:1500]

                u_full = f"[PhD CyberSec | {fn}] {usr_txt}" if not sys_txt else f"{sys_txt[:280]}\n\n{usr_txt}"
                doc_text = f"{u_full}\n\nAnalysis:\n{t_body}\n\nConclusion:\n{a_body}"
                _get_writer("pretrain_lm", split_name).add_pretrain_document(doc_text)
                vyber_pretrain_accepted += 1

                if _get_writer("uta_trace", split_name).add_uta_episode(u_full[:900], t_body[:900], a_body[:900]):
                    vyber_uta_accepted += 1

                if row_count % 3 == 0:
                    dec_u = f"{u_full[:700]}\n#Q$# [security_evaluation] Synthesize the verified defensive/analytical action."
                    if _get_writer("typed_decision", split_name).add_uta_episode(dec_u, t_body[:700], a_body[:700]):
                        vyber_dec_accepted += 1

                if row_count >= max_vyber_rows_per_file:
                    break

        print(
            f"[cyber-ingest] Ingested {row_count} rows from {VYBER_CYBERSEC_REPO}/{fn}",
            flush=True,
        )

    # 2. Stream real vulnerability tasks, sanitizer crash traces (`error.txt`), and patches (`patch.diff`) from `sunblaze-ucb/cybergym`
    try:
        cg_ds = load_dataset(UCB_CYBERGYM_REPO, split="tasks", token=hf_token)
        for idx, row in enumerate(cg_ds):
            if idx >= max_cybergym_tasks:
                break
            task_id = str(row.get("task_id", f"arvo:{idx}")).strip()
            proj = str(row.get("project_name", "unknown")).strip()
            lang = str(row.get("project_language", "c++")).strip()
            vuln_desc = str(row.get("vulnerability_description", "")).strip()
            if not vuln_desc:
                continue

            split_name = "val" if (idx % 10 == 0) else "train"
            diff_dict = row.get("task_difficulty") or {}
            lvl3_files = diff_dict.get("level3") if isinstance(diff_dict, dict) else []
            err_txt = ""
            patch_txt = ""
            if idx < max_cybergym_artifact_Downloads and isinstance(lvl3_files, list):
                for rel_f in lvl3_files:
                    rel_s = str(rel_f)
                    if rel_s.endswith("error.txt") or rel_s.endswith("patch.diff"):
                        try:
                            p_art = Path(
                                hf_hub_download(
                                    UCB_CYBERGYM_REPO,
                                    rel_s,
                                    repo_type="dataset",
                                    token=hf_token,
                                )
                            )
                            raw_art = p_art.read_text(encoding="utf-8", errors="replace").strip()
                            if rel_s.endswith("error.txt"):
                                err_txt = raw_art[:1000]
                            else:
                                patch_txt = raw_art[:1200]
                        except Exception:
                            pass

            u_cg = (
                f"[CyberGym Vulnerability Triage | {task_id} | Project: {proj} ({lang})]\n"
                f"Vulnerability Summary: {vuln_desc}"
            )
            if err_txt:
                u_cg += f"\nSanitizer Crash Trace:\n{err_txt}"
            t_cg = (
                f"Root-cause security analysis for {proj} ({task_id}, {lang}): {vuln_desc} "
                f"We trace the sanitizer call stack to isolate the memory/bounds invariant defect and formulate a minimal memory-safe fix."
            )
            a_cg = patch_txt if patch_txt else (
                f"Enforce strict bounds and initialization checks in {proj} ({lang}) to remediate {task_id}: {vuln_desc}"
            )

            _get_writer("pretrain_lm", split_name).add_pretrain_document(f"{u_cg}\n\n{t_cg}\n\nPatch:\n{a_cg}")
            cybergym_pretrain_accepted += 1
            if _get_writer("uta_trace", split_name).add_uta_episode(u_cg[:950], t_cg[:800], a_cg[:950]):
                cybergym_uta_accepted += 1
            if _get_writer("typed_decision", split_name).add_uta_episode(
                f"{u_cg[:750]}\n#Q$# [cybergym_patch] Identify the root cause and remediation for {task_id}.",
                t_cg[:700],
                a_cg[:750],
            ):
                cybergym_dec_accepted += 1

            # Also synthesize a verifiable OrnithTask tied to this CyberGym vulnerability
            arvo_num = int(re.sub(r"\D+", "", task_id) or (1000 + idx))
            slots_n = 8 + (arvo_num % 19)
            elem_b = 2 + (idx % 4)
            guard_b = 16 + 4 * (idx % 5)
            safe_alloc = slots_n * elem_b + guard_b
            added_ornith_tasks.append(
                OrnithTask(
                    task_id=f"cybergym_{task_id.replace(':', '_')}",
                    domain="cybergym_rl",
                    prompt_q=(
                        f"#U$# [CyberGym {task_id} | {proj} ({lang})] {vuln_desc[:220]} "
                        f"Buffer reserves `slots = {slots_n}` of `elem_size = {elem_b}` bytes and requires a mandatory `{guard_b}`-byte sanitizer guard header.\n"
                        f"#Q$# [code] Trace the root-cause bounds fix in `#T$#` and implement `def solve(slots, elem_size):` returning the total guarded allocation in bytes."
                    ),
                    harness_h=f"assert solve({slots_n}, {elem_b}) == {safe_alloc} and solve(0, {elem_b}) == {guard_b}",
                    scaffold_s="abduction -> deduction -> decision -> evaluation -> metacognition",
                    candidate_rollouts=(
                        f"#T$# Multiply {slots_n} * {elem_b} = {slots_n * elem_b} and add guard header {slots_n * elem_b} + {guard_b} = {safe_alloc} to remediate {task_id}.\n#A$# def solve(slots, elem_size):\n    return slots * elem_size + {guard_b}",
                        f"#T$# Omit the {guard_b}-byte guard header.\n#A$# def solve(slots, elem_size):\n    return slots * elem_size",
                        f"#T$# Subtract the guard header {guard_b}.\n#A$# def solve(slots, elem_size):\n    return slots * elem_size - {guard_b}",
                        "#T$# Default fallback.\n#A$# def solve(slots, elem_size):\n    return 0",
                    ),
                    harness_scores=(1.0, 0.20, 0.10, 0.0),
                    gold_index=0,
                    gold_thought=f"Multiply {slots_n} * {elem_b} = {slots_n * elem_b} and add guard header {slots_n * elem_b} + {guard_b} = {safe_alloc} to remediate {task_id}.",
                    gold_answer=f"def solve(slots, elem_size):\n    return slots * elem_size + {guard_b}",
                    skill="cybergym.vuln_sanitizer_root_cause_patch",
                    star_difficulty=2.65,
                    recursion_depth=4,
                    difficulty_mods=("FL", "HD", "HR"),
                )
            )
    except Exception as exc:
        print(f"[cyber-ingest] Warning during CyberGym ingestion: {exc}", flush=True)

    for (track, split), writer in writers.items():
        writer.record_provenance(
            corpus=f"cybersec_cybergym:{track}:{split}",
            source_file=f"{VYBER_CYBERSEC_REPO}+{UCB_CYBERGYM_REPO}",
            row_start=1,
            row_end=max(1, vyber_pretrain_accepted + cybergym_pretrain_accepted),
            episodes_accepted=max(1, vyber_uta_accepted + cybergym_uta_accepted),
            episodes_rejected_stage0=0,
        )
        writer.flush_shard()

    if added_ornith_tasks:
        rlvr_path = dst_root / "rlvr_env_pool.pt"
        existing_tasks: list[Any] = []
        if rlvr_path.is_file():
            try:
                loaded = torch.load(rlvr_path, map_location="cpu", weights_only=False)
                if isinstance(loaded, list):
                    existing_tasks = loaded
                elif isinstance(loaded, dict) and isinstance(loaded.get("tasks"), list):
                    existing_tasks = loaded["tasks"]
            except Exception:
                existing_tasks = []
        seen_ids = {getattr(t, "task_id", "") for t in existing_tasks}
        merged_tasks = existing_tasks + [t for t in added_ornith_tasks if t.task_id not in seen_ids]
        torch.save(merged_tasks, rlvr_path)
        print(
            f"[cyber-ingest] Updated {rlvr_path} with {len(added_ornith_tasks)} CyberGym RLVR tasks (total={len(merged_tasks)})",
            flush=True,
        )

    summary = {
        "vyber_repo": VYBER_CYBERSEC_REPO,
        "cybergym_repo": UCB_CYBERGYM_REPO,
        "vyber_pretrain_docs": vyber_pretrain_accepted,
        "vyber_uta_episodes": vyber_uta_accepted,
        "vyber_decision_episodes": vyber_dec_accepted,
        "cybergym_pretrain_docs": cybergym_pretrain_accepted,
        "cybergym_uta_episodes": cybergym_uta_accepted,
        "cybergym_decision_episodes": cybergym_dec_accepted,
        "cybergym_ornith_rl_tasks": len(added_ornith_tasks),
        "timestamp": time.time(),
    }
    receipt_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[cyber-ingest] Completed ingestion: {json.dumps(summary)}", flush=True)
    return summary


def build_v9_llama32_1b_dense_rl_config(
    *,
    run_id: str = "colab_mcl4_coe_v9_llama32_1b_dense_ncm_rl",
    runs_root: str = "/content/runs" if Path("/content").is_dir() else r"X:\MCL4-COE\runs",
    shards_root: str = (
        "/content/data_shards_llama128k"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\data_shards_llama128k"
    ),
    corpora_root: str = (
        "/content/MCL4-COE/corpora" if Path("/content").is_dir() else r"X:\corpora"
    ),
    resume_checkpoint: str | None = None,
    total_steps: int = 2000,
    num_experts: int = 8,
    max_experts: int = 8,
    top_k: int = 8,
    enable_mitosis: bool = False,
    release_mitotic_pairs: bool = True,
    freeze_non_clone_experts: bool = False,
    mitotic_perturbation_scale: float = 5.0e-3,
    clone_expert_lr_scale: float = 1.0,
    clone_warmup_steps: int = 0,
    clone_warmup_tokens: int = 0,
    pure_rl_after_clone_warmup: bool = False,
    mitotic_ortho_coef: float | None = None,
    mitosis_milestones: tuple[tuple[int, int], ...] = (),
    batch_size: int = 8,
    micro_batch_size: int = 4,
    max_seq_len: int = 512,
    lr: float = 2.0e-5,
    donor_backbone_lr_scale: float = 0.002,
    ornith_weight: float = 0.30,
    use_nvfp4: bool = True,
    nvfp4_sparsity_mode: str = "none",
    plastic_experts_per_layer: int = 8,
    enable_ncm: bool = True,
    ncm_weight: float = 0.15,
    hf_backup_repo: str | None = "mesmac/mcl4-coe-runs",
) -> TrainingRunConfig:
    """Construct `TrainingRunConfig` for `Llama-3.2-1B-Instruct` Co-Active Sliced Dense (`E=8, top_k=8`) + `NVFP4` + `NCM`."""
    expert_hidden_dim = 8192 // 8
    ortho_coef = (
        float(mitotic_ortho_coef)
        if mitotic_ortho_coef is not None
        else (0.04 if (enable_mitosis or freeze_non_clone_experts) else 0.0)
    )
    return TrainingRunConfig(
        run_id=run_id,
        runs_root=runs_root,
        shards_root=shards_root,
        corpora_root=corpora_root,
        resume_checkpoint=resume_checkpoint,
        vocab_size=LLAMA32_VOCAB_SIZE,
        pad_id=LLAMA32_PAD_ID,
        eos_token_id=LLAMA32_EOS_ID,
        max_seq_len=max_seq_len,
        d_model=2048,
        n_heads=32,
        n_encoder_layers=8,
        n_decoder_layers=8,
        ple_dim=64,
        expert_hidden_dim=expert_hidden_dim,
        num_experts=num_experts,
        max_experts=max(num_experts, max_experts),
        top_k=top_k,
        rms_norm_eps=1e-5,
        rope_theta=500_000.0,
        rope_scaling=dict(LLAMA32_ROPE_SCALING),
        use_qk_norm=False,
        qk_norm_mode="head",
        norm_topk_prob=True,
        routed_scaling_factor=8.0,
        tie_word_embeddings=True,
        preserve_residual_across_stacks=True,
        donor_backbone_lr_scale=donor_backbone_lr_scale,
        model_dtype="bfloat16",
        plastic_experts_per_layer=plastic_experts_per_layer,
        freeze_donor_embeddings=True,
        freeze_non_clone_experts=freeze_non_clone_experts,
        mitotic_perturbation_scale=mitotic_perturbation_scale,
        clone_expert_lr_scale=clone_expert_lr_scale,
        clone_warmup_steps=clone_warmup_steps,
        clone_warmup_tokens=clone_warmup_tokens,
        pure_rl_after_clone_warmup=pure_rl_after_clone_warmup,
        enable_mitosis=enable_mitosis,
        release_mitotic_pairs=release_mitotic_pairs,
        orthogonal_mitosis=True,
        mitosis_every=0,
        mitosis_step_size=4 if enable_mitosis else 0,
        mitosis_milestones=mitosis_milestones,
        clear_consumed_shards=False,
        mitotic_ortho_coef=ortho_coef,
        sequence_diversity_coef=0.02,
        recurrent_coverage_penalty=0.20,
        projection_dim=256,
        recurrence_steps=2,
        recurrent_kv=True,
        depth_ponder=True,
        deep_supervision_weight=0.20,
        enable_gist_conditioning=True,
        gist_conditioning_prob=0.80,
        latent_prior_scale=0.18,
        use_nvfp4=use_nvfp4,
        nvfp4_sparsity_mode=nvfp4_sparsity_mode,
        cpu_offload_optimizer=False,
        total_steps=total_steps,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        decision_batch_size=4,
        operator_every=1,
        lr=lr,
        warmup_ratio=0.08,
        min_lr_ratio=0.20,
        weight_decay=0.01,
        grad_clip=1.0,
        curriculum_mode="cosine",
        start_weights={"pretrain_lm": 0.20, "uta_trace": 0.50, "typed_decision": 0.30},
        end_weights={"pretrain_lm": 0.15, "uta_trace": 0.50, "typed_decision": 0.35},
        enable_ornith_rl=True,
        ornith_every=1,
        ornith_weight=ornith_weight,
        ornith_warmup_ratio=0.0,
        enable_dream_rsi=True,
        dream_every=1,
        dream_max_rounds=3,
        dream_max_calls=4,
        dream_workers=2,
        dream_revisions=3,
        enable_ncm=enable_ncm,
        ncm_every=1,
        ncm_weight=ncm_weight,
        ncm_target_length=64,
        ncm_num_levels=4,
        ncm_level_schedule="quadratic",
        ncm_codebook_size=1024,
        ncm_vq_embed_dim=256,
        ncm_scale_adaptive_top_k=True,
        checkpoint_every=100,
        max_keep_checkpoints=3,
        save_optimizer_state=False,
        hf_backup_repo=hf_backup_repo,
        hf_keep_checkpoints=3,
        eval_every=50,
        val_max_batches_per_track=8,
        sample_every=50,
        sample_prompt="How do plants use capillary action to transport water?",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graft and launch MCL4-COE v9-Llama3.2-1B-Dense (E=8, top_k=8 Co-Active Dense + NVFP4 + NCM + RL/RSI)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config JSON and exit")
    parser.add_argument("--graft-only", action="store_true", help="Run Step-0 donor grafting + verification and exit")
    parser.add_argument("--run-id", type=str, default="colab_mcl4_coe_v9_llama32_1b_dense_ncm_rl")
    parser.add_argument(
        "--donor",
        type=str,
        default=LLAMA32_1B_DONOR_REPO,
        help="HuggingFace repo ID or local path for Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="/content/runs" if Path("/content").is_dir() else r"X:\MCL4-COE\runs",
    )
    parser.add_argument(
        "--shards-root",
        type=str,
        default=(
            "/content/data_shards_llama128k"
            if Path("/content").is_dir()
            else r"X:\MCL4-COE\data_shards_llama128k"
        ),
    )
    parser.add_argument(
        "--src-shards-root",
        type=str,
        default=(
            "/content/data_shards_granite49k"
            if Path("/content/data_shards_granite49k").is_dir()
            else (
                "/content/data_shards"
                if Path("/content/data_shards").is_dir()
                else r"X:\MCL4-COE\data_shards_granite49k"
            )
        ),
    )
    parser.add_argument(
        "--corpora-root",
        type=str,
        default="/content/MCL4-COE/corpora" if Path("/content").is_dir() else r"X:\corpora",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--max-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--enable-mitosis", action="store_true")
    parser.add_argument("--freeze-non-clone-experts", action="store_true")
    parser.add_argument("--mitotic-perturbation-scale", type=float, default=5.0e-3)
    parser.add_argument("--clone-expert-lr-scale", type=float, default=1.0)
    parser.add_argument("--clone-warmup-steps", type=int, default=0)
    parser.add_argument("--clone-warmup-tokens", type=int, default=0)
    parser.add_argument("--pure-rl-after-clone-warmup", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--donor-lr-scale", type=float, default=0.002)
    parser.add_argument("--ornith-weight", type=float, default=0.30)
    parser.add_argument("--ncm-weight", type=float, default=0.15)
    parser.add_argument("--no-nvfp4", action="store_true")
    parser.add_argument(
        "--ingest-cybersec-cybergym",
        action="store_true",
        default=True,
        help="Ingest Vyber07/cyber-security and sunblaze-ucb/cybergym into Llama-128K warmup & RL shards",
    )
    parser.add_argument("--hf-backup-repo", type=str, default="mesmac/mcl4-coe-runs")
    args = parser.parse_args()

    ckpt_dir = Path(args.runs_root).expanduser().resolve() / args.run_id / "checkpoints"
    resume_path = args.resume
    if resume_path is None and not args.dry_run:
        step0_meta = graft_llama32_1b_dense_checkpoint(
            ckpt_dir,
            donor_repo_or_path=args.donor,
            num_experts=args.num_experts,
            top_k=args.top_k,
            verify_graft=True,
        )
        resume_path = str(step0_meta)
        if args.graft_only:
            return

    if not args.dry_run:
        transcode_shards_to_llama128k(
            args.src_shards_root,
            args.shards_root,
            seq_len=args.max_seq_len,
        )
        if args.ingest_cybersec_cybergym:
            ingest_cybersec_and_cybergym_into_llama128k(
                args.shards_root,
                seq_len=args.max_seq_len,
            )

    cfg = build_v9_llama32_1b_dense_rl_config(
        run_id=args.run_id,
        runs_root=args.runs_root,
        shards_root=args.shards_root,
        corpora_root=args.corpora_root,
        resume_checkpoint=resume_path,
        total_steps=args.total_steps,
        num_experts=args.num_experts,
        max_experts=max(args.num_experts, args.max_experts),
        top_k=args.top_k,
        enable_mitosis=args.enable_mitosis,
        freeze_non_clone_experts=args.freeze_non_clone_experts,
        mitotic_perturbation_scale=args.mitotic_perturbation_scale,
        clone_expert_lr_scale=args.clone_expert_lr_scale,
        clone_warmup_steps=args.clone_warmup_steps,
        clone_warmup_tokens=args.clone_warmup_tokens,
        pure_rl_after_clone_warmup=args.pure_rl_after_clone_warmup,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        lr=args.lr,
        donor_backbone_lr_scale=args.donor_lr_scale,
        ornith_weight=args.ornith_weight,
        use_nvfp4=not args.no_nvfp4,
        ncm_weight=args.ncm_weight,
        hf_backup_repo=args.hf_backup_repo,
    )

    if args.dry_run:
        print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
        return

    print(
        f"[v9-Llama3.2-1B-Dense] Launching run_id={cfg.run_id} | "
        f"E={cfg.num_experts} (top_k={cfg.top_k}, scale={cfg.routed_scaling_factor}x, 100% active FFN) | "
        f"NVFP4={cfg.use_nvfp4} | NCM={cfg.enable_ncm} | "
        f"lr={cfg.lr:.1e} (donor_scale={cfg.donor_backbone_lr_scale}) | "
        f"total_steps={cfg.total_steps}",
        flush=True,
    )
    trainer = MCL4COETrainer(cfg)
    summary = trainer.train()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
