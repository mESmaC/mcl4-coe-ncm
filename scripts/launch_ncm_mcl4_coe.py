"""Launch & Benchmark Next-Concept Modeling (NCM) on MCL4-COE (`ncm` branch).

Supports:
1. Donor-Anchored or Fresh `MCL4-COE` + `NCMSystem` initialization.
2. Stage-1 Multiscale Residual VQ-VAE (`DonorAnchoredMultiscaleVQVAE`) warmup with k-means
   codebook seeding, Laplace-smoothed EMA updates, and nested dropout (`fine_drop`).
3. Stage-2 Single-Pass Block-Diagonal Next-Concept Generator (`NCMGeneratorAdapter`) training
   with distance-tied codebook heads (`TiedCodebookHead`), multiscale `COE` `Abstraction`
   alignment, and `Ornith-2.5` (`pp = p * 0.95^(n-1)`) hierarchical GRPO (`step_ncm_self_improvement`).
4. Coarse-to-fine $K$-step parallel generation with scale-adaptive `SparseMoE` `top_k(k)`,
   mid-hierarchy 9-operator `COE` verification, and level-by-level concept decoding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mcl4_coe import (
    COEConfig,
    CognitiveOperatorEnsemble,
    MCL4CEDBackbone,
    MCL4CEDConfig,
    NCMConfig,
    OrnithSelfImprover,
    get_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Next-Concept Modeling (NCM) on MCL4-COE")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional MCL4-COE .pt checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--target-length", type=int, default=32)
    parser.add_argument("--num-levels", type=int, default=4)
    parser.add_argument("--level-schedule", type=str, default="quadratic")
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--vq-embed-dim", type=int, default=64)
    parser.add_argument("--vqvae-steps", type=int, default=8, help="Stage-1 VQ-VAE warmup steps")
    parser.add_argument("--ncm-steps", type=int, default=12, help="Stage-2 NCM + Ornith-2.5 steps")
    parser.add_argument("--ornith-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--prompt",
        type=str,
        default="How do plants use capillary action to transport water?",
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42)

    if args.checkpoint and Path(args.checkpoint).is_file():
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        b_cfg = MCL4CEDConfig(**raw["backbone_config"])
        c_cfg = COEConfig(**raw["coe_config"])
        coe = CognitiveOperatorEnsemble(MCL4CEDBackbone(b_cfg), c_cfg)
        coe.load_state_dict(raw["model_state"], strict=False)
    else:
        b_cfg = MCL4CEDConfig(
            vocab_size=260,
            pad_id=256,
            d_model=args.d_model,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            ple_dim=32,
            expert_hidden_dim=args.d_model * 2,
            num_experts=args.num_experts,
            top_k=args.top_k,
        )
        c_cfg = COEConfig(projection_dim=32, recurrence_steps=2)
        coe = CognitiveOperatorEnsemble(MCL4CEDBackbone(b_cfg), c_cfg)

    coe = coe.to(device)
    tokenizer = get_tokenizer(b_cfg.vocab_size)

    ncm_cfg = NCMConfig(
        target_length=args.target_length,
        num_levels=args.num_levels,
        level_schedule=args.level_schedule,
        codebook_size=args.codebook_size,
        vq_embed_dim=args.vq_embed_dim,
        d_model=b_cfg.d_model,
        vocab_size=b_cfg.vocab_size,
        pad_id=b_cfg.pad_id,
        scale_adaptive_top_k=True,
        coarse_top_k=min(b_cfg.num_experts, max(4, b_cfg.top_k * 2)),
        mid_top_k=min(b_cfg.num_experts, max(2, b_cfg.top_k)),
        fine_top_k=min(b_cfg.num_experts, b_cfg.top_k),
    )
    ncm_sys = coe.create_ncm_system(ncm_cfg)

    print(
        f"[NCM // MCL4-COE] Initialized NCMSystem on {device} | "
        f"levels={ncm_cfg.levels} | codebook={ncm_cfg.codebook_size}x{ncm_cfg.vq_embed_dim}",
        flush=True,
    )

    sample_texts = [
        (
            "#U$# How do plants use capillary action to transport water?",
            "#T$# Cohesion and adhesion pull water molecules up xylem vessels.\n#A$# Plants transport water via xylem capillary action and transpiration pull.",
        ),
        (
            "#U$# Compute 6 * 7 and verify the result.",
            "#T$# 6 * 7 = 42. Check 42 / 6 = 7.\n#A$# 42",
        ),
        (
            "#U$# Explain why coarse-to-fine concept generation improves planning.",
            "#T$# Coarse concept scales lock in global sentence structure before token realization.\n#A$# Coarse scales plan global semantics before fine lexical refinement.",
        ),
        (
            "#U$# What is the derivative of x^3 + 2x?",
            "#T$# d/dx(x^3) = 3x^2 and d/dx(2x) = 2.\n#A$# 3x^2 + 2",
        ),
    ]

    p_rows = []
    t_rows = []
    for p_txt, t_txt in sample_texts:
        p_toks = tokenizer.encode(p_txt, add_bos=True, add_eos=False)[: args.target_length]
        p_toks = p_toks + [b_cfg.pad_id] * max(0, args.target_length - len(p_toks))
        t_toks = tokenizer.encode(t_txt, add_bos=False, add_eos=True)[: args.target_length]
        t_toks = t_toks + [b_cfg.pad_id] * max(0, args.target_length - len(t_toks))
        p_rows.append(p_toks)
        t_rows.append(t_toks)

    prompt_batch = torch.tensor(p_rows, dtype=torch.long, device=device)
    target_batch = torch.tensor(t_rows, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(
        [p for p in coe.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    improver = OrnithSelfImprover(
        projection_dim=c_cfg.projection_dim,
        max_seq_len=128,
        seed=42,
        enable_dream_rsi=True,
        dream_every=2,
    )

    # 1. Stage-1 VQ-VAE Warmup
    t0 = time.perf_counter()
    vq_history = []
    for step in range(1, args.vqvae_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        vq_out = ncm_sys.forward_tokenizer_step(target_batch)
        vq_out.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(coe.parameters(), 1.0)
        optimizer.step()
        vq_history.append(
            {
                "step": step,
                "recon_loss": round(float(vq_out.recon_loss.item()), 4),
                "vq_loss": round(float(vq_out.vq_loss.item()), 4),
                "recon_acc": round(float(vq_out.recon_accuracy), 4),
                "utilization": round(float(vq_out.quantizer_out.global_utilization), 4),
            }
        )

    # 2. Stage-2 Single-Pass All-Level NCM + Ornith-2.5 Hierarchical RL
    ncm_history = []
    for step in range(1, args.ncm_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        ncm_out = ncm_sys.forward_training_step(prompt_batch, target_batch)
        loss = ncm_out.total_loss
        ornith_pp = 0.0
        if args.ornith_every > 0 and step % args.ornith_every == 0:
            ornith_out = improver.step_ncm_self_improvement(ncm_sys, batch_size=2)
            loss = loss + 0.15 * ornith_out.loss
            ornith_pp = float(ornith_out.total_pp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(coe.parameters(), 1.0)
        optimizer.step()
        ncm_history.append(
            {
                "step": step,
                "total_loss": round(float(ncm_out.total_loss.item()), 4),
                "ncm_ce_loss": round(float(ncm_out.ncm_ce_loss.item()), 4),
                "recon_acc": round(float(ncm_out.recon_accuracy), 4),
                "per_level_acc": list(ncm_out.per_level_acc),
                "ornith_total_pp": round(ornith_pp, 4),
            }
        )

    # 3. Coarse-to-Fine Generation Demo
    coe.eval()
    gen_prompt_ids = torch.tensor(
        [tokenizer.encode(f"#U$# {args.prompt}", add_bos=True, add_eos=False)],
        dtype=torch.long,
        device=device,
    )
    gen_out = ncm_sys.generate(
        gen_prompt_ids,
        tokenizer=tokenizer,
        target_length=args.target_length,
        temperature=0.0,
        enable_mid_hierarchy_coe=True,
        enable_early_exit=False,
        decode_all_levels=True,
    )
    elapsed = round(time.perf_counter() - t0, 3)

    summary = {
        "levels": list(ncm_cfg.levels),
        "elapsed_s": elapsed,
        "vqvae_final": vq_history[-1] if vq_history else {},
        "ncm_final": ncm_history[-1] if ncm_history else {},
        "generation": {
            "prompt": args.prompt,
            "levels_executed": gen_out.levels_executed,
            "per_level_active_top_k": gen_out.per_level_active_top_k,
            "per_level_metacognition": gen_out.per_level_metacognition,
            "per_level_deduction": gen_out.per_level_deduction,
            "level_decoded_texts": gen_out.level_decoded_texts,
            "final_text": gen_out.text,
        },
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
