r"""Pure RL + Dream-RSI Launcher for Unsplit Granite 3.1 1B-A400M (`MCL4-COE v7-Granite-E32-Pure-RL`).

Runs pure verified-execution GRPO (`OrnithSelfImprover`, `ornith_every=1`), `Dream-RSI` (`dream_every=1`),
and 9-Operator `COE` (`operator_every=1`) on the pre-mitosis 32-expert Granite 3.1 checkpoint
(`colab_mcl4_coe_v7_granite_slingshot_r2/checkpoints/step_00400.pt`, `1.45B` parameters, `E=32`, `top_k=8`)
with **zero mitotic expansion (`enable_mitosis=False`)** and strict donor trunk protection:
- `lr = 2.0e-5`, `donor_backbone_lr_scale = 0.002` (`4.0e-8` effective LR on donor experts/attention)
- `freeze_donor_embeddings = True` (locks vocabulary & output head calibration)
- `enable_gist_conditioning = False` (no artificial input corruption)
- Preserves exact Granite 3.1 muP multipliers (`attention_multiplier=0.015625`, `embedding_multiplier=12.0`,
  `residual_multiplier=0.22`, `logits_scaling=6.0`, `rms_norm_eps=1e-6`, `rope_theta=1500000.0`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mcl4_coe.trainer import MCL4COETrainer, TrainingRunConfig

GRANITE_VOCAB_SIZE = 49155


def build_v7_granite_pure_rl_config(
    *,
    run_id: str = "colab_mcl4_coe_v7_granite_e32_pure_rl",
    runs_root: str = "/content/runs" if Path("/content").is_dir() else r"X:\MCL4-COE\runs",
    shards_root: str = (
        "/content/data_shards_granite49k"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\data_shards_granite49k"
    ),
    corpora_root: str = (
        "/content/MCL4-COE/corpora" if Path("/content").is_dir() else r"X:\corpora"
    ),
    resume_checkpoint: str = (
        "/content/runs/colab_mcl4_coe_v7_granite_slingshot_r2/checkpoints/step_00400.meta.json"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\runs\colab_mcl4_coe_v7_granite_slingshot_r2\checkpoints\step_00400.meta.json"
    ),
    total_steps: int = 1400,
    num_experts: int = 32,
    top_k: int = 8,
    batch_size: int = 8,
    micro_batch_size: int = 8,
    max_seq_len: int = 512,
    lr: float = 2.0e-5,
    donor_backbone_lr_scale: float = 0.002,
    ornith_weight: float = 0.30,
    use_nvfp4: bool = True,
    nvfp4_sparsity_mode: str = "none",
    plastic_experts_per_layer: int = 8,
    hf_backup_repo: str | None = "mesmac/mcl4-coe-runs",
) -> TrainingRunConfig:
    """Construct pure RL + Dream-RSI `TrainingRunConfig` for `Granite 3.1 1B-A400M` (`E=32`, no mitosis)."""
    return TrainingRunConfig(
        run_id=run_id,
        runs_root=runs_root,
        shards_root=shards_root,
        corpora_root=corpora_root,
        resume_checkpoint=resume_checkpoint,
        vocab_size=GRANITE_VOCAB_SIZE,
        pad_id=4,
        eos_token_id=0,
        max_seq_len=max_seq_len,
        d_model=1024,
        n_heads=16,
        n_encoder_layers=12,
        n_decoder_layers=12,
        ple_dim=64,
        expert_hidden_dim=512,
        num_experts=num_experts,
        max_experts=num_experts,
        top_k=top_k,
        rms_norm_eps=1e-6,
        rope_theta=1_500_000.0,
        use_qk_norm=False,
        norm_topk_prob=True,
        attention_multiplier=0.015625,
        embedding_multiplier=12.0,
        residual_multiplier=0.22,
        logits_scaling=6.0,
        preserve_residual_across_stacks=True,
        donor_backbone_lr_scale=donor_backbone_lr_scale,
        plastic_experts_per_layer=plastic_experts_per_layer,
        freeze_donor_embeddings=True,
        enable_mitosis=False,
        release_mitotic_pairs=False,
        mitosis_every=0,
        mitosis_step_size=0,
        mitosis_milestones=(),
        clear_consumed_shards=False,
        mitotic_ortho_coef=0.0,
        sequence_diversity_coef=0.02,
        recurrent_coverage_penalty=0.20,
        projection_dim=256,
        recurrence_steps=1,
        recurrent_kv=False,
        depth_ponder=False,
        deep_supervision_weight=0.20,
        enable_gist_conditioning=False,
        gist_conditioning_prob=0.50,
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
        description="Launch Pure RL + Dream-RSI on MCL4-COE Granite 3.1 (No Mitosis)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config JSON and exit")
    parser.add_argument("--run-id", type=str, default="colab_mcl4_coe_v7_granite_e32_pure_rl")
    parser.add_argument(
        "--runs-root",
        type=str,
        default="/content/runs" if Path("/content").is_dir() else r"X:\MCL4-COE\runs",
    )
    parser.add_argument(
        "--shards-root",
        type=str,
        default="/content/data_shards_granite49k"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\data_shards_granite49k",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="/content/runs/colab_mcl4_coe_v7_granite_slingshot_r2/checkpoints/step_00400.meta.json"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\runs\colab_mcl4_coe_v7_granite_slingshot_r2\checkpoints\step_00400.meta.json",
    )
    parser.add_argument("--total-steps", type=int, default=1400)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--donor-lr-scale", type=float, default=0.002)
    parser.add_argument("--ornith-weight", type=float, default=0.30)
    parser.add_argument("--hf-backup-repo", type=str, default="mesmac/mcl4-coe-runs")
    args = parser.parse_args()

    cfg = build_v7_granite_pure_rl_config(
        run_id=args.run_id,
        runs_root=args.runs_root,
        shards_root=args.shards_root,
        resume_checkpoint=args.resume,
        total_steps=args.total_steps,
        num_experts=args.num_experts,
        top_k=args.top_k,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        lr=args.lr,
        donor_backbone_lr_scale=args.donor_lr_scale,
        ornith_weight=args.ornith_weight,
        hf_backup_repo=args.hf_backup_repo or None,
    )
    print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return

    def _on_step(rec: dict[str, Any]) -> None:
        st = int(rec.get("step", 0))
        if st == 401 or st % 10 == 0 or "val_bpb" in rec:
            orn_str = (
                f" | orn_rw={rec.get('ornith_harness_reward', 0.0):.3f} "
                f"pass={rec.get('ornith_pass_rate', 0.0):.3f} "
                f"drm_cyc={rec.get('dream_cycle', 0)} "
                f"drm_acc={rec.get('dream_accepted_revisions', 0)}"
                if "ornith_harness_reward" in rec
                else ""
            )
            val_str = (
                f" | val_bpb={rec.get('val_bpb'):.4f} val_mcqa={rec.get('val_mcqa_acc', 0.0):.4f}"
                if "val_bpb" in rec
                else ""
            )
            print(
                f"[step {st:4d}/{cfg.total_steps}] loss={rec.get('total_loss'):.4f} "
                f"(lm={rec.get('lm_loss'):.4f} bpb={rec.get('train_bpb'):.4f} dec={rec.get('decision_loss'):.4f}) | "
                f"E={rec.get('num_experts')} k={rec.get('top_k')} fp4={rec.get('nvfp4_gemms', 0)} | "
                f"{rec.get('tok_per_s'):.0f} tok/s | vram={rec.get('vram_allocated_mb'):.0f}MB"
                f"{orn_str}{val_str}",
                flush=True,
            )

    trainer = MCL4COETrainer(cfg, on_step_callback=_on_step)
    trainer.train()


if __name__ == "__main__":
    main()
