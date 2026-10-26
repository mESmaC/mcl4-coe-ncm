"""Launch script for Progressive Mitotic v6 (`colab_mcl4_coe_v6_mitotic`) on Colab RTX PRO 6000 Blackwell.

Implements high-velocity sparse scaling:
- Base trunk: `d_model=512, n_heads=8, n_encoder_layers=4, n_decoder_layers=4, ple_dim=128, expert_hidden_dim=1024`
  with `vocab_size=16384` (MorphCode-16K) and `max_seq_len=1024` (`markov_query_chunk_size=1024`).
- Starts at `N=4` experts (`77.71M` total / `52.55M` active parameters, `~333 ms/step`, `~49k tok/s`)
  for rapid Stage-1 convergence and verification on the sanitized `87.52M`-token MorphCode-16K shards,
  and expands capacity via Exact-Ratio Twin-Gated Mitosis (`N = 4 -> 8 -> 12 -> 16 -> ... -> 56`,
  reaching `228.8M` in Stage 1 and `631.5M - 732.2M+` in Stage 2+) while keeping active compute flat.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from colab_hf_cloud import run_cloud_training  # noqa: E402
from mcl4_coe import TrainingRunConfig  # noqa: E402


def build_v6_mitotic_config(
    *,
    run_id: str = "colab_mcl4_coe_v6_mitotic",
    runs_root: str = "/content/runs",
    shards_root: str = "/content/data_shards_morph16k",
    corpora_root: str = "/content/MCL4-COE/corpora",
    resume_checkpoint: str | None = None,
    num_experts: int = 4,
    max_experts: int = 16,
    mitosis_every: int = 1000,
    mitosis_step_size: int = 4,
    total_steps: int = 4000,
    lr: float = 1.2e-3,
) -> TrainingRunConfig:
    return TrainingRunConfig(
        run_id=run_id,
        runs_root=runs_root,
        shards_root=shards_root,
        corpora_root=corpora_root,
        resume_checkpoint=resume_checkpoint,
        vocab_size=16384,
        max_seq_len=1024,
        d_model=512,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=4,
        ple_dim=128,
        expert_hidden_dim=1024,
        num_experts=num_experts,
        top_k=2,
        enable_mitosis=True,
        max_experts=max_experts,
        mitosis_every=mitosis_every,
        mitosis_step_size=mitosis_step_size,
        projection_dim=128,
        recurrence_steps=2,
        recurrent_kv=True,
        depth_ponder=True,
        deep_supervision_weight=0.20,
        enable_gist_conditioning=True,
        gist_conditioning_prob=0.35,
        latent_prior_scale=0.35,
        use_nvfp4=False,
        cpu_offload_optimizer=False,
        total_steps=total_steps,
        batch_size=16,
        micro_batch_size=16,
        decision_batch_size=4,
        lr=lr,
        warmup_ratio=0.06,
        min_lr_ratio=0.10,
        weight_decay=0.01,
        grad_clip=1.0,
        curriculum_mode="cosine",
        start_weights={"pretrain_lm": 0.50, "uta_trace": 0.35, "typed_decision": 0.15},
        end_weights={"pretrain_lm": 0.25, "uta_trace": 0.55, "typed_decision": 0.20},
        enable_ornith_rl=True,
        ornith_every=8,
        ornith_weight=0.20,
        ornith_warmup_ratio=0.65,
        enable_dream_rsi=True,
        dream_every=10,
        dream_max_rounds=3,
        dream_max_calls=4,
        dream_workers=2,
        dream_revisions=3,
        checkpoint_every=250,
        eval_every=200,
        val_max_batches_per_track=8,
        sample_every=200,
        sample_prompt=(
            "Locate MCL4CEDBackbone in src/mcl4_coe/backbone.py and explain how "
            "recurrent depth pondering refines latent representations."
        ),
    )


def build_v6_mitotic_stage2_config(
    *,
    run_id: str = "colab_mcl4_coe_v6_mitotic",
    runs_root: str = "/content/runs",
    shards_root: str = "/content/data_shards_morph16k",
    corpora_root: str = "/content/MCL4-COE/corpora",
    resume_checkpoint: str = "/content/runs/colab_mcl4_coe_v6_mitotic/checkpoints/step_04000.pt",
    num_experts: int = 24,
    max_experts: int = 56,
    mitosis_every: int = 800,
    mitosis_step_size: int = 8,
    total_steps: int = 8000,
    lr: float = 1.2e-3,
) -> TrainingRunConfig:
    cfg = build_v6_mitotic_config(
        run_id=run_id,
        runs_root=runs_root,
        shards_root=shards_root,
        corpora_root=corpora_root,
        resume_checkpoint=resume_checkpoint,
        num_experts=num_experts,
        max_experts=max_experts,
        mitosis_every=mitosis_every,
        mitosis_step_size=mitosis_step_size,
        total_steps=total_steps,
        lr=lr,
    )
    cfg.min_lr_ratio = 0.25
    cfg.seed = 43
    return cfg


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch Progressive Mitotic v6 training")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()

    cfg = build_v6_mitotic_stage2_config() if args.stage == 2 else build_v6_mitotic_config()
    print(
        f"[v6-Mitotic Stage {args.stage}] Starting {cfg.run_id} (E={cfg.num_experts} -> {cfg.max_experts}) "
        f"on {cfg.shards_root} for {cfg.total_steps} steps...",
        flush=True,
    )
    manifest = run_cloud_training(
        cfg,
        hf_run_repo="mESmaC/mcl4-coe-runs",
        drive_root=None,
        resume_from_cloud=True,
    )
    print(f"[v6-Mitotic Stage {args.stage}] Run complete!", manifest.get("step"), flush=True)


if __name__ == "__main__":
    main()

