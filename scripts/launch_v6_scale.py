"""Launch script for v6-Scale (550.01M -> 776.55M params) on Colab RTX PRO 6000 Blackwell."""

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


def build_v6_scale_config(
    *,
    runs_root: str = "/content/runs",
    shards_root: str = "/content/data_shards_morph16k",
    corpora_root: str = "/content/MCL4-COE/corpora",
    resume_checkpoint: str | None = None,
) -> TrainingRunConfig:
    return TrainingRunConfig(
        run_id="colab_mcl4_coe_v6_scale",
        runs_root=runs_root,
        shards_root=shards_root,
        corpora_root=corpora_root,
        resume_checkpoint=resume_checkpoint,
        vocab_size=16384,
        max_seq_len=1024,
        d_model=768,
        n_heads=12,
        n_encoder_layers=8,
        n_decoder_layers=8,
        ple_dim=256,
        expert_hidden_dim=1536,
        num_experts=8,
        top_k=2,
        enable_mitosis=True,
        max_experts=12,
        mitosis_every=1000,
        mitosis_step_size=2,
        projection_dim=192,
        recurrence_steps=2,
        recurrent_kv=True,
        depth_ponder=True,
        deep_supervision_weight=0.20,
        enable_gist_conditioning=True,
        gist_conditioning_prob=0.35,
        latent_prior_scale=0.35,
        use_nvfp4=False,
        cpu_offload_optimizer=False,
        total_steps=6000,
        batch_size=16,
        micro_batch_size=8,
        decision_batch_size=4,
        lr=8e-4,
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


def main() -> None:
    cfg = build_v6_scale_config()
    print(
        f"[v6-Scale] Starting {cfg.run_id} (550.01M -> 776.55M params) "
        f"on {cfg.shards_root} for {cfg.total_steps} steps...",
        flush=True,
    )
    manifest = run_cloud_training(
        cfg,
        hf_run_repo="mESmaC/mcl4-coe-runs",
        resume_from_cloud=True,
    )
    print("[v6-Scale] Run complete!", manifest.get("step"), flush=True)


if __name__ == "__main__":
    main()
