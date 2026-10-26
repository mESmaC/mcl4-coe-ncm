from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import torch

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mcl4_coe.trainer import MCL4COETrainer, TrainingRunConfig

GRANITE_VOCAB_SIZE = 49155


def build_v7_granite_ncm_pure_rl_config(
    *,
    run_id: str = "colab_mcl4_coe_v7_granite_ncm_e32_pure_rl",
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
        "/content/runs/colab_mcl4_coe_v7_granite_e32_pure_rl/checkpoints/step_05000.meta.json"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\runs\colab_mcl4_coe_v7_granite_e32_pure_rl\checkpoints\step_05000.meta.json"
    ),
    total_steps: int = 5500,
    num_experts: int = 32,
    top_k: int = 8,
    batch_size: int = 8,
    micro_batch_size: int = 8,
    max_seq_len: int = 512,
    lr: float = 2.0e-5,
    donor_backbone_lr_scale: float = 0.002,
    ornith_weight: float = 0.30,
    ncm_weight: float = 0.12,
    ncm_target_length: int = 64,
    ncm_num_levels: int = 6,
    ncm_codebook_size: int = 1024,
    ncm_vq_embed_dim: int = 256,
    use_nvfp4: bool = True,
    nvfp4_sparsity_mode: str = "none",
    plastic_experts_per_layer: int = 8,
    hf_backup_repo: str | None = "YYCore/mcl4-coe-runs",
) -> TrainingRunConfig:
    """Construct NCM + Pure RL + Dream-RSI `TrainingRunConfig` for `Granite 3.1 1B-A400M` (`E=32`, no mitosis)."""
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
        mitotic_ortho_coef=0.02 if num_experts > 32 else 0.0,
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
        warmup_ratio=0.05,
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
        enable_ncm=True,
        ncm_every=1,
        ncm_weight=ncm_weight,
        ncm_target_length=ncm_target_length,
        ncm_num_levels=ncm_num_levels,
        ncm_level_schedule="quadratic",
        ncm_codebook_size=ncm_codebook_size,
        ncm_vq_embed_dim=ncm_vq_embed_dim,
        ncm_scale_adaptive_top_k=True,
        checkpoint_every=50,
        max_keep_checkpoints=3,
        save_optimizer_state=False,
        hf_backup_repo=hf_backup_repo,
        hf_keep_checkpoints=4,
        eval_every=25,
        val_max_batches_per_track=8,
        sample_every=25,
        sample_prompt="How do plants use capillary action to transport water?",
    )


def fit_ncm_hierarchy_on_frozen_donor(
    trainer: MCL4COETrainer,
    *,
    fit_steps: int = 250,
    fit_lr: float = 1.5e-3,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Fit Stage-1 `DonorAnchoredMultiscaleVQVAE` + Stage-2 `NCMGeneratorAdapter` on frozen Granite 3.1.

    1. Freezes 100% of `backbone.*` and existing `COE` heads (`requires_grad=False`), leaving only
       `ncm_system.vqvae.*` and `ncm_system.generator.*` trainable.
    2. Initializes the $K$-level residual VQ codebooks via K-means on real Granite 3.1 token slices.
    3. Trains `ncm_system.forward_training_step` for `fit_steps` with cosine-annealed AdamW (`fit_lr`)
       so the multiscale concept codebooks and coarse-to-fine generator lock onto the Granite 3.1
       manifold before joint pure RL begins.
    """
    ncm_sys = trainer.ncm_system
    if ncm_sys is None or fit_steps <= 0:
        return {}

    device = trainer.device
    ncm_len = int(ncm_sys.config.target_length)
    pad_id = int(trainer.config.pad_id)
    use_amp = device.type == "cuda"

    # Collect token windows directly from MultiTrackMixtureDataset
    n_ds = len(trainer.dataset)

    def _sample_ncm_pair(step_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p_list: list[torch.Tensor] = []
        t_list: list[torch.Tensor] = []
        for b in range(batch_size):
            idx = (step_idx * batch_size + b * 97) % max(1, n_ds)
            item = trainer.dataset[idx]
            inp = item["input_ids"]
            valid_mask = inp != pad_id
            valid_toks = inp[valid_mask]
            if valid_toks.numel() >= ncm_len * 2:
                max_offset = max(0, int(valid_toks.numel()) - ncm_len * 2)
                off = (step_idx * 13 + b * 7) % (max_offset + 1)
                p_row = valid_toks[off : off + ncm_len]
                t_row = valid_toks[off + ncm_len : off + ncm_len * 2]
            else:
                p_row = inp[:ncm_len]
                t_row = inp[ncm_len : ncm_len * 2]
                if t_row.numel() < ncm_len:
                    t_row = torch.cat(
                        [t_row, torch.full((ncm_len - t_row.numel(),), pad_id, dtype=inp.dtype)]
                    )
            p_list.append(p_row)
            t_list.append(t_row)
        return (
            torch.stack(p_list, dim=0).to(device, non_blocking=True),
            torch.stack(t_list, dim=0).to(device, non_blocking=True),
        )

    # Save original requires_grad state and freeze everything except ncm_system.*
    orig_req_grad = {name: p.requires_grad for name, p in trainer.model.named_parameters()}
    ncm_params: list[torch.nn.Parameter] = []
    for name, p in trainer.model.named_parameters():
        is_ncm = name.startswith("ncm_system.")
        p.requires_grad_(is_ncm)
        if is_ncm:
            ncm_params.append(p)

    # Step 0: Multiscale K-Means initialization on real Granite token embeddings
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        _, init_targets = _sample_ncm_pair(0)
        _, more_targets = _sample_ncm_pair(1)
        kmeans_targets = torch.cat([init_targets, more_targets], dim=0)
        z_init = ncm_sys.vqvae.encode_pre_quant(kmeans_targets)
        ncm_sys.vqvae.vq.init_codebooks_from_z(z_init.float())

    fit_opt = torch.optim.AdamW(
        ncm_params,
        lr=fit_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=(device.type == "cuda"),
    )

    print(
        f"[NCM Fit] Initialized K={ncm_sys.config.num_levels} codebooks ({ncm_sys.config.levels}) via K-means. "
        f"Fitting {sum(p.numel() for p in ncm_params):,} NCM adapter params for {fit_steps} steps "
        f"(Granite 3.1 donor trunk 100% frozen)...",
        flush=True,
    )

    last_info: dict[str, Any] = {}
    for s in range(1, fit_steps + 1):
        # Re-run K-means at step 25 once enc_proj has aligned with dec_out_proj
        if s == 25:
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                _, km_t1 = _sample_ncm_pair(s)
                _, km_t2 = _sample_ncm_pair(s + 1)
                z_re = ncm_sys.vqvae.encode_pre_quant(torch.cat([km_t1, km_t2], dim=0))
                ncm_sys.vqvae.vq.init_codebooks_from_z(z_re.float())

        prog = s / float(max(1, fit_steps))
        cur_lr = fit_lr * (0.20 + 0.80 * 0.5 * (1.0 + float(torch.cos(torch.tensor(prog * 3.14159265)).item())))
        for g in fit_opt.param_groups:
            g["lr"] = cur_lr

        fit_opt.zero_grad(set_to_none=True)
        p_ids, t_ids = _sample_ncm_pair(s)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            ncm_out = ncm_sys.forward_training_step(
                p_ids,
                t_ids,
                train_vqvae_jointly=True,
                include_coe_abstraction_loss=True,
            )
            loss = ncm_out.total_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ncm_params, 1.0)
        fit_opt.step()

        if s == 1 or s % 25 == 0 or s == fit_steps:
            elapsed = time.perf_counter() - t0
            last_info = {
                "fit_step": s,
                "ncm_loss": round(float(ncm_out.total_loss.detach().item()), 4),
                "ncm_ce_loss": round(float(ncm_out.ncm_ce_loss.item()), 4),
                "vq_recon_loss": round(float(ncm_out.vq_recon_loss.item()), 4),
                "recon_acc": round(float(ncm_out.recon_accuracy), 4),
                "codebook_util": round(float(ncm_out.codebook_utilization), 4),
                "per_level_acc": list(ncm_out.per_level_acc),
                "elapsed_s": round(elapsed, 1),
            }
            print(
                f"  [NCM Fit {s:3d}/{fit_steps}] loss={last_info['ncm_loss']:.4f} "
                f"(ce={last_info['ncm_ce_loss']:.4f} vq_rec={last_info['vq_recon_loss']:.4f}) | "
                f"recon_acc={last_info['recon_acc']:.4f} cb_util={last_info['codebook_util']:.4f} | "
                f"lvl_acc={last_info['per_level_acc']} | {elapsed:.1f}s",
                flush=True,
            )

    # Restore original plasticity policy & rebuild joint optimizer for Pure RL
    for name, p in trainer.model.named_parameters():
        if name in orig_req_grad:
            p.requires_grad_(orig_req_grad[name])
    trainer._apply_plasticity_policy(getattr(trainer, "_ema_expert_loads", None))
    trainer._create_optimizer()
    return last_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit NCM Hierarchy & Launch Pure Ornith-2.5 RL + Dream-RSI on Granite 3.1 (No Mitosis)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config JSON and exit")
    parser.add_argument("--run-id", type=str, default="colab_mcl4_coe_v7_granite_ncm_e32_pure_rl")
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
        default="/content/runs/colab_mcl4_coe_v7_granite_e32_pure_rl/checkpoints/step_05000.meta.json"
        if Path("/content").is_dir()
        else r"X:\MCL4-COE\runs\colab_mcl4_coe_v7_granite_e32_pure_rl\checkpoints\step_05000.meta.json",
    )
    parser.add_argument("--total-steps", type=int, default=5500)
    parser.add_argument("--fit-steps", type=int, default=250)
    parser.add_argument("--fit-lr", type=float, default=1.5e-3)
    parser.add_argument(
        "--force-fit",
        action="store_true",
        help="Run NCM fitting even when resuming from an existing NCM run checkpoint",
    )
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--donor-lr-scale", type=float, default=0.002)
    parser.add_argument("--ornith-weight", type=float, default=0.30)
    parser.add_argument("--ncm-weight", type=float, default=0.12)
    parser.add_argument("--hf-backup-repo", type=str, default="YYCore/mcl4-coe-runs")
    args = parser.parse_args()

    # Automatically resume from latest checkpoint in <runs_root>/<run_id>/checkpoints if present
    target_ckpt_dir = Path(args.runs_root).resolve() / args.run_id / "checkpoints"
    existing_ckpts = sorted(target_ckpt_dir.glob("step_*.meta.json")) if target_ckpt_dir.is_dir() else []
    resume_ckpt = args.resume
    fit_steps = int(args.fit_steps)
    if existing_ckpts and not args.force_fit:
        resume_ckpt = str(existing_ckpts[-1])
        fit_steps = 0
        if not args.dry_run:
            print(
                f"[Auto-Resume] Detected existing NCM checkpoint {resume_ckpt}; "
                f"resuming with fit_steps={fit_steps}.",
                flush=True,
            )

    cfg = build_v7_granite_ncm_pure_rl_config(
        run_id=args.run_id,
        runs_root=args.runs_root,
        shards_root=args.shards_root,
        resume_checkpoint=resume_ckpt,
        total_steps=args.total_steps,
        num_experts=args.num_experts,
        top_k=args.top_k,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        lr=args.lr,
        donor_backbone_lr_scale=args.donor_lr_scale,
        ornith_weight=args.ornith_weight,
        ncm_weight=args.ncm_weight,
        hf_backup_repo=args.hf_backup_repo or None,
    )
    print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return

    # Copy existing dream_state.json from step_05000 run if present so Dream-RSI continues seamlessly
    res_path = Path(resume_ckpt).resolve()
    donor_dream = res_path.parent.parent / "dream" / "dream_state.json"
    target_dream = Path(args.runs_root).resolve() / args.run_id / "dream" / "dream_state.json"
    if donor_dream.is_file() and not target_dream.is_file():
        target_dream.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(donor_dream, target_dream)

    def _on_step(rec: dict[str, Any]) -> None:
        st = int(rec.get("step", 0))
        if st == 5001 or st % 10 == 0 or "val_bpb" in rec:
            ncm_str = (
                f" | ncm={rec.get('ncm_loss', 0.0):.3f} "
                f"rec_acc={rec.get('ncm_recon_accuracy', 0.0):.3f} "
                f"cb_u={rec.get('ncm_codebook_utilization', 0.0):.3f}"
                if "ncm_loss" in rec
                else ""
            )
            orn_str = (
                f" | orn_rw={rec.get('ornith_harness_reward', 0.0):.3f} "
                f"pass={rec.get('ornith_pass_rate', 0.0):.3f} "
                f"pp={rec.get('ornith_total_pp', 0.0):.1f} "
                f"drm_cyc={rec.get('dream_cycle', 0)}"
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
                f"(lm={rec.get('lm_loss'):.4f} bpb={rec.get('train_bpb'):.4f} dec={rec.get('decision_loss'):.4f})"
                f"{ncm_str} | E={rec.get('num_experts')} k={rec.get('top_k')} | "
                f"{rec.get('tok_per_s'):.0f} tok/s | vram={rec.get('vram_allocated_mb'):.0f}MB"
                f"{orn_str}{val_str}",
                flush=True,
            )

    trainer = MCL4COETrainer(cfg, on_step_callback=_on_step)
    if fit_steps > 0:
        fit_ncm_hierarchy_on_frozen_donor(
            trainer,
            fit_steps=fit_steps,
            fit_lr=args.fit_lr,
        )
    trainer.train()


if __name__ == "__main__":
    main()
