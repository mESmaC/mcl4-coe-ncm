"""End-to-End Validation of the Sharded `.bin` + Index Receipt + Multi-Track Curriculum Pipeline on RTX 5070.

Verifies:
1. Loading `pretrain_lm`, `uta_trace`, and `typed_decision` `index_receipt.json` files into
   `MultiTrackMixtureDataset` (`3,712` total `2048`-token blocks across `233` `.bin` shards).
2. Smooth curriculum annealing via `CurriculumMixtureSchedule` + `CurriculumMixtureSampler`
   paired with background `HostBatchPrefetcher`.
3. Streaming structured `DecisionExample` batches directly from `X:\\corpora\\text\\tasksource-jev-typed-decisions`
   via `TypedDecisionShardStreamer` into `compute_joint_coe_loss` (`MCL4-CED` + `COE` + `CPUOffloadAdamW`)
   on the RTX 5070.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcl4_coe import (
    COEConfig,
    CPUOffloadAdamW,
    CognitiveOperatorEnsemble,
    CurriculumMixtureSampler,
    CurriculumMixtureSchedule,
    HostBatchPrefetcher,
    MCL4CEDBackbone,
    MCL4CEDConfig,
    MultiTrackMixtureDataset,
    TypedDecisionShardStreamer,
    compute_joint_coe_loss,
    ensure_sharded_tracks_ready,
    prepare_model,
    resolve_corpora_root,
    resolve_shards_root,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== MCL4-COE Sharded Data Pipeline & Multi-Track Curriculum Validation on {device} ===")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} (capability={torch.cuda.get_device_capability(0)})")

    shards_root = resolve_shards_root(r"X:\MCL4-COE\data_shards")
    corpora_root = resolve_corpora_root(r"X:\corpora")
    track_receipts = ensure_sharded_tracks_ready(shards_root, corpora_root=corpora_root, seq_len=2048)

    mix_ds = MultiTrackMixtureDataset(track_receipts)
    print(
        f"\n[1] Loaded MultiTrackMixtureDataset (seq_len={mix_ds.seq_len}):\n"
        f"    Total Blocks Across All Tracks: {len(mix_ds):,}\n"
        + "\n".join(
            f"    - {name:<16}: {count:>6,} blocks ({len(mix_ds.track_datasets[name].shards):>3} shards)"
            for name, count in zip(mix_ds.track_names, mix_ds.track_block_counts)
        )
    )

    schedule = CurriculumMixtureSchedule(
        start_weights={"pretrain_lm": 0.65, "uta_trace": 0.25, "typed_decision": 0.10},
        end_weights={"pretrain_lm": 0.15, "uta_trace": 0.50, "typed_decision": 0.35},
        mode="cosine",
    )
    sampler = CurriculumMixtureSampler(mix_ds, schedule, seed=42)
    prefetcher = HostBatchPrefetcher(mix_ds, micro_batch_size=4, pin_memory=(device.type == "cuda"))

    dec_cache = shards_root / "typed_decision" / "decision_examples_cache.pt"
    streamer = TypedDecisionShardStreamer(corpora_root, max_seq_len=512, max_options=6, cache_path=dec_cache)
    real_decision_batch = streamer.sample_batch(batch_size=4, device=device)
    print(
        f"\n[2] Streamed Real `tasksource-jev-typed-decisions` Batch via TypedDecisionShardStreamer:\n"
        f"    context_ids shape: {tuple(real_decision_batch['context_ids'].shape)}\n"
        f"    option_ids shape:  {tuple(real_decision_batch['option_ids'].shape)}"
    )

    cfg = MCL4CEDConfig(
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        ple_dim=32,
        expert_hidden_dim=256,
        num_experts=4,
        top_k=2,
        max_seq_len=mix_ds.seq_len,
    )
    model = CognitiveOperatorEnsemble(
        MCL4CEDBackbone(cfg),
        COEConfig(projection_dim=32, initial_scale=15.0, recurrence_steps=1),
    )
    prepare_model(model, device=device, bf16_working_weights=(device.type == "cuda"))
    optimizer = CPUOffloadAdamW(model.parameters(), lr=2e-3, weight_decay=0.01, transfer_mode="batched")

    total_steps = 12
    batch_size = 8
    track_counts_drawn = {name: 0 for name in mix_ds.track_names}
    total_unmasked_tokens = 0

    try:
        # Kick off step 1 prefetch
        next_indices = sampler.take(batch_size, progress=0.0)
        prefetcher.schedule(next_indices)

        t0 = time.perf_counter()
        first_loss = 0.0
        last_loss = 0.0

        for step in range(1, total_steps + 1):
            cur_indices = next_indices
            progress = step / total_steps
            if step < total_steps:
                next_indices = sampler.take(batch_size, progress=progress)

            micro_batches = prefetcher.get(cur_indices)
            if step < total_steps:
                prefetcher.schedule(next_indices)

            optimizer.zero_grad(set_to_none=True)
            step_loss_val = 0.0
            for mb in micro_batches:
                inp = mb["input_ids"].to(device, non_blocking=True)
                lbl = mb["labels"].to(device, non_blocking=True)
                t_ids = mb["track_ids"].tolist()
                for tid in t_ids:
                    track_counts_drawn[mix_ds.track_names[tid]] += 1
                total_unmasked_tokens += int((lbl != -100).sum().item())

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    joint = compute_joint_coe_loss(
                        model,
                        lm_batch=(inp, lbl),
                        decision_batch=real_decision_batch,
                    )
                    loss = joint.total_loss / len(micro_batches)
                loss.backward()
                step_loss_val += float(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1:
                first_loss = step_loss_val
            last_loss = step_loss_val

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    finally:
        prefetcher.close()

    tok_per_sec = total_unmasked_tokens / max(elapsed, 1e-6)
    print(
        f"\n[3] Live Multi-Track Curriculum Training on {device} ({total_steps} steps, seq_len={mix_ds.seq_len}):\n"
        f"    Blocks Drawn by Track:   {track_counts_drawn}\n"
        f"    Supervised Target Toks:  {total_unmasked_tokens:,} ({tok_per_sec:,.0f} unmasked target tok/s)\n"
        f"    Joint COE + LM Loss:     {first_loss:.4f} -> {last_loss:.4f} (wall_time={elapsed:.2f}s)"
    )


if __name__ == "__main__":
    main()
