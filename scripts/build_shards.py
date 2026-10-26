"""CLI tool to inventory `X:\\corpora` and build sharded `.bin` + `index_receipt.json` datasets.

Examples:
1. Inventory all corpora in `X:\\corpora` and write `corpus_inventory_receipt.json`:
   python scripts/build_shards.py --inventory-only

2. Build sample shards across all three tracks (`pretrain_lm`, `uta_trace`, `typed_decision`)
   from `X:\\corpora` and verify `ShardedTokenBlockDataset` + `HostBatchPrefetcher`:
   python scripts/build_shards.py --max-files-per-corpus 1 --max-records-per-file 200 --max-blocks-per-shard 16

3. Build full production shards (resumable via `index_receipt.json`):
   python scripts/build_shards.py --output-dir X:\\MCL4-COE\\shards --seq-len 2048 --max-blocks-per-shard 16384
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcl4_coe import (
    ByteTokenizer,
    EpochBlockSampler,
    HostBatchPrefetcher,
    ShardedTokenBlockDataset,
    build_corpus_shards,
    discover_corpora,
    resolve_corpora_root,
    resolve_shards_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MCL4-COE sharded .bin datasets and index receipts.")
    parser.add_argument("--corpora-root", type=str, default=r"X:\corpora", help="Root corpora directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"X:\MCL4-COE\data_shards",
        help="Destination directory for sharded .bin files and receipts",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["pretrain_lm", "uta_trace", "typed_decision"],
        choices=["pretrain_lm", "uta_trace", "typed_decision"],
        help="Tracks to build",
    )
    parser.add_argument(
        "--corpus-filter",
        nargs="*",
        default=None,
        help="Optional substring filter for corpus names",
    )
    parser.add_argument("--seq-len", type=int, default=512, help="Block sequence length")
    parser.add_argument(
        "--max-blocks-per-shard",
        type=int,
        default=16_384,
        help="Maximum blocks per .bin shard before rolling over to the next shard",
    )
    parser.add_argument(
        "--max-files-per-corpus",
        type=int,
        default=None,
        help="Optional cap on source files processed per corpus",
    )
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        default=None,
        help="Optional cap on records streamed per source file",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Only scan corpora and emit corpus_inventory_receipt.json",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    corpora_root = resolve_corpora_root(args.corpora_root)
    out_dir = resolve_shards_root(args.output_dir)
    specs = discover_corpora(corpora_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    inv_path = out_dir / "corpus_inventory_receipt.json"
    inv_payload = {
        "corpora_root": str(corpora_root),
        "total_corpora": len(specs),
        "total_bytes": sum(s.total_bytes for s in specs),
        "corpora": [asdict(s) for s in specs],
    }
    inv_path.write_text(json.dumps(inv_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"=== Discovered {len(specs)} Valid Corpora in {corpora_root} (TinyStories Excluded) ===")
    for s in specs:
        gb = s.total_bytes / (1024**3)
        print(f"  [{s.track:<14}] {s.name:<48} | files={len(s.files):>6} | {gb:>6.2f} GB ({s.format_hint})")
    print(f"Inventory Receipt written to: {inv_path}")

    if args.inventory_only:
        return

    print(f"\n=== Building Shards into {out_dir} (seq_len={args.seq_len}, max_blocks_per_shard={args.max_blocks_per_shard}) ===")
    receipts = build_corpus_shards(
        args.corpora_root,
        out_dir,
        tracks=args.tracks,
        corpus_filter=args.corpus_filter,
        seq_len=args.seq_len,
        max_blocks_per_shard=args.max_blocks_per_shard,
        max_files_per_corpus=args.max_files_per_corpus,
        max_records_per_file=args.max_records_per_file,
    )
    elapsed = time.perf_counter() - t0
    tokenizer = ByteTokenizer()

    for track_name, receipt in receipts.items():
        totals = receipt.totals
        index_file = out_dir / track_name / "index_receipt.json"
        print(
            f"\n[{track_name}] Index Receipt: {index_file}\n"
            f"  Shards:                 {totals.get('num_shards', 0)}\n"
            f"  Episodes Packed:        {totals.get('num_episodes', 0):,}\n"
            f"  Blocks (seq_len={receipt.seq_len}):   {totals.get('num_blocks', 0):,}\n"
            f"  Total Tokens:           {totals.get('num_tokens', 0):,}\n"
            f"  Unmasked Target Tokens: {totals.get('unmasked_target_tokens', 0):,}\n"
            f"  Stage-0 Rejected:       {totals.get('rejected_stage0', 0):,}"
        )
        if totals.get("num_blocks", 0) > 0:
            ds = ShardedTokenBlockDataset(index_file, track=track_name, split="train")
            sampler = EpochBlockSampler(len(ds), seed=42)
            prefetcher = HostBatchPrefetcher(ds, micro_batch_size=min(4, len(ds)), pin_memory=False)
            try:
                batch_list = prefetcher.get(sampler.take(min(4, len(ds))))
                sample_in = batch_list[0]["input_ids"][0]
                sample_lbl = batch_list[0]["labels"][0]
                unmasked_tokens = [t for t in sample_lbl.tolist() if t != -100]
                preview_in = tokenizer.decode(sample_in[:140]).replace("\n", "\\n")
                preview_tgt = tokenizer.decode(unmasked_tokens[:100]).replace("\n", "\\n")
                print(f"  Verified ShardedTokenBlockDataset ({len(ds)} blocks across {len(ds.shards)} shards):")
                print(f"    Sample Input  : {preview_in!r}")
                print(f"    Sample Target : {preview_tgt!r}")
            finally:
                prefetcher.close()

    print(f"\nCompleted shard build & verification in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
