"""Cloud Pipeline for `MCL4-COE` on Google Colab + Hugging Face Pro + Google Drive.

Provides four end-to-end cloud workflows (callable from CLI or directly from Colab cells):
1. `build-cloud-shards`: Streams curated Hugging Face datasets directly into block-aligned
   `.bin` shards + `index_receipt.json` + `decision_examples_cache.pt` across all 3 tracks
   (`pretrain_lm`, `uta_trace`, `typed_decision`), with optional automatic push to a private
   Hugging Face Pro dataset repo (`--hf-dataset-repo`) and/or Google Drive (`--drive-root`).
2. `pull-shards`: Stages pre-built `.bin` shards from a Hugging Face dataset repo or Google
   Drive onto Colab's local NVMe SSD (`/content/data_shards`) for zero-bottleneck `mmap` I/O.
3. `train`: Runs `MCL4COETrainer` on Colab GPU (`A100` / `L4` / `T4`) with automatic checkpoint
   and telemetry syncing (`step_XXXXX.pt`, `step_XXXXX.meta.json`, `metrics.jsonl`, `dream_out/`)
   to Hugging Face Hub (`--hf-run-repo`) and/or Google Drive (`--drive-root`).
4. `ablate`: Executes the 4-Arm Context Hierarchy & Backbone Ablation Suite (`Arm A` Standard
   Attention vs. `Arm B` `MCL4-CED` vs. `Arm C` `+TurboQuant/Context-LM` vs. `Arm D` `+A.TXY`
   `COEAddressRouter`) and writes a publication-ready JSON + Markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcl4_coe import (
    ATXYPageStore,
    BlockAlignedShardWriter,
    COEAddressRouter,
    COEConfig,
    CPUOffloadAdamW,
    CognitiveOperatorEnsemble,
    ContextLMCompressor,
    DecisionExample,
    MCL4CEDBackbone,
    MCL4CEDConfig,
    MCL4COETrainer,
    MultiTrackMixtureDataset,
    TrainingRunConfig,
    compress_encoded_state,
    ensure_sharded_tracks_ready,
    extract_pretrain_text,
    extract_typed_decision_uta,
    extract_uta_episodes,
    index_python_repo_to_atxy,
    infer_github_code_language,
    list_checkpoints,
    normalize_github_code_record,
    parse_mcqa_decision_fields,
    prepare_model,
    synthesize_repository_uta_traces,
)
from mcl4_coe.curation import sanitize_trace_markers

# Default "Nemotron-Core" + "GitHub-Code" + "ArXiv-C++" + "Physics/Quantum/Math" + "Tool-Use" 3-Track HF recipe
DEFAULT_HF_TRACK_SOURCES: dict[str, list[dict[str, Any]]] = {
    "pretrain_lm": [
        {"path": "nvidia/Nemotron-ClimbMix", "name": None, "split": "train"},
        {"path": "codeparrot/github-code", "name": "C+Rust+C+++Python-all", "split": "train"},
        {"path": "AlgorithmicResearchGroup/arxiv_cplusplus_research_code", "name": "default", "split": "train"},
        {"path": "camel-ai/physics", "name": None, "split": "train"},
        {"path": "camel-ai/math", "name": None, "split": "train"},
        {
            "path": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
            "name": "Nemotron-Pretraining-Code-Concepts",
            "split": "train",
        },
        {
            "path": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
            "name": "Nemotron-Pretraining-Formal-Logic",
            "split": "train",
        },
        {
            "path": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
            "name": "Nemotron-Pretraining-Unconditional-Algorithmic",
            "split": "train",
        },
        {
            "path": "nvidia/Nemotron-Pretraining-Dataset-sample",
            "name": "Nemotron-CC-High-Quality-Synthetic",
            "split": "train",
        },
        {
            "path": "nvidia/Nemotron-Pretraining-Dataset-sample",
            "name": "Nemotron-CC-MATH",
            "split": "train",
        },
        {"path": "nvidia/Nemotron-MIND", "name": None, "split": "train"},
    ],
    "uta_trace": [
        {"path": "nvidia/Nemotron-Cascade-2-SFT-Data", "name": "math", "split": "train"},
        {"path": "nvidia/Nemotron-Cascade-2-SFT-Data", "name": "science", "split": "train"},
        {"path": "nvidia/Nemotron-Cascade-2-SFT-Data", "name": "instruction_following", "split": "train"},
        {"path": "nvidia/Nemotron-Cascade-2-SFT-Data", "name": "chat", "split": "train"},
        {"path": "nvidia/Nemotron-Post-Training-Dataset-v1", "name": "default", "split": "math"},
        {"path": "nvidia/Nemotron-Post-Training-Dataset-v1", "name": "default", "split": "stem"},
        {"path": "nvidia/Nemotron-Pretraining-Dataset-sample", "name": "Nemotron-SFT-MATH", "split": "train"},
        {"path": "nvidia/Nemotron-Pretraining-Dataset-sample", "name": "Nemotron-SFT-Code", "split": "train"},
        {"path": "codeparrot/github-code", "name": "C+Rust+C+++Python-all", "split": "train"},
        {"path": "AlgorithmicResearchGroup/arxiv_cplusplus_research_code", "name": "default", "split": "train"},
        {"path": "camel-ai/physics", "name": None, "split": "train"},
        {"path": "camel-ai/math", "name": None, "split": "train"},
        {"path": "NousResearch/hermes-function-calling-v1", "name": None, "split": "train"},
    ],
    "typed_decision": [
        {"path": "nvidia/Nemotron-RL-knowledge-mcqa", "name": "default", "split": "train"},
        {"path": "nvidia/Nemotron-CrossThink", "name": "default", "split": "train_qa"},
        {
            "path": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
            "name": "Nemotron-Pretraining-Multiple-Choice",
            "split": "train",
        },
    ],
}


def resolve_hf_token(
    explicit_token: str | None = None,
    *,
    require_userdata: bool = False,
) -> str | None:
    """Resolve `HF_TOKEN` from argument or environment (and optionally Colab `userdata` when pushing)."""
    if explicit_token:
        return explicit_token.strip()
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()
    if require_userdata:
        try:
            from google.colab import userdata  # type: ignore[import-not-found]

            secret = userdata.get("HF_TOKEN")
            if secret and str(secret).strip():
                os.environ["HF_TOKEN"] = str(secret).strip()
                return str(secret).strip()
        except Exception:
            pass
    return None


def _convert_record_to_decision_example(
    record: dict[str, Any],
    idx: int,
    max_seq_len: int,
    max_options: int = 10,
) -> DecisionExample | None:
    """Convert a raw HF or Nemotron MCQA record into a bounded, strictly symmetric `DecisionExample`."""
    parsed = parse_mcqa_decision_fields(record)
    if parsed is None:
        return None
    state, kind, question, clean_opts, best_idx, _rationale = parsed
    if len(clean_opts) < 2 or len(clean_opts) > max_options:
        return None

    budget = max(64, max_seq_len - 16)
    ctx_raw = f"#U$# {sanitize_trace_markers(state)}\n#Q$# [{kind}] {sanitize_trace_markers(question)}"
    ctx_str = ctx_raw.encode("utf-8")[:budget].decode("utf-8", errors="ignore")

    descs: list[str] = []
    for opt_text in clean_opts:
        # Strict candidate symmetry: never leak gold rationale into a single option description
        d_raw = f"#A$# {sanitize_trace_markers(opt_text)}"
        d_str = d_raw.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
        descs.append(d_str)

    return DecisionExample(
        question_id=f"hf_dec_{idx:06d}",
        kind="binary" if len(clean_opts) == 2 else "choice",
        context=ctx_str,
        descriptions=tuple(descs),
        values=tuple(f"opt_{i}" for i in range(len(clean_opts))),
        target=best_idx,
    )


def sanitize_decision_caches(shards_root: str | Path) -> dict[str, int]:
    """Scrub any asymmetric `#T$#` prefixes from cached `DecisionExample` files so all options are `#A$# {opt}`."""
    root = Path(shards_root).expanduser().resolve()
    td_dir = root / "typed_decision"
    updated_counts: dict[str, int] = {}
    for fname in ("decision_examples_cache.pt", "decision_examples_val_cache.pt"):
        fpath = td_dir / fname
        if not fpath.is_file():
            continue
        try:
            pool = torch.load(fpath, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(pool, list) or not pool:
            continue
        changed = False
        cleaned_pool: list[DecisionExample] = []
        for ex in pool:
            new_descs: list[str] = []
            for d in ex.descriptions:
                if "\n#A$# " in d:
                    new_d = "#A$# " + d.split("\n#A$# ", 1)[1].strip()
                    changed = True
                elif not d.startswith("#A$# "):
                    new_d = f"#A$# {d.strip()}"
                    changed = True
                else:
                    new_d = d
                new_descs.append(new_d)
            cleaned_pool.append(
                DecisionExample(
                    question_id=ex.question_id,
                    kind=ex.kind,
                    context=ex.context,
                    descriptions=tuple(new_descs),
                    values=ex.values,
                    target=ex.target,
                )
            )
        if changed:
            tmp = fpath.with_suffix(".pt.tmp")
            torch.save(cleaned_pool, tmp)
            os.replace(tmp, fpath)
        updated_counts[fname] = len(cleaned_pool)
    return updated_counts



def _configure_hf_transfer() -> None:
    """Enable `hf_transfer` only when the `hf_transfer` package is actually installed."""
    import importlib.util

    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    else:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)


def _iter_hf_source_rows(
    ds_path: str,
    ds_name: str | None,
    ds_split: str,
    max_rows: int,
    token: str | None,
    load_dataset_fn: Any,
):
    """Stream up to `max_rows` records via native Parquet streaming (`load_dataset`) with `/rows` fallback.

    For `codeparrot/github-code` (which uses a deprecated `github-code.py` builder script blocked by
    `datasets >= 4.0`), streams directly from `hf://datasets/codeparrot/github-code/data/train-*-of-01126.parquet`
    and normalizes each row into `{code, repo_name, path, language, license, size}`.
    """
    import urllib.parse
    import urllib.request

    yielded = 0
    if load_dataset_fn is not None and ds_path in ("codeparrot/github-code", "codeparrot/github-code-clean"):
        try:
            allowed_langs: set[str] | None = None
            if ds_name and ds_name != "all-all":
                lang_part = ds_name.split("-")[0]
                if lang_part and lang_part != "all":
                    allowed_langs = {l.strip() for l in lang_part.split("+") if l.strip()}
            rows_per_shard_est = 250 if allowed_langs is not None else 1500
            num_shards = min(1126, max(8, (max_rows // rows_per_shard_est) + 6))
            parquet_urls = [
                f"hf://datasets/codeparrot/github-code/data/train-{i:05d}-of-01126.parquet"
                for i in range(num_shards)
            ]
            stream = load_dataset_fn(
                "parquet",
                data_files=parquet_urls,
                split="train",
                streaming=True,
                token=token,
            )
            for raw_row in stream:
                if not isinstance(raw_row, dict):
                    continue
                norm_row = normalize_github_code_record(raw_row)
                if allowed_langs is not None and norm_row.get("language") not in allowed_langs:
                    continue
                yielded += 1
                yield norm_row
                if yielded >= max_rows:
                    return
            if yielded > 0:
                return
        except Exception:
            if yielded >= max_rows:
                return

    if load_dataset_fn is not None and ds_path.startswith("camel-ai/"):
        try:
            parquet_url = f"hf://datasets/{ds_path}@refs/convert/parquet/default/train/0000.parquet"
            stream = load_dataset_fn(
                "parquet",
                data_files=[parquet_url],
                split="train",
                streaming=True,
                token=token,
            )
            for row in stream:
                if isinstance(row, dict):
                    yielded += 1
                    yield row
                    if yielded >= max_rows:
                        return
            if yielded > 0:
                return
        except Exception:
            if yielded >= max_rows:
                return

    if load_dataset_fn is not None and ds_path == "AlgorithmicResearchGroup/arxiv_cplusplus_research_code":
        try:
            num_shards = min(44, max(2, (max_rows // 10000) + 2))
            parquet_urls = [
                f"hf://datasets/{ds_path}/data/train-{i:05d}-of-00044.parquet"
                for i in range(num_shards)
            ]
            stream = load_dataset_fn(
                "parquet",
                data_files=parquet_urls,
                split="train",
                streaming=True,
                token=token,
            )
            for raw_row in stream:
                if isinstance(raw_row, dict):
                    yielded += 1
                    yield normalize_github_code_record(raw_row)
                    if yielded >= max_rows:
                        return
            if yielded > 0:
                return
        except Exception:
            if yielded >= max_rows:
                return

    if load_dataset_fn is not None:
        try:
            stream = load_dataset_fn(
                ds_path,
                name=ds_name,
                split=ds_split,
                streaming=True,
                token=token,
            )
            for row in stream:
                if isinstance(row, dict):
                    yielded += 1
                    yield row
                    if yielded >= max_rows:
                        return
            if yielded > 0:
                return
        except Exception:
            if yielded >= max_rows:
                return

    cfg = ds_name or "default"
    try:
        while yielded < max_rows:
            length = min(100, max_rows - yielded)
            qs = urllib.parse.urlencode(
                {
                    "dataset": ds_path,
                    "config": cfg,
                    "split": ds_split,
                    "offset": yielded,
                    "length": length,
                }
            )
            req = urllib.request.Request(f"https://datasets-server.huggingface.co/rows?{qs}")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            rows = payload.get("rows", [])
            if not rows:
                break
            for item in rows:
                if isinstance(item, dict) and isinstance(item.get("row"), dict):
                    yielded += 1
                    yield item["row"]
                    if yielded >= max_rows:
                        return
            if len(rows) < length:
                return
    except Exception:
        return


def append_github_code_and_repo_shards(
    shards_root: str | Path,
    *,
    records_to_stream: int = 2000,
    seq_len: int | None = None,
    val_ratio: float = 0.05,
    hf_token: str | None = None,
    include_repo_ast: bool = True,
) -> dict[str, Any]:
    """Stream multi-domain expansion corpora (`ArXiv-C++`, `C+Rust`, `Physics/Quantum`, `Math`, `Tool-Use`) into `shards_root`."""
    token = resolve_hf_token(hf_token)
    _configure_hf_transfer()
    out_root = Path(shards_root).expanduser().resolve()
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        load_dataset = None

    val_mod = max(5, int(round(1.0 / max(0.01, min(0.25, val_ratio))))) if val_ratio > 0 else 0
    report: dict[str, Any] = {"shards_root": str(out_root), "tracks_updated": {}}

    expansion_sources: dict[str, list[tuple[str, str | None, str]]] = {
        "pretrain_lm": [
            ("AlgorithmicResearchGroup/arxiv_cplusplus_research_code", "default", "train"),
            ("codeparrot/github-code", "C+Rust-all", "train"),
            ("camel-ai/physics", None, "train"),
            ("camel-ai/math", None, "train"),
        ],
        "uta_trace": [
            ("AlgorithmicResearchGroup/arxiv_cplusplus_research_code", "default", "train"),
            ("codeparrot/github-code", "C+Rust-all", "train"),
            ("camel-ai/physics", None, "train"),
            ("camel-ai/math", None, "train"),
            ("NousResearch/hermes-function-calling-v1", None, "train"),
        ],
    }

    for track, src_list in expansion_sources.items():
        t_dir = out_root / track
        t_dir.mkdir(parents=True, exist_ok=True)
        idx_path = t_dir / "index_receipt.json"
        existing_vocab = 260
        target_seq_len = seq_len or 2048
        if idx_path.is_file():
            raw_idx = json.loads(idx_path.read_text(encoding="utf-8"))
            target_seq_len = seq_len or int(raw_idx.get("seq_len", 2048))
            if raw_idx.get("token_encoding") == "morphcode_16k" or int(raw_idx.get("vocab_size", 260)) == 16384:
                existing_vocab = 16384

        train_writer = BlockAlignedShardWriter(
            t_dir,
            track=track,
            split="train",
            seq_len=target_seq_len,
            max_blocks_per_shard=2048,
            vocab_size=existing_vocab,
        )
        val_writer = BlockAlignedShardWriter(
            t_dir,
            track=track,
            split="val",
            seq_len=target_seq_len,
            max_blocks_per_shard=512,
            vocab_size=existing_vocab,
        )

        by_source: dict[str, dict[str, int]] = {}
        total_acc = 0
        total_rej = 0
        for ds_path, ds_name, ds_split in src_list:
            accepted = 0
            rejected = 0
            row_idx = 0
            src_key = f"{ds_path}:{ds_name or 'default'}"
            for row in _iter_hf_source_rows(
                ds_path,
                ds_name,
                ds_split,
                records_to_stream,
                token,
                load_dataset,
            ):
                row_idx += 1
                is_val = bool(val_mod > 0 and ((total_acc + accepted + 1) % val_mod == 0))
                active_w = val_writer if is_val else train_writer
                if track == "pretrain_lm":
                    text, rej = extract_pretrain_text(row)
                    if rej or not text:
                        rejected += 1
                    else:
                        active_w.add_pretrain_document(text)
                        accepted += 1
                else:
                    eps = extract_uta_episodes(row, max_seq_len=target_seq_len)
                    if not eps:
                        rejected += 1
                    else:
                        added_any = False
                        for u_s, t_s, a_s in eps:
                            if active_w.add_uta_episode(u_s, t_s, a_s):
                                added_any = True
                        if added_any:
                            accepted += 1
                        else:
                            rejected += 1

            by_source[src_key] = {"accepted": accepted, "rejected_stage0": rejected}
            total_acc += accepted
            total_rej += rejected
            if accepted > 0 or rejected > 0:
                train_writer.record_provenance(
                    corpus=ds_path,
                    source_file=f"hf://{ds_path}/{ds_name or ds_split}",
                    row_start=1,
                    row_end=row_idx,
                    episodes_accepted=accepted,
                    episodes_rejected_stage0=rejected,
                )

        repo_acc = 0
        if track == "uta_trace" and include_repo_ast:
            repo_eps = synthesize_repository_uta_traces(REPO_ROOT, max_episodes=512)
            for idx_ep, (u_p, t_r, a_r) in enumerate(repo_eps):
                target_w = val_writer if (idx_ep % 10 == 0) else train_writer
                if target_w.add_uta_episode(u_p, t_r, a_r):
                    repo_acc += 1
            if repo_acc > 0:
                train_writer.record_provenance(
                    corpus="local_ast_codebase_uta",
                    source_file=str(REPO_ROOT / "src" / "mcl4_coe"),
                    row_start=1,
                    row_end=len(repo_eps),
                    episodes_accepted=repo_acc,
                    episodes_rejected_stage0=len(repo_eps) - repo_acc,
                )

        train_writer.flush_shard()
        val_writer.flush_shard()
        final_idx = json.loads((t_dir / "index_receipt.json").read_text(encoding="utf-8"))
        report["tracks_updated"][track] = {
            "total_accepted": total_acc,
            "total_rejected_stage0": total_rej,
            "by_source": by_source,
            "repo_ast_uta_accepted": repo_acc,
            "totals": final_idx.get("totals", {}),
        }
    return report


def build_cloud_shards(
    shards_root: str | Path,
    *,
    seq_len: int = 2048,
    max_blocks_per_shard: int = 2048,
    records_per_source: int = 4000,
    val_ratio: float = 0.05,
    hf_token: str | None = None,
    hf_dataset_repo: str | None = None,
    drive_root: str | Path | None = None,
) -> dict[str, Any]:
    """Stream datasets from Hugging Face Hub directly into train/val `.bin` shards + `index_receipt.json`."""
    token = resolve_hf_token(hf_token)
    _configure_hf_transfer()
    out_root = Path(shards_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        load_dataset = None

    summary: dict[str, Any] = {"shards_root": str(out_root), "tracks": {}}
    train_decision_pool: list[DecisionExample] = []
    val_decision_pool: list[DecisionExample] = []
    val_mod = max(5, int(round(1.0 / max(0.01, min(0.25, val_ratio))))) if val_ratio > 0 else 0

    for track, sources in DEFAULT_HF_TRACK_SOURCES.items():
        t_dir = out_root / track
        t_dir.mkdir(parents=True, exist_ok=True)
        train_writer = BlockAlignedShardWriter(
            t_dir,
            track=track,
            split="train",
            seq_len=seq_len,
            max_blocks_per_shard=max_blocks_per_shard,
        )
        val_writer = BlockAlignedShardWriter(
            t_dir,
            track=track,
            split="val",
            seq_len=seq_len,
            max_blocks_per_shard=max(64, max_blocks_per_shard // 4),
        )
        track_accepted = 0
        track_rejected = 0

        for src in sources:
            ds_path = src["path"]
            ds_name = src.get("name")
            ds_split = src.get("split", "train")
            print(f"[build-cloud-shards] Streaming {ds_path} ({ds_name or 'default'}/{ds_split}) -> {track} ...")
            accepted = 0
            rejected = 0
            row_idx = 0
            try:
                for row in _iter_hf_source_rows(
                    ds_path, ds_name, ds_split, records_per_source, token, load_dataset
                ):
                    row_idx += 1
                    is_val_row = bool(val_mod > 0 and ((accepted + 1) % val_mod == 0))
                    active_writer = val_writer if is_val_row else train_writer

                    if track == "pretrain_lm":
                        text, rej = extract_pretrain_text(row)
                        if rej or not text:
                            rejected += 1
                        else:
                            active_writer.add_pretrain_document(text)
                            accepted += 1
                    elif track == "uta_trace":
                        eps = extract_uta_episodes(row, max_seq_len=seq_len)
                        if not eps:
                            rejected += 1
                        else:
                            added_any = False
                            for u_s, t_s, a_s in eps:
                                if active_writer.add_uta_episode(u_s, t_s, a_s):
                                    added_any = True
                            if added_any:
                                accepted += 1
                            else:
                                rejected += 1
                    else:
                        uta = extract_typed_decision_uta(row, max_seq_len=seq_len, require_rationale=True)
                        total_dec_seen = len(train_decision_pool) + len(val_decision_pool)
                        dec_ex = _convert_record_to_decision_example(
                            row, total_dec_seen, max_seq_len=min(512, seq_len)
                        )
                        if dec_ex is not None:
                            # Route every 5th 4-way MCQA example into the uncontaminated held-out benchmark (up to 512)
                            if (
                                len(dec_ex.descriptions) == 4
                                and len(val_decision_pool) < 512
                                and (total_dec_seen % 5 == 0)
                            ):
                                val_decision_pool.append(dec_ex)
                                active_writer = val_writer
                            elif len(train_decision_pool) < 8192:
                                train_decision_pool.append(dec_ex)
                                active_writer = train_writer
                        if uta is not None and active_writer.add_uta_episode(*uta):
                            accepted += 1
                        else:
                            eps = extract_uta_episodes(row, max_seq_len=seq_len)
                            if eps and active_writer.add_uta_episode(*eps[0]):
                                accepted += 1
                            else:
                                rejected += 1
            except Exception as exc:
                print(f"  [skip] {ds_path}: {exc}")
                continue

            if accepted > 0 or rejected > 0:
                train_writer.record_provenance(
                    corpus=ds_path,
                    source_file=f"hf://{ds_path}/{ds_name or ds_split}",
                    row_start=1,
                    row_end=row_idx,
                    episodes_accepted=accepted,
                    episodes_rejected_stage0=rejected,
                )
                track_accepted += accepted
                track_rejected += rejected
                print(f"  -> accepted={accepted:,}, rejected_stage0={rejected:,}")

        if track == "uta_trace":
            repo_eps = synthesize_repository_uta_traces(REPO_ROOT, max_episodes=256)
            repo_acc = 0
            for idx_ep, (u_p, t_r, a_r) in enumerate(repo_eps):
                target_w = val_writer if (idx_ep % 10 == 0) else train_writer
                if target_w.add_uta_episode(u_p, t_r, a_r):
                    repo_acc += 1
            if repo_acc > 0:
                train_writer.record_provenance(
                    corpus="local_ast_codebase_uta",
                    source_file=str(REPO_ROOT / "src" / "mcl4_coe"),
                    row_start=1,
                    row_end=len(repo_eps),
                    episodes_accepted=repo_acc,
                    episodes_rejected_stage0=len(repo_eps) - repo_acc,
                )
                track_accepted += repo_acc
                print(f"  [local_ast_codebase_uta] -> accepted={repo_acc:,} AST-grounded repository episodes")

        train_writer.flush_shard()
        val_writer.flush_shard()
        final_idx = json.loads((t_dir / "index_receipt.json").read_text(encoding="utf-8"))
        t_totals = dict(final_idx.get("totals", {}))
        summary["tracks"][track] = {
            "accepted_episodes": track_accepted,
            "rejected_stage0": track_rejected,
            "num_blocks": t_totals.get("num_blocks", 0),
            "num_tokens": t_totals.get("num_tokens", 0),
            "train_blocks": t_totals.get("train_blocks", 0),
            "val_blocks": t_totals.get("val_blocks", 0),
            "num_shards": t_totals.get("num_shards", 0),
        }

    if train_decision_pool:
        dec_cache = out_root / "typed_decision" / "decision_examples_cache.pt"
        dec_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(train_decision_pool, dec_cache)
        summary["decision_examples_train_cached"] = len(train_decision_pool)

    if val_decision_pool:
        val_dec_cache = out_root / "typed_decision" / "decision_examples_val_cache.pt"
        val_dec_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(val_decision_pool, val_dec_cache)
        summary["decision_examples_val_cached"] = len(val_decision_pool)

    # Guarantee every track has at least 1 valid .bin shard (portable self-healing fallback)
    healed = ensure_sharded_tracks_ready(out_root, seq_len=seq_len)
    for t_name, idx_path in healed.items():
        raw_idx = json.loads(idx_path.read_text(encoding="utf-8"))
        totals = dict(raw_idx.get("totals", {}))
        if t_name in summary["tracks"]:
            totals["streamed_accepted"] = summary["tracks"][t_name].get("accepted_episodes", 0)
            totals["streamed_rejected_stage0"] = summary["tracks"][t_name].get("rejected_stage0", 0)
        summary["tracks"][t_name] = totals

    if drive_root:
        drive_shards = Path(drive_root).expanduser().resolve() / "data_shards"
        drive_shards.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_root, drive_shards, dirs_exist_ok=True)
        summary["drive_mirror"] = str(drive_shards)

    if hf_dataset_repo:
        try:
            uploaded_url = push_folder_to_hf(
                out_root,
                repo_id=hf_dataset_repo,
                repo_type="dataset",
                hf_token=token,
                commit_message=f"Update MCL4-COE sharded tracks (seq_len={seq_len})",
            )
            summary["hf_dataset_repo"] = uploaded_url
        except Exception as exc:
            print(f"[build-cloud-shards] Warning: HF upload skipped ({exc})")
            summary["hf_upload_skipped"] = str(exc)

    return summary


def push_folder_to_hf(
    local_dir: str | Path,
    *,
    repo_id: str,
    repo_type: str = "dataset",
    path_in_repo: str | None = None,
    hf_token: str | None = None,
    commit_message: str = "Sync MCL4-COE artifacts",
) -> str:
    """Create (if needed) and upload `local_dir` to a private Hugging Face repo."""
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    token = resolve_hf_token(hf_token, require_userdata=True)
    _configure_hf_transfer()
    api = HfApi(token=token)
    target_repo = repo_id
    try:
        who = api.whoami()
        hf_user = str(who.get("name") or "").strip()
        if hf_user:
            short_name = repo_id.split("/")[-1]
            target_repo = f"{hf_user}/{short_name}"
    except Exception:
        pass
    api.create_repo(repo_id=target_repo, repo_type=repo_type, private=True, exist_ok=True)
    info = api.upload_folder(
        repo_id=target_repo,
        repo_type=repo_type,
        folder_path=str(Path(local_dir).expanduser().resolve()),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    return str(info)


def pull_shards_to_local_nvme(
    local_shards_root: str | Path = "/content/data_shards",
    *,
    hf_dataset_repo: str | None = None,
    drive_root: str | Path | None = None,
    hf_token: str | None = None,
    seq_len: int = 2048,
) -> dict[str, Path]:
    """Stage `.bin` shards onto local NVMe from HF Hub or Google Drive (or heal portably)."""
    dst = Path(local_shards_root).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    token = resolve_hf_token(hf_token, require_userdata=bool(hf_dataset_repo))
    _configure_hf_transfer()

    if hf_dataset_repo:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        print(f"[pull-shards] Downloading {hf_dataset_repo} -> {dst} via HF Hub ...")
        snapshot_download(
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            local_dir=str(dst),
            token=token,
        )
    elif drive_root:
        src = Path(drive_root).expanduser().resolve() / "data_shards"
        if src.is_dir():
            print(f"[pull-shards] Staging {src} -> {dst} onto local NVMe ...")
            shutil.copytree(src, dst, dirs_exist_ok=True)

    return ensure_sharded_tracks_ready(dst, seq_len=seq_len)


def run_cloud_training(
    config: TrainingRunConfig,
    *,
    hf_run_repo: str | None = None,
    drive_root: str | Path | None = None,
    hf_token: str | None = None,
    resume_from_cloud: bool = True,
) -> dict[str, Any]:
    """Run `MCL4COETrainer` and automatically sync checkpoints + telemetry to HF Hub and/or Drive."""
    token = resolve_hf_token(hf_token, require_userdata=bool(hf_run_repo))
    _configure_hf_transfer()
    runs_root = Path(config.runs_root).expanduser().resolve()
    run_dir = runs_root / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Optionally restore prior run checkpoints from Drive or HF Hub before starting
    if resume_from_cloud:
        if drive_root:
            drive_run = Path(drive_root).expanduser().resolve() / "runs" / config.run_id
            if drive_run.is_dir():
                shutil.copytree(drive_run, run_dir, dirs_exist_ok=True)
        if hf_run_repo:
            try:
                from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

                snapshot_download(
                    repo_id=hf_run_repo,
                    repo_type="model",
                    allow_patterns=[f"{config.run_id}/**"],
                    local_dir=str(runs_root),
                    token=token,
                )
            except Exception:
                pass
        existing_ckpts = list_checkpoints(run_dir)
        if existing_ckpts and not config.resume_checkpoint:
            latest_meta = existing_ckpts[0]
            ckpt_pt = Path(str(latest_meta.get("pt_file") or latest_meta.get("checkpoint_path") or ""))
            if not ckpt_pt.is_file():
                ckpt_pt = run_dir / "checkpoints" / ckpt_pt.name
            if ckpt_pt.is_file():
                config.resume_checkpoint = str(ckpt_pt)
                print(f"[cloud-train] Resuming from checkpoint: {ckpt_pt.name} (step={latest_meta['step']})")

    import threading

    existing_list = list_checkpoints(run_dir)
    last_synced_ckpt_step = int(existing_list[0]["step"]) if existing_list else 0
    active_sync_thread: threading.Thread | None = None

    def _sync_worker(step: int) -> None:
        if drive_root:
            drive_run = Path(drive_root).expanduser().resolve() / "runs" / config.run_id
            drive_run.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_dir, drive_run, dirs_exist_ok=True)
        if hf_run_repo:
            try:
                push_folder_to_hf(
                    run_dir,
                    repo_id=hf_run_repo,
                    repo_type="model",
                    path_in_repo=config.run_id,
                    hf_token=token,
                    commit_message=f"Checkpoint sync for {config.run_id} at step {step}",
                )
            except Exception as exc:
                print(f"[cloud-train] Warning: HF upload skipped at step {step}: {exc}")

    def _sync_run_artifacts(step: int, *, blocking: bool = False) -> None:
        nonlocal active_sync_thread
        if blocking:
            if active_sync_thread is not None and active_sync_thread.is_alive():
                active_sync_thread.join(timeout=120)
            _sync_worker(step)
            return
        if active_sync_thread is not None and active_sync_thread.is_alive():
            return
        active_sync_thread = threading.Thread(target=_sync_worker, args=(step,), daemon=True)
        active_sync_thread.start()

    def _on_step(metrics: dict[str, Any]) -> None:
        nonlocal last_synced_ckpt_step
        step = int(metrics.get("step", 0))
        loss = float(metrics.get("total_loss", 0.0))
        lm = float(metrics.get("lm_loss", 0.0))
        tps = float(metrics.get("tok_per_s", metrics.get("tokens_per_sec", 0.0)))
        if "mitosis_event" in metrics:
            me = metrics["mitosis_event"]
            cleared_str = f" | cleared_shards={me['cleared_shards']}" if me.get("cleared_shards") else ""
            print(
                f"  [mitosis @ step {step}] action={me.get('action')} | "
                f"experts={me.get('old_experts', me.get('num_experts'))}->{me.get('new_experts', me.get('num_experts'))} | "
                f"pairs={len(me.get('mitotic_pairs', []))} | params={me.get('parameter_count', 0):,}{cleared_str}"
            )
        elif "cleared_shards" in metrics:
            print(f"  [shard-evict @ step {step}] cleared_shards={metrics['cleared_shards']}")
        if step == 1 or step % 10 == 0 or step == config.total_steps or "mitosis_event" in metrics:
            val_str = ""
            if "val_bpb" in metrics:
                val_str += f" | val_bpb={float(metrics['val_bpb']):.4f}"
            if "val_mcqa_acc" in metrics:
                val_str += (
                    f" | val_mcqa={float(metrics['val_mcqa_acc']) * 100.0:.1f}%"
                    f" (n={metrics.get('val_mcqa_n', 0)})"
                )
            n_exp = int(metrics.get("num_experts", config.num_experts))
            print(
                f"  [step {step:>4}/{config.total_steps}] "
                f"E={n_exp} | total_loss={loss:.4f} | lm_loss={lm:.4f}{val_str} | tok/s={tps:,.0f}"
            )
        cur_ckpts = list_checkpoints(run_dir)
        newest_ckpt_step = int(cur_ckpts[0]["step"]) if cur_ckpts else 0
        if newest_ckpt_step > last_synced_ckpt_step:
            last_synced_ckpt_step = newest_ckpt_step
            _sync_run_artifacts(step, blocking=False)

    sanitize_decision_caches(config.shards_root)
    trainer = MCL4COETrainer(config, on_step_callback=_on_step)
    manifest = trainer.train()
    _sync_run_artifacts(int(manifest.get("step", config.total_steps)), blocking=True)
    return manifest


def run_context_hierarchy_ablation(
    output_dir: str | Path,
    *,
    shards_root: str | Path | None = None,
    d_model: int = 192,
    n_heads: int = 6,
    steps: int = 24,
    seq_len: int = 256,
) -> dict[str, Any]:
    """Execute the 5-Arm Controlled Context Hierarchy, Attention & NVFP4 Ablation Suite."""
    from mcl4_coe import ShardedTokenBlockDataset, collate_decisions
    from mcl4_coe.nvfp4 import convert_linear_to_nvfp4, get_gemm_counts, reset_gemm_counts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if shards_root is not None:
        resolved_shards = Path(shards_root).expanduser().resolve()
    elif Path("/content/data_shards_v2").is_dir():
        resolved_shards = Path("/content/data_shards_v2")
    else:
        resolved_shards = REPO_ROOT / "data_shards"

    sanitize_decision_caches(resolved_shards)
    track_receipts = ensure_sharded_tracks_ready(resolved_shards, seq_len=seq_len)
    ds = MultiTrackMixtureDataset(track_receipts, split="train")

    val_datasets: dict[str, Any] = {}
    for t_name, r_path in track_receipts.items():
        try:
            val_datasets[t_name] = ShardedTokenBlockDataset(r_path, track=t_name, split="val")
        except ValueError:
            val_datasets[t_name] = ds.track_datasets[t_name]

    train_dec_path = resolved_shards / "typed_decision" / "decision_examples_cache.pt"
    val_dec_path = resolved_shards / "typed_decision" / "decision_examples_val_cache.pt"
    max_mc_len = min(512, seq_len)

    def _load_filtered_mcqa(p: Path, limit: int) -> list[Any]:
        if not p.is_file():
            return []
        try:
            loaded_p = torch.load(p, map_location="cpu", weights_only=False)
            return [
                ex
                for ex in loaded_p
                if len(ex.descriptions) == 4
                and len(ex.context.encode("utf-8")) + 2 <= max_mc_len
                and all(len(d.encode("utf-8")) + 2 <= max_mc_len for d in ex.descriptions)
            ][:limit]
        except Exception:
            return []

    train_mcqa_pool = _load_filtered_mcqa(train_dec_path, 256)
    val_mcqa_pool = _load_filtered_mcqa(val_dec_path if val_dec_path.is_file() else train_dec_path, 128)

    arms = [
        ("Arm_A_Standard_Attention", "standard", False, False, False),
        ("Arm_B_MCL4_CED", "markov_local4", False, False, False),
        ("Arm_C_MCL4_TurboQuant_ContextLM", "markov_local4", True, False, False),
        ("Arm_D_MCL4_Full_4Tier_ATXY", "markov_local4", True, True, False),
        ("Arm_E_MCL4_NVFP4_4Tier_ATXY", "markov_local4", True, True, True),
    ]

    store: ATXYPageStore = index_python_repo_to_atxy(REPO_ROOT, subsystems=("src",))
    results: dict[str, Any] = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "shards_root": str(resolved_shards),
        "d_model": d_model,
        "steps": steps,
        "seq_len": seq_len,
        "atxy_records_indexed": len(store),
        "arms": {},
    }

    for arm_name, attn_type, use_compression, use_atxy, use_nvfp4 in arms:
        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cfg = MCL4CEDConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=2,
            n_decoder_layers=2,
            ple_dim=min(64, max(32, d_model // 3)),
            expert_hidden_dim=d_model * 2,
            num_experts=4,
            top_k=2,
            max_seq_len=seq_len,
            attention_type=attn_type,
        )
        model = CognitiveOperatorEnsemble(
            MCL4CEDBackbone(cfg),
            COEConfig(projection_dim=min(64, max(32, d_model // 3)), initial_scale=15.0, recurrence_steps=1),
        )
        nvfp4_converted = 0
        if use_nvfp4 and device.type == "cuda":
            reset_gemm_counts()
            nvfp4_converted = convert_linear_to_nvfp4(
                model,
                min_features=32,
                experts_only=True,
                sparsity_mode="2to4_transposable",
                sr_ste_decay=1e-4,
                causal_activation_scale=True,
            )
            model.to(device)
            opt: Any = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)
        else:
            prepare_model(model, device=device, bf16_working_weights=(device.type == "cuda"))
            opt = CPUOffloadAdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01, transfer_mode="batched")

        ctx_compressor = (
            ContextLMCompressor(d_model=d_model, chunk_size=64, gist_tokens_per_chunk=8).to(device)
            if use_compression
            else None
        )

        t0 = time.perf_counter()
        first_loss = 0.0
        final_loss = 0.0
        kv_ratio = 1.0
        total_toks = 0

        for step_idx in range(1, steps + 1):
            batch_items = [ds[(step_idx * 7 + b) % len(ds)] for b in range(4)]
            input_ids = torch.stack([b["input_ids"][:seq_len] for b in batch_items]).to(device)
            targets = torch.stack([b["labels"][:seq_len] for b in batch_items]).to(device)

            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and not use_nvfp4)):
                lm_out = model.forward_inference(input_ids, targets=targets)
                loss = lm_out.loss if lm_out.loss is not None else lm_out.aux_loss

                if ctx_compressor is not None:
                    valid = input_ids != cfg.pad_id
                    gist_state = ctx_compressor(model.backbone, input_ids, valid)
                    comp_kv = compress_encoded_state(gist_state)
                    kv_ratio = (seq_len / max(1, gist_state.hidden.shape[1])) * comp_kv.compression_ratio
                    loss = loss + 0.05 * gist_state.aux_loss

                if train_mcqa_pool and step_idx % 2 == 0:
                    mc_start = ((step_idx // 2) * 4) % len(train_mcqa_pool)
                    mc_sub = [train_mcqa_pool[(mc_start + k) % len(train_mcqa_pool)] for k in range(4)]
                    mc_b = {k: v.to(device) for k, v in collate_decisions(mc_sub, max_length=min(512, seq_len)).items()}
                    op_out = model.forward_all_operators(**mc_b)
                    loss = loss + 0.35 * op_out.total_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_toks += int(targets.ne(-100).sum().item())
            lv = float(loss.detach().item())
            if step_idx == 1:
                first_loss = lv
            final_loss = lv

        elapsed = time.perf_counter() - t0
        tok_per_s = total_toks / max(1e-6, elapsed)
        vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0

        # Evaluate held-out val_bpb across tracks
        model.eval()
        val_nll_sum = 0.0
        val_tok_sum = 0
        with torch.no_grad():
            for t_name, v_ds in val_datasets.items():
                n_eval = min(16, len(v_ds))
                if n_eval <= 0:
                    continue
                v_batch = v_ds.batch(list(range(n_eval)))
                v_inp = v_batch["input_ids"][:, :seq_len].to(device)
                v_lbl = v_batch["labels"][:, :seq_len].to(device)
                v_toks = int(v_lbl.ne(-100).sum().item())
                if v_toks <= 0:
                    continue
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and not use_nvfp4)):
                    v_out = model.forward_inference(v_inp, targets=v_lbl)
                if v_out.ce_loss is not None:
                    val_nll_sum += float(v_out.ce_loss.item()) * v_toks
                    val_tok_sum += v_toks
        val_ce = val_nll_sum / max(1, val_tok_sum)
        val_bpb = val_ce / math.log(2.0)

        # Evaluate symmetric held-out MCQA accuracy
        val_mcqa_acc = 0.0
        if val_mcqa_pool:
            mc_correct = 0
            mc_total = 0
            with torch.no_grad():
                for i in range(0, min(64, len(val_mcqa_pool)), 16):
                    sub = val_mcqa_pool[i : i + 16]
                    mc_b = {k: v.to(device) for k, v in collate_decisions(sub, max_length=min(512, seq_len)).items()}
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and not use_nvfp4)):
                        op_out = model.forward_all_operators(**mc_b)
                    preds = op_out.decision.probabilities.argmax(dim=-1)
                    mc_correct += int((preds == mc_b["targets"]).sum().item())
                    mc_total += len(sub)
            val_mcqa_acc = mc_correct / max(1, mc_total)

        atxy_route_conf = 0.0
        atxy_selected = None
        if use_atxy:
            router = COEAddressRouter(store, max_seq_len=min(128, seq_len))
            r_out = router.route_query(
                model,
                "Locate TurboQuant compression and A.TXY virtual memory page store in mcl4_coe",
                top_k_candidates=6,
            )
            atxy_route_conf = float(r_out["confidence"])
            atxy_selected = str(r_out["address"])

        gemm_counts = get_gemm_counts() if use_nvfp4 else {"nvfp4_gemms": 0, "nvfp4_sparse_2to4_gemms": 0}
        results["arms"][arm_name] = {
            "attention_type": attn_type,
            "use_nvfp4": use_nvfp4,
            "nvfp4_layers_converted": nvfp4_converted,
            "nvfp4_sparse_2to4_gemms": int(gemm_counts.get("nvfp4_sparse_2to4_gemms", 0)),
            "first_loss": round(first_loss, 4),
            "final_loss": round(final_loss, 4),
            "train_bpb": round(final_loss / math.log(2.0), 4),
            "held_out_val_bpb": round(val_bpb, 4),
            "symmetric_val_mcqa_acc": round(val_mcqa_acc, 4),
            "effective_context_compression_ratio": round(kv_ratio, 2),
            "atxy_routed_address": atxy_selected,
            "atxy_route_confidence": round(atxy_route_conf, 4),
            "tok_per_s": round(tok_per_s, 1),
            "vram_allocated_mb": round(vram_mb, 1),
            "wall_time_sec": round(elapsed, 2),
        }
        print(
            f"[{arm_name}] loss: {first_loss:.4f} -> {final_loss:.4f} | val_bpb={val_bpb:.4f} | "
            f"val_mcqa={val_mcqa_acc*100:.1f}% | tok/s={tok_per_s:,.0f} | "
            f"compression={kv_ratio:.1f}x | atxy={atxy_selected} ({elapsed:.2f}s)"
        )

    json_file = out_path / "context_hierarchy_ablation.json"
    json_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def pull_run_from_cloud(
    run_id: str,
    *,
    runs_root: str | Path = "runs",
    hf_run_repo: str = "mESmaC/mcl4-coe-runs",
    drive_root: str | Path | None = None,
    hf_token: str | None = None,
) -> Path:
    """Download a cloud-trained run directory (`checkpoints/`, `metrics.jsonl`, `samples.jsonl`) locally."""
    token = resolve_hf_token(hf_token, require_userdata=False)
    _configure_hf_transfer()
    dst_root = Path(runs_root).expanduser().resolve()
    run_dir = dst_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if drive_root:
        src = Path(drive_root).expanduser().resolve() / "runs" / run_id
        if src.is_dir():
            shutil.copytree(src, run_dir, dirs_exist_ok=True)
            return run_dir

    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    snapshot_download(
        repo_id=hf_run_repo,
        repo_type="model",
        allow_patterns=[f"{run_id}/**"],
        local_dir=str(dst_root),
        token=token,
    )
    return run_dir


def evaluate_trained_checkpoint(
    checkpoint_or_run: str | Path,
    *,
    shards_root: str | Path = "runs/nemotron_core_shards",
    prompt: str = "Explain why Markov Local-4 routing achieves O(T) complexity while preserving local context.",
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Run generative `<UTA>`, held-out Nemotron 4-way MCQA, and `A.TXY` routing evaluation on a checkpoint."""
    from mcl4_coe import collate_decisions, evaluate_checkpoint_prompt

    target = Path(checkpoint_or_run).expanduser().resolve()
    if target.is_dir():
        ckpts = list_checkpoints(target)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {target}")
        ckpt_path = Path(str(ckpts[0]["pt_file"]))
    else:
        ckpt_path = target

    gen_eval = evaluate_checkpoint_prompt(ckpt_path, prompt=prompt, max_new_tokens=max_new_tokens)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    b_cfg = MCL4CEDConfig(**payload["backbone_config"])
    c_cfg = COEConfig(**payload["coe_config"])
    model = CognitiveOperatorEnsemble(MCL4CEDBackbone(b_cfg), c_cfg)
    model.load_state_dict(payload["model_state"], strict=False)
    saved_pairs = payload.get("mitotic_pairs")
    if isinstance(saved_pairs, list) and saved_pairs:
        parsed_pairs = [(int(p[0]), int(p[1])) for p in saved_pairs if len(p) == 2]
        for layer in list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers):
            if not layer.moe.mitotic_pairs:
                layer.moe.mitotic_pairs = list(parsed_pairs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    mcqa_eval: dict[str, Any] = {}
    shards_dir = Path(shards_root).expanduser().resolve()
    sanitize_decision_caches(shards_dir)
    val_dec_cache = shards_dir / "typed_decision" / "decision_examples_val_cache.pt"
    dec_cache = val_dec_cache if val_dec_cache.is_file() else (shards_dir / "typed_decision" / "decision_examples_cache.pt")
    split_tag = "held_out_val" if val_dec_cache.is_file() else "train_pool_fallback"
    if not dec_cache.is_file():
        alt = REPO_ROOT / "data_shards" / "typed_decision" / "decision_examples_cache.pt"
        if alt.is_file():
            dec_cache = alt
    if dec_cache.is_file():
        dec_pool = torch.load(dec_cache, map_location="cpu", weights_only=False)
        max_len = min(512, b_cfg.max_seq_len)
        held_out = [
            ex
            for ex in dec_pool
            if len(ex.descriptions) == 4
            and len(ex.context.encode("utf-8")) + 2 <= max_len
            and all(len(d.encode("utf-8")) + 2 <= max_len for d in ex.descriptions)
        ][:500]
        if held_out:
            bsz_eval = 16
            total_correct = 0
            total_n = 0
            sum_ce = 0.0
            sum_ded = 0.0
            sum_eval = 0.0
            sum_meta = 0.0
            sum_abs = 0.0
            with torch.no_grad():
                for i in range(0, len(held_out), bsz_eval):
                    chunk = held_out[i : i + bsz_eval]
                    raw_b = collate_decisions(chunk, max_length=max_len)
                    dec_batch = {k: v.to(device) for k, v in raw_b.items()}
                    op_out = model.forward_all_operators(**dec_batch)
                    n_c = len(chunk)
                    total_correct += int((op_out.decision.logits.argmax(dim=-1) == dec_batch["targets"]).sum().item())
                    total_n += n_c
                    sum_ce += float(op_out.decision.ce_loss.item()) * n_c
                    sum_ded += float(op_out.deduction_loss.item()) * n_c
                    sum_eval += float(op_out.evaluation_loss.item()) * n_c
                    sum_meta += float(op_out.metacognition_loss.item()) * n_c
                    sum_abs += float(op_out.abstraction_loss.item()) * n_c
            mcqa_eval = {
                "split": split_tag,
                "num_examples": total_n,
                "decision_ce_loss": round(sum_ce / max(1, total_n), 4),
                "random_4way_ce_baseline": 1.3863,
                "decision_top1_accuracy": round(total_correct / max(1, total_n), 4),
                "deduction_loss": round(sum_ded / max(1, total_n), 4),
                "evaluation_loss": round(sum_eval / max(1, total_n), 4),
                "metacognition_loss": round(sum_meta / max(1, total_n), 4),
                "abstraction_loss": round(sum_abs / max(1, total_n), 4),
            }

    # Evaluate held-out validation BPB per track if val shards exist
    import math
    from mcl4_coe.pipeline import ShardedTokenBlockDataset

    val_bpb_eval: dict[str, Any] = {}
    ln2 = math.log(2.0)
    total_val_nll = 0.0
    total_val_toks = 0
    for t_name in ("pretrain_lm", "uta_trace", "typed_decision"):
        idx_p = shards_dir / t_name / "index_receipt.json"
        if not idx_p.is_file():
            continue
        try:
            v_ds = ShardedTokenBlockDataset(idx_p, track=t_name, split="val")
        except Exception:
            continue
        n_b = min(len(v_ds), 32)
        if n_b <= 0:
            continue
        t_nll = 0.0
        t_toks = 0
        with torch.no_grad():
            for s_i in range(0, n_b, 8):
                sub = v_ds.batch(list(range(s_i, min(n_b, s_i + 8))), pin_memory=False)
                inp = sub["input_ids"][:, : b_cfg.max_seq_len].to(device)
                lbl = sub["labels"][:, : b_cfg.max_seq_len].to(device)
                cnt = int((lbl != -100).sum().item())
                if cnt <= 0:
                    continue
                lm_out = model.forward_inference(inp, targets=lbl)
                assert lm_out.ce_loss is not None
                t_nll += float(lm_out.ce_loss.item()) * cnt
                t_toks += cnt
        if t_toks > 0:
            ce_m = t_nll / t_toks
            val_bpb_eval[f"val_loss_{t_name}"] = round(ce_m, 5)
            val_bpb_eval[f"val_bpb_{t_name}"] = round(ce_m / ln2, 5)
            total_val_nll += t_nll
            total_val_toks += t_toks
    if total_val_toks > 0:
        overall_ce = total_val_nll / total_val_toks
        val_bpb_eval["val_lm_loss"] = round(overall_ce, 5)
        val_bpb_eval["val_bpb"] = round(overall_ce / ln2, 5)
        val_bpb_eval["val_tokens_evaluated"] = total_val_toks

    store = index_python_repo_to_atxy(REPO_ROOT, subsystems=("src",))
    router = COEAddressRouter(store, max_seq_len=128)
    r_out = router.route_query(
        model,
        "Locate parse_mcqa_decision_fields and Nemotron-Core shard pipeline in mcl4_coe",
        top_k_candidates=6,
    )
    rec = r_out.get("record")
    atxy_eval = {
        "address": r_out["address"],
        "confidence": round(float(r_out["confidence"]), 4),
        "candidates": r_out["candidates"],
        "resolved_payload": rec.payload[:160] if rec is not None else None,
        "resolved_fields": rec.fields if rec is not None else None,
    }

    return {
        "checkpoint_path": str(ckpt_path),
        "generative_uta": gen_eval,
        "held_out_val_bpb": val_bpb_eval,
        "held_out_mcqa": mcqa_eval,
        "atxy_routing": atxy_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MCL4-COE Cloud Pipeline (Colab + HF Pro + Drive)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-cloud-shards", help="Stream HF corpora into .bin shards")
    p_build.add_argument("--shards-root", default="/content/data_shards")
    p_build.add_argument("--seq-len", type=int, default=2048)
    p_build.add_argument("--max-blocks-per-shard", type=int, default=2048)
    p_build.add_argument("--records-per-source", type=int, default=4000)
    p_build.add_argument("--val-ratio", type=float, default=0.05)
    p_build.add_argument("--hf-dataset-repo", default=None)
    p_build.add_argument("--drive-root", default=None)

    p_pull = sub.add_parser("pull-shards", help="Stage .bin shards onto local NVMe")
    p_pull.add_argument("--shards-root", default="/content/data_shards")
    p_pull.add_argument("--hf-dataset-repo", default=None)
    p_pull.add_argument("--drive-root", default=None)
    p_pull.add_argument("--seq-len", type=int, default=2048)

    p_train = sub.add_parser("train", help="Run MCL4COETrainer with cloud checkpoint sync")
    p_train.add_argument("--run-id", default="colab_mcl4_coe_v1")
    p_train.add_argument("--shards-root", default="/content/data_shards")
    p_train.add_argument("--runs-root", default="/content/runs")
    p_train.add_argument("--resume-checkpoint", default=None)
    p_train.add_argument("--total-steps", type=int, default=200)
    p_train.add_argument("--checkpoint-every", type=int, default=50)
    p_train.add_argument("--eval-every", type=int, default=100)
    p_train.add_argument("--seq-len", type=int, default=2048)
    p_train.add_argument("--lr", type=float, default=1.5e-3)
    p_train.add_argument("--d-model", type=int, default=384)
    p_train.add_argument("--n-heads", type=int, default=6)
    p_train.add_argument("--n-encoder-layers", type=int, default=4)
    p_train.add_argument("--n-decoder-layers", type=int, default=4)
    p_train.add_argument("--num-experts", type=int, default=8)
    p_train.add_argument("--enable-mitosis", action="store_true")
    p_train.add_argument("--max-experts", type=int, default=12)
    p_train.add_argument("--mitosis-every", type=int, default=300)
    p_train.add_argument("--mitosis-step-size", type=int, default=2)
    p_train.add_argument("--recurrence-steps", type=int, default=1)
    p_train.add_argument("--recurrent-kv", action="store_true")
    p_train.add_argument("--depth-ponder", action="store_true")
    p_train.add_argument("--deep-supervision-weight", type=float, default=0.20)
    p_train.add_argument("--projection-dim", type=int, default=0)
    p_train.add_argument("--enable-gist-conditioning", action="store_true")
    p_train.add_argument("--gist-conditioning-prob", type=float, default=0.50)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--micro-batch-size", type=int, default=4)
    p_train.add_argument("--ornith-warmup-ratio", type=float, default=0.80)
    p_train.add_argument("--use-nvfp4", action="store_true")
    p_train.add_argument("--hf-run-repo", default=None)
    p_train.add_argument("--drive-root", default=None)

    p_pull_run = sub.add_parser("pull-run", help="Pull a cloud-trained run directory locally")
    p_pull_run.add_argument("--run-id", default="colab_mcl4_coe_v1")
    p_pull_run.add_argument("--runs-root", default="runs")
    p_pull_run.add_argument("--hf-run-repo", default="mESmaC/mcl4-coe-runs")
    p_pull_run.add_argument("--drive-root", default=None)

    p_eval = sub.add_parser("eval-checkpoint", help="Evaluate a trained checkpoint (UTA + MCQA + A.TXY)")
    p_eval.add_argument("--target", default="runs/nemotron_core_v1")
    p_eval.add_argument("--shards-root", default="runs/nemotron_core_shards")
    p_eval.add_argument(
        "--prompt",
        default="Explain why Markov Local-4 routing achieves O(T) complexity while preserving local context.",
    )
    p_eval.add_argument("--max-new-tokens", type=int, default=96)

    p_ablate = sub.add_parser("ablate", help="Run 5-Arm Context Hierarchy, Attention & NVFP4 Ablation")
    p_ablate.add_argument("--output-dir", default="runs/ablations")
    p_ablate.add_argument("--shards-root", default=None)
    p_ablate.add_argument("--d-model", type=int, default=192)
    p_ablate.add_argument("--n-heads", type=int, default=6)
    p_ablate.add_argument("--steps", type=int, default=24)
    p_ablate.add_argument("--seq-len", type=int, default=256)

    args = parser.parse_args()
    if args.cmd == "build-cloud-shards":
        res = build_cloud_shards(
            args.shards_root,
            seq_len=args.seq_len,
            max_blocks_per_shard=args.max_blocks_per_shard,
            records_per_source=args.records_per_source,
            val_ratio=args.val_ratio,
            hf_dataset_repo=args.hf_dataset_repo,
            drive_root=args.drive_root,
        )
        print(json.dumps(res, indent=2))
    elif args.cmd == "pull-shards":
        receipts = pull_shards_to_local_nvme(
            args.shards_root,
            hf_dataset_repo=args.hf_dataset_repo,
            drive_root=args.drive_root,
            seq_len=args.seq_len,
        )
        print(json.dumps({k: str(v) for k, v in receipts.items()}, indent=2))
    elif args.cmd == "train":
        proj_dim = args.projection_dim if args.projection_dim > 0 else max(32, args.d_model // 4)
        cfg = TrainingRunConfig(
            run_id=args.run_id,
            shards_root=args.shards_root,
            runs_root=args.runs_root,
            resume_checkpoint=args.resume_checkpoint,
            total_steps=args.total_steps,
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            sample_every=max(10, args.checkpoint_every // 2),
            max_seq_len=args.seq_len,
            lr=args.lr,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            ple_dim=min(128, args.d_model // 2),
            expert_hidden_dim=args.d_model * 2,
            num_experts=args.num_experts,
            top_k=min(2, args.num_experts),
            enable_mitosis=bool(args.enable_mitosis),
            max_experts=args.max_experts,
            mitosis_every=args.mitosis_every,
            mitosis_step_size=args.mitosis_step_size,
            recurrence_steps=args.recurrence_steps,
            recurrent_kv=bool(args.recurrent_kv),
            depth_ponder=bool(args.depth_ponder),
            deep_supervision_weight=args.deep_supervision_weight,
            projection_dim=proj_dim,
            enable_gist_conditioning=bool(args.enable_gist_conditioning or args.d_model >= 256),
            gist_conditioning_prob=args.gist_conditioning_prob,
            use_nvfp4=bool(args.use_nvfp4),
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            enable_ornith_rl=True,
            ornith_every=10,
            ornith_warmup_ratio=args.ornith_warmup_ratio,
            enable_dream_rsi=True,
            dream_every=25,
        )
        manifest = run_cloud_training(
            cfg,
            hf_run_repo=args.hf_run_repo,
            drive_root=args.drive_root,
        )
        print(json.dumps(manifest, indent=2))
    elif args.cmd == "pull-run":
        out_dir = pull_run_from_cloud(
            args.run_id,
            runs_root=args.runs_root,
            hf_run_repo=args.hf_run_repo,
            drive_root=args.drive_root,
        )
        print(json.dumps({"pulled_run_dir": str(out_dir), "checkpoints": len(list_checkpoints(out_dir))}, indent=2))
    elif args.cmd == "eval-checkpoint":
        res = evaluate_trained_checkpoint(
            args.target,
            shards_root=args.shards_root,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(res, indent=2))
    elif args.cmd == "ablate":
        res = run_context_hierarchy_ablation(
            args.output_dir,
            shards_root=args.shards_root,
            d_model=args.d_model,
            n_heads=args.n_heads,
            steps=args.steps,
            seq_len=args.seq_len,
        )
        print(json.dumps(res, indent=2))

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

