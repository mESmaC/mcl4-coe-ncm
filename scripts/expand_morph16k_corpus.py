#!/usr/bin/env python3
"""High-speed Stage-0 sanitized MorphCode-16K corpus expansion (~87.5M -> ~295M+ tokens)."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, "/content/MCL4-COE/src")
from mcl4_coe.curation import ByteSieveConfig, evaluate_byte_sieve, normalize_prose_unwrapping, sanitize_trace_markers
from mcl4_coe.data import DecisionExample, get_tokenizer
from mcl4_coe.pipeline import (
    BlockAlignedShardWriter,
    IndexReceipt,
    _compact_uta_triplet,
    _is_clean_uta_episode,
    _split_thought_and_answer,
)

SHARDS_ROOT = Path("/content/data_shards_morph16k")
SEQ_LEN = 1024
VOCAB_SIZE = 16384
MAX_BLOCKS_PER_SHARD = 16384

_OT_THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>\s*(?:<\|begin_of_solution\|>(.*?)(?:<\|end_of_solution\|>|$)|(.*))",
    re.DOTALL,
)
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL | re.IGNORECASE)

_WORKER_TOK = None
_WORKER_SIEVE = None


def _init_worker() -> None:
    global _WORKER_TOK, _WORKER_SIEVE
    _WORKER_TOK = get_tokenizer(VOCAB_SIZE)
    _WORKER_SIEVE = ByteSieveConfig()


def _worker_encode_pretrain(raw_text: str) -> tuple[bytes, list[int]] | None:
    if not raw_text or len(raw_text) < 120:
        return None
    if len(raw_text) > 32000:
        raw_text = raw_text[:32000]
    dec = evaluate_byte_sieve(raw_text, _WORKER_SIEVE)
    if not dec.keep or not dec.sanitized_text:
        return None
    clean = normalize_prose_unwrapping(dec.sanitized_text).strip()
    if len(clean) < 120:
        return None
    digest = hashlib.blake2b(clean.encode("utf-8", "replace"), digest_size=12).digest()
    tokens = _WORKER_TOK.encode(clean, add_bos=True, add_eos=True)
    if len(tokens) < 32:
        return None
    return digest, tokens


def _clean_special_cot_tags(text: str) -> str:
    for tag in (
        "<|begin_of_thought|>",
        "<|end_of_thought|>",
        "<|begin_of_solution|>",
        "<|end_of_solution|>",
        "<think>",
        "</think>",
    ):
        text = text.replace(tag, "")
    return text.strip()


def _worker_encode_uta(item: tuple[str, str, str | None]) -> tuple[bytes, list[int]] | None:
    u_raw, a_raw, explicit_t = item
    if not u_raw or (not a_raw and not explicit_t):
        return None

    m_ot = _OT_THOUGHT_RE.search(a_raw)
    if m_ot:
        explicit_t = m_ot.group(1).strip()
        a_raw = (m_ot.group(2) or m_ot.group(3) or "").strip()
    else:
        m_th = _THINK_RE.search(a_raw)
        if m_th:
            explicit_t = m_th.group(1).strip()
            a_raw = m_th.group(2).strip()

    u_raw = _clean_special_cot_tags(u_raw)
    a_raw = _clean_special_cot_tags(a_raw)
    if explicit_t:
        explicit_t = _clean_special_cot_tags(explicit_t)

    # Check Stage-0 byte sieve on combined text
    combined = f"{u_raw}\n{explicit_t or ''}\n{a_raw}"
    if len(combined) > 24000:
        combined = combined[:24000]
    sieve = evaluate_byte_sieve(combined, _WORKER_SIEVE)
    if not sieve.keep:
        return None

    t_part, a_part = _split_thought_and_answer(a_raw, explicit_thought=explicit_t)
    # Try generous MorphCode-16K budget first (~2600 chars), then tighter fallbacks so len(tokens) <= 1024
    for max_b, max_c in ((2650, 2550), (2050, 1950), (1550, 1450)):
        u_c, t_c, a_c = _compact_uta_triplet(
            u_raw, t_part, a_part, max_total_chars=max_c, max_total_bytes=max_b
        )
        if not _is_clean_uta_episode(u_c, t_c, a_c, track_name="uta_trace", seq_len=SEQ_LEN):
            return None
        tokens = _WORKER_TOK.encode_masked_turn(u_c, t_c, a_c, seq_len=None)
        if len(tokens) <= SEQ_LEN:
            full_str = f"#U$# {u_c}\n#T$# {t_c}\n#A$# {a_c}"
            digest = hashlib.blake2b(full_str.encode("utf-8", "replace"), digest_size=12).digest()
            return digest, tokens
    return None


def _worker_encode_decision(
    item: tuple[str, str, str, list[str], int, str | None]
) -> tuple[bytes, list[int], DecisionExample | None] | None:
    qid, kind, question, opts, best_idx, rationale = item
    if not question or len(opts) < 2 or best_idx < 0 or best_idx >= len(opts):
        return None
    clean_q = sanitize_trace_markers(question.strip())
    clean_opts = [sanitize_trace_markers(str(o).strip()) for o in opts]
    if not clean_q or any(not o for o in clean_opts):
        return None

    winning_opt = clean_opts[best_idx]
    opt_lines = "\n".join(f"#H$# [{i}] {opt}" for i, opt in enumerate(clean_opts))
    prompt = f"{clean_q}\n#Q$# [{kind}] Select the best option.\n{opt_lines}"
    if rationale and rationale.strip():
        r_clean = sanitize_trace_markers(rationale.strip())
        thought = f"{r_clean}\nTherefore, option [{best_idx}] ({winning_opt}) is correct."
    else:
        thought = f"Evaluate hypotheses against question stem -> [{best_idx}] {winning_opt}"
    answer = f"[{best_idx}] {winning_opt}"

    for max_b, max_c in ((2600, 2500), (1950, 1850), (1450, 1350)):
        u_c, t_c, a_c = _compact_uta_triplet(
            prompt, thought, answer, max_total_chars=max_c, max_total_bytes=max_b
        )
        tokens = _WORKER_TOK.encode_masked_turn(u_c, t_c, a_c, seq_len=None)
        if len(tokens) <= SEQ_LEN:
            full_str = f"#U$# {u_c}\n#T$# {t_c}\n#A$# {a_c}"
            digest = hashlib.blake2b(full_str.encode("utf-8", "replace"), digest_size=12).digest()
            dec_ex = None
            ctx = f"#U$# {clean_q}\n#Q$# [{kind}] Select the best option."
            if (
                len(clean_opts) <= 8
                and len(ctx.encode("utf-8")) + 2 <= 512
                and all(len(o.encode("utf-8")) + 2 <= 480 for o in clean_opts)
            ):
                dec_ex = DecisionExample(
                    question_id=qid,
                    kind="choice",
                    context=ctx,
                    descriptions=tuple(f"#A$# {o}" for o in clean_opts),
                    values=tuple(f"opt_{i}" for i in range(len(clean_opts))),
                    target=best_idx,
                )
            return digest, tokens, dec_ex
    return None


def _pack_pretrain_tokens(writer: BlockAlignedShardWriter, tokens: list[int]) -> int:
    blocks_added = 0
    offset = 0
    n = len(tokens)
    while offset < n:
        space = writer.seq_len - len(writer._current_block_buf)
        take = min(space, n - offset)
        writer._current_block_buf.extend(tokens[offset : offset + take])
        offset += take
        if len(writer._current_block_buf) == writer.seq_len:
            writer._flush_block_buffer()
            blocks_added += 1
    writer._current_episodes += 1
    return blocks_added


def _pack_uta_tokens(writer: BlockAlignedShardWriter, tokens: list[int]) -> int:
    if len(tokens) > writer.seq_len:
        return 0
    blocks_before = len(writer._current_blocks)
    shards_before = len(writer.index_receipt.shards)
    if len(writer._current_block_buf) + len(tokens) > writer.seq_len:
        writer._flush_block_buffer()
    writer._current_block_buf.extend(tokens)
    writer._current_episodes += 1
    if len(writer._current_block_buf) == writer.seq_len:
        writer._flush_block_buffer()
    blocks_after = len(writer._current_blocks) + (
        (len(writer.index_receipt.shards) - shards_before) * writer.max_blocks_per_shard
    )
    return max(0, blocks_after - blocks_before)


def _prepare_track_dir(track: str, initial_train_shards: int) -> tuple[BlockAlignedShardWriter, set[bytes]]:
    track_dir = SHARDS_ROOT / track
    idx_path = track_dir / "index_receipt.json"
    backup_path = track_dir / "index_receipt.stage1_backup.json"
    if not backup_path.exists():
        shutil.copy2(idx_path, backup_path)
    else:
        shutil.copy2(backup_path, idx_path)

    # Remove any extra train shards beyond initial_train_shards from a prior partial run
    for bf in sorted(track_dir.glob(f"{track}_train_shard_*.bin")):
        s_idx = int(bf.stem.split("_")[-1])
        if s_idx >= initial_train_shards:
            bf.unlink(missing_ok=True)
            bf.with_suffix(".receipt.json").unlink(missing_ok=True)

    tok = get_tokenizer(VOCAB_SIZE)
    seen: set[bytes] = set()
    for bf in sorted(track_dir.glob("*.bin")):
        raw = np.fromfile(bf, dtype="<u2") & 0x7FFF
        bos = np.nonzero(raw == tok.bos_id)[0]
        eos = np.nonzero(raw == tok.eos_id)[0]
        ec = 0
        for b in bos:
            while ec < eos.size and eos[ec] <= b:
                ec += 1
            if ec >= eos.size:
                break
            txt = tok.decode(raw[b + 1 : eos[ec]].tolist()).strip()
            if txt:
                seen.add(hashlib.blake2b(txt.encode("utf-8", "replace"), digest_size=12).digest())

    writer = BlockAlignedShardWriter(
        track_dir,
        track=track,
        split="train",
        seq_len=SEQ_LEN,
        max_blocks_per_shard=MAX_BLOCKS_PER_SHARD,
        vocab_size=VOCAB_SIZE,
    )
    return writer, seen


def expand_pretrain_lm(pool: mp.pool.Pool, target_new_blocks: int = 98_304) -> None:
    t0 = time.time()
    writer, seen = _prepare_track_dir("pretrain_lm", initial_train_shards=4)
    start_blocks = int(writer.index_receipt.totals.get("train_blocks", 50299))
    print(
        f"[pretrain_lm] Starting expansion from {start_blocks:,} blocks ({len(seen):,} existing eps) -> target +{target_new_blocks:,} blocks",
        flush=True,
    )

    p_cosmo = hf_hub_download(
        repo_id="HuggingFaceTB/smollm-corpus",
        repo_type="dataset",
        filename="cosmopedia-v2/train-00000-of-00104.parquet",
    )
    p_math = hf_hub_download(
        repo_id="open-web-math/open-web-math",
        repo_type="dataset",
        filename="data/train-00000-of-00114-5a023365406cb9c4.parquet",
    )
    p_fw = hf_hub_download(
        repo_id="HuggingFaceTB/smollm-corpus",
        repo_type="dataset",
        filename="fineweb-edu-dedup/train-00000-of-00234.parquet",
    )
    p_oss = hf_hub_download(
        repo_id="ise-uiuc/Magicoder-OSS-Instruct-75K",
        repo_type="dataset",
        filename="data-oss_instruct-decontaminated.jsonl",
    )

    sources = [
        ("HuggingFaceTB/smollm-corpus:cosmopedia-v2", p_cosmo, "parquet", 44_000),
        ("open-web-math/open-web-math", p_math, "parquet", 24_000),
        ("ise-uiuc/Magicoder-OSS-Instruct-75K:code_docs", p_oss, "jsonl_code", 10_000),
        ("HuggingFaceTB/smollm-corpus:fineweb-edu-dedup", p_fw, "parquet_fw", 22_000),
    ]

    added_blocks = 0
    for corpus_name, fpath, kind, quota_blocks in sources:
        if added_blocks >= target_new_blocks:
            break
        src_blocks = 0
        accepted = 0
        rejected = 0
        row_idx = 0

        def _iter_raw():
            nonlocal row_idx
            if kind.startswith("parquet"):
                pf = pq.ParquetFile(fpath)
                for batch in pf.iter_batches(batch_size=1024):
                    for r in batch.to_pylist():
                        row_idx += 1
                        if kind == "parquet_fw":
                            meta = r.get("metadata")
                            if isinstance(meta, dict) and meta.get("int_score", 3) < 4:
                                continue
                        txt = r.get("text")
                        if isinstance(txt, str) and len(txt) >= 160:
                            yield txt
            elif kind == "jsonl_code":
                with open(fpath, "r", encoding="utf-8") as fh:
                    for line in fh:
                        row_idx += 1
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        prob = str(r.get("problem") or "").strip()
                        sol = str(r.get("solution") or "").strip()
                        lang = str(r.get("lang") or "code").strip()
                        if prob and sol:
                            yield f"# Language: {lang}\n# Specification:\n{prob}\n\n{sol}"

        for res in pool.imap(_worker_encode_pretrain, _iter_raw(), chunksize=128):
            if res is None:
                rejected += 1
                continue
            digest, tokens = res
            if digest in seen:
                rejected += 1
                continue
            seen.add(digest)
            accepted += 1
            prev_total = int(writer.index_receipt.totals.get("train_blocks", start_blocks)) + len(writer._current_blocks)
            _pack_pretrain_tokens(writer, tokens)
            now_total = int(writer.index_receipt.totals.get("train_blocks", start_blocks)) + len(writer._current_blocks)
            delta = now_total - prev_total
            added_blocks += delta
            src_blocks += delta
            if accepted % 4000 == 0:
                el = time.time() - t0
                tot_tok = (start_blocks + added_blocks) * SEQ_LEN
                print(
                    f"[pretrain_lm] +{added_blocks:,}/{target_new_blocks:,} new blocks | total={tot_tok/1e6:.2f}M tok | src={corpus_name} ({src_blocks:,}/{quota_blocks:,} blk) | {added_blocks*SEQ_LEN/el:,.0f} tok/s",
                    flush=True,
                )
            if src_blocks >= quota_blocks or added_blocks >= target_new_blocks:
                break

        writer.record_provenance(
            corpus=corpus_name,
            source_file=Path(fpath).name,
            row_start=1,
            row_end=row_idx,
            episodes_accepted=accepted,
            episodes_rejected_stage0=rejected,
        )

    writer.flush_shard()
    tot_b = int(writer.index_receipt.totals.get("train_blocks", 0))
    print(
        f"[pretrain_lm] DONE in {time.time()-t0:.1f}s! Train blocks: {tot_b:,} ({tot_b*SEQ_LEN/1e6:.2f}M tokens across {sum(1 for s in writer.index_receipt.shards if s['split']=='train')} train shards)",
        flush=True,
    )


def expand_uta_trace(pool: mp.pool.Pool, target_new_blocks: int = 81_920) -> None:
    t0 = time.time()
    writer, seen = _prepare_track_dir("uta_trace", initial_train_shards=2)
    start_blocks = int(writer.index_receipt.totals.get("train_blocks", 29276))
    print(
        f"[uta_trace] Starting expansion from {start_blocks:,} blocks ({len(seen):,} existing eps) -> target +{target_new_blocks:,} blocks",
        flush=True,
    )

    p_ot = hf_hub_download(
        repo_id="open-thoughts/OpenThoughts-114k",
        repo_type="dataset",
        filename="data/train-00000-of-00006.parquet",
    )
    p_bs = hf_hub_download(
        repo_id="bespokelabs/Bespoke-Stratos-17k",
        repo_type="dataset",
        filename="data/train-00000-of-00001.parquet",
    )
    p_oss = hf_hub_download(
        repo_id="ise-uiuc/Magicoder-OSS-Instruct-75K",
        repo_type="dataset",
        filename="data-oss_instruct-decontaminated.jsonl",
    )
    p_evol = hf_hub_download(
        repo_id="ise-uiuc/Magicoder-Evol-Instruct-110K",
        repo_type="dataset",
        filename="data-evol_instruct-decontaminated.jsonl",
    )
    p_cf = hf_hub_download(
        repo_id="m-a-p/CodeFeedback-Filtered-Instruction",
        repo_type="dataset",
        filename="CodeFeedback-Filtered-Instruction.jsonl",
    )
    p_mm = hf_hub_download(
        repo_id="meta-math/MetaMathQA",
        repo_type="dataset",
        filename="MetaMathQA-395K.json",
    )

    sources = [
        ("open-thoughts/OpenThoughts-114k", p_ot, "parquet_conv", 12_000),
        ("bespokelabs/Bespoke-Stratos-17k", p_bs, "parquet_conv", 10_000),
        ("ise-uiuc/Magicoder-OSS-Instruct-75K", p_oss, "jsonl_oss", 16_000),
        ("ise-uiuc/Magicoder-Evol-Instruct-110K", p_evol, "jsonl_evol", 18_000),
        ("m-a-p/CodeFeedback-Filtered-Instruction", p_cf, "jsonl_cf", 15_000),
        ("meta-math/MetaMathQA", p_mm, "json_mm", 15_000),
    ]

    added_blocks = 0
    for corpus_name, fpath, kind, quota_blocks in sources:
        if added_blocks >= target_new_blocks:
            break
        src_blocks = 0
        accepted = 0
        rejected = 0
        row_idx = 0

        def _iter_items():
            nonlocal row_idx
            if kind == "parquet_conv":
                pf = pq.ParquetFile(fpath)
                for batch in pf.iter_batches(batch_size=512):
                    for r in batch.to_pylist():
                        row_idx += 1
                        convs = r.get("conversations") or []
                        if len(convs) >= 2 and isinstance(convs[0], dict) and isinstance(convs[1], dict):
                            u = str(convs[0].get("value") or "").strip()
                            a = str(convs[1].get("value") or "").strip()
                            if u and a:
                                yield (u, a, None)
            elif kind == "jsonl_oss":
                with open(fpath, "r", encoding="utf-8") as fh:
                    for line in fh:
                        row_idx += 1
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        yield (str(r.get("problem") or "").strip(), str(r.get("solution") or "").strip(), None)
            elif kind == "jsonl_evol":
                with open(fpath, "r", encoding="utf-8") as fh:
                    for line in fh:
                        row_idx += 1
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        yield (str(r.get("instruction") or "").strip(), str(r.get("response") or "").strip(), None)
            elif kind == "jsonl_cf":
                with open(fpath, "r", encoding="utf-8") as fh:
                    for line in fh:
                        row_idx += 1
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        yield (str(r.get("query") or "").strip(), str(r.get("answer") or "").strip(), None)
            elif kind == "json_mm":
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for r in data:
                    row_idx += 1
                    yield (str(r.get("query") or "").strip(), str(r.get("response") or "").strip(), None)

        for res in pool.imap(_worker_encode_uta, _iter_items(), chunksize=128):
            if res is None:
                rejected += 1
                continue
            digest, tokens = res
            if digest in seen:
                rejected += 1
                continue
            seen.add(digest)
            accepted += 1
            prev_total = int(writer.index_receipt.totals.get("train_blocks", start_blocks)) + len(writer._current_blocks)
            _pack_uta_tokens(writer, tokens)
            now_total = int(writer.index_receipt.totals.get("train_blocks", start_blocks)) + len(writer._current_blocks)
            delta = now_total - prev_total
            added_blocks += delta
            src_blocks += delta
            if accepted % 5000 == 0:
                el = time.time() - t0
                tot_tok = (start_blocks + added_blocks) * SEQ_LEN
                print(
                    f"[uta_trace] +{added_blocks:,}/{target_new_blocks:,} new blocks | total={tot_tok/1e6:.2f}M tok | eps={accepted:,} | src={corpus_name} ({src_blocks:,}/{quota_blocks:,} blk) | {added_blocks*SEQ_LEN/el:,.0f} tok/s",
                    flush=True,
                )
            if src_blocks >= quota_blocks or added_blocks >= target_new_blocks:
                break

        writer.record_provenance(
            corpus=corpus_name,
            source_file=Path(fpath).name,
            row_start=1,
            row_end=row_idx,
            episodes_accepted=accepted,
            episodes_rejected_stage0=rejected,
        )

    writer.flush_shard()
    tot_b = int(writer.index_receipt.totals.get("train_blocks", 0))
    print(
        f"[uta_trace] DONE in {time.time()-t0:.1f}s! Train blocks: {tot_b:,} ({tot_b*SEQ_LEN/1e6:.2f}M tokens across {sum(1 for s in writer.index_receipt.shards if s['split']=='train')} train shards)",
        flush=True,
    )


def expand_typed_decision(pool: mp.pool.Pool, target_new_blocks: int = 24_576) -> None:
    t0 = time.time()
    writer, seen = _prepare_track_dir("typed_decision", initial_train_shards=1)
    start_blocks = int(writer.index_receipt.totals.get("train_blocks", 5891))
    print(
        f"[typed_decision] Starting expansion from {start_blocks:,} blocks ({len(seen):,} existing eps) -> target +{target_new_blocks:,} blocks",
        flush=True,
    )

    cache_path = SHARDS_ROOT / "typed_decision" / "decision_examples_cache.pt"
    cache_backup = SHARDS_ROOT / "typed_decision" / "decision_examples_cache.stage1_backup.pt"
    if not cache_backup.exists():
        shutil.copy2(cache_path, cache_backup)
    base_dec_examples: list[DecisionExample] = list(
        torch.load(cache_backup, map_location="cpu", weights_only=False)
    )
    new_dec_examples: list[DecisionExample] = []

    p_sciq = hf_hub_download(repo_id="allenai/sciq", repo_type="dataset", filename="data/train-00000-of-00001.parquet")
    p_obqa = hf_hub_download(repo_id="allenai/openbookqa", repo_type="dataset", filename="additional/train-00000-of-00001.parquet")
    p_arcc = hf_hub_download(repo_id="allenai/ai2_arc", repo_type="dataset", filename="ARC-Challenge/train-00000-of-00001.parquet")
    p_arce = hf_hub_download(repo_id="allenai/ai2_arc", repo_type="dataset", filename="ARC-Easy/train-00000-of-00001.parquet")
    p_aqua = hf_hub_download(repo_id="deepmind/aqua_rat", repo_type="dataset", filename="raw/train-00000-of-00001.parquet")
    p_mmlu = hf_hub_download(repo_id="cais/mmlu", repo_type="dataset", filename="all/auxiliary_train-00000-of-00001.parquet")

    sources = [
        ("allenai/sciq", p_sciq, "sciq", 5_000),
        ("allenai/openbookqa", p_obqa, "obqa", 2_000),
        ("allenai/ai2_arc:Challenge", p_arcc, "arc", 1_000),
        ("allenai/ai2_arc:Easy", p_arce, "arc", 1_500),
        ("deepmind/aqua_rat", p_aqua, "aqua", 10_000),
        ("cais/mmlu:auxiliary_train", p_mmlu, "mmlu", 10_000),
    ]

    added_blocks = 0
    rng = random.Random(42)

    for corpus_name, fpath, kind, quota_blocks in sources:
        if added_blocks >= target_new_blocks:
            break
        src_blocks = 0
        accepted = 0
        rejected = 0
        row_idx = 0

        def _iter_dec():
            nonlocal row_idx
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=1024):
                for r in batch.to_pylist():
                    row_idx += 1
                    qid = f"{kind}_{row_idx:06d}"
                    if kind == "sciq":
                        q = str(r.get("question") or "").strip()
                        corr = str(r.get("correct_answer") or "").strip()
                        d1 = str(r.get("distractor1") or "").strip()
                        d2 = str(r.get("distractor2") or "").strip()
                        d3 = str(r.get("distractor3") or "").strip()
                        sup = str(r.get("support") or "").strip()
                        if q and corr and d1 and d2 and d3:
                            opts = [corr, d1, d2, d3]
                            # Deterministic permutation by row_idx so target isn't always 0
                            perm = [0, 1, 2, 3]
                            random.Random(row_idx).shuffle(perm)
                            shuf_opts = [opts[p] for p in perm]
                            best_idx = perm.index(0)
                            yield (qid, "science_mcqa", q, shuf_opts, best_idx, sup or None)
                    elif kind in ("obqa", "arc"):
                        q = str(r.get("question_stem") or r.get("question") or "").strip()
                        ch = r.get("choices") or {}
                        texts = list(ch.get("text") or [])
                        labels = [str(l).strip().upper() for l in (ch.get("label") or [])]
                        ak = str(r.get("answerKey") or "").strip().upper()
                        fact = str(r.get("fact1") or "").strip()
                        if q and len(texts) >= 2 and ak in labels:
                            yield (qid, "science_mcqa", q, texts, labels.index(ak), fact or None)
                    elif kind == "aqua":
                        q = str(r.get("question") or "").strip()
                        raw_opts = list(r.get("options") or [])
                        rat = str(r.get("rationale") or "").strip()
                        corr = str(r.get("correct") or "").strip().upper()
                        letters = ["A", "B", "C", "D", "E"]
                        clean_o = [re.sub(r"^[A-E][\)\.\:]\s*", "", str(o).strip()) for o in raw_opts]
                        if q and len(clean_o) == 5 and corr in letters:
                            yield (qid, "math_mcqa", q, clean_o, letters.index(corr), rat or None)
                    elif kind == "mmlu":
                        q = str(r.get("question") or "").strip()
                        opts = [str(o).strip() for o in (r.get("choices") or [])]
                        ans = r.get("answer")
                        subj = str(r.get("subject") or "general").strip()
                        if q and len(opts) >= 2 and isinstance(ans, int) and 0 <= ans < len(opts):
                            yield (qid, f"mmlu_{subj}" if subj else "mcqa", q, opts, ans, None)

        for res in pool.imap(_worker_encode_decision, _iter_dec(), chunksize=128):
            if res is None:
                rejected += 1
                continue
            digest, tokens, dec_ex = res
            if digest in seen:
                rejected += 1
                continue
            seen.add(digest)
            accepted += 1
            if dec_ex is not None and len(new_dec_examples) < 10_000 and (accepted % 5 == 0):
                new_dec_examples.append(dec_ex)
            prev_total = int(writer.index_receipt.totals.get("train_blocks", start_blocks)) + len(writer._current_blocks)
            _pack_uta_tokens(writer, tokens)
            now_total = int(writer.index_receipt.totals.get("train_blocks", start_blocks)) + len(writer._current_blocks)
            delta = now_total - prev_total
            added_blocks += delta
            src_blocks += delta
            if accepted % 5000 == 0:
                el = time.time() - t0
                tot_tok = (start_blocks + added_blocks) * SEQ_LEN
                print(
                    f"[typed_decision] +{added_blocks:,}/{target_new_blocks:,} new blocks | total={tot_tok/1e6:.2f}M tok | eps={accepted:,} | src={corpus_name} ({src_blocks:,}/{quota_blocks:,} blk) | {added_blocks*SEQ_LEN/el:,.0f} tok/s",
                    flush=True,
                )
            if src_blocks >= quota_blocks or added_blocks >= target_new_blocks:
                break

        writer.record_provenance(
            corpus=corpus_name,
            source_file=Path(fpath).name,
            row_start=1,
            row_end=row_idx,
            episodes_accepted=accepted,
            episodes_rejected_stage0=rejected,
        )

    writer.flush_shard()
    rng.shuffle(new_dec_examples)
    combined_dec = base_dec_examples + new_dec_examples[:10_000]
    tmp_pt = cache_path.with_suffix(".pt.tmp")
    torch.save(combined_dec, tmp_pt)
    os.replace(tmp_pt, cache_path)
    tot_b = int(writer.index_receipt.totals.get("train_blocks", 0))
    print(
        f"[typed_decision] DONE in {time.time()-t0:.1f}s! Train blocks: {tot_b:,} ({tot_b*SEQ_LEN/1e6:.2f}M tokens), decision_examples_cache.pt: {len(base_dec_examples):,} -> {len(combined_dec):,} examples",
        flush=True,
    )


def main() -> None:
    t_start = time.time()
    print("=== STARTING MORPHCODE-16K CORPUS EXPANSION ===", flush=True)
    with mp.Pool(processes=12, initializer=_init_worker) as pool:
        expand_typed_decision(pool, target_new_blocks=24_576)

    # Print final summary across all 3 tracks
    grand_tok = 0
    grand_unmasked = 0
    grand_blocks = 0
    grand_eps = 0
    print("\n=== FINAL MORPHCODE-16K CORPUS RECEIPT AUDIT ===", flush=True)
    for track in ("pretrain_lm", "uta_trace", "typed_decision"):
        idx = IndexReceipt.from_file(SHARDS_ROOT / track / "index_receipt.json")
        tr_shards = [s for s in idx.shards if s["split"] == "train"]
        va_shards = [s for s in idx.shards if s["split"] == "val"]
        t_tok = sum(s["num_tokens"] for s in tr_shards)
        t_unm = sum(s["unmasked_target_tokens"] for s in tr_shards)
        t_blk = sum(s["num_blocks"] for s in tr_shards)
        t_eps = sum(s["num_episodes"] for s in tr_shards)
        grand_tok += t_tok
        grand_unmasked += t_unm
        grand_blocks += t_blk
        grand_eps += t_eps
        print(
            f"  {track:16s}: {len(tr_shards):2d} train shards | {t_blk:7,d} blocks | {t_tok/1e6:7.2f}M raw tok | {t_unm/1e6:7.2f}M target tok | {t_eps:7,d} eps | val_shards={len(va_shards)} ({sum(s['num_blocks'] for s in va_shards)} blk)",
            flush=True,
        )
    print(
        f"  {'TOTAL TRAIN':16s}: {grand_blocks:7,d} blocks | {grand_tok/1e6:7.2f}M raw tok | {grand_unmasked/1e6:7.2f}M target tok | {grand_eps:7,d} eps (elapsed {time.time()-t_start:.1f}s)",
        flush=True,
    )
    print("=== ALL SHARDS BUILT SUCCESSFULLY ===", flush=True)


if __name__ == "__main__":
    main()
