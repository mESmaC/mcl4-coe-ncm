"""Sharded Binary Corpus Pipeline & Index Receipt System for `MCL4-COE`.

Turns raw multi-hundred-gigabyte corpora (`X:\\corpora\\chat` and `X:\\corpora\\text`)
into deterministic, resumable `.bin` shards (`uint16` / `.tok16`) paired with:
1. Per-shard receipts (`shard_XXXXX.receipt.json`) recording token/block/episode counts,
   unmasked target token counts, SHA-256 checksums, and exact file/row provenance.
2. A master Index Receipt (`index_receipt.json`) that `ShardedTokenBlockDataset` reads
   to memory-map and stream blocks across hundreds of shards with $O(\\log N_{\\text{shards}})$
   lookup and zero episode severing across `seq_len` block boundaries.
"""

from __future__ import annotations

import codecs
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
import operator
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .curation import (
    ByteSieveConfig,
    build_head_tail_window,
    evaluate_byte_sieve,
    normalize_prose_unwrapping,
    sanitize_trace_markers,
)
from .data import TRACE_MARKERS, ByteTokenizer, get_tokenizer

EXCLUDED_CORPORA: frozenset[str] = frozenset({"tinystories"})

_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL | re.IGNORECASE)
_GSM_HASH_RE = re.compile(r"^(.*?)\n*####\s*(.+)$", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")

METADATA_JSON_FILENAMES: frozenset[str] = frozenset(
    {
        "dataset_infos.json",
        "dataset_manifest.json",
        "quality_audit.json",
        "build-manifest.json",
        "build-report.jsonl",
        "build-summary.json",
        "failed-tasks.json",
        "fixed-source-audit.json",
        "outdated-datasets.json",
        "release-audit.json",
        "webinstruct-addition.json",
        "sha256sums.json",
        "label_stats.json",
    }
)


@dataclass(frozen=True)
class CorpusSpec:
    """Processing specification for a discovered corpus directory."""

    name: str
    root_path: str
    track: str  # "pretrain_lm", "uta_trace", or "typed_decision"
    files: tuple[str, ...]
    total_bytes: int
    format_hint: str


@dataclass(frozen=True)
class ShardProvenance:
    """Exact source provenance entry recorded inside a shard receipt."""

    corpus: str
    source_file: str
    row_start: int
    row_end: int
    episodes_accepted: int
    episodes_rejected_stage0: int


@dataclass(frozen=True)
class ShardReceipt:
    """Per-shard receipt (`shard_XXXXX.receipt.json`) for a packed `.bin` shard."""

    shard_id: str
    bin_file: str
    receipt_file: str
    track: str
    split: str
    dtype: str
    byte_order: str
    seq_len: int
    num_tokens: int
    num_blocks: int
    num_episodes: int
    unmasked_target_tokens: int
    num_bytes: int
    sha256: str
    created_at_unix: float
    provenance: tuple[ShardProvenance, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = [asdict(p) for p in self.provenance]
        return payload


@dataclass
class IndexReceipt:
    """Master index receipt (`index_receipt.json`) governing a sharded dataset build."""

    format_version: int = 1
    pipeline: str = "mcl4_coe_sharded_v1"
    dtype: str = "uint16"
    byte_order: str = "little"
    seq_len: int = 2048
    max_tokens_per_shard: int = 67_108_864  # 64M tokens = 128 MB per shard
    trace_markers: dict[str, str] = field(default_factory=lambda: dict(TRACE_MARKERS))
    excluded_corpora: list[str] = field(default_factory=lambda: sorted(EXCLUDED_CORPORA))
    shards: list[dict[str, Any]] = field(default_factory=list)
    completed_source_files: list[str] = field(default_factory=list)
    totals: dict[str, int] = field(
        default_factory=lambda: {
            "num_shards": 0,
            "num_tokens": 0,
            "num_blocks": 0,
            "num_episodes": 0,
            "unmasked_target_tokens": 0,
            "rejected_stage0": 0,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str | Path) -> IndexReceipt:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            format_version=int(raw.get("format_version", 1)),
            pipeline=str(raw.get("pipeline", "mcl4_coe_sharded_v1")),
            dtype=str(raw.get("dtype", "uint16")),
            byte_order=str(raw.get("byte_order", "little")),
            seq_len=int(raw["seq_len"]),
            max_tokens_per_shard=int(raw.get("max_tokens_per_shard", 67_108_864)),
            trace_markers=dict(raw.get("trace_markers", TRACE_MARKERS)),
            excluded_corpora=list(raw.get("excluded_corpora", sorted(EXCLUDED_CORPORA))),
            shards=list(raw.get("shards", [])),
            completed_source_files=list(raw.get("completed_source_files", [])),
            totals=dict(raw.get("totals", {})),
        )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tmp.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _classify_corpus_track(corpus_rel_path: str) -> str:
    lower = corpus_rel_path.lower().replace("\\", "/")
    name = lower.split("/")[-1]
    if name in EXCLUDED_CORPORA:
        return "excluded"
    if "typed-decisions" in name or "tasksource" in name:
        return "typed_decision"
    if lower.startswith("chat/") or name in {
        "supra-reasoning-cot-dataset-v2",
        "mathinstruct",
        "metamathqa",
        "code-alpaca-20k",
        "github-codereview",
        "no_robots",
        "oasst1",
        "hh-rlhf",
        "nemotron-math-proofs-v2",
    }:
        return "uta_trace"
    return "pretrain_lm"


def _is_valid_data_file(path: Path) -> bool:
    if not path.is_file():
        return False
    parts_lower = [part.lower() for part in path.parts]
    if any(part.startswith(".") for part in parts_lower):
        return False
    if path.name.lower() in METADATA_JSON_FILENAMES:
        return False

    # Corpus-specific deduplication & hygiene rules:
    # 1. tasksource-jev-typed-decisions: use `filtered-full/*.parquet` + `data/validation-*` + `data/test-*`
    #    (skip duplicate `full/`, `vision/`, and `quality/excluded.parquet`).
    if "tasksource-jev-typed-decisions" in parts_lower:
        if path.suffix.lower() != ".parquet":
            return False
        if "filtered-full" in parts_lower:
            return True
        if "data" in parts_lower and (
            path.name.startswith("validation-") or path.name.startswith("test-")
        ):
            return True
        return False

    # 2. amazon-c11-distillation-filtered: data lives in `datasets/**/*.parquet`, not `blocks/block-*.json`.
    if "amazon-c11-distillation-filtered" in parts_lower:
        return path.suffix.lower() == ".parquet" and "datasets" in parts_lower

    # 3. wikipedia-en: data lives in `20231101.en/*.parquet`, not sha1 `.json` metadata files.
    if "wikipedia-en" in parts_lower:
        return path.suffix.lower() == ".parquet"

    # 4. precinct6-cybersecurity: use `.parquet` and `.jsonl`, skip `build/` graph/summary `.json`.
    if "precinct6-cybersecurity" in parts_lower:
        return path.suffix.lower() in {".parquet", ".jsonl"}

    name_lower = path.name.lower()
    if name_lower.endswith(".jsonl.zst"):
        uncompressed = path.with_suffix("")
        if uncompressed.exists():
            return False
        return True
    if "sample_1k" in name_lower:
        return False
    return path.suffix.lower() in {".parquet", ".jsonl", ".json", ".txt"}


def discover_corpora(corpora_root: str | Path = r"X:\corpora") -> list[CorpusSpec]:
    """Scan `X:\\corpora` (or custom root) and produce an inventory Receipt of valid corpora."""
    root = Path(corpora_root).expanduser().resolve(strict=True)
    specs: list[CorpusSpec] = []

    candidates: list[Path] = []
    for category in ("chat", "text"):
        cat_dir = root / category
        if cat_dir.is_dir():
            candidates.extend(sorted(p for p in cat_dir.iterdir() if p.is_dir()))
    if not candidates:
        candidates = sorted(p for p in root.iterdir() if p.is_dir())

    for sub in candidates:
        rel = str(sub.relative_to(root)).replace("\\", "/")
        track = _classify_corpus_track(rel)
        if track == "excluded":
            continue
        files = sorted(p for p in sub.rglob("*") if _is_valid_data_file(p))
        if not files:
            continue
        total_bytes = sum(p.stat().st_size for p in files)
        exts = sorted({p.suffix.lower() for p in files})
        specs.append(
            CorpusSpec(
                name=rel,
                root_path=str(sub),
                track=track,
                files=tuple(str(p) for p in files),
                total_bytes=total_bytes,
                format_hint=",".join(exts),
            )
        )
    return specs


def _stream_zstd_lines(raw_stream: Any) -> Iterator[str]:
    import zstandard  # type: ignore[import-not-found]

    decompressor = zstandard.ZstdDecompressor().decompressobj()
    decoder = io.IncrementalNewlineDecoder(codecs.getincrementaldecoder("utf-8-sig")(), True)
    pending = ""
    while chunk := raw_stream.read(1024 * 1024):
        while chunk:
            if decompressor.eof:
                decompressor = zstandard.ZstdDecompressor().decompressobj()
            decoded = decompressor.decompress(chunk)
            if decoded:
                lines = (pending + decoder.decode(decoded)).split("\n")
                pending = lines.pop()
                for line in lines:
                    yield line
            chunk = decompressor.unused_data
    if tail := decompressor.flush():
        lines = (pending + decoder.decode(tail)).split("\n")
        pending = lines.pop()
        for line in lines:
            yield line
    pending += decoder.decode(b"", final=True)
    if pending:
        yield pending


def iter_source_records(path: str | Path, *, batch_size: int = 256) -> Iterator[tuple[int, Any]]:
    """Stream records from `.parquet`, `.jsonl`, `.jsonl.zst`, `.json`, or `.txt` files."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    name_lower = file_path.name.lower()

    if suffix == ".parquet":
        import pyarrow.parquet as pq  # type: ignore[import-not-found]

        with pq.ParquetFile(file_path) as pf:
            row_idx = 0
            for batch in pf.iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    row_idx += 1
                    yield row_idx, row
        return

    if name_lower.endswith(".jsonl.zst"):
        with file_path.open("rb") as raw:
            for line_idx, line in enumerate(_stream_zstd_lines(raw), start=1):
                if line.strip():
                    try:
                        yield line_idx, json.loads(line)
                    except json.JSONDecodeError:
                        continue
        return

    if suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line_idx, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield line_idx, json.loads(line)
                    except json.JSONDecodeError:
                        continue
        return

    if suffix == ".json":
        with file_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            first_char = ""
            while True:
                ch = handle.read(1)
                if not ch or not ch.isspace():
                    first_char = ch
                    break
            handle.seek(0)
            if first_char == "[":
                data = json.load(handle)
                if isinstance(data, list):
                    for idx, item in enumerate(data, start=1):
                        yield idx, item
            else:
                # JSON-lines stored with a .json extension (e.g. arxiv-papers-0000.json)
                for line_idx, line in enumerate(handle, start=1):
                    if line.strip():
                        try:
                            yield line_idx, json.loads(line)
                        except json.JSONDecodeError:
                            continue
        return

    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8-sig", errors="replace")
        for chunk_idx, chunk_text in enumerate(_clean_and_chunk_gutenberg_txt(text), start=1):
            yield chunk_idx, {"text": chunk_text}
        return


_GUTENBERG_LANG_RE = re.compile(r"^Language\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\r\n]*\*\*\*",
    re.IGNORECASE,
)
_GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\r\n]*\*\*\*|"
    r"End of (?:the )?Project Gutenberg(?:'s)? (?:EBook|Etext)",
    re.IGNORECASE,
)


def _clean_and_chunk_gutenberg_txt(
    raw_text: str,
    *,
    max_chunk_chars: int = 24_000,
) -> list[str]:
    """Filter non-English Project Gutenberg `.txt` books, strip license headers/footers, unwrap 72-char lines, and chunk."""
    if not raw_text or not raw_text.strip():
        return []

    header_window = raw_text[:3500]
    m_lang = _GUTENBERG_LANG_RE.search(header_window)
    if m_lang and "english" not in m_lang.group(1).lower():
        return []

    body = raw_text
    m_start = _GUTENBERG_START_RE.search(body)
    if m_start:
        body = body[m_start.end() :]
    m_end = _GUTENBERG_END_RE.search(body)
    if m_end:
        body = body[: m_end.start()]

    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not body:
        return []

    raw_paras = re.split(r"\n\s*\n+", body)
    clean_paras: list[str] = []
    for para in raw_paras:
        lines = [ln for ln in para.splitlines() if ln.strip()]
        if not lines:
            continue
        if any(ln.startswith("  ") or ln.startswith("\t") for ln in lines):
            clean_paras.append("\n".join(ln.rstrip() for ln in lines))
        else:
            clean_paras.append(" ".join(ln.strip() for ln in lines))

    chunks: list[str] = []
    cur_buf: list[str] = []
    cur_len = 0
    for para in clean_paras:
        p_len = len(para)
        if p_len > max_chunk_chars:
            if cur_buf:
                chunks.append("\n\n".join(cur_buf))
                cur_buf.clear()
                cur_len = 0
            for off in range(0, p_len, max_chunk_chars):
                sub = para[off : off + max_chunk_chars].strip()
                if sub:
                    chunks.append(sub)
            continue
        if cur_buf and cur_len + p_len + 2 > max_chunk_chars:
            chunks.append("\n\n".join(cur_buf))
            cur_buf = [para]
            cur_len = p_len
        else:
            cur_buf.append(para)
            cur_len += p_len + (2 if len(cur_buf) > 1 else 0)
    if cur_buf:
        chunks.append("\n\n".join(cur_buf))
    return chunks


def _split_thought_and_answer(raw_response: str, explicit_thought: str | None = None) -> tuple[str, str]:
    """Ensure every reasoning trace has both a non-empty `#T$#` scratchpad and `#A$#` answer."""
    resp = raw_response.strip()
    if explicit_thought and explicit_thought.strip():
        thought = explicit_thought.strip()
        answer = resp if resp else thought.split("\n")[-1].strip()
        return thought, answer

    # 1. Check `<think>...</think>`
    m_think = _THINK_TAG_RE.search(resp)
    if m_think:
        t_part = m_think.group(1).strip()
        a_part = m_think.group(2).strip()
        if t_part and a_part:
            return t_part, a_part
        if t_part and not a_part:
            lines = [ln.strip() for ln in t_part.splitlines() if ln.strip()]
            return t_part, (lines[-1] if lines else t_part)

    # 2. Check GSM8K / Math `#### <final_answer>`
    m_gsm = _GSM_HASH_RE.search(resp)
    if m_gsm and m_gsm.group(1).strip() and m_gsm.group(2).strip():
        return m_gsm.group(1).strip(), m_gsm.group(2).strip()

    # 3. Check `\boxed{...}` in math solutions
    m_box = _BOXED_RE.findall(resp)
    if m_box and len(resp) > len(m_box[-1]) + 10:
        return resp, m_box[-1].strip()

    # 4. Code responses: preserve leading indentation and keep the complete executable block in `#A$#`
    raw_lines = [ln.rstrip() for ln in resp.splitlines() if ln.strip()]
    has_code_indent = any(ln.startswith(("    ", "\t")) for ln in raw_lines)
    starts_with_code = resp.startswith(("def ", "class ", "import ", "from ", "#include", "function ", "const ", "let ", "public ", "```"))
    if starts_with_code or has_code_indent or "```" in resp:
        if "```" in resp:
            parts = resp.split("```", 1)
            prose_head = parts[0].strip()
            code_body = "```" + parts[1].strip()
            if prose_head:
                return prose_head, code_body
        first_line = raw_lines[0].strip() if raw_lines else "implementation"
        return f"Outline the structure and control flow for `{first_line}`.", "\n".join(raw_lines)

    # 5. Multi-paragraph or multi-line prose derivation: use preceding steps as `#T$#` and final block as `#A$#`
    paragraphs = [p.strip() for p in resp.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        return "\n\n".join(paragraphs[:-1]), paragraphs[-1]

    if len(raw_lines) >= 2:
        return "\n".join(raw_lines[:-1]), raw_lines[-1].strip()

    # 6. Single-sentence fallback: split sentences or clause boundaries without duplicating resp or injecting boilerplate
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", resp) if s.strip()]
    if len(sentences) >= 2:
        return " ".join(sentences[:-1]), sentences[-1]
    for sep in ("; ", ": ", ", so ", ", therefore ", " because ", ", which ", ", "):
        if sep in resp:
            left, right = resp.rsplit(sep, 1)
            if left.strip() and right.strip():
                return left.strip(), right.strip()
    words = resp.split()
    if len(words) >= 6:
        mid = len(words) // 2
        return " ".join(words[:mid]), " ".join(words[mid:])
    return resp, resp


def _truncate_contiguous_sentences(text: str, max_chars: int) -> str:
    """Keep contiguous complete sentences from the start of `text` up to `max_chars` (never splicing `\n...\n`)."""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    window = stripped[:max_chars]
    # Snap to the last sentence boundary (., !, ?, or newline) if past 55% of max_chars
    best_end = -1
    for m in re.finditer(r"[.!?](?:\s|$)|\n", window):
        if m.end() >= int(max_chars * 0.55):
            best_end = m.end()
    if best_end > 0:
        return window[:best_end].strip()
    # Otherwise snap to the last word boundary
    last_space = window.rfind(" ")
    if last_space >= int(max_chars * 0.50):
        return window[:last_space].strip()
    return window.strip()


def _truncate_utf8_bytes(text: str, max_bytes: int) -> str:
    """Safely truncate `text` on a contiguous sentence or word boundary so UTF-8 byte length is `<= max_bytes`."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    prefix = encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()
    return _truncate_contiguous_sentences(prefix, len(prefix))


def _compact_uta_triplet(
    u: str,
    t: str,
    a: str,
    *,
    max_total_chars: int = 1900,
    max_total_bytes: int = 1960,
) -> tuple[str, str, str]:
    """Sanitize and compact `#U$#`, `#T$#`, and `#A$#` contiguously so every episode fits inside `seq_len`."""
    u_s = sanitize_trace_markers(u).strip()
    t_s = sanitize_trace_markers(t).strip()
    a_s = sanitize_trace_markers(a).strip()

    u_cap = min(640, max(64, int(max_total_bytes * 0.34)))
    a_cap = min(560, max(48, int(max_total_bytes * 0.28)))
    t_min = min(240, max(48, int(max_total_bytes * 0.20)))

    # 1. Cap oversized prompt (#U$#) and answer (#A$#) contiguously at sentence boundaries
    if len(u_s) > u_cap + 60:
        u_s = _truncate_contiguous_sentences(u_s, u_cap)
    if len(a_s) > a_cap + 60:
        a_s = _truncate_contiguous_sentences(a_s, a_cap)

    # 2. Compact verbose scratchpad (#T$#) contiguously to fit remaining character budget
    if len(u_s) + len(t_s) + len(a_s) > max_total_chars and len(t_s) > t_min:
        budget = max(t_min, max_total_chars - len(u_s) - len(a_s))
        t_s = _truncate_contiguous_sentences(t_s, budget)

    # 3. Guarantee UTF-8 byte length <= max_total_bytes (accounting for multibyte math/Unicode)
    u_b = len(u_s.encode("utf-8"))
    t_b = len(t_s.encode("utf-8"))
    a_b = len(a_s.encode("utf-8"))
    if u_b + t_b + a_b > max_total_bytes:
        u_s = _truncate_utf8_bytes(u_s, min(u_b, u_cap))
        a_s = _truncate_utf8_bytes(a_s, min(a_b, a_cap))
        rem = max(48, max_total_bytes - len(u_s.encode("utf-8")) - len(a_s.encode("utf-8")))
        t_s = _truncate_utf8_bytes(t_s, rem)
    return (u_s, t_s, a_s)


def extract_uta_episodes(
    record: Any,
    *,
    max_seq_len: int = 2048,
) -> list[tuple[str, str, str]]:
    """Extract `(prompt, thought, answer)` tuples from heterogeneous chat/CoT records."""
    if not isinstance(record, dict):
        return []

    max_b = max(180, max_seq_len - 88)
    max_c = max(160, max_seq_len - 148)

    def _pack(u_raw: str, t_raw: str, a_raw: str) -> tuple[str, str, str]:
        return _compact_uta_triplet(
            u_raw,
            t_raw,
            a_raw,
            max_total_chars=max_c,
            max_total_bytes=max_b,
        )

    # Case 1: Supra-Reasoning-CoT (`user`, `CoT`, `assistant`)
    if "user" in record and ("CoT" in record or "cot" in record) and "assistant" in record:
        u = str(record.get("user") or "").strip()
        t = str(record.get("CoT") or record.get("cot") or "").strip()
        a = str(record.get("assistant") or "").strip()
        if u and (t or a):
            t_clean, a_clean = _split_thought_and_answer(a, explicit_thought=t)
            return [_pack(u, t_clean, a_clean)]

    # Case 2: Fable-5 (`query`, `thinking`, optional `answer`/`response`)
    if "query" in record and "thinking" in record:
        u = str(record.get("query") or "").strip()
        t = str(record.get("thinking") or "").strip()
        a = str(record.get("answer") or record.get("response") or "").strip()
        if u and (t or a):
            t_clean, a_clean = _split_thought_and_answer(a, explicit_thought=t)
            return [_pack(u, t_clean, a_clean)]

    # Case 3: Instruction / Query / Formal Statement + Response / Output / Proof
    for q_key in (
        "instruction",
        "query",
        "problem",
        "prompt",
        "question",
        "formal_statement",
        "informal_statement",
        "theorem",
        "observation",
        "task",
    ):
        if isinstance(record.get(q_key), str) and record[q_key].strip():
            explicit_t = str(
                record.get("reasoning")
                or record.get("thought")
                or record.get("thinking")
                or record.get("analysis")
                or record.get("informal_proof")
                or ""
            ).strip()
            for a_key in (
                "response",
                "output",
                "solution",
                "answer",
                "final_answer",
                "conclusion",
                "lean4_code",
                "formal_proof",
                "proof",
                "completion",
            ):
                if isinstance(record.get(a_key), str) and record[a_key].strip():
                    u = record[q_key].strip()
                    inp_extra = str(record.get("input") or "").strip()
                    if inp_extra and inp_extra not in u:
                        u = f"{u}\n{inp_extra}"
                    t_clean, a_clean = _split_thought_and_answer(
                        record[a_key], explicit_thought=explicit_t or None
                    )
                    return [_pack(u, t_clean, a_clean)]

    # Case 4: `messages` or `conversations` array (`AM-DeepSeek-R1`, `Nemotron-Cascade-2`, `Nemotron-Post-Training`, etc.)
    msg_list = record.get("messages") or record.get("conversations")
    if isinstance(msg_list, list) and msg_list:
        episodes: list[tuple[str, str, str]] = []
        current_user: list[str] = []
        has_real_user = False
        for msg in msg_list:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("from") or "").lower()
            content = str(msg.get("content") or msg.get("value") or msg.get("text") or "").strip()
            reasoning = str(
                msg.get("reasoning_content")
                or msg.get("thinking")
                or msg.get("reasoning")
                or msg.get("thought")
                or ""
            ).strip()
            if role in ("system", "developer") and content:
                current_user.append(content)
            elif role in ("user", "human", "prompter") and content and content != "-" and len(content) >= 4:
                current_user.append(content)
                has_real_user = True
            elif role in ("tool", "observation", "function", "environment") and content:
                current_user.append(f"[Observation]\n{content}")
                has_real_user = True
            elif role in ("assistant", "gpt", "model", "bot") and (content or reasoning) and has_real_user:
                prompt_str = "\n\n".join(current_user)
                t_clean, a_clean = _split_thought_and_answer(content, explicit_thought=reasoning)
                episodes.append(_pack(prompt_str, t_clean, a_clean))
                current_user.clear()
                has_real_user = False
        return episodes

    # Case 5: Anthropic hh-rlhf (`chosen` multi-turn `Human:` / `Assistant:` transcript)
    if isinstance(record.get("chosen"), str) and "Human:" in record["chosen"] and "Assistant:" in record["chosen"]:
        parts = re.split(r"(?:^|\n\n)(Human|Assistant):\s*", record["chosen"].strip())
        if len(parts) >= 5:
            last_u = ""
            last_a = ""
            for idx in range(1, len(parts) - 1, 2):
                speaker, body = parts[idx], parts[idx + 1].strip()
                if speaker == "Human" and body:
                    last_u = body
                elif speaker == "Assistant" and body and last_u:
                    last_a = body
            if last_u and last_a:
                t_clean, a_clean = _split_thought_and_answer(last_a)
                return [_pack(last_u, t_clean, a_clean)]

    # Case 6: GitHub CodeReview (`before_code`, `reviewer_comment`, `after_code`)
    if "before_code" in record and "reviewer_comment" in record and "after_code" in record:
        before = str(record.get("before_code") or "").strip()
        comment = str(record.get("reviewer_comment") or "").strip()
        after = str(record.get("after_code") or "").strip()
        if before and comment and after:
            return [_pack(f"Review and patch code:\n{before}", comment, after)]

    # Case 7: Nemotron SFT (`text` formatted as `input: ... output: ...`)
    raw_text = record.get("text")
    if isinstance(raw_text, str) and raw_text.lstrip().lower().startswith("input:"):
        parts = re.split(r"\s+output\s*:\s*", raw_text.lstrip()[len("input:") :], maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            t_clean, a_clean = _split_thought_and_answer(parts[1].strip())
            return [_pack(parts[0].strip(), t_clean, a_clean)]

    # Case 8: `codeparrot/github-code` or `arxiv_cplusplus_research_code` (`repo_name`/`repo`, `path`/`file`, `code`/`content`)
    if (
        ("repo_name" in record or "repo" in record)
        and ("path" in record or "file" in record)
        and ("code" in record or "content" in record)
    ):
        norm_rec = normalize_github_code_record(record)
        code_path = str(norm_rec.get("path") or "").strip()
        repo_name = str(norm_rec.get("repo_name") or "").strip()
        lang = str(norm_rec.get("language") or "Code").strip()
        code_str = str(norm_rec.get("code") or "").strip()
        if code_path and code_str and not _is_noisy_code_path(code_path):
            sieve = evaluate_byte_sieve(code_str)
            if sieve.keep and sieve.sanitized_text:
                clean_code = sieve.sanitized_text.replace("\r\n", "\n").replace("\r", "\n").strip()
                decl_lines: list[str] = []
                import_lines: list[str] = []
                for ln in clean_code.splitlines():
                    s_ln = ln.strip()
                    if not s_ln or s_ln.startswith(("#", "//", "/*", "*")) and not s_ln.startswith("#include"):
                        continue
                    if s_ln.startswith(("import ", "from ", "#include", "using ", "require(", "package ", "use ")):
                        if len(import_lines) < 3:
                            import_lines.append(s_ln[:80])
                    elif any(
                        s_ln.startswith(p)
                        for p in (
                            "def ",
                            "async def ",
                            "class ",
                            "function ",
                            "export ",
                            "const ",
                            "pub fn ",
                            "fn ",
                            "func ",
                            "struct ",
                            "template",
                            "namespace ",
                            "void ",
                            "int ",
                            "double ",
                            "float ",
                            "bool ",
                            "auto ",
                            "inline ",
                            "static ",
                            "interface ",
                            "type ",
                            "public ",
                            "private ",
                        )
                    ):
                        if len(decl_lines) < 4:
                            decl_lines.append(s_ln.rstrip("{:").strip()[:90])
                if decl_lines or len(clean_code.splitlines()) >= 6:
                    sym_desc = ", ".join(f"`{d}`" for d in decl_lines[:2]) if decl_lines else f"`{Path(code_path).name}`"
                    imp_desc = f" Dependencies: {', '.join(f'`{i}`' for i in import_lines)}." if import_lines else ""
                    u_code = f"In `{repo_name}` (`{code_path}`, {lang}), implement {sym_desc}."
                    t_code = (
                        f"Target module `{code_path}` ({lang}) in `{repo_name}`.{imp_desc} "
                        f"Key declarations: {'; '.join(decl_lines) if decl_lines else Path(code_path).name}."
                    )
                    return [_pack(u_code, t_code, clean_code)]

    # Case 9: `camel-ai/physics` & `camel-ai/math` (`message_1`, `message_2`, `topic;`, `sub_topic`)
    if isinstance(record.get("message_1"), str) and isinstance(record.get("message_2"), str):
        m1 = record["message_1"].strip()
        m2 = record["message_2"].strip()
        if m1 and m2:
            topic = str(record.get("topic;") or record.get("topic") or "").strip()
            sub_topic = str(record.get("sub_topic") or "").strip()
            header = f"[{topic}: {sub_topic}]\n" if topic and sub_topic else (f"[{topic}]\n" if topic else "")
            t_clean, a_clean = _split_thought_and_answer(m2)
            return [_pack(f"{header}{m1}", t_clean, a_clean)]

    return []


_GITHUB_CODE_LANG_TO_EXT: dict[str, tuple[str, ...]] = {
    "Assembly": (".asm",),
    "Batchfile": (".bat", ".cmd"),
    "C": (".c", ".h"),
    "C#": (".cs",),
    "C++": (".cpp", ".hpp", ".c++", ".h++", ".cc", ".hh", ".cxx", ".hxx", ".C", ".H"),
    "CMake": (".cmake",),
    "CSS": (".css",),
    "Dockerfile": (".dockerfile", "Dockerfile"),
    "FORTRAN": (".f90", ".f", ".f03", ".f08", ".f77", ".f95", ".for", ".fpp"),
    "GO": (".go",),
    "Haskell": (".hs",),
    "HTML": (".html",),
    "Java": (".java",),
    "JavaScript": (".js",),
    "Julia": (".jl",),
    "Lua": (".lua",),
    "Makefile": ("Makefile",),
    "Markdown": (".md", ".markdown"),
    "PHP": (".php", ".php3", ".php4", ".php5", ".phps", ".phpt"),
    "Perl": (".pl", ".pm", ".pod", ".perl"),
    "PowerShell": (".ps1", ".psd1", ".psm1"),
    "Python": (".py",),
    "Ruby": (".rb",),
    "Rust": (".rs",),
    "SQL": (".sql",),
    "Scala": (".scala",),
    "Shell": (".sh", ".bash", ".command", ".zsh"),
    "TypeScript": (".ts", ".tsx"),
    "TeX": (".tex",),
    "Visual Basic": (".vb",),
}

_GITHUB_CODE_EXT_TO_LANG: dict[str, str] = {
    ext: lang for lang, exts in _GITHUB_CODE_LANG_TO_EXT.items() for ext in exts
}

_NOISY_CODE_PATH_SUBSTRINGS: tuple[str, ...] = (
    "node_modules/",
    "bower_components/",
    "vendor/",
    "third_party/",
    "third-party/",
    "/dist/",
    "/build/",
    ".min.js",
    ".min.css",
    ".bundle.js",
    "package-lock.json",
    "yarn.lock",
    "cargo.lock",
    "__pycache__/",
    "_pb2.py",
    ".pb.go",
    ".pb.cc",
    ".pb.h",
)


def infer_github_code_language(path: str) -> str:
    """Infer programming language from file path using the `codeparrot/github-code` extension table."""
    clean = str(path or "").strip()
    for ext, lang in _GITHUB_CODE_EXT_TO_LANG.items():
        if clean.endswith(ext):
            return lang
    return "Unknown"


def _is_noisy_code_path(path: str) -> bool:
    """Return True if `path` belongs to vendored dependencies, minified bundles, or generated lockfiles."""
    norm = f"/{str(path or '').replace(chr(92), '/').lower().lstrip('/')}"
    return any(tok in norm for tok in _NOISY_CODE_PATH_SUBSTRINGS)


def normalize_github_code_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw Parquet or dataset rows from `codeparrot/github-code` or `arxiv_cplusplus_research_code`.

    Canonical fields: `code`, `repo_name`, `path`, `language`, `license`, `size`.
    """
    code_val = record.get("code")
    if not isinstance(code_val, str) or not code_val:
        code_val = record.get("content") or ""
    path_val = str(record.get("path") or record.get("file") or "")
    repo_val = str(record.get("repo_name") or record.get("repo") or "")
    lang_val = str(record.get("language") or "").strip()
    if not lang_val:
        ext_type = str(record.get("extension_type") or "").strip().lower().lstrip(".")
        if ext_type in ("cpp", "cxx", "cc", "hpp", "hxx", "h"):
            lang_val = "C++"
        else:
            lang_val = infer_github_code_language(path_val)
    size_raw = record.get("size") if record.get("size") is not None else record.get("file_length")
    try:
        size_val = int(size_raw) if size_raw is not None else len(str(code_val))
    except (TypeError, ValueError):
        size_val = len(str(code_val))
    return {
        "code": str(code_val),
        "repo_name": repo_val,
        "path": path_val,
        "language": lang_val,
        "license": str(record.get("license") or ""),
        "size": size_val,
    }


_MCQA_OPTION_LINE_RE = re.compile(r"^([A-Z])[\.\:]\s+(.+)$")
_MCQA_ANSWER_LINE_RE = re.compile(r"^Answer\s*:\s*([A-Z])\b", re.IGNORECASE)
_GPT2_DECODER: Any = None


def _decode_gpt2_tokens(tokens: list[int]) -> str | None:
    """Decode GPT-2 BPE token IDs (e.g. from `nvidia/Nemotron-ClimbMix`) into UTF-8 text."""
    global _GPT2_DECODER
    if _GPT2_DECODER is False:
        return None
    if _GPT2_DECODER is None:
        try:
            import tiktoken  # type: ignore[import-not-found]

            _GPT2_DECODER = ("tiktoken", tiktoken.get_encoding("gpt2"))
        except Exception:
            try:
                from transformers import AutoTokenizer  # type: ignore[import-not-found]

                _GPT2_DECODER = (
                    "transformers",
                    AutoTokenizer.from_pretrained("gpt2", model_max_length=1_000_000),
                )
            except Exception:
                _GPT2_DECODER = False
                return None
    kind, dec = _GPT2_DECODER
    try:
        if kind == "tiktoken":
            return str(dec.decode(tokens))
        return str(dec.decode(tokens, skip_special_tokens=True))
    except Exception:
        return None


def parse_mcqa_decision_fields(
    record: Any,
) -> tuple[str, str, str, list[str], int, str | None] | None:
    """Normalize heterogeneous MCQA / decision records into `(state, kind, question, options, best_idx, rationale)`.

    Supports:
    1. `jev/typed-decisions` (`state`, `kind`, `question`, `options`, `target`)
    2. `nvidia/Nemotron-RL-knowledge-mcqa` (`responses_create_params`, `options` list of letter dicts, `expected_answer`)
    3. `nvidia/Nemotron-CrossThink` (`meta_data.question` + `reward_model.ground_truth`)
    4. `nvidia/Nemotron-Pretraining-Specialized-v1.1` (`Nemotron-Pretraining-Multiple-Choice` `text` with `Answer: X` + rationale)
    """
    if not isinstance(record, dict):
        return None

    # Schema 2: `nvidia/Nemotron-RL-knowledge-mcqa`
    if "responses_create_params" in record and "expected_answer" in record and isinstance(record.get("options"), list):
        rcp = record.get("responses_create_params")
        raw_input = rcp.get("input") if isinstance(rcp, dict) else None
        raw_content = (
            str(raw_input[0].get("content") or "").strip()
            if isinstance(raw_input, list) and raw_input and isinstance(raw_input[0], dict)
            else ""
        )
        letters: list[str] = []
        opts: list[str] = []
        for item in record["options"]:
            if isinstance(item, dict):
                for k, v in sorted(item.items()):
                    if isinstance(v, str) and v.strip():
                        letters.append(str(k).strip().upper())
                        opts.append(sanitize_trace_markers(v.strip()))
                        break
        exp_ans = str(record.get("expected_answer") or "").strip().upper()
        if raw_content and len(opts) >= 2 and exp_ans in letters:
            best_idx = letters.index(exp_ans)
            lines = [ln.rstrip() for ln in raw_content.splitlines()]
            stem_lines: list[str] = []
            for ln in lines:
                s_ln = ln.strip()
                if s_ln.startswith("Answer the following multiple choice question"):
                    continue
                if _MCQA_OPTION_LINE_RE.match(s_ln):
                    break
                stem_lines.append(ln)
            state = "\n".join(stem_lines).strip() or raw_content
            return state, "mcqa", "Select the option that best answers the question.", opts, best_idx, None

    # Schema 3: `nvidia/Nemotron-CrossThink`
    if "reward_model" in record and isinstance(record.get("reward_model"), dict):
        meta = record.get("meta_data") if isinstance(record.get("meta_data"), dict) else {}
        q_raw = str(meta.get("question") or "").strip()
        if not q_raw and isinstance(record.get("prompt"), list) and record["prompt"]:
            first_p = record["prompt"][0]
            if isinstance(first_p, dict):
                q_raw = str(first_p.get("content") or "").strip()
        gt = str(record["reward_model"].get("ground_truth") or "").strip()
        gt_letter = gt.split(":", 1)[0].split(".", 1)[0].strip().upper()
        if q_raw and gt_letter:
            stem_lines = []
            letters = []
            opts = []
            in_opts = False
            for ln in q_raw.splitlines():
                m = _MCQA_OPTION_LINE_RE.match(ln.strip())
                if m:
                    in_opts = True
                    letters.append(m.group(1).upper())
                    opts.append(sanitize_trace_markers(m.group(2).strip()))
                elif not in_opts:
                    stem_lines.append(ln)
            if len(opts) >= 2 and gt_letter in letters:
                best_idx = letters.index(gt_letter)
                state = "\n".join(stem_lines).strip()
                if "User:" in state:
                    state = state.split("User:", 1)[-1].strip()
                return state, "mcqa", "Select the best hypothesis.", opts, best_idx, None

    # Schema 4: `nvidia/Nemotron-Pretraining-Specialized-v1.1` (`Nemotron-Pretraining-Multiple-Choice`)
    if isinstance(record.get("text"), str) and "\nAnswer:" in record["text"]:
        lines = record["text"].strip().splitlines()
        stem_lines = []
        letters = []
        opts = []
        ans_letter = ""
        rationale_lines: list[str] = []
        mode = "stem"
        for ln in lines:
            s_ln = ln.strip()
            if mode in ("stem", "opts"):
                m_ans = _MCQA_ANSWER_LINE_RE.match(s_ln)
                if m_ans:
                    ans_letter = m_ans.group(1).upper()
                    mode = "rationale"
                    continue
                m_opt = _MCQA_OPTION_LINE_RE.match(s_ln)
                if m_opt:
                    mode = "opts"
                    letters.append(m_opt.group(1).upper())
                    opts.append(sanitize_trace_markers(m_opt.group(2).strip()))
                    continue
                if mode == "stem":
                    stem_lines.append(ln)
                elif opts:
                    opts[-1] = f"{opts[-1]} {sanitize_trace_markers(s_ln)}".strip()
            else:
                rationale_lines.append(ln)
        if len(opts) >= 2 and ans_letter in letters:
            best_idx = letters.index(ans_letter)
            state = "\n".join(stem_lines).strip()
            rationale = "\n".join(rationale_lines).strip() or None
            if state:
                return state, "mcqa", "Determine the correct multiple-choice option.", opts, best_idx, rationale

    # Schema 1: `jev/typed-decisions` / `tasksource`
    state = str(record.get("state") or record.get("inputs") or record.get("instruction") or "").strip()
    kind = str(record.get("kind") or "choice").strip()
    question = str(record.get("question") or "Select the best hypothesis.").strip()
    options = record.get("options") or record.get("choices")
    target = record.get("target") or record.get("label")

    if kind == "noul" and (not isinstance(options, list) or len(options) == 0):
        options = ["The answer is no.", "The answer is yes."]
        if isinstance(target, list) and len(target) == 1:
            p = float(target[0])
            target = [1.0 - p, p]

    if not state or not isinstance(options, list) or len(options) < 2:
        return None

    best_idx = -1
    if isinstance(target, list) and len(target) == len(options) and target:
        best_idx = int(np.argmax(np.asarray(target, dtype=np.float64)))
    elif isinstance(target, int) and 0 <= target < len(options):
        best_idx = target
    if best_idx < 0:
        return None

    clean_opts = [sanitize_trace_markers(str(opt).strip()) for opt in options]
    if any(not opt for opt in clean_opts):
        return None
    return state, kind, question, clean_opts, best_idx, None


def extract_typed_decision_uta(
    record: Any,
    *,
    max_seq_len: int = 2048,
    require_rationale: bool = False,
) -> tuple[str, str, str] | None:
    """Convert a typed decision or Nemotron MCQA record into a `#U$#/#Q$#/#H$# -> #T$# -> #A$#` episode."""
    parsed = parse_mcqa_decision_fields(record)
    if parsed is None:
        return None
    state, kind, question, clean_opts, best_idx, rationale = parsed
    if require_rationale and not (rationale and rationale.strip()):
        return None
    winning_opt = clean_opts[best_idx]
    opt_lines = "\n".join(f"#H$# [{i}] {opt}" for i, opt in enumerate(clean_opts))
    prompt = (
        f"{sanitize_trace_markers(state)}\n"
        f"#Q$# [{kind}] {sanitize_trace_markers(question)}\n"
        f"{opt_lines}"
    )
    thought = (
        sanitize_trace_markers(rationale)
        if rationale and rationale.strip()
        else f"{sanitize_trace_markers(question)} -> [{best_idx}] {winning_opt}"
    )
    answer = f"[{best_idx}] {winning_opt}"
    return _compact_uta_triplet(
        prompt,
        thought,
        answer,
        max_total_chars=max(160, max_seq_len - 148),
        max_total_bytes=max(180, max_seq_len - 88),
    )



def extract_pretrain_text(
    record: Any,
    *,
    sieve_config: ByteSieveConfig = ByteSieveConfig(),
) -> tuple[str | None, bool]:
    """Extract and Stage-0 filter an expository or source-code pretraining document (`(text, rejected_stage0)`)."""
    if not isinstance(record, dict):
        return None, True

    is_code_record = bool(
        isinstance(record.get("code"), str)
        or ("repo_name" in record and "path" in record and ("code" in record or "content" in record))
    )
    if is_code_record:
        norm_rec = normalize_github_code_record(record)
        code_path = str(norm_rec.get("path") or "").strip()
        if code_path and _is_noisy_code_path(code_path):
            return None, True
        raw_code = str(norm_rec.get("code") or "")
        if not raw_code.strip():
            return None, True
        decision = evaluate_byte_sieve(raw_code, sieve_config)
        if not decision.keep or decision.sanitized_text is None:
            return None, True
        clean_code = decision.sanitized_text.replace("\r\n", "\n").replace("\r", "\n").strip()
        repo_name = str(norm_rec.get("repo_name") or "").strip()
        lang = str(norm_rec.get("language") or "").strip()
        if repo_name and code_path:
            header = f"# Repo: {repo_name} | Path: {code_path}"
            if lang and lang != "Unknown":
                header += f" | Language: {lang}"
            return f"{header}\n{clean_code}", False
        return clean_code, False

    raw = (
        record.get("text")
        or record.get("Text")
        or record.get("report_text")
        or record.get("content")
    )
    if (
        (not isinstance(raw, str) or not raw.strip())
        and isinstance(record.get("message_1"), str)
        and isinstance(record.get("message_2"), str)
    ):
        topic = str(record.get("topic;") or record.get("topic") or "").strip()
        sub_topic = str(record.get("sub_topic") or "").strip()
        header = f"Topic: {topic} — {sub_topic}\n\n" if topic and sub_topic else ""
        raw = f"{header}Problem:\n{record['message_1'].strip()}\n\nDerivation:\n{record['message_2'].strip()}"
    if (not isinstance(raw, str) or not raw.strip()) and isinstance(record.get("tokens"), list) and record["tokens"]:
        raw = _decode_gpt2_tokens(record["tokens"])
    if not isinstance(raw, str) or not raw.strip():
        return None, True
    decision = evaluate_byte_sieve(raw, sieve_config)
    if not decision.keep or decision.sanitized_text is None:
        return None, True
    return normalize_prose_unwrapping(decision.sanitized_text), False


class BlockAlignedShardWriter:
    """Streams encoded episodes into fixed-size `.bin` shards + atomic `.receipt.json` receipts.

    Guarantees for `MCL4-COE`:
    1. In `uta_trace` and `typed_decision` tracks, every `#U$# -> #T$# -> #A$#` episode is
       packed into `seq_len` blocks without **ever** severing an episode across a block
       boundary or shard boundary. Multiple short episodes can pack into the same `seq_len`
       block; any remaining tail bytes in a block are padded with `pad_id = 256` before the
       next block begins.
    2. Each shard (`shard_XXXXX.bin`) has an exact multiple of `seq_len` tokens (plus 1
       trailing `pad_id` token so `TokenBlockDataset` has `num_blocks = (token_count - 1) // seq_len`
       complete causal blocks).
    3. Every flushed shard writes `shard_XXXXX.receipt.json` and atomically updates the master
       `index_receipt.json`.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        track: str = "uta_trace",
        split: str = "train",
        seq_len: int = 2048,
        max_blocks_per_shard: int = 32_768,
        vocab_size: int = 260,
    ) -> None:
        if seq_len < 16:
            raise ValueError("seq_len must be >= 16")
        if max_blocks_per_shard < 1:
            raise ValueError("max_blocks_per_shard must be >= 1")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.track = track
        self.split = split
        self.seq_len = seq_len
        self.max_blocks_per_shard = max_blocks_per_shard
        self.vocab_size = vocab_size
        self.tokenizer = get_tokenizer(vocab_size)
        self.shard_dtype = "uint32" if vocab_size > 32768 else "uint16"
        self._np_dtype = "<u4" if vocab_size > 32768 else "<u2"

        self.index_path = self.output_dir / "index_receipt.json"
        if self.index_path.exists():
            self.index_receipt = IndexReceipt.from_file(self.index_path)
        else:
            self.index_receipt = IndexReceipt(
                dtype=self.shard_dtype,
                seq_len=seq_len,
                max_tokens_per_shard=max_blocks_per_shard * seq_len,
            )
        self.index_receipt.totals["pad_id"] = int(self.tokenizer.pad_id)
        self.index_receipt.totals["vocab_size"] = int(self.vocab_size)

        self._current_blocks: list[np.ndarray] = []
        self._current_block_buf: list[int] = []
        self._current_episodes = 0
        self._current_unmasked_targets = 0
        self._current_provenance: list[ShardProvenance] = []

    def _count_unmasked_targets(self, block_tokens: np.ndarray) -> int:
        # In a block of length seq_len, causal targets come from positions 1..seq_len-1
        # plus the boundary token. Any token with prompt_mask_bit == 0 and != pad_id is supervised.
        nxt = block_tokens[1:]
        pad_id = int(getattr(self.tokenizer, "pad_id", ByteTokenizer.pad_id))
        mask_bit = int(getattr(self.tokenizer, "prompt_mask_bit", ByteTokenizer.prompt_mask_bit))
        unmasked = (nxt != pad_id) & ((nxt & mask_bit) == 0)
        return int(np.count_nonzero(unmasked))

    def _flush_block_buffer(self) -> None:
        if not self._current_block_buf:
            return
        pad_needed = self.seq_len - len(self._current_block_buf)
        if pad_needed > 0:
            self._current_block_buf.extend([self.tokenizer.pad_id] * pad_needed)
        arr = np.asarray(self._current_block_buf, dtype=self._np_dtype)
        self._current_unmasked_targets += self._count_unmasked_targets(arr)
        self._current_blocks.append(arr)
        self._current_block_buf = []
        if len(self._current_blocks) >= self.max_blocks_per_shard:
            self.flush_shard()

    def add_uta_episode(self, prompt: str, thought: str, answer: str) -> bool:
        """Pack a `#U$# -> #T$# -> #A$#` episode without severing across `seq_len` blocks."""
        tokens = self.tokenizer.encode_masked_turn(prompt, thought, answer, seq_len=None)
        if len(tokens) > self.seq_len:
            # Never sever a reasoning trace mid-thought across blocks
            return False
        if len(self._current_block_buf) + len(tokens) > self.seq_len:
            self._flush_block_buffer()
        self._current_block_buf.extend(tokens)
        self._current_episodes += 1
        if len(self._current_block_buf) == self.seq_len:
            self._flush_block_buffer()
        return True

    def add_pretrain_document(self, text: str) -> int:
        """Pack an expository pretraining document (`<bos> + UTF-8 + <eos>`) into blocks."""
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        if tokens and getattr(self.tokenizer, "is_hf_donor", False):
            tokens[0] = int(tokens[0]) | int(self.tokenizer.prompt_mask_bit)
        blocks_added = 0
        offset = 0
        n = len(tokens)
        while offset < n:
            space = self.seq_len - len(self._current_block_buf)
            take = min(space, n - offset)
            self._current_block_buf.extend(tokens[offset : offset + take])
            offset += take
            if len(self._current_block_buf) == self.seq_len:
                self._flush_block_buffer()
                blocks_added += 1
        self._current_episodes += 1
        return blocks_added

    def record_provenance(
        self,
        corpus: str,
        source_file: str,
        row_start: int,
        row_end: int,
        episodes_accepted: int,
        episodes_rejected_stage0: int = 0,
    ) -> None:
        self._current_provenance.append(
            ShardProvenance(
                corpus=corpus,
                source_file=source_file,
                row_start=row_start,
                row_end=row_end,
                episodes_accepted=episodes_accepted,
                episodes_rejected_stage0=episodes_rejected_stage0,
            )
        )
        self.index_receipt.totals["rejected_stage0"] = (
            self.index_receipt.totals.get("rejected_stage0", 0) + episodes_rejected_stage0
        )

    def flush_shard(self) -> ShardReceipt | None:
        """Flush buffered blocks to `shard_XXXXX.bin` + `shard_XXXXX.receipt.json` and update `index_receipt.json`."""
        if self._current_block_buf:
            self._flush_block_buffer()
        if not self._current_blocks:
            return None

        if self.index_path.exists():
            try:
                latest_idx = IndexReceipt.from_file(self.index_path)
                self.index_receipt.shards = list(latest_idx.shards)
                merged_totals = dict(latest_idx.totals)
                merged_totals["rejected_stage0"] = max(
                    merged_totals.get("rejected_stage0", 0),
                    self.index_receipt.totals.get("rejected_stage0", 0),
                )
                self.index_receipt.totals = merged_totals
            except Exception:
                pass

        shard_idx = sum(1 for s in self.index_receipt.shards if s.get("split") == self.split)
        shard_id = f"{self.track}_{self.split}_shard_{shard_idx:05d}"
        bin_name = f"{shard_id}.bin"
        receipt_name = f"{shard_id}.receipt.json"
        bin_path = self.output_dir / bin_name
        receipt_path = self.output_dir / receipt_name

        # Append 1 trailing pad_id token so TokenBlockDataset sees `num_blocks` full causal windows
        trailing = np.asarray([self.tokenizer.pad_id], dtype=self._np_dtype)
        flat = np.concatenate([*self._current_blocks, trailing], axis=0)
        payload_bytes = flat.tobytes()
        digest = hashlib.sha256(payload_bytes).hexdigest()

        tmp_bin = bin_path.with_suffix(".bin.tmp")
        with tmp_bin.open("wb") as handle:
            handle.write(payload_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_bin, bin_path)

        num_blocks = len(self._current_blocks)
        num_tokens = int(flat.size)
        receipt = ShardReceipt(
            shard_id=shard_id,
            bin_file=bin_name,
            receipt_file=receipt_name,
            track=self.track,
            split=self.split,
            dtype=self.shard_dtype,
            byte_order="little",
            seq_len=self.seq_len,
            num_tokens=num_tokens,
            num_blocks=num_blocks,
            num_episodes=self._current_episodes,
            unmasked_target_tokens=self._current_unmasked_targets,
            num_bytes=len(payload_bytes),
            sha256=digest,
            created_at_unix=time.time(),
            provenance=tuple(self._current_provenance),
        )
        _atomic_write_json(receipt_path, receipt.to_dict())

        self.index_receipt.shards.append(receipt.to_dict())
        totals = self.index_receipt.totals
        totals["pad_id"] = int(self.tokenizer.pad_id)
        totals["vocab_size"] = int(self.vocab_size)
        totals["num_shards"] = len(self.index_receipt.shards)
        totals["num_tokens"] = totals.get("num_tokens", 0) + num_tokens
        totals["num_blocks"] = totals.get("num_blocks", 0) + num_blocks
        totals["num_episodes"] = totals.get("num_episodes", 0) + self._current_episodes
        totals["unmasked_target_tokens"] = (
            totals.get("unmasked_target_tokens", 0) + self._current_unmasked_targets
        )
        totals[f"{self.split}_blocks"] = totals.get(f"{self.split}_blocks", 0) + num_blocks
        totals[f"{self.split}_tokens"] = totals.get(f"{self.split}_tokens", 0) + num_tokens
        _atomic_write_json(self.index_path, self.index_receipt.to_dict())

        self._current_blocks.clear()
        self._current_episodes = 0
        self._current_unmasked_targets = 0
        self._current_provenance.clear()
        return receipt


class ShardedTokenBlockDataset(Dataset):
    """Multi-shard zero-copy `uint16`/`uint32` dataset driven by `index_receipt.json`.

    Drop-in replacement for `TokenBlockDataset` that maps global block indices
    across arbitrarily many `.bin` shards in $O(\\log N_{\\text{shards}})$ time using
    cumulative block offsets from `index_receipt.json`. Works seamlessly with
    `EpochBlockSampler` and `HostBatchPrefetcher`.
    """

    def __init__(
        self,
        index_receipt_path: str | Path,
        *,
        track: str | None = None,
        split: str | None = "train",
    ) -> None:
        self.index_path = Path(index_receipt_path).expanduser().resolve(strict=True)
        self.base_dir = self.index_path.parent
        self.receipt = IndexReceipt.from_file(self.index_path)
        self.seq_len = self.receipt.seq_len

        selected = []
        for s in self.receipt.shards:
            if track is not None and s.get("track") != track:
                continue
            if split is not None and s.get("split") != split:
                continue
            if int(s.get("num_blocks", 0)) > 0:
                selected.append(s)
        if not selected:
            raise ValueError(
                f"No shards in {self.index_path} matched track={track!r}, split={split!r}"
            )

        self.shards = selected
        self._shard_paths = [self.base_dir / s["bin_file"] for s in selected]
        block_counts = [int(s["num_blocks"]) for s in selected]
        self._cum_blocks = np.cumsum(block_counts, dtype=np.int64)
        self.total_blocks = int(self._cum_blocks[-1])
        self._datasets: list[Dataset | None] = [None] * len(selected)

    def __len__(self) -> int:
        return self.total_blocks

    def _get_shard_dataset(self, shard_idx: int):
        from .data import TokenBlockDataset

        ds = self._datasets[shard_idx]
        if ds is None:
            shard_path = self._shard_paths[shard_idx]
            if not shard_path.is_file():
                # If an early shard was evicted after consumption, fall back to the nearest surviving shard on disk
                fallback_idx = next(
                    (i for i in range(shard_idx + 1, len(self._shard_paths)) if self._shard_paths[i].is_file()),
                    next(
                        (i for i in range(len(self._shard_paths) - 1, -1, -1) if self._shard_paths[i].is_file()),
                        None,
                    ),
                )
                if fallback_idx is None:
                    raise FileNotFoundError(f"All shard .bin files for {self.index_path} have been evicted")
                return self._get_shard_dataset(fallback_idx)
            shard_dtype = str(self.shards[shard_idx].get("dtype", self.receipt.dtype))
            ds = TokenBlockDataset(
                shard_path,
                seq_len=self.seq_len,
                dtype=shard_dtype,
            )
            self._datasets[shard_idx] = ds
        return ds

    def evict_consumed_shards(
        self,
        consumed_block_cursor: int,
        *,
        delete_from_disk: bool = False,
    ) -> list[str]:
        """Release memory (and optionally unlink `.bin` files) for all shards with `cum_blocks <= consumed_block_cursor`.

        Always retains at least the final shard on disk so the track always has a valid backing `.bin` file.
        """
        cursor = int(consumed_block_cursor)
        if cursor <= 0 or len(self.shards) <= 1:
            return []
        evicted_files: list[str] = []
        receipt_dirty = False
        for s_idx in range(len(self.shards) - 1):
            if int(self._cum_blocks[s_idx]) <= cursor:
                ds = self._datasets[s_idx]
                if ds is not None:
                    setattr(ds, "_tokens", None)
                    self._datasets[s_idx] = None
                bin_name = str(self.shards[s_idx].get("bin_file", f"shard_{s_idx:05d}.bin"))
                shard_path = self._shard_paths[s_idx]
                if delete_from_disk:
                    if shard_path.is_file():
                        try:
                            shard_path.unlink(missing_ok=True)
                            evicted_files.append(bin_name)
                        except OSError:
                            pass
                    if not self.shards[s_idx].get("evicted"):
                        self.shards[s_idx]["evicted"] = True
                        receipt_dirty = True
                else:
                    if not self.shards[s_idx].get("evicted_mem"):
                        self.shards[s_idx]["evicted_mem"] = True
                        evicted_files.append(bin_name)
        if receipt_dirty:
            try:
                _atomic_write_json(self.index_path, self.receipt.to_dict())
            except OSError:
                pass
        return evicted_files

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        idx = operator.index(index)
        if idx < 0:
            idx += self.total_blocks
        if idx < 0 or idx >= self.total_blocks:
            raise IndexError(index)
        return {k: v[0] for k, v in self.batch([idx]).items()}

    def batch(self, indices: Iterable[int], *, pin_memory: bool = False) -> dict[str, torch.Tensor]:
        idx_arr = np.asarray(indices, dtype=np.int64)
        if idx_arr.ndim != 1 or idx_arr.size == 0:
            raise ValueError("indices must be a non-empty 1D sequence")
        if np.any(idx_arr < 0) or np.any(idx_arr >= self.total_blocks):
            raise IndexError("Global block index out of range")

        shard_ids = np.searchsorted(self._cum_blocks, idx_arr, side="right")
        prev_cum = np.where(shard_ids > 0, self._cum_blocks[shard_ids - 1], 0)
        local_indices = idx_arr - prev_cum

        bsz = idx_arr.size
        inputs = torch.empty((bsz, self.seq_len), dtype=torch.long)
        labels = torch.empty((bsz, self.seq_len), dtype=torch.long)

        for s_id in np.unique(shard_ids):
            mask = np.nonzero(shard_ids == s_id)[0]
            ds = self._get_shard_dataset(int(s_id))
            safe_local = np.mod(local_indices[mask], max(1, len(ds)))
            sub_batch = ds.batch(safe_local, pin_memory=False)
            pos = torch.from_numpy(mask).long()
            inputs[pos] = sub_batch["input_ids"]
            labels[pos] = sub_batch["labels"]

        result = {"input_ids": inputs, "labels": labels}
        if pin_memory and torch.cuda.is_available():
            result = {k: v.pin_memory() for k, v in result.items()}
        return result

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_datasets"] = [None] * len(self.shards)
        return state


def build_corpus_shards(
    corpora_root: str | Path,
    output_dir: str | Path,
    *,
    tracks: Iterable[str] = ("pretrain_lm", "uta_trace", "typed_decision"),
    corpus_filter: Iterable[str] | None = None,
    seq_len: int = 2048,
    max_blocks_per_shard: int = 16_384,
    max_files_per_corpus: int | None = None,
    max_records_per_file: int | None = None,
    sieve_config: ByteSieveConfig = ByteSieveConfig(),
    vocab_size: int = 260,
) -> dict[str, IndexReceipt]:
    """Stream corpora from `corpora_root` into per-track `.bin` shards and `index_receipt.json` files."""
    specs = discover_corpora(corpora_root)
    allowed_tracks = set(tracks)
    name_filter = {c.lower().replace("\\", "/") for c in corpus_filter} if corpus_filter else None

    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Save the discovery inventory receipt first so the pipeline always has a full map of sources
    inventory_payload = {
        "corpora_root": str(Path(corpora_root).resolve()),
        "excluded_corpora": sorted(EXCLUDED_CORPORA),
        "discovered_corpora": [asdict(s) for s in specs],
    }
    _atomic_write_json(out_root / "corpus_inventory_receipt.json", inventory_payload)

    writers: dict[str, BlockAlignedShardWriter] = {}
    for spec in specs:
        if spec.track not in allowed_tracks:
            continue
        if name_filter is not None and not any(
            f in spec.name.lower() for f in name_filter
        ):
            continue

        track_dir = out_root / spec.track
        writer = writers.get(spec.track)
        if writer is None:
            writer = BlockAlignedShardWriter(
                track_dir,
                track=spec.track,
                split="train",
                seq_len=seq_len,
                max_blocks_per_shard=max_blocks_per_shard,
                vocab_size=vocab_size,
            )
            writers[spec.track] = writer

        completed_set = set(writer.index_receipt.completed_source_files)
        files_to_process = list(spec.files)
        if max_files_per_corpus is not None:
            files_to_process = files_to_process[:max_files_per_corpus]

        for fpath in files_to_process:
            if fpath in completed_set:
                continue
            accepted = 0
            rejected_s0 = 0
            first_row = 0
            last_row = 0
            for row_idx, rec in iter_source_records(fpath):
                if first_row == 0:
                    first_row = row_idx
                last_row = row_idx

                if spec.track == "pretrain_lm":
                    text, rej = extract_pretrain_text(rec, sieve_config=sieve_config)
                    if rej or text is None:
                        rejected_s0 += 1
                    else:
                        writer.add_pretrain_document(text)
                        accepted += 1
                elif spec.track == "uta_trace":
                    episodes = extract_uta_episodes(rec)
                    if not episodes:
                        rejected_s0 += 1
                    else:
                        for u, t, a in episodes:
                            if writer.add_uta_episode(u, t, a):
                                accepted += 1
                            else:
                                rejected_s0 += 1
                elif spec.track == "typed_decision":
                    ep = extract_typed_decision_uta(rec)
                    if ep is None:
                        rejected_s0 += 1
                    else:
                        if writer.add_uta_episode(*ep):
                            accepted += 1
                        else:
                            rejected_s0 += 1

                if max_records_per_file is not None and row_idx >= max_records_per_file:
                    break

            if last_row > 0:
                writer.record_provenance(
                    corpus=spec.name,
                    source_file=fpath,
                    row_start=first_row,
                    row_end=last_row,
                    episodes_accepted=accepted,
                    episodes_rejected_stage0=rejected_s0,
                )
            if max_records_per_file is None:
                writer.index_receipt.completed_source_files.append(fpath)

    receipts: dict[str, IndexReceipt] = {}
    for track_name, writer in writers.items():
        writer.flush_shard()
        receipts[track_name] = writer.index_receipt
    return receipts


class MultiTrackMixtureDataset(Dataset):
    """Unified multi-track `uint16` dataset combining multiple `index_receipt.json` tracks.

    Maps a single global index space `[0, total_blocks)` across `K` tracks
    (e.g., `pretrain_lm`, `uta_trace`, `typed_decision`) and returns
    `{"input_ids": [B, L], "labels": [B, L], "track_ids": [B]}` so the training
    loop can use a single `HostBatchPrefetcher` while still auditing per-track
    losses and unmasked target token rates.
    """

    def __init__(
        self,
        track_receipts: dict[str, str | Path],
        *,
        split: str | None = "train",
    ) -> None:
        if not track_receipts:
            raise ValueError("track_receipts must be a non-empty mapping of track_name -> index_receipt.json")

        self.track_names: tuple[str, ...] = tuple(track_receipts.keys())
        self.track_datasets: dict[str, ShardedTokenBlockDataset] = {}
        seq_lens: set[int] = set()
        track_counts: list[int] = []

        for name, path in track_receipts.items():
            ds = ShardedTokenBlockDataset(path, track=name, split=split)
            self.track_datasets[name] = ds
            seq_lens.add(ds.seq_len)
            track_counts.append(len(ds))

        if len(seq_lens) != 1:
            raise ValueError(f"All tracks in MultiTrackMixtureDataset must share seq_len, got {seq_lens}")

        self.seq_len: int = next(iter(seq_lens))
        self.track_block_counts: tuple[int, ...] = tuple(track_counts)
        self._cum_blocks = np.cumsum(track_counts, dtype=np.int64)
        self.total_blocks = int(self._cum_blocks[-1])

    def track_offset(self, track_name: str) -> int:
        idx = self.track_names.index(track_name)
        return 0 if idx == 0 else int(self._cum_blocks[idx - 1])

    def __len__(self) -> int:
        return self.total_blocks

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        idx = operator.index(index)
        if idx < 0:
            idx += self.total_blocks
        if idx < 0 or idx >= self.total_blocks:
            raise IndexError(index)
        return {k: v[0] for k, v in self.batch([idx]).items()}

    def batch(self, indices: Iterable[int], *, pin_memory: bool = False) -> dict[str, torch.Tensor]:
        idx_arr = np.asarray(indices, dtype=np.int64)
        if idx_arr.ndim != 1 or idx_arr.size == 0:
            raise ValueError("indices must be a non-empty 1D sequence")
        if np.any(idx_arr < 0) or np.any(idx_arr >= self.total_blocks):
            raise IndexError("Global mixture block index out of range")

        track_ids = np.searchsorted(self._cum_blocks, idx_arr, side="right")
        prev_cum = np.where(track_ids > 0, self._cum_blocks[track_ids - 1], 0)
        local_indices = idx_arr - prev_cum

        bsz = idx_arr.size
        inputs = torch.empty((bsz, self.seq_len), dtype=torch.long)
        labels = torch.empty((bsz, self.seq_len), dtype=torch.long)

        for t_id in np.unique(track_ids):
            mask = np.nonzero(track_ids == t_id)[0]
            t_name = self.track_names[int(t_id)]
            ds = self.track_datasets[t_name]
            sub = ds.batch(local_indices[mask], pin_memory=False)
            pos = torch.from_numpy(mask).long()
            inputs[pos] = sub["input_ids"]
            labels[pos] = sub["labels"]

        result = {
            "input_ids": inputs,
            "labels": labels,
            "track_ids": torch.from_numpy(track_ids.astype(np.int64, copy=False)),
        }
        if pin_memory and torch.cuda.is_available():
            result = {k: v.pin_memory() for k, v in result.items()}
        return result


@dataclass(frozen=True)
class CurriculumMixtureSchedule:
    """Declarative mixture schedule across tracks (`start_weights -> end_weights` or multi-stage `stage_weights`)."""

    start_weights: dict[str, float]
    end_weights: dict[str, float] | None = None
    mode: str = "cosine"  # "constant", "linear", "cosine", or "mitotic_staged"
    stage_weights: tuple[dict[str, float], ...] | None = None

    def __post_init__(self) -> None:
        if not self.start_weights or any(v < 0.0 for v in self.start_weights.values()):
            raise ValueError("start_weights must be non-empty and non-negative")
        if sum(self.start_weights.values()) <= 0.0:
            raise ValueError("start_weights must have positive sum")
        if self.end_weights is not None:
            if set(self.end_weights.keys()) != set(self.start_weights.keys()):
                raise ValueError("end_weights must have the exact same track keys as start_weights")
            if any(v < 0.0 for v in self.end_weights.values()) or sum(self.end_weights.values()) <= 0.0:
                raise ValueError("end_weights must be non-negative with positive sum")
        if self.stage_weights is not None:
            clean_stages = tuple(dict(s) for s in self.stage_weights if s)
            object.__setattr__(self, "stage_weights", clean_stages)
            for st in clean_stages:
                if set(st.keys()) != set(self.start_weights.keys()):
                    raise ValueError("Each entry in stage_weights must match start_weights keys")
                if any(v < 0.0 for v in st.values()) or sum(st.values()) <= 0.0:
                    raise ValueError("Each entry in stage_weights must be non-negative with positive sum")
        if self.mode not in ("constant", "linear", "cosine", "mitotic_staged"):
            raise ValueError("mode must be 'constant', 'linear', 'cosine', or 'mitotic_staged'")

    @staticmethod
    def _normalize_map(w: dict[str, float]) -> dict[str, float]:
        tot = sum(w.values())
        return {k: v / tot for k, v in w.items()}

    def weights_at(
        self,
        progress: float = 0.0,
        *,
        stage_idx: int | None = None,
        stage_progress: float | None = None,
    ) -> dict[str, float]:
        """Return normalized track weights summing to `1.0` at training progress $u \\in [0, 1]$ or mitotic stage."""
        import math

        u = max(0.0, min(1.0, float(progress)))
        s_norm = self._normalize_map(self.start_weights)

        if self.stage_weights and len(self.stage_weights) >= 2:
            n_stages = len(self.stage_weights)
            if stage_idx is not None:
                cur_i = max(0, min(n_stages - 1, int(stage_idx)))
                next_i = max(0, min(n_stages - 1, cur_i + 1))
                local_u = (
                    max(0.0, min(1.0, float(stage_progress)))
                    if stage_progress is not None
                    else u
                )
            else:
                scaled = u * (n_stages - 1)
                cur_i = min(n_stages - 1, int(math.floor(scaled)))
                next_i = min(n_stages - 1, cur_i + 1)
                local_u = scaled - cur_i
            w0 = self._normalize_map(self.stage_weights[cur_i])
            w1 = self._normalize_map(self.stage_weights[next_i])
            alpha = 0.5 * (1.0 - math.cos(math.pi * local_u))
            blended = {k: (1.0 - alpha) * w0[k] + alpha * w1[k] for k in w0}
            return self._normalize_map(blended)

        if self.end_weights is None or self.mode == "constant":
            return s_norm

        e_norm = self._normalize_map(self.end_weights)
        if self.mode == "mitotic_staged" and stage_idx is not None:
            local_u = (
                max(0.0, min(1.0, float(stage_progress)))
                if stage_progress is not None
                else u
            )
            eff_u = min(1.0, 0.35 * int(stage_idx) + 0.35 * local_u)
            alpha = 0.5 * (1.0 - math.cos(math.pi * eff_u))
        else:
            alpha = u if self.mode == "linear" else 0.5 * (1.0 - math.cos(math.pi * u))
        blended = {k: (1.0 - alpha) * s_norm[k] + alpha * e_norm[k] for k in s_norm}
        return self._normalize_map(blended)


class CurriculumMixtureSampler:
    """Deterministic, resumable multi-track block sampler using Hamilton's Largest Remainder quotas.

    Each track maintains its own independent `EpochBlockSampler` so every block
    inside a track is visited without replacement before that track advances to
    its next epoch, even when tracks have wildly different sizes (e.g., 3,162
    `pretrain_lm` blocks vs. 504 `uta_trace` blocks vs. 46 `typed_decision` blocks).
    """

    def __init__(
        self,
        dataset: MultiTrackMixtureDataset,
        schedule: CurriculumMixtureSchedule,
        *,
        seed: int = 42,
    ) -> None:
        from .data import EpochBlockSampler

        if set(schedule.start_weights.keys()) != set(dataset.track_names):
            raise ValueError(
                f"Schedule tracks {set(schedule.start_weights.keys())} do not match "
                f"dataset tracks {set(dataset.track_names)}"
            )
        self.dataset = dataset
        self.schedule = schedule
        self.seed = seed
        self.step_count = 0
        self.track_samplers: dict[str, EpochBlockSampler] = {
            name: EpochBlockSampler(
                len(dataset.track_datasets[name]),
                seed=seed + idx * 1009,
                shard_boundaries=(
                    dataset.track_datasets[name]._cum_blocks.tolist()
                    if len(dataset.track_datasets[name]) > 32768
                    else None
                ),
            )
            for idx, name in enumerate(dataset.track_names)
        }

    def evict_consumed_shards(
        self,
        *,
        delete_from_disk: bool = False,
    ) -> dict[str, list[str]]:
        """Evict all fully consumed shards across tracks based on each track sampler's cursor."""
        cleared: dict[str, list[str]] = {}
        for name, sampler in self.track_samplers.items():
            ds = self.dataset.track_datasets[name]
            eff_cursor = sampler.num_blocks if sampler.epoch >= 1 else sampler.cursor
            evicted = ds.evict_consumed_shards(
                eff_cursor,
                delete_from_disk=delete_from_disk,
            )
            if evicted:
                cleared[name] = evicted
        return cleared

    def allocate_quotas(
        self,
        batch_size: int,
        progress: float = 0.0,
        *,
        stage_idx: int | None = None,
        stage_progress: float | None = None,
    ) -> dict[str, int]:
        """Compute exact integer block counts per track summing to `batch_size` (Hamilton's method)."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        weights = self.schedule.weights_at(
            progress, stage_idx=stage_idx, stage_progress=stage_progress
        )
        exact = {name: weights[name] * batch_size for name in self.dataset.track_names}
        floors = {name: int(exact[name]) for name in self.dataset.track_names}
        remainder = batch_size - sum(floors.values())
        if remainder > 0:
            ranked = sorted(
                self.dataset.track_names,
                key=lambda k: (exact[k] - floors[k], weights[k]),
                reverse=True,
            )
            for name in ranked[:remainder]:
                floors[name] += 1
        return floors

    def take(
        self,
        batch_size: int,
        *,
        progress: float = 0.0,
        stage_idx: int | None = None,
        stage_progress: float | None = None,
    ) -> torch.Tensor:
        """Draw `batch_size` global block indices across tracks according to `schedule.weights_at(...)`."""
        quotas = self.allocate_quotas(
            batch_size,
            progress=progress,
            stage_idx=stage_idx,
            stage_progress=stage_progress,
        )
        global_chunks: list[torch.Tensor] = []
        for name in self.dataset.track_names:
            q = quotas[name]
            if q <= 0:
                continue
            sampler = self.track_samplers[name]
            offset = self.dataset.track_offset(name)
            remaining = q
            while remaining > 0:
                drawn = sampler.take(remaining)
                global_chunks.append(drawn + offset)
                remaining -= int(drawn.numel())
        self.step_count += 1
        return torch.cat(global_chunks, dim=0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "step_count": self.step_count,
            "tracks": {
                name: {"epoch": s.epoch, "cursor": s.cursor}
                for name, s in self.track_samplers.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_count = int(state.get("step_count", 0))
        track_states = state.get("tracks", {})
        for name, s in self.track_samplers.items():
            if name in track_states:
                s.epoch = int(track_states[name]["epoch"])
                raw_cursor = int(track_states[name]["cursor"])
                if s.num_blocks > 0 and raw_cursor >= s.num_blocks:
                    s.epoch += raw_cursor // s.num_blocks
                    raw_cursor = raw_cursor % s.num_blocks
                s.cursor = max(0, raw_cursor)
                s._order = None
                if s.cursor > 0:
                    _ = s.peek(1)


class TypedDecisionShardStreamer:
    """Streams structured `DecisionExample` batches from `tasksource-jev-typed-decisions`.

    Automatically caches parsed `DecisionExample` records to `cache_path` on first load
    so training runs stream pre-sanitized decision batches from RAM in `<0.1 ms` without
    re-reading raw Parquet files during training steps.
    """

    def __init__(
        self,
        corpora_root: str | Path = r"X:\corpora",
        *,
        max_seq_len: int = 512,
        max_options: int = 8,
        cache_path: str | Path | None = None,
        max_cached_examples: int = 512,
        vocab_size: int = 260,
    ) -> None:
        self.corpora_root = Path(corpora_root).expanduser().resolve()
        self.max_seq_len = max_seq_len
        self.max_options = max_options
        self.max_cached_examples = max_cached_examples
        self.vocab_size = vocab_size
        self.tokenizer = get_tokenizer(vocab_size)

        ts_dir = self.corpora_root / "text" / "tasksource-jev-typed-decisions"
        self.files = (
            sorted(p for p in ts_dir.rglob("*.parquet") if _is_valid_data_file(p))
            if ts_dir.is_dir()
            else []
        )
        self.cache_path = Path(cache_path).expanduser().resolve() if cache_path else None
        self._cached_examples: list[Any] | None = None
        self._cursor: int = 0

    def _scan_raw_examples(self):
        from .data import DecisionExample

        for fpath in self.files:
            for row_idx, rec in iter_source_records(fpath):
                if not isinstance(rec, dict):
                    continue
                state = str(rec.get("state") or "").strip()
                kind = str(rec.get("kind") or "choice").strip()
                question = str(rec.get("question") or "").strip()
                options = rec.get("options")
                target = rec.get("target")
                if not state or not question or not isinstance(target, list) or not target:
                    continue

                if kind == "noul" and (not isinstance(options, list) or len(options) == 0):
                    options = ["The answer is no.", "The answer is yes."]
                    if len(target) == 1:
                        p = float(target[0])
                        target = [1.0 - p, p]
                    values: tuple[Any, ...] = (False, True)
                elif kind == "choice":
                    if not isinstance(options, list) or not (2 <= len(options) <= self.max_options):
                        continue
                    if len(target) != len(options):
                        continue
                    values = tuple(f"opt_{i}" for i in range(len(options)))
                else:
                    continue

                clean_opts = tuple(sanitize_trace_markers(str(o).strip()) for o in options)
                if any(not o for o in clean_opts):
                    continue

                context = (
                    f"#U$# {sanitize_trace_markers(state)}\n"
                    f"#Q$# [{kind}] {sanitize_trace_markers(question)}"
                )
                if len(context.encode("utf-8")) + 2 > self.max_seq_len:
                    continue
                if any(len(o.encode("utf-8")) + 2 > self.max_seq_len for o in clean_opts):
                    continue

                best_idx = int(np.argmax(np.asarray(target, dtype=np.float64)))
                qid = str(rec.get("id") or f"{fpath.stem}:{row_idx}")
                yield DecisionExample(
                    question_id=qid,
                    kind=kind,
                    context=context,
                    descriptions=clean_opts,
                    values=values,
                    target=best_idx,
                )

    def _get_cached_pool(self) -> list[Any]:
        if self._cached_examples is not None:
            return self._cached_examples
        if self.cache_path is not None and self.cache_path.is_file():
            try:
                loaded = torch.load(self.cache_path, map_location="cpu", weights_only=False)
                if isinstance(loaded, list) and len(loaded) > 0:
                    filtered = [
                        ex
                        for ex in loaded
                        if len(ex.descriptions) <= self.max_options
                        and len(ex.context.encode("utf-8")) + 2 <= self.max_seq_len
                        and all(len(d.encode("utf-8")) + 2 <= self.max_seq_len for d in ex.descriptions)
                    ]
                    if filtered:
                        self._cached_examples = filtered
                        return filtered
            except Exception:
                pass
        pool = []
        if self.files:
            for ex in self._scan_raw_examples():
                pool.append(ex)
                if len(pool) >= self.max_cached_examples:
                    break
        if not pool:
            pool = _build_portable_decision_examples(max_seq_len=self.max_seq_len)
        self._cached_examples = pool
        if self.cache_path is not None and pool:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_path.with_suffix(".pt.tmp")
                torch.save(pool, tmp)
                os.replace(tmp, self.cache_path)
            except Exception:
                pass
        return pool

    def iter_decision_examples(self):
        pool = self._get_cached_pool()
        for ex in pool:
            yield ex

    def sample_batch(
        self,
        batch_size: int = 8,
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        from .data import collate_decisions

        pool = self._get_cached_pool()
        if not pool:
            raise RuntimeError("No valid DecisionExamples found in TypedDecisionShardStreamer")
        n = len(pool)
        examples = [pool[(self._cursor + i) % n] for i in range(batch_size)]
        self._cursor = (self._cursor + batch_size) % n
        return collate_decisions(examples, self.max_seq_len, device=device, tokenizer=self.tokenizer)


def _build_portable_decision_examples(max_seq_len: int = 512) -> list[Any]:
    """Build portable `DecisionExample` records when `X:\\corpora` is not mounted (Laptop / Colab)."""
    from .data import DecisionExample, parse_request

    examples: list[DecisionExample] = []
    repo_root = Path(__file__).resolve().parents[2]
    sibling_mcattn = repo_root.parent / "mcattn" / "examples" / "association_train.jsonl"
    if sibling_mcattn.is_file():
        for line in sibling_mcattn.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                try:
                    examples.extend(parse_request(json.loads(line), require_targets=True))
                except Exception:
                    continue

    animals = ("rabbit", "otter", "falcon", "badger")
    colors = ("crimson", "cobalt", "emerald", "amber")
    tools = ("compass", "lantern", "chronometer", "telescope")
    for a_i, animal in enumerate(animals):
        for c_i, color in enumerate(colors):
            t_i = (a_i + c_i) % len(tools)
            other_a = animals[(a_i + 1) % len(animals)]
            other_c = colors[(c_i + 1) % len(colors)]
            other_t = tools[(t_i + 1) % len(tools)]
            state = (
                f"The {color} {animal} calibrated the {tools[t_i]} before dawn, "
                f"while the {other_c} {other_a} inspected the {other_t}."
            )
            opts = [
                f"#T$# Matches the {color} {animal} in the first clause.\n#A$# {tools[t_i]}",
                f"#T$# Belongs to the {other_c} {other_a}, not the {animal}.\n#A$# {other_t}",
                f"#T$# Unmentioned instrument.\n#A$# barometer",
            ]
            examples.append(
                DecisionExample(
                    question_id=f"portable_{a_i}_{c_i}",
                    kind="choice",
                    context=f"#U$# {state}\n#Q$# [choice] Which instrument did the {color} {animal} calibrate?",
                    descriptions=tuple(opts),
                    values=(tools[t_i], other_t, "barometer"),
                    target=0,
                )
            )
    return [
        ex
        for ex in examples
        if len(ex.context.encode("utf-8")) + 2 <= max_seq_len
        and all(len(d.encode("utf-8")) + 2 <= max_seq_len for d in ex.descriptions)
    ]


def _track_shards_complete(index_file: Path) -> bool:
    if not index_file.is_file():
        return False
    try:
        receipt = IndexReceipt.from_file(index_file)
        if not receipt.shards:
            return False
        base_dir = index_file.parent
        surviving = 0
        for s in receipt.shards:
            if (base_dir / s["bin_file"]).is_file():
                surviving += 1
            elif not s.get("evicted"):
                return False
        return surviving > 0
    except Exception:
        return False


def ensure_sharded_tracks_ready(
    shards_root: str | Path,
    *,
    corpora_root: str | Path | None = None,
    seq_len: int = 512,
    vocab_size: int = 260,
) -> dict[str, Path]:
    """Ensure `pretrain_lm`, `uta_trace`, and `typed_decision` have valid `.bin` shards + `index_receipt.json`."""
    root = Path(shards_root).expanduser().resolve()
    track_names = ("pretrain_lm", "uta_trace", "typed_decision")
    existing: dict[str, Path] = {}
    for t_name in track_names:
        idx = root / t_name / "index_receipt.json"
        if _track_shards_complete(idx):
            existing[t_name] = idx

    if len(existing) == len(track_names):
        return existing

    # If corpora_root is mounted and contains real corpora, build from corpora_root
    if corpora_root is not None:
        c_path = Path(corpora_root).expanduser()
        if c_path.is_dir() and discover_corpora(c_path):
            build_corpus_shards(
                c_path,
                root,
                tracks=track_names,
                seq_len=seq_len,
                max_blocks_per_shard=64,
                max_files_per_corpus=1,
                max_records_per_file=200,
                vocab_size=vocab_size,
            )
            for t_name in track_names:
                idx = root / t_name / "index_receipt.json"
                if _track_shards_complete(idx):
                    existing[t_name] = idx
            if len(existing) == len(track_names):
                return existing

    # Portable bootstrap for Laptop / Google Colab when X:\corpora is not mounted
    repo_root = Path(__file__).resolve().parents[2]
    if root == (repo_root / "data_shards").resolve():
        target_root = (root / "_portable_morph16k") if vocab_size > 1024 else (root / "_portable")
    else:
        target_root = root
    dec_examples = _build_portable_decision_examples(max_seq_len=max(256, seq_len))
    main_dec_cache = root / "typed_decision" / "decision_examples_cache.pt"
    if not main_dec_cache.is_file():
        main_dec_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dec_examples, main_dec_cache)

    for t_name in track_names:
        t_dir = target_root / t_name
        idx = t_dir / "index_receipt.json"
        if _track_shards_complete(idx):
            existing[t_name] = idx
            continue
        if t_dir.is_dir():
            for old_f in t_dir.glob("*"):
                if old_f.is_file() and old_f.name != "decision_examples_cache.pt":
                    old_f.unlink(missing_ok=True)

        writer = BlockAlignedShardWriter(
            t_dir,
            track=t_name,
            split="train",
            seq_len=seq_len,
            max_blocks_per_shard=16,
            vocab_size=vocab_size,
        )
        if t_name == "pretrain_lm":
            docs: list[str] = []
            for md_path in sorted((repo_root / "docs").glob("*.md")):
                try:
                    docs.append(md_path.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    continue
            for py_path in sorted((repo_root / "src" / "mcl4_coe").glob("*.py")):
                try:
                    docs.append(py_path.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    continue
            if not docs:
                docs = [
                    (
                        "In signal processing and machine learning, a matched filter correlates "
                        "a reference template against an observed state to maximize signal-to-noise ratio. "
                    )
                    * 12
                ]
            accepted = 0
            for d_text in docs[:4]:
                writer.add_pretrain_document(d_text[: max(512, seq_len * 4)])
                accepted += 1
            writer.record_provenance(
                corpus="portable_repo_corpus",
                source_file="docs+src",
                row_start=1,
                row_end=accepted,
                episodes_accepted=accepted,
                episodes_rejected_stage0=0,
            )
        elif t_name == "uta_trace":
            accepted = 0
            if seq_len >= 256:
                try:
                    from .context_memory import synthesize_repository_uta_traces

                    repo_eps = synthesize_repository_uta_traces(
                        Path(__file__).resolve().parents[2],
                        max_episodes=64,
                    )
                    for u_r, t_r, a_r in repo_eps:
                        if writer.add_uta_episode(u_r, t_r, a_r):
                            accepted += 1
                except Exception:
                    pass
            for i in range(1, 96):
                a = 3 + (i * 7) % 23
                b = 2 + (i * 11) % 17
                c = 1 + (i * 5) % 9
                val = a * b + c
                if seq_len < 384:
                    u_str = f"solve({a},{b},{c})"
                    t_str = f"{a}*{b}+{c}={val}"
                    a_str = f"{val}"
                else:
                    u_str = f"Implement `solve(x, y, z) = x * y + z` and evaluate `solve({a}, {b}, {c})`."
                    t_str = (
                        f"[plan: abduction -> deduction -> decision -> evaluation] "
                        f"First compute product {a} * {b} = {a * b}. Next add offset {c} to obtain {val}."
                    )
                    a_str = f"def solve(x, y, z):\n    return x * y + z\n# solve({a}, {b}, {c}) == {val}"
                if not writer.add_uta_episode(u_str, t_str, a_str):
                    if writer.add_uta_episode(f"solve({a},{b},{c})", f"{a}*{b}+{c}={val}", f"{val}"):
                        accepted += 1
                else:
                    accepted += 1
            writer.record_provenance(
                corpus="portable_uta_traces",
                source_file="synthetic_verified_uta",
                row_start=1,
                row_end=accepted,
                episodes_accepted=accepted,
                episodes_rejected_stage0=0,
            )
        else:
            dec_cache = t_dir / "decision_examples_cache.pt"
            if not dec_cache.is_file():
                torch.save(dec_examples, dec_cache)
            accepted = 0
            for _ in range(4):
                for idx_ex, ex in enumerate(dec_examples):
                    u_str = ex.context.replace("#U$#", "").strip()
                    gold_desc = ex.descriptions[ex.target]
                    t_clean, a_clean = _split_thought_and_answer(gold_desc)
                    if not writer.add_uta_episode(u_str, t_clean, a_clean):
                        writer.add_uta_episode(f"Q{idx_ex}", "Compare options.", f"[{ex.target}]")
                    accepted += 1
            writer.record_provenance(
                corpus="portable_typed_decisions",
                source_file="mcattn+compositional",
                row_start=1,
                row_end=accepted,
                episodes_accepted=accepted,
                episodes_rejected_stage0=0,
            )
        writer.flush_shard()
        existing[t_name] = idx

    return existing


def _is_clean_uta_episode(
    u_clean: str,
    t_clean: str,
    a_clean: str,
    *,
    track_name: str = "uta_trace",
    seq_len: int = 1024,
) -> bool:
    """Return True if `(u_clean, t_clean, a_clean)` is a genuine, non-stub reasoning trace."""
    if not u_clean or not t_clean or not a_clean:
        return False
    if track_name != "uta_trace" or seq_len < 256:
        return True
    if "Score Candidate " in u_clean or "Reply only with a JSON array" in u_clean:
        return False
    if t_clean.startswith("Evaluate step-by-step:") or (
        t_clean.startswith("[") and len(t_clean) <= 6
    ):
        return False
    if (
        u_clean.startswith("{'")
        or u_clean.startswith('{"')
        or a_clean.startswith("{'")
        or a_clean.startswith('{"')
    ):
        return False
    if len(t_clean) < 12 or len(a_clean) < 2:
        return False
    if len(u_clean) >= 32 and u_clean[:60] == a_clean[:60]:
        return False
    return True


def transcode_byte_shards_to_morphcode(
    src_shards_root: str | Path,
    dst_shards_root: str | Path,
    *,
    seq_len: int | None = None,
    max_shards_per_track: int | None = None,
) -> dict[str, IndexReceipt]:
    """Losslessly transcode Stage-0 curated `ByteTokenizer` (`vocab_size=260`) shards into `MorphCode-16K` (`vocab_size=16,384`) shards.

    Reconstructs every `<bos> ... <eos>` pretrain document and every `0x8000`-masked `#U$# -> #T$# -> #A$#`
    episode from the existing `.bin` shards (`train` and `val` splits), deduplicates episodes across
    legacy `local_*` and standard shards, filters out synthetic rubric/dict stubs in `uta_trace`, and
    packs them into `MorphCode-16K` block-aligned shards without requiring `X:\\corpora` to be mounted.
    """
    import shutil

    src_root = Path(src_shards_root).expanduser().resolve(strict=True)
    dst_root = Path(dst_shards_root).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    byte_tok = ByteTokenizer()

    inv_file = src_root / "corpus_inventory_receipt.json"
    if inv_file.is_file():
        shutil.copy2(inv_file, dst_root / "corpus_inventory_receipt.json")

    receipts: dict[str, IndexReceipt] = {}
    for track_name in ("pretrain_lm", "uta_trace", "typed_decision"):
        src_track_dir = src_root / track_name
        src_idx_path = src_track_dir / "index_receipt.json"
        if not src_idx_path.is_file():
            continue

        src_receipt = IndexReceipt.from_file(src_idx_path)
        target_seq_len = seq_len or src_receipt.seq_len
        dst_track_dir = dst_root / track_name
        dst_track_dir.mkdir(parents=True, exist_ok=True)

        # Clean any stale .bin / .receipt.json / index_receipt.json in destination track dir before transcoding
        for old_file in dst_track_dir.glob("*"):
            if old_file.is_file() and not old_file.name.endswith(".pt"):
                old_file.unlink(missing_ok=True)

        # Copy decision example caches (`decision_examples_cache.pt`, `decision_examples_val_cache.pt`)
        for pt_file in src_track_dir.glob("*.pt"):
            shutil.copy2(pt_file, dst_track_dir / pt_file.name)

        # Group shards by split ("train", "val"), preferring canonical non-`local_` shards first
        splits: dict[str, list[dict[str, Any]]] = {}
        ordered_shards = sorted(
            src_receipt.shards,
            key=lambda s: (1 if str(s.get("bin_file", "")).startswith("local_") else 0, str(s.get("bin_file", ""))),
        )
        for s in ordered_shards:
            bin_p = src_track_dir / s["bin_file"]
            if bin_p.is_file():
                sp = str(s.get("split", "train"))
                splits.setdefault(sp, []).append(s)

        for sp, shard_list in splits.items():
            if max_shards_per_track is not None and sp == "train":
                shard_list = shard_list[:max_shards_per_track]
            writer = BlockAlignedShardWriter(
                dst_track_dir,
                track=track_name,
                split=sp,
                seq_len=target_seq_len,
                max_blocks_per_shard=16_384,
                vocab_size=16_384,
            )
            seen_hashes: set[bytes] = set()
            for s_meta in shard_list:
                bin_p = src_track_dir / s_meta["bin_file"]
                raw_u16 = np.fromfile(bin_p, dtype="<u2")
                clean_ids = raw_u16 & 0x7FFF
                # Locate episode boundaries: bos_id (257) to eos_id (258)
                bos_positions = np.nonzero(clean_ids == byte_tok.bos_id)[0]
                eos_positions = np.nonzero(clean_ids == byte_tok.eos_id)[0]
                if bos_positions.size == 0 or eos_positions.size == 0:
                    continue

                eos_cursor = 0
                n_eos = eos_positions.size
                accepted = 0
                rejected = 0
                for b_pos in bos_positions:
                    while eos_cursor < n_eos and eos_positions[eos_cursor] <= b_pos:
                        eos_cursor += 1
                    if eos_cursor >= n_eos:
                        break
                    e_pos = eos_positions[eos_cursor]
                    span_u16 = raw_u16[b_pos + 1 : e_pos]
                    span_clean = span_u16 & 0x7FFF
                    valid_bytes = span_clean[span_clean < 256].astype(np.uint8).tobytes()
                    if not valid_bytes:
                        continue
                    digest = hashlib.blake2b(valid_bytes, digest_size=12).digest()
                    if digest in seen_hashes:
                        rejected += 1
                        continue
                    text = valid_bytes.decode("utf-8", errors="replace")
                    if track_name == "pretrain_lm":
                        seen_hashes.add(digest)
                        writer.add_pretrain_document(text)
                        accepted += 1
                    else:
                        # Reconstruct (prompt, thought, answer) from `#U$# ... \n#T$# ... \n#A$# ...`
                        if "#T$#" in text and "#A$#" in text:
                            u_part, rest = text.split("#T$#", 1)
                            t_part, a_part = rest.split("#A$#", 1)
                            u_clean = u_part.replace("#U$#", "", 1).strip()
                            t_clean = t_part.strip()
                            a_clean = a_part.strip()
                            if _is_clean_uta_episode(
                                u_clean,
                                t_clean,
                                a_clean,
                                track_name=track_name,
                                seq_len=target_seq_len,
                            ):
                                seen_hashes.add(digest)
                                if writer.add_uta_episode(u_clean, t_clean, a_clean):
                                    accepted += 1
                                else:
                                    rejected += 1
                            else:
                                rejected += 1

                writer.record_provenance(
                    corpus=f"transcoded_morph16k:{track_name}:{sp}",
                    source_file=s_meta["bin_file"],
                    row_start=1,
                    row_end=accepted + rejected,
                    episodes_accepted=accepted,
                    episodes_rejected_stage0=rejected,
                )
            writer.flush_shard()
            receipts[track_name] = writer.index_receipt

    return receipts


def sanitize_morphcode_shards(
    shards_root: str | Path,
    *,
    tracks: Iterable[str] = ("pretrain_lm", "uta_trace", "typed_decision"),
    vocab_size: int = 16_384,
) -> dict[str, dict[str, int]]:
    """Deduplicate episodes and strip corrupted rubric/dict stubs in-place across `shards_root`."""
    import shutil

    root = Path(shards_root).expanduser().resolve(strict=True)
    tok = get_tokenizer(vocab_size)
    summary: dict[str, dict[str, int]] = {}

    for track_name in tracks:
        track_dir = root / track_name
        idx_path = track_dir / "index_receipt.json"
        if not idx_path.is_file():
            continue
        src_receipt = IndexReceipt.from_file(idx_path)
        target_seq_len = src_receipt.seq_len
        tmp_dir = root / f".tmp_sanitize_{track_name}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for pt_file in track_dir.glob("*.pt"):
            shutil.copy2(pt_file, tmp_dir / pt_file.name)

        splits: dict[str, list[dict[str, Any]]] = {}
        ordered_shards = sorted(
            src_receipt.shards,
            key=lambda s: (1 if str(s.get("bin_file", "")).startswith("local_") else 0, str(s.get("bin_file", ""))),
        )
        for s in ordered_shards:
            bin_p = track_dir / s["bin_file"]
            if bin_p.is_file():
                sp = str(s.get("split", "train"))
                splits.setdefault(sp, []).append(s)

        kept_total = 0
        dropped_total = 0
        for sp, shard_list in splits.items():
            writer = BlockAlignedShardWriter(
                tmp_dir,
                track=track_name,
                split=sp,
                seq_len=target_seq_len,
                max_blocks_per_shard=16_384,
                vocab_size=vocab_size,
            )
            seen_hashes: set[bytes] = set()
            for s_meta in shard_list:
                bin_p = track_dir / s_meta["bin_file"]
                raw_u16 = np.fromfile(bin_p, dtype="<u2")
                clean_ids = raw_u16 & 0x7FFF
                bos_positions = np.nonzero(clean_ids == tok.bos_id)[0]
                eos_positions = np.nonzero(clean_ids == tok.eos_id)[0]
                if bos_positions.size == 0 or eos_positions.size == 0:
                    continue

                eos_cursor = 0
                n_eos = eos_positions.size
                accepted = 0
                rejected = 0
                for b_pos in bos_positions:
                    while eos_cursor < n_eos and eos_positions[eos_cursor] <= b_pos:
                        eos_cursor += 1
                    if eos_cursor >= n_eos:
                        break
                    e_pos = eos_positions[eos_cursor]
                    span_clean = clean_ids[b_pos + 1 : e_pos]
                    if span_clean.size == 0:
                        continue
                    text = tok.decode(span_clean.tolist())
                    if not text.strip():
                        continue
                    digest = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=12).digest()
                    if digest in seen_hashes:
                        rejected += 1
                        dropped_total += 1
                        continue
                    if track_name == "pretrain_lm":
                        seen_hashes.add(digest)
                        writer.add_pretrain_document(text)
                        accepted += 1
                        kept_total += 1
                    else:
                        if "#T$#" in text and "#A$#" in text:
                            u_part, rest = text.split("#T$#", 1)
                            t_part, a_part = rest.split("#A$#", 1)
                            u_clean = u_part.replace("#U$#", "", 1).strip()
                            t_clean = t_part.strip()
                            a_clean = a_part.strip()
                            if _is_clean_uta_episode(
                                u_clean,
                                t_clean,
                                a_clean,
                                track_name=track_name,
                                seq_len=target_seq_len,
                            ):
                                seen_hashes.add(digest)
                                if writer.add_uta_episode(u_clean, t_clean, a_clean):
                                    accepted += 1
                                    kept_total += 1
                                else:
                                    rejected += 1
                                    dropped_total += 1
                            else:
                                rejected += 1
                                dropped_total += 1
                        else:
                            rejected += 1
                            dropped_total += 1

                writer.record_provenance(
                    corpus=f"sanitized_morph16k:{track_name}:{sp}",
                    source_file=s_meta["bin_file"],
                    row_start=1,
                    row_end=accepted + rejected,
                    episodes_accepted=accepted,
                    episodes_rejected_stage0=rejected,
                )
            writer.flush_shard()

        shutil.rmtree(track_dir)
        tmp_dir.rename(track_dir)
        new_idx = IndexReceipt.from_file(track_dir / "index_receipt.json")
        summary[track_name] = {
            "kept_episodes": kept_total,
            "dropped_episodes": dropped_total,
            "num_blocks": int(new_idx.totals.get("num_blocks", 0)),
            "train_blocks": int(new_idx.totals.get("train_blocks", 0)),
            "val_blocks": int(new_idx.totals.get("val_blocks", 0)),
        }

    return summary







