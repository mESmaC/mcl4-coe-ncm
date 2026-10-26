"""Block-aligned byte tokenization, 0x8000 prompt masking, and typed decision collation.

Unifies:
- `mini/src/mini_moe/data.py` (`ByteTokenizer`, `TokenBlockDataset` with `0x8000`
  high-bit prompt masking, `EpochBlockSampler`, `HostBatchPrefetcher`), and
- `mcattn/src/mcattn/decision.py` (`DecisionExample`, `parse_request`, `collate_decisions`,
  `format_answers`).
"""

from __future__ import annotations

import json
import math
import operator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


TRACE_MARKERS: dict[str, str] = {
    "user": "#U$#",
    "question": "#Q$#",
    "schedule": "#S$#",
    "hypothesis": "#H$#",
    "thought": "#T$#",
    "answer": "#A$#",
}


class ByteTokenizer:
    """UTF-8 byte tokenizer with special control IDs (`pad=256, bos=257, eos=258, unk=259`)."""

    pad_id: int = 256
    bos_id: int = 257
    eos_id: int = 258
    unk_id: int = 259
    vocab_size: int = 260
    prompt_mask_bit: int = 0x8000

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.bos_id)
        tokens.extend(text.encode("utf-8"))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def encode_masked_turn(
        self,
        prompt: str,
        thought: str,
        answer: str,
        *,
        seq_len: int | None = None,
    ) -> list[int]:
        """Encode a `#U$# -> #T$# -> #A$#` micro-trace with `0x8000` high-bit masking on `#U$#`.

        Why 4-byte symmetric markers (`#U$#`, `#Q$#`, `#S$#`, `#H$#`, `#T$#`, `#A$#`):
        - Zero collision with programming-language identifiers, shell/Perl/PHP `$VAR` syntax,
          or regex end-of-line `T$` anchors.
        - Exact 4-byte footprint (`[35, X, 36, 35]`) fits 100% inside a single `MCL4`
          mandatory local window (`markov_local_tokens = 4`, `{q, q-1, q-2, q-3}`) at Layer 0.
        - Rule for `MCL4-COE`: 100% of reasoning traces must include a non-empty `#T$#`
          scratchpad before `#A$#` (no blind `#U$# -> #A$#` guessing).
        """
        if not prompt.strip():
            raise ValueError("prompt (#U$#) must be non-empty")
        if not thought.strip():
            raise ValueError("thought (#T$#) scratchpad is mandatory in MCL4-COE micro-traces")
        if not answer.strip():
            raise ValueError("answer (#A$#) must be non-empty")

        prefix_str = f"#U$# {prompt.strip()}\n"
        completion_str = f"#T$# {thought.strip()}\n#A$# {answer.strip()}"

        prefix_ids = [self.bos_id, *prefix_str.encode("utf-8")]
        masked_prefix = [tok | self.prompt_mask_bit for tok in prefix_ids]
        completion_ids = [*completion_str.encode("utf-8"), self.eos_id]
        full = masked_prefix + completion_ids

        if seq_len is not None:
            if len(full) > seq_len:
                raise ValueError(
                    f"Micro-trace length ({len(full)}) exceeds block seq_len={seq_len}; "
                    "never sever traces mid-thought across blocks"
                )
            full.extend([self.pad_id] * (seq_len - len(full)))
        return full

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        raw_bytes = bytearray()
        for tid in token_ids:
            clean = int(tid) & 0x7FFF
            if 0 <= clean < 256:
                raw_bytes.append(clean)
        return raw_bytes.decode("utf-8", errors="replace")

    def count_utf8_bytes(self, token_ids: torch.Tensor) -> int:
        valid_mask = (token_ids >= 0) & ((token_ids & 0x7FFF) < 256)
        return int(valid_mask.sum().item())


class DonorHFTokenizer:
    """HuggingFace BPE donor tokenizer wrapper (default: `ibm-granite/granite-3.1-1b-a400m-instruct`)."""

    _CACHE: dict[str, tuple[Any, np.ndarray]] = {}

    def __init__(
        self,
        model_id: str | None = None,
        *,
        vocab_size: int = 49155,
        pad_id: int | None = None,
        bos_id: int | None = None,
        eos_id: int | None = None,
        unk_id: int | None = None,
    ) -> None:
        if model_id is None:
            if int(vocab_size) == 128256:
                model_id = "unsloth/Llama-3.2-1B-Instruct"
            elif int(vocab_size) == 50304:
                model_id = "allenai/OLMoE-1B-7B-0125"
            else:
                model_id = "ibm-granite/granite-3.1-1b-a400m-instruct"
        self.model_id = model_id
        self.vocab_size = int(vocab_size)
        if int(vocab_size) == 128256:
            self.pad_id = int(128004 if pad_id is None else pad_id)
            self.bos_id = int(128000 if bos_id is None else bos_id)
            self.eos_id = int(128009 if eos_id is None else eos_id)
            self.unk_id = int(128004 if unk_id is None else unk_id)
        elif int(vocab_size) == 50304:
            self.pad_id = int(1 if pad_id is None else pad_id)
            self.bos_id = int(50279 if bos_id is None else bos_id)
            self.eos_id = int(50279 if eos_id is None else eos_id)
            self.unk_id = int(0 if unk_id is None else unk_id)
        else:
            self.pad_id = int(4 if pad_id is None else pad_id)
            self.bos_id = int(0 if bos_id is None else bos_id)
            self.eos_id = int(0 if eos_id is None else eos_id)
            self.unk_id = int(4 if unk_id is None else unk_id)
        self.is_hf_donor = True
        self.prompt_mask_bit = 0x80000000
        self.value_mask = 0x7FFFFFFF
        if model_id not in self._CACHE:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]

            hf_tok = AutoTokenizer.from_pretrained(model_id)
            decoded_pieces = hf_tok.batch_decode(
                [[i] for i in range(self.vocab_size)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            byte_lens = np.zeros(self.vocab_size, dtype=np.int32)
            special_ids = set(getattr(hf_tok, "all_special_ids", []) or [])
            special_ids.update({self.pad_id, self.bos_id, self.eos_id, self.unk_id})
            for idx, piece in enumerate(decoded_pieces):
                if idx in special_ids:
                    byte_lens[idx] = 0
                else:
                    byte_lens[idx] = max(1, len(piece.encode("utf-8")))
            self._CACHE[model_id] = (hf_tok, byte_lens)
        self.hf_tok, self._token_byte_lens = self._CACHE[model_id]

        def _first_id(s: str, default: int) -> int:
            ids = self.hf_tok.encode(s, add_special_tokens=False)
            return int(ids[0]) if ids else default

        self.period_id = _first_id(".", 15 if self.vocab_size == 50304 else 32)
        self.colon_id = _first_id(":", 27 if self.vocab_size == 50304 else 44)
        self.hash_id = _first_id("#", 4 if self.vocab_size == 50304 else 21)
        self.lt_id = _first_id("<", 29 if self.vocab_size == 50304 else 46)
        self.newline_id = _first_id("\n", 187 if self.vocab_size == 50304 else 203)
        self._newline_ids = frozenset({self.newline_id})
        self._sentence_end_ids = frozenset(
            {
                self.period_id,
                _first_id("!", 2 if self.vocab_size == 50304 else 19),
                _first_id("?", 32 if self.vocab_size == 50304 else 49),
                self.newline_id,
            }
        )
        self.marker_char_ids = tuple(
            _first_id(c, d)
            for c, d in (("U", 54), ("T", 53), ("Q", 50), ("S", 52), ("H", 41))
        )
        self.a_char_id = _first_id("A", 34 if self.vocab_size == 50304 else 51)
        if self.vocab_size == 128256:
            self.invalid_logit_slice_low = (0, 0)
            self.invalid_logit_start_high = 128000
        elif self.vocab_size == 50304:
            self.invalid_logit_slice_low = (0, 2)
            self.invalid_logit_start_high = 50254
        else:
            self.invalid_logit_slice_low = (1, 19)
            self.invalid_logit_start_high = 49152

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.bos_id)
        tokens.extend(int(t) for t in self.hf_tok.encode(text, add_special_tokens=False))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def encode_masked_turn(
        self,
        prompt: str,
        thought: str,
        answer: str,
        *,
        seq_len: int | None = None,
    ) -> list[int]:
        if not prompt.strip():
            raise ValueError("prompt (#U$#) must be non-empty")
        if not thought.strip():
            raise ValueError("thought (#T$#) scratchpad is mandatory in MCL4-COE micro-traces")
        if not answer.strip():
            raise ValueError("answer (#A$#) must be non-empty")

        prefix_str = f"#U$# {prompt.strip()}\n"
        completion_str = f"#T$# {thought.strip()}\n#A$# {answer.strip()}"

        prefix_ids = [
            self.bos_id,
            *(int(t) for t in self.hf_tok.encode(prefix_str, add_special_tokens=False)),
        ]
        masked_prefix = [tok | self.prompt_mask_bit for tok in prefix_ids]
        completion_ids = [
            *(int(t) for t in self.hf_tok.encode(completion_str, add_special_tokens=False)),
            self.eos_id,
        ]
        full = masked_prefix + completion_ids

        if seq_len is not None:
            if len(full) > seq_len:
                raise ValueError(
                    f"Micro-trace length ({len(full)}) exceeds block seq_len={seq_len}; "
                    "never sever traces mid-thought across blocks"
                )
            full.extend([self.pad_id] * (seq_len - len(full)))
        return full

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        clean_ids: list[int] = []
        for tid in token_ids:
            if int(tid) < 0 and (int(tid) & self.prompt_mask_bit) == 0:
                continue
            clean = int(tid) & self.value_mask
            if clean in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if 0 <= clean < self.vocab_size:
                clean_ids.append(clean)
        if not clean_ids:
            return ""
        return str(
            self.hf_tok.decode(
                clean_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def count_utf8_bytes(self, token_ids: list[int] | torch.Tensor) -> int:
        arr = (
            token_ids.detach().cpu().numpy().reshape(-1)
            if isinstance(token_ids, torch.Tensor)
            else np.asarray(list(token_ids), dtype=np.int64).reshape(-1)
        )
        valid = arr >= 0
        if not np.any(valid):
            return 0
        clean = arr[valid] & self.value_mask
        in_range = (clean >= 0) & (clean < self.vocab_size)
        return int(self._token_byte_lens[clean[in_range]].sum())


_TOKENIZER_INSTANCES: dict[int, Any] = {}


def get_tokenizer(vocab_size: int = 260) -> Any:
    """Return `DonorHFTokenizer` when `vocab_size > 16384`, `MorphCodeTokenizer` when `vocab_size > 1024`, else `ByteTokenizer`."""
    key = int(vocab_size)
    cached = _TOKENIZER_INSTANCES.get(key)
    if cached is not None:
        return cached
    if key > 16384:
        tok: Any = DonorHFTokenizer(vocab_size=key)
    elif key > 1024:
        from .morph_tokenizer import MorphCodeTokenizer

        tok = MorphCodeTokenizer()
    else:
        tok = ByteTokenizer()
    _TOKENIZER_INSTANCES[key] = tok
    return tok


class TokenBlockDataset(Dataset):
    """Memory-mapped uint16/uint32 block dataset with high-bit prompt masking."""

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        *,
        include_partial: bool = False,
        dtype: str | None = None,
        pad_id: int | None = None,
    ) -> None:
        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        self.path = Path(path).expanduser().resolve(strict=True)
        self.seq_len = seq_len
        self.include_partial = include_partial

        resolved_dtype = dtype
        resolved_pad_id = pad_id
        if resolved_dtype is None or resolved_pad_id is None:
            receipt_candidate = self.path.with_suffix(".receipt.json")
            if receipt_candidate.is_file():
                try:
                    r_data = json.loads(receipt_candidate.read_text(encoding="utf-8"))
                    if resolved_dtype is None:
                        resolved_dtype = str(r_data.get("dtype", "uint16"))
                    if resolved_pad_id is None and r_data.get("pad_id") is not None:
                        resolved_pad_id = int(r_data["pad_id"])
                except Exception:
                    pass
            if resolved_dtype is None or resolved_pad_id is None:
                idx_candidate = self.path.parent / "index_receipt.json"
                if idx_candidate.is_file():
                    try:
                        i_data = json.loads(idx_candidate.read_text(encoding="utf-8"))
                        if resolved_dtype is None:
                            resolved_dtype = str(i_data.get("dtype", "uint16"))
                        if resolved_pad_id is None:
                            if i_data.get("pad_id") is not None:
                                resolved_pad_id = int(i_data["pad_id"])
                            elif isinstance(i_data.get("totals"), dict) and i_data["totals"].get("pad_id") is not None:
                                resolved_pad_id = int(i_data["totals"]["pad_id"])
                    except Exception:
                        pass

        if resolved_dtype in ("uint32", "<u4", "u4"):
            self._dtype_str = "<u4"
            self.prompt_mask_bit = 0x80000000
            self.value_mask = 0x7FFFFFFF
            self.pad_id = 4 if resolved_pad_id is None else int(resolved_pad_id)
        else:
            self._dtype_str = "<u2"
            self.prompt_mask_bit = 0x8000
            self.value_mask = 0x7FFF
            self.pad_id = ByteTokenizer.pad_id if resolved_pad_id is None else int(resolved_pad_id)

        itemsize = np.dtype(self._dtype_str).itemsize
        size = self.path.stat().st_size
        if size % itemsize:
            raise ValueError(f"Malformed {self._dtype_str} token file: {self.path}")
        self.token_count = size // itemsize
        minimum_tokens = 2 if include_partial else self.seq_len + 1
        if self.token_count < minimum_tokens:
            raise ValueError(
                f"Insufficient tokens in {self.path}: found {self.token_count}, need >= {minimum_tokens}"
            )
        self._tokens: np.memmap | None = None

    def __len__(self) -> int:
        targets = self.token_count - 1
        return (
            (targets + self.seq_len - 1) // self.seq_len
            if self.include_partial
            else targets // self.seq_len
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        batch = self.batch([index])
        return {name: tensor[0] for name, tensor in batch.items()}

    def batch(self, indices, *, pin_memory: bool = False) -> dict[str, torch.Tensor]:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not indices.size:
            raise ValueError("indices must be a nonempty 1D sequence")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Block index out of range")
        if self._tokens is None:
            self._tokens = np.memmap(self.path, mode="r", dtype=self._dtype_str)
        offsets = indices[:, None] * self.seq_len + np.arange(self.seq_len + 1)
        valid = offsets < self.token_count
        raw_block = np.asarray(
            self._tokens[np.minimum(offsets, self.token_count - 1)], dtype=np.int64
        )
        masked_targets = (raw_block[:, 1:] & self.prompt_mask_bit) != 0
        block = raw_block & self.value_mask
        inputs = np.array(block[:, :-1], copy=True)
        labels = np.array(block[:, 1:], copy=True)
        inputs[~valid[:, :-1]] = self.pad_id
        labels[
            (~valid[:, 1:])
            | masked_targets
            | (labels == self.pad_id)
            | (inputs == self.pad_id)
        ] = -100
        result = {"input_ids": torch.from_numpy(inputs), "labels": torch.from_numpy(labels)}
        return (
            {name: tensor.pin_memory() for name, tensor in result.items()} if pin_memory else result
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state


class EpochBlockSampler:
    """Deterministic full-epoch (or shard-windowed) block shuffle with resumable cursor."""

    def __init__(
        self,
        num_blocks: int,
        seed: int = 42,
        *,
        shard_boundaries: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self.shard_boundaries: tuple[int, ...] | None = (
            tuple(int(b) for b in shard_boundaries if int(b) > 0)
            if shard_boundaries
            else None
        )
        self._order: torch.Tensor | None = None

    def _build_order(self) -> torch.Tensor:
        if self.shard_boundaries and len(self.shard_boundaries) > 1:
            chunks: list[torch.Tensor] = []
            start_b = 0
            for s_idx, end_b in enumerate(self.shard_boundaries):
                clamped_end = min(self.num_blocks, int(end_b))
                span = clamped_end - start_b
                if span > 0:
                    gen = torch.Generator(device="cpu").manual_seed(
                        self.seed + self.epoch * 10007 + s_idx
                    )
                    chunks.append(start_b + torch.randperm(span, generator=gen))
                start_b = clamped_end
            if start_b < self.num_blocks:
                gen = torch.Generator(device="cpu").manual_seed(
                    self.seed + self.epoch * 10007 + len(self.shard_boundaries)
                )
                chunks.append(start_b + torch.randperm(self.num_blocks - start_b, generator=gen))
            return torch.cat(chunks, dim=0) if chunks else torch.arange(self.num_blocks)
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        return torch.randperm(self.num_blocks, generator=generator)

    def peek(self, count: int) -> torch.Tensor:
        if count < 1:
            raise ValueError("count must be positive")
        if self.cursor >= self.num_blocks:
            return torch.empty(0, dtype=torch.int64)
        if self._order is None:
            self._order = self._build_order()
        end = min(self.cursor + count, self.num_blocks)
        return self._order[self.cursor : end]

    def take(self, count: int) -> torch.Tensor:
        if count < 1:
            raise ValueError("count must be positive")
        if self.cursor >= self.num_blocks:
            self.epoch += 1
            self.cursor = 0
            self._order = None
        indices = self.peek(count)
        self.cursor += len(indices)
        return indices


class HostBatchPrefetcher:
    """Single-worker background host batch prefetcher with optional pinned memory."""

    def __init__(self, dataset: TokenBlockDataset, micro_batch_size: int, *, pin_memory: bool = True):
        if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size < 1:
            raise ValueError("micro_batch_size must be a positive integer")
        self.dataset = dataset
        self.micro_batch_size = micro_batch_size
        self.pin_memory = pin_memory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcl4-coe-prefetch")
        self._pending: Future | None = None
        self._indices: np.ndarray | None = None
        self._closed = False

    def _load(self, indices: np.ndarray) -> list[dict[str, torch.Tensor]]:
        return [
            self.dataset.batch(
                indices[start : start + self.micro_batch_size], pin_memory=self.pin_memory
            )
            for start in range(0, len(indices), self.micro_batch_size)
        ]

    def schedule(self, indices) -> None:
        if self._closed:
            raise RuntimeError("HostBatchPrefetcher is closed")
        requested = np.asarray(indices, dtype=np.int64).copy()
        if self._pending is not None and np.array_equal(self._indices, requested):
            return
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        if requested.size:
            self._indices = requested
            self._pending = self._executor.submit(self._load, requested)

    def get(self, indices) -> list[dict[str, torch.Tensor]]:
        self.schedule(indices)
        assert self._pending is not None
        try:
            return self._pending.result()
        finally:
            self._pending = None
            self._indices = None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True)
class DecisionExample:
    question_id: str
    kind: str
    context: str
    descriptions: tuple[str, ...]
    values: tuple
    target: int | None = None


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def parse_request(
    request: dict,
    *,
    require_targets: bool = False,
    compact_framing: bool = True,
) -> list[DecisionExample]:
    """Parse a typed decision request (`choice`, `noul`, `score`) into DecisionExamples."""
    if not isinstance(request, dict) or not {"state", "questions"} <= request.keys():
        raise ValueError("A request requires state and questions")
    if set(request) - {"state", "questions", "targets"}:
        raise ValueError("Unknown request fields")
    if not isinstance(request["state"], (str, dict, list)):
        raise ValueError("state must be text, a JSON object, or a JSON array")
    state_str = (
        request["state"] if isinstance(request["state"], str) else _json(request["state"])
    )
    questions, targets = request["questions"], request.get("targets", {})
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty object")
    if not isinstance(targets, dict) or set(targets) - set(questions):
        raise ValueError("targets must map known question IDs to answers")

    examples: list[DecisionExample] = []
    for question_id, question in questions.items():
        _text(question_id, "question ID")
        if not isinstance(question, dict):
            raise ValueError("Each question must be an object")
        kind = question.get("type")
        instructions = _text(question.get("instructions"), "instructions")
        common = {"type", "instructions"}
        if kind == "choice":
            if set(question) - common - {"options"}:
                raise ValueError("Unknown choice fields")
            options = question.get("options")
            if not isinstance(options, list) or len(options) < 2:
                raise ValueError("choice requires at least two options")
            if any(
                not isinstance(item, dict) or set(item) != {"id", "description"} for item in options
            ):
                raise ValueError("Each option requires exactly id and description")
            values = tuple(_text(item["id"], "option ID") for item in options)
            descriptions = tuple(_text(item["description"], "description") for item in options)
            if len(set(values)) != len(values):
                raise ValueError("Choice IDs must be unique")
        elif kind == "noul":
            if set(question) - common - {"false_description", "true_description"}:
                raise ValueError("Unknown noul fields")
            values = (False, True)
            descriptions = (
                _text(question.get("false_description", "The answer is no."), "false_description"),
                _text(question.get("true_description", "The answer is yes."), "true_description"),
            )
        elif kind == "score":
            if set(question) - common - {"levels"}:
                raise ValueError("Unknown score fields")
            levels = question.get("levels")
            if not isinstance(levels, list) or len(levels) < 2:
                raise ValueError("score requires at least two ordered levels")
            if any(
                not isinstance(item, dict) or set(item) != {"value", "description"}
                for item in levels
            ):
                raise ValueError("Each level requires exactly value and description")
            values = tuple(item["value"] for item in levels)
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in values
            ):
                raise ValueError("Score values must be finite numbers")
            if any(a >= b for a, b in zip(values, values[1:])):
                raise ValueError("Score levels must be strictly increasing")
            descriptions = tuple(_text(item["description"], "description") for item in levels)
        else:
            raise ValueError("Question type must be choice, noul, or score")

        if compact_framing:
            context = f"#U$# {state_str}\n#Q$# [{kind}] {instructions}"
        else:
            context = _json({"state": request["state"], "question": instructions, "type": kind})

        target = None
        if question_id in targets:
            value = targets[question_id]
            if kind == "noul" and not isinstance(value, bool):
                raise ValueError("noul targets must be booleans")
            if kind == "score" and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError("score targets must be declared numerical levels")
            if value not in values:
                raise ValueError(f"Target for {question_id} is not a declared answer")
            target = values.index(value)
        elif require_targets:
            raise ValueError(f"Missing target for {question_id}")
        examples.append(DecisionExample(question_id, kind, context, descriptions, values, target))
    return examples


def _clean_bpe_decision_context(ctx: str) -> str:
    text = ctx.strip()
    if text.startswith("#U$#"):
        text = text[4:].strip()
    if "\n#Q$#" in text:
        u_part, q_part = text.split("\n#Q$#", 1)
        q_part = q_part.strip()
        if q_part.startswith("[") and "]" in q_part:
            q_part = q_part.split("]", 1)[1].strip()
        u_part = u_part.strip()
        if not u_part or u_part == q_part or q_part.endswith(u_part) or u_part in q_part:
            out = q_part or u_part or ctx
        elif not q_part:
            out = u_part
        else:
            out = f"{u_part}\n{q_part}"
    else:
        out = text or ctx
    if out.endswith("?") and "Answer:" not in out and "#S$#" not in out:
        out = f"Question: {out}\nAnswer:" if not out.startswith("Question:") else f"{out}\nAnswer:"
    return out


def _clean_bpe_decision_option(desc: str) -> str:
    stripped = desc.strip()
    if not stripped or stripped.startswith("#"):
        return desc
    return " " + stripped


def collate_decisions(
    examples: list[DecisionExample],
    max_length: int,
    *,
    device: str | torch.device = "cpu",
    tokenizer: Any | None = None,
    vocab_size: int = 260,
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("At least one decision is required")
    if tokenizer is None:
        tokenizer = get_tokenizer(vocab_size)
    is_donor_bpe = int(getattr(tokenizer, "vocab_size", vocab_size)) > 16384
    if is_donor_bpe:
        bos_tok = int(getattr(tokenizer, "bos_id", 128000))
        contexts = []
        for item in examples:
            c_ids = tokenizer.encode(_clean_bpe_decision_context(item.context))
            if len(c_ids) > max_length:
                c_ids = [bos_tok] + c_ids[-(max_length - 1) :]
            contexts.append(c_ids)
        candidates = [
            [
                tokenizer.encode(_clean_bpe_decision_option(description))[:max_length]
                for description in item.descriptions
            ]
            for item in examples
        ]
    else:
        contexts = [tokenizer.encode(item.context) for item in examples]
        candidates = [
            [tokenizer.encode(description) for description in item.descriptions]
            for item in examples
        ]
    if any(len(row) > max_length for row in contexts) or any(
        len(row) > max_length for options in candidates for row in options
    ):
        raise ValueError(f"Decision context or candidate exceeds {max_length} byte tokens")

    batch, count = len(examples), max(map(len, candidates))
    context_ids = torch.full((batch, max(map(len, contexts))), tokenizer.pad_id, dtype=torch.long)
    option_ids = torch.full(
        (batch, count, max(len(row) for options in candidates for row in options)),
        tokenizer.pad_id,
        dtype=torch.long,
    )
    context_lengths = torch.tensor(list(map(len, contexts)), dtype=torch.long)
    option_lengths = torch.zeros(batch, count, dtype=torch.long)
    for index, (context, options) in enumerate(zip(contexts, candidates)):
        context_ids[index, : len(context)] = torch.tensor(context, dtype=torch.long)
        for option_index, tokens in enumerate(options):
            option_ids[index, option_index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            option_lengths[index, option_index] = len(tokens)
    batch_data = dict(
        context_ids=context_ids,
        context_lengths=context_lengths,
        option_ids=option_ids,
        option_lengths=option_lengths,
    )
    if all(item.target is not None for item in examples):
        batch_data["targets"] = torch.tensor([item.target for item in examples], dtype=torch.long)
    elif any(item.target is not None for item in examples):
        raise ValueError("A batch must have either all targets or none")
    return {key: value.to(device) for key, value in batch_data.items()}


def format_answers(
    examples: list[DecisionExample],
    probabilities: list[list[float]],
    similarities: list[list[float]],
) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    for example, probs, scores in zip(examples, probabilities, similarities):
        rows = [
            dict(value=value, probability=float(p), association_score=float(s))
            for value, p, s in zip(example.values, probs, scores)
        ]
        answer: dict[str, Any] = {
            "type": example.kind,
            "ranking": sorted(rows, key=lambda row: row["probability"], reverse=True),
        }
        if example.kind == "choice":
            answer["choice"] = answer["ranking"][0]["value"]
        elif example.kind == "noul":
            answer["noul"] = rows[1]["probability"]
        else:
            answer["score"] = sum(row["value"] * row["probability"] for row in rows)
        answers[example.question_id] = answer
    return answers
