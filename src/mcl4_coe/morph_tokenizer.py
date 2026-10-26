"""MorphCode-16K (`vocab_size = 16,384`): Hybrid Morpheme-Aware + Code/Math Byte-Fallback Tokenizer.

Designed specifically for `MCL4-COE` (`MarkovLocal4Router` + `PerLayerEmbedding` + `CognitiveOperatorEnsemble`):
1. **100% Lossless UTF-8 Byte Fallback (`0..255`)**:
   Every possible string round-trips identically (`decode(encode(text)) == text`) with zero `<unk>` drops.
2. **Exact Control ID & `uint16` `0x8000` Masking Compatibility**:
   - `pad_id = 256`, `bos_id = 257`, `eos_id = 258`, `unk_id = 259`
   - `vocab_size = 16,384` (`0..16,383`), fitting inside bits `0..13` of `uint16` so bit 15 (`0x8000 = 32,768`)
     continues to serve as the zero-overhead `#U$#` prompt mask (`tok | 0x8000`, `tok & 0x7FFF`).
3. **Atomic `MCL4-COE` Trace & Operator Markers (`260..275`)**:
   - `#U$#`, `#Q$#`, `#S$#`, `#H$#`, `#T$#`, `#A$#` and their space/newline-prefixed forms
     (`\n#T$# `, `\n#A$# `, `#U$# `, `#T$# `, `#A$# `) are single atomic token IDs.
4. **Single-Digit Arithmetic Invariance (`'0'..'9'`)**:
   - Digits `'0'..'9'` (`48..57`) are never merged into multi-digit tokens, preserving strict
     place-value alignment for arithmetic and symbolic math.
5. **Morpheme-Constrained Word Decomposition (`prefix + root_stem + suffix`)**:
   - Inflected and derived English/scientific words split along true morphological boundaries
     (e.g., `" acceleration" -> [" acceler", "ation"]`, `" recalculating" -> [" re", "calcul", "ating"]`,
     `" transported" -> [" transport", "ed"]`), so a whole derived word spans `1..3` tokens inside
     `PerLayerEmbedding`'s `kernel_size = 4` causal depthwise convolution and shares an exact
     root stem ID inside `AttentiveMatchedFilterPool`.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Iterable

import torch

from .data import TRACE_MARKERS

VOCAB_JSON_PATH = Path(__file__).with_name("morphcode_16k.json")
MORPHCODE_16K_VOCAB_SIZE = 16384

# Atomic trace marker tokens at fixed IDs 260..275
ATOMIC_TRACE_TOKENS: tuple[str, ...] = (
    "#U$#",
    "#Q$#",
    "#S$#",
    "#H$#",
    "#T$#",
    "#A$#",
    "\n#U$#",
    "\n#Q$#",
    "\n#S$#",
    "\n#H$#",
    "\n#T$#",
    "\n#A$#",
    "#U$# ",
    "#Q$# ",
    "#T$# ",
    "#A$# ",
)

# Productive English + Greco-Latin scientific prefixes (ordered longest-first for greedy stem check)
MORPHEME_PREFIXES: tuple[str, ...] = (
    "counter",
    "electro",
    "thermo",
    "pseudo",
    "hetero",
    "magnet",
    "neuro",
    "micro",
    "macro",
    "multi",
    "inter",
    "intra",
    "trans",
    "super",
    "under",
    "hyper",
    "infra",
    "ultra",
    "proto",
    "quasi",
    "retro",
    "cyber",
    "extra",
    "hydro",
    "ortho",
    "photo",
    "socio",
    "anti",
    "auto",
    "over",
    "post",
    "semi",
    "para",
    "peri",
    "poly",
    "mono",
    "meta",
    "mega",
    "mini",
    "nano",
    "omni",
    "tele",
    "fore",
    "homo",
    "non",
    "sub",
    "pre",
    "pro",
    "dis",
    "mis",
    "out",
    "uni",
    "tri",
    "bio",
    "geo",
    "eco",
    "iso",
    "syn",
    "sym",
    "mid",
    "un",
    "re",
    "de",
    "in",
    "im",
    "il",
    "ir",
    "en",
    "em",
    "co",
    "bi",
    "ex",
    "ad",
)

# Productive derivational & inflectional suffixes (ordered longest-first)
MORPHEME_SUFFIXES: tuple[str, ...] = (
    "ization",
    "isation",
    "ational",
    "fulness",
    "lessness",
    "ability",
    "ibility",
    "atively",
    "itively",
    "ionally",
    "ically",
    "ations",
    "itions",
    "aments",
    "ements",
    "anness",
    "enness",
    "iously",
    "eously",
    "ation",
    "ition",
    "ating",
    "iting",
    "izing",
    "ising",
    "ifying",
    "ified",
    "ifies",
    "ingly",
    "ments",
    "ances",
    "ences",
    "ities",
    "ously",
    "ively",
    "fully",
    "lessly",
    "wards",
    "ional",
    "logy",
    "graphy",
    "metry",
    "cracy",
    "lysis",
    "pathy",
    "scope",
    "meter",
    "tion",
    "sion",
    "ment",
    "ness",
    "ance",
    "ence",
    "ancy",
    "ency",
    "able",
    "ible",
    "ated",
    "ates",
    "ized",
    "izes",
    "ised",
    "ises",
    "ical",
    "ally",
    "ious",
    "eous",
    "less",
    "ship",
    "hood",
    "ward",
    "wise",
    "like",
    "fold",
    "most",
    "ings",
    "ists",
    "isms",
    "ives",
    "ables",
    "ibles",
    "ity",
    "ive",
    "ative",
    "itive",
    "ous",
    "ful",
    "ate",
    "ize",
    "ise",
    "ify",
    "ing",
    "est",
    "ism",
    "ist",
    "ery",
    "ary",
    "ory",
    "ure",
    "age",
    "ant",
    "ent",
    "ian",
    "ics",
    "ial",
    "ies",
    "ied",
    "ier",
    "ily",
    "ers",
    "ors",
    "ed",
    "er",
    "or",
    "en",
    "al",
    "ic",
    "ly",
    "es",
    "ty",
    "ry",
    "th",
    "ion",
    "'s",
    "s",
    "d",
    "y",
)

# Short closed-class / atomic root words that should NEVER be split by prefix/suffix heuristics
ATOMIC_SHORT_WORDS: frozenset[str] = frozenset(
    {
        "the", "and", "that", "with", "from", "into", "for", "are", "was", "were",
        "have", "has", "had", "will", "would", "could", "should", "can", "may", "might",
        "must", "shall", "not", "this", "these", "those", "which", "what", "where", "when",
        "why", "how", "who", "whom", "whose", "because", "while", "between", "through",
        "during", "before", "after", "above", "below", "under", "over", "again", "further",
        "then", "once", "here", "there", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "only", "own", "same", "than", "too", "very",
        "just", "now", "also", "well", "back", "even", "still", "way", "take", "make",
        "made", "does", "did", "done", "doing", "goes", "went", "gone", "going", "being",
        "been", "having", "water", "number", "order", "under", "over", "after", "other",
        "never", "ever", "every", "power", "layer", "inner", "outer", "upper", "lower",
        "first", "second", "third", "state", "states", "value", "values", "model", "models",
        "token", "tokens", "index", "matrix", "vector", "tensor", "input", "output",
        "error", "class", "while", "where", "return", "import", "assert", "except",
        "false", "true", "none", "self", "super", "async", "await", "yield", "raise",
        "break", "pass", "print", "range", "float", "tuple", "bytes", "train", "valid",
        "plant", "force", "space", "point", "group", "field", "level", "table", "graph",
        "chain", "block", "local", "total", "final", "score", "scale", "phase", "stage",
        "trace", "shard", "batch", "epoch", "query", "route", "router", "expert",
        "solve", "using", "xylem", "narrow", "upward", "against", "gravity", "capillary",
    }
)

# Regex to split around atomic MCL4-COE control markers first (preserving the space after `#X$#` for the following word!)
_TRACE_MARKER_SPLIT_RE = re.compile(
    r"(\n#U\$#|\n#Q\$#|\n#S\$#|\n#H\$#|\n#T\$#|\n#A\$#|#U\$#|#Q\$#|#S\$#|#H\$#|#T\$#|#A\$#)"
)

# Master pre-tokenization scanner within non-marker text:
# 1. Newline + indentation or multi-newline / multi-space runs
# 2. Code fence or multi-char string delimiters
# 3. LaTeX command (\alpha, \frac{, \boxed{, etc.)
# 4. Single digit (never merged with adjacent digits!)
# 5. Word with optional single leading space (including contractions like don't / it's)
# 6. Multi-char operators with optional single leading space (never eating trailing space!)
# 7. Punctuation / operator / symbol chunk or fallback char
_MASTER_SCAN_RE = re.compile(
    r"\n[ ]{1,24}|\n\t+|\n\n+|[ ]{2,24}"
    r"|```python\n|```python|```\n|```|\"\"\"|'''|\):\n"
    r"|\\[a-zA-Z]+\{?"
    r"|[0-9]"
    r"| ?[a-zA-Z]+(?:'[a-zA-Z]+)?"
    r"| ?(?:===|!==|==|!=|<=|>=|->|=>|\+=|-=|\*=|/=|//|\*\*|:=|::|\.\.\.|&&|\|\||<<|>>|[=+\-*/<>])"
    r"|[^\s0-9a-zA-Z]"
    r"|\s"
)


class MorphCodeTokenizer:
    """`MorphCode-16K` (`vocab_size = 16,384`) Morpheme + Code + Math Tokenizer with UTF-8 Byte Fallback."""

    pad_id: int = 256
    bos_id: int = 257
    eos_id: int = 258
    unk_id: int = 259
    vocab_size: int = 16_384
    prompt_mask_bit: int = 0x8000

    _SHARED_DATA: dict | None = None

    def __init__(self, vocab_path: str | Path | None = None) -> None:
        path = Path(vocab_path) if vocab_path is not None else VOCAB_JSON_PATH
        data = self._load_vocab_data(str(path.resolve()))
        self._id_to_bytes: tuple[bytes, ...] = data["id_to_bytes"]
        self._str_to_id: dict[str, int] = data["str_to_id"]
        self._stem_set: frozenset[str] = data["stem_set"]
        self._prefix_set: frozenset[str] = data["prefix_set"]
        self._suffix_set: frozenset[str] = data["suffix_set"]
        self._byte_lengths_tensor: torch.Tensor = data["byte_lengths_tensor"]
        self._sentence_end_ids: frozenset[int] = data["sentence_end_ids"]
        self._clause_end_ids: frozenset[int] = data["clause_end_ids"]
        self._newline_ids: frozenset[int] = data["newline_ids"]
        self._hash_start_ids: frozenset[int] = data["hash_start_ids"]
        self._xml_open_ids: frozenset[int] = data["xml_open_ids"]

    @classmethod
    @lru_cache(maxsize=4)
    def _load_vocab_data(cls, resolved_path: str) -> dict:
        p = Path(resolved_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"MorphCode-16K vocabulary file not found at {p}. "
                "Run `py.exe scripts/build_morphcode_vocab.py` first."
            )
        raw = json.loads(p.read_text(encoding="utf-8"))
        pieces: list[str] = raw["pieces"]
        if len(pieces) != cls.vocab_size:
            raise ValueError(f"Expected {cls.vocab_size} tokens in {p}, got {len(pieces)}")

        id_to_bytes: list[bytes] = []
        str_to_id: dict[str, int] = {}
        byte_lengths: list[int] = []

        for idx, piece in enumerate(pieces):
            if idx < 256:
                b = bytes([idx])
                id_to_bytes.append(b)
                byte_lengths.append(1)
            elif idx in (cls.pad_id, cls.bos_id, cls.eos_id, cls.unk_id):
                id_to_bytes.append(b"")
                byte_lengths.append(0)
            else:
                b = piece.encode("utf-8")
                id_to_bytes.append(b)
                byte_lengths.append(len(b))
                # Map string piece -> token ID (preferring specialized multi-byte IDs >= 260)
                if piece not in str_to_id:
                    str_to_id[piece] = idx

        # Also map single ASCII printable characters (except digits, which stay 48..57) for 1-char fallback
        for b_idx in range(256):
            ch = chr(b_idx)
            if b_idx < 128 and ch not in str_to_id:
                str_to_id[ch] = b_idx

        stem_set = frozenset(raw.get("stems", []))
        prefix_set = frozenset(MORPHEME_PREFIXES)
        suffix_set = frozenset(MORPHEME_SUFFIXES)

        sentence_end_ids: set[int] = {ord("."), ord("!"), ord("?"), 10}
        clause_end_ids: set[int] = {10, ord("."), ord("!"), ord("?"), ord(","), ord(";"), ord(":")}
        newline_ids: set[int] = {10}
        hash_start_ids: set[int] = {ord("#")}
        xml_open_ids: set[int] = {ord("<")}

        for idx in range(260, cls.vocab_size):
            piece = pieces[idx]
            if not piece:
                continue
            stripped = piece.rstrip(" ")
            if stripped.endswith((".", "!", "?")) or piece.endswith("\n"):
                sentence_end_ids.add(idx)
                clause_end_ids.add(idx)
            elif stripped.endswith((",", ";", ":")):
                clause_end_ids.add(idx)
            if "\n" in piece:
                newline_ids.add(idx)
            if piece.lstrip().startswith("#"):
                hash_start_ids.add(idx)
            if piece.lstrip().startswith("<"):
                xml_open_ids.add(idx)

        return {
            "id_to_bytes": tuple(id_to_bytes),
            "str_to_id": str_to_id,
            "stem_set": stem_set,
            "prefix_set": prefix_set,
            "suffix_set": suffix_set,
            "byte_lengths_tensor": torch.tensor(byte_lengths, dtype=torch.long),
            "sentence_end_ids": frozenset(sentence_end_ids),
            "clause_end_ids": frozenset(clause_end_ids),
            "newline_ids": frozenset(newline_ids),
            "hash_start_ids": frozenset(hash_start_ids),
            "xml_open_ids": frozenset(xml_open_ids),
        }

    @property
    def byte_lengths(self) -> torch.Tensor:
        """Return `[vocab_size]` tensor mapping each token ID to its UTF-8 byte length."""
        return self._byte_lengths_tensor

    def count_utf8_bytes(self, token_ids: torch.Tensor) -> int:
        """Compute total UTF-8 bytes represented by valid (`>= 0`, `!= pad_id`) token IDs."""
        valid_mask = (token_ids >= 0) & (token_ids != self.pad_id)
        if not bool(valid_mask.any()):
            return 0
        clean_ids = (token_ids[valid_mask] & 0x7FFF).clamp(0, self.vocab_size - 1).cpu()
        return int(self._byte_lengths_tensor.index_select(0, clean_ids).sum().item())

    def _encode_subword_fallback(self, span: str) -> tuple[int, ...]:
        """Longest-prefix match in `_str_to_id` with raw UTF-8 byte fallback (`0..255`)."""
        str_to_id = self._str_to_id
        if span in str_to_id:
            return (str_to_id[span],)
        out: list[int] = []
        n = len(span)
        i = 0
        while i < n:
            matched_id = -1
            matched_len = 0
            max_end = min(n, i + 16)
            for end in range(max_end, i, -1):
                sub = span[i:end]
                tid = str_to_id.get(sub)
                if tid is not None and (end - i > 1 or ord(sub[0]) < 128):
                    matched_id = tid
                    matched_len = end - i
                    break
            if matched_id >= 0:
                out.append(matched_id)
                i += matched_len
            else:
                out.extend(span[i].encode("utf-8"))
                i += 1
        return tuple(out)

    @lru_cache(maxsize=65536)
    def _encode_word_morphemic(self, word_span: str) -> tuple[int, ...]:
        """Segment a word (`'?' + letters`) into `[prefix] + root_stem + [suffix]` or vocabulary pieces."""
        str_to_id = self._str_to_id
        has_space = word_span.startswith(" ")
        bare = word_span[1:] if has_space else word_span
        lower_bare = bare.lower()

        # 1. Short closed-class / atomic root words or very short words (`<= 4` chars):
        #    if directly in vocabulary, emit as a single atomic token.
        if (len(bare) <= 4 or lower_bare in ATOMIC_SHORT_WORDS) and word_span in str_to_id:
            return (str_to_id[word_span],)

        # 2. If bare word is itself a registered canonical root stem and exists directly in vocab,
        #    emit as a single root token (e.g., `" transport"`, `" capillary"`, `" function"`, `" network"`).
        if lower_bare in self._stem_set and word_span in str_to_id:
            return (str_to_id[word_span],)

        # 3. Try morpheme decomposition: `[optional prefix] + canonical_stem + [optional suffix]`
        #    We prioritize decompositions where every piece (`prefix_str`, `stem_str`, `suffix_str`)
        #    is a single token in `str_to_id` and `stem` is at least 3 chars long.
        if bare.islower() or (bare[0].isupper() and bare[1:].islower()):
            is_title = bare[0].isupper() and bare[1:].islower()

            # 3a. Check `stem + suffix` first (2-token morphological split, e.g. `" acceler" + "ation"`, `" plant" + "s"`)
            for suf in MORPHEME_SUFFIXES:
                if len(bare) >= len(suf) + 3 and bare.endswith(suf) and suf in str_to_id:
                    stem_part = bare[: -len(suf)]
                    stem_lower = stem_part.lower()
                    if stem_lower in self._stem_set or len(stem_part) >= 4:
                        stem_tok_str = (" " + stem_part) if has_space else stem_part
                        if stem_tok_str in str_to_id:
                            return (str_to_id[stem_tok_str], str_to_id[suf])
                        # Orthographic doubled-consonant / dropped-e check: e.g. `running -> runn + ing`
                        if (
                            len(stem_part) >= 4
                            and stem_part[-1] == stem_part[-2]
                            and stem_part[:-1].lower() in self._stem_set
                        ):
                            base_str = (" " + stem_part[:-1]) if has_space else stem_part[:-1]
                            tail_str = stem_part[-1] + suf
                            if base_str in str_to_id and tail_str in str_to_id:
                                return (str_to_id[base_str], str_to_id[tail_str])

            # 3b. Check `prefix + stem` (2-token split, e.g. `" re" + "build"`, `" un" + "known"`)
            for pref in MORPHEME_PREFIXES:
                pref_cased = pref.capitalize() if is_title else pref
                if len(bare) >= len(pref) + 3 and bare.startswith(pref_cased):
                    rem = bare[len(pref) :]
                    if rem in str_to_id and (rem.lower() in self._stem_set or len(rem) >= 4):
                        pref_tok_str = (" " + pref_cased) if has_space else pref_cased
                        if pref_tok_str in str_to_id:
                            return (str_to_id[pref_tok_str], str_to_id[rem])

            # 3c. Check `prefix + stem + suffix` (3-token split, e.g. `" re" + "calcul" + "ating"`)
            for pref in MORPHEME_PREFIXES:
                pref_cased = pref.capitalize() if is_title else pref
                if len(bare) >= len(pref) + 5 and bare.startswith(pref_cased):
                    pref_tok_str = (" " + pref_cased) if has_space else pref_cased
                    if pref_tok_str not in str_to_id:
                        continue
                    after_pref = bare[len(pref) :]
                    for suf in MORPHEME_SUFFIXES:
                        if (
                            len(after_pref) >= len(suf) + 3
                            and after_pref.endswith(suf)
                            and suf in str_to_id
                        ):
                            mid_stem = after_pref[: -len(suf)]
                            if mid_stem in str_to_id and (
                                mid_stem.lower() in self._stem_set or len(mid_stem) >= 4
                            ):
                                return (
                                    str_to_id[pref_tok_str],
                                    str_to_id[mid_stem],
                                    str_to_id[suf],
                                )

        # 4. If whole `word_span` is in vocab (e.g., high-frequency word or identifier), emit it directly
        if word_span in str_to_id:
            return (str_to_id[word_span],)

        # 5. If `has_space` and bare word is in vocab (or decomposes cleanly without leading space),
        #    check if splitting the space or keeping longest subword is more compact
        return self._encode_subword_fallback(word_span)

    def _encode_plain_span(self, span: str) -> list[int]:
        if not span:
            return []
        str_to_id = self._str_to_id
        tokens: list[int] = []
        for match in _MASTER_SCAN_RE.finditer(span):
            chunk = match.group(0)
            c0 = chunk[0]
            if len(chunk) == 1 and "0" <= c0 <= "9":
                # Single-digit rule: digits '0'..'9' are ALWAYS ASCII byte IDs 48..57
                tokens.append(ord(c0))
            elif c0.isalpha() or (
                c0 == " " and len(chunk) > 1 and chunk[1].isalpha()
            ):
                tokens.extend(self._encode_word_morphemic(chunk))
            elif chunk in str_to_id:
                tokens.append(str_to_id[chunk])
            else:
                tokens.extend(self._encode_subword_fallback(chunk))
        return tokens

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Encode `text` into `MorphCode-16K` token IDs (`0..16,383`)."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.bos_id)
        if text:
            str_to_id = self._str_to_id
            parts = _TRACE_MARKER_SPLIT_RE.split(text)
            for idx, part in enumerate(parts):
                if not part:
                    continue
                if idx % 2 == 1 and part in str_to_id:
                    tokens.append(str_to_id[part])
                else:
                    tokens.extend(self._encode_plain_span(part))
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
        """Encode a `#U$# -> #T$# -> #A$#` micro-trace with `0x8000` high-bit masking on `#U$#`."""
        if not prompt.strip():
            raise ValueError("prompt (#U$#) must be non-empty")
        if not thought.strip():
            raise ValueError("thought (#T$#) scratchpad is mandatory in MCL4-COE micro-traces")
        if not answer.strip():
            raise ValueError("answer (#A$#) must be non-empty")

        prefix_str = f"#U$# {prompt.strip()}\n"
        completion_str = f"#T$# {thought.strip()}\n#A$# {answer.strip()}"

        prefix_ids = self.encode(prefix_str, add_bos=True, add_eos=False)
        masked_prefix = [tok | self.prompt_mask_bit for tok in prefix_ids]
        completion_ids = self.encode(completion_str, add_bos=False, add_eos=True)
        full = masked_prefix + completion_ids

        if seq_len is not None:
            if len(full) > seq_len:
                raise ValueError(
                    f"Micro-trace length ({len(full)}) exceeds block seq_len={seq_len}; "
                    "never sever traces mid-thought across blocks"
                )
            full.extend([self.pad_id] * (seq_len - len(full)))
        return full

    def decode(self, token_ids: Iterable[int] | torch.Tensor) -> str:
        """Decode `MorphCode-16K` token IDs back into a UTF-8 string (stripping `0x8000` prompt mask)."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        id_to_bytes = self._id_to_bytes
        vocab_n = self.vocab_size
        chunks: list[bytes] = []
        for tid in token_ids:
            clean = int(tid) & 0x7FFF
            if 0 <= clean < vocab_n:
                b = id_to_bytes[clean]
                if b:
                    chunks.append(b)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def tokenize_to_pieces(self, text: str) -> list[str]:
        """Return the human-readable list of token pieces produced for `text` (without `<bos>`/`<eos>`)."""
        ids = self.encode(text, add_bos=False, add_eos=False)
        return [self._id_to_bytes[i].decode("utf-8", errors="replace") for i in ids]

    def tokenize_to_strings(self, text: str) -> list[str]:
        """Alias for `tokenize_to_pieces(text)` returning decoded UTF-8 token pieces."""
        return self.tokenize_to_pieces(text)

    def compression_ratio(self, text: str) -> float:
        """Return UTF-8 byte count divided by `MorphCode-16K` token count (excluding `<bos>`/`<eos>`)."""
        if not text:
            return 1.0
        n_bytes = len(text.encode("utf-8"))
        n_toks = max(1, len(self.encode(text, add_bos=False, add_eos=False)))
        return n_bytes / float(n_toks)

    def count_utf8_bytes(self, token_ids: Iterable[int] | torch.Tensor) -> int:
        """Count total UTF-8 bytes represented by non-special, unmasked (`!= -100`) token IDs."""
        if isinstance(token_ids, torch.Tensor):
            flat = token_ids.detach().cpu().view(-1).tolist()
        else:
            flat = [int(t) for t in token_ids]
        id_to_bytes = self._id_to_bytes
        vocab_n = self.vocab_size
        total = 0
        for tid in flat:
            if tid < 0:
                continue
            clean = tid & 0x7FFF
            if 0 <= clean < vocab_n:
                total += len(id_to_bytes[clean])
        return total

