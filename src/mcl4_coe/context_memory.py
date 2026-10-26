"""Four-Tier Context & Virtual Memory Hierarchy for `MCL4-COE`.

Unifies:
1. Tier 2 (`TurboQuantKV`): Data-oblivious randomized Walsh-Hadamard rotation (`H * D`) +
   3-bit Lloyd-Max hypersphere quantization + 1-bit Quantized Johnson-Lindenstrauss (`QJL`)
   residual sign sketch for ~4.2x KV / `EncodedState` compression with unbiased inner products.
2. Tier 3 (`ContextLMCompressor`): Multi-chunk continuous Context Language Model compressor
   that condenses long byte streams into compact gist vectors and `RecurrentFeedback` states.
3. Tier 4 (`A.TXY` Virtual Memory & `COEAddressRouter`): Base-36 factorized `A.TXY` virtual
   memory (`1,679,616` logical addresses across Planes `A0..A5`), SHA-256 verified page-table
   lookup, `N5` CRC32 append-only WAL writes, `N1` 1-position factorized address embeddings,
   `N6` single-register pointer programs (`READ`, `FOLLOW`, `UP`, `DOWN`, `PROJECT`, `FILTER`,
   `FIND`, `COUNT`, `COMPARE`), and `COEAddressRouter` (solving `N3b-primary` natural-language
   candidate routing via `Association` + `Decision`).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time
import unicodedata
from typing import Any, Iterable, Sequence
import zlib

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .backbone import EncodedState, MCL4CEDBackbone, RecurrentFeedback
from .data import DecisionExample, collate_decisions, get_tokenizer
from .operators import CognitiveOperatorEnsemble, pearson_similarity

# ---------------------------------------------------------------------------
# Tier 2: TurboQuant (Walsh-Hadamard + 3-bit Polar/Lloyd-Max + 1-bit QJL)
# ---------------------------------------------------------------------------

# Optimal 8-level (3-bit) Lloyd-Max centroids for a standard normal N(0, 1) variable
_LLOYD_MAX_3BIT_NORMAL: tuple[float, ...] = (
    -2.1520,
    -1.3439,
    -0.7560,
    -0.2451,
    0.2451,
    0.7560,
    1.3439,
    2.1520,
)
_LLOYD_MAX_3BIT_BOUNDS: tuple[float, ...] = (
    -1.7480,
    -1.0500,
    -0.5006,
    0.0000,
    0.5006,
    1.0500,
    1.7480,
)


def _rademacher_signs(dim: int, seed: int, device: torch.device) -> Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bits = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.int64)
    signs = (bits * 2 - 1).to(dtype=torch.float32, device=device)
    return signs


def fast_walsh_hadamard(x: Tensor, signs: Tensor | None = None, *, inverse: bool = False) -> Tensor:
    """Orthonormal Fast Walsh-Hadamard Transform ($O(D \\log D)$) along the last dimension.

    For non-power-of-two `D`, pads to the next power of two or applies block-diagonal FWHT
    on the largest power-of-two factor. When `signs` is provided:
    - forward (`inverse=False`): computes $H (s \\odot x) / \\sqrt{D}$
    - inverse (`inverse=True`): computes $s \\odot (H x / \\sqrt{D})$
    """
    orig_dtype = x.dtype
    h = x.float()
    dim = h.shape[-1]
    if dim <= 0:
        raise ValueError(f"fast_walsh_hadamard requires positive last dim, got {dim}")
    block_dim = dim & -dim

    if not inverse and signs is not None:
        h = h * signs.to(device=h.device, dtype=torch.float32)

    if block_dim > 1:
        n_blocks = dim // block_dim
        h = h.reshape(*x.shape[:-1], n_blocks, block_dim)
        step = 1
        while step < block_dim:
            h = h.view(*x.shape[:-1], n_blocks, block_dim // (step * 2), 2, step)
            a = h[..., 0, :]
            b = h[..., 1, :]
            h = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], n_blocks, block_dim)
            step *= 2
        h = (h * (1.0 / math.sqrt(block_dim))).reshape(*x.shape)

    if inverse and signs is not None:
        h = h * signs.to(device=h.device, dtype=torch.float32)
    return h.to(orig_dtype)


@dataclass
class CompressedKV:
    """TurboQuant compressed representation of a `[..., D]` key/value or encoder state tensor."""

    codes_3bit: Tensor          # uint8 in [0, 7], shape [..., D]
    qjl_signs: Tensor           # uint8 in {0, 1} representing {-1, +1}, shape [..., D]
    row_norms: Tensor           # float16/bfloat16 row L2 norms, shape [..., 1]
    residual_norms: Tensor      # float16/bfloat16 residual L2 norms, shape [..., 1]
    seed: int
    orig_dtype: torch.dtype
    orig_shape: tuple[int, ...]

    @property
    def uncompressed_bytes(self) -> int:
        elem_size = 2 if self.orig_dtype in (torch.bfloat16, torch.float16) else 4
        n_elems = math.prod(self.orig_shape)
        return n_elems * elem_size

    @property
    def packed_logical_bytes(self) -> int:
        """Theoretical bit-packed footprint: 3 bits (polar/Lloyd-Max) + 1 bit (QJL) + 2 FP16 scalars per row."""
        n_elems = math.prod(self.orig_shape)
        n_rows = n_elems // self.orig_shape[-1]
        return (n_elems * 4 + 7) // 8 + n_rows * 4

    @property
    def compression_ratio(self) -> float:
        return self.uncompressed_bytes / max(1, self.packed_logical_bytes)


def turboquant_compress(x: Tensor, *, seed: int = 42) -> CompressedKV:
    """Compress `x` (`[..., D]`) via randomized FWHT + 3-bit Lloyd-Max + 1-bit QJL residual sketch."""
    orig_shape = tuple(x.shape)
    orig_dtype = x.dtype
    dim = orig_shape[-1]
    xf = x.float()

    row_norms = xf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    unit_x = xf / row_norms

    signs = _rademacher_signs(dim, seed, x.device)
    rotated = fast_walsh_hadamard(unit_x, signs, inverse=False)

    # Scale by sqrt(D) so coordinates match standard N(0, 1) variance on the unit sphere
    scaled = rotated * math.sqrt(dim)
    bounds = torch.tensor(_LLOYD_MAX_3BIT_BOUNDS, device=x.device, dtype=torch.float32)
    centroids = torch.tensor(_LLOYD_MAX_3BIT_NORMAL, device=x.device, dtype=torch.float32)

    codes = torch.bucketize(scaled, bounds).to(torch.uint8)
    quant_rotated = centroids[codes.long()] / math.sqrt(dim)

    residual = rotated - quant_rotated
    res_norms = residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qjl_bits = (residual >= 0).to(torch.uint8)

    return CompressedKV(
        codes_3bit=codes,
        qjl_signs=qjl_bits,
        row_norms=row_norms.to(torch.float16),
        residual_norms=res_norms.to(torch.float16),
        seed=seed,
        orig_dtype=orig_dtype,
        orig_shape=orig_shape,
    )


def turboquant_decompress(comp: CompressedKV) -> Tensor:
    """Reconstruct `[..., D]` tensor from `CompressedKV` with QJL residual correction."""
    device = comp.codes_3bit.device
    dim = comp.orig_shape[-1]
    centroids = torch.tensor(_LLOYD_MAX_3BIT_NORMAL, device=device, dtype=torch.float32)

    quant_rotated = centroids[comp.codes_3bit.long()] / math.sqrt(dim)
    qjl_pm1 = comp.qjl_signs.float() * 2.0 - 1.0
    # Unbiased 1-bit QJL reconstruction factor sqrt(pi / (2 * D)) * ||residual||_2 * sign(residual)
    qjl_correction = math.sqrt(math.pi / (2.0 * dim)) * comp.residual_norms.float() * qjl_pm1
    reconstructed_rotated = quant_rotated + qjl_correction

    signs = _rademacher_signs(dim, comp.seed, device)
    unit_x = fast_walsh_hadamard(reconstructed_rotated, signs, inverse=True)
    out = unit_x * comp.row_norms.float()
    return out.view(*comp.orig_shape).to(comp.orig_dtype)


def turboquant_inner_product(queries: Tensor, comp_keys: CompressedKV) -> Tensor:
    """Compute unbiased attention/similarity inner products `<queries, keys>` directly in rotated space."""
    device = queries.device
    dim = queries.shape[-1]
    signs = _rademacher_signs(dim, comp_keys.seed, device)
    q_rot = fast_walsh_hadamard(queries.float(), signs, inverse=False)

    centroids = torch.tensor(_LLOYD_MAX_3BIT_NORMAL, device=device, dtype=torch.float32)
    k_polar = centroids[comp_keys.codes_3bit.long()] / math.sqrt(dim)
    k_qjl = (
        math.sqrt(math.pi / (2.0 * dim))
        * comp_keys.residual_norms.float()
        * (comp_keys.qjl_signs.float() * 2.0 - 1.0)
    )
    k_rot = (k_polar + k_qjl) * comp_keys.row_norms.float()
    return torch.matmul(q_rot, k_rot.transpose(-1, -2)).to(queries.dtype)


@dataclass
class CompressedEncodedState:
    """TurboQuant-compressed `EncodedState` for low-VRAM multi-pass `COE` & long-context caching."""

    compressed_hidden: CompressedKV
    valid: Tensor
    aux_loss: Tensor
    router_z_loss: Tensor
    expert_load: Tensor

    @property
    def uncompressed_bytes(self) -> int:
        return self.compressed_hidden.uncompressed_bytes

    @property
    def compressed_bytes(self) -> int:
        return self.compressed_hidden.packed_logical_bytes

    @property
    def compression_ratio(self) -> float:
        return self.compressed_hidden.compression_ratio

    def decompress(self) -> EncodedState:
        return EncodedState(
            hidden=turboquant_decompress(self.compressed_hidden),
            valid=self.valid,
            aux_loss=self.aux_loss,
            router_z_loss=self.router_z_loss,
            expert_load=self.expert_load,
        )


def compress_encoded_state(state: EncodedState, *, seed: int = 42) -> CompressedEncodedState:
    """Compress `EncodedState.hidden` with TurboQuant while preserving exact validity masks and MoE stats."""
    return CompressedEncodedState(
        compressed_hidden=turboquant_compress(state.hidden, seed=seed),
        valid=state.valid,
        aux_loss=state.aux_loss,
        router_z_loss=state.router_z_loss,
        expert_load=state.expert_load,
    )


# ---------------------------------------------------------------------------
# Tier 3: Context-LM Multi-Chunk Recurrent State Compressor
# ---------------------------------------------------------------------------


class ContextLMCompressor(nn.Module):
    """Streaming Context-LM compressor that condenses long byte streams into compact gist states.

    Splits an arbitrarily long input `[B, L_total]` into windows of `chunk_size`, encodes each
    chunk through `MCL4CEDBackbone.encode_state`, blends running cross-chunk state via
    `MCL4CEDBackbone.feedback` (`RecurrentFeedback`), and pools each chunk into
    `gist_tokens_per_chunk` summary vectors so downstream `COE` operators or decoders can attend
    to an $O(N_{\\text{chunks}} \\cdot G)$ compressed context.
    """

    def __init__(
        self,
        d_model: int,
        *,
        chunk_size: int = 256,
        gist_tokens_per_chunk: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.gist_tokens_per_chunk = gist_tokens_per_chunk
        self.gist_queries = nn.Parameter(torch.randn(gist_tokens_per_chunk, d_model) * 0.02)
        self.gist_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        backbone: MCL4CEDBackbone,
        token_ids: Tensor,
        valid_mask: Tensor | None = None,
    ) -> EncodedState:
        batch_size, total_len = token_ids.shape
        if valid_mask is None:
            valid_mask = torch.ones_like(token_ids, dtype=torch.bool)

        gist_chunks: list[Tensor] = []
        gist_valids: list[Tensor] = []
        running_gist: Tensor | None = None
        total_aux = token_ids.new_zeros((), dtype=torch.float32)
        total_z = token_ids.new_zeros((), dtype=torch.float32)
        loads: list[Tensor] = []

        for start in range(0, total_len, self.chunk_size):
            end = min(total_len, start + self.chunk_size)
            c_ids = token_ids[:, start:end]
            c_val = valid_mask[:, start:end]
            if not c_val.any():
                continue

            row_has_valid = c_val.any(dim=-1)
            c_val_safe = c_val.clone()
            c_val_safe[~row_has_valid, 0] = True

            enc = backbone.encode_state(c_ids, c_val_safe, causal=False)
            total_aux = total_aux + enc.aux_loss
            total_z = total_z + enc.router_z_loss
            loads.append(enc.expert_load)

            # Cross-attend `gist_queries` over the chunk's valid token representations
            q = self.gist_queries.unsqueeze(0).expand(batch_size, -1, -1).to(enc.hidden.dtype)
            scores = torch.matmul(q.float(), enc.hidden.float().transpose(-1, -2)) / math.sqrt(
                self.d_model
            )
            scores = scores.masked_fill(~c_val_safe.unsqueeze(1), -1e9)
            attn = F.softmax(scores, dim=-1).to(enc.hidden.dtype)
            pooled = torch.matmul(attn, enc.hidden).float()
            chunk_gist = (0.85 * pooled + 0.15 * self.gist_norm(pooled)).to(enc.hidden.dtype)

            chunk_has_valid = row_has_valid.unsqueeze(-1).expand(
                batch_size, self.gist_tokens_per_chunk
            )
            chunk_gist = chunk_gist.masked_fill(~chunk_has_valid.unsqueeze(-1), 0)
            if running_gist is not None:
                safe_valid = chunk_has_valid.clone()
                safe_valid[~row_has_valid, 0] = True
                updated_gist = backbone.feedback(chunk_gist, running_gist, valid=safe_valid)
                chunk_gist = torch.where(
                    chunk_has_valid.unsqueeze(-1),
                    0.75 * chunk_gist + 0.25 * updated_gist,
                    running_gist,
                )
            running_gist = chunk_gist

            gist_chunks.append(chunk_gist)
            gist_valids.append(chunk_has_valid)

        if not gist_chunks:
            return backbone.encode_state(token_ids[:, :1], valid_mask[:, :1])

        compressed_hidden = torch.cat(gist_chunks, dim=1)
        compressed_valid = torch.cat(gist_valids, dim=1)
        avg_load = torch.stack(loads, dim=0).mean(dim=0)
        return EncodedState(
            hidden=compressed_hidden,
            valid=compressed_valid,
            aux_loss=total_aux / max(1, len(loads)),
            router_z_loss=total_z / max(1, len(loads)),
            expert_load=avg_load,
        )

    def compress_encoded_states(
        self,
        backbone: MCL4CEDBackbone,
        states: Sequence[EncodedState],
    ) -> EncodedState:
        """Fold a sequence of pre-encoded or decompressed `EncodedState` file pages into recurrent soft gist tokens."""
        if not states:
            raise ValueError("states must be non-empty in compress_encoded_states")
        device = states[0].hidden.device
        dtype = states[0].hidden.dtype
        batch_size = states[0].hidden.shape[0]

        gist_chunks: list[Tensor] = []
        gist_valids: list[Tensor] = []
        running_gist: Tensor | None = None
        total_aux = torch.zeros((), dtype=torch.float32, device=device)
        total_z = torch.zeros((), dtype=torch.float32, device=device)
        loads: list[Tensor] = []

        for enc in states:
            h = enc.hidden.to(device=device, dtype=dtype)
            c_val = enc.valid.to(device=device, dtype=torch.bool)
            row_has_valid = c_val.any(dim=-1)
            c_val_safe = c_val.clone()
            c_val_safe[~row_has_valid, 0] = True

            total_aux = total_aux + enc.aux_loss.to(device=device).float()
            total_z = total_z + enc.router_z_loss.to(device=device).float()
            loads.append(enc.expert_load.to(device=device).float())

            q = self.gist_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)
            scores = torch.matmul(q.float(), h.float().transpose(-1, -2)) / math.sqrt(self.d_model)
            scores = scores.masked_fill(~c_val_safe.unsqueeze(1), -1e9)
            attn = F.softmax(scores, dim=-1).to(dtype)
            pooled = torch.matmul(attn, h).float()
            chunk_gist = (0.85 * pooled + 0.15 * self.gist_norm(pooled)).to(dtype)

            chunk_has_valid = row_has_valid.unsqueeze(-1).expand(
                batch_size, self.gist_tokens_per_chunk
            )
            chunk_gist = chunk_gist.masked_fill(~chunk_has_valid.unsqueeze(-1), 0)
            if running_gist is not None:
                safe_valid = chunk_has_valid.clone()
                safe_valid[~row_has_valid, 0] = True
                updated_gist = backbone.feedback(chunk_gist, running_gist, valid=safe_valid)
                chunk_gist = torch.where(
                    chunk_has_valid.unsqueeze(-1),
                    (0.75 * chunk_gist + 0.25 * updated_gist).to(dtype),
                    running_gist,
                )
            running_gist = chunk_gist

            gist_chunks.append(chunk_gist)
            gist_valids.append(chunk_has_valid)

        compressed_hidden = torch.cat(gist_chunks, dim=1)
        compressed_valid = torch.cat(gist_valids, dim=1)
        avg_load = torch.stack(loads, dim=0).mean(dim=0)
        return EncodedState(
            hidden=compressed_hidden,
            valid=compressed_valid,
            aux_loss=total_aux / max(1, len(loads)),
            router_z_loss=total_z / max(1, len(loads)),
            expert_load=avg_load,
        )

    def compress_turboquant_pages(
        self,
        backbone: MCL4CEDBackbone,
        pages: Sequence[CompressedEncodedState],
    ) -> EncodedState:
        """Decompress multiple 4-bit `TurboQuant` file pages and fold them into recurrent soft gist tokens."""
        decompressed = [p.decompress() for p in pages]
        return self.compress_encoded_states(backbone, decompressed)


# ---------------------------------------------------------------------------
# Tier 4: A.TXY Factorized Virtual Memory, Pointer Programs & COE Router
# ---------------------------------------------------------------------------

ATXY_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ATXY_BASE = len(ATXY_ALPHABET)  # 36
ATXY_N_ADDRESSES = ATXY_BASE**4  # 1,679,616
_ATXY_VAL = {ch: idx for idx, ch in enumerate(ATXY_ALPHABET)}

ATXY_OPS: tuple[str, ...] = (
    "READ",
    "FOLLOW",
    "UP",
    "DOWN",
    "PROJECT",
    "FILTER",
    "FIND",
    "FOLLOW_SET",
    "COUNT",
    "COMPARE",
)


@dataclass(frozen=True)
class ATXYAddress:
    """Four-field base-36 semantic address `A.TXY` (`A`: plane, `T`: table, `X`: row, `Y`: col)."""

    A: int
    T: int
    X: int
    Y: int

    def __post_init__(self) -> None:
        for name, val in (("A", self.A), ("T", self.T), ("X", self.X), ("Y", self.Y)):
            if not isinstance(val, int) or not (0 <= val < ATXY_BASE):
                raise ValueError(f"ATXY field {name}={val!r} out of range [0, {ATXY_BASE})")

    @property
    def packed(self) -> int:
        return ((self.A * ATXY_BASE + self.T) * ATXY_BASE + self.X) * ATXY_BASE + self.Y

    @property
    def directory_index(self) -> int:
        return self.A * ATXY_BASE + self.T

    @property
    def cell(self) -> int:
        return self.X * ATXY_BASE + self.Y

    def format(self, *, dot: bool = True, alphabet: str = ATXY_ALPHABET) -> str:
        body = alphabet[self.A] + alphabet[self.T] + alphabet[self.X] + alphabet[self.Y]
        return f"{body[0]}.{body[1:]}" if dot else body

    def __str__(self) -> str:
        return self.format(dot=True)


def pack_atxy(A: int, T: int, X: int, Y: int) -> int:
    return ATXYAddress(A, T, X, Y).packed


def unpack_atxy(packed: int) -> ATXYAddress:
    if not isinstance(packed, int) or not (0 <= packed < ATXY_N_ADDRESSES):
        raise ValueError(f"packed={packed!r} out of range [0, {ATXY_N_ADDRESSES})")
    Y = packed % ATXY_BASE
    X = (packed // ATXY_BASE) % ATXY_BASE
    T = (packed // (ATXY_BASE * ATXY_BASE)) % ATXY_BASE
    A = (packed // (ATXY_BASE**3)) % ATXY_BASE
    return ATXYAddress(A, T, X, Y)


def parse_atxy(text: str, *, alphabet: str = ATXY_ALPHABET) -> ATXYAddress:
    s = text.strip()
    if alphabet == ATXY_ALPHABET:
        s = s.upper()
    if "." in s:
        if len(s) != 5 or s[1] != ".":
            raise ValueError(f"Invalid dotted A.TXY address: {text!r}")
        chars = s[0] + s[2:]
    else:
        if len(s) != 4:
            raise ValueError(f"Invalid undotted ATXY address: {text!r}")
        chars = s
    val_map = _ATXY_VAL if alphabet == ATXY_ALPHABET else {c: i for i, c in enumerate(alphabet)}
    try:
        A, T, X, Y = (val_map[c] for c in chars)
    except KeyError as exc:
        raise ValueError(f"Character {exc} not in alphabet") from None
    return ATXYAddress(A, T, X, Y)


@dataclass(frozen=True)
class ATXYRecord:
    """Verified canonical record stored at an `A.TXY` cell."""

    address: str
    kind: str
    payload: str
    summary: str = ""
    up: tuple[str, ...] = ()
    down: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    rels: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fields: dict[str, int] = field(default_factory=dict)
    version: int = 1
    schema_version: int = 1
    provenance: tuple[str, ...] = ()

    @property
    def parsed_address(self) -> ATXYAddress:
        return parse_atxy(self.address)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "address": str(parse_atxy(self.address)),
            "fields": {str(k): int(v) for k, v in sorted(self.fields.items())},
            "kind": str(self.kind),
            "payload": str(self.payload),
            "provenance": list(self.provenance),
            "related": [str(parse_atxy(x)) for x in self.related],
            "rels": {
                str(k): sorted(str(parse_atxy(x)) for x in v)
                for k, v in sorted(self.rels.items())
            },
            "schema_version": int(self.schema_version),
            "summary": str(self.summary),
            "up": [str(parse_atxy(x)) for x in self.up],
            "down": [str(parse_atxy(x)) for x in self.down],
            "version": int(self.version),
        }

    def digest(self) -> str:
        addr = self.parsed_address
        h = hashlib.sha256()
        h.update(struct.pack("<III", self.schema_version, self.version, addr.packed))
        raw = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        h.update(raw)
        return h.hexdigest()

    def to_candidate_description(self) -> str:
        """Render a compact `#T$# ... #A$#` candidate description for `COE` routing."""
        rel_str = ", ".join(f"{k}:{len(v)}" for k, v in sorted(self.rels.items()) if v)
        meta = f"[{self.kind} @{self.address}] {self.summary}".strip()
        if rel_str:
            meta = f"{meta} ({rel_str})"
        return f"#T$# {meta}\n#A$# {self.payload}"


class ATXYFactorizedEmbedding(nn.Module):
    """`elem` N1 validated 1-position factorized address embedding: $E_A(A) + E_T(T) + E_X(X) + E_Y(Y)$."""

    def __init__(self, d_model: int, *, base: int = ATXY_BASE) -> None:
        super().__init__()
        self.base = base
        self.d_model = d_model
        self.plane_emb = nn.Embedding(base, d_model)
        self.table_emb = nn.Embedding(base, d_model)
        self.row_emb = nn.Embedding(base, d_model)
        self.col_emb = nn.Embedding(base, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, coords: Tensor) -> Tensor:
        """Encode `[..., 4]` integer coordinates `(A, T, X, Y)` into `[..., d_model]` single-position vectors."""
        if coords.shape[-1] != 4:
            raise ValueError(f"Expected last dim 4 for (A, T, X, Y), got {coords.shape}")
        a = coords[..., 0]
        t = coords[..., 1]
        x = coords[..., 2]
        y = coords[..., 3]
        emb = self.plane_emb(a) + self.table_emb(t) + self.row_emb(x) + self.col_emb(y)
        return self.norm(emb)

    def encode_addresses(
        self,
        addresses: Sequence[str | int | ATXYAddress],
        *,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        rows = []
        for item in addresses:
            addr = (
                item
                if isinstance(item, ATXYAddress)
                else (unpack_atxy(item) if isinstance(item, int) else parse_atxy(item))
            )
            rows.append([addr.A, addr.T, addr.X, addr.Y])
        tensor_coords = torch.tensor(rows, dtype=torch.long, device=device)
        return self.forward(tensor_coords)


class ATXYPageStore:
    """Two-level `(A, T) -> (X, Y)` virtual memory page store with SHA-256 verification and `N5` WAL writes."""

    def __init__(self, records: Iterable[ATXYRecord | dict[str, Any]] = ()) -> None:
        self._directory: dict[int, dict[int, tuple[ATXYRecord, str]]] = {}
        self._aliases: dict[str, int | None] = {}
        self._wal_frames: list[bytes] = []
        for rec in records:
            self.put(rec, record_wal=False)

    @staticmethod
    def _normalize_alias(text: str) -> str:
        return unicodedata.normalize("NFC", text).strip().casefold()

    def _coerce_record(self, rec: ATXYRecord | dict[str, Any]) -> ATXYRecord:
        if isinstance(rec, ATXYRecord):
            return rec
        return ATXYRecord(
            address=str(parse_atxy(str(rec["address"]))),
            kind=str(rec.get("kind", "concept")),
            payload=str(rec["payload"]),
            summary=str(rec.get("summary", "")),
            up=tuple(str(parse_atxy(x)) for x in rec.get("up", ())),
            down=tuple(str(parse_atxy(x)) for x in rec.get("down", ())),
            related=tuple(str(parse_atxy(x)) for x in rec.get("related", ())),
            rels={
                str(k): tuple(str(parse_atxy(x)) for x in v)
                for k, v in (rec.get("rels") or {}).items()
            },
            fields={str(k): int(v) for k, v in (rec.get("fields") or {}).items()},
            version=int(rec.get("version", 1)),
            schema_version=int(rec.get("schema_version", 1)),
            provenance=tuple(str(x) for x in rec.get("provenance", ())),
        )

    def put(self, rec: ATXYRecord | dict[str, Any], *, record_wal: bool = False) -> ATXYRecord:
        canonical = self._coerce_record(rec)
        addr = canonical.parsed_address
        dig = canonical.digest()
        page = self._directory.setdefault(addr.directory_index, {})
        page[addr.cell] = (canonical, dig)

        norm = self._normalize_alias(canonical.payload)
        if norm:
            if norm not in self._aliases:
                self._aliases[norm] = addr.packed
            elif self._aliases[norm] != addr.packed:
                self._aliases[norm] = None  # Ambiguous alias excluded (fail-closed)

        if record_wal:
            payload_bytes = json.dumps(canonical.canonical_dict(), sort_keys=True).encode("utf-8")
            crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
            frame = struct.pack("<II", crc, len(payload_bytes)) + payload_bytes
            self._wal_frames.append(frame)
        return canonical

    def commit_write(
        self,
        rec: ATXYRecord | dict[str, Any],
        *,
        wal_path: str | Path | None = None,
    ) -> ATXYRecord:
        """`elem` N5 durable append-only versioned write with CRC32 WAL framing."""
        canonical = self._coerce_record(rec)
        existing = self.lookup(canonical.address)
        if existing is not None and canonical.version <= existing.version:
            canonical = ATXYRecord(
                **{**canonical.__dict__, "version": existing.version + 1}
            )
        committed = self.put(canonical, record_wal=True)
        if wal_path is not None:
            p = Path(wal_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("ab") as fh:
                fh.write(self._wal_frames[-1])
                fh.flush()
        return committed

    def load_wal(self, wal_path: str | Path) -> int:
        """Replay a framed CRC32 write-ahead log (`N5` session boundary recovery). Fails closed on corruption."""
        raw = Path(wal_path).read_bytes()
        off = 0
        applied = 0
        while off < len(raw):
            if off + 8 > len(raw):
                raise ValueError("Torn WAL header frame detected")
            crc_expected, length = struct.unpack_from("<II", raw, off)
            off += 8
            if off + length > len(raw):
                raise ValueError("Torn WAL payload frame detected")
            body = raw[off : off + length]
            off += length
            crc_actual = zlib.crc32(body) & 0xFFFFFFFF
            if crc_actual != crc_expected:
                raise ValueError("WAL frame CRC32 mismatch (corruption)")
            data = json.loads(body.decode("utf-8"))
            self.put(data, record_wal=True)
            applied += 1
        return applied

    def lookup(self, addr: str | int | ATXYAddress) -> ATXYRecord | None:
        """O(1) verified two-level directory/page lookup. Fails closed on digest or relocation mismatch."""
        a = (
            addr
            if isinstance(addr, ATXYAddress)
            else (unpack_atxy(addr) if isinstance(addr, int) else parse_atxy(addr))
        )
        page = self._directory.get(a.directory_index)
        if page is None:
            return None
        entry = page.get(a.cell)
        if entry is None:
            return None
        record, stored_digest = entry
        if record.parsed_address.packed != a.packed:
            raise ValueError(f"Relocation error at {a}: holds {record.address}")
        if record.digest() != stored_digest:
            raise ValueError(f"Digest corruption at {record.address}")
        return record

    def resolve(self, surface_text: str) -> ATXYRecord | None:
        """Exact deterministic reverse alias lookup (`surface_text -> ATXYRecord`)."""
        packed = self._aliases.get(self._normalize_alias(surface_text))
        if packed is None:
            return None
        return self.lookup(packed)

    def delete(self, addr: str | int | ATXYAddress) -> bool:
        """Remove an `A.TXY` cell from its page directory and clean up its reverse alias."""
        a = (
            addr
            if isinstance(addr, ATXYAddress)
            else (unpack_atxy(addr) if isinstance(addr, int) else parse_atxy(addr))
        )
        page = self._directory.get(a.directory_index)
        if page is None or a.cell not in page:
            return False
        rec, _ = page.pop(a.cell)
        norm = self._normalize_alias(rec.payload)
        if norm and self._aliases.get(norm) == a.packed:
            self._aliases.pop(norm, None)
        return True

    def records(self) -> list[ATXYRecord]:
        out: list[ATXYRecord] = []
        for di in sorted(self._directory):
            page = self._directory[di]
            for cell in sorted(page):
                out.append(page[cell][0])
        return out

    def __len__(self) -> int:
        return sum(len(p) for p in self._directory.values())


class ATXYPointerExecutor:
    """`elem` N6 sequential single-register pointer program executor over `ATXYPageStore`."""

    def __init__(self, store: ATXYPageStore) -> None:
        self.store = store
        self.register: Any = None
        self.register_kind: str = "empty"
        self.audit: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.register = None
        self.register_kind = "empty"
        self.audit.clear()

    def step(self, op: str, operand: Any = "REG", arg: Any = None) -> tuple[str, Any]:
        op_u = op.upper()
        if op_u not in ATXY_OPS:
            raise ValueError(f"Unsupported pointer op: {op!r}")

        src = self.register if operand == "REG" else operand

        if op_u == "READ":
            rec = self.store.lookup(src)
            if rec is None:
                raise KeyError(f"READ miss at {src!r}")
            self.register = rec.payload
            self.register_kind = "value"
        elif op_u in ("UP", "DOWN"):
            rec = self.store.lookup(src)
            if rec is None:
                raise KeyError(f"{op_u} miss at {src!r}")
            links = rec.up if op_u == "UP" else rec.down
            idx = int(arg or 0)
            if idx >= len(links):
                raise IndexError(f"No {op_u}[{idx}] link on {rec.address}")
            self.register = links[idx]
            self.register_kind = "addr"
        elif op_u == "FOLLOW":
            rec = self.store.lookup(src)
            if rec is None:
                raise KeyError(f"FOLLOW miss at {src!r}")
            rel_name = str(arg or "RELATED").upper()
            targets = rec.related if rel_name == "RELATED" else rec.rels.get(rel_name, ())
            if not targets:
                raise KeyError(f"No {rel_name} edge from {rec.address}")
            self.register = targets[0]
            self.register_kind = "addr"
        elif op_u == "FOLLOW_SET":
            addrs = src if isinstance(src, (list, tuple, frozenset, set)) else [src]
            rel_name = str(arg or "RELATED").upper()
            out_set: list[str] = []
            for a_item in addrs:
                rec = self.store.lookup(a_item)
                if rec is None:
                    continue
                edges = (
                    rec.up
                    if rel_name == "UP"
                    else (
                        rec.down
                        if rel_name == "DOWN"
                        else (rec.related if rel_name == "RELATED" else rec.rels.get(rel_name, ()))
                    )
                )
                for dst in edges:
                    if dst not in out_set:
                        out_set.append(dst)
            self.register = tuple(out_set)
            self.register_kind = "set"
        elif op_u == "PROJECT":
            rec = self.store.lookup(src)
            if rec is None:
                raise KeyError(f"PROJECT miss at {src!r}")
            field_name = str(arg)
            if field_name not in rec.fields:
                raise KeyError(f"Missing field {field_name!r} at {rec.address}")
            self.register = rec.fields[field_name]
            self.register_kind = "scalar"
        elif op_u == "FIND":
            plane_filter = arg.get("plane") if isinstance(arg, dict) else None
            kind_filter = arg.get("kind") if isinstance(arg, dict) else None
            matches = []
            for r in self.store.records():
                if plane_filter is not None and r.parsed_address.A != int(plane_filter):
                    continue
                if kind_filter is not None and r.kind != str(kind_filter):
                    continue
                matches.append(r.address)
            self.register = tuple(matches)
            self.register_kind = "set"
        elif op_u == "FILTER":
            addrs = src if isinstance(src, (list, tuple, frozenset, set)) else []
            field_name = str(arg["field"])
            op_cmp = str(arg.get("op", "EQ")).upper()
            val = int(arg["val"])
            kept = []
            for a_item in addrs:
                rec = self.store.lookup(a_item)
                if rec is None or field_name not in rec.fields:
                    continue
                fv = rec.fields[field_name]
                if (op_cmp == "EQ" and fv == val) or (op_cmp == "LT" and fv < val) or (op_cmp == "GT" and fv > val):
                    kept.append(rec.address)
            self.register = tuple(kept)
            self.register_kind = "set"
        elif op_u == "COUNT":
            self.register = len(src) if isinstance(src, (list, tuple, frozenset, set)) else 0
            self.register_kind = "scalar"
        elif op_u == "COMPARE":
            self.register = bool(self.register == arg)
            self.register_kind = "bool"

        self.audit.append(
            {
                "op": op_u,
                "operand": operand,
                "arg": arg,
                "result_kind": self.register_kind,
                "result": self.register,
            }
        )
        return self.register_kind, self.register

    def run(self, program: Sequence[tuple[str, Any, Any]]) -> tuple[str, Any]:
        self.reset()
        for op, operand, arg in program:
            self.step(op, operand, arg)
        return self.register_kind, self.register


def _code_tokens_and_stems(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens, snake/camel splits, and 4-char stems for code routing."""
    # Split camelCase into words before lowercasing
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    raw = re.findall(r"[a-z0-9]+", expanded.lower())
    out: set[str] = set()
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "when", "where",
        "which", "what", "how", "does", "select", "exact", "record", "matching", "request",
        "self", "none", "true", "false", "return", "def", "class",
    }
    for tok in raw:
        if len(tok) >= 2 and tok not in stop:
            out.add(tok)
            if len(tok) >= 5:
                out.add(f"stem:{tok[:4]}")
    return out


def _truncate_to_token_budget(text: str, vocab_size: int, max_tokens: int) -> str:
    """Truncate `text` so its tokenized representation (plus BOS/EOS) fits strictly within `max_tokens`."""
    budget = max(8, max_tokens - 4)
    if vocab_size <= 260:
        return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
    tok = get_tokenizer(vocab_size)
    ids = tok.encode(text, add_bos=False, add_eos=False)[:budget]
    return tok.decode(ids)


class COEAddressRouter:
    """Resolves `elem`'s `N3b-primary` natural-language routing bottleneck via `MCL4-COE`.

    Two-stage exact routing over `ATXYPageStore`:
    1. Stage 1 (`Association` + lexical/morpheme stem sketch): Filters all occupied `A.TXY` records
       into the top-$K$ candidate cells.
    2. Stage 2 (`Decision`, shared-state `MCL4-CED` cross-attention + `AttentiveMatchedFilterPool`):
       Scores the $K$ candidate records against the natural-language query in a single softmax
       with bitwise candidate-permutation invariance.
    """

    def __init__(
        self,
        store: ATXYPageStore,
        *,
        max_seq_len: int = 256,
    ) -> None:
        self.store = store
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def route_query(
        self,
        coe: CognitiveOperatorEnsemble,
        query_text: str,
        *,
        top_k_candidates: int = 8,
        plane_filter: int | None = None,
        recurrence_steps: int | None = None,
        depth_ponder: bool | None = None,
        margin_halt_threshold: float | None = None,
    ) -> dict[str, Any]:
        all_recs = [
            r
            for r in self.store.records()
            if plane_filter is None or r.parsed_address.A == plane_filter
        ]
        if not all_recs:
            raise ValueError("No records available in ATXYPageStore for routing")

        device = next(coe.parameters()).device
        vocab_size = int(getattr(coe.backbone.config, "vocab_size", 260))
        k = min(top_k_candidates, len(all_recs))

        # If store has more than top_k_candidates records, pre-filter via IDF-weighted token/stem sketch
        if len(all_recs) > k:
            q_tokens = _code_tokens_and_stems(query_text)
            doc_token_sets = [
                _code_tokens_and_stems(f"{r.address} {r.kind} {r.payload} {r.summary}")
                for r in all_recs
            ]
            n_docs = len(all_recs)
            df: dict[str, int] = {}
            for dset in doc_token_sets:
                for tok in dset:
                    df[tok] = df.get(tok, 0) + 1

            sketch_scores = []
            q_lower = query_text.lower()
            for r, d_tokens in zip(all_recs, doc_token_sets):
                shared = q_tokens & d_tokens
                score = sum(math.log(1.0 + n_docs / max(1, df.get(t, 1))) for t in shared)
                # Bonus if exact symbol or file name appears in query
                sym_part = r.payload.split(":")[-1].lower()
                if len(sym_part) >= 3 and sym_part in q_lower:
                    score += 4.0
                sketch_scores.append(score)
            ranked_idx = sorted(
                range(len(all_recs)),
                key=lambda i: sketch_scores[i],
                reverse=True,
            )[:k]
            cand_recs = [all_recs[i] for i in ranked_idx]
            cand_sketch = [sketch_scores[i] for i in ranked_idx]
        else:
            cand_recs = all_recs
            cand_sketch = [0.0 for _ in cand_recs]

        max_len = min(self.max_seq_len, int(coe.backbone.config.max_seq_len))
        raw_ctx = f"#U$# {query_text}\n#Q$# Select the exact A.TXY record matching the request."
        ctx_str = _truncate_to_token_budget(raw_ctx, vocab_size, max_len)
        desc_strs = tuple(
            _truncate_to_token_budget(r.to_candidate_description(), vocab_size, max_len)
            for r in cand_recs
        )
        example = DecisionExample(
            question_id="atxy_route",
            kind="choice",
            context=ctx_str,
            descriptions=desc_strs,
            values=tuple(r.address for r in cand_recs),
            target=0,
        )
        batch = collate_decisions(
            [example],
            max_length=max_len,
            device=device,
            vocab_size=vocab_size,
        )
        dec_out = coe.forward_decision(
            batch["context_ids"],
            batch["context_lengths"],
            batch["option_ids"],
            batch["option_lengths"],
            recurrence_steps=recurrence_steps,
            depth_ponder=depth_ponder,
            margin_halt_threshold=margin_halt_threshold,
        )
        dec_sim = dec_out.similarities[0].float()
        if any(s > 0.0 for s in cand_sketch):
            sk_t = torch.tensor(cand_sketch, dtype=torch.float32, device=device)
            sk_z = (sk_t - sk_t.mean()) / sk_t.std(unbiased=False).clamp_min(1e-5)
            scale_eff = (coe._scale().detach().float() * 0.40).clamp(1.2, 4.0)
            joint_logits = scale_eff * (0.65 * dec_sim + 0.45 * sk_z)
            probs = joint_logits.softmax(dim=-1)
        else:
            probs = dec_out.probabilities[0]
        best_idx = int(probs.argmax(dim=-1).item())
        chosen_rec = cand_recs[best_idx]
        return {
            "address": chosen_rec.address,
            "record": chosen_rec,
            "confidence": float(probs[best_idx].item()),
            "candidates": [r.address for r in cand_recs],
            "probabilities": [round(float(p), 5) for p in probs.cpu().tolist()],
        }


def _summarize_ast_symbol(node: ast.AST, rel: str) -> str:
    """Build a compact semantic summary of an AST function or class for `ATXYRecord`."""
    kind = "class" if isinstance(node, ast.ClassDef) else "function"
    doc_lines = [ln.strip() for ln in (ast.get_docstring(node) or "").strip().splitlines() if ln.strip()]
    doc_joined = (
        " ".join(doc_lines[:3])[:220]
        if doc_lines
        else f"{kind} {getattr(node, 'name', '')} in {Path(rel).name}"
    )

    sig_str = ""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arg_names = [a.arg for a in node.args.args if a.arg != "self"][:6]
        kw_names = [a.arg for a in node.args.kwonlyargs][:4]
        all_args = arg_names + [f"*{a}" for a in kw_names]
        sig_str = f"({', '.join(all_args)})"
    elif isinstance(node, ast.ClassDef):
        methods = [
            n.name
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("__")
        ][:6]
        if methods:
            sig_str = f"[methods: {', '.join(methods)}]"

    skip_ids = {
        "self", "cls", "None", "True", "False", "int", "float", "str", "bool",
        "list", "dict", "tuple", "set", "Any", "Tensor", "torch", "range", "len",
        "min", "max", "sum", "zip", "enumerate", "isinstance", "getattr", "ValueError",
    }
    body_terms: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.attr, str):
            if sub.attr not in skip_ids and not sub.attr.startswith("_") and sub.attr not in body_terms:
                body_terms.append(sub.attr)
        elif isinstance(sub, ast.Name) and isinstance(sub.id, str):
            if sub.id not in skip_ids and len(sub.id) >= 4 and not sub.id.startswith("_") and sub.id not in body_terms:
                body_terms.append(sub.id)
        if len(body_terms) >= 10:
            break

    parts = [p for p in (sig_str, doc_joined) if p]
    if body_terms:
        parts.append(f"refs: {', '.join(body_terms[:6])}")
    return " | ".join(parts)


def index_python_repo_to_atxy(
    repo_root: str | Path,
    *,
    subsystems: Sequence[str] = ("src", "scripts", "tests"),
) -> ATXYPageStore:
    """Compile a Python repository's AST into an `ATXYPageStore` (`A=3` subsystem, `A=2` file, `A=1` symbol)."""
    root = Path(repo_root).expanduser().resolve()
    records: list[ATXYRecord] = []
    file_coords: dict[str, tuple[int, int]] = {}
    sym_index: dict[str, list[str]] = {}
    parsed_files: dict[str, tuple[list[tuple[str, str, str, ast.AST]], list[str]]] = {}

    for t_idx, sub in enumerate(subsystems[:ATXY_BASE]):
        sub_dir = root / sub
        if not sub_dir.is_dir():
            continue
        py_files = sorted(p for p in sub_dir.rglob("*.py") if "__pycache__" not in p.parts)[
            :ATXY_BASE
        ]
        for x_idx, py_path in enumerate(py_files):
            rel = py_path.relative_to(root).as_posix()
            file_coords[rel] = (t_idx, x_idx)
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                parsed_files[rel] = ([], [])
                continue

            syms: list[tuple[str, str, str, ast.AST]] = []
            imports: list[str] = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    summary = _summarize_ast_symbol(node, rel)
                    syms.append((node.name, kind, summary, node))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)

            # Prioritize public classes and functions before private underscore helpers
            syms.sort(key=lambda item: item[0].startswith("_"))
            syms = syms[:ATXY_BASE]
            parsed_files[rel] = (syms, imports)
            for y_idx, (s_name, _kind, _sum, _nd) in enumerate(syms):
                sym_addr = str(ATXYAddress(1, t_idx, x_idx, y_idx))
                sym_index.setdefault(s_name, []).append(sym_addr)

    for t_idx, sub in enumerate(subsystems[:ATXY_BASE]):
        sub_rels = [r for r, (ti, _) in file_coords.items() if ti == t_idx]
        if not sub_rels:
            continue
        sub_addr = str(ATXYAddress(3, t_idx, 0, 0))
        file_addrs = tuple(str(ATXYAddress(2, t_idx, file_coords[r][1], 0)) for r in sub_rels)
        records.append(
            ATXYRecord(
                address=sub_addr,
                kind="subsystem",
                payload=sub,
                summary=f"Subsystem {sub} ({len(sub_rels)} files)",
                down=file_addrs,
                fields={"file_count": len(sub_rels)},
                provenance=("ast:subsystem",),
            )
        )
        for rel in sub_rels:
            _, x_idx = file_coords[rel]
            f_addr = str(ATXYAddress(2, t_idx, x_idx, 0))
            syms, _ = parsed_files.get(rel, ([], []))
            sym_addrs = tuple(str(ATXYAddress(1, t_idx, x_idx, y_idx)) for y_idx in range(len(syms)))
            records.append(
                ATXYRecord(
                    address=f_addr,
                    kind="file",
                    payload=rel,
                    summary=f"Module {rel} ({len(syms)} top-level symbols)",
                    up=(sub_addr,),
                    down=sym_addrs,
                    fields={"symbol_count": len(syms)},
                    provenance=("ast:file",),
                )
            )
            for y_idx, (s_name, s_kind, s_sum, s_node) in enumerate(syms):
                s_addr = str(ATXYAddress(1, t_idx, x_idx, y_idx))
                callees: list[str] = []
                for n in ast.walk(s_node):
                    if isinstance(n, ast.Call):
                        cname = (
                            n.func.id
                            if isinstance(n.func, ast.Name)
                            else (n.func.attr if isinstance(n.func, ast.Attribute) else None)
                        )
                        if cname and cname != s_name and len(sym_index.get(cname, ())) == 1:
                            tgt = sym_index[cname][0]
                            if tgt not in callees:
                                callees.append(tgt)
                records.append(
                    ATXYRecord(
                        address=s_addr,
                        kind=s_kind,
                        payload=f"{rel}:{s_name}",
                        summary=s_sum,
                        up=(f_addr,),
                        rels={"CALLS": tuple(callees)} if callees else {},
                        fields={"lineno": int(getattr(s_node, "lineno", 1))},
                        provenance=("ast:symbol",),
                    )
                )

    return ATXYPageStore(records)


# ---------------------------------------------------------------------------
# Pre-Prompt Bounded Codebase RL Warmup + TurboQuant / A.TXY Session Engine
# ---------------------------------------------------------------------------


@dataclass
class CodebaseWarmupReport:
    """Telemetry from a bounded pre-prompt codebase RL exploration & `TurboQuant` indexing session."""

    repo_root: str
    total_files: int
    total_symbols: int
    total_repo_bytes: int
    total_repo_tokens: int
    uncompressed_state_bytes: int
    turboquant_state_bytes: int
    compression_ratio: float
    mean_turboquant_cosine: float
    warmup_steps: int
    warmup_elapsed_sec: float
    trainable_params: int
    frozen_backbone_params: int
    pre_route_acc: float
    post_route_acc: float
    pre_route_conf: float
    post_route_conf: float
    pre_verify_acc: float
    post_verify_acc: float
    pre_verify_conf: float
    post_verify_conf: float
    pre_loss: float
    post_loss: float
    step_losses: list[float] = field(default_factory=list)


@dataclass
class MultiFileSynthesisResult:
    """Result of Tier-4 (`A.TXY` + `CALLS`) + Tier-2 (`TurboQuant`) + Tier-3 (`ContextLMCompressor`) multi-file synthesis."""

    query: str
    primary_address: str
    primary_symbol: str
    primary_kind: str
    primary_summary: str
    primary_confidence: float
    primary_source: str
    focal_statement: str
    focal_invariant: str
    callees: tuple[dict[str, str], ...]
    participating_files: tuple[str, ...]
    participating_file_addresses: tuple[str, ...]
    num_files_folded: int
    raw_source_tokens: int
    encoded_page_tokens: int
    compressed_gist_tokens: int
    uncompressed_page_bytes: int
    turboquant_page_bytes: int
    token_folding_ratio: float
    byte_compression_ratio: float
    compressed_state: EncodedState
    multifile_latent_prior: Tensor


@dataclass
class DirtyPageRefreshReport:
    """Telemetry from an incremental dirty-file AST + `A.TXY` WAL + `TurboQuant` page refresh."""

    dirty_files: tuple[str, ...]
    updated_file_addresses: tuple[str, ...]
    updated_symbol_addresses: tuple[str, ...]
    added_symbols: tuple[str, ...]
    modified_symbols: tuple[str, ...]
    removed_symbols: tuple[str, ...]
    wal_frames_appended: int
    recompressed_pages: int
    mean_refresh_cosine: float
    micro_warmup_steps: int
    micro_warmup_loss: float
    elapsed_ms: float


@dataclass
class CodebaseSessionState:
    """Session-scoped virtual memory (`A.TXY` + `TurboQuant`) and reversible fast-weight snapshot."""

    repo_root: str
    store: ATXYPageStore
    router: COEAddressRouter
    pointer_executor: ATXYPointerExecutor
    symbol_sources: dict[str, str]
    file_turboquant_pages: dict[str, CompressedEncodedState]
    file_latent_priors: dict[str, Tensor]
    total_repo_files: int
    total_repo_symbols: int
    total_repo_bytes: int
    total_repo_tokens: int
    uncompressed_state_bytes: int
    compressed_state_bytes: int
    mean_turboquant_cosine: float
    base_operator_state: dict[str, Tensor]
    train_route_tasks: list[DecisionExample] = field(default_factory=list)
    eval_route_tasks: list[DecisionExample] = field(default_factory=list)
    train_verify_tasks: list[DecisionExample] = field(default_factory=list)
    eval_verify_tasks: list[DecisionExample] = field(default_factory=list)
    context_compressor: ContextLMCompressor | None = None
    base_compressor_state: dict[str, Tensor] = field(default_factory=dict)
    file_token_counts: dict[str, int] = field(default_factory=dict)
    file_digests: dict[str, str] = field(default_factory=dict)
    file_addresses: dict[str, str] = field(default_factory=dict)
    subsystems: tuple[str, ...] = ("src",)

    @property
    def compression_ratio(self) -> float:
        return float(self.uncompressed_state_bytes) / max(1.0, float(self.compressed_state_bytes))

    def restore_base_weights(self, coe: CognitiveOperatorEnsemble) -> None:
        """Instantly restore `coe`'s operator & pooling heads (and `context_compressor`) to their pre-session weights."""
        device = next(coe.parameters()).device
        with torch.no_grad():
            param_map = dict(coe.named_parameters())
            for name, saved_tensor in self.base_operator_state.items():
                if name in param_map:
                    param_map[name].copy_(saved_tensor.to(device=device, dtype=param_map[name].dtype))
            if self.context_compressor is not None and self.base_compressor_state:
                comp_map = dict(self.context_compressor.named_parameters())
                for name, saved_tensor in self.base_compressor_state.items():
                    if name in comp_map:
                        comp_map[name].copy_(
                            saved_tensor.to(device=comp_map[name].device, dtype=comp_map[name].dtype)
                        )


def _extract_focal_line_and_context(snippet: str) -> tuple[str, str]:
    """Split a function snippet into shared setup context and the focal executable invariant line(s)."""
    raw_lines = [ln for ln in snippet.splitlines() if ln.strip()]
    if not raw_lines:
        return snippet, snippet

    # Prefer the last non-trivial line containing an operator, return, assignment, or call
    op_markers = ("return ", " = ", " == ", " != ", " + ", " - ", " * ", " / ", " <= ", " >= ", " < ", " > ")
    focal_idx = len(raw_lines) - 1
    for idx in range(len(raw_lines) - 1, 0, -1):
        s = raw_lines[idx].strip()
        if s.startswith(("#", '"""', "'''", "@")):
            continue
        if any(m in s for m in op_markers):
            focal_idx = idx
            break

    focal_line = raw_lines[focal_idx].strip()
    ctx_lines = [
         raw_lines[i] if i != focal_idx else "    # <<< FOCAL INVARIANT LINE >>>"
        for i in range(len(raw_lines))
    ]
    return "\n".join(ctx_lines), focal_line


def _summarize_code_invariants(focal_line: str) -> str:
    """Extract a deterministic structural/AST invariant summary from a focal code statement."""
    patterns = (
        "+", "-", "*", "/", "==", "!=", "<=", ">=", "<", ">",
        "is not None", "is None", "True", "False", "dim=-1", "dim=0",
        "argmax", "argmin", "max", "min", "sum", "mean", "None",
    )
    found = [p for p in patterns if p in focal_line]
    ops_str = ", ".join(found[:6]) if found else "expr"
    kind = "return" if focal_line.startswith("return ") else ("assign" if "=" in focal_line else "call")
    return f"stmt={kind} | tokens=[{ops_str}]"


def _mutate_focal_line(focal_line: str, *, mode: str = "train") -> list[str]:
    """Generate 3 realistic buggy mutations of a focal Python statement."""
    rules: list[tuple[str, str]] = [
        (" + ", " - "),
        (" - ", " + "),
        (" * ", " / "),
        (" / ", " * "),
        (" == ", " != "),
        (" != ", " == "),
        (" <= ", " > "),
        (" >= ", " < "),
        (" < ", " > "),
        (" > ", " < "),
        ("is not None", "is None"),
        ("is None", "is not None"),
        ("True", "False"),
        ("False", "True"),
        ("dim=-1", "dim=0"),
        ("dim=0", "dim=-1"),
        ("argmax", "argmin"),
        ("max(", "min("),
        ("min(", "max("),
        ("sum(", "mean("),
        ("1", "0"),
        ("0", "1"),
    ]
    mutations: list[str] = []
    for old_pat, new_pat in rules:
        if old_pat in focal_line:
            mut = focal_line.replace(old_pat, new_pat, 1)
            if mut != focal_line and mut not in mutations:
                mutations.append(mut)

    # Fallback realistic corruptions of the focal statement
    fallbacks_train = [
        "return None  # dropped return value",
        f"not ({focal_line[7:]})" if focal_line.startswith("return ") else f"{focal_line}  # unverified",
        "raise ValueError('unexpected state in focal invariant')",
    ]
    fallbacks_eval = [
        "return 0  # zeroed fallback",
        "pass  # no-op mutation",
        "return False  # inverted boolean guard",
    ]
    fb_list = fallbacks_train if mode == "train" else fallbacks_eval
    for fb in fb_list:
        if fb != focal_line and fb not in mutations:
            mutations.append(fb)

    if mode == "eval" and len(mutations) >= 4:
        # Select distinct mutations for eval when multiple single-token mutations exist
        return mutations[1:4]
    return mutations[:3]


class CodebaseWarmupEngine:
    """Pre-Prompt Bounded Codebase RL Warmup & `TurboQuant` + `A.TXY` Context Engine.

    At the start of a coding/debugging session, `CodebaseWarmupEngine`:
    1. Compiles the repository's AST into an `ATXYPageStore` (`A=3` subsystems, `A=2` files,
       `A=1` functions/classes + `CALLS` graph).
    2. Encodes and compresses each repository module into `TurboQuant` 4-bit `CompressedEncodedState`
       pages (`4.2x` memory reduction) and extracts `384-d` `Abstraction` latent priors.
    3. Synthesizes AST-verified `DecisionExample` tasks from the repository:
       - **Symbol & Call-Graph Localization (`route_tasks`)**: routing natural-language architecture
         and debugging queries to exact `A.TXY` symbol records against hard-negative repo symbols.
       - **Real-vs-Mutated Code Bug Discrimination (`verify_tasks`)**: distinguishing authentic
         repository function implementations from mutated buggy variants (`+` vs `-`, `<` vs `>`,
         boundary flips, axis swaps).
    4. Runs a **bounded pre-prompt RL/COE warmup session** (`warmup_session`) that freezes
       `coe.backbone` (preserving 100% of generation fluency) and adapts only the session-scoped
       `COE` operator & `AttentiveMatchedFilterPool` heads (`~600K` params, resettable in `<1ms`).
    """

    def __init__(
        self,
        *,
        subsystems: Sequence[str] = ("src",),
        max_seq_len: int = 192,
        max_symbol_lines: int = 18,
        num_route_options: int = 6,
    ) -> None:
        self.subsystems = tuple(subsystems)
        self.max_seq_len = max_seq_len
        self.max_symbol_lines = max_symbol_lines
        self.num_route_options = max(4, num_route_options)

    @staticmethod
    def _operator_parameter_names(coe: CognitiveOperatorEnsemble) -> list[str]:
        """Return parameter names belonging to the lightweight `COE` operator & pooling heads."""
        return [
            name
            for name, _ in coe.named_parameters()
            if not name.startswith("backbone.")
        ]

    @torch.no_grad()
    def index_repository(
        self,
        coe: CognitiveOperatorEnsemble,
        repo_root: str | Path,
        *,
        max_tasks_per_family: int = 256,
    ) -> CodebaseSessionState:
        """Index `repo_root` into `ATXYPageStore` + `TurboQuant` pages and synthesize AST self-RL tasks."""
        root = Path(repo_root).expanduser().resolve()
        store = index_python_repo_to_atxy(root, subsystems=self.subsystems)
        router = COEAddressRouter(store, max_seq_len=self.max_seq_len)
        pointer_executor = ATXYPointerExecutor(store)

        device = next(coe.parameters()).device
        vocab_size = int(getattr(coe.backbone.config, "vocab_size", 260))
        tokenizer = get_tokenizer(vocab_size)
        max_len = min(self.max_seq_len, int(coe.backbone.config.max_seq_len))

        # Snapshot base operator weights on CPU for instant session reset
        op_names = set(self._operator_parameter_names(coe))
        base_operator_state = {
            name: param.detach().cpu().clone()
            for name, param in coe.named_parameters()
            if name in op_names
        }

        symbol_sources: dict[str, str] = {}
        file_turboquant_pages: dict[str, CompressedEncodedState] = {}
        file_latent_priors: dict[str, Tensor] = {}
        file_token_counts: dict[str, int] = {}
        file_digests: dict[str, str] = {}
        file_addresses: dict[str, str] = {}

        d_model = int(coe.backbone.config.d_model)
        n_gist_per_chunk = int(getattr(coe.config, "gist_tokens_per_chunk", 6))
        context_compressor = ContextLMCompressor(
            d_model=d_model,
            chunk_size=max_len,
            gist_tokens_per_chunk=n_gist_per_chunk,
        ).to(device=device)
        if hasattr(coe, "gist_queries") and coe.gist_queries.shape == context_compressor.gist_queries.shape:
            with torch.no_grad():
                context_compressor.gist_queries.copy_(coe.gist_queries.detach().to(device=device))
                context_compressor.gist_norm.weight.copy_(coe.gist_norm.weight.detach().to(device=device))
                context_compressor.gist_norm.bias.copy_(coe.gist_norm.bias.detach().to(device=device))
        base_compressor_state = {
            name: param.detach().cpu().clone()
            for name, param in context_compressor.named_parameters()
        }

        total_repo_bytes = 0
        total_repo_tokens = 0
        uncompressed_state_bytes = 0
        compressed_state_bytes = 0
        tq_cosines: list[float] = []

        file_recs = [r for r in store.records() if r.parsed_address.A == 2]
        sym_recs = [r for r in store.records() if r.parsed_address.A == 1]

        # Extract symbol source snippets & compress module states with TurboQuant
        for idx, frec in enumerate(file_recs):
            fpath = root / frec.payload
            if not fpath.is_file():
                continue
            src_text = fpath.read_text(encoding="utf-8", errors="replace")
            src_bytes = src_text.encode("utf-8")
            src_lines = src_text.splitlines()
            total_repo_bytes += len(src_bytes)
            file_digests[frec.payload] = hashlib.sha256(src_bytes).hexdigest()
            file_addresses[frec.payload] = frec.address
            toks = tokenizer.encode(src_text, add_bos=False, add_eos=False)
            total_repo_tokens += len(toks)
            file_token_counts[frec.address] = len(toks)

            # Parse AST to extract clean per-symbol source snippets
            try:
                tree = ast.parse(src_text)
                ast_map = {
                    node.name: node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                }
            except SyntaxError:
                ast_map = {}

            module_summary_lines = [f"# Module {frec.payload}"]
            for s_addr in frec.down:
                srec = store.lookup(s_addr)
                if srec is None:
                    continue
                sym_name = srec.payload.split(":")[-1]
                node = ast_map.get(sym_name)
                if node is not None:
                    start_ln = max(0, int(getattr(node, "lineno", 1)) - 1)
                    end_ln = min(
                        len(src_lines),
                        int(getattr(node, "end_lineno", start_ln + self.max_symbol_lines)),
                        start_ln + self.max_symbol_lines,
                    )
                    snippet = "\n".join(src_lines[start_ln:end_ln]).strip()
                else:
                    snippet = f"def {sym_name}(): # {srec.summary}"
                symbol_sources[srec.address] = snippet
                module_summary_lines.append(f"{sym_name}: {srec.summary}")

            # Encode module summary chunk through lower-CED & compress via 4-bit TurboQuant
            mod_doc = "\n".join(module_summary_lines)
            mod_ids = tokenizer.encode(mod_doc, add_bos=True, add_eos=True)[:max_len]
            inp_t = torch.tensor([mod_ids], dtype=torch.long, device=device)
            len_t = torch.tensor([len(mod_ids)], dtype=torch.long, device=device)
            valid_mask = torch.ones((1, len(mod_ids)), dtype=torch.bool, device=device)
            enc_state = coe.backbone.encode_state(inp_t, valid_mask, causal=True)
            comp_state = compress_encoded_state(enc_state, seed=1337 + idx * 17)
            deq_state = comp_state.decompress()

            cos = float(
                F.cosine_similarity(
                    enc_state.hidden.float().reshape(-1, enc_state.hidden.shape[-1]),
                    deq_state.hidden.float().reshape(-1, deq_state.hidden.shape[-1]),
                    dim=-1,
                )
                .mean()
                .item()
            )
            tq_cosines.append(cos)
            uncompressed_state_bytes += comp_state.uncompressed_bytes
            compressed_state_bytes += comp_state.compressed_bytes
            file_turboquant_pages[frec.address] = comp_state

            z_abs = coe.compress_abstraction(inp_t, len_t, level="standard")[0].detach()
            file_latent_priors[frec.address] = z_abs

        # Synthesize AST-verified Train & Held-Out Evaluation Tasks
        train_route, eval_route, train_verify, eval_verify = self._build_codebase_tasks(
            store,
            symbol_sources,
            vocab_size=vocab_size,
            max_len=max_len,
            max_tasks=max_tasks_per_family,
        )

        return CodebaseSessionState(
            repo_root=str(root),
            store=store,
            router=router,
            pointer_executor=pointer_executor,
            symbol_sources=symbol_sources,
            file_turboquant_pages=file_turboquant_pages,
            file_latent_priors=file_latent_priors,
            total_repo_files=len(file_recs),
            total_repo_symbols=len(sym_recs),
            total_repo_bytes=total_repo_bytes,
            total_repo_tokens=total_repo_tokens,
            uncompressed_state_bytes=uncompressed_state_bytes,
            compressed_state_bytes=compressed_state_bytes,
            mean_turboquant_cosine=sum(tq_cosines) / max(1, len(tq_cosines)),
            base_operator_state=base_operator_state,
            train_route_tasks=train_route,
            eval_route_tasks=eval_route,
            train_verify_tasks=train_verify,
            eval_verify_tasks=eval_verify,
            context_compressor=context_compressor,
            base_compressor_state=base_compressor_state,
            file_token_counts=file_token_counts,
            file_digests=file_digests,
            file_addresses=file_addresses,
            subsystems=self.subsystems,
        )

    def _build_codebase_tasks(
        self,
        store: ATXYPageStore,
        symbol_sources: dict[str, str],
        *,
        vocab_size: int,
        max_len: int,
        max_tasks: int,
    ) -> tuple[
        list[DecisionExample],
        list[DecisionExample],
        list[DecisionExample],
        list[DecisionExample],
    ]:
        sym_recs = [r for r in store.records() if r.parsed_address.A == 1]

        train_route: list[DecisionExample] = []
        eval_route: list[DecisionExample] = []
        train_verify: list[DecisionExample] = []
        eval_verify: list[DecisionExample] = []

        if len(sym_recs) < 4:
            return train_route, eval_route, train_verify, eval_verify

        # Precompute token sets for selecting hard-negative distractors per symbol
        sym_toks = [
            _code_tokens_and_stems(f"{r.payload} {r.summary}")
            for r in sym_recs
        ]
        k_opts = min(self.num_route_options, len(sym_recs))

        # Stratify selection evenly across all repository modules so every file is explored
        n_sel = min(max_tasks, len(sym_recs))
        selected_indices = [int(i * len(sym_recs) / n_sel) for i in range(n_sel)]
        neg_counts: dict[int, int] = {j: 0 for j in selected_indices}

        for task_i, idx in enumerate(selected_indices):
            rec = sym_recs[idx]
            rel_path, sym_name = (
                rec.payload.split(":", 1) if ":" in rec.payload else (rec.payload, rec.payload)
            )
            mod_short = Path(rel_path).name

            # Rank hard-negative distractors within selected_indices with degree-balancing penalty
            # so every explored symbol appears equally often as positive and negative (zero prior bias)
            scored_neg = []
            for j in selected_indices:
                if j == idx:
                    continue
                other = sym_recs[j]
                same_mod = 1.5 if other.up == rec.up else 0.0
                ov = len(sym_toks[idx] & sym_toks[j]) / max(1, len(sym_toks[idx]))
                scored_neg.append((same_mod + ov - 0.65 * neg_counts[j], j))
            scored_neg.sort(key=lambda pair: pair[0], reverse=True)
            neg_indices = [j for _, j in scored_neg[: k_opts - 1]]
            for j in neg_indices:
                neg_counts[j] += 1

            # Deterministically place target at index (task_i % k_opts)
            target_pos = task_i % k_opts
            cand_list = [sym_recs[j] for j in neg_indices]
            cand_list.insert(target_pos, rec)
            desc_tuple = tuple(
                _truncate_to_token_budget(c.to_candidate_description(), vocab_size, max_len)
                for c in cand_list
            )
            val_tuple = tuple(c.address for c in cand_list)

            # Extract callees for call-graph context
            callee_names = []
            for cal_addr in rec.rels.get("CALLS", ()):
                crec = store.lookup(cal_addr)
                if crec is not None:
                    callee_names.append(crec.payload.split(":")[-1])
            call_hint = f" (calls {', '.join(callee_names[:3])})" if callee_names else ""

            redacted_summary = re.sub(
                re.escape(sym_name),
                f"the target {rec.kind}",
                rec.summary,
                flags=re.IGNORECASE,
            )

            # 1. Self-exploration training query (AST signature + summary + call-graph)
            train_q = (
                f"#U$# Explore {mod_short}: locate {rec.kind} `{sym_name}` — {rec.summary}{call_hint}.\n"
                f"#Q$# Select the exact A.TXY record matching the request."
            )
            train_route.append(
                DecisionExample(
                    question_id=f"route_train_{task_i}",
                    kind="choice",
                    context=_truncate_to_token_budget(train_q, vocab_size, max_len),
                    descriptions=desc_tuple,
                    values=val_tuple,
                    target=target_pos,
                )
            )

            # 2. Held-out natural-language developer debugging query (redacts direct symbol name where possible)
            eval_q = (
                f"#U$# Debugging `{mod_short}`: which {rec.kind} in `{rel_path}` is responsible for "
                f"{redacted_summary}{call_hint}?\n"
                f"#Q$# Select the exact A.TXY record matching the request."
            )
            eval_route.append(
                DecisionExample(
                    question_id=f"route_eval_{task_i}",
                    kind="choice",
                    context=_truncate_to_token_budget(eval_q, vocab_size, max_len),
                    descriptions=desc_tuple,
                    values=val_tuple,
                    target=target_pos,
                )
            )

            # 3. Real-vs-Mutated Code Bug Verification Task (shared setup in lower-CED, focal line in upper-CED)
            snippet = symbol_sources.get(rec.address, "")
            if len(snippet.splitlines()) >= 3:
                shared_setup, focal_line = _extract_focal_line_and_context(snippet)
                auth_inv = _summarize_code_invariants(focal_line)
                v_target = (task_i + 1) % 4

                # Train task: Mutation Set A on focal statement
                muts_train = _mutate_focal_line(focal_line, mode="train")
                v_cands_tr = list(muts_train[:3])
                v_cands_tr.insert(v_target, focal_line)
                v_descs_tr = tuple(
                    _truncate_to_token_budget(
                        f"#T$# [{sym_name}] {_summarize_code_invariants(stmt)}\n#A$# {stmt}",
                        vocab_size,
                        max_len,
                    )
                    for stmt in v_cands_tr
                )
                v_ctx_tr = (
                    f"#U$# Pre-session AST verification for `{sym_name}` in `{rel_path}` ({rec.summary} | expected {auth_inv}):\n"
                    f"{shared_setup}\n"
                    f"#Q$# Select the authentic bug-free focal statement for `{sym_name}`."
                )
                train_verify.append(
                    DecisionExample(
                        question_id=f"verify_train_{task_i}",
                        kind="choice",
                        context=_truncate_to_token_budget(v_ctx_tr, vocab_size, max_len),
                        descriptions=v_descs_tr,
                        values=("opt_0", "opt_1", "opt_2", "opt_3"),
                        target=v_target,
                    )
                )

                # Held-out Eval task: Unseen Mutation Set B + developer code-review prompt
                muts_eval = _mutate_focal_line(focal_line, mode="eval")
                v_cands_ev = list(muts_eval[:3])
                v_cands_ev.insert(v_target, focal_line)
                v_descs_ev = tuple(
                    _truncate_to_token_budget(
                        f"#T$# [{sym_name}] {_summarize_code_invariants(stmt)}\n#A$# {stmt}",
                        vocab_size,
                        max_len,
                    )
                    for stmt in v_cands_ev
                )
                v_ctx_ev = (
                    f"#U$# Code review on `{rel_path}`: verify `{sym_name}` ({rec.summary} | expected {auth_inv}):\n"
                    f"{shared_setup}\n"
                    f"#Q$# Select the authentic bug-free focal statement for `{sym_name}`."
                )
                eval_verify.append(
                    DecisionExample(
                        question_id=f"verify_eval_{task_i}",
                        kind="choice",
                        context=_truncate_to_token_budget(v_ctx_ev, vocab_size, max_len),
                        descriptions=v_descs_ev,
                        values=("opt_0", "opt_1", "opt_2", "opt_3"),
                        target=v_target,
                    )
                )

        return train_route, eval_route, train_verify, eval_verify

    @torch.no_grad()
    def evaluate_tasks(
        self,
        coe: CognitiveOperatorEnsemble,
        examples: Sequence[DecisionExample],
        *,
        batch_size: int = 8,
        recurrence_steps: int | None = None,
        depth_ponder: bool | None = None,
        margin_halt_threshold: float | None = None,
    ) -> dict[str, float]:
        """Evaluate `Decision` top-1 accuracy, mean target confidence, and multi-operator loss."""
        if not examples:
            return {"accuracy": 0.0, "confidence": 0.0, "loss": 0.0}
        coe.eval()
        device = next(coe.parameters()).device
        vocab_size = int(getattr(coe.backbone.config, "vocab_size", 260))
        max_len = min(self.max_seq_len, int(coe.backbone.config.max_seq_len))

        correct = 0
        total = 0
        conf_sum = 0.0
        loss_sum = 0.0
        n_batches = 0

        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            batch = collate_decisions(
                chunk,
                max_length=max_len,
                device=device,
                vocab_size=vocab_size,
            )
            op_out = coe.forward_all_operators(
                batch["context_ids"],
                batch["context_lengths"],
                batch["option_ids"],
                batch["option_lengths"],
                batch["targets"],
                recurrence_steps=recurrence_steps,
                depth_ponder=depth_ponder,
                margin_halt_threshold=margin_halt_threshold,
            )
            probs = op_out.decision.probabilities
            targets = batch["targets"].long()
            preds = probs.argmax(dim=-1)
            correct += int((preds == targets).sum().item())
            b_rows = torch.arange(len(chunk), device=device)
            conf_sum += float(probs[b_rows, targets].sum().item())
            total += len(chunk)
            loss_sum += float(op_out.total_loss.item())
            n_batches += 1

        return {
            "accuracy": correct / max(1, total),
            "confidence": conf_sum / max(1, total),
            "loss": loss_sum / max(1, n_batches),
        }

    def warmup_session(
        self,
        coe: CognitiveOperatorEnsemble,
        session: CodebaseSessionState,
        *,
        steps: int = 24,
        lr: float = 2.0e-3,
        batch_size: int = 8,
    ) -> CodebaseWarmupReport:
        """Run a bounded pre-prompt RL/COE warmup session on the indexed repository.

        Freezes `coe.backbone` (`requires_grad=False`) to preserve 100% of language generation
        fluency and pre-caches the frozen backbone's `EncodedState` + `DecodedCandidates` across
        the repository's AST tasks so 24+ multi-operator RL updates execute in ~1-2 seconds.
        """
        t0 = time.perf_counter()
        device = next(coe.parameters()).device
        vocab_size = int(getattr(coe.backbone.config, "vocab_size", 260))
        max_len = min(self.max_seq_len, int(coe.backbone.config.max_seq_len))

        # 1. Measure Pre-Warmup baseline on held-out routing & code verification tasks
        pre_route = self.evaluate_tasks(coe, session.eval_route_tasks, batch_size=batch_size)
        pre_verify = self.evaluate_tasks(coe, session.eval_verify_tasks, batch_size=batch_size)

        # 2. Freeze backbone, enable gradients only on lightweight COE operator & pooling heads
        orig_grad_flags = {name: p.requires_grad for name, p in coe.named_parameters()}
        op_names = set(self._operator_parameter_names(coe))
        trainable_params_list: list[nn.Parameter] = []
        frozen_count = 0
        trainable_count = 0

        for name, p in coe.named_parameters():
            if name in op_names:
                p.requires_grad_(True)
                trainable_params_list.append(p)
                trainable_count += p.numel()
            else:
                p.requires_grad_(False)
                frozen_count += p.numel()

        train_examples = list(session.train_route_tasks) + list(session.train_verify_tasks)
        step_losses: list[float] = []

        if train_examples and steps > 0 and trainable_params_list:
            # Pre-encode frozen backbone states once across all training mini-batches for ultra-fast RL warmup
            coe.eval()
            cached_batches: list[dict[str, Any]] = []
            with torch.no_grad():
                for start in range(0, len(train_examples), batch_size):
                    chunk = train_examples[start : start + batch_size]
                    b = collate_decisions(
                        chunk,
                        max_length=max_len,
                        device=device,
                        vocab_size=vocab_size,
                    )
                    context_ids = b["context_ids"]
                    context_lengths = b["context_lengths"]
                    option_ids = b["option_ids"]
                    option_lengths = b["option_lengths"]
                    targets = b["targets"].long()

                    valid_options = option_lengths > 0
                    ctx_width = int(context_lengths.max().item())
                    trimmed_ctx_ids = context_ids[:, :ctx_width].long()
                    ctx_valid = (
                        torch.arange(ctx_width, device=device)[None, :] < context_lengths[:, None]
                    )
                    encoded_state = coe.backbone.encode_state(trimmed_ctx_ids, ctx_valid, causal=True)

                    valid_lengths = option_lengths[valid_options]
                    opt_width = int(valid_lengths.max().item())
                    flat_opt_ids = option_ids[valid_options][:, :opt_width].long()
                    owner_rows = valid_options.nonzero(as_tuple=False)[:, 0]

                    decoded_options = coe.backbone.decode_candidates(
                        encoded_state,
                        flat_opt_ids,
                        (torch.arange(opt_width, device=device)[None, :] < valid_lengths[:, None]),
                        owner_rows=owner_rows,
                        recurrence_steps=coe.config.recurrence_steps,
                        causal_local=True,
                    )
                    cached_batches.append(
                        {
                            "encoded_hidden": encoded_state.hidden.detach(),
                            "trimmed_ctx_ids": trimmed_ctx_ids,
                            "ctx_valid": ctx_valid,
                            "decoded_hidden": decoded_options.hidden.detach(),
                            "decoded_anchor": (
                                decoded_options.anchor.detach()
                                if decoded_options.anchor is not None
                                else None
                            ),
                            "flat_opt_ids": flat_opt_ids,
                            "opt_valid": (
                                torch.arange(opt_width, device=device)[None, :]
                                < valid_lengths[:, None]
                            ),
                            "owner_rows": owner_rows,
                            "valid_options": valid_options,
                            "targets": targets,
                            "batch_size": context_ids.shape[0],
                            "num_opts": option_ids.shape[1],
                        }
                    )

            optimizer = torch.optim.AdamW(trainable_params_list, lr=lr, weight_decay=1e-4)
            n_cb = float(len(cached_batches))
            for _step_idx in range(steps):
                optimizer.zero_grad(set_to_none=True)
                epoch_loss_val = 0.0

                for cb in cached_batches:
                    paired_state_repr, cand_repr = coe.pooler.matched_pair_representations(
                        cb["encoded_hidden"],
                        cb["trimmed_ctx_ids"],
                        cb["ctx_valid"],
                        cb["decoded_hidden"],
                        cb["flat_opt_ids"],
                        cb["opt_valid"],
                        cb["owner_rows"],
                        cand_anchor=cb["decoded_anchor"],
                        strip_boundary=coe.config.strip_boundary_tokens_in_pool,
                        bos_id=coe.config.bos_id,
                        eos_id=coe.config.eos_id,
                    )
                    state_proj = coe.state_projection(paired_state_repr)
                    cand_proj = coe.output_projection(cand_repr)
                    valid_scores = coe._similarity(state_proj, cand_proj)

                    valid_options = cb["valid_options"]
                    similarities = valid_scores.new_zeros(valid_options.shape).masked_scatter(
                        valid_options, valid_scores
                    )
                    bsz = cb["batch_size"]
                    num_opts = cb["num_opts"]
                    d_m = paired_state_repr.shape[-1]
                    mask_3d = valid_options.unsqueeze(-1).expand(bsz, num_opts, d_m)
                    state_grid = paired_state_repr.new_zeros((bsz, num_opts, d_m)).masked_scatter(
                        mask_3d, paired_state_repr
                    )
                    opt_grid = cand_repr.new_zeros((bsz, num_opts, d_m)).masked_scatter(
                        mask_3d, cand_repr
                    )

                    b_rows = torch.arange(bsz, device=device)
                    t_idx = cb["targets"]
                    s_pos = state_grid[b_rows, t_idx]
                    c_pos = opt_grid[b_rows, t_idx]

                    d_p = state_proj.shape[-1]
                    mask_3d_p = valid_options.unsqueeze(-1).expand(bsz, num_opts, d_p)
                    s_proj_grid = state_proj.new_zeros((bsz, num_opts, d_p)).masked_scatter(
                        mask_3d_p, state_proj
                    )
                    c_proj_grid = cand_proj.new_zeros((bsz, num_opts, d_p)).masked_scatter(
                        mask_3d_p, cand_proj
                    )
                    s_pos_proj = s_proj_grid[b_rows, t_idx]
                    c_pos_proj = c_proj_grid[b_rows, t_idx]

                    distractor_mask = valid_options.clone()
                    distractor_mask[b_rows, t_idx] = False
                    neg_scores = similarities.float().masked_fill(~distractor_mask, -torch.inf)
                    d_idx = neg_scores.argmax(dim=-1)
                    s_neg = state_grid[b_rows, d_idx]
                    c_neg = opt_grid[b_rows, d_idx]

                    s_Pair = torch.cat([s_pos, s_neg], dim=0)
                    c_Pair = torch.cat([c_pos, c_neg], dim=0)
                    ded_joint = coe.deduction_query_proj(torch.cat([s_Pair, c_Pair], dim=-1))

                    with torch.autocast(device_type=device.type, enabled=False):
                        scale = coe._scale()
                        logits = (similarities.float() * scale).masked_fill(~valid_options, -torch.inf)
                        dec_loss = F.cross_entropy(logits, t_idx)

                        # Symmetric in-batch InfoNCE prevents unconditional candidate prior bias
                        if bsz > 1:
                            cross_sims = pearson_similarity(
                                s_pos_proj.float().unsqueeze(1),
                                c_pos_proj.float().unsqueeze(0),
                                eps=coe.config.eps,
                            ) * scale
                            infonce_loss = 0.5 * (
                                F.cross_entropy(cross_sims, b_rows)
                                + F.cross_entropy(cross_sims.transpose(0, 1), b_rows)
                            )
                        else:
                            infonce_loss = dec_loss.new_zeros(())

                        ded_sims = pearson_similarity(
                            ded_joint.unsqueeze(1),
                            coe.deduction_prototypes.unsqueeze(0),
                            eps=coe.config.eps,
                        )
                        ded_logits = ded_sims.float() * scale
                        ded_targets = torch.cat(
                            [
                                torch.zeros(bsz, device=device, dtype=torch.long),
                                torch.ones(bsz, device=device, dtype=torch.long),
                            ],
                            dim=0,
                        )
                        ded_loss = F.cross_entropy(ded_logits, ded_targets)

                        eval_logits = (
                            valid_scores.float() * coe.assoc_alpha.float()
                            + coe.assoc_beta.float()
                        )
                        target_grid = F.one_hot(t_idx, num_classes=num_opts).float()
                        eval_loss = F.binary_cross_entropy_with_logits(
                            eval_logits, target_grid[valid_options]
                        )

                        batch_loss = (
                            dec_loss + 0.35 * infonce_loss + 0.30 * ded_loss + 0.20 * eval_loss
                        ) / n_cb

                    batch_loss.backward()
                    epoch_loss_val += float(batch_loss.item())

                torch.nn.utils.clip_grad_norm_(trainable_params_list, 1.0)
                optimizer.step()
                step_losses.append(epoch_loss_val)

            # Also warm up the Tier-3 ContextLMCompressor gist queries on decompressed TurboQuant file pages
            if session.context_compressor is not None and session.file_turboquant_pages:
                comp_mod = session.context_compressor.to(device=device)
                comp_mod.train()
                for p in comp_mod.parameters():
                    p.requires_grad_(True)
                comp_opt = torch.optim.AdamW(comp_mod.parameters(), lr=lr * 1.5, weight_decay=1e-4)
                file_addrs = list(session.file_turboquant_pages.keys())
                deq_pages = [session.file_turboquant_pages[fa].decompress() for fa in file_addrs]
                prior_targets = torch.stack(
                    [session.file_latent_priors[fa].to(device=device).float() for fa in file_addrs],
                    dim=0,
                )
                for _c_step in range(min(12, max(4, steps // 2))):
                    comp_opt.zero_grad(set_to_none=True)
                    folded_means = []
                    for enc_p in deq_pages:
                        h = enc_p.hidden.to(device=device)
                        c_val = enc_p.valid.to(device=device, dtype=torch.bool)
                        q = comp_mod.gist_queries.unsqueeze(0).to(device=device, dtype=h.dtype)
                        scores = torch.matmul(q.float(), h.float().transpose(-1, -2)) / math.sqrt(
                            comp_mod.d_model
                        )
                        scores = scores.masked_fill(~c_val.unsqueeze(1), -1e9)
                        attn = F.softmax(scores, dim=-1)
                        pooled = torch.matmul(attn, h.float())
                        chunk_gist = 0.85 * pooled + 0.15 * comp_mod.gist_norm(pooled)
                        folded_means.append(chunk_gist.mean(dim=1).squeeze(0))
                    folded_stack = torch.stack(folded_means, dim=0)
                    comp_loss = (
                        1.0 - F.cosine_similarity(folded_stack, prior_targets, dim=-1).mean()
                    )
                    comp_loss.backward()
                    comp_opt.step()
                comp_mod.eval()

        # Restore original requires_grad flags
        for name, p in coe.named_parameters():
            p.requires_grad_(orig_grad_flags.get(name, True))
        coe.eval()

        # 3. Measure Post-Warmup performance on held-out routing & code verification tasks
        post_route = self.evaluate_tasks(coe, session.eval_route_tasks, batch_size=batch_size)
        post_verify = self.evaluate_tasks(coe, session.eval_verify_tasks, batch_size=batch_size)
        elapsed = time.perf_counter() - t0

        return CodebaseWarmupReport(
            repo_root=session.repo_root,
            total_files=session.total_repo_files,
            total_symbols=session.total_repo_symbols,
            total_repo_bytes=session.total_repo_bytes,
            total_repo_tokens=session.total_repo_tokens,
            uncompressed_state_bytes=session.uncompressed_state_bytes,
            turboquant_state_bytes=session.compressed_state_bytes,
            compression_ratio=session.compression_ratio,
            mean_turboquant_cosine=session.mean_turboquant_cosine,
            warmup_steps=steps,
            warmup_elapsed_sec=elapsed,
            trainable_params=trainable_count,
            frozen_backbone_params=frozen_count,
            pre_route_acc=pre_route["accuracy"],
            post_route_acc=post_route["accuracy"],
            pre_route_conf=pre_route["confidence"],
            post_route_conf=post_route["confidence"],
            pre_verify_acc=pre_verify["accuracy"],
            post_verify_acc=post_verify["accuracy"],
            pre_verify_conf=pre_verify["confidence"],
            post_verify_conf=post_verify["confidence"],
            pre_loss=0.5 * (pre_route["loss"] + pre_verify["loss"]),
            post_loss=0.5 * (post_route["loss"] + post_verify["loss"]),
            step_losses=step_losses,
        )

    @torch.no_grad()
    def query_codebase(
        self,
        coe: CognitiveOperatorEnsemble,
        session: CodebaseSessionState,
        user_query: str,
        *,
        top_k_candidates: int = 6,
        recurrence_steps: int | None = None,
        depth_ponder: bool | None = None,
        margin_halt_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Route a user debugging/codebase query via warmed `COEAddressRouter`, traverse `A.TXY` `CALLS` edges, and retrieve `TurboQuant` page state."""
        route_res = session.router.route_query(
            coe,
            user_query,
            top_k_candidates=top_k_candidates,
            plane_filter=1,
            recurrence_steps=recurrence_steps,
            depth_ponder=depth_ponder,
            margin_halt_threshold=margin_halt_threshold,
        )
        primary_rec: ATXYRecord = route_res["record"]
        primary_source = session.symbol_sources.get(primary_rec.address, "")

        # Traverse UP to parent file and FOLLOW_SET("CALLS") via ATXYPointerExecutor
        parent_file_addr = primary_rec.up[0] if primary_rec.up else None
        callees: list[dict[str, str]] = []
        if "CALLS" in primary_rec.rels:
            _, callee_addrs = session.pointer_executor.run(
                [("FOLLOW_SET", [primary_rec.address], "CALLS")]
            )
            for c_addr in callee_addrs:
                crec = session.store.lookup(c_addr)
                if crec is not None:
                    callees.append(
                        {
                            "address": crec.address,
                            "symbol": crec.payload,
                            "summary": crec.summary,
                        }
                    )

        tq_page = (
            session.file_turboquant_pages.get(parent_file_addr)
            if parent_file_addr is not None
            else None
        )
        decompressed_state = tq_page.decompress() if tq_page is not None else None
        latent_prior = (
            session.file_latent_priors.get(parent_file_addr)
            if parent_file_addr is not None
            else None
        )

        return {
            "query": user_query,
            "address": primary_rec.address,
            "symbol": primary_rec.payload,
            "kind": primary_rec.kind,
            "summary": primary_rec.summary,
            "confidence": route_res["confidence"],
            "candidates": route_res["candidates"],
            "probabilities": route_res["probabilities"],
            "source_snippet": primary_source,
            "parent_file_address": parent_file_addr,
            "callees": callees,
            "turboquant_page_compressed_bytes": (
                tq_page.compressed_bytes if tq_page is not None else 0
            ),
            "turboquant_decompressed_shape": (
                tuple(decompressed_state.hidden.shape) if decompressed_state is not None else ()
            ),
            "latent_prior": latent_prior,
        }

    @torch.no_grad()
    def synthesize_multifile_context(
        self,
        coe: CognitiveOperatorEnsemble,
        session: CodebaseSessionState,
        user_query: str,
        *,
        max_files: int = 3,
        top_k_candidates: int = 6,
        recurrence_steps: int | None = None,
        depth_ponder: bool | None = None,
        margin_halt_threshold: float | None = None,
    ) -> MultiFileSynthesisResult:
        """Synthesize multi-file context via `A.TXY` routing, `CALLS` traversal, `TurboQuant` page decompression, and `ContextLMCompressor` folding."""
        device = next(coe.parameters()).device
        route_res = session.router.route_query(
            coe,
            user_query,
            top_k_candidates=top_k_candidates,
            plane_filter=1,
            recurrence_steps=recurrence_steps,
            depth_ponder=depth_ponder,
            margin_halt_threshold=margin_halt_threshold,
        )
        primary_rec: ATXYRecord = route_res["record"]
        primary_source = session.symbol_sources.get(primary_rec.address, "")
        _, focal_stmt = _extract_focal_line_and_context(primary_source)
        focal_inv = _summarize_code_invariants(focal_stmt)

        participating_file_addrs: list[str] = []
        if primary_rec.up and primary_rec.up[0] in session.file_turboquant_pages:
            participating_file_addrs.append(primary_rec.up[0])

        callees: list[dict[str, str]] = []
        if "CALLS" in primary_rec.rels:
            _, callee_addrs = session.pointer_executor.run(
                [("FOLLOW_SET", [primary_rec.address], "CALLS")]
            )
            for c_addr in callee_addrs:
                crec = session.store.lookup(c_addr)
                if crec is not None:
                    callees.append(
                        {
                            "address": crec.address,
                            "symbol": crec.payload,
                            "summary": crec.summary,
                        }
                    )
                    if (
                        crec.up
                        and crec.up[0] in session.file_turboquant_pages
                        and crec.up[0] not in participating_file_addrs
                        and len(participating_file_addrs) < max_files
                    ):
                        participating_file_addrs.append(crec.up[0])

        # Fill remaining multi-file slots from secondary routed candidates across distinct modules
        for cand_addr in route_res["candidates"]:
            if len(participating_file_addrs) >= max_files:
                break
            cand_rec = session.store.lookup(cand_addr)
            if cand_rec is not None and cand_rec.up:
                f_addr = cand_rec.up[0]
                if (
                    f_addr in session.file_turboquant_pages
                    and f_addr not in participating_file_addrs
                ):
                    participating_file_addrs.append(f_addr)

        if not participating_file_addrs and session.file_turboquant_pages:
            participating_file_addrs.append(next(iter(session.file_turboquant_pages.keys())))

        participating_files: list[str] = []
        tq_pages: list[CompressedEncodedState] = []
        raw_source_tokens = 0
        encoded_page_tokens = 0
        uncompressed_bytes = 0
        turboquant_bytes = 0

        for f_addr in participating_file_addrs:
            frec = session.store.lookup(f_addr)
            if frec is not None:
                participating_files.append(frec.payload)
            page = session.file_turboquant_pages[f_addr]
            tq_pages.append(page)
            raw_source_tokens += int(session.file_token_counts.get(f_addr, 0))
            encoded_page_tokens += int(page.valid.sum().item())
            uncompressed_bytes += int(page.uncompressed_bytes)
            turboquant_bytes += int(page.compressed_bytes)

        compressor = session.context_compressor
        if compressor is None:
            compressor = ContextLMCompressor(
                d_model=int(coe.backbone.config.d_model),
                chunk_size=self.max_seq_len,
                gist_tokens_per_chunk=6,
            ).to(device=device)
            session.context_compressor = compressor
        else:
            compressor = compressor.to(device=device)

        compressed_state = compressor.compress_turboquant_pages(coe.backbone, tq_pages)
        compressed_gist_tokens = int(compressed_state.hidden.shape[1])

        # Fuse multi-file latent priors with ContextLMCompressor gist representation
        prior_list = [
            session.file_latent_priors[f_addr].to(device=device).float()
            for f_addr in participating_file_addrs
            if f_addr in session.file_latent_priors
        ]
        gist_mean = compressed_state.hidden.float().mean(dim=1)  # [1, d_model]
        if prior_list:
            if len(prior_list) == 1:
                file_prior = prior_list[0].unsqueeze(0)
            else:
                sec_mean = torch.stack(prior_list[1:], dim=0).mean(dim=0)
                file_prior = (0.60 * prior_list[0] + 0.40 * sec_mean).unsqueeze(0)
            raw_prior = 0.70 * file_prior + 0.30 * gist_mean
        else:
            raw_prior = gist_mean

        norm_dtype = next(coe.abstraction_norm.parameters()).dtype
        multifile_latent_prior = coe.abstraction_norm(raw_prior.to(dtype=norm_dtype)).float()

        return MultiFileSynthesisResult(
            query=user_query,
            primary_address=primary_rec.address,
            primary_symbol=primary_rec.payload,
            primary_kind=primary_rec.kind,
            primary_summary=primary_rec.summary,
            primary_confidence=float(route_res["confidence"]),
            primary_source=primary_source,
            focal_statement=focal_stmt,
            focal_invariant=focal_inv,
            callees=tuple(callees),
            participating_files=tuple(participating_files),
            participating_file_addresses=tuple(participating_file_addrs),
            num_files_folded=len(participating_file_addrs),
            raw_source_tokens=raw_source_tokens,
            encoded_page_tokens=encoded_page_tokens,
            compressed_gist_tokens=compressed_gist_tokens,
            uncompressed_page_bytes=uncompressed_bytes,
            turboquant_page_bytes=turboquant_bytes,
            token_folding_ratio=float(raw_source_tokens) / max(1.0, float(compressed_gist_tokens)),
            byte_compression_ratio=float(uncompressed_bytes) / max(1.0, float(turboquant_bytes)),
            compressed_state=compressed_state,
            multifile_latent_prior=multifile_latent_prior,
        )

    def detect_dirty_files(self, session: CodebaseSessionState) -> list[str]:
        """Scan `session.repo_root` and return repo-relative POSIX paths of added, modified, or deleted `.py` files."""
        root = Path(session.repo_root).expanduser().resolve()
        subsystems = session.subsystems or self.subsystems
        dirty: set[str] = set()
        seen_on_disk: set[str] = set()

        for sub in subsystems[:ATXY_BASE]:
            sub_dir = root / sub
            if not sub_dir.is_dir():
                continue
            py_files = sorted(p for p in sub_dir.rglob("*.py") if "__pycache__" not in p.parts)[
                :ATXY_BASE
            ]
            for py_path in py_files:
                rel = py_path.relative_to(root).as_posix()
                seen_on_disk.add(rel)
                src_bytes = py_path.read_text(encoding="utf-8", errors="replace").encode("utf-8")
                digest = hashlib.sha256(src_bytes).hexdigest()
                if session.file_digests.get(rel) != digest:
                    dirty.add(rel)

        for rel in list(session.file_digests.keys()):
            if rel not in seen_on_disk and not (root / rel).is_file():
                dirty.add(rel)

        return sorted(dirty)

    def refresh_dirty_files(
        self,
        coe: CognitiveOperatorEnsemble,
        session: CodebaseSessionState,
        dirty_paths: Sequence[str | Path] | None = None,
        *,
        micro_warmup_steps: int = 3,
        lr: float = 2.0e-3,
        wal_path: str | Path | None = None,
    ) -> DirtyPageRefreshReport:
        """Incrementally re-index dirty `.py` files into `ATXYPageStore` (`N5` WAL), re-quantize only their 4-bit `TurboQuant` pages, and run a targeted micro-RL fast-weight update."""
        t0 = time.perf_counter()
        root = Path(session.repo_root).expanduser().resolve()
        subsystems = session.subsystems or self.subsystems

        if dirty_paths is None:
            dirty_rels = self.detect_dirty_files(session)
        else:
            norm_rels: list[str] = []
            for p in dirty_paths:
                pp = Path(p)
                if pp.is_absolute():
                    rel = pp.resolve().relative_to(root).as_posix()
                else:
                    rel = pp.as_posix().lstrip("./")
                if rel not in norm_rels:
                    norm_rels.append(rel)
            dirty_rels = norm_rels

        if not dirty_rels:
            return DirtyPageRefreshReport(
                dirty_files=(),
                updated_file_addresses=(),
                updated_symbol_addresses=(),
                added_symbols=(),
                modified_symbols=(),
                removed_symbols=(),
                wal_frames_appended=0,
                recompressed_pages=0,
                mean_refresh_cosine=session.mean_turboquant_cosine,
                micro_warmup_steps=0,
                micro_warmup_loss=0.0,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        device = next(coe.parameters()).device
        vocab_size = int(getattr(coe.backbone.config, "vocab_size", 260))
        tokenizer = get_tokenizer(vocab_size)
        max_len = min(self.max_seq_len, int(coe.backbone.config.max_seq_len))
        initial_wal_len = len(session.store._wal_frames)

        dirty_set = set(dirty_rels)
        added_symbols: list[str] = []
        modified_symbols: list[str] = []
        removed_symbols: list[str] = []
        updated_file_addrs: list[str] = []
        updated_sym_addrs: list[str] = []

        # 1. Build symbol index of unchanged files so CALLS edges resolve across the full repo
        sym_index: dict[str, list[str]] = {}
        for rec in session.store.records():
            if rec.parsed_address.A == 1 and ":" in rec.payload:
                r_rel, s_name = rec.payload.split(":", 1)
                if r_rel not in dirty_set:
                    sym_index.setdefault(s_name, []).append(rec.address)

        parsed_dirty: dict[
            str,
            tuple[
                str,
                int,
                int,
                str,
                list[str],
                list[tuple[str, str, str, ast.AST]],
            ],
        ] = {}

        # 2. Parse dirty files (or clean up deleted files) and register their A.TXY coordinates
        for rel in dirty_rels:
            fpath = root / rel
            old_f_addr = session.file_addresses.get(rel)
            old_frec = session.store.lookup(old_f_addr) if old_f_addr else None

            if not fpath.is_file():
                # File was deleted on disk
                if old_frec is not None:
                    for s_addr in old_frec.down:
                        srec = session.store.lookup(s_addr)
                        if srec is not None:
                            removed_symbols.append(srec.payload)
                        session.store.delete(s_addr)
                        session.symbol_sources.pop(s_addr, None)
                    session.store.delete(old_frec.address)
                    session.file_turboquant_pages.pop(old_frec.address, None)
                    session.file_latent_priors.pop(old_frec.address, None)
                    session.file_token_counts.pop(old_frec.address, None)
                session.file_digests.pop(rel, None)
                session.file_addresses.pop(rel, None)
                continue

            if old_f_addr is not None:
                parsed_f = parse_atxy(old_f_addr)
                t_idx, x_idx = parsed_f.T, parsed_f.X
                f_addr = old_f_addr
            else:
                # Allocate new A=2 file slot in matching subsystem
                t_idx = 0
                for idx_sub, sub in enumerate(subsystems[:ATXY_BASE]):
                    if rel == sub or rel.startswith(sub.rstrip("/") + "/"):
                        t_idx = idx_sub
                        break
                existing_x = [
                    r.parsed_address.X
                    for r in session.store.records()
                    if r.parsed_address.A == 2 and r.parsed_address.T == t_idx
                ]
                x_idx = (max(existing_x) + 1) if existing_x else 0
                f_addr = str(ATXYAddress(2, t_idx, x_idx, 0))
                session.file_addresses[rel] = f_addr

                sub_addr = str(ATXYAddress(3, t_idx, 0, 0))
                sub_rec = session.store.lookup(sub_addr)
                if sub_rec is not None and f_addr not in sub_rec.down:
                    new_down = tuple(list(sub_rec.down) + [f_addr])
                    session.store.commit_write(
                        ATXYRecord(
                            address=sub_addr,
                            kind="subsystem",
                            payload=sub_rec.payload,
                            summary=f"Subsystem {sub_rec.payload} ({len(new_down)} files)",
                            down=new_down,
                            fields={"file_count": len(new_down)},
                            version=sub_rec.version + 1,
                            provenance=("ast:subsystem", "refresh:dirty"),
                        ),
                        wal_path=wal_path,
                    )

            src_text = fpath.read_text(encoding="utf-8", errors="replace")
            src_bytes = src_text.encode("utf-8")
            src_lines = src_text.splitlines()
            session.file_digests[rel] = hashlib.sha256(src_bytes).hexdigest()

            syms: list[tuple[str, str, str, ast.AST]] = []
            try:
                tree = ast.parse(src_text)
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        kind = "class" if isinstance(node, ast.ClassDef) else "function"
                        summary = _summarize_ast_symbol(node, rel)
                        syms.append((node.name, kind, summary, node))
            except SyntaxError:
                syms = []

            syms.sort(key=lambda item: item[0].startswith("_"))
            syms = syms[:ATXY_BASE]
            parsed_dirty[rel] = (f_addr, t_idx, x_idx, src_text, src_lines, syms)

            for y_idx, (s_name, _kind, _sum, _nd) in enumerate(syms):
                s_addr = str(ATXYAddress(1, t_idx, x_idx, y_idx))
                sym_index.setdefault(s_name, []).append(s_addr)

        # 3. Commit updated A=2 file and A=1 symbol records to ATXYPageStore + re-compress 4-bit TurboQuant pages
        refresh_cosines: list[float] = []
        with torch.no_grad():
            for rel, (f_addr, t_idx, x_idx, src_text, src_lines, syms) in parsed_dirty.items():
                old_frec = session.store.lookup(f_addr)
                old_sym_map: dict[str, tuple[str, str, str]] = {}
                if old_frec is not None:
                    for old_s_addr in old_frec.down:
                        old_srec = session.store.lookup(old_s_addr)
                        if old_srec is not None:
                            old_sym_map[old_srec.payload] = (
                                old_s_addr,
                                old_srec.summary,
                                session.symbol_sources.get(old_s_addr, ""),
                            )

                new_sym_addrs = tuple(
                    str(ATXYAddress(1, t_idx, x_idx, y_idx)) for y_idx in range(len(syms))
                )
                if old_frec is not None:
                    for old_s_addr in old_frec.down:
                        if old_s_addr not in new_sym_addrs:
                            session.store.delete(old_s_addr)
                            session.symbol_sources.pop(old_s_addr, None)

                new_payloads = {f"{rel}:{s_name}" for s_name, _, _, _ in syms}
                for old_payload in old_sym_map:
                    if old_payload not in new_payloads:
                        removed_symbols.append(old_payload)

                sub_addr = str(ATXYAddress(3, t_idx, 0, 0))
                session.store.commit_write(
                    ATXYRecord(
                        address=f_addr,
                        kind="file",
                        payload=rel,
                        summary=f"Module {rel} ({len(syms)} top-level symbols)",
                        up=(sub_addr,),
                        down=new_sym_addrs,
                        fields={"symbol_count": len(syms)},
                        version=(old_frec.version + 1 if old_frec is not None else 1),
                        provenance=("ast:file", "refresh:dirty"),
                    ),
                    wal_path=wal_path,
                )
                updated_file_addrs.append(f_addr)

                toks = tokenizer.encode(src_text, add_bos=False, add_eos=False)
                session.file_token_counts[f_addr] = len(toks)

                module_summary_lines = [f"# Module {rel}"]
                for y_idx, (s_name, s_kind, s_sum, s_node) in enumerate(syms):
                    s_addr = str(ATXYAddress(1, t_idx, x_idx, y_idx))
                    payload = f"{rel}:{s_name}"

                    start_ln = max(0, int(getattr(s_node, "lineno", 1)) - 1)
                    end_ln = min(
                        len(src_lines),
                        int(getattr(s_node, "end_lineno", start_ln + self.max_symbol_lines)),
                        start_ln + self.max_symbol_lines,
                    )
                    snippet = "\n".join(src_lines[start_ln:end_ln]).strip()

                    callees: list[str] = []
                    for n in ast.walk(s_node):
                        if isinstance(n, ast.Call):
                            cname = (
                                n.func.id
                                if isinstance(n.func, ast.Name)
                                else (n.func.attr if isinstance(n.func, ast.Attribute) else None)
                            )
                            if cname and cname != s_name and len(sym_index.get(cname, ())) == 1:
                                tgt = sym_index[cname][0]
                                if tgt not in callees:
                                    callees.append(tgt)

                    is_added = payload not in old_sym_map
                    is_modified = (
                        not is_added
                        and (
                            old_sym_map[payload][1] != s_sum
                            or old_sym_map[payload][2] != snippet
                        )
                    )
                    if is_added:
                        added_symbols.append(payload)
                        updated_sym_addrs.append(s_addr)
                    elif is_modified:
                        modified_symbols.append(payload)
                        updated_sym_addrs.append(s_addr)

                    old_cell_rec = session.store.lookup(s_addr)
                    session.store.commit_write(
                        ATXYRecord(
                            address=s_addr,
                            kind=s_kind,
                            payload=payload,
                            summary=s_sum,
                            up=(f_addr,),
                            rels={"CALLS": tuple(callees)} if callees else {},
                            fields={"lineno": int(getattr(s_node, "lineno", 1))},
                            version=(old_cell_rec.version + 1 if old_cell_rec is not None else 1),
                            provenance=("ast:symbol", "refresh:dirty"),
                        ),
                        wal_path=wal_path,
                    )
                    session.symbol_sources[s_addr] = snippet
                    module_summary_lines.append(f"{s_name}: {s_sum}")

                # Re-encode only this dirty module through lower-CED & quantize via 4-bit TurboQuant
                mod_doc = "\n".join(module_summary_lines)
                mod_ids = tokenizer.encode(mod_doc, add_bos=True, add_eos=True)[:max_len]
                inp_t = torch.tensor([mod_ids], dtype=torch.long, device=device)
                len_t = torch.tensor([len(mod_ids)], dtype=torch.long, device=device)
                valid_mask = torch.ones((1, len(mod_ids)), dtype=torch.bool, device=device)
                enc_state = coe.backbone.encode_state(inp_t, valid_mask, causal=True)
                comp_state = compress_encoded_state(enc_state, seed=1337 + x_idx * 17 + t_idx * 101)
                deq_state = comp_state.decompress()

                cos = float(
                    F.cosine_similarity(
                        enc_state.hidden.float().reshape(-1, enc_state.hidden.shape[-1]),
                        deq_state.hidden.float().reshape(-1, deq_state.hidden.shape[-1]),
                        dim=-1,
                    )
                    .mean()
                    .item()
                )
                refresh_cosines.append(cos)
                session.file_turboquant_pages[f_addr] = comp_state
                session.file_latent_priors[f_addr] = coe.compress_abstraction(
                    inp_t, len_t, level="standard"
                )[0].detach()

        # Recompute aggregate session statistics
        all_file_recs = [r for r in session.store.records() if r.parsed_address.A == 2]
        all_sym_recs = [r for r in session.store.records() if r.parsed_address.A == 1]
        session.total_repo_files = len(all_file_recs)
        session.total_repo_symbols = len(all_sym_recs)
        session.total_repo_tokens = sum(session.file_token_counts.values())
        session.uncompressed_state_bytes = sum(
            p.uncompressed_bytes for p in session.file_turboquant_pages.values()
        )
        session.compressed_state_bytes = sum(
            p.compressed_bytes for p in session.file_turboquant_pages.values()
        )
        if refresh_cosines:
            session.mean_turboquant_cosine = 0.85 * session.mean_turboquant_cosine + 0.15 * (
                sum(refresh_cosines) / len(refresh_cosines)
            )

        # 4. Targeted Micro-RL Fast-Weight Adaptation exclusively on dirty symbols & pages
        micro_loss_val = 0.0
        actual_micro_steps = 0
        if micro_warmup_steps > 0 and updated_sym_addrs and len(all_sym_recs) >= 4:
            target_set = set(updated_sym_addrs)
            sym_toks = [_code_tokens_and_stems(f"{r.payload} {r.summary}") for r in all_sym_recs]
            k_opts = min(self.num_route_options, len(all_sym_recs))
            micro_tasks: list[DecisionExample] = []

            for task_i, idx in enumerate(
                [i for i, r in enumerate(all_sym_recs) if r.address in target_set]
            ):
                rec = all_sym_recs[idx]
                rel_path, sym_name = (
                    rec.payload.split(":", 1) if ":" in rec.payload else (rec.payload, rec.payload)
                )
                mod_short = Path(rel_path).name
                scored_neg = []
                for j, other in enumerate(all_sym_recs):
                    if j == idx:
                        continue
                    same_mod = 1.5 if other.up == rec.up else 0.0
                    ov = len(sym_toks[idx] & sym_toks[j]) / max(1, len(sym_toks[idx]))
                    scored_neg.append((same_mod + ov, j))
                scored_neg.sort(key=lambda pair: pair[0], reverse=True)
                neg_indices = [j for _, j in scored_neg[: k_opts - 1]]
                target_pos = task_i % k_opts
                cand_list = [all_sym_recs[j] for j in neg_indices]
                cand_list.insert(target_pos, rec)
                desc_tuple = tuple(
                    _truncate_to_token_budget(c.to_candidate_description(), vocab_size, max_len)
                    for c in cand_list
                )
                val_tuple = tuple(c.address for c in cand_list)
                q_text = (
                    f"#U$# Explore {mod_short}: locate {rec.kind} `{sym_name}` — {rec.summary}.\n"
                    f"#Q$# Select the exact A.TXY record matching the request."
                )
                micro_tasks.append(
                    DecisionExample(
                        question_id=f"dirty_route_{task_i}",
                        kind="choice",
                        context=_truncate_to_token_budget(q_text, vocab_size, max_len),
                        descriptions=desc_tuple,
                        values=val_tuple,
                        target=target_pos,
                    )
                )

            if micro_tasks:
                op_names = set(self._operator_parameter_names(coe))
                orig_grad = {name: p.requires_grad for name, p in coe.named_parameters()}
                trainable_params = []
                for name, p in coe.named_parameters():
                    if name in op_names:
                        p.requires_grad_(True)
                        trainable_params.append(p)
                    else:
                        p.requires_grad_(False)
                coe.train()
                coe.backbone.eval()

                with torch.enable_grad():
                    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
                    batch = collate_decisions(
                        micro_tasks,
                        max_length=max_len,
                        device=device,
                        vocab_size=vocab_size,
                        tokenizer=tokenizer,
                    )
                    for _ in range(micro_warmup_steps):
                        opt.zero_grad(set_to_none=True)
                        dec_out = coe.forward_decision(
                            batch["context_ids"],
                            batch["context_lengths"],
                            batch["option_ids"],
                            batch["option_lengths"],
                            recurrence_steps=1,
                        )
                        scale = coe._scale()
                        valid_opts = batch["option_lengths"] > 0
                        logits = (dec_out.similarities.float() * scale).masked_fill(
                            ~valid_opts, -torch.inf
                        )
                        loss = F.cross_entropy(logits, batch["targets"])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        opt.step()
                        micro_loss_val = float(loss.item())
                        actual_micro_steps += 1

                    if session.context_compressor is not None and updated_file_addrs:
                        comp_mod = session.context_compressor.to(device=device)
                        comp_mod.train()
                        comp_opt = torch.optim.AdamW(comp_mod.parameters(), lr=lr, weight_decay=1e-4)
                        deq_pages = [
                            session.file_turboquant_pages[fa].decompress()
                            for fa in updated_file_addrs
                            if fa in session.file_turboquant_pages
                        ]
                        prior_targets = torch.stack(
                            [
                                session.file_latent_priors[fa].to(device=device).float()
                                for fa in updated_file_addrs
                                if fa in session.file_latent_priors
                            ],
                            dim=0,
                        )
                        for _ in range(min(2, micro_warmup_steps)):
                            comp_opt.zero_grad(set_to_none=True)
                            folded_means = []
                            for enc_p in deq_pages:
                                h = enc_p.hidden.to(device=device)
                                c_val = enc_p.valid.to(device=device, dtype=torch.bool)
                                q = comp_mod.gist_queries.unsqueeze(0).to(
                                    device=device, dtype=h.dtype
                                )
                                scores = torch.matmul(
                                    q.float(), h.float().transpose(-1, -2)
                                ) / math.sqrt(comp_mod.d_model)
                                scores = scores.masked_fill(~c_val.unsqueeze(1), -1e9)
                                attn = F.softmax(scores, dim=-1)
                                pooled = torch.matmul(attn, h.float())
                                chunk_gist = 0.85 * pooled + 0.15 * comp_mod.gist_norm(pooled)
                                folded_means.append(chunk_gist.mean(dim=1).squeeze(0))
                            folded_stack = torch.stack(folded_means, dim=0)
                            c_loss = (
                                1.0
                                - F.cosine_similarity(folded_stack, prior_targets, dim=-1).mean()
                            )
                            c_loss.backward()
                            comp_opt.step()
                        comp_mod.eval()

                for name, p in coe.named_parameters():
                    p.requires_grad_(orig_grad.get(name, True))
                coe.eval()

        wal_appended = len(session.store._wal_frames) - initial_wal_len
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return DirtyPageRefreshReport(
            dirty_files=tuple(dirty_rels),
            updated_file_addresses=tuple(updated_file_addrs),
            updated_symbol_addresses=tuple(updated_sym_addrs),
            added_symbols=tuple(added_symbols),
            modified_symbols=tuple(modified_symbols),
            removed_symbols=tuple(removed_symbols),
            wal_frames_appended=wal_appended,
            recompressed_pages=len(refresh_cosines),
            mean_refresh_cosine=(
                sum(refresh_cosines) / len(refresh_cosines)
                if refresh_cosines
                else session.mean_turboquant_cosine
            ),
            micro_warmup_steps=actual_micro_steps,
            micro_warmup_loss=round(micro_loss_val, 4),
            elapsed_ms=round(elapsed_ms, 2),
        )


def synthesize_repository_uta_traces(
    repo_root: str | Path,
    *,
    subsystems: Sequence[str] = ("src",),
    max_symbol_lines: int = 16,
    max_episodes: int = 512,
) -> list[tuple[str, str, str]]:
    """Synthesize AST-grounded `(prompt, thought, answer)` codebase episodes from `repo_root`.

    Generates both:
    1. **Architecture & Call-Graph Localization traces**: `#U$#` asks to locate and explain a module/symbol;
       `#T$#` traces the `A.TXY` address, call-graph edges, and focal invariant; `#A$#` provides the
       grounded symbol explanation and source implementation.
    2. **Bug Discrimination & Invariant Repair traces**: `#U$#` presents a candidate mutated line;
       `#T$#` identifies the violated focal invariant against the AST ground truth; `#A$#` provides
       the verified repair.
    """
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        return []
    store = index_python_repo_to_atxy(root, subsystems=subsystems)
    file_recs = [r for r in store.records() if r.parsed_address.A == 2]
    if not file_recs and tuple(subsystems) == ("src",):
        store = index_python_repo_to_atxy(root, subsystems=(".",))
        file_recs = [r for r in store.records() if r.parsed_address.A == 2]
    episodes: list[tuple[str, str, str]] = []

    for frec in file_recs:
        fpath = root / frec.payload
        if not fpath.is_file():
            continue
        src_text = fpath.read_text(encoding="utf-8", errors="replace")
        src_lines = src_text.splitlines()
        try:
            tree = ast.parse(src_text)
            ast_map = {
                node.name: node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
        except SyntaxError:
            ast_map = {}

        for s_addr in frec.down:
            srec = store.lookup(s_addr)
            if srec is None:
                continue
            sym_name = srec.payload.split(":")[-1]
            node = ast_map.get(sym_name)
            if node is not None:
                start_ln = max(0, int(getattr(node, "lineno", 1)) - 1)
                end_ln = min(
                    len(src_lines),
                    int(getattr(node, "end_lineno", start_ln + max_symbol_lines)),
                    start_ln + max_symbol_lines,
                )
                snippet = "\n".join(src_lines[start_ln:end_ln]).strip()
            else:
                snippet = f"def {sym_name}(): # {srec.summary}"

            _, focal_stmt = _extract_focal_line_and_context(snippet)
            focal_inv = _summarize_code_invariants(focal_stmt)
            callee_addrs = srec.rels.get("CALLS", ())
            callee_names = [
                store.lookup(ca).payload.split(":")[-1]
                for ca in callee_addrs[:4]
                if store.lookup(ca) is not None
            ]
            callee_str = (
                f" Downstream callees: {', '.join(callee_names)}."
                if callee_names
                else ""
            )

            # 1. Architecture & Call-Graph Localization Episode
            u_arch = f"Locate `{sym_name}` in `{frec.payload}` and explain its implementation invariant."
            t_arch = (
                f"Target `{frec.payload}:{sym_name}` at A.TXY `{srec.address}` ({srec.summary}). "
                f"Focal statement: `{focal_stmt}`. Invariant: {focal_inv}.{callee_str}"
            )
            a_arch = (
                f"`{sym_name}` in `{frec.payload}` ({srec.summary}) enforces {focal_inv} via `{focal_stmt}`:\n"
                f"{snippet}"
            )
            episodes.append((u_arch, t_arch, a_arch))
            if len(episodes) >= max_episodes:
                return episodes

            # 2. Bug Verification & Repair Episode
            muts = _mutate_focal_line(focal_stmt, mode="train")
            if muts:
                bug_line = muts[0]
                u_bug = (
                    f"In `{frec.payload}:{sym_name}`, verify whether `{bug_line}` preserves the symbol's invariant."
                )
                t_bug = (
                    f"Checking `{frec.payload}:{sym_name}` (`{srec.address}`): required invariant is {focal_inv}. "
                    f"Candidate `{bug_line}` corrupts the authentic focal statement `{focal_stmt}`."
                )
                a_bug = (
                    f"`{bug_line}` violates the invariant of `{sym_name}` ({focal_inv}). "
                    f"Restore `{focal_stmt}` in `{frec.payload}`:\n{snippet}"
                )
                episodes.append((u_bug, t_bug, a_bug))
                if len(episodes) >= max_episodes:
                    return episodes

    return episodes


# ---------------------------------------------------------------------------
# Tier 5: Windowless 1M+ Demand-Paged Context & Zero-Forgetting Working Ledger
# ---------------------------------------------------------------------------


@dataclass
class WorkingStateLedger:
    """Zero-forgetting pinned working-state ledger permanently anchored in `CED` `n_prefix` slot 0.

    Stores the model's current high-level objective, active plan step, and verified key-value
    invariants/symbols. Only re-encodes when modified (`_dirty = True`), and its compressed
    `EncodedState` is prepended to `gist_state` on every forward pass so the model physically
    cannot forget what it is working on even after millions of streamed tokens.
    """

    objective: str = ""
    active_plan_step: str = ""
    active_step: str = ""
    pinned_facts: dict[str, str] = field(default_factory=dict)
    _cached_compressed: CompressedEncodedState | None = field(default=None, repr=False)
    _dirty: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        if self.active_step and not self.active_plan_step:
            self.active_plan_step = self.active_step
        elif self.active_plan_step and not self.active_step:
            self.active_step = self.active_plan_step

    @property
    def compressed_state(self) -> CompressedEncodedState | None:
        return self._cached_compressed

    def update(
        self,
        *,
        objective: str | None = None,
        active_plan_step: str | None = None,
        active_step: str | None = None,
        facts: dict[str, str] | None = None,
    ) -> None:
        if objective is not None and objective != self.objective:
            self.objective = objective
            self._dirty = True
        step_val = active_plan_step if active_plan_step is not None else active_step
        if step_val is not None and step_val != self.active_plan_step:
            self.active_plan_step = step_val
            self.active_step = step_val
            self._dirty = True
        if facts:
            for k, v in facts.items():
                if self.pinned_facts.get(k) != v:
                    self.pinned_facts[str(k)] = str(v)
                    self._dirty = True

    def is_empty(self) -> bool:
        return (
            not (self.objective.strip() or self.active_plan_step.strip() or self.pinned_facts)
            and self._cached_compressed is None
        )

    def to_text(self) -> str:
        parts: list[str] = []
        if self.objective.strip():
            parts.append(f"[OBJECTIVE] {self.objective.strip()}")
        if self.active_plan_step.strip():
            parts.append(f"[STEP] {self.active_plan_step.strip()}")
        if self.pinned_facts:
            fact_str = "; ".join(f"{k}={v}" for k, v in sorted(self.pinned_facts.items()))
            parts.append(f"[INVARIANTS] {fact_str}")
        return " | ".join(parts)

    def format_ledger_text(self) -> str:
        return self.to_text()

    @torch.no_grad()
    def encode_from_tokens(
        self,
        backbone: MCL4CEDBackbone,
        token_ids: Tensor,
        *,
        seed: int = 42,
    ) -> EncodedState:
        device = backbone.token_embedding.weight.device
        inp = token_ids.to(device=device, dtype=torch.long)
        if inp.ndim == 1:
            inp = inp.unsqueeze(0)
        elif inp.shape[0] > 1:
            inp = inp[:1]
        val = inp != backbone.config.pad_id
        if not val.any():
            val[:, 0] = True
        enc = backbone.encode_state(inp, val, causal=False)
        self._cached_compressed = compress_encoded_state(enc, seed=seed)
        self._dirty = False
        return enc

    def get_encoded_state(
        self,
        backbone: MCL4CEDBackbone,
        *,
        max_ledger_tokens: int = 256,
        seed: int = 42,
    ) -> EncodedState | None:
        if self.is_empty():
            return None
        device = backbone.token_embedding.weight.device
        if not self._dirty and self._cached_compressed is not None:
            dec = self._cached_compressed.decompress()
            return EncodedState(
                hidden=dec.hidden.to(device=device, dtype=backbone.token_embedding.weight.dtype),
                valid=dec.valid.to(device=device),
                aux_loss=dec.aux_loss.to(device=device),
                router_z_loss=dec.router_z_loss.to(device=device),
                expert_load=dec.expert_load.to(device=device),
            )
        tok = get_tokenizer(backbone.config.vocab_size)
        ids = tok.encode(self.to_text())[: max(8, int(max_ledger_tokens))]
        if not ids:
            return None
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        return self.encode_from_tokens(backbone, inp, seed=seed)


@dataclass
class PagedMemoryEntry:
    """Single demand-paged context page stored in `InfiniteContextPager`."""

    page_id: int
    token_span: tuple[int, int]
    compressed_state: CompressedEncodedState
    landmark_keys: CompressedKV
    source_tag: str = ""
    sha256: str = ""


class InfiniteContextPager:
    """Demand-paged windowless context manager for $1\\text{M}\\to\\infty$ effective context length.

    Maintains:
    1. A `WorkingStateLedger` permanently pinned in `CED` `n_prefix` slot 0 (zero-forgetting task state).
    2. An unbounded table of `TurboQuant`-compressed `PagedMemoryEntry` pages (`page_size` tokens each,
       encoded with local RoPE `[0 .. page_size]` so positional embeddings never go out of distribution).
    3. Multi-landmark `TurboQuant` QJL inner-product routing (`retrieve_top_k_pages`) that scores
       thousands of pages in a single vectorized matmul and decompresses ONLY the top-`B` relevant pages
       into position-invariant `CED` cross-attention (`n_prefix`), preventing softmax entropy dilution.
    """

    def __init__(
        self,
        *,
        page_size: int = 512,
        landmarks_per_page: int = 4,
        top_k_pages: int = 4,
        working_state: WorkingStateLedger | None = None,
        seed: int = 42,
    ) -> None:
        if page_size < 4:
            raise ValueError("page_size must be >= 4")
        if landmarks_per_page < 1:
            raise ValueError("landmarks_per_page must be >= 1")
        if top_k_pages < 1:
            raise ValueError("top_k_pages must be >= 1")
        self.page_size = int(page_size)
        self.landmarks_per_page = int(landmarks_per_page)
        self.top_k_pages = int(top_k_pages)
        self.seed = int(seed)
        self.ledger = working_state if working_state is not None else WorkingStateLedger()
        self.working_state = self.ledger
        self.pages: list[PagedMemoryEntry] = []
        self._sha_to_page_id: dict[str, int] = {}
        self.total_ingested_tokens: int = 0

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def total_compressed_bytes(self) -> int:
        tot = sum(
            p.compressed_state.compressed_bytes + p.landmark_keys.packed_logical_bytes
            for p in self.pages
        )
        if self.ledger._cached_compressed is not None:
            tot += self.ledger._cached_compressed.compressed_bytes
        return tot

    @property
    def total_uncompressed_bytes(self) -> int:
        tot = sum(p.compressed_state.uncompressed_bytes for p in self.pages)
        if self.ledger._cached_compressed is not None:
            tot += self.ledger._cached_compressed.uncompressed_bytes
        return tot

    @torch.no_grad()
    def update_working_state(
        self,
        backbone: MCL4CEDBackbone,
        ledger_tokens: Tensor | None = None,
        *,
        objective: str | None = None,
        active_step: str | None = None,
        facts: dict[str, str] | None = None,
    ) -> EncodedState | None:
        """Update and compress the pinned `WorkingStateLedger`."""
        if objective is not None or active_step is not None or facts is not None:
            self.ledger.update(objective=objective, active_step=active_step, facts=facts)
        if ledger_tokens is not None:
            return self.ledger.encode_from_tokens(backbone, ledger_tokens, seed=self.seed)
        return self.ledger.get_encoded_state(backbone, seed=self.seed)

    def _compute_page_landmarks(self, hidden: Tensor, valid: Tensor) -> Tensor:
        """Extract `G = landmarks_per_page` multi-resolution landmark vectors `[B, G, D]` from a page `[B, L, D]`."""
        bsz, seq_len, dim = hidden.shape
        g = min(self.landmarks_per_page, seq_len)
        v_f = valid.to(hidden.dtype).unsqueeze(-1)
        masked_h = hidden * v_f
        landmarks: list[Tensor] = []
        boundaries = torch.linspace(0, seq_len, steps=g + 1, device=hidden.device).long()
        for idx in range(g):
            s = int(boundaries[idx].item())
            e = max(s + 1, int(boundaries[idx + 1].item()))
            sub_h = masked_h[:, s:e, :]
            sub_v = v_f[:, s:e, :].sum(dim=1).clamp_min(1.0)
            landmarks.append(sub_h.sum(dim=1) / sub_v)
        while len(landmarks) < self.landmarks_per_page:
            landmarks.append(landmarks[-1])
        return F.normalize(torch.stack(landmarks, dim=1).float(), p=2.0, dim=-1).to(hidden.dtype)

    @torch.no_grad()
    def ingest_tokens(
        self,
        backbone: MCL4CEDBackbone,
        token_ids: Tensor | Sequence[int],
        *,
        source_tag: str = "",
        deduplicate: bool = True,
    ) -> list[int]:
        """Slice `token_ids` into `page_size` pages, encode with local RoPE `[0..page_size]`, and store compressed."""
        device = backbone.token_embedding.weight.device
        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim >= 2:
                flat_ids = token_ids[0].detach().reshape(-1).to(device=device, dtype=torch.long)
            else:
                flat_ids = token_ids.detach().reshape(-1).to(device=device, dtype=torch.long)
        else:
            flat_ids = torch.tensor(list(token_ids), dtype=torch.long, device=device)

        n_tok = int(flat_ids.numel())
        if n_tok == 0:
            return []

        added_page_ids: list[int] = []
        for start in range(0, n_tok, self.page_size):
            end = min(n_tok, start + self.page_size)
            chunk = flat_ids[start:end]
            raw_bytes = chunk.detach().cpu().numpy().tobytes()
            digest = hashlib.sha256(raw_bytes).hexdigest()[:24]
            if deduplicate and digest in self._sha_to_page_id:
                added_page_ids.append(self._sha_to_page_id[digest])
                continue

            chunk_2d = chunk.unsqueeze(0)
            valid_2d = chunk_2d != backbone.config.pad_id
            if not valid_2d.any():
                valid_2d[:, 0] = True
            enc = backbone.encode_state(chunk_2d, valid_2d, causal=True)
            landmarks = self._compute_page_landmarks(enc.hidden, enc.valid)
            comp_state = compress_encoded_state(enc, seed=self.seed + len(self.pages))
            comp_landmarks = turboquant_compress(landmarks, seed=self.seed)

            pid = len(self.pages)
            span = (self.total_ingested_tokens + start, self.total_ingested_tokens + end)
            entry = PagedMemoryEntry(
                page_id=pid,
                token_span=span,
                compressed_state=comp_state,
                landmark_keys=comp_landmarks,
                source_tag=source_tag or f"page_{pid:05d}",
                sha256=digest,
            )
            self.pages.append(entry)
            self._sha_to_page_id[digest] = pid
            added_page_ids.append(pid)

        self.total_ingested_tokens += n_tok
        return added_page_ids

    @torch.no_grad()
    def _resolve_query_hidden(
        self,
        backbone_or_query: Any,
        query: Tensor | None = None,
        query_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if isinstance(backbone_or_query, torch.Tensor):
            q_hid = backbone_or_query
            return q_hid, query_valid
        backbone = backbone_or_query
        assert query is not None
        if query.ndim == 2 and query.dtype in (torch.long, torch.int64, torch.int32):
            val = query != backbone.config.pad_id if query_valid is None else query_valid
            if not val.any(dim=-1).all():
                val = val.clone()
                val[:, 0] = True
            enc = backbone.encode_state(query, val, causal=True)
            return enc.hidden, enc.valid
        return query, query_valid

    @torch.no_grad()
    def retrieve_top_k_pages(
        self,
        backbone_or_query: Any,
        query_or_valid: Tensor | None = None,
        query_valid: Tensor | None = None,
        *,
        top_k: int | None = None,
        always_include_recent: int = 1,
    ) -> list[int]:
        """Score all compressed pages via `turboquant_inner_product` and return top-`k` page IDs in chronological order."""
        if not self.pages:
            return []
        k_req = min(len(self.pages), max(1, int(top_k if top_k is not None else self.top_k_pages)))
        if len(self.pages) <= k_req:
            return [p.page_id for p in self.pages]

        if isinstance(backbone_or_query, torch.Tensor):
            query_hidden = backbone_or_query
            q_val = query_or_valid
        else:
            query_hidden, q_val = self._resolve_query_hidden(
                backbone_or_query, query_or_valid, query_valid
            )

        if q_val is not None:
            v_f = q_val.to(query_hidden.dtype).unsqueeze(-1)
            q_mean = (query_hidden * v_f).sum(dim=1) / v_f.sum(dim=1).clamp_min(1.0)
            q_tail = query_hidden[:, -min(64, query_hidden.shape[1]) :, :].mean(dim=1)
        else:
            q_mean = query_hidden.mean(dim=1)
            q_tail = query_hidden[:, -min(64, query_hidden.shape[1]) :, :].mean(dim=1)

        q_vecs = F.normalize(
            torch.stack([q_mean.mean(dim=0), q_tail.mean(dim=0)], dim=0).float(),
            p=2.0,
            dim=-1,
        )  # [2, D]

        page_scores: list[tuple[float, int]] = []
        n_recent = max(0, min(int(always_include_recent), k_req - 1))
        recent_set = {p.page_id for p in self.pages[-n_recent:]} if n_recent > 0 else set()

        for entry in self.pages:
            if entry.page_id in recent_set:
                continue
            sim = turboquant_inner_product(
                q_vecs.unsqueeze(0), entry.landmark_keys
            )  # [1, 2, G]
            score = float(sim.max().item())
            page_scores.append((score, entry.page_id))

        page_scores.sort(key=lambda item: item[0], reverse=True)
        chosen = [pid for _, pid in page_scores[: k_req - len(recent_set)]] + list(recent_set)
        return sorted(set(chosen))

    @torch.no_grad()
    def build_gist_state(
        self,
        backbone: MCL4CEDBackbone,
        query: Tensor,
        query_valid: Tensor | None = None,
        *,
        top_k: int | None = None,
    ) -> EncodedState | None:
        """Assemble `[WorkingStateLedger (Pinned) || Top-B Demand-Paged Pages]` as a single position-invariant `EncodedState`."""
        query_hidden, q_val = self._resolve_query_hidden(backbone, query, query_valid)
        device = query_hidden.device
        dtype = query_hidden.dtype
        bsz = int(query_hidden.shape[0])

        chunks_h: list[Tensor] = []
        chunks_v: list[Tensor] = []
        aux_list: list[Tensor] = []
        z_list: list[Tensor] = []
        loads: list[Tensor] = []

        pinned_enc = self.ledger.get_encoded_state(backbone, seed=self.seed)
        if pinned_enc is not None:
            ph = pinned_enc.hidden.to(device=device, dtype=dtype)
            pv = pinned_enc.valid.to(device=device, dtype=torch.bool)
            if ph.shape[0] == 1 and bsz > 1:
                ph = ph.expand(bsz, -1, -1)
                pv = pv.expand(bsz, -1)
            chunks_h.append(ph)
            chunks_v.append(pv)
            aux_list.append(pinned_enc.aux_loss.to(device=device))
            z_list.append(pinned_enc.router_z_loss.to(device=device))
            loads.append(pinned_enc.expert_load.to(device=device))

        selected_pids = self.retrieve_top_k_pages(
            query_hidden, q_val, top_k=top_k
        )
        for pid in selected_pids:
            dec = self.pages[pid].compressed_state.decompress()
            dh = dec.hidden.to(device=device, dtype=dtype)
            dv = dec.valid.to(device=device, dtype=torch.bool)
            if dh.shape[0] == 1 and bsz > 1:
                dh = dh.expand(bsz, -1, -1)
                dv = dv.expand(bsz, -1)
            chunks_h.append(dh)
            chunks_v.append(dv)
            aux_list.append(dec.aux_loss.to(device=device))
            z_list.append(dec.router_z_loss.to(device=device))
            loads.append(dec.expert_load.to(device=device))

        if not chunks_h:
            return None

        return EncodedState(
            hidden=torch.cat(chunks_h, dim=1),
            valid=torch.cat(chunks_v, dim=1),
            aux_loss=torch.stack(aux_list).mean(),
            router_z_loss=torch.stack(z_list).mean(),
            expert_load=torch.stack(loads).mean(dim=0),
        )




