"""Per-Layer Token Embeddings (PLE) with FP32 RMSNorm and state-gated injection.

Adapted from mini (PerLayerTokenEmbedding) and oracle/ple.py (Gemma-style PLE).
Injects fresh token identity into every layer's residual stream at O(1) lookup
cost per token rather than widening O(d_model * d_ffn) expert matrices.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """FP32 root-mean-square normalization with learned scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError("eps must be positive")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        values = x.float()
        normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class PerLayerEmbedding(nn.Module):
    """Layer-local token identity lookup and state-conditioned residual gate.

    Supports both discrete token IDs (`token_ids >= 0`, with `pad_id` masked out)
    and continuous input projections (`token_ids == -1` or `None`).
    """

    def __init__(
        self,
        d_model: int,
        ple_dim: int,
        vocab_size: int = 260,
        pad_id: int = 256,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        for name, value in (
            ("d_model", d_model),
            ("ple_dim", ple_dim),
            ("vocab_size", vocab_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.d_model = d_model
        self.ple_dim = ple_dim
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.input_projection = nn.Linear(d_model, ple_dim, bias=False)
        self.projection_norm = RMSNorm(ple_dim, eps=rms_norm_eps)
        self.factorized_vocab = vocab_size > 1024
        if self.factorized_vocab:
            self.token_embedding = nn.Embedding(512, ple_dim)
            self.bucket_embedding: nn.Embedding | None = nn.Embedding(256, ple_dim)
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.bucket_embedding.weight, mean=0.0, std=0.02)
        else:
            self.token_embedding = nn.Embedding(
                vocab_size,
                ple_dim,
                padding_idx=pad_id if 0 <= pad_id < vocab_size else None,
            )
            self.bucket_embedding = None
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
            if 0 <= pad_id < vocab_size:
                with torch.no_grad():
                    self.token_embedding.weight[pad_id].zero_()

        self.ngram_conv = nn.Conv1d(ple_dim, ple_dim, kernel_size=4, groups=ple_dim, bias=False)
        self.ngram_scale = nn.Parameter(torch.full((ple_dim,), 0.15))
        self.state_gate = nn.Linear(d_model, ple_dim, bias=False)
        self.output_projection = nn.Linear(ple_dim, d_model, bias=False)
        self.output_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.gate_scale = nn.Parameter(torch.full((d_model,), 0.1))

        nn.init.constant_(self.ngram_conv.weight, 0.25)

    def embed(
        self,
        original_inputs: Tensor,
        token_ids: Tensor | None = None,
        valid: Tensor | None = None,
    ) -> Tensor:
        """Compute [batch, seq_len, ple_dim] per-layer features from immutable inputs."""
        active = (
            torch.ones(original_inputs.shape[:2], dtype=torch.bool, device=original_inputs.device)
            if valid is None
            else valid
        )
        if token_ids is not None:
            active = active & (token_ids != self.pad_id) & (token_ids >= 0)
        clean_inputs = original_inputs.masked_fill(~active.unsqueeze(-1), 0)
        projected = self.projection_norm(
            self.input_projection(clean_inputs) / math.sqrt(self.d_model)
        )
        if token_ids is not None:
            safe_ids = token_ids.clamp(0, self.vocab_size - 1)
            if self.factorized_vocab and self.bucket_embedding is not None:
                fine = self.token_embedding(safe_ids.remainder(512))
                coarse = self.bucket_embedding(
                    torch.div(safe_ids, 64, rounding_mode="floor").remainder(256)
                )
                lookup = (fine + coarse) * math.sqrt(0.5 * self.ple_dim)
            else:
                lookup = self.token_embedding(safe_ids) * math.sqrt(self.ple_dim)
            lookup = lookup.masked_fill(~active.unsqueeze(-1), 0)
            if lookup.shape[1] >= 1:
                padded = F.pad(lookup.transpose(1, 2), (3, 0))
                ngram_feat = self.ngram_conv(padded).transpose(1, 2)
                lookup = lookup + ngram_feat * self.ngram_scale.to(lookup.dtype)
            combined = (projected + lookup) / math.sqrt(2.0)
            projected = torch.where(active.unsqueeze(-1), combined, projected)
        return projected.masked_fill(~active.unsqueeze(-1), 0)

    def forward(
        self,
        state: Tensor,
        original_inputs: Tensor,
        token_ids: Tensor | None = None,
        valid: Tensor | None = None,
    ) -> Tensor:
        """Inject state-gated per-layer embedding into `state`, preserving padded rows."""
        active = (
            torch.ones(state.shape[:2], dtype=torch.bool, device=state.device)
            if valid is None
            else valid
        )
        per_layer_input = self.embed(original_inputs, token_ids=token_ids, valid=active)
        mask = active.unsqueeze(-1)
        clean_state = state.masked_fill(~mask, 0)
        gated = F.gelu(self.state_gate(clean_state), approximate="tanh") * per_layer_input
        update = (
            self.output_norm(self.output_projection(gated)).float() * self.gate_scale.float()
        ).to(state.dtype)
        return torch.where(mask, state + update, state)
