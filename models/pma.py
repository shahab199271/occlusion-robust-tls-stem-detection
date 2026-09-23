"""Pooling by Multihead Attention (PMA) for subgraph-level context.

Author: Shahab Alaedin Baloochi

Refined point embeddings are treated as an unordered set and pooled with two
learned seed queries. The two seed descriptors are concatenated and passed to
FiLM. The implementation follows the Set Transformer PMA/MAB structure with a
residual row-wise feed-forward update. use_layer_norm remains configurable,
and no post-PMA SAB is added.

PMA operates independently within each connected subgraph and does not modify
the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PMAOutput:
    """PMA seed descriptors, their concatenation, and optional attention maps."""

    seed_descriptors: Tensor
    context: Tensor
    attention_weights: Tensor


class PoolingByMultiheadAttention(nn.Module):
    """Set Transformer-style PMA with learned seed queries.

    Parameters
    ----------
    model_dim:
        Channel dimension of each refined point embedding.
    num_heads:
        Number of attention heads. ``model_dim`` must be divisible by it.
    use_layer_norm:
        Selects whether the two MAB LayerNorm operations are used.
    num_seeds:
        Number of learned PMA seed queries; the default is 2.
    qkv_bias, out_bias, feedforward_bias:
        Bias choices are explicit because they are not specified by the
        manuscript.
    attention_dropout, feedforward_dropout:
        Optional attention and feed-forward dropout probabilities.

    Input shapes
    ------------
    ``x`` can be either ``(N, D)`` for one subgraph or ``(B, N, D)`` for a batch.
    ``point_mask`` is optional and uses the intuitive convention ``True=valid``;
    it has shape ``(N,)`` or ``(B, N)`` respectively.

    Output shapes
    -------------
    For unbatched input and two seeds:
      * seed_descriptors: ``(2, D)``
      * context: ``(2 * D,)``
      * attention_weights: ``(H, 2, N)`` when requested

    For batched input:
      * seed_descriptors: ``(B, 2, D)``
      * context: ``(B, 2 * D)``
      * attention_weights: ``(B, H, 2, N)`` when requested
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        use_layer_norm: bool,
        num_seeds: int = 2,
        qkv_bias: bool = True,
        out_bias: bool = True,
        feedforward_bias: bool = True,
        attention_dropout: float = 0.0,
        feedforward_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if num_seeds <= 0:
            raise ValueError("num_seeds must be positive.")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        for name, value in (
            ("attention_dropout", attention_dropout),
            ("feedforward_dropout", feedforward_dropout),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must satisfy 0 <= p < 1.")

        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_seeds = int(num_seeds)
        self.head_dim = self.model_dim // self.num_heads
        # Lee et al. (2019) scale Set Transformer attention by sqrt(d), where d
        # is the full model dimension; the official code mirrors this choice.
        self.scale = 1.0 / sqrt(float(self.model_dim))
        self.use_layer_norm = bool(use_layer_norm)

        self.seed_vectors = nn.Parameter(torch.empty(1, self.num_seeds, self.model_dim))
        nn.init.xavier_uniform_(self.seed_vectors)

        self.query_projection = nn.Linear(self.model_dim, self.model_dim, bias=qkv_bias)
        self.key_projection = nn.Linear(self.model_dim, self.model_dim, bias=qkv_bias)
        self.value_projection = nn.Linear(self.model_dim, self.model_dim, bias=qkv_bias)
        # The formal Set Transformer multi-head definition includes W^O.
        self.output_projection = nn.Linear(self.model_dim, self.model_dim, bias=out_bias)

        # Row-wise residual feed-forward update. The official Set Transformer
        # implementation uses a same-width Linear + ReLU here, so no extra
        # hidden width is introduced.
        self.feedforward = nn.Linear(
            self.model_dim,
            self.model_dim,
            bias=feedforward_bias,
        )

        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.feedforward_dropout = nn.Dropout(float(feedforward_dropout))
        self.norm_after_attention = (
            nn.LayerNorm(self.model_dim) if self.use_layer_norm else nn.Identity()
        )
        self.norm_after_feedforward = (
            nn.LayerNorm(self.model_dim) if self.use_layer_norm else nn.Identity()
        )

    @property
    def context_channels(self) -> int:
        """Number of channels after concatenating all seed descriptors."""
        return self.num_seeds * self.model_dim

    def _validate_and_batch(
        self,
        x: Tensor,
        point_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, bool]:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch tensor.")
        if x.ndim not in (2, 3):
            raise ValueError(f"x must have shape (N,D) or (B,N,D), got {tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("x must be floating point.")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or infinite values.")

        unbatched = x.ndim == 2
        xb = x.unsqueeze(0) if unbatched else x
        batch_size, num_points, channels = map(int, xb.shape)
        if batch_size <= 0 or num_points <= 0:
            raise ValueError("x must contain at least one subgraph and one point.")
        if channels != self.model_dim:
            raise ValueError(
                f"Expected model_dim={self.model_dim}, got input dimension {channels}."
            )

        if point_mask is None:
            mask = torch.ones(
                (batch_size, num_points),
                dtype=torch.bool,
                device=xb.device,
            )
        else:
            if not isinstance(point_mask, Tensor):
                raise TypeError("point_mask must be a torch tensor when provided.")
            if point_mask.dtype != torch.bool:
                raise TypeError("point_mask must have dtype torch.bool (True=valid point).")
            if point_mask.device != xb.device:
                raise ValueError("x and point_mask must be on the same device.")
            expected_shape = (num_points,) if unbatched else (batch_size, num_points)
            if tuple(point_mask.shape) != expected_shape:
                raise ValueError(
                    f"point_mask must have shape {expected_shape}, got {tuple(point_mask.shape)}."
                )
            mask = point_mask.unsqueeze(0) if unbatched else point_mask

        valid_counts = mask.sum(dim=1)
        if torch.any(valid_counts == 0):
            bad = torch.nonzero(valid_counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(f"Every subgraph must contain at least one valid point; empty batch item(s): {bad}.")

        return xb, mask, unbatched

    def _project_heads(self, values: Tensor, projection: nn.Linear) -> Tensor:
        batch_size, length, _ = values.shape
        projected = projection(values)
        return projected.reshape(batch_size, length, self.num_heads, self.head_dim)

    def forward(
        self,
        x: Tensor,
        point_mask: Optional[Tensor] = None,
        *,
        return_attention: bool = False,
    ) -> Tensor | PMAOutput:
        """Pool one or more unordered point sets into learned seed descriptors."""
        xb, valid_mask, unbatched = self._validate_and_batch(x, point_mask)
        batch_size, num_points, _ = xb.shape

        seeds = self.seed_vectors.expand(batch_size, -1, -1)
        q = self._project_heads(seeds, self.query_projection)  # (B, K, H, Dh)
        k = self._project_heads(xb, self.key_projection)       # (B, N, H, Dh)
        v = self._project_heads(xb, self.value_projection)     # (B, N, H, Dh)

        # scores: (B, H, K, N)
        scores = torch.einsum("bkhd,bnhd->bhkn", q, k) * self.scale
        scores = scores.masked_fill(~valid_mask[:, None, None, :], -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        if not torch.isfinite(attention).all():
            raise RuntimeError("PMA attention produced NaN or infinite values.")

        dropped_attention = self.attention_dropout(attention)
        # weighted: (B, K, H, Dh)
        weighted = torch.einsum("bhkn,bnhd->bkhd", dropped_attention, v)
        multihead = weighted.reshape(batch_size, self.num_seeds, self.model_dim)
        multihead = self.output_projection(multihead)

        # MAB residual uses the learned seed queries themselves, not projected Q.
        hidden = self.norm_after_attention(seeds + multihead)
        ff_update = torch.relu(self.feedforward(hidden))
        descriptors = self.norm_after_feedforward(
            hidden + self.feedforward_dropout(ff_update)
        )

        if not torch.isfinite(descriptors).all():
            raise RuntimeError("PMA produced NaN or infinite seed descriptors.")

        context = descriptors.reshape(batch_size, self.context_channels)

        if unbatched:
            descriptors_out = descriptors[0]
            context_out = context[0]
            attention_out = attention[0] if return_attention else x.new_empty((0,))
        else:
            descriptors_out = descriptors
            context_out = context
            attention_out = attention if return_attention else x.new_empty((0,))

        if return_attention:
            return PMAOutput(
                seed_descriptors=descriptors_out,
                context=context_out,
                attention_weights=attention_out,
            )
        return context_out


__all__ = [
    "PMAOutput",
    "PoolingByMultiheadAttention",
]
