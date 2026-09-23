"""Fixed-graph local multi-head attention for TLS stem detection.

Author: Shahab Alaedin Baloochi

The block refines the concatenated EdgeConv representation over the supplied
fixed k-NN graph. Attention combines scaled query-key similarity with a
learnable bias from relative XYZ and adds the update through a residual
connection. Head count, head width, and positional-bias network widths are
constructor parameters.

For an edge j -> i:

    score_ij^h = <q_i^h, k_j^h> / sqrt(d_h) + b_h(p_j - p_i)
    alpha_ij^h = softmax_j(score_ij^h)
    a_i^h = sum_j alpha_ij^h v_j^h
    y_i = x_i + W_o concat_h(a_i^h)

No graph reconstruction or learned-space neighbour search is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LocalAttentionOutput:
    """Refined features plus optional edge-wise attention weights."""

    refined: Tensor
    attention_weights: Tensor


def make_relative_position_bias_mlp(
    num_heads: int,
    hidden_channels: Sequence[int],
    *,
    activation_factory: Callable[[], nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a learnable relative-XYZ to per-head attention-bias network.

    Hidden widths are supplied by the caller.
    """
    if num_heads <= 0:
        raise ValueError("num_heads must be positive.")
    hidden = tuple(int(c) for c in hidden_channels)
    if any(c <= 0 for c in hidden):
        raise ValueError("All hidden_channels must be positive.")

    widths = (3, *hidden, int(num_heads))
    layers: list[nn.Module] = []
    for idx, (c_in, c_out) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(c_in, c_out))
        if idx < len(widths) - 2:
            layers.append(activation_factory())
    return nn.Sequential(*layers)


class FixedLocalMultiheadAttention(nn.Module):
    """Multi-head local self-attention on a precomputed fixed graph.

    ``x`` contains the multi-scale point representation from the EdgeConv
    backbone. ``positions`` contains metric XYZ coordinates (tree-local or
    global coordinates are both valid because only relative differences are
    used). ``edge_index`` is the already-computed fixed graph and is consumed
    without modification.

    The attention update is projected back to ``in_channels`` and added to the
    original ``x`` through a residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        head_dim: int,
        position_bias_mlp: nn.Module,
        *,
        qkv_bias: bool = False,
        out_bias: bool = False,
        attention_dropout: float = 0.0,
        output_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive.")
        if not isinstance(position_bias_mlp, nn.Module):
            raise TypeError("position_bias_mlp must be a torch.nn.Module.")
        for name, value in (
            ("attention_dropout", attention_dropout),
            ("output_dropout", output_dropout),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must satisfy 0 <= p < 1.")

        self.in_channels = int(in_channels)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_channels = self.num_heads * self.head_dim
        self.scale = 1.0 / sqrt(float(self.head_dim))

        self.query = nn.Linear(self.in_channels, self.inner_channels, bias=qkv_bias)
        self.key = nn.Linear(self.in_channels, self.inner_channels, bias=qkv_bias)
        self.value = nn.Linear(self.in_channels, self.inner_channels, bias=qkv_bias)
        self.position_bias_mlp = position_bias_mlp
        self.output_projection = nn.Linear(
            self.inner_channels,
            self.in_channels,
            bias=out_bias,
        )
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.output_dropout = nn.Dropout(float(output_dropout))

    def _validate_inputs(
        self,
        x: Tensor,
        positions: Tensor,
        edge_index: Tensor,
    ) -> tuple[int, int]:
        if not isinstance(x, Tensor) or not isinstance(positions, Tensor):
            raise TypeError("x and positions must be torch tensors.")
        if not isinstance(edge_index, Tensor):
            raise TypeError("edge_index must be a torch tensor.")
        if x.ndim != 2:
            raise ValueError(f"x must have shape (N, C), got {tuple(x.shape)}.")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (N, 3), got {tuple(positions.shape)}."
            )
        if x.shape[0] != positions.shape[0]:
            raise ValueError("x and positions must contain the same number of nodes.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )
        if x.shape[0] <= 0:
            raise ValueError("x must contain at least one node.")
        if not x.is_floating_point() or not positions.is_floating_point():
            raise TypeError("x and positions must be floating point.")
        if x.dtype != positions.dtype:
            raise TypeError("x and positions must use the same floating-point dtype.")
        if x.device != positions.device or x.device != edge_index.device:
            raise ValueError("x, positions and edge_index must be on the same device.")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or infinite values.")
        if not torch.isfinite(positions).all():
            raise ValueError("positions contains NaN or infinite values.")

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}."
            )
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise TypeError("edge_index must use an integer dtype (int32 or int64).")

        num_nodes = int(x.shape[0])
        num_edges = int(edge_index.shape[1])
        if num_edges:
            if torch.any(edge_index < 0) or torch.any(edge_index >= num_nodes):
                raise ValueError("edge_index contains an out-of-range node index.")
            if torch.any(edge_index[0] == edge_index[1]):
                raise ValueError("edge_index contains self-edges; fixed k-NN excludes self.")
        return num_nodes, num_edges

    @staticmethod
    def _segment_softmax(scores: Tensor, target: Tensor, num_nodes: int) -> Tensor:
        """Stable softmax over incoming edges, independently per node and head."""
        if scores.ndim != 2:
            raise ValueError("scores must have shape (E, H).")
        num_edges, num_heads = scores.shape
        if num_edges == 0:
            return scores

        scatter_index = target[:, None].expand(-1, num_heads)
        maxima = scores.new_full((num_nodes, num_heads), -torch.inf)
        maxima.scatter_reduce_(
            0,
            scatter_index,
            scores,
            reduce="amax",
            include_self=True,
        )
        shifted = scores - maxima[target]
        exponentials = torch.exp(shifted)
        denominator = scores.new_zeros((num_nodes, num_heads))
        denominator.scatter_add_(0, scatter_index, exponentials)
        tiny = torch.finfo(scores.dtype).tiny
        return exponentials / denominator[target].clamp_min(tiny)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | LocalAttentionOutput:
        num_nodes, num_edges = self._validate_inputs(x, positions, edge_index)

        if num_edges == 0:
            empty_attention = x.new_empty((0, self.num_heads))
            if return_attention:
                return LocalAttentionOutput(refined=x, attention_weights=empty_attention)
            return x

        source = edge_index[0].to(dtype=torch.long)
        target = edge_index[1].to(dtype=torch.long)

        q = self.query(x).reshape(num_nodes, self.num_heads, self.head_dim)
        k = self.key(x).reshape(num_nodes, self.num_heads, self.head_dim)
        v = self.value(x).reshape(num_nodes, self.num_heads, self.head_dim)

        relative_xyz = positions[source] - positions[target]
        position_bias = self.position_bias_mlp(relative_xyz)
        if position_bias.ndim != 2 or position_bias.shape != (num_edges, self.num_heads):
            raise RuntimeError(
                "position_bias_mlp must return shape "
                f"(E, num_heads)=({num_edges}, {self.num_heads}), got "
                f"{tuple(position_bias.shape)}."
            )
        if not position_bias.is_floating_point():
            raise RuntimeError("position_bias_mlp must return floating-point values.")
        if position_bias.device != x.device:
            raise RuntimeError("position_bias_mlp output must be on the same device as x.")
        if not torch.isfinite(position_bias).all():
            raise RuntimeError("position_bias_mlp produced NaN or infinite values.")

        content_scores = (q[target] * k[source]).sum(dim=-1) * self.scale
        # Under automatic mixed precision, the projection and positional MLP
        # may legitimately run in a lower-precision floating dtype even when
        # the residual input ``x`` remains float32. Align only the score dtype
        # here instead of rejecting AMP-compatible execution.
        position_bias = position_bias.to(dtype=content_scores.dtype)
        scores = content_scores + position_bias
        attention = self._segment_softmax(scores, target, num_nodes)
        dropped_attention = self.attention_dropout(attention)

        weighted_values = dropped_attention[:, :, None] * v[source]
        aggregated = v.new_zeros((num_nodes, self.num_heads, self.head_dim))
        scatter_index = target[:, None, None].expand(
            -1,
            self.num_heads,
            self.head_dim,
        )
        aggregated.scatter_add_(0, scatter_index, weighted_values)

        update = self.output_projection(aggregated.reshape(num_nodes, self.inner_channels))
        refined = x + self.output_dropout(update)

        if not torch.isfinite(refined).all():
            raise RuntimeError("Local attention produced NaN or infinite values.")

        if return_attention:
            return LocalAttentionOutput(refined=refined, attention_weights=attention)
        return refined


__all__ = [
    "FixedLocalMultiheadAttention",
    "LocalAttentionOutput",
    "make_relative_position_bias_mlp",
]
