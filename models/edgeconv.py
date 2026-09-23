"""Fixed-graph EdgeConv backbone for TLS stem detection.

Author: Shahab Alaedin Baloochi

Edge messages use [x_i, x_j - x_i] and are aggregated with a channel-wise
maximum over the supplied fixed graph. The module does not recompute k-NN
neighbourhoods. Three EdgeConv layer outputs are concatenated for the
multi-scale representation. Message-MLP widths are configured by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EdgeConvBackboneOutput:
    """Outputs of the three-layer EdgeConv backbone."""

    multi_scale: Tensor
    layer_outputs: tuple[Tensor, Tensor, Tensor]


def make_message_mlp(
    in_channels: int,
    hidden_channels: Sequence[int],
    out_channels: int,
    *,
    activation_factory: Callable[[], nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build an EdgeConv message MLP.

    The pair representation has 2 * in_channels channels. Hidden widths are
    supplied by the caller.
    """
    if in_channels <= 0 or out_channels <= 0:
        raise ValueError("in_channels and out_channels must be positive.")

    hidden = tuple(int(c) for c in hidden_channels)
    if any(c <= 0 for c in hidden):
        raise ValueError("All hidden_channels must be positive.")

    widths = (2 * int(in_channels), *hidden, int(out_channels))
    layers: list[nn.Module] = []
    for idx, (c_in, c_out) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(c_in, c_out))
        if idx < len(widths) - 2:
            layers.append(activation_factory())
    return nn.Sequential(*layers)


class FixedEdgeConv(nn.Module):
    """EdgeConv on a precomputed, fixed graph.

    For a directed edge ``j -> i`` represented by ``edge_index[:, e] = [j, i]``,
    the pairwise message input is

    ``[x_i, x_j - x_i]``.

    The supplied ``message_mlp`` maps that pair representation to
    ``out_channels`` values. Messages arriving at each target node are then
    reduced with a channel-wise maximum.

    No graph-building routine exists in this class. In particular, there is no
    dynamic k-NN recomputation in learned feature space.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        message_mlp: nn.Module,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if not isinstance(message_mlp, nn.Module):
            raise TypeError("message_mlp must be a torch.nn.Module.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.message_mlp = message_mlp

    @staticmethod
    def _validate_inputs(x: Tensor, edge_index: Tensor) -> tuple[int, int]:
        if not isinstance(x, Tensor) or not isinstance(edge_index, Tensor):
            raise TypeError("x and edge_index must be torch tensors.")
        if x.ndim != 2:
            raise ValueError(f"x must have shape (N, C), got {tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("x must be floating point.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}."
            )
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise TypeError("edge_index must use an integer dtype (int32 or int64).")
        if edge_index.device != x.device:
            raise ValueError("x and edge_index must be on the same device.")

        num_nodes = int(x.shape[0])
        num_edges = int(edge_index.shape[1])
        if num_nodes <= 0:
            raise ValueError("x must contain at least one node.")

        if num_edges:
            if torch.any(edge_index < 0) or torch.any(edge_index >= num_nodes):
                raise ValueError("edge_index contains an out-of-range node index.")
            if torch.any(edge_index[0] == edge_index[1]):
                raise ValueError("edge_index contains self-edges; fixed k-NN excludes self.")
        return num_nodes, num_edges

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes, num_edges = self._validate_inputs(x, edge_index)
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )

        # A singleton component can legitimately have no internal graph edge.
        # We do not add a synthetic self-edge because the manuscript uses the
        # fixed precomputed graph. Its EdgeConv response is therefore zero.
        if num_edges == 0:
            return x.new_zeros((num_nodes, self.out_channels))

        source = edge_index[0].to(dtype=torch.long)
        target = edge_index[1].to(dtype=torch.long)
        x_i = x[target]
        x_j = x[source]
        pair_features = torch.cat((x_i, x_j - x_i), dim=-1)

        messages = self.message_mlp(pair_features)
        if messages.ndim != 2:
            raise RuntimeError(
                "message_mlp must return a rank-2 tensor with shape (E, out_channels)."
            )
        if messages.shape != (num_edges, self.out_channels):
            raise RuntimeError(
                "message_mlp returned shape "
                f"{tuple(messages.shape)}; expected ({num_edges}, {self.out_channels})."
            )
        if not messages.is_floating_point():
            raise RuntimeError("message_mlp must return floating-point messages.")

        # Channel-wise max over all neighbours j that send a message to target i.
        output = messages.new_full(
            (num_nodes, self.out_channels),
            -torch.inf,
        )
        scatter_index = target[:, None].expand(-1, self.out_channels)
        output.scatter_reduce_(
            dim=0,
            index=scatter_index,
            src=messages,
            reduce="amax",
            include_self=True,
        )

        # No incoming edge means an isolated node in the current local subgraph.
        # Keep it finite without changing the fixed graph by assigning zero.
        incoming = torch.bincount(target, minlength=num_nodes)
        isolated = incoming == 0
        if torch.any(isolated):
            output = torch.where(isolated[:, None], torch.zeros_like(output), output)

        return output


class ThreeLayerFixedEdgeConvBackbone(nn.Module):
    """Three fixed-graph EdgeConv layers with multi-scale concatenation.

    Layer ``l+1`` receives the output of layer ``l``. The same ``edge_index`` is
    passed unchanged to every layer. The three layer outputs are concatenated
    along the feature dimension, matching the multi-scale aggregation described
    in the manuscript.
    """

    def __init__(
        self,
        layer1: FixedEdgeConv,
        layer2: FixedEdgeConv,
        layer3: FixedEdgeConv,
    ) -> None:
        super().__init__()
        if layer2.in_channels != layer1.out_channels:
            raise ValueError("layer2.in_channels must equal layer1.out_channels.")
        if layer3.in_channels != layer2.out_channels:
            raise ValueError("layer3.in_channels must equal layer2.out_channels.")

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    @property
    def in_channels(self) -> int:
        return self.layer1.in_channels

    @property
    def multi_scale_channels(self) -> int:
        return (
            self.layer1.out_channels
            + self.layer2.out_channels
            + self.layer3.out_channels
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> EdgeConvBackboneOutput:
        h1 = self.layer1(x, edge_index)
        h2 = self.layer2(h1, edge_index)
        h3 = self.layer3(h2, edge_index)
        multi_scale = torch.cat((h1, h2, h3), dim=-1)
        return EdgeConvBackboneOutput(
            multi_scale=multi_scale,
            layer_outputs=(h1, h2, h3),
        )


__all__ = [
    "EdgeConvBackboneOutput",
    "FixedEdgeConv",
    "ThreeLayerFixedEdgeConvBackbone",
    "make_message_mlp",
]
