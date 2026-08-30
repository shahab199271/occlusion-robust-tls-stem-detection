"""Complete point-graph stem detector assembly.

Author: Shahab Alaedin Baloochi

This module connects the model components described in Section 2.3 of the
manuscript:

    10D point features
        -> three fixed-graph EdgeConv layers
        -> multi-scale concatenation
        -> fixed-neighbour local multi-head attention + residual
        -> PMA with two learned seed queries
        -> concatenated subgraph context
        -> feature-wise linear modulation (FiLM)
        -> point-wise MLP classifier
        -> one stem logit per point

Reproducibility boundary
------------------------
The manuscript explicitly reports:
  * 10 input channels (metric local XYZ + 7 engineered descriptors),
  * three EdgeConv layers whose outputs are concatenated,
  * local attention on the unchanged fixed Euclidean k-NN graph,
  * PMA with exactly two learned seed queries,
  * FiLM using learned channel-wise scale gamma(g) and shift beta(g),
  * classifier dimensions 448 -> 256 -> 128 -> 64 -> 1,
  * ReLU activations,
  * dropout between classifier layers, with dropout = 0.15 reported in
    the optimisation settings,
  * logits during training and sigmoid probabilities during evaluation.

The manuscript does NOT publish the individual EdgeConv widths, attention
head configuration, PMA head configuration, or the exact neural mapping used
to generate FiLM gamma/beta from the pooled context. Those details are not
invented here. Instead, the already-configured backbone, attention and PMA
modules are injected explicitly, and FiLM receives caller-supplied gamma/beta
generators. A minimal direct-linear FiLM generator helper is provided for
experimentation, but is not claimed to reproduce an unpublished original
implementation detail.

Because the reported classifier begins at 448 channels, this assembly enforces
a 448-channel refined point representation before FiLM/classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

try:
    from .edgeconv import EdgeConvBackboneOutput, ThreeLayerFixedEdgeConvBackbone
    from .local_attention import FixedLocalMultiheadAttention
    from .pma import PoolingByMultiheadAttention
except ImportError:  # allow direct execution/import from models/
    from edgeconv import EdgeConvBackboneOutput, ThreeLayerFixedEdgeConvBackbone
    from local_attention import FixedLocalMultiheadAttention
    from pma import PoolingByMultiheadAttention


PAPER_INPUT_CHANNELS = 10
PAPER_REFINED_CHANNELS = 448
PAPER_CLASSIFIER_CHANNELS = (448, 256, 128, 64, 1)
PAPER_DROPOUT = 0.15
PAPER_PMA_SEEDS = 2


@dataclass(frozen=True)
class FiLMOutput:
    """Conditioned point features and the generated FiLM parameters."""

    conditioned: Tensor
    gamma: Tensor
    beta: Tensor


@dataclass(frozen=True)
class StemDetectorOutput:
    """Optional detailed outputs from one subgraph forward pass."""

    logits: Tensor
    probabilities: Optional[Tensor]
    multi_scale_features: Tensor
    refined_features: Tensor
    global_context: Tensor
    gamma: Tensor
    beta: Tensor
    conditioned_features: Tensor


def make_linear_film_generators(
    context_channels: int,
    feature_channels: int,
    *,
    bias: bool = True,
) -> tuple[nn.Linear, nn.Linear]:
    """Create direct linear context->gamma and context->beta generators.

    This is a minimal standard FiLM realization and a convenience for
    experiments. The stem-detection manuscript states learned channel-wise
    scaling and shifting, but does not publish the original generator
    architecture; therefore this helper is not used implicitly by
    ``StemDetector``.
    """
    if context_channels <= 0 or feature_channels <= 0:
        raise ValueError("context_channels and feature_channels must be positive.")
    return (
        nn.Linear(int(context_channels), int(feature_channels), bias=bias),
        nn.Linear(int(context_channels), int(feature_channels), bias=bias),
    )


class FeatureWiseLinearModulation(nn.Module):
    """Apply FiLM conditioning: gamma(g) * h_i + beta(g).

    One context vector ``g`` is shared by all points in the current connected
    subgraph, while gamma and beta are learned channel-wise functions of that
    context.
    """

    def __init__(
        self,
        feature_channels: int,
        context_channels: int,
        gamma_generator: nn.Module,
        beta_generator: nn.Module,
    ) -> None:
        super().__init__()
        if feature_channels <= 0 or context_channels <= 0:
            raise ValueError("feature_channels and context_channels must be positive.")
        if not isinstance(gamma_generator, nn.Module):
            raise TypeError("gamma_generator must be a torch.nn.Module.")
        if not isinstance(beta_generator, nn.Module):
            raise TypeError("beta_generator must be a torch.nn.Module.")

        self.feature_channels = int(feature_channels)
        self.context_channels = int(context_channels)
        self.gamma_generator = gamma_generator
        self.beta_generator = beta_generator

    def forward(self, point_features: Tensor, context: Tensor) -> FiLMOutput:
        if not isinstance(point_features, Tensor) or not isinstance(context, Tensor):
            raise TypeError("point_features and context must be torch tensors.")
        if point_features.ndim != 2:
            raise ValueError(
                f"point_features must have shape (N,C), got {tuple(point_features.shape)}."
            )
        if context.ndim != 1:
            raise ValueError(f"context must have shape (G,), got {tuple(context.shape)}.")
        if point_features.shape[0] <= 0:
            raise ValueError("point_features must contain at least one point.")
        if point_features.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected {self.feature_channels} point-feature channels, "
                f"got {point_features.shape[1]}."
            )
        if context.shape[0] != self.context_channels:
            raise ValueError(
                f"Expected context size {self.context_channels}, got {context.shape[0]}."
            )
        if not point_features.is_floating_point() or not context.is_floating_point():
            raise TypeError("point_features and context must be floating point.")
        if point_features.device != context.device:
            raise ValueError("point_features and context must be on the same device.")
        if not torch.isfinite(point_features).all():
            raise ValueError("point_features contains NaN or infinite values.")
        if not torch.isfinite(context).all():
            raise ValueError("context contains NaN or infinite values.")

        gamma = self.gamma_generator(context)
        beta = self.beta_generator(context)

        expected = (self.feature_channels,)
        if gamma.ndim != 1 or tuple(gamma.shape) != expected:
            raise RuntimeError(
                f"gamma_generator must return shape {expected}, got {tuple(gamma.shape)}."
            )
        if beta.ndim != 1 or tuple(beta.shape) != expected:
            raise RuntimeError(
                f"beta_generator must return shape {expected}, got {tuple(beta.shape)}."
            )
        if not gamma.is_floating_point() or not beta.is_floating_point():
            raise RuntimeError("FiLM generators must return floating-point tensors.")
        if gamma.device != point_features.device or beta.device != point_features.device:
            raise RuntimeError("FiLM generator outputs must be on the feature device.")
        if not torch.isfinite(gamma).all() or not torch.isfinite(beta).all():
            raise RuntimeError("FiLM generator produced NaN or infinite values.")

        # AMP may produce lower-precision generator outputs while the residual
        # feature tensor remains float32. Align dtypes before modulation.
        gamma = gamma.to(dtype=point_features.dtype)
        beta = beta.to(dtype=point_features.dtype)
        conditioned = point_features * gamma.unsqueeze(0) + beta.unsqueeze(0)

        if not torch.isfinite(conditioned).all():
            raise RuntimeError("FiLM conditioning produced NaN or infinite values.")

        return FiLMOutput(conditioned=conditioned, gamma=gamma, beta=beta)


class StemPointClassifier(nn.Module):
    """Paper-reported point-wise MLP: 448 -> 256 -> 128 -> 64 -> 1."""

    input_channels = PAPER_REFINED_CHANNELS

    def __init__(self, dropout: float = PAPER_DROPOUT) -> None:
        super().__init__()
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= p < 1.")
        self.dropout_probability = float(dropout)

        self.network = nn.Sequential(
            nn.Linear(448, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(64, 1),
        )

    def forward(self, conditioned_features: Tensor) -> Tensor:
        if not isinstance(conditioned_features, Tensor):
            raise TypeError("conditioned_features must be a torch tensor.")
        if conditioned_features.ndim != 2:
            raise ValueError(
                "conditioned_features must have shape (N,448), got "
                f"{tuple(conditioned_features.shape)}."
            )
        if conditioned_features.shape[0] <= 0:
            raise ValueError("conditioned_features must contain at least one point.")
        if conditioned_features.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, "
                f"got {conditioned_features.shape[1]}."
            )
        if not conditioned_features.is_floating_point():
            raise TypeError("conditioned_features must be floating point.")
        if not torch.isfinite(conditioned_features).all():
            raise ValueError("conditioned_features contains NaN or infinite values.")

        logits = self.network(conditioned_features).squeeze(-1)
        if logits.shape != (conditioned_features.shape[0],):
            raise RuntimeError("Classifier did not produce one logit per point.")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Classifier produced NaN or infinite logits.")
        return logits


class StemDetector(nn.Module):
    """Assemble the complete paper-described detector for one connected subgraph.

    The class deliberately receives configured submodules rather than guessing
    unpublished widths/head counts. It performs strict cross-module dimension
    checks so incompatible configurations fail at construction time.
    """

    def __init__(
        self,
        backbone: ThreeLayerFixedEdgeConvBackbone,
        local_attention: FixedLocalMultiheadAttention,
        pma: PoolingByMultiheadAttention,
        film: FeatureWiseLinearModulation,
        classifier: Optional[StemPointClassifier] = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, ThreeLayerFixedEdgeConvBackbone):
            raise TypeError("backbone must be ThreeLayerFixedEdgeConvBackbone.")
        if not isinstance(local_attention, FixedLocalMultiheadAttention):
            raise TypeError("local_attention must be FixedLocalMultiheadAttention.")
        if not isinstance(pma, PoolingByMultiheadAttention):
            raise TypeError("pma must be PoolingByMultiheadAttention.")
        if not isinstance(film, FeatureWiseLinearModulation):
            raise TypeError("film must be FeatureWiseLinearModulation.")

        classifier = StemPointClassifier() if classifier is None else classifier
        if not isinstance(classifier, StemPointClassifier):
            raise TypeError("classifier must be StemPointClassifier.")

        if backbone.in_channels != PAPER_INPUT_CHANNELS:
            raise ValueError(
                f"The proposed paper model expects {PAPER_INPUT_CHANNELS} input channels; "
                f"backbone has {backbone.in_channels}. Use the separate XYZ-only baseline "
                "implementation for the ablation model."
            )
        if backbone.multi_scale_channels != PAPER_REFINED_CHANNELS:
            raise ValueError(
                "The three EdgeConv outputs must concatenate to 448 channels to match "
                "the reported 448->256->128->64->1 classifier; got "
                f"{backbone.multi_scale_channels}."
            )
        if local_attention.in_channels != PAPER_REFINED_CHANNELS:
            raise ValueError(
                f"local_attention.in_channels must be {PAPER_REFINED_CHANNELS}."
            )
        if pma.model_dim != PAPER_REFINED_CHANNELS:
            raise ValueError(f"pma.model_dim must be {PAPER_REFINED_CHANNELS}.")
        if pma.num_seeds != PAPER_PMA_SEEDS:
            raise ValueError(
                f"The manuscript specifies exactly {PAPER_PMA_SEEDS} PMA seed queries."
            )
        if film.feature_channels != PAPER_REFINED_CHANNELS:
            raise ValueError(f"film.feature_channels must be {PAPER_REFINED_CHANNELS}.")
        if film.context_channels != pma.context_channels:
            raise ValueError(
                "film.context_channels must equal the concatenated PMA context size "
                f"({pma.context_channels})."
            )
        if classifier.input_channels != PAPER_REFINED_CHANNELS:
            raise ValueError(f"classifier input must be {PAPER_REFINED_CHANNELS} channels.")

        self.backbone = backbone
        self.local_attention = local_attention
        self.pma = pma
        self.film = film
        self.classifier = classifier

    @property
    def input_channels(self) -> int:
        return self.backbone.in_channels

    @property
    def refined_channels(self) -> int:
        return self.backbone.multi_scale_channels

    def _validate_inputs(
        self,
        node_features: Tensor,
        positions: Tensor,
        edge_index: Tensor,
    ) -> None:
        if not isinstance(node_features, Tensor):
            raise TypeError("node_features must be a torch tensor.")
        if not isinstance(positions, Tensor):
            raise TypeError("positions must be a torch tensor.")
        if not isinstance(edge_index, Tensor):
            raise TypeError("edge_index must be a torch tensor.")
        if node_features.ndim != 2:
            raise ValueError(
                f"node_features must have shape (N,10), got {tuple(node_features.shape)}."
            )
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (N,3), got {tuple(positions.shape)}."
            )
        if node_features.shape[0] != positions.shape[0]:
            raise ValueError("node_features and positions must contain the same points.")
        if node_features.shape[0] <= 0:
            raise ValueError("At least one point is required.")
        if node_features.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"got {node_features.shape[1]}."
            )
        if not node_features.is_floating_point() or not positions.is_floating_point():
            raise TypeError("node_features and positions must be floating point.")
        if node_features.dtype != positions.dtype:
            raise TypeError("node_features and positions must use the same dtype.")
        if node_features.device != positions.device or node_features.device != edge_index.device:
            raise ValueError("node_features, positions and edge_index must share a device.")
        if not torch.isfinite(node_features).all():
            raise ValueError("node_features contains NaN or infinite values.")
        if not torch.isfinite(positions).all():
            raise ValueError("positions contains NaN or infinite values.")

    def forward(
        self,
        node_features: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        *,
        return_details: bool = False,
        return_probabilities: bool = False,
    ) -> Tensor | StemDetectorOutput:
        """Return per-point logits, with optional evaluation probabilities/details."""
        self._validate_inputs(node_features, positions, edge_index)

        backbone_output: EdgeConvBackboneOutput = self.backbone(
            node_features,
            edge_index,
        )
        multi_scale = backbone_output.multi_scale

        # Under automatic mixed precision, EdgeConv linear layers may emit a
        # lower-precision dtype while the geometric positions remain float32.
        # The attention block requires aligned floating dtypes, so cast only the
        # attention-view of positions; the original input tensor is not modified.
        attention_positions = (
            positions
            if positions.dtype == multi_scale.dtype
            else positions.to(dtype=multi_scale.dtype)
        )
        refined = self.local_attention(
            multi_scale,
            attention_positions,
            edge_index,
        )
        context = self.pma(refined)

        film_output = self.film(refined, context)
        logits = self.classifier(film_output.conditioned)

        probabilities = torch.sigmoid(logits) if return_probabilities else None

        if return_details:
            return StemDetectorOutput(
                logits=logits,
                probabilities=probabilities,
                multi_scale_features=multi_scale,
                refined_features=refined,
                global_context=context,
                gamma=film_output.gamma,
                beta=film_output.beta,
                conditioned_features=film_output.conditioned,
            )
        if return_probabilities:
            if probabilities is None:  # defensive; logically unreachable
                raise RuntimeError("Probability output was requested but not computed.")
            return probabilities
        return logits

    def predict_proba(
        self,
        node_features: Tensor,
        positions: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Return sigmoid stem probabilities while preserving the module's mode."""
        logits = self.forward(node_features, positions, edge_index)
        if not isinstance(logits, Tensor):
            raise RuntimeError("Unexpected detailed output in predict_proba.")
        return torch.sigmoid(logits)


__all__ = [
    "PAPER_INPUT_CHANNELS",
    "PAPER_REFINED_CHANNELS",
    "PAPER_CLASSIFIER_CHANNELS",
    "PAPER_DROPOUT",
    "PAPER_PMA_SEEDS",
    "FiLMOutput",
    "StemDetectorOutput",
    "FeatureWiseLinearModulation",
    "StemPointClassifier",
    "StemDetector",
    "make_linear_film_generators",
]
