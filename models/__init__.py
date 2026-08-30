"""Model components for occlusion-robust TLS stem detection.

Author: Shahab Alaedin Baloochi

This package exposes the public model interface used by the repository:
fixed-graph EdgeConv, fixed-neighbour local attention, PMA pooling, FiLM
conditioning, and the complete point-wise stem detector assembly.
"""

from .edgeconv import (
    EdgeConvBackboneOutput,
    FixedEdgeConv,
    ThreeLayerFixedEdgeConvBackbone,
    make_message_mlp,
)
from .local_attention import (
    FixedLocalMultiheadAttention,
    LocalAttentionOutput,
    make_relative_position_bias_mlp,
)
from .pma import (
    PMAOutput,
    PoolingByMultiheadAttention,
)
from .stem_detector import (
    PAPER_INPUT_CHANNELS,
    PAPER_REFINED_CHANNELS,
    PAPER_CLASSIFIER_CHANNELS,
    PAPER_DROPOUT,
    PAPER_PMA_SEEDS,
    FiLMOutput,
    StemDetectorOutput,
    FeatureWiseLinearModulation,
    StemPointClassifier,
    StemDetector,
    make_linear_film_generators,
)

__all__ = [
    "EdgeConvBackboneOutput",
    "FixedEdgeConv",
    "ThreeLayerFixedEdgeConvBackbone",
    "make_message_mlp",
    "FixedLocalMultiheadAttention",
    "LocalAttentionOutput",
    "make_relative_position_bias_mlp",
    "PMAOutput",
    "PoolingByMultiheadAttention",
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
