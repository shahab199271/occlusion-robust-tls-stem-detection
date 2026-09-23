"""Post-processing utilities for TLS stem detection.

Author: Shahab Alaedin Baloochi

This package contains Step 1 high-confidence core identification and Step 2
two-pass axis-envelope expansion. TreeQSM-style Step 3 filtering is outside
this package.
"""

from .axis_envelope import (
    PAPER_BIN_HEIGHT_M,
    PAPER_MAX_BINS,
    PAPER_MAX_RADIUS_M,
    PAPER_MIN_BINS,
    PAPER_MIN_RADIUS_M,
    PAPER_NUM_EXPANSION_PASSES,
    PAPER_RADIUS_QUANTILE,
    PAPER_SMOOTH_WINDOW,
    AxisEnvelope,
    EnvelopeMembership,
    ExpansionPassResult,
    TwoPassExpansionResult,
    adaptive_num_bins,
    envelope_membership,
    estimate_axis_envelope,
    expand_connected_candidate,
    two_pass_axis_envelope_expansion,
)
from .postprocess import (
    CoreEnvelopePostprocessingResult,
    HighConfidenceComponent,
    HighConfidenceCoreResult,
    identify_high_confidence_core,
    postprocess_core_envelope,
    validate_core_envelope_postprocessing_result,
    validate_high_confidence_core_result,
)

__all__ = [
    "PAPER_BIN_HEIGHT_M",
    "PAPER_MIN_BINS",
    "PAPER_MAX_BINS",
    "PAPER_RADIUS_QUANTILE",
    "PAPER_MIN_RADIUS_M",
    "PAPER_MAX_RADIUS_M",
    "PAPER_SMOOTH_WINDOW",
    "PAPER_NUM_EXPANSION_PASSES",
    "AxisEnvelope",
    "EnvelopeMembership",
    "ExpansionPassResult",
    "TwoPassExpansionResult",
    "adaptive_num_bins",
    "estimate_axis_envelope",
    "envelope_membership",
    "expand_connected_candidate",
    "two_pass_axis_envelope_expansion",
    "HighConfidenceComponent",
    "HighConfidenceCoreResult",
    "CoreEnvelopePostprocessingResult",
    "identify_high_confidence_core",
    "validate_high_confidence_core_result",
    "postprocess_core_envelope",
    "validate_core_envelope_postprocessing_result",
]
