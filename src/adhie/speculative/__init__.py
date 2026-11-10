"""Speculative decoding module — controller, draft sweep, and adaptive k."""

from adhie.speculative.controller import SpecConfig, SpeculativeController
from adhie.speculative.adaptive import AdaptiveDraftController, AdaptiveState
from adhie.speculative.draft_sweep import DraftLengthSweep, SweepResult, VariantSweepResult

__all__ = [
    "SpecConfig",
    "SpeculativeController",
    "AdaptiveDraftController",
    "AdaptiveState",
    "DraftLengthSweep",
    "SweepResult",
    "VariantSweepResult",
]

