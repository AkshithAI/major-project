"""Runtime Adaptation Manager — safe parameter adjustment between requests.

Phase B of the ADHIE integrated pipeline. Only adapts parameters that
can change without requiring a model reload:

  - Draft length (k) via AdaptiveDraftController
  - Speculation enable/disable
  - Future: KV precision, CPU thread budget (between requests)

The manager is called after each inference request completes, and
returns a dict of recommended parameter adjustments for the next request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from adhie.speculative.adaptive import AdaptiveDraftController, AdaptiveState
from adhie.types import InferenceMetrics, SpecMetrics

logger = logging.getLogger(__name__)


@dataclass
class AdaptationResult:
    """Result of runtime adaptation analysis."""

    adjustments: dict[str, object] = field(default_factory=dict)
    adaptive_state: AdaptiveState | None = None
    reason: str = ""


class RuntimeAdaptationManager:
    """Wraps adaptive controllers for safe runtime parameter tuning.

    Only adjusts parameters that can change without model reload.
    Currently manages:
      - Draft length (k) via AdaptiveDraftController
      - Speculation enable/disable

    Args:
        spec_controller: The adaptive draft controller for speculation.
        enable_speculation: Whether speculation is currently enabled.
    """

    def __init__(
        self,
        spec_controller: AdaptiveDraftController | None = None,
        enable_speculation: bool = True,
    ) -> None:
        self.spec_controller = spec_controller
        self._speculation_enabled = enable_speculation
        self._request_count = 0
        self._total_adjustments = 0

    @property
    def speculation_enabled(self) -> bool:
        """Whether speculative decoding is currently enabled."""
        return self._speculation_enabled

    def on_request_complete(
        self,
        metrics: InferenceMetrics,
        spec_metrics: SpecMetrics | None = None,
    ) -> AdaptationResult:
        """Analyse completed request and return parameter adjustments.

        Called after each inference request. Uses the metrics to decide
        whether to adjust draft length, disable speculation, etc.

        Args:
            metrics: Inference timing metrics from the completed request.
            spec_metrics: Speculative decoding metrics, if applicable.

        Returns:
            AdaptationResult with a dict of adjustments and reasoning.
        """
        self._request_count += 1
        adjustments: dict[str, object] = {}
        reasons: list[str] = []

        # -- Speculative decoding adaptation --
        if self.spec_controller and spec_metrics and self._speculation_enabled:
            new_k = self.spec_controller.update(spec_metrics)
            adjustments["draft_length"] = new_k

            if self.spec_controller.should_disable():
                self._speculation_enabled = False
                adjustments["speculation_enabled"] = False
                reasons.append(
                    f"Disabled speculation: avg speedup below threshold"
                )
            else:
                state = self.spec_controller.state()
                if state.reason:
                    reasons.append(f"Draft k={new_k}: {state.reason}")

        # -- Future: thread budget adaptation --
        # Could analyze cpu_utilization_pct and adjust thread counts
        # between requests without reload.

        if adjustments:
            self._total_adjustments += 1

        adaptive_state = (
            self.spec_controller.state() if self.spec_controller else None
        )

        return AdaptationResult(
            adjustments=adjustments,
            adaptive_state=adaptive_state,
            reason="; ".join(reasons) if reasons else "No adjustments needed",
        )

    def reset(self) -> None:
        """Reset all adaptation state (e.g., on model change)."""
        if self.spec_controller:
            self.spec_controller.reset()
        self._speculation_enabled = True
        self._request_count = 0
        self._total_adjustments = 0

    def summary(self) -> dict[str, object]:
        """Return a summary of adaptation activity."""
        state = self.spec_controller.state() if self.spec_controller else None
        return {
            "requests_processed": self._request_count,
            "total_adjustments": self._total_adjustments,
            "speculation_enabled": self._speculation_enabled,
            "current_k": state.current_k if state else None,
            "avg_acceptance_rate": state.avg_acceptance_rate if state else None,
            "k_changes": state.k_changes if state else None,
        }

