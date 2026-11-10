"""Adaptive Draft Controller — runtime adaptation of speculation draft length.

Monitors recent speculative decoding performance and adjusts the draft
length (k) to maximize throughput. Uses a simple rule-based policy:

  - Track rolling window of last N cycles
  - If acceptance_rate > threshold_high → increase k (try longer drafts)
  - If acceptance_rate < threshold_low → decrease k (shorter drafts)
  - If speedup_vs_baseline < 1.0 → disable speculation entirely
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean

from adhie.types import SpecMetrics


@dataclass
class AdaptiveState:
    """Observable state from the adaptive controller."""

    current_k: int = 4
    avg_acceptance_rate: float = 0.0
    avg_speedup: float = 0.0
    total_cycles: int = 0
    k_changes: int = 0
    speculation_disabled: bool = False
    reason: str = ""


class AdaptiveDraftController:
    """Runtime adaptation of draft length based on recent acceptance rates.

    The controller maintains a sliding window of recent SpecMetrics and
    adjusts k (draft length) based on observed acceptance rates.

    Args:
        candidate_lengths: Ordered list of k values to choose from.
        window_size: Number of recent cycles to consider.
        threshold_high: Acceptance rate above which k is increased.
        threshold_low: Acceptance rate below which k is decreased.
        disable_threshold: Speedup below which speculation is disabled.
        min_samples_for_change: Minimum window samples before allowing k changes.
    """

    def __init__(
        self,
        candidate_lengths: list[int] | None = None,
        window_size: int = 20,
        threshold_high: float = 0.7,
        threshold_low: float = 0.3,
        disable_threshold: float = 0.95,
        min_samples_for_change: int = 5,
    ) -> None:
        self.candidates = candidate_lengths or [2, 4, 6, 8]
        self.window_size = window_size
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.disable_threshold = disable_threshold
        self.min_samples_for_change = min_samples_for_change

        self._window: deque[SpecMetrics] = deque(maxlen=window_size)
        self._current_k_idx: int = self._default_k_idx()
        self._total_cycles: int = 0
        self._k_changes: int = 0
        self._disabled: bool = False

    def _default_k_idx(self) -> int:
        """Default starting index: pick k=4 if available, else middle."""
        if 4 in self.candidates:
            return self.candidates.index(4)
        return len(self.candidates) // 2

    @property
    def current_k(self) -> int:
        """Current draft length."""
        return self.candidates[self._current_k_idx]

    def update(self, metrics: SpecMetrics) -> int:
        """Record new cycle metrics and return the next k to use.

        Args:
            metrics: Metrics from the most recent speculative cycle.

        Returns:
            The draft length k to use for the next cycle.
        """
        self._window.append(metrics)
        self._total_cycles += 1

        if len(self._window) < self.min_samples_for_change:
            return self.current_k

        avg_acceptance = mean(m.acceptance_rate for m in self._window)

        old_k_idx = self._current_k_idx

        if avg_acceptance > self.threshold_high:
            # Good acceptance — try longer drafts
            if self._current_k_idx < len(self.candidates) - 1:
                self._current_k_idx += 1
        elif avg_acceptance < self.threshold_low:
            # Poor acceptance — try shorter drafts
            if self._current_k_idx > 0:
                self._current_k_idx -= 1

        if self._current_k_idx != old_k_idx:
            self._k_changes += 1

        return self.current_k

    def should_disable(self) -> bool:
        """Check if speculation should be disabled entirely.

        Speculation is disabled if the rolling average speedup vs baseline
        is consistently below the disable threshold (e.g., 0.95x).
        """
        if len(self._window) < self.min_samples_for_change * 2:
            return False

        speedups = [
            m.speedup_vs_baseline
            for m in self._window
            if m.speedup_vs_baseline is not None
        ]
        if not speedups:
            return False

        avg_speedup = mean(speedups)
        self._disabled = avg_speedup < self.disable_threshold
        return self._disabled

    def reset(self) -> None:
        """Reset the controller state (e.g., when model or context changes)."""
        self._window.clear()
        self._current_k_idx = self._default_k_idx()
        self._total_cycles = 0
        self._k_changes = 0
        self._disabled = False

    def state(self) -> AdaptiveState:
        """Return the current observable state of the controller."""
        avg_acceptance = 0.0
        avg_speedup = 0.0

        if self._window:
            avg_acceptance = mean(m.acceptance_rate for m in self._window)
            speedups = [
                m.speedup_vs_baseline
                for m in self._window
                if m.speedup_vs_baseline is not None
            ]
            if speedups:
                avg_speedup = mean(speedups)

        reason = ""
        if self._disabled:
            reason = f"Disabled: avg speedup {avg_speedup:.2f}x < {self.disable_threshold}"
        elif avg_acceptance > self.threshold_high:
            reason = f"High acceptance ({avg_acceptance:.2f}), trying longer drafts"
        elif avg_acceptance < self.threshold_low:
            reason = f"Low acceptance ({avg_acceptance:.2f}), using shorter drafts"
        else:
            reason = f"Stable acceptance ({avg_acceptance:.2f})"

        return AdaptiveState(
            current_k=self.current_k,
            avg_acceptance_rate=avg_acceptance,
            avg_speedup=avg_speedup,
            total_cycles=self._total_cycles,
            k_changes=self._k_changes,
            speculation_disabled=self._disabled,
            reason=reason,
        )

    def acceptance_trend(self) -> list[float]:
        """Return the sequence of acceptance rates in the current window."""
        return [m.acceptance_rate for m in self._window]

