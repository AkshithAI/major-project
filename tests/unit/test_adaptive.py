"""Tests for the Adaptive Draft Controller."""

from __future__ import annotations

import pytest

from adhie.speculative.adaptive import AdaptiveDraftController, AdaptiveState
from adhie.types import SpecMetrics


def _make_metrics(acceptance: float = 0.5, speedup: float = 1.2) -> SpecMetrics:
    return SpecMetrics(
        draft_time_ms=10.0,
        verify_time_ms=20.0,
        proposed_tokens=4,
        accepted_tokens=int(4 * acceptance),
        acceptance_rate=acceptance,
        effective_tokens_per_s=30.0,
        speedup_vs_baseline=speedup,
    )


class TestAdaptiveControllerInit:
    """Test initialization and defaults."""

    def test_default_k(self) -> None:
        ctrl = AdaptiveDraftController()
        assert ctrl.current_k == 4  # Default start at k=4

    def test_custom_candidates(self) -> None:
        ctrl = AdaptiveDraftController(candidate_lengths=[3, 5, 7])
        # No 4 in candidates, so picks middle
        assert ctrl.current_k in [3, 5, 7]

    def test_custom_thresholds(self) -> None:
        ctrl = AdaptiveDraftController(
            threshold_high=0.8,
            threshold_low=0.2,
            disable_threshold=0.9,
        )
        assert ctrl.threshold_high == 0.8
        assert ctrl.threshold_low == 0.2


class TestAdaptiveUpdate:
    """Test k adaptation based on acceptance rates."""

    def test_high_acceptance_increases_k(self) -> None:
        ctrl = AdaptiveDraftController(
            candidate_lengths=[2, 4, 6, 8],
            min_samples_for_change=3,
            threshold_high=0.7,
        )
        # Feed high-acceptance metrics
        for _ in range(5):
            ctrl.update(_make_metrics(acceptance=0.8))

        assert ctrl.current_k > 4  # Should have increased from 4

    def test_low_acceptance_decreases_k(self) -> None:
        ctrl = AdaptiveDraftController(
            candidate_lengths=[2, 4, 6, 8],
            min_samples_for_change=3,
            threshold_low=0.3,
        )
        # Feed low-acceptance metrics
        for _ in range(5):
            ctrl.update(_make_metrics(acceptance=0.1))

        assert ctrl.current_k < 4  # Should have decreased from 4

    def test_stable_acceptance_no_change(self) -> None:
        ctrl = AdaptiveDraftController(
            candidate_lengths=[2, 4, 6, 8],
            min_samples_for_change=3,
            threshold_high=0.7,
            threshold_low=0.3,
        )
        # Feed stable-acceptance metrics (between thresholds)
        for _ in range(10):
            ctrl.update(_make_metrics(acceptance=0.5))

        assert ctrl.current_k == 4  # Should stay at 4

    def test_k_does_not_exceed_bounds(self) -> None:
        ctrl = AdaptiveDraftController(
            candidate_lengths=[2, 4, 6, 8],
            min_samples_for_change=2,
            threshold_high=0.5,
        )
        # Push k up repeatedly
        for _ in range(50):
            ctrl.update(_make_metrics(acceptance=0.9))

        assert ctrl.current_k == 8  # Max

    def test_k_does_not_go_below_min(self) -> None:
        ctrl = AdaptiveDraftController(
            candidate_lengths=[2, 4, 6, 8],
            min_samples_for_change=2,
            threshold_low=0.5,
        )
        # Push k down repeatedly
        for _ in range(50):
            ctrl.update(_make_metrics(acceptance=0.1))

        assert ctrl.current_k == 2  # Min

    def test_no_change_before_min_samples(self) -> None:
        ctrl = AdaptiveDraftController(
            candidate_lengths=[2, 4, 6, 8],
            min_samples_for_change=10,
        )
        # Only 3 samples — should not change
        for _ in range(3):
            ctrl.update(_make_metrics(acceptance=0.9))

        assert ctrl.current_k == 4  # Still at initial


class TestShouldDisable:
    """Test speculation disable logic."""

    def test_disable_when_speedup_low(self) -> None:
        ctrl = AdaptiveDraftController(
            min_samples_for_change=3,
            disable_threshold=0.95,
        )
        # Feed metrics with speedup < 0.95
        for _ in range(10):
            ctrl.update(_make_metrics(acceptance=0.5, speedup=0.8))

        assert ctrl.should_disable() is True

    def test_no_disable_when_speedup_good(self) -> None:
        ctrl = AdaptiveDraftController(
            min_samples_for_change=3,
            disable_threshold=0.95,
        )
        for _ in range(10):
            ctrl.update(_make_metrics(acceptance=0.5, speedup=1.3))

        assert ctrl.should_disable() is False

    def test_no_disable_with_few_samples(self) -> None:
        ctrl = AdaptiveDraftController(min_samples_for_change=5)
        for _ in range(3):
            ctrl.update(_make_metrics(acceptance=0.5, speedup=0.5))

        assert ctrl.should_disable() is False  # Not enough samples

    def test_no_speedup_data(self) -> None:
        ctrl = AdaptiveDraftController(min_samples_for_change=3)
        for _ in range(10):
            ctrl.update(SpecMetrics(
                acceptance_rate=0.5,
                speedup_vs_baseline=None,  # No baseline comparison
            ))

        assert ctrl.should_disable() is False


class TestReset:
    """Test controller reset."""

    def test_reset_clears_state(self) -> None:
        ctrl = AdaptiveDraftController()
        for _ in range(10):
            ctrl.update(_make_metrics(acceptance=0.9))

        ctrl.reset()
        assert ctrl.current_k == 4
        state = ctrl.state()
        assert state.total_cycles == 0
        assert state.k_changes == 0


class TestState:
    """Test state reporting."""

    def test_initial_state(self) -> None:
        ctrl = AdaptiveDraftController()
        state = ctrl.state()
        assert isinstance(state, AdaptiveState)
        assert state.current_k == 4
        assert state.total_cycles == 0
        assert state.speculation_disabled is False

    def test_state_after_updates(self) -> None:
        ctrl = AdaptiveDraftController(min_samples_for_change=2)
        for _ in range(5):
            ctrl.update(_make_metrics(acceptance=0.8, speedup=1.3))

        state = ctrl.state()
        assert state.total_cycles == 5
        assert state.avg_acceptance_rate > 0
        assert state.avg_speedup > 0
        assert state.reason != ""

    def test_state_reports_k_changes(self) -> None:
        ctrl = AdaptiveDraftController(
            candidate_lengths=[2, 4, 6, 8],
            min_samples_for_change=2,
            threshold_high=0.7,
        )
        for _ in range(5):
            ctrl.update(_make_metrics(acceptance=0.9))

        state = ctrl.state()
        assert state.k_changes > 0


class TestAcceptanceTrend:
    """Test acceptance trend reporting."""

    def test_empty_trend(self) -> None:
        ctrl = AdaptiveDraftController()
        assert ctrl.acceptance_trend() == []

    def test_trend_matches_input(self) -> None:
        ctrl = AdaptiveDraftController()
        rates = [0.3, 0.5, 0.7, 0.9]
        for r in rates:
            ctrl.update(_make_metrics(acceptance=r))

        trend = ctrl.acceptance_trend()
        assert len(trend) == 4
        assert trend == rates

