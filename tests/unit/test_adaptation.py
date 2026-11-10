"""Tests for the Runtime Adaptation Manager."""

from __future__ import annotations

import pytest

from adhie.runtime.adaptation import (
    AdaptationResult,
    RuntimeAdaptationManager,
)
from adhie.speculative.adaptive import AdaptiveDraftController
from adhie.types import InferenceMetrics, SpecMetrics


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_metrics(tpot: float = 80.0) -> InferenceMetrics:
    return InferenceMetrics(tpot_ms=tpot, ttft_ms=400.0)


def _make_spec_metrics(
    acceptance: float = 0.6,
    speedup: float | None = 1.2,
) -> SpecMetrics:
    return SpecMetrics(
        acceptance_rate=acceptance,
        speedup_vs_baseline=speedup,
        proposed_tokens=4,
        accepted_tokens=int(4 * acceptance),
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestAdaptationManagerInit:
    def test_default_state(self) -> None:
        mgr = RuntimeAdaptationManager()
        assert mgr.speculation_enabled is True
        assert mgr._request_count == 0

    def test_with_controller(self) -> None:
        ctrl = AdaptiveDraftController()
        mgr = RuntimeAdaptationManager(spec_controller=ctrl)
        assert mgr.spec_controller is ctrl

    def test_disabled_speculation(self) -> None:
        mgr = RuntimeAdaptationManager(enable_speculation=False)
        assert mgr.speculation_enabled is False


class TestOnRequestComplete:
    def test_no_controller_no_adjustments(self) -> None:
        mgr = RuntimeAdaptationManager()
        result = mgr.on_request_complete(_make_metrics())

        assert isinstance(result, AdaptationResult)
        assert result.adjustments == {}
        assert result.adaptive_state is None

    def test_with_spec_metrics_updates_k(self) -> None:
        ctrl = AdaptiveDraftController(
            candidate_lengths=[2, 4, 6, 8],
            min_samples_for_change=1,
        )
        mgr = RuntimeAdaptationManager(spec_controller=ctrl)

        # Send enough high-acceptance metrics to trigger k increase
        for _ in range(3):
            result = mgr.on_request_complete(
                _make_metrics(),
                _make_spec_metrics(acceptance=0.9),
            )

        assert "draft_length" in result.adjustments
        # With high acceptance, k should have increased from default 4
        assert result.adjustments["draft_length"] >= 4

    def test_no_spec_metrics_no_adjustments(self) -> None:
        ctrl = AdaptiveDraftController()
        mgr = RuntimeAdaptationManager(spec_controller=ctrl)

        result = mgr.on_request_complete(_make_metrics())
        assert "draft_length" not in result.adjustments

    def test_request_count_increments(self) -> None:
        mgr = RuntimeAdaptationManager()
        mgr.on_request_complete(_make_metrics())
        mgr.on_request_complete(_make_metrics())
        assert mgr._request_count == 2

    def test_speculation_disabled_skips_update(self) -> None:
        ctrl = AdaptiveDraftController()
        mgr = RuntimeAdaptationManager(
            spec_controller=ctrl,
            enable_speculation=False,
        )

        result = mgr.on_request_complete(
            _make_metrics(),
            _make_spec_metrics(),
        )
        # Speculation is disabled so should not adjust draft_length
        assert "draft_length" not in result.adjustments


class TestSpeculationDisable:
    def test_disable_on_low_speedup(self) -> None:
        ctrl = AdaptiveDraftController(
            min_samples_for_change=2,
            disable_threshold=1.0,
        )
        mgr = RuntimeAdaptationManager(spec_controller=ctrl)

        # Feed low-speedup samples; should_disable needs min_samples * 2 = 4
        disabled_result = None
        for _ in range(20):
            result = mgr.on_request_complete(
                _make_metrics(),
                _make_spec_metrics(acceptance=0.5, speedup=0.8),
            )
            if result.adjustments.get("speculation_enabled") is False:
                disabled_result = result
                break

        # Must have been disabled at some point
        assert disabled_result is not None, "Speculation was never disabled"
        assert mgr.speculation_enabled is False


class TestReset:
    def test_reset_restores_defaults(self) -> None:
        ctrl = AdaptiveDraftController()
        mgr = RuntimeAdaptationManager(spec_controller=ctrl)

        # Do some work
        mgr.on_request_complete(_make_metrics(), _make_spec_metrics())
        mgr._speculation_enabled = False

        mgr.reset()
        assert mgr.speculation_enabled is True
        assert mgr._request_count == 0
        assert mgr._total_adjustments == 0


class TestSummary:
    def test_summary_without_controller(self) -> None:
        mgr = RuntimeAdaptationManager()
        s = mgr.summary()
        assert s["requests_processed"] == 0
        assert s["current_k"] is None

    def test_summary_with_controller(self) -> None:
        ctrl = AdaptiveDraftController(candidate_lengths=[2, 4, 6])
        mgr = RuntimeAdaptationManager(spec_controller=ctrl)
        mgr.on_request_complete(_make_metrics(), _make_spec_metrics())

        s = mgr.summary()
        assert s["requests_processed"] == 1
        assert s["current_k"] == 4
        assert s["speculation_enabled"] is True


class TestAdaptationResult:
    def test_defaults(self) -> None:
        r = AdaptationResult()
        assert r.adjustments == {}
        assert r.adaptive_state is None
        assert r.reason == ""

