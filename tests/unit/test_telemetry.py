"""Tests for the TelemetryCollector — timing metrics capture."""

from __future__ import annotations

import time

import pytest

from adhie.telemetry.collector import TelemetryCollector


class TestTelemetryCollectorBasic:
    """Basic lifecycle tests."""

    def test_start_and_finalize_empty(self) -> None:
        """Finalize immediately after start — no tokens recorded."""
        tc = TelemetryCollector()
        tc.start(prompt_tokens=10)
        metrics = tc.finalize()
        assert metrics.total_prompt_tokens == 10
        assert metrics.total_output_tokens == 0
        assert metrics.total_time_s > 0  # some small elapsed time

    def test_single_token(self) -> None:
        tc = TelemetryCollector()
        tc.start(prompt_tokens=5)
        time.sleep(0.01)  # simulate prompt processing
        tc.record_first_token()
        metrics = tc.finalize()
        assert metrics.total_output_tokens == 1
        assert metrics.ttft_ms > 0

    def test_multiple_tokens(self) -> None:
        tc = TelemetryCollector()
        tc.start(prompt_tokens=5)
        time.sleep(0.01)
        tc.record_first_token()
        for _ in range(9):
            time.sleep(0.001)
            tc.record_token()
        metrics = tc.finalize()
        assert metrics.total_output_tokens == 10
        assert metrics.ttft_ms > 0
        assert metrics.tpot_ms > 0
        assert metrics.output_tokens_per_s > 0

    def test_batch_tokens(self) -> None:
        """record_tokens() for speculative/batch decoding."""
        tc = TelemetryCollector()
        tc.start()
        tc.record_tokens(5)
        metrics = tc.finalize()
        assert metrics.total_output_tokens == 5
        assert metrics.ttft_ms > 0  # first token time auto-set


class TestTelemetryITL:
    """Inter-token latency (ITL) tests."""

    def test_itl_list_populated(self) -> None:
        tc = TelemetryCollector()
        tc.start()
        tc.record_first_token()
        for _ in range(4):
            time.sleep(0.002)
            tc.record_token()
        metrics = tc.finalize()
        assert len(metrics.itl_ms_list) == 4  # gaps between 5 token events
        assert all(itl > 0 for itl in metrics.itl_ms_list)

    def test_itl_p95(self) -> None:
        tc = TelemetryCollector()
        tc.start()
        tc.record_first_token()
        for _ in range(19):
            time.sleep(0.001)
            tc.record_token()
        metrics = tc.finalize()
        assert metrics.itl_p95_ms > 0

    def test_no_itl_for_single_token(self) -> None:
        tc = TelemetryCollector()
        tc.start()
        tc.record_first_token()
        metrics = tc.finalize()
        assert len(metrics.itl_ms_list) == 0


class TestTelemetryPromptMetrics:
    """Prompt throughput metrics."""

    def test_prompt_tokens_per_s(self) -> None:
        tc = TelemetryCollector()
        tc.start(prompt_tokens=100)
        time.sleep(0.02)  # ~20ms "prompt processing"
        tc.record_first_token()
        metrics = tc.finalize()
        assert metrics.prompt_tokens_per_s > 0

    def test_no_prompt_tokens_no_throughput(self) -> None:
        tc = TelemetryCollector()
        tc.start(prompt_tokens=0)
        tc.record_first_token()
        metrics = tc.finalize()
        # prompt_tokens_per_s stays at 0 because no prompt tokens
        assert metrics.prompt_tokens_per_s == 0.0


class TestTelemetryResourceSampling:
    """Resource sampling tests."""

    def test_peak_ram_tracked(self) -> None:
        tc = TelemetryCollector()
        tc.start()  # no PID → uses system-wide fallback
        tc.record_first_token()
        metrics = tc.finalize()
        # peak_ram_bytes should be non-zero (system memory is always used)
        assert metrics.peak_ram_bytes > 0

    def test_invalid_pid_handled(self) -> None:
        """Passing a PID that doesn't exist should not crash."""
        tc = TelemetryCollector()
        tc.start(pid=999999999)  # unlikely PID
        tc.record_first_token()
        metrics = tc.finalize()
        # Should still complete without error
        assert metrics.total_output_tokens == 1

