"""Tests for the BenchmarkRunner — metric aggregation and result formatting."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from adhie.benchmark.runner import BenchmarkRunner
from adhie.runtime.base import GenerateRequest, GenerateResponse, RuntimeAdapter
from adhie.types import (
    BenchmarkResult,
    HardwareProfile,
    InferenceMetrics,
    ModelProfile,
    RunPlan,
    WorkloadConfig,
)


# ---------------------------------------------------------------------------
# Fake runtime adapter for testing (no subprocess, no network)
# ---------------------------------------------------------------------------

class FakeAdapter(RuntimeAdapter):
    """A mock runtime adapter that returns canned responses."""

    def __init__(self, metrics_sequence: list[InferenceMetrics] | None = None) -> None:
        self._metrics_sequence = metrics_sequence or []
        self._call_idx = 0
        self._loaded = False
        self._last: InferenceMetrics | None = None

    def load(self, plan: RunPlan) -> None:
        self._loaded = True

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        if not self._loaded:
            raise RuntimeError("Not loaded")
        if self._call_idx < len(self._metrics_sequence):
            m = self._metrics_sequence[self._call_idx]
        else:
            m = InferenceMetrics(
                ttft_ms=50.0, tpot_ms=20.0,
                output_tokens_per_s=50.0,
                total_output_tokens=10,
            )
        self._call_idx += 1
        self._last = m
        return GenerateResponse(
            text="Hello world " * 10,
            prompt_tokens=20,
            output_tokens=10,
            metrics=m,
        )

    def generate_simple(self, prompt: str, max_tokens: int = 256) -> str:
        return "Hello"

    def health_check(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    def last_metrics(self) -> InferenceMetrics | None:
        return self._last


def _make_plan(**kwargs) -> RunPlan:
    defaults = dict(
        model_path="/models/test.gguf",
        backend="llama.cpp",
        gpu_layers=20,
        cpu_threads=4,
        context_length=4096,
        batch_size=512,
        ubatch_size=128,
    )
    defaults.update(kwargs)
    return RunPlan(**defaults)


class TestBenchmarkRunnerSingle:
    """Test run_single with a fake adapter."""

    def test_run_single_basic(self) -> None:
        metrics = [
            InferenceMetrics(ttft_ms=50.0, tpot_ms=20.0, output_tokens_per_s=50.0,
                             total_output_tokens=10, total_prompt_tokens=20,
                             total_time_s=0.5, peak_ram_bytes=1_000_000),
        ]
        adapter = FakeAdapter(metrics * 4)  # warmup + timed
        runner = BenchmarkRunner(adapter, warmup_runs=1, timed_runs=3,
                                 cooldown_seconds=0)
        plan = _make_plan()
        result = runner.run_single(plan, "Tell me a joke", max_tokens=64)
        assert isinstance(result, BenchmarkResult)
        assert result.metrics.ttft_ms == 50.0
        assert result.run_id.startswith("run-")

    def test_run_single_zero_warmup(self) -> None:
        adapter = FakeAdapter()
        runner = BenchmarkRunner(adapter, warmup_runs=0, timed_runs=1,
                                 cooldown_seconds=0)
        plan = _make_plan()
        result = runner.run_single(plan, "Hello")
        assert isinstance(result, BenchmarkResult)


class TestAggregateMetrics:
    """Test _aggregate_metrics median computation."""

    def test_empty_list(self) -> None:
        adapter = FakeAdapter()
        runner = BenchmarkRunner(adapter)
        result = runner._aggregate_metrics([])
        assert result.ttft_ms == 0.0
        assert result.total_output_tokens == 0

    def test_single_metric(self) -> None:
        adapter = FakeAdapter()
        runner = BenchmarkRunner(adapter)
        m = InferenceMetrics(ttft_ms=100.0, tpot_ms=25.0, output_tokens_per_s=40.0)
        result = runner._aggregate_metrics([m])
        assert result.ttft_ms == 100.0
        assert result.tpot_ms == 25.0

    def test_median_of_three(self) -> None:
        adapter = FakeAdapter()
        runner = BenchmarkRunner(adapter)
        metrics = [
            InferenceMetrics(ttft_ms=100.0, tpot_ms=20.0, output_tokens_per_s=50.0,
                             prompt_tokens_per_s=1000.0,
                             total_output_tokens=10, total_prompt_tokens=20,
                             total_time_s=0.5, peak_vram_bytes=1_000_000,
                             peak_ram_bytes=2_000_000, itl_ms_list=[20.0, 21.0]),
            InferenceMetrics(ttft_ms=120.0, tpot_ms=22.0, output_tokens_per_s=45.0,
                             prompt_tokens_per_s=900.0,
                             total_output_tokens=10, total_prompt_tokens=20,
                             total_time_s=0.6, peak_vram_bytes=1_100_000,
                             peak_ram_bytes=2_100_000, itl_ms_list=[22.0, 23.0]),
            InferenceMetrics(ttft_ms=110.0, tpot_ms=21.0, output_tokens_per_s=48.0,
                             prompt_tokens_per_s=950.0,
                             total_output_tokens=10, total_prompt_tokens=20,
                             total_time_s=0.55, peak_vram_bytes=1_050_000,
                             peak_ram_bytes=2_050_000, itl_ms_list=[21.0, 22.0]),
        ]
        result = runner._aggregate_metrics(metrics)
        # Median TTFT should be 110 (middle of 100, 110, 120)
        assert result.ttft_ms == 110.0
        assert result.tpot_ms == 21.0
        assert result.output_tokens_per_s == 48.0
        # Peak VRAM/RAM should be max across runs
        assert result.peak_vram_bytes == 1_100_000
        assert result.peak_ram_bytes == 2_100_000
        # All ITLs pooled
        assert len(result.itl_ms_list) == 6


class TestPrintComparison:
    """Test comparison table rendering doesn't crash."""

    def test_print_comparison_no_crash(self) -> None:
        adapter = FakeAdapter()
        runner = BenchmarkRunner(adapter)

        results = [
            BenchmarkResult(
                run_id="run-1",
                timestamp="2026-01-01T00:00:00Z",
                hardware_fingerprint="hw-abc",
                model_fingerprint="mdl-xyz",
                plan=_make_plan(plan_id="plan-A", execution_mode="gpu_partial"),
                workload=WorkloadConfig(context_length=4096, output_tokens=256),
                metrics=InferenceMetrics(
                    ttft_ms=100.0, tpot_ms=20.0, itl_p95_ms=25.0,
                    output_tokens_per_s=50.0,
                    peak_vram_bytes=2_000_000_000,
                    peak_ram_bytes=4_000_000_000,
                ),
            ),
            BenchmarkResult(
                run_id="run-2",
                timestamp="2026-01-01T00:01:00Z",
                hardware_fingerprint="hw-abc",
                model_fingerprint="mdl-xyz",
                plan=_make_plan(plan_id="plan-B", execution_mode="gpu_resident",
                                gpu_layers=40, kv_type_k="q8_0"),
                workload=WorkloadConfig(context_length=4096, output_tokens=256),
                metrics=InferenceMetrics(
                    ttft_ms=80.0, tpot_ms=15.0, itl_p95_ms=18.0,
                    output_tokens_per_s=65.0,
                    peak_vram_bytes=3_000_000_000,
                    peak_ram_bytes=3_500_000_000,
                ),
            ),
        ]

        # Should not raise; we redirect console to avoid noise
        runner.print_comparison(results)


class TestFakeAdapterContract:
    """Verify FakeAdapter satisfies the RuntimeAdapter contract."""

    def test_full_lifecycle(self) -> None:
        adapter = FakeAdapter()
        plan = _make_plan()
        adapter.load(plan)
        assert adapter.health_check() is True
        resp = adapter.generate(GenerateRequest(prompt="test"))
        assert resp.text != ""
        assert adapter.last_metrics() is not None
        adapter.unload()
        assert adapter.health_check() is False

