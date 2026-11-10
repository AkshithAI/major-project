"""Tests for the Autotuner.

Tests use mock RuntimeAdapters to avoid needing a real backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adhie.planner.autotuner import (
    Autotuner,
    AutotunerConfig,
    TuneResult,
    _aggregate_metrics,
    _make_benchmark_prompt,
)
from adhie.runtime.base import GenerateRequest, GenerateResponse, RuntimeAdapter
from adhie.types import InferenceMetrics, RunPlan, WorkloadConfig


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_plan(gpu_layers: int = 10, mode: str = "hybrid") -> RunPlan:
    return RunPlan(
        plan_id=f"plan-{gpu_layers}",
        gpu_layers=gpu_layers,
        execution_mode=mode,
        kv_type_k="f16",
        estimated_vram_bytes=5_000_000_000,
    )


def _make_metrics(tpot: float = 80.0, ttft: float = 500.0, vram: int = 0) -> InferenceMetrics:
    return InferenceMetrics(
        tpot_ms=tpot,
        ttft_ms=ttft,
        output_tokens_per_s=1000.0 / tpot,
        total_output_tokens=128,
        total_time_s=tpot * 128 / 1000.0,
        peak_vram_bytes=vram,
    )


def _make_mock_adapter(
    metrics: InferenceMetrics | None = None,
    fail_load: bool = False,
    fail_health: bool = False,
) -> MagicMock:
    """Create a mock RuntimeAdapter."""
    adapter = MagicMock(spec=RuntimeAdapter)

    if fail_load:
        adapter.load.side_effect = RuntimeError("OOM")
    else:
        adapter.load.return_value = None

    adapter.health_check.return_value = not fail_health
    adapter.unload.return_value = None

    m = metrics or _make_metrics()
    adapter.generate.return_value = GenerateResponse(
        text="Test output",
        output_tokens=128,
        metrics=m,
    )
    adapter.last_metrics.return_value = m

    return adapter


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestAutotunerConfig:
    def test_defaults(self) -> None:
        cfg = AutotunerConfig()
        assert cfg.max_candidates == 5
        assert cfg.benchmark_rounds == 3
        assert cfg.w_tpot == 0.5

    def test_custom(self) -> None:
        cfg = AutotunerConfig(max_candidates=3, w_tpot=0.6)
        assert cfg.max_candidates == 3
        assert cfg.w_tpot == 0.6


class TestAutotunerScoring:
    def test_score_basic(self) -> None:
        adapter = _make_mock_adapter()
        tuner = Autotuner(adapter)
        plan = _make_plan()
        metrics = _make_metrics(tpot=100.0, ttft=500.0)

        score = tuner._compute_score(plan, metrics)
        # 0.5 * 100 + 0.2 * 500 + 0.2 * 0 + 0.1 * 0 = 50 + 100 = 150
        assert score == pytest.approx(150.0)

    def test_score_with_memory_penalty(self) -> None:
        adapter = _make_mock_adapter()
        tuner = Autotuner(adapter)
        plan = _make_plan()
        plan.estimated_vram_bytes = 1_000_000_000
        metrics = _make_metrics(tpot=100.0, ttft=500.0, vram=950_000_000)

        score = tuner._compute_score(plan, metrics)
        # vram_ratio = 0.95 → penalty = (0.95 - 0.9) * 1000 = 50
        # 0.5*100 + 0.2*500 + 0.2*50 + 0.1*0 = 50 + 100 + 10 = 160
        assert score == pytest.approx(160.0)

    def test_score_zero_vram_no_penalty(self) -> None:
        """No memory penalty when VRAM data is missing."""
        adapter = _make_mock_adapter()
        tuner = Autotuner(adapter)
        plan = _make_plan()
        plan.estimated_vram_bytes = 0
        metrics = _make_metrics(tpot=80.0, ttft=400.0, vram=0)

        score = tuner._compute_score(plan, metrics)
        # No mem penalty since estimated_vram = 0
        expected = 0.5 * 80 + 0.2 * 400
        assert score == pytest.approx(expected)


class TestAutotunerTune:
    def test_single_candidate(self) -> None:
        metrics = _make_metrics(tpot=80.0, ttft=400.0)
        adapter = _make_mock_adapter(metrics)
        tuner = Autotuner(adapter, AutotunerConfig(benchmark_rounds=1))

        candidates = [_make_plan(gpu_layers=20)]
        workload = WorkloadConfig()

        result = tuner.tune(candidates, workload)
        assert isinstance(result, TuneResult)
        assert result.error is None
        assert result.plan.gpu_layers == 20
        assert result.score > 0

    def test_multiple_candidates_selects_best(self) -> None:
        """The candidate with the best (lowest) score should be selected."""
        call_count = 0

        def side_effect(req):
            nonlocal call_count
            call_count += 1
            # First plan → fast, second plan → slow
            # warmup + 2 rounds per plan = 3 calls each → 6 total
            if call_count <= 3:  # First plan's calls
                m = _make_metrics(tpot=50.0, ttft=200.0)
            else:  # Second plan's calls
                m = _make_metrics(tpot=200.0, ttft=800.0)
            return GenerateResponse(text="test", output_tokens=128, metrics=m)

        adapter = _make_mock_adapter()
        adapter.generate.side_effect = side_effect

        tuner = Autotuner(
            adapter,
            AutotunerConfig(benchmark_rounds=2, max_candidates=2),
        )

        candidates = [
            _make_plan(gpu_layers=20, mode="hybrid"),
            _make_plan(gpu_layers=0, mode="cpu_first"),
        ]

        result = tuner.tune(candidates, WorkloadConfig())
        # First plan should win (lower TPOT and TTFT)
        assert result.plan.gpu_layers == 20

    def test_all_candidates_fail_raises(self) -> None:
        adapter = _make_mock_adapter(fail_load=True)
        tuner = Autotuner(adapter)

        with pytest.raises(RuntimeError, match="No candidate"):
            tuner.tune([_make_plan()], WorkloadConfig())

    def test_health_check_failure_skips_candidate(self) -> None:
        """A plan whose backend fails health check is skipped."""
        adapter = _make_mock_adapter(fail_health=True)
        tuner = Autotuner(adapter)

        with pytest.raises(RuntimeError, match="No candidate"):
            tuner.tune([_make_plan()], WorkloadConfig())

    def test_max_candidates_limit(self) -> None:
        adapter = _make_mock_adapter()
        tuner = Autotuner(adapter, AutotunerConfig(max_candidates=2))

        plans = [_make_plan(gpu_layers=i) for i in range(5)]
        result = tuner.tune(plans, WorkloadConfig())

        # Should only load/unload for max 2 candidates
        assert adapter.load.call_count == 2
        assert adapter.unload.call_count == 2

    def test_unload_called_even_on_error(self) -> None:
        """adapter.unload() is called even if benchmark fails."""
        adapter = _make_mock_adapter()
        adapter.generate.side_effect = RuntimeError("inference error")

        tuner = Autotuner(adapter)

        # Should not raise — returns error in TuneResult
        with pytest.raises(RuntimeError, match="No candidate"):
            tuner.tune([_make_plan()], WorkloadConfig())

        adapter.unload.assert_called()


class TestAggregateMetrics:
    def test_empty_list(self) -> None:
        result = _aggregate_metrics([])
        assert result.tpot_ms == 0.0

    def test_single_metric(self) -> None:
        m = _make_metrics(tpot=100.0, ttft=500.0)
        result = _aggregate_metrics([m])
        assert result.tpot_ms == 100.0

    def test_median_of_three(self) -> None:
        metrics = [
            _make_metrics(tpot=100.0, ttft=500.0),
            _make_metrics(tpot=80.0, ttft=300.0),
            _make_metrics(tpot=120.0, ttft=700.0),
        ]
        result = _aggregate_metrics(metrics)
        assert result.tpot_ms == 100.0   # median
        assert result.ttft_ms == 500.0   # median

    def test_peak_vram_takes_max(self) -> None:
        metrics = [
            _make_metrics(vram=1000),
            _make_metrics(vram=3000),
            _make_metrics(vram=2000),
        ]
        result = _aggregate_metrics(metrics)
        assert result.peak_vram_bytes == 3000


class TestBenchmarkPrompt:
    def test_chat_prompt(self) -> None:
        wl = WorkloadConfig(workload_type="chat")
        prompt = _make_benchmark_prompt(wl)
        assert "attention" in prompt.lower() or "explain" in prompt.lower()

    def test_code_prompt(self) -> None:
        wl = WorkloadConfig(workload_type="code")
        prompt = _make_benchmark_prompt(wl)
        assert "python" in prompt.lower() or "function" in prompt.lower()

    def test_unknown_defaults_to_chat(self) -> None:
        wl = WorkloadConfig(workload_type="unknown_type")
        prompt = _make_benchmark_prompt(wl)
        assert len(prompt) > 10  # Should still return a usable prompt


class TestTuneResult:
    def test_defaults(self) -> None:
        r = TuneResult(
            plan=_make_plan(),
            metrics=_make_metrics(),
            score=42.0,
        )
        assert r.error is None
        assert r.score == 42.0

    def test_with_error(self) -> None:
        r = TuneResult(
            plan=_make_plan(),
            metrics=InferenceMetrics(),
            score=float("inf"),
            error="OOM",
        )
        assert r.error == "OOM"
        assert r.score == float("inf")

