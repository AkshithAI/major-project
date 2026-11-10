"""Tests for the Draft Length Sweep."""

from __future__ import annotations

import pytest

from adhie.runtime.base import GenerateRequest, GenerateResponse, RuntimeAdapter
from adhie.speculative.controller import SpecConfig, SpeculativeController
from adhie.speculative.draft_sweep import (
    DraftLengthSweep,
    SweepResult,
    VariantSweepResult,
)
from adhie.types import (
    HardwareProfile,
    InferenceMetrics,
    ModelProfile,
    RunPlan,
)


# ---------------------------------------------------------------------------
# Fake adapter for testing (no subprocess / network)
# ---------------------------------------------------------------------------

class FakeSweepAdapter(RuntimeAdapter):
    """Mock adapter that returns canned metrics based on plan config."""

    def __init__(self) -> None:
        self._loaded = False
        self._last_metrics: InferenceMetrics | None = None
        self.load_count = 0
        self.unload_count = 0

    def load(self, plan: RunPlan) -> None:
        self._loaded = True
        self.load_count += 1
        # Simulate: more GPU layers → faster, higher k → diminishing returns
        base_tok_s = 20.0 + plan.gpu_layers * 0.5
        k_bonus = max(0, plan.draft_length * 2 - plan.draft_length ** 0.5)
        self._last_metrics = InferenceMetrics(
            ttft_ms=200.0 - plan.gpu_layers * 2,
            tpot_ms=50.0 - plan.gpu_layers * 0.5,
            output_tokens_per_s=base_tok_s + k_bonus,
            total_output_tokens=64,
            total_prompt_tokens=20,
            total_time_s=1.0,
        )

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        if not self._loaded:
            raise RuntimeError("Not loaded")
        return GenerateResponse(
            text="Hello " * 64,
            prompt_tokens=20,
            output_tokens=64,
            metrics=self._last_metrics,
        )

    def generate_simple(self, prompt: str, max_tokens: int = 256) -> str:
        return "Hello"

    def health_check(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False
        self.unload_count += 1

    def last_metrics(self) -> InferenceMetrics | None:
        return self._last_metrics


def _make_hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Test GPU", vram_total_bytes=int(6e9),
        vram_free_bytes=int(5.5e9), gpu_count=1,
        cpu_physical_cores=8, cpu_logical_cores=16,
        ram_total_bytes=int(16e9), ram_available_bytes=int(11e9),
    )


def _make_target() -> ModelProfile:
    return ModelProfile(
        model_id="target-14B", model_path="/models/target.gguf",
        file_size_bytes=int(8e9), parameter_count=int(14e9),
        layer_count=40, hidden_size=5120, weight_dtype="q4_k_m",
    )


def _make_draft() -> ModelProfile:
    return ModelProfile(
        model_id="draft-0.5B", model_path="/models/draft.gguf",
        file_size_bytes=int(0.3e9), parameter_count=int(0.5e9),
        layer_count=24, hidden_size=896, weight_dtype="q4_k_m",
    )


def _make_base_plan() -> RunPlan:
    return RunPlan(
        model_path="/models/target.gguf",
        gpu_layers=20, cpu_threads=4,
        context_length=8192, batch_size=2048, ubatch_size=512,
    )


class TestSweepResult:
    """Test SweepResult dataclass."""

    def test_defaults(self) -> None:
        sr = SweepResult()
        assert sr.variant_id == ""
        assert sr.error is None

    def test_with_error(self) -> None:
        sr = SweepResult(variant_id="B-k4", error="Server crashed")
        assert sr.error is not None


class TestVariantSweepResult:
    """Test VariantSweepResult properties."""

    def test_is_beneficial(self) -> None:
        vr = VariantSweepResult(best_tokens_per_s=25.0)
        assert vr.is_beneficial is True

    def test_is_not_beneficial(self) -> None:
        vr = VariantSweepResult(best_tokens_per_s=0.0)
        assert vr.is_beneficial is False


class TestDraftLengthSweepVariant:
    """Test run_variant_sweep with a fake adapter."""

    def test_sweep_four_k_values(self) -> None:
        adapter = FakeSweepAdapter()
        hw = _make_hw()
        target = _make_target()
        draft = _make_draft()
        controller = SpeculativeController(hw, target, draft)

        # Create B-variant configs with k=2,4,6,8
        configs = [
            SpecConfig(
                variant_id=f"B-k{k}",
                variant_name=f"CPU draft + GPU target (k={k})",
                enabled=True,
                draft_model_path="/models/draft.gguf",
                draft_device="cpu",
                draft_length=k,
                draft_threads=4,
                spec_type="sd",
                gpu_layers=20,
            )
            for k in [2, 4, 6, 8]
        ]

        sweep = DraftLengthSweep(adapter, controller, runner=None)
        result = sweep.run_variant_sweep(configs, _make_base_plan())

        assert result.variant_id == "B"
        assert len(result.sweep_results) == 4
        assert result.best_k > 0
        assert result.best_tokens_per_s > 0

    def test_empty_configs(self) -> None:
        adapter = FakeSweepAdapter()
        hw = _make_hw()
        target = _make_target()
        controller = SpeculativeController(hw, target)

        sweep = DraftLengthSweep(adapter, controller)
        result = sweep.run_variant_sweep([], _make_base_plan())
        assert result.variant_id == ""
        assert len(result.sweep_results) == 0


class TestFullMatrix:
    """Test run_full_matrix across all variants."""

    def test_full_matrix_runs_all_variants(self) -> None:
        adapter = FakeSweepAdapter()
        hw = _make_hw()
        target = _make_target()
        draft = _make_draft()
        controller = SpeculativeController(hw, target, draft)

        matrix = controller.build_experiment_matrix(20)
        sweep = DraftLengthSweep(adapter, controller, runner=None)

        results = sweep.run_full_matrix(matrix, _make_base_plan())

        # Should have at least baseline (A) and one variant
        assert len(results) >= 2
        variant_ids = [r.variant_id for r in results]
        assert "A" in variant_ids

    def test_print_sweep_summary_no_crash(self) -> None:
        """Ensure summary printing doesn't crash."""
        adapter = FakeSweepAdapter()
        hw = _make_hw()
        target = _make_target()
        draft = _make_draft()
        controller = SpeculativeController(hw, target, draft)

        matrix = controller.build_experiment_matrix(20)
        sweep = DraftLengthSweep(adapter, controller, runner=None)
        results = sweep.run_full_matrix(matrix, _make_base_plan())

        # Should not raise
        sweep.print_sweep_summary(results)

