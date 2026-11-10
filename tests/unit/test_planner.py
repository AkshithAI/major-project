"""Tests for the Plan Generator."""

from __future__ import annotations

import io

import pytest

from adhie.planner.plan_generator import PlanGenerator
from adhie.estimator.memory_estimator import MemoryEstimator
from adhie.types import HardwareProfile, ModelProfile, WorkloadConfig


def _make_hardware(vram_gb: float = 6.0, ram_gb: float = 32.0, has_gpu: bool = True) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Test GPU" if has_gpu else None,
        gpu_count=1 if has_gpu else 0,
        vram_total_bytes=int(vram_gb * 1e9),
        vram_free_bytes=int(vram_gb * 1e9 * 0.92),
        ram_total_bytes=int(ram_gb * 1e9),
        ram_available_bytes=int(ram_gb * 1e9 * 0.8),
        cpu_physical_cores=8,
        cpu_logical_cores=16,
    )


def _make_model_14b() -> ModelProfile:
    return ModelProfile(
        model_id="test-14b-q4km",
        model_path="/fake/model.gguf",
        file_size_bytes=int(8.5e9),
        architecture="qwen2",
        parameter_count=14_700_000_000,
        layer_count=40,
        hidden_size=5120,
        intermediate_size=13824,
        vocab_size=152064,
        num_attention_heads=40,
        num_kv_heads=8,
        head_dim=128,
        max_context=32768,
        weight_dtype="q4_k_m",
        bits_per_weight=4.8,
    )


def _make_model_small() -> ModelProfile:
    """A small model that fits fully on a 6 GB GPU."""
    return ModelProfile(
        model_id="test-1b-q4km",
        model_path="/fake/small_model.gguf",
        file_size_bytes=int(0.8e9),
        architecture="qwen2",
        parameter_count=500_000_000,
        layer_count=24,
        hidden_size=2048,
        intermediate_size=5504,
        vocab_size=152064,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        max_context=32768,
        weight_dtype="q4_k_m",
        bits_per_weight=4.8,
    )


class TestPlanGeneration:
    def setup_method(self) -> None:
        self.gen = PlanGenerator()
        self.hw_6gb = _make_hardware(vram_gb=6.0)
        self.hw_24gb = _make_hardware(vram_gb=24.0)
        self.hw_no_gpu = _make_hardware(has_gpu=False)
        self.model_14b = _make_model_14b()
        self.model_small = _make_model_small()
        self.workload = WorkloadConfig(context_length=8192)

    def test_produces_at_least_one_plan(self) -> None:
        """Should find at least one feasible plan for 14B on 6 GB + 32 GB RAM."""
        plans = self.gen.generate_candidates(self.hw_6gb, self.model_14b, self.workload)
        assert len(plans) >= 1

    def test_no_full_gpu_plan_for_14b_on_6gb(self) -> None:
        """14B should not get a gpu_resident plan on 6 GB."""
        plans = self.gen.generate_candidates(self.hw_6gb, self.model_14b, self.workload)
        gpu_resident = [p for p in plans if p.execution_mode == "gpu_resident"]
        assert len(gpu_resident) == 0

    def test_full_gpu_plan_for_small_model_on_6gb(self) -> None:
        """Small model should get a gpu_resident plan on 6 GB."""
        plans = self.gen.generate_candidates(self.hw_6gb, self.model_small, self.workload)
        gpu_resident = [p for p in plans if p.execution_mode == "gpu_resident"]
        assert len(gpu_resident) >= 1

    def test_full_gpu_plan_for_14b_on_24gb(self) -> None:
        """14B should fit fully on 24 GB GPU."""
        plans = self.gen.generate_candidates(self.hw_24gb, self.model_14b, self.workload)
        gpu_resident = [p for p in plans if p.execution_mode == "gpu_resident"]
        assert len(gpu_resident) >= 1

    def test_no_gpu_only_cpu_plans(self) -> None:
        """Without GPU, all plans should be cpu_first."""
        plans = self.gen.generate_candidates(self.hw_no_gpu, self.model_14b, self.workload)
        for plan in plans:
            assert plan.gpu_layers == 0
            assert plan.execution_mode == "cpu_first"

    def test_plans_are_sorted_best_first(self) -> None:
        """First plan should have the best (lowest) heuristic score."""
        plans = self.gen.generate_candidates(self.hw_6gb, self.model_14b, self.workload)
        if len(plans) >= 2:
            # Best plan should have more GPU layers than worst
            assert plans[0].gpu_layers >= plans[-1].gpu_layers or \
                   plans[0].kv_offload != plans[-1].kv_offload

    def test_all_plans_are_feasible(self) -> None:
        """Generator should only return feasible plans."""
        estimator = MemoryEstimator()
        plans = self.gen.generate_candidates(self.hw_6gb, self.model_14b, self.workload)
        for plan in plans:
            est = estimator.estimate(
                self.hw_6gb, self.model_14b,
                context_length=plan.context_length,
                gpu_layers=plan.gpu_layers,
                kv_dtype=plan.kv_type_k,
                kv_offload=plan.kv_offload,
            )
            assert est.is_feasible, (
                f"Plan {plan.plan_id} (gpu_layers={plan.gpu_layers}, "
                f"kv={plan.kv_type_k}, offload={plan.kv_offload}) is not feasible"
            )

    def test_plans_have_rationale(self) -> None:
        """Every plan should have a non-empty rationale."""
        plans = self.gen.generate_candidates(self.hw_6gb, self.model_14b, self.workload)
        for plan in plans:
            assert plan.rationale, f"Plan {plan.plan_id} has empty rationale"

    def test_plans_have_memory_estimates(self) -> None:
        """Every plan should have non-zero memory estimates."""
        plans = self.gen.generate_candidates(self.hw_6gb, self.model_14b, self.workload)
        for plan in plans:
            assert plan.estimated_ram_bytes > 0 or plan.estimated_vram_bytes > 0

    def test_context_clamped_to_model_max(self) -> None:
        """If workload requests more than model max, plans use model max."""
        big_ctx = WorkloadConfig(context_length=999999)
        plans = self.gen.generate_candidates(self.hw_24gb, self.model_14b, big_ctx)
        for plan in plans:
            assert plan.context_length <= self.model_14b.max_context

    def test_hybrid_plans_have_mixed_layers(self) -> None:
        """Hybrid plans should have 0 < gpu_layers < total layers."""
        plans = self.gen.generate_candidates(self.hw_6gb, self.model_14b, self.workload)
        hybrid = [p for p in plans if p.execution_mode == "hybrid"]
        for plan in hybrid:
            assert 0 < plan.gpu_layers < self.model_14b.layer_count


class TestPlanFields:
    def setup_method(self) -> None:
        self.gen = PlanGenerator()

    def test_backend_is_llamacpp(self) -> None:
        hw = _make_hardware()
        model = _make_model_14b()
        plans = self.gen.generate_candidates(hw, model, WorkloadConfig())
        for plan in plans:
            assert plan.backend == "llamacpp"

    def test_model_path_set(self) -> None:
        hw = _make_hardware()
        model = _make_model_14b()
        plans = self.gen.generate_candidates(hw, model, WorkloadConfig())
        for plan in plans:
            assert plan.model_path == model.model_path

    def test_kv_type_k_and_v_match(self) -> None:
        hw = _make_hardware()
        model = _make_model_14b()
        plans = self.gen.generate_candidates(hw, model, WorkloadConfig())
        for plan in plans:
            assert plan.kv_type_k == plan.kv_type_v

    def test_cpu_threads_positive(self) -> None:
        hw = _make_hardware()
        model = _make_model_14b()
        plans = self.gen.generate_candidates(hw, model, WorkloadConfig())
        for plan in plans:
            assert plan.cpu_threads >= 2


class TestPlanReport:
    def test_print_candidates_no_crash(self) -> None:
        from rich.console import Console
        gen = PlanGenerator()
        hw = _make_hardware()
        model = _make_model_14b()
        plans = gen.generate_candidates(hw, model, WorkloadConfig())
        console = Console(file=io.StringIO())
        gen.print_candidates(plans, console)

    def test_print_empty_candidates(self) -> None:
        from rich.console import Console
        gen = PlanGenerator()
        console = Console(file=io.StringIO())
        gen.print_candidates([], console)

