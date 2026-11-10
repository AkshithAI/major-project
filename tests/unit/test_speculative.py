"""Tests for the Speculative Decoding Controller."""

from __future__ import annotations

import pytest

from adhie.speculative.controller import (
    SpecConfig,
    SpeculativeController,
    _estimate_draft_ram,
    _estimate_draft_vram,
)
from adhie.types import HardwareProfile, ModelProfile, RunPlan


def _make_hw(vram_gb: float = 6.0, ram_gb: float = 16.0, cores: int = 8) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Test GPU",
        vram_total_bytes=int(vram_gb * 1e9),
        vram_free_bytes=int(vram_gb * 1e9 * 0.95),
        gpu_count=1,
        cpu_physical_cores=cores,
        cpu_logical_cores=cores * 2,
        ram_total_bytes=int(ram_gb * 1e9),
        ram_available_bytes=int(ram_gb * 1e9 * 0.7),
    )


def _make_target(layers: int = 40, param_b: float = 14.0, size_gb: float = 8.0) -> ModelProfile:
    return ModelProfile(
        model_id="target-14B",
        model_path="/models/target.gguf",
        file_size_bytes=int(size_gb * 1e9),
        parameter_count=int(param_b * 1e9),
        layer_count=layers,
        hidden_size=5120,
        num_kv_heads=8,
        head_dim=128,
        weight_dtype="q4_k_m",
        bits_per_weight=4.8,
    )


def _make_draft(param_b: float = 0.5, size_gb: float = 0.3) -> ModelProfile:
    return ModelProfile(
        model_id="draft-0.5B",
        model_path="/models/draft.gguf",
        file_size_bytes=int(size_gb * 1e9),
        parameter_count=int(param_b * 1e9),
        layer_count=24,
        hidden_size=896,
        num_kv_heads=2,
        head_dim=64,
        weight_dtype="q4_k_m",
        bits_per_weight=4.8,
    )


def _make_base_plan() -> RunPlan:
    return RunPlan(
        model_path="/models/target.gguf",
        backend="llama.cpp",
        gpu_layers=20,
        cpu_threads=4,
        context_length=8192,
        batch_size=2048,
        ubatch_size=512,
    )


class TestSpecConfig:
    """Test SpecConfig dataclass."""

    def test_defaults(self) -> None:
        config = SpecConfig()
        assert config.enabled is False
        assert config.draft_length == 4
        assert config.spec_type == "none"

    def test_custom(self) -> None:
        config = SpecConfig(
            variant_id="B-k6",
            enabled=True,
            draft_device="cpu",
            draft_length=6,
            spec_type="sd",
        )
        assert config.variant_id == "B-k6"
        assert config.draft_length == 6
        assert config.draft_device == "cpu"


class TestEstimateHelpers:
    """Test memory estimation helpers."""

    def test_draft_ram_estimate(self) -> None:
        draft = _make_draft(size_gb=0.3)
        ram = _estimate_draft_ram(draft)
        assert ram > draft.file_size_bytes
        assert ram < draft.file_size_bytes + 1e9  # reasonable overhead

    def test_draft_vram_estimate(self) -> None:
        draft = _make_draft(size_gb=0.3)
        vram = _estimate_draft_vram(draft)
        assert vram > draft.file_size_bytes
        assert vram < draft.file_size_bytes + 1e9


class TestSpeculativeController:
    """Test experiment matrix generation."""

    def test_no_draft_returns_baseline_only(self) -> None:
        hw = _make_hw()
        target = _make_target()
        controller = SpeculativeController(hw, target, draft_profile=None)
        matrix = controller.build_experiment_matrix(20)

        assert len(matrix) == 1
        assert matrix[0].variant_id == "A"
        assert matrix[0].enabled is False

    def test_with_draft_includes_variant_b(self) -> None:
        hw = _make_hw(ram_gb=16.0)
        target = _make_target()
        draft = _make_draft()
        controller = SpeculativeController(hw, target, draft)
        matrix = controller.build_experiment_matrix(20)

        variant_ids = [c.variant_id for c in matrix]
        assert "A" in variant_ids
        assert any(v.startswith("B-") for v in variant_ids)

    def test_variant_b_has_four_k_values(self) -> None:
        hw = _make_hw(ram_gb=16.0)
        target = _make_target()
        draft = _make_draft()
        controller = SpeculativeController(hw, target, draft)
        matrix = controller.build_experiment_matrix(20)

        b_configs = [c for c in matrix if c.variant_id.startswith("B-")]
        assert len(b_configs) == 4
        k_values = [c.draft_length for c in b_configs]
        assert k_values == [2, 4, 6, 8]

    def test_variant_c_requires_vram(self) -> None:
        """Variant C (GPU draft) only included if VRAM has room."""
        # With 6GB VRAM and 20/40 layers (4GB used), ~1.5GB left
        # Draft is 0.3GB + overhead = ~0.43GB — should fit
        hw = _make_hw(vram_gb=6.0)
        target = _make_target()
        draft = _make_draft(size_gb=0.3)
        controller = SpeculativeController(hw, target, draft)
        matrix = controller.build_experiment_matrix(20)

        c_configs = [c for c in matrix if c.variant_id.startswith("C-")]
        assert len(c_configs) > 0

    def test_variant_c_excluded_when_no_vram(self) -> None:
        """Variant C should be excluded when VRAM is too small."""
        hw = _make_hw(vram_gb=6.0)
        target = _make_target()
        draft = _make_draft(size_gb=5.0)  # Huge draft that won't fit
        controller = SpeculativeController(hw, target, draft)
        matrix = controller.build_experiment_matrix(30)  # Many layers on GPU

        c_configs = [c for c in matrix if c.variant_id.startswith("C-")]
        assert len(c_configs) == 0

    def test_variant_d_all_cpu(self) -> None:
        hw = _make_hw(ram_gb=32.0)
        target = _make_target()
        draft = _make_draft()
        controller = SpeculativeController(hw, target, draft)
        matrix = controller.build_experiment_matrix(20)

        d_configs = [c for c in matrix if c.variant_id.startswith("D-")]
        assert len(d_configs) > 0
        for d in d_configs:
            assert d.gpu_layers == 0

    def test_variant_d_excluded_when_low_ram(self) -> None:
        """Variant D needs both models in RAM."""
        hw = _make_hw(ram_gb=4.0)  # Not enough for target + draft
        target = _make_target(size_gb=8.0)
        draft = _make_draft(size_gb=0.3)
        controller = SpeculativeController(hw, target, draft)
        matrix = controller.build_experiment_matrix(20)

        d_configs = [c for c in matrix if c.variant_id.startswith("D-")]
        assert len(d_configs) == 0

    def test_all_spec_configs_have_sd_type(self) -> None:
        hw = _make_hw(ram_gb=32.0)
        target = _make_target()
        draft = _make_draft()
        controller = SpeculativeController(hw, target, draft)
        matrix = controller.build_experiment_matrix(20)

        for c in matrix:
            if c.enabled:
                assert c.spec_type == "sd"
                assert c.draft_model_path is not None


class TestConfigToPlan:
    """Test converting SpecConfig to RunPlan."""

    def test_baseline_plan(self) -> None:
        hw = _make_hw()
        target = _make_target()
        controller = SpeculativeController(hw, target)
        config = SpecConfig(variant_id="A", gpu_layers=20)
        base = _make_base_plan()

        plan = controller.config_to_plan(config, base)
        assert plan.plan_id == "spec-A"
        assert plan.gpu_layers == 20
        assert plan.draft_model_path is None
        assert plan.spec_type == "none"

    def test_cpu_draft_plan(self) -> None:
        hw = _make_hw()
        target = _make_target()
        draft = _make_draft()
        controller = SpeculativeController(hw, target, draft)
        config = SpecConfig(
            variant_id="B-k4",
            variant_name="CPU draft + GPU target (k=4)",
            enabled=True,
            draft_model_path="/models/draft.gguf",
            draft_device="cpu",
            draft_length=4,
            draft_threads=4,
            spec_type="sd",
            gpu_layers=20,
        )
        base = _make_base_plan()

        plan = controller.config_to_plan(config, base)
        assert plan.plan_id == "spec-B-k4"
        assert plan.draft_model_path == "/models/draft.gguf"
        assert plan.draft_device == "cpu"
        assert plan.draft_length == 4
        assert plan.spec_type == "sd"
        assert plan.execution_mode == "hybrid"

    def test_cpu_only_plan(self) -> None:
        hw = _make_hw()
        target = _make_target()
        controller = SpeculativeController(hw, target)
        config = SpecConfig(variant_id="D-k2", gpu_layers=0)
        base = _make_base_plan()

        plan = controller.config_to_plan(config, base)
        assert plan.execution_mode == "cpu_first"

    def test_gpu_resident_plan(self) -> None:
        hw = _make_hw()
        target = _make_target(layers=40)
        controller = SpeculativeController(hw, target)
        config = SpecConfig(variant_id="test", gpu_layers=40)
        base = _make_base_plan()

        plan = controller.config_to_plan(config, base)
        assert plan.execution_mode == "gpu_resident"

    def test_base_plan_fields_preserved(self) -> None:
        hw = _make_hw()
        target = _make_target()
        controller = SpeculativeController(hw, target)
        config = SpecConfig(variant_id="A", gpu_layers=20)
        base = _make_base_plan()

        plan = controller.config_to_plan(config, base)
        assert plan.model_path == base.model_path
        assert plan.context_length == base.context_length
        assert plan.batch_size == base.batch_size


class TestFeasibilitySummary:
    """Test feasibility_variants_summary."""

    def test_no_draft_summary(self) -> None:
        hw = _make_hw()
        target = _make_target()
        controller = SpeculativeController(hw, target)
        summary = controller.feasible_variants_summary()

        assert len(summary) == 2
        assert summary[0]["feasible"] == "yes"
        assert summary[1]["feasible"] == "no"

    def test_with_draft_summary(self) -> None:
        hw = _make_hw(ram_gb=32.0, vram_gb=8.0)
        target = _make_target()
        draft = _make_draft()
        controller = SpeculativeController(hw, target, draft)
        summary = controller.feasible_variants_summary()

        assert len(summary) == 4
        variants = {s["variant"]: s["feasible"] for s in summary}
        assert variants["A (baseline)"] == "yes"
        assert variants["B (CPU draft)"] == "yes"

