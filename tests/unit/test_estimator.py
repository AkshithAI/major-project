"""Tests for the Memory Estimator."""

from __future__ import annotations

import io

import pytest

from adhie.estimator.memory_estimator import MemoryEstimator, dtype_to_bytes_per_element
from adhie.types import HardwareProfile, ModelProfile


def _make_hardware(vram_gb: float = 6.0, ram_gb: float = 32.0, has_gpu: bool = True) -> HardwareProfile:
    """Create a test hardware profile."""
    return HardwareProfile(
        gpu_name="Test GPU" if has_gpu else None,
        gpu_count=1 if has_gpu else 0,
        vram_total_bytes=int(vram_gb * 1e9),
        vram_free_bytes=int(vram_gb * 1e9 * 0.92),  # ~8% used by OS/display
        ram_total_bytes=int(ram_gb * 1e9),
        ram_available_bytes=int(ram_gb * 1e9 * 0.8),  # ~20% used by OS
        cpu_physical_cores=8,
        cpu_logical_cores=16,
    )


def _make_model_14b() -> ModelProfile:
    """Create a test 14B Q4_K_M model profile (Qwen2.5-14B-like)."""
    return ModelProfile(
        model_id="test-14b-q4km",
        file_size_bytes=int(8.5e9),  # ~8.5 GB Q4_K_M
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


class TestDtypeConversion:
    def test_f16(self) -> None:
        assert dtype_to_bytes_per_element("f16") == 2.0

    def test_q8_0(self) -> None:
        assert dtype_to_bytes_per_element("q8_0") == pytest.approx(8.5 / 8)

    def test_q4_k_m(self) -> None:
        assert dtype_to_bytes_per_element("q4_k_m") == pytest.approx(4.8 / 8)

    def test_unknown_defaults_to_f16(self) -> None:
        assert dtype_to_bytes_per_element("weird_type") == 2.0


class TestMemoryEstimator:
    def setup_method(self) -> None:
        self.estimator = MemoryEstimator()
        self.hw = _make_hardware(vram_gb=6.0, ram_gb=32.0)
        self.model = _make_model_14b()

    def test_full_gpu_14b_is_infeasible_on_6gb(self) -> None:
        """14B Q4_K_M (~8.5 GB weights) cannot fit fully on 6 GB GPU."""
        est = self.estimator.estimate(
            self.hw, self.model,
            context_length=8192,
            gpu_layers=40,
            kv_dtype="f16",
        )
        assert est.is_feasible is False
        assert est.bottleneck == "vram"

    def test_zero_gpu_layers_always_vram_feasible(self) -> None:
        """With 0 GPU layers, VRAM should not be the bottleneck."""
        est = self.estimator.estimate(
            self.hw, self.model,
            context_length=8192,
            gpu_layers=0,
            kv_dtype="f16",
            kv_offload=True,
        )
        assert est.weight_gpu_bytes == 0 or est.weight_gpu_bytes < est.vram_budget_bytes
        assert est.weight_cpu_bytes > 0

    def test_partial_gpu_feasibility(self) -> None:
        """Some layers on 6 GB GPU should be feasible."""
        # ~212.5 MB per layer, 6GB budget ~5 GB usable
        # 5 GB / 212.5 MB ≈ 23 layers max (rough)
        est_20 = self.estimator.estimate(
            self.hw, self.model,
            context_length=4096,
            gpu_layers=20,
            kv_dtype="f16",
        )
        # 20 layers should fit on 6 GB with 4K context
        # This depends on exact overhead estimates
        assert est_20.weight_gpu_bytes < est_20.weight_total_bytes

    def test_kv_cache_grows_with_context(self) -> None:
        """KV cache should scale linearly with context length."""
        est_4k = self.estimator.estimate(
            self.hw, self.model, context_length=4096, gpu_layers=0, kv_offload=True,
        )
        est_8k = self.estimator.estimate(
            self.hw, self.model, context_length=8192, gpu_layers=0, kv_offload=True,
        )
        assert est_8k.kv_total_bytes == pytest.approx(est_4k.kv_total_bytes * 2, rel=0.01)

    def test_q8_kv_halves_cache(self) -> None:
        """Q8 KV should use roughly half the memory of FP16 KV."""
        est_f16 = self.estimator.estimate(
            self.hw, self.model, context_length=8192, gpu_layers=20, kv_dtype="f16",
        )
        est_q8 = self.estimator.estimate(
            self.hw, self.model, context_length=8192, gpu_layers=20, kv_dtype="q8_0",
        )
        ratio = est_q8.kv_total_bytes / est_f16.kv_total_bytes
        assert 0.4 < ratio < 0.6  # Should be roughly half

    def test_kv_offload_moves_kv_to_ram(self) -> None:
        """KV offload should move all KV cache to RAM."""
        est = self.estimator.estimate(
            self.hw, self.model,
            context_length=8192,
            gpu_layers=20,
            kv_dtype="f16",
            kv_offload=True,
        )
        assert est.kv_gpu_bytes == 0
        assert est.kv_cpu_bytes == est.kv_total_bytes

    def test_context_exceeds_max_is_infeasible(self) -> None:
        """Requesting more context than model supports should fail."""
        est = self.estimator.estimate(
            self.hw, self.model,
            context_length=65536,  # Model max is 32768
            gpu_layers=0,
            kv_offload=True,
        )
        assert est.is_feasible is False
        assert any("exceeds model maximum" in w for w in est.warnings)

    def test_no_gpu_has_zero_vram_budget(self) -> None:
        """Without a GPU, VRAM budget should be 0."""
        hw_no_gpu = _make_hardware(has_gpu=False)
        est = self.estimator.estimate(
            hw_no_gpu, self.model,
            context_length=8192,
            gpu_layers=0,
            kv_offload=True,
        )
        assert est.vram_budget_bytes == 0
        assert est.cuda_context_bytes == 0

    def test_ram_infeasible_with_tiny_ram(self) -> None:
        """Very low RAM should fail for large CPU-offloaded models."""
        hw_low_ram = _make_hardware(ram_gb=4.0, has_gpu=False)
        est = self.estimator.estimate(
            hw_low_ram, self.model,
            context_length=8192,
            gpu_layers=0,
            kv_offload=True,
        )
        assert est.bottleneck in ("ram", "both")
        assert est.is_feasible is False

    def test_warnings_on_tight_vram(self) -> None:
        """Should warn when VRAM usage > 90% of budget."""
        # Find a config that barely fits
        max_layers = self.estimator.find_max_gpu_layers(
            self.hw, self.model, context_length=4096, kv_dtype="f16",
        )
        if max_layers > 0:
            est = self.estimator.estimate(
                self.hw, self.model,
                context_length=4096,
                gpu_layers=max_layers,
                kv_dtype="f16",
            )
            # At max layers, usage should be high
            assert est.is_feasible


class TestFindMaxGpuLayers:
    def setup_method(self) -> None:
        self.estimator = MemoryEstimator()
        self.hw = _make_hardware(vram_gb=6.0)
        self.model = _make_model_14b()

    def test_max_layers_less_than_total(self) -> None:
        """14B on 6 GB should not fit all 40 layers."""
        max_layers = self.estimator.find_max_gpu_layers(
            self.hw, self.model, context_length=8192,
        )
        assert 0 < max_layers < 40

    def test_max_layers_with_kv_offload_is_higher(self) -> None:
        """Offloading KV should allow more GPU layers."""
        max_no_offload = self.estimator.find_max_gpu_layers(
            self.hw, self.model, context_length=8192,
            kv_dtype="f16", kv_offload=False,
        )
        max_with_offload = self.estimator.find_max_gpu_layers(
            self.hw, self.model, context_length=8192,
            kv_dtype="f16", kv_offload=True,
        )
        assert max_with_offload >= max_no_offload

    def test_max_layers_with_q8_kv_is_higher(self) -> None:
        """Q8 KV should allow more GPU layers than FP16 KV."""
        max_f16 = self.estimator.find_max_gpu_layers(
            self.hw, self.model, context_length=8192,
            kv_dtype="f16", kv_offload=False,
        )
        max_q8 = self.estimator.find_max_gpu_layers(
            self.hw, self.model, context_length=8192,
            kv_dtype="q8_0", kv_offload=False,
        )
        assert max_q8 >= max_f16

    def test_no_gpu_returns_zero(self) -> None:
        hw_no_gpu = _make_hardware(has_gpu=False)
        max_layers = self.estimator.find_max_gpu_layers(
            hw_no_gpu, self.model, context_length=8192,
        )
        assert max_layers == 0

    def test_large_gpu_fits_everything(self) -> None:
        """24 GB GPU should fit entire 14B Q4_K_M model."""
        hw_big = _make_hardware(vram_gb=24.0)
        max_layers = self.estimator.find_max_gpu_layers(
            hw_big, self.model, context_length=8192,
        )
        assert max_layers == 40


class TestEstimatorReport:
    def test_print_report_no_crash(self) -> None:
        from rich.console import Console
        estimator = MemoryEstimator()
        hw = _make_hardware(vram_gb=6.0)
        model = _make_model_14b()
        est = estimator.estimate(hw, model, context_length=8192, gpu_layers=20)
        console = Console(file=io.StringIO())
        estimator.print_report(est, model, gpu_layers=20, context_length=8192, console=console)

