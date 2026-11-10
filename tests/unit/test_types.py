"""Tests for shared ADHIE types — serialization, computation, round-trips."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from adhie.types import (
    BenchmarkResult,
    HardwareProfile,
    InferenceMetrics,
    MemoryEstimate,
    ModelProfile,
    RunPlan,
    WorkloadConfig,
)


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------

class TestHardwareProfile:
    def test_defaults(self) -> None:
        hp = HardwareProfile()
        assert hp.gpu_count == 0
        assert hp.vram_total_bytes == 0
        assert hp.has_gpu() is False

    def test_has_gpu(self) -> None:
        hp = HardwareProfile(gpu_count=1, vram_total_bytes=6_000_000_000)
        assert hp.has_gpu() is True

    def test_usable_vram(self) -> None:
        hp = HardwareProfile(
            vram_free_bytes=6_000_000_000,
            vram_safety_margin_bytes=512 * 1024 * 1024,
        )
        expected = 6_000_000_000 - 512 * 1024 * 1024
        assert hp.usable_vram_bytes() == expected

    def test_usable_vram_clamps_to_zero(self) -> None:
        hp = HardwareProfile(
            vram_free_bytes=100_000_000,
            vram_safety_margin_bytes=512 * 1024 * 1024,
        )
        assert hp.usable_vram_bytes() == 0

    def test_fingerprint_stable(self) -> None:
        hp = HardwareProfile(gpu_name="RTX 3060", vram_total_bytes=6_000_000_000)
        f1 = hp.fingerprint()
        f2 = hp.fingerprint()
        assert f1 == f2
        assert len(f1) == 16

    def test_yaml_round_trip(self) -> None:
        hp = HardwareProfile(
            gpu_name="RTX 4090",
            gpu_compute_capability=(8, 9),
            vram_total_bytes=24_000_000_000,
            vram_free_bytes=22_000_000_000,
            gpu_count=1,
            cpu_model="AMD Ryzen 9 7950X",
            cpu_physical_cores=16,
            cpu_logical_cores=32,
            ram_total_bytes=64_000_000_000,
            ram_available_bytes=55_000_000_000,
            h2d_bandwidth_bytes_s=12_000_000_000.0,
            cuda_version="12.4",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hw.yaml"
            hp.to_yaml(path)
            loaded = HardwareProfile.from_yaml(path)
            assert loaded.gpu_name == hp.gpu_name
            assert loaded.gpu_compute_capability == (8, 9)
            assert loaded.vram_total_bytes == hp.vram_total_bytes
            assert loaded.cpu_model == hp.cpu_model
            assert loaded.h2d_bandwidth_bytes_s == hp.h2d_bandwidth_bytes_s


# ---------------------------------------------------------------------------
# ModelProfile
# ---------------------------------------------------------------------------

class TestModelProfile:
    def test_kv_bytes_per_token(self) -> None:
        """Test KV cache calculation for a GQA model."""
        mp = ModelProfile(
            layer_count=40,
            num_kv_heads=8,
            head_dim=128,
        )
        # FP16: 2 * 40 * 8 * 128 * 2 = 163840 bytes per token
        result = mp.kv_bytes_per_token(kv_dtype_bits=16)
        assert result == 2 * 40 * 8 * 128 * 2

    def test_kv_bytes_per_token_q8(self) -> None:
        mp = ModelProfile(layer_count=40, num_kv_heads=8, head_dim=128)
        result_q8 = mp.kv_bytes_per_token(kv_dtype_bits=8)
        result_f16 = mp.kv_bytes_per_token(kv_dtype_bits=16)
        assert result_q8 == result_f16 / 2

    def test_weight_bytes_per_layer(self) -> None:
        mp = ModelProfile(file_size_bytes=8_500_000_000, layer_count=40)
        per_layer = mp.weight_bytes_per_layer()
        assert per_layer == pytest.approx(8_500_000_000 / 40)

    def test_weight_bytes_per_layer_zero_layers(self) -> None:
        mp = ModelProfile(file_size_bytes=8_500_000_000, layer_count=0)
        assert mp.weight_bytes_per_layer() == 0.0

    def test_yaml_round_trip(self) -> None:
        mp = ModelProfile(
            model_id="qwen2.5-14b-q4_k_m",
            architecture="qwen2",
            parameter_count=14_700_000_000,
            layer_count=40,
            num_kv_heads=8,
            head_dim=128,
            max_context=32768,
            weight_dtype="q4_k_m",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.yaml"
            mp.to_yaml(path)
            loaded = ModelProfile.from_yaml(path)
            assert loaded.model_id == mp.model_id
            assert loaded.parameter_count == mp.parameter_count
            assert loaded.num_kv_heads == 8


# ---------------------------------------------------------------------------
# WorkloadConfig
# ---------------------------------------------------------------------------

class TestWorkloadConfig:
    def test_defaults(self) -> None:
        wc = WorkloadConfig()
        assert wc.context_length == 8192
        assert wc.temperature == 0.0
        assert wc.seed == 42

    def test_yaml_round_trip(self) -> None:
        wc = WorkloadConfig(context_length=16384, output_tokens=512)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wl.yaml"
            wc.to_yaml(path)
            loaded = WorkloadConfig.from_yaml(path)
            assert loaded.context_length == 16384
            assert loaded.output_tokens == 512


# ---------------------------------------------------------------------------
# RunPlan
# ---------------------------------------------------------------------------

class TestRunPlan:
    def test_defaults(self) -> None:
        rp = RunPlan()
        assert rp.backend == "llamacpp"
        assert rp.gpu_layers == 0
        assert rp.draft_device == "none"
        assert rp.execution_mode == "unknown"

    def test_yaml_round_trip(self) -> None:
        rp = RunPlan(
            plan_id="test-plan-1",
            gpu_layers=24,
            context_length=8192,
            kv_type_k="q8_0",
            execution_mode="hybrid",
            rationale="24/40 layers fit in 5.6GB VRAM budget",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plan.yaml"
            rp.to_yaml(path)
            loaded = RunPlan.from_yaml(path)
            assert loaded.gpu_layers == 24
            assert loaded.execution_mode == "hybrid"


# ---------------------------------------------------------------------------
# MemoryEstimate
# ---------------------------------------------------------------------------

class TestMemoryEstimate:
    def test_feasibility_flag(self) -> None:
        me = MemoryEstimate(is_feasible=True, bottleneck="none")
        assert me.is_feasible is True
        assert me.bottleneck == "none"


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def test_json_round_trip(self) -> None:
        br = BenchmarkResult(
            run_id="test-001",
            timestamp="2026-11-10T10:00:00Z",
            plan=RunPlan(gpu_layers=20, context_length=8192),
            metrics=InferenceMetrics(ttft_ms=1200.0, tpot_ms=85.0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            br.to_json(path)
            loaded = BenchmarkResult.from_json(path)
            assert loaded.run_id == "test-001"
            assert loaded.plan.gpu_layers == 20
            assert loaded.metrics.ttft_ms == 1200.0

