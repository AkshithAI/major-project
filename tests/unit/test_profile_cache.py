"""Tests for the Profile Cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from adhie.planner.profile_cache import ProfileCache
from adhie.types import (
    HardwareProfile,
    InferenceMetrics,
    ModelProfile,
    RunPlan,
    WorkloadConfig,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_profiles"


@pytest.fixture
def cache(cache_dir: Path) -> ProfileCache:
    return ProfileCache(cache_dir=cache_dir)


@pytest.fixture
def hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="RTX 3050",
        vram_total_bytes=6_000_000_000,
        vram_free_bytes=5_500_000_000,
        gpu_compute_capability=(8, 6),
        cpu_model="Intel i7",
        cpu_physical_cores=6,
        cpu_logical_cores=12,
        ram_total_bytes=16_000_000_000,
        ram_available_bytes=10_000_000_000,
    )


@pytest.fixture
def model() -> ModelProfile:
    return ModelProfile(
        model_id="Qwen2.5-14B-Instruct",
        model_path="/models/qwen.gguf",
        file_size_bytes=9_000_000_000,
        parameter_count=14_700_000_000,
        architecture="qwen2",
        layer_count=40,
        hidden_size=5120,
        num_kv_heads=8,
        head_dim=128,
    )


@pytest.fixture
def workload() -> WorkloadConfig:
    return WorkloadConfig(context_length=8192, workload_type="chat")


@pytest.fixture
def plan() -> RunPlan:
    return RunPlan(
        plan_id="test-plan-1",
        gpu_layers=24,
        execution_mode="hybrid",
        kv_type_k="q8_0",
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestProfileCacheLookup:
    def test_miss_returns_none(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, workload: WorkloadConfig,
    ) -> None:
        assert cache.lookup(hw, model, workload) is None

    def test_hit_after_store(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, workload: WorkloadConfig, plan: RunPlan,
    ) -> None:
        cache.store(hw, model, workload, plan)
        result = cache.lookup(hw, model, workload)

        assert result is not None
        assert result.gpu_layers == 24
        assert result.execution_mode == "hybrid"
        assert result.kv_type_k == "q8_0"

    def test_different_workload_is_miss(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, plan: RunPlan,
    ) -> None:
        wl1 = WorkloadConfig(context_length=8192, workload_type="chat")
        wl2 = WorkloadConfig(context_length=4096, workload_type="code")

        cache.store(hw, model, wl1, plan)
        assert cache.lookup(hw, model, wl2) is None

    def test_different_model_is_miss(
        self, cache: ProfileCache, hw: HardwareProfile,
        workload: WorkloadConfig, plan: RunPlan,
    ) -> None:
        model1 = ModelProfile(model_id="ModelA", parameter_count=7_000_000_000,
                              layer_count=32, hidden_size=4096, num_kv_heads=8, head_dim=128)
        model2 = ModelProfile(model_id="ModelB", parameter_count=14_000_000_000,
                              layer_count=40, hidden_size=5120, num_kv_heads=8, head_dim=128)

        cache.store(hw, model1, workload, plan)
        assert cache.lookup(hw, model2, workload) is None


class TestProfileCacheStore:
    def test_creates_cache_dir(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, workload: WorkloadConfig, plan: RunPlan,
    ) -> None:
        assert not cache.cache_dir.exists()
        cache.store(hw, model, workload, plan)
        assert cache.cache_dir.exists()

    def test_returns_path(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, workload: WorkloadConfig, plan: RunPlan,
    ) -> None:
        path = cache.store(hw, model, workload, plan)
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_with_metrics(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, workload: WorkloadConfig, plan: RunPlan,
    ) -> None:
        metrics = InferenceMetrics(tpot_ms=85.0, ttft_ms=1200.0)
        path = cache.store(hw, model, workload, plan, metrics)
        assert path.exists()

        # Verify YAML content includes metrics
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "metrics" in data
        assert data["metrics"]["tpot_ms"] == 85.0

    def test_overwrite_existing(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, workload: WorkloadConfig,
    ) -> None:
        plan1 = RunPlan(plan_id="plan-1", gpu_layers=20)
        plan2 = RunPlan(plan_id="plan-2", gpu_layers=30)

        cache.store(hw, model, workload, plan1)
        cache.store(hw, model, workload, plan2)

        result = cache.lookup(hw, model, workload)
        assert result is not None
        assert result.gpu_layers == 30  # Updated


class TestProfileCacheInvalidate:
    def test_invalidate_existing(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, workload: WorkloadConfig, plan: RunPlan,
    ) -> None:
        cache.store(hw, model, workload, plan)
        assert cache.lookup(hw, model, workload) is not None

        removed = cache.invalidate(hw, model, workload)
        assert removed is True
        assert cache.lookup(hw, model, workload) is None

    def test_invalidate_missing(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, workload: WorkloadConfig,
    ) -> None:
        assert cache.invalidate(hw, model, workload) is False


class TestProfileCacheList:
    def test_empty_cache(self, cache: ProfileCache) -> None:
        assert cache.list_entries() == []

    def test_list_after_store(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, workload: WorkloadConfig, plan: RunPlan,
    ) -> None:
        cache.store(hw, model, workload, plan)
        entries = cache.list_entries()
        assert len(entries) == 1
        assert entries[0]["model_id"] == "Qwen2.5-14B-Instruct"


class TestProfileCacheClear:
    def test_clear_empty(self, cache: ProfileCache) -> None:
        assert cache.clear() == 0

    def test_clear_populated(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, plan: RunPlan,
    ) -> None:
        wl1 = WorkloadConfig(context_length=4096)
        wl2 = WorkloadConfig(context_length=8192)
        cache.store(hw, model, wl1, plan)
        cache.store(hw, model, wl2, plan)

        count = cache.clear()
        assert count == 2
        assert cache.list_entries() == []


class TestHashKeyDeterminism:
    def test_same_inputs_same_key(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile, workload: WorkloadConfig,
    ) -> None:
        key1 = cache._hash_key(hw, model, workload)
        key2 = cache._hash_key(hw, model, workload)
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex

    def test_different_inputs_different_key(
        self, cache: ProfileCache, hw: HardwareProfile,
        model: ModelProfile,
    ) -> None:
        wl1 = WorkloadConfig(context_length=4096)
        wl2 = WorkloadConfig(context_length=8192)
        assert cache._hash_key(hw, model, wl1) != cache._hash_key(hw, model, wl2)

