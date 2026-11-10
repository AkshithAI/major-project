"""Tests for benchmark report and ablation modules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adhie.benchmark.report import (
    AblationEntry,
    ComparisonRow,
    ablation_to_markdown,
    comparison_to_markdown,
    compute_ablation,
    generate_summary,
    get_plot_data,
    load_results,
    print_ablation_table,
    print_comparison_table,
    results_to_comparison,
)
from adhie.benchmark.ablation import (
    AblationConfig,
    build_ablation_configs,
    plan_pairs_from_configs,
    results_to_ablation_entries,
)
from adhie.types import (
    BenchmarkResult,
    HardwareProfile,
    InferenceMetrics,
    ModelProfile,
    RunPlan,
    WorkloadConfig,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _make_result(
    plan_id: str = "test-plan",
    gpu_layers: int = 10,
    mode: str = "hybrid",
    kv_type: str = "f16",
    tpot: float = 80.0,
    ttft: float = 400.0,
    tok_s: float = 12.5,
    vram: int = 4_000_000_000,
    spec_type: str = "none",
    draft_length: int = 0,
) -> BenchmarkResult:
    return BenchmarkResult(
        run_id=f"run-{plan_id}",
        timestamp="2026-01-01T00:00:00Z",
        hardware_fingerprint="hw-test",
        model_fingerprint="model-test",
        plan=RunPlan(
            plan_id=plan_id,
            gpu_layers=gpu_layers,
            execution_mode=mode,
            kv_type_k=kv_type,
            spec_type=spec_type,
            draft_length=draft_length,
        ),
        workload=WorkloadConfig(context_length=8192),
        metrics=InferenceMetrics(
            tpot_ms=tpot,
            ttft_ms=ttft,
            output_tokens_per_s=tok_s,
            itl_p95_ms=tpot * 1.2,
            peak_vram_bytes=vram,
            peak_ram_bytes=8_000_000_000,
        ),
    )


def _make_hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="RTX 3050",
        gpu_count=1,
        vram_total_bytes=6_000_000_000,
        vram_free_bytes=5_500_000_000,
        gpu_compute_capability=(8, 6),
        ram_total_bytes=16_000_000_000,
        ram_available_bytes=10_000_000_000,
    )


def _make_model() -> ModelProfile:
    return ModelProfile(
        model_id="TestModel-14B",
        parameter_count=14_000_000_000,
        layer_count=40,
        hidden_size=5120,
        num_kv_heads=8,
        head_dim=128,
    )


# ------------------------------------------------------------------
# Report Tests
# ------------------------------------------------------------------

class TestComparisonRow:
    def test_from_result(self) -> None:
        results = [_make_result()]
        rows = results_to_comparison(results, labels=["Test"])
        assert len(rows) == 1
        assert rows[0].label == "Test"
        assert rows[0].tpot_ms == 80.0
        assert rows[0].tokens_per_s == 12.5

    def test_multiple_results(self) -> None:
        results = [
            _make_result(plan_id="fast", tpot=50.0),
            _make_result(plan_id="slow", tpot=200.0),
        ]
        rows = results_to_comparison(results)
        assert len(rows) == 2


class TestAblationEntry:
    def test_compute_ablation(self) -> None:
        baseline = _make_result(tpot=100.0, ttft=500.0, tok_s=10.0)
        variant = _make_result(tpot=80.0, ttft=400.0, tok_s=12.5)
        entry = compute_ablation("Test Feature", baseline, variant)

        assert entry.feature_name == "Test Feature"
        assert entry.tpot_delta_pct < 0  # Variant is faster
        assert entry.tokens_per_s_delta_pct > 0  # Higher throughput
        assert "Beneficial" in entry.verdict

    def test_harmful_feature(self) -> None:
        baseline = _make_result(tpot=80.0, tok_s=12.5)
        variant = _make_result(tpot=120.0, tok_s=8.0)
        entry = compute_ablation("Bad Feature", baseline, variant)
        assert "Harmful" in entry.verdict

    def test_neutral_feature(self) -> None:
        baseline = _make_result(tpot=80.0, tok_s=12.5)
        variant = _make_result(tpot=81.0, tok_s=12.3)
        entry = compute_ablation("Neutral Feature", baseline, variant)
        assert "Neutral" in entry.verdict


class TestPrintTables:
    def test_comparison_table_no_crash(self) -> None:
        import io
        from rich.console import Console
        con = Console(file=io.StringIO())
        rows = results_to_comparison([_make_result()])
        print_comparison_table(rows, console=con)

    def test_ablation_table_no_crash(self) -> None:
        import io
        from rich.console import Console
        con = Console(file=io.StringIO())
        entries = [AblationEntry(
            feature_name="Test",
            baseline_label="OFF",
            variant_label="ON",
            baseline_tpot_ms=100.0,
            variant_tpot_ms=80.0,
            baseline_ttft_ms=500.0,
            variant_ttft_ms=400.0,
            baseline_tokens_per_s=10.0,
            variant_tokens_per_s=12.5,
        )]
        print_ablation_table(entries, console=con)


class TestMarkdown:
    def test_comparison_markdown(self) -> None:
        rows = results_to_comparison([_make_result()], labels=["MyPlan"])
        md = comparison_to_markdown(rows)
        assert "MyPlan" in md
        assert "|" in md

    def test_ablation_markdown(self) -> None:
        entries = [AblationEntry(
            feature_name="Test",
            baseline_label="OFF",
            variant_label="ON",
            baseline_tpot_ms=100.0,
            variant_tpot_ms=80.0,
            baseline_ttft_ms=500.0,
            variant_ttft_ms=400.0,
            baseline_tokens_per_s=10.0,
            variant_tokens_per_s=12.5,
        )]
        md = ablation_to_markdown(entries)
        assert "Test" in md
        assert "Beneficial" in md


class TestSummary:
    def test_empty(self) -> None:
        s = generate_summary([])
        assert s["count"] == 0

    def test_with_results(self) -> None:
        results = [
            _make_result(plan_id="a", tpot=100.0, tok_s=10.0),
            _make_result(plan_id="b", tpot=50.0, tok_s=20.0),
        ]
        s = generate_summary(results)
        assert s["count"] == 2
        assert s["best_tpot_ms"] == 50.0
        assert s["best_tokens_per_s"] == 20.0


class TestPlotData:
    def test_extract(self) -> None:
        results = [_make_result(plan_id="p1", tpot=80.0)]
        data = get_plot_data(results)
        assert len(data["labels"]) == 1
        assert data["tpot_ms"][0] == 80.0


class TestLoadResults:
    def test_load_from_empty_dir(self, tmp_path: Path) -> None:
        assert load_results(tmp_path) == []

    def test_load_nonexistent_dir(self, tmp_path: Path) -> None:
        assert load_results(tmp_path / "nope") == []

    def test_load_saved_results(self, tmp_path: Path) -> None:
        r = _make_result(plan_id="saved")
        r.to_json(tmp_path / "test.json")
        loaded = load_results(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].plan.plan_id == "saved"


# ------------------------------------------------------------------
# Ablation Config Tests
# ------------------------------------------------------------------

class TestBuildAblationConfigs:
    def test_basic_configs_without_draft(self) -> None:
        hw = _make_hw()
        model = _make_model()
        plan = RunPlan(gpu_layers=20, execution_mode="hybrid")
        configs = build_ablation_configs(plan, hw, model)

        names = [c.name for c in configs]
        assert "ADHIE vs Default" in names
        assert "KV Quantization" in names
        assert "KV Offload" in names
        # No speculation configs without draft
        assert "Speculation" not in names

    def test_configs_with_draft(self) -> None:
        hw = _make_hw()
        model = _make_model()
        plan = RunPlan(
            gpu_layers=20,
            execution_mode="hybrid",
            draft_model_path="/models/draft.gguf",
            draft_length=4,
            spec_type="draft-simple",
        )
        configs = build_ablation_configs(plan, hw, model)
        names = [c.name for c in configs]
        assert "Speculation" in names
        assert "Adaptive vs Fixed k" in names

    def test_no_kv_offload_without_gpu(self) -> None:
        hw = HardwareProfile()  # No GPU
        model = _make_model()
        plan = RunPlan(gpu_layers=0, execution_mode="cpu_first")
        configs = build_ablation_configs(plan, hw, model)
        names = [c.name for c in configs]
        assert "KV Offload" not in names


class TestPlanPairs:
    def test_generates_pairs(self) -> None:
        hw = _make_hw()
        model = _make_model()
        plan = RunPlan(gpu_layers=20, execution_mode="hybrid")
        configs = build_ablation_configs(plan, hw, model)
        pairs = plan_pairs_from_configs(configs, plan)

        for name, baseline, variant in pairs:
            assert isinstance(baseline, RunPlan)
            assert isinstance(variant, RunPlan)
            assert baseline.plan_id != variant.plan_id or name == "ADHIE vs Default"

    def test_adhie_vs_default_baseline_is_cpu(self) -> None:
        hw = _make_hw()
        model = _make_model()
        plan = RunPlan(gpu_layers=20, execution_mode="hybrid")
        configs = build_ablation_configs(plan, hw, model)
        pairs = plan_pairs_from_configs(configs, plan)

        # First config is ADHIE vs Default
        name, baseline, variant = pairs[0]
        assert name == "ADHIE vs Default"
        assert baseline.gpu_layers == 0
        assert variant.gpu_layers == 20


class TestResultsToAblationEntries:
    def test_match_results(self) -> None:
        hw = _make_hw()
        model = _make_model()
        plan = RunPlan(gpu_layers=20, execution_mode="hybrid")
        configs = build_ablation_configs(plan, hw, model)

        baselines = [_make_result(tpot=100.0, tok_s=10.0) for _ in configs]
        variants = [_make_result(tpot=80.0, tok_s=12.5) for _ in configs]

        entries = results_to_ablation_entries(configs, baselines, variants)
        assert len(entries) == len(configs)
        for e in entries:
            assert "Beneficial" in e.verdict

