"""Benchmark Report — compare results, generate ablation tables and summary reports.

Loads BenchmarkResult JSON files, computes deltas, and produces:
  - Comparison tables (Rich and Markdown)
  - Ablation delta tables (feature on/off)
  - Summary statistics
  - Data suitable for matplotlib plotting
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from adhie.types import BenchmarkResult, InferenceMetrics


@dataclass
class ComparisonRow:
    """One row in a comparison table."""

    label: str
    plan_id: str
    execution_mode: str
    gpu_layers: int
    kv_type: str
    ttft_ms: float
    tpot_ms: float
    itl_p95_ms: float
    tokens_per_s: float
    peak_vram_gb: float
    peak_ram_gb: float
    spec_type: str = "none"
    draft_length: int = 0


@dataclass
class AblationEntry:
    """One ablation study: feature ON vs OFF."""

    feature_name: str
    baseline_label: str
    variant_label: str
    baseline_tpot_ms: float
    variant_tpot_ms: float
    baseline_ttft_ms: float
    variant_ttft_ms: float
    baseline_tokens_per_s: float
    variant_tokens_per_s: float
    tpot_delta_pct: float = 0.0
    ttft_delta_pct: float = 0.0
    tokens_per_s_delta_pct: float = 0.0
    verdict: str = ""

    def __post_init__(self) -> None:
        if self.baseline_tpot_ms > 0:
            self.tpot_delta_pct = (
                (self.variant_tpot_ms - self.baseline_tpot_ms)
                / self.baseline_tpot_ms * 100
            )
        if self.baseline_ttft_ms > 0:
            self.ttft_delta_pct = (
                (self.variant_ttft_ms - self.baseline_ttft_ms)
                / self.baseline_ttft_ms * 100
            )
        if self.baseline_tokens_per_s > 0:
            self.tokens_per_s_delta_pct = (
                (self.variant_tokens_per_s - self.baseline_tokens_per_s)
                / self.baseline_tokens_per_s * 100
            )
        # Positive tokens_per_s_delta means variant is faster
        if self.tokens_per_s_delta_pct > 5:
            self.verdict = "✓ Beneficial"
        elif self.tokens_per_s_delta_pct < -5:
            self.verdict = "✗ Harmful"
        else:
            self.verdict = "— Neutral"


def load_results(directory: Path) -> list[BenchmarkResult]:
    """Load all BenchmarkResult JSON files from a directory."""
    results = []
    if not directory.exists():
        return results
    for f in sorted(directory.glob("*.json")):
        try:
            results.append(BenchmarkResult.from_json(f))
        except Exception:
            continue
    return results


def results_to_comparison(
    results: list[BenchmarkResult],
    labels: list[str] | None = None,
) -> list[ComparisonRow]:
    """Convert BenchmarkResults to ComparisonRows."""
    rows = []
    for i, r in enumerate(results):
        label = labels[i] if labels and i < len(labels) else r.run_id
        rows.append(ComparisonRow(
            label=label,
            plan_id=r.plan.plan_id,
            execution_mode=r.plan.execution_mode,
            gpu_layers=r.plan.gpu_layers,
            kv_type=r.plan.kv_type_k,
            ttft_ms=r.metrics.ttft_ms,
            tpot_ms=r.metrics.tpot_ms,
            itl_p95_ms=r.metrics.itl_p95_ms,
            tokens_per_s=r.metrics.output_tokens_per_s,
            peak_vram_gb=r.metrics.peak_vram_bytes / 1e9,
            peak_ram_gb=r.metrics.peak_ram_bytes / 1e9,
            spec_type=r.plan.spec_type,
            draft_length=r.plan.draft_length,
        ))
    return rows


def compute_ablation(
    feature_name: str,
    baseline: BenchmarkResult,
    variant: BenchmarkResult,
    baseline_label: str = "OFF",
    variant_label: str = "ON",
) -> AblationEntry:
    """Compute ablation deltas between two results."""
    return AblationEntry(
        feature_name=feature_name,
        baseline_label=baseline_label,
        variant_label=variant_label,
        baseline_tpot_ms=baseline.metrics.tpot_ms,
        variant_tpot_ms=variant.metrics.tpot_ms,
        baseline_ttft_ms=baseline.metrics.ttft_ms,
        variant_ttft_ms=variant.metrics.ttft_ms,
        baseline_tokens_per_s=baseline.metrics.output_tokens_per_s,
        variant_tokens_per_s=variant.metrics.output_tokens_per_s,
    )


def print_comparison_table(
    rows: list[ComparisonRow],
    console: Console | None = None,
    title: str = "Benchmark Comparison",
) -> None:
    """Print a Rich comparison table."""
    con = console or Console()
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Label", style="bold")
    table.add_column("Mode")
    table.add_column("GPU Layers", justify="right")
    table.add_column("KV", justify="center")
    table.add_column("Spec")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("TPOT (ms)", justify="right")
    table.add_column("ITL P95", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("VRAM (GB)", justify="right")
    table.add_column("RAM (GB)", justify="right")

    for r in rows:
        spec = r.spec_type
        if r.draft_length > 0:
            spec += f" k={r.draft_length}"
        table.add_row(
            r.label, r.execution_mode,
            str(r.gpu_layers), r.kv_type, spec,
            f"{r.ttft_ms:.0f}", f"{r.tpot_ms:.1f}", f"{r.itl_p95_ms:.1f}",
            f"[bold green]{r.tokens_per_s:.1f}[/bold green]",
            f"{r.peak_vram_gb:.2f}" if r.peak_vram_gb > 0 else "—",
            f"{r.peak_ram_gb:.2f}" if r.peak_ram_gb > 0 else "—",
        )
    con.print(table)


def print_ablation_table(
    entries: list[AblationEntry],
    console: Console | None = None,
    title: str = "Ablation Study",
) -> None:
    """Print a Rich ablation results table."""
    con = console or Console()
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Feature", style="bold")
    table.add_column("Baseline")
    table.add_column("Variant")
    table.add_column("TPOT Δ", justify="right")
    table.add_column("TTFT Δ", justify="right")
    table.add_column("tok/s Δ", justify="right")
    table.add_column("Verdict")

    for e in entries:
        # Color deltas: green = improvement, red = degradation
        tpot_style = "green" if e.tpot_delta_pct < 0 else "red" if e.tpot_delta_pct > 0 else ""
        ttft_style = "green" if e.ttft_delta_pct < 0 else "red" if e.ttft_delta_pct > 0 else ""
        toks_style = "green" if e.tokens_per_s_delta_pct > 0 else "red" if e.tokens_per_s_delta_pct < 0 else ""

        table.add_row(
            e.feature_name,
            e.baseline_label,
            e.variant_label,
            f"[{tpot_style}]{e.tpot_delta_pct:+.1f}%[/{tpot_style}]",
            f"[{ttft_style}]{e.ttft_delta_pct:+.1f}%[/{ttft_style}]",
            f"[{toks_style}]{e.tokens_per_s_delta_pct:+.1f}%[/{toks_style}]",
            e.verdict,
        )
    con.print(table)


def comparison_to_markdown(rows: list[ComparisonRow]) -> str:
    """Generate a Markdown comparison table."""
    lines = [
        "| Label | Mode | GPU Layers | KV | Spec | TTFT (ms) | TPOT (ms) | tok/s | VRAM (GB) |",
        "|-------|------|-----------|-----|------|-----------|-----------|-------|-----------|",
    ]
    for r in rows:
        spec = r.spec_type
        if r.draft_length > 0:
            spec += f" k={r.draft_length}"
        lines.append(
            f"| {r.label} | {r.execution_mode} | {r.gpu_layers} | {r.kv_type} | "
            f"{spec} | {r.ttft_ms:.0f} | {r.tpot_ms:.1f} | "
            f"{r.tokens_per_s:.1f} | {r.peak_vram_gb:.2f} |"
        )
    return "\n".join(lines)


def ablation_to_markdown(entries: list[AblationEntry]) -> str:
    """Generate a Markdown ablation table."""
    lines = [
        "| Feature | Baseline | Variant | TPOT Δ | TTFT Δ | tok/s Δ | Verdict |",
        "|---------|----------|---------|--------|--------|---------|---------|",
    ]
    for e in entries:
        lines.append(
            f"| {e.feature_name} | {e.baseline_label} | {e.variant_label} | "
            f"{e.tpot_delta_pct:+.1f}% | {e.ttft_delta_pct:+.1f}% | "
            f"{e.tokens_per_s_delta_pct:+.1f}% | {e.verdict} |"
        )
    return "\n".join(lines)


def generate_summary(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Generate a summary dict from a list of benchmark results."""
    if not results:
        return {"count": 0}

    best_tpot = min(r.metrics.tpot_ms for r in results)
    best_ttft = min(r.metrics.ttft_ms for r in results)
    best_tok_s = max(r.metrics.output_tokens_per_s for r in results)

    best_result = min(results, key=lambda r: r.metrics.tpot_ms)

    return {
        "count": len(results),
        "best_tpot_ms": best_tpot,
        "best_ttft_ms": best_ttft,
        "best_tokens_per_s": best_tok_s,
        "best_plan_id": best_result.plan.plan_id,
        "best_execution_mode": best_result.plan.execution_mode,
        "best_gpu_layers": best_result.plan.gpu_layers,
        "best_kv_type": best_result.plan.kv_type_k,
        "hardware_fingerprint": results[0].hardware_fingerprint,
        "model_fingerprint": results[0].model_fingerprint,
    }


def get_plot_data(results: list[BenchmarkResult]) -> dict[str, list]:
    """Extract data suitable for matplotlib bar charts.

    Returns dict with lists of: labels, tpot_ms, ttft_ms, tokens_per_s,
    peak_vram_gb — all aligned by index.
    """
    return {
        "labels": [r.plan.plan_id or r.run_id for r in results],
        "tpot_ms": [r.metrics.tpot_ms for r in results],
        "ttft_ms": [r.metrics.ttft_ms for r in results],
        "tokens_per_s": [r.metrics.output_tokens_per_s for r in results],
        "peak_vram_gb": [r.metrics.peak_vram_bytes / 1e9 for r in results],
        "gpu_layers": [r.plan.gpu_layers for r in results],
    }

