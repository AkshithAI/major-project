#!/usr/bin/env python3
"""Generate performance comparison plots from ADHIE benchmark results.

Usage:
    python scripts/generate_plots.py --input benchmarks/results/demo
    python scripts/generate_plots.py --input benchmarks/results --output benchmarks/plots
    python scripts/generate_plots.py --input benchmarks/results --ablation benchmarks/results/ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Attempt matplotlib import — bail gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print(
        "ERROR: matplotlib is required for plot generation.\n"
        "Install it with: pip install matplotlib",
        file=sys.stderr,
    )
    sys.exit(1)

from adhie.benchmark.report import get_plot_data, load_results

# ---------------------------------------------------------------------------
# Colour palette (Google-ish)
# ---------------------------------------------------------------------------
COLORS = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D01", "#46BDC6"]


def _save(fig: plt.Figure, path: Path, *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_tpot(data: dict, out_dir: Path) -> None:
    """Bar chart — Time Per Output Token (ms)."""
    labels = data["labels"]
    values = data["tpot_ms"]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, values, color=COLORS[: len(labels)], width=0.6)
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("Time Per Output Token")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylim(0, max(values) * 1.25)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, nbins=8))
    fig.tight_layout()
    _save(fig, out_dir / "tpot_comparison.png")


def plot_throughput(data: dict, out_dir: Path) -> None:
    """Bar chart — Throughput (tokens/s)."""
    labels = data["labels"]
    values = data["tokens_per_s"]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, values, color=COLORS[: len(labels)], width=0.6)
    ax.set_ylabel("Tokens / second")
    ax.set_title("Generation Throughput")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    _save(fig, out_dir / "throughput_comparison.png")


def plot_ttft(data: dict, out_dir: Path) -> None:
    """Bar chart — Time To First Token (ms)."""
    labels = data["labels"]
    values = data["ttft_ms"]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, values, color=COLORS[: len(labels)], width=0.6)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time To First Token")
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    _save(fig, out_dir / "ttft_comparison.png")


def plot_vram(data: dict, out_dir: Path) -> None:
    """Bar chart — Peak VRAM usage (GB)."""
    labels = data["labels"]
    values = data["peak_vram_gb"]
    if not labels or all(v == 0 for v in values):
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, values, color=COLORS[: len(labels)], width=0.6)
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("GPU Memory Usage")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1.0)
    fig.tight_layout()
    _save(fig, out_dir / "vram_comparison.png")


def plot_gpu_layers(data: dict, out_dir: Path) -> None:
    """Bar chart — GPU layers per configuration."""
    labels = data["labels"]
    values = data["gpu_layers"]
    if not labels or all(v == 0 for v in values):
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, values, color=COLORS[: len(labels)], width=0.6)
    ax.set_ylabel("GPU Layers")
    ax.set_title("Layer Placement (GPU vs CPU)")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_ylim(0, max(values) * 1.25)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    _save(fig, out_dir / "gpu_layers.png")


def plot_ablation(ablation_dir: Path, out_dir: Path) -> None:
    """Grouped bar chart from ablation report JSON files."""
    # Try to find ablation result pairs
    ablation_files = sorted(ablation_dir.glob("*.json"))
    if not ablation_files:
        print("  ⚠ No ablation JSON files found, skipping ablation plots.")
        return

    results = load_results(ablation_dir)
    if len(results) < 2:
        print("  ⚠ Need at least 2 ablation results for comparison plot.")
        return

    data = get_plot_data(results)
    labels = data["labels"]
    tpot = data["tpot_ms"]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(labels, tpot, color=COLORS[: len(labels)], width=0.5)
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("Ablation Study — Feature Impact on TPOT")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylim(0, max(tpot) * 1.25)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, out_dir / "ablation_tpot.png")


def plot_combined_overview(data: dict, out_dir: Path) -> None:
    """2×2 subplot combining TPOT, throughput, TTFT, and VRAM."""
    labels = data["labels"]
    if not labels:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("ADHIE Benchmark Overview", fontsize=14, fontweight="bold")

    metrics = [
        ("TPOT (ms)", data["tpot_ms"], "%.1f"),
        ("Tokens/s", data["tokens_per_s"], "%.1f"),
        ("TTFT (ms)", data["ttft_ms"], "%.0f"),
        ("Peak VRAM (GB)", data["peak_vram_gb"], "%.2f"),
    ]

    for ax, (ylabel, values, fmt) in zip(axes.flat, metrics):
        bars = ax.bar(labels, values, color=COLORS[: len(labels)], width=0.5)
        ax.set_ylabel(ylabel)
        ax.bar_label(bars, fmt=fmt, fontsize=8, padding=2)
        if values and max(values) > 0:
            ax.set_ylim(0, max(values) * 1.3)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir / "overview.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ADHIE benchmark plots")
    parser.add_argument(
        "--input", "-i", required=True,
        help="Directory containing benchmark result JSON files",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Directory for plots (default: <input>/../plots)",
    )
    parser.add_argument(
        "--ablation", "-a", default=None,
        help="Directory with ablation JSON results",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_dir.parent / "plots"

    print(f"Loading results from: {input_dir}")
    results = load_results(input_dir)

    if not results:
        print("ERROR: No benchmark result JSON files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(results)} result(s). Generating plots...\n")

    data = get_plot_data(results)

    plot_tpot(data, output_dir)
    plot_throughput(data, output_dir)
    plot_ttft(data, output_dir)
    plot_vram(data, output_dir)
    plot_gpu_layers(data, output_dir)
    plot_combined_overview(data, output_dir)

    # Ablation plots
    ablation_dir = Path(args.ablation) if args.ablation else input_dir / "ablation"
    if ablation_dir.exists():
        print("\nGenerating ablation plots...")
        plot_ablation(ablation_dir, output_dir)

    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()

