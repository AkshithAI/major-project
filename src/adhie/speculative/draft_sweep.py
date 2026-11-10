"""Draft Length Sweep — benchmark speculative decoding across k values.

Runs a systematic sweep of draft lengths (k ∈ {2, 4, 6, 8}) for each
speculative decoding variant, collecting per-k metrics to identify
the optimal draft length for a given hardware/model/workload combination.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from adhie.benchmark.runner import BenchmarkRunner
from adhie.runtime.base import GenerateRequest, RuntimeAdapter
from adhie.speculative.controller import SpecConfig, SpeculativeController
from adhie.types import (
    BenchmarkResult,
    HardwareProfile,
    InferenceMetrics,
    ModelProfile,
    RunPlan,
    SpecMetrics,
    WorkloadConfig,
)

console = Console()


@dataclass
class SweepResult:
    """Result of one k-value within a sweep."""

    variant_id: str = ""
    draft_length: int = 0
    metrics: InferenceMetrics = field(default_factory=InferenceMetrics)
    spec_metrics: SpecMetrics | None = None
    benchmark_result: BenchmarkResult | None = None
    error: str | None = None


@dataclass
class VariantSweepResult:
    """Complete sweep results for one variant across all k values."""

    variant_id: str = ""
    variant_name: str = ""
    sweep_results: list[SweepResult] = field(default_factory=list)
    best_k: int = 0
    best_tokens_per_s: float = 0.0

    @property
    def is_beneficial(self) -> bool:
        """Whether any k value achieved positive speedup over baseline."""
        return self.best_tokens_per_s > 0


class DraftLengthSweep:
    """Sweep draft lengths for speculative decoding experiments.

    Usage:
        sweep = DraftLengthSweep(adapter, controller, runner)
        results = sweep.run_variant_sweep(
            variant_configs,   # list of SpecConfigs with different k values
            base_plan,
            workload,
        )
    """

    def __init__(
        self,
        adapter: RuntimeAdapter,
        controller: SpeculativeController,
        runner: BenchmarkRunner | None = None,
        prompt: str = "Explain the concept of speculative decoding in LLMs.",
        max_tokens: int = 128,
    ) -> None:
        self.adapter = adapter
        self.controller = controller
        self.runner = runner
        self.prompt = prompt
        self.max_tokens = max_tokens

    def run_variant_sweep(
        self,
        configs: list[SpecConfig],
        base_plan: RunPlan,
        hardware_profile: HardwareProfile | None = None,
        model_profile: ModelProfile | None = None,
    ) -> VariantSweepResult:
        """Sweep all k values for a single variant (e.g., all B-k2, B-k4, B-k6, B-k8).

        Args:
            configs: SpecConfigs sharing the same variant letter but different k values.
            base_plan: Template plan to overlay speculation settings onto.

        Returns:
            VariantSweepResult with per-k results and the best k.
        """
        if not configs:
            return VariantSweepResult()

        # Extract variant letter (e.g., "B" from "B-k4")
        variant_letter = configs[0].variant_id.split("-")[0]
        variant_name = configs[0].variant_name.rsplit("(", 1)[0].strip()

        result = VariantSweepResult(
            variant_id=variant_letter,
            variant_name=variant_name,
        )

        console.print(f"\n[bold cyan]Sweeping variant {variant_letter}:[/bold cyan] {variant_name}")

        for config in configs:
            plan = self.controller.config_to_plan(config, base_plan)
            k = config.draft_length

            console.print(f"  [dim]Testing k={k}...[/dim]")

            sweep_item = SweepResult(
                variant_id=config.variant_id,
                draft_length=k,
            )

            try:
                if self.runner:
                    # Full benchmark with warmup + timed runs
                    bench_result = self.runner.run_single(
                        plan=plan,
                        prompt=self.prompt,
                        max_tokens=self.max_tokens,
                        hardware_profile=hardware_profile,
                        model_profile=model_profile,
                    )
                    sweep_item.metrics = bench_result.metrics
                    sweep_item.benchmark_result = bench_result
                else:
                    # Quick single-run measurement
                    self.adapter.load(plan)
                    request = GenerateRequest(
                        prompt=self.prompt,
                        max_tokens=self.max_tokens,
                        temperature=0.0,
                        seed=42,
                        stream=True,
                    )
                    response = self.adapter.generate(request)
                    if response.metrics:
                        sweep_item.metrics = response.metrics
                    self.adapter.unload()

                console.print(
                    f"    k={k}: "
                    f"TTFT={sweep_item.metrics.ttft_ms:.0f}ms | "
                    f"TPOT={sweep_item.metrics.tpot_ms:.1f}ms | "
                    f"tok/s={sweep_item.metrics.output_tokens_per_s:.1f}"
                )

            except Exception as e:
                sweep_item.error = str(e)
                console.print(f"    k={k}: [red]ERROR: {e}[/red]")
                try:
                    self.adapter.unload()
                except Exception:
                    pass

            result.sweep_results.append(sweep_item)

        # Find best k by tokens/s
        valid = [s for s in result.sweep_results if s.error is None]
        if valid:
            best = max(valid, key=lambda s: s.metrics.output_tokens_per_s)
            result.best_k = best.draft_length
            result.best_tokens_per_s = best.metrics.output_tokens_per_s

        return result

    def run_full_matrix(
        self,
        experiment_matrix: list[SpecConfig],
        base_plan: RunPlan,
        hardware_profile: HardwareProfile | None = None,
        model_profile: ModelProfile | None = None,
    ) -> list[VariantSweepResult]:
        """Run sweeps for all variants in the experiment matrix.

        Groups configs by variant letter (A, B, C, D) and sweeps each.
        """
        # Group by variant letter
        groups: dict[str, list[SpecConfig]] = {}
        for config in experiment_matrix:
            letter = config.variant_id.split("-")[0]
            groups.setdefault(letter, []).append(config)

        results: list[VariantSweepResult] = []

        for letter in sorted(groups.keys()):
            configs = groups[letter]
            if letter == "A":
                # Baseline has no k sweep — run once
                plan = self.controller.config_to_plan(configs[0], base_plan)
                console.print(f"\n[bold cyan]Variant A:[/bold cyan] Baseline (no speculation)")

                baseline_result = SweepResult(
                    variant_id="A",
                    draft_length=0,
                )
                try:
                    if self.runner:
                        bench = self.runner.run_single(
                            plan=plan,
                            prompt=self.prompt,
                            max_tokens=self.max_tokens,
                            hardware_profile=hardware_profile,
                            model_profile=model_profile,
                        )
                        baseline_result.metrics = bench.metrics
                        baseline_result.benchmark_result = bench
                    else:
                        self.adapter.load(plan)
                        response = self.adapter.generate(
                            GenerateRequest(
                                prompt=self.prompt,
                                max_tokens=self.max_tokens,
                                temperature=0.0,
                                seed=42,
                            )
                        )
                        if response.metrics:
                            baseline_result.metrics = response.metrics
                        self.adapter.unload()

                    console.print(
                        f"  Baseline: "
                        f"TTFT={baseline_result.metrics.ttft_ms:.0f}ms | "
                        f"tok/s={baseline_result.metrics.output_tokens_per_s:.1f}"
                    )
                except Exception as e:
                    baseline_result.error = str(e)
                    console.print(f"  Baseline: [red]ERROR: {e}[/red]")

                results.append(VariantSweepResult(
                    variant_id="A",
                    variant_name="No speculation (baseline)",
                    sweep_results=[baseline_result],
                    best_k=0,
                    best_tokens_per_s=baseline_result.metrics.output_tokens_per_s,
                ))
            else:
                variant_result = self.run_variant_sweep(
                    configs, base_plan, hardware_profile, model_profile,
                )
                results.append(variant_result)

        return results

    def print_sweep_summary(self, results: list[VariantSweepResult]) -> None:
        """Print a summary table comparing all variants and k values."""
        table = Table(
            title="Speculative Decoding Sweep Results",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Variant", style="bold")
        table.add_column("k")
        table.add_column("TTFT", justify="right")
        table.add_column("TPOT", justify="right")
        table.add_column("Tokens/s", justify="right")
        table.add_column("vs Baseline", justify="right")
        table.add_column("Status")

        # Find baseline tok/s
        baseline_tok_s = 0.0
        for r in results:
            if r.variant_id == "A" and r.sweep_results:
                baseline_tok_s = r.sweep_results[0].metrics.output_tokens_per_s

        for variant in results:
            first_row = True
            for sr in variant.sweep_results:
                if sr.error:
                    table.add_row(
                        variant.variant_name if first_row else "",
                        str(sr.draft_length) if sr.draft_length > 0 else "—",
                        "—", "—", "—", "—",
                        f"[red]Error[/red]",
                    )
                else:
                    m = sr.metrics
                    speedup = ""
                    if baseline_tok_s > 0 and m.output_tokens_per_s > 0:
                        ratio = m.output_tokens_per_s / baseline_tok_s
                        color = "green" if ratio > 1.05 else "red" if ratio < 0.95 else "yellow"
                        speedup = f"[{color}]{ratio:.2f}x[/{color}]"

                    is_best = (
                        sr.draft_length == variant.best_k
                        and variant.variant_id != "A"
                    )
                    status = "[bold green]★ best[/bold green]" if is_best else ""

                    table.add_row(
                        variant.variant_name if first_row else "",
                        str(sr.draft_length) if sr.draft_length > 0 else "—",
                        f"{m.ttft_ms:.0f} ms",
                        f"{m.tpot_ms:.1f} ms",
                        f"{m.output_tokens_per_s:.1f}",
                        speedup,
                        status,
                    )
                first_row = False

            if variant != results[-1]:
                table.add_section()

        console.print()
        console.print(table)

        # Print recommendation
        if baseline_tok_s > 0:
            best_variant = max(
                results,
                key=lambda r: r.best_tokens_per_s if r.sweep_results else 0,
            )
            if best_variant.variant_id != "A":
                speedup = best_variant.best_tokens_per_s / baseline_tok_s
                if speedup > 1.05:
                    console.print(
                        f"\n[bold green]Recommendation:[/bold green] "
                        f"Variant {best_variant.variant_id} with k={best_variant.best_k} "
                        f"achieves {speedup:.2f}x speedup over baseline."
                    )
                else:
                    console.print(
                        f"\n[bold yellow]Recommendation:[/bold yellow] "
                        f"Speculation does not significantly improve performance "
                        f"for this workload. Consider disabling it."
                    )

