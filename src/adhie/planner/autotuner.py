"""Autotuner — benchmark top candidates and select the optimal RunPlan.

Phase A of ADHIE's integrated pipeline:
  1. PlanGenerator produces candidate RunPlans
  2. MemoryEstimator has already rejected infeasible ones
  3. Autotuner benchmarks top-N candidates with short warm-up runs
  4. Scores each using a weighted objective function
  5. Selects and caches the best plan

The objective function (from spec §11.3):
    J(c) = w1·P95_TPOT + w2·P95_TTFT + w3·MemPenalty + w4·QualPenalty

Lower is better.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from adhie.runtime.base import GenerateRequest, RuntimeAdapter
from adhie.telemetry.collector import TelemetryCollector
from adhie.types import InferenceMetrics, RunPlan, WorkloadConfig

logger = logging.getLogger(__name__)


@dataclass
class TuneResult:
    """Result of evaluating one candidate plan."""

    plan: RunPlan
    metrics: InferenceMetrics
    score: float
    error: str | None = None


@dataclass
class AutotunerConfig:
    """Configuration for the Autotuner."""

    max_candidates: int = 5
    warmup_tokens: int = 64
    benchmark_tokens: int = 128
    benchmark_rounds: int = 3

    # Objective function weights
    w_tpot: float = 0.5
    w_ttft: float = 0.2
    w_mem_penalty: float = 0.2
    w_quality_penalty: float = 0.1


class Autotuner:
    """Benchmark candidates and select the best RunPlan.

    Manages the load → benchmark → score → unload cycle for each
    candidate plan, selecting the one with the lowest objective score.

    Args:
        adapter: Runtime backend to load models on.
        config: Autotuner configuration.
        console: Rich console for status output.
    """

    def __init__(
        self,
        adapter: RuntimeAdapter,
        config: AutotunerConfig | None = None,
        console: Console | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or AutotunerConfig()
        self.console = console or Console()

    def tune(
        self,
        candidates: list[RunPlan],
        workload: WorkloadConfig,
    ) -> TuneResult:
        """Benchmark top-N candidates and return the best.

        Args:
            candidates: Feasible RunPlans, sorted by heuristic score (best first).
            workload: The target workload configuration.

        Returns:
            TuneResult containing the best plan, its metrics, and score.

        Raises:
            RuntimeError: If no candidate could be loaded and benchmarked.
        """
        top_n = candidates[: self.config.max_candidates]
        results: list[TuneResult] = []

        self.console.print(
            f"\n[bold cyan]ADHIE Autotuner[/bold cyan] — benchmarking "
            f"{len(top_n)} candidate(s)...\n"
        )

        for i, plan in enumerate(top_n, 1):
            self.console.print(
                f"  [{i}/{len(top_n)}] {plan.execution_mode} · "
                f"{plan.gpu_layers} GPU layers · "
                f"KV={plan.kv_type_k}"
            )
            result = self._evaluate_candidate(plan, workload)
            if result.error:
                self.console.print(
                    f"       [red]✗ {result.error}[/red]"
                )
            else:
                self.console.print(
                    f"       [green]✓[/green] TPOT={result.metrics.tpot_ms:.0f}ms  "
                    f"TTFT={result.metrics.ttft_ms:.0f}ms  "
                    f"score={result.score:.2f}"
                )
                results.append(result)

        if not results:
            raise RuntimeError(
                "Autotuner: No candidate could be loaded and benchmarked. "
                "All plans failed with errors."
            )

        # Select the best (lowest score)
        results.sort(key=lambda r: r.score)
        best = results[0]

        self.console.print(
            f"\n  [bold green]Selected:[/bold green] {best.plan.execution_mode} · "
            f"{best.plan.gpu_layers} GPU layers · "
            f"score={best.score:.2f}\n"
        )

        return best

    def _evaluate_candidate(
        self,
        plan: RunPlan,
        workload: WorkloadConfig,
    ) -> TuneResult:
        """Load a plan, run a short benchmark, and compute its score."""
        try:
            self.adapter.load(plan)
        except Exception as e:
            return TuneResult(
                plan=plan,
                metrics=InferenceMetrics(),
                score=float("inf"),
                error=f"Load failed: {e}",
            )

        try:
            # Wait for backend readiness
            if not self.adapter.health_check():
                return TuneResult(
                    plan=plan,
                    metrics=InferenceMetrics(),
                    score=float("inf"),
                    error="Backend health check failed",
                )

            # Run benchmark rounds
            all_metrics: list[InferenceMetrics] = []

            # Warmup run (discarded)
            self._run_single(
                "Warmup prompt for ADHIE autotuner.",
                self.config.warmup_tokens,
            )

            # Timed runs
            for _ in range(self.config.benchmark_rounds):
                metrics = self._run_single(
                    _make_benchmark_prompt(workload),
                    self.config.benchmark_tokens,
                )
                if metrics:
                    all_metrics.append(metrics)

            if not all_metrics:
                return TuneResult(
                    plan=plan,
                    metrics=InferenceMetrics(),
                    score=float("inf"),
                    error="No valid metrics from benchmark runs",
                )

            # Aggregate metrics
            aggregated = _aggregate_metrics(all_metrics)
            score = self._compute_score(plan, aggregated)

            return TuneResult(
                plan=plan,
                metrics=aggregated,
                score=score,
            )

        except Exception as e:
            logger.warning("Autotuner error for plan %s: %s", plan.plan_id, e)
            return TuneResult(
                plan=plan,
                metrics=InferenceMetrics(),
                score=float("inf"),
                error=f"Benchmark error: {e}",
            )
        finally:
            try:
                self.adapter.unload()
            except Exception:
                pass

    def _run_single(
        self,
        prompt: str,
        max_tokens: int,
    ) -> InferenceMetrics | None:
        """Run a single generation and return the metrics."""
        request = GenerateRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=42,
        )
        response = self.adapter.generate(request)
        return response.metrics or self.adapter.last_metrics()

    def _compute_score(
        self,
        plan: RunPlan,
        metrics: InferenceMetrics,
    ) -> float:
        """Compute the objective function J(c).

        J(c) = w1·TPOT + w2·TTFT + w3·MemPenalty + w4·QualPenalty

        Memory penalty grows when peak VRAM exceeds 90% of the estimate,
        to penalize plans that risk OOM.
        """
        cfg = self.config

        # Memory penalty: penalise if actual VRAM usage > 90% of estimate
        mem_penalty = 0.0
        if plan.estimated_vram_bytes > 0 and metrics.peak_vram_bytes > 0:
            vram_ratio = metrics.peak_vram_bytes / plan.estimated_vram_bytes
            if vram_ratio > 0.9:
                mem_penalty = (vram_ratio - 0.9) * 1000.0

        # Quality penalty: placeholder (0 for now; can be extended for
        # greedy equivalence or perplexity checks)
        quality_penalty = 0.0

        score = (
            cfg.w_tpot * metrics.tpot_ms
            + cfg.w_ttft * metrics.ttft_ms
            + cfg.w_mem_penalty * mem_penalty
            + cfg.w_quality_penalty * quality_penalty
        )
        return score

    def print_comparison(self, results: list[TuneResult]) -> None:
        """Print a Rich comparison table of all evaluated candidates."""
        table = Table(title="Autotuner Candidates", show_header=True,
                      header_style="bold cyan")
        table.add_column("#", justify="right", width=3)
        table.add_column("Mode")
        table.add_column("GPU Layers", justify="right")
        table.add_column("KV Type")
        table.add_column("TPOT (ms)", justify="right")
        table.add_column("TTFT (ms)", justify="right")
        table.add_column("tok/s", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Status")

        for i, r in enumerate(results, 1):
            if r.error:
                table.add_row(
                    str(i), r.plan.execution_mode,
                    str(r.plan.gpu_layers), r.plan.kv_type_k,
                    "—", "—", "—", "—", f"[red]{r.error}[/red]",
                )
            else:
                table.add_row(
                    str(i), r.plan.execution_mode,
                    str(r.plan.gpu_layers), r.plan.kv_type_k,
                    f"{r.metrics.tpot_ms:.0f}",
                    f"{r.metrics.ttft_ms:.0f}",
                    f"{r.metrics.output_tokens_per_s:.1f}",
                    f"{r.score:.2f}",
                    "[green]✓[/green]",
                )

        self.console.print(table)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_benchmark_prompt(workload: WorkloadConfig) -> str:
    """Create a benchmark prompt matching the workload type."""
    prompts = {
        "chat": "Explain the concept of attention mechanisms in transformer models.",
        "code": "Write a Python function that implements binary search on a sorted list.",
        "summarize": (
            "Summarize the following text: The development of large language "
            "models has progressed rapidly in recent years, with models like "
            "GPT-4, Claude, and Llama 3 demonstrating remarkable capabilities "
            "across a wide range of tasks. These models use transformer "
            "architectures with billions of parameters."
        ),
    }
    return prompts.get(workload.workload_type, prompts["chat"])


def _aggregate_metrics(metrics_list: list[InferenceMetrics]) -> InferenceMetrics:
    """Aggregate multiple inference metrics by taking medians.

    For the autotuner, median is more robust than mean to outliers
    from cold caches or GC pauses.
    """
    if not metrics_list:
        return InferenceMetrics()

    if len(metrics_list) == 1:
        return metrics_list[0]

    def median(values: list[float]) -> float:
        s = sorted(values)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    return InferenceMetrics(
        ttft_ms=median([m.ttft_ms for m in metrics_list]),
        tpot_ms=median([m.tpot_ms for m in metrics_list]),
        itl_p95_ms=median([m.itl_p95_ms for m in metrics_list]),
        output_tokens_per_s=median([m.output_tokens_per_s for m in metrics_list]),
        prompt_tokens_per_s=median([m.prompt_tokens_per_s for m in metrics_list]),
        total_output_tokens=int(median([m.total_output_tokens for m in metrics_list])),
        total_prompt_tokens=int(median([m.total_prompt_tokens for m in metrics_list])),
        total_time_s=median([m.total_time_s for m in metrics_list]),
        peak_vram_bytes=max(m.peak_vram_bytes for m in metrics_list),
        peak_ram_bytes=max(m.peak_ram_bytes for m in metrics_list),
        cpu_utilization_pct=median([m.cpu_utilization_pct for m in metrics_list]),
    )

