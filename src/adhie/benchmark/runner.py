"""Benchmark Runner — execute reproducible benchmarks across model and hardware configurations.

Runs baseline, ADHIE-selected, or parameter sweeps.
Measures TTFT, TPOT, ITL, output tokens/s, peak VRAM/RAM, and writes
structured JSON results with median and P95 metrics.
"""

from __future__ import annotations

import datetime
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from adhie.runtime.base import GenerateRequest, RuntimeAdapter
from adhie.types import (
    BenchmarkResult,
    HardwareProfile,
    InferenceMetrics,
    ModelProfile,
    RunPlan,
    WorkloadConfig,
)

console = Console()


class BenchmarkRunner:
    """Orchestrates benchmark runs using a runtime adapter."""

    def __init__(
        self,
        adapter: RuntimeAdapter,
        warmup_runs: int = 1,
        timed_runs: int = 3,
        cooldown_seconds: float = 2.0,
    ) -> None:
        self.adapter = adapter
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs
        self.cooldown_seconds = cooldown_seconds

    def run_single(
        self,
        plan: RunPlan,
        prompt: str,
        max_tokens: int = 256,
        hardware_profile: HardwareProfile | None = None,
        model_profile: ModelProfile | None = None,
    ) -> BenchmarkResult:
        """Run warmups and timed benchmark iterations for a single plan."""
        console.print(f"[bold cyan]Benchmarking Plan:[/bold cyan] {plan.plan_id or 'default'}")
        console.print(f"  Mode: {plan.execution_mode} | GPU layers: {plan.gpu_layers} | KV: {plan.kv_type_k}")

        # Ensure server is loaded with this plan
        self.adapter.load(plan)

        request = GenerateRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=True,
            temperature=0.0,
            seed=42,
        )

        # 1. Warm-up runs (not recorded in final averages)
        if self.warmup_runs > 0:
            console.print(f"  [dim]Running {self.warmup_runs} warm-up iteration(s)...[/dim]")
            for i in range(self.warmup_runs):
                self.adapter.generate(request)
                time.sleep(1.0)

        # 2. Timed runs
        console.print(f"  [dim]Running {self.timed_runs} timed iteration(s)...[/dim]")
        metrics_list: list[InferenceMetrics] = []

        for i in range(self.timed_runs):
            resp = self.adapter.generate(request)
            if resp.metrics:
                metrics_list.append(resp.metrics)
                console.print(
                    f"    Run {i + 1}/{self.timed_runs}: "
                    f"TTFT={resp.metrics.ttft_ms:.0f}ms | "
                    f"TPOT={resp.metrics.tpot_ms:.1f}ms | "
                    f"tok/s={resp.metrics.output_tokens_per_s:.1f} | "
                    f"tokens={resp.metrics.total_output_tokens}"
                )
            if i < self.timed_runs - 1:
                time.sleep(self.cooldown_seconds)

        # 3. Aggregate metrics
        aggregated = self._aggregate_metrics(metrics_list)

        workload = WorkloadConfig(
            context_length=plan.context_length,
            output_tokens=max_tokens,
            prompt_text=prompt[:100] + "..." if len(prompt) > 100 else prompt,
        )

        result = BenchmarkResult(
            run_id=f"run-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            hardware_fingerprint=hardware_profile.fingerprint() if hardware_profile else "unknown",
            model_fingerprint=model_profile.fingerprint() if model_profile else "unknown",
            plan=plan,
            workload=workload,
            metrics=aggregated,
        )

        return result

    def _aggregate_metrics(self, metrics_list: list[InferenceMetrics]) -> InferenceMetrics:
        """Compute median values across multiple benchmark iterations."""
        if not metrics_list:
            return InferenceMetrics()

        if len(metrics_list) == 1:
            return metrics_list[0]

        # Use medians for robustness against system spikes
        med_ttft = statistics.median(m.ttft_ms for m in metrics_list)
        med_tpot = statistics.median(m.tpot_ms for m in metrics_list)
        med_tok_s = statistics.median(m.output_tokens_per_s for m in metrics_list)
        med_prompt_tok_s = statistics.median(m.prompt_tokens_per_s for m in metrics_list)
        med_out_tokens = int(statistics.median(m.total_output_tokens for m in metrics_list))
        med_prompt_tokens = int(statistics.median(m.total_prompt_tokens for m in metrics_list))
        med_total_time = statistics.median(m.total_time_s for m in metrics_list)

        # Maximum resource utilization across runs
        peak_vram = max(m.peak_vram_bytes for m in metrics_list)
        peak_ram = max(m.peak_ram_bytes for m in metrics_list)

        # Pool all ITLs to compute global P95
        all_itls: list[float] = []
        for m in metrics_list:
            all_itls.extend(m.itl_ms_list)

        itl_p95 = 0.0
        if all_itls:
            sorted_itls = sorted(all_itls)
            p95_idx = min(len(sorted_itls) - 1, int(len(sorted_itls) * 0.95))
            itl_p95 = sorted_itls[p95_idx]

        return InferenceMetrics(
            ttft_ms=med_ttft,
            tpot_ms=med_tpot,
            itl_ms_list=all_itls,
            itl_p95_ms=itl_p95,
            output_tokens_per_s=med_tok_s,
            prompt_tokens_per_s=med_prompt_tok_s,
            total_output_tokens=med_out_tokens,
            total_prompt_tokens=med_prompt_tokens,
            total_time_s=med_total_time,
            peak_vram_bytes=peak_vram,
            peak_ram_bytes=peak_ram,
        )

    def print_comparison(self, results: list[BenchmarkResult]) -> None:
        """Print a formatted comparison table of multiple benchmark runs."""
        table = Table(
            title="Benchmark Comparison",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Plan", style="bold")
        table.add_column("Mode")
        table.add_column("GPU Layers", justify="right")
        table.add_column("KV Type")
        table.add_column("TTFT", justify="right")
        table.add_column("TPOT", justify="right")
        table.add_column("ITL P95", justify="right")
        table.add_column("Tokens/s", justify="right")
        table.add_column("Peak VRAM", justify="right")
        table.add_column("Peak RAM", justify="right")

        for r in results:
            m = r.metrics
            table.add_row(
                r.plan.plan_id or "default",
                r.plan.execution_mode,
                str(r.plan.gpu_layers),
                r.plan.kv_type_k,
                f"{m.ttft_ms:.0f} ms",
                f"{m.tpot_ms:.1f} ms",
                f"{m.itl_p95_ms:.1f} ms",
                f"[bold green]{m.output_tokens_per_s:.1f}[/bold green]",
                f"{m.peak_vram_bytes / 1e9:.2f} GB" if m.peak_vram_bytes > 0 else "—",
                f"{m.peak_ram_bytes / 1e9:.2f} GB" if m.peak_ram_bytes > 0 else "—",
            )

        console.print()
        console.print(table)

