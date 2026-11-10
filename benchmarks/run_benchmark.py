#!/usr/bin/env python3
"""ADHIE Full Benchmark Matrix Runner.

Sweeps across the configuration space defined in the spec and saves
JSON results for each run.  Use `adhie compare` or `scripts/generate_plots.py`
to analyse and visualise the output.

Usage (single):
    python benchmarks/run_benchmark.py MODEL_REF

Usage (full matrix):
    python benchmarks/run_benchmark.py MODEL_REF --matrix

Usage (custom):
    python benchmarks/run_benchmark.py MODEL_REF \
        --context 2048 8192 16384 \
        --tokens 128 256 \
        --gpu-pct 0 50 100 \
        --kv-types f16 q8_0 \
        --runs 3 \
        --output benchmarks/results/sweep
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.progress import track

from adhie.benchmark.runner import BenchmarkRunner
from adhie.inspector.model_inspector import ModelInspector
from adhie.inspector.model_resolver import resolve_model
from adhie.planner.plan_generator import PlanGenerator
from adhie.estimator.memory_estimator import MemoryEstimator
from adhie.profiler.hardware_profiler import HardwareProfiler
from adhie.runtime.llamacpp_adapter import LlamaCppAdapter, find_llama_server
from adhie.types import RunPlan, WorkloadConfig

console = Console()


# ---------------------------------------------------------------------------
# Matrix definitions
# ---------------------------------------------------------------------------

DEFAULT_CONTEXTS = [4096]
DEFAULT_TOKENS = [128]
DEFAULT_GPU_PCT = [0, 100]  # % of model layers on GPU
DEFAULT_KV_TYPES = ["f16"]
DEFAULT_KV_OFFLOAD = [False]
DEFAULT_SPEC_MODES = ["none"]  # "none", "fixed-4", "adaptive"

FULL_CONTEXTS = [2048, 8192, 16384]
FULL_TOKENS = [128, 256]
FULL_GPU_PCT = [0, 50, 100]    # plus "adhie" — handled specially
FULL_KV_TYPES = ["f16", "q8_0"]
FULL_KV_OFFLOAD = [True, False]
FULL_SPEC_MODES = ["none", "fixed-4", "adaptive"]


def _pct_to_layers(pct: int, total: int) -> int:
    """Convert a percentage to a layer count."""
    return max(0, min(total, round(total * pct / 100)))


def _build_matrix(
    model_layers: int,
    *,
    contexts: list[int],
    tokens: list[int],
    gpu_pcts: list[int],
    kv_types: list[str],
    kv_offloads: list[bool],
    spec_modes: list[str],
) -> list[dict]:
    """Generate all configuration combinations."""
    configs = []
    for ctx, tok, pct, kvt, kvo, spec in itertools.product(
        contexts, tokens, gpu_pcts, kv_types, kv_offloads, spec_modes,
    ):
        gpu_layers = _pct_to_layers(pct, model_layers)
        label = (
            f"ctx{ctx}_tok{tok}_gpu{pct}pct_kv{kvt}"
            f"{'_offload' if kvo else ''}"
            f"{'_' + spec if spec != 'none' else ''}"
        )
        configs.append({
            "label": label,
            "context": ctx,
            "tokens": tok,
            "gpu_layers": gpu_layers,
            "kv_type": kvt,
            "kv_offload": kvo,
            "spec_mode": spec,
        })
    return configs


def _config_to_plan(cfg: dict, base_plan: RunPlan) -> RunPlan:
    """Override a base plan with the matrix configuration."""
    plan = replace(
        base_plan,
        plan_id=f"matrix-{cfg['label']}",
        gpu_layers=cfg["gpu_layers"],
        kv_type_k=cfg["kv_type"],
        kv_type_v=cfg["kv_type"],
        kv_offload=cfg["kv_offload"],
    )

    if cfg["spec_mode"] == "none":
        plan = replace(
            plan,
            spec_type="none",
            draft_model_path=None,
            draft_length=0,
        )
    elif cfg["spec_mode"] == "fixed-4":
        plan = replace(plan, draft_length=4)
    elif cfg["spec_mode"] == "adaptive":
        plan = replace(plan, draft_length=6)  # adaptive will tune

    return plan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADHIE Full Benchmark Matrix Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", help="GGUF path or HuggingFace repo identifier")
    parser.add_argument("--matrix", action="store_true",
                        help="Run the full benchmark matrix from the spec")
    parser.add_argument("--context", nargs="+", type=int, default=None,
                        help="Context lengths to test")
    parser.add_argument("--tokens", nargs="+", type=int, default=None,
                        help="Output token counts to test")
    parser.add_argument("--gpu-pct", nargs="+", type=int, default=None,
                        help="GPU layer percentages (0, 50, 100)")
    parser.add_argument("--kv-types", nargs="+", default=None,
                        help="KV cache types (f16, q8_0)")
    parser.add_argument("--kv-offload", nargs="+", type=int, default=None,
                        help="KV offload flags (0=no, 1=yes)")
    parser.add_argument("--spec-modes", nargs="+", default=None,
                        help="Speculation modes (none, fixed-4, adaptive)")
    parser.add_argument("--draft", "-d", default=None,
                        help="Draft model for speculative decoding")
    parser.add_argument("--runs", "-r", type=int, default=3,
                        help="Timed runs per config (default: 3)")
    parser.add_argument("--warmup", "-w", type=int, default=1,
                        help="Warm-up runs")
    parser.add_argument("--output", "-o", default="benchmarks/results",
                        help="Output directory for JSON results")
    parser.add_argument("--server", default=None,
                        help="Path to llama-server binary")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running benchmarks")
    args = parser.parse_args()

    # ── Select matrix dimensions ─────────────────────────────────
    if args.matrix:
        contexts = args.context or FULL_CONTEXTS
        tokens = args.tokens or FULL_TOKENS
        gpu_pcts = args.gpu_pct or FULL_GPU_PCT
        kv_types = args.kv_types or FULL_KV_TYPES
        kv_offloads = [bool(x) for x in args.kv_offload] if args.kv_offload else FULL_KV_OFFLOAD
        spec_modes = args.spec_modes or FULL_SPEC_MODES
    else:
        contexts = args.context or DEFAULT_CONTEXTS
        tokens = args.tokens or DEFAULT_TOKENS
        gpu_pcts = args.gpu_pct or DEFAULT_GPU_PCT
        kv_types = args.kv_types or DEFAULT_KV_TYPES
        kv_offloads = [bool(x) for x in args.kv_offload] if args.kv_offload else DEFAULT_KV_OFFLOAD
        spec_modes = args.spec_modes or DEFAULT_SPEC_MODES

    # Filter spec modes if no draft model supplied
    if not args.draft:
        spec_modes = ["none"]

    # ── Resolve model & inspect ──────────────────────────────────
    console.print("[bold blue]ADHIE Benchmark Matrix[/bold blue]\n")
    model_path = resolve_model(args.model)
    model = ModelInspector().inspect(model_path)
    console.print(f"Model: {model.model_id} ({model.parameter_count / 1e9:.1f}B, "
                  f"{model.layer_count}L, {model.weight_dtype})")

    if args.draft:
        draft_path = resolve_model(args.draft)
        model.draft_model_path = draft_path
        console.print(f"Draft: {draft_path}")

    # ── Profile hardware ─────────────────────────────────────────
    hw = HardwareProfiler(verbose=False).profile()
    console.print(f"GPU: {hw.gpu_name or 'None'} | "
                  f"VRAM: {hw.vram_free_bytes / 1e9:.1f} GB | "
                  f"RAM: {hw.ram_available_bytes / 1e9:.1f} GB\n")

    # ── Build matrix ─────────────────────────────────────────────
    configs = _build_matrix(
        model.layer_count,
        contexts=contexts,
        tokens=tokens,
        gpu_pcts=gpu_pcts,
        kv_types=kv_types,
        kv_offloads=kv_offloads,
        spec_modes=spec_modes,
    )

    # Also add the ADHIE-selected plan for each context/token combo
    estimator = MemoryEstimator()
    generator = PlanGenerator(estimator=estimator)
    for ctx, tok in itertools.product(contexts, tokens):
        workload = WorkloadConfig(context_length=ctx, output_tokens=tok)
        candidates = generator.generate_candidates(hw, model, workload)
        if candidates:
            best = candidates[0]
            configs.append({
                "label": f"ctx{ctx}_tok{tok}_adhie_selected",
                "context": ctx,
                "tokens": tok,
                "gpu_layers": best.gpu_layers,
                "kv_type": best.kv_type_k,
                "kv_offload": best.kv_offload,
                "spec_mode": "none",
                "_plan_override": best,
            })

    console.print(f"[bold]Total configurations: {len(configs)}[/bold]\n")

    if args.dry_run:
        for i, cfg in enumerate(configs, 1):
            console.print(f"  {i:3d}. {cfg['label']}")
        console.print(f"\n[yellow]Dry run — no benchmarks executed.[/yellow]")
        return

    # ── Generate base plan ───────────────────────────────────────
    workload = WorkloadConfig(context_length=contexts[0], output_tokens=tokens[0])
    candidates = generator.generate_candidates(hw, model, workload)
    if not candidates:
        console.print("[bold red]No feasible plans found![/bold red]")
        sys.exit(1)
    base_plan = candidates[0]

    # ── Run benchmarks ───────────────────────────────────────────
    server_path = args.server or find_llama_server()
    adapter = LlamaCppAdapter(server_path=server_path)
    runner = BenchmarkRunner(adapter, warmup_runs=args.warmup, timed_runs=args.runs)

    prompt = (
        "Write a comprehensive explanation of how heterogeneous CPU and GPU "
        "computing works for large language model inference, covering memory "
        "management, layer placement, and speculative decoding."
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    try:
        for cfg in track(configs, description="Benchmarking..."):
            if "_plan_override" in cfg:
                plan = cfg["_plan_override"]
            else:
                plan = _config_to_plan(cfg, base_plan)

            # Validate feasibility (skip infeasible)
            est = estimator.estimate(
                hw, model, cfg["context"], cfg["gpu_layers"],
                cfg["kv_type"], cfg["kv_offload"],
            )
            if not est.feasible:
                console.print(f"  [yellow]SKIP (infeasible): {cfg['label']}[/yellow]")
                continue

            try:
                result = runner.run_single(
                    plan=plan,
                    prompt=prompt,
                    max_tokens=cfg["tokens"],
                    hardware_profile=hw,
                    model_profile=model,
                )
                results.append(result)

                out_file = out_dir / f"{result.run_id}.json"
                result.to_json(out_file)

            except Exception as exc:
                console.print(f"  [red]ERROR: {cfg['label']}: {exc}[/red]")
            finally:
                adapter.unload()

        # Print summary
        if results:
            runner.print_comparison(results)
            console.print(f"\n[green]Saved {len(results)} results to {out_dir}/[/green]")
        else:
            console.print("[yellow]No benchmark results produced.[/yellow]")

    finally:
        adapter.unload()


if __name__ == "__main__":
    main()
