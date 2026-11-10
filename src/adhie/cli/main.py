"""ADHIE Command-Line Interface.

Usage:
    adhie profile                       Profile local hardware
    adhie inspect MODEL                 Inspect model metadata
    adhie estimate MODEL                Estimate memory requirements
    adhie plan MODEL                    Generate execution plan candidates
    adhie run MODEL                     Profile, plan, and run inference
    adhie benchmark MODEL               Run benchmark suite
    adhie experiment MODEL              Run speculative decoding experiments
    adhie compare FILE [FILE ...]       Compare benchmark results
    adhie ablation MODEL                Run ablation studies
    adhie models                        List cached models

MODEL can be:
    /path/to/model.gguf                 Local GGUF file
    Qwen/Qwen2.5-14B-Instruct-GGUF     HuggingFace repo (auto-downloads)
    Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m   HF repo with quant hint
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from adhie import __version__

console = Console()


def _resolve_model(model_ref: str) -> str:
    """Resolve a model reference (local path or HF identifier) to a local GGUF path."""
    from adhie.inspector.model_resolver import resolve_model
    return resolve_model(model_ref, interactive=True)


@click.group()
@click.version_option(version=__version__, prog_name="adhie")
def cli() -> None:
    """ADHIE — Adaptive Heterogeneous Inference Engine for Local LLMs."""
    pass


@cli.command()
@click.option("--output", "-o", default=None, help="Save profile to YAML file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed profiling info")
def profile(output: str | None, verbose: bool) -> None:
    """Profile local hardware: GPU, CPU, RAM, bandwidths."""
    from adhie.profiler.hardware_profiler import HardwareProfiler

    console.print("[bold blue]ADHIE Hardware Profiler[/bold blue]")
    console.print()

    profiler = HardwareProfiler(verbose=verbose)
    hw = profiler.profile()

    profiler.print_report(hw, console)

    if output:
        from pathlib import Path
        hw.to_yaml(Path(output))
        console.print(f"\n[green]Profile saved to {output}[/green]")


@cli.command()
@click.argument("model_ref")
@click.option("--output", "-o", default=None, help="Save profile to YAML file")
def inspect(model_ref: str, output: str | None) -> None:
    """Inspect model metadata from a GGUF file or HuggingFace repo.

    MODEL_REF can be a local path (/path/to/model.gguf),
    a HuggingFace repo ID (Qwen/Qwen2.5-14B-Instruct-GGUF),
    or a repo with quant hint (Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m).
    """
    from adhie.inspector.model_inspector import ModelInspector

    console.print("[bold blue]ADHIE Model Inspector[/bold blue]")
    console.print()

    # Resolve model reference to local path (downloads if HF)
    model_path = _resolve_model(model_ref)
    console.print(f"[dim]Model file: {model_path}[/dim]")
    console.print()

    inspector = ModelInspector()
    model = inspector.inspect(model_path)

    inspector.print_report(model, console)

    if output:
        from pathlib import Path
        model.to_yaml(Path(output))
        console.print(f"\n[green]Model profile saved to {output}[/green]")


@cli.command()
@click.argument("model_ref")
@click.option("--context", "-c", default=8192, help="Context length in tokens")
@click.option("--draft", "-d", default=None, help="Draft model (path or HF repo)")
@click.option("--output", "-o", default=None, help="Save plans to YAML file")
def plan(model_ref: str, context: int, draft: str | None, output: str | None) -> None:
    """Generate and rank execution plan candidates.

    MODEL_REF can be a local path or HuggingFace repo ID.
    """
    from adhie.profiler.hardware_profiler import HardwareProfiler
    from adhie.inspector.model_inspector import ModelInspector
    from adhie.planner.plan_generator import PlanGenerator
    from adhie.estimator.memory_estimator import MemoryEstimator
    from adhie.types import WorkloadConfig

    console.print("[bold blue]ADHIE Plan Generator[/bold blue]")
    console.print()

    # Resolve models
    model_path = _resolve_model(model_ref)
    draft_path = _resolve_model(draft) if draft else None

    console.print("[dim]Profiling hardware...[/dim]")
    hw = HardwareProfiler(verbose=False).profile()
    console.print(f"  GPU: {hw.gpu_name or 'None'} | "
                  f"VRAM: {hw.vram_free_bytes / 1e9:.1f} GB | "
                  f"RAM: {hw.ram_available_bytes / 1e9:.1f} GB")
    console.print()

    console.print("[dim]Inspecting model...[/dim]")
    model = ModelInspector().inspect(model_path)
    if draft_path:
        model.draft_model_path = draft_path
    console.print(f"  {model.model_id} | {model.parameter_count / 1e9:.1f}B | "
                  f"{model.layer_count}L | {model.weight_dtype}")
    console.print()

    # Show memory estimate for max GPU layers
    estimator = MemoryEstimator()
    max_layers = estimator.find_max_gpu_layers(hw, model, context)
    console.print(f"[dim]Max GPU layers: {max_layers}/{model.layer_count}[/dim]")
    console.print()

    workload = WorkloadConfig(context_length=context)
    generator = PlanGenerator(estimator=estimator)
    candidates = generator.generate_candidates(hw, model, workload)

    generator.print_candidates(candidates, console)

    if output:
        from pathlib import Path
        import yaml
        from dataclasses import asdict
        data = [asdict(c) for c in candidates]
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        console.print(f"\n[green]Plans saved to {output}[/green]")


@cli.command()
@click.argument("model_ref")
@click.option("--context", "-c", default=8192, help="Context length in tokens")
@click.option("--gpu-layers", "-g", default=None, type=int, help="GPU layers (auto if omitted)")
@click.option("--kv-type", default="f16", help="KV cache type (f16, q8_0)")
@click.option("--kv-offload", is_flag=True, help="Offload KV cache to RAM")
def estimate(model_ref: str, context: int, gpu_layers: int | None,
             kv_type: str, kv_offload: bool) -> None:
    """Estimate memory requirements for a model configuration.

    MODEL_REF can be a local path or HuggingFace repo ID.
    """
    from adhie.profiler.hardware_profiler import HardwareProfiler
    from adhie.inspector.model_inspector import ModelInspector
    from adhie.estimator.memory_estimator import MemoryEstimator

    console.print("[bold blue]ADHIE Memory Estimator[/bold blue]")
    console.print()

    model_path = _resolve_model(model_ref)
    hw = HardwareProfiler(verbose=False).profile()
    model = ModelInspector().inspect(model_path)

    estimator = MemoryEstimator()

    if gpu_layers is None:
        gpu_layers = estimator.find_max_gpu_layers(
            hw, model, context, kv_dtype=kv_type, kv_offload=kv_offload,
        )
        console.print(f"[dim]Auto-detected max GPU layers: {gpu_layers}/{model.layer_count}[/dim]")
        console.print()

    est = estimator.estimate(hw, model, context, gpu_layers, kv_type, kv_offload)
    estimator.print_report(est, model, gpu_layers, context, console)


@cli.command()
@click.argument("model_ref")
@click.option("--context", "-c", default=8192, help="Context length")
@click.option("--draft", "-d", default=None, help="Draft model (path or HF repo)")
@click.option("--backend-path", default=None, help="Path to llama-server binary")
@click.option("--verbose", "-v", is_flag=True)
def run(model_ref: str, context: int, draft: str | None,
        backend_path: str | None, verbose: bool) -> None:
    """Profile hardware, plan, and run inference with the best configuration.

    MODEL_REF can be a local path or HuggingFace repo ID.
    """
    from adhie.profiler.hardware_profiler import HardwareProfiler
    from adhie.inspector.model_inspector import ModelInspector
    from adhie.planner.plan_generator import PlanGenerator
    from adhie.runtime.llamacpp_adapter import LlamaCppAdapter
    from adhie.types import WorkloadConfig

    console.print("[bold blue]ADHIE — Adaptive Heterogeneous Inference Engine[/bold blue]")
    console.print()

    # Step 1: Resolve models
    console.print("[bold]Step 1:[/bold] Resolving model...")
    model_path = _resolve_model(model_ref)
    draft_path = _resolve_model(draft) if draft else None
    console.print(f"  Target: {model_path}")
    if draft_path:
        console.print(f"  Draft:  {draft_path}")
    console.print()

    # Step 2: Profile
    console.print("[bold]Step 2:[/bold] Profiling hardware...")
    hw = HardwareProfiler(verbose=verbose).profile()
    console.print(f"  GPU: {hw.gpu_name or 'None'} | "
                  f"VRAM: {hw.vram_free_bytes / 1e9:.1f} GB free | "
                  f"RAM: {hw.ram_available_bytes / 1e9:.1f} GB available")
    console.print()

    # Step 3: Inspect model
    console.print("[bold]Step 3:[/bold] Inspecting model...")
    model = ModelInspector().inspect(model_path)
    if draft_path:
        model.draft_model_path = draft_path
    console.print(f"  Model: {model.model_id} | "
                  f"Params: {model.parameter_count / 1e9:.1f}B | "
                  f"Layers: {model.layer_count} | "
                  f"Quant: {model.weight_dtype}")
    console.print()

    # Step 4: Generate plan
    console.print("[bold]Step 4:[/bold] Generating execution plan...")
    workload = WorkloadConfig(context_length=context)
    candidates = PlanGenerator().generate_candidates(hw, model, workload)

    if not candidates:
        console.print("[bold red]No feasible execution plans found![/bold red]")
        console.print("The model may be too large for available memory.")
        raise SystemExit(1)

    best_plan = candidates[0]
    console.print(f"  Mode: {best_plan.execution_mode} | "
                  f"GPU layers: {best_plan.gpu_layers}/{model.layer_count} | "
                  f"KV: {best_plan.kv_type_k}")
    console.print(f"  Rationale: {best_plan.rationale}")
    console.print()

    # Step 5: Launch
    console.print("[bold]Step 5:[/bold] Launching inference server...")
    from adhie.runtime.llamacpp_adapter import LlamaCppAdapter, find_llama_server
    server_path = backend_path or find_llama_server()
    adapter = LlamaCppAdapter(server_path)

    try:
        adapter.load(best_plan)
        console.print("[green]Server ready. Starting interactive session...[/green]")
        console.print("Type your prompt (Ctrl+D to exit):\n")

        _interactive_loop(adapter, console)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    finally:
        adapter.unload()
        console.print("[dim]Server stopped.[/dim]")


@cli.command()
def models() -> None:
    """List all locally cached models."""
    from adhie.inspector.model_resolver import list_cached_models, DEFAULT_CACHE_DIR

    console.print("[bold blue]ADHIE Model Cache[/bold blue]")
    console.print(f"[dim]Cache directory: {DEFAULT_CACHE_DIR}[/dim]")
    console.print()

    cached = list_cached_models()

    if not cached:
        console.print("[yellow]No cached models found.[/yellow]")
        console.print("Download a model with: adhie inspect Qwen/Qwen2.5-14B-Instruct-GGUF")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Repository")
    table.add_column("File")
    table.add_column("Size", justify="right")
    table.add_column("Path")

    for m in cached:
        table.add_row(m["repo"], m["file"], f"{m['size_gb']} GB", m["path"])

    console.print(table)


@cli.command()
@click.argument("model_ref")
@click.option("--context", "-c", default=8192, help="Context length")
@click.option("--draft", "-d", default=None, help="Draft model (path or HF repo)")
@click.option("--output", "-o", default="benchmarks/results", help="Results directory")
@click.option("--mode", type=click.Choice(["baseline", "auto", "sweep"]), default="auto")
@click.option("--backend-path", default=None)
@click.option("--runs", "-r", default=3, help="Number of timed iterations")
@click.option("--tokens", "-t", default=128, help="Output tokens to generate")
def benchmark(model_ref: str, context: int, draft: str | None,
              output: str, mode: str, backend_path: str | None,
              runs: int, tokens: int) -> None:
    """Run the benchmark suite and save results.

    MODEL_REF can be a local path or HuggingFace repo ID.
    """
    from pathlib import Path
    from adhie.profiler.hardware_profiler import HardwareProfiler
    from adhie.inspector.model_inspector import ModelInspector
    from adhie.planner.plan_generator import PlanGenerator
    from adhie.runtime.llamacpp_adapter import LlamaCppAdapter, find_llama_server
    from adhie.benchmark.runner import BenchmarkRunner
    from adhie.types import WorkloadConfig

    console.print("[bold blue]ADHIE Benchmark Suite[/bold blue]")
    console.print(f"Mode: {mode} | Runs: {runs} | Tokens: {tokens}")
    console.print()

    # Resolve model
    model_path = _resolve_model(model_ref)
    draft_path = _resolve_model(draft) if draft else None

    # Profile & inspect
    hw = HardwareProfiler(verbose=False).profile()
    model = ModelInspector().inspect(model_path)
    if draft_path:
        model.draft_model_path = draft_path

    # Generate plans
    workload = WorkloadConfig(context_length=context, output_tokens=tokens)
    generator = PlanGenerator()
    candidates = generator.generate_candidates(hw, model, workload)

    if not candidates:
        console.print("[bold red]No feasible plans found![/bold red]")
        raise SystemExit(1)

    # Select plans to benchmark based on mode
    if mode == "baseline":
        # CPU-only baseline: find the plan with gpu_layers=0
        cpu_plans = [p for p in candidates if p.gpu_layers == 0]
        plans_to_test = cpu_plans[:1] if cpu_plans else candidates[:1]
    elif mode == "sweep":
        # Test top 3 diverse plans
        plans_to_test = candidates[:3]
    else:
        # auto: test the best plan
        plans_to_test = candidates[:1]

    # Run benchmarks
    server_path = backend_path or find_llama_server()
    adapter = LlamaCppAdapter(server_path)
    runner = BenchmarkRunner(adapter, warmup_runs=1, timed_runs=runs)

    prompt = (
        "Write a comprehensive explanation of how heterogeneous CPU and GPU "
        "computing works for large language model inference, covering memory "
        "management, layer placement, and speculative decoding."
    )

    results = []
    try:
        for plan in plans_to_test:
            result = runner.run_single(
                plan=plan,
                prompt=prompt,
                max_tokens=tokens,
                hardware_profile=hw,
                model_profile=model,
            )
            results.append(result)
            adapter.unload()

        # Print comparison
        runner.print_comparison(results)

        # Save results
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            out_file = out_dir / f"{r.run_id}.json"
            r.to_json(out_file)
            console.print(f"[green]Saved: {out_file}[/green]")

    finally:
        adapter.unload()


@cli.command()
@click.argument("model_ref")
@click.option("--draft", "-d", required=True, help="Draft model (path or HF repo)")
@click.option("--context", "-c", default=8192, help="Context length")
@click.option("--backend-path", default=None, help="Path to llama-server binary")
@click.option("--runs", "-r", default=3, help="Timed runs per configuration")
@click.option("--tokens", "-t", default=128, help="Output tokens per run")
@click.option("--output", "-o", default="benchmarks/results/spec", help="Results directory")
def experiment(model_ref: str, draft: str, context: int,
               backend_path: str | None, runs: int, tokens: int,
               output: str) -> None:
    """Run speculative decoding experiments (variants A-D × k sweep).

    Requires both a target model and a draft model.
    """
    from pathlib import Path
    from adhie.profiler.hardware_profiler import HardwareProfiler
    from adhie.inspector.model_inspector import ModelInspector
    from adhie.estimator.memory_estimator import MemoryEstimator
    from adhie.planner.plan_generator import PlanGenerator
    from adhie.runtime.llamacpp_adapter import LlamaCppAdapter, find_llama_server
    from adhie.benchmark.runner import BenchmarkRunner
    from adhie.speculative.controller import SpeculativeController
    from adhie.speculative.draft_sweep import DraftLengthSweep
    from adhie.types import WorkloadConfig

    console.print("[bold blue]ADHIE Speculative Decoding Experiment[/bold blue]")
    console.print()

    # Resolve models
    model_path = _resolve_model(model_ref)
    draft_path = _resolve_model(draft)

    # Profile & inspect
    console.print("[dim]Profiling hardware...[/dim]")
    hw = HardwareProfiler(verbose=False).profile()
    console.print(f"  GPU: {hw.gpu_name or 'None'} | VRAM: {hw.vram_free_bytes / 1e9:.1f} GB")

    console.print("[dim]Inspecting target model...[/dim]")
    target = ModelInspector().inspect(model_path)
    console.print(f"  Target: {target.model_id} ({target.parameter_count / 1e9:.1f}B)")

    console.print("[dim]Inspecting draft model...[/dim]")
    draft_profile = ModelInspector().inspect(draft_path)
    console.print(f"  Draft: {draft_profile.model_id} ({draft_profile.parameter_count / 1e9:.1f}B)")
    console.print()

    # Check compatibility
    inspector = ModelInspector()
    compat, issues = inspector.check_draft_compatibility(target, draft_profile)
    if issues:
        for issue in issues:
            console.print(f"  [yellow]⚠ {issue}[/yellow]")
    if not compat:
        console.print("[bold red]Draft model is incompatible with target![/bold red]")
        raise SystemExit(1)
    console.print("[green]Draft model is compatible.[/green]")
    console.print()

    # Find optimal GPU layers for target
    estimator = MemoryEstimator()
    target_gpu_layers = estimator.find_max_gpu_layers(hw, target, context)
    console.print(f"[dim]Target GPU layers: {target_gpu_layers}/{target.layer_count}[/dim]")

    # Build experiment matrix
    controller = SpeculativeController(hw, target, draft_profile)
    matrix = controller.build_experiment_matrix(target_gpu_layers)

    # Show feasibility summary
    console.print()
    console.print("[bold]Variant Feasibility:[/bold]")
    for item in controller.feasible_variants_summary():
        status = "[green]✓[/green]" if item["feasible"] == "yes" else "[red]✗[/red]"
        console.print(f"  {status} {item['variant']}: {item['reason']}")
    console.print(f"\n[bold]Total configurations to test:[/bold] {len(matrix)}")
    console.print()

    # Build base plan
    workload = WorkloadConfig(context_length=context, output_tokens=tokens)
    candidates = PlanGenerator(estimator=estimator).generate_candidates(hw, target, workload)
    if not candidates:
        console.print("[bold red]No feasible base plan![/bold red]")
        raise SystemExit(1)
    base_plan = candidates[0]

    # Run experiments
    server_path = backend_path or find_llama_server()
    adapter = LlamaCppAdapter(server_path)
    runner = BenchmarkRunner(adapter, warmup_runs=1, timed_runs=runs, cooldown_seconds=1.0)

    sweep = DraftLengthSweep(
        adapter=adapter,
        controller=controller,
        runner=runner,
        max_tokens=tokens,
    )

    try:
        results = sweep.run_full_matrix(
            matrix, base_plan,
            hardware_profile=hw,
            model_profile=target,
        )
        sweep.print_sweep_summary(results)

        # Save results
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        import json
        from dataclasses import asdict
        summary_data = []
        for variant in results:
            for sr in variant.sweep_results:
                summary_data.append({
                    "variant_id": sr.variant_id,
                    "draft_length": sr.draft_length,
                    "ttft_ms": sr.metrics.ttft_ms,
                    "tpot_ms": sr.metrics.tpot_ms,
                    "tokens_per_s": sr.metrics.output_tokens_per_s,
                    "error": sr.error,
                })

        summary_file = out_dir / "spec_sweep_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        console.print(f"\n[green]Results saved to {summary_file}[/green]")

    finally:
        adapter.unload()


@cli.command()
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Save Markdown report to file")
def compare(result_files: tuple[str, ...], output: str | None) -> None:
    """Compare benchmark results from JSON files.

    Pass two or more result JSON files to see a side-by-side comparison table.
    """
    from pathlib import Path
    from adhie.benchmark.report import (
        comparison_to_markdown,
        print_comparison_table,
        results_to_comparison,
        generate_summary,
    )

    console.print("[bold blue]ADHIE Benchmark Comparison[/bold blue]")
    console.print()

    results = []
    for f in result_files:
        from adhie.types import BenchmarkResult
        results.append(BenchmarkResult.from_json(Path(f)))

    if not results:
        console.print("[red]No valid result files found.[/red]")
        raise SystemExit(1)

    rows = results_to_comparison(results)
    print_comparison_table(rows, console)

    summary = generate_summary(results)
    console.print()
    console.print(f"[bold]Best configuration:[/bold] {summary.get('best_plan_id', '?')}")
    console.print(f"  TPOT: {summary.get('best_tpot_ms', 0):.1f} ms")
    console.print(f"  tok/s: {summary.get('best_tokens_per_s', 0):.1f}")

    if output:
        md = comparison_to_markdown(rows)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write("# ADHIE Benchmark Comparison\n\n")
            f.write(md)
            f.write("\n")
        console.print(f"\n[green]Report saved to {output}[/green]")


@cli.command()
@click.argument("model_ref")
@click.option("--context", "-c", default=8192, help="Context length")
@click.option("--draft", "-d", default=None, help="Draft model (path or HF repo)")
@click.option("--backend-path", default=None, help="Path to llama-server binary")
@click.option("--runs", "-r", default=3, help="Timed runs per configuration")
@click.option("--tokens", "-t", default=128, help="Output tokens per run")
@click.option("--output", "-o", default="benchmarks/results/ablation", help="Results directory")
def ablation(model_ref: str, context: int, draft: str | None,
             backend_path: str | None, runs: int, tokens: int,
             output: str) -> None:
    """Run ablation studies to isolate each feature's contribution.

    Systematically enables/disables features and benchmarks the delta.
    MODEL_REF can be a local path or HuggingFace repo ID.
    """
    from pathlib import Path
    from adhie.profiler.hardware_profiler import HardwareProfiler
    from adhie.inspector.model_inspector import ModelInspector
    from adhie.planner.plan_generator import PlanGenerator
    from adhie.runtime.llamacpp_adapter import LlamaCppAdapter, find_llama_server
    from adhie.benchmark.runner import BenchmarkRunner
    from adhie.benchmark.ablation import (
        build_ablation_configs,
        plan_pairs_from_configs,
        results_to_ablation_entries,
    )
    from adhie.benchmark.report import (
        ablation_to_markdown,
        print_ablation_table,
    )
    from adhie.types import WorkloadConfig

    console.print("[bold blue]ADHIE Ablation Study[/bold blue]")
    console.print()

    # Resolve models
    model_path = _resolve_model(model_ref)
    draft_path = _resolve_model(draft) if draft else None

    # Profile & inspect
    hw = HardwareProfiler(verbose=False).profile()
    model = ModelInspector().inspect(model_path)
    if draft_path:
        model.draft_model_path = draft_path

    # Generate the ADHIE-selected plan as the base
    workload = WorkloadConfig(context_length=context, output_tokens=tokens)
    candidates = PlanGenerator().generate_candidates(hw, model, workload)
    if not candidates:
        console.print("[bold red]No feasible plans found![/bold red]")
        raise SystemExit(1)
    base_plan = candidates[0]

    # Build ablation configs
    configs = build_ablation_configs(base_plan, hw, model)
    pairs = plan_pairs_from_configs(configs, base_plan)

    console.print(f"[bold]Ablation studies to run:[/bold] {len(configs)}")
    for cfg in configs:
        console.print(f"  • {cfg.name}: {cfg.description}")
    console.print()

    # Run each pair
    server_path = backend_path or find_llama_server()
    adapter = LlamaCppAdapter(server_path)
    runner = BenchmarkRunner(adapter, warmup_runs=1, timed_runs=runs)
    prompt = "Explain the concept of attention mechanisms in transformer models."

    baseline_results = []
    variant_results = []

    try:
        for name, baseline_plan, variant_plan in pairs:
            console.print(f"\n[bold cyan]━━━ {name} ━━━[/bold cyan]")

            # Run baseline
            console.print(f"  [dim]Baseline: {baseline_plan.plan_id}[/dim]")
            b_result = runner.run_single(
                baseline_plan, prompt, tokens,
                hardware_profile=hw, model_profile=model,
            )
            baseline_results.append(b_result)
            adapter.unload()

            # Run variant
            console.print(f"  [dim]Variant: {variant_plan.plan_id}[/dim]")
            v_result = runner.run_single(
                variant_plan, prompt, tokens,
                hardware_profile=hw, model_profile=model,
            )
            variant_results.append(v_result)
            adapter.unload()

        # Compute ablation entries
        entries = results_to_ablation_entries(configs, baseline_results, variant_results)

        # Print results
        console.print()
        print_ablation_table(entries, console)

        # Save
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        md = ablation_to_markdown(entries)
        md_file = out_dir / "ablation_report.md"
        with open(md_file, "w") as f:
            f.write("# ADHIE Ablation Study\n\n")
            f.write(md)
            f.write("\n")
        console.print(f"\n[green]Report saved to {md_file}[/green]")

        # Save individual results
        for r in baseline_results + variant_results:
            r.to_json(out_dir / f"{r.run_id}.json")

    finally:
        adapter.unload()


def _interactive_loop(adapter: "LlamaCppAdapter", console: Console) -> None:
    """Simple interactive chat loop."""
    while True:
        try:
            prompt = console.input("[bold cyan]>>> [/bold cyan]")
        except EOFError:
            break

        if not prompt.strip():
            continue

        console.print()
        response = adapter.generate_simple(prompt)
        console.print(response)
        console.print()

        metrics = adapter.last_metrics()
        if metrics:
            console.print(
                f"[dim]TTFT: {metrics.ttft_ms:.0f}ms | "
                f"TPOT: {metrics.tpot_ms:.1f}ms | "
                f"Tokens/s: {metrics.output_tokens_per_s:.1f} | "
                f"Output tokens: {metrics.total_output_tokens}[/dim]"
            )
            console.print()


if __name__ == "__main__":
    cli()
