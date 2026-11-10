"""ADHIE Command-Line Interface.

Usage:
    adhie profile                       Profile local hardware
    adhie inspect MODEL                 Inspect model metadata
    adhie plan MODEL                    Generate execution plan candidates
    adhie run MODEL                     Profile, plan, and run inference
    adhie benchmark MODEL               Run benchmark suite
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
    from adhie.types import WorkloadConfig

    console.print("[bold blue]ADHIE Plan Generator[/bold blue]")
    console.print()

    # Resolve models
    model_path = _resolve_model(model_ref)
    draft_path = _resolve_model(draft) if draft else None

    hw = HardwareProfiler().profile()
    model = ModelInspector().inspect(model_path)
    if draft_path:
        model.draft_model_path = draft_path

    workload = WorkloadConfig(context_length=context)
    generator = PlanGenerator()
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
    server_path = backend_path or _find_llama_server()
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
def benchmark(model_ref: str, context: int, draft: str | None,
              output: str, mode: str, backend_path: str | None) -> None:
    """Run the benchmark suite and save results.

    MODEL_REF can be a local path or HuggingFace repo ID.
    """
    console.print("[bold blue]ADHIE Benchmark Suite[/bold blue]")
    console.print(f"Mode: {mode}")
    console.print("(Benchmark runner will be fully implemented in Module 3)")


def _find_llama_server() -> str:
    """Locate llama-server binary."""
    import shutil
    from pathlib import Path

    # Check vendor directory first
    vendor_path = (Path(__file__).parent.parent.parent.parent
                   / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server")
    if vendor_path.exists():
        return str(vendor_path)

    # Check PATH
    path = shutil.which("llama-server")
    if path:
        return path

    raise FileNotFoundError(
        "llama-server not found. Either:\n"
        "  1. Run ./scripts/setup_backend.sh to build it, or\n"
        "  2. Pass --backend-path /path/to/llama-server"
    )


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
