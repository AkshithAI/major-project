"""Memory Estimator — estimate VRAM/RAM requirements for a model configuration.

Computes weight memory, KV cache memory, runtime overhead, and determines
feasibility for a given hardware + model + workload + placement combination.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from adhie.types import HardwareProfile, MemoryEstimate, ModelProfile, WorkloadConfig


# Bytes-per-element for common KV/weight dtype names
DTYPE_BITS: dict[str, float] = {
    "f32": 32.0,
    "f16": 16.0,
    "bf16": 16.0,
    "q8_0": 8.5,
    "q8_1": 9.0,
    "q6_k": 6.6,
    "q5_k_m": 5.7,
    "q5_k_s": 5.5,
    "q5_0": 5.5,
    "q5_1": 6.0,
    "q4_k_m": 4.8,
    "q4_k_s": 4.5,
    "q4_0": 4.5,
    "q4_1": 5.0,
    "q3_k_m": 3.9,
    "q3_k_l": 4.3,
    "q3_k_s": 3.5,
    "q2_k": 3.35,
    "iq4_xs": 4.25,
    "iq4_nl": 4.5,
    "iq3_s": 3.44,
    "iq3_xxs": 3.06,
    "iq2_xs": 2.31,
    "iq2_xxs": 2.06,
    "iq1_s": 1.56,
}

# Default overhead constants (conservative estimates)
DEFAULT_CUDA_CONTEXT_BYTES = 300 * 1024 * 1024      # ~300 MB for CUDA context
DEFAULT_EXECUTION_WORKSPACE_BYTES = 200 * 1024 * 1024  # ~200 MB for activations/scratch
DEFAULT_SAFETY_MARGIN_BYTES = 512 * 1024 * 1024      # ~512 MB safety margin


def dtype_to_bytes_per_element(dtype: str) -> float:
    """Convert a dtype name to bytes per element."""
    bits = DTYPE_BITS.get(dtype.lower(), 16.0)
    return bits / 8.0


class MemoryEstimator:
    """Estimate memory requirements for model inference configurations."""

    def __init__(
        self,
        cuda_context_bytes: int = DEFAULT_CUDA_CONTEXT_BYTES,
        execution_workspace_bytes: int = DEFAULT_EXECUTION_WORKSPACE_BYTES,
        safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES,
    ) -> None:
        self.cuda_context_bytes = cuda_context_bytes
        self.execution_workspace_bytes = execution_workspace_bytes
        self.safety_margin_bytes = safety_margin_bytes

    def estimate(
        self,
        hardware: HardwareProfile,
        model: ModelProfile,
        context_length: int,
        gpu_layers: int,
        kv_dtype: str = "f16",
        kv_offload: bool = False,
    ) -> MemoryEstimate:
        """Compute a memory estimate for the given configuration.

        Args:
            hardware: Machine capabilities.
            model: Model metadata.
            context_length: Requested context length in tokens.
            gpu_layers: Number of layers placed on GPU.
            kv_dtype: Data type for KV cache entries.
            kv_offload: Whether KV cache is offloaded to host RAM.

        Returns:
            MemoryEstimate with feasibility analysis.
        """
        est = MemoryEstimate()
        warnings: list[str] = []

        # ------------------------------------------------------------------
        # Weight memory
        # ------------------------------------------------------------------
        weight_per_layer = model.weight_bytes_per_layer()

        # Clamp gpu_layers to valid range
        gpu_layers = max(0, min(gpu_layers, model.layer_count))

        est.weight_total_bytes = model.file_size_bytes
        est.weight_gpu_bytes = int(weight_per_layer * gpu_layers)
        est.weight_cpu_bytes = int(weight_per_layer * (model.layer_count - gpu_layers))

        # Non-layer weights (embedding, output head, norms) — rough estimate
        # These are typically a small fraction; assume ~5% of total or
        # the difference between total file size and per-layer sum
        non_layer_bytes = max(
            0,
            model.file_size_bytes - int(weight_per_layer * model.layer_count)
        )
        # Embedding/output head goes on GPU if any layers are on GPU
        if gpu_layers > 0:
            est.weight_gpu_bytes += non_layer_bytes
        else:
            est.weight_cpu_bytes += non_layer_bytes

        # ------------------------------------------------------------------
        # KV cache memory
        # ------------------------------------------------------------------
        kv_bits = DTYPE_BITS.get(kv_dtype.lower(), 16.0)
        est.kv_per_token_bytes = model.kv_bytes_per_token(kv_dtype_bits=int(kv_bits))
        est.kv_total_bytes = int(est.kv_per_token_bytes * context_length)

        if kv_offload:
            est.kv_gpu_bytes = 0
            est.kv_cpu_bytes = est.kv_total_bytes
        else:
            est.kv_gpu_bytes = est.kv_total_bytes
            est.kv_cpu_bytes = 0

        # ------------------------------------------------------------------
        # Runtime overhead
        # ------------------------------------------------------------------
        if hardware.has_gpu() and gpu_layers > 0:
            est.cuda_context_bytes = self.cuda_context_bytes
            est.execution_workspace_bytes = self.execution_workspace_bytes
        else:
            est.cuda_context_bytes = 0
            est.execution_workspace_bytes = self.execution_workspace_bytes // 2

        est.safety_margin_bytes = self.safety_margin_bytes

        # ------------------------------------------------------------------
        # Budgets
        # ------------------------------------------------------------------
        if hardware.has_gpu():
            est.vram_budget_bytes = max(0, (
                hardware.vram_free_bytes
                - est.cuda_context_bytes
                - est.safety_margin_bytes
            ))
        else:
            est.vram_budget_bytes = 0

        est.ram_budget_bytes = max(0, (
            hardware.ram_available_bytes
            - self.safety_margin_bytes  # RAM safety margin
        ))

        # ------------------------------------------------------------------
        # Required totals
        # ------------------------------------------------------------------
        est.vram_required_bytes = (
            est.weight_gpu_bytes
            + est.kv_gpu_bytes
            + est.execution_workspace_bytes
        )

        est.ram_required_bytes = (
            est.weight_cpu_bytes
            + est.kv_cpu_bytes
            + self.execution_workspace_bytes // 2  # CPU-side scratch
        )

        # ------------------------------------------------------------------
        # Feasibility
        # ------------------------------------------------------------------
        vram_ok = est.vram_required_bytes <= est.vram_budget_bytes or gpu_layers == 0
        ram_ok = est.ram_required_bytes <= est.ram_budget_bytes

        if not vram_ok and not ram_ok:
            est.bottleneck = "both"
        elif not vram_ok:
            est.bottleneck = "vram"
        elif not ram_ok:
            est.bottleneck = "ram"
        else:
            est.bottleneck = "none"

        est.is_feasible = vram_ok and ram_ok

        # ------------------------------------------------------------------
        # Warnings
        # ------------------------------------------------------------------
        if gpu_layers == 0 and not kv_offload and hardware.has_gpu():
            warnings.append(
                "KV cache placed on GPU but no layers on GPU — "
                "this is contradictory; setting kv_offload is recommended"
            )

        if context_length > model.max_context and model.max_context > 0:
            warnings.append(
                f"Requested context ({context_length}) exceeds model maximum "
                f"({model.max_context})"
            )
            est.is_feasible = False

        vram_usage_pct = (
            (est.vram_required_bytes / est.vram_budget_bytes * 100)
            if est.vram_budget_bytes > 0 else 0
        )
        if vram_usage_pct > 90 and est.is_feasible:
            warnings.append(
                f"VRAM usage is {vram_usage_pct:.0f}% of budget — "
                "tight margin, OOM risk under load"
            )

        kv_fraction = (
            est.kv_total_bytes / max(1, est.kv_total_bytes + est.weight_gpu_bytes)
        )
        if kv_fraction > 0.5 and context_length > 8192:
            warnings.append(
                f"KV cache is {kv_fraction:.0%} of GPU memory usage — "
                "consider Q8 KV or KV offload for long contexts"
            )

        est.warnings = warnings
        return est

    def find_max_gpu_layers(
        self,
        hardware: HardwareProfile,
        model: ModelProfile,
        context_length: int,
        kv_dtype: str = "f16",
        kv_offload: bool = False,
    ) -> int:
        """Find the maximum number of layers that fit on GPU.

        Binary search for the highest gpu_layers value that produces
        a feasible estimate.
        """
        if not hardware.has_gpu():
            return 0

        low, high = 0, model.layer_count
        best = 0

        while low <= high:
            mid = (low + high) // 2
            est = self.estimate(hardware, model, context_length, mid, kv_dtype, kv_offload)
            if est.is_feasible:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        return best

    def print_report(
        self,
        est: MemoryEstimate,
        model: ModelProfile,
        gpu_layers: int,
        context_length: int,
        console: Console,
    ) -> None:
        """Print a formatted memory estimate report."""
        table = Table(title="Memory Estimate", show_header=True,
                      header_style="bold cyan")
        table.add_column("Category", style="bold")
        table.add_column("Component")
        table.add_column("Size", justify="right")
        table.add_column("Device", justify="center")

        # Weights
        table.add_row(
            "Weights", "GPU-resident layers",
            _fmt_bytes(est.weight_gpu_bytes),
            "GPU" if est.weight_gpu_bytes > 0 else "—",
        )
        table.add_row(
            "", "CPU-offloaded layers",
            _fmt_bytes(est.weight_cpu_bytes),
            "RAM" if est.weight_cpu_bytes > 0 else "—",
        )
        table.add_row(
            "", f"Total ({model.weight_dtype})",
            _fmt_bytes(est.weight_total_bytes), "",
        )

        table.add_section()

        # KV Cache
        table.add_row(
            "KV Cache", f"Per token",
            f"{est.kv_per_token_bytes / 1024:.1f} KB", "",
        )
        table.add_row(
            "", f"Total ({context_length:,} tokens)",
            _fmt_bytes(est.kv_total_bytes),
            "GPU" if est.kv_gpu_bytes > 0 else "RAM",
        )

        table.add_section()

        # Runtime
        if est.cuda_context_bytes > 0:
            table.add_row(
                "Runtime", "CUDA context",
                _fmt_bytes(est.cuda_context_bytes), "GPU",
            )
        table.add_row(
            "", "Execution workspace",
            _fmt_bytes(est.execution_workspace_bytes),
            "GPU" if est.cuda_context_bytes > 0 else "CPU",
        )
        table.add_row(
            "", "Safety margin",
            _fmt_bytes(est.safety_margin_bytes), "Reserved",
        )

        table.add_section()

        # Totals
        table.add_row(
            "VRAM", "Required",
            _fmt_bytes(est.vram_required_bytes), "",
        )
        table.add_row(
            "", "Budget",
            _fmt_bytes(est.vram_budget_bytes), "",
        )
        headroom = est.vram_budget_bytes - est.vram_required_bytes
        headroom_style = "green" if headroom > 0 else "red"
        table.add_row(
            "", "Headroom",
            f"[{headroom_style}]{_fmt_bytes(abs(headroom))} "
            f"{'✓' if headroom >= 0 else '✗ OVER'}[/{headroom_style}]",
            "",
        )

        table.add_section()

        table.add_row(
            "RAM", "Required",
            _fmt_bytes(est.ram_required_bytes), "",
        )
        table.add_row(
            "", "Budget",
            _fmt_bytes(est.ram_budget_bytes), "",
        )

        table.add_section()

        # Feasibility
        if est.is_feasible:
            table.add_row(
                "Feasibility", "Status",
                "[bold green]FEASIBLE ✓[/bold green]", "",
            )
        else:
            table.add_row(
                "Feasibility", "Status",
                f"[bold red]INFEASIBLE ✗[/bold red] (bottleneck: {est.bottleneck})",
                "",
            )

        console.print(table)

        # Warnings
        if est.warnings:
            console.print()
            for w in est.warnings:
                console.print(f"  [yellow]⚠ {w}[/yellow]")


def _fmt_bytes(b: int | float) -> str:
    """Format bytes as a human-readable string."""
    b = abs(b)
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    elif b >= 1e6:
        return f"{b / 1e6:.0f} MB"
    elif b >= 1e3:
        return f"{b / 1e3:.0f} KB"
    return f"{b:.0f} B"

