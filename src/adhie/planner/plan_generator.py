"""Plan Generator — produce ranked feasible execution plans.

Takes hardware profile, model profile, and workload config, then:
1. Enumerates a small set of candidate configurations
2. Estimates memory for each via MemoryEstimator
3. Rejects infeasible candidates
4. Scores and ranks the survivors
5. Returns a sorted list of RunPlan objects with rationale
"""

from __future__ import annotations

import uuid
from dataclasses import replace

from rich.console import Console
from rich.table import Table

from adhie.estimator.memory_estimator import MemoryEstimator, DTYPE_BITS
from adhie.types import (
    HardwareProfile,
    MemoryEstimate,
    ModelProfile,
    RunPlan,
    WorkloadConfig,
)


# GPU layer split fractions to enumerate
DEFAULT_GPU_SPLIT_FRACTIONS = [1.0, 0.75, 0.50, 0.25, 0.0]

# KV cache type options (only use what llama.cpp supports)
DEFAULT_KV_TYPES = ["f16", "q8_0"]

# KV offload options
DEFAULT_KV_OFFLOAD = [False, True]

# Default CPU thread heuristic: leave 2 cores for OS/draft
MIN_CPU_THREADS = 2
DRAFT_THREAD_RESERVE = 4


class PlanGenerator:
    """Generate and rank feasible execution plans."""

    def __init__(
        self,
        estimator: MemoryEstimator | None = None,
        gpu_split_fractions: list[float] | None = None,
        kv_types: list[str] | None = None,
    ) -> None:
        self.estimator = estimator or MemoryEstimator()
        self.gpu_split_fractions = gpu_split_fractions or DEFAULT_GPU_SPLIT_FRACTIONS
        self.kv_types = kv_types or DEFAULT_KV_TYPES

    def generate_candidates(
        self,
        hardware: HardwareProfile,
        model: ModelProfile,
        workload: WorkloadConfig,
    ) -> list[RunPlan]:
        """Generate, filter, and rank candidate execution plans.

        Returns a list of RunPlan sorted by score (best first).
        """
        # Step 1: Enumerate raw candidates
        raw = self._enumerate_candidates(hardware, model, workload)

        # Step 2: Estimate memory and filter infeasible
        feasible: list[tuple[RunPlan, MemoryEstimate]] = []
        for plan in raw:
            est = self.estimator.estimate(
                hardware=hardware,
                model=model,
                context_length=plan.context_length,
                gpu_layers=plan.gpu_layers,
                kv_dtype=plan.kv_type_k,
                kv_offload=plan.kv_offload,
            )
            if est.is_feasible:
                # Update plan with actual estimates
                plan.estimated_vram_bytes = est.vram_required_bytes
                plan.estimated_ram_bytes = est.ram_required_bytes
                if est.vram_budget_bytes > 0:
                    plan.memory_headroom_pct = (
                        (est.vram_budget_bytes - est.vram_required_bytes)
                        / est.vram_budget_bytes * 100
                    )
                feasible.append((plan, est))

        # Step 3: Score and sort
        scored = [(plan, est, self._score(plan, est, hardware, model))
                  for plan, est in feasible]
        scored.sort(key=lambda x: x[2])  # Lower score = better

        # Step 4: Attach rationale and return
        result = []
        for plan, est, score in scored:
            plan.rationale = self._build_rationale(plan, est, model, hardware)
            result.append(plan)

        return result

    # ------------------------------------------------------------------
    # Candidate enumeration
    # ------------------------------------------------------------------

    def _enumerate_candidates(
        self,
        hardware: HardwareProfile,
        model: ModelProfile,
        workload: WorkloadConfig,
    ) -> list[RunPlan]:
        """Enumerate the candidate configuration space."""
        candidates: list[RunPlan] = []

        # Determine CPU threads: prefer physical cores to avoid hyperthread contention
        if hardware.cpu_physical_cores > 0:
            target_threads = max(MIN_CPU_THREADS, hardware.cpu_physical_cores)
        else:
            total_threads = hardware.cpu_logical_cores
            target_threads = max(MIN_CPU_THREADS, total_threads - 2)  # Reserve 2 for OS

        # Context length
        context = workload.context_length
        if model.max_context > 0:
            context = min(context, model.max_context)

        for kv_type in self.kv_types:
            for kv_offload in DEFAULT_KV_OFFLOAD:
                layer_counts = set()
                for frac in self.gpu_split_fractions:
                    layer_counts.add(int(model.layer_count * frac))

                if hardware.has_gpu():
                    max_gpu = self.estimator.find_max_gpu_layers(
                        hardware, model, context, kv_dtype=kv_type, kv_offload=kv_offload
                    )
                    if max_gpu > 0:
                        layer_counts.add(max_gpu)

                for gpu_layers in sorted(layer_counts, reverse=True):
                    # Skip GPU configs if no GPU
                    if gpu_layers > 0 and not hardware.has_gpu():
                        continue

                    # Skip nonsensical combos
                    if self._should_skip(gpu_layers, kv_type, kv_offload, hardware):
                        continue

                    # Determine execution mode
                    if gpu_layers == model.layer_count:
                        exec_mode = "gpu_resident"
                    elif gpu_layers == 0:
                        exec_mode = "cpu_first"
                    else:
                        exec_mode = "hybrid"

                    # CPU threads for target
                    cpu_threads = target_threads
                    if model.draft_model_path and gpu_layers > 0:
                        # Reserve threads for draft model
                        cpu_threads = max(
                            MIN_CPU_THREADS,
                            target_threads - DRAFT_THREAD_RESERVE,
                        )

                    plan = RunPlan(
                        plan_id=str(uuid.uuid4())[:8],
                        backend="llamacpp",
                        model_path=model.model_path,
                        model_format="gguf",
                        gpu_layers=gpu_layers,
                        cpu_threads=cpu_threads,
                        context_length=context,
                        batch_size=min(2048, context),
                        ubatch_size=min(512, context),
                        kv_type_k=kv_type,
                        kv_type_v=kv_type,
                        kv_offload=kv_offload,
                        execution_mode=exec_mode,
                        # Speculation configured separately (Module 4)
                        draft_model_path=model.draft_model_path,
                        draft_device="none",
                        draft_length=0,
                        spec_type="none",
                    )
                    candidates.append(plan)

        return candidates

    def _should_skip(
        self,
        gpu_layers: int,
        kv_type: str,
        kv_offload: bool,
        hardware: HardwareProfile,
    ) -> bool:
        """Filter out nonsensical or redundant configurations."""
        # No GPU layers but KV on GPU — contradictory
        if gpu_layers == 0 and not kv_offload:
            return True

        # All layers on GPU — KV offload wastes bandwidth
        # (keep one kv_offload=False variant, skip offload when fully GPU-resident)
        # Actually, KV offload with all layers on GPU can still be useful
        # for very long contexts. Keep it.

        # If no GPU, skip non-offloaded KV
        if not hardware.has_gpu() and not kv_offload:
            return True

        return False

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        plan: RunPlan,
        est: MemoryEstimate,
        hardware: HardwareProfile,
        model: ModelProfile,
    ) -> float:
        """Score a plan — lower is better.

        Heuristic scoring before autotuning (Module 6 adds measured scoring).
        Prioritizes:
          1. More GPU layers (faster compute)
          2. KV on GPU (avoids PCIe traffic during attention)
          3. Lower KV quantization only when needed
          4. Memory headroom (avoid OOM)
        """
        score = 0.0

        # Prefer more GPU layers (normalized 0-100)
        if model.layer_count > 0:
            gpu_fraction = plan.gpu_layers / model.layer_count
            score += (1.0 - gpu_fraction) * 100  # 0 = all GPU, 100 = all CPU

        # Small penalty for KV offload to host RAM
        if plan.kv_offload:
            score += 10

        # Small penalty for quantized KV (quality preservation)
        if plan.kv_type_k != "f16":
            score += 5

        # Prefer more headroom (safety)
        if est.vram_budget_bytes > 0:
            usage_ratio = est.vram_required_bytes / est.vram_budget_bytes
            if usage_ratio > 0.98:
                score += 10
            elif usage_ratio > 0.90:
                score += 5

        # Small bonus for hybrid over pure CPU
        if plan.execution_mode == "cpu_first":
            score += 5

        return score

    # ------------------------------------------------------------------
    # Rationale
    # ------------------------------------------------------------------

    def _build_rationale(
        self,
        plan: RunPlan,
        est: MemoryEstimate,
        model: ModelProfile,
        hardware: HardwareProfile,
    ) -> str:
        """Build a human-readable explanation of why this plan was chosen."""
        parts: list[str] = []

        # GPU layer placement
        if plan.gpu_layers == model.layer_count:
            parts.append(
                f"All {model.layer_count} layers fit on GPU "
                f"({est.weight_gpu_bytes / 1e9:.1f} GB weights)"
            )
        elif plan.gpu_layers == 0:
            parts.append(
                f"Model weights ({est.weight_total_bytes / 1e9:.1f} GB) "
                f"exceed GPU budget; running on CPU"
            )
        else:
            parts.append(
                f"{plan.gpu_layers}/{model.layer_count} layers on GPU "
                f"({est.weight_gpu_bytes / 1e9:.1f} GB), "
                f"remaining on CPU ({est.weight_cpu_bytes / 1e9:.1f} GB)"
            )

        # KV cache
        if plan.kv_type_k != "f16":
            parts.append(
                f"{plan.kv_type_k} KV cache saves "
                f"~{(est.kv_total_bytes * 0.5) / 1e6:.0f} MB vs FP16"
            )
        if plan.kv_offload:
            parts.append("KV cache offloaded to RAM to free GPU memory")

        # Memory situation
        if est.vram_budget_bytes > 0:
            headroom_mb = (est.vram_budget_bytes - est.vram_required_bytes) / 1e6
            parts.append(f"VRAM headroom: {headroom_mb:.0f} MB")

        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_candidates(
        self,
        candidates: list[RunPlan],
        console: Console,
    ) -> None:
        """Print a ranked table of candidate plans."""
        if not candidates:
            console.print("[bold red]No feasible plans found.[/bold red]")
            return

        table = Table(
            title=f"Execution Plan Candidates ({len(candidates)} feasible)",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Rank", justify="center", width=4)
        table.add_column("Mode", width=12)
        table.add_column("GPU Layers", justify="right", width=10)
        table.add_column("KV Type", width=8)
        table.add_column("KV Offload", justify="center", width=10)
        table.add_column("Est. VRAM", justify="right", width=10)
        table.add_column("Est. RAM", justify="right", width=10)
        table.add_column("Headroom", justify="right", width=10)

        for i, plan in enumerate(candidates):
            style = "bold green" if i == 0 else ""
            rank = f"{'→ ' if i == 0 else ''}{i + 1}"
            table.add_row(
                rank,
                plan.execution_mode,
                str(plan.gpu_layers),
                plan.kv_type_k,
                "Yes" if plan.kv_offload else "No",
                f"{plan.estimated_vram_bytes / 1e9:.1f} GB",
                f"{plan.estimated_ram_bytes / 1e9:.1f} GB",
                f"{plan.memory_headroom_pct:.0f}%",
                style=style,
            )

        console.print(table)

        # Show rationale for the top pick
        if candidates:
            console.print()
            console.print(f"[bold green]Recommended:[/bold green] Plan #{1}")
            console.print(f"  {candidates[0].rationale}")

