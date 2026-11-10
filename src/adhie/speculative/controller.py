"""Speculative Decoding Controller — configure and manage speculation experiments.

Builds experiment matrices for the 5 speculative decoding variants (A-E):
  A: No speculation (baseline)
  B: CPU draft + GPU target (ADHIE hypothesis)
  C: GPU draft + GPU target (if VRAM permits)
  D: CPU draft + CPU target (low-VRAM comparison)
  E: Prompt-lookup / n-gram draft (no external model)

The controller determines which variants are feasible for a given
hardware + model combination and generates the corresponding RunPlans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adhie.types import (
    HardwareProfile,
    ModelProfile,
    RunPlan,
    WorkloadConfig,
)


# ---------------------------------------------------------------------------
# Spec Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpecConfig:
    """Configuration for one speculative decoding variant."""

    variant_id: str = ""           # e.g., "A", "B-k4", "C-k6"
    variant_name: str = ""         # Human-readable label
    enabled: bool = False
    draft_model_path: str | None = None
    draft_device: str = "none"     # "cpu", "gpu", "none"
    draft_length: int = 4          # k value
    draft_threads: int = 4
    spec_type: str = "none"        # "sd" (speculative decoding), "none"
    gpu_layers: int = 0            # Target model GPU layers
    kv_type: str = "f16"


# ---------------------------------------------------------------------------
# Feasibility Helpers
# ---------------------------------------------------------------------------

def _estimate_draft_ram(draft_profile: ModelProfile) -> int:
    """Rough estimate of RAM needed to run draft model on CPU."""
    return draft_profile.file_size_bytes + 256 * 1024 * 1024  # weights + overhead


def _estimate_draft_vram(draft_profile: ModelProfile) -> int:
    """Rough estimate of VRAM needed to run draft model on GPU."""
    # All layers on GPU + small overhead
    return draft_profile.file_size_bytes + 128 * 1024 * 1024


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class SpeculativeController:
    """Generates experiment configurations for speculative decoding."""

    DRAFT_LENGTHS = [2, 4, 6, 8]

    def __init__(
        self,
        hardware: HardwareProfile,
        target_profile: ModelProfile,
        draft_profile: ModelProfile | None = None,
    ) -> None:
        self.hardware = hardware
        self.target = target_profile
        self.draft = draft_profile

    def build_experiment_matrix(
        self,
        target_gpu_layers: int,
        kv_type: str = "f16",
    ) -> list[SpecConfig]:
        """Generate the experiment matrix of speculative decoding variants.

        Returns a list of SpecConfigs representing each feasible variant × k.
        """
        configs: list[SpecConfig] = []

        # --- Variant A: No speculation (baseline) ---
        configs.append(SpecConfig(
            variant_id="A",
            variant_name="No speculation (baseline)",
            enabled=False,
            gpu_layers=target_gpu_layers,
            kv_type=kv_type,
        ))

        if self.draft is None:
            return configs

        draft_ram = _estimate_draft_ram(self.draft)
        draft_vram = _estimate_draft_vram(self.draft)
        available_ram = self.hardware.ram_available_bytes
        usable_vram = self.hardware.usable_vram_bytes()

        # Calculate how much VRAM is left after target model
        target_vram_usage = int(
            target_gpu_layers * self.target.weight_bytes_per_layer()
        )
        vram_remaining = usable_vram - target_vram_usage

        cpu_threads = max(2, self.hardware.cpu_physical_cores // 3)

        # --- Variant B: CPU draft + GPU target (ADHIE core hypothesis) ---
        if available_ram > draft_ram:
            for k in self.DRAFT_LENGTHS:
                configs.append(SpecConfig(
                    variant_id=f"B-k{k}",
                    variant_name=f"CPU draft + GPU target (k={k})",
                    enabled=True,
                    draft_model_path=self.draft.model_path,
                    draft_device="cpu",
                    draft_length=k,
                    draft_threads=cpu_threads,
                    spec_type="sd",
                    gpu_layers=target_gpu_layers,
                    kv_type=kv_type,
                ))

        # --- Variant C: GPU draft + GPU target (if VRAM permits) ---
        if vram_remaining > draft_vram:
            for k in self.DRAFT_LENGTHS:
                configs.append(SpecConfig(
                    variant_id=f"C-k{k}",
                    variant_name=f"GPU draft + GPU target (k={k})",
                    enabled=True,
                    draft_model_path=self.draft.model_path,
                    draft_device="gpu",
                    draft_length=k,
                    draft_threads=0,
                    spec_type="sd",
                    gpu_layers=target_gpu_layers,
                    kv_type=kv_type,
                ))

        # --- Variant D: CPU draft + CPU target (low-VRAM fallback) ---
        if available_ram > draft_ram + self.target.file_size_bytes:
            for k in self.DRAFT_LENGTHS:
                configs.append(SpecConfig(
                    variant_id=f"D-k{k}",
                    variant_name=f"CPU draft + CPU target (k={k})",
                    enabled=True,
                    draft_model_path=self.draft.model_path,
                    draft_device="cpu",
                    draft_length=k,
                    draft_threads=cpu_threads,
                    spec_type="sd",
                    gpu_layers=0,  # All CPU
                    kv_type=kv_type,
                ))

        # --- Variant E: Prompt-lookup (no external draft model) ---
        # llama.cpp doesn't support prompt-lookup natively yet;
        # placeholder for future integration
        # configs.append(SpecConfig(
        #     variant_id="E",
        #     variant_name="Prompt-lookup (n-gram)",
        #     enabled=True,
        #     draft_device="none",
        #     spec_type="prompt-lookup",
        # ))

        return configs

    def config_to_plan(
        self,
        config: SpecConfig,
        base_plan: RunPlan,
    ) -> RunPlan:
        """Convert a SpecConfig into a RunPlan by overlaying spec settings
        on top of a base plan.

        The base plan provides model_path, context_length, batch sizes, etc.
        The SpecConfig overrides speculation and GPU layer settings.
        """
        from dataclasses import replace

        plan = replace(
            base_plan,
            plan_id=f"spec-{config.variant_id}",
            gpu_layers=config.gpu_layers,
            kv_type_k=config.kv_type,
            kv_type_v=config.kv_type,
            draft_model_path=config.draft_model_path,
            draft_device=config.draft_device,
            draft_length=config.draft_length,
            draft_threads=config.draft_threads if config.draft_threads else None,
            spec_type=config.spec_type,
            rationale=config.variant_name,
        )

        # Set execution mode
        if config.gpu_layers == 0:
            plan.execution_mode = "cpu_first"
        elif config.gpu_layers >= self.target.layer_count:
            plan.execution_mode = "gpu_resident"
        else:
            plan.execution_mode = "hybrid"

        return plan

    def feasible_variants_summary(self) -> list[dict[str, str]]:
        """Return a summary of which variants are feasible and why."""
        summary = []
        summary.append({
            "variant": "A (baseline)",
            "feasible": "yes",
            "reason": "No draft model needed",
        })

        if self.draft is None:
            summary.append({
                "variant": "B/C/D (speculation)",
                "feasible": "no",
                "reason": "No draft model provided",
            })
            return summary

        draft_ram = _estimate_draft_ram(self.draft)
        draft_vram = _estimate_draft_vram(self.draft)
        available_ram = self.hardware.ram_available_bytes
        usable_vram = self.hardware.usable_vram_bytes()

        summary.append({
            "variant": "B (CPU draft)",
            "feasible": "yes" if available_ram > draft_ram else "no",
            "reason": (
                f"Draft needs ~{draft_ram / 1e9:.1f} GB RAM, "
                f"{available_ram / 1e9:.1f} GB available"
            ),
        })

        summary.append({
            "variant": "C (GPU draft)",
            "feasible": "yes" if usable_vram > draft_vram else "no",
            "reason": (
                f"Draft needs ~{draft_vram / 1e9:.1f} GB VRAM, "
                f"{usable_vram / 1e9:.1f} GB usable"
            ),
        })

        total_cpu_needed = draft_ram + self.target.file_size_bytes
        summary.append({
            "variant": "D (all CPU)",
            "feasible": "yes" if available_ram > total_cpu_needed else "no",
            "reason": (
                f"Both models need ~{total_cpu_needed / 1e9:.1f} GB RAM, "
                f"{available_ram / 1e9:.1f} GB available"
            ),
        })

        return summary

