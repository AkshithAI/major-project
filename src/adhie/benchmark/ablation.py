"""Ablation Study Driver — systematically isolate feature contributions.

Defines a set of ablation configurations (feature ON vs OFF pairs),
generates the corresponding RunPlans, and orchestrates benchmark runs.
The result is a list of AblationEntry objects showing each feature's
impact on latency, throughput, and memory.

Ablation studies from the spec:
  1. ADHIE-selected vs backend default (all CPU)
  2. Speculation ON vs OFF
  3. Adaptive k vs fixed k=4
  4. Q8 KV vs F16 KV
  5. KV offload vs GPU KV
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from rich.console import Console

from adhie.benchmark.report import AblationEntry, compute_ablation
from adhie.types import (
    BenchmarkResult,
    HardwareProfile,
    InferenceMetrics,
    ModelProfile,
    RunPlan,
    WorkloadConfig,
)


@dataclass
class AblationConfig:
    """Describes one ablation pair: baseline vs variant."""

    name: str
    description: str
    baseline_label: str
    variant_label: str
    make_baseline: Any = None   # Callable[[RunPlan], RunPlan]
    make_variant: Any = None    # Callable[[RunPlan], RunPlan]


def build_ablation_configs(
    base_plan: RunPlan,
    hw: HardwareProfile,
    model: ModelProfile,
) -> list[AblationConfig]:
    """Build the standard set of ablation configurations.

    Each config produces a pair of plans: baseline (feature OFF)
    and variant (feature ON), derived from the base_plan.
    """
    configs: list[AblationConfig] = []

    # 1. ADHIE-selected vs backend default (CPU-only)
    configs.append(AblationConfig(
        name="ADHIE vs Default",
        description="ADHIE-selected plan vs all-CPU backend default",
        baseline_label="CPU-only (default)",
        variant_label="ADHIE-selected",
        make_baseline=lambda p: replace(p,
            plan_id="ablation-default",
            gpu_layers=0,
            execution_mode="cpu_first",
            kv_type_k="f16",
            kv_type_v="f16",
            kv_offload=True,
            spec_type="none",
            draft_model_path=None,
            draft_length=0,
        ),
        make_variant=lambda p: replace(p,
            plan_id="ablation-adhie",
        ),
    ))

    # 2. Speculation ON vs OFF
    if base_plan.draft_model_path:
        configs.append(AblationConfig(
            name="Speculation",
            description="Speculative decoding ON vs OFF",
            baseline_label="No speculation",
            variant_label=f"Spec k={base_plan.draft_length}",
            make_baseline=lambda p: replace(p,
                plan_id="ablation-no-spec",
                spec_type="none",
                draft_model_path=None,
                draft_length=0,
                draft_device="none",
            ),
            make_variant=lambda p: replace(p,
                plan_id="ablation-with-spec",
            ),
        ))

    # 3. Q8 KV vs F16 KV
    configs.append(AblationConfig(
        name="KV Quantization",
        description="Q8_0 KV cache vs F16 KV cache",
        baseline_label="KV F16",
        variant_label="KV Q8_0",
        make_baseline=lambda p: replace(p,
            plan_id="ablation-kv-f16",
            kv_type_k="f16",
            kv_type_v="f16",
        ),
        make_variant=lambda p: replace(p,
            plan_id="ablation-kv-q8",
            kv_type_k="q8_0",
            kv_type_v="q8_0",
        ),
    ))

    # 4. KV offload vs GPU KV (only if GPU is available)
    if hw.has_gpu() and base_plan.gpu_layers > 0:
        configs.append(AblationConfig(
            name="KV Offload",
            description="KV cache on CPU (offloaded) vs GPU",
            baseline_label="KV on GPU",
            variant_label="KV offloaded to RAM",
            make_baseline=lambda p: replace(p,
                plan_id="ablation-kv-gpu",
                kv_offload=False,
            ),
            make_variant=lambda p: replace(p,
                plan_id="ablation-kv-offload",
                kv_offload=True,
            ),
        ))

    # 5. Adaptive k vs fixed k=4 (only if speculation is enabled)
    if base_plan.draft_model_path and base_plan.draft_length > 0:
        fixed_k = 4
        configs.append(AblationConfig(
            name="Adaptive vs Fixed k",
            description=f"Adaptive draft length vs fixed k={fixed_k}",
            baseline_label=f"Fixed k={fixed_k}",
            variant_label="Adaptive k",
            make_baseline=lambda p: replace(p,
                plan_id="ablation-fixed-k",
                draft_length=fixed_k,
            ),
            make_variant=lambda p: replace(p,
                plan_id="ablation-adaptive-k",
                # Adaptive k is the default behavior
            ),
        ))

    return configs


def plan_pairs_from_configs(
    configs: list[AblationConfig],
    base_plan: RunPlan,
) -> list[tuple[str, RunPlan, RunPlan]]:
    """Generate (name, baseline_plan, variant_plan) tuples."""
    pairs = []
    for cfg in configs:
        baseline = cfg.make_baseline(base_plan)
        variant = cfg.make_variant(base_plan)
        pairs.append((cfg.name, baseline, variant))
    return pairs


def results_to_ablation_entries(
    configs: list[AblationConfig],
    baseline_results: list[BenchmarkResult],
    variant_results: list[BenchmarkResult],
) -> list[AblationEntry]:
    """Match configs with results and compute ablation entries."""
    entries = []
    for i, cfg in enumerate(configs):
        if i < len(baseline_results) and i < len(variant_results):
            entry = compute_ablation(
                feature_name=cfg.name,
                baseline=baseline_results[i],
                variant=variant_results[i],
                baseline_label=cfg.baseline_label,
                variant_label=cfg.variant_label,
            )
            entries.append(entry)
    return entries

