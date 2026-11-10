"""Shared data types for the ADHIE engine.

All subsystems communicate through these dataclasses. Each is serializable
to/from YAML for persistence and reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Hardware Profile
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    """Snapshot of the host machine's compute and memory resources."""

    # GPU
    gpu_name: str | None = None
    gpu_compute_capability: tuple[int, int] | None = None
    vram_total_bytes: int = 0
    vram_free_bytes: int = 0
    gpu_count: int = 0

    # CPU
    cpu_model: str = "unknown"
    cpu_physical_cores: int = 0
    cpu_logical_cores: int = 0
    cpu_freq_mhz: float | None = None
    simd_flags: list[str] = field(default_factory=list)

    # System memory
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0

    # Measured bandwidths (bytes/s) — filled by microbenchmarks
    h2d_bandwidth_bytes_s: float | None = None
    d2h_bandwidth_bytes_s: float | None = None
    ram_bandwidth_bytes_s: float | None = None
    gpu_mem_bandwidth_bytes_s: float | None = None

    # PCIe
    pcie_gen: int | None = None
    pcie_width: int | None = None

    # Storage
    storage_type: str = "unknown"
    storage_read_mbps: float | None = None

    # Runtime versions
    cuda_version: str | None = None
    driver_version: str | None = None

    # Policy
    vram_safety_margin_bytes: int = 512 * 1024 * 1024  # 512 MB default

    def usable_vram_bytes(self) -> int:
        """VRAM available for model/KV after safety margin."""
        return max(0, self.vram_free_bytes - self.vram_safety_margin_bytes)

    def has_gpu(self) -> bool:
        return self.gpu_count > 0 and self.vram_total_bytes > 0

    def fingerprint(self) -> str:
        """Stable hash for cache keying."""
        key_fields = (
            self.gpu_name,
            self.vram_total_bytes,
            self.cpu_model,
            self.cpu_physical_cores,
            self.ram_total_bytes,
        )
        return hashlib.sha256(str(key_fields).encode()).hexdigest()[:16]

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        # Convert tuple to list for YAML safe serialization
        if data.get("gpu_compute_capability") is not None:
            data["gpu_compute_capability"] = list(data["gpu_compute_capability"])
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> HardwareProfile:
        with open(path) as f:
            data = yaml.safe_load(f)
        # Convert list back to tuple
        cc = data.get("gpu_compute_capability")
        if cc and isinstance(cc, list):
            data["gpu_compute_capability"] = tuple(cc)
        return cls(**data)


# ---------------------------------------------------------------------------
# Model Profile
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    """Metadata extracted from a model file (GGUF or config)."""

    model_id: str = "unknown"
    model_path: str = ""
    file_size_bytes: int = 0

    # Architecture
    architecture: str = "unknown"
    model_family: str = "unknown"
    parameter_count: int = 0
    layer_count: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0

    # Attention
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    max_context: int = 0
    attention_type: str = "full"  # "full", "sliding_window", "hybrid"
    sliding_window_size: int | None = None

    # Quantization
    weight_dtype: str = "unknown"
    bits_per_weight: float = 16.0

    # Tokenizer
    tokenizer_id: str = "unknown"
    chat_template: str | None = None

    # MoE
    is_moe: bool = False
    num_experts: int | None = None
    num_active_experts: int | None = None

    # Draft model
    draft_model_path: str | None = None
    draft_compatible: bool | None = None

    def weight_bytes_per_layer(self) -> float:
        """Rough per-layer weight size estimate."""
        if self.layer_count == 0:
            return 0.0
        return self.file_size_bytes / self.layer_count

    def kv_bytes_per_token(self, kv_dtype_bits: int = 16) -> float:
        """KV cache bytes per token per sequence (all layers).

        Formula from spec §3.2:
            M_KV ≈ 2 × L × H_KV × D_head × B_elem
        per token.
        """
        bytes_per_elem = kv_dtype_bits / 8
        return (
            2.0
            * self.layer_count
            * self.num_kv_heads
            * self.head_dim
            * bytes_per_elem
        )

    def fingerprint(self) -> str:
        key_fields = (
            self.model_id,
            self.parameter_count,
            self.layer_count,
            self.weight_dtype,
            self.num_kv_heads,
            self.head_dim,
        )
        return hashlib.sha256(str(key_fields).encode()).hexdigest()[:16]

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> ModelProfile:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Workload Configuration
# ---------------------------------------------------------------------------

@dataclass
class WorkloadConfig:
    """Describes the inference workload for planning and benchmarking."""

    context_length: int = 8192
    output_tokens: int = 256
    active_sequences: int = 1
    batch_size: int = 1

    # Workload type hint for benchmark prompt selection
    workload_type: str = "chat"  # "chat", "code", "summarize"

    # Benchmark prompt references
    prompt_file: str | None = None
    prompt_text: str | None = None

    # Sampling (for reproducibility)
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42

    # Latency targets (optional, for scoring)
    target_tpot_ms: float | None = None
    target_ttft_ms: float | None = None

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> WorkloadConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Run Plan
# ---------------------------------------------------------------------------

@dataclass
class RunPlan:
    """A concrete execution configuration selected by the planner."""

    plan_id: str = ""
    rationale: str = ""

    # Backend
    backend: str = "llamacpp"
    model_path: str = ""
    model_format: str = "gguf"

    # GPU/CPU placement
    gpu_layers: int = 0
    cpu_threads: int = 4
    cpu_threads_draft: int | None = None

    # Context
    context_length: int = 8192
    batch_size: int = 2048
    ubatch_size: int = 512

    # KV cache
    kv_type_k: str = "f16"
    kv_type_v: str = "f16"
    kv_offload: bool = False

    # Speculative decoding
    draft_model_path: str | None = None
    draft_device: str = "none"  # "cpu", "gpu", "none"
    draft_length: int = 0
    draft_threads: int | None = None
    spec_type: str = "none"  # "draft-simple", "none"

    # Memory estimates (filled by estimator)
    estimated_vram_bytes: int = 0
    estimated_ram_bytes: int = 0
    memory_headroom_pct: float = 0.0

    # Predicted execution mode (§7)
    execution_mode: str = "unknown"  # "gpu_resident", "hybrid", "cpu_first", "streaming"

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> RunPlan:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Inference Metrics
# ---------------------------------------------------------------------------

@dataclass
class InferenceMetrics:
    """Measurements captured during one inference request."""

    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    itl_ms_list: list[float] = field(default_factory=list)
    itl_p95_ms: float = 0.0
    output_tokens_per_s: float = 0.0
    prompt_tokens_per_s: float = 0.0
    total_output_tokens: int = 0
    total_prompt_tokens: int = 0
    total_time_s: float = 0.0

    peak_vram_bytes: int = 0
    peak_ram_bytes: int = 0
    cpu_utilization_pct: float = 0.0
    gpu_utilization_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Speculative Decoding Metrics
# ---------------------------------------------------------------------------

@dataclass
class SpecMetrics:
    """Metrics specific to one speculative decoding cycle or aggregated run."""

    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    acceptance_by_position: list[float] = field(default_factory=list)
    effective_tokens_per_s: float = 0.0
    speedup_vs_baseline: float | None = None


# ---------------------------------------------------------------------------
# Memory Estimate
# ---------------------------------------------------------------------------

@dataclass
class MemoryEstimate:
    """Result of the memory estimator for a given model/hardware/config."""

    # Weights
    weight_total_bytes: int = 0
    weight_gpu_bytes: int = 0
    weight_cpu_bytes: int = 0

    # KV cache
    kv_per_token_bytes: float = 0.0
    kv_total_bytes: int = 0
    kv_gpu_bytes: int = 0
    kv_cpu_bytes: int = 0

    # Runtime overhead
    cuda_context_bytes: int = 300 * 1024 * 1024  # ~300 MB default
    execution_workspace_bytes: int = 200 * 1024 * 1024  # ~200 MB default
    safety_margin_bytes: int = 512 * 1024 * 1024  # ~512 MB default

    # Budgets
    vram_budget_bytes: int = 0
    vram_required_bytes: int = 0
    ram_budget_bytes: int = 0
    ram_required_bytes: int = 0

    # Feasibility
    is_feasible: bool = False
    bottleneck: str = "unknown"  # "vram", "ram", "both", "none"
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Benchmark Result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """One complete benchmark run (plan + metrics + metadata)."""

    run_id: str = ""
    timestamp: str = ""
    hardware_fingerprint: str = ""
    model_fingerprint: str = ""
    plan: RunPlan = field(default_factory=RunPlan)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    metrics: InferenceMetrics = field(default_factory=InferenceMetrics)
    spec_metrics: SpecMetrics | None = None

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> BenchmarkResult:
        with open(path) as f:
            data = json.load(f)
        # Reconstruct nested dataclasses
        data["plan"] = RunPlan(**data["plan"])
        data["workload"] = WorkloadConfig(**data["workload"])
        data["metrics"] = InferenceMetrics(**data["metrics"])
        if data.get("spec_metrics"):
            data["spec_metrics"] = SpecMetrics(**data["spec_metrics"])
        return cls(**data)
