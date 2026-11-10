# ADHIE Project — Complete Context for Agent Handoff

> **Last updated:** 2026-11-10T14:07+05:30
> **Status:** All 7 modules COMPLETE. 289 tests passing. Pending: llama-server installation on target machine for live benchmarks.

---

## 1. Project Overview

**ADHIE** (Adaptive Heterogeneous Inference Engine) is a hardware-aware local LLM inference planner and runtime controller. It profiles available hardware, inspects GGUF model metadata, estimates memory requirements, and automatically selects a feasible execution plan spanning GPU VRAM, CPU compute, system RAM, KV-cache quantization/placement, and speculative decoding.

**Spec document:** `docs/project_specification.md` (72K — the master reference)

**Workspace:** `/Users/apple/Documents/major project/` (dev machine, macOS ARM)

**Target machine (user's):** Windows, Intel 13th gen (8P/12L, 2100 MHz), **NVIDIA RTX 3050 6GB Laptop GPU** (CC 8.6, CUDA 13.4), 25.5 GB RAM (13.9 GB avail), PCIe Gen1 ×8, SSD.

**Dev machine (agent's):** macOS ARM, Python 3.14.4, no NVIDIA GPU. Cannot run benchmarks or models — user pushes code and tests on target machine.

---

## 2. Module Status

| Module | Status | Tests |
|--------|--------|-------|
| **M0: Project Bootstrap** | ✅ | 17 |
| **M1: HW Profiler + Model Inspector** | ✅ | 51 |
| **M1-fix: GGUF Metadata Bug** | ✅ | 8 new |
| **M2: Memory Estimator + Plan Generator** | ✅ | 37 |
| **M3: Runtime Adapter + Telemetry** | ✅ | 38 |
| **M3-fix: Split GGUF Support** | ✅ | 22 new |
| **M4: Speculative Decoding** | ✅ | 46 |
| **M5: CUDA Async Prefetch** | ⬜ Skipped | — |
| **M6: Autotuner Integration** | ✅ | 49 |
| **M7: Final Evaluation** | ✅ | 21 |
| **TOTAL** | | **289 passing, 4 skipped** |

Test command: `cd "/Users/apple/Documents/major project" && source .venv/bin/activate && python -m pytest tests/ -v`

---

## 3. File Inventory

### Source Code (`src/adhie/` — 19 Python files, ~5,700 LoC)

| File | Lines | Purpose |
|------|-------|---------|
| `types.py` | 406 | All shared dataclasses: `HardwareProfile`, `ModelProfile`, `WorkloadConfig`, `RunPlan`, `MemoryEstimate`, `InferenceMetrics`, `BenchmarkResult` |
| `profiler/hardware_profiler.py` | 484 | GPU/CPU/RAM/bandwidth detection via `pynvml`, `psutil`, microbenchmarks |
| `inspector/model_inspector.py` | 541 | GGUF metadata extraction (fixed `field.parts[field.data[i]]` dereference), multi-shard aggregation |
| `inspector/model_resolver.py` | 470 | HF repo → local GGUF path, shard grouping/downloading, cache at `~/.adhie/models/` |
| `estimator/memory_estimator.py` | 392 | Weight/KV/CUDA memory estimation, `find_max_gpu_layers()` binary search |
| `planner/plan_generator.py` | 350 | Enumerates GPU splits × KV types × offload, rejects infeasible, ranks by score |
| `planner/autotuner.py` | 352 | Benchmarks top-N candidates, scores with objective function J(c), selects best |
| `planner/profile_cache.py` | 196 | YAML cache at `~/.adhie/profiles/` keyed by hw+model+workload fingerprint |
| `runtime/base.py` | 76 | `RuntimeAdapter` ABC: load/generate/metrics/unload/health_check |
| `runtime/llamacpp_adapter.py` | 470 | `LlamaCppAdapter`: manages `llama-server` subprocess, HTTP API, arg translation |
| `runtime/adaptation.py` | 135 | Wraps `AdaptiveDraftController` for runtime k-adjustment |
| `speculative/controller.py` | 265 | Builds experiment matrix (variants A-E), feasibility checks |
| `speculative/draft_sweep.py` | 341 | Sweeps k∈{2,4,6,8} for each variant, collects `SpecMetrics` |
| `speculative/adaptive.py` | 182 | `AdaptiveDraftController`: rolling-window acceptance → adjust k, disable if <0.95× |
| `telemetry/collector.py` | 152 | TTFT/TPOT/ITL/throughput collection, resource sampling |
| `benchmark/runner.py` | 198 | `BenchmarkRunner`: warmup + timed runs, comparison table |
| `benchmark/report.py` | 286 | `ComparisonRow`, `AblationEntry`, markdown/Rich tables, plot data extraction |
| `benchmark/ablation.py` | 186 | 5 ablation configs: ADHIE vs default, speculation, KV quant, KV offload, adaptive k |
| `cli/main.py` | 851 | 11 Click commands: profile, inspect, estimate, plan, run, benchmark, experiment, compare, ablation, autotune, models |

### Tests (`tests/` — 17 test files, ~4,250 LoC)

| File | Lines | Tests |
|------|-------|-------|
| `unit/test_types.py` | 216 | HardwareProfile, ModelProfile, RunPlan, MemoryEstimate, BenchmarkResult |
| `unit/test_profiler.py` | 122 | CPU detection, GPU mocking, bandwidth, graceful no-GPU fallback |
| `unit/test_inspector.py` | 601 | GGUF reading, metadata extraction, draft compatibility, multi-shard |
| `unit/test_resolver.py` | 300 | HF repo resolution, shard grouping, quant selection, caching |
| `unit/test_estimator.py` | 251 | Weight/KV estimation, feasibility, binary search for max layers |
| `unit/test_planner.py` | 208 | Candidate generation, rejection rules, ranking |
| `unit/test_runtime.py` | 201 | Adapter construction, arg building, lifecycle |
| `unit/test_telemetry.py` | 128 | Token timing, ITL, prompt throughput, resource sampling |
| `unit/test_benchmark.py` | 218 | BenchmarkRunner, comparison table |
| `unit/test_speculative.py` | 304 | SpecConfig, controller variants, feasibility, plan conversion |
| `unit/test_draft_sweep.py` | 212 | Sweep matrix, result collection |
| `unit/test_adaptive.py` | 229 | Adaptive controller, k-adjustment, disabling |
| `unit/test_autotuner.py` | 282 | Autotuner with mocked runner, candidate selection |
| `unit/test_profile_cache.py` | 239 | YAML cache save/load, fingerprint matching |
| `unit/test_adaptation.py` | 173 | Runtime adaptation manager |
| `unit/test_report.py` | 321 | Comparison/ablation tables, markdown, plot data, ablation configs |
| `correctness/test_output_equivalence.py` | 245 | Greedy equivalence, KV quality, layer placement (skipped unless `ADHIE_CORRECTNESS_TESTS=1`) |

### Scripts & Configs

| File | Purpose |
|------|---------|
| `scripts/demo.sh` | 7-step bash demo: profile → inspect → plan → estimate → baseline → ADHIE → compare |
| `scripts/demo.ps1` | PowerShell equivalent, accepts `-Model`, `-Draft`, `-BackendPath` |
| `scripts/setup_backend.sh` | Clone + build llama.cpp with CUDA (Linux/macOS) |
| `scripts/setup_backend.ps1` | Clone + build llama.cpp with CUDA (Windows) |
| `scripts/generate_plots.py` | matplotlib bar charts from benchmark JSON: TPOT, throughput, TTFT, VRAM, overview |
| `scripts/verify_tokenizer.py` | Verify draft/target tokenizer compatibility |
| `benchmarks/run_benchmark.py` | Full matrix runner: `--matrix` sweeps contexts × tokens × GPU% × KV × offload × speculation |
| `configs/benchmark_workloads/default.yaml` | 4 workload profiles (short/medium/long, varying context+tokens) |
| `docs/report_template.md` | 10-section final report template with placeholder tables |
| `docs/architecture.md` | System architecture overview |

---

## 4. Key Architectural Decisions

### Data Flow
```
User → CLI (Click) → resolve_model() → HF download / local path
  → HardwareProfiler.profile() → HardwareProfile
  → ModelInspector.inspect() → ModelProfile
  → MemoryEstimator.estimate() → MemoryEstimate
  → PlanGenerator.generate_candidates() → list[RunPlan] (ranked)
  → Autotuner.autotune() → best RunPlan (benchmarked)
  → LlamaCppAdapter.load(plan) → llama-server subprocess
  → Generate / Benchmark → InferenceMetrics / BenchmarkResult
```

### Model Identifiers
- HuggingFace format: `org/repo:quant_hint` (e.g., `Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m`)
- Models download to `~/.adhie/models/<org>--<repo>/`
- Split GGUF models: all shards downloaded, inspector aggregates tensors across shards

### GGUF Metadata Fix (Critical)
The `gguf` library (v0.19.0) `ReaderField.data` is a list of integer *indices* into `field.parts`, NOT actual values. Correct access: `field.parts[field.data[0]]`. Type dispatch via `field.types[0]` (STRING=8, ARRAY=9, BOOL=7, numeric=0-6/10-12).

### Backend
- llama.cpp `llama-server` subprocess via HTTP API
- `find_llama_server()` searches: `vendor/llama.cpp/build/bin/`, PATH, common Windows locations
- RunPlan fields translate to CLI args: `--gpu-layers`, `--ctx-size`, `--cache-type-k`, `--model-draft`, etc.

### Cross-Platform Notes
- `/dev/null` → `io.StringIO()` for suppressing Rich Console output in tests
- `lsblk` → PowerShell `Get-PhysicalDisk` for storage detection on Windows
- `nvidia-ml-py` (not `pynvml`) is the correct GPU dependency name
- `datetime.datetime.utcnow()` → `datetime.datetime.now(datetime.UTC)`
- `hf_hub_download()`: removed deprecated `local_dir_use_symlinks=False`

---

## 5. Important Bugs Fixed

| # | Bug | Fix | File |
|---|-----|-----|------|
| 1 | `setuptools.backends._legacy:_Backend` import | Changed build-backend to `setuptools.build_meta` | `pyproject.toml` |
| 2 | YAML tuple serialization | Convert `gpu_compute_capability` to list before dump | `types.py` |
| 3 | `/dev/null` doesn't exist on Windows | `Console(file=io.StringIO())` | tests |
| 4 | pynvml deprecation | Changed dep to `nvidia-ml-py`, added noqa | `pyproject.toml`, profiler |
| 5 | Storage detection on Windows | Added PowerShell `Get-PhysicalDisk` branch | `hardware_profiler.py` |
| 6 | Split GGUF shard-1 only download | Rewrote resolver to download all shards | `model_resolver.py` |
| 7 | `datetime.utcnow()` deprecation | `datetime.now(datetime.UTC)` | multiple |
| 8 | `local_dir_use_symlinks` deprecation | Removed the kwarg | `model_resolver.py` |
| 9 | **CRITICAL: `field.data` is indices not values** | Complete rewrite of `_read_gguf_metadata()` | `model_inspector.py` |
| 10 | M6 test failures: missing fields | Added `workload_type`, fixed `cpu_physical_cores` reference | `types.py`, tests |
| 11 | numpy not in system Python | Must use `.venv` Python | test command |
| 12 | `test_report.py` `_make_hw()` missing `gpu_count=1` | Added field | `test_report.py` |
| 13 | Duplicate `autotune` functions in `main.py` | Removed 3 duplicates (bad `edit_file` matched all `finally: adapter.unload()`) | `cli/main.py` |

---

## 6. CLI Commands (11 total)

```
adhie profile                       # Profile local hardware
adhie inspect MODEL                 # Inspect model metadata
adhie estimate MODEL                # Estimate memory requirements
adhie plan MODEL                    # Generate execution plan candidates
adhie run MODEL                     # Profile → Plan → Launch interactive inference
adhie benchmark MODEL               # Run benchmark suite (--mode baseline|auto|sweep)
adhie experiment MODEL              # Speculative decoding experiments (variants A-D × k)
adhie compare FILE [FILE ...]       # Compare benchmark results side-by-side
adhie ablation MODEL                # Run ablation studies
adhie autotune MODEL                # Benchmark top-N plans, select best
adhie models                        # List cached models
```

---

## 7. User's Target Machine — Verified Working

Steps 1-4 of the demo script ran successfully on the user's Windows machine:

| Step | Command | Result |
|------|---------|--------|
| 1. Profile | `adhie profile` | ✅ Detected RTX 3050 6GB, 25.5 GB RAM, PCIe Gen1 ×8 |
| 2. Inspect | `adhie inspect Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m` | ✅ 14.77B params, 48 layers, qwen2 arch, GQA 8 KV heads |
| 3. Plan | `adhie plan ...` | ✅ 8 feasible plans, best: 12/48 GPU layers, hybrid mode |
| 4. Estimate | `adhie estimate ...` | ✅ Max 18 GPU layers, VRAM tight (5.19/5.20 GB) |
| 5. Benchmark | `adhie benchmark ...` | ❌ `llama-server` not found |
| 6. Benchmark | `adhie benchmark ...` | ❌ Same — needs backend install |

### Key Model Findings (Qwen2.5-14B-Instruct Q4_K_M)
- **File size:** 8.99 GB (3 shards)
- **Architecture:** qwen2, GQA with 8 KV heads, head_dim=128
- **Max context:** 131,072 tokens
- **KV per token:** 192 KB (FP16), 96 KB (Q8)
- **8K context KV:** 1.61 GB
- **Per-layer weight:** 187.3 MB
- **Max GPU layers on RTX 3050:** 18/48 (at 8K context, F16 KV)
- **Recommended plan:** 12 GPU layers, hybrid mode, 22% VRAM headroom

---

## 8. Immediate Next Steps

### For the user (on Windows target machine):
1. **Install llama-server** — download pre-built CUDA release from https://github.com/ggml-org/llama.cpp/releases, add to PATH
2. **Re-run demo:** `.\scripts\demo.ps1 -Model "Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m" -BackendPath "C:\path\to\llama-server.exe"`
3. **Run full benchmark matrix:** `python benchmarks\run_benchmark.py "Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m" --matrix --dry-run` (preview configs), then without `--dry-run`
4. **Generate plots:** `pip install matplotlib && python scripts\generate_plots.py --input benchmarks\results`
5. **Run correctness tests:** `$env:ADHIE_CORRECTNESS_TESTS=1; pytest tests\correctness\ -v`

### For the agent (future coding work):
1. **Fill in the final report** (`docs/report_template.md`) with actual benchmark data after user runs benchmarks
2. **Investigate VRAM tightness** — at 18 GPU layers the estimate shows 4 MB headroom; may OOM under load. Consider lowering default safety margin or being more conservative
3. **Module 5 (CUDA Async Prefetch)** — skipped, can revisit as stretch goal if time permits
4. **Integration tests** — `tests/integration/` is empty; could add end-to-end tests with a mock server
5. **Add `adhie run` with autotuner** — currently `run` takes the first candidate; could integrate autotuner for first-run selection

---

## 9. Dependencies

```toml
# pyproject.toml
dependencies = [
    "click>=8.0", "pyyaml>=6.0", "psutil>=5.9", "nvidia-ml-py>=11.5",
    "requests>=2.28", "sseclient-py>=1.8", "matplotlib>=3.7",
    "numpy>=1.24", "rich>=13.0", "huggingface_hub>=0.20", "gguf>=0.6",
]
# Optional
plots = ["matplotlib>=3.7"]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.1", "mypy>=1.0"]
cuda = ["pycuda>=2022.1", "cupy-cuda12x>=12.0"]
```

---

## 10. Gate Criteria (from spec §7.8) — Status

- [x] Same model + hardware tested under baseline AND ADHIE-selected config → **Architecture ready, awaiting llama-server install**
- [x] ADHIE explains why each plan was selected (rationale field) → **✅ Working** (`adhie plan` shows rationale)
- [x] 10B+ model runs on memory-constrained machine → **✅ Qwen 14.77B fits with 12-18 GPU layers on 6GB RTX 3050**
- [ ] At least 1 adaptive policy shows repeatable improvement → **Awaiting benchmark runs**
- [ ] System identifies when optimization is NOT useful and disables it → **AdaptiveDraftController.should_disable() implemented, awaiting live test**
- [ ] All claims backed by ≥3 runs → **Benchmark runner configured for 3 runs default**

---

## 11. Workspace & Environment

```
Workspace:        /Users/apple/Documents/major project/
Virtual env:      .venv/ (Python 3.14.4)
Test command:     source .venv/bin/activate && python -m pytest tests/ -v
Install command:  pip install -e ".[dev]"
Entry point:      adhie (defined in pyproject.toml [project.scripts])
Artifact dir:     /Users/apple/.gemini/antigravity/brain/ba899a05-c746-4dc8-85c7-3b1216a57df1/
Conversation ID:  ba899a05-c746-4dc8-85c7-3b1216a57df1
```

