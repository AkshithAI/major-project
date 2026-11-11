# ADHIE Project — Complete Context for Agent Handoff

> **Last updated:** 2026-11-10T20:56+05:30  
> **Status:** All 7 modules COMPLETE. 289 tests passing. Backend, CUDA offloading, telemetry, and end-to-end demo script fully functional and verified on target machine.

---

## 1. Project Overview

**ADHIE** (Adaptive Heterogeneous Inference Engine) is a hardware-aware local LLM inference planner and runtime controller. It profiles available hardware, inspects GGUF model metadata, estimates memory requirements, and automatically selects an optimal feasible execution plan spanning GPU VRAM, CPU compute, system RAM, KV-cache quantization/placement, and speculative decoding.

**Spec document:** `docs/project_specification.md` (72K — the master reference)

**Target machine (active workspace):** `d:\major-project` (Windows, Intel 13th Gen i5 [8P/12L], **NVIDIA RTX 3050 6GB Laptop GPU** [CC 8.6, CUDA 13.4 driver, CUDA 12.4 runtime], 25.5 GB RAM, PCIe Gen1 ×8, SSD).

**Python environment:** Python 3.13.2 in `d:\major-project\.venv`

---

## 2. Module Status

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| **M0: Project Bootstrap** | ✅ | 17 | Packaging, typing, project structure |
| **M1: HW Profiler + Model Inspector** | ✅ | 51 | pynvml, psutil, GGUF metadata parser |
| **M1-fix: GGUF Metadata Bug** | ✅ | 8 | Fixed `field.parts[field.data[i]]` dereference |
| **M2: Memory Estimator + Plan Generator** | ✅ | 37 | Memory budgeting, dynamic layer offload, hybrid plans |
| **M3: Runtime Adapter + Telemetry** | ✅ | 38 | `LlamaCppAdapter`, SSE streaming, token timing |
| **M3-fix: Split GGUF Support** | ✅ | 22 | Multi-shard tensor aggregation & resolution |
| **M4: Speculative Decoding** | ✅ | 46 | Draft matrix sweep, adaptive k controller |
| **M5: CUDA Async Prefetch** | ⬜ Skipped | — | Optional stretch goal |
| **M6: Autotuner Integration** | ✅ | 49 | Objective function J(c), candidate scoring |
| **M7: Final Evaluation** | ✅ | 21 | Benchmark runner, comparison & ablation reports |
| **TOTAL** | | **289 passing, 4 skipped** | All tests pass in ~30s |

Test command:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ -v
```

---

## 3. File Inventory

### Source Code (`src/adhie/` — 19 Python files, ~5,800 LoC)

| File | Purpose |
|------|---------|
| `types.py` | All shared dataclasses: `HardwareProfile`, `ModelProfile`, `WorkloadConfig`, `RunPlan`, `MemoryEstimate`, `InferenceMetrics`, `BenchmarkResult` |
| `profiler/hardware_profiler.py` | GPU/CPU/RAM/bandwidth detection via `nvidia-ml-py` (`pynvml`), `psutil`, microbenchmarks |
| `inspector/model_inspector.py` | GGUF metadata extraction, multi-shard aggregation |
| `inspector/model_resolver.py` | HF repo → local GGUF path, offline cache check, shard grouping/downloading at `~/.adhie/models/` |
| `estimator/memory_estimator.py` | Weight/KV/CUDA memory estimation, `find_max_gpu_layers()` binary search |
| `planner/plan_generator.py` | Dynamic max GPU layer enumeration, physical CPU thread sizing, plan ranking |
| `planner/autotuner.py` | Benchmarks top-N candidates, scores with objective function J(c), selects best |
| `planner/profile_cache.py` | YAML cache at `~/.adhie/profiles/` keyed by hw+model+workload fingerprint |
| `runtime/base.py` | `RuntimeAdapter` ABC: load/generate/metrics/unload/health_check |
| `runtime/llamacpp_adapter.py` | `LlamaCppAdapter`: manages `llama-server` subprocess, `--flash-attn auto`, `--no-kv-offload`, SSE streaming telemetry |
| `runtime/adaptation.py` | Wraps `AdaptiveDraftController` for runtime k-adjustment |
| `speculative/controller.py` | Builds experiment matrix (variants A-E), feasibility checks |
| `speculative/draft_sweep.py` | Sweeps k∈{2,4,6,8} for each variant, collects `SpecMetrics` |
| `speculative/adaptive.py` | `AdaptiveDraftController`: rolling-window acceptance → adjust k, disable if <0.95× |
| `telemetry/collector.py` | TTFT/TPOT/ITL/throughput collection, prompt tokens/s, resource sampling |
| `benchmark/runner.py` | `BenchmarkRunner`: warmup + timed runs, streaming telemetry, comparison table with PP tok/s |
| `benchmark/report.py` | `ComparisonRow`, `AblationEntry`, Markdown/Rich tables with PP tok/s, plot data extraction |
| `benchmark/ablation.py` | 5 ablation configs: ADHIE vs default, speculation, KV quant, KV offload, adaptive k |
| `cli/main.py` | 11 Click commands: profile, inspect, estimate, plan, run, benchmark, experiment, compare, ablation, autotune, models |

### Key Scripts & Vendor Files

| File | Purpose |
|------|---------|
| `scripts/demo.ps1` | 7-step PowerShell demo with `Invoke-Adhie` wrapper, venv auto-detection, and clean run comparison |
| `scripts/generate_plots.py` | Matplotlib chart generator (TPOT, throughput, TTFT, VRAM, overview) |
| `vendor/llama.cpp/llama-server.exe` | Pre-built llama.cpp server (v0.9.0-dev build 12244) |
| `vendor/llama.cpp/llama-bench.exe` | Low-level llama.cpp benchmarking tool |
| `vendor/llama.cpp/ggml-cuda.dll` + CUDA runtime DLLs | `cublas64_12.dll`, `cublasLt64_12.dll`, `cudart64_12.dll` |

---

## 4. Key Architectural Decisions

### Data Flow
```
User → CLI (Click) → resolve_model() → local cache check / HF download
  → HardwareProfiler.profile() → HardwareProfile
  → ModelInspector.inspect() → ModelProfile
  → MemoryEstimator.estimate() → MemoryEstimate
  → PlanGenerator.generate_candidates() → list[RunPlan] (ranked by score)
  → Autotuner.autotune() → best RunPlan (benchmarked)
  → LlamaCppAdapter.load(plan) → llama-server subprocess
  → Generate / Benchmark → InferenceMetrics / BenchmarkResult
```

### Model Identifiers & Offline Support
- HF format: `org/repo:quant_hint` (e.g., `Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m`)
- Cached locally at `~/.adhie/models/<org>--<repo>/`
- `resolve_model()` checks local disk first, enabling offline operation in sandboxed environments

### Backend Configuration
- `find_llama_server()` searches `vendor/llama.cpp/llama-server.exe`, `vendor/llama.cpp/build/bin/`, PATH
- CLI flags passed to `llama-server`:
  - `--gpu-layers N`
  - `--ctx-size N`
  - `--threads N` (physical core count)
  - `--no-kv-offload` (when `plan.kv_offload=True`, keeping KV cache in host RAM)
  - `--flash-attn auto`
  - `--cache-type-k` / `--cache-type-v` (when not `f16`)

---

## 5. Important Bugs Fixed

| # | Bug | Root Cause | Fix | File |
|---|-----|------------|-----|------|
| 1 | `setuptools.backends._legacy:_Backend` import | Deprecated backend | Changed build-backend to `setuptools.build_meta` | `pyproject.toml` |
| 2 | YAML tuple serialization | PyYAML cannot dump tuples | Convert `gpu_compute_capability` to list | `types.py` |
| 3 | `/dev/null` missing on Windows | Unix-specific path | `Console(file=io.StringIO())` | tests |
| 4 | `pynvml` deprecation warning | Renamed package | Changed dep to `nvidia-ml-py` | `pyproject.toml`, profiler |
| 5 | Storage detection on Windows | `lsblk` missing | Added PowerShell `Get-PhysicalDisk` branch | `hardware_profiler.py` |
| 6 | Split GGUF shard download incomplete | Shard 1 only | Group and download all split shards | `model_resolver.py` |
| 7 | `datetime.utcnow()` deprecation | Python 3.12+ | `datetime.now(datetime.UTC)` | multiple |
| 8 | `field.data` is index list not values | GGUF v0.19 parser | Complete rewrite to dereference `field.parts[field.data[i]]` | `model_inspector.py` |
| 9 | Duplicate `autotune` functions in `main.py` | Bad string replacement | Removed duplicate blocks | `cli/main.py` |
| 10 | `llama-server not found` | Release zip extracted to `vendor/llama.cpp/` directly | Added direct `vendor/llama.cpp/` to search paths | `llamacpp_adapter.py` |
| 11 | `--flash-attn` argument error | Newer llama.cpp requires value | Changed to `--flash-attn auto` | `llamacpp_adapter.py` |
| 12 | Demo script ignores exit codes | PowerShell `$LASTEXITCODE` not trapped by `$ErrorActionPreference` | Added `Invoke-Adhie` wrapper to abort on non-zero exit | `demo.ps1` |
| 13 | `adhie` not in PATH | Venv not active in new shell | Added `.venv\Scripts` auto-detection | `demo.ps1` |
| 14 | `llama-cli --list-devices` showed `(none)` | Missing CUDA 12.4 runtime companion DLLs | Added `cublas64_12.dll`, `cudart64_12.dll`, `cublasLt64_12.dll` | `vendor/llama.cpp/` |
| 15 | **Coarse split fractions capped GPU offload at 12 layers** | `[1.0, 0.75, 0.50, 0.25, 0.0]` checked only 48, 36, 24, 12, 0. 24 failed with FP16 KV on GPU, so it selected 12 layers (75% on CPU) | Integrated `find_max_gpu_layers()` into candidate generation, unlocking 24 layers (KV in RAM) and 16-20 layers | `plan_generator.py` |
| 16 | **Missing `--no-kv-offload` flag** | In ADHIE `kv_offload=True` means RAM offload, but llama-server defaults to GPU KV offload | Added `--no-kv-offload` flag to `_build_args` when `plan.kv_offload=True` | `llamacpp_adapter.py` |
| 17 | **Thread contention on hybrid CPU (Intel 13th Gen)** | Planner allocated 10 threads (logical cores - 2), causing thread stalls across P-cores and hyperthreads | Sized threads to `hardware.cpu_physical_cores` (8 threads), increasing generation speed by ~19% | `plan_generator.py` |
| 18 | **Missing streaming usage & prompt telemetry** | Missing `"stream_options": {"include_usage": True}` and parser skipped `choices: []` chunk | Added option and parsed `usage` chunk before checking `choices`, restoring prompt tokens/sec | `llamacpp_adapter.py` |
| 19 | **Comparison table omitted prompt throughput** | Only showed output `tok/s` (TG), hiding 35%-66% prompt prefill acceleration | Added `prompt_tokens_per_s` and `PP tok/s` column to Rich and Markdown tables | `report.py`, `runner.py` |
| 20 | **Model resolver crashed offline** | Queried `HfApi.list_repo_tree()` before checking local disk cache | Added local directory check before network requests | `model_resolver.py` |
| 21 | **Demo script compared old historical runs** | Gathered all existing JSONs in folder | Cleaned prior demo `run-*.json` files at script start | `demo.ps1` |

---

## 6. CLI Commands (11 total)

```bash
adhie profile                       # Profile local hardware
adhie inspect MODEL                 # Inspect model metadata
adhie estimate MODEL                # Estimate memory requirements
adhie plan MODEL                    # Generate execution plan candidates
adhie run MODEL                     # Profile → Plan → Launch interactive inference
adhie benchmark MODEL               # Run benchmark suite (--mode baseline|auto|sweep, --prompt)
adhie experiment MODEL              # Speculative decoding experiments (variants A-D × k)
adhie compare FILE [FILE ...]       # Compare benchmark results side-by-side
adhie ablation MODEL                # Run ablation studies
adhie autotune MODEL                # Benchmark top-N plans, select best
adhie models                        # List cached models
```

---

## 7. Target Machine Verification & Performance Summary

### Verified Steps (demo.ps1)
All 7 steps run end-to-end successfully:
1. `adhie profile` → RTX 3050 6GB Laptop GPU, 25.5 GB RAM, PCIe Gen1 ×8
2. `adhie inspect` → Qwen2.5-14B-Instruct Q4_K_M (14.77B params, 48 layers)
3. `adhie plan` → 10 feasible candidates, recommended: **24/48 GPU layers** (4.5 GB VRAM)
4. `adhie estimate` → Feasibility confirmed with 120 MB VRAM headroom beyond 512 MB safety margin
5. `adhie benchmark --mode baseline` → CPU-only run (0 GPU layers)
6. `adhie benchmark --mode auto` → ADHIE-optimized hybrid run (24 GPU layers, 8 threads, `--no-kv-offload`)
7. `adhie compare` → Side-by-side comparison table

### Measured Hardware Performance (Qwen2.5-14B-Instruct Q4_K_M)
* **Isolated `llama-bench.exe` ground truth:**
  - `ngl 0` (CPU only, 8 threads): Prompt prefill: **80.3 tok/s**, Token gen: **4.75 tok/s**
  - `ngl 16` (16 GPU layers, 8 threads): Prompt prefill: **109.0 tok/s** (+36%), Token gen: **5.25 tok/s** (+10.5%)
  - `ngl 20` (20 GPU layers, 8 threads): Prompt prefill: **133.2 tok/s** (+66%), Token gen: **5.4+ tok/s**
* **ADHIE live benchmark (`adhie compare`):**
  - CPU baseline: TPOT = **213.6 ms** (4.6 tok/s), VRAM = 1.03 GB
  - Hybrid (24 GPU layers): TPOT = **202.2 ms** (4.9 tok/s), VRAM = 5.46 GB, PP = **424 tok/s**

---

## 8. Immediate Next Steps

1. **Run full demo script:**
   ```powershell
   .\scripts\demo.ps1 -Model "Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m"
   ```
2. **Run speculative decoding demo (Target + Draft):**
   ```powershell
   .\scripts\demo.ps1 -Model "Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m" -Draft "Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m"
   ```
3. **Generate final plots:**
   ```powershell
   python scripts\generate_plots.py --input benchmarks\results\demo
   ```
4. **Fill in Final Evaluation Report:**
   Populate `docs/report_template.md` with final benchmark data and generated plots.
