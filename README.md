# ADHIE — Adaptive Heterogeneous Inference Engine

**Make useful 10B+ parameter models practical on hardware that cannot hold the entire model in GPU memory.**

ADHIE is a hardware-aware local inference planner and runtime controller. It profiles your machine, inspects the model, estimates memory requirements, and automatically selects a feasible execution plan across GPU VRAM, CPU compute, system RAM, quantization, KV-cache placement, and speculative decoding.

## Quick Start

```bash
# 1. Install ADHIE
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -e ".[dev]"

# 2. Profile your hardware
adhie profile

# 3. Inspect a model (downloads from HuggingFace automatically)
adhie inspect Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m

# 4. Generate execution plan
adhie plan Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m --context 8192

# 5. Estimate memory usage
adhie estimate Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m

# 6. Run with automatic configuration (requires llama-server — see below)
adhie run Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m
```

## Setting Up the Backend (llama-server)

ADHIE uses [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` as its inference backend. You need to install it before running `adhie run`, `adhie benchmark`, or `adhie experiment`.

### Option A: Download Pre-built Release (Recommended)

1. Go to [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)
2. Download the release matching your platform:
   - **Windows + NVIDIA GPU**: `llama-*-bin-win-cuda-cu*.*.zip`
   - **Linux + NVIDIA GPU**: `llama-*-bin-ubuntu-x64-cuda-cu*.*.tar.gz`
3. Extract and add the `bin/` folder to your system `PATH`
4. Verify: `llama-server --help`

### Option B: Build from Source

```bash
# Linux / macOS
./scripts/setup_backend.sh

# Windows PowerShell
.\scripts\setup_backend.ps1
```

### Option C: Pass the Path Directly

```bash
adhie benchmark MODEL --backend-path "C:\path\to\llama-server.exe"
adhie run MODEL --backend-path /path/to/llama-server
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `adhie profile` | Profile local hardware (GPU, CPU, RAM, bandwidths) |
| `adhie inspect MODEL` | Inspect model metadata from GGUF / HuggingFace |
| `adhie estimate MODEL` | Estimate memory requirements for a configuration |
| `adhie plan MODEL` | Generate and rank execution plan candidates |
| `adhie run MODEL` | Profile → Plan → Launch interactive inference |
| `adhie benchmark MODEL` | Run benchmark suite and save results |
| `adhie experiment MODEL` | Run speculative decoding experiments |
| `adhie compare FILE...` | Compare benchmark results side-by-side |
| `adhie ablation MODEL` | Run ablation studies isolating feature contributions |
| `adhie models` | List locally cached models |

**MODEL** can be:
- A local GGUF path: `/path/to/model.gguf`
- A HuggingFace repo: `Qwen/Qwen2.5-14B-Instruct-GGUF`
- A repo with quant hint: `Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m`

## Running the Full Demo

```bash
# Linux / macOS
./scripts/demo.sh Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m

# Windows PowerShell
.\scripts\demo.ps1 -Model "Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m"

# With speculative decoding (draft model)
.\scripts\demo.ps1 -Model "Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m" -Draft "Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m"
```

## Running the Benchmark Matrix

```bash
# Quick benchmark (best plan vs CPU-only)
adhie benchmark MODEL --mode sweep --runs 3

# Full matrix sweep (all contexts × KV types × GPU splits × speculation)
python benchmarks/run_benchmark.py MODEL --matrix --runs 3

# Dry run to see all configurations
python benchmarks/run_benchmark.py MODEL --matrix --dry-run
```

## Generating Plots

```bash
pip install matplotlib
python scripts/generate_plots.py --input benchmarks/results/demo
```

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Hardware    │──▶│    Model     │──▶│   Memory     │
│  Profiler    │   │  Inspector   │   │  Estimator   │
└──────────────┘   └──────────────┘   └──────┬───────┘
                                             │
                                      ┌──────▼───────┐
                                      │    Plan      │
                                      │  Generator   │
                                      └──────┬───────┘
                                             │
                                      ┌──────▼───────┐
                                      │  Autotuner   │
                                      │  (Top-N)     │
                                      └──────┬───────┘
                                             │
                                      ┌──────▼───────┐
                                      │   Runtime    │
                                      │  (llama.cpp) │
                                      └──────────────┘
```

## Project Structure

```
adhie/
├── src/adhie/              # Core engine (6,200+ LoC)
│   ├── cli/                # Command-line interface (10 commands)
│   ├── profiler/           # Hardware detection & benchmarking
│   ├── inspector/          # Model metadata extraction + HF resolver
│   ├── estimator/          # Memory & cost estimation
│   ├── planner/            # Plan generation, autotuning, profile cache
│   ├── runtime/            # Backend adapters (llama.cpp)
│   ├── speculative/        # Speculative decoding controller
│   ├── benchmark/          # Benchmark runner, report, ablation
│   ├── telemetry/          # Metrics collection
│   └── types.py            # Shared dataclasses (400+ lines)
├── tests/                  # 289 tests (4,200+ LoC)
│   ├── unit/               # 16 unit test modules
│   └── correctness/        # Output equivalence tests
├── scripts/                # Setup, demo, and plotting scripts
├── benchmarks/             # Full benchmark matrix runner
├── configs/                # Hardware/model/workload configs
└── docs/                   # Spec, architecture, decisions, report template
```

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (CPU fallback available)
- llama.cpp `llama-server` binary (for inference/benchmarking)
- Sufficient RAM to hold offloaded model weights

## Development

```bash
# Run all tests
python -m pytest tests/ -v

# Run correctness tests (requires llama-server + models)
ADHIE_CORRECTNESS_TESTS=1 pytest tests/correctness/ -v

# Lint
ruff check src/ tests/
```

## License

MIT
