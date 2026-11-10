# ADHIE — Adaptive Heterogeneous Inference Engine

**Make useful 10B+ parameter models practical on hardware that cannot hold the entire model in GPU memory.**

ADHIE is a hardware-aware local inference planner and runtime controller. It profiles your machine, inspects the model, estimates memory requirements, and automatically selects a feasible execution plan across GPU VRAM, CPU compute, system RAM, quantization, KV-cache placement, and speculative decoding.

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Profile your hardware
adhie profile

# Inspect a model
adhie inspect /path/to/model.gguf

# Generate execution plan
adhie plan /path/to/model.gguf --context 8192

# Run with automatic configuration
adhie run /path/to/model.gguf
```

## Architecture

```
USER / LOCAL APP
       |
   CLI + API
       |
  ADHIE Controller
       |
  ┌────┼────────────────┐
  │    │                 │
Profiler  Inspector  Workload
  │    │                 │
  └────┼────────────────┘
       |
  Memory Estimator
       |
  Plan Generator → Autotuner
       |
  Runtime Adapter (llama.cpp)
       |
  CPU + GPU + Host Memory
```

## Project Structure

```
adhie/
├── src/adhie/          # Core engine
│   ├── cli/            # Command-line interface
│   ├── profiler/       # Hardware detection & benchmarking
│   ├── inspector/      # Model metadata extraction
│   ├── estimator/      # Memory & cost estimation
│   ├── planner/        # Plan generation & autotuning
│   ├── runtime/        # Backend adapters (llama.cpp)
│   ├── speculative/    # Speculative decoding controller
│   └── telemetry/      # Metrics collection
├── configs/            # Hardware/model/workload configs
├── benchmarks/         # Benchmark scripts & results
├── cuda_experiments/   # CUDA async prefetch PoC
├── tests/              # Unit, integration, correctness tests
├── docs/               # Documentation & decision records
└── scripts/            # Setup & utility scripts
```

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (CPU fallback available)
- llama.cpp built with CUDA support
- Sufficient RAM to hold offloaded model weights

## License

MIT

