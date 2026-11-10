# ADHIE — Final Evaluation Report

> **Adaptive Heterogeneous Inference Engine for Local LLMs**
>
> *[Author Name]*
> *[Date]*

---

## Abstract

This report presents a comprehensive evaluation of ADHIE (Adaptive
Heterogeneous Inference Engine), a system that optimises local LLM inference
on heterogeneous CPU + GPU hardware.  ADHIE automatically profiles available
hardware, inspects model metadata, estimates memory requirements, and
generates an execution plan that maximises throughput within memory
constraints.  We evaluate the engine on a consumer-grade NVIDIA RTX 3050
(6 GB) laptop, demonstrating measurable improvements in time-per-output-token
(TPOT) and generation throughput over naïve CPU-only and default
configurations.

---

## 1. Introduction

### 1.1 Problem Statement

<!-- Describe the challenge of running 10B+ models on consumer hardware with
     limited VRAM.  Motivate the need for intelligent layer placement,
     KV cache management, and speculative decoding. -->

### 1.2 Contributions

1. A hardware profiler that detects CPU, GPU, RAM, and bandwidth.
2. A GGUF model inspector that reads architecture metadata.
3. A memory estimator and plan generator that explores the configuration space.
4. An autotuner that benchmarks top-N candidates and selects the best.
5. Speculative decoding with adaptive draft length.
6. Ablation studies isolating each feature's contribution.

---

## 2. Background & Related Work

### 2.1 llama.cpp & GGUF

<!-- Overview of the backend, quantisation formats, GPU offloading. -->

### 2.2 Speculative Decoding

<!-- Leviathan et al., Chen et al., draft/target paradigm. -->

### 2.3 KV Cache Quantisation

<!-- F16 vs Q8_0 trade-offs, memory savings, perplexity impact. -->

---

## 3. System Architecture

<!-- Include the architecture diagram from docs/architecture.md.
     Describe the pipeline: Profile → Inspect → Estimate → Plan → Run. -->

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

---

## 4. Experimental Setup

### 4.1 Hardware

| Component | Specification           |
|-----------|-------------------------|
| CPU       | Intel 13th Gen (8P/12L) |
| GPU       | NVIDIA RTX 3050 6 GB    |
| RAM       | 25.5 GB (11.3 GB avail) |
| Bus       | PCIe Gen1 ×8            |

### 4.2 Models

| Model                     | Parameters | Quantisation | File Size |
|---------------------------|------------|-------------|-----------|
| <!-- target model -->     |            |             |           |
| <!-- draft model -->      |            |             |           |

### 4.3 Benchmark Matrix

| Dimension      | Values                               |
|----------------|--------------------------------------|
| Context length | 2 048, 8 192, 16 384                 |
| Output tokens  | 128, 256                             |
| GPU layers     | 0 %, 50 %, ADHIE-selected, 100 %     |
| KV type        | F16, Q8_0                            |
| KV offload     | On, Off                              |
| Speculation    | None, CPU-draft k=4, Adaptive        |
| Runs per cfg   | 3 (minimum)                          |

### 4.4 Metrics

- **TTFT** — Time to first token (ms)
- **TPOT** — Time per output token (ms)
- **Throughput** — Output tokens per second (tok/s)
- **Peak VRAM** — Maximum GPU memory used (GB)
- **Peak RAM** — Maximum system memory used (GB)
- **ITL P95** — 95th-percentile inter-token latency (ms)

---

## 5. Results

### 5.1 Baseline vs ADHIE-Selected

<!-- Paste the output of `adhie compare` here.
     Include the comparison table and the plots from generate_plots.py. -->

| Configuration  | TPOT (ms) | tok/s | TTFT (ms) | VRAM (GB) | GPU Layers |
|----------------|-----------|-------|-----------|-----------|------------|
| CPU-only       |           |       |           |           |            |
| ADHIE-selected |           |       |           |           |            |
| Δ              |           |       |           |           |            |

<!-- ![TPOT Comparison](benchmarks/plots/tpot_comparison.png) -->
<!-- ![Throughput Comparison](benchmarks/plots/throughput_comparison.png) -->

### 5.2 Context Length Scaling

<!-- How does performance change from 2K → 8K → 16K context? -->

### 5.3 KV Cache Quantisation Impact

<!-- Q8_0 vs F16: memory saved, TPOT change, throughput change. -->

### 5.4 GPU Layer Placement Curve

<!-- 0 → 50% → ADHIE → 100% layers on GPU: throughput vs VRAM. -->

### 5.5 Speculative Decoding

<!-- Speculation ON vs OFF.  Acceptance rate.  Adaptive k vs fixed k=4. -->

---

## 6. Ablation Studies

Run the ablation suite with:
```bash
adhie ablation MODEL --draft DRAFT --runs 3
```

### 6.1 Summary Table

<!-- Paste the output of `adhie ablation` here. -->

| Feature          | Baseline  | Variant    | Δ TPOT  | Δ tok/s | Verdict   |
|------------------|-----------|------------|---------|---------|-----------|
| ADHIE vs Default |           |            |         |         |           |
| Speculation      |           |            |         |         |           |
| KV Quantisation  |           |            |         |         |           |
| KV Offload       |           |            |         |         |           |
| Adaptive k       |           |            |         |         |           |

### 6.2 Analysis

<!-- Which features help?  Which are neutral or harmful?  Under what conditions? -->

<!-- ![Ablation Plot](benchmarks/plots/ablation_tpot.png) -->

---

## 7. Plan Explanation & Rationale

<!-- Show the output of `adhie plan MODEL` for the best configuration.
     ADHIE must explain WHY each plan was selected (rationale field). -->

### 7.1 Why ADHIE Chose This Configuration

```yaml
# Paste plan YAML here
```

### 7.2 Feature Interaction

<!-- How does the interplay between GPU layers, KV type, and speculation
     affect the final plan?  When does ADHIE disable a feature? -->

---

## 8. Correctness Verification

### 8.1 Greedy Output Equivalence

<!-- Baseline vs speculative greedy output must be identical (same seed). -->

### 8.2 KV Quantisation Quality

<!-- Q8 KV produces valid, coherent output — factual check. -->

### 8.3 Layer Placement Consistency

<!-- 0 GPU layers vs partial GPU layers give the same greedy output. -->

To run correctness tests:
```bash
ADHIE_CORRECTNESS_TESTS=1 pytest tests/correctness/ -v
```

---

## 9. Discussion

### 9.1 Key Findings

<!-- Summarise the most important takeaways. -->

1. **Finding 1:** ...
2. **Finding 2:** ...
3. **Finding 3:** ...

### 9.2 Limitations

- PCIe Gen1 ×8 bus bandwidth limits GPU offloading benefit.
- Single GPU only — no multi-GPU support.
- GGUF format only — no safetensors/AWQ/GPTQ.
- llama.cpp backend only — no vLLM/exllamav2.

### 9.3 When ADHIE Does NOT Help

<!-- Document cases where optimisation provides no benefit or is harmful.
     This is a spec requirement: system must identify when optimisation
     is NOT useful and disable it. -->

### 9.4 Future Work

- CUDA async prefetch (M5, skipped in this iteration).
- Multi-GPU support.
- Adaptive context windowing.
- Continuous profiling and online re-planning.

---

## 10. Reproducibility Checklist

- [ ] Same model + hardware tested under baseline AND ADHIE-selected config
- [ ] ADHIE explains why each plan was selected (rationale field)
- [ ] 10B+ model runs on memory-constrained machine (≤6 GB VRAM)
- [ ] At least 1 adaptive policy shows repeatable improvement
- [ ] System identifies when optimisation is NOT useful and disables it
- [ ] All claims backed by ≥3 runs
- [ ] Correctness tests pass
- [ ] Hardware profile matches expected specs

---

## Appendix A — Full Benchmark Data

<!-- Link to or paste full JSON results. -->

```
benchmarks/results/demo/
├── hardware_profile.yaml
├── model_profile.yaml
├── plans.yaml
├── *.json               ← individual run results
├── comparison_report.md
└── ablation/
    ├── ablation_report.md
    └── *.json
```

## Appendix B — How to Reproduce

```bash
# 1. Install
pip install -e .

# 2. Setup backend (llama.cpp)
bash scripts/setup_backend.sh

# 3. Run the full demo
bash scripts/demo.sh Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m

# 4. Generate plots
pip install matplotlib
python scripts/generate_plots.py --input benchmarks/results/demo

# 5. Run correctness tests
ADHIE_CORRECTNESS_TESTS=1 pytest tests/correctness/ -v
```

## Appendix C — System Versions

| Component   | Version |
|-------------|---------|
| Python      |         |
| llama.cpp   |         |
| CUDA        |         |
| ADHIE       |         |
| OS          |         |

