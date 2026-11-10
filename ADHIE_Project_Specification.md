# ADHIE — Adaptive Heterogeneous Inference Engine for Local LLMs

**Project specification, architecture proposal, research plan, and implementation roadmap**  
**Team:** 3 B.Tech students  
**Available time:** At most 6 months; developed alongside placements and college work  
**Primary deployment target:** Single-user or very-low-concurrency local inference on consumer hardware  
**Minimum hardware class to investigate:** NVIDIA GPU with approximately 4 GB VRAM, plus available system RAM; support for stronger consumer systems should scale automatically  
**Primary ambition:** Make useful 10B+ parameter models practical on hardware that cannot hold the entire model in GPU memory

> **Working thesis:** Build a local inference runtime that profiles the machine, understands the model and workload, and automatically selects a feasible execution plan across GPU VRAM, CPU compute, system RAM, quantization, KV-cache placement, and speculative decoding. The engine should optimize for interactive single-user latency and useful model capacity—not cloud-style high-concurrency throughput.

---

## 1. Executive summary

Running a modern language model locally is not simply a question of whether its weights fit in GPU VRAM. The system must also accommodate the KV cache, temporary activations, CUDA/runtime allocations, and the operating system. On a consumer machine with a small GPU, a model may need to be quantized and split across GPU and host memory. That can make it load successfully, but it does not automatically make token generation fast: moving weights or KV data across PCIe, executing layers on a CPU, and repeatedly reading large weight tensors can become the dominant costs.

ADHIE (Adaptive Distributed/Heterogeneous Inference Engine; the implementation may use the simpler name **Adaptive Heterogeneous Inference Engine**) will make those decisions explicit and measurable. Its first version will target one machine and one active request. It will:

1. **Profile hardware** — GPU model, VRAM, CPU cores, RAM, available memory, storage, and measured transfer/compute characteristics.
2. **Inspect the model** — parameter count, architecture, layer count, attention/KV dimensions, quantization, tokenizer, and context requirements.
3. **Estimate feasibility** — model-weight footprint, KV-cache footprint, runtime reserve, and safe VRAM/RAM budgets.
4. **Choose an execution plan** — GPU-resident or hybrid CPU/GPU layer placement, weight precision, KV-cache precision and placement, and a decoding strategy.
5. **Run inference through a proven backend** — initially use an established backend such as `llama.cpp` rather than attempting to recreate every transformer kernel.
6. **Experiment with CPU-draft/GPU-target speculative decoding** — place the draft model on the CPU and preserve as much GPU capacity as possible for the target, then compare it with no speculation and alternative placements.
7. **Prototype asynchronous layer prefetch** — use a dedicated non-default CUDA transfer stream to stage upcoming weights while the target computes the current layer, but only integrate this if profiling proves it helps.
8. **Adapt safely** — adjust draft length, KV policy, and other low-risk runtime choices from measured behavior; avoid expensive model-weight migrations in the token-generation hot path.
9. **Benchmark rigorously** — TTFT, TPOT, inter-token latency, output tokens/s, memory use, CPU/GPU utilization, transfer volume, and quality/accuracy trade-offs.

The project is deliberately **not** a full clone of vLLM, TensorRT-LLM, or llama.cpp. Those systems already implement much of the low-level inference machinery. ADHIE's defensible contribution is the **hardware-aware planner, policy selection, and measured coordination of heterogeneous resources**, with one or two carefully scoped custom mechanisms.

---

## 2. Problem statement

### 2.1 Problem

Consumer GPUs often have much less VRAM than is required to hold a useful 10B+ parameter model at high precision. Even with weight quantization, the full model may not fit. Long prompts further increase KV-cache memory, and CPU offloading can introduce substantial latency. The user is forced to understand many low-level settings—quantization, GPU layer count, CPU placement, KV-cache type, context length, and speculative decoding—and manually tune them for each machine and model.

Most users need a straightforward experience:

```bash
adhie run <model>
```

The system should inspect the local machine, determine what is feasible, select a reasonable execution plan, and explain the expected trade-offs. It should make the largest useful model/context combination practical without claiming that every model can run quickly on every device.

### 2.2 Core research/engineering question

**Given a target model, a consumer hardware configuration, and an interactive workload, can an adaptive planner select a CPU/GPU/memory/decoding configuration that improves usable model capacity and interactive latency over static or manually selected configurations?**

### 2.3 What “usable” means

A model merely loading is not success. A model is considered usable only when it meets a stated quality floor and a measurable interaction target. For example:

- the model loads without out-of-memory failures;
- the chosen context length is supported;
- the output quality is not silently degraded by an unsuitable quantization/cache setting;
- TTFT and TPOT are reported;
- generation is responsive enough for the chosen task and hardware;
- memory use remains within a configurable safety margin.

Do not promise a universal minimum token rate. Define “usable” per hardware class and workload, and publish measured results.

---

## 3. Constraints and project assumptions

These constraints determine the architecture and scope.

| Constraint | Design implication |
|---|---|
| Maximum duration is about six months | Build a narrow, working end-to-end system first; extensions are conditional. |
| Work happens alongside college and placement preparation | Use existing inference backends; avoid writing an entire serving engine from scratch. |
| Three team members | Give each member a distinct subsystem, but integrate against shared interfaces from the beginning. |
| Local inference is the primary use case | Optimize for one active user / roughly 1–4 active sequences, not high-QPS server throughput. |
| Consumer devices vary widely | Detect and profile capabilities instead of hard-coding a particular GPU. |
| Minimum GPU class to investigate is around 4 GB VRAM | Treat host RAM as an important resource; 4 GB VRAM is not enough to hold most 10B+ models in full. |
| Main objective is 10B+ models | Optimize for quantized weights and partial GPU residency; measure performance, not just successful loading. |
| NVIDIA/CUDA kernel work is desired | Keep a CUDA/Triton experimental path, but do not make the entire project unusable without CUDA. |
| Distributed nodes are optional | First support heterogeneous resources inside one PC; multi-node inference is a stretch goal. |

### 3.1 Reality check: 10B+ parameters on a 4 GB GPU

A rough lower-bound estimate for raw weight storage is:

\[
M_{weights} \approx N_{params} \times \frac{b}{8}
\]

where `b` is the average number of bits per parameter. For a 10B-parameter model:

| Weight representation | Raw arithmetic estimate for 10B parameters | Practical interpretation |
|---|---:|---|
| FP16/BF16 (16 bits) | ~20 GB decimal | Far beyond 4 GB VRAM. |
| INT8 (8 bits) | ~10 GB decimal | Still beyond 4 GB VRAM. |
| 4-bit (4 bits) | ~5 GB decimal before scales/metadata/alignment | Usually still larger than 4 GB VRAM, and runtime/KV memory is additional. |

Real quantized formats have metadata, scales, block structure, alignment, and implementation-specific overhead. In addition, VRAM must hold some KV cache, temporary buffers, CUDA context allocations, and possibly display allocations. Therefore, **a 10B+ model on a 4 GB GPU generally requires some combination of host-RAM residency, CPU execution, and/or partial GPU offload**.

That can make the model technically runnable, but speed depends heavily on CPU performance, RAM bandwidth, PCIe bandwidth, GPU memory bandwidth, quantization kernels, and the fraction of work performed on each device. A 4 GB GPU is a minimum hardware class for experimentation—not a promise of fast 10B+ inference.

For MoE models, distinguish **total parameters** from **active parameters per token**. Only a subset of experts may be active for a token, but the full expert weights may still need to be stored somewhere. Memory and compute behavior therefore differ from dense models.

### 3.2 KV-cache size estimate

For a conventional decoder-only transformer, an approximate per-sequence KV-cache size is:

\[
M_{KV} \approx 2 \times L \times T \times H_{KV} \times D_{head} \times B_{elem}
\]

where:

- `2` accounts for keys and values;
- `L` is the number of transformer layers;
- `T` is cached sequence length in tokens;
- `H_KV` is the number of KV heads (not necessarily the number of query heads, especially with GQA/MQA);
- `D_head` is the head dimension;
- `B_elem` is bytes per stored element.

This is an estimate, not an exact allocator forecast. Sliding-window attention, latent attention, hybrid attention layers, quantization metadata, block rounding, and model-specific cache layouts change the result. The model inspector should use architecture-specific fields whenever possible.

### 3.3 The objective is not “run any 10B model at any cost”

The objective is to **maximize useful local inference under a hardware budget**. The planner must be allowed to say:

- this configuration is feasible and likely responsive;
- this configuration will load but be slow because host/device transfer dominates;
- this context length is not safe with the current KV format;
- this model is not supported by the current backend;
- a smaller model or more aggressive quantization is the better option.

Honest feasibility estimates are a feature, not a failure.

---

## 4. Goals, non-goals, and definition of success

### 4.1 Must-have goals

1. Hardware discovery and a reproducible hardware profile.
2. Model metadata inspection and memory estimation.
3. A working local inference path on at least one supported backend.
4. GPU/CPU placement or offload configuration based on available resources.
5. KV-cache precision/offload configuration and memory accounting.
6. A CPU-draft/GPU-target speculative decoding experiment with a no-speculation baseline.
7. A simple autotuner that tests a small candidate set and recommends a configuration.
8. A benchmark suite that can reproduce results.
9. A CLI and a useful report/dashboard showing what the planner selected and why.

### 4.2 Strongly preferred goals

- adaptive draft length based on observed drafting latency and acceptance rate;
- a custom CUDA prototype for asynchronous layer-weight prefetch;
- safe runtime adaptation for selected policies;
- comparison against at least one established local inference engine on identical hardware/model settings.

### 4.3 Non-goals for the six-month core

- cloud-scale continuous batching or high-concurrency serving;
- a complete replacement for llama.cpp/vLLM/TensorRT-LLM;
- implementing tensor parallelism, pipeline parallelism, MoE expert parallelism, and RDMA simultaneously;
- a general distributed cluster scheduler;
- SSD as an active per-token memory tier;
- training a new foundation model;
- guaranteeing fast inference on every 4 GB GPU;
- adding reinforcement learning before a measured heuristic baseline exists.

### 4.4 Success criteria

The project is successful if it can demonstrate all of the following:

- the same model and hardware can be tested under a documented baseline and ADHIE-selected configuration;
- ADHIE can explain why its plan was selected (memory feasibility, expected bottleneck, selected cache/quantization/offload settings);
- it can run a chosen 10B+ quantized model on at least one memory-constrained test machine, subject to available system RAM and backend compatibility;
- at least one adaptive policy produces a repeatable improvement in a clearly defined metric on at least one hardware/workload regime, without violating the quality constraint;
- the system can identify regimes where an optimization is not useful and disable it;
- all claims are backed by repeatable measurements rather than a single demo run.

---

## 5. Positioning against existing systems and prior work

The project must explicitly acknowledge the existing systems. Reimplementing a known feature without a new policy or a meaningful evaluation is not a sufficient contribution.

| System / work | What it already establishes | How ADHIE should use it |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Local quantized inference, CPU/GPU layer offload, GPU layer count, CPU placement options for dense FFN/MoE weights, KV-cache types, draft-model options, and RPC support. | Primary practical baseline and likely first runtime backend. Use its documented options before writing replacements. |
| [KTransformers](https://www.ktransformers.net/en/docs/technical-work/heterogeneous-inference) | Heterogeneous CPU/GPU inference, including mapping parts of large MoE models to suitable devices and long-context strategies. | Reference for CPU/GPU placement and MoE-oriented extensions. Do not assume its performance transfers to every dense model or PC. |
| [Hugging Face Transformers KV-cache strategies](https://huggingface.co/docs/transformers/main/kv_cache) | Dynamic/static/quantized/offloaded cache strategies; offloading can asynchronously prefetch layer cache but can reduce throughput due to data movement. | Reference for cache-policy behavior and quality/performance trade-offs. |
| [vLLM speculative decoding documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/) | Speculative decoding is especially relevant to memory-bound, low-to-medium-QPS latency workloads; offers draft models, n-gram, EAGLE/MTP and adaptive variants. | Reference for methods and measurement. ADHIE is low-concurrency, so latency is a relevant objective. |
| [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180) | Block-based KV memory management reduces fragmentation and enables flexible cache sharing in multi-request serving. | Reference for page/block design. ADHIE's single-user MVP can use a smaller allocator and defer multi-request sharing. |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware exact attention reduces data movement between GPU HBM and on-chip SRAM. | A kernel-level reference; do not reimplement attention unless profiling identifies it as a bottleneck. |
| [KVQuant](https://arxiv.org/abs/2401.18079) | KV quantization can substantially reduce cache memory, with method-specific quality/performance trade-offs. | Research reference for possible low-bit KV work; use backend-supported quantized cache first. |
| [Dovetail, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.879/) | Studies heterogeneous CPU/GPU speculative decoding on consumer hardware. Its design puts the **draft model on GPU and target model on CPU**, the reverse of ADHIE's initial CPU-draft/GPU-target hypothesis. | Important prior art and a reason to benchmark device placement rather than assume one direction is universally optimal. |

**Differentiation:** ADHIE's intended contribution is not “we support quantization/offloading/speculative decoding.” Existing engines do. The contribution should be the **hardware- and workload-aware policy that selects and coordinates those existing mechanisms**, with an empirical study of CPU-draft/GPU-target placement and asynchronous transfer overlap.

---

## 6. Proposed system architecture

### 6.1 High-level architecture

```text
                         USER / LOCAL APP
                                |
                    CLI + Optional Local API
                                |
                       Request / Model Spec
                                |
                +---------------v----------------+
                |       ADHIE Controller          |
                |  request profile + policy state |
                +---------------+----------------+
                                |
              +-----------------+-------------------+
              |                 |                   |
      Hardware Profiler   Model Inspector     Workload Analyzer
      GPU / CPU / RAM     params / layers     context / output
      PCIe / disk         KV dimensions      latency preference
              |                 |                   |
              +-----------------+-------------------+
                                |
                      Memory / Cost Estimator
                                |
                         Plan Generator
                                |
                       Candidate Evaluator
                                |
                       Selected Run Plan
                                |
       +------------------------+-----------------------------+
       |                        |                             |
  Runtime Adapter          KV Manager                 Decode Controller
  llama.cpp first           placement / format         CPU draft / GPU target
  later custom backend      offload / cache             adaptive draft length
       |                        |                             |
       +------------------------+-----------------------------+
                                |
                    Execution Backend / Runtime
                     CPU + GPU + host memory
                                |
               +----------------+----------------+
               |                |                |
             VRAM              RAM              SSD
         model/KV/hot       weights/KV/cold    model storage
         workspaces         host staging       cold spill only
                                |
                      Metrics / Telemetry
                                |
                  CLI report / Grafana (optional)
```

### 6.2 Main subsystems

#### A. Hardware Profiler

Collect static and dynamic properties:

- GPU name, compute capability, total/free VRAM, device count;
- CPU model, physical/logical cores, SIMD/AMX availability where relevant;
- total/free system RAM;
- NUMA topology where available;
- PCIe generation/width where discoverable, with measured H2D/D2H bandwidth preferred over trusting reported link data;
- storage type and read throughput (mostly for model loading, not token generation);
- measured CPU memory bandwidth and a small representative matrix/vector benchmark;
- GPU memory bandwidth/compute characteristics via small warm-up microbenchmarks;
- CUDA copy/compute overlap capability where queryable;
- current memory pressure and reserved safety margin.

Do not make hardware discovery depend on one GPU vendor. The first optimized implementation may be NVIDIA/CUDA-centric, but the planner should have a CPU fallback and backend capability flags.

#### B. Model Inspector

Read or require a model manifest containing:

- architecture and model family;
- total parameters and, for MoE, active parameters per token and expert count;
- layer count, hidden size, intermediate size;
- query-head count, KV-head count, head dimension;
- vocab size and tokenizer identity;
- model context limit and attention pattern (full, sliding window, hybrid, etc.);
- weight dtype/quantization and file size;
- whether a compatible draft model is available;
- chat template and model-specific runtime requirements.

The model inspector must distinguish what can be read from metadata from what needs to be benchmarked or supplied by the user.

#### C. Memory and Cost Estimator

Estimate:

```text
VRAM budget = total VRAM
             - operating-system/display usage
             - CUDA/runtime reserve
             - execution workspaces
             - safety margin

GPU model budget = VRAM budget - GPU KV budget

Host RAM budget = free RAM
                  - operating-system reserve
                  - other applications
                  - pinned staging buffers
                  - CPU runtime reserve
```

The estimator should generate a range, not a fake exact number. Validate estimates against actual peak memory during warm-up and update the profile.

#### D. Plan Generator / Autotuner

Generate a small set of feasible candidates, not every possible combination. Candidate settings can include:

- backend and model format;
- weight quantization variant (select among existing model files/formats rather than quantizing weights online in v1);
- number/fraction of GPU-resident layers;
- CPU thread count and affinity;
- KV-cache precision;
- KV-cache device/offload mode;
- context length and batch/ubatch values supported by backend;
- draft model location (CPU/GPU) and draft length;
- optional async weight-prefetch setting (experimental backend only).

Reject candidates that violate memory safety, unsupported architecture, context compatibility, or quality constraints.

#### E. Runtime Adapter

Use an interface that separates ADHIE's planner from the inference engine. The first adapter should wrap a mature backend. Do not write a model parser, tokenizer, quantized GEMM library, attention implementation, and sampler from scratch during the MVP.

Potential initial backend: **llama.cpp** with GGUF quantized models and CUDA support. It already exposes useful controls for GPU layer residency, CPU FFN/MoE placement, K/V cache types, and draft model placement/CPU thread affinity. Validate the actual options against the exact pinned build because command-line options can change.

A later experimental backend can use PyTorch/C++/CUDA for a narrow layer-streaming proof-of-concept. The main product should not depend on that custom backend unless it outperforms the stable path.

---

## 7. Runtime execution modes

ADHIE should choose among a small number of understandable modes.

### Mode 1 — GPU-resident target

```text
GPU: target weights + KV + execution buffers
CPU: tokenizer, controller, optional CPU draft
```

Use when the model and chosen context fit safely in VRAM. This is normally the best starting point for latency.

### Mode 2 — Hybrid layer placement

```text
GPU: selected target layers + GPU KV/workspace
RAM: offloaded target layers and/or host KV
CPU: compute layers assigned to CPU
```

Use when the full model cannot fit in VRAM. Test several layer splits; more GPU layers are not automatically better if the GPU has to exchange data with the CPU frequently or runs out of memory.

### Mode 3 — CPU-first with partial GPU acceleration

```text
RAM: most model weights
CPU: most target execution
GPU: selected layers / acceleration / possibly draft model
```

This may be more appropriate on a very low-VRAM GPU or when model weights cannot be staged efficiently. It should be included as a comparison mode, not dismissed in advance.

### Mode 4 — Experimental weight-streaming pipeline

```text
Host RAM --async H2D--> GPU staging buffer
                               |
                         compute current layer
                               |
                      wait for next layer ready
```

Use only when the runtime/backend provides direct control over layer weights and profiling confirms a meaningful overlap. This is not automatically faster than CPU execution for offloaded layers.

### Mode 5 — Optional multi-GPU / multi-node

Not part of the core MVP. Add only after a single machine is reliable. Remote execution is especially sensitive to network bandwidth and synchronization; a network link is not equivalent to local VRAM.

---

## 8. KV-cache subsystem

The KV cache grows with the context. In long-context use, it can consume a significant fraction of memory even when model weights are quantized.

### 8.1 KV-cache policy options

1. **Precision selection:** FP16/BF16 where supported; Q8/Q4 or backend-supported alternatives when memory pressure requires it.
2. **Placement:** GPU-resident cache when capacity allows; host-RAM offload when VRAM is scarce.
3. **Block/page allocation:** fixed-size token blocks to manage variable-length contexts and avoid large contiguous allocations. Begin with a simple block allocator; do not build a full vLLM-like prefix-sharing system in v1.
4. **Prefetch:** stage the next layer's KV block before it is required, if the backend can expose layerwise execution and profiling shows transfer overlap.
5. **Eviction:** only discard/recompute/approximate KV under a clearly defined semantic policy. Ordinary exact autoregressive generation cannot arbitrarily drop old KV without changing model behavior.
6. **Prefix reuse:** stretch goal for repeated identical system prompts or shared document prefixes. Cache keys must include model, tokenizer, adapter, RoPE/context settings, and exact tokenized prefix.

### 8.2 Suggested page metadata

```text
page_id
request_id
layer_id
token_start
token_end
key_dtype
value_dtype
byte_size
memory_tier       # VRAM / pinned RAM / pageable RAM / SSD-cold
last_access_time
owner / reference count
transfer_state
ready_event
```

The MVP may only need `page_id`, `layer_id`, token range, device, dtype, and transfer state.

### 8.3 KV offload policy

A simple initial policy:

```text
if GPU memory estimate fits with safety margin:
    keep KV on GPU
elif backend supports efficient KV offload:
    use host offload
elif context is too large even with offload:
    reduce requested context or recommend a model/cache configuration
```

Do not immediately implement attention-aware eviction, learned token importance, remote KV, and disk paging. These are research-sized additions and introduce quality/correctness risks.

### 8.4 SSD policy

Treat SSD as **cold storage**, not “extra VRAM.” Use it primarily for model files, model download/cache, and potentially cold checkpoint/swap experiments. Repeatedly transferring active KV pages or model layers from SSD during each token step is likely to make latency poor. A RAM staging tier should precede any storage transfer, and the SSD path should remain a stretch experiment.

### 8.5 KV quantization correctness

Quantizing K/V can reduce cache footprint, but may affect output quality and/or latency. The quantization overhead may not be worthwhile for a short context that already fits in VRAM. Benchmark each cache format at multiple context lengths and compare task quality/perplexity where practical. Never assume “lower bits = faster.”

---

## 9. CPU-draft / GPU-target speculative decoding

This is one of the most interesting project-specific experiments.

### 9.1 Proposed placement

```text
                 HOST CPU + RAM
          +---------------------------+
          | Small draft model         |
          | Draft KV cache             |
          | CPU threads / sampler      |
          +-------------+-------------+
                        |
                 proposed token IDs
                        |
                        v
                GPU TARGET RUNTIME
          +---------------------------+
          | Large target model        |
          | Target KV cache           |
          | Parallel verification     |
          +---------------------------+
```

**Hypothesis:** keep the GPU focused on the target model and run the smaller draft model on CPU, avoiding GPU VRAM/compute contention from a second model.

This is plausible when:

- the target model has substantial GPU-resident execution;
- the CPU can draft a block quickly enough;
- the draft model is small and sufficiently predictive;
- CPU resources are not already saturated by target layers offloaded to CPU;
- tokenization and vocabulary are compatible;
- target verification is substantially more efficient over a block than one token at a time.

It may lose when the CPU draft is too slow, the CPU is already executing target layers, the draft acceptance rate is low, or the target model is not actually GPU-resident. Dovetail is directly relevant prior art, but it tests the **opposite** allocation (GPU draft, CPU target), so ADHIE must experimentally compare placements rather than cite it as proof of the proposed direction.

### 9.2 Speculative decoding cycle

For a draft length `k`:

1. CPU draft model autoregressively proposes `k` candidate tokens.
2. Candidate IDs and required sampling metadata are passed to the target runtime.
3. The target computes/verifies the candidate block in a batched/parallel forward path.
4. A correct rejection-sampling/verification procedure accepts a prefix and emits the appropriate next token(s).
5. The accepted prefix becomes the context for the next draft cycle.
6. The controller measures draft time, target verification time, accepted length, and output latency.

Use a well-understood lossless speculative decoding algorithm. For stochastic generation, preserve the target model's intended sampling distribution with the correct rejection-sampling procedure; do not compare only greedy output and claim correctness for all sampling modes.

### 9.3 Important dependency: drafting and verification are not freely concurrent

For a single sequence, the CPU needs the accepted prefix before it can correctly draft the next proposal block. Therefore, the basic cycle is sequential at the block level:

```text
CPU draft block -> GPU verify block -> accepted prefix -> CPU draft next block
```

Do **not** describe this as CPU drafting and GPU target verification of the same next block occurring simultaneously. They are causally dependent. The valuable overlap in the proposed architecture is more realistically:

- GPU target computation overlapping with independent H2D weight/KV prefetch;
- CPU draft work overlapping with unrelated host preparation or independent requests, if any;
- other CPU work such as tokenization / scheduling overlapping with GPU execution.

For the primary single-user/single-sequence workload, benchmark the full end-to-end cycle. Do not count just GPU verification time.

### 9.4 Draft length should be adaptive

Do not hard-code `k=8` or `k=16`. Start by testing a small set, for example:

```text
k in {2, 4, 6, 8}
```

Expand only if results justify it. Record:

- draft time per block;
- target verification time per block;
- accepted tokens per block;
- acceptance rate by draft position;
- emitted tokens per second;
- TPOT and ITL;
- CPU utilization and target GPU utilization.

Approximate speculative-cycle speedup can be reasoned about as:

\[
Speedup \approx \frac{E[A] \times T_{target,1}}{T_{draft}(k)+T_{verify}(k)}
\]

where `E[A]` is expected emitted tokens per speculative cycle (accepted draft tokens plus the appropriate target token), and `T_target,1` is baseline target-only time per emitted token. This is a conceptual estimate; measure actual end-to-end time because block verification, memory traffic, and sampling costs are not perfectly linear.

If speculative decoding is slower, disable it. The controller should learn that a CPU draft is not beneficial on every CPU/GPU/model combination.

### 9.5 Required comparisons

At minimum compare:

| Variant | Draft placement | Target placement | Purpose |
|---|---|---|---|
| A | None | GPU/hybrid | Baseline autoregressive target-only decoding |
| B | CPU | GPU/hybrid | Proposed ADHIE hypothesis |
| C | GPU, if it fits | GPU/hybrid | Measures GPU resource contention vs faster drafting |
| D | CPU | CPU/hybrid | Useful low-VRAM comparison if target is CPU-heavy |
| E | N-gram/suffix if supported | GPU/hybrid | Low-overhead non-neural draft baseline |

Keep target model, target quantization, prompt, output length, sampling parameters, and context identical. Report draft and target resource usage separately.

### 9.6 Practical compatibility requirements

- Prefer a draft checkpoint from the same model family and tokenizer/vocabulary as the target.
- Verify tokenizer ID compatibility; do not assume same vocabulary because model names look related.
- Verify chat template and special-token behavior.
- Measure draft acceptance on the actual task prompts, not just a synthetic random prompt.
- Reserve CPU cores/threads for the draft model, and measure contention with any target layers executing on CPU.
- Begin with backend-supported speculative decoding. A custom sampler/verifier is a later task only if necessary.

### 9.7 Example llama.cpp experiment command

The current llama.cpp server documentation exposes separate draft-model device and CPU-thread controls. A starting experiment may look conceptually like:

```bash
llama-server \
  --model /models/target.gguf \
  --model-draft /models/draft.gguf \
  --spec-type draft-simple \
  --device-draft none \
  --threads-draft 6 \
  --spec-draft-n-max 4 \
  --gpu-layers 24 \
  --ctx-size 8192 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

This is an illustrative starting point, not a guaranteed valid command for every llama.cpp build or model. Confirm current flag names with the pinned build's `llama-server --help`; tune target GPU layers, context, draft thread count, and draft length for the actual machine. `--device-draft none` is intended to keep draft execution off the GPU; verify the startup log/device placement rather than assuming the flag behaved as expected.

---

## 10. CUDA compute-stream + transfer-stream layer prefetch

The proposed CUDA idea is **technically plausible**, but it is a separate optimization from speculative decoding and should be validated independently.

### 10.1 Intended pipeline

Suppose the target model has a sequence of layers and some layer weights are staged from host RAM to VRAM on demand.

```text
TIME -------------------------------------------------------------->

COMPUTE STREAM:   [compute L0] [compute L1] [compute L2] [compute L3]

TRANSFER STREAM:       [H2D L1]    [H2D L2]    [H2D L3]    [H2D L4]

Dependency: each layer's compute must wait until its own weights are fully ready.
```

The goal is to transfer layer `N+1` while the GPU computes layer `N`. In the idealized case, the steady-state layer time approaches:

\[
T_{layer} \approx \max(T_{compute}, T_{transfer})
\]

rather than their sum. This is only an ideal pipeline approximation. Real performance depends on transfer volume, PCIe bandwidth, copy-engine capability, compute duration, launch overhead, buffer capacity, and runtime scheduling.

### 10.2 Use explicit non-default streams

Prefer:

```text
compute_stream  = explicit non-blocking CUDA stream
transfer_stream = explicit non-blocking CUDA stream
```

Do not rely on the legacy default stream for the target compute path when overlap is the goal. The legacy default stream can introduce implicit synchronization with other blocking streams. Use CUDA events to express the dependency between transfer completion and layer consumption.

Conceptual C++ skeleton:

```cpp
cudaStream_t compute_stream;
cudaStream_t transfer_stream;

cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking);

// Host buffers used for asynchronous copies should be pinned/page-locked.
// Device staging buffers should be preallocated and reused.

cudaMemcpyAsync(
    d_next_layer_weights,
    h_next_layer_weights_pinned,
    bytes,
    cudaMemcpyHostToDevice,
    transfer_stream
);

cudaEventRecord(next_layer_ready, transfer_stream);
cudaStreamWaitEvent(compute_stream, next_layer_ready, 0);

// Launch the kernel(s) for the layer only after its weights are ready.
launch_layer_kernel(..., compute_stream);
```

This is a conceptual sketch, not a complete race-free allocator. A production implementation also needs buffer lifetime management, error handling, double/triple buffering, per-layer events, and protection against overwriting a staging buffer while a kernel still reads it.

### 10.3 Required implementation details

- Allocate/reuse pinned host staging buffers; page-locked memory is required for genuinely asynchronous host-to-device copies on standard CUDA paths.
- Preallocate device buffers; avoid hot-path `cudaMalloc`/free.
- Use double buffering at minimum where memory permits:
  - buffer A is consumed by the compute stream;
  - buffer B receives the next layer on the transfer stream;
  - swap only after the event confirms the consumer is finished.
- Record an event after H2D completion; make the compute stream wait on that event immediately before the relevant layer.
- Avoid global synchronization (`cudaDeviceSynchronize`) in the steady-state pipeline.
- Use NVTX ranges and Nsight Systems to verify actual H2D/kernel overlap; an asynchronous API call alone does not prove overlap.
- Query relevant device properties and test on the target GPU; hardware does not guarantee that every transfer and kernel schedule overlaps as desired.
- Reuse pinned buffers carefully; pinned memory is a limited system resource and excessive allocation can degrade the host.

### 10.4 The bandwidth catch

Overlapping transfer with compute does not make the transfer free. If the target's offloaded weights are copied from RAM to VRAM for every generated token, the amount of data moved per token may be enormous.

A useful lower-bound intuition is:

\[
T_{H2D,token} \geq \frac{W_{offloaded}}{B_{effective}}
\]

where `W_offloaded` is the amount of weight data that must cross PCIe per token and `B_effective` is measured effective H2D bandwidth. Even perfect overlap cannot exceed the available transfer bandwidth. If transfer time exceeds useful compute time, the pipeline is transfer-bound.

This leads to an important design choice:

- **CPU-execute an offloaded layer:** weights remain in host memory and CPU performs its operations. This avoids re-uploading the entire layer to GPU each token, but CPU execution may be slower.
- **GPU-stream an offloaded layer:** weights are staged to GPU and GPU executes them. This may use faster GPU compute but incurs host-to-device traffic and requires enough compute time to hide it.
- **Keep the layer resident on GPU:** best when VRAM budget permits.

The planner must compare these choices from measurements. The transfer-stream idea is not automatically better than CPU layer execution.

### 10.5 Do not conflate draft/verify overlap with weight/compute overlap

There are three distinct activities:

1. CPU draft model generates a candidate block.
2. GPU target verifies that block.
3. GPU copy engine stages data needed by future target layers.

(1) must precede (2) for the same proposal block. (3) may overlap with (2) if target layer weights are streamed and dependencies are correct. It is misleading to claim that CPU draft, GPU verification, and H2D weight transfers can all run freely in parallel for one sequence at every step.

### 10.6 Go/no-go test

Before integrating this into ADHIE:

1. Build a synthetic CUDA benchmark with representative layer-sized H2D copies and compute kernels.
2. Confirm overlap visually in Nsight Systems.
3. Measure isolated transfer time, isolated compute time, and overlapped time.
4. Move to one real transformer layer or a backend-specific layer hook.
5. Compare against CPU execution of the same offloaded layer.
6. Integrate only if end-to-end token latency improves on a defined hardware/model regime.

This is a high-value stretch contribution, not a prerequisite for the first working end-to-end engine.

---

## 11. Autotuner / adaptive controller

This is the central ADHIE contribution.

### 11.1 Two operating phases

**Phase A — startup planning (coarse autotuning):** before loading the model fully, generate a small list of candidate configurations, reject infeasible ones using memory estimates, then benchmark the most promising candidates briefly.

**Phase B — safe runtime adaptation:** while generating, update only parameters that can be changed safely and cheaply. Examples:

- speculative draft length;
- whether speculative decoding remains enabled;
- future-request KV precision/offload choice;
- CPU thread budget for the next request;
- whether optional prefetch is enabled for future layers/requests.

Do not dynamically migrate all target weights or rebuild the model graph in the token hot path. Weight placement should normally be selected at load time and remain stable until the model is unloaded.

### 11.2 Candidate search

A practical search procedure:

```text
1. Read hardware and model metadata.
2. Estimate VRAM/RAM feasibility for the requested context.
3. Enumerate a small candidate set:
      - GPU layer split: low / medium / high
      - KV type: native / Q8 (and Q4 only if supported and justified)
      - KV placement: GPU / host-offloaded
      - draft mode: off / CPU draft / GPU draft if available
      - draft length: 2 / 4 / 6 / 8
4. Reject candidates exceeding memory or compatibility limits.
5. Run short warm-up and benchmark trials for the remaining candidates.
6. Score each candidate using TTFT, TPOT, memory headroom, and quality floor.
7. Store selected configuration plus the measured profile.
```

Do not test the full Cartesian product. A few carefully selected candidates are enough for a six-month project.

### 11.3 Objective function

For a single-user interactive workload, one possible score is:

\[
J(c) = w_1 \cdot P95(TPOT_c) + w_2 \cdot P95(TTFT_c)
       + w_3 \cdot MemoryPenalty_c + w_4 \cdot QualityPenalty_c
\]

subject to:

\[
PeakVRAM_c \leq VRAM_{budget}, \quad
PeakRAM_c \leq RAM_{budget}, \quad
Quality_c \geq Q_{min}
\]

Weights `w1...w4` should be chosen for the target use case and clearly documented. Start with a rule-based score; Bayesian optimization or contextual bandits are optional later work.

### 11.4 The “usable model capacity” metric

Define **Usable Model Capacity (UMC)** as the largest tested model/configuration that meets a defined interaction and quality target on a given machine.

For example, report:

```text
Hardware profile: 4 GB VRAM / 32 GB RAM / CPU XYZ
Model: model family + parameter count + quantization
Context: 8K / 16K / 32K
Output: 256 tokens
P95 TTFT: measured
P95 TPOT: measured
Peak VRAM/RAM: measured
Quality check: measured
UMC: largest model/configuration that satisfies the agreed limits
```

UMC is more meaningful than claiming a model “runs” because it connects model size to latency, memory, and quality.

---

## 12. Benchmark and evaluation framework

### 12.1 Baselines

At minimum:

1. **Backend default** — same backend with its automatic/default configuration.
2. **Manual tuned backend** — a human-selected configuration after reasonable tuning.
3. **ADHIE selected configuration** — planner-selected configuration.
4. **Speculation ablations** — target-only, CPU draft, GPU draft where feasible.
5. **KV/offload ablations** — default KV, quantized KV, offloaded KV where supported.

Use llama.cpp as the main practical baseline initially. If a model/hardware combination is supported by KTransformers or Transformers, add them as references, but do not force an apples-to-oranges comparison.

### 12.2 Metrics

| Metric | Meaning | Why it matters |
|---|---|---|
| TTFT | Time from request submission to first streamed output token | User-perceived prompt latency. |
| TPOT | Per-request average time per output token after the first | Useful interactive generation metric. |
| ITL | Time between successive streamed output events | Captures pauses; interpret carefully under speculative decoding. |
| Output tokens/s | Generated tokens divided by generation time | Familiar throughput measure for one user. |
| Prompt processing tokens/s | Prompt tokens processed per second | Important for long prompts and context-heavy work. |
| Peak VRAM | Maximum GPU memory use | Feasibility and OOM headroom. |
| Peak RAM | Maximum process/system RAM use | Feasibility on consumer machines. |
| CPU/GPU utilization | Resource occupancy | Identifies idle or saturated resources. |
| H2D/D2H bytes and bandwidth | Host-device traffic | Detects transfer-bound execution. |
| Transfer/compute overlap | Fraction/time where H2D and compute overlap | Verifies whether CUDA pipeline works. |
| KV-cache bytes / occupancy | Memory used by KV | Context capacity and cache efficiency. |
| Draft acceptance | Accepted draft tokens / proposed draft tokens, plus accepted length distribution | Determines speculative usefulness. |
| Energy / power (optional) | Energy or average power per output token | Valuable consumer-device metric if reliable telemetry exists. |
| Quality/perplexity | Task accuracy or language-model quality proxy | Prevents speed-only regressions. |

**Important speculative-decoding measurement detail:** ITL can differ from TPOT because one streamed output event may contain multiple accepted tokens. TPOT amortizes generation time over all output tokens, while ITL measures gaps between streamed outputs. Report both and define the measurement point.

### 12.3 Workload matrix

Keep the main workload single-user. Start with:

| Dimension | Suggested values |
|---|---|
| Active sequences | 1; optionally 2 and 4 as secondary tests |
| Prompt/context length | 2K, 8K, 16K, 32K; add 64K only where model/backend/memory permit |
| Generated output | 128 and 256 tokens; optionally 512 |
| Model sizes | one small development model; one 10B–14B main model; larger model only if hardware permits |
| Weight formats | one practical 4-bit format; compare another quantization only if time permits |
| KV cache | backend default, Q8, lower-bit only if supported and quality checked |
| Draft length | 2, 4, 6, 8; adaptive after fixed-length tests |
| Hardware classes | low-VRAM, midrange, and high-VRAM machine if the team can access them |

Do not make the benchmark matrix so large that the team spends the whole project running tests. Prioritize experiments that answer the research questions.

### 12.4 Reproducibility protocol

- Pin backend versions/commit hashes, driver, CUDA version, model revision, tokenizer, and quantization file.
- Use exactly the same prompt tokens and generation settings for baseline comparisons.
- Warm up before timing; report cold model-load time separately from warm inference time.
- Run multiple trials; report median and P95 where sample size permits.
- Record GPU clocks/temperature and avoid thermal throttling where possible.
- Note background processes and available RAM/VRAM.
- Reset prefix/KV caches between trials when cache reuse is not intended.
- Record raw per-request output and timing data in JSON/CSV.
- Make benchmark scripts runnable from the repository.

### 12.5 Quality evaluation

At minimum:

- check greedy output equivalence for lossless speculative decoding against target-only decoding under identical settings;
- compare output behavior for sampling settings with the correct verification algorithm;
- compare a small set of task-level outputs relevant to local use (e.g., code completion, summarization, and long-context retrieval);
- if KV/weight quantization changes, add perplexity or an established evaluation harness on a manageable sample.

Never attribute a quality drop to “randomness” without fixing seeds or documenting sampling variance.

---

## 13. Research questions and hypotheses

The final report should focus on answering a few testable questions, not claiming every feature is novel.

### RQ1 — Hardware-aware configuration

**Can a lightweight startup autotuner select a better CPU/GPU/KV configuration than backend defaults or a fixed manual configuration across different consumer hardware profiles?**

- H1: a small candidate search will improve at least one of TTFT/TPOT or feasible context size on some tested profiles, while avoiding memory failures.

### RQ2 — CPU draft with GPU target

**When does CPU-resident speculative drafting improve single-user decoding latency for a GPU-accelerated target model?**

- H2: CPU drafting helps when the CPU draft cycle is sufficiently cheap and target GPU verification accepts enough tokens; it hurts when CPU drafting or CPU offloaded target layers dominate.

### RQ3 — Adaptive draft length

**Can acceptance-aware draft-length selection outperform one fixed draft length across different prompts and hardware?**

- H3: adaptive `k` will improve average end-to-end TPOT in workloads where acceptance characteristics vary.

### RQ4 — KV cache placement/precision

**When is host offload or low-bit KV more beneficial than keeping the cache at native precision on GPU?**

- H4: the best choice changes with context length, available VRAM, and device transfer bandwidth; aggressive KV quantization can hurt latency/quality for short contexts.

### RQ5 — Asynchronous layer prefetch

**Can H2D transfer of upcoming target-layer weights be hidden behind current-layer computation on the selected hardware, and does this improve end-to-end token latency over CPU execution of the offloaded layer?**

- H5: overlap is beneficial only in a measurable regime where copy time and compute time are balanced and the backend exposes enough control. Otherwise, the feature should remain disabled.

---

## 14. Six-month implementation roadmap

The project is part-time. The roadmap is organized around deliverable gates, not a promise that every stretch feature will be completed.

### Phase 0 — Scope lock and environment setup (Weeks 1–2)

**Tasks**
- Inventory all available team machines: GPU/VRAM, CPU, RAM, OS, storage, CUDA capability.
- Pick one primary target machine and one fallback machine.
- Select a main 10B–14B quantized model that can be legally downloaded and run by the team; record model family, license, tokenizer, quantization, and context limit.
- Select a compatible small draft model from the same family if available.
- Build and pin llama.cpp (or selected backend) with the relevant hardware support.
- Define benchmark prompts and output length.

**Deliverables**
- hardware inventory;
- model manifest;
- baseline inference command and successful local run;
- initial repository and issue board.

**Gate:** The team can run the same model and prompt reliably on the primary machine before custom optimization begins.

### Phase 1 — Baseline and measurement foundation (Weeks 3–4)

**Tasks**
- Implement hardware profiler.
- Implement model metadata/config loader.
- Implement memory estimator for weights and KV cache.
- Capture backend baseline TTFT, TPOT, ITL, tokens/s, peak VRAM/RAM.
- Test a few GPU-layer splits and context sizes.
- Build a simple JSON/CSV benchmark runner.

**Deliverables**
- profiler report;
- memory estimate report;
- baseline benchmark data;
- short design review identifying the actual bottleneck.

**Gate:** Measurements are repeatable and the team knows whether the main limitation is VRAM capacity, GPU memory bandwidth, PCIe transfer, CPU compute, or KV size.

### Phase 2 — Practical heterogeneous runtime and KV policy (Weeks 5–8)

**Tasks**
- Integrate a backend adapter around llama.cpp.
- Generate feasible GPU layer/offload settings from memory estimates.
- Test host-RAM placement and CPU layer execution.
- Test backend-supported KV precision options and KV offload.
- Implement a simple startup plan generator with safe memory margin.
- Keep SSD out of the active inference path.

**Deliverables**
- working `adhie run` prototype;
- configuration report explaining selected GPU layers/KV mode;
- baseline-vs-planner comparison.

**Gate:** ADHIE can launch a model with a selected configuration and explain why it was chosen.

### Phase 3 — CPU draft / GPU target speculative decoding (Weeks 9–12)

**Tasks**
- Verify draft/target tokenizer and architecture compatibility.
- Establish target-only baseline.
- Run CPU draft + GPU target via supported backend facilities where possible.
- Compare GPU draft (if it fits) and CPU-heavy alternatives where available.
- Sweep fixed draft lengths (initially 2/4/6/8).
- Record draft latency, verification latency, acceptance, TTFT, TPOT, ITL, CPU and GPU utilization.
- Implement a simple adaptive rule: enable speculation only when recent measured speedup is positive; choose draft length from a small tested set.

**Deliverables**
- speculative decoding experiment report;
- adaptive draft-length controller;
- explicit regimes where speculation helps and hurts.

**Gate:** End-to-end benefit is demonstrated on at least one hardware/workload regime, or the system correctly disables speculation when it is slower.

### Phase 4 — CUDA asynchronous prefetch proof-of-concept (Weeks 13–16)

**Tasks**
- Build isolated H2D + compute microbenchmark.
- Use pinned host buffers, explicit non-blocking compute/transfer streams, events, and double buffering.
- Profile with Nsight Systems.
- Compare transfer-only, compute-only, overlap, and CPU-execution alternatives.
- If the result is promising, integrate into one experimental layer path or a backend hook. Do not rewrite the entire runtime.

**Deliverables**
- Nsight timeline screenshots/traces;
- copy/compute overlap measurements;
- go/no-go decision for integration.

**Gate:** Only proceed if the overlap is real and end-to-end latency improves. If not, document the bottleneck and preserve the microbenchmark as a negative/diagnostic result.

### Phase 5 — Autotuner integration and stabilization (Weeks 17–20)

**Tasks**
- Combine hardware profiler, model inspector, memory estimator, candidate generator, and benchmark evaluator.
- Implement startup autotuning over a deliberately small candidate set.
- Implement safe runtime adaptation for draft length and optionally future-request KV policy.
- Save per-machine profiles and selected configurations.
- Add basic CLI status output and optional Prometheus/Grafana metrics if time permits.

**Deliverables**
- integrated ADHIE prototype;
- reusable machine profile and run-plan format;
- end-to-end demo script.

**Gate:** A new run can select a plan from hardware/model information without manual tuning of every parameter.

### Phase 6 — Final evaluation, ablation, and report (Weeks 21–24)

**Tasks**
- Repeat the most informative benchmark matrix.
- Run ablations: no speculation, CPU draft, GPU draft if feasible, no KV offload, quantized/offloaded KV, no prefetch vs prefetch.
- Evaluate quality and correctness.
- Profile bottlenecks and explain failure regimes.
- Stabilize installation and documentation.
- Prepare thesis/report, architecture diagrams, graphs, and live demo.

**Deliverables**
- final code release;
- benchmark dataset/results and plotting scripts;
- ablation tables;
- final report/presentation;
- reproducible local demonstration.

---

## 15. Three-member ownership model

The ownership should be clear, but the interfaces and benchmark methodology must be shared.

### Member 1 — Runtime integration and memory/KV subsystem

**Owns**
- backend adapter and model launch;
- memory estimator integration;
- CPU/GPU layer placement experiments;
- KV precision/offload experiments;
- CUDA layer-prefetch proof-of-concept if the team has the required CUDA experience.

**Core learning areas**
- llama.cpp runtime and GGUF model format;
- CUDA streams, events, pinned host memory;
- GPU memory and PCIe behavior;
- KV-cache layout and memory accounting;
- Nsight Systems and PyTorch/CUDA profiling.

**Primary deliverable:** stable inference backend plus memory/KV execution options; optional custom prefetch experiment.

### Member 2 — Speculative decoding and CPU-side execution

**Owns**
- draft model selection and compatibility;
- CPU draft configuration and thread affinity;
- draft length sweep and acceptance tracking;
- target verification measurements;
- adaptive draft controller;
- correctness checks for speculative decoding.

**Core learning areas**
- speculative decoding algorithms and rejection sampling;
- target/draft tokenizer compatibility;
- CPU inference and thread placement;
- latency decomposition and acceptance metrics.

**Primary deliverable:** CPU-draft/GPU-target evaluation and adaptive speculative policy.

### Member 3 — Hardware profiler, autotuner, and benchmarking

**Owns**
- hardware discovery and benchmark micro-tests;
- model metadata schema;
- memory/cost estimator;
- candidate generation and configuration scoring;
- benchmark runner, JSON/CSV logging, plots, and dashboard/CLI report;
- end-to-end evaluation protocol.

**Core learning areas**
- GPU/CPU performance profiling;
- memory estimation for transformer architectures;
- experiment design and statistics;
- backend interfaces and orchestration.

**Primary deliverable:** ADHIE planner/autotuner and reproducible benchmark suite.

### Shared engineering rules

- Agree on data structures and interfaces before parallel implementation.
- Every optimization must have a baseline and an ablation.
- Every subsystem must expose metrics to the benchmark runner.
- All three members should be able to build and run the integrated system.
- Do not let one member's experimental branch become a blocker for the main runnable version.

---

## 16. Suggested interfaces and repository layout

### 16.1 Core interfaces (illustrative)

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareProfile:
    gpu_name: Optional[str]
    vram_total_bytes: int
    vram_free_bytes: int
    ram_total_bytes: int
    ram_available_bytes: int
    cpu_physical_cores: int
    cpu_logical_cores: int
    h2d_bandwidth_bytes_s: Optional[float]
    d2h_bandwidth_bytes_s: Optional[float]

@dataclass
class ModelProfile:
    model_id: str
    architecture: str
    parameter_count: int
    layer_count: int
    num_kv_heads: Optional[int]
    head_dim: Optional[int]
    max_context: Optional[int]
    weight_format: str
    tokenizer_id: str

@dataclass
class RunPlan:
    backend: str
    gpu_layers: int
    context_length: int
    kv_type_k: str
    kv_type_v: str
    kv_offload: bool
    draft_model: Optional[str]
    draft_device: str
    draft_length: int
    cpu_threads: int
    rationale: str

class HardwareProfiler:
    def profile(self) -> HardwareProfile: ...

class ModelInspector:
    def inspect(self, model_path: str) -> ModelProfile: ...

class MemoryEstimator:
    def estimate(self, hardware, model, workload): ...

class PlanGenerator:
    def generate_candidates(self, hardware, model, workload): ...

class CandidateEvaluator:
    def benchmark(self, plan, workload): ...

class RuntimeAdapter:
    def load(self, plan): ...
    def generate(self, request): ...
    def metrics(self): ...
    def unload(self): ...
```

The code is an interface sketch. Use backend-specific types internally and validate all model metadata. Do not put vendor-specific fields in the generic API unless they are optional capabilities.

### 16.2 Suggested repository structure

```text
adhie/
├── README.md
├── docs/
│   ├── project_specification.md
│   ├── architecture.md
│   ├── experiments.md
│   └── decisions/
├── configs/
│   ├── hardware_profiles/
│   ├── model_profiles/
│   └── benchmark_workloads/
├── src/
│   ├── cli/
│   ├── profiler/
│   ├── model_inspector/
│   ├── memory_estimator/
│   ├── planner/
│   ├── runtime/
│   │   ├── base.py
│   │   └── llamacpp_adapter.py
│   ├── kv_policy/
│   ├── speculative/
│   ├── telemetry/
│   └── api/
├── cuda_experiments/
│   ├── async_copy_compute/
│   └── layer_prefetch/
├── benchmarks/
│   ├── run_benchmark.py
│   ├── workloads/
│   ├── results/
│   └── plots/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── correctness/
└── scripts/
    ├── setup_backend.sh
    ├── profile_hardware.sh
    └── reproduce_results.sh
```

### 16.3 Configuration/run-plan example

```yaml
model:
  path: /models/target-model.gguf
  draft_path: /models/draft-model.gguf
  context_length: 16384

runtime:
  backend: llamacpp
  gpu_layers: auto
  cpu_threads: auto

kv_cache:
  key_type: q8_0
  value_type: q8_0
  offload: backend_auto

speculative:
  enabled: auto
  draft_device: cpu
  draft_tokens: auto
  candidate_lengths: [2, 4, 6, 8]

policy:
  vram_safety_margin_mb: 1024
  optimize_for: interactive_latency
```

This is a conceptual ADHIE configuration. Exact accepted settings must be translated to the pinned backend version; do not pass unsupported flags.

---

## 17. Technology stack recommendation

| Area | Recommended starting point | Why |
|---|---|---|
| Main language | Python | Planner, orchestration, profiling, CLI, experiment scripts. |
| Low-level work | C++ / CUDA | Stream/event orchestration and any narrow custom GPU path. |
| GPU kernel experimentation | Triton and/or CUDA C++ | Use for measured bottlenecks, not as a blanket rewrite. |
| Inference backend | llama.cpp first | Mature quantized local inference, CPU/GPU placement and draft support. |
| Model metadata | GGUF metadata initially; optional Hugging Face config support later | Reduces initial model-loader work. |
| API | CLI first; optional FastAPI local endpoint | A polished CLI matters more than building a web server for a local runtime. |
| Monitoring | Structured logs + JSON/CSV first; Prometheus/Grafana optional | Do not lose time building a dashboard before measuring correctly. |
| Profiling | Nsight Systems, Nsight Compute, CUDA events, `nvidia-smi`, `perf` | Needed to identify compute, memory, and transfer bottlenecks. |
| Plotting | Python / matplotlib | Reproducible plots from benchmark output. |

**PyTorch note:** If a custom model execution experiment is needed, PyTorch can be used for a focused prototype. Do not build the core project around a full Python/Hugging Face eager execution path if it makes low-level layer placement and quantized CPU/GPU execution harder than using the selected backend. The planner should be backend-independent even if the first implementation uses one backend.

---

## 18. Risks and mitigation

| Risk | Why it matters | Mitigation |
|---|---|---|
| Scope explosion | A full inference runtime is too much for three part-time students in six months. | Use an existing backend; enforce must-have/stretch separation. |
| 4 GB GPU performance is poor | Model may load but weight traffic/CPU execution dominates. | Report speed honestly; optimize for the best feasible model and context, not universal performance promises. |
| CPU draft is slower than expected | Draft stage can become the bottleneck. | Sweep draft lengths; benchmark CPU/GPU placement; disable speculation when it loses. |
| CPU draft competes with CPU target layers | Hybrid target execution can saturate CPU cores/RAM bandwidth. | Reserve CPU threads, profile contention, compare CPU/GPU split choices. |
| Weight streaming is transfer-bound | PCIe transfer of large layer weights per generated token may dominate. | Calculate bytes/token, profile real overlap, compare CPU layer execution; keep streaming experimental. |
| CUDA overlap is not achieved | Default stream, pageable buffers, synchronization, or hardware limitations can serialize work. | Explicit non-blocking streams, pinned memory, events, Nsight Systems verification. |
| KV quantization hurts quality or latency | Lower precision is not always faster or acceptable. | Evaluate quality and latency at multiple contexts; use only when benefit is measured. |
| Backend internals change | CLI options/API can change between releases. | Pin exact versions and write adapter tests. |
| Draft tokenizer mismatch | Speculative verification may be invalid or incompatible. | Select same-family compatible checkpoints and validate token IDs/templates. |
| Not enough test machines | Hardware diversity claims become weak. | Use available machines as separate hardware profiles; add reproducible synthetic bandwidth tests, but do not claim universal support. |
| Benchmark noise | Thermal state, cache reuse, and background work distort results. | Warmup, repeated runs, fixed prompt, record environment, separate cold/warm metrics. |
| Team integration delay | Parallel tasks can become three disconnected projects. | Define interfaces early, integrate a runnable version every 2–3 weeks. |

---

## 19. Must-answer technical decisions before implementation

These questions should be resolved by the team during Weeks 1–2; they are not reasons to delay starting the profiler/backend baseline.

1. **What exact GPU/CPU/RAM machines are available?** Include VRAM, total/free RAM, CPU model/core count, OS, GPU architecture, and storage.
2. **Is the primary implementation NVIDIA-only?** The custom CUDA work assumes NVIDIA; the planner and CPU backend should still have graceful fallbacks.
3. **Which exact 10B–14B target model and quantized format will be used first?** Model architecture determines attention/KV behavior and backend compatibility.
4. **Is a tokenizer-compatible draft model available?** If not, test an n-gram/suffix method or remove the draft-model requirement from the first milestone.
5. **Will llama.cpp be the production runtime adapter?** Strong recommendation: yes for v1; custom CUDA experiments should be isolated until validated.
6. **What is the main workload?** Coding assistant, document analysis, long-context summarization, or general chat. This determines prompt patterns and quality tests.
7. **What is the team's actual weekly time budget?** The roadmap is six months, but should be trimmed if the sustainable weekly effort is low.

---

## 20. Recommended final demonstration

A useful demonstration should show the engine making and justifying decisions, not merely printing tokens.

### Demo flow

1. Start ADHIE on a machine with limited VRAM.
2. Run hardware profiling and show available VRAM/RAM and measured transfer bandwidth.
3. Select a 10B+ quantized model and a compatible draft model.
4. Ask ADHIE to serve a long-context prompt.
5. Show the generated run plan:
   - backend and model format;
   - selected GPU layer count;
   - host-RAM placement;
   - KV type/placement;
   - speculative mode and draft device/length;
   - memory estimate and safety margin.
6. Run baseline and ADHIE-selected configuration using the same prompt.
7. Show TTFT, TPOT, ITL, tokens/s, peak VRAM/RAM, CPU/GPU utilization, and draft acceptance.
8. Change one constraint (context length, available VRAM budget, or draft length) and rerun planning.
9. If the asynchronous prefetch prototype passes its go/no-go gate, show the Nsight timeline and its measured end-to-end impact. Otherwise, show the negative result and explain why the planner does not enable it.

### What to claim

A strong, defensible final claim looks like:

> “On the tested hardware profiles and model/workload combinations, ADHIE automatically selected heterogeneous execution and decoding configurations that improved [specific measured metric] or enabled [specific larger context/model configuration] relative to [named baseline], while maintaining [defined quality check]. We also identified hardware regimes where CPU drafting, KV offload, or asynchronous weight prefetch did not improve performance and disabled those options.”

Avoid claims such as “runs every 100B model on a 4 GB GPU” or “always faster than llama.cpp.”

---

## 21. Recommended project scope at a glance

### Core release (must finish)

- [ ] hardware profiler;
- [ ] model inspector and weight/KV memory estimator;
- [ ] llama.cpp runtime adapter;
- [ ] automatic candidate generation for GPU/CPU placement and KV settings;
- [ ] CPU-draft/GPU-target speculative decoding benchmark;
- [ ] simple adaptive draft-length/enable-disable policy;
- [ ] reproducible benchmark runner and reports;
- [ ] 10B+ model demonstration on at least one memory-constrained machine.

### Stretch release (only after core is stable)

- [ ] custom CUDA compute/transfer stream pipeline;
- [ ] layer-weight prefetch integrated into an actual target execution path;
- [ ] custom KV page allocator/prefetch policy;
- [ ] prefix cache;
- [ ] MoE expert placement;
- [ ] multi-GPU or remote-node extension;
- [ ] SSD cold-tier KV experiment;
- [ ] Bayesian optimization or learned controller;
- [ ] Prometheus/Grafana dashboard.

---

## 22. References and learning resources

The following are starting points. Pin the documentation/version that matches the backend and CUDA toolkit actually used by the team.

### Inference runtimes and heterogeneous execution

1. **llama.cpp repository and server options** — CPU/GPU offload, GPU layer count, KV types, draft model options, CPU affinity, and runtime flags:  
   https://github.com/ggml-org/llama.cpp  
   https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
2. **KTransformers heterogeneous inference documentation** — CPU/GPU mapping, expert placement, layerwise prefill, and MoE-oriented execution:  
   https://www.ktransformers.net/en/docs/technical-work/heterogeneous-inference
3. **Hugging Face Transformers KV cache guide** — dynamic/static, quantized and offloaded cache strategies:  
   https://huggingface.co/docs/transformers/main/kv_cache
4. **vLLM speculative decoding guide** — draft model, EAGLE/MTP, n-gram, dynamic speculation, and low-QPS latency context:  
   https://docs.vllm.ai/en/latest/features/speculative_decoding/
5. **vLLM benchmark CLI guide** — TTFT, TPOT, ITL, benchmark caveats and per-request measurements:  
   https://docs.vllm.ai/en/latest/benchmarking/cli/

### CUDA and profiling

6. **NVIDIA CUDA C++ Programming Guide** — streams, asynchronous execution, page-locked host memory, events and overlapping copies with kernels:  
   https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
7. **PyTorch CUDA semantics** — non-blocking copies, pinned memory, stream dependencies, and buffer lifetime/synchronization guidance:  
   https://docs.pytorch.org/docs/stable/notes/cuda.html
8. **NVIDIA Nsight Systems** — timeline-based profiling of CUDA kernels, memory copies, CPU threads, and synchronization:  
   https://docs.nvidia.com/nsight-systems/
9. **NVIDIA Nsight Compute** — kernel-level performance analysis:  
   https://docs.nvidia.com/nsight-compute/

### Research papers

10. **Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM Inference (EMNLP 2025)** — relevant consumer-hardware heterogeneous speculation; note its GPU-draft/CPU-target design:  
    https://aclanthology.org/2025.emnlp-main.879/
11. **Efficient Memory Management for LLM Serving with PagedAttention (2023)** — block-based KV-cache management:  
    https://arxiv.org/abs/2309.06180
12. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)** — IO-aware attention and GPU memory hierarchy:  
    https://arxiv.org/abs/2205.14135
13. **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization (2024)** — low-bit KV-cache quantization methods and trade-offs:  
    https://arxiv.org/abs/2401.18079
14. **Speculative Decoding (Leviathan et al., 2023)** — foundational speculative sampling method:  
    https://arxiv.org/abs/2211.17192

---

## 23. Final recommendation

Build ADHIE as a **hardware-aware local inference planner and runtime controller**, not as a general-purpose distributed serving framework.

The highest-value core is:

1. **Automatic hardware/model/memory profiling and configuration planning.**
2. **Practical CPU/GPU heterogeneous execution with a mature backend.**
3. **A rigorous CPU-draft/GPU-target speculative decoding study with adaptive draft length.**
4. **KV precision/offload selection based on context and resource constraints.**
5. **A CUDA asynchronous layer-prefetch prototype, integrated only if measurements justify it.**

This has a realistic six-month path, divides well across three members, and produces a meaningful systems evaluation. The optional distributed-node path can remain a future extension. Your differentiator is not the number of inference features you implement; it is whether ADHIE can **choose the right combination for a given consumer machine and prove that the choice helps**.

---

*Research and documentation links checked on 10 November 2026. Backend flags and runtime APIs are version-sensitive; pin exact versions in the project repository before publishing commands or performance claims.*
