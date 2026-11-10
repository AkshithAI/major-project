# ADHIE Architecture

See [Project Specification](./project_specification.md) for the full design.

## System Overview

ADHIE is structured as a pipeline of independent subsystems that communicate
through shared dataclasses defined in `src/adhie/types.py`.

```
HardwareProfiler → HardwareProfile ─┐
ModelInspector   → ModelProfile    ──┤
WorkloadConfig   ───────────────────┤
                                     ↓
                              MemoryEstimator → MemoryEstimate
                                     ↓
                              PlanGenerator → list[RunPlan]
                                     ↓
                              Autotuner → best RunPlan
                                     ↓
                              RuntimeAdapter (llama.cpp)
                                     ↓
                              TelemetryCollector → InferenceMetrics
```

## Module Boundaries

| Module | Source | Owner |
|--------|--------|-------|
| Hardware Profiler | `src/adhie/profiler/` | M3 |
| Model Inspector | `src/adhie/inspector/` | M1 |
| Memory Estimator | `src/adhie/estimator/` | M3 |
| Plan Generator | `src/adhie/planner/` | M3 |
| Runtime Adapter | `src/adhie/runtime/` | M1 |
| Speculative Decoding | `src/adhie/speculative/` | M2 |
| Telemetry | `src/adhie/telemetry/` | M1 |
| CLI | `src/adhie/cli/` | All |

