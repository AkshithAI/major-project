#!/usr/bin/env python3
"""Standalone Benchmark Script for ADHIE.

Usage:
    python benchmarks/run_benchmark.py MODEL_REF [--context 8192] [--runs 3] [--output results/]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adhie.benchmark.runner import BenchmarkRunner
from adhie.inspector.model_inspector import ModelInspector
from adhie.inspector.model_resolver import resolve_model
from adhie.planner.plan_generator import PlanGenerator
from adhie.profiler.hardware_profiler import HardwareProfiler
from adhie.runtime.llamacpp_adapter import LlamaCppAdapter, find_llama_server
from adhie.types import WorkloadConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="ADHIE Benchmark Runner")
    parser.add_argument("model", help="Path to GGUF model or HuggingFace repo identifier")
    parser.add_argument("--context", "-c", type=int, default=4096, help="Context length")
    parser.add_argument("--tokens", "-t", type=int, default=128, help="Output tokens to generate")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Timed runs per configuration")
    parser.add_argument("--warmup", "-w", type=int, default=1, help="Warm-up runs")
    parser.add_argument("--output", "-o", default="benchmarks/results", help="Directory for JSON results")
    parser.add_argument("--server", default=None, help="Path to llama-server binary")
    args = parser.parse_args()

    # 1. Resolve model & inspect
    model_path = resolve_model(args.model)
    model = ModelInspector().inspect(model_path)

    # 2. Profile hardware
    hw = HardwareProfiler(verbose=False).profile()

    # 3. Locate backend
    server_path = args.server or find_llama_server()
    adapter = LlamaCppAdapter(server_path=server_path)

    # 4. Generate plans
    workload = WorkloadConfig(context_length=args.context, output_tokens=args.tokens)
    generator = PlanGenerator()
    candidates = generator.generate_candidates(hw, model, workload)

    if not candidates:
        print("ERROR: No feasible plans found for this model/hardware combination.")
        sys.exit(1)

    best_plan = candidates[0]

    # 5. Run benchmark
    prompt = "Write a comprehensive explanation of how heterogeneous CPU and GPU computing works for large language models."
    runner = BenchmarkRunner(adapter, warmup_runs=args.warmup, timed_runs=args.runs)

    try:
        result = runner.run_single(
            plan=best_plan,
            prompt=prompt,
            max_tokens=args.tokens,
            hardware_profile=hw,
            model_profile=model,
        )
        runner.print_comparison([result])

        # 6. Save JSON
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{result.run_id}.json"
        result.to_json(out_file)
        print(f"\nResult saved to: {out_file}")

    finally:
        adapter.unload()


if __name__ == "__main__":
    main()

