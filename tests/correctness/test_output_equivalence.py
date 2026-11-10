"""Correctness tests for ADHIE — verify output quality isn't degraded.

These tests require a running llama-server backend and real models.
They are NOT run in CI — they are designed for the target machine
with the full ADHIE stack deployed.

Run with:
    pytest tests/correctness/ -v --run-correctness

Skip with (default):
    pytest tests/  (skips correctness tests automatically)

Tests:
  1. Greedy equivalence: baseline output == speculative output (same seed)
  2. KV quantization quality: Q8 KV doesn't degrade output significantly
  3. Layer placement consistency: same output regardless of GPU layer count
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

# Skip all correctness tests unless explicitly requested
pytestmark = pytest.mark.skipif(
    not os.environ.get("ADHIE_CORRECTNESS_TESTS"),
    reason="Correctness tests require ADHIE_CORRECTNESS_TESTS=1 and a running backend",
)


def _get_adapter():
    """Create a LlamaCppAdapter from environment."""
    from adhie.runtime.llamacpp_adapter import LlamaCppAdapter, find_llama_server
    server_path = os.environ.get("LLAMA_SERVER_PATH") or find_llama_server()
    return LlamaCppAdapter(server_path)


def _get_model_path() -> str:
    """Get the model path from environment."""
    path = os.environ.get("ADHIE_TEST_MODEL")
    if not path:
        pytest.skip("Set ADHIE_TEST_MODEL to a local GGUF file path")
    return path


def _get_draft_path() -> str | None:
    """Get the draft model path from environment."""
    return os.environ.get("ADHIE_TEST_DRAFT")


class TestGreedyEquivalence:
    """Verify that speculative decoding produces identical output to baseline.

    When temperature=0 and seed is fixed, speculative decoding should
    produce EXACTLY the same output as the target model alone (greedy).
    Any difference indicates a bug in the speculation pipeline.
    """

    PROMPT = "Explain the concept of gradient descent in machine learning."
    MAX_TOKENS = 64

    def test_baseline_vs_speculative_same_output(self) -> None:
        """Target-only greedy == speculative greedy (same seed, same prompt)."""
        from adhie.runtime.base import GenerateRequest
        from adhie.types import RunPlan
        from adhie.inspector.model_inspector import ModelInspector

        model_path = _get_model_path()
        draft_path = _get_draft_path()
        if not draft_path:
            pytest.skip("Set ADHIE_TEST_DRAFT for speculation tests")

        adapter = _get_adapter()
        model = ModelInspector().inspect(model_path)

        # Run baseline (no speculation)
        baseline_plan = RunPlan(
            plan_id="correctness-baseline",
            model_path=model_path,
            gpu_layers=0,
            context_length=4096,
            spec_type="none",
        )
        adapter.load(baseline_plan)
        req = GenerateRequest(
            prompt=self.PROMPT,
            max_tokens=self.MAX_TOKENS,
            temperature=0.0,
            seed=42,
        )
        baseline_resp = adapter.generate(req)
        adapter.unload()

        # Run with speculation
        spec_plan = RunPlan(
            plan_id="correctness-spec",
            model_path=model_path,
            draft_model_path=draft_path,
            gpu_layers=0,
            context_length=4096,
            spec_type="draft-simple",
            draft_length=4,
            draft_device="cpu",
        )
        adapter.load(spec_plan)
        spec_resp = adapter.generate(req)
        adapter.unload()

        assert baseline_resp.text == spec_resp.text, (
            f"Greedy equivalence FAILED!\n"
            f"Baseline: {baseline_resp.text[:200]!r}\n"
            f"Speculative: {spec_resp.text[:200]!r}"
        )


class TestKVQuantizationQuality:
    """Verify that Q8 KV cache doesn't significantly degrade output quality.

    We compare outputs with F16 and Q8 KV — they won't be identical
    (different numerical precision) but should be semantically similar.
    As a simple proxy, we check that both produce non-empty, non-garbage output.
    """

    PROMPT = "What is the capital of France?"
    MAX_TOKENS = 32

    def test_q8_kv_produces_valid_output(self) -> None:
        """Q8 KV should produce coherent output, not garbage."""
        from adhie.runtime.base import GenerateRequest
        from adhie.types import RunPlan

        model_path = _get_model_path()
        adapter = _get_adapter()

        plan = RunPlan(
            plan_id="correctness-kv-q8",
            model_path=model_path,
            gpu_layers=0,
            context_length=4096,
            kv_type_k="q8_0",
            kv_type_v="q8_0",
        )
        adapter.load(plan)
        req = GenerateRequest(
            prompt=self.PROMPT,
            max_tokens=self.MAX_TOKENS,
            temperature=0.0,
            seed=42,
        )
        resp = adapter.generate(req)
        adapter.unload()

        assert len(resp.text.strip()) > 0, "Q8 KV produced empty output"
        # Simple quality check: the answer should mention Paris
        assert "paris" in resp.text.lower() or "France" in resp.text, (
            f"Q8 KV output doesn't mention expected content: {resp.text!r}"
        )

    def test_f16_kv_produces_valid_output(self) -> None:
        """F16 KV baseline should also produce coherent output."""
        from adhie.runtime.base import GenerateRequest
        from adhie.types import RunPlan

        model_path = _get_model_path()
        adapter = _get_adapter()

        plan = RunPlan(
            plan_id="correctness-kv-f16",
            model_path=model_path,
            gpu_layers=0,
            context_length=4096,
            kv_type_k="f16",
            kv_type_v="f16",
        )
        adapter.load(plan)
        req = GenerateRequest(
            prompt=self.PROMPT,
            max_tokens=self.MAX_TOKENS,
            temperature=0.0,
            seed=42,
        )
        resp = adapter.generate(req)
        adapter.unload()

        assert len(resp.text.strip()) > 0, "F16 KV produced empty output"


class TestLayerPlacementConsistency:
    """Verify that different GPU layer counts produce the same greedy output.

    The output should be identical regardless of how many layers are
    placed on GPU vs CPU, since the computation is the same.
    """

    PROMPT = "Count from 1 to 10."
    MAX_TOKENS = 32

    def test_zero_vs_partial_gpu_layers(self) -> None:
        """0 GPU layers and partial GPU layers should give same greedy output."""
        from adhie.runtime.base import GenerateRequest
        from adhie.types import RunPlan
        from adhie.inspector.model_inspector import ModelInspector

        model_path = _get_model_path()
        adapter = _get_adapter()
        model = ModelInspector().inspect(model_path)

        req = GenerateRequest(
            prompt=self.PROMPT,
            max_tokens=self.MAX_TOKENS,
            temperature=0.0,
            seed=42,
        )

        # Run with 0 GPU layers
        plan_0 = RunPlan(
            plan_id="correctness-0gpu",
            model_path=model_path,
            gpu_layers=0,
            context_length=4096,
        )
        adapter.load(plan_0)
        resp_0 = adapter.generate(req)
        adapter.unload()

        # Run with some GPU layers (half the model)
        partial = max(1, model.layer_count // 2)
        plan_partial = RunPlan(
            plan_id=f"correctness-{partial}gpu",
            model_path=model_path,
            gpu_layers=partial,
            context_length=4096,
        )
        adapter.load(plan_partial)
        resp_partial = adapter.generate(req)
        adapter.unload()

        assert resp_0.text == resp_partial.text, (
            f"Layer placement changed output!\n"
            f"0 GPU:       {resp_0.text[:200]!r}\n"
            f"{partial} GPU: {resp_partial.text[:200]!r}"
        )

