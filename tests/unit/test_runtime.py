"""Tests for the LlamaCpp Runtime Adapter — CLI arg building and lifecycle."""

from __future__ import annotations

import pytest

from adhie.runtime.base import GenerateRequest, GenerateResponse, RuntimeAdapter
from adhie.runtime.llamacpp_adapter import LlamaCppAdapter, find_llama_server
from adhie.types import RunPlan


class TestGenerateRequest:
    """Test GenerateRequest dataclass defaults."""

    def test_defaults(self) -> None:
        req = GenerateRequest()
        assert req.prompt == ""
        assert req.max_tokens == 256
        assert req.temperature == 0.0
        assert req.stream is True
        assert req.seed == 42

    def test_custom_values(self) -> None:
        req = GenerateRequest(
            prompt="Hello",
            max_tokens=128,
            temperature=0.7,
            top_p=0.9,
            seed=123,
            stream=False,
            stop=["END"],
        )
        assert req.prompt == "Hello"
        assert req.max_tokens == 128
        assert req.stop == ["END"]

    def test_messages_format(self) -> None:
        req = GenerateRequest(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert len(req.messages) == 2


class TestGenerateResponse:
    """Test GenerateResponse dataclass."""

    def test_defaults(self) -> None:
        resp = GenerateResponse()
        assert resp.text == ""
        assert resp.finish_reason == "stop"
        assert resp.metrics is None

    def test_custom(self) -> None:
        resp = GenerateResponse(
            text="Hello world",
            prompt_tokens=10,
            output_tokens=5,
            finish_reason="length",
        )
        assert resp.output_tokens == 5


class TestBuildArgs:
    """Test _build_args translates RunPlan into correct CLI arguments."""

    def _make_adapter(self) -> LlamaCppAdapter:
        return LlamaCppAdapter("/path/to/llama-server", host="127.0.0.1", port=8787)

    def _make_plan(self, **kwargs) -> RunPlan:
        defaults = dict(
            model_path="/models/test.gguf",
            backend="llama.cpp",
            gpu_layers=20,
            cpu_threads=4,
            context_length=4096,
            batch_size=512,
            ubatch_size=128,
            kv_type_k="f16",
            kv_type_v="f16",
        )
        defaults.update(kwargs)
        return RunPlan(**defaults)

    def test_basic_args(self) -> None:
        adapter = self._make_adapter()
        plan = self._make_plan()
        args = adapter._build_args(plan)
        assert args[0] == "/path/to/llama-server"
        assert "--model" in args
        assert args[args.index("--model") + 1] == "/models/test.gguf"
        assert "--gpu-layers" in args
        assert args[args.index("--gpu-layers") + 1] == "20"
        assert "--ctx-size" in args
        assert args[args.index("--ctx-size") + 1] == "4096"
        assert "--threads" in args
        assert args[args.index("--threads") + 1] == "4"
        assert "--flash-attn" in args

    def test_kv_type_not_f16(self) -> None:
        adapter = self._make_adapter()
        plan = self._make_plan(kv_type_k="q8_0", kv_type_v="q8_0")
        args = adapter._build_args(plan)
        assert "--cache-type-k" in args
        assert args[args.index("--cache-type-k") + 1] == "q8_0"
        assert "--cache-type-v" in args
        assert args[args.index("--cache-type-v") + 1] == "q8_0"

    def test_kv_type_f16_omitted(self) -> None:
        adapter = self._make_adapter()
        plan = self._make_plan(kv_type_k="f16", kv_type_v="f16")
        args = adapter._build_args(plan)
        assert "--cache-type-k" not in args
        assert "--cache-type-v" not in args

    def test_speculative_decoding_args(self) -> None:
        adapter = self._make_adapter()
        plan = self._make_plan(
            draft_model_path="/models/draft.gguf",
            draft_device="gpu",
            spec_type="sd",
            draft_length=8,
        )
        args = adapter._build_args(plan)
        assert "--model-draft" in args
        assert args[args.index("--model-draft") + 1] == "/models/draft.gguf"
        assert "--spec-type" in args
        assert args[args.index("--spec-type") + 1] == "sd"
        assert "--spec-draft-n-max" in args
        assert args[args.index("--spec-draft-n-max") + 1] == "8"

    def test_no_draft_when_none(self) -> None:
        adapter = self._make_adapter()
        plan = self._make_plan()  # defaults: no draft
        args = adapter._build_args(plan)
        assert "--model-draft" not in args
        assert "--spec-type" not in args

    def test_draft_on_cpu(self) -> None:
        adapter = self._make_adapter()
        plan = self._make_plan(
            draft_model_path="/models/draft.gguf",
            draft_device="cpu",
            spec_type="sd",
            draft_threads=2,
        )
        args = adapter._build_args(plan)
        assert "--device-draft" in args
        assert args[args.index("--device-draft") + 1] == "none"
        assert "--threads-draft" in args
        assert args[args.index("--threads-draft") + 1] == "2"

    def test_host_and_port(self) -> None:
        adapter = LlamaCppAdapter("/path/llama-server", host="0.0.0.0", port=9999)
        plan = self._make_plan()
        args = adapter._build_args(plan)
        assert "--host" in args
        assert args[args.index("--host") + 1] == "0.0.0.0"
        assert "--port" in args
        assert args[args.index("--port") + 1] == "9999"


class TestAdapterLifecycle:
    """Test adapter construction and state."""

    def test_construction(self) -> None:
        adapter = LlamaCppAdapter("/path/to/server")
        assert adapter.server_path == "/path/to/server"
        assert adapter.process is None
        assert adapter.last_metrics() is None

    def test_health_check_when_not_running(self) -> None:
        """health_check returns False when no server is running."""
        adapter = LlamaCppAdapter("/path/to/server", port=19999)
        assert adapter.health_check() is False

    def test_unload_when_not_loaded(self) -> None:
        """unload() should be a no-op when nothing is loaded."""
        adapter = LlamaCppAdapter("/path/to/server")
        adapter.unload()  # should not raise

    def test_generate_without_load_raises(self) -> None:
        """generate() should fail if server isn't running."""
        adapter = LlamaCppAdapter("/path/to/server", port=19999)
        req = GenerateRequest(prompt="test")
        with pytest.raises(RuntimeError, match="not ready"):
            adapter.generate(req)


class TestRuntimeAdapterABC:
    """Verify RuntimeAdapter is a proper ABC."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            RuntimeAdapter()  # type: ignore

    def test_llamacpp_is_subclass(self) -> None:
        assert issubclass(LlamaCppAdapter, RuntimeAdapter)

