"""Tests for the Model Inspector.

Since we may not have a real GGUF file in CI, these tests use mocking
for the GGUF reader and test the metadata-to-profile logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adhie.inspector.model_inspector import (
    ModelInspector,
    QUANT_TYPE_MAP,
    _detect_split_shards,
)
from adhie.types import ModelProfile


class TestModelInspectorMetadataPopulation:
    """Test _populate_from_metadata with synthetic metadata dicts."""

    def setup_method(self) -> None:
        self.inspector = ModelInspector()

    def _make_qwen2_metadata(self) -> dict:
        """Synthetic metadata matching a Qwen2.5-14B model."""
        return {
            "general.name": "Qwen2.5-14B-Instruct",
            "general.architecture": "qwen2",
            "qwen2.block_count": 40,
            "qwen2.embedding_length": 5120,
            "qwen2.feed_forward_length": 13824,
            "qwen2.vocab_size": 152064,
            "qwen2.attention.head_count": 40,
            "qwen2.attention.head_count_kv": 8,
            "qwen2.attention.key_length": 128,
            "qwen2.context_length": 32768,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.chat_template": "{% for msg in messages %}...",
            "_tensor_quant_types": {15: 300, 1: 10},  # Mostly q4_k_m
            "_tensor_count": 310,
            "_total_params": 14_700_000_000,
        }

    def test_basic_architecture_fields(self) -> None:
        meta = self._make_qwen2_metadata()
        profile = ModelProfile()
        self.inspector._populate_from_metadata(profile, meta)

        assert profile.model_id == "Qwen2.5-14B-Instruct"
        assert profile.architecture == "qwen2"
        assert profile.layer_count == 40
        assert profile.hidden_size == 5120
        assert profile.intermediate_size == 13824
        assert profile.vocab_size == 152064

    def test_attention_config(self) -> None:
        meta = self._make_qwen2_metadata()
        profile = ModelProfile()
        self.inspector._populate_from_metadata(profile, meta)

        assert profile.num_attention_heads == 40
        assert profile.num_kv_heads == 8  # GQA
        assert profile.head_dim == 128
        assert profile.max_context == 32768
        assert profile.attention_type == "full"

    def test_quantization_detection(self) -> None:
        meta = self._make_qwen2_metadata()
        profile = ModelProfile()
        self.inspector._populate_from_metadata(profile, meta)

        assert profile.weight_dtype == "q4_k_m"
        assert profile.bits_per_weight == 4.8

    def test_parameter_count(self) -> None:
        meta = self._make_qwen2_metadata()
        profile = ModelProfile()
        self.inspector._populate_from_metadata(profile, meta)

        assert profile.parameter_count == 14_700_000_000

    def test_tokenizer_info(self) -> None:
        meta = self._make_qwen2_metadata()
        profile = ModelProfile()
        self.inspector._populate_from_metadata(profile, meta)

        assert profile.tokenizer_id == "gpt2"
        assert profile.chat_template is not None

    def test_no_moe_by_default(self) -> None:
        meta = self._make_qwen2_metadata()
        profile = ModelProfile()
        self.inspector._populate_from_metadata(profile, meta)

        assert profile.is_moe is False
        assert profile.num_experts is None

    def test_moe_detection(self) -> None:
        meta = self._make_qwen2_metadata()
        meta["qwen2.expert_count"] = 64
        meta["qwen2.expert_used_count"] = 8
        profile = ModelProfile()
        self.inspector._populate_from_metadata(profile, meta)

        assert profile.is_moe is True
        assert profile.num_experts == 64
        assert profile.num_active_experts == 8

    def test_sliding_window_detection(self) -> None:
        meta = self._make_qwen2_metadata()
        meta["qwen2.attention.sliding_window"] = 4096
        profile = ModelProfile()
        self.inspector._populate_from_metadata(profile, meta)

        assert profile.attention_type == "sliding_window"
        assert profile.sliding_window_size == 4096

    def test_head_dim_calculated_from_hidden_size(self) -> None:
        """When key_length is not in metadata, compute head_dim."""
        meta = self._make_qwen2_metadata()
        del meta["qwen2.attention.key_length"]
        profile = ModelProfile()
        self.inspector._populate_from_metadata(profile, meta)

        # 5120 / 40 = 128
        assert profile.head_dim == 128

    def test_kv_heads_defaults_to_attention_heads(self) -> None:
        """When kv heads missing, assume MHA."""
        meta = self._make_qwen2_metadata()
        del meta["qwen2.attention.head_count_kv"]
        profile = ModelProfile()
        self.inspector._populate_from_metadata(profile, meta)

        assert profile.num_kv_heads == 40  # Same as attention heads

    def test_llama_architecture(self) -> None:
        """Test with a llama-style metadata."""
        meta = {
            "general.name": "Llama-3.1-8B-Instruct",
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.feed_forward_length": 14336,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.attention.key_length": 128,
            "llama.context_length": 131072,
            "tokenizer.ggml.model": "gpt2",
            "_tensor_quant_types": {15: 250},
            "_tensor_count": 250,
            "_total_params": 8_000_000_000,
        }
        profile = ModelProfile()
        self.inspector._populate_from_metadata(profile, meta)

        assert profile.architecture == "llama"
        assert profile.layer_count == 32
        assert profile.num_kv_heads == 8


class TestDraftCompatibility:
    def setup_method(self) -> None:
        self.inspector = ModelInspector()

    def test_compatible_models(self) -> None:
        target = ModelProfile(
            tokenizer_id="gpt2",
            vocab_size=152064,
            parameter_count=14_700_000_000,
            architecture="qwen2",
        )
        draft = ModelProfile(
            tokenizer_id="gpt2",
            vocab_size=152064,
            parameter_count=500_000_000,
            architecture="qwen2",
        )
        compat, issues = self.inspector.check_draft_compatibility(target, draft)
        assert compat is True
        assert len(issues) == 0

    def test_tokenizer_mismatch(self) -> None:
        target = ModelProfile(tokenizer_id="gpt2", vocab_size=152064,
                              parameter_count=14_000_000_000)
        draft = ModelProfile(tokenizer_id="llama", vocab_size=32000,
                             parameter_count=500_000_000)
        compat, issues = self.inspector.check_draft_compatibility(target, draft)
        assert compat is False
        assert any("Tokenizer mismatch" in i for i in issues)

    def test_draft_larger_than_target(self) -> None:
        target = ModelProfile(parameter_count=8_000_000_000, tokenizer_id="gpt2",
                              vocab_size=32000)
        draft = ModelProfile(parameter_count=14_000_000_000, tokenizer_id="gpt2",
                             vocab_size=32000)
        compat, issues = self.inspector.check_draft_compatibility(target, draft)
        assert any("not smaller" in i for i in issues)


class TestQuantTypeMap:
    def test_common_types_present(self) -> None:
        assert 15 in QUANT_TYPE_MAP  # q4_k_m
        assert 8 in QUANT_TYPE_MAP   # q8_0
        assert 1 in QUANT_TYPE_MAP   # f16

    def test_q4_k_m_values(self) -> None:
        name, bits = QUANT_TYPE_MAP[15]
        assert name == "q4_k_m"
        assert bits == 4.8


class TestDetectSplitShards:
    """Test _detect_split_shards for split GGUF file detection."""

    def test_non_split_returns_empty(self, tmp_path) -> None:
        f = tmp_path / "model-q4_k_m.gguf"
        f.write_bytes(b"\x00" * 10)
        assert _detect_split_shards(str(f)) == []

    def test_split_finds_all_shards(self, tmp_path) -> None:
        """When all shards are present, returns sorted list."""
        for i in range(1, 4):
            f = tmp_path / f"model-q4_k_m-{i:05d}-of-00003.gguf"
            f.write_bytes(b"\x00" * 10)
        result = _detect_split_shards(
            str(tmp_path / "model-q4_k_m-00001-of-00003.gguf")
        )
        assert len(result) == 3
        assert "00001" in result[0]
        assert "00003" in result[2]

    def test_split_partial_shards(self, tmp_path) -> None:
        """When only some shards exist, returns only existing ones."""
        for i in [1, 3]:
            f = tmp_path / f"model-q4_k_m-{i:05d}-of-00003.gguf"
            f.write_bytes(b"\x00" * 10)
        result = _detect_split_shards(
            str(tmp_path / "model-q4_k_m-00001-of-00003.gguf")
        )
        assert len(result) == 2  # shards 1 and 3


class TestRefineSplitQuant:
    """Test _refine_split_quant corrects quant from filename."""

    def test_corrects_wrong_quant(self) -> None:
        inspector = ModelInspector()
        profile = ModelProfile(weight_dtype="q3_k_m", bits_per_weight=3.9)
        inspector._refine_split_quant(
            profile,
            "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
        )
        assert profile.weight_dtype == "q4_k_m"
        assert profile.bits_per_weight == 4.8

    def test_no_change_when_correct(self) -> None:
        inspector = ModelInspector()
        profile = ModelProfile(weight_dtype="q4_k_m", bits_per_weight=4.8)
        inspector._refine_split_quant(
            profile,
            "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
        )
        assert profile.weight_dtype == "q4_k_m"

    def test_no_quant_in_filename(self) -> None:
        inspector = ModelInspector()
        profile = ModelProfile(weight_dtype="q3_k_m", bits_per_weight=3.9)
        inspector._refine_split_quant(profile, "model-custom-00001-of-00002.gguf")
        # Should not change since we can't extract a quant from "custom"
        assert profile.weight_dtype == "q3_k_m"
