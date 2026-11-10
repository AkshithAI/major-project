"""Tests for the Model Inspector.

Since we may not have a real GGUF file in CI, these tests use mocking
for the GGUF reader and test the metadata-to-profile logic.

Includes tests for the critical _read_gguf_metadata method that must
correctly dereference field.parts[field.data[i]] to extract values
from the gguf library's ReaderField structure.
"""

from __future__ import annotations

from collections import namedtuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from adhie.inspector.model_inspector import (
    ModelInspector,
    QUANT_TYPE_MAP,
    _detect_split_shards,
)
from adhie.types import ModelProfile


# ------------------------------------------------------------------
# Mock GGUFReaderField matching the real gguf >= 0.9 structure:
#   ReaderField(offset, name, parts, data, types)
#
#   parts: list of numpy arrays (type tags + data payloads)
#   data:  list of int *indices* into parts pointing to actual values
#   types: list of GGUFValueType enum ints
# ------------------------------------------------------------------
ReaderField = namedtuple("ReaderField", ["offset", "name", "parts", "data", "types"])


def _make_string_field(name: str, value: str, offset: int = 0) -> ReaderField:
    """Create a mock ReaderField for a STRING value."""
    # parts[0..3] are type tags; parts[4] is the actual string data
    # data = [4] → parts[4] is the payload
    encoded = np.frombuffer(value.encode("utf-8"), dtype=np.uint8)
    return ReaderField(
        offset=offset,
        name=name,
        parts=[
            np.array([8], dtype=np.uint32),   # type tag: STRING=8
            np.array([len(value)], dtype=np.uint64),  # string length
            np.array([], dtype=np.uint8),      # padding
            np.array([], dtype=np.uint8),      # padding
            encoded,                            # the actual string bytes
        ],
        data=[4],       # index into parts → parts[4] = encoded string
        types=[8],      # GGUFValueType.STRING = 8
    )


def _make_uint32_field(name: str, value: int, offset: int = 0) -> ReaderField:
    """Create a mock ReaderField for a UINT32 value."""
    return ReaderField(
        offset=offset,
        name=name,
        parts=[
            np.array([4], dtype=np.uint32),    # type tag: UINT32=4
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.uint8),
            np.array([value], dtype=np.uint32),  # the actual value
        ],
        data=[4],       # index into parts → parts[4]
        types=[4],      # GGUFValueType.UINT32 = 4
    )


def _make_float32_field(name: str, value: float, offset: int = 0) -> ReaderField:
    """Create a mock ReaderField for a FLOAT32 value."""
    return ReaderField(
        offset=offset,
        name=name,
        parts=[
            np.array([6], dtype=np.uint32),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.uint8),
            np.array([value], dtype=np.float32),
        ],
        data=[4],
        types=[6],      # GGUFValueType.FLOAT32 = 6
    )


def _make_bool_field(name: str, value: bool, offset: int = 0) -> ReaderField:
    """Create a mock ReaderField for a BOOL value."""
    return ReaderField(
        offset=offset,
        name=name,
        parts=[
            np.array([7], dtype=np.uint32),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.uint8),
            np.array([int(value)], dtype=np.uint8),
        ],
        data=[4],
        types=[7],      # GGUFValueType.BOOL = 7
    )


def _make_string_array_field(name: str, values: list[str], offset: int = 0) -> ReaderField:
    """Create a mock ReaderField for an ARRAY of STRING values."""
    parts = [
        np.array([9], dtype=np.uint32),   # type tag: ARRAY=9
        np.array([8], dtype=np.uint32),   # element type: STRING=8
        np.array([len(values)], dtype=np.uint64),  # array length
    ]
    data_indices = []
    for s in values:
        idx = len(parts)
        parts.append(np.array([len(s)], dtype=np.uint64))  # string length
        idx_data = len(parts)
        parts.append(np.frombuffer(s.encode("utf-8"), dtype=np.uint8))
        data_indices.append(idx_data)
    return ReaderField(
        offset=offset,
        name=name,
        parts=parts,
        data=data_indices,
        types=[9, 8],   # ARRAY of STRING
    )


def _make_mock_tensor(tensor_type: int, shape: list[int]):
    """Create a mock tensor object."""
    t = MagicMock()
    t.tensor_type = tensor_type
    t.shape = np.array(shape, dtype=np.uint64)
    return t


def _make_qwen2_reader_mock():
    """Build a mock GGUFReader with Qwen2.5-14B-like fields.

    Uses the real ReaderField structure where field.data contains
    indices into field.parts.
    """
    fields = {
        "general.name": _make_string_field("general.name", "Qwen2.5-14B-Instruct"),
        "general.architecture": _make_string_field("general.architecture", "qwen2"),
        "qwen2.block_count": _make_uint32_field("qwen2.block_count", 40),
        "qwen2.embedding_length": _make_uint32_field("qwen2.embedding_length", 5120),
        "qwen2.feed_forward_length": _make_uint32_field("qwen2.feed_forward_length", 13824),
        "qwen2.vocab_size": _make_uint32_field("qwen2.vocab_size", 152064),
        "qwen2.attention.head_count": _make_uint32_field("qwen2.attention.head_count", 40),
        "qwen2.attention.head_count_kv": _make_uint32_field("qwen2.attention.head_count_kv", 8),
        "qwen2.attention.key_length": _make_uint32_field("qwen2.attention.key_length", 128),
        "qwen2.context_length": _make_uint32_field("qwen2.context_length", 32768),
        "tokenizer.ggml.model": _make_string_field("tokenizer.ggml.model", "gpt2"),
        "tokenizer.chat_template": _make_string_field(
            "tokenizer.chat_template", "{% for msg in messages %}..."
        ),
    }

    # Create mock tensors — mostly q4_k_m (type=15), a few f16 (type=1)
    tensors = []
    for _ in range(300):
        tensors.append(_make_mock_tensor(15, [4096, 4096]))  # q4_k_m
    for _ in range(10):
        tensors.append(_make_mock_tensor(1, [4096, 4096]))   # f16

    reader = MagicMock()
    reader.fields = fields
    reader.tensors = tensors
    return reader


class TestReadGGUFMetadata:
    """Test _read_gguf_metadata correctly dereferences field.parts[field.data[i]].

    This is the critical test class for the bug where field.data (a list
    of indices) was being treated as the actual values, causing all
    metadata (name, architecture, layer count, etc.) to be wrong.
    """

    def setup_method(self) -> None:
        self.inspector = ModelInspector()

    @patch("adhie.inspector.model_inspector.GGUFReader", create=True)
    def test_string_fields_extracted_correctly(self, mock_reader_cls) -> None:
        """String values must be decoded from field.parts[field.data[0]]."""
        reader = _make_qwen2_reader_mock()
        mock_reader_cls.return_value = reader

        # We need to patch the import inside the method
        with patch.dict("sys.modules", {"gguf": MagicMock(GGUFReader=mock_reader_cls)}):
            metadata = self.inspector._read_gguf_metadata("/fake/path.gguf")

        assert metadata["general.name"] == "Qwen2.5-14B-Instruct"
        assert metadata["general.architecture"] == "qwen2"
        assert metadata["tokenizer.ggml.model"] == "gpt2"
        assert "{% for msg in messages %}" in metadata["tokenizer.chat_template"]

    @patch("adhie.inspector.model_inspector.GGUFReader", create=True)
    def test_uint32_fields_extracted_correctly(self, mock_reader_cls) -> None:
        """Numeric values must be read as Python ints, not index values."""
        reader = _make_qwen2_reader_mock()
        mock_reader_cls.return_value = reader

        with patch.dict("sys.modules", {"gguf": MagicMock(GGUFReader=mock_reader_cls)}):
            metadata = self.inspector._read_gguf_metadata("/fake/path.gguf")

        # These MUST NOT be the index value (e.g. 4), they must be actual values
        assert metadata["qwen2.block_count"] == 40
        assert metadata["qwen2.embedding_length"] == 5120
        assert metadata["qwen2.feed_forward_length"] == 13824
        assert metadata["qwen2.vocab_size"] == 152064
        assert metadata["qwen2.attention.head_count"] == 40
        assert metadata["qwen2.attention.head_count_kv"] == 8
        assert metadata["qwen2.attention.key_length"] == 128
        assert metadata["qwen2.context_length"] == 32768

    @patch("adhie.inspector.model_inspector.GGUFReader", create=True)
    def test_tensor_aggregation(self, mock_reader_cls) -> None:
        """Tensor counts and params should be aggregated from reader.tensors."""
        reader = _make_qwen2_reader_mock()
        mock_reader_cls.return_value = reader

        with patch.dict("sys.modules", {"gguf": MagicMock(GGUFReader=mock_reader_cls)}):
            metadata = self.inspector._read_gguf_metadata("/fake/path.gguf")

        assert metadata["_tensor_count"] == 310
        assert metadata["_tensor_quant_types"][15] == 300  # q4_k_m
        assert metadata["_tensor_quant_types"][1] == 10    # f16
        # 310 tensors × 4096 × 4096 each
        assert metadata["_total_params"] == 310 * 4096 * 4096

    @patch("adhie.inspector.model_inspector.GGUFReader", create=True)
    def test_end_to_end_inspect_with_mock(self, mock_reader_cls) -> None:
        """Full pipeline: _read_gguf_metadata → _populate_from_metadata → profile.

        This validates that the metadata extraction produces values that
        _populate_from_metadata can use correctly.
        """
        reader = _make_qwen2_reader_mock()
        mock_reader_cls.return_value = reader

        with patch.dict("sys.modules", {"gguf": MagicMock(GGUFReader=mock_reader_cls)}):
            metadata = self.inspector._read_gguf_metadata("/fake/path.gguf")

        profile = ModelProfile(model_path="/fake/path.gguf", file_size_bytes=9_000_000_000)
        self.inspector._populate_from_metadata(profile, metadata)

        assert profile.model_id == "Qwen2.5-14B-Instruct"
        assert profile.architecture == "qwen2"
        assert profile.layer_count == 40
        assert profile.hidden_size == 5120
        assert profile.intermediate_size == 13824
        assert profile.vocab_size == 152064
        assert profile.num_attention_heads == 40
        assert profile.num_kv_heads == 8
        assert profile.head_dim == 128
        assert profile.max_context == 32768
        assert profile.tokenizer_id == "gpt2"
        assert profile.weight_dtype == "q4_k_m"

        # Crucially, these must be non-zero so downstream estimator works
        assert profile.weight_bytes_per_layer() > 0
        assert profile.kv_bytes_per_token(16) > 0

    @patch("adhie.inspector.model_inspector.GGUFReader", create=True)
    def test_bool_field(self, mock_reader_cls) -> None:
        """Boolean fields are correctly extracted."""
        reader = MagicMock()
        reader.fields = {
            "general.name": _make_string_field("general.name", "test"),
            "general.architecture": _make_string_field("general.architecture", "llama"),
            "some.bool.field": _make_bool_field("some.bool.field", True),
        }
        reader.tensors = []
        mock_reader_cls.return_value = reader

        with patch.dict("sys.modules", {"gguf": MagicMock(GGUFReader=mock_reader_cls)}):
            metadata = self.inspector._read_gguf_metadata("/fake/path.gguf")

        assert metadata["some.bool.field"] is True

    @patch("adhie.inspector.model_inspector.GGUFReader", create=True)
    def test_float32_field(self, mock_reader_cls) -> None:
        """Float fields are correctly extracted."""
        reader = MagicMock()
        reader.fields = {
            "general.name": _make_string_field("general.name", "test"),
            "some.float": _make_float32_field("some.float", 3.14),
        }
        reader.tensors = []
        mock_reader_cls.return_value = reader

        with patch.dict("sys.modules", {"gguf": MagicMock(GGUFReader=mock_reader_cls)}):
            metadata = self.inspector._read_gguf_metadata("/fake/path.gguf")

        assert abs(metadata["some.float"] - 3.14) < 0.01

    @patch("adhie.inspector.model_inspector.GGUFReader", create=True)
    def test_string_array_field(self, mock_reader_cls) -> None:
        """Array of strings (e.g., tokenizer.ggml.tokens) are correctly extracted."""
        reader = MagicMock()
        reader.fields = {
            "general.name": _make_string_field("general.name", "test"),
            "tokenizer.ggml.tokens": _make_string_array_field(
                "tokenizer.ggml.tokens", ["hello", "world", "test"]
            ),
        }
        reader.tensors = []
        mock_reader_cls.return_value = reader

        with patch.dict("sys.modules", {"gguf": MagicMock(GGUFReader=mock_reader_cls)}):
            metadata = self.inspector._read_gguf_metadata("/fake/path.gguf")

        assert metadata["tokenizer.ggml.tokens"] == ["hello", "world", "test"]

    @patch("adhie.inspector.model_inspector.GGUFReader", create=True)
    def test_field_data_indices_not_used_as_values(self, mock_reader_cls) -> None:
        """Regression: field.data=[4] must NOT produce value=4.

        This was the original bug: field.data is a list of indices, not values.
        When data=[4], the old code returned 4 as the value instead of
        dereferencing parts[4] to get the actual value.
        """
        reader = MagicMock()
        # field.data=[4] but actual value at parts[4] is 40
        reader.fields = {
            "general.name": _make_string_field("general.name", "TestModel"),
            "general.architecture": _make_string_field("general.architecture", "qwen2"),
            "qwen2.block_count": _make_uint32_field("qwen2.block_count", 40),
        }
        reader.tensors = []
        mock_reader_cls.return_value = reader

        with patch.dict("sys.modules", {"gguf": MagicMock(GGUFReader=mock_reader_cls)}):
            metadata = self.inspector._read_gguf_metadata("/fake/path.gguf")

        # The value MUST be 40 (from parts[4]), NOT 4 (the index in data)
        assert metadata["qwen2.block_count"] == 40
        assert metadata["qwen2.block_count"] != 4  # explicit regression check
        assert metadata["general.name"] == "TestModel"
        assert metadata["general.name"] != "4"  # explicit regression check


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
