"""Tests for the Model Resolver — HuggingFace ID parsing and file selection."""

from __future__ import annotations

import pytest

from adhie.inspector.model_resolver import (
    is_hf_identifier,
    parse_model_ref,
    _select_gguf_file,
    QUANT_PREFERENCE,
)


class TestIsHfIdentifier:
    """Test HuggingFace identifier detection."""

    def test_standard_hf_repo(self) -> None:
        assert is_hf_identifier("Qwen/Qwen2.5-14B-Instruct-GGUF") is True

    def test_hf_repo_with_quant_hint(self) -> None:
        assert is_hf_identifier("Qwen/Qwen2.5-14B-Instruct-GGUF") is True

    def test_hf_various_orgs(self) -> None:
        assert is_hf_identifier("TheBloke/Llama-2-7B-GGUF") is True
        assert is_hf_identifier("meta-llama/Llama-3.1-8B-Instruct-GGUF") is True
        assert is_hf_identifier("bartowski/gemma-2-9b-it-GGUF") is True

    def test_local_absolute_path(self) -> None:
        assert is_hf_identifier("/home/user/models/model.gguf") is False

    def test_local_relative_path(self) -> None:
        assert is_hf_identifier("./models/model.gguf") is False
        assert is_hf_identifier("../model.gguf") is False

    def test_local_home_path(self) -> None:
        assert is_hf_identifier("~/models/model.gguf") is False

    def test_gguf_extension(self) -> None:
        assert is_hf_identifier("model.gguf") is False

    def test_no_slash(self) -> None:
        assert is_hf_identifier("just-a-name") is False

    def test_too_many_slashes(self) -> None:
        assert is_hf_identifier("org/sub/model") is False


class TestParseModelRef:
    """Test model reference parsing."""

    def test_plain_hf_repo(self) -> None:
        repo, hint = parse_model_ref("Qwen/Qwen2.5-14B-Instruct-GGUF")
        assert repo == "Qwen/Qwen2.5-14B-Instruct-GGUF"
        assert hint is None

    def test_hf_repo_with_quant(self) -> None:
        repo, hint = parse_model_ref("Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m")
        assert repo == "Qwen/Qwen2.5-14B-Instruct-GGUF"
        assert hint == "q4_k_m"

    def test_hf_repo_with_uppercase_quant(self) -> None:
        repo, hint = parse_model_ref("Qwen/Qwen2.5-14B-Instruct-GGUF:Q4_K_M")
        assert repo == "Qwen/Qwen2.5-14B-Instruct-GGUF"
        assert hint == "q4_k_m"

    def test_local_path_unchanged(self) -> None:
        repo, hint = parse_model_ref("/models/target.gguf")
        assert repo == "/models/target.gguf"
        assert hint is None

    def test_local_path_with_colon_in_name(self) -> None:
        # Edge case: local path should not be split on colon
        repo, hint = parse_model_ref("/path/to/model.gguf")
        assert repo == "/path/to/model.gguf"
        assert hint is None


class TestSelectGgufFile:
    """Test GGUF file selection logic."""

    def test_single_file_returns_it(self) -> None:
        files = ["model-q4_k_m.gguf"]
        result = _select_gguf_file(files, None, interactive=False)
        assert result == "model-q4_k_m.gguf"

    def test_quant_hint_exact_match(self) -> None:
        files = [
            "model-Q2_K.gguf",
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
            "model-F16.gguf",
        ]
        result = _select_gguf_file(files, "q4_k_m", interactive=False)
        assert result == "model-Q4_K_M.gguf"

    def test_quant_hint_q8(self) -> None:
        files = [
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
        ]
        result = _select_gguf_file(files, "q8_0", interactive=False)
        assert result == "model-Q8_0.gguf"

    def test_quant_hint_no_match_raises(self) -> None:
        files = ["model-Q4_K_M.gguf", "model-Q8_0.gguf"]
        with pytest.raises(ValueError, match="did not match"):
            _select_gguf_file(files, "f32", interactive=False)

    def test_auto_select_prefers_q4_k_m(self) -> None:
        files = [
            "model-Q2_K.gguf",
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
            "model-F16.gguf",
        ]
        result = _select_gguf_file(files, None, interactive=False)
        assert result == "model-Q4_K_M.gguf"

    def test_auto_select_falls_back_to_q5_k_m(self) -> None:
        files = [
            "model-Q5_K_M.gguf",
            "model-Q8_0.gguf",
            "model-F16.gguf",
        ]
        result = _select_gguf_file(files, None, interactive=False)
        assert result == "model-Q5_K_M.gguf"

    def test_auto_select_falls_back_to_first(self) -> None:
        """When no quant pattern is recognized, pick the first file."""
        files = [
            "model-custom-v1.gguf",
            "model-custom-v2.gguf",
        ]
        result = _select_gguf_file(files, None, interactive=False)
        assert result == "model-custom-v1.gguf"

    def test_quant_hint_with_dashes(self) -> None:
        """Quant hint with dashes should match underscored filenames."""
        files = ["model-Q4_K_M.gguf", "model-Q8_0.gguf"]
        result = _select_gguf_file(files, "q4-k-m", interactive=False)
        assert result == "model-Q4_K_M.gguf"


class TestQuantPreference:
    """Validate the quant preference order."""

    def test_q4_k_m_is_first(self) -> None:
        assert QUANT_PREFERENCE[0] == "q4_k_m"

    def test_f16_is_present(self) -> None:
        assert "f16" in QUANT_PREFERENCE

    def test_no_duplicates(self) -> None:
        assert len(QUANT_PREFERENCE) == len(set(QUANT_PREFERENCE))

