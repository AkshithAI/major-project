"""Tests for the Model Resolver — HuggingFace ID parsing, file selection, and split GGUF handling."""

from __future__ import annotations

import pytest

from adhie.inspector.model_resolver import (
    is_hf_identifier,
    parse_model_ref,
    _select_gguf_file,
    _parse_split_shard,
    _group_split_files,
    _extract_quant,
    _shard1_of_group,
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

    def test_windows_drive_path(self) -> None:
        """Windows paths like D:\\models should not be treated as HF identifiers."""
        assert is_hf_identifier("D:/models/model.gguf") is False


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
        repo, hint = parse_model_ref("/path/to/model.gguf")
        assert repo == "/path/to/model.gguf"
        assert hint is None


class TestParseSplitShard:
    """Test split shard filename parsing."""

    def test_standard_split_shard(self) -> None:
        result = _parse_split_shard("model-q4_k_m-00001-of-00003.gguf")
        assert result == ("model-q4_k_m", 1, 3)

    def test_shard_2_of_3(self) -> None:
        result = _parse_split_shard("model-q4_k_m-00002-of-00003.gguf")
        assert result == ("model-q4_k_m", 2, 3)

    def test_non_split_file(self) -> None:
        result = _parse_split_shard("model-q4_k_m.gguf")
        assert result is None

    def test_real_qwen_filename(self) -> None:
        result = _parse_split_shard(
            "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
        )
        assert result == ("qwen2.5-14b-instruct-q4_k_m", 1, 3)

    def test_case_insensitive(self) -> None:
        result = _parse_split_shard("Model-Q4_K_M-00001-of-00002.GGUF")
        assert result is not None
        assert result[1] == 1
        assert result[2] == 2


class TestGroupSplitFiles:
    """Test grouping of split GGUF files."""

    def test_mixed_singles_and_splits(self) -> None:
        files = [
            "model-q4_k_m.gguf",                        # single
            "model-q8_0-00001-of-00002.gguf",           # split shard 1
            "model-q8_0-00002-of-00002.gguf",           # split shard 2
            "model-f16.gguf",                            # single
        ]
        singles, groups = _group_split_files(files)
        assert set(singles) == {"model-q4_k_m.gguf", "model-f16.gguf"}
        assert "model-q8_0" in groups
        assert len(groups["model-q8_0"]) == 2

    def test_all_singles(self) -> None:
        files = ["model-q4_k_m.gguf", "model-q8_0.gguf"]
        singles, groups = _group_split_files(files)
        assert len(singles) == 2
        assert len(groups) == 0

    def test_all_splits(self) -> None:
        files = [
            "model-q4_k_m-00001-of-00003.gguf",
            "model-q4_k_m-00002-of-00003.gguf",
            "model-q4_k_m-00003-of-00003.gguf",
        ]
        singles, groups = _group_split_files(files)
        assert len(singles) == 0
        assert "model-q4_k_m" in groups
        assert len(groups["model-q4_k_m"]) == 3

    def test_multiple_split_groups(self) -> None:
        files = [
            "model-q4_k_m-00001-of-00002.gguf",
            "model-q4_k_m-00002-of-00002.gguf",
            "model-q8_0-00001-of-00003.gguf",
            "model-q8_0-00002-of-00003.gguf",
            "model-q8_0-00003-of-00003.gguf",
        ]
        singles, groups = _group_split_files(files)
        assert len(singles) == 0
        assert len(groups) == 2
        assert len(groups["model-q4_k_m"]) == 2
        assert len(groups["model-q8_0"]) == 3


class TestExtractQuant:
    """Test quant type extraction from filenames."""

    def test_q4_k_m(self) -> None:
        assert _extract_quant("model-q4_k_m.gguf") == "q4_k_m"

    def test_q8_0(self) -> None:
        assert _extract_quant("model-Q8_0.gguf") == "q8_0"

    def test_f16(self) -> None:
        assert _extract_quant("model-F16.gguf") == "f16"

    def test_from_split_stem(self) -> None:
        assert _extract_quant("qwen2.5-14b-instruct-q4_k_m") == "q4_k_m"

    def test_no_quant(self) -> None:
        assert _extract_quant("model-custom.gguf") is None

    def test_iq4_xs(self) -> None:
        assert _extract_quant("model-IQ4_XS.gguf") == "iq4_xs"


class TestSelectGgufFileNew:
    """Test the updated _select_gguf_file with split support."""

    def _call(self, singles, split_groups, quant_hint=None, interactive=False):
        """Helper to call _select_gguf_file and return (target, is_split, files)."""
        return _select_gguf_file(singles, split_groups, quant_hint, interactive)

    def test_single_file_returns_it(self) -> None:
        target, is_split, files = self._call(["model-q4_k_m.gguf"], {})
        assert target == "model-q4_k_m.gguf"
        assert is_split is False

    def test_quant_hint_matches_single(self) -> None:
        singles = ["model-Q4_K_M.gguf", "model-Q8_0.gguf"]
        target, is_split, files = self._call(singles, {}, "q4_k_m")
        assert target == "model-Q4_K_M.gguf"
        assert is_split is False

    def test_quant_hint_matches_split(self) -> None:
        singles = ["model-q4_k_m.gguf"]
        split_groups = {
            "model-q8_0": [
                "model-q8_0-00001-of-00002.gguf",
                "model-q8_0-00002-of-00002.gguf",
            ]
        }
        target, is_split, files = self._call(singles, split_groups, "q8_0")
        assert is_split is True
        assert len(files) == 2
        assert "00001" in target

    def test_auto_select_prefers_single_over_split(self) -> None:
        """When both a single and split exist for the same quant, prefer single."""
        singles = ["model-q4_k_m.gguf"]
        split_groups = {
            "model-q4_k_m": [
                "model-q4_k_m-00001-of-00002.gguf",
                "model-q4_k_m-00002-of-00002.gguf",
            ]
        }
        target, is_split, files = self._call(singles, split_groups)
        assert target == "model-q4_k_m.gguf"
        assert is_split is False

    def test_auto_select_picks_split_when_only_option(self) -> None:
        singles = []
        split_groups = {
            "model-q4_k_m": [
                "model-q4_k_m-00001-of-00003.gguf",
                "model-q4_k_m-00002-of-00003.gguf",
                "model-q4_k_m-00003-of-00003.gguf",
            ]
        }
        target, is_split, files = self._call(singles, split_groups)
        assert is_split is True
        assert len(files) == 3

    def test_quant_hint_no_match_raises(self) -> None:
        singles = ["model-Q4_K_M.gguf", "model-Q8_0.gguf"]
        with pytest.raises(ValueError, match="did not match"):
            self._call(singles, {}, "f32")

    def test_auto_select_by_preference_order(self) -> None:
        singles = ["model-Q8_0.gguf", "model-Q5_K_M.gguf"]
        target, is_split, files = self._call(singles, {})
        # q5_k_m comes before q8_0 in QUANT_PREFERENCE
        assert target == "model-Q5_K_M.gguf"

    def test_quant_hint_with_dashes(self) -> None:
        singles = ["model-Q4_K_M.gguf", "model-Q8_0.gguf"]
        target, is_split, files = self._call(singles, {}, "q4-k-m")
        assert target == "model-Q4_K_M.gguf"

    def test_only_split_group_single_candidate(self) -> None:
        """A single split group is the only candidate — returns it."""
        singles = []
        split_groups = {
            "model-q4_k_m": [
                "model-q4_k_m-00001-of-00002.gguf",
                "model-q4_k_m-00002-of-00002.gguf",
            ]
        }
        target, is_split, files = self._call(singles, split_groups)
        assert is_split is True
        assert len(files) == 2


class TestShard1OfGroup:
    """Test shard-1 extraction."""

    def test_finds_shard_1(self) -> None:
        shards = [
            "model-q4_k_m-00001-of-00003.gguf",
            "model-q4_k_m-00002-of-00003.gguf",
            "model-q4_k_m-00003-of-00003.gguf",
        ]
        assert _shard1_of_group(shards) == "model-q4_k_m-00001-of-00003.gguf"

    def test_out_of_order_input(self) -> None:
        shards = [
            "model-q4_k_m-00003-of-00003.gguf",
            "model-q4_k_m-00001-of-00003.gguf",
            "model-q4_k_m-00002-of-00003.gguf",
        ]
        assert _shard1_of_group(shards) == "model-q4_k_m-00001-of-00003.gguf"


class TestQuantPreference:
    """Validate the quant preference order."""

    def test_q4_k_m_is_first(self) -> None:
        assert QUANT_PREFERENCE[0] == "q4_k_m"

    def test_f16_is_present(self) -> None:
        assert "f16" in QUANT_PREFERENCE

    def test_no_duplicates(self) -> None:
        assert len(QUANT_PREFERENCE) == len(set(QUANT_PREFERENCE))
