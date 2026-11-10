"""Model Inspector — extract metadata from GGUF model files.

Reads GGUF file headers to populate a ModelProfile with architecture,
parameter count, attention config, quantization info, and tokenizer details.
"""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.table import Table

from adhie.types import ModelProfile


# Mapping from GGUF quantization type IDs to human-readable names and average bits
QUANT_TYPE_MAP: dict[int, tuple[str, float]] = {
    0: ("f32", 32.0),
    1: ("f16", 16.0),
    2: ("q4_0", 4.5),
    3: ("q4_1", 5.0),
    6: ("q5_0", 5.5),
    7: ("q5_1", 6.0),
    8: ("q8_0", 8.5),
    9: ("q8_1", 9.0),
    10: ("q2_k", 3.35),
    11: ("q3_k_s", 3.5),
    12: ("q3_k_m", 3.9),
    13: ("q3_k_l", 4.3),
    14: ("q4_k_s", 4.5),
    15: ("q4_k_m", 4.8),
    16: ("q5_k_s", 5.5),
    17: ("q5_k_m", 5.7),
    18: ("q6_k", 6.6),
    19: ("iq2_xxs", 2.06),
    20: ("iq2_xs", 2.31),
    21: ("iq3_xxs", 3.06),
    22: ("iq1_s", 1.56),
    23: ("iq4_nl", 4.5),
    24: ("iq3_s", 3.44),
    25: ("iq2_s", 2.5),
    26: ("iq4_xs", 4.25),
    28: ("bf16", 16.0),
    29: ("q4_0_4_4", 4.5),
    30: ("q4_0_4_8", 4.5),
    31: ("q4_0_8_8", 4.5),
}

# Mapping from GGUF type enum values to Python types for reading metadata
GGUF_METADATA_VALUE_TYPE = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}


class ModelInspector:
    """Extract model metadata from GGUF files."""

    def inspect(self, model_path: str) -> ModelProfile:
        """Read a GGUF file and return a ModelProfile."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not path.suffix.lower() == ".gguf":
            raise ValueError(f"Expected .gguf file, got: {path.suffix}")

        file_size = path.stat().st_size
        metadata = self._read_gguf_metadata(model_path)

        profile = ModelProfile(
            model_path=str(path.resolve()),
            file_size_bytes=file_size,
        )

        self._populate_from_metadata(profile, metadata)
        return profile

    def _read_gguf_metadata(self, model_path: str) -> dict:
        """Read all key-value metadata from a GGUF file."""
        try:
            from gguf import GGUFReader
        except ImportError:
            raise ImportError(
                "The 'gguf' package is required. Install with: pip install gguf"
            )

        reader = GGUFReader(model_path)
        metadata: dict = {}

        for field in reader.fields.values():
            name = field.name
            try:
                # Try to extract a usable value
                if len(field.parts) > field.parts_count:
                    # Has data parts beyond type descriptors
                    data_parts = field.parts[field.parts_count:]
                    if len(data_parts) == 1:
                        val = data_parts[0]
                        if hasattr(val, "__len__") and not isinstance(val, (int, float)):
                            # It's an array-like (bytes or ndarray)
                            try:
                                val = bytes(val).decode("utf-8")
                            except (UnicodeDecodeError, TypeError):
                                try:
                                    val = int(val[0]) if len(val) == 1 else val.tolist()
                                except Exception:
                                    val = str(val)
                        metadata[name] = val
                    else:
                        # Multiple data parts — likely a token list or array
                        metadata[name] = data_parts
                elif len(field.parts) == 1:
                    val = field.parts[0]
                    if hasattr(val, "item"):
                        metadata[name] = val.item()
                    else:
                        metadata[name] = val
            except Exception:
                continue

        # Also gather tensor info for quantization detection
        tensor_quant_types: dict[int, int] = {}
        for tensor in reader.tensors:
            qt = int(tensor.tensor_type)
            tensor_quant_types[qt] = tensor_quant_types.get(qt, 0) + 1

        metadata["_tensor_quant_types"] = tensor_quant_types
        metadata["_tensor_count"] = len(reader.tensors)

        # Calculate total parameter count from tensors
        total_params = 0
        for tensor in reader.tensors:
            size = 1
            for dim in tensor.shape:
                size *= dim
            total_params += size
        metadata["_total_params"] = total_params

        return metadata

    def _populate_from_metadata(self, profile: ModelProfile, meta: dict) -> None:
        """Fill ModelProfile fields from GGUF metadata."""
        # Model identity
        profile.model_id = self._get_str(meta, "general.name", "unknown")
        arch = self._get_str(meta, "general.architecture", "unknown")
        profile.architecture = arch
        profile.model_family = profile.model_id.split("-")[0] if profile.model_id != "unknown" else arch

        # Use architecture prefix for metadata keys
        prefix = arch

        # Layer count
        profile.layer_count = self._get_int(meta, f"{prefix}.block_count", 0)

        # Sizes
        profile.hidden_size = self._get_int(meta, f"{prefix}.embedding_length", 0)
        profile.intermediate_size = self._get_int(meta, f"{prefix}.feed_forward_length", 0)
        profile.vocab_size = self._get_int(meta, f"{prefix}.vocab_size",
                                           self._get_int(meta, "tokenizer.ggml.vocab_size", 0))

        # If vocab_size still 0, try counting tokens
        if profile.vocab_size == 0:
            tokens_data = meta.get("tokenizer.ggml.tokens")
            if tokens_data and isinstance(tokens_data, (list, tuple)):
                profile.vocab_size = len(tokens_data)

        # Attention configuration
        profile.num_attention_heads = self._get_int(
            meta, f"{prefix}.attention.head_count", 0)
        profile.num_kv_heads = self._get_int(
            meta, f"{prefix}.attention.head_count_kv",
            profile.num_attention_heads)  # Default to MHA

        # Head dimension
        head_dim = self._get_int(meta, f"{prefix}.attention.key_length", 0)
        if head_dim == 0 and profile.num_attention_heads > 0 and profile.hidden_size > 0:
            head_dim = profile.hidden_size // profile.num_attention_heads
        profile.head_dim = head_dim

        # Context length
        profile.max_context = self._get_int(meta, f"{prefix}.context_length", 0)

        # Sliding window
        sw = self._get_int(meta, f"{prefix}.attention.sliding_window", 0)
        if sw > 0:
            profile.attention_type = "sliding_window"
            profile.sliding_window_size = sw
        else:
            profile.attention_type = "full"

        # Parameter count (from tensor analysis)
        profile.parameter_count = meta.get("_total_params", 0)

        # Quantization (determined from most common tensor type)
        quant_types = meta.get("_tensor_quant_types", {})
        if quant_types:
            # Find the dominant quantization type (most tensors)
            dominant_type = max(quant_types, key=quant_types.get)
            type_name, bits = QUANT_TYPE_MAP.get(dominant_type, ("unknown", 16.0))
            profile.weight_dtype = type_name
            profile.bits_per_weight = bits
        else:
            profile.weight_dtype = "unknown"
            profile.bits_per_weight = 16.0

        # Tokenizer
        profile.tokenizer_id = self._get_str(meta, "tokenizer.ggml.model", "unknown")

        # Chat template
        ct = meta.get("tokenizer.chat_template")
        if isinstance(ct, str):
            profile.chat_template = ct

        # MoE detection
        num_experts = self._get_int(meta, f"{prefix}.expert_count", 0)
        if num_experts > 0:
            profile.is_moe = True
            profile.num_experts = num_experts
            profile.num_active_experts = self._get_int(
                meta, f"{prefix}.expert_used_count", 1)

    def check_draft_compatibility(
        self, target: ModelProfile, draft: ModelProfile
    ) -> tuple[bool, list[str]]:
        """Check if a draft model is compatible with a target for speculative decoding.

        Returns (is_compatible, list_of_issues).
        """
        issues: list[str] = []

        # Tokenizer model must match
        if target.tokenizer_id != draft.tokenizer_id:
            issues.append(
                f"Tokenizer mismatch: target={target.tokenizer_id}, "
                f"draft={draft.tokenizer_id}"
            )

        # Vocab size should match
        if target.vocab_size != draft.vocab_size and target.vocab_size > 0 and draft.vocab_size > 0:
            issues.append(
                f"Vocab size mismatch: target={target.vocab_size}, "
                f"draft={draft.vocab_size}"
            )

        # Draft should be smaller
        if draft.parameter_count >= target.parameter_count and draft.parameter_count > 0:
            issues.append(
                f"Draft model ({draft.parameter_count/1e9:.1f}B) is not smaller than "
                f"target ({target.parameter_count/1e9:.1f}B)"
            )

        # Architecture family should ideally match
        if target.architecture != draft.architecture:
            issues.append(
                f"Architecture mismatch: target={target.architecture}, "
                f"draft={draft.architecture} (may still work if tokenizer matches)"
            )

        compatible = len([i for i in issues if "mismatch" in i.lower()
                         and "architecture" not in i.lower()]) == 0
        return compatible, issues

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_str(meta: dict, key: str, default: str = "") -> str:
        val = meta.get(key, default)
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        return str(val) if val is not None else default

    @staticmethod
    def _get_int(meta: dict, key: str, default: int = 0) -> int:
        val = meta.get(key, default)
        if val is None:
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, profile: ModelProfile, console: Console) -> None:
        """Print a formatted model profile report."""
        table = Table(title="Model Profile", show_header=True,
                      header_style="bold cyan")
        table.add_column("Category", style="bold")
        table.add_column("Property")
        table.add_column("Value", justify="right")

        # Identity
        table.add_row("Identity", "Name", profile.model_id)
        table.add_row("", "Architecture", profile.architecture)
        table.add_row("", "Family", profile.model_family)
        table.add_row("", "File Size", f"{profile.file_size_bytes / 1e9:.2f} GB")

        table.add_section()

        # Architecture
        table.add_row("Architecture", "Parameters",
                      f"{profile.parameter_count / 1e9:.2f}B")
        table.add_row("", "Layers", str(profile.layer_count))
        table.add_row("", "Hidden Size", str(profile.hidden_size))
        table.add_row("", "Intermediate Size", str(profile.intermediate_size))
        table.add_row("", "Vocab Size", f"{profile.vocab_size:,}")

        table.add_section()

        # Attention
        table.add_row("Attention", "Query Heads",
                      str(profile.num_attention_heads))
        table.add_row("", "KV Heads", str(profile.num_kv_heads))
        gqa = ("GQA" if profile.num_kv_heads < profile.num_attention_heads
               else "MHA" if profile.num_kv_heads == profile.num_attention_heads
               else "MQA" if profile.num_kv_heads == 1 else "?")
        table.add_row("", "Type", gqa)
        table.add_row("", "Head Dim", str(profile.head_dim))
        table.add_row("", "Max Context", f"{profile.max_context:,}")
        table.add_row("", "Attention Pattern", profile.attention_type)
        if profile.sliding_window_size:
            table.add_row("", "Sliding Window", str(profile.sliding_window_size))

        table.add_section()

        # Quantization
        table.add_row("Quantization", "Weight Type", profile.weight_dtype)
        table.add_row("", "Bits/Weight", f"{profile.bits_per_weight:.1f}")

        table.add_section()

        # Tokenizer
        table.add_row("Tokenizer", "Model", profile.tokenizer_id)
        table.add_row("", "Has Chat Template",
                      "Yes" if profile.chat_template else "No")

        # MoE
        if profile.is_moe:
            table.add_section()
            table.add_row("MoE", "Experts", str(profile.num_experts))
            table.add_row("", "Active Experts", str(profile.num_active_experts))

        table.add_section()

        # Memory estimates
        kv_f16 = profile.kv_bytes_per_token(16)
        kv_q8 = profile.kv_bytes_per_token(8)
        table.add_row("KV Cache", "Per-Token (FP16)",
                      f"{kv_f16 / 1024:.1f} KB")
        table.add_row("", "Per-Token (Q8)",
                      f"{kv_q8 / 1024:.1f} KB")
        table.add_row("", "8K Context (FP16)",
                      f"{kv_f16 * 8192 / 1e9:.2f} GB")
        table.add_row("", "Per-Layer Weight",
                      f"{profile.weight_bytes_per_layer() / 1e6:.1f} MB")
        table.add_row("", "Fingerprint", profile.fingerprint())

        console.print(table)

