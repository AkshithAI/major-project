#!/usr/bin/env python3
"""Verify tokenizer compatibility between target and draft models.

Checks that the draft model can safely be used for speculative decoding
with the target model by comparing vocabularies, special tokens, and
chat templates.

Usage:
    python scripts/verify_tokenizer.py TARGET.gguf DRAFT.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def read_gguf_tokenizer_info(model_path: str) -> dict:
    """Extract tokenizer metadata from a GGUF file."""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: 'gguf' package not installed. Run: pip install gguf")
        sys.exit(1)

    reader = GGUFReader(model_path)
    info: dict = {}

    for field in reader.fields.values():
        name = field.name
        if name.startswith("tokenizer.ggml."):
            short = name.removeprefix("tokenizer.ggml.")
            if short == "tokens":
                # Token list — extract as strings
                info["vocab_size"] = len(field.parts) - field.parts_count
                # Get the actual tokens for comparison
                tokens = []
                for part in field.parts[field.parts_count:]:
                    try:
                        tokens.append(bytes(part).decode("utf-8", errors="replace"))
                    except Exception:
                        tokens.append(str(part))
                info["tokens"] = tokens
                info["vocab_size"] = len(tokens)
            elif short == "model":
                info["model"] = str(bytes(field.parts[-1]), "utf-8")
            elif short in ("bos_token_id", "eos_token_id", "pad_token_id",
                           "sep_token_id", "cls_token_id", "unk_token_id"):
                try:
                    info[short] = int(field.parts[-1])
                except (ValueError, IndexError):
                    info[short] = None
            elif short == "chat_template":
                try:
                    info["chat_template"] = str(bytes(field.parts[-1]), "utf-8")
                except Exception:
                    info["chat_template"] = None

    return info


def compare_tokenizers(target_path: str, draft_path: str) -> bool:
    """Compare two models' tokenizers and report compatibility."""
    print(f"Target: {target_path}")
    print(f"Draft:  {draft_path}")
    print("=" * 60)

    target = read_gguf_tokenizer_info(target_path)
    draft = read_gguf_tokenizer_info(draft_path)

    passed = True

    # 1. Tokenizer model type
    t_model = target.get("model", "unknown")
    d_model = draft.get("model", "unknown")
    if t_model == d_model:
        print(f"✓ Tokenizer model: {t_model}")
    else:
        print(f"✗ Tokenizer model mismatch: target={t_model}, draft={d_model}")
        passed = False

    # 2. Vocab size
    t_vocab = target.get("vocab_size", 0)
    d_vocab = draft.get("vocab_size", 0)
    if t_vocab == d_vocab:
        print(f"✓ Vocab size: {t_vocab}")
    else:
        print(f"⚠ Vocab size differs: target={t_vocab}, draft={d_vocab}")
        # Not necessarily fatal — draft may have a subset

    # 3. Special tokens
    for token_name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        t_val = target.get(token_name)
        d_val = draft.get(token_name)
        if t_val == d_val:
            print(f"✓ {token_name}: {t_val}")
        elif t_val is None or d_val is None:
            print(f"⚠ {token_name}: target={t_val}, draft={d_val} (one not set)")
        else:
            print(f"✗ {token_name} mismatch: target={t_val}, draft={d_val}")
            passed = False

    # 4. Token overlap (first N tokens — core vocabulary)
    t_tokens = target.get("tokens", [])
    d_tokens = draft.get("tokens", [])
    if t_tokens and d_tokens:
        check_count = min(1000, len(t_tokens), len(d_tokens))
        matches = sum(1 for i in range(check_count) if t_tokens[i] == d_tokens[i])
        pct = (matches / check_count) * 100
        if pct > 99:
            print(f"✓ First {check_count} tokens: {pct:.1f}% match")
        elif pct > 95:
            print(f"⚠ First {check_count} tokens: {pct:.1f}% match (minor differences)")
        else:
            print(f"✗ First {check_count} tokens: {pct:.1f}% match (significant divergence)")
            passed = False

    # 5. Chat template
    t_chat = target.get("chat_template")
    d_chat = draft.get("chat_template")
    if t_chat and d_chat:
        if t_chat == d_chat:
            print("✓ Chat templates match")
        else:
            print("⚠ Chat templates differ (may affect instruction-following)")
    elif t_chat and not d_chat:
        print("⚠ Draft model has no chat template (target does)")
    else:
        print("– Chat template: not compared")

    print("=" * 60)
    if passed:
        print("RESULT: ✓ Compatible — draft model is suitable for speculative decoding")
    else:
        print("RESULT: ✗ Incompatible — do NOT use this draft model without further investigation")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify tokenizer compatibility between target and draft models"
    )
    parser.add_argument("target", help="Path to target model GGUF file")
    parser.add_argument("draft", help="Path to draft model GGUF file")
    args = parser.parse_args()

    if not Path(args.target).exists():
        print(f"ERROR: Target model not found: {args.target}")
        sys.exit(1)
    if not Path(args.draft).exists():
        print(f"ERROR: Draft model not found: {args.draft}")
        sys.exit(1)

    compatible = compare_tokenizers(args.target, args.draft)
    sys.exit(0 if compatible else 1)


if __name__ == "__main__":
    main()

