# ADR-002: Model Selection

## Status
Proposed

## Context
We need a primary target model (10B–14B parameters) and a compatible draft
model for speculative decoding experiments.

## Decision

### Target Model
- **Model:** Qwen2.5-14B-Instruct (or similar 10–14B model)
- **Format:** GGUF, Q4_K_M quantization
- **Expected file size:** ~8.5 GB
- **Layer count:** 40
- **KV heads:** 8 (GQA)
- **Head dim:** 128
- **Max context:** 32768
- **Tokenizer:** Qwen2.5 family

### Draft Model
- **Model:** Qwen2.5-0.5B-Instruct
- **Format:** GGUF, Q4_K_M quantization
- **Expected file size:** ~0.4 GB
- **Tokenizer:** Same Qwen2.5 family (must verify with verify_tokenizer.py)

### Alternatives Considered
- **Llama-3.1-8B:** Good ecosystem support, but 8B may be too small to
  stress the planner on constrained hardware.
- **Mistral-7B:** Well-tested in llama.cpp but uses sliding window attention.
- **Gemma-2-9B:** Google model, may have different tokenizer behavior.

## Verification
Run `python scripts/verify_tokenizer.py target.gguf draft.gguf` before
committing to this model pair.

## Download
```bash
# Example using huggingface-cli (update with actual GGUF source)
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GGUF qwen2.5-14b-instruct-q4_k_m.gguf
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf
```

