"""Model registry: context windows and capability hints per Ollama model.

Used by the context manager to decide when to compress chat history.
The map is intentionally conservative — when an exact model name is not
found, we look up by family prefix, then fall back to a safe default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata about an LLM available through Ollama."""

    name: str
    context_window: int
    supports_tools: bool = True
    family: str = ""
    abliterated: bool = False
    notes: str = ""


# Conservative context windows. Many of these can be raised by passing
# `num_ctx` to Ollama at runtime, but defaults match what the model is
# typically deployed with.
KNOWN_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("llama3.1:8b", 128_000, family="llama"),
    ModelSpec("llama3.1:70b", 128_000, family="llama"),
    ModelSpec("llama3.2:3b", 128_000, family="llama"),
    ModelSpec("llama3.2:1b", 128_000, family="llama"),
    ModelSpec("llama3:8b", 8_192, family="llama"),
    ModelSpec("qwen2.5-coder:7b", 32_768, family="qwen"),
    ModelSpec("qwen2.5-coder:14b", 32_768, family="qwen"),
    ModelSpec("qwen2.5-coder:32b", 32_768, family="qwen"),
    ModelSpec("qwen2.5:7b", 32_768, family="qwen"),
    ModelSpec("qwen2.5:14b", 32_768, family="qwen"),
    ModelSpec("mistral:7b", 32_768, family="mistral"),
    ModelSpec("mistral-nemo", 128_000, family="mistral"),
    ModelSpec("phi3:mini", 4_096, family="phi"),
    ModelSpec("phi3:medium", 4_096, family="phi"),
    ModelSpec("phi3.5", 128_000, family="phi"),
    ModelSpec("gemma2:9b", 8_192, family="gemma"),
    ModelSpec("gemma2:27b", 8_192, family="gemma"),
    ModelSpec("deepseek-coder-v2:16b", 128_000, family="deepseek"),
    ModelSpec("deepseek-r1:7b", 64_000, family="deepseek"),
    # Abliterated / uncensored families (OBLITERATUS-style).
    ModelSpec(
        "huihui_ai/llama3.1-abliterated",
        128_000,
        family="llama",
        abliterated=True,
        notes="Refusal directions removed via abliteration.",
    ),
    ModelSpec(
        "huihui_ai/qwen2.5-coder-abliterate",
        32_768,
        family="qwen",
        abliterated=True,
        notes="Refusal directions removed via abliteration.",
    ),
    ModelSpec(
        "huihui_ai/qwen2.5-abliterate",
        32_768,
        family="qwen",
        abliterated=True,
        notes="Refusal directions removed via abliteration.",
    ),
)


DEFAULT_CONTEXT_WINDOW = 8_192


def get_model_spec(name: str) -> ModelSpec:
    """Return a `ModelSpec` for `name`, falling back to a safe default."""
    needle = name.strip()
    for spec in KNOWN_MODELS:
        if spec.name == needle:
            return spec
    # Match by family prefix (e.g. "llama3.1:8b-instruct-q4_0" -> "llama3.1:8b").
    for spec in KNOWN_MODELS:
        if needle.startswith(spec.name):
            return spec
    family = needle.split(":", 1)[0].split("/", 1)[-1]
    return ModelSpec(name=needle, context_window=DEFAULT_CONTEXT_WINDOW, family=family)


def suggest_abliterated() -> list[ModelSpec]:
    """Return the registered abliterated/uncensored models for `models suggest`."""
    return [spec for spec in KNOWN_MODELS if spec.abliterated]
