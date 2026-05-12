"""Inference backends for devin-local.

A backend is anything that can take a list of chat messages and produce an
assistant response (optionally with tool calls). All backends implement
`InferenceBackend` so the rest of the agent (loop, tools, streaming UI,
compaction, plugins, MCP) is backend-agnostic.

Current backends:

- `OllamaBackend` — talks to a local Ollama daemon via HTTP. Default. Fastest
  on commodity hardware because Ollama's runtime handles GGUF quantization
  and KV caching for you.
- `LayeredBackend` — AirLLM-style layer-by-layer quantized loading. Lets you
  run massive (30B–70B+) models on machines that could never hold them whole
  in RAM, by loading one transformer block at a time, running it, and freeing
  it before loading the next. Slow on CPU, but it actually works.
- `HFBackend` — HuggingFace `transformers` + `accelerate` with optional
  `bitsandbytes` 4-bit. Useful when you have a real GPU and want full
  in-RAM inference without an Ollama daemon.

Backends are constructed via `build_backend(kind, **opts)`. Importing this
module never imports torch / transformers / airllm — those are loaded lazily
inside the corresponding backend, so `pip install devin-local` stays light.
"""

# Only re-export lightweight, side-effect-free names here. The factory and
# the concrete backends are loaded lazily via __getattr__ so that importing
# `devin_local.inference` does NOT pull in torch / transformers / airllm.
from __future__ import annotations

from typing import Any

from devin_local.inference.backend import (
    BackendUnavailableError,
    ChatChunk,
    ChatResponse,
    InferenceBackend,
)
from devin_local.inference.types import ChatMessage, ToolCall, ToolDefinition

__all__ = [
    "BackendUnavailableError",
    "ChatChunk",
    "ChatMessage",
    "ChatResponse",
    "InferenceBackend",
    "ToolCall",
    "ToolDefinition",
    "build_backend",
    "list_available_backends",
]

_LAZY = {"build_backend", "list_available_backends"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from devin_local.inference.factory import build_backend, list_available_backends

        globals()["build_backend"] = build_backend
        globals()["list_available_backends"] = list_available_backends
        return globals()[name]
    raise AttributeError(f"module 'devin_local.inference' has no attribute {name!r}")
