"""AirLLM-style layer-by-layer quantized loading backend.

This backend lets you run massive models (30B–70B+) on machines that could
never hold the whole model in RAM/VRAM, by loading **one transformer layer
at a time**, running it, freeing it, then loading the next. The price is
speed: on CPU this is genuinely slow (~tens of seconds per token for a
70B). On a low-VRAM GPU it's faster but still slow.

We use the [AirLLM](https://github.com/lyogavin/airllm) library when
available because it handles the hard parts (memory-mapped weight shards,
correct KV cache eviction between layers, prefill chunking) for the Llama,
Qwen, Mistral and Mixtral families.

If `airllm` is not installed, the backend raises `BackendUnavailableError`
with a clear hint on how to install it. We never silently fall back — the
user explicitly asked for layered inference and should get layered inference
or a useful error, not a different model on a different code path.

To stay light, this module is only imported lazily by the factory.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator
from typing import Any

from devin_local.inference.backend import (
    BackendUnavailableError,
    ChatChunk,
    ChatResponse,
    InferenceBackend,
)
from devin_local.inference.chat_template import format_chatml_prompt
from devin_local.inference.tool_parser import parse_tool_calls, tool_calls_to_wire
from devin_local.inference.types import ChatMessage

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
_STOP_TAGS = ("<|im_end|>", "<|endoftext|>")


class LayeredBackend(InferenceBackend):
    """AirLLM-backed inference, one transformer layer at a time."""

    name = "layered"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        compression: str | None = "4bit",
        profiling_mode: bool = False,
        layer_cache_dir: str | None = None,
        hf_token: str | None = None,
        max_new_tokens_default: int = 1024,
    ) -> None:
        self.model_id = model_id
        self.compression = compression
        self.profiling_mode = profiling_mode
        self.layer_cache_dir = layer_cache_dir
        self.hf_token = hf_token
        self.max_new_tokens_default = max_new_tokens_default
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._lock = threading.Lock()

    # ---------- private: lazy model load ----------

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer
            try:
                from airllm import AutoModel  # type: ignore[import-not-found]
            except ImportError as exc:
                raise BackendUnavailableError(
                    "Layered backend needs the `airllm` package. "
                    "Install with: pip install devin-local[layered] "
                    "(or: pip install airllm torch transformers accelerate)."
                ) from exc

            kwargs: dict[str, Any] = {}
            if self.compression:
                kwargs["compression"] = self.compression
            if self.profiling_mode:
                kwargs["profiling_mode"] = True
            if self.layer_cache_dir:
                kwargs["layer_shards_saving_path"] = self.layer_cache_dir
            if self.hf_token:
                kwargs["hf_token"] = self.hf_token

            log.info("Loading AirLLM model %s with %r", self.model_id, kwargs)
            model = AutoModel.from_pretrained(self.model_id, **kwargs)
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is None:
                from transformers import AutoTokenizer  # type: ignore[import-not-found]

                tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)
            self._model = model
            self._tokenizer = tokenizer
            return self._model, self._tokenizer

    # ---------- public API ----------

    def is_available(self) -> bool:
        try:
            __import__("airllm")
        except ImportError:
            return False
        return True

    def list_models(self) -> list[str]:
        # AirLLM doesn't enumerate; surface the configured target so the GUI
        # has something to show. Users add other models via --model-path.
        return [self.model_id]

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> ChatResponse:
        # Allow the caller to override the configured model id per-call.
        effective_model = model or self.model_id
        if effective_model != self.model_id:
            self.model_id = effective_model
            self._model = None
            self._tokenizer = None

        air_model, tokenizer = self._ensure_loaded()
        prompt = format_chatml_prompt(messages, tools=tools)
        opts = options or {}
        max_new_tokens = int(opts.get("num_predict") or self.max_new_tokens_default)

        text = _generate_full(air_model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        text = _trim_stops(text)
        clean, calls = parse_tool_calls(text)
        assistant_msg = ChatMessage(
            role="assistant",
            content=clean,
            tool_calls=tool_calls_to_wire(calls),
        )
        return ChatResponse(
            message=assistant_msg,
            raw={"backend": "layered", "model_id": effective_model, "raw_text": text},
            done=True,
        )

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> Iterator[ChatChunk]:
        effective_model = model or self.model_id
        if effective_model != self.model_id:
            self.model_id = effective_model
            self._model = None
            self._tokenizer = None

        air_model, tokenizer = self._ensure_loaded()
        prompt = format_chatml_prompt(messages, tools=tools)
        opts = options or {}
        max_new_tokens = int(opts.get("num_predict") or self.max_new_tokens_default)

        # AirLLM doesn't have an official token-stream callback; we generate in
        # chunks of N tokens and yield the difference, which is good enough to
        # paint a streaming UI without busy-waiting on the model.
        chunk_size = max(8, min(64, max_new_tokens // 16 or 8))
        produced_text = ""
        budget = max_new_tokens
        while budget > 0:
            step = min(chunk_size, budget)
            piece = _generate_full(
                air_model,
                tokenizer,
                prompt + produced_text,
                max_new_tokens=step,
            )
            if not piece:
                break
            delta = piece
            produced_text += delta
            yield ChatChunk(delta=delta, done=False)
            if any(tag in produced_text for tag in _STOP_TAGS):
                break
            budget -= step

        final_text = _trim_stops(produced_text)
        clean, calls = parse_tool_calls(final_text)
        final_msg = ChatMessage(
            role="assistant",
            content=clean,
            tool_calls=tool_calls_to_wire(calls),
        )
        yield ChatChunk(
            delta="",
            done=True,
            message=final_msg,
            raw={"backend": "layered", "model_id": effective_model, "raw_text": final_text},
        )

    def summarize(
        self,
        model: str,
        text: str,
        max_tokens: int = 512,
        instruction: str | None = None,
    ) -> str:
        instr = instruction or (
            "Summarize the following transcript into a dense, faithful note. "
            "Preserve all decisions, file paths, identifiers, and pending actions. "
            "Omit chit-chat. Use compact prose, not bullet points."
        )
        msgs = [
            ChatMessage(role="system", content=instr),
            ChatMessage(role="user", content=text),
        ]
        resp = self.chat(model=model, messages=msgs, options={"num_predict": max_tokens})
        return resp.message.content.strip()

    def close(self) -> None:
        # AirLLM model objects hold mmaps to layer shards on disk; drop our
        # reference and let GC handle the rest. Calling explicit teardown is
        # safe but slows app shutdown noticeably.
        self._model = None
        self._tokenizer = None


# ---------- helpers ----------


def _generate_full(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    """Run AirLLM generation and return *only the new text*.

    AirLLM's `model.generate()` returns the prompt + completion; we strip the
    prompt prefix before returning. Falls back gracefully if generation
    returns a tensor.
    """
    input_tokens = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "return_dict_in_generate": True,
    }
    output = model.generate(input_tokens.get("input_ids"), **generation_kwargs)
    sequences = getattr(output, "sequences", output)
    text = tokenizer.decode(sequences[0], skip_special_tokens=False)
    if text.startswith(prompt):
        text = text[len(prompt) :]
    return text


def _trim_stops(text: str) -> str:
    for tag in _STOP_TAGS:
        idx = text.find(tag)
        if idx != -1:
            text = text[:idx]
    return re.sub(r"<\|im_start\|>.*$", "", text, flags=re.DOTALL).strip()
