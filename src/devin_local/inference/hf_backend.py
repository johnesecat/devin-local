"""HuggingFace transformers backend with optional bitsandbytes 4-bit / 8-bit.

This is the "I have a real GPU and want full in-RAM inference without an
Ollama daemon" path. It's also the fallback when AirLLM isn't installed but
the user still wants to drive an HF model directly.

Streaming uses `TextIteratorStreamer` from `transformers`, so tokens reach
the UI as soon as they're generated.

Lazy-imported by the factory; no torch/transformers in your environment
unless you `pip install devin-local[hf]`.
"""

from __future__ import annotations

import logging
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

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
_STOP_TAGS = ("<|im_end|>", "<|endoftext|>")


class HFBackend(InferenceBackend):
    """HuggingFace transformers + accelerate device-map, optional bnb quant."""

    name = "hf"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device_map: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
        hf_token: str | None = None,
        max_new_tokens_default: int = 1024,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code
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
                import torch  # noqa: F401  # type: ignore[import-not-found]
                from transformers import (  # type: ignore[import-not-found]
                    AutoModelForCausalLM,
                    AutoTokenizer,
                )
            except ImportError as exc:
                raise BackendUnavailableError(
                    "HF backend needs `torch` + `transformers`. "
                    "Install with: pip install devin-local[hf]"
                ) from exc

            kwargs: dict[str, Any] = {
                "device_map": self.device_map,
                "trust_remote_code": self.trust_remote_code,
            }
            if self.hf_token:
                kwargs["token"] = self.hf_token
            if self.load_in_4bit or self.load_in_8bit:
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
                except ImportError as exc:
                    raise BackendUnavailableError(
                        "4-bit / 8-bit quantization needs `bitsandbytes` + recent transformers. "
                        "Install with: pip install bitsandbytes"
                    ) from exc
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit and not self.load_in_4bit,
                )

            log.info("Loading HF model %s with %r", self.model_id, kwargs)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, token=self.hf_token, trust_remote_code=self.trust_remote_code
            )
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            model.eval()
            self._model = model
            self._tokenizer = tokenizer
            return self._model, self._tokenizer

    # ---------- public API ----------

    def is_available(self) -> bool:
        try:
            __import__("torch")
            __import__("transformers")
        except ImportError:
            return False
        return True

    def list_models(self) -> list[str]:
        return [self.model_id]

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> ChatResponse:
        effective_model = model or self.model_id
        if effective_model != self.model_id:
            self.model_id = effective_model
            self._model = None
            self._tokenizer = None

        hf_model, tokenizer = self._ensure_loaded()
        prompt = format_chatml_prompt(messages, tools=tools)
        opts = options or {}
        max_new_tokens = int(opts.get("num_predict") or self.max_new_tokens_default)
        temperature = float(opts.get("temperature", 0.2))

        text = _hf_generate_full(
            hf_model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )
        text = _trim_stops(text)
        clean, calls = parse_tool_calls(text)
        return ChatResponse(
            message=ChatMessage(
                role="assistant", content=clean, tool_calls=tool_calls_to_wire(calls)
            ),
            raw={"backend": "hf", "model_id": effective_model, "raw_text": text},
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

        hf_model, tokenizer = self._ensure_loaded()
        prompt = format_chatml_prompt(messages, tools=tools)
        opts = options or {}
        max_new_tokens = int(opts.get("num_predict") or self.max_new_tokens_default)
        temperature = float(opts.get("temperature", 0.2))

        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(hf_model, "device"):
            inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        gen_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0.0,
            "streamer": streamer,
        }
        thread = threading.Thread(target=hf_model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        accumulated: list[str] = []
        for delta in streamer:
            if not delta:
                continue
            accumulated.append(delta)
            yield ChatChunk(delta=delta, done=False)
            if any(tag in delta for tag in _STOP_TAGS):
                break
        thread.join(timeout=5.0)

        full_text = _trim_stops("".join(accumulated))
        clean, calls = parse_tool_calls(full_text)
        yield ChatChunk(
            delta="",
            done=True,
            message=ChatMessage(
                role="assistant", content=clean, tool_calls=tool_calls_to_wire(calls)
            ),
            raw={"backend": "hf", "model_id": effective_model, "raw_text": full_text},
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
        self._model = None
        self._tokenizer = None


# ---------- helpers ----------


def _hf_generate_full(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=False)
    if text.startswith(prompt):
        text = text[len(prompt) :]
    return text


def _trim_stops(text: str) -> str:
    for tag in _STOP_TAGS:
        idx = text.find(tag)
        if idx != -1:
            text = text[:idx]
    return text.strip()
