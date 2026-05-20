"""Ollama model service: list installed models, browse the Ollama library, and
stream pull progress.

Wraps the Ollama HTTP API:

- ``GET /api/tags`` -> installed models
- ``POST /api/pull`` -> streaming NDJSON progress events while a model
  downloads

A curated list of popular models is hardcoded for the "Browse library" UI
because Ollama has no public "list everything in the registry" endpoint. The
list is intentionally small and biased toward tool-calling-capable models that
work with this agent.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class LibraryModel:
    """A curated entry in the Ollama library browser."""

    name: str
    size: str
    description: str
    tool_calling: bool


# Curated, conservatively sized. Each entry must be a real Ollama tag so
# `ollama pull <name>` works. Update sparingly.
POPULAR_MODELS: tuple[LibraryModel, ...] = (
    LibraryModel(
        "llama3.1:8b",
        "~4.7 GB",
        "Meta's Llama 3.1 8B. Default. Reliable tool calling.",
        tool_calling=True,
    ),
    LibraryModel(
        "llama3.2:3b",
        "~2.0 GB",
        "Small Llama 3.2. Fastest tool-capable option for 8 GB RAM machines.",
        tool_calling=True,
    ),
    LibraryModel(
        "llama3.2:1b",
        "~1.3 GB",
        "Tiny Llama 3.2. Useful for quick smoke tests; tool calls can be brittle.",
        tool_calling=True,
    ),
    LibraryModel(
        "qwen2.5:7b",
        "~4.4 GB",
        "Alibaba Qwen 2.5 7B. Strong reasoning + tool calling.",
        tool_calling=True,
    ),
    LibraryModel(
        "qwen2.5:14b",
        "~9.0 GB",
        "Qwen 2.5 14B. Needs ~12 GB RAM at runtime.",
        tool_calling=True,
    ),
    LibraryModel(
        "qwen2.5-coder:7b",
        "~4.4 GB",
        "Code-specialized Qwen. Emits structured tool calls.",
        tool_calling=True,
    ),
    LibraryModel(
        "mistral:7b",
        "~4.1 GB",
        "Mistral 7B v0.3. Tool calling supported.",
        tool_calling=True,
    ),
    LibraryModel(
        "mistral-nemo:12b",
        "~7.1 GB",
        "Mistral Nemo 12B. Strong tool use; needs ~10 GB RAM.",
        tool_calling=True,
    ),
    LibraryModel(
        "phi3.5:3.8b",
        "~2.2 GB",
        "Microsoft Phi-3.5 mini. Tool calling supported.",
        tool_calling=True,
    ),
    LibraryModel(
        "gemma2:9b",
        "~5.4 GB",
        "Google Gemma 2 9B. No structured tool calls (text only).",
        tool_calling=False,
    ),
    LibraryModel(
        "deepseek-coder:6.7b",
        "~3.8 GB",
        "DeepSeek Coder 6.7B. Code-only; structured tool calls limited.",
        tool_calling=False,
    ),
)


@dataclass
class InstalledModel:
    """A locally installed Ollama model."""

    name: str
    size_bytes: int = 0
    modified_at: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""

    @property
    def size_human(self) -> str:
        return _human_bytes(self.size_bytes)


def _human_bytes(n: int) -> str:
    if n <= 0:
        return "?"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


@dataclass
class PullProgress:
    """One step of an `/api/pull` stream.

    Ollama's pull events look like::

        {"status": "pulling manifest"}
        {"status": "downloading", "digest": "sha256:...", "total": N, "completed": M}
        {"status": "success"}
    """

    status: str = ""
    digest: str = ""
    total: int = 0
    completed: int = 0
    done: bool = False
    error: str = ""

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, max(0.0, self.completed / self.total))


class OllamaModelService:
    """Sync wrapper over Ollama's model-management HTTP endpoints.

    Designed to be called from a Qt worker thread (long-running pulls block).
    The streaming methods are plain generators so callers can attach progress
    callbacks (and cancel by stopping iteration).
    """

    def __init__(self, host: str = DEFAULT_HOST, client: httpx.Client | None = None) -> None:
        self.host = host.rstrip("/")
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def is_available(self) -> bool:
        try:
            resp = self._client.get(f"{self.host}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def list_installed(self) -> list[InstalledModel]:
        resp = self._client.get(f"{self.host}/api/tags", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        models: list[InstalledModel] = []
        for entry in data.get("models", []):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("model") or ""
            if not name:
                continue
            details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
            models.append(
                InstalledModel(
                    name=name,
                    size_bytes=int(entry.get("size") or 0),
                    modified_at=str(entry.get("modified_at") or ""),
                    family=str(details.get("family") or ""),
                    parameter_size=str(details.get("parameter_size") or ""),
                    quantization=str(details.get("quantization_level") or ""),
                )
            )
        models.sort(key=lambda m: m.name)
        return models

    def list_library(self) -> list[LibraryModel]:
        return list(POPULAR_MODELS)

    def pull(self, name: str) -> PullStream:
        """Begin a streaming pull. Returns a context-manager-like iterator that
        yields :class:`PullProgress` events as bytes arrive from Ollama.
        """
        return PullStream(self._client, self.host, name)

    def delete(self, name: str) -> None:
        resp = self._client.request(
            "DELETE", f"{self.host}/api/delete", json={"name": name}, timeout=30.0
        )
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"delete failed: {resp.status_code} {resp.text}",
                request=resp.request,
                response=resp,
            )


class PullStream:
    """Iterator wrapper around `/api/pull` streaming response.

    Usage::

        with service.pull("llama3.1:8b") as stream:
            for event in stream:
                if event.done or event.error:
                    break
                progress_bar.set(event.fraction)

    Stopping iteration (or exiting the ``with`` block) closes the underlying
    HTTP response so the daemon stops sending bytes.
    """

    def __init__(self, client: httpx.Client, host: str, name: str) -> None:
        self._client = client
        self._host = host.rstrip("/")
        self._name = name
        self._cm = None  # type: ignore[assignment]
        self._resp: httpx.Response | None = None
        self._closed = False

    def __enter__(self) -> PullStream:
        self._cm = self._client.stream(
            "POST",
            f"{self._host}/api/pull",
            json={"name": self._name, "stream": True},
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=5.0),
        )
        self._resp = self._cm.__enter__()
        if self._resp.status_code >= 400:
            body = self._resp.read().decode("utf-8", errors="replace")
            self.close()
            raise RuntimeError(f"pull failed: {self._resp.status_code} {body[:300]}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __iter__(self):
        if self._resp is None:
            self.__enter__()
        assert self._resp is not None
        for line in self._resp.iter_lines():
            if self._closed:
                break
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield _progress_from_event(obj)
            if obj.get("status") == "success" or obj.get("error"):
                break

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._cm is not None:
            with contextlib.suppress(Exception):
                self._cm.__exit__(None, None, None)
            self._cm = None
            self._resp = None


def _progress_from_event(event: dict[str, Any]) -> PullProgress:
    err = event.get("error")
    if err:
        return PullProgress(status="error", error=str(err), done=True)
    status = str(event.get("status") or "")
    done = status == "success"
    return PullProgress(
        status=status,
        digest=str(event.get("digest") or ""),
        total=int(event.get("total") or 0),
        completed=int(event.get("completed") or 0),
        done=done,
    )
