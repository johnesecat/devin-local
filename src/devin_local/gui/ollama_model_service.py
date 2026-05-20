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
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_HOST = "http://127.0.0.1:11434"
HF_API_BASE = "https://huggingface.co/api"
HF_BASE = "https://huggingface.co"


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


# ---------------------------------------------------------------------------
# Hugging Face GGUF catalog
# ---------------------------------------------------------------------------


_GGUF_QUANT_RE = re.compile(
    r"\.(?:(IQ|Q)\d_[A-Z0-9_]+|F16|FP16|BF16|F32|FP32)(?:\.|$)",
    re.IGNORECASE,
)


def _quant_from_filename(name: str) -> str:
    """Pull the quantization label out of a GGUF filename.

    Examples::

        Llama-3.2-3B-Instruct-Q4_K_M.gguf       -> "Q4_K_M"
        gemma-2-9b-it.F16.gguf                  -> "F16"
        Qwen2.5-7B-Instruct.IQ3_M.gguf          -> "IQ3_M"

    Returns "?" if no recognizable quant is in the name.
    """
    base = name.lower()
    for marker in (
        "iq4_nl",
        "iq4_xs",
        "iq3_xs",
        "iq3_xxs",
        "iq3_s",
        "iq3_m",
        "iq2_xs",
        "iq2_xxs",
        "iq2_s",
        "iq2_m",
        "iq1_s",
        "iq1_m",
        "q2_k",
        "q3_k_s",
        "q3_k_m",
        "q3_k_l",
        "q4_0",
        "q4_1",
        "q4_k_s",
        "q4_k_m",
        "q5_0",
        "q5_1",
        "q5_k_s",
        "q5_k_m",
        "q6_k",
        "q8_0",
        "fp16",
        "f16",
        "bf16",
        "fp32",
        "f32",
    ):
        if marker in base:
            return marker.upper()
    return "?"


@dataclass
class HuggingFaceGGUFFile:
    """One GGUF artifact attached to a Hugging Face model repository."""

    filename: str
    size_bytes: int = 0
    quant: str = "?"

    @property
    def size_human(self) -> str:
        return _human_bytes(self.size_bytes)


@dataclass
class HuggingFaceGGUFEntry:
    """A Hugging Face model that ships GGUF files.

    The browser shows one of these per repo; clicking expands the list of
    GGUF files inside the repo (each one is independently downloadable).
    """

    repo_id: str  # e.g. "bartowski/Llama-3.2-3B-Instruct-GGUF"
    author: str = ""
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    last_modified: str = ""
    pipeline_tag: str = ""
    library_name: str = ""
    gguf_files: list[HuggingFaceGGUFFile] = field(default_factory=list)

    @property
    def display_family(self) -> str:
        """Cheap family inference from tags + name (llama / qwen / mistral / …)."""
        text = (self.repo_id + " " + " ".join(self.tags)).lower()
        for family in (
            "llama",
            "qwen",
            "mistral",
            "phi",
            "gemma",
            "deepseek",
            "yi",
            "falcon",
            "command-r",
            "codellama",
            "starcoder",
        ):
            if family in text:
                return family
        return "other"

    def ollama_name(self, file: HuggingFaceGGUFFile) -> str:
        """Construct the Ollama tag we'll create for ``file``.

        Format: ``hf.<author>-<repo>:<quant-lowercased>``. Stable across
        re-runs so the same GGUF doesn't get registered twice.
        """
        repo_safe = self.repo_id.replace("/", "-").replace("_", "-").lower()
        quant = file.quant.lower() if file.quant != "?" else "gguf"
        return f"hf.{repo_safe}:{quant}"


class HuggingFaceCatalog:
    """Read-only Hugging Face Hub client, scoped to GGUF model discovery.

    Hits ``GET huggingface.co/api/models?library=gguf&...`` and
    ``GET huggingface.co/api/models/{repo_id}`` to enumerate GGUF files.
    No auth is required for public models. Downloads stream the bytes
    straight to disk so a 4 GB GGUF doesn't blow up RAM.
    """

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0),
            follow_redirects=True,
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def search(
        self,
        *,
        query: str = "",
        limit: int = 60,
        sort: str = "downloads",
        direction: int = -1,
    ) -> list[HuggingFaceGGUFEntry]:
        """List GGUF-tagged repos on the Hub. Newest popular first by default."""
        params: dict[str, Any] = {
            "library": "gguf",
            "limit": limit,
            "sort": sort,
            "direction": direction,
            "full": "false",
        }
        if query:
            params["search"] = query
        resp = self._client.get(f"{HF_API_BASE}/models", params=params, timeout=15.0)
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list):
            return []
        out: list[HuggingFaceGGUFEntry] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            repo_id = str(row.get("modelId") or row.get("id") or "")
            if not repo_id:
                continue
            tags_raw = row.get("tags") or []
            tags = [str(t) for t in tags_raw if isinstance(t, str)]
            out.append(
                HuggingFaceGGUFEntry(
                    repo_id=repo_id,
                    author=str(row.get("author") or repo_id.split("/", 1)[0]),
                    downloads=int(row.get("downloads") or 0),
                    likes=int(row.get("likes") or 0),
                    tags=tags,
                    last_modified=str(row.get("lastModified") or ""),
                    pipeline_tag=str(row.get("pipeline_tag") or ""),
                    library_name=str(row.get("library_name") or "gguf"),
                )
            )
        return out

    def fetch_files(self, repo_id: str) -> list[HuggingFaceGGUFFile]:
        """Load the list of GGUF files (with sizes) for one repo."""
        resp = self._client.get(
            f"{HF_API_BASE}/models/{repo_id}",
            params={"blobs": "true"},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        siblings = data.get("siblings") if isinstance(data, dict) else None
        if not isinstance(siblings, list):
            return []
        out: list[HuggingFaceGGUFFile] = []
        for sib in siblings:
            if not isinstance(sib, dict):
                continue
            name = str(sib.get("rfilename") or "")
            if not name.lower().endswith(".gguf"):
                continue
            size_int = 0
            size_raw = sib.get("size")
            if isinstance(size_raw, (int, float)) and size_raw:
                size_int = int(size_raw)
            elif isinstance(sib.get("lfs"), dict):
                lfs_size = sib["lfs"].get("size")
                if isinstance(lfs_size, (int, float)) and lfs_size:
                    size_int = int(lfs_size)
            out.append(
                HuggingFaceGGUFFile(
                    filename=name,
                    size_bytes=size_int,
                    quant=_quant_from_filename(name),
                )
            )
        out.sort(key=lambda f: f.filename)
        return out

    def download_gguf(
        self,
        repo_id: str,
        filename: str,
        dest: Path,
        *,
        progress: Any = None,
    ) -> Path:
        """Stream a GGUF file to ``dest``. ``progress`` is an optional
        callable ``(completed: int, total: int) -> None`` for live updates.

        Idempotent: skips the download if ``dest`` already exists with a
        size matching the server-reported ``Content-Length``.
        """
        url = f"{HF_BASE}/{repo_id}/resolve/main/{filename}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        head = self._client.head(url, timeout=15.0)
        head.raise_for_status()
        expected = int(head.headers.get("Content-Length") or 0)
        if dest.exists() and expected and dest.stat().st_size == expected:
            if progress is not None:
                progress(expected, expected)
            return dest
        tmp = dest.with_suffix(dest.suffix + ".part")
        with self._client.stream("GET", url, timeout=None) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length") or expected or 0)
            completed = 0
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    completed += len(chunk)
                    if progress is not None:
                        progress(completed, total)
        tmp.replace(dest)
        return dest


def install_gguf_via_ollama(
    gguf_path: Path,
    ollama_name: str,
    *,
    ollama_bin: str = "ollama",
) -> None:
    """Register a downloaded GGUF with the local Ollama daemon.

    Writes a minimal Modelfile next to the GGUF and runs
    ``ollama create <ollama_name> -f <Modelfile>``. Raises if the CLI is
    missing or the create command fails.

    The Modelfile keeps it boring: just ``FROM <path>``. The user can edit
    the entry afterwards with ``ollama show --modelfile`` if they want a
    custom template / system prompt / parameters.
    """
    if not gguf_path.exists():
        raise FileNotFoundError(f"gguf not found: {gguf_path}")
    with tempfile.TemporaryDirectory(prefix="devin-local-modelfile-") as td:
        modelfile = Path(td) / "Modelfile"
        modelfile.write_text(f'FROM "{gguf_path.as_posix()}"\n', encoding="utf-8")
        try:
            subprocess.run(
                [ollama_bin, "create", ollama_name, "-f", str(modelfile)],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"`{ollama_bin}` CLI not found on PATH. Install Ollama from https://ollama.com."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"`ollama create` failed: {detail[:400]}") from exc
