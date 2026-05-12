"""Backend factory: pick the right `InferenceBackend` by name."""

from __future__ import annotations

import contextlib
from typing import Any

from devin_local.inference.backend import BackendUnavailableError, InferenceBackend

# Names users can pass to --backend / build_backend. Kept here so the CLI and
# the GUI surface exactly the same set without grepping the code.
SUPPORTED_BACKENDS: tuple[str, ...] = ("ollama", "layered", "hf")


def build_backend(kind: str, **options: Any) -> InferenceBackend:
    """Construct a backend by name.

    Arguments per backend:
      - ``ollama``: ``host``, ``timeout``
      - ``layered``: ``model_id``, ``compression`` ("4bit"|"8bit"|None),
        ``profiling_mode`` (bool), ``layer_cache_dir`` (path|None),
        ``hf_token`` (str|None)
      - ``hf``: ``model_id``, ``device_map`` ("auto"|"cpu"|"cuda"|...),
        ``load_in_4bit`` (bool), ``load_in_8bit`` (bool),
        ``trust_remote_code`` (bool), ``hf_token`` (str|None)
    """
    kind = (kind or "ollama").lower()
    if kind == "ollama":
        from devin_local.inference.ollama_backend import OllamaBackend

        return OllamaBackend(**options)
    if kind == "layered":
        from devin_local.inference.layered_backend import LayeredBackend

        return LayeredBackend(**options)
    if kind == "hf":
        from devin_local.inference.hf_backend import HFBackend

        return HFBackend(**options)
    raise BackendUnavailableError(
        f"Unknown backend {kind!r}. Supported: {', '.join(SUPPORTED_BACKENDS)}."
    )


def list_available_backends() -> list[dict[str, Any]]:
    """Probe each backend to report status.

    Returns a list of {name, available, details} entries. ``available``
    means "this backend can serve a request without throwing immediately":

    - ``ollama``: daemon reachable.
    - ``layered``: `airllm` package installed (model loads on first call).
    - ``hf``: `torch` + `transformers` installed.

    Useful for the `doctor` command and the GUI's backend dropdown.
    """
    results: list[dict[str, Any]] = []
    for name in SUPPORTED_BACKENDS:
        entry: dict[str, Any] = {"name": name, "available": False, "details": ""}
        try:
            backend = build_backend(name)
        except BackendUnavailableError as exc:
            entry["details"] = str(exc)
            results.append(entry)
            continue
        try:
            entry["available"] = backend.is_available()
            entry["details"] = "ready" if entry["available"] else _hint_for(name)
        except BackendUnavailableError as exc:
            entry["details"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            entry["details"] = f"probe failed: {exc!r}"
        finally:
            with contextlib.suppress(Exception):
                backend.close()
        results.append(entry)
    return results


def _hint_for(backend: str) -> str:
    return {
        "ollama": "daemon not reachable \u2014 run `ollama serve`",
        "layered": "`airllm` not installed \u2014 `pip install devin-local[layered]`",
        "hf": "`transformers` not installed \u2014 `pip install devin-local[hf]`",
    }.get(backend, "not available")
