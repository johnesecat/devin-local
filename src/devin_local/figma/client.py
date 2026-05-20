"""Minimal Figma REST client (read-only).

Why this module exists, and what it deliberately does NOT do:

- Figma's public REST API supports **read** operations (file tree, image
  export, comments) and a handful of write operations (post / update
  comments, dev-resource pins). It does **NOT** support creating files,
  frames, components, or styles. To author content programmatically you
  need a Figma desktop plugin (TypeScript). We are explicit about that
  limitation in :doc:`/docs/figma.md`.
- This client therefore does only what the REST API can actually do:
  fetch a file by key, list components / styles, and export node renders
  to PNG / SVG / PDF.

Auth is via a Personal Access Token. The token is **never** baked into
the codebase; it is read from ``~/.devin-local/figma.json`` (operator's
machine, ``0600`` perms) or the ``FIGMA_TOKEN`` env var.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import httpx
except Exception:  # pragma: no cover - httpx is in deps, but allow stub for tests
    httpx = None  # type: ignore[assignment]


FIGMA_API_BASE = "https://api.figma.com/v1"
USER_AGENT = "devin-local-figma/0.1"


class FigmaError(RuntimeError):
    """Raised on transport, auth, or response-shape errors."""


@dataclass
class FigmaSettings:
    """Operator-local Figma settings (token + last-used file)."""

    token: str = ""
    last_file_key: str = ""
    last_file_url: str = ""


def _figma_settings_path() -> Path:
    from devin_local.settings import settings_dir

    return settings_dir() / "figma.json"


def load_figma_settings() -> FigmaSettings:
    """Read ``~/.devin-local/figma.json`` (or env var) into a settings struct.

    Order of precedence: explicit file → ``FIGMA_TOKEN`` env var → empty.
    """
    path = _figma_settings_path()
    s = FigmaSettings()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        s.token = str(raw.get("token", "") or "")
        s.last_file_key = str(raw.get("last_file_key", "") or "")
        s.last_file_url = str(raw.get("last_file_url", "") or "")
    if not s.token:
        s.token = os.environ.get("FIGMA_TOKEN", "")
    return s


def save_figma_settings(settings: FigmaSettings) -> Path:
    """Persist Figma settings with ``0600`` perms. The token is treated as a secret."""
    path = _figma_settings_path()
    payload = {
        "_warning": "Contains a Figma PAT. Never commit this file. 0600 perms enforced.",
        "token": settings.token,
        "last_file_key": settings.last_file_key,
        "last_file_url": settings.last_file_url,
        "rotate_recommended": False,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with contextlib.suppress(OSError, NotImplementedError):
        os.chmod(tmp, 0o600)
    tmp.replace(path)
    return path


def figma_token_from_disk() -> str:
    """Convenience helper for callers that just need the bare token."""
    return load_figma_settings().token


_FILE_KEY_RE = re.compile(r"/(?:file|design)/([A-Za-z0-9]+)")


def parse_file_key(url_or_key: str) -> str:
    """Extract a Figma file key from a Figma URL or pass through a bare key.

    Supports both legacy ``figma.com/file/<key>/...`` and current
    ``figma.com/design/<key>/...`` URL formats.
    """
    if not url_or_key:
        raise FigmaError("empty file key/url")
    if "://" not in url_or_key and "/" not in url_or_key:
        return url_or_key
    parsed = urlparse(url_or_key)
    m = _FILE_KEY_RE.search(parsed.path or url_or_key)
    if not m:
        raise FigmaError(f"could not extract Figma file key from {url_or_key!r}")
    return m.group(1)


@dataclass
class FigmaClient:
    """Tiny synchronous Figma REST client."""

    token: str
    base_url: str = FIGMA_API_BASE
    timeout: float = 30.0

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise FigmaError(
                "No Figma PAT configured. Set one in Settings or ~/.devin-local/figma.json."
            )
        return {"X-Figma-Token": self.token, "User-Agent": USER_AGENT}

    def _client(self) -> Any:
        if httpx is None:
            raise FigmaError(
                "httpx is not available in this environment; cannot reach the Figma API."
            )
        return httpx.Client(timeout=self.timeout)

    def me(self) -> dict[str, Any]:
        """Hit ``/v1/me`` — used to verify the PAT works."""
        with self._client() as client:
            resp = client.get(f"{self.base_url}/me", headers=self._headers())
        if resp.status_code != 200:
            raise FigmaError(f"Figma /me returned HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def get_file(self, file_key_or_url: str, *, depth: int | None = None) -> dict[str, Any]:
        """Fetch a Figma file's node tree.

        ``depth`` is an optional API parameter — pass a small number when
        you only want top-level frames (cuts payload size massively).
        """
        key = parse_file_key(file_key_or_url)
        params: dict[str, Any] = {}
        if depth is not None:
            params["depth"] = depth
        with self._client() as client:
            resp = client.get(
                f"{self.base_url}/files/{key}",
                headers=self._headers(),
                params=params,
            )
        if resp.status_code != 200:
            raise FigmaError(
                f"Figma file fetch failed (HTTP {resp.status_code}): {resp.text[:200]}"
            )
        return resp.json()

    def export_images(
        self,
        file_key_or_url: str,
        node_ids: list[str],
        *,
        format: str = "png",
        scale: float = 2.0,
    ) -> dict[str, str]:
        """Ask Figma to render the given nodes. Returns ``{node_id: url}``.

        The URLs are short-lived; callers should download immediately. The
        empty-string URL is returned for nodes Figma refused to render.
        """
        if format not in {"png", "svg", "pdf", "jpg"}:
            raise FigmaError(f"unsupported format: {format!r}")
        if not node_ids:
            return {}
        key = parse_file_key(file_key_or_url)
        params = {
            "ids": ",".join(node_ids),
            "format": format,
            "scale": str(scale),
        }
        with self._client() as client:
            resp = client.get(
                f"{self.base_url}/images/{key}",
                headers=self._headers(),
                params=params,
            )
        if resp.status_code != 200:
            raise FigmaError(
                f"Figma image render failed (HTTP {resp.status_code}): {resp.text[:200]}"
            )
        payload = resp.json()
        if payload.get("err"):
            raise FigmaError(f"Figma image render returned err: {payload['err']}")
        return {k: v or "" for k, v in (payload.get("images") or {}).items()}

    def download(self, url: str, dest: Path) -> Path:
        """Download a previously-issued render URL to ``dest``."""
        if httpx is None:
            raise FigmaError("httpx is not available; cannot download Figma renders.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url)
        if resp.status_code != 200:
            raise FigmaError(f"download failed (HTTP {resp.status_code}): {url}")
        dest.write_bytes(resp.content)
        return dest

    def styles(self, file_key_or_url: str) -> list[dict[str, Any]]:
        """Return the file's named styles (colors / text / effects / grids)."""
        key = parse_file_key(file_key_or_url)
        with self._client() as client:
            resp = client.get(
                f"{self.base_url}/files/{key}/styles",
                headers=self._headers(),
            )
        if resp.status_code != 200:
            raise FigmaError(
                f"Figma styles fetch failed (HTTP {resp.status_code}): {resp.text[:200]}"
            )
        payload = resp.json()
        return list((payload.get("meta") or {}).get("styles") or [])


def remember_last_file(file_key_or_url: str) -> None:
    """Persist the last-used file so the UI can default to it next time."""
    settings = load_figma_settings()
    settings.last_file_url = file_key_or_url
    with contextlib.suppress(FigmaError):
        settings.last_file_key = parse_file_key(file_key_or_url)
    save_figma_settings(settings)


def now_ts() -> float:
    return time.time()
