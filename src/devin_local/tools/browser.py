"""Lightweight browser / web tools.

We deliberately avoid making `playwright` a hard dependency — the default
fetch tool uses `httpx` only. If the user installs the `[browser]` extra,
playwright-backed automation can be added later.
"""

from __future__ import annotations

import html
import re
from typing import Any

import httpx

from devin_local.tools.base import (
    Tool,
    ToolResult,
    error_result,
    optional_str,
    require_str,
    text_result,
    truncate,
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def _html_to_text(body: str) -> str:
    """Crude but dependency-free HTML-to-text conversion."""
    cleaned = re.sub(r"<script[\s\S]*?</script>", " ", body, flags=re.IGNORECASE)
    cleaned = re.sub(r"<style[\s\S]*?</style>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = _TAG_RE.sub("\n", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = _WS_RE.sub(" ", cleaned)
    cleaned = _NL_RE.sub("\n\n", cleaned)
    return cleaned.strip()


class WebFetchTool(Tool):
    """Fetch a URL and return the response body (rendered as text for HTML)."""

    name = "web_fetch"
    description = (
        "Fetch a URL via HTTP(S). For HTML, returns visible text only "
        "(scripts/styles stripped). For JSON or text, returns as-is. "
        "Honors the user's local network/proxy settings. 30s timeout."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "method": {"type": "string", "enum": ["GET", "POST"]},
            "body": {"type": "string", "description": "Optional request body for POST."},
        },
        "required": ["url"],
    }

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=30.0, follow_redirects=True)

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        url = require_str(arguments, "url")
        method = optional_str(arguments, "method", "GET").upper() or "GET"
        body = optional_str(arguments, "body")
        try:
            if method == "GET":
                resp = self._client.get(url)
            else:
                resp = self._client.post(url, content=body)
        except httpx.HTTPError as exc:
            return error_result(f"HTTP error fetching {url}: {exc}")
        ctype = resp.headers.get("content-type", "").lower()
        if "html" in ctype:
            text = _html_to_text(resp.text)
        else:
            text = resp.text
        header = f"[{resp.status_code}] {method} {url} ({ctype or 'unknown'})\n"
        return ToolResult(
            ok=200 <= resp.status_code < 400,
            output=truncate(header + text),
            data={
                "status_code": resp.status_code,
                "content_type": ctype,
                "url": str(resp.url),
            },
        )


class WebSearchTool(Tool):
    """Search the web via DuckDuckGo's HTML endpoint (no API key required)."""

    name = "web_search"
    description = (
        "Search the web using DuckDuckGo. Returns up to 10 result titles, "
        "URLs, and snippets. No API key required, no tracking."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 25},
        },
        "required": ["query"],
    }

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (devin-local)"},
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = require_str(arguments, "query")
        max_results = int(arguments.get("max_results") or 10)
        try:
            resp = self._client.post(
                "https://duckduckgo.com/html/",
                data={"q": query, "kl": "us-en"},
            )
        except httpx.HTTPError as exc:
            return error_result(f"Search failed: {exc}")
        if resp.status_code != 200:
            return error_result(f"Search returned HTTP {resp.status_code}.")
        body = resp.text
        result_re = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
            r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        matches = result_re.findall(body)
        lines: list[str] = []
        for url, title, snippet in matches[:max_results]:
            lines.append(
                f"- {_html_to_text(title).strip()}\n  {url}\n  {_html_to_text(snippet).strip()}"
            )
        if not lines:
            return error_result("No results parsed (DuckDuckGo HTML format may have changed).")
        return text_result("\n".join(lines), count=len(lines))
