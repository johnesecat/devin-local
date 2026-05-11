"""Stdio MCP client wrapper.

This module bridges MCP servers (each exposing a list of tools and a
`call_tool` RPC) into devin-local's `ToolRegistry`. Server tools are
registered with the prefix `mcp__<server>__<tool>` so they never collide
with built-in tools.

Configuration lives in `mcp_servers.json` at the workspace root:

    {
      "servers": [
        {
          "name": "filesystem",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        }
      ]
    }

If the `mcp` SDK is not installed we still parse the config and surface a
clear startup message, but no tools are registered.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devin_local.tools.base import Tool, ToolResult, error_result, text_result, truncate


def mcp_available() -> bool:
    """Return True if the `mcp` Python SDK is importable."""
    try:
        import mcp  # type: ignore[import-not-found]  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass
class McpServerSpec:
    """One MCP server's launch configuration."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


def load_mcp_servers(config_path: Path) -> list[McpServerSpec]:
    """Parse `mcp_servers.json` (returns empty if missing/invalid)."""
    if not config_path.exists():
        return []
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    raw_servers = data.get("servers", []) or []
    return [
        McpServerSpec(
            name=str(s.get("name") or f"server-{i}"),
            command=str(s.get("command", "")),
            args=list(s.get("args", []) or []),
            env=dict(s.get("env", {}) or {}),
            cwd=s.get("cwd"),
        )
        for i, s in enumerate(raw_servers)
        if s.get("command")
    ]


class _McpToolProxy(Tool):
    """A `Tool` that forwards calls to an MCP server tool."""

    def __init__(
        self,
        manager: MCPClientManager,
        server: str,
        original_name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self._manager = manager
        self._server = server
        self._original_name = original_name
        self.name = f"mcp__{server}__{original_name}"
        self.description = description or f"MCP tool {original_name} on server {server}"
        self.parameters = parameters or {"type": "object", "properties": {}}

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            payload = self._manager.call_tool(self._server, self._original_name, arguments)
        except Exception as exc:  # noqa: BLE001
            return error_result(f"MCP tool {self.name} failed: {exc}")
        if isinstance(payload, str):
            return text_result(truncate(payload))
        return text_result(truncate(json.dumps(payload, default=str, indent=2)))


class MCPClientManager:
    """Manages connections to MCP servers and exposes their tools.

    The MCP Python SDK is async-first; we run a dedicated background event
    loop in a thread so that synchronous code (the agent loop) can call
    `call_tool()` without becoming async itself.
    """

    def __init__(self) -> None:
        self._servers: dict[str, dict[str, Any]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            self._started.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_runner, name="mcp-loop", daemon=True)
        self._thread.start()
        self._started.wait(timeout=5)
        assert self._loop is not None
        return self._loop

    def connect_all(self, specs: list[McpServerSpec]) -> list[_McpToolProxy]:
        """Connect to every server in `specs` and return their tool proxies."""
        if not specs:
            return []
        if not mcp_available():
            return []
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(self._connect_all_async(specs), loop)
        return future.result(timeout=60)

    async def _connect_all_async(self, specs: list[McpServerSpec]) -> list[_McpToolProxy]:
        from mcp import ClientSession, StdioServerParameters  # type: ignore[import-not-found]
        from mcp.client.stdio import stdio_client  # type: ignore[import-not-found]

        proxies: list[_McpToolProxy] = []
        for spec in specs:
            params = StdioServerParameters(
                command=spec.command,
                args=spec.args,
                env=spec.env or None,
                cwd=spec.cwd,
            )
            ctx = stdio_client(params)
            read, write = await ctx.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            tools = await session.list_tools()
            self._servers[spec.name] = {
                "session": session,
                "ctx": ctx,
                "tools": tools,
            }
            for tool in getattr(tools, "tools", []) or []:
                proxies.append(
                    _McpToolProxy(
                        self,
                        spec.name,
                        tool.name,
                        getattr(tool, "description", "") or "",
                        getattr(tool, "inputSchema", {}) or {"type": "object"},
                    )
                )
        return proxies

    def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        state = self._servers.get(server)
        if state is None:
            raise RuntimeError(f"Unknown MCP server: {server}")
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(state["session"].call_tool(tool, arguments), loop)
        result = future.result(timeout=120)
        # MCP returns a list of content items; flatten to text.
        if hasattr(result, "content"):
            chunks: list[str] = []
            for item in result.content or []:
                text = getattr(item, "text", None)
                if text is not None:
                    chunks.append(text)
            return "\n".join(chunks) if chunks else result
        return result

    def shutdown(self) -> None:
        if self._loop is None:
            return

        async def _close() -> None:
            for state in self._servers.values():
                with contextlib.suppress(Exception):
                    await state["session"].__aexit__(None, None, None)
                with contextlib.suppress(Exception):
                    await state["ctx"].__aexit__(None, None, None)

        asyncio.run_coroutine_threadsafe(_close(), self._loop).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
