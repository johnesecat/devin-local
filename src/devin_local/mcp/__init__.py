"""MCP (Model Context Protocol) integration.

We use stdio-transport MCP servers — they're launched as subprocesses and
expose tools via a JSON-RPC over stdin/stdout. The official `mcp` Python SDK
is an optional dependency (the `[mcp]` extra). If it isn't installed, MCP
support is gracefully no-op'd at startup.
"""

from __future__ import annotations

from devin_local.mcp.client import MCPClientManager, McpServerSpec, mcp_available

__all__ = ["MCPClientManager", "McpServerSpec", "mcp_available"]
