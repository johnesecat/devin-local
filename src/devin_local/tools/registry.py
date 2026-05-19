"""Tool registry: discovery, dispatch, and Ollama schema export."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devin_local.emulators.desktop_emulator import DesktopEmulator
from devin_local.emulators.terminal_emulator import TerminalEmulator
from devin_local.tools.base import Tool, ToolError, ToolResult, error_result
from devin_local.tools.browser import WebFetchTool, WebSearchTool
from devin_local.tools.desktop import (
    DesktopClickTool,
    DesktopKeyTool,
    DesktopScreenshotTool,
    DesktopTypeTool,
)
from devin_local.tools.file_tools import (
    EditFileTool,
    FileToolContext,
    FindFilesTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from devin_local.tools.python_repl import PythonReplTool
from devin_local.tools.terminal import ShellExecTool, ShellSessionTool


@dataclass
class ToolRegistry:
    """Registry of tools available to the agent for the current session."""

    tools: dict[str, Tool]

    @classmethod
    def empty(cls) -> ToolRegistry:
        return cls(tools={})

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ToolError("Tool must define a name.")
        if tool.name in self.tools:
            raise ToolError(f"Duplicate tool name: {tool.name}")
        self.tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self.tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __iter__(self):
        return iter(self.tools.values())

    def names(self) -> list[str]:
        return sorted(self.tools.keys())

    def to_ollama_schemas(self) -> list[dict[str, Any]]:
        return [t.to_ollama_schema() for t in self.tools.values()]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return error_result(f"Unknown tool {name!r}. Available: {', '.join(self.names())}")
        try:
            return tool.run(arguments or {})
        except ToolError as exc:
            return error_result(str(exc))
        except Exception as exc:  # noqa: BLE001 - we never want a tool to crash the loop
            return error_result(f"Unhandled error in tool {name}: {exc!r}")

    def dispatch_many(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        max_workers: int = 4,
    ) -> list[ToolResult]:
        """Dispatch multiple tool calls concurrently, preserving order.

        When the model emits several tool_calls in one turn (e.g. read three
        files at once), running them serially wastes wall-clock time. Each
        call still goes through the same `dispatch()` path \u2014 same error
        handling, same observers \u2014 just in a worker thread.
        """
        if not calls:
            return []
        if len(calls) == 1:
            name, arguments = calls[0]
            return [self.dispatch(name, arguments)]
        workers = min(max_workers, len(calls))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tool") as pool:
            futures = [pool.submit(self.dispatch, name, arguments) for name, arguments in calls]
            return [f.result() for f in futures]


def build_default_registry(
    workspace: Path,
    *,
    enable_desktop: bool = True,
    enable_browser: bool = True,
) -> ToolRegistry:
    """Build the standard toolbelt rooted at `workspace`."""
    workspace = workspace.resolve()
    registry = ToolRegistry.empty()
    file_ctx = FileToolContext(workspace=workspace)
    for tool in (
        ReadFileTool(file_ctx),
        WriteFileTool(file_ctx),
        EditFileTool(file_ctx),
        ListDirTool(file_ctx),
        FindFilesTool(file_ctx),
        GrepTool(file_ctx),
        ShellExecTool(workspace),
        PythonReplTool(workspace),
    ):
        registry.register(tool)

    terminal = TerminalEmulator()
    registry.register(ShellSessionTool(workspace, terminal))

    if enable_browser:
        registry.register(WebFetchTool())
        registry.register(WebSearchTool())

    if enable_desktop:
        desktop = DesktopEmulator(workspace=workspace)
        registry.register(DesktopScreenshotTool(desktop))
        registry.register(DesktopClickTool(desktop))
        registry.register(DesktopTypeTool(desktop))
        registry.register(DesktopKeyTool(desktop))

    return registry
