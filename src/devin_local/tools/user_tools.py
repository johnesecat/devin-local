"""User-defined Python tools loaded dynamically from disk.

Drop a ``.py`` file under ``~/.devin-local/tools/`` (or the operator's
configured user-tools directory) that exposes one or more functions
decorated with :func:`tool` and the agent will register them on its next
init / reload pass. Each tool function receives an ``arguments`` dict
matching the JSON-Schema declared on the decorator and returns either a
string (treated as ``ok=True`` output) or a :class:`ToolResult`.

Example file (``~/.devin-local/tools/wordcount.py``)::

    from devin_local.tools.user_tools import tool

    @tool(
        name="wordcount",
        description="Count words in a string.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )
    def wordcount(args):
        return f"{len(args['text'].split())} words"

The decorator stores metadata on the function object; the loader
collects every decorated function it finds in the directory's ``.py``
files and registers them with the agent's :class:`ToolRegistry`.

This module is *user-facing* — the operator edits these files directly,
so the API surface is kept deliberately small and self-explanatory.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import logging
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devin_local.tools.base import Tool, ToolError, ToolResult, error_result, text_result

log = logging.getLogger(__name__)

_TOOL_ATTR = "__devin_local_tool_spec__"


@dataclass
class ToolSpec:
    """Metadata recorded by the :func:`tool` decorator on a user function."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class LoadedUserTool:
    """A user tool actually mounted into the agent's registry.

    Tracks the source path + last-modified mtime so the loader can detect
    edits and re-load without restarting the GUI.
    """

    tool: Tool
    source_path: Path
    mtime: float


@dataclass
class UserToolError:
    """Loader-side error for a user tool file (so the GUI can surface it)."""

    source_path: Path
    message: str
    traceback: str = ""


@dataclass
class UserToolsReport:
    """Result of one ``load_user_tools()`` pass."""

    loaded: list[LoadedUserTool] = field(default_factory=list)
    errors: list[UserToolError] = field(default_factory=list)


def tool(
    *,
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[[dict[str, Any]], Any]], Callable[[dict[str, Any]], Any]]:
    """Decorator marking a function as a devin-local user tool.

    ``parameters`` must be a JSON-Schema object (same shape Ollama expects
    in its ``tools`` array). When omitted we default to an empty object
    schema — equivalent to ``{"type": "object", "properties": {}}``.
    """

    if not name or not name.replace("_", "").isalnum():
        raise ToolError(f"@tool name must be non-empty and alphanumeric (got {name!r})")

    schema = parameters or {"type": "object", "properties": {}, "additionalProperties": False}

    def decorator(fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
        spec = ToolSpec(name=name, description=description, parameters=schema)
        setattr(fn, _TOOL_ATTR, spec)
        return fn

    return decorator


class _UserFunctionTool(Tool):
    """Adapter wrapping a user @tool function as a :class:`Tool` instance."""

    def __init__(self, spec: ToolSpec, fn: Callable[[dict[str, Any]], Any]) -> None:
        self.name = spec.name
        self.description = spec.description
        self.parameters = spec.parameters
        self._fn = fn

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = self._fn(arguments or {})
        except ToolError as exc:
            return error_result(str(exc))
        except Exception as exc:  # noqa: BLE001
            return error_result(f"user tool {self.name!r} raised: {exc!r}")
        if isinstance(result, ToolResult):
            return result
        if result is None:
            return text_result("")
        return text_result(str(result))


def _load_module(py_file: Path) -> Any | None:
    """Import (or re-import) a single ``.py`` file as an anonymous module."""
    mod_name = f"devin_local_user_tool_{py_file.stem}_{int(py_file.stat().st_mtime)}"
    spec = importlib.util.spec_from_file_location(mod_name, py_file)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def load_user_tools(directory: Path) -> UserToolsReport:
    """Import every ``.py`` file under ``directory`` and collect @tool funcs.

    Returns a report with both the successfully loaded tools and per-file
    error details (so the GUI can show a "tool failed to load" banner
    without crashing the agent).
    """
    report = UserToolsReport()
    if not directory.exists() or not directory.is_dir():
        return report
    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            module = _load_module(py_file)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(
                UserToolError(
                    source_path=py_file,
                    message=f"import failed: {exc!r}",
                    traceback=traceback.format_exc(),
                )
            )
            log.warning("user tool import failed (%s): %s", py_file, exc)
            continue
        if module is None:
            continue
        # Pull every @tool-decorated function in the module.
        found_any = False
        for _, attr in inspect.getmembers(module):
            spec = getattr(attr, _TOOL_ATTR, None)
            if spec is None or not callable(attr):
                continue
            found_any = True
            wrapped = _UserFunctionTool(spec, attr)
            report.loaded.append(
                LoadedUserTool(
                    tool=wrapped,
                    source_path=py_file,
                    mtime=py_file.stat().st_mtime,
                )
            )
        if not found_any:
            report.errors.append(
                UserToolError(
                    source_path=py_file,
                    message=f"{py_file.name} contains no @tool functions",
                )
            )
    return report


def register_user_tools(registry: Any, directory: Path) -> UserToolsReport:
    """Load + register user tools onto ``registry``.

    Duplicate names overwrite previously-registered user tools (so an edit
    can be reloaded without restarting); they refuse to overwrite a
    builtin to avoid the operator accidentally shadowing core tools. A
    descriptive entry is added to ``report.errors`` when that happens.
    """
    report = load_user_tools(directory)
    # Drop previously-registered user tools FIRST so we can correctly
    # snapshot the set of actual builtins. Without this, a tool reloaded
    # for the second time would see its own previous registration as a
    # "builtin" and refuse to overwrite itself.
    previous_user = getattr(registry, "_user_tool_paths", {})
    for old_name in list(previous_user.keys()):
        if old_name in registry:
            registry.unregister(old_name)
    builtin_names = set(registry.names()) if hasattr(registry, "names") else set()
    new_user_paths: dict[str, Path] = {}
    for loaded in report.loaded:
        name = loaded.tool.name
        if name in builtin_names:
            report.errors.append(
                UserToolError(
                    source_path=loaded.source_path,
                    message=(
                        f"refusing to overwrite builtin tool {name!r} — pick a different @tool name"
                    ),
                )
            )
            continue
        if name in registry:
            registry.unregister(name)
        registry.register(loaded.tool)
        new_user_paths[name] = loaded.source_path
    # Stash on the registry so a later reload can clear our previous additions.
    with contextlib.suppress(Exception):
        registry._user_tool_paths = new_user_paths  # noqa: SLF001 - intentional
    return report


def list_user_tool_files(directory: Path) -> list[Path]:
    """Return every ``.py`` file in ``directory`` sorted by name."""
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.py") if not p.name.startswith("_"))


_STARTER_TEMPLATE = '''\
"""User tool: {name}.

Generated by the devin-local Settings → Tools tab on {when}.
Edit this file directly, or use the dialog to update / delete it.
"""

from devin_local.tools.user_tools import tool


@tool(
    name="{name}",
    description="{description}",
    parameters={{
        "type": "object",
        "properties": {{
            "text": {{"type": "string", "description": "Input text."}}
        }},
        "required": ["text"],
        "additionalProperties": False,
    }},
)
def {name}(arguments):
    """Implement your tool here. Return a string or a ToolResult."""
    text = arguments["text"]
    return f"You sent: {{text!r}}"
'''


def starter_template(name: str, description: str) -> str:
    """Return a self-contained @tool starter for a new user tool file."""
    safe = "".join(c for c in name if c.isalnum() or c == "_") or "my_tool"
    desc = description.strip() or "TODO: describe what this tool does."
    return _STARTER_TEMPLATE.format(
        name=safe, description=desc.replace('"', '\\"'), when=time.strftime("%Y-%m-%d")
    )
