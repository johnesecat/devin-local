"""Example plugin: adds a `current_time` tool to the agent's toolbelt.

Drop this file under the workspace's `plugins/` directory (which is where it
already lives in this repo) and `devin-local` will load it automatically on
startup.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from devin_local.tools.base import Tool, ToolResult, text_result


class CurrentTimeTool(Tool):
    name = "current_time"
    description = "Return the current local date and time as an ISO-8601 string."
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        return text_result(datetime.now().isoformat(timespec="seconds"))


def register(agent) -> None:  # noqa: ANN001 - duck-typed
    """Plugin entry point. Receives the live `Agent` instance."""
    agent.registry.register(CurrentTimeTool())
