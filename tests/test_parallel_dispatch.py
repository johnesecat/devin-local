"""Tests for ToolRegistry.dispatch_many parallel dispatch."""

from __future__ import annotations

import time

from devin_local.tools.base import Tool, ToolResult
from devin_local.tools.registry import ToolRegistry


class _SleepTool(Tool):
    """Test-only tool that sleeps for `duration` seconds and reports it back."""

    description = "Sleep for the given duration."

    def __init__(self, name: str, duration: float) -> None:
        self.name = name
        self._duration = duration

    def to_ollama_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def run(self, arguments: dict) -> ToolResult:
        time.sleep(self._duration)
        return ToolResult(ok=True, output=f"slept {self._duration}s")


def test_dispatch_many_runs_in_parallel() -> None:
    registry = ToolRegistry.empty()
    registry.register(_SleepTool("sleep_a", 0.3))
    registry.register(_SleepTool("sleep_b", 0.3))
    registry.register(_SleepTool("sleep_c", 0.3))

    calls = [("sleep_a", {}), ("sleep_b", {}), ("sleep_c", {})]
    start = time.monotonic()
    results = registry.dispatch_many(calls)
    elapsed = time.monotonic() - start

    # All three results returned in order.
    assert len(results) == 3
    assert all(r.ok for r in results)
    # If we ran serially, elapsed >= 0.9s. Parallel should be ~0.3s.
    # Allow generous slack for CI: < 0.7s proves parallelism.
    assert elapsed < 0.7, f"expected parallel dispatch (~0.3s), got {elapsed:.2f}s"


def test_dispatch_many_preserves_order() -> None:
    registry = ToolRegistry.empty()
    registry.register(_SleepTool("a", 0.1))
    registry.register(_SleepTool("b", 0.01))
    registry.register(_SleepTool("c", 0.05))

    calls = [("a", {}), ("b", {}), ("c", {})]
    results = registry.dispatch_many(calls)
    outputs = [r.output for r in results]
    assert outputs == ["slept 0.1s", "slept 0.01s", "slept 0.05s"]


def test_dispatch_many_with_single_call_short_circuits() -> None:
    registry = ToolRegistry.empty()
    registry.register(_SleepTool("solo", 0.01))
    [result] = registry.dispatch_many([("solo", {})])
    assert result.ok


def test_dispatch_many_with_empty_list_returns_empty() -> None:
    registry = ToolRegistry.empty()
    assert registry.dispatch_many([]) == []
