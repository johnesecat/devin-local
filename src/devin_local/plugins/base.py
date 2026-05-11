"""Plugin ABCs and registration record types."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginRegistration:
    """A record of one plugin that has been loaded into the agent."""

    name: str
    source: str  # "entry_point" | "folder" | "programmatic"
    register_fn: Callable[[Any], None]
    metadata: dict[str, Any] = field(default_factory=dict)


class Plugin(abc.ABC):
    """Optional base class for class-style plugins.

    Plugins do not need to inherit from `Plugin` — a free function named
    `register(agent)` in a module is also accepted. This class exists for
    plugins that want a cleaner OOP surface.
    """

    name: str = ""

    @abc.abstractmethod
    def register(self, agent: Any) -> None:
        """Hook into the agent. Add tools, system-prompt sections, etc."""
