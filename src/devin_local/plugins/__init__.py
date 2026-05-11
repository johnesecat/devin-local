"""Plugin system.

A devin-local plugin is a Python module that exposes a top-level
`register(agent)` function. The function may:

    - call `agent.registry.register(MyTool(...))` to add tools,
    - append to `agent.system_prompt_sections` to extend the system prompt,
    - read configuration from `agent.config`.

Plugins are discovered from three sources:

    1. The `plugins/` directory at the workspace root (any `*.py` file).
    2. Any installed package registered under the
       `devin_local.plugins` entry-points group.
    3. Programmatic registration via `Agent.use_plugin(callable)`.
"""

from __future__ import annotations

from devin_local.plugins.base import Plugin, PluginRegistration
from devin_local.plugins.loader import discover_plugins, load_plugins_from_dir

__all__ = ["Plugin", "PluginRegistration", "discover_plugins", "load_plugins_from_dir"]
