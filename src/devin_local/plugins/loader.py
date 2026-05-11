"""Plugin discovery and loading."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from importlib.metadata import entry_points
from pathlib import Path

from devin_local.plugins.base import PluginRegistration

ENTRY_POINT_GROUP = "devin_local.plugins"


def load_plugins_from_dir(directory: Path) -> list[PluginRegistration]:
    """Load every `*.py` file in `directory` as a plugin module."""
    results: list[PluginRegistration] = []
    if not directory.exists() or not directory.is_dir():
        return results
    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        mod_name = f"devin_local_plugin_{py_file.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, py_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[plugins] failed to load {py_file}: {exc}\n")
            continue
        register_fn = getattr(module, "register", None)
        if not callable(register_fn):
            continue
        results.append(
            PluginRegistration(
                name=py_file.stem,
                source="folder",
                register_fn=register_fn,
                metadata={"path": str(py_file)},
            )
        )
    return results


def load_plugins_from_entry_points() -> list[PluginRegistration]:
    """Discover plugins registered via packaging entry points."""
    results: list[PluginRegistration] = []
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:
        # Python < 3.10 fallback (we don't target it, but be defensive).
        all_eps = entry_points()
        eps = all_eps.get(ENTRY_POINT_GROUP, [])  # type: ignore[assignment]
    for ep in eps:
        try:
            obj = ep.load()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[plugins] failed to load entry point {ep.name}: {exc}\n")
            continue
        if not callable(obj):
            continue
        results.append(
            PluginRegistration(
                name=ep.name,
                source="entry_point",
                register_fn=obj,
                metadata={"module": ep.value},
            )
        )
    return results


def discover_plugins(workspace_plugins_dir: Path) -> list[PluginRegistration]:
    """Combine folder-based and entry-point-based plugins."""
    return [
        *load_plugins_from_dir(workspace_plugins_dir),
        *load_plugins_from_entry_points(),
    ]
