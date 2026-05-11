"""Desktop emulation tools (screenshot, mouse, keyboard).

These wrap `pyautogui` from the optional `[desktop]` extra. We import it
lazily so the rest of the agent works fine without it installed (useful for
headless CI). All actions are funneled through `DesktopEmulator`.
"""

from __future__ import annotations

from typing import Any

from devin_local.emulators.desktop_emulator import DesktopEmulator, DesktopUnavailableError
from devin_local.tools.base import (
    Tool,
    ToolResult,
    error_result,
    optional_str,
    require_str,
    text_result,
)


class DesktopScreenshotTool(Tool):
    name = "desktop_screenshot"
    description = (
        "Capture a screenshot of the primary display and save it under "
        "screenshots/. Returns the saved path."
    )
    parameters = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Optional filename relative to screenshots/.",
            },
        },
    }

    def __init__(self, emulator: DesktopEmulator) -> None:
        self.emulator = emulator

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = self.emulator.screenshot(optional_str(arguments, "filename") or None)
        except DesktopUnavailableError as exc:
            return error_result(str(exc))
        return text_result(f"Saved screenshot to {path}", path=str(path))


class DesktopClickTool(Tool):
    name = "desktop_click"
    description = "Click the mouse at (x, y) coordinates on the primary display."
    parameters = {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "button": {"type": "string", "enum": ["left", "right", "middle"]},
            "clicks": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["x", "y"],
    }

    def __init__(self, emulator: DesktopEmulator) -> None:
        self.emulator = emulator

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            self.emulator.click(
                int(arguments["x"]),
                int(arguments["y"]),
                button=optional_str(arguments, "button", "left") or "left",
                clicks=int(arguments.get("clicks") or 1),
            )
        except DesktopUnavailableError as exc:
            return error_result(str(exc))
        return text_result(f"Clicked ({arguments['x']}, {arguments['y']}).")


class DesktopTypeTool(Tool):
    name = "desktop_type"
    description = "Type a string into the active window."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "interval": {"type": "number", "description": "Delay between keystrokes (s)."},
        },
        "required": ["text"],
    }

    def __init__(self, emulator: DesktopEmulator) -> None:
        self.emulator = emulator

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            self.emulator.type_text(
                require_str(arguments, "text"),
                interval=float(arguments.get("interval") or 0.02),
            )
        except DesktopUnavailableError as exc:
            return error_result(str(exc))
        return text_result("Typed text.")


class DesktopKeyTool(Tool):
    name = "desktop_key"
    description = "Press a keyboard shortcut, e.g. 'ctrl+s' or 'win+r'."
    parameters = {
        "type": "object",
        "properties": {
            "combo": {"type": "string"},
        },
        "required": ["combo"],
    }

    def __init__(self, emulator: DesktopEmulator) -> None:
        self.emulator = emulator

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            self.emulator.hotkey(require_str(arguments, "combo"))
        except DesktopUnavailableError as exc:
            return error_result(str(exc))
        return text_result(f"Pressed {arguments['combo']}.")
