"""Terminal and desktop emulation primitives used by the agent's tools."""

from __future__ import annotations

from devin_local.emulators.desktop_emulator import DesktopEmulator, DesktopUnavailableError
from devin_local.emulators.terminal_emulator import TerminalEmulator

__all__ = ["DesktopEmulator", "DesktopUnavailableError", "TerminalEmulator"]
