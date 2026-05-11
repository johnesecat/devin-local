"""Desktop emulator. Wraps `pyautogui` with graceful fallback.

`pyautogui` is in the optional `[desktop]` extra. On headless CI (no DISPLAY,
no Win32 desktop), we surface a `DesktopUnavailableError` instead of crashing
on import.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class DesktopUnavailableError(RuntimeError):
    """Raised when the desktop emulator cannot operate (missing deps / no display)."""


class DesktopEmulator:
    """Thin wrapper over `pyautogui` for screenshot + input automation."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.screenshot_dir = workspace / "screenshots"
        self._pyautogui = None  # lazy import

    def _load(self):  # noqa: ANN202 - return type is the pyautogui module
        if self._pyautogui is not None:
            return self._pyautogui
        try:
            import pyautogui  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise DesktopUnavailableError(
                "pyautogui is not installed. Install the desktop extra: "
                "`pip install 'devin-local[desktop]'`."
            ) from exc
        # Disable the safety corner since we're driving programmatically and
        # the agent is responsible for sanity-checking its own clicks.
        pyautogui.FAILSAFE = False
        self._pyautogui = pyautogui
        return pyautogui

    def screenshot(self, filename: str | None = None) -> Path:
        pag = self._load()
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        if not filename:
            filename = datetime.now().strftime("screenshot-%Y%m%d-%H%M%S.png")
        if not filename.lower().endswith(".png"):
            filename += ".png"
        out = self.screenshot_dir / filename
        try:
            img = pag.screenshot()
        except Exception as exc:  # noqa: BLE001
            raise DesktopUnavailableError(f"Could not capture screen: {exc}") from exc
        img.save(out)
        return out

    def click(self, x: int, y: int, *, button: str = "left", clicks: int = 1) -> None:
        pag = self._load()
        try:
            pag.click(x=x, y=y, button=button, clicks=max(1, min(3, clicks)))
        except Exception as exc:  # noqa: BLE001
            raise DesktopUnavailableError(f"Click failed: {exc}") from exc

    def type_text(self, text: str, *, interval: float = 0.02) -> None:
        pag = self._load()
        try:
            pag.typewrite(text, interval=interval)
        except Exception as exc:  # noqa: BLE001
            raise DesktopUnavailableError(f"Typing failed: {exc}") from exc

    def hotkey(self, combo: str) -> None:
        pag = self._load()
        keys = [k.strip() for k in combo.replace("-", "+").split("+") if k.strip()]
        if not keys:
            raise DesktopUnavailableError(f"Empty key combo: {combo!r}")
        try:
            pag.hotkey(*keys)
        except Exception as exc:  # noqa: BLE001
            raise DesktopUnavailableError(f"Hotkey failed: {exc}") from exc


__all__ = ["DesktopEmulator", "DesktopUnavailableError"]
