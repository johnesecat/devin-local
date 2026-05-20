"""Icon helpers for the devin-local GUI.

Tries ``qtawesome`` (Material Symbols / Font Awesome) first. Falls back to
Unicode glyphs so the app stays usable when the optional icon font isn't
installed. Each icon is referenced by a short logical name (e.g.
``"send"``, ``"settings"``) — never reach into ``qtawesome`` directly from
widgets so we can rename the icon set in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from devin_local.gui.design_tokens import TOKENS

if TYPE_CHECKING:
    from PySide6.QtGui import QIcon

# Logical name -> (qtawesome name, fallback unicode glyph)
ICON_MAP: dict[str, tuple[str, str]] = {
    "send": ("mdi.send", "↑"),
    "settings": ("mdi.cog", "⚙"),
    "plus": ("mdi.plus", "+"),
    "minus": ("mdi.minus", "−"),
    "trash": ("mdi.trash-can-outline", "🗑"),
    "refresh": ("mdi.refresh", "↻"),
    "github": ("mdi.github", ""),
    "robot": ("mdi.robot", "◆"),
    "user": ("mdi.account-circle", "●"),
    "play": ("mdi.play", "▶"),
    "check": ("mdi.check", "✓"),
    "x": ("mdi.close", "✕"),
    "warn": ("mdi.alert", "!"),
    "spark": ("mdi.lightning-bolt", "⚡"),
    "folder": ("mdi.folder-outline", "▣"),
    "file": ("mdi.file-document-outline", "▤"),
    "terminal": ("mdi.console", "❯_"),
    "sandbox": ("mdi.cube-outline", "▢"),
    "plan": ("mdi.format-list-checks", "≡"),
    "copy": ("mdi.content-copy", "⎘"),
    "search": ("mdi.magnify", "⌕"),
    "link": ("mdi.link-variant", "↗"),
    "tool": ("mdi.tools", "✦"),
    "model": ("mdi.brain", "◊"),
    "backend": ("mdi.server", "▤"),
    "mcp": ("mdi.transit-connection-variant", "⇄"),
    "appearance": ("mdi.palette-outline", "❖"),
    "general": ("mdi.tune", "≡"),
    "figma": ("mdi.alpha-f-box", "F"),
    "knowledge": ("mdi.book-open-page-variant", "📖"),
    "tools": ("mdi.tools", "⚙"),
    "import": ("mdi.download-outline", "⤓"),
    "export": ("mdi.upload-outline", "⤒"),
    "design": ("mdi.shape-outline", "◇"),
    "ui_creator": ("mdi.gesture-tap", "⬚"),
    "session": ("mdi.message-text-outline", "✎"),
    "edit": ("mdi.pencil-outline", "✎"),
    "open": ("mdi.folder-open-outline", "📂"),
    "save": ("mdi.content-save", "💾"),
}


def has_qtawesome() -> bool:
    try:
        import qtawesome  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def icon(name: str, *, color: str | None = None) -> QIcon | None:
    """Return a ``QIcon`` for the named glyph, or ``None`` if PySide6 isn't loaded.

    Pass ``color`` to tint the icon (default: accent color from tokens).
    """
    try:
        from PySide6.QtGui import QIcon  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    if not has_qtawesome():
        return None
    import qtawesome as qta

    fa_name, _glyph = ICON_MAP.get(name, ("", ""))
    if not fa_name:
        return None
    tint = color or TOKENS.palette.accent
    try:
        return qta.icon(fa_name, color=tint)
    except Exception:  # noqa: BLE001
        return None


def glyph(name: str) -> str:
    """Return the Unicode fallback glyph (always works, no deps)."""
    _, fallback = ICON_MAP.get(name, ("", "?"))
    return fallback
