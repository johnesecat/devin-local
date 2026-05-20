"""Design tokens for the devin-local GUI.

This module is the single source of truth for colors, spacing, typography,
radii, shadows, and motion. ``gui/theme.py`` reads from here to build the
Qt stylesheet, and individual widgets reach in for one-off values (e.g.
icon tint).

The design-system spec lives at ``docs/design-system.md`` — when you change
a token here, mirror the change there so the visual system stays
documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Palette:
    # Matches the existing tokyo-night palette in ``gui/theme.py`` so old QSS
    # and new design-tokened code render identically.
    bg_root: str = "#0d0f17"
    bg_chrome: str = "#13151f"
    bg_panel: str = "#1a1b26"
    bg_panel_alt: str = "#24283b"
    bg_input: str = "#13151f"
    border: str = "#2f3549"
    border_strong: str = "#3b4261"
    text: str = "#c0caf5"
    text_muted: str = "#9aa5ce"
    text_subtle: str = "#565f89"
    accent: str = "#7aa2f7"
    accent_hover: str = "#bb9af7"
    accent_press: str = "#5e87df"
    success: str = "#9ece6a"
    warn: str = "#e0af68"
    error: str = "#f7768e"
    info: str = "#7dcfff"
    code_bg: str = "#11141b"


@dataclass(frozen=True)
class Spacing:
    xxs: int = 2
    xs: int = 4
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 24
    xxl: int = 32


@dataclass(frozen=True)
class Radii:
    sm: int = 4
    md: int = 8
    lg: int = 12
    pill: int = 999


@dataclass(frozen=True)
class Typography:
    family_ui: str = "'Inter', 'Segoe UI', 'SF Pro Text', system-ui, sans-serif"
    family_mono: str = "'JetBrains Mono', 'Cascadia Mono', Consolas, 'Liberation Mono', monospace"
    size_xs: int = 11
    size_sm: int = 12
    size_md: int = 13
    size_lg: int = 15
    size_xl: int = 18
    size_xxl: int = 22
    weight_regular: int = 400
    weight_medium: int = 500
    weight_bold: int = 600


@dataclass(frozen=True)
class Motion:
    fast_ms: int = 120
    medium_ms: int = 200
    slow_ms: int = 320


@dataclass(frozen=True)
class DesignTokens:
    palette: Palette = field(default_factory=Palette)
    spacing: Spacing = field(default_factory=Spacing)
    radii: Radii = field(default_factory=Radii)
    typography: Typography = field(default_factory=Typography)
    motion: Motion = field(default_factory=Motion)


TOKENS = DesignTokens()
"""The active token set. Replace at import time if/when we add light theme."""


def as_css_vars(tokens: DesignTokens = TOKENS) -> dict[str, str]:
    """Flat dict of ``--name: value`` mappings, useful for stylesheet templating."""
    palette = tokens.palette
    spacing = tokens.spacing
    radii = tokens.radii
    typography = tokens.typography
    return {
        "--color-bg-root": palette.bg_root,
        "--color-bg-chrome": palette.bg_chrome,
        "--color-bg-panel": palette.bg_panel,
        "--color-bg-panel-alt": palette.bg_panel_alt,
        "--color-bg-input": palette.bg_input,
        "--color-border": palette.border,
        "--color-border-strong": palette.border_strong,
        "--color-text": palette.text,
        "--color-text-muted": palette.text_muted,
        "--color-text-subtle": palette.text_subtle,
        "--color-accent": palette.accent,
        "--color-accent-hover": palette.accent_hover,
        "--color-accent-press": palette.accent_press,
        "--color-success": palette.success,
        "--color-warn": palette.warn,
        "--color-error": palette.error,
        "--color-info": palette.info,
        "--color-code-bg": palette.code_bg,
        "--space-xxs": f"{spacing.xxs}px",
        "--space-xs": f"{spacing.xs}px",
        "--space-sm": f"{spacing.sm}px",
        "--space-md": f"{spacing.md}px",
        "--space-lg": f"{spacing.lg}px",
        "--space-xl": f"{spacing.xl}px",
        "--space-xxl": f"{spacing.xxl}px",
        "--radius-sm": f"{radii.sm}px",
        "--radius-md": f"{radii.md}px",
        "--radius-lg": f"{radii.lg}px",
        "--radius-pill": f"{radii.pill}px",
        "--font-ui": typography.family_ui,
        "--font-mono": typography.family_mono,
        "--size-xs": f"{typography.size_xs}px",
        "--size-sm": f"{typography.size_sm}px",
        "--size-md": f"{typography.size_md}px",
        "--size-lg": f"{typography.size_lg}px",
        "--size-xl": f"{typography.size_xl}px",
        "--size-xxl": f"{typography.size_xxl}px",
    }
