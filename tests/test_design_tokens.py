"""Tests for the design-tokens module."""

from __future__ import annotations

import re

from devin_local.gui.design_tokens import TOKENS, as_css_vars


def test_palette_colors_are_hex() -> None:
    palette = TOKENS.palette
    hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
    for field_name in ("bg_root", "bg_chrome", "bg_panel", "text", "accent", "success", "error"):
        value = getattr(palette, field_name)
        assert hex_re.match(value), f"{field_name}={value!r} should be 6-digit hex"


def test_spacing_is_monotonic() -> None:
    s = TOKENS.spacing
    sizes = [s.xxs, s.xs, s.sm, s.md, s.lg, s.xl, s.xxl]
    assert sizes == sorted(sizes), "spacing scale must be non-decreasing"
    assert len(set(sizes)) == len(sizes), "spacing values must be distinct"


def test_typography_scale_is_monotonic() -> None:
    t = TOKENS.typography
    sizes = [t.size_xs, t.size_sm, t.size_md, t.size_lg, t.size_xl, t.size_xxl]
    assert sizes == sorted(sizes), "type scale must be non-decreasing"


def test_css_vars_has_all_tokens() -> None:
    vars_ = as_css_vars()
    for required in (
        "--color-bg-root",
        "--color-accent",
        "--color-error",
        "--space-md",
        "--radius-pill",
        "--font-mono",
        "--size-md",
    ):
        assert required in vars_, f"missing CSS var {required}"


def test_token_palette_matches_theme_palette() -> None:
    """The Figma-style tokens should agree with the existing QSS palette so
    we have ONE color system, not two."""
    from devin_local.gui.theme import DARK_PALETTE

    palette = TOKENS.palette
    assert palette.accent == DARK_PALETTE["accent"]
    assert palette.success == DARK_PALETTE["ok"]
    assert palette.error == DARK_PALETTE["err"]
    assert palette.warn == DARK_PALETTE["warn"]
