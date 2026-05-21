"""Walk a Figma file's node tree and emit a flat design-token dict.

Specifically what we extract:

- **Colors** (``fills`` with type=SOLID on any node, plus any
  ``COLOR``/``FILL`` styles defined in the file's ``styles`` table).
- **Typography** (``style`` on TEXT nodes: family + weight + size +
  lineHeight + letterSpacing, plus any TEXT styles in the styles table).
- **Spacing** (``itemSpacing`` and ``padding*`` on auto-layout frames).
- **Radii** (``cornerRadius`` on frames / rects).
- **Shadows** (``effects`` with type=DROP_SHADOW or INNER_SHADOW).

Output is a :class:`DesignTokens` dataclass, plus a helper
:func:`tokens_to_python_module` that renders them as a drop-in replacement
for ``devin_local.gui.design_tokens``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


def _rgba_to_hex(rgba: dict[str, float]) -> str:
    """Convert Figma's 0..1 float RGBA to ``#RRGGBB`` or ``#RRGGBBAA``."""
    r = max(0, min(255, int(round(float(rgba.get("r", 0)) * 255))))
    g = max(0, min(255, int(round(float(rgba.get("g", 0)) * 255))))
    b = max(0, min(255, int(round(float(rgba.get("b", 0)) * 255))))
    a = float(rgba.get("a", 1.0))
    if a >= 1.0:
        return f"#{r:02x}{g:02x}{b:02x}"
    ai = max(0, min(255, int(round(a * 255))))
    return f"#{r:02x}{g:02x}{b:02x}{ai:02x}"


@dataclass
class DesignTokens:
    """Collected design tokens from a Figma file."""

    colors: dict[str, str] = field(default_factory=dict)
    typography: dict[str, dict[str, Any]] = field(default_factory=dict)
    radii: dict[str, float] = field(default_factory=dict)
    spacing: dict[str, float] = field(default_factory=dict)
    shadows: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_file_key: str = ""
    source_file_name: str = ""


def _safe_key(s: str, fallback: str) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in (s or "").strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _iter_nodes(node: dict[str, Any]):
    """Depth-first walk over a Figma node and its children."""
    if not isinstance(node, dict):
        return
    yield node
    for child in node.get("children") or []:
        yield from _iter_nodes(child)


def extract_design_tokens(file_payload: dict[str, Any]) -> DesignTokens:
    """Run the extraction on a payload returned by :meth:`FigmaClient.get_file`."""
    tokens = DesignTokens()
    tokens.source_file_key = str(file_payload.get("key", "") or "")
    tokens.source_file_name = str(file_payload.get("name", "") or "")
    document = file_payload.get("document") or {}
    styles_table = file_payload.get("styles") or {}

    color_counter = 0
    type_counter = 0
    shadow_counter = 0
    radius_counter = 0
    spacing_counter = 0
    seen_colors: dict[str, str] = OrderedDict()

    for node in _iter_nodes(document):
        # Colors via fills.
        for fill in node.get("fills") or []:
            if fill.get("type") != "SOLID":
                continue
            hex_value = _rgba_to_hex(fill.get("color") or {})
            if hex_value in seen_colors:
                continue
            # Prefer a named style if Figma attached one to this fill on the
            # node.
            style_id = (node.get("styles") or {}).get("fill")
            name = (styles_table.get(style_id) or {}).get("name", "") if style_id else ""
            color_counter += 1
            key = _safe_key(name, f"color_{color_counter:02d}")
            tokens.colors[key] = hex_value
            seen_colors[hex_value] = key

        # Typography via TEXT nodes.
        if node.get("type") == "TEXT":
            style = node.get("style") or {}
            if style:
                type_counter += 1
                style_id = (node.get("styles") or {}).get("text")
                name = (styles_table.get(style_id) or {}).get("name", "") if style_id else ""
                key = _safe_key(name, f"text_{type_counter:02d}")
                tokens.typography[key] = {
                    "family": style.get("fontFamily", ""),
                    "weight": int(style.get("fontWeight", 400) or 400),
                    "size": float(style.get("fontSize", 14.0) or 14.0),
                    "line_height_px": float(
                        style.get("lineHeightPx") or style.get("fontSize") or 14.0
                    ),
                    "letter_spacing": float(style.get("letterSpacing", 0.0) or 0.0),
                }

        # Radii.
        cr = node.get("cornerRadius")
        if isinstance(cr, (int, float)) and cr:
            radius_counter += 1
            tokens.radii[f"radius_{radius_counter:02d}"] = float(cr)

        # Auto-layout spacing.
        for key in ("itemSpacing", "paddingLeft", "paddingTop", "paddingRight", "paddingBottom"):
            value = node.get(key)
            if isinstance(value, (int, float)) and value:
                spacing_counter += 1
                tokens.spacing[f"spacing_{spacing_counter:02d}_{key}"] = float(value)

        # Effects (drop / inner shadows).
        for effect in node.get("effects") or []:
            if not effect.get("visible", True):
                continue
            kind = effect.get("type")
            if kind not in {"DROP_SHADOW", "INNER_SHADOW"}:
                continue
            shadow_counter += 1
            offset = effect.get("offset") or {"x": 0, "y": 0}
            tokens.shadows[f"shadow_{shadow_counter:02d}"] = {
                "kind": kind.lower(),
                "color": _rgba_to_hex(effect.get("color") or {}),
                "radius": float(effect.get("radius", 0.0) or 0.0),
                "offset_x": float(offset.get("x", 0.0) or 0.0),
                "offset_y": float(offset.get("y", 0.0) or 0.0),
                "spread": float(effect.get("spread", 0.0) or 0.0),
            }

    # Also fold standalone color styles defined in the file (some files
    # define swatches as styles without any visible nodes using them).
    for _style_id, style in styles_table.items():
        if style.get("styleType") == "FILL":
            color_counter += 1
            tokens.colors.setdefault(
                _safe_key(style.get("name", ""), f"color_{color_counter:02d}"),
                "#000000",
            )

    return tokens


def tokens_to_python_module(tokens: DesignTokens) -> str:
    """Render :class:`DesignTokens` as a Python module string.

    The output is meant to be diffed against the hand-authored
    ``design_tokens.py`` module — the operator (or the agent) can choose
    which tokens to adopt.
    """
    lines: list[str] = [
        '"""Auto-generated from a Figma file via `devin_local.figma.tokens`.\n\n'
        f"Source: {tokens.source_file_name!r} (key={tokens.source_file_key!r})\n"
        "Do not hand-edit; regenerate from Settings \u2192 Appearance \u2192 'Refresh from Figma'.\n"
        '"""\n',
        "from __future__ import annotations\n",
        "FIGMA_COLORS = {",
    ]
    for key, value in tokens.colors.items():
        lines.append(f"    {key!r}: {value!r},")
    lines.append("}\n")

    lines.append("FIGMA_TYPOGRAPHY = {")
    for key, payload in tokens.typography.items():
        lines.append(f"    {key!r}: {payload!r},")
    lines.append("}\n")

    lines.append("FIGMA_RADII = {")
    for key, value in tokens.radii.items():
        lines.append(f"    {key!r}: {value!r},")
    lines.append("}\n")

    lines.append("FIGMA_SPACING = {")
    for key, value in tokens.spacing.items():
        lines.append(f"    {key!r}: {value!r},")
    lines.append("}\n")

    lines.append("FIGMA_SHADOWS = {")
    for key, payload in tokens.shadows.items():
        lines.append(f"    {key!r}: {payload!r},")
    lines.append("}\n")
    return "\n".join(lines) + "\n"
