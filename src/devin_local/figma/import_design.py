"""Convert a Figma file payload into a :class:`UiDocument`.

This is a *best-effort* import: Figma's data model is richer than the UI
Creator's, so we map node types onto our element types using simple
heuristics. The result is a starting point — the operator is expected to
tweak it inside the UI Creator. The conversion is fully local and
deterministic; no network calls happen here.

Mapping rules
-------------
- ``TEXT``  → ``label`` (or ``heading`` if ``fontSize >= 22``).
- ``RECTANGLE`` / ``FRAME`` / ``COMPONENT`` / ``INSTANCE`` with
  ``cornerRadius`` and a fill, sized roughly like a button (height ≤ 64
  px), → ``button``.
- ``FRAME`` / ``GROUP`` / ``COMPONENT`` containers → recursed into
  (their children are imported as siblings at the same depth, offset by
  the parent's bounds).
- Anything else is skipped.

We use ``absoluteBoundingBox`` for geometry and translate every element
so the chosen root frame sits at ``(0, 0)``.
"""

from __future__ import annotations

from typing import Any

from devin_local.ui_creator.document import UiDocument, UiElement


def _bbox(node: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bb = node.get("absoluteBoundingBox")
    if not isinstance(bb, dict):
        return None
    try:
        return (
            float(bb.get("x", 0)),
            float(bb.get("y", 0)),
            float(bb.get("width", 0)),
            float(bb.get("height", 0)),
        )
    except (TypeError, ValueError):
        return None


def _pick_root_frame(file_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first non-trivial top-level frame in the file."""
    doc = file_payload.get("document") or {}
    for page in doc.get("children") or []:
        for child in page.get("children") or []:
            if child.get("type") in {"FRAME", "COMPONENT", "COMPONENT_SET", "SECTION"}:
                bb = _bbox(child)
                if bb and bb[2] >= 32 and bb[3] >= 32:
                    return child
    return None


def _looks_like_button(node: dict[str, Any]) -> bool:
    if node.get("type") not in {"RECTANGLE", "FRAME", "COMPONENT", "INSTANCE"}:
        return False
    cr = node.get("cornerRadius")
    if not isinstance(cr, (int, float)) or cr <= 0:
        return False
    bb = _bbox(node)
    if not bb:
        return False
    _, _, w, h = bb
    return 24 <= h <= 72 and 40 <= w <= 480


def _text_props(node: dict[str, Any]) -> dict[str, Any]:
    style = node.get("style") or {}
    text = str(node.get("characters", "") or "").strip() or "Text"
    return {
        "text": text,
        "font_pt": int(round(float(style.get("fontSize", 11) or 11))),
        "bold": int(style.get("fontWeight", 400) or 400) >= 600,
        "italic": str(style.get("italic") or "").lower() == "true",
    }


def import_figma_file_as_document(
    file_payload: dict[str, Any], *, document_name: str = ""
) -> UiDocument:
    """Build a :class:`UiDocument` from a Figma ``get_file`` payload.

    If we can't find a usable root frame, returns an empty document.
    """
    name = document_name or str(file_payload.get("name", "") or "Figma import")
    doc = UiDocument.new(name=name)

    root = _pick_root_frame(file_payload)
    if root is None:
        return doc
    root_bb = _bbox(root)
    if root_bb is None:
        return doc
    ox, oy, rw, rh = root_bb
    doc.canvas_width = int(max(400, min(2000, rw)))
    doc.canvas_height = int(max(300, min(2400, rh)))

    def _walk(node: dict[str, Any]) -> None:
        bb = _bbox(node)
        kind = node.get("type")
        if bb is not None and kind in {"TEXT"}:
            x, y, w, h = bb
            props = _text_props(node)
            type_id = "heading" if props["font_pt"] >= 22 or props["bold"] else "label"
            element = UiElement.new(type_id, x=int(x - ox), y=int(y - oy))
            element.width = max(40, int(w))
            element.height = max(16, int(h))
            element.props.update(props)
            element.name = str(node.get("name", "") or "").strip()
            doc.elements.append(element)
            return
        if bb is not None and _looks_like_button(node):
            x, y, w, h = bb
            label = ""
            for child in node.get("children") or []:
                if child.get("type") == "TEXT":
                    label = str(child.get("characters", "") or "").strip()
                    if label:
                        break
            element = UiElement.new("button", x=int(x - ox), y=int(y - oy))
            element.width = max(60, int(w))
            element.height = max(28, int(h))
            element.props["text"] = label or str(node.get("name", "") or "Button").strip()
            element.name = str(node.get("name", "") or "").strip()
            doc.elements.append(element)
            return
        # Containers: recurse into children at the same flattened depth.
        if kind in {"FRAME", "GROUP", "COMPONENT", "COMPONENT_SET", "INSTANCE", "SECTION"}:
            for child in node.get("children") or []:
                _walk(child)

    for child in root.get("children") or []:
        _walk(child)
    return doc
