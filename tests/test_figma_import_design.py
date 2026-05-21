"""Tests for the Figma → UiDocument import logic.

We construct fake Figma file payloads (matching the shape returned by
``GET /v1/files/{key}``) and assert the import maps them to the expected
:class:`UiDocument` elements. No network calls.
"""

from __future__ import annotations

from devin_local.figma.import_design import import_figma_file_as_document


def _make_payload(frame_children: list[dict]) -> dict:
    return {
        "name": "Test design",
        "document": {
            "children": [
                {
                    "type": "CANVAS",
                    "children": [
                        {
                            "type": "FRAME",
                            "name": "Home",
                            "absoluteBoundingBox": {
                                "x": 100,
                                "y": 200,
                                "width": 800,
                                "height": 600,
                            },
                            "children": frame_children,
                        }
                    ],
                }
            ]
        },
    }


def test_import_promotes_large_text_to_heading() -> None:
    payload = _make_payload(
        [
            {
                "type": "TEXT",
                "name": "Title",
                "absoluteBoundingBox": {"x": 116, "y": 220, "width": 200, "height": 32},
                "characters": "Hello",
                "style": {"fontSize": 28, "fontWeight": 700},
            }
        ]
    )
    doc = import_figma_file_as_document(payload)
    assert len(doc.elements) == 1
    el = doc.elements[0]
    assert el.type == "heading"
    assert el.props["text"] == "Hello"
    assert el.props["bold"] is True
    # geometry translated relative to root frame origin
    assert el.x == 16
    assert el.y == 20


def test_import_uses_label_for_small_text() -> None:
    payload = _make_payload(
        [
            {
                "type": "TEXT",
                "name": "Body",
                "absoluteBoundingBox": {"x": 116, "y": 280, "width": 400, "height": 24},
                "characters": "Body copy.",
                "style": {"fontSize": 14, "fontWeight": 400},
            }
        ]
    )
    doc = import_figma_file_as_document(payload)
    assert len(doc.elements) == 1
    el = doc.elements[0]
    assert el.type == "label"
    assert el.props["bold"] is False


def test_import_detects_button() -> None:
    payload = _make_payload(
        [
            {
                "type": "RECTANGLE",
                "name": "CTA",
                "absoluteBoundingBox": {"x": 116, "y": 340, "width": 160, "height": 40},
                "cornerRadius": 8,
                "fills": [{"type": "SOLID", "color": {"r": 0.5, "g": 0.5, "b": 1, "a": 1}}],
            }
        ]
    )
    doc = import_figma_file_as_document(payload)
    assert len(doc.elements) == 1
    el = doc.elements[0]
    assert el.type == "button"
    assert el.props["text"] == "CTA"
    assert el.width == 160
    assert el.height == 40


def test_import_uses_button_label_from_nested_text() -> None:
    payload = _make_payload(
        [
            {
                "type": "FRAME",
                "name": "CTA wrapper",
                "absoluteBoundingBox": {"x": 116, "y": 340, "width": 160, "height": 40},
                "cornerRadius": 8,
                "fills": [{"type": "SOLID", "color": {"r": 0.5, "g": 0.5, "b": 1, "a": 1}}],
                "children": [
                    {
                        "type": "TEXT",
                        "name": "Label",
                        "absoluteBoundingBox": {
                            "x": 130,
                            "y": 348,
                            "width": 130,
                            "height": 24,
                        },
                        "characters": "Click me",
                        "style": {"fontSize": 14, "fontWeight": 600},
                    }
                ],
            }
        ]
    )
    doc = import_figma_file_as_document(payload)
    types = [e.type for e in doc.elements]
    assert "button" in types
    button = next(e for e in doc.elements if e.type == "button")
    assert button.props["text"] == "Click me"


def test_import_returns_empty_doc_for_no_frame() -> None:
    payload = {"name": "Empty", "document": {"children": []}}
    doc = import_figma_file_as_document(payload)
    assert doc.elements == []


def test_import_walks_nested_frames() -> None:
    payload = _make_payload(
        [
            {
                "type": "FRAME",
                "name": "Section",
                "absoluteBoundingBox": {"x": 120, "y": 250, "width": 600, "height": 400},
                "children": [
                    {
                        "type": "TEXT",
                        "name": "Nested",
                        "absoluteBoundingBox": {
                            "x": 130,
                            "y": 260,
                            "width": 200,
                            "height": 30,
                        },
                        "characters": "Nested text",
                        "style": {"fontSize": 16, "fontWeight": 500},
                    }
                ],
            }
        ]
    )
    doc = import_figma_file_as_document(payload)
    assert any(e.props.get("text") == "Nested text" for e in doc.elements)
