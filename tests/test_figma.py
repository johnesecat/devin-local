"""Figma client + token extractor — no live network."""

from __future__ import annotations

import json

import pytest

from devin_local.figma import (
    FigmaClient,
    FigmaError,
    extract_design_tokens,
    parse_file_key,
    tokens_to_python_module,
)


def test_parse_file_key_from_design_url():
    assert parse_file_key("https://www.figma.com/design/AbCd1234/Demo?node-id=1-2") == "AbCd1234"


def test_parse_file_key_from_legacy_file_url():
    assert parse_file_key("https://www.figma.com/file/XYZ987/Old-Style") == "XYZ987"


def test_parse_file_key_passthrough_bare_key():
    assert parse_file_key("AbCd1234") == "AbCd1234"


def test_parse_file_key_rejects_empty():
    with pytest.raises(FigmaError):
        parse_file_key("")


def test_figma_client_requires_token():
    client = FigmaClient(token="")
    with pytest.raises(FigmaError):
        client.me()


def test_extract_design_tokens_collects_colors_and_typography():
    payload = json.loads(
        json.dumps(
            {
                "key": "k",
                "name": "Demo",
                "document": {
                    "id": "0:0",
                    "children": [
                        {
                            "id": "1:1",
                            "type": "FRAME",
                            "cornerRadius": 8,
                            "itemSpacing": 12,
                            "paddingLeft": 16,
                            "fills": [
                                {
                                    "type": "SOLID",
                                    "color": {
                                        "r": 0.1,
                                        "g": 0.2,
                                        "b": 0.9,
                                        "a": 1.0,
                                    },
                                }
                            ],
                            "effects": [
                                {
                                    "type": "DROP_SHADOW",
                                    "visible": True,
                                    "color": {"r": 0, "g": 0, "b": 0, "a": 0.5},
                                    "offset": {"x": 0, "y": 4},
                                    "radius": 12,
                                    "spread": 0,
                                }
                            ],
                        },
                        {
                            "id": "2:1",
                            "type": "TEXT",
                            "style": {
                                "fontFamily": "Inter",
                                "fontWeight": 600,
                                "fontSize": 14,
                                "lineHeightPx": 18,
                            },
                            "styles": {"text": "tStyle"},
                        },
                    ],
                },
                "styles": {"tStyle": {"styleType": "TEXT", "name": "Heading/H1"}},
            }
        )
    )
    tokens = extract_design_tokens(payload)
    assert tokens.source_file_name == "Demo"
    assert any(v == "#1a33e6" for v in tokens.colors.values())
    assert tokens.typography["heading_h1"]["family"] == "Inter"
    assert tokens.radii  # cornerRadius captured
    assert tokens.spacing  # paddingLeft + itemSpacing captured
    assert tokens.shadows  # drop shadow captured
    module = tokens_to_python_module(tokens)
    assert "FIGMA_COLORS" in module
    assert "FIGMA_TYPOGRAPHY" in module
    assert "FIGMA_RADII" in module
