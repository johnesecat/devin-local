"""Figma REST client + design-token extractor.

The Figma REST API is **read-only** — there is no public endpoint to create
or modify Figma files. This module therefore focuses on what the API
*does* support:

- :func:`get_file` — fetch a Figma file's full node tree by key.
- :func:`export_image` — render one or more nodes as PNG / SVG / JPG.
- :func:`extract_design_tokens` — walk a file's node tree and emit a
  flat dict of color / typography / spacing tokens for the app's theme.

We also document, prominently, where the token comes from (operator's
local config; never committed) and how to rotate it.
"""

from devin_local.figma.client import (
    FigmaClient,
    FigmaError,
    figma_token_from_disk,
    load_figma_settings,
    parse_file_key,
)
from devin_local.figma.tokens import (
    DesignTokens,
    extract_design_tokens,
    tokens_to_python_module,
)

__all__ = [
    "DesignTokens",
    "FigmaClient",
    "FigmaError",
    "extract_design_tokens",
    "figma_token_from_disk",
    "load_figma_settings",
    "parse_file_key",
    "tokens_to_python_module",
]
