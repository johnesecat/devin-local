"""Tests for the GUI syntax module: markdown parsing + code highlighting."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from devin_local.gui.syntax import (  # noqa: E402 - import after importorskip
    CodeBlock,
    TextBlock,  # noqa: F401 - re-exported for downstream tests
    highlight_code_html,
    parse_markdown_blocks,
)


def test_parse_returns_text_only_when_no_fence() -> None:
    blocks = parse_markdown_blocks("Just prose here.")
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == "Just prose here."


def test_parse_extracts_single_code_block() -> None:
    text = "Here's the code:\n```python\nprint('hi')\n```\nDone."
    blocks = parse_markdown_blocks(text)
    assert len(blocks) == 3
    assert isinstance(blocks[0], TextBlock)
    assert isinstance(blocks[1], CodeBlock)
    assert blocks[1].lang == "python"
    assert blocks[1].code == "print('hi')"
    assert isinstance(blocks[2], TextBlock)


def test_parse_handles_no_language() -> None:
    text = "```\nplain code\n```"
    blocks = parse_markdown_blocks(text)
    assert len(blocks) == 1
    assert isinstance(blocks[0], CodeBlock)
    assert blocks[0].lang == ""
    assert blocks[0].code == "plain code"


def test_parse_extracts_multiple_blocks() -> None:
    text = "```py\na=1\n```\n\nmiddle\n\n```js\nlet b=2;\n```"
    blocks = parse_markdown_blocks(text)
    langs = [b.lang for b in blocks if isinstance(b, CodeBlock)]
    assert langs == ["py", "js"]


def test_highlight_python_returns_html_with_class() -> None:
    html = highlight_code_html("print('hi')", "python")
    assert html is not None
    # Should contain a Pygments class span and the literal token.
    assert 'class="highlight"' in html
    assert "print" in html


def test_highlight_unknown_language_returns_none() -> None:
    html = highlight_code_html("blah", "completely-made-up-language-12345")
    assert html is None
