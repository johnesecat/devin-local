"""Syntax highlighting + markdown-aware rendering for chat bubbles.

The chat pane composes a `MessageBubble` from a sequence of mixed blocks:

- ``TextBlock``: plain markdown text rendered with ``QTextEdit.setMarkdown``
- ``CodeBlock``: fenced ``` ``` ``` code block rendered with Pygments and
  shown in a dedicated panel with a copy button + language label.

Splitting the bubble into discrete child widgets (rather than dumping
everything into one QTextEdit) gives us:

- Per-language syntax colors that match the dark theme
- A copy button per code block
- Reliable height measurement on streaming updates

If Pygments is not installed (the ``[gui]`` extra always installs it, but
unit tests run with a slim env) the code block falls back to a monospace
QPlainTextEdit with no colors. The widget tree is otherwise identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

try:  # Pygments is in the [gui] extra but keep the dep optional.
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.util import ClassNotFound

    _HAS_PYGMENTS = True
except Exception:  # noqa: BLE001
    _HAS_PYGMENTS = False


_FENCE_RE = re.compile(r"(?ms)^[ \t]*```([\w+\-.]*)\s*\n(.*?)\n?[ \t]*```[ \t]*$")


@dataclass
class TextBlock:
    """A run of prose (possibly markdown) between code fences."""

    text: str


@dataclass
class CodeBlock:
    """A fenced code block with optional language tag."""

    lang: str
    code: str


def parse_markdown_blocks(text: str) -> list[TextBlock | CodeBlock]:
    """Split text into a sequence of TextBlock / CodeBlock entries.

    Triple-backtick fences with an optional language tag are extracted into
    CodeBlock; everything else becomes TextBlock(s). Empty text segments are
    dropped.
    """
    blocks: list[TextBlock | CodeBlock] = []
    pos = 0
    for match in _FENCE_RE.finditer(text):
        if match.start() > pos:
            chunk = text[pos : match.start()]
            if chunk.strip():
                blocks.append(TextBlock(chunk))
        lang = (match.group(1) or "").strip()
        code = match.group(2) or ""
        blocks.append(CodeBlock(lang=lang, code=code))
        pos = match.end()
    if pos < len(text):
        chunk = text[pos:]
        if chunk.strip():
            blocks.append(TextBlock(chunk))
    if not blocks and text:
        blocks.append(TextBlock(text))
    return blocks


# Pygments style colors tuned to match the GUI dark theme (tokyo-night-ish).
# Each entry is (token-class, hex-color).
_PYGMENTS_CSS = """
.highlight pre { background: #1a1b26; color: #c0caf5; }
.highlight .hll { background-color: #2f3549; }
.highlight .c, .highlight .ch, .highlight .cm, .highlight .cp, .highlight .cpf,
.highlight .c1, .highlight .cs { color: #565f89; font-style: italic; }
.highlight .err { color: #f7768e; }
.highlight .k, .highlight .kc, .highlight .kd, .highlight .kn, .highlight .kp,
.highlight .kr, .highlight .kt { color: #bb9af7; }
.highlight .l, .highlight .ld, .highlight .m, .highlight .mb, .highlight .mf,
.highlight .mh, .highlight .mi, .highlight .il, .highlight .mo { color: #ff9e64; }
.highlight .s, .highlight .sa, .highlight .sb, .highlight .sc, .highlight .dl,
.highlight .sd, .highlight .s2, .highlight .se, .highlight .sh, .highlight .si,
.highlight .sx, .highlight .sr, .highlight .s1, .highlight .ss { color: #9ece6a; }
.highlight .n, .highlight .nb, .highlight .bp, .highlight .nc, .highlight .no,
.highlight .nd, .highlight .ni, .highlight .ne, .highlight .nf, .highlight .fm,
.highlight .py, .highlight .nl, .highlight .nn, .highlight .nx, .highlight .nt,
.highlight .nv, .highlight .vc, .highlight .vg, .highlight .vi, .highlight .vm,
.highlight .w { color: #c0caf5; }
.highlight .o, .highlight .ow { color: #89ddff; }
.highlight .p { color: #c0caf5; }
.highlight .ge { font-style: italic; }
.highlight .gs { font-weight: bold; }
.highlight .gh, .highlight .gu { color: #7aa2f7; font-weight: bold; }
.highlight .nt { color: #7aa2f7; }
"""


def highlight_code_html(code: str, lang: str) -> str | None:
    """Return an HTML fragment of `code` highlighted with Pygments, or None
    if Pygments is unavailable or the language is unknown.
    """
    if not _HAS_PYGMENTS:
        return None
    try:
        if lang:
            lexer = get_lexer_by_name(lang, stripall=False)
        else:
            try:
                lexer = guess_lexer(code)
            except ClassNotFound:
                return None
    except ClassNotFound:
        return None
    formatter = HtmlFormatter(nowrap=False, noclasses=False, cssclass="highlight")
    body = highlight(code, lexer, formatter)
    return f"<style>{_PYGMENTS_CSS}</style>{body}"


class CodeBlockWidget(QFrame):
    """A self-contained code block: lang badge, copy button, syntax-highlighted body."""

    def __init__(self, lang: str, code: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CodeBlock")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.lang = lang
        self.code = code

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QWidget()
        header.setObjectName("CodeBlockHeader")
        h = QHBoxLayout(header)
        h.setContentsMargins(10, 4, 6, 4)
        h.setSpacing(6)
        self._lang_label = QLabel(lang or "text")
        self._lang_label.setObjectName("Muted")
        h.addWidget(self._lang_label)
        h.addStretch(1)
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setObjectName("Ghost")
        self._copy_btn.setFlat(True)
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.clicked.connect(self._copy_to_clipboard)
        h.addWidget(self._copy_btn)
        outer.addWidget(header)

        body_html = highlight_code_html(code, lang)
        if body_html is not None:
            from PySide6.QtWidgets import QTextBrowser  # local import; QTextBrowser is heavy

            self._body: QPlainTextEdit | QTextBrowser = QTextBrowser()
            self._body.setOpenExternalLinks(False)
            self._body.setHtml(body_html)
        else:
            self._body = QPlainTextEdit()
            self._body.setPlainText(code)
            self._body.setReadOnly(True)
            mono = QFont("JetBrains Mono", 10)
            mono.setStyleHint(QFont.StyleHint.Monospace)
            self._body.setFont(mono)
        self._body.setFrameShape(QFrame.Shape.NoFrame)
        self._body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._body.setObjectName("CodeBlockBody")
        outer.addWidget(self._body)

        self._adjust_height()

    def _copy_to_clipboard(self) -> None:
        cb = QGuiApplication.clipboard()
        if cb is not None:
            cb.setText(self.code)
            self._copy_btn.setText("Copied")

    def _adjust_height(self) -> None:
        # Lines * line height + padding. Cap at 24 lines; if longer, the user
        # can scroll inside the block.
        line_count = self.code.count("\n") + 1
        line_h = self.fontMetrics().lineSpacing()
        height = min(24, line_count) * line_h + 24
        self._body.setMinimumHeight(max(48, height))
