"""Reusable widgets for the devin-local desktop GUI.

The chat pane composes:

- :class:`MessageBubble` — chat bubble that splits its text into prose
  (markdown) and fenced code blocks. Streaming deltas accumulate into the
  raw text; the visual children are rebuilt only at safe boundaries (end of
  delta when the buffer's outstanding fences are balanced) so the user sees
  live tokens without the page jumping.
- :class:`ToolCard` — collapsible card showing one tool invocation. Has a
  status pill (running / ok / error), pretty-printed JSON args (collapsible),
  result panel with file-preview shortcuts for write_file/edit_file and
  stdout/stderr blocks for shell_exec, and per-call timing in ms.
- :class:`ChatPane` — scrollable column of bubbles + tool cards.
- :class:`Composer` — multi-line input + send button (Ctrl+Enter).

Kept dependency-light: PySide6 + Pygments (via syntax.py). Markdown rendering
uses Qt's built-in QTextDocument.setMarkdown.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from devin_local.gui.syntax import (
    CodeBlock,
    CodeBlockWidget,
    parse_markdown_blocks,
)
from devin_local.tools.base import ToolResult

_FILE_PATH_KEYS = ("path", "file_path", "filename", "file")


class _ProseView(QTextEdit):
    """A read-only QTextEdit tuned for markdown bubble prose."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.SansSerif)
        font.setPointSize(11)
        self.setFont(font)

    def set_markdown(self, text: str) -> None:
        # Qt 6's setMarkdown handles CommonMark + tables + lists nicely; for
        # streaming partial markdown we still need to fall back to plain text
        # when the document looks unbalanced (e.g. an open list at end of
        # stream). The split-by-fence step has already removed code blocks.
        try:
            self.document().setMarkdown(text)
        except Exception:  # noqa: BLE001 - fall back if Qt rejects partial md
            self.setPlainText(text)
        self._adjust_height()

    def set_plain(self, text: str) -> None:
        self.setPlainText(text)
        self._adjust_height()

    def _adjust_height(self) -> None:
        doc = self.document()
        doc.setTextWidth(self.viewport().width())
        h = int(doc.size().height()) + 14
        self.setMinimumHeight(max(28, h))


class MessageBubble(QFrame):
    """A single chat bubble (user or assistant) with mixed prose / code blocks."""

    def __init__(self, role: str, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.role = role
        self.setObjectName("MessageBubbleUser" if role == "user" else "MessageBubbleAssistant")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self._avatar = QLabel("You" if role == "user" else "devin-local")
        self._avatar.setObjectName("BubbleAvatar")
        header.addWidget(self._avatar)
        header.addStretch(1)
        outer.addLayout(header)

        # Container that holds the dynamic prose/code block children. Rebuilt
        # on each set_text/finalize. During streaming we keep a single
        # _ProseView as the tail child so tokens visibly accrue.
        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        outer.addLayout(self._content_layout)

        self._raw_text = ""
        self._is_streaming = False
        self._stream_view: _ProseView | None = None
        if text:
            self.set_text(text)

    # ---------- public API ----------

    def text(self) -> str:
        return self._raw_text

    def append_delta(self, delta: str) -> None:
        """Append streaming token text. Avoids rebuilding the widget tree on
        every chunk by maintaining a single tail _ProseView until the bubble
        is finalized.
        """
        if not delta:
            return
        self._raw_text += delta
        if not self._is_streaming:
            self._clear_content()
            self._stream_view = _ProseView()
            self._content_layout.addWidget(self._stream_view)
            self._is_streaming = True
        assert self._stream_view is not None
        # During streaming we render as plain text (markdown parsing on every
        # token would re-flow the document constantly). The finalize step
        # below switches to a fully parsed mixed-content view.
        self._stream_view.set_plain(self._raw_text)

    def set_text(self, text: str) -> None:
        """Replace the bubble content with `text`, parsing markdown + code
        fences. Ends streaming mode.
        """
        self._raw_text = text
        self._is_streaming = False
        self._stream_view = None
        self._clear_content()
        blocks = parse_markdown_blocks(text) if text else []
        for block in blocks:
            if isinstance(block, CodeBlock):
                self._content_layout.addWidget(CodeBlockWidget(block.lang, block.code))
            else:
                view = _ProseView()
                view.set_markdown(block.text)
                self._content_layout.addWidget(view)
        if not blocks:
            view = _ProseView()
            view.set_plain("")
            self._content_layout.addWidget(view)

    # ---------- internals ----------

    def _clear_content(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


class _Pill(QLabel):
    """A small status pill (running / ok / error / warn)."""

    def __init__(self, text: str = "", kind: str = "running") -> None:
        super().__init__(text)
        self.set_kind(kind)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setContentsMargins(0, 0, 0, 0)

    def set_kind(self, kind: str) -> None:
        # Object name drives the QSS styling.
        mapping = {
            "running": "PillRunning",
            "ok": "PillOk",
            "error": "PillErr",
            "warn": "PillWarn",
        }
        self.setObjectName(mapping.get(kind, "PillRunning"))
        # Re-apply stylesheet for object-name selectors.
        self.style().unpolish(self)
        self.style().polish(self)


class _Collapser(QWidget):
    """A click-to-expand header + a single body widget."""

    def __init__(self, title: str, body: QWidget, expanded: bool = False) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._toggle = QPushButton(("\u25bc " if expanded else "\u25b6 ") + title)
        self._toggle.setObjectName("Ghost")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setFlat(True)
        self._toggle.setStyleSheet("text-align: left; padding: 2px 0;")
        self._toggle.clicked.connect(self._on_toggled)
        layout.addWidget(self._toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self._body = body
        self._body.setVisible(expanded)
        self._expanded = expanded
        self._title = title
        layout.addWidget(self._body)

    def _on_toggled(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._toggle.setText(("\u25bc " if self._expanded else "\u25b6 ") + self._title)


class ToolCard(QFrame):
    """A collapsible card showing one tool invocation + its result."""

    SHELL_TOOLS = {"shell_exec", "shell_session", "python"}
    WRITE_TOOLS = {"write_file", "edit_file"}

    def __init__(
        self,
        name: str,
        arguments: dict[str, Any],
        workspace: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ToolCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._name = name
        self._arguments = arguments or {}
        self._workspace = workspace
        self._started_at = time.monotonic()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        # Header: icon + name + spacer + timing + status pill
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self._icon = QLabel(_tool_icon(name))
        self._icon.setObjectName("ToolIcon")
        header.addWidget(self._icon)
        self._name_label = QLabel(name)
        self._name_label.setObjectName("ToolName")
        header.addWidget(self._name_label)
        self._summary = QLabel(_summarize_args(name, self._arguments))
        self._summary.setObjectName("Muted")
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self._summary, 1)
        self._timing = QLabel("…")
        self._timing.setObjectName("Muted")
        header.addWidget(self._timing)
        self._status = _Pill("running", kind="running")
        header.addWidget(self._status)
        outer.addLayout(header)

        # Args (collapsed by default).
        args_view = QPlainTextEdit()
        args_view.setReadOnly(True)
        args_view.setPlainText(_pretty_json(self._arguments))
        args_view.setFrameShape(QFrame.Shape.NoFrame)
        args_view.setObjectName("CodeBlockBody")
        mono = QFont("JetBrains Mono", 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        args_view.setFont(mono)
        args_view.setMinimumHeight(60)
        args_view.setMaximumHeight(220)
        outer.addWidget(_Collapser(f"arguments ({len(self._arguments)} field(s))", args_view))

        # Result panel — created on finish().
        self._result_container = QWidget()
        rc = QVBoxLayout(self._result_container)
        rc.setContentsMargins(0, 0, 0, 0)
        rc.setSpacing(8)
        outer.addWidget(self._result_container)

    # ---------- public API ----------

    def finish(self, result: ToolResult) -> None:
        elapsed = time.monotonic() - self._started_at
        self._timing.setText(_format_elapsed(elapsed))
        ok = bool(result.ok)
        self._status.setText("ok" if ok else "error")
        self._status.set_kind("ok" if ok else "error")

        rc_layout = self._result_container.layout()
        if self._name in self.WRITE_TOOLS:
            self._add_write_result(rc_layout, result)
        elif self._name in self.SHELL_TOOLS:
            self._add_shell_result(rc_layout, result)
        else:
            self._add_generic_result(rc_layout, result)

    # ---------- result renderers ----------

    def _add_write_result(self, layout: QVBoxLayout, result: ToolResult) -> None:
        path_str = _first_path_arg(self._arguments)
        full_path = self._resolve(path_str) if path_str else None
        meta_line = self._make_file_meta(full_path, path_str)
        layout.addWidget(meta_line)

        body_widget: QWidget
        if full_path is not None and full_path.exists() and full_path.is_file():
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                content = f"<could not read file: {exc}>"
            lang = _lang_for_path(full_path)
            body_widget = CodeBlockWidget(lang, content)
        else:
            preview = result.to_chat_payload()
            body_widget = _result_text_widget(preview)
        layout.addWidget(_Collapser("preview", body_widget, expanded=False))

    def _add_shell_result(self, layout: QVBoxLayout, result: ToolResult) -> None:
        # ShellExec ToolResult.output is the merged transcript; we still show
        # it as a single code block (bash) but with a meta line.
        cmd = self._arguments.get("command") or self._arguments.get("cmd") or ""
        if cmd:
            meta = QLabel(f"$ <code>{_escape_html(str(cmd))}</code>")
            meta.setObjectName("Muted")
            meta.setTextFormat(Qt.TextFormat.RichText)
            meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            meta.setWordWrap(True)
            layout.addWidget(meta)
        text = result.output or result.to_chat_payload()
        layout.addWidget(_Collapser("output", CodeBlockWidget("bash", text), expanded=True))

    def _add_generic_result(self, layout: QVBoxLayout, result: ToolResult) -> None:
        payload = result.to_chat_payload()
        layout.addWidget(_Collapser("result", _result_text_widget(payload), expanded=False))

    # ---------- helpers ----------

    def _resolve(self, rel: str) -> Path | None:
        if not rel:
            return None
        p = Path(rel)
        if p.is_absolute():
            return p
        if self._workspace is None:
            return None
        return (self._workspace / rel).resolve()

    def _make_file_meta(self, full_path: Path | None, raw: str) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        path_label = QLabel(f"\U0001f4c4  <code>{_escape_html(raw or '?')}</code>")
        path_label.setTextFormat(Qt.TextFormat.RichText)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        h.addWidget(path_label)
        if full_path is not None and full_path.exists():
            try:
                size = full_path.stat().st_size
                lines = sum(1 for _ in full_path.open("rb"))
                meta = QLabel(f"{_human_bytes(size)}  ·  {lines} line(s)")
            except OSError:
                meta = QLabel("(could not stat)")
            meta.setObjectName("Muted")
            h.addWidget(meta)
        h.addStretch(1)
        return row


class ChatPane(QScrollArea):
    """Scrollable column of MessageBubble + ToolCard widgets."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(20, 20, 20, 20)
        self._layout.setSpacing(12)
        self._layout.addStretch(1)
        self.setWidget(self._container)
        self._current_assistant: MessageBubble | None = None
        self._workspace: Path | None = None

    def set_workspace(self, workspace: Path) -> None:
        self._workspace = workspace

    def add_user(self, text: str) -> MessageBubble:
        bubble = MessageBubble("user", text)
        self._insert(bubble)
        self._current_assistant = None
        return bubble

    def start_assistant(self) -> MessageBubble:
        bubble = MessageBubble("assistant", "")
        self._insert(bubble)
        self._current_assistant = bubble
        return bubble

    def append_assistant_delta(self, delta: str) -> None:
        if self._current_assistant is None:
            self.start_assistant()
        assert self._current_assistant is not None
        self._current_assistant.append_delta(delta)
        self._scroll_to_bottom()

    def finish_assistant(self, final_text: str | None = None) -> None:
        if self._current_assistant is None:
            return
        if final_text is not None:
            self._current_assistant.set_text(final_text)
        else:
            # Convert the streaming plain-text view into a proper
            # markdown + code-block layout.
            self._current_assistant.set_text(self._current_assistant.text())
        self._current_assistant = None
        self._scroll_to_bottom()

    def add_tool(self, name: str, arguments: dict[str, Any]) -> ToolCard:
        card = ToolCard(name, arguments, workspace=self._workspace)
        self._insert(card)
        return card

    def add_system_notice(self, text: str) -> None:
        label = QLabel(text)
        label.setObjectName("Muted")
        label.setWordWrap(True)
        self._insert(label)

    def clear(self) -> None:
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._current_assistant = None

    def _insert(self, widget: QWidget) -> None:
        self._layout.insertWidget(self._layout.count() - 1, widget)

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())


class Composer(QWidget):
    """Multi-line input box + send button."""

    submitted = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Composer")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)
        self._input = QPlainTextEdit()
        self._input.setPlaceholderText("Ask devin-local anything\u2026  (Ctrl+Enter to send)")
        self._input.setMinimumHeight(64)
        self._input.setMaximumHeight(180)
        self._send = QPushButton("Send")
        self._send.setObjectName("Primary")
        self._send.setMinimumWidth(96)
        self._send.clicked.connect(self._emit)
        layout.addWidget(self._input, 1)
        layout.addWidget(self._send, 0, Qt.AlignmentFlag.AlignBottom)

    def keyPressEvent(self, event) -> None:  # noqa: D401, N802
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._emit()
            return
        super().keyPressEvent(event)

    def set_busy(self, busy: bool) -> None:
        self._send.setEnabled(not busy)
        self._send.setText("\u2026" if busy else "Send")

    def _emit(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self.submitted.emit(text)


# ---------- module helpers ----------


def _pretty_json(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _summarize_args(name: str, args: dict[str, Any]) -> str:
    """One-line summary shown on the card header. Picks the most identifying arg."""
    if not args:
        return ""
    if name in ToolCard.WRITE_TOOLS or name in ("read_file", "list_dir"):
        for key in _FILE_PATH_KEYS:
            if key in args:
                return f"  {args[key]}"
    if name in ToolCard.SHELL_TOOLS:
        cmd = args.get("command") or args.get("cmd")
        if cmd:
            short = str(cmd).splitlines()[0]
            if len(short) > 80:
                short = short[:77] + "\u2026"
            return f"  {short}"
    # Fallback: first scalar value, truncated.
    for k, v in args.items():
        if isinstance(v, (str, int, float, bool)):
            preview = str(v)
            if len(preview) > 60:
                preview = preview[:57] + "\u2026"
            return f"  {k}={preview}"
    return ""


def _tool_icon(name: str) -> str:
    """Unicode icon for a tool, chosen for clarity in a small label."""
    if name in {"write_file", "edit_file"}:
        return "\u270e"  # pencil
    if name == "read_file":
        return "\U0001f4d6"  # book
    if name == "list_dir":
        return "\U0001f4c1"  # folder
    if name in {"find_files", "grep"}:
        return "\U0001f50d"  # magnifier
    if name in {"shell_exec", "shell_session"}:
        return "\u25b6"  # play
    if name == "python":
        return "\U0001f40d"  # python
    if name in {"web_fetch", "web_search"}:
        return "\U0001f310"  # globe
    if name.startswith("desktop_"):
        return "\U0001f5a5"  # desktop
    return "\u2699"  # gear


def _first_path_arg(args: dict[str, Any]) -> str:
    for key in _FILE_PATH_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def _lang_for_path(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mapping = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "tsx": "tsx",
        "jsx": "jsx",
        "rs": "rust",
        "go": "go",
        "java": "java",
        "kt": "kotlin",
        "c": "c",
        "h": "c",
        "cpp": "cpp",
        "cc": "cpp",
        "hpp": "cpp",
        "cs": "csharp",
        "rb": "ruby",
        "sh": "bash",
        "bash": "bash",
        "zsh": "bash",
        "ps1": "powershell",
        "json": "json",
        "yaml": "yaml",
        "yml": "yaml",
        "toml": "toml",
        "md": "markdown",
        "html": "html",
        "css": "css",
        "sql": "sql",
        "xml": "xml",
        "dockerfile": "dockerfile",
    }
    return mapping.get(suffix, "")


def _human_bytes(n: int) -> str:
    if n <= 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    val = float(n)
    for unit in units:
        if val < 1024 or unit == units[-1]:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


def _format_elapsed(seconds: float) -> str:
    if seconds < 1:
        return f"{int(seconds * 1000)} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _result_text_widget(text: str) -> QWidget:
    """Long text result, capped + monospace."""
    body = QPlainTextEdit()
    body.setReadOnly(True)
    body.setPlainText(text if len(text) <= 4000 else text[:4000] + f"\n… ({len(text)} chars total)")
    body.setFrameShape(QFrame.Shape.NoFrame)
    body.setObjectName("CodeBlockBody")
    mono = QFont("JetBrains Mono", 10)
    mono.setStyleHint(QFont.StyleHint.Monospace)
    body.setFont(mono)
    body.setMinimumHeight(80)
    body.setMaximumHeight(360)
    return body
