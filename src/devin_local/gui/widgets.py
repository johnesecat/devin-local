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
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
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
        # Time-stamped at construction so the header chip is meaningful even
        # when the bubble streams over many seconds.
        self._created_at = time.localtime()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        avatar_text = "You" if role == "user" else "devin-local"
        self._avatar = QLabel(avatar_text)
        self._avatar.setObjectName("BubbleAvatarUser" if role == "user" else "BubbleAvatar")
        header.addWidget(self._avatar)
        self._timestamp = QLabel(time.strftime("%H:%M", self._created_at))
        self._timestamp.setObjectName("BubbleTimestamp")
        header.addWidget(self._timestamp)
        header.addStretch(1)
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setObjectName("BubbleCopy")
        self._copy_btn.setFlat(True)
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.clicked.connect(self._copy_to_clipboard)
        header.addWidget(self._copy_btn)
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

    def _copy_to_clipboard(self) -> None:
        cb = QGuiApplication.clipboard()
        if cb is not None:
            cb.setText(self._raw_text)
            self._copy_btn.setText("Copied")


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

    SHELL_TOOLS = {"shell_exec", "shell_session", "python", "python_exec"}
    WRITE_TOOLS = {"write_file", "edit_file"}
    DIR_TOOLS = {"list_dir", "find_files"}
    READ_TOOLS = {"read_file"}

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
        elif self._name in self.READ_TOOLS:
            self._add_read_result(rc_layout, result)
        elif self._name in self.DIR_TOOLS:
            self._add_directory_result(rc_layout, result)
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

    def _add_read_result(self, layout: QVBoxLayout, result: ToolResult) -> None:
        """read_file: header with path, optional syntax-highlighted preview."""
        path_str = _first_path_arg(self._arguments)
        full_path = self._resolve(path_str) if path_str else None
        meta_line = self._make_file_meta(full_path, path_str)
        layout.addWidget(meta_line)
        text = result.output or result.to_chat_payload()
        lang = _lang_for_path(full_path) if full_path else ""
        body_widget = CodeBlockWidget(lang, text)
        layout.addWidget(_Collapser("contents", body_widget, expanded=False))

    def _add_directory_result(self, layout: QVBoxLayout, result: ToolResult) -> None:
        """list_dir / find_files: render a folder tree card."""
        path_str = self._arguments.get("path") or self._arguments.get("root") or "."
        pattern = self._arguments.get("pattern")
        raw = result.output or result.to_chat_payload()
        entries = _parse_directory_listing(raw, self._name)

        meta = QWidget()
        h = QHBoxLayout(meta)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        icon_char = "\U0001f4c1"
        label_text = (
            f"{icon_char}  <code>{_escape_html(str(path_str))}</code>"
            if not pattern
            else f"{icon_char}  <code>{_escape_html(str(path_str))}</code>  \u00b7  pattern: <code>{_escape_html(str(pattern))}</code>"
        )
        path_label = QLabel(label_text)
        path_label.setTextFormat(Qt.TextFormat.RichText)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        h.addWidget(path_label)
        count_label = QLabel(f"{len(entries)} entry(ies)")
        count_label.setObjectName("Muted")
        h.addWidget(count_label)
        h.addStretch(1)
        layout.addWidget(meta)

        if not entries:
            empty = QLabel("(empty)")
            empty.setObjectName("Muted")
            layout.addWidget(empty)
            return

        tree = _FolderTreeWidget(entries)
        layout.addWidget(_Collapser("entries", tree, expanded=True))

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
        self.setObjectName("ChatPane")
        self.setWidgetResizable(True)
        # Match the rest of the app's dark theme. Without these, the
        # QScrollArea viewport defaults to a near-white system color and
        # produces the "beige chat area" effect under the dark sidebar.
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        viewport = self.viewport()
        if viewport is not None:
            viewport.setObjectName("ChatPaneViewport")
        self._container = QWidget()
        self._container.setObjectName("ChatPaneContainer")
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


# ---------- folder tree rendering ----------


def parse_list_dir_output(text: str) -> list[tuple[str, str, int]]:
    """Parse the output of ``list_dir`` into ``(kind, name, size)`` tuples.

    ``list_dir`` returns lines like ``"dir         0  src"`` or
    ``"file      1234  README.md"``. Returns an empty list for ``"(empty)"``.
    """
    entries: list[tuple[str, str, int]] = []
    if not text or text.strip() == "(empty)":
        return entries
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        kind, size_str, name = parts[0], parts[1], parts[2]
        if kind not in {"dir", "file"}:
            continue
        try:
            size = int(size_str)
        except ValueError:
            size = 0
        entries.append((kind, name, size))
    return entries


def parse_find_files_output(text: str) -> list[tuple[str, str, int]]:
    """Parse the output of ``find_files`` into ``(kind, path, size)`` tuples.

    ``find_files`` returns one workspace-relative file path per line.
    """
    entries: list[tuple[str, str, int]] = []
    if not text or text.strip() == "(no matches)":
        return entries
    for line in text.splitlines():
        path = line.strip()
        if not path:
            continue
        entries.append(("file", path, 0))
    return entries


def _parse_directory_listing(text: str, tool_name: str) -> list[tuple[str, str, int]]:
    if tool_name == "list_dir":
        return parse_list_dir_output(text)
    if tool_name == "find_files":
        return parse_find_files_output(text)
    return []


class _FolderTreeWidget(QTreeWidget):
    """A small tree view showing folder/file entries with icons + sizes.

    For ``list_dir`` results we show a single flat level (the tool returns one
    level). For ``find_files`` results we group by directory so the chat shows
    a real visual tree of matched paths.
    """

    def __init__(self, entries: list[tuple[str, str, int]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FolderTree")
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setIndentation(14)
        self.setUniformRowHeights(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        mono = QFont("JetBrains Mono", 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(mono)
        self._populate(entries)
        # Height: roughly one row per visible item, capped at 12 rows.
        row_h = max(18, self.fontMetrics().lineSpacing() + 4)
        rows = min(12, max(1, self._visible_row_estimate(entries)))
        self.setMinimumHeight(row_h * rows + 6)
        self.setMaximumHeight(row_h * 14 + 6)

    def _visible_row_estimate(self, entries: list[tuple[str, str, int]]) -> int:
        # When the entries already contain "/" we'll group; count both dirs and
        # files we'll show.
        if any("/" in name or "\\" in name for _, name, _ in entries):
            dirs: set[str] = set()
            for _kind, name, _size in entries:
                norm = name.replace("\\", "/")
                head = norm.rsplit("/", 1)[0] if "/" in norm else ""
                if head:
                    dirs.add(head)
            return len(entries) + len(dirs)
        return len(entries)

    def _populate(self, entries: list[tuple[str, str, int]]) -> None:
        # Decide flat vs grouped based on whether any entries contain a path
        # separator.
        if any("/" in name or "\\" in name for _, name, _ in entries):
            self._populate_grouped(entries)
        else:
            self._populate_flat(entries)
        self.expandAll()

    def _populate_flat(self, entries: list[tuple[str, str, int]]) -> None:
        for kind, name, size in entries:
            item = QTreeWidgetItem(self)
            icon = "\U0001f4c1" if kind == "dir" else _file_glyph_for_name(name)
            label = f"{icon}  {name}"
            if kind == "file" and size > 0:
                label += f"   {_human_bytes(size)}"
            item.setText(0, label)

    def _populate_grouped(self, entries: list[tuple[str, str, int]]) -> None:
        # Group file paths by their parent directory; show parents as folder
        # items, children indented underneath. Order parents alphabetically.
        groups: dict[str, list[tuple[str, str, int]]] = {}
        for kind, name, size in entries:
            norm = name.replace("\\", "/")
            if "/" in norm:
                parent, leaf = norm.rsplit("/", 1)
            else:
                parent, leaf = "", norm
            groups.setdefault(parent, []).append((kind, leaf, size))
        for parent in sorted(groups.keys()):
            if parent:
                root = QTreeWidgetItem(self)
                root.setText(0, f"\U0001f4c1  {parent}/")
            else:
                root = None
            for kind, leaf, size in sorted(groups[parent], key=lambda x: x[1].lower()):
                glyph = "\U0001f4c1" if kind == "dir" else _file_glyph_for_name(leaf)
                label = f"{glyph}  {leaf}"
                if kind == "file" and size > 0:
                    label += f"   {_human_bytes(size)}"
                if root is None:
                    QTreeWidgetItem(self).setText(0, label)
                else:
                    child = QTreeWidgetItem(root)
                    child.setText(0, label)


def _file_glyph_for_name(name: str) -> str:
    """Return a single-char glyph that hints the file kind based on extension."""
    lower = name.lower()
    if lower.endswith((".py",)):
        return "\U0001f40d"  # python
    if lower.endswith((".md", ".markdown", ".rst", ".txt")):
        return "\U0001f4dd"  # memo
    if lower.endswith((".json", ".yaml", ".yml", ".toml", ".ini")):
        return "\u2699"  # gear (config)
    if lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp")):
        return "\U0001f5bc"  # framed picture
    if lower.endswith((".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs")):
        return "\U0001f7e8"  # yellow square (web)
    if lower.endswith((".rs",)):
        return "\U0001f980"  # crab
    if lower.endswith((".go",)):
        return "\U0001f439"  # mouse (gopher-ish)
    if lower.endswith((".sh", ".bash", ".zsh", ".ps1")):
        return "\u25b6"  # play
    return "\U0001f4c4"  # page
