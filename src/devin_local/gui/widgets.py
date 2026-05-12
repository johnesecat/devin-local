"""Reusable widgets for the devin-local desktop GUI.

Kept small and dependency-free (PySide6 only). The main window in `app.py`
composes these into the final layout.
"""

from __future__ import annotations

import json
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

from devin_local.tools.base import ToolResult


class MessageBubble(QFrame):
    """A single chat bubble (user or assistant)."""

    def __init__(self, role: str, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.role = role
        self.setObjectName("MessageBubbleUser" if role == "user" else "MessageBubbleAssistant")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        self._header = QLabel("You" if role == "user" else "Assistant")
        self._header.setObjectName("H2")
        layout.addWidget(self._header)
        self._body = QTextEdit()
        self._body.setReadOnly(True)
        self._body.setFrameShape(QFrame.Shape.NoFrame)
        self._body.setStyleSheet("background: transparent; border: none;")
        self._body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.SansSerif)
        font.setPointSize(11)
        self._body.setFont(font)
        self._body.setPlainText(text)
        layout.addWidget(self._body)
        self._raw_text = text
        self._adjust_height()

    def append_delta(self, delta: str) -> None:
        self._raw_text += delta
        self._body.setPlainText(self._raw_text)
        self._adjust_height()

    def set_text(self, text: str) -> None:
        self._raw_text = text
        self._body.setPlainText(text)
        self._adjust_height()

    def text(self) -> str:
        return self._raw_text

    def _adjust_height(self) -> None:
        doc = self._body.document()
        doc.setTextWidth(self._body.viewport().width())
        h = int(doc.size().height()) + 14
        self._body.setMinimumHeight(max(28, h))


class ToolCard(QFrame):
    """A collapsible card showing a single tool invocation + result."""

    def __init__(
        self,
        name: str,
        arguments: dict[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ToolCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(4)
        header = QHBoxLayout()
        self._name_label = QLabel(f"<b>{name}</b>")
        self._status_label = QLabel("running\u2026")
        self._status_label.setObjectName("Muted")
        header.addWidget(self._name_label)
        header.addStretch(1)
        header.addWidget(self._status_label)
        outer.addLayout(header)
        self._args_label = QLabel(_pretty_json(arguments))
        self._args_label.setObjectName("Muted")
        self._args_label.setWordWrap(True)
        self._args_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self._args_label)
        self._result_label: QLabel | None = None

    def finish(self, result: ToolResult) -> None:
        ok = bool(result.ok)
        self._status_label.setText("ok" if ok else "error")
        self._status_label.setObjectName("BadgeOk" if ok else "BadgeErr")
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)
        payload = result.to_chat_payload()
        # Trim very long payloads but keep them inspectable.
        if len(payload) > 600:
            payload = payload[:600] + f"\u2026  ({len(payload)} chars total)"
        self._result_label = QLabel(payload)
        self._result_label.setWordWrap(True)
        self._result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.layout().addWidget(self._result_label)


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
            # Replace streamed text with the canonical post-processed text
            # (with tool-call markup stripped) so the user sees clean output.
            self._current_assistant.set_text(final_text)
        self._current_assistant = None
        self._scroll_to_bottom()

    def add_tool(self, name: str, arguments: dict[str, Any]) -> ToolCard:
        card = ToolCard(name, arguments)
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
        # Insert before the trailing stretch so new items appear at the bottom.
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


def _pretty_json(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)
