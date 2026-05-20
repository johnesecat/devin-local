"""UI Creator runtime: mounts a :class:`UiDocument` as a live Qt widget.

The runtime is what makes the designer *functional* instead of cosmetic.
When the operator clicks "Mount" (or when something programmatic asks for
a custom panel) we instantiate this widget; it walks the document, builds
the real Qt children, and connects each element's event signals to the
action dispatcher.

Element values can flow between actions: a text input named ``q`` becomes
``{q}`` in any action template; combo / checkbox / file-picker values are
exposed the same way. The dispatcher also supports a small set of
built-in variables: ``{workspace}``, ``{now}``, ``{user}``.
"""

from __future__ import annotations

import contextlib
import logging
import re
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QWidget,
)

from devin_local.ui_creator.document import UiAction, UiDocument, UiElement

log = logging.getLogger(__name__)

VAR_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass
class UiRuntimeContext:
    """External hooks the runtime can call.

    All callables are optional — when missing, the corresponding actions
    become no-ops with a logged warning. This keeps the runtime usable in
    unit tests that don't want to spin up a full agent.
    """

    workspace: Path = field(default_factory=lambda: Path.cwd())
    send_to_agent: Callable[[str], None] | None = None
    dispatch_tool: Callable[[str, dict[str, Any]], str] | None = None
    open_file: Callable[[Path], None] | None = None
    extra_vars: dict[str, Any] = field(default_factory=dict)


class UiRuntime(QWidget):
    """Live widget that renders a :class:`UiDocument` and runs its actions."""

    action_dispatched = Signal(str, dict)  # (kind, payload) — for tests/observers

    def __init__(
        self,
        document: UiDocument,
        ctx: UiRuntimeContext | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._doc = document
        self._ctx = ctx or UiRuntimeContext()
        self._widgets: dict[str, QWidget] = {}
        self._variables: dict[str, Any] = dict(self._ctx.extra_vars)
        self.setObjectName("UiRuntime")
        self.setFixedSize(document.canvas_width, document.canvas_height)
        self.setStyleSheet(f"#UiRuntime {{ background-color: {document.background}; }}")
        self._render()

    def _render(self) -> None:
        # Clear any previous children.
        for child in list(self.children()):
            if isinstance(child, QWidget):
                child.setParent(None)
                child.deleteLater()
        self._widgets.clear()

        for element in self._doc.elements:
            try:
                widget = self._build_widget(element)
            except Exception as exc:  # noqa: BLE001 - never blow up the host
                log.warning("UiRuntime: skipped element %s: %s", element.id, exc)
                continue
            widget.setParent(self)
            widget.setGeometry(element.x, element.y, element.width, element.height)
            widget.show()
            self._widgets[element.id] = widget

    # ---------- widget construction ----------

    def _build_widget(self, element: UiElement) -> QWidget:  # noqa: PLR0915
        props = element.props
        t = element.type
        if t == "label":
            w = QLabel(str(props.get("text", "")))
            w.setStyleSheet(self._label_style(props))
            return w
        if t == "heading":
            w = QLabel(str(props.get("text", "")))
            w.setStyleSheet(self._label_style(props))
            return w
        if t == "button":
            w = QPushButton(str(props.get("text", "Button")))
            if props.get("primary", True):
                w.setObjectName("Primary")
            w.clicked.connect(lambda _=False, eid=element.id: self._fire(eid, "clicked"))
            return w
        if t == "line_edit":
            w = QLineEdit()
            w.setPlaceholderText(str(props.get("placeholder", "")))
            w.setText(str(props.get("default", "")))
            w.textChanged.connect(lambda _, eid=element.id: self._fire(eid, "changed"))
            return w
        if t == "text_area":
            w = QPlainTextEdit()
            w.setPlaceholderText(str(props.get("placeholder", "")))
            w.setPlainText(str(props.get("default", "")))
            w.textChanged.connect(lambda eid=element.id: self._fire(eid, "changed"))
            return w
        if t == "combo":
            w = QComboBox()
            items = props.get("items") or []
            for item in items:
                w.addItem(str(item))
            default = str(props.get("default", ""))
            idx = w.findText(default)
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.currentTextChanged.connect(lambda _, eid=element.id: self._fire(eid, "changed"))
            return w
        if t == "checkbox":
            w = QCheckBox(str(props.get("text", "")))
            w.setChecked(bool(props.get("checked", False)))
            w.toggled.connect(lambda _, eid=element.id: self._fire(eid, "changed"))
            return w
        if t == "file_picker":
            return self._build_file_picker(element)
        if t == "image":
            w = QLabel("")
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            path = props.get("path") or ""
            if path:
                pm = QPixmap(str(path))
                if not pm.isNull():
                    fit = props.get("fit", "contain")
                    if fit == "contain":
                        pm = pm.scaled(
                            element.width,
                            element.height,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    w.setPixmap(pm)
                else:
                    w.setText(f"(image not found: {path})")
            else:
                w.setText("(no image)")
            return w
        if t == "markdown":
            w = QTextBrowser()
            md = str(props.get("source", "") or "")
            w.setMarkdown(md)
            w.setOpenExternalLinks(True)
            return w
        if t == "separator":
            w = QFrame()
            orient = props.get("orientation", "horizontal")
            w.setFrameShape(QFrame.Shape.HLine if orient == "horizontal" else QFrame.Shape.VLine)
            w.setFrameShadow(QFrame.Shadow.Sunken)
            return w
        if t == "chat_send":
            w = QPushButton(str(props.get("text", "Send to agent")))
            w.setObjectName("Primary")
            w.clicked.connect(lambda _=False, eid=element.id: self._fire(eid, "clicked"))
            return w
        return QLabel(f"(unsupported element type {t!r})")

    @staticmethod
    def _label_style(props: dict[str, Any]) -> str:
        bits = [f"font-size: {int(props.get('font_pt', 11))}pt"]
        if props.get("bold"):
            bits.append("font-weight: 600")
        if props.get("italic"):
            bits.append("font-style: italic")
        return "; ".join(bits)

    def _build_file_picker(self, element: UiElement) -> QWidget:
        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        line = QLineEdit()
        line.setReadOnly(True)
        line.setPlaceholderText(str(element.props.get("placeholder", "")))
        browse = QPushButton("Browse\u2026")

        mode = element.props.get("mode", "open")

        def _on_browse() -> None:
            if mode == "save":
                chosen, _ = QFileDialog.getSaveFileName(
                    wrapper, "Pick a file", str(self._ctx.workspace)
                )
            else:
                chosen, _ = QFileDialog.getOpenFileName(
                    wrapper, "Pick a file", str(self._ctx.workspace)
                )
            if chosen:
                line.setText(chosen)
                self._fire(element.id, "changed")

        browse.clicked.connect(_on_browse)
        row.addWidget(line, 1)
        row.addWidget(browse, 0)
        wrapper.setProperty("file_line", line)
        return wrapper

    # ---------- value extraction ----------

    def value_of(self, element_id: str) -> str:
        widget = self._widgets.get(element_id)
        if widget is None:
            return ""
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, QPlainTextEdit):
            return widget.toPlainText()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QCheckBox):
            return "true" if widget.isChecked() else "false"
        if isinstance(widget, QLabel):
            return widget.text()
        # file_picker wrapper carries the QLineEdit as a property.
        line = widget.property("file_line") if hasattr(widget, "property") else None
        if line is not None:
            with contextlib.suppress(Exception):
                return line.text()
        return ""

    def _all_values(self) -> dict[str, str]:
        """Map every named element to its current value (plus builtin vars)."""
        values: dict[str, str] = {}
        for element in self._doc.elements:
            if element.name:
                values[element.name] = self.value_of(element.id)
            values[element.id] = self.value_of(element.id)
        values.update({k: str(v) for k, v in self._variables.items()})
        values.setdefault("workspace", str(self._ctx.workspace))
        values.setdefault("now", time.strftime("%Y-%m-%d %H:%M:%S"))
        return values

    def _substitute(self, template: str, values: dict[str, str]) -> str:
        def repl(match: re.Match[str]) -> str:
            return values.get(match.group(1), match.group(0))

        return VAR_RE.sub(repl, template)

    # ---------- action dispatch ----------

    def _fire(self, element_id: str, event: str) -> None:
        element = self._doc.get(element_id)
        if element is None:
            return
        actions = element.on.get(event) or []
        if not actions:
            return
        for action in actions:
            # Recollect values between actions so set_var results are visible
            # to the next action in the chain.
            values = self._all_values()
            try:
                self._run_action(action, values, element)
            except Exception as exc:  # noqa: BLE001
                log.warning("UiRuntime action %s failed: %s", action.kind, exc)

    def _run_action(self, action: UiAction, values: dict[str, str], element: UiElement) -> None:
        kind = action.kind
        self.action_dispatched.emit(kind, {"element": element.id, "action": action.__dict__})
        if kind == "agent.send":
            text = self._substitute(
                action.template or "{value}", {**values, "value": self.value_of(element.id)}
            )
            if self._ctx.send_to_agent is None:
                log.warning("UiRuntime: agent.send action but no send_to_agent hook registered")
                return
            self._ctx.send_to_agent(text)
            return
        if kind == "agent.run_tool":
            if self._ctx.dispatch_tool is None:
                log.warning("UiRuntime: agent.run_tool but no dispatch_tool hook")
                return
            substituted_args = {
                k: self._substitute(str(v), values) for k, v in (action.args or {}).items()
            }
            output = self._ctx.dispatch_tool(action.tool, substituted_args)
            self._write_target(action.target, output)
            return
        if kind == "shell.run":
            cmd = self._substitute(action.command, values).strip()
            if not cmd:
                return
            try:
                completed = subprocess.run(  # noqa: S603 - operator-authored command
                    shlex.split(cmd),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(self._ctx.workspace),
                )
                merged = completed.stdout + (
                    f"\n[stderr]\n{completed.stderr}" if completed.stderr else ""
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
                merged = f"[error: {exc}]"
            self._write_target(action.target, merged.rstrip())
            return
        if kind == "set_var":
            if not action.var:
                return
            self._variables[action.var] = self._substitute(
                action.template or "{value}", {**values, "value": self.value_of(element.id)}
            )
            return
        if kind == "python":
            expr = self._substitute(action.expr, values)
            try:
                # Limited eval — only a small whitelist of builtins.
                safe_globals: dict[str, Any] = {"__builtins__": {}}
                safe_locals: dict[str, Any] = {
                    "values": values,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                }
                result = eval(expr, safe_globals, safe_locals)  # noqa: S307 - sandboxed
            except Exception as exc:  # noqa: BLE001
                result = f"[python error: {exc}]"
            self._write_target(action.target, str(result))
            return
        if kind == "open_file":
            path = self._substitute(action.path, values).strip()
            if not path:
                return
            target = (self._ctx.workspace / path).resolve()
            if self._ctx.open_file is not None:
                self._ctx.open_file(target)
            return
        log.warning("UiRuntime: unknown action kind %r", kind)

    def _write_target(self, target_id_or_name: str, value: str) -> None:
        if not target_id_or_name:
            return
        target = self._doc.get(target_id_or_name)
        if target is None:
            for e in self._doc.elements:
                if e.name == target_id_or_name:
                    target = e
                    break
        if target is None:
            return
        widget = self._widgets.get(target.id)
        if widget is None:
            return
        if isinstance(widget, QLineEdit):
            widget.setText(value)
        elif isinstance(widget, QPlainTextEdit):
            widget.setPlainText(value)
        elif isinstance(widget, QLabel):
            widget.setText(value)
        elif isinstance(widget, QTextBrowser):
            widget.setMarkdown(value)

    def document(self) -> UiDocument:
        return self._doc
