"""Sandbox + plan pane for the right inspector.

The inspector is rebuilt around four collapsible sections:

1. **Plan** — live todo list emitted by the agent's planning loop.
2. **Workspace** — file tree + path + free-space hint.
3. **Sandbox** — terminal emulator state (cwd, last cmd), desktop emulator
   state (available?), enabled tool names, security flags (network on/off).
4. **Backend** — backend dropdown + model dropdown (model_selector widget).

This file owns sections 1 and 3; the workspace tree and backend/model
selectors are composed by ``app.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from devin_local.agent_planning import PlanStep


class PlanPane(QFrame):
    """A live-updating todo list showing the agent's plan for the current task."""

    plan_cleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PlanPane")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("Plan")
        title.setObjectName("H2")
        header.addWidget(title)
        header.addStretch(1)
        self._summary = QLabel("\u2014")
        self._summary.setObjectName("Muted")
        header.addWidget(self._summary)
        layout.addLayout(header)

        self._list = QListWidget()
        self._list.setObjectName("PlanList")
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setSelectionMode(self._list.SelectionMode.NoSelection)
        layout.addWidget(self._list, 1)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("Ghost")
        clear_btn.setFlat(True)
        clear_btn.clicked.connect(self._clear)
        layout.addWidget(clear_btn, 0, Qt.AlignmentFlag.AlignLeft)

    # ---------- public API ----------

    def set_plan(self, steps: Iterable[PlanStep]) -> None:
        steps = list(steps)
        self._list.clear()
        for step in steps:
            self._list.addItem(self._make_item(step))
        self._refresh_summary(steps)

    def _refresh_summary(self, steps: list[PlanStep]) -> None:
        if not steps:
            self._summary.setText("\u2014")
            return
        done = sum(1 for s in steps if s.status == "completed")
        self._summary.setText(f"{done}/{len(steps)}")

    def _make_item(self, step: PlanStep) -> QListWidgetItem:
        glyph = {
            "pending": "\u25cb",  # ○
            "in_progress": "\u25d0",  # ◐
            "completed": "\u2714",  # ✔
            "failed": "\u2716",  # ✖
        }.get(step.status, "\u25cb")
        text = f"  {glyph}  {step.text}"
        item = QListWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        if step.status == "completed":
            item.setForeground(Qt.GlobalColor.gray)
        elif step.status == "failed":
            item.setForeground(Qt.GlobalColor.red)
        return item

    def _clear(self) -> None:
        self._list.clear()
        self._summary.setText("\u2014")
        self.plan_cleared.emit()


class SandboxPanel(QFrame):
    """Live snapshot of the sandbox: workspace, terminal, desktop, tools."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SandboxPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        title = QLabel("Sandbox")
        title.setObjectName("H2")
        layout.addWidget(title)

        self._workspace_lbl = QLabel("workspace: \u2014")
        self._workspace_lbl.setWordWrap(True)
        layout.addWidget(self._workspace_lbl)

        self._term_lbl = QLabel("terminal: idle")
        self._term_lbl.setObjectName("Muted")
        self._term_lbl.setWordWrap(True)
        layout.addWidget(self._term_lbl)

        self._desktop_lbl = QLabel("desktop: \u2014")
        self._desktop_lbl.setObjectName("Muted")
        self._desktop_lbl.setWordWrap(True)
        layout.addWidget(self._desktop_lbl)

        self._tools_lbl = QLabel("tools: \u2014")
        self._tools_lbl.setObjectName("Muted")
        self._tools_lbl.setWordWrap(True)
        layout.addWidget(self._tools_lbl)

        self._flags_lbl = QLabel("flags: \u2014")
        self._flags_lbl.setObjectName("Muted")
        self._flags_lbl.setWordWrap(True)
        layout.addWidget(self._flags_lbl)

    # ---------- public API ----------

    def set_workspace(self, path: Path) -> None:
        self._workspace_lbl.setText(f"workspace: <code>{path}</code>")
        self._workspace_lbl.setTextFormat(Qt.TextFormat.RichText)

    def set_terminal_state(self, cwd: str, last_command: str | None) -> None:
        if last_command:
            preview = last_command if len(last_command) < 60 else last_command[:57] + "\u2026"
            self._term_lbl.setText(
                f"terminal: cwd=<code>{cwd}</code>  ·  last: <code>{preview}</code>"
            )
        else:
            self._term_lbl.setText(f"terminal: cwd=<code>{cwd}</code>  ·  idle")
        self._term_lbl.setTextFormat(Qt.TextFormat.RichText)

    def set_desktop_state(self, available: bool, last_action: str | None = None) -> None:
        if not available:
            self._desktop_lbl.setText("desktop: <i>disabled</i>")
        elif last_action:
            self._desktop_lbl.setText(f"desktop: ready  ·  last: <code>{last_action}</code>")
        else:
            self._desktop_lbl.setText("desktop: ready")
        self._desktop_lbl.setTextFormat(Qt.TextFormat.RichText)

    def set_tools(self, tool_names: list[str]) -> None:
        if not tool_names:
            self._tools_lbl.setText("tools: (none)")
            return
        # Group into a short preview.
        names = sorted(tool_names)
        preview = ", ".join(names[:6])
        if len(names) > 6:
            preview += f", \u2026 (+{len(names) - 6})"
        self._tools_lbl.setText(f"tools ({len(names)}): {preview}")

    def set_flags(self, *, network: bool, browser: bool, desktop: bool) -> None:
        bits = (("net", network), ("browser", browser), ("desktop", desktop))
        chunks = []
        for name, on in bits:
            color = "#9ece6a" if on else "#565f89"
            chunks.append(f"<span style='color: {color}'>{name}={'on' if on else 'off'}</span>")
        self._flags_lbl.setText("flags: " + "  ".join(chunks))
        self._flags_lbl.setTextFormat(Qt.TextFormat.RichText)
