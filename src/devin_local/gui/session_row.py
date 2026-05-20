"""Sidebar session row widget.

Each row shows the session name, a small timestamp, and a gear button on
the right. Click the body to select the session; click the gear to open
:class:`PerSessionSettingsDialog` for *that* session.

This is intentionally Qt-only and observer-free — the main window connects
to the row's ``selected`` / ``settings_clicked`` / ``delete_clicked``
signals to react.
"""

from __future__ import annotations

import time
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from devin_local.gui.icons import icon
from devin_local.sessions import SessionInfo


class SessionRow(QWidget):
    """Single row in the sidebar's session list."""

    selected = Signal(str)  # session id
    settings_clicked = Signal(str)
    delete_clicked = Signal(str)

    def __init__(
        self, info: SessionInfo, active: bool = False, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._info = info
        self._active = active

        self.setObjectName("SessionRow")
        self.setProperty("active", active)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 8, 8)
        row.setSpacing(6)

        body = QWidget(self)
        body.setObjectName("SessionRowBody")
        body.setCursor(Qt.CursorShape.PointingHandCursor)
        bcol = QVBoxLayout(body)
        bcol.setContentsMargins(0, 0, 0, 0)
        bcol.setSpacing(2)
        self._name_label = QLabel(info.name or "(untitled)")
        self._name_label.setObjectName("SessionRowName")
        bcol.addWidget(self._name_label)
        self._meta_label = QLabel(self._format_meta(info))
        self._meta_label.setObjectName("SessionRowMeta")
        bcol.addWidget(self._meta_label)
        body.mousePressEvent = self._on_body_clicked  # type: ignore[assignment]
        row.addWidget(body, 1)

        gear_btn = QPushButton(self)
        gear_btn.setObjectName("SessionRowGear")
        gear_btn.setFlat(True)
        gear_btn.setFixedSize(28, 28)
        gear_btn.setToolTip("Configure this session (system prompt, knowledge, model…)")
        gear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        gear_icon = icon("settings")
        if gear_icon is not None:
            gear_btn.setIcon(gear_icon)
        else:
            gear_btn.setText("\u2699")  # gear glyph fallback
        gear_btn.clicked.connect(lambda: self.settings_clicked.emit(self._info.id))
        row.addWidget(gear_btn, 0)

        delete_btn = QPushButton(self)
        delete_btn.setObjectName("SessionRowDelete")
        delete_btn.setFlat(True)
        delete_btn.setFixedSize(28, 28)
        delete_btn.setToolTip("Delete this session")
        delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        del_icon = icon("trash")
        if del_icon is not None:
            delete_btn.setIcon(del_icon)
        else:
            delete_btn.setText("\u00d7")
        delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self._info.id))
        row.addWidget(delete_btn, 0)

    @staticmethod
    def _format_meta(info: SessionInfo) -> str:
        ts = info.last_used_ts or info.created_ts or time.time()
        when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        bits: list[str] = [when]
        if info.model:
            bits.append(info.model)
        if info.system_prompt_override:
            bits.append("custom prompt")
        if info.knowledge_dir:
            bits.append("kb")
        return " · ".join(bits)

    def _on_body_clicked(self, _event: object) -> None:
        self.selected.emit(self._info.id)

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        self.setProperty("active", active)
        # Re-polish so the stylesheet picks up the property change.
        self.style().unpolish(self)
        self.style().polish(self)

    def update_info(self, info: SessionInfo) -> None:
        self._info = info
        self._name_label.setText(info.name or "(untitled)")
        self._meta_label.setText(self._format_meta(info))

    def info(self) -> SessionInfo:
        return self._info
