"""The drag-and-drop UI designer dialog.

Layout::

    +----------+---------------------------+--------------------+
    | Palette  |         Canvas            |  Properties        |
    | -------- |                           |                    |
    |  Label   |  [drop area]              |  General           |
    |  Button  |                           |  Style             |
    |  Input   |                           |  Actions           |
    |  ...     |                           |                    |
    +----------+---------------------------+--------------------+
                  [Save]  [Open\u2026]  [Mount preview]

Drag a palette entry onto the canvas to drop an element. Click an element
to select it; drag to move it; drag the bottom-right corner to resize it.
The right panel reflects the selected element's properties and lets you
add / remove actions per event.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QMimeData, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QDrag, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from devin_local.ui_creator.document import (
    ELEMENT_TYPES,
    UiAction,
    UiDocument,
    UiElement,
)
from devin_local.ui_creator.runtime import UiRuntime, UiRuntimeContext

log = logging.getLogger(__name__)

UI_CREATOR_DIR_NAME = ".devin-local/ui-creator"


class _PaletteItem(QListWidgetItem):
    def __init__(self, type_id: str, spec: dict[str, Any]) -> None:
        super().__init__(f"  {spec['label']}")
        self.type_id = type_id
        self.setData(Qt.ItemDataRole.UserRole, type_id)


class _Palette(QListWidget):
    """Left-rail palette. Drag items onto the canvas to drop them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("UiCreatorPalette")
        self.setDragEnabled(True)
        self.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        for type_id, spec in ELEMENT_TYPES.items():
            self.addItem(_PaletteItem(type_id, spec))

    def startDrag(self, _supportedActions: Qt.DropAction) -> None:  # type: ignore[override]
        item = self.currentItem()
        if not isinstance(item, _PaletteItem):
            return
        mime = QMimeData()
        mime.setData("application/x-devinlocal-uicreator", item.type_id.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class _CanvasElement(QFrame):
    """A draggable / resizable proxy widget on the canvas.

    Note: this is the *designer-side* widget. It is not the same widget
    the runtime mounts when the operator clicks "Mount". The runtime walks
    the document directly.
    """

    GRIP = 12

    def __init__(self, element: UiElement, canvas: _Canvas) -> None:
        super().__init__(canvas)
        self._element = element
        self._canvas = canvas
        self.setObjectName("UiCanvasElement")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "#UiCanvasElement { background-color: rgba(255,255,255,12); "
            "border: 1px solid rgba(122,162,247,80); border-radius: 6px; } "
            "#UiCanvasElement[selected='true'] { border: 1.5px solid #7aa2f7; }"
        )
        self.setProperty("selected", False)
        self.setGeometry(element.x, element.y, element.width, element.height)
        self._refresh_label()
        self._drag_origin: QPoint | None = None
        self._resize_origin: QPoint | None = None
        self._start_rect: QRect | None = None

    def _refresh_label(self) -> None:
        spec = ELEMENT_TYPES.get(self._element.type, {})
        label = spec.get("label", self._element.type)
        text = self._element.name or self._element.props.get("text") or label
        if hasattr(self, "_label"):
            self._label.setText(f"{label}: {text}")
        else:
            self._label = QLabel(f"{label}: {text}", self)
            self._label.setStyleSheet("background: transparent; color: #c0caf5; padding: 4px;")
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self._label)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._canvas.select(self._element.id)
        pos = event.position().toPoint()
        if pos.x() >= self.width() - self.GRIP and pos.y() >= self.height() - self.GRIP:
            self._resize_origin = event.globalPosition().toPoint()
            self._start_rect = self.geometry()
        else:
            self._drag_origin = event.globalPosition().toPoint()
            self._start_rect = self.geometry()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_origin is not None and self._start_rect is not None:
            delta = event.globalPosition().toPoint() - self._drag_origin
            new_rect = self._start_rect.translated(delta)
            new_rect.moveLeft(max(0, new_rect.left()))
            new_rect.moveTop(max(0, new_rect.top()))
            self.setGeometry(new_rect)
            self._element.x = new_rect.x()
            self._element.y = new_rect.y()
        elif self._resize_origin is not None and self._start_rect is not None:
            delta = event.globalPosition().toPoint() - self._resize_origin
            new_w = max(40, self._start_rect.width() + delta.x())
            new_h = max(20, self._start_rect.height() + delta.y())
            self.resize(new_w, new_h)
            self._element.width = new_w
            self._element.height = new_h

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._drag_origin = None
        self._resize_origin = None
        self._start_rect = None
        self._canvas.notify_geometry_change(self._element.id)

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(Qt.GlobalColor.gray)
        pen.setWidth(1)
        painter.setPen(pen)
        # Resize grip in the bottom-right corner.
        x = self.width() - self.GRIP
        y = self.height() - self.GRIP
        painter.drawLine(x, self.height(), self.width(), y)
        painter.drawLine(x + 4, self.height(), self.width(), y + 4)


class _Canvas(QFrame):
    """The drop area. Accepts drags from the palette and hosts canvas elements."""

    element_added = Signal(UiElement)
    element_selected = Signal(str)
    element_changed = Signal(str)

    def __init__(self, document: UiDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("UiCanvas")
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"#UiCanvas {{ background-color: {document.background}; "
            "border: 1px dashed rgba(122,162,247,80); border-radius: 8px; }}"
        )
        self.setFixedSize(document.canvas_width, document.canvas_height)
        self._doc = document
        self._elements: dict[str, _CanvasElement] = {}
        self._selected_id: str | None = None
        for element in document.elements:
            self._mount(element)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasFormat("application/x-devinlocal-uicreator"):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):  # noqa: N802
        if event.mimeData().hasFormat("application/x-devinlocal-uicreator"):
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        if not event.mimeData().hasFormat("application/x-devinlocal-uicreator"):
            return
        type_id = bytes(event.mimeData().data("application/x-devinlocal-uicreator")).decode()
        pos = event.position().toPoint()
        element = UiElement.new(type_id, x=pos.x(), y=pos.y())
        self._doc.add(element)
        self._mount(element)
        self.element_added.emit(element)
        self.select(element.id)
        event.acceptProposedAction()

    def _mount(self, element: UiElement) -> None:
        widget = _CanvasElement(element, self)
        widget.show()
        self._elements[element.id] = widget

    def select(self, element_id: str) -> None:
        if self._selected_id == element_id:
            return
        if self._selected_id is not None:
            old = self._elements.get(self._selected_id)
            if old is not None:
                old.set_selected(False)
        self._selected_id = element_id
        new = self._elements.get(element_id)
        if new is not None:
            new.set_selected(True)
        self.element_selected.emit(element_id)

    def notify_geometry_change(self, element_id: str) -> None:
        self.element_changed.emit(element_id)

    def selected_id(self) -> str | None:
        return self._selected_id

    def remove(self, element_id: str) -> None:
        widget = self._elements.pop(element_id, None)
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        self._doc.remove(element_id)
        if self._selected_id == element_id:
            self._selected_id = None

    def refresh_label(self, element_id: str) -> None:
        widget = self._elements.get(element_id)
        if widget is not None:
            widget._refresh_label()  # noqa: SLF001


class _PropertiesPanel(QWidget):
    """Properties + Actions panel on the right side of the designer."""

    element_changed = Signal(str)

    def __init__(self, document: UiDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._doc = document
        self._element_id: str | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        self._tabs = QTabWidget()
        self._props_tab = QWidget()
        self._actions_tab = QWidget()
        self._tabs.addTab(self._props_tab, "Properties")
        self._tabs.addTab(self._actions_tab, "Actions")
        outer.addWidget(self._tabs, 1)
        self._stack_props = QStackedWidget()
        plays_props = QVBoxLayout(self._props_tab)
        plays_props.addWidget(self._stack_props, 1)
        self._stack_actions = QStackedWidget()
        plays_actions = QVBoxLayout(self._actions_tab)
        plays_actions.addWidget(self._stack_actions, 1)

        # Placeholder when nothing is selected.
        empty = QLabel("(no element selected)\nClick an element on the canvas.")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack_props.addWidget(empty)
        empty2 = QLabel("(no element selected)")
        empty2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack_actions.addWidget(empty2)

    def show_element(self, element_id: str | None) -> None:
        self._element_id = element_id
        if element_id is None:
            self._stack_props.setCurrentIndex(0)
            self._stack_actions.setCurrentIndex(0)
            return
        element = self._doc.get(element_id)
        if element is None:
            return
        # Rebuild properties tab.
        for stack in (self._stack_props, self._stack_actions):
            while stack.count() > 1:
                widget = stack.widget(1)
                stack.removeWidget(widget)
                widget.deleteLater()
        props_widget = self._build_props_form(element)
        actions_widget = self._build_actions_panel(element)
        self._stack_props.addWidget(props_widget)
        self._stack_props.setCurrentWidget(props_widget)
        self._stack_actions.addWidget(actions_widget)
        self._stack_actions.setCurrentWidget(actions_widget)

    def _build_props_form(self, element: UiElement) -> QWidget:
        wrapper = QWidget()
        form = QFormLayout(wrapper)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("Type", QLabel(element.type))
        form.addRow("ID", QLabel(element.id))

        name_edit = QLineEdit(element.name)
        name_edit.setPlaceholderText("(optional variable name — use {name} in actions)")

        def _on_name_changed(text: str) -> None:
            element.name = text.strip()
            self.element_changed.emit(element.id)

        name_edit.textChanged.connect(_on_name_changed)
        form.addRow("Name", name_edit)

        # Geometry
        x_spin = QSpinBox()
        x_spin.setRange(0, 4000)
        x_spin.setValue(element.x)
        y_spin = QSpinBox()
        y_spin.setRange(0, 4000)
        y_spin.setValue(element.y)
        w_spin = QSpinBox()
        w_spin.setRange(20, 4000)
        w_spin.setValue(element.width)
        h_spin = QSpinBox()
        h_spin.setRange(20, 4000)
        h_spin.setValue(element.height)

        def _apply_geom() -> None:
            element.x = x_spin.value()
            element.y = y_spin.value()
            element.width = w_spin.value()
            element.height = h_spin.value()
            self.element_changed.emit(element.id)

        for sp in (x_spin, y_spin, w_spin, h_spin):
            sp.valueChanged.connect(lambda _=0: _apply_geom())
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("x"))
        row1.addWidget(x_spin)
        row1.addWidget(QLabel("y"))
        row1.addWidget(y_spin)
        geom_wrap = QWidget()
        geom_wrap.setLayout(row1)
        form.addRow("Position", geom_wrap)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("w"))
        row2.addWidget(w_spin)
        row2.addWidget(QLabel("h"))
        row2.addWidget(h_spin)
        geom_wrap2 = QWidget()
        geom_wrap2.setLayout(row2)
        form.addRow("Size", geom_wrap2)

        # Per-type prop editors.
        for prop_name, prop_value in list(element.props.items()):
            editor = self._editor_for(element, prop_name, prop_value)
            form.addRow(prop_name, editor)

        delete_btn = QPushButton("Delete element")
        delete_btn.setObjectName("Danger")

        def _on_delete() -> None:
            self.element_changed.emit("__delete__:" + element.id)

        delete_btn.clicked.connect(_on_delete)
        form.addRow("", delete_btn)
        return wrapper

    def _editor_for(self, element: UiElement, key: str, value: Any) -> QWidget:
        if isinstance(value, bool):
            cb = QCheckBox()
            cb.setChecked(value)

            def _on_toggle(state: bool) -> None:
                element.props[key] = state
                self.element_changed.emit(element.id)

            cb.toggled.connect(_on_toggle)
            return cb
        if isinstance(value, (int, float)):
            spin = QSpinBox() if isinstance(value, int) else QDoubleSpinBox()
            spin.setRange(-100000, 100000)
            spin.setValue(value)

            def _on_change(v: Any) -> None:
                element.props[key] = v
                self.element_changed.emit(element.id)

            spin.valueChanged.connect(_on_change)
            return spin
        if isinstance(value, list):
            line = QLineEdit(", ".join(map(str, value)))

            def _on_change(text: str) -> None:
                element.props[key] = [s.strip() for s in text.split(",") if s.strip()]
                self.element_changed.emit(element.id)

            line.textChanged.connect(_on_change)
            return line
        if key in ("source", "default") and isinstance(value, str) and len(value) > 30:
            edit = QPlainTextEdit(str(value))

            def _on_change_pt() -> None:
                element.props[key] = edit.toPlainText()
                self.element_changed.emit(element.id)

            edit.textChanged.connect(_on_change_pt)
            return edit
        line = QLineEdit(str(value))

        def _on_change_text(text: str) -> None:
            element.props[key] = text
            self.element_changed.emit(element.id)

        line.textChanged.connect(_on_change_text)
        return line

    def _build_actions_panel(self, element: UiElement) -> QWidget:
        wrapper = QWidget()
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)

        # Pick the event we're editing. Buttons default to "clicked"; inputs
        # default to "changed". The operator can switch via the combo.
        event_combo = QComboBox()
        for event in ("clicked", "changed"):
            event_combo.addItem(event)
        outer.addWidget(event_combo)

        list_widget = QListWidget()
        outer.addWidget(list_widget, 1)

        def reload_list() -> None:
            list_widget.clear()
            current_event = event_combo.currentText()
            for idx, action in enumerate(element.on.get(current_event) or []):
                summary = self._summarize_action(action)
                list_widget.addItem(f"{idx + 1}. {action.kind}  \u2014  {summary}")

        reload_list()
        event_combo.currentTextChanged.connect(lambda _=0: reload_list())

        btns = QHBoxLayout()
        add = QPushButton("Add\u2026")
        edit = QPushButton("Edit\u2026")
        rm = QPushButton("Remove")
        btns.addWidget(add)
        btns.addWidget(edit)
        btns.addWidget(rm)
        outer.addLayout(btns)

        def add_action() -> None:
            current_event = event_combo.currentText()
            action = _prompt_action(self, None)
            if action is None:
                return
            element.on.setdefault(current_event, []).append(action)
            reload_list()
            self.element_changed.emit(element.id)

        def edit_action() -> None:
            current_event = event_combo.currentText()
            idx = list_widget.currentRow()
            actions = element.on.get(current_event) or []
            if idx < 0 or idx >= len(actions):
                return
            updated = _prompt_action(self, actions[idx])
            if updated is None:
                return
            actions[idx] = updated
            reload_list()
            self.element_changed.emit(element.id)

        def remove_action() -> None:
            current_event = event_combo.currentText()
            idx = list_widget.currentRow()
            actions = element.on.get(current_event) or []
            if idx < 0 or idx >= len(actions):
                return
            actions.pop(idx)
            reload_list()
            self.element_changed.emit(element.id)

        add.clicked.connect(add_action)
        edit.clicked.connect(edit_action)
        rm.clicked.connect(remove_action)
        return wrapper

    @staticmethod
    def _summarize_action(action: UiAction) -> str:
        if action.kind == "agent.send":
            return f"send: {action.template[:48]}"
        if action.kind == "agent.run_tool":
            return f"tool: {action.tool}({json.dumps(action.args)[:36]})"
        if action.kind == "shell.run":
            return f"shell: {action.command[:48]}"
        if action.kind == "set_var":
            return f"set {action.var} = {action.template[:36]}"
        if action.kind == "python":
            return f"python: {action.expr[:48]}"
        if action.kind == "open_file":
            return f"open: {action.path}"
        return action.kind


def _prompt_action(parent: QWidget, existing: UiAction | None) -> UiAction | None:
    """Modal sub-dialog to add or edit a :class:`UiAction`."""

    dialog = QDialog(parent)
    dialog.setWindowTitle("Action")
    dialog.setMinimumWidth(420)
    form = QFormLayout(dialog)
    kind = QComboBox()
    kind.addItems(
        [
            "agent.send",
            "agent.run_tool",
            "shell.run",
            "set_var",
            "python",
            "open_file",
        ]
    )
    if existing is not None:
        idx = kind.findText(existing.kind)
        if idx >= 0:
            kind.setCurrentIndex(idx)
    form.addRow("Kind", kind)

    template = QLineEdit(existing.template if existing else "{value}")
    template.setPlaceholderText("Use {var} placeholders; {value} = current element's value.")
    form.addRow("Template", template)

    tool = QLineEdit(existing.tool if existing else "")
    tool.setPlaceholderText("e.g. shell_exec, read_file, knowledge_search")
    form.addRow("Tool", tool)

    args = QPlainTextEdit(json.dumps(existing.args, indent=2) if existing else "{}")
    args.setPlaceholderText('{"command": "ls {workspace}"}')
    args.setMaximumHeight(120)
    form.addRow("Args (JSON)", args)

    command = QLineEdit(existing.command if existing else "")
    command.setPlaceholderText("e.g. python -m pytest -q")
    form.addRow("Command", command)

    target = QLineEdit(existing.target if existing else "")
    target.setPlaceholderText("Element id/name to write result into")
    form.addRow("Target", target)

    var = QLineEdit(existing.var if existing else "")
    form.addRow("Var name", var)

    expr = QLineEdit(existing.expr if existing else "")
    form.addRow("Python expr", expr)

    path = QLineEdit(existing.path if existing else "")
    form.addRow("Path", path)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None

    try:
        parsed_args = json.loads(args.toPlainText()) if args.toPlainText().strip() else {}
        if not isinstance(parsed_args, dict):
            parsed_args = {}
    except json.JSONDecodeError:
        QMessageBox.warning(parent, "Action", "Args must be valid JSON. Resetting to {}.")
        parsed_args = {}

    return UiAction(
        kind=kind.currentText(),
        template=template.text(),
        tool=tool.text(),
        args=parsed_args,
        command=command.text(),
        target=target.text(),
        var=var.text(),
        expr=expr.text(),
        path=path.text(),
    )


class UiCreatorDialog(QDialog):
    """The drag-and-drop UI designer."""

    def __init__(
        self,
        workspace: Path,
        agent: Any | None = None,
        request_submit: Signal | None = None,
        document: UiDocument | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("UI Creator")
        self.resize(1200, 800)
        self._workspace = workspace
        self._agent = agent
        self._request_submit = request_submit
        self._doc = document or UiDocument.new()
        self._save_dir = workspace / UI_CREATOR_DIR_NAME
        self._save_dir.mkdir(parents=True, exist_ok=True)

        outer = QVBoxLayout(self)

        header = QHBoxLayout()
        self._name_edit = QLineEdit(self._doc.name)
        self._name_edit.setPlaceholderText("Name this UI")
        header.addWidget(QLabel("Name:"))
        header.addWidget(self._name_edit, 1)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._palette = _Palette()
        self._palette.setFixedWidth(180)
        splitter.addWidget(self._palette)

        canvas_scroll = QScrollArea()
        canvas_scroll.setWidgetResizable(False)
        self._canvas = _Canvas(self._doc)
        canvas_scroll.setWidget(self._canvas)
        splitter.addWidget(canvas_scroll)

        self._props = _PropertiesPanel(self._doc)
        self._props.setMinimumWidth(320)
        splitter.addWidget(self._props)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)

        # Wire signals.
        self._canvas.element_added.connect(lambda e: self._props.show_element(e.id))
        self._canvas.element_selected.connect(self._props.show_element)
        self._canvas.element_changed.connect(self._on_canvas_element_changed)
        self._props.element_changed.connect(self._on_props_changed)

        # Bottom button row.
        btn_row = QHBoxLayout()
        self._open_btn = QPushButton("Open\u2026")
        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("Primary")
        self._mount_btn = QPushButton("Mount preview\u2026")
        self._open_btn.clicked.connect(self._on_open)
        self._save_btn.clicked.connect(self._on_save)
        self._mount_btn.clicked.connect(self._on_mount)
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._mount_btn)
        btn_row.addWidget(self._save_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    # ---------- handlers ----------

    def _on_canvas_element_changed(self, element_id: str) -> None:
        self._canvas.refresh_label(element_id)

    def _on_props_changed(self, marker: str) -> None:
        if marker.startswith("__delete__:"):
            target_id = marker.split(":", 1)[1]
            self._canvas.remove(target_id)
            self._props.show_element(None)
            return
        self._canvas.refresh_label(marker)

    def _on_save(self) -> None:
        self._doc.name = self._name_edit.text().strip() or self._doc.name
        path = self._doc.save(self._save_dir)
        QMessageBox.information(
            self,
            "Saved",
            f"UI saved to:\n\n{path}\n\nMount it later via 'Open\u2026' or load it programmatically.",
        )

    def _on_open(self) -> None:
        files = sorted(self._save_dir.glob("*.json"))
        if not files:
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Open UI", str(self._save_dir), "JSON (*.json)"
            )
            if not chosen:
                return
            path = Path(chosen)
        else:
            choice, ok = QInputDialog.getItem(
                self,
                "Open UI",
                "Pick a saved UI:",
                [p.name for p in files],
                0,
                False,
            )
            if not ok or not choice:
                return
            path = self._save_dir / choice
        try:
            self._doc = UiDocument.load(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Open UI", f"Could not load:\n\n{exc}")
            return
        self._name_edit.setText(self._doc.name)
        # Rebuild canvas + props from scratch.
        old = self._canvas
        new_canvas = _Canvas(self._doc)
        new_canvas.element_added.connect(lambda e: self._props.show_element(e.id))
        new_canvas.element_selected.connect(self._props.show_element)
        new_canvas.element_changed.connect(self._on_canvas_element_changed)
        parent_layout = old.parent()
        if isinstance(parent_layout, QScrollArea):
            parent_layout.setWidget(new_canvas)
        self._canvas = new_canvas
        self._props = _PropertiesPanel(self._doc)
        self._props.element_changed.connect(self._on_props_changed)
        QMessageBox.information(self, "Open UI", f"Loaded {path.name}.")

    def _on_mount(self) -> None:
        ctx = self._build_runtime_context()
        runtime = UiRuntime(self._doc, ctx)
        preview = QDialog(self)
        preview.setWindowTitle(f"Preview \u2014 {self._doc.name}")
        layout = QVBoxLayout(preview)
        layout.addWidget(runtime)
        close_btn = QPushButton("Close preview")
        close_btn.clicked.connect(preview.accept)
        layout.addWidget(close_btn)
        preview.exec()

    def _build_runtime_context(self) -> UiRuntimeContext:
        def send_to_agent(text: str) -> None:
            if self._request_submit is not None:
                self._request_submit.emit(text)
            else:
                QMessageBox.information(self, "Agent", f"Would send to agent:\n\n{text}")

        def dispatch_tool(name: str, args: dict[str, Any]) -> str:
            if self._agent is None:
                return "[no agent attached]"
            result = self._agent.registry.dispatch(name, args)
            return result.to_chat_payload()

        return UiRuntimeContext(
            workspace=self._workspace,
            send_to_agent=send_to_agent,
            dispatch_tool=dispatch_tool,
            open_file=None,
        )
