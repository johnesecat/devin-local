"""Model selector widget + library browser dialog.

The right-inspector exposes a combobox that lists locally installed Ollama
models (pulled live from ``GET /api/tags``). A "Browse library…" entry opens
a dialog with a curated list of popular models; selecting one streams the
``POST /api/pull`` events into a live progress bar.

The pull runs on a Qt worker thread so the UI never blocks. Cancel is wired
through the streaming generator's ``close()`` call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from devin_local.gui.ollama_model_service import (
        InstalledModel,
        LibraryModel,
        OllamaModelService,
    )

log = logging.getLogger(__name__)


class ModelSelector(QWidget):
    """Compact combobox + refresh + browse buttons for picking an Ollama model."""

    model_changed = Signal(str)
    browse_requested = Signal()
    refresh_requested = Signal()

    def __init__(self, current: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current = current
        self._installed: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._combo = QComboBox()
        self._combo.setEditable(False)
        self._combo.addItem(current, current)
        self._combo.currentIndexChanged.connect(self._on_index_changed)
        layout.addWidget(self._combo)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._refresh = QPushButton("\u21bb Refresh")
        self._refresh.setObjectName("Ghost")
        self._refresh.setFlat(True)
        self._refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh.clicked.connect(self.refresh_requested.emit)
        row.addWidget(self._refresh)
        self._browse = QPushButton("Browse library\u2026")
        self._browse.setObjectName("Ghost")
        self._browse.setFlat(True)
        self._browse.setCursor(Qt.CursorShape.PointingHandCursor)
        self._browse.clicked.connect(self.browse_requested.emit)
        row.addWidget(self._browse)
        row.addStretch(1)
        layout.addLayout(row)

        self._status = QLabel("")
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    # ---------- public API ----------

    def set_installed(self, names: list[str]) -> None:
        """Repopulate the combobox with `names` (locally installed models).

        The current selection is preserved if still present; otherwise the
        first item is selected.
        """
        self._installed = list(names)
        previous = self._current
        self._combo.blockSignals(True)
        self._combo.clear()
        for name in names:
            self._combo.addItem(name, name)
        if previous and previous not in names:
            # Keep the unavailable previous selection visible (greyed out)
            # so the user knows what's missing.
            self._combo.addItem(f"{previous}  (not installed)", previous)
        idx = self._combo.findData(previous)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)
        if not names:
            self._status.setText("No models installed — click \u2018Browse library\u2026\u2019")
        else:
            self._status.setText(f"{len(names)} installed model(s)")

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_current(self, name: str) -> None:
        self._current = name
        idx = self._combo.findData(name)
        if idx < 0:
            self._combo.addItem(name, name)
            idx = self._combo.count() - 1
        self._combo.setCurrentIndex(idx)

    def current(self) -> str:
        data = self._combo.currentData()
        return str(data) if data is not None else self._current

    # ---------- handlers ----------

    def _on_index_changed(self, _idx: int) -> None:
        name = self.current()
        if name and name != self._current:
            self._current = name
            self.model_changed.emit(name)


# ---------------------------------------------------------------------------
# Library browser dialog with streaming pull progress
# ---------------------------------------------------------------------------


class _PullWorker(QObject):
    """Streams `OllamaModelService.pull` events; emits Qt signals for each."""

    progress = Signal(str, int, int)  # status, completed, total
    finished = Signal(bool, str)  # ok, message

    def __init__(self, service: OllamaModelService, name: str) -> None:
        super().__init__()
        self._service = service
        self._name = name
        self._stream = None  # type: ignore[assignment]
        self._cancelled = False

    @Slot()
    def cancel(self) -> None:
        self._cancelled = True
        if self._stream is not None:
            self._stream.close()

    @Slot()
    def run(self) -> None:
        try:
            with self._service.pull(self._name) as stream:
                self._stream = stream
                for event in stream:
                    if self._cancelled:
                        self.finished.emit(False, "cancelled")
                        return
                    if event.error:
                        self.finished.emit(False, event.error)
                        return
                    self.progress.emit(event.status, event.completed, event.total)
                    if event.done:
                        break
            self.finished.emit(True, "success")
        except Exception as exc:  # noqa: BLE001
            log.exception("pull worker failed")
            self.finished.emit(False, f"{type(exc).__name__}: {exc}")


class LibraryBrowserDialog(QDialog):
    """Dialog showing curated + installed models, with streaming pull."""

    pulled = Signal(str)  # emitted with model name after a successful pull

    def __init__(
        self,
        service: OllamaModelService,
        installed: list[InstalledModel],
        library: list[LibraryModel],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Browse Ollama models")
        self.resize(640, 520)
        self._service = service
        self._installed = {m.name for m in installed}
        self._worker: _PullWorker | None = None
        self._thread: QThread | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        intro = QLabel(
            "Pick a model to pull from the Ollama library. Pull progress streams "
            "live. You can also enter any tag from <a href='https://ollama.com/library'>"
            "ollama.com/library</a>."
        )
        intro.setOpenExternalLinks(True)
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(False)
        for lib in library:
            item = QListWidgetItem(self._format_library_row(lib))
            item.setData(Qt.ItemDataRole.UserRole, lib.name)
            tip = lib.description + (
                "\n\n(Tool-calling capable)"
                if lib.tool_calling
                else "\n\n(Text-only — limited tool calling)"
            )
            item.setToolTip(tip)
            self._list.addItem(item)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list, 1)

        # Custom-pull row
        custom_row = QHBoxLayout()
        custom_row.setContentsMargins(0, 0, 0, 0)
        custom_row.setSpacing(6)
        custom_row.addWidget(QLabel("Custom tag:"))
        self._custom_input = QLineEdit()
        self._custom_input.setPlaceholderText("e.g. qwen2.5-coder:14b")
        custom_row.addWidget(self._custom_input, 1)
        self._pull_btn = QPushButton("Pull")
        self._pull_btn.setObjectName("Primary")
        self._pull_btn.clicked.connect(self._on_pull_clicked)
        custom_row.addWidget(self._pull_btn)
        layout.addLayout(custom_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # Standard buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._cancel_btn = QPushButton("Cancel pull")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_active_pull)
        buttons.addButton(self._cancel_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    # ---------- helpers ----------

    def _format_library_row(self, lib: LibraryModel) -> str:
        marker = "\u2713" if lib.name in self._installed else " "
        tag = "\U0001f527" if lib.tool_calling else "\U0001f4ac"
        return f"  [{marker}]  {tag}  {lib.name:24s}  {lib.size:10s}  {lib.description}"

    def _selected_name(self) -> str:
        custom = self._custom_input.text().strip()
        if custom:
            return custom
        item = self._list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    # ---------- pull lifecycle ----------

    def _on_double_click(self, _item: QListWidgetItem) -> None:
        self._on_pull_clicked()

    def _on_pull_clicked(self) -> None:
        name = self._selected_name()
        if not name:
            self._status.setText("Pick a model from the list or enter a custom tag.")
            return
        if self._thread is not None:
            self._status.setText("Another pull is already running.")
            return
        self._status.setText(f"Pulling {name}\u2026")
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._pull_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)

        worker = _PullWorker(self._service, name)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(lambda ok, msg, n=name: self._on_finished(ok, msg, n))
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        self._worker = worker
        self._thread = thread

    def _cancel_active_pull(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("Cancelling\u2026")

    def _on_progress(self, status: str, completed: int, total: int) -> None:
        if total > 0:
            pct = int((completed / total) * 100)
            self._progress.setMaximum(100)
            self._progress.setValue(pct)
            self._status.setText(f"{status}  ({_human(completed)} / {_human(total)})")
        else:
            # Indeterminate phase (manifest, verify, etc.)
            self._progress.setMaximum(0)
            self._status.setText(status)

    def _on_finished(self, ok: bool, msg: str, name: str) -> None:
        self._progress.setMaximum(100)
        self._progress.setValue(100 if ok else 0)
        self._pull_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._worker = None
        self._thread = None
        if ok:
            self._status.setText(f"Installed {name}.")
            self._installed.add(name)
            # Refresh the visible markers.
            for i in range(self._list.count()):
                item = self._list.item(i)
                lib_name = str(item.data(Qt.ItemDataRole.UserRole) or "")
                marker = "\u2713" if lib_name in self._installed else " "
                text = item.text()
                # Replace the first occurrence of "[X]" with the new marker.
                if text[:5] in ("  [✓]", "  [ ]"):
                    item.setText(f"  [{marker}]" + text[5:])
            self.pulled.emit(name)
        else:
            self._status.setText(f"Pull failed: {msg}")

    # ---------- cleanup ----------

    def reject(self) -> None:  # noqa: D401 - Qt override
        self._cancel_active_pull()
        super().reject()


def _human(n: int) -> str:
    if n <= 0:
        return "?"
    units = ("B", "KB", "MB", "GB", "TB")
    val = float(n)
    for unit in units:
        if val < 1024 or unit == units[-1]:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"
