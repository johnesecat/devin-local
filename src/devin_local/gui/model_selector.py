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
from typing import TYPE_CHECKING, Any

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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Runtime imports: needed for isinstance / dataclass construction in the
# HF browse tab. Kept here (not lazily) so the tab module is self-contained
# and can be unit-tested without standing up the rest of the dialog.
from devin_local.gui.ollama_model_service import (
    HuggingFaceGGUFEntry,
    HuggingFaceGGUFFile,
)

if TYPE_CHECKING:
    from devin_local.gui.ollama_model_service import (
        HuggingFaceCatalog,
        InstalledModel,
        LibraryModel,
        OllamaModelService,
    )

log = logging.getLogger(__name__)

_BROWSE_SENTINEL = "__devin_local__browse_library__"


class ModelComboBox(QComboBox):
    """Compact, reusable model dropdown.

    Drop-in replacement for the free-text ``QLineEdit`` model fields in the
    global Settings dialog and the per-session Behavior tab. Populated from
    a list of installed Ollama model names; optionally appends a sentinel
    "Browse library\u2026" entry at the bottom.

    Emits :attr:`browse_requested` when the user picks the sentinel — the
    parent dialog wires this to :class:`LibraryBrowserDialog`.
    """

    browse_requested = Signal()

    def __init__(
        self,
        installed: list[str] | None = None,
        current: str = "",
        *,
        allow_browse: bool = True,
        allow_custom_text: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._allow_browse = allow_browse
        self._installed: list[str] = []
        self._current = current
        self.setEditable(allow_custom_text)
        if allow_custom_text:
            self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.populate(installed or [], current)
        self.currentIndexChanged.connect(self._on_index_changed)

    def populate(self, installed: list[str], current: str | None = None) -> None:
        """Repopulate with the given installed model names.

        If ``current`` is not in ``installed``, it's added as a "(not
        installed)" entry so the user can still see what's selected.
        """
        if current is None:
            current = self._current
        self._installed = list(installed)
        self.blockSignals(True)
        self.clear()
        if not installed:
            self.addItem("(no models installed)", "")
        for name in installed:
            self.addItem(name, name)
        if current and current not in installed and current.strip():
            self.addItem(f"{current}  (not installed)", current)
        if self._allow_browse:
            self.insertSeparator(self.count())
            self.addItem("\U0001f4da  Browse library\u2026", _BROWSE_SENTINEL)
        idx = self.findData(current)
        if idx >= 0:
            self.setCurrentIndex(idx)
        elif installed:
            self.setCurrentIndex(0)
        self.blockSignals(False)
        self._current = self.current_model()

    def current_model(self) -> str:
        data = self.currentData()
        if data == _BROWSE_SENTINEL:
            return self._current
        if isinstance(data, str) and data:
            return data
        text = self.currentText().strip()
        return text if text and not text.startswith("(") else self._current

    def set_current_model(self, name: str) -> None:
        idx = self.findData(name)
        if idx < 0:
            # Insert above the separator/browse item.
            insert_at = self.count()
            if self._allow_browse and insert_at >= 2:
                insert_at = max(0, insert_at - 2)
            self.insertItem(insert_at, f"{name}  (not installed)", name)
            idx = insert_at
        self.setCurrentIndex(idx)
        self._current = name

    def _on_index_changed(self, _idx: int) -> None:
        data = self.currentData()
        if data == _BROWSE_SENTINEL:
            # Don't actually select the sentinel; bounce back to previous.
            prev_idx = self.findData(self._current)
            if prev_idx >= 0:
                self.blockSignals(True)
                self.setCurrentIndex(prev_idx)
                self.blockSignals(False)
            self.browse_requested.emit()
            return
        self._current = self.current_model()


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
    """Tabbed model browser: Ollama curated library + Hugging Face GGUF.

    Both tabs end up registering the chosen model with the local Ollama
    daemon, so the rest of the app only needs to know about Ollama tags.
    """

    pulled = Signal(str)  # emitted with model name after a successful install

    def __init__(
        self,
        service: OllamaModelService,
        installed: list[InstalledModel],
        library: list[LibraryModel],
        parent: QWidget | None = None,
        hf_catalog: HuggingFaceCatalog | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Browse model library")
        self.resize(820, 620)
        self._service = service
        self._installed = {m.name for m in installed}
        self._worker: _PullWorker | None = None
        self._thread: QThread | None = None
        # Lazy: only construct an HF client when the HF tab is touched.
        self._hf_catalog = hf_catalog

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_ollama_tab(library), "Ollama library")
        self._hf_tab = _HFBrowseTab(self._get_hf_catalog, parent=self)
        self._hf_tab.installed.connect(self._on_hf_installed)
        self._tabs.addTab(self._hf_tab, "Hugging Face GGUF")
        layout.addWidget(self._tabs, 1)

        # Standard buttons (shared across tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._cancel_btn = QPushButton("Cancel pull")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_active_pull)
        buttons.addButton(self._cancel_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    # ---------- Ollama tab ----------

    def _build_ollama_tab(self, library: list[LibraryModel]) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(12, 12, 12, 12)
        page_layout.setSpacing(10)

        intro = QLabel(
            "Pick a model to pull from the Ollama library. Pull progress streams "
            "live. You can also enter any tag from <a href='https://ollama.com/library'>"
            "ollama.com/library</a>."
        )
        intro.setOpenExternalLinks(True)
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        page_layout.addWidget(intro)

        # Search + tool-capable filter row
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(8)
        filter_row.addWidget(QLabel("Search:"))
        self._ollama_search = QLineEdit()
        self._ollama_search.setPlaceholderText("filter by name, e.g. qwen")
        self._ollama_search.textChanged.connect(self._refilter_ollama_list)
        filter_row.addWidget(self._ollama_search, 1)
        self._ollama_filter = QComboBox()
        self._ollama_filter.addItems(["All", "Tool-capable only", "Text-only"])
        self._ollama_filter.currentIndexChanged.connect(self._refilter_ollama_list)
        filter_row.addWidget(QLabel("Capability:"))
        filter_row.addWidget(self._ollama_filter)
        page_layout.addLayout(filter_row)

        self._library_models = list(library)
        self._list = QListWidget()
        self._list.setAlternatingRowColors(False)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._populate_ollama_list()
        page_layout.addWidget(self._list, 1)

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
        page_layout.addLayout(custom_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        page_layout.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        page_layout.addWidget(self._status)
        return page

    def _populate_ollama_list(self) -> None:
        self._list.clear()
        for lib in self._library_models:
            item = QListWidgetItem(self._format_library_row(lib))
            item.setData(Qt.ItemDataRole.UserRole, lib.name)
            tip = lib.description + (
                "\n\n(Tool-calling capable)"
                if lib.tool_calling
                else "\n\n(Text-only — limited tool calling)"
            )
            item.setToolTip(tip)
            self._list.addItem(item)

    def _refilter_ollama_list(self, *_: Any) -> None:
        query = self._ollama_search.text().strip().lower()
        cap_mode = self._ollama_filter.currentText()
        for i in range(self._list.count()):
            item = self._list.item(i)
            name = str(item.data(Qt.ItemDataRole.UserRole) or "")
            lib = next((m for m in self._library_models if m.name == name), None)
            if lib is None:
                item.setHidden(True)
                continue
            text_ok = query in name.lower() or query in lib.description.lower()
            cap_ok = (
                cap_mode == "All"
                or (cap_mode == "Tool-capable only" and lib.tool_calling)
                or (cap_mode == "Text-only" and not lib.tool_calling)
            )
            item.setHidden(not (text_ok and cap_ok))

    # ---------- HF tab plumbing ----------

    def _get_hf_catalog(self) -> HuggingFaceCatalog:
        if self._hf_catalog is None:
            from devin_local.gui.ollama_model_service import HuggingFaceCatalog

            self._hf_catalog = HuggingFaceCatalog()
        return self._hf_catalog

    def _on_hf_installed(self, name: str) -> None:
        self._installed.add(name)
        self.pulled.emit(name)

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
        if hasattr(self, "_hf_tab"):
            self._hf_tab.cancel_active_install()

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


# ---------------------------------------------------------------------------
# Hugging Face GGUF browse tab
# ---------------------------------------------------------------------------


class _HFSearchWorker(QObject):
    """Background `HuggingFaceCatalog.search` call."""

    results = Signal(list)  # list[HuggingFaceGGUFEntry]
    failed = Signal(str)

    def __init__(self, catalog_factory, query: str, limit: int = 60) -> None:
        super().__init__()
        self._catalog_factory = catalog_factory
        self._query = query
        self._limit = limit

    @Slot()
    def run(self) -> None:
        try:
            catalog = self._catalog_factory()
            entries = catalog.search(query=self._query, limit=self._limit)
            self.results.emit(entries)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _HFFilesWorker(QObject):
    """Background `HuggingFaceCatalog.fetch_files` call."""

    files = Signal(str, list)  # repo_id, list[HuggingFaceGGUFFile]
    failed = Signal(str, str)  # repo_id, error

    def __init__(self, catalog_factory, repo_id: str) -> None:
        super().__init__()
        self._catalog_factory = catalog_factory
        self._repo_id = repo_id

    @Slot()
    def run(self) -> None:
        try:
            catalog = self._catalog_factory()
            files = catalog.fetch_files(self._repo_id)
            self.files.emit(self._repo_id, files)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._repo_id, f"{type(exc).__name__}: {exc}")


class _HFInstallWorker(QObject):
    """Streams a GGUF download then `ollama create`s the result."""

    progress = Signal(str, int, int)  # status, completed, total
    finished = Signal(bool, str, str)  # ok, ollama_tag, message

    def __init__(
        self,
        catalog_factory,
        repo_id: str,
        filename: str,
        ollama_tag: str,
        dest_dir,
    ) -> None:
        super().__init__()
        self._catalog_factory = catalog_factory
        self._repo_id = repo_id
        self._filename = filename
        self._ollama_tag = ollama_tag
        self._dest_dir = dest_dir
        self._cancelled = False

    @Slot()
    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        from pathlib import Path as _Path

        from devin_local.gui.ollama_model_service import install_gguf_via_ollama

        try:
            catalog = self._catalog_factory()
            dest = _Path(self._dest_dir) / self._filename

            def _on_progress(completed: int, total: int) -> None:
                if self._cancelled:
                    raise RuntimeError("cancelled by user")
                self.progress.emit("downloading", completed, total)

            self.progress.emit("downloading", 0, 0)
            catalog.download_gguf(self._repo_id, self._filename, dest, progress=_on_progress)
            if self._cancelled:
                self.finished.emit(False, self._ollama_tag, "cancelled")
                return
            self.progress.emit("registering with ollama", 0, 0)
            install_gguf_via_ollama(dest, self._ollama_tag)
            self.finished.emit(True, self._ollama_tag, "success")
        except Exception as exc:  # noqa: BLE001
            log.exception("hf install worker failed")
            self.finished.emit(False, self._ollama_tag, f"{type(exc).__name__}: {exc}")


class _HFBrowseTab(QWidget):
    """Hugging Face GGUF browser: search, expand, download, register with Ollama."""

    installed = Signal(str)  # ollama tag after a successful install

    def __init__(self, catalog_factory, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._catalog_factory = catalog_factory
        self._entries: list[HuggingFaceGGUFEntry] = []
        self._search_thread: QThread | None = None
        self._files_thread: QThread | None = None
        self._install_thread: QThread | None = None
        self._install_worker: _HFInstallWorker | None = None
        self._selected_repo: str = ""
        self._selected_file: HuggingFaceGGUFFile | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro = QLabel(
            "Browse <a href='https://huggingface.co/models?library=gguf'>Hugging Face GGUF</a> "
            "models. Pick a repo, choose a quantization, click <b>Install</b> \u2014 the file "
            "is downloaded and registered with the local Ollama daemon as a new tag."
        )
        intro.setOpenExternalLinks(True)
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)

        # Search + filter row
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(8)
        filter_row.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("e.g. llama-3.2 instruct gguf")
        self._search_input.returnPressed.connect(self._on_search_clicked)
        filter_row.addWidget(self._search_input, 1)
        self._search_btn = QPushButton("Search")
        self._search_btn.clicked.connect(self._on_search_clicked)
        filter_row.addWidget(self._search_btn)

        filter_row.addWidget(QLabel("Family:"))
        self._family_filter = QComboBox()
        self._family_filter.addItems(
            ["All", "llama", "qwen", "mistral", "phi", "gemma", "deepseek", "other"]
        )
        self._family_filter.currentIndexChanged.connect(self._refilter_results)
        filter_row.addWidget(self._family_filter)
        layout.addLayout(filter_row)

        # Results list
        self._results = QListWidget()
        self._results.setAlternatingRowColors(False)
        self._results.currentItemChanged.connect(self._on_result_selected)
        layout.addWidget(self._results, 1)

        # Files (quantizations) for the selected repo
        files_label = QLabel("Quantizations in selected repo:")
        files_label.setObjectName("Muted")
        layout.addWidget(files_label)
        self._files_list = QListWidget()
        self._files_list.setMaximumHeight(140)
        self._files_list.currentItemChanged.connect(self._on_file_selected)
        layout.addWidget(self._files_list)

        # Install row
        install_row = QHBoxLayout()
        install_row.setContentsMargins(0, 0, 0, 0)
        install_row.setSpacing(6)
        install_row.addWidget(QLabel("New Ollama tag:"))
        self._tag_preview = QLabel("(select a file)")
        self._tag_preview.setObjectName("Muted")
        install_row.addWidget(self._tag_preview, 1)
        self._install_btn = QPushButton("Install")
        self._install_btn.setObjectName("Primary")
        self._install_btn.setEnabled(False)
        self._install_btn.clicked.connect(self._on_install_clicked)
        install_row.addWidget(self._install_btn)
        layout.addLayout(install_row)

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

        # Kick off an initial search for the most-downloaded GGUF repos.
        self._on_search_clicked()

    # ---------- search ----------

    def _on_search_clicked(self) -> None:
        if self._search_thread is not None:
            return
        query = self._search_input.text().strip()
        self._status.setText("Searching Hugging Face\u2026")
        self._results.clear()
        worker = _HFSearchWorker(self._catalog_factory, query)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.results.connect(self._on_search_results)
        worker.failed.connect(self._on_search_failed)
        worker.results.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_search_thread)
        thread.start()
        self._search_thread = thread

    def _clear_search_thread(self) -> None:
        self._search_thread = None

    def _on_search_results(self, entries: list) -> None:
        self._entries = list(entries)
        self._refilter_results()
        self._status.setText(f"Found {len(self._entries)} repo(s).")

    def _on_search_failed(self, message: str) -> None:
        self._entries = []
        self._results.clear()
        self._status.setText(f"Hugging Face search failed: {message}")

    def _refilter_results(self, *_: Any) -> None:
        family = self._family_filter.currentText()
        self._results.clear()
        for entry in self._entries:
            if family != "All" and entry.display_family != family:
                continue
            item = QListWidgetItem(self._format_entry_row(entry))
            item.setData(Qt.ItemDataRole.UserRole, entry.repo_id)
            item.setToolTip(", ".join(entry.tags[:8]))
            self._results.addItem(item)

    def _format_entry_row(self, entry: HuggingFaceGGUFEntry) -> str:
        downloads = entry.downloads
        return f"  {entry.repo_id:48s}  {downloads:>8} dl  {entry.display_family}"

    # ---------- file expansion ----------

    def _on_result_selected(self, current: QListWidgetItem | None, _prev: Any) -> None:
        if current is None:
            self._selected_repo = ""
            self._files_list.clear()
            return
        repo_id = str(current.data(Qt.ItemDataRole.UserRole) or "")
        self._selected_repo = repo_id
        if not repo_id:
            return
        if self._files_thread is not None:
            return
        self._files_list.clear()
        self._files_list.addItem(QListWidgetItem("Loading files\u2026"))
        worker = _HFFilesWorker(self._catalog_factory, repo_id)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.files.connect(self._on_files_loaded)
        worker.failed.connect(self._on_files_failed)
        worker.files.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_files_thread)
        thread.start()
        self._files_thread = thread

    def _clear_files_thread(self) -> None:
        self._files_thread = None

    def _on_files_loaded(self, repo_id: str, files: list) -> None:
        if repo_id != self._selected_repo:
            return
        self._files_list.clear()
        if not files:
            self._files_list.addItem(QListWidgetItem("(no .gguf files in this repo)"))
            return
        for file in files:
            label = f"  {file.quant:8s}  {file.size_human:>10}  {file.filename}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, file)
            self._files_list.addItem(item)

    def _on_files_failed(self, repo_id: str, message: str) -> None:
        if repo_id != self._selected_repo:
            return
        self._files_list.clear()
        self._files_list.addItem(QListWidgetItem(f"(failed: {message})"))

    def _on_file_selected(self, current: QListWidgetItem | None, _prev: Any) -> None:
        if current is None:
            self._selected_file = None
            self._install_btn.setEnabled(False)
            self._tag_preview.setText("(select a file)")
            return
        file = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(file, HuggingFaceGGUFFile):
            self._install_btn.setEnabled(False)
            self._tag_preview.setText("(select a file)")
            return
        self._selected_file = file
        entry = next(
            (e for e in self._entries if e.repo_id == self._selected_repo),
            None,
        )
        if entry is None:
            self._tag_preview.setText("(unknown repo)")
            return
        tag = entry.ollama_name(file)
        self._tag_preview.setText(tag)
        self._install_btn.setEnabled(True)

    # ---------- install ----------

    def _on_install_clicked(self) -> None:
        if self._install_thread is not None:
            return
        if self._selected_file is None or not self._selected_repo:
            self._status.setText("Select a repo and a .gguf file first.")
            return
        entry = next(
            (e for e in self._entries if e.repo_id == self._selected_repo),
            None,
        )
        if entry is None:
            self._status.setText("Selected repo is not in the result list anymore.")
            return
        from devin_local.settings import settings_dir as _settings_dir

        tag = entry.ollama_name(self._selected_file)
        dest_dir = _settings_dir() / "hf-models" / entry.repo_id.replace("/", "_")
        dest_dir.mkdir(parents=True, exist_ok=True)

        self._status.setText(f"Downloading {self._selected_file.filename}\u2026")
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._install_btn.setEnabled(False)

        worker = _HFInstallWorker(
            self._catalog_factory,
            self._selected_repo,
            self._selected_file.filename,
            tag,
            dest_dir,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_install_progress)
        worker.finished.connect(self._on_install_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_install_thread)
        thread.start()
        self._install_worker = worker
        self._install_thread = thread

    def cancel_active_install(self) -> None:
        if self._install_worker is not None:
            self._install_worker.cancel()
            self._status.setText("Cancelling\u2026")

    def _clear_install_thread(self) -> None:
        self._install_worker = None
        self._install_thread = None

    def _on_install_progress(self, status: str, completed: int, total: int) -> None:
        if total > 0:
            pct = int((completed / total) * 100)
            self._progress.setMaximum(100)
            self._progress.setValue(pct)
            self._status.setText(f"{status}  ({_human(completed)} / {_human(total)})")
        else:
            self._progress.setMaximum(0)
            self._status.setText(status)

    def _on_install_finished(self, ok: bool, tag: str, message: str) -> None:
        self._progress.setMaximum(100)
        self._progress.setValue(100 if ok else 0)
        self._install_btn.setEnabled(True)
        if ok:
            self._status.setText(f"Installed {tag}.")
            self.installed.emit(tag)
        else:
            self._status.setText(f"Install failed: {message}")


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
