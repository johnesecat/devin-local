"""Main window for the devin-local desktop GUI.

Layout::

    +-----------------------------------------------------+
    | Sidebar |  Chat pane                  |  Inspector  |
    |---------|----------------------------|              |
    | New     |  [user bubble]             |  Workspace   |
    | session |  [assistant bubble]        |  - files     |
    |         |  [tool card: write_file]   |  Plugins     |
    | History |  [assistant bubble]        |  MCP servers |
    |         |                            |              |
    |         |----------------------------|              |
    |         |  Composer + Send           |              |
    +-----------------------------------------------------+
    | Status bar: backend  |  model  |  workspace  |  ... |
    +-----------------------------------------------------+

The chat pane and inspector are inside a `QSplitter` so the user can resize
columns. The sidebar holds a session list (persisted via `SessionStore`).

The agent runs on a background thread (`AgentWorker`), so the UI stays
responsive even when the layered backend is grinding through 70B layers.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFileSystemModel,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from devin_local import __version__
from devin_local.agent import Agent, AgentConfig
from devin_local.gui.agent_worker import AgentWorker, run_in_worker_thread
from devin_local.gui.theme import stylesheet
from devin_local.gui.widgets import ChatPane, Composer, ToolCard
from devin_local.inference.factory import SUPPORTED_BACKENDS, list_available_backends
from devin_local.tools.base import ToolResult

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Top-level devin-local window."""

    request_submit = Signal(str)

    def __init__(
        self,
        workspace: Path,
        backend: str = "ollama",
        model: str = "llama3.1:8b",
        host: str = "http://127.0.0.1:11434",
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"devin-local \u2014 {workspace.name}")
        self.resize(1280, 800)
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        self._backend_name = backend
        self._model_name = model
        self._host = host
        self._agent: Agent | None = None
        self._worker: AgentWorker | None = None
        self._worker_thread = None
        self._pending_tool_cards: dict[str, ToolCard] = {}

        self._build_ui()
        self.setStyleSheet(stylesheet())
        self._initialize_agent()

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_main_pane())
        splitter.addWidget(self._build_inspector())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 760, 280])
        outer.addWidget(splitter, 1)
        outer.addWidget(self._build_status_bar(), 0)

        self.setCentralWidget(central)
        self._build_menus()

    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("Sidebar")
        side.setMinimumWidth(200)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(10)

        title = QLabel("devin-local")
        title.setObjectName("H1")
        layout.addWidget(title)

        version_label = QLabel(f"v{__version__}")
        version_label.setObjectName("Muted")
        layout.addWidget(version_label)

        new_btn = QPushButton("+ New session")
        new_btn.setObjectName("Primary")
        new_btn.clicked.connect(self._new_session)
        layout.addWidget(new_btn)

        layout.addSpacing(8)
        sessions_label = QLabel("Sessions")
        sessions_label.setObjectName("H2")
        layout.addWidget(sessions_label)
        self._session_list = QListWidget()
        self._session_list.itemDoubleClicked.connect(self._session_double_clicked)
        layout.addWidget(self._session_list, 1)
        self._refresh_session_list()

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._open_settings)
        layout.addWidget(settings_btn)

        return side

    def _build_main_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._chat = ChatPane()
        self._composer = Composer()
        self._composer.submitted.connect(self._on_user_submit)
        layout.addWidget(self._chat, 1)
        layout.addWidget(self._composer, 0)
        return pane

    def _build_inspector(self) -> QWidget:
        inspector = QWidget()
        inspector.setObjectName("Inspector")
        inspector.setMinimumWidth(240)
        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(_section_label("Workspace"))
        self._fs_model = QFileSystemModel()
        self._fs_model.setRootPath(str(self.workspace))
        self._fs_tree = QTreeView()
        self._fs_tree.setModel(self._fs_model)
        self._fs_tree.setRootIndex(self._fs_model.index(str(self.workspace)))
        self._fs_tree.setHeaderHidden(True)
        for column in range(1, self._fs_model.columnCount()):
            self._fs_tree.hideColumn(column)
        layout.addWidget(self._fs_tree, 1)

        layout.addWidget(_section_label("Backend"))
        self._backend_combo = QComboBox()
        for entry in list_available_backends():
            badge = " (ready)" if entry["available"] else " (missing)"
            self._backend_combo.addItem(entry["name"] + badge, entry["name"])
        self._select_backend_in_combo(self._backend_name)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        layout.addWidget(self._backend_combo)

        layout.addWidget(_section_label("Model"))
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.addItem(self._model_name)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(self._model_combo)

        return inspector

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("StatusBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(20)
        self._status_state = QLabel("idle")
        self._status_backend = QLabel(f"backend: {self._backend_name}")
        self._status_model = QLabel(f"model: {self._model_name}")
        self._status_workspace = QLabel(f"workspace: {self.workspace}")
        self._status_workspace.setObjectName("Muted")
        layout.addWidget(self._status_state)
        layout.addWidget(self._status_backend)
        layout.addWidget(self._status_model)
        layout.addStretch(1)
        layout.addWidget(self._status_workspace)
        return bar

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("&Open workspace\u2026", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._choose_workspace)
        file_menu.addAction(open_action)
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("&About devin-local", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ---------- agent wiring ----------

    def _initialize_agent(self) -> None:
        config = AgentConfig(
            model=self._model_name,
            workspace=self.workspace,
            backend=self._backend_name,
            ollama_host=self._host,
        )
        try:
            agent = Agent(config)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Backend unavailable",
                f"Failed to initialize backend {self._backend_name!r}:\n\n{exc}",
            )
            return
        worker = AgentWorker(agent)
        worker.token.connect(self._on_token)
        worker.tool_finished.connect(self._on_tool_finished)
        worker.turn_finished.connect(self._on_turn_finished)
        worker.error.connect(self._on_error)
        worker.state_changed.connect(self._on_state_changed)
        self.request_submit.connect(worker.submit)
        thread = run_in_worker_thread(self, worker)
        self._agent = agent
        self._worker = worker
        self._worker_thread = thread

    def _shutdown_agent(self) -> None:
        if self._worker is not None:
            with contextlib.suppress(Exception):
                self._worker.shutdown()
        if self._worker_thread is not None:
            with contextlib.suppress(Exception):
                self._worker_thread.quit()
                self._worker_thread.wait(2000)
        self._agent = None
        self._worker = None
        self._worker_thread = None

    # ---------- handlers ----------

    def _on_user_submit(self, text: str) -> None:
        if self._worker is None:
            QMessageBox.warning(self, "No backend", "Agent is not initialized.")
            return
        self._chat.add_user(text)
        self._chat.start_assistant()
        self._composer.set_busy(True)
        self.request_submit.emit(text)

    def _on_token(self, delta: str) -> None:
        self._chat.append_assistant_delta(delta)

    def _on_tool_finished(self, name: str, args: dict, result: ToolResult) -> None:
        card = self._chat.add_tool(name, args)
        card.finish(result)

    def _on_turn_finished(self, final_text: str, tool_count: int, elapsed_s: float) -> None:
        self._chat.finish_assistant(final_text or None)
        self._composer.set_busy(False)
        self._status_state.setText(
            f"idle  \u00b7  last turn: {elapsed_s:.1f}s  \u00b7  tools: {tool_count}"
        )

    def _on_error(self, message: str) -> None:
        self._chat.add_system_notice(f"\u26a0  {message}")
        self._composer.set_busy(False)
        self._status_state.setText("error")

    def _on_state_changed(self, state: str) -> None:
        self._status_state.setText(state)

    def _on_backend_changed(self, _index: int) -> None:
        chosen = self._backend_combo.currentData()
        if not chosen or chosen == self._backend_name:
            return
        self._backend_name = chosen
        self._status_backend.setText(f"backend: {chosen}")
        self._shutdown_agent()
        self._initialize_agent()
        self._chat.add_system_notice(f"Switched to backend: {chosen}")

    def _on_model_changed(self, model: str) -> None:
        model = model.strip()
        if not model or model == self._model_name:
            return
        self._model_name = model
        self._status_model.setText(f"model: {model}")
        if self._agent is not None:
            self._agent.config.model = model

    def _new_session(self) -> None:
        self._chat.clear()
        if self._agent is not None:
            self._agent.messages = []
            self._agent._initialized = False  # rebuild system prompt on next turn
        self._chat.add_system_notice("New session.")

    def _open_settings(self) -> None:
        QMessageBox.information(
            self,
            "Settings",
            "Backend + model are live-editable in the right inspector. "
            "Advanced settings (4-bit quantization, layer cache, MCP hosts, "
            "plugins) live in `devin_local/config.json` in your workspace.",
        )

    def _choose_workspace(self) -> None:
        new_path = QFileDialog.getExistingDirectory(self, "Choose workspace", str(self.workspace))
        if not new_path:
            return
        self._shutdown_agent()
        self.workspace = Path(new_path)
        self.setWindowTitle(f"devin-local \u2014 {self.workspace.name}")
        self._status_workspace.setText(f"workspace: {self.workspace}")
        self._fs_model.setRootPath(str(self.workspace))
        self._fs_tree.setRootIndex(self._fs_model.index(str(self.workspace)))
        self._refresh_session_list()
        self._chat.clear()
        self._initialize_agent()

    def _refresh_session_list(self) -> None:
        self._session_list.clear()
        sessions_dir = self.workspace / "sessions"
        if not sessions_dir.exists():
            return
        for path in sorted(sessions_dir.glob("*.jsonl"), reverse=True):
            item = QListWidgetItem(path.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self._session_list.addItem(item)

    def _session_double_clicked(self, item: QListWidgetItem) -> None:
        # Sessions are append-only JSONL files; loading would require building
        # the full transcript into the chat pane. For now we just surface the
        # path so the user can inspect it externally; loading is a follow-up.
        path = item.data(Qt.ItemDataRole.UserRole)
        QMessageBox.information(self, "Session", f"Session log:\n\n{path}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About devin-local",
            f"<h3>devin-local v{__version__}</h3>"
            "<p>A 100% local, free, private autonomous AI software-engineer agent.</p>"
            "<p>Backends: <code>ollama</code>, <code>layered</code> (AirLLM), "
            "<code>hf</code> (HuggingFace transformers).</p>"
            "<p><a href='https://github.com/johnesecat/devin-local'>"
            "github.com/johnesecat/devin-local</a></p>",
        )

    def _select_backend_in_combo(self, name: str) -> None:
        for i in range(self._backend_combo.count()):
            if self._backend_combo.itemData(i) == name:
                self._backend_combo.setCurrentIndex(i)
                return

    def closeEvent(self, event) -> None:  # noqa: D401, N802
        self._shutdown_agent()
        super().closeEvent(event)


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("H2")
    return label


def launch_gui(
    workspace: Path | None = None,
    backend: str = "ollama",
    model: str = "llama3.1:8b",
    host: str = "http://127.0.0.1:11434",
) -> int:
    """Boot a QApplication, show the main window, run the event loop."""
    if backend not in SUPPORTED_BACKENDS:
        backend = "ollama"
    workspace = workspace or Path.cwd()
    app = QApplication.instance() or QApplication(sys.argv)
    # High-DPI scaling: Qt 6 has this on by default, but enabling it
    # explicitly doesn't hurt and makes the intent clear.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    window = MainWindow(workspace=workspace, backend=backend, model=model, host=host)
    window.show()
    # Give Qt a moment to render before any agent work starts.
    QTimer.singleShot(0, lambda: None)
    return app.exec()


__all__ = ["launch_gui", "MainWindow"]


# Silence "unused import" warning for QIcon (kept for future window icons).
_ = QIcon
_ = Any
