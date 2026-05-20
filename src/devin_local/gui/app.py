"""Main window for the devin-local desktop GUI.

Layout::

    +-------------------------------------------------------+
    | Sidebar   |  Chat pane                |  Inspector    |
    |-----------|---------------------------|---------------|
    | + New     |  [user bubble]            |  Backend      |
    |           |  [plan card]              |  Model picker |
    | Sessions  |  [assistant bubble]       |  Plan pane    |
    |           |  [tool card: write_file]  |  Workspace    |
    |           |                           |  Sandbox      |
    |           |---------------------------|               |
    |           |  Composer + Send          |               |
    +-------------------------------------------------------+
    | Status bar: state · backend · model · workspace · …   |
    +-------------------------------------------------------+

Sections are inside a `QSplitter` so the user can resize columns. The agent
runs on a background thread (`AgentWorker`), so the UI stays responsive even
when the layered backend grinds through 70B layers.

This window also owns:

- The model selector, which talks to Ollama's ``/api/tags`` to list installed
  models and opens the library browser dialog to pull new ones.
- The backend dropdown, which prompts the user to install missing extras
  (``layered`` / ``hf``) via ``InstallBackendDialog`` rather than just
  erroring.
- The plan pane, which reflects the agent's live ``<plan>`` blocks.
- The sandbox panel, which exposes the workspace path, terminal/desktop
  state, enabled tools, and security flags.
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
    QScrollArea,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from devin_local import __version__
from devin_local.agent import Agent, AgentConfig
from devin_local.agent_planning import Plan
from devin_local.gui.agent_worker import AgentWorker, run_in_worker_thread
from devin_local.gui.backend_installer import (
    BACKEND_EXTRAS,
    InstallBackendDialog,
    backend_dep_probe,
)
from devin_local.gui.model_selector import LibraryBrowserDialog, ModelSelector
from devin_local.gui.ollama_model_service import (
    InstalledModel,
    OllamaModelService,
)
from devin_local.gui.sandbox_panel import PlanPane, SandboxPanel
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
        self.resize(1360, 860)
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        self._backend_name = backend
        self._model_name = model
        self._host = host
        self._agent: Agent | None = None
        self._worker: AgentWorker | None = None
        self._worker_thread = None
        self._pending_tool_cards: dict[str, ToolCard] = {}
        self._open_tool_cards: list[ToolCard] = []

        self._model_service = OllamaModelService(host=host)

        self._build_ui()
        self.setStyleSheet(stylesheet())
        self._initialize_agent()
        # Refresh installed Ollama models in the background after the window
        # paints so first-paint stays snappy.
        QTimer.singleShot(50, self._refresh_installed_models)

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
        splitter.setSizes([220, 760, 360])
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
        sessions_label = QLabel("SESSIONS")
        sessions_label.setObjectName("H2")
        layout.addWidget(sessions_label)
        self._session_list = QListWidget()
        self._session_list.itemDoubleClicked.connect(self._session_double_clicked)
        layout.addWidget(self._session_list, 1)
        self._refresh_session_list()

        settings_btn = QPushButton("Settings")
        from devin_local.gui.icons import icon as _icon

        gear = _icon("settings")
        if gear is not None:
            settings_btn.setIcon(gear)
        settings_btn.clicked.connect(self._open_settings)
        layout.addWidget(settings_btn)

        return side

    def _build_main_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._chat = ChatPane()
        self._chat.set_workspace(self.workspace)
        self._composer = Composer()
        self._composer.submitted.connect(self._on_user_submit)
        layout.addWidget(self._chat, 1)
        layout.addWidget(self._composer, 0)
        return pane

    def _build_inspector(self) -> QWidget:
        # The inspector is a scrollable column so it doesn't crush its children
        # when the window is narrow.
        inspector = QScrollArea()
        inspector.setObjectName("Inspector")
        inspector.setWidgetResizable(True)
        inspector.setMinimumWidth(280)
        inspector.setFrameShape(inspector.Shape.NoFrame)

        container = QWidget()
        container.setObjectName("Inspector")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(14)

        # --- Backend section -------------------------------------------------
        layout.addWidget(_section_label("BACKEND"))
        self._backend_combo = QComboBox()
        for entry in list_available_backends():
            badge = " · ready" if entry["available"] else " · install"
            self._backend_combo.addItem(entry["name"] + badge, entry["name"])
        self._select_backend_in_combo(self._backend_name)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        layout.addWidget(self._backend_combo)

        self._install_btn = QPushButton("Install backend dependencies\u2026")
        self._install_btn.setObjectName("Ghost")
        self._install_btn.setFlat(True)
        self._install_btn.clicked.connect(self._open_install_dialog)
        layout.addWidget(self._install_btn)
        self._update_install_btn_visibility()

        # --- Model section ---------------------------------------------------
        layout.addWidget(_section_label("MODEL"))
        self._model_selector = ModelSelector(self._model_name)
        self._model_selector.model_changed.connect(self._on_model_changed)
        self._model_selector.refresh_requested.connect(self._refresh_installed_models)
        self._model_selector.browse_requested.connect(self._open_model_browser)
        layout.addWidget(self._model_selector)

        # --- Plan ------------------------------------------------------------
        layout.addWidget(_section_label("PLAN"))
        self._plan_pane = PlanPane()
        layout.addWidget(self._plan_pane)

        # --- Workspace tree --------------------------------------------------
        layout.addWidget(_section_label("WORKSPACE"))
        self._fs_model = QFileSystemModel()
        self._fs_model.setRootPath(str(self.workspace))
        self._fs_tree = QTreeView()
        self._fs_tree.setModel(self._fs_model)
        self._fs_tree.setRootIndex(self._fs_model.index(str(self.workspace)))
        self._fs_tree.setHeaderHidden(True)
        for column in range(1, self._fs_model.columnCount()):
            self._fs_tree.hideColumn(column)
        self._fs_tree.setMinimumHeight(180)
        layout.addWidget(self._fs_tree, 1)

        # --- Sandbox ---------------------------------------------------------
        layout.addWidget(_section_label("SANDBOX"))
        self._sandbox = SandboxPanel()
        self._sandbox.set_workspace(self.workspace)
        layout.addWidget(self._sandbox)
        self._refresh_sandbox()

        inspector.setWidget(container)
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

        tools_menu = self.menuBar().addMenu("&Tools")
        browse_action = QAction("Browse Ollama models\u2026", self)
        browse_action.triggered.connect(self._open_model_browser)
        tools_menu.addAction(browse_action)
        refresh_action = QAction("&Refresh installed models", self)
        refresh_action.setShortcut("Ctrl+R")
        refresh_action.triggered.connect(self._refresh_installed_models)
        tools_menu.addAction(refresh_action)
        install_action = QAction("Install backend dependencies\u2026", self)
        install_action.triggered.connect(self._open_install_dialog)
        tools_menu.addAction(install_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("&About devin-local", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ---------- agent wiring ----------

    def _initialize_agent(self) -> None:
        # If the chosen backend's deps are missing, prompt to install instead
        # of crashing with a stack trace.
        ok, details = backend_dep_probe(self._backend_name)
        if not ok:
            self._chat.add_system_notice(
                f"Backend {self._backend_name!r} not installed ({details}). "
                "Use Tools \u2192 Install backend dependencies\u2026 or pick another backend."
            )
            return

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
        worker.tool_started.connect(self._on_tool_started)
        worker.tool_finished.connect(self._on_tool_finished)
        worker.plan_updated.connect(self._on_plan_updated)
        worker.turn_finished.connect(self._on_turn_finished)
        worker.error.connect(self._on_error)
        worker.state_changed.connect(self._on_state_changed)
        self.request_submit.connect(worker.submit)
        thread = run_in_worker_thread(self, worker)
        self._agent = agent
        self._worker = worker
        self._worker_thread = thread
        self._refresh_sandbox()

    def _shutdown_agent(self) -> None:
        if self._worker is not None:
            with contextlib.suppress(Exception):
                self.request_submit.disconnect(self._worker.submit)
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
            QMessageBox.warning(
                self,
                "No backend",
                "Agent is not initialized. Pick a working backend in the right inspector.",
            )
            return
        self._chat.add_user(text)
        self._chat.start_assistant()
        self._composer.set_busy(True)
        self._plan_pane.set_plan([])
        self._open_tool_cards = []
        self.request_submit.emit(text)

    def _on_token(self, delta: str) -> None:
        self._chat.append_assistant_delta(delta)

    def _on_tool_started(self, name: str, args: dict) -> None:
        card = self._chat.add_tool(name, args)
        self._open_tool_cards.append(card)

    def _on_tool_finished(self, name: str, args: dict, result: ToolResult) -> None:
        # Match to the most recent open card with this tool name, otherwise add one.
        for card in reversed(self._open_tool_cards):
            if card._name == name and card._arguments == args:  # noqa: SLF001 - intentional
                card.finish(result)
                self._open_tool_cards.remove(card)
                return
        # No matching start event (older flows that skipped tool_started):
        card = self._chat.add_tool(name, args)
        card.finish(result)
        self._refresh_sandbox()

    def _on_plan_updated(self, plan: Plan) -> None:
        self._plan_pane.set_plan(plan.steps)

    def _on_turn_finished(self, final_text: str, tool_count: int, elapsed_s: float) -> None:
        self._chat.finish_assistant(final_text or None)
        self._composer.set_busy(False)
        self._status_state.setText(
            f"idle  \u00b7  last turn: {elapsed_s:.1f}s  \u00b7  tools: {tool_count}"
        )
        self._refresh_sandbox()

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
        self._update_install_btn_visibility()
        ok, details = backend_dep_probe(chosen)
        if not ok:
            self._chat.add_system_notice(
                f"Backend {chosen!r} requires installing extras. {details}. "
                "Click the 'Install backend dependencies' button below the backend dropdown."
            )
            self._shutdown_agent()
            return
        self._shutdown_agent()
        self._initialize_agent()
        self._chat.add_system_notice(f"Switched to backend: {chosen}")

    def _on_model_changed(self, model: str) -> None:
        model = (model or "").strip()
        if not model or model == self._model_name:
            return
        self._model_name = model
        self._status_model.setText(f"model: {model}")
        if self._agent is not None:
            self._agent.config.model = model

    def _update_install_btn_visibility(self) -> None:
        ok, _ = backend_dep_probe(self._backend_name)
        self._install_btn.setVisible(not ok and self._backend_name in BACKEND_EXTRAS)

    def _open_install_dialog(self) -> None:
        if self._backend_name not in BACKEND_EXTRAS:
            QMessageBox.information(
                self,
                "Nothing to install",
                f"Backend {self._backend_name!r} doesn't have any extras to install.",
            )
            return
        dlg = InstallBackendDialog(self._backend_name, parent=self)
        dlg.installed.connect(self._on_backend_installed)
        dlg.exec()

    def _on_backend_installed(self, name: str) -> None:
        self._chat.add_system_notice(f"Installed backend {name!r}. Switching to it now\u2026")
        self._update_install_btn_visibility()
        self._shutdown_agent()
        self._initialize_agent()

    def _refresh_installed_models(self) -> None:
        try:
            models: list[InstalledModel] = self._model_service.list_installed()
        except Exception as exc:  # noqa: BLE001
            self._model_selector.set_status(f"Could not reach Ollama: {exc}")
            return
        self._model_selector.set_installed([m.name for m in models])
        self._model_selector.set_current(self._model_name)

    def _open_model_browser(self) -> None:
        try:
            installed = self._model_service.list_installed()
        except Exception:  # noqa: BLE001
            installed = []
        library = self._model_service.list_library()
        dlg = LibraryBrowserDialog(self._model_service, installed, library, parent=self)
        dlg.pulled.connect(self._on_model_pulled)
        dlg.exec()
        self._refresh_installed_models()

    def _on_model_pulled(self, name: str) -> None:
        self._chat.add_system_notice(f"Installed model {name!r}.")
        self._refresh_installed_models()

    def _new_session(self) -> None:
        self._chat.clear()
        self._plan_pane.set_plan([])
        if self._agent is not None:
            self._agent.messages = []
            self._agent.plan = Plan()
            self._agent._initialized = False  # rebuild system prompt on next turn
        self._chat.add_system_notice("New session.")

    def _open_settings(self) -> None:
        from devin_local.gui.settings_dialog import SettingsDialog

        dlg = SettingsDialog(self)
        dlg.settings_saved.connect(self._on_settings_saved)
        dlg.exec()

    def _on_settings_saved(self, settings: object) -> None:
        from devin_local.settings import Settings

        if not isinstance(settings, Settings):
            return
        # Apply changes that don't require an agent restart.
        if settings.general.workspace and Path(settings.general.workspace) != self.workspace:
            self._shutdown_agent()
            self.workspace = Path(settings.general.workspace)
            self.setWindowTitle(f"devin-local \u2014 {self.workspace.name}")
            self._status_workspace.setText(f"workspace: {self.workspace}")
            self._fs_model.setRootPath(str(self.workspace))
            self._fs_tree.setRootIndex(self._fs_model.index(str(self.workspace)))
            self._refresh_session_list()
            self._chat.clear()
            self._chat.set_workspace(self.workspace)
            self._sandbox.set_workspace(self.workspace)
        self._chat.add_system_notice(
            f"Settings saved. Defaults: model={settings.general.default_model}, "
            f"backend={settings.general.default_backend}."
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
        self._chat.set_workspace(self.workspace)
        self._sandbox.set_workspace(self.workspace)
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

    def _refresh_sandbox(self) -> None:
        if self._agent is None:
            self._sandbox.set_terminal_state(str(self.workspace), None)
            self._sandbox.set_desktop_state(False)
            self._sandbox.set_tools([])
            self._sandbox.set_flags(network=False, browser=False, desktop=False)
            return
        registry = self._agent.registry
        names = registry.names()
        self._sandbox.set_tools(names)
        # Pull terminal state from the registry's shell_session emulator, if any.
        cwd = str(self.workspace)
        last_cmd = None
        session_tool = registry.get("shell_session")
        if session_tool is not None:
            emul = getattr(session_tool, "emulator", None)
            if emul is not None:
                cwd = str(getattr(emul, "cwd", cwd) or cwd)
                last_cmd = getattr(emul, "last_command", None)
        self._sandbox.set_terminal_state(cwd, last_cmd)
        desktop_enabled = self._agent.config.enable_desktop
        self._sandbox.set_desktop_state(desktop_enabled)
        self._sandbox.set_flags(
            network=self._agent.config.enable_browser,
            browser=self._agent.config.enable_browser,
            desktop=desktop_enabled,
        )

    def _select_backend_in_combo(self, name: str) -> None:
        for i in range(self._backend_combo.count()):
            if self._backend_combo.itemData(i) == name:
                self._backend_combo.setCurrentIndex(i)
                return

    def closeEvent(self, event) -> None:  # noqa: D401, N802
        self._shutdown_agent()
        with contextlib.suppress(Exception):
            self._model_service.close()
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
