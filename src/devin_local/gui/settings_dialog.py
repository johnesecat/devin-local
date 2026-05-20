"""Settings dialog for the devin-local desktop app.

Tabs:
- **General**: workspace, default model + backend, Ollama host, parallel
  tool calls toggle, OBLITERATUS toggle, planning toggle.
- **MCP**: list, add, edit, remove, test MCP servers. Persisted to
  ``~/.devin-local/mcp_servers.json``.
- **GitHub**: paste a PAT, "Test connection" hits ``/user``. Persisted to
  ``~/.devin-local/github.json`` (0600).
- **Backends**: probe ``ollama`` / ``layered`` / ``hf`` and offer one-click
  install for missing extras (same plumbing as ``devin-local install``).
- **Appearance**: theme name, font scale, accent color.

The dialog mutates a ``Settings`` instance in memory and only writes to disk
on **Save**. Cancel discards.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from devin_local.gui.backend_installer import backend_dep_probe
from devin_local.gui.icons import icon
from devin_local.settings import (
    MCPServer,
    Settings,
    load_github,
    load_mcp_servers,
    save_github,
    save_mcp_servers,
    verify_github_pat,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


class _GeneralTab(QWidget):
    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.workspace = QLineEdit(settings.general.workspace)
        self.workspace.setPlaceholderText("(empty = current working directory)")
        ws_row = QHBoxLayout()
        ws_row.addWidget(self.workspace, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_workspace)
        ws_row.addWidget(browse)
        ws_wrap = QWidget()
        ws_wrap.setLayout(ws_row)
        form.addRow("Workspace", ws_wrap)

        self.default_model = QLineEdit(settings.general.default_model)
        form.addRow("Default model", self.default_model)

        self.default_backend = QComboBox()
        self.default_backend.addItems(["ollama", "layered", "hf"])
        idx = self.default_backend.findText(settings.general.default_backend)
        if idx >= 0:
            self.default_backend.setCurrentIndex(idx)
        form.addRow("Default backend", self.default_backend)

        self.ollama_host = QLineEdit(settings.general.ollama_host)
        form.addRow("Ollama host", self.ollama_host)

        self.parallel = QCheckBox("Dispatch multiple tool calls in parallel")
        self.parallel.setChecked(settings.general.parallel_tool_calls)
        form.addRow("", self.parallel)

        self.obliteratus = QCheckBox("Include OBLITERATUS directive in system prompt")
        self.obliteratus.setChecked(settings.general.enable_obliteratus)
        form.addRow("", self.obliteratus)

        self.planning = QCheckBox("Plan-first workflow (emit & track <plan> blocks)")
        self.planning.setChecked(settings.general.enable_planning)
        form.addRow("", self.planning)

    def _on_browse_workspace(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose workspace")
        if chosen:
            self.workspace.setText(chosen)

    def apply_to(self, settings: Settings) -> None:
        settings.general = replace(
            settings.general,
            workspace=self.workspace.text().strip(),
            default_model=self.default_model.text().strip() or settings.general.default_model,
            default_backend=self.default_backend.currentText(),
            ollama_host=self.ollama_host.text().strip() or settings.general.ollama_host,
            parallel_tool_calls=self.parallel.isChecked(),
            enable_obliteratus=self.obliteratus.isChecked(),
            enable_planning=self.planning.isChecked(),
        )


class _MCPEditorRow(QDialog):
    def __init__(self, server: MCPServer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Edit MCP server: {server.name or 'new'}")
        self._server = MCPServer(**asdict(server))
        form = QFormLayout()
        self.name = QLineEdit(server.name)
        self.transport = QComboBox()
        self.transport.addItems(["stdio", "http", "sse"])
        idx = self.transport.findText(server.transport)
        if idx >= 0:
            self.transport.setCurrentIndex(idx)
        self.command = QLineEdit(server.command)
        self.command.setPlaceholderText("/path/to/mcp-server (stdio only)")
        self.args = QLineEdit(" ".join(server.args))
        self.args.setPlaceholderText("space-separated args (stdio only)")
        self.url = QLineEdit(server.url)
        self.url.setPlaceholderText("http://127.0.0.1:8765/mcp (http/sse only)")
        self.enabled = QCheckBox("Enabled")
        self.enabled.setChecked(server.enabled)
        form.addRow("Name", self.name)
        form.addRow("Transport", self.transport)
        form.addRow("Command", self.command)
        form.addRow("Args", self.args)
        form.addRow("URL", self.url)
        form.addRow("", self.enabled)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def server(self) -> MCPServer:
        return MCPServer(
            name=self.name.text().strip() or "unnamed",
            transport=self.transport.currentText(),
            command=self.command.text().strip(),
            args=[a for a in self.args.text().split() if a],
            url=self.url.text().strip(),
            enabled=self.enabled.isChecked(),
            env={},
        )


class _MCPTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._servers: list[MCPServer] = load_mcp_servers()
        layout = QVBoxLayout(self)
        hint = QLabel(
            "MCP servers add extra tools to the agent's toolbelt. Add a "
            "stdio command or an HTTP/SSE URL."
        )
        hint.setWordWrap(True)
        hint.setObjectName("Hint")
        layout.addWidget(hint)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget, 1)

        row = QHBoxLayout()
        self.add_btn = QPushButton("Add…")
        plus = icon("plus")
        if plus:
            self.add_btn.setIcon(plus)
        self.add_btn.clicked.connect(self._on_add)
        self.edit_btn = QPushButton("Edit…")
        self.edit_btn.clicked.connect(self._on_edit)
        self.remove_btn = QPushButton("Remove")
        trash = icon("trash")
        if trash:
            self.remove_btn.setIcon(trash)
        self.remove_btn.clicked.connect(self._on_remove)
        self.test_btn = QPushButton("Test connection")
        self.test_btn.clicked.connect(self._on_test)
        row.addWidget(self.add_btn)
        row.addWidget(self.edit_btn)
        row.addWidget(self.remove_btn)
        row.addStretch(1)
        row.addWidget(self.test_btn)
        layout.addLayout(row)

        self._refresh()

    def _refresh(self) -> None:
        self.list_widget.clear()
        for s in self._servers:
            tag = s.transport
            target = s.url if s.transport != "stdio" else (s.command or "(no command)")
            text = f"{s.name}  ·  {tag}  ·  {target}{'' if s.enabled else '  (disabled)'}"
            self.list_widget.addItem(QListWidgetItem(text))

    def _selected_index(self) -> int:
        items = self.list_widget.selectedIndexes()
        return items[0].row() if items else -1

    def _on_add(self) -> None:
        dlg = _MCPEditorRow(MCPServer(name=""), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._servers.append(dlg.server())
            self._refresh()

    def _on_edit(self) -> None:
        idx = self._selected_index()
        if idx < 0:
            return
        dlg = _MCPEditorRow(self._servers[idx], parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._servers[idx] = dlg.server()
            self._refresh()

    def _on_remove(self) -> None:
        idx = self._selected_index()
        if idx < 0:
            return
        del self._servers[idx]
        self._refresh()

    def _on_test(self) -> None:
        idx = self._selected_index()
        if idx < 0:
            QMessageBox.information(self, "Test", "Select a server first.")
            return
        server = self._servers[idx]
        if server.transport == "stdio":
            from shutil import which

            if not server.command:
                QMessageBox.warning(self, "Test", "No command set on this stdio server.")
                return
            resolved = which(server.command) if "/" not in server.command else server.command
            if not resolved:
                QMessageBox.warning(
                    self,
                    "Test",
                    f"Command not found on PATH: {server.command}",
                )
                return
            QMessageBox.information(
                self,
                "Test",
                f"Command resolves to: {resolved}\n\nFull MCP handshake is "
                f"performed when the agent starts.",
            )
            return
        if not server.url:
            QMessageBox.warning(self, "Test", "No URL set on this server.")
            return
        try:
            import httpx

            response = httpx.get(server.url, timeout=3.0)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Test", f"Connection failed: {exc}")
            return
        QMessageBox.information(self, "Test", f"GET {server.url} → HTTP {response.status_code}")

    def apply(self) -> None:
        save_mcp_servers(self._servers)


class _GitHubVerifyWorker(QObject):
    finished = Signal(bool, str)  # ok, message

    def __init__(self, pat: str) -> None:
        super().__init__()
        self._pat = pat

    def run(self) -> None:
        try:
            data = verify_github_pat(self._pat)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))
            return
        login = data.get("login", "?")
        self.finished.emit(True, login)


class _GitHubTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._gh = load_github()
        form = QFormLayout(self)
        self.username = QLineEdit(self._gh.username)
        self.username.setPlaceholderText("(optional — auto-filled by Test)")
        self.pat = QLineEdit(self._gh.pat)
        self.pat.setEchoMode(QLineEdit.EchoMode.Password)
        self.pat.setPlaceholderText("ghp_… or github_pat_… (stored locally with 0600 perms)")
        self.status = QLabel(
            f"Last verified: @{self._gh.last_verified_login}"
            if self._gh.last_verified_login
            else "Not verified."
        )
        self.status.setObjectName("Hint")

        self.test_btn = QPushButton("Test connection")
        gh_icon = icon("github")
        if gh_icon:
            self.test_btn.setIcon(gh_icon)
        self.test_btn.clicked.connect(self._on_test)

        form.addRow("Username", self.username)
        form.addRow("Personal Access Token", self.pat)
        form.addRow("", self.test_btn)
        form.addRow("", self.status)

        self._thread: QThread | None = None
        self._worker: _GitHubVerifyWorker | None = None

    def _on_test(self) -> None:
        pat = self.pat.text().strip()
        if not pat:
            QMessageBox.warning(self, "Test", "Paste a PAT first.")
            return
        self.test_btn.setEnabled(False)
        self.status.setText("Verifying…")
        thread = QThread(self)
        worker = _GitHubVerifyWorker(pat)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_verified)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_verified(self, ok: bool, message: str) -> None:
        self.test_btn.setEnabled(True)
        if ok:
            self.status.setText(f"OK — logged in as @{message}")
            self.username.setText(message)
            self._gh.last_verified_login = message
        else:
            self.status.setText(f"Failed: {message}")

    def apply(self) -> None:
        self._gh.pat = self.pat.text().strip()
        self._gh.username = self.username.text().strip()
        save_github(self._gh)


class _BackendsTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Each backend has its own optional dependencies. Click Install "
            "to pip-install the right extras for missing backends. ollama "
            "needs no Python deps but does need the Ollama daemon running."
        )
        hint.setWordWrap(True)
        hint.setObjectName("Hint")
        layout.addWidget(hint)
        self._rows: dict[str, QLabel] = {}
        for name in ("ollama", "layered", "hf"):
            row = QHBoxLayout()
            label = QLabel(name)
            label.setMinimumWidth(80)
            status = QLabel("(probing…)")
            status.setObjectName("Hint")
            install = QPushButton("Install")
            install.clicked.connect(lambda _=False, n=name: self._install(n))
            if name == "ollama":
                install.setEnabled(False)
            row.addWidget(label)
            row.addWidget(status, 1)
            row.addWidget(install)
            self._rows[name] = status
            wrapper = QWidget()
            wrapper.setLayout(row)
            layout.addWidget(wrapper)
        layout.addStretch(1)
        self._probe()

    def _probe(self) -> None:

        for name, status in self._rows.items():
            ok, detail = backend_dep_probe(name)
            tag = "ready" if ok else "missing"
            status.setText(f"{tag} — {detail}")

    def _install(self, backend: str) -> None:
        from devin_local.gui.backend_installer import InstallBackendDialog

        dlg = InstallBackendDialog(backend, parent=self)
        dlg.exec()
        self._probe()

    def apply(self) -> None:  # no-op
        pass


class _AppearanceTab(QWidget):
    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        form = QFormLayout(self)
        self.theme = QComboBox()
        self.theme.addItems(["tokyo-night"])
        idx = self.theme.findText(settings.appearance.theme)
        if idx >= 0:
            self.theme.setCurrentIndex(idx)
        self.font_scale = QDoubleSpinBox()
        self.font_scale.setRange(0.75, 1.75)
        self.font_scale.setSingleStep(0.05)
        self.font_scale.setValue(settings.appearance.font_scale)
        self.accent = QLineEdit(settings.appearance.accent)
        self.show_sandbox = QCheckBox("Show sandbox panel")
        self.show_sandbox.setChecked(settings.appearance.show_sandbox_panel)
        self.show_plan = QCheckBox("Show plan pane")
        self.show_plan.setChecked(settings.appearance.show_plan_pane)
        form.addRow("Theme", self.theme)
        form.addRow("Font scale", self.font_scale)
        form.addRow("Accent (hex)", self.accent)
        form.addRow("", self.show_sandbox)
        form.addRow("", self.show_plan)

    def apply_to(self, settings: Settings) -> None:
        settings.appearance = replace(
            settings.appearance,
            theme=self.theme.currentText(),
            font_scale=float(self.font_scale.value()),
            accent=self.accent.text().strip() or settings.appearance.accent,
            show_sandbox_panel=self.show_sandbox.isChecked(),
            show_plan_pane=self.show_plan.isChecked(),
        )


class SettingsDialog(QDialog):
    """Multi-tab settings dialog. Returns ``QDialog.Accepted`` on save."""

    settings_saved = Signal(object)  # emits the new Settings instance

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(640, 520)
        self._settings = Settings.load()
        self._tabs = QTabWidget()

        self.general_tab = _GeneralTab(self._settings)
        self.mcp_tab = _MCPTab()
        self.github_tab = _GitHubTab()
        self.backends_tab = _BackendsTab()
        self.appearance_tab = _AppearanceTab(self._settings)

        for name, widget, icon_name in (
            ("General", self.general_tab, "general"),
            ("MCP", self.mcp_tab, "mcp"),
            ("GitHub", self.github_tab, "github"),
            ("Backends", self.backends_tab, "backend"),
            ("Appearance", self.appearance_tab, "appearance"),
        ):
            ico = icon(icon_name)
            if ico is not None:
                self._tabs.addTab(widget, ico, name)
            else:
                self._tabs.addTab(widget, name)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs, 1)
        layout.addWidget(buttons)

    def _on_save(self) -> None:
        self.general_tab.apply_to(self._settings)
        self.appearance_tab.apply_to(self._settings)
        self._settings.save()
        self.mcp_tab.apply()
        self.github_tab.apply()
        self.backends_tab.apply()
        self.settings_saved.emit(self._settings)
        self.accept()

    def current_settings(self) -> Settings:
        return self._settings
