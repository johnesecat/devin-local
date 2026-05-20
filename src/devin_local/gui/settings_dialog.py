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
from pathlib import Path
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
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from devin_local.figma.client import (
    FigmaClient,
    FigmaError,
    FigmaSettings,
    load_figma_settings,
    parse_file_key,
    remember_last_file,
    save_figma_settings,
)
from devin_local.figma.tokens import extract_design_tokens, tokens_to_python_module
from devin_local.gui.backend_installer import backend_dep_probe
from devin_local.gui.icons import icon
from devin_local.knowledge.store import KnowledgeStore
from devin_local.settings import (
    MCPServer,
    Settings,
    load_github,
    load_mcp_servers,
    save_github,
    save_mcp_servers,
    user_knowledge_path,
    user_tools_dir,
    verify_github_pat,
)
from devin_local.tools.user_tools import (
    list_user_tool_files,
    load_user_tools,
    starter_template,
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

        self.verbose_prompt = QCheckBox(
            "Verbose system prompt (~13 KB; slower local inference, more guidance)"
        )
        self.verbose_prompt.setChecked(settings.general.verbose_prompt)
        self.verbose_prompt.setToolTip(
            "When off, devin-local uses the slim ~3 KB system prompt. The slim "
            "prompt still carries every identity, honesty, security, planning, "
            "and tool-use rule — only the long elaboration is dropped. Recommended "
            "for CPU-only inference on small models."
        )
        form.addRow("", self.verbose_prompt)

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
            verbose_prompt=self.verbose_prompt.isChecked(),
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


class _AddNoteDialog(QDialog):
    """Compose a single knowledge note (title + body + scope/tags)."""

    def __init__(self, parent: QWidget | None = None, *, title: str = "", body: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Add knowledge note")
        self.setMinimumSize(560, 420)
        form = QFormLayout()
        self.title_edit = QLineEdit(title)
        self.title_edit.setPlaceholderText("Short title — e.g. 'Project build commands'")
        self.scope_edit = QLineEdit()
        self.scope_edit.setPlaceholderText("Comma-separated hints — e.g. 'rust, build, ci'")
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("Comma-separated tags")
        self.body_edit = QPlainTextEdit(body)
        self.body_edit.setPlaceholderText(
            "Paste the knowledge content. Markdown is fine — it will be embedded "
            "verbatim into the system prompt at session start."
        )
        form.addRow("Title", self.title_edit)
        form.addRow("Scope", self.scope_edit)
        form.addRow("Tags", self.tags_edit)
        form.addRow("Body", self.body_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form, 1)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str, str, list[str]]:
        tags_raw = self.tags_edit.text().strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        return (
            self.title_edit.text().strip() or "Untitled",
            self.body_edit.toPlainText().strip(),
            self.scope_edit.text().strip(),
            tags,
        )


class _KnowledgeTab(QWidget):
    """Manage the user-level knowledge store (``~/.devin-local/knowledge``).

    Everything here is embedded into the system prompt at session start, so
    the agent does NOT need to re-search this corpus per turn. Users can
    upload files, paste text, or remove notes — changes apply on the next
    agent turn.
    """

    knowledge_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = KnowledgeStore.open(user_knowledge_path())

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Knowledge notes are embedded directly into the agent's system "
            "prompt at session start. The agent does NOT re-search this "
            "corpus on every turn — that saves tokens and keeps responses "
            "fast. Upload anything you want the agent to remember across "
            "sessions: project conventions, API keys to NOT touch, build "
            "commands, runbooks, etc."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.list = QListWidget()
        self.list.setObjectName("KnowledgeList")
        self.list.setSelectionMode(self.list.SelectionMode.SingleSelection)
        layout.addWidget(self.list, 1)

        button_row = QHBoxLayout()
        self.upload_btn = QPushButton("Upload file…")
        self.upload_btn.clicked.connect(self._on_upload_file)
        self.paste_btn = QPushButton("Add text…")
        self.paste_btn.clicked.connect(self._on_add_text)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete)
        self.delete_btn.setEnabled(False)
        button_row.addWidget(self.upload_btn)
        button_row.addWidget(self.paste_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.delete_btn)
        layout.addLayout(button_row)

        self.usage_label = QLabel()
        self.usage_label.setObjectName("Hint")
        layout.addWidget(self.usage_label)

        self.list.itemSelectionChanged.connect(self._on_selection_changed)
        self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        total_chars = 0
        for note in self._store.all():
            item = QListWidgetItem(self._format_item_label(note))
            item.setData(Qt.ItemDataRole.UserRole, note.id)
            self.list.addItem(item)
            total_chars += len(note.body)
        self.usage_label.setText(
            f"{len(self._store.all())} note(s) · {total_chars:,} chars total "
            f"(embedded into every new session's system prompt)"
        )

    @staticmethod
    def _format_item_label(note) -> str:  # noqa: ANN001 - KnowledgeNote
        scope = f" — scope: {note.scope}" if note.scope else ""
        size = len(note.body)
        return f"{note.title}{scope}  ({size:,} chars)"

    def _on_selection_changed(self) -> None:
        self.delete_btn.setEnabled(bool(self.list.selectedItems()))

    def _on_upload_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Upload knowledge file",
            "",
            "Text files (*.md *.txt *.rst *.json *.yaml *.yml);;All files (*)",
        )
        if not chosen:
            return
        try:
            with open(chosen, encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError as exc:
            QMessageBox.warning(self, "Upload failed", f"Could not read file:\n{exc}")
            return
        from pathlib import Path as _Path

        # Default title from the filename — user can change it via "Add text".
        default_title = _Path(chosen).stem
        title, ok = QInputDialog.getText(
            self,
            "Note title",
            "Title for this knowledge note:",
            text=default_title,
        )
        if not ok or not title.strip():
            return
        self._store.add(title=title.strip(), body=body)
        self._refresh()
        self.knowledge_changed.emit()

    def _on_add_text(self) -> None:
        dlg = _AddNoteDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        title, body, scope, tags = dlg.values()
        if not body.strip():
            QMessageBox.information(self, "Empty", "Body is empty — nothing saved.")
            return
        self._store.add(title=title, body=body, scope=scope, tags=tags)
        self._refresh()
        self.knowledge_changed.emit()

    def _on_delete(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        note_id = item.data(Qt.ItemDataRole.UserRole)
        confirmed = QMessageBox.question(
            self,
            "Delete note",
            f"Delete '{item.text()}'? This is permanent.",
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        if self._store.remove(note_id):
            self._refresh()
            self.knowledge_changed.emit()

    def apply(self) -> None:  # no-op; we persist on every action.
        pass


class _FigmaVerifyWorker(QObject):
    """Background worker for Figma API calls so the UI doesn't freeze."""

    finished = Signal(bool, str)  # ok, message (login or error)

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token

    def run(self) -> None:
        try:
            client = FigmaClient(token=self._token)
            data = client.me()
        except FigmaError as exc:
            self.finished.emit(False, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, f"unexpected: {exc}")
            return
        handle = data.get("handle") or data.get("email") or data.get("id") or "?"
        self.finished.emit(True, str(handle))


class _FigmaImportWorker(QObject):
    """Background worker for the actual file import + token extraction."""

    finished = Signal(bool, str, object)
    # ok, message, payload-dict (with summary counts + output path) | None

    def __init__(self, token: str, file_url: str, output_path: str) -> None:
        super().__init__()
        self._token = token
        self._file_url = file_url
        self._output_path = output_path

    def run(self) -> None:
        try:
            client = FigmaClient(token=self._token)
            payload = client.get_file(self._file_url, depth=4)
        except FigmaError as exc:
            self.finished.emit(False, str(exc), None)
            return
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, f"fetch failed: {exc}", None)
            return
        try:
            tokens = extract_design_tokens(payload)
            module_text = tokens_to_python_module(tokens)
            from pathlib import Path as _P

            out = _P(self._output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(module_text, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, f"token extraction failed: {exc}", None)
            return
        summary = {
            "file_name": tokens.source_file_name,
            "file_key": tokens.source_file_key,
            "colors": len(tokens.colors),
            "typography": len(tokens.typography),
            "radii": len(tokens.radii),
            "spacing": len(tokens.spacing),
            "shadows": len(tokens.shadows),
            "output_path": str(out),
        }
        self.finished.emit(True, "imported", summary)


class _FigmaTab(QWidget):
    """Settings tab to manage the Figma PAT and import design tokens."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fs: FigmaSettings = load_figma_settings()
        form = QFormLayout(self)

        self.token = QLineEdit(self._fs.token)
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText(
            "figd_… or fig_… (stored locally under ~/.devin-local/figma.json)"
        )

        self.file_url = QLineEdit(self._fs.last_file_url or self._fs.last_file_key)
        self.file_url.setPlaceholderText(
            "https://www.figma.com/file/<KEY>/<name> — or just the file key"
        )

        self.output_path = QLineEdit("design_tokens_figma.py")
        self.output_path.setPlaceholderText("Relative path under your workspace")

        self.test_btn = QPushButton("Test connection")
        self.test_btn.clicked.connect(self._on_test)
        fig_icon = icon("figma")
        if fig_icon:
            self.test_btn.setIcon(fig_icon)

        self.import_btn = QPushButton("Import design tokens")
        self.import_btn.setObjectName("Primary")
        self.import_btn.clicked.connect(self._on_import)
        imp_icon = icon("import")
        if imp_icon:
            self.import_btn.setIcon(imp_icon)

        self.status = QLabel(
            f"Last used file: {self._fs.last_file_key}"
            if self._fs.last_file_key
            else "No file imported yet."
        )
        self.status.setObjectName("Hint")
        self.status.setWordWrap(True)

        form.addRow("Personal access token", self.token)
        form.addRow("File URL or key", self.file_url)
        form.addRow("Output module", self.output_path)
        row = QHBoxLayout()
        row.addWidget(self.test_btn)
        row.addStretch(1)
        row.addWidget(self.import_btn)
        wrapper = QWidget(self)
        wrapper.setLayout(row)
        form.addRow("", wrapper)
        form.addRow("", self.status)

        # Friendly hint about Figma personal access tokens.
        hint = QLabel(
            "Get a PAT at https://www.figma.com/developers/personal-access-token — "
            "scopes: 'File content' (read)."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        form.addRow("", hint)

        self._thread: QThread | None = None
        self._worker: QObject | None = None

    # ---- handlers ------------------------------------------------------

    def _on_test(self) -> None:
        tok = self.token.text().strip()
        if not tok:
            QMessageBox.warning(self, "Test", "Paste a Figma PAT first.")
            return
        self.test_btn.setEnabled(False)
        self.status.setText("Verifying token via /v1/me…")
        thread = QThread(self)
        worker = _FigmaVerifyWorker(tok)
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
            self.status.setText(f"OK — signed in as @{message}")
        else:
            self.status.setText(f"Failed: {message}")

    def _on_import(self) -> None:
        tok = self.token.text().strip()
        url = self.file_url.text().strip()
        rel_out = self.output_path.text().strip() or "design_tokens_figma.py"
        if not tok:
            QMessageBox.warning(self, "Import", "Paste a Figma PAT first.")
            return
        if not url:
            QMessageBox.warning(self, "Import", "Paste a Figma file URL or key.")
            return
        try:
            parse_file_key(url)
        except FigmaError as exc:
            QMessageBox.warning(self, "Import", f"Could not parse file key: {exc}")
            return
        # Resolve output path against the workspace.
        from pathlib import Path as _P

        # Local import so the tab is cheap when unused.
        from devin_local.settings import Settings as _Settings

        s = _Settings.load()
        workspace = _P(s.general.workspace).expanduser()
        out_path = _P(rel_out)
        if not out_path.is_absolute():
            out_path = workspace / out_path

        self.import_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.status.setText(f"Fetching {url}… (this can take a few seconds)")

        thread = QThread(self)
        worker = _FigmaImportWorker(tok, url, str(out_path))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_imported)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_imported(self, ok: bool, message: str, payload: object) -> None:
        self.import_btn.setEnabled(True)
        self.test_btn.setEnabled(True)
        if not ok:
            self.status.setText(f"Import failed: {message}")
            QMessageBox.warning(self, "Import", f"Failed: {message}")
            return
        assert isinstance(payload, dict)
        out = payload.get("output_path", "?")
        self.status.setText(
            "Imported '{name}' \u2192 {colors} colors, {typo} text styles, "
            "{rad} radii, {sp} spacing, {sh} shadows \u2192 {out}".format(
                name=payload.get("file_name", "?"),
                colors=payload.get("colors", 0),
                typo=payload.get("typography", 0),
                rad=payload.get("radii", 0),
                sp=payload.get("spacing", 0),
                sh=payload.get("shadows", 0),
                out=out,
            )
        )
        QMessageBox.information(
            self,
            "Imported",
            f"Wrote {out}\n\nImport the module from this path to apply the tokens.",
        )

    # ---- persistence ---------------------------------------------------

    def apply(self) -> None:
        """Persist token + last file. The PAT is written to disk with 0600 perms
        by ``save_figma_settings`` so we never echo it back into logs.
        """
        self._fs.token = self.token.text().strip()
        url = self.file_url.text().strip()
        if url:
            try:
                self._fs.last_file_key = parse_file_key(url)
                self._fs.last_file_url = url
            except FigmaError:
                self._fs.last_file_url = url
        save_figma_settings(self._fs)
        # Convenience: also remember in case the user typed a URL but didn't
        # press Import. This keeps the field sticky between sessions.
        if url:
            import contextlib

            with contextlib.suppress(FigmaError):
                remember_last_file(url)


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


class _ToolsTab(QWidget):
    """Manage user-defined Python tools at ``~/.devin-local/tools/``.

    Capabilities:

    - List every ``.py`` file in the user-tools directory.
    - For each file: name, source path, and which @tool functions it
      exposes (or the loader error if the file failed to import).
    - **New tool**: prompts for a name + description, drops a starter
      ``.py`` skeleton into the directory.
    - **Open**: opens the file in the system editor.
    - **Edit code in app**: inline code editor for quick tweaks; Save
      writes back to disk.
    - **Delete**: removes the ``.py``.
    - **Upload**: copies an existing ``.py`` from disk into the directory.
    - **Reload**: re-imports every file and refreshes the list.

    The agent picks up changes on the next ``initialize()`` or via
    ``Agent.reload_user_tools()``.
    """

    tools_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._directory = user_tools_dir()
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QLabel(
            "Custom Python tools registered with the @tool decorator. "
            "Drop a .py file under "
            f"<code>{self._directory}</code> "
            "or use the buttons below. The agent picks the right tool for "
            "each task on its own — you do not need to mention them."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)

        self._list = QListWidget()
        self._list.setObjectName("UserToolsList")
        self._list.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self._list, 1)

        button_row = QHBoxLayout()
        self._new_btn = QPushButton("New tool…")
        self._new_btn.clicked.connect(self._on_new)
        button_row.addWidget(self._new_btn)
        self._upload_btn = QPushButton("Upload .py…")
        self._upload_btn.clicked.connect(self._on_upload)
        button_row.addWidget(self._upload_btn)
        self._open_btn = QPushButton("Open file")
        self._open_btn.clicked.connect(self._on_open)
        button_row.addWidget(self._open_btn)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setObjectName("Danger")
        self._delete_btn.clicked.connect(self._on_delete)
        button_row.addWidget(self._delete_btn)
        self._reload_btn = QPushButton("Reload")
        self._reload_btn.clicked.connect(self._on_reload)
        button_row.addWidget(self._reload_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        editor_label = QLabel("Edit selected file:")
        layout.addWidget(editor_label)
        self._editor = QPlainTextEdit()
        self._editor.setObjectName("CodeBlockBody")
        self._editor.setPlaceholderText("Select a tool above to view / edit its source.")
        layout.addWidget(self._editor, 1)

        save_row = QHBoxLayout()
        save_row.addStretch(1)
        self._save_btn = QPushButton("Save edits")
        self._save_btn.clicked.connect(self._on_save_edits)
        save_row.addWidget(self._save_btn)
        layout.addLayout(save_row)

        self._refresh()

    # ---- internal ---------------------------------------------------------

    def _current_path(self) -> Path | None:
        item = self._list.currentItem()
        if item is None:
            return None
        return Path(item.data(Qt.ItemDataRole.UserRole))

    def _refresh(self) -> None:
        self._list.clear()
        report = load_user_tools(self._directory)
        # Group tools by their source file.
        tools_by_file: dict[str, list[str]] = {}
        for loaded in report.loaded:
            tools_by_file.setdefault(str(loaded.source_path), []).append(loaded.tool.name)
        errors_by_file = {str(err.source_path): err.message for err in report.errors}
        for py in list_user_tool_files(self._directory):
            names = tools_by_file.get(str(py), [])
            err = errors_by_file.get(str(py))
            if names:
                label = f"{py.name}  —  {', '.join(names)}"
            elif err:
                label = f"{py.name}  —  ⚠ {err}"
            else:
                label = f"{py.name}  —  (no @tool functions)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(py))
            self._list.addItem(item)
        self._editor.clear()
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _on_selection(self) -> None:
        path = self._current_path()
        if path is None or not path.exists():
            self._editor.clear()
            return
        try:
            self._editor.setPlainText(path.read_text(encoding="utf-8"))
        except OSError as exc:
            self._editor.setPlainText(f"# could not read {path}: {exc}")

    def _on_new(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "New user tool",
            "Tool name (alphanumeric + underscore):",
        )
        if not ok or not name.strip():
            return
        safe = "".join(c for c in name.strip() if c.isalnum() or c == "_")
        if not safe:
            QMessageBox.warning(self, "Invalid name", "Tool name must be alphanumeric.")
            return
        description, ok = QInputDialog.getText(
            self, "Description", "One-line description of the tool:"
        )
        if not ok:
            description = ""
        path = Path(self._directory) / f"{safe}.py"
        if path.exists():
            QMessageBox.warning(self, "Already exists", f"{path.name} already exists.")
            return
        path.write_text(starter_template(safe, description), encoding="utf-8")
        self._refresh()
        self._select_path(path)
        self.tools_changed.emit()

    def _on_upload(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Python tool file",
            "",
            "Python files (*.py)",
        )
        if not chosen:
            return
        src = Path(chosen)
        dest = Path(self._directory) / src.name
        if dest.exists():
            confirm = QMessageBox.question(
                self,
                "Overwrite?",
                f"{dest.name} already exists. Overwrite?",
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        self._refresh()
        self._select_path(dest)
        self.tools_changed.emit()

    def _on_open(self) -> None:
        path = self._current_path()
        if path is None:
            return
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Open failed", str(exc))

    def _on_delete(self) -> None:
        path = self._current_path()
        if path is None:
            return
        confirm = QMessageBox.question(
            self,
            "Delete tool",
            f"Delete {path.name}? This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            path.unlink()
        except OSError as exc:
            QMessageBox.warning(self, "Delete failed", str(exc))
            return
        self._refresh()
        self.tools_changed.emit()

    def _on_reload(self) -> None:
        self._refresh()
        self.tools_changed.emit()

    def _on_save_edits(self) -> None:
        path = self._current_path()
        if path is None:
            QMessageBox.information(self, "No file selected", "Select a tool file above first.")
            return
        try:
            path.write_text(self._editor.toPlainText(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._refresh()
        self._select_path(path)
        self.tools_changed.emit()

    def _select_path(self, target: Path) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if Path(item.data(Qt.ItemDataRole.UserRole)) == target:
                self._list.setCurrentRow(i)
                return


class SettingsDialog(QDialog):
    """Multi-tab settings dialog. Returns ``QDialog.Accepted`` on save."""

    settings_saved = Signal(object)  # emits the new Settings instance
    knowledge_changed = Signal()  # re-emitted from the Knowledge tab
    tools_changed = Signal()  # re-emitted from the Tools tab

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(680, 560)
        self._settings = Settings.load()
        self._tabs = QTabWidget()

        self.general_tab = _GeneralTab(self._settings)
        self.mcp_tab = _MCPTab()
        self.github_tab = _GitHubTab()
        self.backends_tab = _BackendsTab()
        self.knowledge_tab = _KnowledgeTab()
        self.tools_tab = _ToolsTab()
        self.figma_tab = _FigmaTab()
        self.appearance_tab = _AppearanceTab(self._settings)
        # Re-emit so the main window can refresh the running agent's prompt.
        self.knowledge_tab.knowledge_changed.connect(self.knowledge_changed)
        self.tools_tab.tools_changed.connect(self.tools_changed)

        for name, widget, icon_name in (
            ("General", self.general_tab, "general"),
            ("MCP", self.mcp_tab, "mcp"),
            ("GitHub", self.github_tab, "github"),
            ("Backends", self.backends_tab, "backend"),
            ("Knowledge", self.knowledge_tab, "knowledge"),
            ("Tools", self.tools_tab, "tools"),
            ("Figma", self.figma_tab, "figma"),
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
        self.knowledge_tab.apply()
        self.figma_tab.apply()
        self.settings_saved.emit(self._settings)
        self.accept()

    def current_settings(self) -> Settings:
        return self._settings
