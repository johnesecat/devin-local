"""Per-session settings dialog.

Opened from the gear button on each session row in the sidebar. Edits a
single :class:`SessionInfo` instance; the main window saves it to disk and
applies the changes to the running agent (if this is the active session)
when the dialog emits :attr:`session_saved`.

Tabs:

- **Identity** — session name, free-form notes.
- **Agent** — per-session system-prompt override (with "Reset to global
  default" to clear it).
- **Knowledge** — per-session knowledge directory picker; toggles a
  "Browse" button to choose an existing folder.
- **Behavior** — per-session model, backend, OBLITERATUS toggle, planning
  toggle, parallel tool calls, max iterations, temperature.

Each per-session field can be reset to "inherit from global Settings" by
clearing the checkbox to the left of it. The dialog is Qt-only but does
not require a running event loop — tests can instantiate it offscreen.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from PySide6.QtCore import Qt, Signal
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
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from devin_local.gui.model_selector import LibraryBrowserDialog, ModelComboBox
from devin_local.gui.ollama_model_service import OllamaModelService
from devin_local.sessions import SessionInfo
from devin_local.settings import Settings


def _safe_list_installed(host: str) -> list[str]:
    """Best-effort list of installed Ollama models; ``[]`` on failure."""
    try:
        service = OllamaModelService(host=host)
        try:
            return [m.name for m in service.list_installed()]
        finally:
            service.close()
    except Exception:  # noqa: BLE001
        return []


class _OverrideRow(QWidget):
    """A small helper: checkbox + editor; unchecked => inherit (None).

    The editor is provided by the caller; this widget wires the checkbox to
    enable/disable the editor and exposes ``is_overridden`` / ``set_overridden``.
    """

    def __init__(
        self,
        label: str,
        editor: QWidget,
        overridden: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._cb = QCheckBox(label)
        self._cb.setToolTip(
            "Tick to override the global default for this session only. "
            "Untick to inherit from main Settings."
        )
        self._cb.setChecked(overridden)
        self._editor = editor
        self._editor.setEnabled(overridden)
        self._cb.toggled.connect(self._editor.setEnabled)
        row.addWidget(self._cb, 0)
        row.addWidget(self._editor, 1)

    def is_overridden(self) -> bool:
        return self._cb.isChecked()

    def set_overridden(self, on: bool) -> None:
        self._cb.setChecked(on)
        self._editor.setEnabled(on)


class _IdentityTab(QWidget):
    def __init__(self, info: SessionInfo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.name = QLineEdit(info.name)
        form.addRow("Session name", self.name)
        self.notes = QPlainTextEdit(info.notes)
        self.notes.setPlaceholderText("Free-form notes for this session (not sent to the model).")
        self.notes.setMinimumHeight(100)
        form.addRow("Notes", self.notes)


class _AgentTab(QWidget):
    """System-prompt override editor for the session."""

    def __init__(
        self,
        info: SessionInfo,
        global_settings: Settings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._global = global_settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(
            QLabel(
                "Per-session system prompt override. Leave EMPTY to use the "
                "global default. Anything you put here is PREPENDED to the "
                "rest of the assembled prompt; the agent's identity, tool "
                "discipline, and safety rules still apply."
            )
        )

        self.editor = QPlainTextEdit(info.system_prompt_override)
        self.editor.setPlaceholderText(
            "Example: 'You are reviewing legal contracts. Always cite "
            "clause numbers when answering.'"
        )
        self.editor.setMinimumHeight(220)
        layout.addWidget(self.editor, 1)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Reset to global default")
        clear_btn.clicked.connect(self.editor.clear)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)


class _KnowledgeTab(QWidget):
    """Per-session knowledge directory picker."""

    def __init__(self, info: SessionInfo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(
            QLabel(
                "Pick a directory containing knowledge files (.md / .txt) "
                "for THIS session. Files in that directory are NOT embedded "
                "in the system prompt — only a tiny manifest (path + title) "
                "is. The agent loads bodies on demand via the "
                "`knowledge_search` and `knowledge_read` tools. Different "
                "sessions can point at different directories."
            )
        )

        row = QHBoxLayout()
        self.path_edit = QLineEdit(info.knowledge_dir or "")
        self.path_edit.setPlaceholderText("(empty = no knowledge directory for this session)")
        row.addWidget(self.path_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        row.addWidget(browse)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.path_edit.clear)
        row.addWidget(clear)
        layout.addLayout(row)

        layout.addStretch(1)

    def _on_browse(self) -> None:
        current = self.path_edit.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Knowledge directory", current)
        if chosen:
            self.path_edit.setText(chosen)


class _BehaviorTab(QWidget):
    """Per-session model / backend / behavior overrides."""

    def __init__(
        self,
        info: SessionInfo,
        global_settings: Settings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._global_settings = global_settings
        installed = _safe_list_installed(global_settings.general.ollama_host)
        self._model_edit = ModelComboBox(
            installed=installed,
            current=info.model or global_settings.general.default_model,
            allow_browse=True,
        )
        self._model_edit.browse_requested.connect(self._on_browse_models)
        self._model_row = _OverrideRow("Override model", self._model_edit, info.model is not None)
        form.addRow(self._model_row)

        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["ollama", "layered", "hf"])
        current_backend = info.backend or global_settings.general.default_backend
        idx = self._backend_combo.findText(current_backend)
        if idx >= 0:
            self._backend_combo.setCurrentIndex(idx)
        self._backend_row = _OverrideRow(
            "Override backend", self._backend_combo, info.backend is not None
        )
        form.addRow(self._backend_row)

        self._obli_cb = QCheckBox("Include OBLITERATUS directive in this session's prompt")
        self._obli_cb.setChecked(
            info.enable_obliteratus
            if info.enable_obliteratus is not None
            else global_settings.general.enable_obliteratus
        )
        self._obli_row = _OverrideRow(
            "Override OBLITERATUS",
            self._obli_cb,
            info.enable_obliteratus is not None,
        )
        form.addRow(self._obli_row)

        self._plan_cb = QCheckBox("Plan-first workflow")
        self._plan_cb.setChecked(
            info.enable_planning
            if info.enable_planning is not None
            else global_settings.general.enable_planning
        )
        self._plan_row = _OverrideRow(
            "Override planning",
            self._plan_cb,
            info.enable_planning is not None,
        )
        form.addRow(self._plan_row)

        self._parallel_cb = QCheckBox("Dispatch multiple tool calls in parallel")
        self._parallel_cb.setChecked(
            info.parallel_tool_calls
            if info.parallel_tool_calls is not None
            else global_settings.general.parallel_tool_calls
        )
        self._parallel_row = _OverrideRow(
            "Override parallel tools",
            self._parallel_cb,
            info.parallel_tool_calls is not None,
        )
        form.addRow(self._parallel_row)

        self._verbose_cb = QCheckBox("Verbose system prompt (~13 KB; slower but more guidance)")
        self._verbose_cb.setChecked(
            info.verbose_prompt
            if info.verbose_prompt is not None
            else global_settings.general.verbose_prompt
        )
        self._verbose_row = _OverrideRow(
            "Override prompt size",
            self._verbose_cb,
            info.verbose_prompt is not None,
        )
        form.addRow(self._verbose_row)

        self._iters_spin = QSpinBox()
        self._iters_spin.setRange(1, 200)
        self._iters_spin.setValue(info.max_iterations or 20)
        self._iters_row = _OverrideRow(
            "Override max_iterations",
            self._iters_spin,
            info.max_iterations is not None,
        )
        form.addRow(self._iters_row)

        self._temp_spin = QDoubleSpinBox()
        self._temp_spin.setRange(0.0, 2.0)
        self._temp_spin.setSingleStep(0.05)
        self._temp_spin.setDecimals(2)
        self._temp_spin.setValue(info.temperature if info.temperature is not None else 0.2)
        self._temp_row = _OverrideRow(
            "Override temperature",
            self._temp_spin,
            info.temperature is not None,
        )
        form.addRow(self._temp_row)

    # Public accessors used by the dialog when saving.
    def _on_browse_models(self) -> None:
        host = self._global_settings.general.ollama_host
        service = OllamaModelService(host=host)
        try:
            installed = service.list_installed()
        except Exception:  # noqa: BLE001
            installed = []
        dialog = LibraryBrowserDialog(service, installed, service.list_library(), parent=self)
        dialog.pulled.connect(self._on_model_pulled)
        dialog.exec()
        service.close()

    def _on_model_pulled(self, name: str) -> None:
        host = self._global_settings.general.ollama_host
        installed = _safe_list_installed(host)
        if name not in installed:
            installed.append(name)
        self._model_edit.populate(sorted(installed), current=name)
        self._model_row.set_overridden(True)

    def values(self) -> dict[str, object]:
        return {
            "model": self._model_edit.current_model() if self._model_row.is_overridden() else None,
            "backend": (
                self._backend_combo.currentText() if self._backend_row.is_overridden() else None
            ),
            "enable_obliteratus": (
                self._obli_cb.isChecked() if self._obli_row.is_overridden() else None
            ),
            "enable_planning": (
                self._plan_cb.isChecked() if self._plan_row.is_overridden() else None
            ),
            "parallel_tool_calls": (
                self._parallel_cb.isChecked() if self._parallel_row.is_overridden() else None
            ),
            "verbose_prompt": (
                self._verbose_cb.isChecked() if self._verbose_row.is_overridden() else None
            ),
            "max_iterations": (
                int(self._iters_spin.value()) if self._iters_row.is_overridden() else None
            ),
            "temperature": (
                float(self._temp_spin.value()) if self._temp_row.is_overridden() else None
            ),
        }


class PerSessionSettingsDialog(QDialog):
    """Dialog window editing a single :class:`SessionInfo`."""

    session_saved = Signal(object)  # emits the updated SessionInfo

    def __init__(
        self,
        info: SessionInfo,
        global_settings: Settings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Agent settings — {info.name}")
        self.setMinimumSize(720, 540)
        self._info = replace(info)
        self._global = global_settings or Settings.load()

        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        self._identity_tab = _IdentityTab(info)
        self._agent_tab = _AgentTab(info, self._global)
        self._knowledge_tab = _KnowledgeTab(info)
        self._behavior_tab = _BehaviorTab(info, self._global)
        self._tabs.addTab(self._identity_tab, "Identity")
        self._tabs.addTab(self._agent_tab, "Agent")
        self._tabs.addTab(self._knowledge_tab, "Knowledge")
        self._tabs.addTab(self._behavior_tab, "Behavior")
        layout.addWidget(self._tabs, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, 0)

    def _on_save(self) -> None:
        updates: dict[str, object] = {
            "name": self._identity_tab.name.text().strip() or self._info.name,
            "notes": self._identity_tab.notes.toPlainText(),
            "system_prompt_override": self._agent_tab.editor.toPlainText().rstrip(),
            "knowledge_dir": self._knowledge_tab.path_edit.text().strip() or None,
        }
        updates.update(self._behavior_tab.values())
        # Apply to a copy.
        merged_dict = asdict(self._info)
        merged_dict.update(updates)
        merged = SessionInfo(**merged_dict)
        self._info = merged
        self.session_saved.emit(merged)
        self.accept()

    def session(self) -> SessionInfo:
        return self._info
