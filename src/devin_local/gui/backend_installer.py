"""Backend installer: install optional pip extras (`layered`, `hf`) on demand.

When the user picks a backend whose Python deps aren't installed, the GUI
opens an :class:`InstallBackendDialog`. The dialog spawns ``pip install -e
.[extra]`` (or ``pip install devin-local[extra]`` if the package isn't
editable-installed) in a background QThread and pipes stdout/stderr into a
live log view, with cancel.

The CLI also exposes ``devin-local install layered|hf|gui|all`` which calls
the same :func:`install_extras` plumbing for headless setups.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from devin_local.installer import (
    BACKEND_EXTRAS,
    backend_dep_probe,
    pip_command,
)

log = logging.getLogger(__name__)

__all__ = [
    "BACKEND_EXTRAS",
    "InstallBackendDialog",
    "backend_dep_probe",
    "pip_command",
]


class _InstallWorker(QObject):
    """Runs ``pip install`` and streams stdout/stderr line-by-line."""

    line = Signal(str)
    finished = Signal(int, str)  # return_code, message

    def __init__(self, argv: list[str], env: dict[str, str] | None = None) -> None:
        super().__init__()
        self._argv = argv
        self._env = env or {}
        self._proc: subprocess.Popen[str] | None = None
        self._cancelled = False

    @Slot()
    def cancel(self) -> None:
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.terminate()

    @Slot()
    def run(self) -> None:
        full_env = os.environ.copy()
        # Don't let pip's progress redraw mangle the log.
        full_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        full_env["PIP_NO_INPUT"] = "1"
        full_env["PYTHONUNBUFFERED"] = "1"
        full_env.update(self._env)
        self.line.emit(f"$ {' '.join(self._argv)}")
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - argv is constructed in-code
                self._argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=full_env,
            )
        except OSError as exc:
            self.finished.emit(-1, f"failed to spawn pip: {exc}")
            return
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            if self._cancelled:
                break
            self.line.emit(raw.rstrip("\n"))
        rc = self._proc.wait()
        if self._cancelled:
            self.finished.emit(130, "cancelled")
        elif rc == 0:
            self.finished.emit(0, "ok")
        else:
            self.finished.emit(rc, f"pip exited with code {rc}")


class InstallBackendDialog(QDialog):
    """Dialog that installs `extras` via pip and shows live log + status."""

    installed = Signal(str)  # backend name, after a successful install

    def __init__(
        self,
        backend: str,
        extras: tuple[str, ...] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._backend = backend
        self._extras = extras or BACKEND_EXTRAS.get(backend, (backend,))
        self.setWindowTitle(f"Install backend: {backend}")
        self.resize(720, 520)
        self._worker: _InstallWorker | None = None
        self._thread: QThread | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        spec = ",".join(self._extras)
        intro = QLabel(
            f"The <b>{backend}</b> backend requires additional Python packages "
            f"(<code>devin-local[{spec}]</code>). This dialog will pip-install them "
            "into the current Python environment. Expect ~1\u20132 GB of downloads "
            "(torch + transformers + accelerate)."
        )
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        cmd_label = QLabel("<code>" + " ".join(pip_command(self._extras)) + "</code>")
        cmd_label.setTextFormat(Qt.TextFormat.RichText)
        cmd_label.setWordWrap(True)
        cmd_label.setObjectName("Muted")
        layout.addWidget(cmd_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setObjectName("CodeBlockBody")
        layout.addWidget(self._log, 1)

        self._status = QLabel("")
        self._status.setObjectName("Muted")
        layout.addWidget(self._status)

        buttons = QDialogButtonBox()
        self._install_btn = QPushButton("Install")
        self._install_btn.setObjectName("Primary")
        self._install_btn.clicked.connect(self._start)
        buttons.addButton(self._install_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel)
        buttons.addButton(self._cancel_btn, QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(buttons)

    # ---------- handlers ----------

    def _start(self) -> None:
        if self._thread is not None:
            return
        argv = pip_command(self._extras)
        self._install_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._status.setText("Installing\u2026")
        worker = _InstallWorker(argv)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.line.connect(self._on_line)
        worker.finished.connect(self._on_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        self._worker = worker
        self._thread = thread

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("Cancelling\u2026")
        else:
            self.reject()

    def _on_line(self, text: str) -> None:
        self._log.appendPlainText(text)

    def _on_finished(self, rc: int, message: str) -> None:
        self._worker = None
        self._thread = None
        self._progress.setVisible(False)
        self._install_btn.setEnabled(True)
        if rc == 0:
            self._status.setText("Installed. You can close this dialog and switch backends.")
            self.installed.emit(self._backend)
        else:
            self._status.setText(f"Failed: {message}")
