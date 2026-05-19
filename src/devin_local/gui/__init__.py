"""PySide6 desktop GUI for devin-local.

The GUI is an optional install. `pip install devin-local[gui]` pulls in
PySide6. Entry points:

  - ``devin-local gui`` (subcommand) \u2014 has a console window on Windows.
  - ``devin-local-gui`` (script) \u2014 has a console window.
  - ``devin-local-gui-app`` (gui-script) \u2014 *no* console window on Windows.

All three call :func:`devin_local.gui.app.launch_gui`.
"""

__all__ = ["launch_gui"]


def __getattr__(name: str):  # noqa: ANN202
    if name == "launch_gui":
        from devin_local.gui.app import launch_gui

        return launch_gui
    raise AttributeError(name)
