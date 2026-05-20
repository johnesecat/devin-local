"""In-app drag-and-drop UI Creator.

The UI Creator lets the operator design custom panels and dialogs *inside*
devin-local — drop widgets onto a canvas, configure their properties,
wire actions (send-to-agent, run-shell, open-file, …), and save the
result as portable JSON that can be mounted as a real, fully-functional
widget anywhere in the app.

Public surface
--------------

- :class:`UiDocument`  — the JSON-serializable layout document.
- :class:`UiElement`   — one widget on the canvas.
- :class:`UiAction`    — a named action attached to an event.
- :class:`UiRuntime`   — mounts a :class:`UiDocument` as a live Qt widget,
                         dispatching events to the configured actions.
- :class:`UiCreatorDialog` — the designer dialog the GUI exposes.

The serialization format is stable and forward-compatible: every element
carries an explicit ``schema_version`` so future readers can migrate
older saves.
"""

from devin_local.ui_creator.document import (
    ELEMENT_TYPES,
    SCHEMA_VERSION,
    UiAction,
    UiDocument,
    UiElement,
)
from devin_local.ui_creator.runtime import UiRuntime, UiRuntimeContext

__all__ = [
    "ELEMENT_TYPES",
    "SCHEMA_VERSION",
    "UiAction",
    "UiDocument",
    "UiElement",
    "UiRuntime",
    "UiRuntimeContext",
]


def __getattr__(name: str):
    # Lazy-load the dialog so importing the package never pulls in
    # PySide6 unless the GUI actually opens the designer.
    if name == "UiCreatorDialog":
        from devin_local.ui_creator.dialog import UiCreatorDialog as _UCD

        return _UCD
    raise AttributeError(name)
