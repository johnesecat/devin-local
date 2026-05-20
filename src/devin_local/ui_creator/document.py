"""UI Creator document model.

This is the *headless* representation: no Qt, no PySide6. Both the GUI
designer (``ui_creator/dialog.py``) and the runtime renderer
(``ui_creator/runtime.py``) read and write the same :class:`UiDocument`
structure, which means you can author a document programmatically (or by
hand-editing JSON) and mount it without ever opening the designer.

The document is **fully self-describing**: every element carries its
geometry, style, and action bindings inline. No external references, no
implicit lookups. This makes the JSON easy to read, easy to diff, and
easy to port between machines.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Bumped whenever the schema gains a non-backward-compatible field.
SCHEMA_VERSION = 1

# Element type registry. Each entry: (type_id, label, default size, palette icon hint).
# The runtime maps these to real Qt widgets in ``runtime.py``.
ELEMENT_TYPES: dict[str, dict[str, Any]] = {
    "label": {
        "label": "Label",
        "default_size": (200, 32),
        "props": {"text": "Label", "font_pt": 11, "bold": False, "italic": False},
    },
    "heading": {
        "label": "Heading",
        "default_size": (260, 40),
        "props": {"text": "Heading", "font_pt": 18, "bold": True, "italic": False},
    },
    "button": {
        "label": "Button",
        "default_size": (160, 38),
        "props": {"text": "Click me", "primary": True},
    },
    "line_edit": {
        "label": "Text input",
        "default_size": (240, 34),
        "props": {"placeholder": "Type here\u2026", "default": ""},
    },
    "text_area": {
        "label": "Text area",
        "default_size": (320, 140),
        "props": {"placeholder": "Multi-line text\u2026", "default": ""},
    },
    "combo": {
        "label": "Combo box",
        "default_size": (200, 32),
        "props": {"items": ["Option 1", "Option 2", "Option 3"], "default": "Option 1"},
    },
    "checkbox": {
        "label": "Checkbox",
        "default_size": (180, 28),
        "props": {"text": "Enable feature", "checked": False},
    },
    "file_picker": {
        "label": "File picker",
        "default_size": (320, 34),
        "props": {"placeholder": "(no file selected)", "mode": "open"},
    },
    "image": {
        "label": "Image",
        "default_size": (200, 140),
        "props": {"path": "", "fit": "contain"},
    },
    "markdown": {
        "label": "Markdown",
        "default_size": (360, 180),
        "props": {"source": "# Heading\n\nSupports **bold**, *italic*, `code`."},
    },
    "separator": {
        "label": "Separator",
        "default_size": (320, 8),
        "props": {"orientation": "horizontal"},
    },
    "chat_send": {
        "label": "Send to agent",
        "default_size": (200, 40),
        "props": {"text": "Send to agent", "template": "{value}"},
    },
}


@dataclass
class UiAction:
    """A named action attached to an element event.

    ``kind`` is one of:

    - ``"agent.send"``       — submit ``template`` (substituted with element
      values) to the agent as a user message.
    - ``"agent.run_tool"``   — invoke a registered tool by ``tool`` name with
      ``args`` (which may interpolate element values).
    - ``"shell.run"``        — run ``command`` in a subprocess; capture its
      stdout/stderr into a target element.
    - ``"set_var"``          — assign an element's value into a named runtime
      variable for later substitution.
    - ``"python"``           — evaluate a small Python expression (sandboxed
      to the runtime's globals).
    - ``"open_file"``        — open ``path`` in the workspace via read_file.
    """

    kind: str
    template: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    command: str = ""
    target: str = ""  # element id to write the result into
    var: str = ""
    expr: str = ""
    path: str = ""


@dataclass
class UiElement:
    """One widget on the canvas."""

    id: str
    type: str  # key in ELEMENT_TYPES
    x: int = 16
    y: int = 16
    width: int = 200
    height: int = 32
    name: str = ""  # operator-supplied label / variable name
    props: dict[str, Any] = field(default_factory=dict)
    # Event name -> list of actions. Supported events depend on type;
    # ``"clicked"`` for buttons, ``"changed"`` for inputs, etc.
    on: dict[str, list[UiAction]] = field(default_factory=dict)

    @classmethod
    def new(cls, type_: str, *, x: int = 16, y: int = 16) -> UiElement:
        if type_ not in ELEMENT_TYPES:
            raise ValueError(f"Unknown element type {type_!r}")
        spec = ELEMENT_TYPES[type_]
        w, h = spec["default_size"]
        return cls(
            id=f"el_{uuid.uuid4().hex[:8]}",
            type=type_,
            x=x,
            y=y,
            width=int(w),
            height=int(h),
            name="",
            props=dict(spec["props"]),
            on={},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "name": self.name,
            "props": self.props,
            "on": {event: [asdict(a) for a in actions] for event, actions in self.on.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> UiElement:
        on_raw = raw.get("on") or {}
        events: dict[str, list[UiAction]] = {}
        for event, actions in on_raw.items():
            events[event] = [UiAction(**a) for a in actions if isinstance(a, dict)]
        return cls(
            id=str(raw.get("id") or f"el_{uuid.uuid4().hex[:8]}"),
            type=str(raw.get("type") or "label"),
            x=int(raw.get("x", 16) or 16),
            y=int(raw.get("y", 16) or 16),
            width=int(raw.get("width", 200) or 200),
            height=int(raw.get("height", 32) or 32),
            name=str(raw.get("name", "") or ""),
            props=dict(raw.get("props") or {}),
            on=events,
        )


@dataclass
class UiDocument:
    """A saved UI created in the designer."""

    id: str
    name: str = "Custom UI"
    schema_version: int = SCHEMA_VERSION
    canvas_width: int = 720
    canvas_height: int = 480
    background: str = "#1a1b26"
    elements: list[UiElement] = field(default_factory=list)
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)
    notes: str = ""

    @classmethod
    def new(cls, name: str = "Custom UI") -> UiDocument:
        return cls(
            id=f"ui_{int(time.time())}_{uuid.uuid4().hex[:6]}",
            name=name,
        )

    def add(self, element: UiElement) -> UiElement:
        self.elements.append(element)
        self.updated_ts = time.time()
        return element

    def remove(self, element_id: str) -> bool:
        before = len(self.elements)
        self.elements = [e for e in self.elements if e.id != element_id]
        if len(self.elements) < before:
            self.updated_ts = time.time()
            return True
        return False

    def get(self, element_id: str) -> UiElement | None:
        for e in self.elements:
            if e.id == element_id:
                return e
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "schema_version": self.schema_version,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "background": self.background,
            "elements": [e.to_dict() for e in self.elements],
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> UiDocument:
        elements = [UiElement.from_dict(e) for e in raw.get("elements") or []]
        return cls(
            id=str(raw.get("id") or f"ui_{int(time.time())}_{uuid.uuid4().hex[:6]}"),
            name=str(raw.get("name") or "Custom UI"),
            schema_version=int(raw.get("schema_version") or SCHEMA_VERSION),
            canvas_width=int(raw.get("canvas_width", 720) or 720),
            canvas_height=int(raw.get("canvas_height", 480) or 480),
            background=str(raw.get("background", "#1a1b26") or "#1a1b26"),
            elements=elements,
            created_ts=float(raw.get("created_ts") or time.time()),
            updated_ts=float(raw.get("updated_ts") or time.time()),
            notes=str(raw.get("notes", "") or ""),
        )

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.id}.json"
        self.updated_ts = time.time()
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> UiDocument:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)
