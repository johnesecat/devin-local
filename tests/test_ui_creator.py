"""UI Creator document model + runtime."""

from __future__ import annotations

import os

import pytest

from devin_local.ui_creator import (
    ELEMENT_TYPES,
    UiAction,
    UiDocument,
    UiElement,
)


def test_element_types_have_required_keys():
    for type_id, spec in ELEMENT_TYPES.items():
        assert "label" in spec, type_id
        assert "default_size" in spec, type_id
        assert "props" in spec, type_id


def test_element_new_uses_defaults():
    el = UiElement.new("button", x=10, y=20)
    assert el.type == "button"
    assert el.x == 10
    assert el.y == 20
    assert el.props["text"] == "Click me"


def test_element_new_rejects_unknown_type():
    with pytest.raises(ValueError):
        UiElement.new("not-a-real-type")


def test_document_add_remove_and_get():
    doc = UiDocument.new("Demo")
    el = UiElement.new("label")
    doc.add(el)
    assert doc.get(el.id) is el
    assert doc.remove(el.id) is True
    assert doc.get(el.id) is None


def test_document_save_load_roundtrip(tmp_path):
    doc = UiDocument.new("Roundtrip")
    btn = UiElement.new("button")
    btn.name = "send"
    btn.on["clicked"] = [UiAction(kind="agent.send", template="hi {q}")]
    doc.add(btn)
    inp = UiElement.new("line_edit")
    inp.name = "q"
    doc.add(inp)
    path = doc.save(tmp_path)
    assert path.exists()
    loaded = UiDocument.load(path)
    assert loaded.name == "Roundtrip"
    assert len(loaded.elements) == 2
    assert loaded.elements[0].on["clicked"][0].kind == "agent.send"
    assert loaded.elements[0].on["clicked"][0].template == "hi {q}"


@pytest.mark.skipif(
    not os.environ.get("QT_QPA_PLATFORM", "").startswith("offscreen")
    and "DISPLAY" not in os.environ,
    reason="Qt runtime smoke needs an offscreen platform",
)
def test_runtime_fires_send_to_agent_action(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from devin_local.ui_creator.runtime import UiRuntime, UiRuntimeContext

    app = QApplication.instance() or QApplication([])  # noqa: F841 - needed for QWidget
    doc = UiDocument.new("Wire")
    btn = UiElement.new("button", x=0, y=0)
    btn.name = "send_btn"
    btn.on["clicked"] = [UiAction(kind="agent.send", template="hello {q}")]
    inp = UiElement.new("line_edit", x=0, y=50)
    inp.name = "q"
    doc.add(btn)
    doc.add(inp)

    captured: list[str] = []
    rt = UiRuntime(doc, UiRuntimeContext(send_to_agent=captured.append))
    rt._widgets[inp.id].setText("world")
    rt._fire(btn.id, "clicked")
    assert captured == ["hello world"]


@pytest.mark.skipif(
    not os.environ.get("QT_QPA_PLATFORM", "").startswith("offscreen")
    and "DISPLAY" not in os.environ,
    reason="Qt runtime smoke needs an offscreen platform",
)
def test_runtime_set_var_and_python_action(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from devin_local.ui_creator.runtime import UiRuntime, UiRuntimeContext

    app = QApplication.instance() or QApplication([])  # noqa: F841
    doc = UiDocument.new("Vars")
    text_in = UiElement.new("line_edit", x=0, y=0)
    text_in.name = "user"
    out = UiElement.new("label", x=0, y=80)
    out.name = "out"
    btn = UiElement.new("button", x=0, y=120)
    btn.on["clicked"] = [
        UiAction(kind="set_var", var="who", template="{user}"),
        UiAction(kind="python", expr="'hi ' + values['who']", target="out"),
    ]
    doc.add(text_in)
    doc.add(out)
    doc.add(btn)

    rt = UiRuntime(doc, UiRuntimeContext())
    rt._widgets[text_in.id].setText("jacob")
    rt._fire(btn.id, "clicked")
    assert rt._widgets[out.id].text() == "hi jacob"
