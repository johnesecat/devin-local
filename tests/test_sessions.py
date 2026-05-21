"""Per-session storage layer."""

from __future__ import annotations

import json

from devin_local.sessions import SessionInfo, SessionManager


def test_create_session_writes_json_and_jsonl(tmp_path):
    mgr = SessionManager.for_workspace(tmp_path)
    info = mgr.create(name="Hello", model="llama3.1:8b", knowledge_dir=str(tmp_path / "kb"))
    assert info.id
    cfg_path = mgr.root / f"{info.id}.json"
    txn_path = mgr.root / f"{info.id}.jsonl"
    assert cfg_path.exists()
    assert txn_path.exists()
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["name"] == "Hello"
    assert payload["model"] == "llama3.1:8b"
    assert payload["knowledge_dir"] == str(tmp_path / "kb")


def test_list_orders_by_last_used(tmp_path):
    mgr = SessionManager.for_workspace(tmp_path)
    a = mgr.create(name="A")
    mgr.create(name="B")
    mgr.touch(a.id)  # bump a's last_used_ts
    listed = mgr.list()
    names = [s.name for s in listed]
    assert names[0] == "A"
    assert "B" in names


def test_save_updates_existing_session(tmp_path):
    mgr = SessionManager.for_workspace(tmp_path)
    info = mgr.create(name="Original")
    info.system_prompt_override = "Be very brief."
    info.knowledge_dir = str(tmp_path / "kbX")
    mgr.save(info)
    loaded = mgr.load(info.id)
    assert loaded is not None
    assert loaded.system_prompt_override == "Be very brief."
    assert loaded.knowledge_dir == str(tmp_path / "kbX")


def test_delete_removes_config_and_transcript(tmp_path):
    mgr = SessionManager.for_workspace(tmp_path)
    info = mgr.create(name="Doomed")
    assert (mgr.root / f"{info.id}.jsonl").exists()
    assert (mgr.root / f"{info.id}.json").exists()
    mgr.delete(info.id)
    assert not (mgr.root / f"{info.id}.jsonl").exists()
    assert not (mgr.root / f"{info.id}.json").exists()
    assert mgr.load(info.id) is None


def test_session_info_new_generates_unique_ids():
    a = SessionInfo.new(name="X")
    b = SessionInfo.new(name="X")
    assert a.id != b.id
