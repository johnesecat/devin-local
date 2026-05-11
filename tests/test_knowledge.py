"""Tests for the knowledge store."""

from __future__ import annotations

from pathlib import Path

from devin_local.knowledge.store import KnowledgeStore


def test_add_and_search_roundtrip(tmp_path: Path):
    store = KnowledgeStore.open(tmp_path / "store.jsonl")
    store.add("Build", "Run cargo build to compile the rust crate.", scope="rust, build")
    store.add(
        "Pytest",
        "Run python -m pytest to execute the python test suite.",
        scope="python, testing",
        tags=["python"],
    )
    results = store.search("how do I run python tests?")
    assert results
    assert results[0].title == "Pytest"


def test_persistence(tmp_path: Path):
    path = tmp_path / "store.jsonl"
    store1 = KnowledgeStore.open(path)
    store1.add("X", "hello")
    store2 = KnowledgeStore.open(path)
    assert len(store2.notes) == 1
    assert store2.notes[0].title == "X"


def test_remove(tmp_path: Path):
    store = KnowledgeStore.open(tmp_path / "store.jsonl")
    note = store.add("Y", "body")
    assert store.remove(note.id)
    assert not store.notes


def test_search_empty_store_returns_empty(tmp_path: Path):
    store = KnowledgeStore.open(tmp_path / "store.jsonl")
    assert store.search("anything") == []
