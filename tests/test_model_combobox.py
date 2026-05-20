"""ModelComboBox + LibraryBrowserDialog Hugging Face tab tests.

The widgets run on offscreen Qt so these tests work in CI. The HF catalog
is fully mocked via ``respx`` against the real ``httpx.Client`` instance —
no real network is made.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import respx

pytest.importorskip("PySide6")

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from devin_local.gui.model_selector import _BROWSE_SENTINEL, ModelComboBox
from devin_local.gui.ollama_model_service import (
    HuggingFaceCatalog,
    HuggingFaceGGUFEntry,
    HuggingFaceGGUFFile,
    _quant_from_filename,
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_combobox_populate_lists_installed_models() -> None:
    _app()
    cb = ModelComboBox(installed=["llama3.2:3b", "qwen2.5:7b"], current="llama3.2:3b")
    items = [cb.itemData(i) for i in range(cb.count())]
    assert "llama3.2:3b" in items
    assert "qwen2.5:7b" in items
    assert _BROWSE_SENTINEL in items  # default allow_browse=True
    assert cb.current_model() == "llama3.2:3b"


def test_combobox_marks_unselected_current_as_not_installed() -> None:
    _app()
    cb = ModelComboBox(installed=["llama3.2:3b"], current="llama3.1:8b")
    found_not_installed = False
    for i in range(cb.count()):
        if "not installed" in cb.itemText(i):
            found_not_installed = True
            assert cb.itemData(i) == "llama3.1:8b"
    assert found_not_installed
    assert cb.current_model() == "llama3.1:8b"


def test_combobox_browse_sentinel_emits_signal_and_does_not_change_value() -> None:
    _app()
    cb = ModelComboBox(installed=["a", "b"], current="a", allow_browse=True)
    triggered: list[bool] = []
    cb.browse_requested.connect(lambda: triggered.append(True))
    sentinel_idx = next(i for i in range(cb.count()) if cb.itemData(i) == _BROWSE_SENTINEL)
    cb.setCurrentIndex(sentinel_idx)
    assert triggered == [True]
    assert cb.current_model() == "a"  # bounced back


def test_combobox_set_current_model_inserts_missing_entry() -> None:
    _app()
    cb = ModelComboBox(installed=["a"], current="a")
    cb.set_current_model("zzz")
    assert cb.current_model() == "zzz"


def test_combobox_no_browse_when_disabled() -> None:
    _app()
    cb = ModelComboBox(installed=["a"], current="a", allow_browse=False)
    items = [cb.itemData(i) for i in range(cb.count())]
    assert _BROWSE_SENTINEL not in items


def test_combobox_empty_installed_shows_placeholder() -> None:
    _app()
    cb = ModelComboBox(installed=[], current="", allow_browse=False)
    assert cb.itemText(0).startswith("(")


# ---------------------------------------------------------------------------
# Hugging Face catalog
# ---------------------------------------------------------------------------


def test_quant_from_filename_extracts_known_quants() -> None:
    assert _quant_from_filename("Llama-3.2-3B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
    assert _quant_from_filename("gemma-2-9b-it.F16.gguf") == "F16"
    assert _quant_from_filename("Qwen2.5-7B-Instruct.IQ3_M.gguf") == "IQ3_M"
    assert _quant_from_filename("model.Q8_0.gguf") == "Q8_0"
    assert _quant_from_filename("nonsense.gguf") == "?"


@respx.mock
def test_hf_catalog_search_returns_typed_entries() -> None:
    rows = [
        {
            "modelId": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "author": "bartowski",
            "downloads": 12345,
            "likes": 678,
            "tags": ["gguf", "llama"],
            "lastModified": "2024-10-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "library_name": "gguf",
        },
        {
            "modelId": "TheBloke/qwen2.5-7B-Instruct-GGUF",
            "author": "TheBloke",
            "downloads": 6789,
            "likes": 100,
            "tags": ["gguf", "qwen"],
            "lastModified": "2024-09-01T00:00:00.000Z",
            "pipeline_tag": "text-generation",
            "library_name": "gguf",
        },
    ]
    respx.get("https://huggingface.co/api/models").respond(json=rows)

    catalog = HuggingFaceCatalog()
    try:
        results = catalog.search(query="instruct", limit=10)
    finally:
        catalog.close()

    assert len(results) == 2
    assert results[0].repo_id == "bartowski/Llama-3.2-3B-Instruct-GGUF"
    assert results[0].downloads == 12345
    assert results[0].display_family == "llama"
    assert results[1].display_family == "qwen"


@respx.mock
def test_hf_catalog_fetch_files_filters_to_gguf() -> None:
    payload = {
        "siblings": [
            {"rfilename": "README.md"},
            {"rfilename": "config.json"},
            {
                "rfilename": "llama-3.2-3b-instruct.Q4_K_M.gguf",
                "size": 2_000_000_000,
            },
            {
                "rfilename": "llama-3.2-3b-instruct.Q8_0.gguf",
                "lfs": {"size": 3_500_000_000},
            },
        ]
    }
    respx.get("https://huggingface.co/api/models/bartowski/Llama-3.2-3B-Instruct-GGUF").respond(
        json=payload
    )

    catalog = HuggingFaceCatalog()
    try:
        files = catalog.fetch_files("bartowski/Llama-3.2-3B-Instruct-GGUF")
    finally:
        catalog.close()

    names = [f.filename for f in files]
    assert "README.md" not in names
    assert "config.json" not in names
    assert any(f.filename.endswith("Q4_K_M.gguf") for f in files)
    q4 = next(f for f in files if "Q4_K_M" in f.filename)
    assert q4.quant == "Q4_K_M"
    assert q4.size_bytes == 2_000_000_000
    q8 = next(f for f in files if "Q8_0" in f.filename)
    assert q8.size_bytes == 3_500_000_000  # picked up from lfs.size


def test_hf_entry_ollama_name_is_stable_and_safe() -> None:
    entry = HuggingFaceGGUFEntry(repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF")
    file = HuggingFaceGGUFFile(filename="x.Q4_K_M.gguf", size_bytes=1, quant="Q4_K_M")
    tag = entry.ollama_name(file)
    assert tag.startswith("hf.")
    assert "/" not in tag
    assert tag == "hf.bartowski-llama-3.2-3b-instruct-gguf:q4_k_m"


@respx.mock
def test_hf_catalog_download_writes_file_with_progress(tmp_path: Path) -> None:
    body = b"a" * 4096
    respx.head("https://huggingface.co/repo/x/resolve/main/file.gguf").respond(
        headers={"Content-Length": str(len(body))},
    )
    respx.get("https://huggingface.co/repo/x/resolve/main/file.gguf").respond(
        content=body,
        headers={"Content-Length": str(len(body))},
    )

    dest = tmp_path / "file.gguf"
    progress_events: list[tuple[int, int]] = []

    catalog = HuggingFaceCatalog()
    try:
        result = catalog.download_gguf(
            "repo/x",
            "file.gguf",
            dest,
            progress=lambda c, t: progress_events.append((c, t)),
        )
    finally:
        catalog.close()

    assert result == dest
    assert dest.read_bytes() == body
    assert progress_events
    assert progress_events[-1][0] == len(body)


@respx.mock
def test_hf_catalog_download_is_idempotent_when_size_matches(tmp_path: Path) -> None:
    body = b"a" * 1024
    dest = tmp_path / "file.gguf"
    dest.write_bytes(body)

    head_route = respx.head("https://huggingface.co/repo/x/resolve/main/file.gguf").respond(
        headers={"Content-Length": str(len(body))}
    )
    get_route = respx.get("https://huggingface.co/repo/x/resolve/main/file.gguf").respond(
        content=body
    )

    catalog = HuggingFaceCatalog()
    try:
        result = catalog.download_gguf("repo/x", "file.gguf", dest)
    finally:
        catalog.close()

    assert result == dest
    # HEAD was called; GET was NOT called (idempotent skip).
    assert head_route.called
    assert not get_route.called
