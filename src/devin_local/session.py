"""Session persistence (JSONL).

Every chat turn (user, assistant, tool) is appended to a JSONL file so a
crash or restart doesn't lose work. The CLI uses this to support `--resume`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from devin_local.ollama_client import ChatMessage


@dataclass
class SessionStore:
    """Append-only JSONL store of chat messages."""

    path: Path
    messages: list[ChatMessage] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path) -> SessionStore:
        path.parent.mkdir(parents=True, exist_ok=True)
        store = cls(path=path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                store.messages.append(
                    ChatMessage(
                        role=record.get("role", "user"),
                        content=record.get("content", ""),
                        tool_calls=record.get("tool_calls", []),
                        name=record.get("name"),
                    )
                )
        return store

    def append(self, message: ChatMessage) -> None:
        self.messages.append(message)
        record = {
            "ts": time.time(),
            **message.to_dict(),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def replace_all(self, messages: list[ChatMessage]) -> None:
        """Used by the context manager after compaction — rewrites the file."""
        self.messages = list(messages)
        with self.path.open("w", encoding="utf-8") as fh:
            for msg in messages:
                record = {"ts": time.time(), **msg.to_dict()}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
