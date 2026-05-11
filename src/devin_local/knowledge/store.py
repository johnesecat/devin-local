"""Local knowledge store with simple TF-IDF retrieval.

We deliberately avoid pulling in heavyweight embeddings — the goal is fast,
zero-config knowledge upload that works offline. For most uses the TF-IDF
ranker is plenty; users who need semantic search can swap this out via a
plugin.

Notes are stored as JSONL in `knowledge/store.jsonl`:

    {"id": "...", "title": "...", "scope": "...", "body": "...", "added_ts": ...}
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class KnowledgeNote:
    """A single user-added knowledge note."""

    id: str
    title: str
    body: str
    scope: str = ""
    tags: list[str] = field(default_factory=list)
    added_ts: float = field(default_factory=time.time)

    def to_block(self) -> str:
        header = f"### {self.title}"
        if self.scope:
            header += f"\n*Scope: {self.scope}*"
        if self.tags:
            header += f"\n*Tags: {', '.join(self.tags)}*"
        return f"{header}\n\n{self.body.strip()}"


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class KnowledgeStore:
    """Append-only JSONL knowledge store with TF-IDF ranking."""

    path: Path
    notes: list[KnowledgeNote] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path) -> KnowledgeStore:
        path.parent.mkdir(parents=True, exist_ok=True)
        store = cls(path=path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                store.notes.append(KnowledgeNote(**data))
        return store

    def add(
        self,
        title: str,
        body: str,
        scope: str = "",
        tags: list[str] | None = None,
    ) -> KnowledgeNote:
        note = KnowledgeNote(
            id=uuid.uuid4().hex[:12],
            title=title.strip() or "Untitled",
            body=body.strip(),
            scope=scope.strip(),
            tags=list(tags or []),
        )
        self.notes.append(note)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(note), ensure_ascii=False) + "\n")
        return note

    def remove(self, note_id: str) -> bool:
        before = len(self.notes)
        self.notes = [n for n in self.notes if n.id != note_id]
        if len(self.notes) == before:
            return False
        with self.path.open("w", encoding="utf-8") as fh:
            for n in self.notes:
                fh.write(json.dumps(asdict(n), ensure_ascii=False) + "\n")
        return True

    def all(self) -> list[KnowledgeNote]:
        return list(self.notes)

    def search(self, query: str, k: int = 4) -> list[KnowledgeNote]:
        """Return the top-k notes matching `query` by TF-IDF cosine similarity."""
        if not self.notes:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Document term frequencies.
        docs = [
            (note, Counter(_tokenize(f"{note.title}\n{note.scope}\n{note.body}")))
            for note in self.notes
        ]
        df: Counter[str] = Counter()
        for _note, tf in docs:
            for term in tf:
                df[term] += 1
        n_docs = len(docs)

        def idf(term: str) -> float:
            return math.log((n_docs + 1) / (1 + df.get(term, 0))) + 1.0

        scored: list[tuple[float, KnowledgeNote]] = []
        q_tf = Counter(query_tokens)
        for note, tf in docs:
            score = 0.0
            for term, qf in q_tf.items():
                if term in tf:
                    score += qf * tf[term] * (idf(term) ** 2)
            if score > 0:
                scored.append((score, note))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [note for _score, note in scored[:k]]
