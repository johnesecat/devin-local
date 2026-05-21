"""Directory-backed knowledge index with on-demand TF-IDF search.

This is the *file-based* knowledge subsystem the GUI's per-session settings
panel points at. It is intentionally distinct from the legacy
:class:`KnowledgeStore` (JSONL) for a few reasons:

1. **Zero token cost by default.** Only a tiny manifest (relative path +
   1-line summary) is embedded into the system prompt — not the full note
   bodies. Bodies live on disk and are only loaded when the agent calls a
   tool to read them. For a corpus of N notes averaging 5 KB each, this
   trades ~N×5 KB of prompt tokens for ~N×80 chars of manifest tokens.
2. **Windows-friendly.** The corpus is just plain ``.md`` / ``.txt`` files
   under a directory the operator picks (per-session). Dropping files in
   the folder is the entire ingestion workflow — no JSONL editor needed.
3. **Per-session.** Each session in the GUI can point at its own knowledge
   directory, so a "Rust refactor" session and a "Customer support" session
   don't contaminate each other.

The index is rebuilt on demand from the directory contents and cached in
memory for the lifetime of the :class:`KnowledgeIndex` instance.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Same tokenizer as KnowledgeStore so search behavior stays consistent.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
# File extensions we treat as knowledge sources by default.
DEFAULT_EXTENSIONS: tuple[str, ...] = (".md", ".markdown", ".txt", ".rst")
# Max bytes we will read for a single note (defensive — avoid loading huge
# accidental drops).
MAX_NOTE_BYTES = 1_500_000  # 1.5 MB


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _first_heading_or_line(text: str) -> str:
    """Return the first markdown heading or first non-empty line as title."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Markdown headings: strip leading hashes.
        if line.startswith("#"):
            return line.lstrip("#").strip() or line
        return line
    return ""


def _first_summary(text: str, *, max_chars: int = 140) -> str:
    """Return a 1-line summary suitable for the in-prompt manifest.

    We try to grab the first sentence/line after the title, then collapse
    whitespace and truncate to ``max_chars`` characters.
    """
    seen_title = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not seen_title:
            seen_title = True
            continue
        # Skip blank lines / pure markdown formatting noise.
        cleaned = re.sub(r"\s+", " ", line)
        if len(cleaned) > max_chars:
            return cleaned[: max_chars - 1].rstrip() + "\u2026"
        return cleaned
    # No second line — fall back to the title itself.
    line = _first_heading_or_line(text)
    return re.sub(r"\s+", " ", line)[:max_chars]


@dataclass
class IndexedNote:
    """A single knowledge file on disk."""

    relpath: str  # path relative to the index root (forward slashes)
    abs_path: Path
    title: str
    summary: str
    size: int
    mtime: float

    def manifest_line(self) -> str:
        """Render as a single line for the system-prompt manifest."""
        if self.summary and self.summary != self.title:
            return f"- {self.relpath} — {self.title}: {self.summary}"
        return f"- {self.relpath} — {self.title}" if self.title else f"- {self.relpath}"


@dataclass
class KnowledgeSearchHit:
    """One match from :meth:`KnowledgeIndex.search`."""

    relpath: str
    title: str
    score: float
    excerpt: str

    def to_block(self) -> str:
        return f"### {self.relpath}\n*{self.title}*\n\n{self.excerpt.strip()}"


@dataclass
class KnowledgeIndex:
    """Directory-backed knowledge index.

    Use :meth:`open` to build one from a directory; :meth:`manifest` for
    the system-prompt summary; :meth:`search` for on-demand TF-IDF
    retrieval; :meth:`read` to fetch a specific note's body.
    """

    root: Path
    notes: list[IndexedNote] = field(default_factory=list)
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        extensions: tuple[str, ...] | None = None,
    ) -> KnowledgeIndex:
        """Build (or rebuild) an index from ``root``.

        Missing directories are tolerated — the index is simply empty.
        """
        exts = extensions or DEFAULT_EXTENSIONS
        index = cls(root=root, notes=[], extensions=exts)
        index.reload()
        return index

    def reload(self) -> None:
        """Re-scan ``self.root`` and rebuild ``self.notes``."""
        self.notes = []
        if not self.root.exists() or not self.root.is_dir():
            return
        # Sort for stable ordering across platforms (Windows is case-insensitive
        # but we want deterministic test output).
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.extensions:
                continue
            try:
                stat = path.stat()
                if stat.st_size > MAX_NOTE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            relpath = path.relative_to(self.root).as_posix()
            self.notes.append(
                IndexedNote(
                    relpath=relpath,
                    abs_path=path,
                    title=_first_heading_or_line(text) or relpath,
                    summary=_first_summary(text),
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                )
            )

    # ---------- prompt-side ----------

    def manifest(self, *, max_entries: int = 200, header: str | None = None) -> str:
        """Render the manifest text inlined into the system prompt.

        Returns an empty string when the index has no notes. Capped at
        ``max_entries`` to keep the prompt small even if the operator
        drops thousands of files into the directory.
        """
        if not self.notes:
            return ""
        lines = [n.manifest_line() for n in self.notes[:max_entries]]
        overflow = max(0, len(self.notes) - max_entries)
        if overflow:
            lines.append(f"- (+{overflow} more file(s) — call `knowledge_search` to find them)")
        body = "\n".join(lines)
        head = header or (
            f"Knowledge directory: {self.root}\n"
            "These notes live on disk; their full contents are NOT embedded in this prompt. "
            "Use the `knowledge_search` tool to find passages by keyword, or `knowledge_read` "
            "to load a specific file. Prefer searching before reading."
        )
        return f"{head}\n\n{body}"

    # ---------- search-side ----------

    def search(
        self, query: str, *, k: int = 4, excerpt_chars: int = 600
    ) -> list[KnowledgeSearchHit]:
        """TF-IDF top-K search across the corpus.

        Each hit carries a focused excerpt (window around the highest-scoring
        match) so the agent never needs to read the full file just to get a
        relevant passage.
        """
        if not self.notes or not query.strip():
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        # Read note bodies lazily (only while scoring).
        bodies: list[tuple[IndexedNote, str, Counter[str]]] = []
        df: Counter[str] = Counter()
        for note in self.notes:
            try:
                text = note.abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            tf = Counter(_tokenize(f"{note.title}\n{text}"))
            bodies.append((note, text, tf))
            for term in tf:
                df[term] += 1
        n_docs = len(bodies)
        if not n_docs:
            return []

        def idf(term: str) -> float:
            return math.log((n_docs + 1) / (1 + df.get(term, 0))) + 1.0

        q_tf = Counter(q_tokens)
        scored: list[tuple[float, IndexedNote, str]] = []
        for note, text, tf in bodies:
            score = 0.0
            for term, qf in q_tf.items():
                if term in tf:
                    score += qf * tf[term] * (idf(term) ** 2)
            if score > 0:
                scored.append((score, note, text))
        scored.sort(key=lambda triple: triple[0], reverse=True)

        hits: list[KnowledgeSearchHit] = []
        for score, note, text in scored[:k]:
            excerpt = self._best_excerpt(text, q_tokens, max_chars=excerpt_chars)
            hits.append(
                KnowledgeSearchHit(
                    relpath=note.relpath,
                    title=note.title,
                    score=score,
                    excerpt=excerpt,
                )
            )
        return hits

    @staticmethod
    def _best_excerpt(text: str, q_tokens: list[str], *, max_chars: int) -> str:
        """Return a centered window around the densest match in ``text``."""
        lower = text.lower()
        # Find the earliest occurrence of any query token; expand a window
        # around it. This is a cheap heuristic — good enough for retrieval.
        best_pos = -1
        for tok in q_tokens:
            pos = lower.find(tok)
            if pos < 0:
                continue
            if best_pos < 0 or pos < best_pos:
                best_pos = pos
        if best_pos < 0:
            return text[:max_chars]
        half = max_chars // 2
        start = max(0, best_pos - half)
        end = min(len(text), best_pos + half)
        snippet = text[start:end]
        if start > 0:
            snippet = "\u2026 " + snippet
        if end < len(text):
            snippet = snippet + " \u2026"
        return snippet

    # ---------- read-side ----------

    def read(self, relpath: str, *, max_chars: int = 200_000) -> str:
        """Load the full body of a specific note by relative path.

        Path traversal is prevented by resolving against ``self.root``;
        attempts to escape raise :class:`ValueError`.
        """
        if not relpath:
            raise ValueError("relpath is required")
        # Normalize to a relative path; reject absolute paths outright.
        candidate = Path(relpath.replace("\\", "/"))
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise ValueError(f"Refusing to read path outside knowledge root: {relpath!r}")
        target = (self.root / candidate).resolve()
        try:
            root_resolved = self.root.resolve()
        except OSError as exc:
            raise ValueError(f"Knowledge root unavailable: {exc}") from exc
        try:
            target.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"Path escapes knowledge root: {relpath!r}") from exc
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"No such knowledge file: {relpath}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            head = text[: max_chars // 2]
            tail = text[-max_chars // 2 :]
            omitted = len(text) - len(head) - len(tail)
            text = f"{head}\n... [truncated {omitted} characters] ...\n{tail}"
        return text

    # ---------- snapshot ----------

    def fingerprint(self) -> tuple[str, ...]:
        """Stable identity for cache-invalidation in the agent."""
        return tuple(f"{n.relpath}:{n.size}:{int(n.mtime)}" for n in self.notes)
