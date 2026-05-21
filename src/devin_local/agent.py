"""The agent harness: plan → call tools → verify → respond, in a loop.

The harness is intentionally simple and synchronous. It owns:

    - the Ollama client + chosen model,
    - the toolbelt (`ToolRegistry`),
    - the conversation messages (incl. system prompt),
    - the context manager (compaction),
    - the knowledge + skill stores,
    - the loaded plugins.

Each call to `Agent.handle_user(text)` performs at most `max_iterations`
model→tool round-trips for that single user turn, then returns the final
assistant text. Tool calls are dispatched through `ToolRegistry.dispatch`,
results are appended as `role="tool"` messages, and the loop continues
until the model returns a final answer with no further tool calls.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devin_local.agent_planning import Plan, parse_plan_block, strip_plan_block
from devin_local.context_manager import ContextManager, make_default_context_manager
from devin_local.inference.backend import (
    BackendUnavailableError,
    ChatChunk,
    InferenceBackend,
)
from devin_local.inference.factory import build_backend
from devin_local.inference.types import ChatMessage
from devin_local.knowledge.index import KnowledgeIndex
from devin_local.knowledge.store import KnowledgeStore
from devin_local.mcp.client import MCPClientManager, load_mcp_servers
from devin_local.ollama_client import (
    OllamaClient,
    OllamaError,
    parse_tool_call_arguments,
)
from devin_local.plugins.loader import discover_plugins
from devin_local.session import SessionStore
from devin_local.skills.loader import SkillLoader, load_default_skills
from devin_local.system_prompt import PromptContext, build_system_prompt
from devin_local.tools.base import ToolResult
from devin_local.tools.registry import ToolRegistry, build_default_registry

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def _tokenize_for_match(text: str) -> list[str]:
    """Lowercase word tokens for keyword-overlap matching.

    Used by ``Agent._select_relevant_tools``. Drops short/common tokens
    so they don't trigger false positives on every tool description.
    """
    if not text:
        return []
    return [w.lower() for w in _WORD_RE.findall(text)]


@dataclass
class AgentConfig:
    """User-facing knobs for the agent."""

    model: str = "llama3.1:8b"
    workspace: Path = field(default_factory=lambda: Path.cwd())
    backend: str = "ollama"  # "ollama" | "layered" | "hf"
    backend_options: dict[str, Any] = field(default_factory=dict)
    ollama_host: str = "http://127.0.0.1:11434"
    max_iterations: int = 20
    enable_obliteratus: bool = True
    enable_desktop: bool = True
    enable_browser: bool = True
    enable_mcp: bool = True
    enable_plugins: bool = True
    enable_knowledge_injection: bool = True
    enable_skill_injection: bool = True
    # When True (default) the agent loads ALL user + workspace knowledge
    # notes and ALL matching skills into the system prompt ONCE at session
    # start, instead of running a TF-IDF search every turn. This is faster,
    # cheaper on tokens (because the system prompt is cached on the model
    # side), and lets users upload knowledge from the GUI Settings tab
    # knowing it will stay embedded for the whole session.
    embed_all_knowledge: bool = False
    # Soft cap on how many characters of knowledge to inline into the
    # system prompt when ``embed_all_knowledge=True``. Ignored otherwise.
    max_embedded_knowledge_chars: int = 200_000
    # Per-session knowledge directory. Plain ``.md`` / ``.txt`` files dropped
    # here are exposed to the agent via the ``knowledge_search`` and
    # ``knowledge_read`` tools, and a tiny manifest (path — title: summary)
    # is embedded in the system prompt. None means "no directory configured".
    knowledge_dir: Path | None = None
    # Cap on the system-prompt manifest size (entries, not characters). Files
    # past this are referenced as '(+N more)' and discoverable via search.
    max_manifest_entries: int = 200
    # Per-session system-prompt override. Empty string means "use the default".
    system_prompt_override: str = ""
    # When True, render the full upstream-style verbose system prompt
    # (~13 KB). When False (default), use the slim ~3 KB prompt — much
    # faster for local CPU inference, same identity / honesty / planning
    # rules.
    verbose_prompt: bool = False
    # Spill tool outputs larger than this many characters to disk under
    # ``<workspace>/.devin-local/scratch/`` and keep only a short summary in
    # the chat history. Memory-efficient mode.
    max_inline_tool_output_chars: int = 4_000
    # When True, replace the inline tool output with a short marker pointing
    # at the spilled file (recoverable via ``read_file``).
    spill_large_tool_output: bool = True
    temperature: float = 0.2
    session_path: Path | None = None
    num_ctx: int | None = None  # override the backend's context window
    keep_alive: str | int | None = None  # Ollama: keep model resident between turns
    parallel_tool_calls: bool = True
    # How much of the tool schema to ship to the model.
    #
    #   "full"   — every registered tool, regardless of likely relevance.
    #             Power-user default; bigger prompt, more flexibility.
    #   "smart"  — ship core tools (read/write/shell) always, plus any
    #             tools whose names/descriptions match keywords in the
    #             most recent user message. Cuts prefill cost on CPU
    #             without sacrificing capability — if the model needs a
    #             tool we didn't ship, the next turn auto-includes it.
    tools_schema_mode: str = "smart"
    # Tool names that are ALWAYS included regardless of selection mode.
    # The file + shell + knowledge tools are foundational; almost every
    # turn touches at least one of them, so keeping them resident saves
    # the round-trip cost of "fetch the tool you needed and re-prompt".
    always_include_tools: tuple[str, ...] = (
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "find_files",
        "grep",
        "shell_exec",
        "knowledge_search",
        "knowledge_read",
        "knowledge_list",
    )
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurn:
    """The outcome of one user message."""

    assistant_text: str
    tool_results: list[tuple[str, ToolResult]] = field(default_factory=list)
    iterations: int = 0


ToolObserver = Callable[[str, dict[str, Any], ToolResult], None]
StreamObserver = Callable[[ChatChunk], None]
ToolStartObserver = Callable[[str, dict[str, Any]], None]
PlanObserver = Callable[[Plan], None]


class Agent:
    """Core devin-local agent.

    Talks to any :class:`InferenceBackend`; the default is an Ollama backend
    so existing callers don't need to change. Pass ``backend=...`` to inject
    a custom one (e.g. the layered AirLLM backend or a scripted test backend).
    """

    def __init__(
        self,
        config: AgentConfig,
        client: OllamaClient | None = None,
        registry: ToolRegistry | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        self.config = config
        self.workspace = config.workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        # Backend selection: explicit > legacy client > factory.
        if backend is not None:
            self.backend: InferenceBackend = backend
        elif client is not None:
            from devin_local.inference.ollama_backend import OllamaBackend

            self.backend = OllamaBackend(host=config.ollama_host, client=client)
        else:
            opts = dict(config.backend_options)
            if config.backend == "ollama":
                opts.setdefault("host", config.ollama_host)
            self.backend = build_backend(config.backend, **opts)
        # Keep `self.client` for backwards compatibility with code that
        # reaches into the agent for the raw Ollama client (tests, plugins).
        self.client = client
        self.registry = registry or build_default_registry(
            self.workspace,
            enable_desktop=config.enable_desktop,
            enable_browser=config.enable_browser,
        )
        self.knowledge = KnowledgeStore.open(self.workspace / "knowledge" / "store.jsonl")
        # User-level (cross-workspace) knowledge store: GUI uploads land here.
        try:
            from devin_local.settings import user_knowledge_path

            user_path = user_knowledge_path()
        except Exception:  # noqa: BLE001 - settings dir not writable; fall back.
            user_path = self.workspace / ".devin-local-user-knowledge.jsonl"
        self.user_knowledge = KnowledgeStore.open(user_path)
        # File-directory-backed knowledge (per-session, near-zero token cost).
        self.knowledge_index: KnowledgeIndex | None = None
        if config.knowledge_dir is not None:
            try:
                self.knowledge_index = KnowledgeIndex.open(Path(config.knowledge_dir))
            except Exception as exc:  # noqa: BLE001
                log.warning("knowledge_dir open failed (%s): %s", config.knowledge_dir, exc)
                self.knowledge_index = None
        self._scratch_dir = self.workspace / ".devin-local" / "scratch"
        self._spill_turn = 0
        self.skills: SkillLoader = load_default_skills(self.workspace)
        self.mcp_manager: MCPClientManager | None = None
        self.system_prompt_sections: list[str] = []
        self.plugins_loaded: list[str] = []
        self.messages: list[ChatMessage] = []
        self.session: SessionStore | None = None
        self._observers: list[ToolObserver] = []
        self._stream_observers: list[StreamObserver] = []
        self._tool_start_observers: list[ToolStartObserver] = []
        self._plan_observers: list[PlanObserver] = []
        self.plan: Plan = Plan()
        if config.session_path is not None:
            self.session = SessionStore.open(config.session_path)
            self.messages = list(self.session.messages)
        self.context_manager: ContextManager = make_default_context_manager(
            config.model, self._summarize
        )
        self._initialized = False
        self._cached_system_prompt: str | None = None
        self._cached_tool_names_snapshot: tuple[str, ...] | None = None
        # Full cache key: (tool_names, knowledge_snapshot, skill_snapshot).
        # Invalidates the cached system prompt whenever any of those change.
        self._cached_prompt_cache_key: tuple[Any, ...] | None = None
        # Cancellation flag for the in-flight turn. Set by ``request_cancel``;
        # consumed by ``handle_user``. Threading.Event is process-safe and
        # cheap to check between iterations.
        import threading as _threading

        self._cancel_event = _threading.Event()

    # ---------- public API ----------

    def add_tool_observer(self, observer: ToolObserver) -> None:
        self._observers.append(observer)

    def add_stream_observer(self, observer: StreamObserver) -> None:
        """Subscribe to streaming chat chunks (token deltas + final message).

        Used by the CLI and GUI to render tokens as they arrive. The agent
        loop still owns message accumulation \u2014 observers are read-only.
        """
        self._stream_observers.append(observer)

    def add_tool_start_observer(self, observer: ToolStartObserver) -> None:
        """Subscribe to ``(name, arguments)`` events fired just before a tool runs."""
        self._tool_start_observers.append(observer)

    def add_plan_observer(self, observer: PlanObserver) -> None:
        """Subscribe to plan updates emitted by the agent's planning loop."""
        self._plan_observers.append(observer)

    def initialize(self) -> None:
        """One-time setup: load plugins, connect MCP servers, build system prompt."""
        if self._initialized:
            return
        if self.config.enable_plugins:
            self._load_plugins()
        if self.config.enable_mcp:
            self._connect_mcp()
        self._register_knowledge_tools()
        self._load_user_tools()
        self._ensure_system_prompt()
        self._initialized = True

    def reload_user_tools(self) -> Any:
        """Re-scan ``~/.devin-local/tools/`` and refresh registrations.

        Returns the loader report so the GUI can surface any per-file
        errors. Idempotent and safe to call mid-session.
        """
        report = self._load_user_tools()
        # Invalidate the cached prompt so the next turn picks up the new
        # tool list (tool names are part of the cache key).
        self._cached_system_prompt = None
        self._cached_prompt_cache_key = None
        if self.messages and self.messages[0].role == "system":
            self.messages.pop(0)
        self._ensure_system_prompt()
        return report

    def refresh_embedded_knowledge(self) -> None:
        """Re-read the knowledge stores from disk and rebuild the system prompt.

        Called by the GUI's Settings → Knowledge tab after the user uploads,
        edits, or deletes notes so the change is visible without restarting
        the session. The very next ``handle_user`` will use the new prompt.
        """
        # Refresh JSONL stores from disk (legacy path).
        self.knowledge = KnowledgeStore.open(self.knowledge.path)
        self.user_knowledge = KnowledgeStore.open(self.user_knowledge.path)
        # Refresh the directory-backed index too (per-session).
        if self.knowledge_index is not None:
            self.knowledge_index.reload()
        # Reload skills too — they're cheap to scan.
        self.skills.reload()
        # Drop the old system message; _ensure_system_prompt will rebuild.
        if self.messages and self.messages[0].role == "system":
            self.messages.pop(0)
        self._cached_system_prompt = None
        self._cached_prompt_cache_key = None
        # Re-register knowledge tools if the index changed.
        self._register_knowledge_tools()
        self._ensure_system_prompt()

    def set_knowledge_dir(self, path: Path | None) -> None:
        """Swap the session's knowledge directory at runtime.

        Used by the per-session settings dialog after the operator picks a
        new knowledge folder. Rebuilds the index, re-registers the tools,
        and invalidates the cached system prompt.
        """
        self.config.knowledge_dir = path
        if path is None:
            self.knowledge_index = None
        else:
            try:
                self.knowledge_index = KnowledgeIndex.open(Path(path))
            except Exception as exc:  # noqa: BLE001
                log.warning("set_knowledge_dir(%s) failed: %s", path, exc)
                self.knowledge_index = None
        if self.messages and self.messages[0].role == "system":
            self.messages.pop(0)
        self._cached_system_prompt = None
        self._cached_prompt_cache_key = None
        self._register_knowledge_tools()
        self._ensure_system_prompt()

    def set_system_prompt_override(self, prompt: str) -> None:
        """Set the per-session system-prompt override and rebuild the prompt."""
        self.config.system_prompt_override = prompt or ""
        if self.messages and self.messages[0].role == "system":
            self.messages.pop(0)
        self._cached_system_prompt = None
        self._cached_prompt_cache_key = None
        self._ensure_system_prompt()

    def _register_knowledge_tools(self) -> None:
        """(Re-)register the knowledge_search / knowledge_read / knowledge_list tools.

        Removes any previously-registered knowledge tools first so the
        registry always points at the current index.
        """
        for name in ("knowledge_search", "knowledge_read", "knowledge_list"):
            self.registry.unregister(name)
        if self.knowledge_index is None:
            return
        from devin_local.tools.knowledge_tools import (
            KnowledgeListTool,
            KnowledgeReadTool,
            KnowledgeSearchTool,
        )

        self.registry.register(KnowledgeSearchTool(self.knowledge_index))
        self.registry.register(KnowledgeReadTool(self.knowledge_index))
        self.registry.register(KnowledgeListTool(self.knowledge_index))

    def shutdown(self) -> None:
        """Tear down MCP, terminals, and the inference backend."""
        try:
            if self.mcp_manager is not None:
                self.mcp_manager.shutdown()
        except Exception:  # noqa: BLE001
            log.exception("MCP shutdown failed")
        try:
            self.backend.close()
        except Exception:  # noqa: BLE001
            log.exception("inference backend close failed")

    def request_cancel(self) -> None:
        """Signal that the in-flight ``handle_user`` should abort.

        Two-stage cancellation:
        1. Sets a flag the agent loop checks between iterations and tool
           dispatches, so an in-progress turn returns cleanly with a
           "cancelled" final message rather than crashing.
        2. Closes the inference backend's HTTP connection so an in-flight
           streaming response is torn down immediately (otherwise a slow
           CPU prefill blocks the cancel by minutes).
        """
        self._cancel_event.set()
        try:
            self.backend.close()
        except Exception:  # noqa: BLE001
            log.exception("backend close during cancel failed")

    def reset_cancel(self) -> None:
        self._cancel_event.clear()

    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def handle_user(self, text: str, *, stream: bool = False) -> AgentTurn:
        """Run one user → assistant turn (with any tool calls).

        Set ``stream=True`` to receive incremental token deltas via any
        registered stream observers. The returned :class:`AgentTurn` is
        identical either way.
        """
        self.reset_cancel()
        self.initialize()
        user_msg = ChatMessage(role="user", content=text)
        self._append(user_msg)
        self._inject_relevant_context(text)
        self.messages = self.context_manager.maybe_compact(self.messages)
        if self.session is not None:
            self.session.replace_all(self.messages)

        tool_results: list[tuple[str, ToolResult]] = []
        self.plan = Plan()
        for iteration in range(self.config.max_iterations):
            if self._cancel_event.is_set():
                return AgentTurn(
                    assistant_text="(cancelled by operator)",
                    tool_results=tool_results,
                    iterations=iteration,
                )
            try:
                assistant_msg = self._call_model(stream=stream)
            except Exception as exc:  # noqa: BLE001
                if self._cancel_event.is_set():
                    return AgentTurn(
                        assistant_text="(cancelled by operator)",
                        tool_results=tool_results,
                        iterations=iteration,
                    )
                raise exc
            if self._cancel_event.is_set():
                return AgentTurn(
                    assistant_text="(cancelled by operator)",
                    tool_results=tool_results,
                    iterations=iteration + 1,
                )
            self._append(assistant_msg)
            self._maybe_update_plan(assistant_msg.content)

            if not assistant_msg.tool_calls:
                final_text = strip_plan_block(assistant_msg.content).strip()
                return AgentTurn(
                    assistant_text=final_text,
                    tool_results=tool_results,
                    iterations=iteration + 1,
                )

            calls_to_dispatch: list[tuple[str, dict[str, Any]]] = []
            for call in assistant_msg.tool_calls:
                fn = (call.get("function") or {}) if isinstance(call, dict) else {}
                name = fn.get("name") or call.get("name") or ""
                raw_args = fn.get("arguments", {})
                arguments = parse_tool_call_arguments(raw_args)
                calls_to_dispatch.append((name, arguments))

            # Fire tool_started observers before dispatching so GUIs can
            # render a "running" pill / start a timer before the tool's side
            # effects are observable.
            for name, arguments in calls_to_dispatch:
                for sobs in self._tool_start_observers:
                    try:
                        sobs(name, arguments)
                    except Exception:  # noqa: BLE001
                        log.exception("tool start observer raised")

            if self.config.parallel_tool_calls and len(calls_to_dispatch) > 1:
                results = self.registry.dispatch_many(calls_to_dispatch)
            else:
                results = [self.registry.dispatch(n, a) for n, a in calls_to_dispatch]

            for (name, arguments), result in zip(calls_to_dispatch, results, strict=False):
                tool_results.append((name, result))
                for observer in self._observers:
                    try:
                        observer(name, arguments, result)
                    except Exception:  # noqa: BLE001
                        log.exception("tool observer raised")
                payload = result.to_chat_payload()
                payload = self._maybe_spill_tool_output(name, payload)
                self._append(
                    ChatMessage(
                        role="tool",
                        name=name,
                        content=payload,
                    )
                )
            self.messages = self.context_manager.maybe_compact(self.messages)
            if self.session is not None:
                self.session.replace_all(self.messages)

        return AgentTurn(
            assistant_text="(agent reached max iterations without a final answer)",
            tool_results=tool_results,
            iterations=self.config.max_iterations,
        )

    # ---------- internals ----------

    def _maybe_update_plan(self, assistant_text: str) -> None:
        """Parse a `<plan>` block from `assistant_text`, merge into self.plan,
        and notify observers if it changed.
        """
        new_plan = parse_plan_block(assistant_text)
        if new_plan is None:
            return
        if self.plan.is_empty():
            self.plan = new_plan
        else:
            self.plan.merge(new_plan)
        for obs in list(self._plan_observers):
            try:
                obs(self.plan)
            except Exception:  # noqa: BLE001
                log.exception("plan observer raised")

    def _append(self, message: ChatMessage) -> None:
        self.messages.append(message)
        if self.session is not None:
            self.session.append(message)

    def _select_relevant_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pick a relevant subset of ``schemas`` for the current turn.

        Honest, keyword-overlap based selection — no model call, no
        embedding. Saves CPU prefill cost without requiring the user to
        babysit which tools the agent should use.

        Returns ``schemas`` unchanged when:
        - ``tools_schema_mode != "smart"`` (operator opted into full toolbelt)
        - there is no user message yet
        - there are fewer than 6 schemas (no win from filtering)
        """
        if self.config.tools_schema_mode != "smart":
            return schemas
        if not schemas or len(schemas) < 6:
            return schemas
        latest_user_text = ""
        for msg in reversed(self.messages):
            if msg.role == "user" and msg.content:
                latest_user_text = msg.content
                break
        if not latest_user_text:
            return schemas
        user_words = {w for w in _tokenize_for_match(latest_user_text) if len(w) >= 3}
        if not user_words:
            return schemas
        always = set(self.config.always_include_tools)
        chosen: list[dict[str, Any]] = []
        for schema in schemas:
            fn = schema.get("function") or {}
            name = fn.get("name") or schema.get("name") or ""
            if name in always:
                chosen.append(schema)
                continue
            description = fn.get("description") or ""
            tool_words = set(_tokenize_for_match(name + " " + description))
            if user_words & tool_words:
                chosen.append(schema)
        if not chosen:
            return schemas
        return chosen

    def _call_model(self, *, stream: bool = False) -> ChatMessage:
        options: dict[str, Any] = {"temperature": self.config.temperature}
        if self.config.num_ctx is not None:
            options["num_ctx"] = self.config.num_ctx
        options.update(self.config.extra_options)
        tools = self._select_relevant_tools(self.registry.to_ollama_schemas())
        try:
            if stream:
                final: ChatMessage | None = None
                for chunk in self.backend.stream(
                    model=self.config.model,
                    messages=self.messages,
                    tools=tools,
                    options=options,
                    keep_alive=self.config.keep_alive,
                ):
                    for obs in self._stream_observers:
                        try:
                            obs(chunk)
                        except Exception:  # noqa: BLE001
                            log.exception("stream observer raised")
                    if chunk.done and chunk.message is not None:
                        final = chunk.message
                if final is None:
                    return ChatMessage(role="assistant", content="")
                return final
            response = self.backend.chat(
                model=self.config.model,
                messages=self.messages,
                tools=tools,
                options=options,
                keep_alive=self.config.keep_alive,
            )
            return response.message
        except BackendUnavailableError:
            raise
        except OllamaError:
            raise

    def stream_user(self, text: str) -> Iterator[ChatChunk]:
        """Generator variant of `handle_user` that yields chunks as they arrive.

        The agent's turn runs on a background thread; chunks are handed off to
        the calling thread through a thread-safe queue so callers see token
        deltas immediately (not buffered until the turn completes). After the
        generator is exhausted, ``self.messages`` holds the full updated
        transcript and any tool calls have been dispatched.
        """
        import queue
        import threading

        sentinel = object()
        q: queue.Queue[Any] = queue.Queue()

        def _collect(chunk: ChatChunk) -> None:
            q.put(chunk)

        def _run() -> None:
            try:
                self.handle_user(text, stream=True)
            except BaseException as exc:  # noqa: BLE001
                q.put(exc)
            finally:
                q.put(sentinel)

        self.add_stream_observer(_collect)
        worker = threading.Thread(target=_run, daemon=True, name="agent-stream")
        worker.start()
        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            import contextlib as _ctx

            with _ctx.suppress(ValueError):
                self._stream_observers.remove(_collect)
            worker.join(timeout=5.0)

    def _summarize(self, text: str, max_tokens: int) -> str:
        try:
            return self.backend.summarize(self.config.model, text, max_tokens=max_tokens)
        except (BackendUnavailableError, OllamaError) as exc:
            return f"(summary failed: {exc})"

    def _maybe_spill_tool_output(self, name: str, payload: str) -> str:
        """Spill large tool outputs to disk; keep a short pointer in history.

        Saves the operator's context budget. The full body remains
        recoverable via ``read_file`` if the model needs it again.
        Disabled when ``AgentConfig.spill_large_tool_output`` is False.
        """
        if not self.config.spill_large_tool_output:
            return payload
        cap = max(0, self.config.max_inline_tool_output_chars)
        if cap <= 0 or len(payload) <= cap:
            return payload
        try:
            self._scratch_dir.mkdir(parents=True, exist_ok=True)
            self._spill_turn += 1
            spill_path = self._scratch_dir / f"tool-{self._spill_turn:04d}-{name}.txt"
            spill_path.write_text(payload, encoding="utf-8")
        except OSError as exc:
            log.warning("could not spill tool output for %s: %s", name, exc)
            return payload
        head = payload[: cap // 2]
        tail_chars = cap // 4
        tail = payload[-tail_chars:] if tail_chars > 0 else ""
        overflow = len(payload) - len(head) - len(tail)
        marker = (
            f"\n\n[devin-local] output truncated for memory efficiency. "
            f"Full body ({len(payload)} chars) saved to {spill_path}. "
            f"Use read_file with path={spill_path} to load it again.\n"
            f"[{overflow} chars omitted]\n\n"
        )
        return f"{head}{marker}{tail}"

    def _ensure_system_prompt(self) -> None:
        if self.messages and self.messages[0].role == "system":
            return
        # Cache the system prompt: tools and skills are stable across turns,
        # so rebuilding the prompt every turn just wastes CPU and (worse)
        # changes the prompt-cache hash on the backend, defeating Ollama's
        # KV cache. Only rebuild when the tool set, knowledge, or skill
        # snapshot changes.
        tool_names = tuple(self.registry.names())
        knowledge_snapshot = self._knowledge_snapshot()
        skill_snapshot = self._skill_snapshot()
        manifest_snapshot = self._manifest_snapshot()
        override_snapshot = self.config.system_prompt_override or ""
        cache_key = (
            tool_names,
            knowledge_snapshot,
            skill_snapshot,
            manifest_snapshot,
            override_snapshot,
            bool(self.config.verbose_prompt),
        )
        if self._cached_system_prompt is None or self._cached_prompt_cache_key != cache_key:
            knowledge_blocks = self._collect_knowledge_blocks()
            skill_blocks = self._collect_skill_blocks()
            manifest = ""
            if self.knowledge_index is not None and self.config.enable_knowledge_injection:
                manifest = self.knowledge_index.manifest(
                    max_entries=self.config.max_manifest_entries
                )
            ctx = PromptContext(
                workspace=self.workspace,
                model=self.config.model,
                tool_names=list(tool_names),
                knowledge_blocks=knowledge_blocks,
                skill_blocks=skill_blocks,
                knowledge_manifest=manifest,
                system_prompt_override=override_snapshot,
                enable_obliteratus=self.config.enable_obliteratus,
                extra_sections=self.system_prompt_sections,
                verbose=self.config.verbose_prompt,
            )
            self._cached_system_prompt = build_system_prompt(ctx)
            self._cached_prompt_cache_key = cache_key
            self._cached_tool_names_snapshot = tool_names
        self.messages.insert(0, ChatMessage(role="system", content=self._cached_system_prompt))
        if self.session is not None:
            self.session.replace_all(self.messages)

    def _manifest_snapshot(self) -> tuple[str, ...]:
        if self.knowledge_index is None or not self.config.enable_knowledge_injection:
            return ()
        return self.knowledge_index.fingerprint()

    def _knowledge_snapshot(self) -> tuple[str, ...]:
        """Stable identity of the knowledge corpus, used to invalidate cache."""
        if not self.config.embed_all_knowledge or not self.config.enable_knowledge_injection:
            return ()
        ids: list[str] = []
        for note in self.knowledge.all():
            ids.append(f"ws:{note.id}")
        for note in self.user_knowledge.all():
            ids.append(f"user:{note.id}")
        return tuple(sorted(ids))

    def _skill_snapshot(self) -> tuple[str, ...]:
        if not self.config.enable_skill_injection:
            return ()
        return tuple(sorted(s.name for s in self.skills.skills))

    def _collect_knowledge_blocks(self) -> list[str]:
        """Return all knowledge notes as ``to_block()`` strings, capped by
        :attr:`AgentConfig.max_embedded_knowledge_chars`.
        """
        if not self.config.embed_all_knowledge or not self.config.enable_knowledge_injection:
            return []
        # Workspace notes take priority (more specific), then user notes.
        ordered = list(self.knowledge.all()) + list(self.user_knowledge.all())
        blocks: list[str] = []
        used = 0
        cap = max(0, self.config.max_embedded_knowledge_chars)
        for note in ordered:
            block = note.to_block()
            if cap and used + len(block) > cap:
                # Stop adding once we'd blow the soft cap; do not silently
                # truncate mid-note (would corrupt structure).
                continue
            blocks.append(block)
            used += len(block)
        return blocks

    def _collect_skill_blocks(self) -> list[str]:
        """Return all skill bodies as blocks when ``embed_all_knowledge`` is on."""
        if not self.config.embed_all_knowledge or not self.config.enable_skill_injection:
            return []
        return [s.to_block() for s in self.skills.skills]

    def _inject_relevant_context(self, query: str) -> None:
        """Insert top-K knowledge notes + matching skills before the user turn.

        When :attr:`AgentConfig.embed_all_knowledge` is ``True`` (the default)
        the agent has already inlined every note + skill into the cached
        system prompt, so this method is a no-op — that's the whole point of
        embed-once: avoid re-searching the corpus every turn.

        It still runs in legacy mode (``embed_all_knowledge=False``), where
        callers want classic per-turn TF-IDF retrieval (useful when the
        corpus is too large to inline).
        """
        if self.config.embed_all_knowledge:
            return
        if not (self.config.enable_knowledge_injection or self.config.enable_skill_injection):
            return
        injections: list[str] = []
        if self.config.enable_knowledge_injection:
            ws_notes = self.knowledge.search(query, k=3)
            user_notes = self.user_knowledge.search(query, k=3)
            notes = ws_notes + [n for n in user_notes if n not in ws_notes]
            if notes:
                injections.append(
                    "<retrieved-knowledge>\n"
                    + "\n\n".join(n.to_block() for n in notes)
                    + "\n</retrieved-knowledge>"
                )
        if self.config.enable_skill_injection:
            skills = self.skills.match(query, limit=2)
            if skills:
                injections.append(
                    "<retrieved-skills>\n"
                    + "\n\n".join(s.to_block() for s in skills)
                    + "\n</retrieved-skills>"
                )
        if not injections:
            return
        # Insert as a system message right before the new user message.
        injected = ChatMessage(role="system", content="\n\n".join(injections))
        self.messages.insert(len(self.messages) - 1, injected)

    def _load_user_tools(self) -> Any:
        """Register every ``@tool``-decorated function from the user dir."""
        from devin_local.tools.user_tools import register_user_tools

        try:
            from devin_local.settings import user_tools_dir

            udir = user_tools_dir()
        except Exception:  # noqa: BLE001 - settings dir not writable; skip.
            log.debug("user_tools_dir unavailable; skipping user-tool load")
            return None
        try:
            report = register_user_tools(self.registry, udir)
        except Exception:  # noqa: BLE001
            log.exception("user tool registration failed")
            return None
        for err in report.errors:
            log.warning("user tool %s: %s", err.source_path.name, err.message)
        return report

    def _load_plugins(self) -> None:
        plugins = discover_plugins(self.workspace / "plugins")
        for plugin in plugins:
            try:
                plugin.register_fn(self)
            except Exception as exc:  # noqa: BLE001
                log.warning("plugin %s failed to register: %s", plugin.name, exc)
                continue
            self.plugins_loaded.append(plugin.name)

    def _connect_mcp(self) -> None:
        cfg_path = self.workspace / "mcp_servers.json"
        specs = load_mcp_servers(cfg_path)
        if not specs:
            return
        manager = MCPClientManager()
        try:
            proxies = manager.connect_all(specs)
        except Exception as exc:  # noqa: BLE001
            log.warning("MCP connect failed: %s", exc)
            return
        for proxy in proxies:
            if proxy.name in self.registry:
                continue
            try:
                self.registry.register(proxy)
            except Exception:  # noqa: BLE001
                log.exception("Could not register MCP tool %s", proxy.name)
        self.mcp_manager = manager


# ---------- helpers for embedding callers (tests, plugins) ----------


def dump_messages(messages: list[ChatMessage]) -> str:
    """Render a chat history as JSON (useful for debugging)."""
    return json.dumps([m.to_dict() for m in messages], indent=2, default=str)
