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
    temperature: float = 0.2
    session_path: Path | None = None
    num_ctx: int | None = None  # override the backend's context window
    keep_alive: str | int | None = None  # Ollama: keep model resident between turns
    parallel_tool_calls: bool = True
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
        self._ensure_system_prompt()
        self._initialized = True

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

    def handle_user(self, text: str, *, stream: bool = False) -> AgentTurn:
        """Run one user → assistant turn (with any tool calls).

        Set ``stream=True`` to receive incremental token deltas via any
        registered stream observers. The returned :class:`AgentTurn` is
        identical either way.
        """
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
            assistant_msg = self._call_model(stream=stream)
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

            if self.config.parallel_tool_calls and len(calls_to_dispatch) > 1:
                results = self.registry.dispatch_many(calls_to_dispatch)
            else:
                results = [self.registry.dispatch(n, a) for n, a in calls_to_dispatch]

            # Fire tool_started observers before dispatching.
            for name, arguments in calls_to_dispatch:
                for sobs in self._tool_start_observers:
                    try:
                        sobs(name, arguments)
                    except Exception:  # noqa: BLE001
                        log.exception("tool start observer raised")

            for (name, arguments), result in zip(calls_to_dispatch, results, strict=False):
                tool_results.append((name, result))
                for observer in self._observers:
                    try:
                        observer(name, arguments, result)
                    except Exception:  # noqa: BLE001
                        log.exception("tool observer raised")
                self._append(
                    ChatMessage(
                        role="tool",
                        name=name,
                        content=result.to_chat_payload(),
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

    def _call_model(self, *, stream: bool = False) -> ChatMessage:
        options: dict[str, Any] = {"temperature": self.config.temperature}
        if self.config.num_ctx is not None:
            options["num_ctx"] = self.config.num_ctx
        options.update(self.config.extra_options)
        tools = self.registry.to_ollama_schemas()
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

    def _ensure_system_prompt(self) -> None:
        if self.messages and self.messages[0].role == "system":
            return
        # Cache the system prompt: tools and skills are stable across turns,
        # so rebuilding the prompt every turn just wastes CPU and (worse)
        # changes the prompt-cache hash on the backend, defeating Ollama's
        # KV cache. Only rebuild when the tool set changes.
        tool_names = tuple(self.registry.names())
        if self._cached_system_prompt is None or self._cached_tool_names_snapshot != tool_names:
            ctx = PromptContext(
                workspace=self.workspace,
                model=self.config.model,
                tool_names=list(tool_names),
                enable_obliteratus=self.config.enable_obliteratus,
                extra_sections=self.system_prompt_sections,
            )
            self._cached_system_prompt = build_system_prompt(ctx)
            self._cached_tool_names_snapshot = tool_names
        self.messages.insert(0, ChatMessage(role="system", content=self._cached_system_prompt))
        if self.session is not None:
            self.session.replace_all(self.messages)

    def _inject_relevant_context(self, query: str) -> None:
        """Insert top-K knowledge notes + matching skills before the user turn."""
        if not (self.config.enable_knowledge_injection or self.config.enable_skill_injection):
            return
        injections: list[str] = []
        if self.config.enable_knowledge_injection:
            notes = self.knowledge.search(query, k=3)
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
