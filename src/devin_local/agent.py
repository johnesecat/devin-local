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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devin_local.context_manager import ContextManager, make_default_context_manager
from devin_local.knowledge.store import KnowledgeStore
from devin_local.mcp.client import MCPClientManager, load_mcp_servers
from devin_local.ollama_client import (
    ChatMessage,
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
    num_ctx: int | None = None  # override Ollama's context window
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurn:
    """The outcome of one user message."""

    assistant_text: str
    tool_results: list[tuple[str, ToolResult]] = field(default_factory=list)
    iterations: int = 0


ToolObserver = Callable[[str, dict[str, Any], ToolResult], None]


class Agent:
    """Core devin-local agent."""

    def __init__(
        self,
        config: AgentConfig,
        client: OllamaClient | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config
        self.workspace = config.workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.client = client or OllamaClient(host=config.ollama_host)
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
        if config.session_path is not None:
            self.session = SessionStore.open(config.session_path)
            self.messages = list(self.session.messages)
        self.context_manager: ContextManager = make_default_context_manager(
            config.model, self._summarize
        )
        self._initialized = False

    # ---------- public API ----------

    def add_tool_observer(self, observer: ToolObserver) -> None:
        self._observers.append(observer)

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
        """Tear down MCP, terminals, and the HTTP client."""
        try:
            if self.mcp_manager is not None:
                self.mcp_manager.shutdown()
        except Exception:  # noqa: BLE001
            log.exception("MCP shutdown failed")
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            log.exception("Ollama client close failed")

    def handle_user(self, text: str) -> AgentTurn:
        """Run one user → assistant turn (with any tool calls)."""
        self.initialize()
        user_msg = ChatMessage(role="user", content=text)
        self._append(user_msg)
        self._inject_relevant_context(text)
        self.messages = self.context_manager.maybe_compact(self.messages)
        if self.session is not None:
            self.session.replace_all(self.messages)

        tool_results: list[tuple[str, ToolResult]] = []
        for iteration in range(self.config.max_iterations):
            response = self._call_model()
            assistant_msg = response.message
            self._append(assistant_msg)

            if not assistant_msg.tool_calls:
                return AgentTurn(
                    assistant_text=assistant_msg.content.strip(),
                    tool_results=tool_results,
                    iterations=iteration + 1,
                )

            for call in assistant_msg.tool_calls:
                fn = (call.get("function") or {}) if isinstance(call, dict) else {}
                name = fn.get("name") or call.get("name") or ""
                raw_args = fn.get("arguments", {})
                arguments = parse_tool_call_arguments(raw_args)
                result = self.registry.dispatch(name, arguments)
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

    def _append(self, message: ChatMessage) -> None:
        self.messages.append(message)
        if self.session is not None:
            self.session.append(message)

    def _call_model(self):  # noqa: ANN202
        options: dict[str, Any] = {"temperature": self.config.temperature}
        if self.config.num_ctx is not None:
            options["num_ctx"] = self.config.num_ctx
        options.update(self.config.extra_options)
        try:
            return self.client.chat(
                model=self.config.model,
                messages=self.messages,
                tools=self.registry.to_ollama_schemas(),
                options=options,
            )
        except OllamaError:
            raise

    def _summarize(self, text: str, max_tokens: int) -> str:
        try:
            return self.client.summarize(self.config.model, text, max_tokens=max_tokens)
        except OllamaError as exc:
            return f"(summary failed: {exc})"

    def _ensure_system_prompt(self) -> None:
        if self.messages and self.messages[0].role == "system":
            return
        ctx = PromptContext(
            workspace=self.workspace,
            model=self.config.model,
            tool_names=self.registry.names(),
            enable_obliteratus=self.config.enable_obliteratus,
            extra_sections=self.system_prompt_sections,
        )
        prompt = build_system_prompt(ctx)
        self.messages.insert(0, ChatMessage(role="system", content=prompt))
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
