# devin-local

A **100% local, free, and private** autonomous AI software-engineer agent.
No API keys. No logins. No subscriptions. Your code stays on your machine.

Three pluggable inference backends, one agent:

- **Ollama** (default) — fast, daemon-based, recommended for everyday use.
- **Layered (AirLLM)** — layer-by-layer quantized loading so 70B-class
  models fit on 8 GB of RAM. Slow per token, but unlocks big models on
  laptops.
- **HF Transformers** — HuggingFace + `accelerate` + optional 4-bit
  `bitsandbytes`. Good middle ground when you have a small GPU.

devin-local borrows ideas from Devin and Claude Code: a tool-using agent
loop with file, shell, browser, and desktop primitives — plus plugins, MCP
servers, knowledge upload, skills, model-aware context compression, and
streaming + parallel tool dispatch so sessions feel snappy.

Ships both as a terminal CLI and a native **PySide6 desktop app**.

> **Built primarily for Windows 10/11.** Also works on Linux and macOS.

---

## Features

- **Local-first.** Runs entirely against an Ollama daemon on your machine.
  Works fully offline once your model is pulled.
- **Tool-calling agent harness.** File ops (`read_file`, `write_file`,
  `edit_file`, `list_dir`, `find_files`, `grep`), shell exec, persistent
  shell sessions, Python REPL, web fetch + search, and desktop control.
- **Terminal & desktop emulators.** Persistent PowerShell / bash sessions
  the agent can drive across multiple turns, plus screenshot + mouse +
  keyboard control via `pyautogui` (Windows-native).
- **OBLITERATUS framework integration.** A prompt-level directive that
  configures the model as a fully autonomous operator extension, with a
  registry pointing at abliterated/uncensored models on Ollama.
- **Context compression.** Model-aware: looks up each model's context
  window and summarizes older turns once usage crosses a configurable
  threshold (default 70%). Lets long sessions keep running.
- **Plugins.** Drop a Python file in `plugins/` with a `register(agent)`
  function, or register via the `devin_local.plugins` entry-point group.
- **MCP (Model Context Protocol).** Connect to any stdio MCP server via
  `mcp_servers.json`; their tools are exposed under `mcp__<server>__<tool>`.
- **Knowledge upload.** `devin-local knowledge add` (CLI) or **Settings →
  Knowledge** (GUI) ingests text/markdown files. All notes are embedded
  into the system prompt **once at session start** — the agent does NOT
  re-search the corpus per turn, which preserves your token budget and
  keeps the model's prompt cache warm. Workspace notes go in
  `<workspace>/knowledge/store.jsonl`; cross-workspace user notes go in
  `~/.devin-local/knowledge/store.jsonl` (use `--user` on the CLI).
- **Skills.** Markdown-with-YAML-frontmatter recipes; matched skills are
  auto-injected into the system prompt when their scope hits the task.
- **Session persistence.** Optional JSONL transcript per session for
  resuming work after a restart.

---

## Quick start (Windows 10/11)

### 1. Install Ollama

Download and install Ollama from <https://ollama.com/download/windows>.
Then in a new PowerShell window, pull a model:

```powershell
ollama pull llama3.1:8b
# Or, for a small/fast option:
ollama pull qwen2.5-coder:7b
```

Ollama runs as a background service on Windows by default.

### 2. Install devin-local

```powershell
git clone https://github.com/johnesecat/devin-local
cd devin-local
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[all]"
```

The `[all]` extra installs desktop, browser, MCP, and GUI support. Use
`pip install -e .` for a minimal install.

Optional extras (install only what you need):

| Extra       | What it pulls in                                                |
|-------------|------------------------------------------------------------------|
| `[gui]`     | PySide6 — the desktop application.                              |
| `[layered]` | AirLLM + torch + transformers — layer-by-layer quantized loading. |
| `[hf]`      | torch + transformers + bitsandbytes — HF backend with 4-/8-bit. |
| `[desktop]` | pyautogui + pillow — screenshot / mouse / keyboard tools.       |
| `[browser]` | playwright + bs4 — web fetch + search.                          |
| `[mcp]`     | mcp — connect to Model Context Protocol servers.                |

### 3. Run a health check

```powershell
devin-local doctor
```

You should see your Python version, every backend's status (ollama daemon,
layered, hf), and a list of installed Ollama models.

List just the backends:

```powershell
devin-local backends
```

### 4. Launch the desktop app

```powershell
pip install -e ".[gui]"
devin-local-gui
```

The app has:

- **Sidebar**: sessions + Settings.
- **Main chat pane**: streaming token render, syntax-highlighted code
  blocks, collapsible tool-call cards with file-output previews, plus a
  composer (**Ctrl+Enter to send**).
- **Right inspector**: backend dropdown, model selector (live from Ollama
  `/api/tags`, plus a "Browse library…" dialog), live `<plan>` pane,
  sandbox panel (workspace + terminal + desktop + tools + flags), and a
  workspace file tree.
- **Settings dialog** (gear button in the sidebar): tabs for **General**
  (workspace, defaults, parallel tool calls), **MCP** (add/edit/remove/test
  MCP servers), **GitHub** (paste a PAT, "Test connection" hits `/user`),
  **Backends** (probe + one-click install for `layered` / `hf` extras),
  and **Appearance** (theme + font scale + accent).
- The visual system is documented at `docs/design-system.md` and the
  tokens live in `src/devin_local/gui/design_tokens.py`.

On Windows you can also launch `devin-local-gui-app` (no console window),
useful for Start Menu / desktop shortcuts.

---

## Choosing a backend

| Backend  | Best for                                        | RAM      | Speed  | Setup                          |
|----------|-------------------------------------------------|----------|--------|--------------------------------|
| `ollama` | Everyday work, agent loops, fast iteration.     | ~8 GB    | Fast   | Install Ollama, `ollama pull`. |
| `layered`| Running 70B+ models on ≤8 GB of RAM.            | ~8 GB    | Slow   | `pip install ".[layered]"`     |
| `hf`     | Small GPU + 4-bit quantization, or pure CPU.    | ~16 GB+  | Medium | `pip install ".[hf]"`          |

**Two launcher prefixes** so you can switch without typing flags every time:

```powershell
devin-local         run "..."      # Ollama (default)
devin-local-layered run "..."      # Layered / AirLLM
```

The `--backend` flag still works on either entry point if you want to
override per invocation:

```powershell
devin-local         run --backend layered --model-path Qwen/Qwen2.5-7B-Instruct "…"
devin-local-layered run --backend ollama  --model llama3.1:8b "…"
devin-local         run --backend hf --model-path gpt2 --device-map cpu "hello"
```

### Layered (AirLLM) tradeoffs

The layered backend loads **one transformer layer at a time**, runs it,
frees it, and moves on. That's how it fits a 70B model in 8 GB of RAM —
but also why it's slow per token (often tens of seconds per token on
CPU; faster on GPU). Use it when model size matters more than latency.

### Windows notes for `bitsandbytes`

`bitsandbytes >= 0.43` ships CPU kernels and works on Windows out of the
box. Earlier versions require CUDA. If you hit `OSError: cannot find
libbitsandbytes_*.dll`, upgrade with `pip install -U bitsandbytes`.

### 4. Try the verification task

```powershell
devin-local run "Create a file hello.py that prints 'Hello, world!', then run it with python and confirm the output." --model llama3.1:8b
```

You should end up with a `hello.py` in your workspace and a final
assistant message confirming the output.

### 5. Or open an interactive chat

```powershell
devin-local chat --model llama3.1:8b
```

Type your task; the agent will call tools and report back. `/exit` quits,
`/reset` clears the conversation.

---

## Quick start (Linux / macOS)

Identical, except the venv activation is `source .venv/bin/activate` and
Ollama installs via `curl -fsSL https://ollama.com/install.sh | sh`.

---

## CLI reference

```
devin-local doctor                         # health check
devin-local version                        # print version
devin-local models list                    # show installed Ollama models
devin-local models suggest                 # list abliterated / uncensored models
devin-local knowledge add "Title" "Body"   # add a note
devin-local knowledge list                 # show stored notes
devin-local run "task..."                  # one-shot task
devin-local chat                           # interactive REPL
```

Key flags (shared by `run` and `chat`):

| Flag                       | Effect                                        |
| -------------------------- | --------------------------------------------- |
| `--model`                  | Ollama model name (default `llama3.1:8b`).    |
| `--workspace`              | Workspace root for tools (default cwd).       |
| `--host`                   | Ollama URL (default `http://127.0.0.1:11434`).|
| `--obliteratus/--no-obliteratus` | Toggle OBLITERATUS directive.            |
| `--no-desktop`             | Disable desktop emulator tools.               |
| `--no-browser`             | Disable `web_fetch` / `web_search`.           |
| `--no-plugins`             | Skip plugin discovery.                        |
| `--no-mcp`                 | Skip MCP server connections.                  |
| `--no-knowledge`           | Don't auto-inject knowledge notes.            |
| `--no-skills`              | Don't auto-inject skills.                     |
| `--session PATH`           | Persist + resume the transcript at PATH.      |

---

## Configuration

devin-local reads configuration from a few well-known files in your
workspace root:

- **`plugins/*.py`** — Python plugins (see `plugins/example_plugin.py`).
- **`skills/*.md`** — User skills (markdown with YAML frontmatter).
- **`knowledge/store.jsonl`** — Knowledge store (managed by the CLI).
- **`mcp_servers.json`** — MCP server definitions (see the bundled
  example). Requires `pip install -e ".[mcp]"`.

---

## Plugins

```python
# plugins/my_plugin.py
from devin_local.tools.base import Tool, ToolResult, text_result

class CurrentTimeTool(Tool):
    name = "current_time"
    description = "Return the current local date/time."
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments):
        from datetime import datetime
        return text_result(datetime.now().isoformat(timespec="seconds"))

def register(agent):
    agent.registry.register(CurrentTimeTool())
```

Plugins can also extend the system prompt by appending to
`agent.system_prompt_sections` before the first call to
`agent.initialize()`.

---

## MCP

Edit `mcp_servers.json`:

```json
{
  "servers": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  ]
}
```

Install the MCP extra:

```powershell
pip install -e ".[mcp]"
```

Their tools will show up as `mcp__filesystem__<tool>`.

---

## Skills

Add `skills/my-skill.md`:

```markdown
---
name: deploy-frontend
description: How to deploy the frontend.
scope: deploy, frontend, vercel
---

Run `pnpm build` then `vercel --prod`.
```

Skills whose scope matches your task get auto-injected into the system
prompt for that turn.

---

## Knowledge

```
devin-local knowledge add "Project layout" --file ARCHITECTURE.md
devin-local knowledge add "Build" "Run cargo build to compile" --scope rust,build
devin-local knowledge list
```

Top-K relevant notes are injected per turn (TF-IDF retrieval, no
embeddings needed).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          Agent loop                          │
│                                                              │
│   user msg → system prompt + retrieval → Ollama /api/chat    │
│      ↑                                          │            │
│      │                                          ▼            │
│      │                                    tool_calls?        │
│      │                              yes ──────┘ no           │
│      │                              │           │            │
│      │                              ▼           ▼            │
│      │                       ToolRegistry    final reply     │
│      │                       (dispatch)         │            │
│      │                              │           │            │
│      └──── compact context ◀────────┘           ▼            │
│             (when > threshold)            return turn        │
└──────────────────────────────────────────────────────────────┘
```

Key modules:

- `src/devin_local/agent.py` — the harness.
- `src/devin_local/ollama_client.py` — Ollama HTTP wrapper with tool calls.
- `src/devin_local/system_prompt.py` — prompt builder.
- `src/devin_local/obliteratus.py` — OBLITERATUS directive integration.
- `src/devin_local/context_manager.py` — token-budget compaction.
- `src/devin_local/tools/` — built-in tools.
- `src/devin_local/emulators/` — terminal + desktop emulators.
- `src/devin_local/plugins/` — plugin loader.
- `src/devin_local/mcp/` — MCP stdio client.
- `src/devin_local/knowledge/` — knowledge store + retrieval.
- `src/devin_local/skills/` — skill loader.

---

## Development

```powershell
pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

CI runs the same on Ubuntu and Windows for Python 3.10 and 3.12 (see
`.github/workflows/ci.yml`).

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Inspiration

- [Devin](https://devin.ai) for the autonomous-engineer interaction model.
- [Claude Code](https://www.anthropic.com/claude-code) for the tool harness shape.
- [Claw Code](https://github.com/ultraworkers/claw-code) for harness implementation details.
- [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) for the
  "unrestricted local operator" framing.
