---
name: testing-devin-local
description: End-to-end test the devin-local CLI agent against a real local Ollama backend. Use when verifying the agent loop, tool calling, or any change to OllamaClient / Agent / tools.
---

# Testing devin-local end-to-end against real Ollama

The app under test is a CLI that drives a local Ollama model through a
tool-calling loop. The user-visible surface is the terminal, so a recorded
GUI-terminal session is the right way to demonstrate the primary flow.

## What this skill covers

- Installing Ollama on a fresh Linux VM (Windows target ships through CI only)
- Picking a model that (a) fits in low-RAM VMs and (b) emits Ollama-structured
  `tool_calls` (not all tool-capable models do!)
- A known-good Hello World task wording that survives JSON escaping
- Independent re-verification of the agent's claimed file output
- Testing the knowledge-embedding (embed-once) invariant
- Testing CLI streaming flush through a real subprocess pipe

## Environment setup (Linux VM)

```bash
# 1) Ollama needs zstd for its tarball
sudo apt-get install -y zstd wmctrl

# 2) Install Ollama (creates systemd service, binds 127.0.0.1:11434)
curl -fsSL https://ollama.com/install.sh | sudo sh
systemctl is-active ollama   # -> active
curl -s http://127.0.0.1:11434/api/version

# 3) Install devin-local from the cloned repo
cd /home/ubuntu/repos/devin-local
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[all]"
```

Doctor must be green before testing the agent:

```bash
.venv/bin/devin-local doctor
```

Expected: `Python ok`, `Ollama ok (... models at http://127.0.0.1:11434)`, the
chosen model `installed`.

## Model selection — IMPORTANT

Not every Ollama model with a "tools" capability actually emits structured
`tool_calls`. The `OllamaClient` in this repo only parses `message.tool_calls`,
so models that emit tool calls as raw JSON inside `message.content` look like
broken tool-calling to the agent (0 tool calls dispatched).

Verified behavior on Ollama 0.23.2:

| Model            | Size  | Structured `tool_calls`? | Notes                                        |
|------------------|-------|--------------------------|----------------------------------------------|
| `llama3.1:8b`    | ~5GB  | Yes                      | README default. Needs ~6GB RAM during infer. |
| `llama3.2:3b`    | ~2GB  | Yes                      | Tool calls fire; small models occasionally mangle string args. |
| `qwen2.5-coder:3b` | ~2GB | **No**                   | Emits inline JSON in `content`; agent dispatches 0 tools. |

For a small-RAM VM (~8GB), `llama3.2:3b` is the fastest, but `llama3.1:8b`
matches the README default and is the most faithful proof. Pull with:

```bash
ollama pull llama3.1:8b
```

If the model emits inline JSON rather than structured tool_calls, do NOT
assume the agent is broken — switch the model and re-run.

Quick tool-calling probe (no need to run the full agent):

```bash
curl -s http://127.0.0.1:11434/api/chat -d '{
  "model":"<MODEL>",
  "stream":false,
  "messages":[{"role":"user","content":"Use the add tool to compute 17 + 25."}],
  "tools":[{"type":"function","function":{"name":"add","description":"Add","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}}]
}' | python3 -c "import sys,json; r=json.load(sys.stdin); m=r['message']; print('content:', repr(m.get('content',''))[:120]); print('tool_calls:', m.get('tool_calls'))"
```

If `tool_calls` is `None` and the JSON is inside `content`, that model will not
work with devin-local as currently written.

## Primary test — Hello World end-to-end

In a maximized GUI terminal (Konsole on Plasma) — record this:

```bash
rm -rf /tmp/devin_test_ws && mkdir /tmp/devin_test_ws
.venv/bin/devin-local run \
  --model llama3.1:8b \
  --workspace /tmp/devin_test_ws \
  --no-desktop --no-browser --no-plugins --no-mcp --no-knowledge --no-skills \
  "Use the write_file tool to create hello.py in the workspace. The file should contain a python program that prints the text: Hello, world!  Then use the shell_exec tool to run it with: python3 hello.py  Then reply with the captured stdout, prefixed with: STDOUT="
```

Wording matters:

- Avoid having the model emit `print('Hello, world!')` literally inside JSON
  args — small models sometimes truncate at the embedded single quote and
  write `print(` to disk. Describe the desired *output* ("prints the text:
  Hello, world!") and let the model pick the Python syntax.
- Pass `--no-knowledge --no-skills --no-plugins --no-mcp --no-desktop
  --no-browser` to make the system prompt minimal — reduces inference time
  and isolates the test to the core agent loop.

Then verify independently (DO NOT trust the agent's self-report):

```bash
ls -la /tmp/devin_test_ws/
cat /tmp/devin_test_ws/hello.py
python3 /tmp/devin_test_ws/hello.py     # must print exactly: Hello, world!
```

## Pass criteria

- Run footer shows `≥ 2 tool call(s)` — proves write_file + shell_exec fired
- Final `assistant` panel contains the substring `Hello, world!`
- `/tmp/devin_test_ws/hello.py` exists on disk
- `python3 .../hello.py` prints `Hello, world!` and exits 0

## Testing the embed-knowledge-once invariant

When the user uploads knowledge (via CLI `knowledge add` or GUI Settings →
Knowledge), every note is embedded into the system prompt **once** at session
start. The agent must NOT re-search knowledge per turn.

### Unit + integration level

```bash
QT_QPA_PLATFORM=offscreen pytest tests/test_embedded_knowledge.py -v
QT_QPA_PLATFORM=offscreen pytest tests/test_gui_settings_dialog.py -v -k "knowledge"
```

These test files cover:
- Notes appear verbatim in `messages[0].content` after `Agent.initialize()`
- Exactly 1 `role="system"` message after multiple turns (cache is stable)
- Zero `<retrieved-knowledge>` per-turn markers in embed mode
- Legacy mode (`embed_all_knowledge=False`) still does per-turn TF-IDF
- `max_embedded_knowledge_chars` caps without mid-note truncation
- GUI Knowledge tab persistence + `knowledge_changed` signal propagation

### Adversarial runtime tests (shell scripts)

The key invariants to check at runtime:

1. **F1 — note body inlined into `messages[0]`:** Construct an `Agent` with
   a user-store note containing a unique sentinel, call `initialize()`, then
   assert the sentinel is in `messages[0].content` and `## Injected knowledge`
   header is present.
2. **F2 — embed-once holds across N turns:** Drive N fake user turns (mock
   the backend to return empty assistant messages), then assert exactly 1
   `role="system"` message and zero `<retrieved-knowledge>` blocks.
3. **F4 — GUI refresh path:** Open `SettingsDialog` offscreen, add a note
   via the Knowledge tab, emit `knowledge_changed`, then assert the sentinel
   appeared in the running agent's `messages[0]` — still exactly 1 system msg.

These are scriptable Python harnesses that construct `Agent` or `MainWindow`
directly — no subprocess needed. Use `QT_QPA_PLATFORM=offscreen` for any
test that imports PySide6.

### Known CLI bug: `--workspace` flag collision

`knowledge_add` has both `workspace: Path = typer.Option(Path.cwd(), …)` and
`user: bool = typer.Option(False, "--user/--workspace", …)`. Typer binds
`--workspace` to the boolean (negative of `--user`), so the path option is
inaccessible. **Workaround:** use `--user` alone — it writes to
`~/.devin-local/knowledge/store.jsonl` regardless of cwd. The `--workspace`
path option is unreachable until the flags are renamed.

## Testing CLI streaming flush

The fix forces `line_buffering=True` on stdout and calls `console.file.flush()`
after every token delta. To prove this end-to-end through a real OS pipe:

### F3 — unit level

```python
from unittest.mock import MagicMock
from devin_local.cli import _streaming_printer
from devin_local.inference.backend import ChatChunk

console = MagicMock()
console.file = MagicMock()
emit = _streaming_printer(console)
for tok in ("a", "b", "c"):
    emit(ChatChunk(delta=tok, done=False))
assert console.file.flush.call_count == 3  # one per delta
emit(ChatChunk(delta="", done=True))
assert console.file.flush.call_count == 3  # terminator must NOT flush
```

### F6-synthetic — subprocess pipe observation

Spawn a child Python process that imports `_console()` + `_streaming_printer()`
from `devin_local.cli`, emits N `ChatChunk(delta=..., done=False)` with 0.5 s
sleeps, then a `done=True` terminator. In the parent, read stdout via
`select` + `os.read`. A broken flush (no `flush()` call) would buffer all
N deltas (typically < 4 KiB total) into a single chunk at child exit. A
working flush produces N distinct arrivals spread over ~N×0.5 s.

This is the strongest E2E test of the flush fix that doesn't require running
real Ollama inference.

## Regression checks (shell-only, no recording)

```bash
.venv/bin/python -m pytest -q          # expect 171+ passed
.venv/bin/ruff check .                 # expect: All checks passed!
.venv/bin/ruff format --check .        # expect: N files already formatted

# GitHub Actions CI on main
gh api repos/johnesecat/devin-local/actions/runs?branch=main \
  --jq '.workflow_runs[0] | {conclusion,head_sha}'
```

## Performance / resource expectations

- **With embedded prompt (~20k chars / ~5k tokens):** llama3.1:8b on CPU
  takes **>3 min just for prompt evaluation** before the first token. This
  makes live E2E testing impractical on CPU-only VMs. Use
  `--no-knowledge --no-skills` to strip the prompt down for faster runs,
  or use a GPU-backed `OLLAMA_HOST`.
- **Minimal prompt (--no-knowledge --no-skills etc.):** llama3.1:8b on CPU
  ~3 min for a 2-step Hello World task; llama3.2:3b ~2 min.
- Working set during inference: llama3.1:8b ~6.1 GiB RSS; llama3.2:3b ~2.5 GiB.
- When prompt eval dominates wall time, the streaming flush test (F6) can't
  observe bytes at t < 3 s because no tokens have been generated yet. Use
  the synthetic F6 subprocess test instead — it proves the same code path
  without model inference.

## Common pitfalls

- Ollama install fails with `zstd` error → `sudo apt-get install -y zstd` first.
- `wmctrl: command not found` when maximizing → `sudo apt-get install -y wmctrl`.
- `xdotool type --window $WIN --delay 30 "text"` can sometimes leak the
  delay value into the typed text in Konsole; if that happens, retry without
  `--delay`.
- Konsole opens two windows on first launch — close one with
  `wmctrl -ic <wid>` and activate+maximize the other.
- 8B model on a 8GB-RAM VM is tight; if Ollama OOMs, downshift to llama3.2:3b
  but accept slightly lower arg-formatting quality.
- **`--workspace` flag in `knowledge add` is broken** (shadowed by
  `--user/--workspace` boolean). Use `--user` alone or `cd` into the
  workspace directory and omit the flag.
- **Subprocess Popen with `bufsize=0`** gives a `FileIO` object, not
  `BufferedReader`. Use `os.read(proc.stdout.fileno(), N)` instead of
  `proc.stdout.read1(N)` when reading with non-blocking/select loops.
- **QT_QPA_PLATFORM=offscreen** is required for any test that imports PySide6
  on a headless VM. The offscreen backend renders nothing visible so
  screenshots are meaningless — test Qt widgets programmatically instead.

## Devin Secrets Needed

None. Everything is local. The whole point of the project is 100% offline
execution.
