"""CLI entry point for devin-local.

Commands:
    chat                Start an interactive REPL.
    run "task..."       Run a single task to completion, print final reply.
    doctor              Probe environment + Ollama health.
    models list         Show locally installed Ollama models.
    models suggest      Print recommended (incl. abliterated) models.
    knowledge add       Add a file/text to the knowledge store.
    knowledge list      List stored knowledge notes.

The CLI is implemented with `typer` and renders output via `rich`.
"""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from devin_local import __version__
from devin_local.agent import Agent, AgentConfig
from devin_local.inference.backend import BackendUnavailableError, ChatChunk
from devin_local.inference.factory import SUPPORTED_BACKENDS, list_available_backends
from devin_local.knowledge.store import KnowledgeStore
from devin_local.models import get_model_spec, suggest_abliterated
from devin_local.ollama_client import OllamaClient, OllamaError

DEFAULT_BACKEND = "ollama"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="devin-local — a 100% local autonomous AI software engineer (Ollama-backed).",
)
models_app = typer.Typer(help="Inspect / suggest Ollama models.")
knowledge_app = typer.Typer(help="Manage the local knowledge store.")
app.add_typer(models_app, name="models")
app.add_typer(knowledge_app, name="knowledge")


def _console() -> Console:
    """Build a console that handles Windows UTF-8 quirks."""
    if sys.platform.startswith("win"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    return Console()


def _build_agent(
    model: str,
    workspace: Path,
    host: str,
    *,
    obliteratus: bool,
    no_desktop: bool,
    no_browser: bool,
    no_plugins: bool,
    no_mcp: bool,
    no_knowledge: bool,
    no_skills: bool,
    session: Path | None,
    backend: str = DEFAULT_BACKEND,
    backend_options: dict | None = None,
    keep_alive: str | int | None = None,
) -> Agent:
    cfg = AgentConfig(
        model=model,
        workspace=workspace,
        backend=backend,
        backend_options=backend_options or {},
        ollama_host=host,
        keep_alive=keep_alive,
        enable_obliteratus=obliteratus,
        enable_desktop=not no_desktop,
        enable_browser=not no_browser,
        enable_plugins=not no_plugins,
        enable_mcp=not no_mcp,
        enable_knowledge_injection=not no_knowledge,
        enable_skill_injection=not no_skills,
        session_path=session,
    )
    return Agent(cfg)


def _validate_backend(name: str) -> str:
    name = (name or DEFAULT_BACKEND).lower()
    if name not in SUPPORTED_BACKENDS:
        raise typer.BadParameter(
            f"Unknown backend {name!r}. Choose one of: {', '.join(SUPPORTED_BACKENDS)}."
        )
    return name


def _backend_options(
    backend: str,
    *,
    model_path: str | None = None,
    compression: str | None = None,
    layer_cache: Path | None = None,
    device_map: str = "auto",
    load_4bit: bool = False,
    load_8bit: bool = False,
    trust_remote_code: bool = False,
) -> dict:
    """Assemble the kwargs dict passed to `build_backend(backend, **opts)`."""
    if backend == "layered":
        opts: dict = {}
        if model_path:
            opts["model_id"] = model_path
        if compression:
            opts["compression"] = None if compression.lower() == "none" else compression
        if layer_cache:
            opts["layer_cache_dir"] = str(layer_cache)
        return opts
    if backend == "hf":
        opts = {
            "device_map": device_map,
            "load_in_4bit": load_4bit,
            "load_in_8bit": load_8bit,
            "trust_remote_code": trust_remote_code,
        }
        if model_path:
            opts["model_id"] = model_path
        return opts
    return {}


def _parse_keep_alive(raw: str | None) -> str | int | None:
    """Pass numeric `keep_alive` as int, duration strings (e.g. '10m') as str."""
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def _streaming_printer(console: Console):
    """Return a stream-observer that prints token deltas live."""

    def _emit(chunk: ChatChunk) -> None:
        if chunk.done:
            return
        if chunk.delta:
            console.print(chunk.delta, end="", soft_wrap=True, highlight=False)

    return _emit


@app.callback()
def _global() -> None:
    """devin-local — root."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def doctor(
    host: str = typer.Option("http://127.0.0.1:11434", help="Ollama host URL."),
) -> None:
    """Run a health check: Python + every backend + installed models."""
    console = _console()
    table = Table(title="devin-local doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")

    table.add_row("Python", "ok", sys.version.split()[0])

    client = OllamaClient(host=host)
    try:
        available = client.is_available()
    except OllamaError as exc:
        available = False
        err = str(exc)
    else:
        err = ""
    if available:
        try:
            models = client.list_models()
            table.add_row("Ollama", "ok", f"{len(models)} models at {host}")
            for m in models:
                table.add_row("model", "installed", m)
        except OllamaError as exc:
            table.add_row("Ollama", "fail", str(exc))
    else:
        table.add_row("Ollama", "unreachable", err or host)
        table.add_row(
            "Hint",
            "—",
            "Install from https://ollama.com/download and run `ollama serve`.",
        )
    client.close()

    table.add_row("", "", "")
    for entry in list_available_backends():
        status = "ok" if entry["available"] else "missing"
        table.add_row(f"backend:{entry['name']}", status, entry["details"])

    console.print(table)


@models_app.command("list")
def models_list(
    host: str = typer.Option("http://127.0.0.1:11434"),
) -> None:
    """List locally installed Ollama models."""
    console = _console()
    client = OllamaClient(host=host)
    try:
        installed = client.list_models()
    except OllamaError as exc:
        console.print(f"[red]Could not reach Ollama:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        client.close()
    if not installed:
        console.print("No models installed. Run e.g. `ollama pull llama3.1:8b`.")
        return
    table = Table(title="Installed models")
    table.add_column("Name")
    table.add_column("Context")
    table.add_column("Family")
    table.add_column("Abliterated")
    for name in installed:
        spec = get_model_spec(name)
        table.add_row(
            name,
            f"{spec.context_window:,}",
            spec.family or "?",
            "yes" if spec.abliterated else "no",
        )
    console.print(table)


@models_app.command("suggest")
def models_suggest() -> None:
    """Show recommended abliterated / uncensored models for full-autonomy mode."""
    console = _console()
    abliterated = suggest_abliterated()
    if not abliterated:
        console.print("(none registered)")
        return
    table = Table(title="OBLITERATUS-style models")
    table.add_column("Name")
    table.add_column("Context")
    table.add_column("Pull command")
    for spec in abliterated:
        table.add_row(spec.name, f"{spec.context_window:,}", f"ollama pull {spec.name}")
    console.print(table)
    console.print(
        Panel(
            "These models have had refusal directions removed via abliteration "
            "(see https://github.com/elder-plinius/OBLITERATUS for the technique). "
            "They are most useful when you want the agent to operate without "
            "any built-in content gating on a local task.",
            title="About",
        )
    )


@knowledge_app.command("add")
def knowledge_add(
    title: str = typer.Argument(..., help="Short title for the note."),
    body: str | None = typer.Argument(None, help="Body text. If omitted, read from stdin."),
    file: Path | None = typer.Option(None, "--file", "-f", help="Read body from a file."),
    scope: str = typer.Option("", help="Comma-separated scope hints."),
    tags: str = typer.Option("", help="Comma-separated tags."),
    workspace: Path = typer.Option(Path.cwd(), help="Workspace root."),
) -> None:
    """Add a note to the knowledge store."""
    console = _console()
    if file is not None:
        body = file.read_text(encoding="utf-8")
    if not body:
        body = sys.stdin.read()
    if not body.strip():
        console.print("[red]No body supplied.[/red]")
        raise typer.Exit(1)
    store = KnowledgeStore.open(workspace / "knowledge" / "store.jsonl")
    note = store.add(
        title=title,
        body=body,
        scope=scope,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
    )
    console.print(f"Added note [bold]{note.id}[/bold]: {note.title}")


@knowledge_app.command("list")
def knowledge_list(
    workspace: Path = typer.Option(Path.cwd(), help="Workspace root."),
) -> None:
    """List stored notes."""
    console = _console()
    store = KnowledgeStore.open(workspace / "knowledge" / "store.jsonl")
    notes = store.all()
    if not notes:
        console.print("(no notes)")
        return
    table = Table(title=f"{len(notes)} knowledge note(s)")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Scope")
    table.add_column("Tags")
    for note in notes:
        table.add_row(note.id, note.title, note.scope, ", ".join(note.tags))
    console.print(table)


def _print_turn(console: Console, name: str, text: str) -> None:
    console.print(Panel(Markdown(text or "(empty)"), title=name, border_style="cyan"))


@app.command()
def run(
    task: str = typer.Argument(..., help="The task instruction for the agent."),
    model: str = typer.Option("llama3.1:8b", "--model", "-m"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    host: str = typer.Option("http://127.0.0.1:11434", "--host"),
    backend: str = typer.Option(
        DEFAULT_BACKEND,
        "--backend",
        "-b",
        envvar="DEVIN_LOCAL_DEFAULT_BACKEND",
        help="Inference backend: ollama | layered | hf.",
    ),
    model_path: str | None = typer.Option(
        None,
        "--model-path",
        help="For layered/hf: HuggingFace model id or local path.",
    ),
    compression: str = typer.Option("4bit", "--compression", help="Layered: 4bit | 8bit | none."),
    layer_cache: Path | None = typer.Option(
        None, "--layer-cache", help="Layered: directory to cache per-layer shards."
    ),
    device_map: str = typer.Option("auto", "--device-map", help="HF: auto|cpu|cuda|..."),
    load_4bit: bool = typer.Option(False, "--load-4bit", help="HF: bitsandbytes 4-bit."),
    load_8bit: bool = typer.Option(False, "--load-8bit", help="HF: bitsandbytes 8-bit."),
    trust_remote_code: bool = typer.Option(
        False, "--trust-remote-code", help="HF: allow custom model code."
    ),
    keep_alive: str | None = typer.Option(
        None, "--keep-alive", help="Ollama: keep model resident (e.g. '10m', '-1')."
    ),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream tokens."),
    obliteratus: bool = typer.Option(True, "--obliteratus/--no-obliteratus"),
    no_desktop: bool = typer.Option(False, "--no-desktop"),
    no_browser: bool = typer.Option(False, "--no-browser"),
    no_plugins: bool = typer.Option(False, "--no-plugins"),
    no_mcp: bool = typer.Option(False, "--no-mcp"),
    no_knowledge: bool = typer.Option(False, "--no-knowledge"),
    no_skills: bool = typer.Option(False, "--no-skills"),
    session: Path | None = typer.Option(None, "--session"),
) -> None:
    """Run a single task to completion and print the final assistant reply."""
    console = _console()
    backend = _validate_backend(backend)
    backend_options = _backend_options(
        backend,
        model_path=model_path,
        compression=compression,
        layer_cache=layer_cache,
        device_map=device_map,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        trust_remote_code=trust_remote_code,
    )
    agent = _build_agent(
        model=model,
        workspace=workspace,
        host=host,
        backend=backend,
        backend_options=backend_options,
        keep_alive=_parse_keep_alive(keep_alive),
        obliteratus=obliteratus,
        no_desktop=no_desktop,
        no_browser=no_browser,
        no_plugins=no_plugins,
        no_mcp=no_mcp,
        no_knowledge=no_knowledge,
        no_skills=no_skills,
        session=session,
    )

    def _observer(tool_name: str, args, result) -> None:  # noqa: ANN001
        kind = "ok" if result.ok else "err"
        console.print(f"[dim][tool {tool_name} -> {kind}][/dim]")

    agent.add_tool_observer(_observer)
    if stream:
        agent.add_stream_observer(_streaming_printer(console))
    try:
        start = time.monotonic()
        turn = agent.handle_user(task, stream=stream)
        elapsed = time.monotonic() - start
    except (OllamaError, BackendUnavailableError) as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        agent.shutdown()
    if stream:
        console.print()  # flush trailing newline after streamed output
    _print_turn(console, "assistant", turn.assistant_text)
    console.print(
        f"[dim]({turn.iterations} iterations, "
        f"{len(turn.tool_results)} tool call(s), {elapsed:.1f}s, backend={backend})[/dim]"
    )


@app.command()
def chat(
    model: str = typer.Option("llama3.1:8b", "--model", "-m"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    host: str = typer.Option("http://127.0.0.1:11434", "--host"),
    backend: str = typer.Option(
        DEFAULT_BACKEND,
        "--backend",
        "-b",
        envvar="DEVIN_LOCAL_DEFAULT_BACKEND",
        help="Inference backend: ollama | layered | hf.",
    ),
    model_path: str | None = typer.Option(None, "--model-path"),
    compression: str = typer.Option("4bit", "--compression"),
    layer_cache: Path | None = typer.Option(None, "--layer-cache"),
    device_map: str = typer.Option("auto", "--device-map"),
    load_4bit: bool = typer.Option(False, "--load-4bit"),
    load_8bit: bool = typer.Option(False, "--load-8bit"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
    keep_alive: str | None = typer.Option(None, "--keep-alive"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    obliteratus: bool = typer.Option(True, "--obliteratus/--no-obliteratus"),
    no_desktop: bool = typer.Option(False, "--no-desktop"),
    no_browser: bool = typer.Option(False, "--no-browser"),
    no_plugins: bool = typer.Option(False, "--no-plugins"),
    no_mcp: bool = typer.Option(False, "--no-mcp"),
    no_knowledge: bool = typer.Option(False, "--no-knowledge"),
    no_skills: bool = typer.Option(False, "--no-skills"),
    session: Path | None = typer.Option(None, "--session"),
) -> None:
    """Start an interactive chat REPL with the agent."""
    console = _console()
    backend = _validate_backend(backend)
    backend_options = _backend_options(
        backend,
        model_path=model_path,
        compression=compression,
        layer_cache=layer_cache,
        device_map=device_map,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        trust_remote_code=trust_remote_code,
    )
    agent = _build_agent(
        model=model,
        workspace=workspace,
        host=host,
        backend=backend,
        backend_options=backend_options,
        keep_alive=_parse_keep_alive(keep_alive),
        obliteratus=obliteratus,
        no_desktop=no_desktop,
        no_browser=no_browser,
        no_plugins=no_plugins,
        no_mcp=no_mcp,
        no_knowledge=no_knowledge,
        no_skills=no_skills,
        session=session,
    )
    if stream:
        agent.add_stream_observer(_streaming_printer(console))

    def _observer(tool_name: str, args, result) -> None:  # noqa: ANN001
        kind = "[green]ok[/green]" if result.ok else "[red]err[/red]"
        console.print(f"[dim][tool {tool_name} -> {kind}][/dim]")

    agent.add_tool_observer(_observer)
    console.print(
        Panel(
            f"devin-local v{__version__} — model [bold]{model}[/bold] @ {host}\n"
            f"Workspace: {workspace}\n"
            "Type your task. `/exit` to quit, `/reset` to clear history.",
            title="ready",
            border_style="green",
        )
    )
    try:
        from prompt_toolkit import PromptSession  # imported lazily for fast `--help`

        prompter = PromptSession()
    except Exception:  # noqa: BLE001
        prompter = None
    try:
        while True:
            try:
                user = prompter.prompt("you> ") if prompter is not None else input("you> ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not user.strip():
                continue
            if user.strip() in {"/exit", "/quit"}:
                break
            if user.strip() == "/reset":
                agent.messages = []
                agent.initialize()  # rebuild system prompt
                console.print("[dim](history reset)[/dim]")
                continue
            try:
                turn = agent.handle_user(user, stream=stream)
            except (OllamaError, BackendUnavailableError) as exc:
                console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
                continue
            if stream:
                console.print()
            _print_turn(console, "assistant", turn.assistant_text)
    finally:
        agent.shutdown()


@app.command("backends")
def backends_list() -> None:
    """List available inference backends with install / runtime status."""
    console = _console()
    table = Table(title="inference backends")
    table.add_column("Backend")
    table.add_column("Status")
    table.add_column("Details")
    for entry in list_available_backends():
        status = "[green]ready[/green]" if entry["available"] else "[yellow]missing[/yellow]"
        table.add_row(entry["name"], status, entry["details"])
    console.print(table)


@app.command("install")
def install_extras(
    extras: list[str] = typer.Argument(
        ...,
        help="Pip extras to install (e.g. layered hf gui). Use 'all' for everything.",
    ),
) -> None:
    """Install optional dependencies (layered, hf, gui, all) via pip.

    Streams pip's output directly. Equivalent to::

        pip install -e .[extras]   # editable install
        pip install devin-local[extras]   # installed package

    Use this to enable the AirLLM-style layered backend or the HuggingFace
    backend after the base install.
    """
    from devin_local.installer import install_extras_sync, pip_command

    norm: list[str] = []
    for extra in extras:
        for chunk in extra.split(","):
            chunk = chunk.strip()
            if chunk:
                norm.append(chunk)
    if "all" in norm:
        norm = ["layered", "hf", "gui"]
    if not norm:
        typer.echo("Specify at least one extra (e.g. layered, hf, gui).", err=True)
        raise typer.Exit(2)
    typer.echo("pip command: " + " ".join(pip_command(tuple(norm))))
    rc = install_extras_sync(tuple(norm))
    if rc != 0:
        raise typer.Exit(rc)


@app.command("gui")
def gui(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    backend: str = typer.Option(DEFAULT_BACKEND, "--backend", "-b"),
    model: str = typer.Option("llama3.1:8b", "--model", "-m"),
    host: str = typer.Option("http://127.0.0.1:11434", "--host"),
) -> None:
    """Launch the PySide6 desktop GUI."""
    try:
        from devin_local.gui.app import launch_gui
    except ImportError as exc:
        typer.echo(
            "GUI dependencies not installed. Install with:\n"
            "    pip install devin-local[gui]\n"
            f"(import error: {exc})",
            err=True,
        )
        raise typer.Exit(2) from exc
    launch_gui(workspace=workspace, backend=_validate_backend(backend), model=model, host=host)


def main() -> None:
    """Module-level entry point used by the `devin-local` console script."""
    app()


def main_layered() -> None:
    """Console-script entry point for `devin-local-layered`.

    Sets the ``DEVIN_LOCAL_DEFAULT_BACKEND`` env var to ``layered`` so the
    `run` / `chat` commands default to AirLLM-style loading. Users can still
    override per-invocation with ``--backend ollama`` to fall back to the
    Ollama daemon.
    """
    import os

    os.environ["DEVIN_LOCAL_DEFAULT_BACKEND"] = "layered"
    app()


def main_gui() -> None:
    """Console-script entry point for `devin-local-gui` (no subcommand needed)."""
    try:
        from devin_local.gui.app import launch_gui
    except ImportError as exc:
        sys.stderr.write(
            "GUI dependencies not installed. Install with:\n"
            "    pip install devin-local[gui]\n"
            f"(import error: {exc})\n"
        )
        sys.exit(2)
    launch_gui()


if __name__ == "__main__":
    main()
