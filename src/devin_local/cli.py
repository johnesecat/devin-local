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
from devin_local.knowledge.store import KnowledgeStore
from devin_local.models import get_model_spec, suggest_abliterated
from devin_local.ollama_client import OllamaClient, OllamaError

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
) -> Agent:
    cfg = AgentConfig(
        model=model,
        workspace=workspace,
        ollama_host=host,
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
    """Run a health check: Ollama reachable + models installed + python version."""
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
    agent = _build_agent(
        model=model,
        workspace=workspace,
        host=host,
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
    try:
        start = time.monotonic()
        turn = agent.handle_user(task)
        elapsed = time.monotonic() - start
    except OllamaError as exc:
        console.print(f"[red]Ollama error:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        agent.shutdown()
    _print_turn(console, "assistant", turn.assistant_text)
    console.print(
        f"[dim]({turn.iterations} iterations, "
        f"{len(turn.tool_results)} tool call(s), {elapsed:.1f}s)[/dim]"
    )


@app.command()
def chat(
    model: str = typer.Option("llama3.1:8b", "--model", "-m"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    host: str = typer.Option("http://127.0.0.1:11434", "--host"),
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
    agent = _build_agent(
        model=model,
        workspace=workspace,
        host=host,
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
                turn = agent.handle_user(user)
            except OllamaError as exc:
                console.print(f"[red]Ollama error:[/red] {exc}")
                continue
            _print_turn(console, "assistant", turn.assistant_text)
    finally:
        agent.shutdown()


def main() -> None:
    """Module-level entry point used by the console script."""
    app()


if __name__ == "__main__":
    main()
