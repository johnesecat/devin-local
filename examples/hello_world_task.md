# Example: Hello World task

This is the verification task referenced in the README. Run it like so:

```
devin-local run "Create a file hello.py that prints 'Hello, world!', then run it with python and confirm the output." --model llama3.1:8b
```

The agent should:

1. Call `write_file` to create `hello.py` with a single `print(...)` line.
2. Call `shell_exec` with `python hello.py`.
3. Observe stdout `Hello, world!` and return a brief confirmation.

If your machine doesn't have `llama3.1:8b` yet, pull a smaller one:

```
ollama pull qwen2.5-coder:7b
devin-local run "..." --model qwen2.5-coder:7b
```

Smaller models occasionally need a hint — e.g. "Use the write_file tool to
create the file, then use shell_exec to run it" — to chain tool calls
reliably.
