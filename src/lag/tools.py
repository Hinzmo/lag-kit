"""Tool functions for local-agent-kit.

These are plain Python functions with type hints and docstrings so they can be:
  - passed directly to Ollama, which auto-generates the JSON schema, and
  - wrapped unchanged by the FastMCP server (servers/tools_server.py).

One source of truth, two consumers. Security hardening (sandboxing run_shell,
path allow-lists) is deliberately deferred — see the morning brief.
"""
from __future__ import annotations
import json
import os
import platform
import subprocess
from pathlib import Path

_MEMORY = Path("memory/store.json")


def read_file(path: str) -> str:
    """Read a UTF-8 text file and return its contents.

    Args:
        path: Path to the file to read.
    """
    return Path(path).read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    """Write text to a file, creating parent directories as needed.

    Args:
        path: Destination file path.
        content: Text content to write.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def list_dir(path: str = ".") -> str:
    """List the entries in a directory, one per line.

    Args:
        path: Directory to list. Defaults to the current directory.
    """
    return "\n".join(sorted(os.listdir(path))) or "(empty)"


def run_shell(command: str) -> str:
    """Run a shell command and return its combined stdout and stderr.

    Args:
        command: The shell command to execute.
    """
    r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
    return (r.stdout + r.stderr).strip() or f"(exit code {r.returncode}, no output)"


def system_info() -> str:
    """Return basic information about the host system as JSON."""
    return json.dumps(
        {
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        indent=2,
    )


def remember(key: str, value: str) -> str:
    """Store a key/value pair in persistent local memory.

    Args:
        key: The identifier to store under.
        value: The value to remember.
    """
    _MEMORY.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(_MEMORY.read_text()) if _MEMORY.exists() else {}
    store[key] = value
    _MEMORY.write_text(json.dumps(store, indent=2))
    return f"remembered '{key}'"


def recall(key: str) -> str:
    """Retrieve a value previously stored with remember.

    Args:
        key: The identifier to look up.
    """
    if not _MEMORY.exists():
        return "(memory empty)"
    return json.loads(_MEMORY.read_text()).get(key, f"(no memory for '{key}')")


TOOLS = {
    fn.__name__: fn
    for fn in (read_file, write_file, list_dir, run_shell, system_info, remember, recall)
}


class LocalToolProvider:
    """Phase 1 provider: dispatches straight to in-process Python functions.

    Implements the same (schemas, call) interface as MCPToolProvider so the agent
    loop never needs to know which one it's holding.
    """

    def __init__(self, functions: dict | None = None):
        self.functions = functions or TOOLS

    def schemas(self):
        # Ollama derives the JSON schema from each function's signature and
        # docstring, so we hand it the function objects directly.
        return list(self.functions.values())

    def call(self, name: str, args: dict) -> str:
        fn = self.functions.get(name)
        if fn is None:
            return f"unknown tool: {name}"
        try:
            return str(fn(**(args or {})))
        except Exception as e:  # surface the error to the model instead of raising
            return f"error in {name}: {e}"
