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
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

_MEMORY = Path("memory/store.json")

# web_search is the one tool that leaves the machine — the query goes to
# DuckDuckGo. Everything else here is local. Kept keyless and stdlib-only so it
# adds no dependency and no account: the privacy cost is exactly one outbound
# query, and nothing else about the kit's posture changes.
_DDG = "https://html.duckduckgo.com/html/"
_UA = "Mozilla/5.0 (X11; Linux x86_64) lag/0.2 (+local-agent-kit)"
_SEARCH_TIMEOUT = 15


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
    """Store a fact in persistent local memory.

    Use this whenever you learn something worth keeping — a preference, a
    decision, a finding from a search. Stored facts are indexed into the
    knowledge base, so they come back from a plain question later, not only
    from an exact key lookup.

    Args:
        key: A short label for the fact.
        value: The fact to remember.
    """
    _MEMORY.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(_MEMORY.read_text()) if _MEMORY.exists() else {}
    store[key] = value
    _MEMORY.write_text(json.dumps(store, indent=2))

    # Indexing is the default, not an option: a fact that only `recall` can
    # reach is a fact the model has to already know it stored. Kept lazy and
    # non-fatal so a core install (no chromadb, no embedder) still remembers.
    note = _index_memory(key, value)
    return f"remembered '{key}'{note}"


def _index_memory(key: str, value: str) -> str:
    """Mirror a remembered fact into the vector store. Never raises."""
    try:
        from . import rag
    except ImportError:
        return " (not indexed: retrieval extra not installed — pip install -e '.[rag]')"
    try:
        n = rag.index_note(key, value)
    except Exception as e:  # noqa: BLE001 - degrade, never break the loop
        return f" (not indexed: {type(e).__name__})"
    return f" · indexed {n} chunk(s) into the knowledge base" if n else ""


def recall(key: str) -> str:
    """Retrieve a value previously stored with remember.

    Args:
        key: The identifier to look up.
    """
    if not _MEMORY.exists():
        return "(memory empty)"
    return json.loads(_MEMORY.read_text()).get(key, f"(no memory for '{key}')")


class _ResultParser(HTMLParser):
    """Pull result titles, URLs, and snippets out of DuckDuckGo's HTML endpoint.

    A parser rather than a regex: the markup nests tags inside titles, and the
    snippet element is sometimes <a> and sometimes <div>. Matching on the class
    attribute survives both.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._mode: str | None = None
        self._tag: str = ""
        self._depth: int = 0
        self._href: str = ""
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = dict(attrs)
        cls = a.get("class") or ""
        if self._mode:
            # DuckDuckGo wraps query terms in <b> inside both titles and
            # snippets. Count nesting so an inner </b> doesn't end the capture.
            if tag == self._tag:
                self._depth += 1
            return
        if "result__a" in cls:
            self._mode, self._tag, self._depth = "title", tag, 1
            self._href, self._buf = a.get("href") or "", []
        elif "result__snippet" in cls:
            self._mode, self._tag, self._depth, self._buf = "snippet", tag, 1, []

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._mode or tag != self._tag:
            return
        self._depth -= 1
        if self._depth > 0:
            return
        text = unescape(" ".join("".join(self._buf).split()))
        if self._mode == "title" and text:
            self.results.append({"title": text, "url": _unwrap(self._href), "snippet": ""})
        elif self._mode == "snippet" and self.results:
            self.results[-1]["snippet"] = text
        self._mode, self._tag, self._buf = None, "", []


def _unwrap(href: str) -> str:
    """Turn DuckDuckGo's /l/?uddg=<encoded> redirect into the real URL."""
    if "uddg=" not in href:
        return href
    target = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg")
    return urllib.parse.unquote(target[0]) if target else href


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo and return the top results.

    Use this for information that is not on this machine: current events, library
    documentation, unfamiliar APIs. Results are titles, URLs, and snippets — open
    a promising URL only if the snippet is not enough.

    Args:
        query: What to search for.
        max_results: How many results to return, 1-10. Defaults to 5.
    """
    query = (query or "").strip()
    if not query:
        return "(web_search: empty query)"
    try:
        n = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        n = 5

    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request(_DDG, data=data, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=_SEARCH_TIMEOUT) as r:
            html = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"(web_search: search engine returned HTTP {e.code} — try again shortly)"
    except urllib.error.URLError as e:
        # Offline is the expected state for this kit; say so plainly so the model
        # reports the limitation instead of inventing an answer.
        return f"(web_search: no network — {e.reason}. This machine may be offline.)"
    except Exception as e:  # noqa: BLE001 - never raise into the agent loop
        return f"(web_search: {type(e).__name__}: {e})"

    p = _ResultParser()
    p.feed(html)
    hits = [h for h in p.results if h["url"].startswith("http")][:n]
    if not hits:
        return f"(web_search: no results for '{query}')"

    out = [f"{len(hits)} result(s) for '{query}':", ""]
    for i, h in enumerate(hits, 1):
        out.append(f"{i}. {h['title']}")
        out.append(f"   {h['url']}")
        if h["snippet"]:
            out.append(f"   {h['snippet']}")
        out.append("")
    return "\n".join(out).strip()


TOOLS = {
    fn.__name__: fn
    for fn in (read_file, write_file, list_dir, run_shell, system_info,
               remember, recall, web_search)
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
