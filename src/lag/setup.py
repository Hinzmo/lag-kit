"""`lag setup`: from a fresh clone to a working kit in one command.

Installs Ollama if it is missing, starts it, pulls the models the roles need, and
writes config.yaml. Re-running it is how you switch models: name a new tag and it is
pulled and mapped, or name nothing and it pulls whatever config.yaml already says.

Local models move fast, so nothing here is a fixed list. The defaults are just a
setup that was measured to work on a 16 GB Mac; any tag from the Ollama library or
from your own `ollama list` works.
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

DEFAULTS = {
    # One tag for every chat role: a run loads it once and never switches. Picked from
    # a five-model run on the M4 (tools 7/7, both real goals right, 18.6 tok/s); on
    # Apple Silicon Ollama serves it as an MLX build.
    "orchestrator": "qwen3.5:9b",
    "chat": "qwen3.5:9b",
    "embed": "nomic-embed-text",       # retrieval only
}
CHAT_ROLES = ("worker", "coder", "fast")
ROLES = ("orchestrator", *CHAT_ROLES, "embed")
MAC_APP_BINARY = "/Applications/Ollama.app/Contents/Resources/ollama"
DOWNLOAD_URL = "https://ollama.com/download"


def norm(tag: str) -> str:
    """Ollama treats a bare name as `:latest`; compare tags in that form."""
    return tag if ":" in tag else f"{tag}:latest"


def find_ollama(which=shutil.which) -> str | None:
    """The ollama binary, on PATH or inside the macOS app bundle."""
    found = which("ollama")
    if found:
        return found
    return MAC_APP_BINARY if Path(MAC_APP_BINARY).exists() else None


def install_command(system: str | None = None, which=shutil.which) -> list[str] | None:
    """How to install Ollama here, or None when it has to be done by hand."""
    system = system or platform.system()
    if system == "Darwin" and which("brew"):
        return ["brew", "install", "ollama"]
    if system == "Linux" and which("curl"):
        # Ollama's official installer; it asks for sudo itself when it needs it.
        return ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
    return None


def server_up(client) -> bool:
    try:
        client.list()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not reachable yet"
        return False


def start_server(binary: str, client, wait: float = 20.0, popen=subprocess.Popen,
                 sleep=time.sleep) -> bool:
    """Start `ollama serve` detached, then wait for it to answer."""
    popen([binary, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
          start_new_session=True)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if server_up(client):
            return True
        sleep(0.5)
    return server_up(client)


def installed(client) -> set[str]:
    return {norm(m.model) for m in client.list().models}


def pull(client, tag: str, echo) -> None:
    """Pull one tag, echoing progress at most once per 10%."""
    last = None
    for p in client.pull(tag, stream=True):
        status = p.status or ""
        if p.total and p.completed is not None:
            pct = int(100 * p.completed / p.total) // 10 * 10
            line = f"    {tag}: {status} {pct}%"
        else:
            line = f"    {tag}: {status}"
        if line != last:
            echo(line)
            last = line


def plan_roles(current: dict, orchestrator: str | None, chat: str | None,
               embed: str | None) -> dict:
    """The role → tag map to use. A flag wins, then config.yaml, then the defaults,
    so a re-run with no flags never overwrites a mapping you chose."""
    roles = {}
    roles["orchestrator"] = orchestrator or current.get("orchestrator") or DEFAULTS["orchestrator"]
    for r in CHAT_ROLES:
        roles[r] = chat or current.get(r) or DEFAULTS["chat"]
    roles["embed"] = embed or current.get("embed") or DEFAULTS["embed"]
    return roles


_ROLE_LINE = re.compile(r'^(\s+)(' + "|".join(ROLES) + r')(:\s*)(["\']?)([^"\'#\s]*)\4(.*)$')


def write_models(text: str, roles: dict) -> str:
    """Set each role's tag inside the `models:` block, keeping comments and layout."""
    out, in_models = [], False
    for line in text.splitlines(keepends=True):
        if re.match(r"^models:\s*(#.*)?$", line):
            in_models = True
        elif in_models and line.strip() and not line[0].isspace():
            in_models = False
        m = _ROLE_LINE.match(line.rstrip("\n")) if in_models else None
        if m and m.group(2) in roles:
            nl = "\n" if line.endswith("\n") else ""
            indent, role, sep, _, _, rest = m.groups()
            line = f'{indent}{role}{sep}"{roles[role]}"{rest}{nl}'
        out.append(line)
    return "".join(out)


def rival_servers(host: str, make_client=None) -> bool:
    """True when 127.0.0.1 and ::1 on this port are two Ollama servers that disagree.

    This happens on a Mac running both the Ollama app and a Homebrew `ollama serve`:
    the app listens on every address and brew on 127.0.0.1, so `localhost` and the
    `ollama` CLI can land on different servers. Each keeps its own model list in
    memory, so a model pulled through one is invisible to the other until it restarts.
    """
    from urllib.parse import urlsplit

    u = urlsplit(host)
    if u.hostname not in ("localhost", "127.0.0.1", "::1"):
        return False
    port = u.port or 11434
    if make_client is None:
        import ollama
        make_client = lambda h: ollama.Client(host=h)  # noqa: E731
    seen = []
    for h in (f"http://127.0.0.1:{port}", f"http://[::1]:{port}"):
        try:
            seen.append({m.model for m in make_client(h).list().models})
        except Exception:  # noqa: BLE001 - nothing listening there is the normal case
            return False
    return seen[0] != seen[1]


RIVAL_HINT = ("  ! two Ollama servers are answering on this port and their model lists differ\n"
              "    (often the Ollama app plus a Homebrew `ollama serve`). Quit one of them —\n"
              "    `brew services stop ollama`, or quit the app — so pulls show up everywhere.")
