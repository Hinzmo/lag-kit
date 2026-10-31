"""Confinement for agent mode. Not optional.

Supervised `lag run` and an unattended agent are different risks. A supervised run is
one command whose output a human reads; an agent is a loop that runs for hours with
nobody watching, now from a *saved* definition that can be started again tomorrow. The
earlier decision to give agent mode the same unsandboxed shell the CLI has was taken
before agents could be saved and left running, and it does not survive that change.

**The boundary is a container, not a pattern list.** A deny-list cannot be the boundary:
`rm -rf /` and `sh -c $'\\x72\\x6d ...'` are the same command and no regex catches both.
Docker gives a real one — separate filesystem, no host mounts beyond what we pass,
capabilities dropped, no network, and a pid limit so a fork bomb dies inside. The
deny-list kept here turns a likely accident into a clear message *before* it reaches
the container; it is defence in depth, not the thing that makes this safe.

A seatbelt (`sandbox-exec`) backend was tried first and abandoned: a profile strict
enough to be worth having killed `/bin/sh` with SIGABRT, and one permissive enough to
run was not worth trusting. Docker also works on Linux, where seatbelt does not exist
at all.

**It fails closed.** If no container runtime is reachable, agent mode refuses to start
rather than running unconfined — checked before the loop begins, not at the first shell
call when the agent is already several tool calls deep.

Two zones, because one would be useless: the agent READS the project it was pointed at
(mounted read-only) and WRITES only its own workspace. Confining reads to the workspace
would mean an agent that cannot look at the code it is meant to work on; allowing
writes anywhere would defeat the point. Tools that never reach a shell are checked in
Python, since they run in this process and no container covers them.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .providers import ProviderWrapper

SHELL_TIMEOUT = 60
DEFAULT_IMAGE = "python:3.12-slim"

# The egress proxy (egress.py) and the guard it runs (netguard.py), mounted read-only
# into the proxy container. Paths, not an installed package: the proxy image is bare.
_EGRESS_SRC = Path(__file__).with_name("egress.py")
_NETGUARD_SRC = Path(__file__).with_name("netguard.py")
EGRESS_PORT = 3128

# Where the two zones appear inside the container.
WORKSPACE_MNT = "/workspace"
PROJECT_MNT = "/project"

# ── the deny-list ─────────────────────────────────────────────────────────────
# Obvious footguns, refused before they reach the container so the agent gets a clear
# reason rather than a confusing error from inside. NOT the security boundary.
#
# Judged on the PARSED command — which program runs, with which arguments — never on
# substrings. Substring matching was wrong in both directions: `rm -rf /` is a prefix
# of `rm -rf /workspace/build` and `grep shutdown log.txt` contains `shutdown`, so
# ordinary work was refused, while `rm -r -f /` spells the same delete without the
# exact substring, so it passed. A bogus refusal is not free: the agent loses a cycle
# and learns nothing, since the reason names a command it never meant to run.
_PROGRAMS = {
    "sudo": "privilege escalation", "su": "privilege escalation",
    "doas": "privilege escalation",
    "fdisk": "disk administration", "sfdisk": "disk administration",
    "parted": "disk administration",
    "shutdown": "powers off the machine", "poweroff": "powers off the machine",
    "halt": "powers off the machine", "reboot": "restarts the machine",
}
_SHELLS = {"sh", "bash", "zsh", "dash"}
# Prefixes that run the command after them; the program that matters is that one.
_WRAPPERS = {"env", "nohup", "time", "exec", "command", "builtin"}
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_WORLD_WRITABLE = {"777", "0777", "a+rwx", "ugo+rwx"}
_HARMLESS_DEVICES = {"/dev/null", "/dev/zero", "/dev/stdout", "/dev/stderr"}


# `<<WORD`, `<<-WORD`, `<<'WORD'` — but not the here-string `<<<`.
_HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _strip_heredocs(command: str) -> str:
    """Drop heredoc bodies: they are data fed to a command, not commands. Agents
    write files this way, and a setup script that mentions sudo is not sudo."""
    out, pending = [], []
    for line in command.split("\n"):
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
            continue
        out.append(line)
        pending.extend(m.group(2) for m in _HEREDOC.finditer(line))
    return "\n".join(out)


def _tokens(command: str) -> list[str]:
    """Shell words and operators. Quoted text stays one word, so `grep '| sh'` holds
    a search string, not a pipe. Newlines and backticks also separate commands."""
    text = _strip_heredocs(command).replace("\n", " ; ").replace("`", " ; ")
    try:
        lx = shlex.shlex(text, posix=True, punctuation_chars=True)
        lx.whitespace_split = True
        return list(lx)
    except ValueError:        # unbalanced quote: the shell will report it; split crudely
        return re.findall(r"[;&|()<>]+|[^\s;&|()<>]+", text)


def _segments(command: str) -> list[tuple[list[str], bool]]:
    """Simple commands as (words, piped_into), split on ; && || | & ( ) and $( )."""
    out, cur, piped, next_piped = [], [], False, False
    skip_next = False
    for t in _tokens(command):
        if skip_next:                     # the target of a redirection, not an argument
            skip_next = False
            continue
        if t and set(t) <= set(";&|()<>"):
            if set(t) <= set("<>&") and ("<" in t or ">" in t):
                skip_next = True          # >, >>, <, >&, 2>&1's '>&'
                continue
            if cur:
                out.append((cur, piped))
            cur, piped = [], False
            next_piped = t in ("|", "|&")
            continue
        if t == "$":                      # the '$' of '$(' — the '(' splits next
            continue
        if not cur:
            piped, next_piped = next_piped, False
        cur.append(t)
    if cur:
        out.append((cur, piped))
    return out


def _program(words: list[str]) -> tuple[str, list[str]]:
    """(program name, its arguments), past env assignments and wrappers like `env`."""
    i = 0
    while i < len(words):
        w = words[i]
        if _ASSIGN.match(w):
            i += 1
        elif os.path.basename(w).lower() in _WRAPPERS:
            if w == "command" and i + 1 < len(words) and words[i + 1] in ("-v", "-V"):
                return "command", words[i + 1:]      # a lookup; nothing is run
            i += 1
            while i < len(words) and words[i].startswith("-"):
                i += 1
        else:
            return os.path.basename(w).lower(), words[i + 1:]
    return "", []


def _is_root(path: str) -> bool:
    return path == "/*" or (path.startswith("/") and os.path.normpath(path) in ("/", "//"))


def _deny_reason(command: str, depth: int = 0) -> str | None:
    for words, piped in _segments(command):
        prog, args = _program(words)
        if not prog:
            continue
        shown = " ".join(words)[:80]
        if prog in _SHELLS:
            if piped:
                return f"refusing to pipe into {prog} — runs whatever arrives on stdin"
            if "-c" in args and depth < 3:
                i = args.index("-c")
                if i + 1 < len(args):
                    why = _deny_reason(args[i + 1], depth + 1)
                    if why:
                        return why
            continue
        if prog in _PROGRAMS:
            return f"refusing to run {shown!r} — {_PROGRAMS[prog]}"
        if prog.startswith("mkfs"):
            return f"refusing to run {shown!r} — formats a filesystem"
        flags = [a for a in args if a.startswith("-")]
        operands = [a for a in args if not a.startswith("-")]
        if prog == "rm":
            recursive = any(f in ("-r", "-R", "--recursive") or
                            (not f.startswith("--") and ("r" in f or "R" in f))
                            for f in flags)
            if "--no-preserve-root" in flags or (recursive and any(map(_is_root, operands))):
                return f"refusing to run {shown!r} — recursive delete from root"
        if prog == "chmod" and _WORLD_WRITABLE & set(operands) \
                and any(map(_is_root, operands)):
            return f"refusing to run {shown!r} — world-writable root"
        if prog == "dd" and any(a.startswith("of=/dev/") and a[3:] not in _HARMLESS_DEVICES
                                for a in args):
            return f"refusing to run {shown!r} — raw write to a device"
    return None

# Docker Desktop does not always link its CLI onto PATH.
_CANDIDATES = [
    "docker", "podman",
    "/usr/local/bin/docker", "/opt/homebrew/bin/docker",
    str(Path.home() / ".docker/bin/docker"),
    "/Applications/Docker.app/Contents/Resources/bin/docker",
]


def find_runtime() -> str | None:
    """The container CLI, wherever it is installed."""
    for c in _CANDIDATES:
        p = shutil.which(c) if os.path.basename(c) == c else (c if os.path.isfile(c) else None)
        if p and os.access(p, os.X_OK):
            return p
    return None


def _env_for(cli: str) -> dict:
    """PATH with the CLI's own directory on it.

    Docker Desktop keeps `docker-credential-desktop` beside the `docker` binary and
    neither is necessarily on PATH. Invoking docker without its own directory visible
    makes any registry operation fail with a credential-helper error that says nothing
    about PATH — which is exactly how a first pull fails on a fresh machine.
    """
    env = dict(os.environ)
    d = os.path.dirname(os.path.abspath(cli))
    if d and d not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = d + os.pathsep + env.get("PATH", "")
    return env


def available() -> tuple[bool, str]:
    """(can we confine, why not) — the daemon must answer, not just the CLI exist."""
    cli = find_runtime()
    if not cli:
        return False, ("no container runtime found — install Docker Desktop, or start "
                       "it if it is installed")
    try:
        r = subprocess.run([cli, "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True, timeout=20,
                           env=_env_for(cli))
    except Exception as e:  # noqa: BLE001
        return False, f"could not reach the container runtime: {type(e).__name__}: {e}"
    if r.returncode != 0:
        return False, ("container runtime is installed but its daemon is not running — "
                       "start Docker Desktop and try again")
    return True, ""


class Sandbox:
    """A workspace an agent may write to, and a project it may read."""

    def __init__(self, workspace: str | Path, read_root: str | Path | None = None,
                 image: str = DEFAULT_IMAGE, network: bool = False):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.read_root = Path(read_root or Path.cwd()).resolve()
        self.image = image
        # The primitive defaults to sealed (no egress) — the conservative default for a
        # confinement class. The agent LAYER opts in (GoalSpec.network defaults on),
        # because a real agent needs pip/git/curl from its shell.
        #
        # network=True does NOT mean Docker's default bridge. The bridge NATs out
        # through the host, so the container reached everything the host can: measured,
        # `host.docker.internal:11434` answered with the host's Ollama. Instead the
        # container joins its own --internal network (no route anywhere) whose one
        # other member is an egress proxy applying netguard's public-only rule. See
        # egress.py. Download-and-run from the public internet remains possible; that
        # is what "network on" is for, and the deny-list is not a boundary against it.
        self.network = network
        self.cli = find_runtime()
        self.container: str | None = None
        self.proxy: str | None = None
        self.net: str | None = None

    # ── path checks, for tools that run in this process ──────────────────────
    def _inside(self, path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root)
            return True
        except (ValueError, OSError):
            return False

    def check_read(self, path: str) -> str | None:
        p = Path(path)
        if self._inside(p, self.read_root) or self._inside(p, self.workspace):
            return None
        return f"blocked: {path} is outside this agent's readable area ({self.read_root})"

    def check_write(self, path: str) -> str | None:
        if self._inside(Path(path), self.workspace):
            return None
        return (f"blocked: agents may only write inside their workspace "
                f"({self.workspace}); {path} is outside it")

    def check_glob(self, pattern: str) -> str | None:
        """Refuse a glob that tries to leave the directory it is searching.

        The search tools enforce containment themselves (tools._contained); this is the
        readable half. Silently filtering an escape returns "no matches", which a model
        reads as "the file isn't there". Saying why tells it to change approach.
        """
        p = (pattern or "").strip()
        if os.path.isabs(p) or p.startswith("~"):
            return (f"blocked: the glob {p!r} is absolute — pass the directory as "
                    f"`path` and keep the glob relative to it")
        if ".." in p.replace("\\", "/").split("/"):
            return (f"blocked: the glob {p!r} climbs out with '..' — search from a "
                    f"different `path` instead")
        return None

    def check_command(self, command: str) -> str | None:
        why = _deny_reason(command or "")
        return f"blocked: {why}" if why else None

    # ── container lifecycle ──────────────────────────────────────────────────
    def start(self) -> str | None:
        """Bring the container up. Returns an error string, or None on success.

        One long-lived container per run rather than one per command: `docker run` per
        call would cost seconds each time and lose every bit of state between commands,
        so `cd`, a venv, or a half-finished build would vanish between steps.
        """
        ok, why = available()
        if not ok:
            return why
        name = f"lag-agent-{uuid.uuid4().hex[:10]}"
        if self.network:
            err = self._start_egress(name)
            if err:
                self.stop()
                return err
        r = self._docker(self._run_argv(name), timeout=180)
        if r.returncode != 0:
            self.stop()
            err = (r.stderr or r.stdout).strip()
            if "No such image" in err or "not found" in err.lower():
                return (f"sandbox image {self.image!r} is not present — pull it first: "
                        f"docker pull {self.image}")
            return f"could not start the sandbox: {err[:300]}"
        self.container = name
        return None

    def _docker(self, argv: list, timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a docker command; a failure to even launch it comes back as a result."""
        try:
            return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                  env=_env_for(self.cli))
        except Exception as e:  # noqa: BLE001
            return subprocess.CompletedProcess(argv, 1, "", f"{type(e).__name__}: {e}")

    @staticmethod
    def _egress_names(agent: str) -> tuple[str, str]:
        """(internal network, proxy container) paired to one agent container's name."""
        run_id = agent.removeprefix("lag-agent-")
        return f"lag-net-{run_id}", f"lag-egress-{run_id}"

    def _proxy_argv(self, proxy: str) -> list:
        """The proxy container's `docker run` argv. Pure, like _run_argv, so its
        hardening is asserted in tests rather than hoped for.

        It starts on the default bridge — the only routed network it gets — and is
        joined to the agent's internal network afterwards. It sees neither the project
        nor the workspace: two read-only source files are its entire view of the host.
        """
        return [
            self.cli, "run", "-d", "--name", proxy,
            "--network", "bridge",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",                                # nothing to install, ever
            "--pids-limit", "64",
            "--memory", "256m", "--memory-swap", "256m",
            "--cpus", "0.5",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{_EGRESS_SRC}:/opt/egress/egress.py:ro",
            "-v", f"{_NETGUARD_SRC}:/opt/egress/netguard.py:ro",
            self.image, "python", "-u", "/opt/egress/egress.py",
        ]

    def _start_egress(self, agent: str) -> str | None:
        """Internal network + proxy, ready before the agent container joins it."""
        net, proxy = self._egress_names(agent)
        r = self._docker([self.cli, "network", "create", "--internal", net])
        if r.returncode != 0:
            return f"could not create the sandbox network: {(r.stderr or r.stdout).strip()[:300]}"
        self.net = net
        r = self._docker(self._proxy_argv(proxy), timeout=180)
        if r.returncode != 0:
            return f"could not start the egress proxy: {(r.stderr or r.stdout).strip()[:300]}"
        self.proxy = proxy
        r = self._docker([self.cli, "network", "connect", net, proxy])
        if r.returncode != 0:
            return f"could not attach the egress proxy: {(r.stderr or r.stdout).strip()[:300]}"
        # The agent's first pip install must not race the proxy's startup.
        probe = (f"import socket; socket.create_connection(('127.0.0.1', {EGRESS_PORT}), 1)")
        for _ in range(50):
            if self._docker([self.cli, "exec", proxy, "python", "-c", probe]).returncode == 0:
                return None
            time.sleep(0.2)
        logs = self._docker([self.cli, "logs", proxy]).stderr.strip()[-300:]
        return f"the egress proxy did not come up: {logs or 'no output'}"

    def _run_argv(self, name: str) -> list:
        """The `docker run` argv. Pure, so the confinement flags can be unit-tested
        without a daemon — the security of this list is the security of the sandbox.

        No published port (`-p`): egress, when on, is outbound only. `--network none`
        seals it when network is off; when on, the container joins an --internal network
        whose only way out is the egress proxy — never the default bridge.
        """
        cmd = [
            self.cli, "run", "-d", "--name", name,
            "--workdir", WORKSPACE_MNT,
            "-v", f"{self.workspace}:{WORKSPACE_MNT}",
            "-v", f"{self.read_root}:{PROJECT_MNT}:ro",   # the project, read-only
            "--cap-drop", "ALL",                          # no privileged operations
            "--security-opt", "no-new-privileges",        # and none can be regained
            "--pids-limit", "256",                        # a fork bomb dies inside
            "--memory", "2g",
            # Without this, docker lets memory-swap default to 2x --memory, so the cap is
            # really 4g (2g RAM + 2g swap). Equal to --memory means no swap: the 2g cap
            # is the actual ceiling, and a memory bomb is killed instead of thrashing.
            "--memory-swap", "2g",
            "--cpus", "2",                                # a busy loop can't peg every core
        ]
        # Deliberately NOT --read-only: agents pip-install / apt / build in their shell,
        # which needs a writable rootfs, and the container is ephemeral anyway (fresh per
        # run, torn down on stop), so a writable root buys an attacker nothing across runs.
        if self.network:
            net, proxy = self._egress_names(name)
            url = f"http://{proxy}:{EGRESS_PORT}"
            cmd += ["--network", net]                    # internal: no route but the proxy
            # Both spellings: curl reads only lowercase http_proxy, pip/requests read
            # either, and an empty no_proxy keeps everything going through the proxy.
            for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                cmd += ["-e", f"{var}={url}"]
            cmd += ["-e", "NO_PROXY=", "-e", "no_proxy="]
        else:
            cmd += ["--network", "none"]
        cmd += [self.image, "sleep", "infinity"]
        return cmd

    def stop(self) -> None:
        """Remove the container, its proxy and its network. Disposable by design —
        nothing survives but the workspace mount. Safe after a half-finished start."""
        if not self.cli:
            return
        for name in (self.container, self.proxy):
            if name:
                self._docker([self.cli, "rm", "-f", name])
        if self.net:      # after both containers, or docker refuses: endpoints in use
            self._docker([self.cli, "network", "rm", self.net])
        self.container = self.proxy = self.net = None

    def run_shell(self, command: str) -> str:
        """Run a command inside the container. Never touches the host shell."""
        refused = self.check_command(command)
        if refused:
            return refused
        if not self.container:
            return "blocked: the sandbox is not running, so no shell is available"
        try:
            r = subprocess.run(
                [self.cli, "exec", "-w", WORKSPACE_MNT, self.container,
                 "/bin/sh", "-c", command],
                capture_output=True, text=True, timeout=SHELL_TIMEOUT,
                env=_env_for(self.cli))
        except subprocess.TimeoutExpired:
            return f"(timed out after {SHELL_TIMEOUT}s)"
        except Exception as e:  # noqa: BLE001
            return f"run_shell error: {type(e).__name__}: {e}"
        out = (r.stdout + r.stderr).strip()
        return out or f"(exit code {r.returncode}, no output)"

    def describe(self) -> str:
        return (f"container {self.image} · writes {self.workspace} · "
                f"reads {self.read_root} (ro) · network "
                f"{'public internet via filtering proxy' if self.network else 'off'}")


_READ_PATHS = {"read_file": "path", "list_dir": "path",
               "find_files": "path", "grep_files": "path"}
_WRITE_PATHS = {"write_file": "path"}
# The search tools' second file-selecting argument. Checking only `path` approved the
# starting directory while the glob walked out of it.
_GLOB_ARGS = {"find_files": "pattern", "grep_files": "glob"}


class SandboxedProvider(ProviderWrapper):
    """A tool_provider confined to a sandbox.

    Wraps rather than replaces, like every other provider here, so the agent loop is
    unchanged and confinement holds over direct dispatch and MCP alike — including the
    case that matters most, where the tool would otherwise execute in an MCP server
    subprocess no container covers. Blocked calls come back as strings the model can
    work around: an agent told "that path is outside your workspace" writes elsewhere.
    """

    def __init__(self, inner, sandbox: Sandbox):
        super().__init__(inner)
        self.sandbox = sandbox

    def _translate(self, path: str, for_write: bool = False) -> str:
        """Container path -> host path.

        The agent lives in ONE namespace, the container's. Its shell reports
        /workspace, so that is what it then hands to write_file — but the file tools
        run on the host, where /workspace does not exist, and a relative path resolves
        against the repo rather than the workspace. Without translation an agent is
        stuck: every absolute path it knows is rejected and every relative one lands
        outside its writable area. Observed exactly that, looping until its cycle cap.

        The project fallback is for reads only. Together the two rules behave like
        copy-on-write, the way an overlay filesystem does: read README.md and you get
        the project's; write it and the edit lands in the workspace; read again and
        you get yours. Letting writes fall back too sent them to the read-only
        project, where they were refused every time.
        """
        p = (path or "").strip()
        if p == WORKSPACE_MNT or p.startswith(WORKSPACE_MNT + "/"):
            return str(self.sandbox.workspace / p[len(WORKSPACE_MNT):].lstrip("/"))
        if p == PROJECT_MNT or p.startswith(PROJECT_MNT + "/"):
            return str(self.sandbox.read_root / p[len(PROJECT_MNT):].lstrip("/"))
        if not os.path.isabs(p):
            # Relative means relative to the shell's cwd, which is the workspace —
            # except when it names something that only exists in the project, which is
            # how an agent asks to read source it was pointed at.
            in_ws = self.sandbox.workspace / p
            in_proj = self.sandbox.read_root / p
            if not for_write and not in_ws.exists() and in_proj.exists():
                return str(in_proj)
            return str(in_ws)
        return p

    def call(self, name: str, args: dict) -> str:
        args = dict(args or {})
        if name == "run_shell":
            return self.sandbox.run_shell(str(args.get("command", "")))

        key = _WRITE_PATHS.get(name) or _READ_PATHS.get(name)
        if key and args.get(key) is not None:
            args[key] = self._translate(str(args[key]), for_write=name in _WRITE_PATHS)

        key = _WRITE_PATHS.get(name)
        if key:
            refused = self.sandbox.check_write(str(args.get(key, "")))
            if refused:
                return refused
        key = _READ_PATHS.get(name)
        if key and args.get(key) is not None:
            refused = self.sandbox.check_read(str(args.get(key)))
            if refused:
                return refused
        key = _GLOB_ARGS.get(name)
        if key and args.get(key) is not None:
            refused = self.sandbox.check_glob(str(args[key]))
            if refused:
                return refused
        return self.inner.call(name, args)
