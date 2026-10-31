"""Agent mode: one persona pursuing a goal on its own, for as long as you let it.

The pipeline is a *bounded* shape — a fixed set of stages, one pass, done. Agent mode
is the unbounded one: a goal, a toolset, and a loop that keeps working until the goal
is met or you stop it. That difference drives every design choice here.

**It must be able to stop.** An autonomous loop with no termination condition burns
tokens until a step cap and calls that success. `mark_done` is the condition: the model
declares the goal met and the loop exits. A cycle cap and a stop file are the backstops.

**It must survive its own failures.** A 24/7 run that dies on one bad cycle is not 24/7.
Every cycle is wrapped: an exception is recorded as an observation the next cycle can
read and react to, then the loop backs off and continues. That mirrors the tool layer's
rule — errors come back as text the model can work with, never as a raised exception.

**It must be inspectable while running.** State is written to disk after every cycle, so
`lag agent status`, the console, and a human with `cat` all see the same record, and a
restart resumes rather than starting over.

Confinement is mandatory here, not configurable: unlike a supervised `lag run`, an agent
is an unattended loop started from a saved definition that can be run again tomorrow, so
its shell is a container's — read-only on the project it was pointed at, writable only in
its own workspace, no host or LAN access, and at most the public internet through a
filtering proxy (egress.py). See sandbox.py for why that is the boundary
and the deny-list is not. The stop file and the cycle cap remain the brakes on the loop
itself; the container is the brake on what a cycle can touch.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from . import tools as tools_mod
from .config import load_persona
from .ollama_client import OllamaClient
from .providers import ProviderWrapper, ToolsExcept, ToolSubset, schema_name
from .sandbox import Sandbox, SandboxedProvider
from .sandbox import available as sandbox_available
from .tools import LocalToolProvider

try:
    import fcntl
except ImportError:          # Windows has no flock; the web server's own guard still holds
    fcntl = None  # type: ignore[assignment]

# Definitions and run state live together, paired by name, because they are always
# read together and two near-identical directories (agent/ vs agents/) would be a
# standing trap. The definition is yours to keep and commit; the state is per-machine
# and gitignored.
#     agents/<name>.yaml         what the agent IS
#     agents/<name>.state.json   what it has DONE
#     agents/<name>.stop         a request to stop
AGENT_DIR = Path("agents")

# The provider wrappers moved to providers.py; re-exported because the agent loop is
# where most of them are composed, and this is the obvious place to look for them.
_schema_name = schema_name
__all__ = ["AGENT_DIR", "AgentLoop", "AgentState", "GoalSpec", "GoalToolsLocal",
           "ToolSubset", "ToolsExcept", "running_agent", "stop_path"]

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_name(name: str) -> str:
    """Names become file paths, so refuse anything that could escape the directory."""
    if not _SAFE_NAME.match(name or ""):
        raise ValueError(
            f"invalid agent name {name!r}: use letters, digits, dot, dash, underscore")
    return name


@dataclass
class GoalSpec:
    """What the agent is trying to do, and the bounds you put on it."""

    goal: str
    success: str = ""            # what "done" looks like, in the model's own terms
    constraints: str = ""        # what it must not do
    persona: str = "dev_agent"   # who drives; dev_agent is fast and does not ruminate
    interval: int = 0            # seconds to pause between cycles
    max_cycles: int = 20  # backstop PER RUN (resumes get a fresh budget); 0 = no cap
    name: str = "default"
    model: str = ""              # override the persona's tag for this run
    # Two independent circuit breakers. Consecutive, not total: one failure at cycle 3
    # and another at cycle 400 is a healthy agent; five in a row is a broken one.
    max_failures: int = 5        # consecutive cycles that raised
    max_idle: int = 5            # consecutive cycles that did nothing at all
    # A saved agent is a thing you own: it can be switched off without being deleted,
    # and it carries its own note about what it is for.
    enabled: bool = True
    description: str = ""
    mcp: bool = True
    # Where the agent may write. Per-agent so two agents cannot tread on each other,
    # and outside the repo so a runaway cannot fill the project with scratch files.
    workspace: str = ""
    # Whether the agent's shell can reach the PUBLIC internet. On by default: in practice
    # an agent needs to pip-install, clone a repo, or call an API from its shell, and
    # sealing that off made agent mode unworkable in real use. On never means the host or
    # LAN — the container's only route is the egress proxy, which applies the same
    # public-only rule as http_fetch (see egress.py). web_search and http_fetch run in
    # the harness process regardless; this flag governs the CONTAINER shell. Set it
    # false to seal a particular agent completely.
    network: bool = True
    # Allow-list of tool names. EMPTY MEANS EVERY TOOL — the default decides nothing,
    # so narrowing is always a deliberate act rather than something inherited from a
    # template. An unknown name here is a typo that would silently remove a tool, so
    # it is reported rather than ignored.
    tools: list[str] = field(default_factory=list)

    # ── persistence ──────────────────────────────────────────────────────────
    @classmethod
    def path_for(cls, name: str) -> Path:
        return AGENT_DIR / f"{_check_name(name)}.yaml"

    def save(self) -> Path:
        """Write the definition as YAML — human-editable and diffable, like personas."""
        p = self.path_for(self.name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(asdict(self), sort_keys=False, width=100))
        return p

    @classmethod
    def load(cls, name: str) -> "GoalSpec":
        p = cls.path_for(name)
        if not p.exists():
            raise FileNotFoundError(f"no agent named {name!r} (looked in {p})")
        raw = yaml.safe_load(p.read_text()) or {}
        known = {f.name for f in fields(cls)}
        # Ignore unknown keys rather than exploding: these files are hand-edited, and a
        # stray field should not make an agent unloadable.
        return cls(**{k: v for k, v in raw.items() if k in known})

    @classmethod
    def list_all(cls) -> list["GoalSpec"]:
        if not AGENT_DIR.is_dir():
            return []
        out = []
        for f in sorted(AGENT_DIR.glob("*.yaml")):
            try:
                out.append(cls.load(f.stem))
            except Exception:  # noqa: BLE001 - one broken file must not hide the rest
                continue
        return out

    def delete(self) -> None:
        """Remove the definition and everything the run left behind."""
        for f in (self.path_for(self.name),
                  AGENT_DIR / f"{self.name}.state.json",
                  AGENT_DIR / f"{self.name}.stop"):
            f.unlink(missing_ok=True)

    def unknown_tools(self) -> list[str]:
        """Names in `tools` that do not exist — a typo silently removes a tool."""
        return [t for t in self.tools if t not in tools_mod.TOOLS
                and t != "search_knowledge_base"]

    def prompt(self, cycle: int, notes: list[str], last: str = "") -> str:
        """The per-cycle brief.

        Deliberately restates the goal every cycle. The agent has no conversation
        memory across cycles — each one is a fresh turn — so the goal, the progress
        log and the last observation ARE its continuity.
        """
        parts = [
            f"You are working autonomously towards a goal. This is cycle {cycle}.",
            f"\nGOAL: {self.goal}",
        ]
        if self.success:
            parts.append(f"DONE WHEN: {self.success}")
        if self.constraints:
            parts.append(f"CONSTRAINTS: {self.constraints}")
        if notes:
            recent = "\n".join(f"  - {n}" for n in notes[-12:])
            parts.append(f"\nPROGRESS SO FAR:\n{recent}")
        if last:
            parts.append(f"\nLAST CYCLE ENDED WITH:\n{last[:1500]}")
        # The agent must be told the shape of its world, or it will guess from `pwd`
        # and then hand a container path to a host tool.
        shell_net = ("Your shell reaches the PUBLIC internet only, through an HTTP(S) proxy"
                     " that is already configured: pip, git over https, curl and wget work."
                     " The host machine, the LAN and private addresses are refused (403),"
                     " and tools that ignore HTTP_PROXY (raw sockets, ssh) have no route."
                     if self.network else "There is no network in the shell.")
        parts.append(
            "\nYOUR ENVIRONMENT: your shell runs in a container. You may WRITE only in"
            " /workspace (your shell starts there, and bare relative paths mean there)."
            " You may READ the project at /project, but never write to it. "
            + shell_net + " There is no access to the host machine."
        )
        parts.append(
            "\nTake the next concrete step. Use your tools to actually do the work —"
            " do not just describe what you would do. Call note_progress when you"
            " accomplish something so the next cycle knows. When the goal is genuinely"
            " met, call mark_done. If you are blocked, say why and try another route."
        )
        return "\n".join(parts)


@dataclass
class AgentState:
    spec: dict
    cycle: int = 0
    started: str = ""
    updated: str = ""
    done: bool = False
    summary: str = ""
    notes: list[str] = field(default_factory=list)
    last_output: str = ""
    errors: int = 0              # total, for the record
    consecutive_errors: int = 0
    idle_cycles: int = 0         # consecutive cycles with no tool call and no new note
    stopped_reason: str = ""

    @property
    def path(self) -> Path:
        return AGENT_DIR / f"{self.spec.get('name', 'default')}.state.json"

    def save(self) -> None:
        self.updated = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, name: str) -> "AgentState | None":
        f = AGENT_DIR / f"{name}.state.json"
        if not f.exists():
            return None
        try:
            return cls(**json.loads(f.read_text()))
        except Exception:  # noqa: BLE001 - a corrupt state file must not block a run
            return None


# The goal tools are the loop's CONTROL PLANE, not ordinary tools. mark_done decides
# whether the run continues and note_progress is how one cycle talks to the next, so
# both must mutate state in *this* process. Over MCP every tool executes in a separate
# server process, where a mutation is invisible here — which is exactly what happened:
# an agent called mark_done on cycle 1 and kept running to its cycle cap, reporting
# zero notes, because the record it wrote lived in a subprocess that then exited.
GOAL_TOOLS = ("mark_done", "note_progress", "goal_status")


class GoalToolsLocal(ProviderWrapper):
    """Keep the goal tools in-process, whatever transport the rest of them use.

    Schemas still come from the inner provider, so the model sees one consistent
    toolset and the MCP server needs no special case — only the dispatch is diverted.
    """

    def call(self, name: str, args: dict) -> str:
        if name in GOAL_TOOLS:
            try:
                return str(getattr(tools_mod, name)(**(args or {})))
            except Exception as e:  # noqa: BLE001 - tools report, never raise
                return f"{name} error: {type(e).__name__}: {e}"
        return self.inner.call(name, args)


def stop_path(name: str) -> Path:
    """Touch this file to ask a running agent to stop after its current cycle."""
    return AGENT_DIR / f"{name}.stop"


# ── one agent at a time ─────────────────────────────────────────────────────────
# Only one chat model stays resident on this machine, so two agents would evict each
# other's model every turn (the reason pipeline's parallel mode was removed), and two
# on one name would fight over its state file. The console, a second tab and
# `lag agent start` are separate processes, so an in-memory flag cannot see them all.
#
# An OS file lock (flock) can, and it has the property a PID file lacks: the kernel
# releases it when the holder exits, however it exits. A PID file left by a kill -9
# or a power cut says "running" forever, and guessing whether that PID is still ours
# (PIDs get reused) is a bug of its own. The file's CONTENT is only a label saying
# who holds it; the lock is the truth.
def _lock_path() -> Path:
    return AGENT_DIR / ".run.lock"


def _holder(f) -> dict:
    f.seek(0)
    try:
        return json.loads(f.read() or "{}") or {"name": "another agent"}
    except ValueError:
        return {"name": "another agent"}


def _take_run_lock(name: str):
    """(open lock file, None) when acquired, or (None, who holds it)."""
    if fcntl is None:
        return None, None
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    # "a+", never "w": opening with "w" truncates BEFORE the lock is tried, wiping the
    # holder's label even when we are about to be refused.
    f = open(_lock_path(), "a+")  # noqa: SIM115 - the open file IS the lock; run() closes it
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        who = _holder(f)
        f.close()
        return None, who
    f.seek(0)
    f.truncate()
    f.write(json.dumps({"name": name, "pid": os.getpid(),
                        "since": datetime.datetime.now().astimezone()
                        .isoformat(timespec="seconds")}))
    f.flush()
    return f, None


def running_agent() -> dict | None:
    """Who is running an agent on this machine right now ({name, pid, since}), or None.

    Probes by trying the lock, since a leftover label proves nothing. The probe holds
    it for microseconds, so a start racing a probe can be told "busy" once; the
    console simply polls again.
    """
    if fcntl is None or not _lock_path().exists():
        return None
    with open(_lock_path(), "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return _holder(f)
        fcntl.flock(f, fcntl.LOCK_UN)
        return None


class AgentLoop:
    def __init__(self, spec: GoalSpec, cfg: dict, tool_provider=None):
        self.spec = spec
        self.cfg = cfg
        # Confinement is not configurable. An agent runs unattended, from a definition
        # that can be started again tomorrow, so the shell it gets is a container's —
        # never this machine's. See sandbox.py.
        self.sandbox = Sandbox(
            spec.workspace or (AGENT_DIR / "workspaces" / spec.name),
            read_root=Path.cwd(),
            network=spec.network,
        )
        provider = SandboxedProvider(tool_provider or LocalToolProvider(), self.sandbox)
        # Empty allow-list means every tool, so an agent with no `tools:` set is
        # unrestricted and the wrapper is skipped entirely.
        if spec.tools:
            provider = ToolSubset(provider, spec.tools)
        # Outermost, so the control plane survives the allow-list, the sandbox and MCP.
        self.provider = GoalToolsLocal(provider)
        persona = load_persona(spec.persona, cfg)
        self.model = spec.model or persona["model"]
        self.client = OllamaClient.from_config(
            cfg, self.model, system=persona.get("system_prompt"),
            think=persona.get("thinking_mode"),
        )
        prior = AgentState.load(spec.name)
        # Resume rather than restart: a 24/7 agent that forgets everything on restart
        # is just a slow one-shot.
        if prior and not prior.done and prior.spec.get("goal") == spec.goal:
            self.state = prior
            self.state.spec = asdict(spec)
        else:
            self.state = AgentState(spec=asdict(spec))
            self.state.started = datetime.datetime.now().astimezone().isoformat(
                timespec="seconds")

    def stop_requested(self) -> bool:
        return stop_path(self.spec.name).exists()

    def _sync_goal_tools(self) -> None:
        """Point the goal tools at this run so mark_done/note_progress reach it."""
        tools_mod.GOAL.update({
            "goal": self.spec.goal,
            "started": self.state.started,
            "cycle": self.state.cycle,
            "notes": self.state.notes,
            "done": False,
            "summary": "",
        })

    def run(self, on_event=None) -> AgentState:
        """Work the goal until it is met, the cycle cap is hit, or a stop is requested."""
        def emit(ev):
            if on_event:
                on_event(ev)

        # The lock comes FIRST: a refused start must not touch anything the running
        # agent owns — not its stop file, not its state.
        lock, holder = _take_run_lock(self.spec.name)
        if holder is not None:
            pid = f", pid {holder['pid']}" if holder.get("pid") else ""
            self.state.stopped_reason = (
                f"agent {holder.get('name')!r} is already running{pid} — one agent at "
                f"a time on this machine; stop it first")
            emit({"type": "agent_stop", "reason": "busy", "cycle": self.state.cycle,
                  "detail": self.state.stopped_reason})
            emit({"type": "agent_end", "cycle": self.state.cycle, "done": False,
                  "summary": "", "errors": self.state.errors,
                  "stopped_reason": self.state.stopped_reason})
            return self.state
        try:
            return self._run(emit, on_event)
        finally:
            if lock is not None:
                lock.close()               # closing the file releases the lock

    def _run(self, emit, on_event) -> AgentState:
        AGENT_DIR.mkdir(parents=True, exist_ok=True)
        stop_path(self.spec.name).unlink(missing_ok=True)
        # A resume keeps the RECORD (lifetime cycle count, notes, total errors) but not
        # the last run's verdict. Starting again is a request to try again: carried-over
        # breaker counters tripped after one cycle, and a carried-over reason left a
        # finished agent reporting "done" and "stopped: no progress" at once.
        self.state.stopped_reason = ""
        self.state.consecutive_errors = 0
        self.state.idle_cycles = 0

        # Fail closed, and fail BEFORE the loop — refusing at the first shell call
        # would leave the agent several tool calls deep with a half-done job.
        ok, why = sandbox_available()
        if not ok:
            self.state.stopped_reason = f"no sandbox: {why}"
            emit({"type": "agent_stop", "reason": "no_sandbox",
                  "cycle": self.state.cycle, "detail": self.state.stopped_reason})
            emit({"type": "agent_end", "cycle": self.state.cycle, "done": False,
                  "summary": "", "errors": self.state.errors,
                  "stopped_reason": self.state.stopped_reason})
            return self.state

        err = self.sandbox.start()
        if err:
            self.state.stopped_reason = f"sandbox failed to start: {err}"
            emit({"type": "agent_stop", "reason": "no_sandbox",
                  "cycle": self.state.cycle, "detail": self.state.stopped_reason})
            emit({"type": "agent_end", "cycle": self.state.cycle, "done": False,
                  "summary": "", "errors": self.state.errors,
                  "stopped_reason": self.state.stopped_reason})
            return self.state

        emit({"type": "agent_start", "goal": self.spec.goal, "name": self.spec.name,
              "persona": self.spec.persona, "model": self.model,
              "cycle": self.state.cycle, "resumed": self.state.cycle > 0,
              "sandbox": self.sandbox.describe()})

        try:
            self._loop(emit, on_event)
        finally:
            self.sandbox.stop()

        self.state.save()
        emit({"type": "agent_end", "cycle": self.state.cycle, "done": self.state.done,
              "summary": self.state.summary, "errors": self.state.errors,
              "stopped_reason": self.state.stopped_reason})
        return self.state

    def _loop(self, emit, on_event) -> None:
        # The cap bounds THIS run. state.cycle is the lifetime count and survives a
        # resume; comparing the cap against it meant an agent that once reached 20
        # stopped before cycle 1 on every later start, silently, forever.
        first = self.state.cycle
        while not self.state.done:
            ran = self.state.cycle - first
            if self.spec.max_cycles and ran >= self.spec.max_cycles:
                self.state.stopped_reason = (
                    f"reached the cycle cap ({ran} cycles this run)")
                emit({"type": "agent_stop", "reason": "max_cycles",
                      "cycle": self.state.cycle, "detail": self.state.stopped_reason})
                break
            if self.stop_requested():
                self.state.stopped_reason = "stop requested"
                emit({"type": "agent_stop", "reason": "stop_requested",
                      "cycle": self.state.cycle, "detail": self.state.stopped_reason})
                break

            self.state.cycle += 1
            self._sync_goal_tools()
            emit({"type": "cycle", "n": self.state.cycle, "model": self.model,
                  "goal": self.spec.goal})

            notes_before = len(tools_mod.GOAL["notes"])
            used_tools = False
            failed = False

            def watch(ev):
                # An erroring agent is loud; the dangerous 24/7 failure is the silent
                # one — clean cycles that touch nothing and loop forever. Watching the
                # event stream is how we tell "working" from "talking about working".
                nonlocal used_tools
                if ev.get("type") == "tool":
                    used_tools = True
                if on_event:
                    on_event(ev)

            try:
                out = self.client.run(
                    self.spec.prompt(self.state.cycle, self.state.notes,
                                     self.state.last_output),
                    tool_provider=self.provider, on_event=watch,
                )
                self.state.last_output = out or ""
                self.state.consecutive_errors = 0
            except Exception as e:  # noqa: BLE001 - one bad cycle must not end the run
                failed = True
                self.state.errors += 1
                self.state.consecutive_errors += 1
                msg = f"{type(e).__name__}: {e}"
                self.state.last_output = f"(cycle failed: {msg})"
                emit({"type": "cycle_error", "n": self.state.cycle, "message": msg,
                      "errors": self.state.errors,
                      "consecutive": self.state.consecutive_errors})
                # Back off so a hard failure does not spin the CPU for hours.
                time.sleep(min(60, 5 * self.state.consecutive_errors))

            # note_progress/mark_done wrote into the shared goal record during the turn.
            self.state.notes = list(tools_mod.GOAL["notes"])

            # A crashed cycle is not an idle one. Counting it as both let the idle
            # breaker trip first on a purely failing agent, which then reported "no
            # progress" when the truth was "it kept crashing" — the right stop for the
            # wrong reason, and the reason is what you read at 3am.
            progressed = used_tools or len(self.state.notes) > notes_before
            if failed:
                pass                       # counted by consecutive_errors instead
            elif progressed:
                self.state.idle_cycles = 0
            else:
                self.state.idle_cycles += 1
            if tools_mod.GOAL["done"]:
                self.state.done = True
                self.state.summary = tools_mod.GOAL["summary"]
                emit({"type": "agent_done", "cycle": self.state.cycle,
                      "summary": self.state.summary})

            self.state.save()

            # Circuit breakers. Both end the run rather than pausing it: an agent that
            # cannot make progress will not start making progress by being asked again,
            # and burning a laptop overnight to prove that is the failure we are
            # guarding against.
            if (not self.state.done and self.spec.max_failures
                    and self.state.consecutive_errors >= self.spec.max_failures):
                self.state.stopped_reason = (
                    f"{self.state.consecutive_errors} consecutive failed cycles")
                emit({"type": "agent_stop", "reason": "failed_cycles",
                      "cycle": self.state.cycle,
                      "detail": self.state.stopped_reason,
                      "last_error": self.state.last_output})
                break
            if (not self.state.done and self.spec.max_idle
                    and self.state.idle_cycles >= self.spec.max_idle):
                self.state.stopped_reason = (
                    f"{self.state.idle_cycles} consecutive cycles with no tool call "
                    f"and no progress noted")
                emit({"type": "agent_stop", "reason": "no_progress",
                      "cycle": self.state.cycle,
                      "detail": self.state.stopped_reason})
                break

            if not self.state.done and self.spec.interval:
                emit({"type": "idle", "seconds": self.spec.interval})
                # Poll the stop file so a stop is honoured within a second, not an
                # interval — an agent you cannot stop promptly is not one you leave on.
                for _ in range(self.spec.interval):
                    if self.stop_requested():
                        break
                    time.sleep(1)

