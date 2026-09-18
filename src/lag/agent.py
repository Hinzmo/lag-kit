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

Autonomy here is deliberately unrestricted: agent mode gets the same unsandboxed shell
and filesystem the supervised CLI has. That is an explicit decision, not an oversight —
see CLAUDE.md "Agent mode". The stop file and cycle cap are the only brakes.
"""
from __future__ import annotations

import datetime
import json
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from . import tools as tools_mod
from .config import load_persona
from .ollama_client import OllamaClient
from .tools import LocalToolProvider

# Definitions and run state live together, paired by name, because they are always
# read together and two near-identical directories (agent/ vs agents/) would be a
# standing trap. The definition is yours to keep and commit; the state is per-machine
# and gitignored.
#     agents/<name>.yaml         what the agent IS
#     agents/<name>.state.json   what it has DONE
#     agents/<name>.stop         a request to stop
AGENT_DIR = Path("agents")

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
    max_cycles: int | None = None  # None = until done or stopped
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


def _schema_name(schema) -> str:
    """The tool name out of either provider's schema shape.

    LocalToolProvider hands back the function objects themselves; MCPToolProvider hands
    back Ollama-shaped dicts. Both satisfy the same interface, so anything filtering
    them has to read both — the alternative is a filter that works over one transport
    and silently passes everything over the other.
    """
    if callable(schema):
        return getattr(schema, "__name__", "")
    if isinstance(schema, dict):
        return (schema.get("function") or {}).get("name") or schema.get("name") or ""
    return ""


# The goal tools are the loop's CONTROL PLANE, not ordinary tools. mark_done decides
# whether the run continues and note_progress is how one cycle talks to the next, so
# both must mutate state in *this* process. Over MCP every tool executes in a separate
# server process, where a mutation is invisible here — which is exactly what happened:
# an agent called mark_done on cycle 1 and kept running to its cycle cap, reporting
# zero notes, because the record it wrote lived in a subprocess that then exited.
GOAL_TOOLS = ("mark_done", "note_progress", "goal_status")


class GoalToolsLocal:
    """Keep the goal tools in-process, whatever transport the rest of them use.

    Schemas still come from the inner provider, so the model sees one consistent
    toolset and the MCP server needs no special case — only the dispatch is diverted.
    """

    def __init__(self, inner):
        self.inner = inner

    def schemas(self):
        return self.inner.schemas()

    def call(self, name: str, args: dict) -> str:
        if name in GOAL_TOOLS:
            try:
                return str(getattr(tools_mod, name)(**(args or {})))
            except Exception as e:  # noqa: BLE001 - tools report, never raise
                return f"{name} error: {type(e).__name__}: {e}"
        return self.inner.call(name, args)


class ToolSubset:
    """A tool_provider narrowed to an allow-list.

    Wraps rather than replaces, so it works over direct dispatch and MCP alike and the
    agent loop cannot tell the difference — the same duck-typing rule the providers
    themselves follow. A blocked call returns a string the model can read and adapt to,
    never an exception, matching how every other tool failure behaves.
    """

    def __init__(self, inner, allow: list[str]):
        self.inner = inner
        self.allow = set(allow)

    def schemas(self):
        if not self.allow:
            return self.inner.schemas()
        return [s for s in self.inner.schemas() if _schema_name(s) in self.allow]

    def call(self, name: str, args: dict) -> str:
        if self.allow and name not in self.allow:
            return (f"blocked: {name} is not in this agent's allowed tools "
                    f"({', '.join(sorted(self.allow))})")
        return self.inner.call(name, args)


def stop_path(name: str) -> Path:
    """Touch this file to ask a running agent to stop after its current cycle."""
    return AGENT_DIR / f"{name}.stop"


class AgentLoop:
    def __init__(self, spec: GoalSpec, cfg: dict, tool_provider=None):
        self.spec = spec
        self.cfg = cfg
        provider = tool_provider or LocalToolProvider()
        # Empty allow-list means every tool, so an agent with no `tools:` set is
        # unrestricted and the wrapper is skipped entirely.
        if spec.tools:
            provider = ToolSubset(provider, spec.tools)
        # Outermost, so the control plane survives both the allow-list and MCP.
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
        AGENT_DIR.mkdir(parents=True, exist_ok=True)
        stop_path(self.spec.name).unlink(missing_ok=True)

        def emit(ev):
            if on_event:
                on_event(ev)

        emit({"type": "agent_start", "goal": self.spec.goal, "name": self.spec.name,
              "persona": self.spec.persona, "model": self.model,
              "cycle": self.state.cycle, "resumed": self.state.cycle > 0})

        while not self.state.done:
            if self.spec.max_cycles and self.state.cycle >= self.spec.max_cycles:
                emit({"type": "agent_stop", "reason": "max_cycles",
                      "cycle": self.state.cycle})
                break
            if self.stop_requested():
                emit({"type": "agent_stop", "reason": "stop_requested",
                      "cycle": self.state.cycle})
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

        self.state.save()
        emit({"type": "agent_end", "cycle": self.state.cycle, "done": self.state.done,
              "summary": self.state.summary, "errors": self.state.errors,
              "stopped_reason": self.state.stopped_reason})
        return self.state
