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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import tools as tools_mod
from .config import load_persona
from .ollama_client import OllamaClient
from .tools import LocalToolProvider

AGENT_DIR = Path("agent")


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
        return AGENT_DIR / f"{self.spec.get('name', 'default')}.json"

    def save(self) -> None:
        self.updated = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, name: str) -> "AgentState | None":
        f = AGENT_DIR / f"{name}.json"
        if not f.exists():
            return None
        try:
            return cls(**json.loads(f.read_text()))
        except Exception:  # noqa: BLE001 - a corrupt state file must not block a run
            return None


def stop_path(name: str) -> Path:
    """Touch this file to ask a running agent to stop after its current cycle."""
    return AGENT_DIR / f"{name}.stop"


class AgentLoop:
    def __init__(self, spec: GoalSpec, cfg: dict, tool_provider=None):
        self.spec = spec
        self.cfg = cfg
        self.provider = tool_provider or LocalToolProvider()
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
