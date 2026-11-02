"""The goal pipeline: an ordered list of goals the orchestrator works through, one at a time.

You write WHAT should get done; the orchestrator decides HOW. Each goal is one routed
pipeline run — the lead reads it against the roster, picks the stages and briefs them
in the same call — unless you named the agents yourself, in which case they run as a
fixed pipeline. The queue replaced the console's project wizard, which asked you to make
those routing decisions by hand and then did nothing with most of them.

One at a time is not a limitation being worked around: only one chat model stays
resident on this machine (see STATUS.md), so concurrent goals would evict each other's
model every turn. A queue is the honest shape of what the box can do.

Goals run unattended, so they get the same confinement as agent mode — a container
sandbox and the public-only egress proxy — never this machine's shell. They share one
workspace (`agents/workspaces/goals`), so a later goal can build on an earlier one's
files. They also take the machine-wide agent lock while a goal runs, so a goal and an
agent never fight over the one resident model.

The runner lives in the server process and keeps going with the browser closed; the
console is a viewer. The list persists to `agents/goals.json`. A server restart never
resumes work by itself — a goal it interrupted goes back to pending and the queue comes
up paused, because silently restarting unattended work after a crash is a surprise.
"""
from __future__ import annotations

import datetime
import json
import os
import queue
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agent import AGENT_DIR, _take_run_lock

GOALS_FILE = AGENT_DIR / "goals.json"
WORKSPACE = AGENT_DIR / "workspaces" / "goals"
RUN_NAME = "goal pipeline"           # how the machine lock names us to agent mode

FAIL_POLICIES = ("stop", "skip", "ask")
FINISHED = ("done", "failed", "skipped")
MAX_TEXT = 2000


class Cancelled(BaseException):
    """Raised from the event hook to abandon the running goal.

    BaseException, not Exception, on purpose: OllamaClient.run catches Exception around
    every turn so it can recover from malformed tool JSON, and would swallow a cancel
    and carry on generating.
    """


class SandboxUnavailable(Exception):
    """The environment, not the goal, is the problem (Docker not running, image
    missing). The goal goes back to pending rather than being marked failed."""


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Goal:
    id: str
    text: str
    done_when: str = ""
    agents: list[str] = field(default_factory=list)   # empty = the orchestrator picks
    status: str = "pending"          # pending | running | done | failed | skipped
    route: list[str] = field(default_factory=list)    # the stages that actually ran
    answer: str = ""
    error: str = ""
    added: str = ""
    started: str = ""
    finished: str = ""
    check: str = ""                  # the done-when verdict: met | not met: … | unchecked: …

    def task(self) -> str:
        """The text the pipeline sees. "Done when" rides along as the bar to aim for,
        and the answer is checked against it afterwards (see check_done).

        The unattended note comes first because nobody is there to answer: a model that
        ends on "which would you prefer?" has stopped, and the queue would count it done."""
        task = f"{UNATTENDED}\n\nGoal: {self.text}"
        if self.done_when:
            task += f"\n\nThis goal is done when: {self.done_when}"
        return task


# Prepended to every goal. Goals run from a queue with nobody watching, so the agents
# are told to decide rather than ask, and where they can write: the project is mounted
# read-only, so a code change has to land in the workspace as files or a patch.
UNATTENDED = (
    "You are working through a queue unattended. Nobody will read or answer questions "
    "until the whole goal is finished. When you reach a decision, choose the option that "
    "best serves the goal, say in one line which you chose and why, and keep going. "
    "Do the work rather than describing what you would do; finish with the result, "
    "never with a question or a list of options. The project is read-only at /project; "
    "write anything you make (files, a patch, notes) under /workspace."
)

# Spots an answer that stopped to hand a decision back, or announced the next step
# without taking it. Only the tail is checked: a question in the middle of a write-up
# is fine, ending on one is the failure.
_ASKS = re.compile(
    r"\b(would you like|do you want|should i\b|shall i\b|which (one|option|approach)|"
    r"what would you (like|prefer)|how would you like|let me know|please (confirm|choose|"
    r"specify|advise)|your (choice|preference))", re.I)
_DEFERS = re.compile(
    r"\b(to proceed,? (we|i) will|next,? (we|i) will|i will now|we will now|"
    r"(we|i) will (first|start by|begin by))\b", re.I)


def stopped_short(answer: str) -> str:
    """Why an answer handed the goal back unfinished, or "" if it did not."""
    tail = (answer or "").strip()[-400:]
    if not tail:
        return ""
    last = tail.splitlines()[-1].strip()
    if _ASKS.search(tail) or last.endswith("?"):
        return "it ended by asking a question"
    if _DEFERS.search(tail):
        return "it ended by announcing a next step instead of taking it"
    return ""


# How many times the last stage is sent back to finish an answer that stopped short.
NUDGES = 2


def nudge(goal_task: str, answer: str, why: str) -> str:
    """The follow-up turn for a stage whose answer stopped short."""
    return (f"{goal_task}\n\nYour previous reply:\n{answer}\n\n"
            f"That reply did not finish the goal: {why}. Nobody is available to answer. "
            "Pick the option that best serves the goal, state your choice in one line, "
            "then carry it out now and give the finished result.")


_VERDICT = re.compile(r"^\W*(NOT\s+MET|MET)\b\W*(.*)", re.I | re.S)


def check_done(chat, goal: "Goal", answer: str) -> tuple[str, str]:
    """Judge an answer against the goal's "done when". Returns (verdict, reason).

    `chat(prompt) -> str` is one tool-free model call, injected so tests need no
    Ollama. Verdict is "met", "not met" or "unchecked"; a reply that cannot be read
    is "unchecked" rather than a failure, so a flaky judge never sinks good work.
    """
    if not goal.done_when:
        return "unchecked", "no done-when given"
    prompt = (
        "You are checking whether finished work meets its finish line.\n\n"
        f"Goal: {goal.text}\nDone when: {goal.done_when}\n\n"
        f"The answer:\n{(answer or '')[-6000:]}\n\n"
        "Does the answer meet the done-when condition? Judge only that condition, "
        "strictly but fairly. Reply with ONE line: MET, or NOT MET: <the specific "
        "thing that is missing>.")
    try:
        reply = (chat(prompt) or "").strip()
    except Exception as e:  # noqa: BLE001
        return "unchecked", f"the check failed: {type(e).__name__}: {e}"
    m = _VERDICT.match(reply)
    if not m:
        return "unchecked", f"unreadable verdict: {reply[:120]!r}"
    if m.group(1).upper().startswith("NOT"):
        return "not met", (m.group(2).strip().splitlines() or ["no reason given"])[0][:300]
    return "met", ""


def fix_turn(goal_task: str, answer: str, done_when: str, reason: str) -> str:
    """The follow-up turn for an answer the done-when check rejected."""
    return (f"{goal_task}\n\nYour previous reply:\n{answer}\n\n"
            f"A check against the goal's finish line (\"{done_when}\") found it not met: "
            f"{reason}. Fix exactly that, then give the complete finished result.")


class GoalRunner:
    """The queue, its settings, and the worker thread that drains it.

    `execute(goal, settings, emit) -> (route, answer)` does the work for one goal.
    It is injected so tests can drive the runner without Ollama or Docker; the server
    passes `pipeline_executor`. `emit` must be called for progress — it is also where
    a skip or remove takes effect, by raising Cancelled at the next event.
    """

    def __init__(self, execute, path: str | Path | None = None):
        self._execute = execute
        self.path = Path(path) if path else GOALS_FILE
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._subs: list[queue.Queue] = []
        self._thread: threading.Thread | None = None
        self._cancel: str | None = None      # "skip" | "remove" for the current goal

        self.goals: list[Goal] = []
        self.settings = {"plan": True, "fail": "stop", "model": ""}
        self.running = False      # the queue should be worked
        self.pausing = False      # finish the current goal, then stop
        self.current: str | None = None
        self.stage: str = ""      # the stage now talking, for a console that reconnects
        self.model: str = ""      # the model that stage runs on; chat shares it (web.py)
        self.awaiting: str | None = None   # failed goal waiting on you (fail = ask)
        self.note = ""            # why the queue last stopped, in words
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            # A corrupt file must not take the console down; start empty and say so.
            self.note = f"could not read {self.path.name}; starting with an empty list"
            return
        known = set(Goal.__dataclass_fields__)
        for raw in data.get("goals", []):
            g = Goal(**{k: v for k, v in raw.items() if k in known})
            if g.status == "running":       # interrupted by a restart
                g.status, g.started = "pending", ""
            self.goals.append(g)
        s = data.get("settings", {})
        self.settings.update({k: s[k] for k in self.settings if k in s})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"settings": self.settings,
                                   "goals": [asdict(g) for g in self.goals]}, indent=2))
        os.replace(tmp, self.path)          # atomic: a crash mid-write keeps the old list

    # ── viewers ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            return {"type": "goals", "running": self.running, "pausing": self.pausing,
                    "current": self.current, "stage": self.stage,
                    "awaiting": self.awaiting, "note": self.note,
                    "settings": dict(self.settings),
                    "goals": [asdict(g) for g in self.goals]}

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subs.append(q)
            q.put(self.snapshot())
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def _publish(self, ev: dict) -> None:
        for q in list(self._subs):
            q.put(ev)

    def _changed(self) -> None:
        self._save()
        self._publish(self.snapshot())

    # ── edits ────────────────────────────────────────────────────────────────
    def _find(self, gid: str) -> Goal:
        for g in self.goals:
            if g.id == gid:
                return g
        raise KeyError(gid)

    def add(self, text: str, done_when: str = "", agents: list[str] | None = None) -> Goal:
        text = (text or "").strip()
        if not text:
            raise ValueError("a goal needs some text")
        if len(text) > MAX_TEXT or len(done_when or "") > MAX_TEXT:
            raise ValueError(f"keep a goal under {MAX_TEXT} characters")
        g = Goal(id=uuid.uuid4().hex[:12], text=text, done_when=(done_when or "").strip(),
                 agents=[a for a in (agents or []) if a], added=_now())
        with self._lock:
            self.goals.append(g)
            self._changed()
            self._wake.notify_all()          # an idle running queue picks it up now
        return g

    def remove(self, gid: str) -> None:
        with self._lock:
            g = self._find(gid)
            if gid == self.current:
                self._cancel = "remove"      # the worker drops it when it stops
                self.note = "stopping the current goal…"
            else:
                self.goals.remove(g)
                if self.awaiting == gid:
                    self.awaiting = None
            self._changed()

    def move(self, gid: str, delta: int) -> None:
        """Swap a pending goal with its pending neighbour. Finished and running goals
        keep their place — only what is still waiting can be reordered."""
        with self._lock:
            pending = [g for g in self.goals if g.status == "pending"]
            g = self._find(gid)
            if g not in pending:
                raise ValueError("only goals still waiting can be moved")
            i = pending.index(g)
            j = i + (1 if delta > 0 else -1)
            if not 0 <= j < len(pending):
                return
            a, b = self.goals.index(pending[i]), self.goals.index(pending[j])
            self.goals[a], self.goals[b] = self.goals[b], self.goals[a]
            self._changed()

    def retry(self, gid: str) -> None:
        """Put a failed or skipped goal back at the head of the queue."""
        with self._lock:
            g = self._find(gid)
            if g.status not in ("failed", "skipped"):
                raise ValueError("only a failed or skipped goal can be retried")
            g.status, g.error, g.answer, g.route = "pending", "", "", []
            g.started = g.finished = ""
            self.goals.remove(g)
            first = next((i for i, x in enumerate(self.goals) if x.status == "pending"),
                         len(self.goals))
            self.goals.insert(first, g)
            if self.awaiting == gid:
                self.awaiting = None
                self._start()                 # you answered the question: carry on
            self._changed()

    def skip(self) -> None:
        """Abandon the current goal (it stays in the list as skipped) and move on —
        or, while a failed goal waits on you, leave it failed and carry on."""
        with self._lock:
            if self.current:
                self._cancel = "skip"
                self.note = "skipping the current goal…"
            elif self.awaiting:
                self.awaiting = None
                self._start()
            self._changed()

    def clear_finished(self) -> None:
        with self._lock:
            self.goals = [g for g in self.goals if g.status not in FINISHED]
            self.awaiting = None
            self._changed()

    def configure(self, plan: bool | None = None, fail: str | None = None,
                  model: str | None = None) -> None:
        with self._lock:
            if fail is not None:
                if fail not in FAIL_POLICIES:
                    raise ValueError(f"on-fail must be one of {', '.join(FAIL_POLICIES)}")
                self.settings["fail"] = fail
            if plan is not None:
                self.settings["plan"] = bool(plan)
            if model is not None:
                self.settings["model"] = model
            self._changed()

    # ── run / pause ──────────────────────────────────────────────────────────
    def run(self) -> None:
        with self._lock:
            self.awaiting = None
            self._start()
            self._changed()

    def pause(self) -> None:
        """Stop after the current goal. A model cannot be interrupted mid-answer without
        losing the answer, so pause lets the running goal finish; skip abandons it."""
        with self._lock:
            if self.current:
                self.pausing = True
                self.note = "pausing after this goal…"
            else:
                self.running = False
                self.note = "paused"
            self._wake.notify_all()
            self._changed()

    def _start(self) -> None:
        self.running, self.pausing, self.note = True, False, ""
        if not (self._thread and self._thread.is_alive()):
            self._thread = threading.Thread(target=self._loop, name="goal-runner",
                                            daemon=True)
            self._thread.start()
        self._wake.notify_all()

    def _next(self) -> Goal | None:
        return next((g for g in self.goals if g.status == "pending"), None)

    def _stop(self, note: str) -> None:
        self.running, self.pausing, self.note = False, False, note

    # ── the worker ───────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while True:
            with self._lock:
                # Idle while running = waiting for you to add a goal.
                while self.running and not self._next():
                    self._wake.wait(timeout=1.0)
                if not self.running:
                    self._thread = None
                    return
                g = self._next()
                lock, holder = _take_run_lock(RUN_NAME)
                if holder is not None:
                    self._stop(f"paused: agent {holder.get('name')!r} is running — one at "
                               f"a time on this machine; stop it, then press run")
                    self._thread = None
                    self._changed()
                    return
                g.status, g.started, g.error = "running", _now(), ""
                self.current, self.stage, self._cancel = g.id, "", None
                settings = dict(self.settings)
                self._changed()
            try:
                outcome, detail = self._run_one(g, settings)
            finally:
                if lock is not None:
                    lock.close()             # released between goals, never held idle
            with self._lock:
                self._finish(g, outcome, detail)

    def _run_one(self, g: Goal, settings: dict) -> tuple[str, str]:
        def emit(ev: dict) -> None:
            if self._cancel:
                raise Cancelled(self._cancel)
            kind = ev.get("type")
            with self._lock:
                if kind == "routed":
                    g.route = [a["name"] for a in ev.get("agents", [])]
                    self._changed()
                elif kind == "start":
                    g.route = [s["name"] for s in ev.get("stages", [])]
                    self._changed()
                elif kind == "stage":
                    self.stage = ev.get("name", "")
                    self.model = ev.get("model", "") or self.model
                    self._changed()
                elif kind == "routing":
                    self.model = ev.get("model", "") or self.model
            self._publish({**ev, "goal": g.id})

        try:
            res = self._execute(g, settings, emit)
            route, answer = res[0], res[1]
            verdict = res[2] if len(res) > 2 else None
        except Cancelled as c:
            return str(c), ""
        except SandboxUnavailable as e:
            return "env", str(e)
        except Exception as e:  # noqa: BLE001 - a failed goal is data, not a crash
            return "failed", f"{type(e).__name__}: {e}"
        if self._cancel:                     # asked to stop during a silent stretch
            return self._cancel, ""
        if route:
            g.route = list(route)
        g.answer = answer or ""
        if g.answer.startswith("(stopped:"):  # OllamaClient gave up on malformed turns
            return "failed", g.answer
        why = stopped_short(g.answer)
        if why:                              # still handing it back after the nudges
            return "failed", f"not finished: {why}"
        if verdict:
            kind, reason = verdict
            g.check = kind if kind == "met" else f"{kind}: {reason}"
            if kind == "not met":
                return "failed", f"done-when not met: {reason}"
        return "done", ""

    def _finish(self, g: Goal, outcome: str, detail: str) -> None:
        g.finished = _now()
        self.current, self.stage, self.model, self._cancel = None, "", "", None
        if outcome == "remove":
            if g in self.goals:
                self.goals.remove(g)
            self.note = ""
        elif outcome == "skip":
            g.status, self.note = "skipped", ""
        elif outcome == "env":
            g.status, g.started, g.finished = "pending", "", ""
            self._stop(f"paused: {detail}")
        elif outcome == "failed":
            g.status, g.error = "failed", detail
            policy = self.settings.get("fail", "stop")
            if policy == "stop":
                self._stop("stopped: a goal failed — retry it or press run to carry on")
            elif policy == "ask":
                self.awaiting = g.id
                self._stop("a goal failed — retry it, or skip to carry on")
        else:
            g.status = "done"
        if self.pausing:
            self._stop("paused")
        self._changed()


def judge_chat(cfg: dict, model: str):
    """One tool-free, non-thinking call on `model`, for check_done."""
    import ollama
    client = ollama.Client(host=cfg.get("ollama_base_url", "http://127.0.0.1:11434"))
    num_ctx = ((cfg.get("ollama") or {}).get("num_ctx_by_model") or {}).get(
        model, (cfg.get("ollama") or {}).get("num_ctx", 8192))

    def chat(prompt: str) -> str:
        r = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                        think=False, options={"num_ctx": num_ctx, "num_predict": 200,
                                              "temperature": 0})
        return r.message.content or ""
    return chat


def pipeline_executor(cfg: dict, provider_factory):
    """The real work for one goal: a sandboxed, routed (or hand-picked) pipeline run."""
    def execute(goal: Goal, settings: dict, emit):
        from .pipeline import Pipeline
        from .sandbox import Sandbox, SandboxedProvider, available

        ok, why = available()
        if not ok:
            raise SandboxUnavailable(f"no sandbox: {why}")
        sandbox = Sandbox(WORKSPACE, read_root=Path.cwd(), network=True)
        err = sandbox.start()
        if err:
            raise SandboxUnavailable(f"sandbox failed to start: {err}")
        try:
            provider = SandboxedProvider(provider_factory(), sandbox)
            task = goal.task()
            model = settings.get("model") or None
            if goal.agents:
                pipe = Pipeline.from_personas(goal.agents, cfg, provider,
                                              model_override=model)
            else:
                pipe = Pipeline.routed(task, cfg, provider, on_event=emit,
                                       model_override=model)
            emit({"type": "start", "loads": pipe.loads(),
                  "stages": [{"name": s.name, "model": s.model} for s in pipe.stages]})
            answer = pipe.run(task, plan=settings.get("plan", True), on_event=emit)
            # A stage that stops to ask has not finished: send the last one back to
            # decide for itself, a bounded number of times.
            last = pipe.stages[-1]
            for _ in range(NUDGES):
                why = stopped_short(answer)
                if not why:
                    break
                emit({"type": "nudge", "name": last.name, "reason": why})
                emit({"type": "stage", "name": last.name, "model": last.model,
                      "role": "stage", "loaded": False})
                answer = last.run(pipe._fit(nudge(task, answer, why), last, emit),
                                  on_event=emit)
            verdict = None
            if goal.done_when and not stopped_short(answer):
                # Judged on the last stage's model: it is already resident, so the
                # check costs one short call and no reload.
                chat = judge_chat(cfg, last.model)
                verdict = check_done(chat, goal, answer)
                emit({"type": "check", "verdict": verdict[0], "reason": verdict[1]})
                if verdict[0] == "not met":
                    emit({"type": "stage", "name": last.name, "model": last.model,
                          "role": "stage", "loaded": False})
                    answer = last.run(pipe._fit(fix_turn(task, answer, goal.done_when,
                                                         verdict[1]), last, emit),
                                      on_event=emit)
                    verdict = check_done(chat, goal, answer)
                    emit({"type": "check", "verdict": verdict[0], "reason": verdict[1]})
            return [s.name for s in pipe.stages], answer, verdict
        finally:
            sandbox.stop()
    return execute
