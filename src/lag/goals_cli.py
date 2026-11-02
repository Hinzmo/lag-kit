"""`lag goals`: the goal queue from the terminal.

Every command talks to the running `lag serve` over its HTTP API rather than opening
agents/goals.json itself. The server's GoalRunner owns the queue, so there is one
writer and the console and the terminal always agree. Standard library only.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

import typer

DEFAULT_URL = "http://127.0.0.1:8765"
app = typer.Typer(help="The goal queue from the terminal (talks to a running `lag serve`).",
                  no_args_is_help=True)

MARK = {"pending": "·", "running": "▶", "done": "✓", "failed": "✗", "skipped": "↷"}


def _url() -> str:
    return os.environ.get("LAG_URL", DEFAULT_URL).rstrip("/")


def _fail(msg: str) -> None:
    typer.echo(msg, err=True)
    raise typer.Exit(1)


def api(method: str, path: str, body: dict | None = None, opener=urllib.request.urlopen) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(_url() + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with opener(req, timeout=15) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        _fail(f"{method} {path} failed: HTTP {e.code} {e.read()[:200].decode(errors='replace')}")
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        _fail(f"lag serve is not running at {_url()}. Start it with: lag serve")
    return {}


def resolve(snap: dict, ref: str) -> dict:
    """A goal by id, unique id prefix, or 1-based position in the list."""
    goals = snap.get("goals", [])
    if ref.isdigit() and 1 <= int(ref) <= len(goals) and len(ref) < 4:
        return goals[int(ref) - 1]
    hits = [g for g in goals if g["id"].startswith(ref)]
    if len(hits) == 1:
        return hits[0]
    _fail(f"no goal matches {ref!r}" if not hits else f"{ref!r} matches {len(hits)} goals; use more of the id")
    return {}


def _check(snap: dict) -> dict:
    if snap.get("error"):
        _fail(snap["error"])
    return snap


def render(snap: dict) -> str:
    s = snap.get("settings", {})
    state = ("running" if snap.get("running") else "paused") + (" (pausing after this goal)" if snap.get("pausing") else "")
    lines = [f"queue {state} · plan {'on' if s.get('plan') else 'off'} · on fail: {s.get('fail', 'stop')}"
             + (f" · model {s['model']}" if s.get("model") else "")]
    if snap.get("note"):
        lines.append(f"note: {snap['note']}")
    goals = snap.get("goals", [])
    if not goals:
        lines.append("no goals yet. Add one: lag goals add \"…\"")
    for i, g in enumerate(goals, 1):
        who = ", ".join(g.get("route") or g.get("agents") or []) or "orchestrator picks"
        live = f"  [{snap['stage']}]" if g["id"] == snap.get("current") and snap.get("stage") else ""
        lines.append(f"{i:>2} {MARK.get(g['status'], '?')} {g['id'][:6]}  {g['status']:<8} {g['text'][:70]}{live}")
        if g.get("done_when"):
            lines.append(f"{'':12}done when: {g['done_when'][:70]}")
        lines.append(f"{'':12}agents: {who}")
        if g["status"] == "failed" and g.get("error"):
            lines.append(f"{'':12}error: {g['error'][:100]}")
    return "\n".join(lines)


@app.command("ls")
def ls():
    """List the queue."""
    typer.echo(render(api("GET", "/api/goals")))


@app.command()
def add(
    text: str = typer.Argument(..., help="What should get done."),
    done_when: str = typer.Option("", "--done-when", "-d", help="How you will know it is done."),
    agents: str = typer.Option(None, help="Comma-separated agents, in order. Omit to let the orchestrator pick."),
):
    """Add a goal to the end of the queue."""
    names: list[str] = []
    if agents is not None:
        names = [a.strip() for a in agents.split(",") if a.strip()]
        if not names:
            _fail("--agents was given but names no agents. Omit it to let the orchestrator pick.")
    snap = _check(api("POST", "/api/goals", {"text": text, "done_when": done_when, "agents": names}))
    g = snap["goals"][-1]
    typer.echo(f"added {g['id'][:6]}: {g['text'][:70]}")
    if not snap.get("running"):
        typer.echo("the queue is paused. Start it with: lag goals run")


@app.command()
def show(ref: str = typer.Argument(..., help="Goal id, id prefix, or list position.")):
    """Show one goal in full, answer included."""
    g = resolve(api("GET", "/api/goals"), ref)
    typer.echo(f"{g['id']}  {g['status']}\n{g['text']}")
    for k in ("done_when", "check", "added", "started", "finished", "error"):
        if g.get(k):
            typer.echo(f"{k.replace('_', ' ')}: {g[k]}")
    if g.get("route") or g.get("agents"):
        typer.echo("agents: " + ", ".join(g.get("route") or g.get("agents")))
    if g.get("answer"):
        typer.echo("\n" + g["answer"])


def _simple(path: str, said: str):
    snap = _check(api("POST", path))
    typer.echo(said)
    return snap


@app.command()
def run():
    """Start (or resume) working through the queue."""
    _simple("/api/goals/run", "queue running")


@app.command()
def pause():
    """Stop after the current goal finishes."""
    _simple("/api/goals/pause", "pausing after the current goal")


@app.command()
def skip():
    """Abandon the running goal (kept as skipped) and move on."""
    _simple("/api/goals/skip", "skipped the running goal")


@app.command()
def clear():
    """Drop every finished goal: done, failed and skipped."""
    _simple("/api/goals/clear", "cleared finished goals")


@app.command()
def rm(ref: str = typer.Argument(..., help="Goal id, id prefix, or list position.")):
    """Remove a goal. A running one is stopped and dropped."""
    g = resolve(api("GET", "/api/goals"), ref)
    _check(api("DELETE", f"/api/goals/{g['id']}"))
    typer.echo(f"removed {g['id'][:6]}: {g['text'][:70]}")


@app.command()
def retry(ref: str = typer.Argument(..., help="Goal id, id prefix, or list position.")):
    """Put a failed or skipped goal back in the queue."""
    g = resolve(api("GET", "/api/goals"), ref)
    _check(api("POST", f"/api/goals/{g['id']}/retry"))
    typer.echo(f"queued again: {g['text'][:70]}")


@app.command()
def move(
    ref: str = typer.Argument(..., help="Goal id, id prefix, or list position."),
    by: int = typer.Option(-1, help="Positions to move; negative is earlier."),
):
    """Move a pending goal earlier (default) or later in the queue."""
    g = resolve(api("GET", "/api/goals"), ref)
    typer.echo(render(_check(api("POST", f"/api/goals/{g['id']}/move?delta={by}"))))


@app.command("set")
def settings(
    plan: bool = typer.Option(None, "--plan/--no-plan", help="Let the lead write per-stage instructions first."),
    fail: str = typer.Option(None, help="On a failed goal: stop, skip or ask."),
    model: str = typer.Option(None, help="Run every stage on this tag; \"\" for each persona's own."),
):
    """Change how the queue runs."""
    if fail is not None and fail not in ("stop", "skip", "ask"):
        _fail("--fail must be stop, skip or ask")
    if plan is None and fail is None and model is None:
        _fail("nothing to change. See: lag goals set --help")
    snap = _check(api("POST", "/api/goals/settings", {"plan": plan, "fail": fail, "model": model}))
    typer.echo(render(snap).splitlines()[0])


# ── watch ───────────────────────────────────────────────────────────────────

class Printer:
    """Turns the server's goal events into terminal lines. Tokens stream inline;
    everything else starts a line of its own."""

    def __init__(self, out=sys.stdout, thinking: bool = False):
        self.out, self.thinking = out, thinking
        self.inline = False
        self.seen = False
        self.status: dict[str, str] = {}

    def _line(self, s: str) -> None:
        if self.inline:
            self.out.write("\n")
            self.inline = False
        self.out.write(s + "\n")
        self.out.flush()

    def event(self, e: dict) -> None:
        t = e.get("type")
        if t == "goals":
            if not self.seen:
                # The first snapshot is the queue as it already was: nothing in it is news.
                self.seen = True
                self.status = {g["id"]: g["status"] for g in e.get("goals", [])}
                return
            for g in e.get("goals", []):
                before = self.status.get(g["id"])
                if before is not None and before != g["status"]:
                    mark = MARK.get(g["status"], "?")
                    extra = f": {g['error'][:100]}" if g["status"] == "failed" and g.get("error") else ""
                    self._line(f"{mark} {g['id'][:6]} {g['status']}: {g['text'][:70]}{extra}")
                elif before is None:
                    self._line(f"+ {g['id'][:6]} added: {g['text'][:70]}")
                self.status[g["id"]] = g["status"]
        elif t == "routed":
            names = [a.get("name", "?") if isinstance(a, dict) else str(a) for a in e.get("agents", [])]
            self._line(f"  routed → {' → '.join(names)}")
        elif t == "stage":
            self._line(f"── {e.get('name')} ({e.get('model')})")
        elif t == "token":
            self.out.write(e.get("text", ""))
            self.out.flush()
            self.inline = True
        elif t == "thinking" and self.thinking:
            self._line(f"  (thinking) {e.get('text', '')[:300]}")
        elif t == "tool":
            self._line(f"  → {e.get('pretty') or e.get('name')}"[:160])
        elif t == "tool_result":
            first = (e.get("content") or "").strip().splitlines()[:1]
            self._line(f"  ← {e.get('name')}: {first[0][:100] if first else '(empty)'}")
        elif t == "nudge":
            self._line(f"  ↺ sent back to {e.get('name')}: {e.get('reason')}")
        elif t == "check":
            self._line(f"  ✔ done-when {e.get('verdict')}" + (f": {e['reason']}" if e.get("reason") else ""))
        elif t == "final_answer_nudge":
            self._line(f"  ⏱ out of tool turns ({e.get('max_iters')}); answering from what it has")
        elif t in ("turn_error", "error"):
            self._line(f"  ! {e.get('message') or e.get('error')}")


def sse_events(lines):
    """Parse SSE `data:` lines into dicts; comments and blanks are skipped."""
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n") if isinstance(raw, bytes) else raw.rstrip("\r\n")
        if line.startswith("data:"):
            try:
                yield json.loads(line[5:])
            except ValueError:
                continue


@app.command()
def watch(thinking: bool = typer.Option(False, help="Also print the model's reasoning.")):
    """Stream the queue's live work: stages, tool calls, answers, outcomes. Ctrl-C to stop."""
    p = Printer(thinking=thinking)
    req = urllib.request.Request(_url() + "/api/goals/events")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            first = True
            for e in sse_events(r):
                if first and e.get("type") == "goals":
                    typer.echo(render(e) + "\n")
                    first = False
                p.event(e)
    except KeyboardInterrupt:
        typer.echo("")
    except (urllib.error.URLError, ConnectionError):
        _fail(f"lag serve is not running at {_url()}. Start it with: lag serve")
