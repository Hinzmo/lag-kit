"""Local web console: serves ui/console.html plus a small JSON API over the harness.

Fully local — the server binds to localhost and every call goes through the same
library the CLI uses (config, OllamaClient, Pipeline, tools, rag). No cloud, no keys, so
the privacy stance is intact. This exists so the phosphor console reflects the *real*
harness instead of the scripted playback it shipped as.

`/api/run` is the blocking path: it returns the finished answer. `/api/stream` is the
live one — it runs the same pipeline on a worker thread and forwards the loop's events
(thinking, token, tool, tool_result, stage) to the browser as SSE, so the console shows
tokens as they are produced rather than typing out a finished string. It is a POST
returning text/event-stream rather than an EventSource GET, so the prompt travels in the
body instead of the URL.

Running one agent on a small task is not a special case: it is a one-stage pipeline with
plan=False, so clicking a persona in the roster reuses this same path.

The MCP tool provider is built once and cached: spawning both servers per request would
cost a second of startup each time, and a single shared provider is fine for a
single-user console. The server owns it, so the lifespan hook closes it at shutdown —
two server subprocesses and their threads. Run the server from the repo root so the RAG server resolves
`./.chroma` — the same cwd caveat as `lag run --mcp`.
"""
from __future__ import annotations

import contextlib
import json
import queue
import threading
from pathlib import Path

import ollama
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import load_config, load_persona
from .ollama_client import OllamaClient, capabilities
from .providers import ToolsExcept
from .pipeline import Pipeline
from .tools import TOOLS, LocalToolProvider

_ROOT = Path(__file__).resolve().parents[2]
_UI = _ROOT / "ui" / "console.html"

# Built once, reused across requests (see module docstring).
_provider_cache: dict = {}


@contextlib.asynccontextmanager
async def _lifespan(_app):
    yield
    # Shutdown: this process built these providers, so it closes them. A build that
    # FAILED leaves nothing here to close — from_config cleans up after itself, which
    # matters because _provider retries on every request while a server is broken.
    for key in list(_provider_cache):
        close = getattr(_provider_cache.pop(key), "close", None)
        if close:
            close()


app = FastAPI(title="lag console", lifespan=_lifespan)


def _without_retrieval(provider):
    """The provider minus the knowledge base."""
    return ToolsExcept(provider, ["search_knowledge_base"])


def _provider(cfg: dict, mcp: bool):
    key = "mcp" if mcp else "local"
    if key not in _provider_cache:
        if mcp:
            from .mcp_client import MCPToolProvider

            _provider_cache[key] = MCPToolProvider.from_config(cfg)
        else:
            _provider_cache[key] = LocalToolProvider()
    return _provider_cache[key]


# ── status helpers (separated so tests can stub them without touching Ollama) ──

def _ollama_status(cfg: dict) -> tuple[bool, list[str], list[dict]]:
    """(reachable, available model tags, loaded models with size)."""
    client = ollama.Client(host=cfg.get("ollama_base_url", "http://localhost:11434"))
    up, available, loaded = True, [], []
    try:
        available = [m.model for m in client.list().models]
    except Exception:
        up = False
    # A failure here means "cannot tell what is loaded", which the console renders as
    # an empty panel — not a reason to fail the whole status call.
    with contextlib.suppress(Exception):
        loaded = [
            {"name": m.model, "size_gb": round((m.size or 0) / 1e9, 2)}
            for m in client.ps().models
        ]
    return up, available, loaded


def _model_capabilities(cfg: dict, tags: list[str]) -> list[dict]:
    """Per-tag capabilities, so the console never offers a choice that cannot work.

    The agent loop is a tool-calling loop, so a completion-only tag is not a slightly
    worse pick — it is one that cannot reach a file, a shell, retrieval or memory.
    They stay listed rather than hidden, because they are still fine for plain
    generation; the UI marks them and the loop drops tools rather than failing.
    """
    host = cfg.get("ollama_base_url", "http://localhost:11434")
    out = []
    for t in tags:
        caps = sorted(capabilities(host, t))
        out.append({
            "name": t,
            "capabilities": caps,
            "tools": "tools" in caps,
            "thinking": "thinking" in caps,
            "embedding": "embedding" in caps,
        })
    return out


def _chroma_count(cfg: dict) -> dict:
    """Chunk count in the default collection, or a graceful 'unavailable'."""
    try:
        from . import rag

        return {"available": True, "chunks": rag.get_collection().count()}
    except Exception:
        return {"available": False, "chunks": 0}


def _personas(cfg: dict) -> list[dict]:
    base = Path(cfg.get("paths", {}).get("personas", "./personas"))
    out = []
    for p in sorted(base.glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            out.append(
                {
                    "name": data.get("name", p.stem),
                    "role": data.get("role", ""),
                    "model": load_persona(p.stem, cfg).get("model", ""),
                    "tools": data.get("tools", []),
                }
            )
        except Exception:
            continue  # one bad persona shouldn't blank the whole roster
    return out


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(_UI)


@app.get("/api/state")
def state():
    """Everything the console needs to render the real system on load."""
    cfg = load_config()
    up, available, loaded = _ollama_status(cfg)
    models = cfg.get("models", {}) or {}
    chat_model = models.get("orchestrator") or models.get("worker") or ""
    # Report the window this model actually gets, not the global default — with
    # num_ctx_by_model the two differ by 16x here, and the console showing 8192 next
    # to a 131072 orchestrator would be a lie about the running system.
    o = cfg.get("ollama", {}) or {}
    num_ctx = (o.get("num_ctx_by_model") or {}).get(chat_model, o.get("num_ctx", 4096))
    return {
        "version": "0.2.0",
        "ollama": {"up": up, "available": available, "loaded": loaded,
                   "models": _model_capabilities(cfg, available)},
        "chat_model": chat_model,
        "num_ctx": num_ctx,
        "personas": _personas(cfg),
        "tools": list(TOOLS),
        "rag_tools": ["search_knowledge_base"],
        "chroma": _chroma_count(cfg),
        "mcp": {"tools_server": True, "rag_server": True},
    }


class RunRequest(BaseModel):
    prompt: str
    mode: str = "solo"  # solo | pipeline
    agents: str | None = None
    plan: bool = True
    # Run every stage on this tag instead of each persona's configured model. Picking
    # one model is also how you spend zero reloads, since stages then share the
    # resident weights.
    model: str | None = None
    # Let the lead pick the team from the roster instead of running a fixed list.
    # Ignored when `agents` names them explicitly (e.g. a single-agent task from the
    # roster), since that is the caller having already decided.
    route: bool = True
    # Retrieval is on by default. Off withholds search_knowledge_base for this run —
    # every other tool is untouched.
    retrieval: bool = True
    mcp: bool = True


@app.post("/api/run")
def run(req: RunRequest):
    """Run one turn and return the finished answer (blocking in PR1).

    Errors are returned as ``{"error": ...}`` with a 200 so the console can show them
    as a message rather than a failed request — same spirit as the tool layer, which
    returns errors to the model instead of raising.
    """
    cfg = load_config()
    try:
        provider = _provider(cfg, req.mcp)
        if not req.retrieval:
            provider = _without_retrieval(provider)
        if req.mode in (None, "", "solo"):
            model = cfg["models"]["worker"]
            client = OllamaClient.from_config(cfg, model)
            answer = client.run(req.prompt, tool_provider=provider, verbose=True)
            return {"answer": answer, "model": model}

        from .pipeline import Pipeline

        default = cfg.get("pipeline", {}).get(
            "default_agents", "architect_agent,dev_agent,senior_dev_agent"
        )
        names = [a.strip() for a in (req.agents or default).split(",")]
        pipe = Pipeline.from_personas(names, cfg, provider)
        answer = pipe.run(req.prompt, plan=req.plan)
        return {
            # The answer comes from the LAST stage, so label it with that stage's model.
            # Reporting the orchestrator unconditionally misattributes every run whose
            # final stage isn't on it — which is the common case now that the lead only
            # plans and the downstream stages write the answer.
            "answer": answer,
            "model": pipe.stages[-1].model if pipe.stages else "",
            "stages": [{"name": s.name, "model": s.model} for s in pipe.stages],
            "loads": pipe.loads(),
        }
    except Exception as e:  # noqa: BLE001 - surface to the UI, never 500
        return {"error": f"{type(e).__name__}: {e}"}


_DONE = object()


def _stream(work) -> StreamingResponse:
    """Run `work(emit)` on a worker thread and stream what it emits as SSE.

    Both streaming endpoints were repeating the same five things — build a queue,
    wrap the body in try/except/finally, push a sentinel, start a daemon thread,
    construct the response — around a few lines that actually differed. Sharing it
    also means the error contract is stated once: a failure arrives as a final
    `error` EVENT rather than a failed request, so the console shows it as a message,
    the same spirit as the tool layer returning errors instead of raising.
    """
    events: queue.Queue = queue.Queue()

    def runner() -> None:
        try:
            work(events.put)
        except Exception as e:  # noqa: BLE001 - surface to the UI, never 500
            events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            events.put(_DONE)

    threading.Thread(target=runner, daemon=True).start()
    return StreamingResponse(_sse(events), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _sse(events: "queue.Queue"):
    """Drain the event queue as SSE frames, never dying on one bad event.

    An event that will not serialise (a tool argument object, say) used to raise
    inside the generator and silently end the stream mid-run — the browser just sees
    the connection close, with no error and no answer. Encoding per event and falling
    back to a described placeholder keeps the run visible even when one frame is odd.
    """
    while True:
        ev = events.get()
        if ev is _DONE:
            return
        try:
            payload = json.dumps(ev)
        except (TypeError, ValueError):
            payload = json.dumps({"type": ev.get("type", "event") if isinstance(ev, dict)
                                  else "event",
                                  "unserialisable": True, "repr": repr(ev)[:500]})
        yield f"data: {payload}\n\n"


@app.post("/api/stream")
def stream(req: RunRequest):
    """Same run as /api/run, but events are pushed as they happen.

    The harness loop is synchronous, so the work goes on a daemon thread and pushes
    events into a queue the response generator drains. Errors arrive as a final
    `error` event rather than a failed request — the console shows them as a message,
    the same contract /api/run has and the same spirit as the tool layer.
    """
    cfg = load_config()
    def work(emit) -> None:
        provider = _provider(cfg, req.mcp)
        if not req.retrieval:
            provider = _without_retrieval(provider)

        if req.mode in (None, "", "solo"):
            model = req.model or cfg["models"]["worker"]
            emit({"type": "start", "loads": 1,
                  "stages": [{"name": "solo", "model": model}]})
            emit({"type": "stage", "name": "solo", "model": model,
                  "role": "stage", "loaded": True})
            answer = OllamaClient.from_config(cfg, model).run(
                req.prompt, tool_provider=provider, on_event=emit)
            emit({"type": "done", "answer": answer})
            return

        if req.route and not req.agents:
            # The lead reads the task against the roster and returns the ordered
            # subset that fits it, briefing them in the same turn.
            pipe = Pipeline.routed(req.prompt, cfg, provider, on_event=emit,
                                   model_override=req.model)
        else:
            default = cfg.get("pipeline", {}).get(
                "default_agents", "architect_agent,dev_agent,senior_dev_agent")
            names = [a.strip() for a in (req.agents or default).split(",")]
            pipe = Pipeline.from_personas(names, cfg, provider,
                                          model_override=req.model)
        emit({"type": "start", "loads": pipe.loads(),
              "stages": [{"name": s.name, "model": s.model} for s in pipe.stages]})
        # Pipeline emits its own `done` with the final answer.
        pipe.run(req.prompt, plan=req.plan, on_event=emit)

    return _stream(work)


class AgentRequest(BaseModel):
    # Either run a SAVED agent by name, or describe an ad-hoc one inline. A saved
    # agent is run exactly as defined — re-reading fields from the request would let
    # the running copy drift from the saved one.
    saved: str | None = None
    goal: str = ""
    success: str = ""
    constraints: str = ""
    persona: str = "dev_agent"
    interval: int = 0
    max_cycles: int = 20     # per run, a resume gets a fresh budget; 0 = until done
    name: str = "default"
    model: str = ""          # override the persona's tag for this run
    network: bool = True     # public internet via the egress proxy; false seals it
    retrieval: bool = True
    max_failures: int = 5    # consecutive failed cycles before the agent gives up
    max_idle: int = 5        # consecutive cycles doing nothing before it gives up
    mcp: bool = True


# One agent at a time on this machine (see agent.py, "one agent at a time"). Two
# layers, because they guard different things:
#   _RUNNING + _RUN_GUARD  this process. Claimed in the request, under a lock, BEFORE
#                          answering — so a second click is refused at once.
#   the OS lock            every process (the CLI, another server). Taken by
#                          AgentLoop.run itself, so no entry point can skip it.
# The check used to read _RUNNING in the request but add to it in the worker thread;
# two requests landing in that gap both passed, and two loops ran.
_RUNNING: set[str] = set()
_RUN_GUARD = threading.Lock()


def _agent_holder() -> dict | None:
    from .agent import running_agent
    return running_agent()


def _busy() -> str | None:
    """The agent running anywhere on this machine, or None."""
    if _RUNNING:
        return next(iter(_RUNNING))
    held = _agent_holder()
    return held.get("name", "another agent") if held else None


@app.post("/api/agent/start")
def agent_start(req: AgentRequest):
    """Start agent mode and stream its cycles until it finishes or is stopped.

    The connection stays open for the life of the run, which for a 24/7 goal is the
    point — the browser is a viewer, not the owner. Closing the tab does not stop the
    agent; `/api/agent/stop` does, and so does `lag agent stop`.
    """
    from .agent import AgentLoop, GoalSpec

    cfg = load_config()
    run_name = req.saved or req.name
    # Check and claim in one step. 409 (Conflict) rather than a 200 carrying an error:
    # the console reads a 200 as the start of an event stream, found no events in
    # it, and showed nothing at all.
    with _RUN_GUARD:
        busy = _busy()
        if busy:
            return JSONResponse(status_code=409, content={
                "error": f"agent {busy!r} is already running — one agent at a time; "
                         f"stop it first", "running": busy})
        _RUNNING.add(run_name)

    def work(emit) -> None:
        try:
            if req.saved:
                spec = GoalSpec.load(req.saved)
                if not spec.enabled:
                    raise ValueError(f"agent {req.saved!r} is disabled")
            else:
                spec = GoalSpec(goal=req.goal, success=req.success,
                                constraints=req.constraints, persona=req.persona,
                                interval=req.interval,
                                max_cycles=req.max_cycles, name=req.name,
                                model=req.model or "", network=req.network,
                                max_failures=req.max_failures, max_idle=req.max_idle)
            prov = _provider(cfg, spec.mcp if req.saved else req.mcp)
            if not req.retrieval:
                prov = _without_retrieval(prov)
            AgentLoop(spec, cfg, prov).run(on_event=emit)
        finally:
            # Released here rather than in _stream, because "is this agent running"
            # is this endpoint's concern, not the stream helper's.
            _RUNNING.discard(run_name)

    try:
        return _stream(work)
    except BaseException:
        _RUNNING.discard(run_name)         # the worker never started, so release here
        raise


@app.get("/api/agent/running")
def agent_running():
    """Which agent is running on this machine, from any process — the console greys
    out its start buttons on this."""
    return {"running": _busy()}


@app.get("/api/agent/status")
def agent_status(name: str = "default"):
    """The run's own record, plus whether this process is currently driving it."""
    from .agent import AgentState

    st = AgentState.load(name)
    running = _busy() == name
    if not st:
        return {"exists": False, "running": running, "name": name}
    return {"exists": True, "running": running, "name": name,
            "goal": st.spec.get("goal", ""), "persona": st.spec.get("persona", ""),
            "cycle": st.cycle, "done": st.done, "summary": st.summary,
            "notes": st.notes[-20:], "errors": st.errors,
            "stopped_reason": st.stopped_reason,
            "consecutive_errors": st.consecutive_errors, "idle_cycles": st.idle_cycles,
            "started": st.started, "updated": st.updated}


@app.post("/api/agent/stop")
def agent_stop(name: str = "default"):
    """Ask the agent to stop after its current cycle."""
    from .agent import stop_path

    p = stop_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("stop")
    return {"stopping": True, "name": name, "stop_file": str(p)}


class AgentDef(BaseModel):
    """A saved agent's definition, as the console edits it."""

    name: str
    goal: str
    description: str = ""
    success: str = ""
    constraints: str = ""
    persona: str = "dev_agent"
    model: str = ""
    interval: int = 0
    max_cycles: int = 20     # per run, a resume gets a fresh budget; 0 = until done
    network: bool = True     # public internet via the egress proxy; false seals it
    max_failures: int = 5
    max_idle: int = 5
    enabled: bool = True
    mcp: bool = True
    tools: list[str] = []


def _spec_row(sp, running: bool) -> dict:
    """A definition plus whatever its last run left behind — the console shows both."""
    from .agent import AgentState

    st = AgentState.load(sp.name)
    row = {**{k: getattr(sp, k) for k in AgentDef.model_fields},
           "running": running, "unknown_tools": sp.unknown_tools()}
    row["state"] = None if not st else {
        "cycle": st.cycle, "done": st.done, "summary": st.summary,
        "errors": st.errors, "notes": st.notes[-10:],
        "stopped_reason": st.stopped_reason, "updated": st.updated,
    }
    return row


@app.get("/api/agents")
def agents_list():
    """Every saved agent, with its last run's outcome."""
    from .agent import GoalSpec

    busy = _busy()
    return {"agents": [_spec_row(sp, sp.name == busy) for sp in GoalSpec.list_all()],
            "running": busy}


@app.post("/api/agents")
def agents_save(d: AgentDef):
    """Create or update a saved agent. Editing never touches its run history.

    An update MERGES: only the fields the request actually carried change. The console
    form shows a subset of a definition, and replacing the whole thing reset every
    field it lacks to its default — re-saving switched a sealed agent's network back
    on, re-enabled a disabled one, and wiped its description and workspace.
    """
    from dataclasses import asdict

    from .agent import GoalSpec

    try:
        try:
            base = asdict(GoalSpec.load(d.name))
        except FileNotFoundError:
            base = {}                      # a new agent: unsent fields take defaults
        spec = GoalSpec(**{**base, **d.model_dump(exclude_unset=True)})
        path = spec.save()
    except (ValueError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"saved": spec.name, "path": str(path),
            "unknown_tools": spec.unknown_tools()}


@app.post("/api/agents/{name}/enabled")
def agents_set_enabled(name: str, on: bool = True):
    """Turn a saved agent on or off without deleting it."""
    from .agent import GoalSpec

    try:
        spec = GoalSpec.load(name)
    except (FileNotFoundError, ValueError) as e:
        return {"error": str(e)}
    spec.enabled = on
    spec.save()
    return {"name": name, "enabled": on}


@app.delete("/api/agents/{name}")
def agents_delete(name: str):
    """Delete a saved agent, its run state, and its stop flag."""
    from .agent import GoalSpec

    try:
        spec = GoalSpec.load(name)
    except (FileNotFoundError, ValueError) as e:
        return {"error": str(e)}
    if _busy() == name:                    # from this process or any other
        return {"error": f"{name} is running — stop it first"}
    spec.delete()
    return {"deleted": name}
