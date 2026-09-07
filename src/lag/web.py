"""Local web console: serves ui/console.html plus a small JSON API over the harness.

Fully local — the server binds to localhost and every call goes through the same
library the CLI uses (config, OllamaClient, Swarm, tools, rag). No cloud, no keys, so
the privacy stance is intact. This exists so the phosphor console reflects the *real*
harness instead of the scripted playback it shipped as.

PR1 is deliberately blocking: `/api/run` returns the finished answer and the browser
types it out with the animation the mockup already has. Live token/tool/agent streaming
is the next step (it needs an event hook in the loop), tracked separately.

The MCP tool provider is built once and cached: spawning both servers per request would
leak subprocesses (MCPBridge has no close() yet), and a single shared provider is fine
for a single-user console. Run the server from the repo root so the RAG server resolves
`./.chroma` — the same cwd caveat as `lag run --mcp`.
"""
from __future__ import annotations

from pathlib import Path

import ollama
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import load_config, load_persona
from .ollama_client import OllamaClient
from .tools import TOOLS, LocalToolProvider

_ROOT = Path(__file__).resolve().parents[2]
_UI = _ROOT / "ui" / "console.html"

app = FastAPI(title="lag console")

# Built once, reused across requests (see module docstring).
_provider_cache: dict = {}


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
    try:
        loaded = [
            {"name": m.model, "size_gb": round((m.size or 0) / 1e9, 2)}
            for m in client.ps().models
        ]
    except Exception:
        pass
    return up, available, loaded


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
    return {
        "version": "0.2.0",
        "ollama": {"up": up, "available": available, "loaded": loaded},
        "chat_model": chat_model,
        "num_ctx": (cfg.get("ollama", {}) or {}).get("num_ctx", 4096),
        "personas": _personas(cfg),
        "tools": list(TOOLS),
        "rag_tools": ["search_knowledge_base"],
        "chroma": _chroma_count(cfg),
        "mcp": {"tools_server": True, "rag_server": True},
    }


class RunRequest(BaseModel):
    prompt: str
    mode: str = "solo"  # solo | pipeline | parallel | crew
    agents: str | None = None
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
        if req.mode in (None, "", "solo"):
            model = cfg["models"]["worker"]
            client = OllamaClient.from_config(cfg, model)
            answer = client.run(req.prompt, tool_provider=provider, verbose=True)
            return {"answer": answer, "model": model}

        from .swarm import Swarm

        names = (
            [a.strip() for a in req.agents.split(",")]
            if req.agents
            else ["architect_agent", "dev_agent", "senior_dev_agent"]
        )
        answer = Swarm.from_personas(names, cfg, provider).run(req.prompt, mode=req.mode)
        return {"answer": answer, "model": cfg["models"].get("orchestrator", "")}
    except Exception as e:  # noqa: BLE001 - surface to the UI, never 500
        return {"error": f"{type(e).__name__}: {e}"}
