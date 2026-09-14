"""Web console API tests.

The console renders itself entirely from /api/state, and shows /api/run failures as a
message rather than a broken request — so both contracts are worth pinning. Ollama and
Chroma are stubbed so these stay hermetic (no model, no server, no network).
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from lag import web  # noqa: E402

client = TestClient(web.app)


def test_index_serves_the_console():
    r = client.get("/")
    assert r.status_code == 200
    assert "phosphor console" in r.text.lower()


def test_state_reports_the_real_system(monkeypatch):
    monkeypatch.setattr(
        web, "_ollama_status",
        lambda cfg: (True, ["ornith:latest"], [{"name": "ornith:latest", "size_gb": 5.48}]),
    )
    monkeypatch.setattr(web, "_chroma_count", lambda cfg: {"available": True, "chunks": 9})

    d = client.get("/api/state").json()
    assert d["chat_model"]                       # resolved from config
    assert "read_file" in d["tools"]
    assert "search_knowledge_base" in d["rag_tools"]
    assert d["ollama"]["available"] == ["ornith:latest"]
    assert d["chroma"]["chunks"] == 9
    assert d["personas"], "real personas should load from personas/"
    assert all({"name", "role", "model"} <= set(p) for p in d["personas"])


def test_run_surfaces_errors_as_a_message(monkeypatch):
    """A failure must come back as {'error': ...} with 200, not a 500."""
    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(web.OllamaClient, "from_config", staticmethod(boom))
    r = client.post("/api/run", json={"prompt": "hi", "mode": "solo", "mcp": False})
    assert r.status_code == 200
    assert "nope" in r.json().get("error", "")


def test_state_reports_the_window_the_chat_model_actually_gets(monkeypatch):
    """With num_ctx_by_model the global default is not what the orchestrator runs at."""
    from lag import web

    cfg = {
        "ollama_base_url": "http://x",
        "models": {"orchestrator": "wide:9b", "worker": "narrow:8b"},
        "ollama": {"num_ctx": 8192, "num_ctx_by_model": {"wide:9b": 131072}},
        "paths": {"personas": "./personas"},
    }
    monkeypatch.setattr(web, "load_config", lambda: cfg)
    monkeypatch.setattr(web, "_ollama_status", lambda c: (True, [], []))
    monkeypatch.setattr(web, "_personas", lambda c: [])
    monkeypatch.setattr(web, "_chroma_count", lambda c: 0)

    d = web.state()
    assert d["chat_model"] == "wide:9b"
    assert d["num_ctx"] == 131072
