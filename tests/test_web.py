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
        lambda cfg: (True, ["qwen3:14b"], [{"name": "qwen3:14b", "size_gb": 10.31}]),
    )
    monkeypatch.setattr(web, "_chroma_count", lambda cfg: {"available": True, "chunks": 9})

    d = client.get("/api/state").json()
    assert d["chat_model"]                       # resolved from config
    assert "read_file" in d["tools"]
    assert "search_knowledge_base" in d["rag_tools"]
    assert d["ollama"]["available"] == ["qwen3:14b"]
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
