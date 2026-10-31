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


class TestSavedAgents:
    """POST /api/agents — what the console's save button calls.

    The console form shows only some of an agent's fields. Saving used to REPLACE the
    whole definition with what the form sent, so every field it lacks reset to its
    default: re-saving turned a sealed agent's network back on, re-enabled a disabled
    one, and wiped its description and workspace.
    """

    @pytest.fixture(autouse=True)
    def _agents_dir(self, tmp_path, monkeypatch):
        from lag import agent as agent_mod
        monkeypatch.setattr(agent_mod, "AGENT_DIR", tmp_path / "agents")

    def test_resaving_keeps_the_fields_the_form_does_not_send(self, tmp_path):
        from lag.agent import GoalSpec
        GoalSpec(name="sealed", goal="old goal", network=False, enabled=False,
                 description="kept", workspace=str(tmp_path / "ws"), max_idle=9,
                 model="x:1b").save()
        r = client.post("/api/agents", json={"name": "sealed", "goal": "new goal",
                                             "tools": ["read_file"]})
        assert r.json()["saved"] == "sealed"
        sp = GoalSpec.load("sealed")
        assert sp.goal == "new goal" and sp.tools == ["read_file"]   # what was sent
        assert sp.network is False                                     # what was not
        assert sp.enabled is False
        assert sp.description == "kept"
        assert sp.workspace == str(tmp_path / "ws")
        assert sp.max_idle == 9 and sp.model == "x:1b"

    def test_a_field_that_is_sent_still_changes(self):
        from lag.agent import GoalSpec
        GoalSpec(name="a", goal="g").save()
        client.post("/api/agents", json={"name": "a", "goal": "g", "network": False,
                                         "max_cycles": 0})
        sp = GoalSpec.load("a")
        assert sp.network is False and sp.max_cycles == 0

    def test_a_new_agent_gets_the_defaults(self):
        from lag.agent import GoalSpec
        client.post("/api/agents", json={"name": "fresh", "goal": "g"})
        sp = GoalSpec.load("fresh")
        assert sp.network is True and sp.enabled is True and sp.max_cycles == 20


class TestOneAgentAtATime:
    """POST /api/agent/start claims the run BEFORE answering, under a lock.

    It used to check `name in _RUNNING` in the request but only add the name later,
    inside the worker thread. Two clicks landing in that gap both passed the check
    and two loops ran. The fake _stream below never starts the worker, which holds
    that gap open forever and makes the race deterministic.
    """

    @pytest.fixture(autouse=True)
    def _clean(self, tmp_path, monkeypatch):
        from lag import agent as agent_mod
        monkeypatch.setattr(agent_mod, "AGENT_DIR", tmp_path / "agents")
        monkeypatch.setattr(web, "_stream", lambda work: {"started": True})
        web._RUNNING.clear()
        yield
        web._RUNNING.clear()

    def test_a_second_start_is_refused_before_the_first_one_begins(self):
        assert client.post("/api/agent/start", json={"goal": "g", "name": "a"}).json() \
            == {"started": True}
        r = client.post("/api/agent/start", json={"goal": "g", "name": "a"})
        assert r.status_code == 409 and "already running" in r.json()["error"]

    def test_a_different_agent_is_refused_too(self):
        """One at a time on the machine, not one per name: one model is resident."""
        client.post("/api/agent/start", json={"goal": "g", "name": "a"})
        r = client.post("/api/agent/start", json={"goal": "g", "name": "b"})
        assert r.status_code == 409 and "'a'" in r.json()["error"]

    def test_a_run_from_another_process_is_seen(self, monkeypatch):
        """The CLI holds the OS lock, not this server's set — both are checked."""
        monkeypatch.setattr(web, "_agent_holder", lambda: {"name": "cli-agent", "pid": 1})
        r = client.post("/api/agent/start", json={"goal": "g", "name": "a"})
        assert r.status_code == 409 and "cli-agent" in r.json()["error"]

    def test_running_reports_what_is_running(self, monkeypatch):
        assert client.get("/api/agent/running").json()["running"] is None
        client.post("/api/agent/start", json={"goal": "g", "name": "a"})
        assert client.get("/api/agent/running").json()["running"] == "a"
