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
    """Chat runs on the worker, at the worker's own window — not the orchestrator's,
    and not the global default."""
    from lag import web

    cfg = {
        "ollama_base_url": "http://x",
        "models": {"orchestrator": "wide:9b", "worker": "narrow:8b"},
        "ollama": {"num_ctx": 8192,
                   "num_ctx_by_model": {"wide:9b": 131072, "narrow:8b": 32768}},
        "paths": {"personas": "./personas"},
    }
    monkeypatch.setattr(web, "load_config", lambda: cfg)
    monkeypatch.setattr(web, "_ollama_status", lambda c: (True, [], []))
    monkeypatch.setattr(web, "_personas", lambda c: [])
    monkeypatch.setattr(web, "_chroma_count", lambda c: 0)

    d = web.state()
    assert d["chat_model"] == "narrow:8b"
    assert d["num_ctx"] == 32768


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


# ── goal pipeline ─────────────────────────────────────────────────────────────

@pytest.fixture
def goal_runner(tmp_path, monkeypatch):
    """A real GoalRunner on a temp file, with an executor that needs no Ollama."""
    from lag import agent as agent_mod
    from lag.goals import GoalRunner

    monkeypatch.setattr(agent_mod, "AGENT_DIR", tmp_path / "agents")
    runner = GoalRunner(lambda g, s, emit: (["dev_agent"], "ok"), tmp_path / "goals.json")
    monkeypatch.setitem(web._goal_runner_cache, "runner", runner)
    yield runner
    runner.pause()


def test_goals_can_be_added_listed_and_removed(goal_runner):
    d = client.post("/api/goals", json={"text": "ship it", "done_when": "merged"}).json()
    assert [g["text"] for g in d["goals"]] == ["ship it"]
    gid = d["goals"][0]["id"]
    assert client.get("/api/goals").json()["goals"][0]["done_when"] == "merged"
    d = client.delete(f"/api/goals/{gid}").json()
    assert d["goals"] == []


def test_picking_an_agent_that_does_not_exist_is_refused(goal_runner):
    d = client.post("/api/goals", json={"text": "x", "agents": ["ghost_agent"]}).json()
    assert "ghost_agent" in d["error"] and d["goals"] == []


def test_a_refused_goal_edit_is_a_message_not_a_500(goal_runner):
    r = client.post("/api/goals", json={"text": "  "})
    assert r.status_code == 200 and r.json()["error"]
    r = client.delete("/api/goals/nope")
    assert r.status_code == 200 and r.json()["error"] == "no such goal"


def test_goal_settings_round_trip(goal_runner):
    d = client.post("/api/goals/settings", json={"fail": "ask", "plan": False}).json()
    assert d["settings"]["fail"] == "ask" and d["settings"]["plan"] is False
    assert "error" in client.post("/api/goals/settings", json={"fail": "nope"}).json()


def test_run_works_the_queue(goal_runner):
    import time

    client.post("/api/goals", json={"text": "a"})
    client.post("/api/goals/run")
    for _ in range(200):
        if goal_runner.goals[0].status == "done":
            break
        time.sleep(0.01)
    assert goal_runner.goals[0].status == "done"
    assert goal_runner.goals[0].answer == "ok"


# ── live model watch ────────────────────────────────────────────────────────

def _collect(signatures, n_checks):
    """Run _model_events against a scripted sequence of fingerprints."""
    import asyncio

    from lag import web

    sigs, snaps, frames = list(signatures), [], []
    checks = {"n": 0}

    def signature():
        return sigs.pop(0) if sigs else sigs_last[0]
    sigs_last = [signatures[-1]]

    def snapshot():
        snaps.append(1)
        return {"available": [f"m{len(snaps)}"]}

    async def alive():
        checks["n"] += 1
        return checks["n"] <= n_checks

    async def nap(_):
        return None

    async def run():
        async for f in web._model_events(signature, snapshot, alive, sleep=nap):
            frames.append(f)
    asyncio.run(run())
    return frames, len(snaps)


def test_model_watch_pushes_only_on_a_change():
    frames, built = _collect([("a",), ("a",), ("b",), ("b",), ("b",)], 5)
    data = [f for f in frames if f.startswith("data:")]
    assert len(data) == 2 and built == 2   # first sight, then the change; quiet otherwise


def test_model_watch_pushes_ollama_going_away():
    frames, _ = _collect([("a",), None], 2)
    assert len([f for f in frames if f.startswith("data:")]) == 2


def test_model_watch_sends_keep_alives_when_quiet():
    from lag import web
    frames, built = _collect([("a",)], web.HEARTBEAT_EVERY + 1)
    assert built == 1 and frames.count(": keep-alive\n\n") == 1


def test_model_watch_stops_when_the_client_leaves():
    frames, built = _collect([("a",)], 0)
    assert frames == [] and built == 0


def test_model_signature_includes_digests_and_residents(monkeypatch):
    from types import SimpleNamespace as NS

    from lag import web

    class C:
        def __init__(self, host=None): pass
        def list(self): return NS(models=[NS(model="b:1", digest="d2"), NS(model="a:1", digest="d1")])
        def ps(self): return NS(models=[NS(model="a:1")])
    monkeypatch.setattr(web.ollama, "Client", C)
    assert web._models_signature({}) == ((("a:1", "d1"), ("b:1", "d2")), ("a:1",))

    class Down(C):
        def list(self): raise ConnectionError
    monkeypatch.setattr(web.ollama, "Client", Down)
    assert web._models_signature({}) is None


# ── one model in memory: warm-up and sharing ─────────────────────────────────

def test_warm_caches_the_tool_list_with_two_chats_that_part_at_the_question(monkeypatch):
    from lag import web
    from lag import ollama_client
    calls = []

    class C:
        def __init__(self, host=None): pass
        def show(self, model): return type("S", (), {"capabilities": ["completion", "tools"]})()
        def chat(self, **kw): calls.append(kw)
    monkeypatch.setattr(web.ollama, "Client", C)
    monkeypatch.setattr(ollama_client, "_CAPS", {})
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    monkeypatch.setattr(web, "_provider", lambda cfg, mcp: type("P", (), {"schemas": lambda self: tools})())
    monkeypatch.setattr(web, "load_config", lambda: {
        "models": {"worker": "m:1"},
        "ollama": {"num_ctx": 8192, "num_ctx_by_model": {"m:1": 16384}, "keep_alive": "30m"}})
    web._warm()
    # Same model, window, keep_alive and tool list a chat sends; questions differ at once.
    assert [c["messages"][0]["content"][0] for c in calls] == ["a", "b"]
    for c in calls:
        assert c["model"] == "m:1" and c["tools"] == tools and c["keep_alive"] == "30m"
        assert c["options"] == {"num_ctx": 16384, "num_predict": 1}


def test_warm_never_raises(monkeypatch):
    from lag import web

    def boom():
        raise RuntimeError("no config")
    monkeypatch.setattr(web, "load_config", boom)
    web._warm()


def test_chat_shares_the_running_goals_model(monkeypatch):
    import json as _json
    from types import SimpleNamespace as NS

    from lag import web
    seen = {}

    class FakeClient:
        @classmethod
        def from_config(cls, cfg, model):
            seen["model"] = model
            return NS(run=lambda prompt, tool_provider=None, on_event=None: "ok")
    monkeypatch.setattr(web, "OllamaClient", FakeClient)
    monkeypatch.setattr(web, "_provider", lambda cfg, mcp: object())
    monkeypatch.setitem(web._goal_runner_cache, "runner",
                        NS(current="g1", model="gemma4:e4b"))
    r = client.post("/api/stream", json={"prompt": "hi", "mode": "solo", "model": "other:7b"})
    events = [_json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]
    assert seen["model"] == "gemma4:e4b"
    assert {"type": "shared_model", "model": "gemma4:e4b"} in events


def test_chat_keeps_its_model_when_no_goal_runs(monkeypatch):
    from types import SimpleNamespace as NS

    from lag import web
    seen = {}

    class FakeClient:
        @classmethod
        def from_config(cls, cfg, model):
            seen["model"] = model
            return NS(run=lambda prompt, tool_provider=None, on_event=None: "ok")
    monkeypatch.setattr(web, "OllamaClient", FakeClient)
    monkeypatch.setattr(web, "_provider", lambda cfg, mcp: object())
    monkeypatch.setitem(web._goal_runner_cache, "runner", NS(current=None, model=""))
    client.post("/api/stream", json={"prompt": "hi", "mode": "solo", "model": "other:7b"})
    assert seen["model"] == "other:7b"


class TestOnlyKnownCallers:
    """127.0.0.1 keeps other machines out, not other web pages. A page whose name was
    re-pointed at 127.0.0.1 (DNS rebinding) arrives with its own name in Host; a page
    posting a form here arrives with its own Origin."""

    @pytest.fixture(autouse=True)
    def _agents_dir(self, tmp_path, monkeypatch):
        from lag import agent as agent_mod
        monkeypatch.setattr(agent_mod, "AGENT_DIR", tmp_path / "agents")
        monkeypatch.setattr(web, "_extra_hosts", set())

    def test_an_unknown_host_is_refused(self):
        r = TestClient(web.app, base_url="http://rebound.example:8765").get("/")
        assert r.status_code == 403 and "not allowed" in r.json()["error"]

    @pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765", "[::1]:8765",
                                      "LOCALHOST"])
    def test_loopback_names_are_served(self, host):
        # Set as a header: the test client cannot parse a bracketed IPv6 base URL.
        assert client.get("/", headers={"host": host}).status_code == 200

    def test_a_bind_address_can_be_added(self):
        web.allow_host("100.64.0.7")
        assert TestClient(web.app, base_url="http://100.64.0.7:8765").get("/").status_code == 200

    def test_a_cross_site_post_is_refused_and_does_nothing(self, tmp_path):
        r = client.post("/api/agent/stop?name=x", headers={"Origin": "http://evil.example"})
        assert r.status_code == 403
        assert not (tmp_path / "agents" / "x.stop").exists()

    def test_a_null_origin_is_refused(self):
        r = client.post("/api/agent/stop?name=x", headers={"Origin": "null"})
        assert r.status_code == 403

    def test_the_console_itself_may_post(self, tmp_path):
        r = client.post("/api/agent/stop?name=x", headers={"Origin": "http://testserver"})
        assert r.status_code == 200
        assert (tmp_path / "agents" / "x.stop").exists()

    def test_a_post_with_no_origin_is_not_from_a_page(self):
        """curl and `lag goals` send no Origin."""
        assert client.post("/api/agent/stop?name=x").status_code == 200

    def test_cross_site_reads_are_left_to_the_browser(self):
        """A GET is not refused, but nothing grants a foreign page the right to read it."""
        r = client.get("/", headers={"Origin": "http://evil.example"})
        assert r.status_code == 200
        assert "access-control-allow-origin" not in r.headers


class TestRunNamesAreRefusedUpFront:
    @pytest.fixture(autouse=True)
    def _agents_dir(self, tmp_path, monkeypatch):
        from lag import agent as agent_mod
        monkeypatch.setattr(agent_mod, "AGENT_DIR", tmp_path / "agents")

    def test_start_refuses_a_path_name_before_claiming_the_lock(self):
        r = client.post("/api/agent/start", json={"goal": "g", "name": "../../.."})
        assert r.status_code == 400 and "invalid agent name" in r.json()["error"]
        assert not web._RUNNING

    def test_stop_refuses_a_path_name(self, tmp_path):
        r = client.post("/api/agent/stop?name=../../x")
        assert r.status_code == 400
        assert not list(tmp_path.rglob("*.stop"))

    def test_status_for_a_path_name_is_simply_absent(self):
        assert client.get("/api/agent/status?name=../../x").json()["exists"] is False


class TestSharedObjectsAreBuiltOnce:
    """The warm-up thread, chat requests and the goal runner build these from separate
    threads. Unlocked, two could each build one and the cache kept only the last."""

    def test_concurrent_requests_build_one_provider(self, monkeypatch):
        import threading
        import time

        from lag import mcp_client

        built = []

        class SlowProvider:
            alive = True

            @classmethod
            def from_config(cls, cfg):
                built.append(1)
                time.sleep(0.2)                 # starting two servers takes a while
                return cls()

        monkeypatch.setattr(mcp_client, "MCPToolProvider", SlowProvider)
        monkeypatch.setattr(web, "_provider_cache", {})
        got = []
        threads = [threading.Thread(target=lambda: got.append(web._provider({}, True)))
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(built) == 1 and len({id(p) for p in got}) == 1

    def test_a_dead_provider_is_replaced(self, monkeypatch):
        from lag import mcp_client

        closed = []

        class Provider:
            def __init__(self):
                self.alive = True

            @classmethod
            def from_config(cls, cfg):
                return cls()

            def close(self):
                closed.append(self)

        monkeypatch.setattr(mcp_client, "MCPToolProvider", Provider)
        monkeypatch.setattr(web, "_provider_cache", {})
        first = web._provider({}, True)
        first.alive = False                     # a server died
        second = web._provider({}, True)
        assert second is not first and closed == [first]

    def test_concurrent_requests_build_one_goal_runner(self, monkeypatch):
        import threading
        import time

        from lag import goals

        built = []

        class SlowRunner:
            def __init__(self, execute):
                built.append(1)
                time.sleep(0.2)

        monkeypatch.setattr(goals, "GoalRunner", SlowRunner)
        monkeypatch.setattr(goals, "pipeline_executor", lambda cfg, f: None)
        monkeypatch.setattr(web, "load_config", dict)
        monkeypatch.setattr(web, "_goal_runner_cache", {})
        got = []
        threads = [threading.Thread(target=lambda: got.append(web._goal_runner()))
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(built) == 1 and len({id(r) for r in got}) == 1
