"""Bridge tests that do not require a running MCP server.

The schema translation is the load-bearing part of the bridge: MCP hands back
`{name, description, inputSchema}` and Ollama expects
`{type, function: {name, description, parameters}}`. If that mapping drifts,
`--mcp` fails in a way that looks like a model problem rather than a plumbing
one, so it is worth pinning directly.
"""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lag.mcp_client import MCPToolProvider


def _fake_bridge(tools):
    return SimpleNamespace(tools=tools, call=lambda name, args: f"called {name}")


JSON_SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}}}


@pytest.mark.parametrize("attr", ["input_schema", "inputSchema"])
def test_schema_translation_shape(attr):
    """mcp 2.x renamed inputSchema -> input_schema; both must translate."""
    tool = SimpleNamespace(
        name="read_file", description="Read a file.", **{attr: JSON_SCHEMA}
    )
    (schema,) = MCPToolProvider(_fake_bridge([tool])).schemas()

    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "read_file"
    assert fn["description"] == "Read a file."
    # the whole point: the MCP input schema becomes `parameters`, unchanged
    assert fn["parameters"] == JSON_SCHEMA


def test_real_sdk_tool_model_translates():
    """Guard against the next rename by using the SDK's own Tool model."""
    from mcp import Tool

    tool = Tool(name="list_dir", description="List.", inputSchema=JSON_SCHEMA)
    (schema,) = MCPToolProvider(_fake_bridge([tool])).schemas()
    assert schema["function"]["parameters"] == JSON_SCHEMA


def test_absent_schema_falls_back_to_empty_object():
    tool = SimpleNamespace(name="t", description=None)
    (schema,) = MCPToolProvider(_fake_bridge([tool])).schemas()
    assert schema["function"]["parameters"] == {"type": "object", "properties": {}}


def test_missing_description_becomes_empty_string():
    tool = SimpleNamespace(name="t", description=None, input_schema={})
    (schema,) = MCPToolProvider(_fake_bridge([tool])).schemas()
    assert schema["function"]["description"] == ""


def test_call_delegates_to_bridge():
    provider = MCPToolProvider(_fake_bridge([]))
    assert provider.call("list_dir", {"path": "."}) == "called list_dir"


def test_providers_share_the_duck_typed_interface():
    """The agent loop must not be able to tell the two providers apart."""
    from lag.tools import LocalToolProvider

    for provider in (LocalToolProvider(), MCPToolProvider(_fake_bridge([]))):
        assert callable(provider.schemas)
        assert callable(provider.call)


def test_from_config_spawns_both_servers_with_venv_interpreter(monkeypatch):
    """from_config now wires BOTH the tools and rag servers into one provider.

    Also a regression guard: the server was once spawned with a bare "python".
    Ubuntu ships only python3, and even where `python` resolves it is the system
    interpreter, which cannot import lag; the server dies on import and
    stdio_client waits forever on a pipe that never speaks — the historical
    "--mcp hangs" symptom. Every spawn must use sys.executable.
    """
    spawns = []

    class FakeBridge:
        def __init__(self, command, args):
            spawns.append((command, args))

    monkeypatch.setattr("lag.mcp_client.MCPBridge", FakeBridge)
    MCPToolProvider.from_config({})

    assert len(spawns) == 2
    assert all(command == sys.executable for command, _ in spawns)
    assert {Path(args[0]).name for _, args in spawns} == {
        "tools_server.py",
        "rag_server.py",
    }


def test_from_config_server_paths_are_cwd_independent(monkeypatch, tmp_path):
    """Every server path must resolve to the repo, not the caller's cwd."""
    spawns = []

    class FakeBridge:
        def __init__(self, command, args):
            spawns.append(args)

    monkeypatch.setattr("lag.mcp_client.MCPBridge", FakeBridge)
    monkeypatch.chdir(tmp_path)
    MCPToolProvider.from_config({})

    assert spawns  # sanity: something was spawned
    for args in spawns:
        assert Path(args[0]).is_absolute()
        assert Path(args[0]).exists()


# ── multi-bridge behaviour (Option A: MCPToolProvider holds a list of bridges) ──


def _tool(name):
    return SimpleNamespace(name=name, description="", input_schema=JSON_SCHEMA)


def test_multiple_bridges_merge_schemas():
    """schemas() flattens every server's tools into one list the model sees."""
    b1 = _fake_bridge([_tool("read_file"), _tool("run_shell")])
    b2 = _fake_bridge([_tool("search_knowledge_base")])
    names = [s["function"]["name"] for s in MCPToolProvider([b1, b2]).schemas()]
    assert names == ["read_file", "run_shell", "search_knowledge_base"]


def test_call_routes_to_the_owning_bridge():
    """A call goes to whichever server actually exposes that tool."""
    tools = SimpleNamespace(tools=[_tool("read_file")], call=lambda n, a: f"tools:{n}")
    rag = SimpleNamespace(tools=[_tool("search_knowledge_base")], call=lambda n, a: f"rag:{n}")
    provider = MCPToolProvider([tools, rag])
    assert provider.call("read_file", {}) == "tools:read_file"
    assert provider.call("search_knowledge_base", {}) == "rag:search_knowledge_base"


def test_unknown_tool_across_bridges_returns_string_not_raise():
    b1 = _fake_bridge([_tool("read_file")])
    b2 = _fake_bridge([_tool("search_knowledge_base")])
    assert MCPToolProvider([b1, b2]).call("nope", {}) == "unknown tool: nope"


def test_name_collision_first_bridge_wins():
    """A duplicated tool name is emitted once and routed to the first server."""
    first = SimpleNamespace(tools=[_tool("dup")], call=lambda n, a: "first")
    second = SimpleNamespace(tools=[_tool("dup")], call=lambda n, a: "second")
    provider = MCPToolProvider([first, second])
    assert [s["function"]["name"] for s in provider.schemas()] == ["dup"]
    assert provider.call("dup", {}) == "first"


def test_single_bridge_construction_still_works():
    """Passing one bridge (not a list) is normalized — preserves old callers/tests."""
    provider = MCPToolProvider(_fake_bridge([_tool("read_file")]))
    assert len(provider.bridges) == 1
    assert [s["function"]["name"] for s in provider.schemas()] == ["read_file"]


class TestBridgeShutdown:
    """Provider-level closing. The bridge's own teardown is tested against a real
    server in TestBridgeLifecycle below.

    Three tests used to live here that built an MCPBridge with fake context managers
    and checked close() called their __aexit__ in reverse order. They passed while the
    real SDK raised on every close: a fake does not care which task exits it, and
    which task exits it was the whole bug. Their promises are kept below against a
    real server; the ordering one is now guaranteed by nested `async with`.
    """

    def test_the_provider_closes_every_bridge(self):
        from lag.mcp_client import MCPToolProvider

        closed = []

        class FakeBridge:
            def __init__(self, tag): self.tag = tag; self.tools = []
            def close(self): closed.append(self.tag)

        MCPToolProvider([FakeBridge("a"), FakeBridge("b")]).close()
        assert closed == ["a", "b"]

    def test_one_bridge_failing_to_close_does_not_strand_the_others(self):
        from lag.mcp_client import MCPToolProvider

        closed = []

        class Bad:
            tools = []
            def close(self): raise RuntimeError("nope")

        class Good:
            tools = []
            def close(self): closed.append(1)

        MCPToolProvider([Bad(), Good()]).close()
        assert closed == [1]


# ── lifecycle: start, stop, and failing to start ────────────────────────────────
import contextlib  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from lag import mcp_client  # noqa: E402
from lag.mcp_client import MCPBridge  # noqa: E402
from mcp.shared.exceptions import MCPError  # noqa: E402

TOOLS_SERVER = str(Path(__file__).resolve().parents[1] / "servers" / "tools_server.py")


def _my_servers() -> int:
    """Server subprocesses whose parent is this test process."""
    out = subprocess.run(["ps", "-A", "-o", "ppid=,command="],
                         capture_output=True, text=True).stdout
    me = str(os.getpid())
    return sum(1 for line in out.splitlines()
               if line.split()[:1] == [me] and "_server.py" in line)


def _settle(pred, timeout=5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


@pytest.fixture
def broken_server(tmp_path):
    p = tmp_path / "broken_server.py"
    p.write_text("import sys; sys.exit('server died on import')\n")
    return str(p)


class TestBridgeLifecycle:
    def test_close_is_clean_not_lucky(self, monkeypatch):
        """close() used to exit the SDK's context managers from a different task than
        the one that entered them. anyio refuses that ("Attempted to exit cancel scope
        in a different task"), and close() swallowed the error; the server only died
        because the SDK happened to kill it before raising. Record what gets swallowed."""
        swallowed = []

        class Recording(contextlib.AbstractContextManager):
            def __init__(self, *exc):
                self.exc = exc

            def __exit__(self, t, v, tb):
                if t is not None:
                    swallowed.append(f"{t.__name__}: {v}")
                return t is not None and issubclass(t, self.exc)

        monkeypatch.setattr(mcp_client.contextlib, "suppress", Recording)
        threads = threading.active_count()
        b = MCPBridge(sys.executable, [TOOLS_SERVER])
        assert _my_servers() == 1
        b.close()
        assert swallowed == []
        assert getattr(b, "close_error", None) is None
        assert _settle(lambda: _my_servers() == 0), "the server outlived close()"
        assert _settle(lambda: threading.active_count() <= threads), "a thread outlived close()"

    def test_closing_twice_is_harmless(self):
        b = MCPBridge(sys.executable, [TOOLS_SERVER])
        b.close()
        b.close()
        assert b.close_error is None and not b._thread.is_alive()

    def test_a_server_that_already_died_does_not_make_close_raise(self):
        """Teardown also runs on failure paths, so it cannot itself fail."""
        import signal

        threads = threading.active_count()
        b = MCPBridge(sys.executable, [TOOLS_SERVER])
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,command="],
                             capture_output=True, text=True).stdout
        (pid,) = [int(r.split()[0]) for r in out.splitlines()
                  if r.split()[1] == str(os.getpid()) and "tools_server.py" in r]
        os.kill(pid, signal.SIGKILL)
        b.close()                                   # must not raise
        assert _settle(lambda: threading.active_count() <= threads)

    def test_a_tool_call_works_between_open_and_close(self):
        """The refactor must not cost the thing the bridge is for."""
        b = MCPBridge(sys.executable, [TOOLS_SERVER])
        try:
            assert any(t.name == "now" for t in b.tools)
            assert b.call("now", {}).strip()
            assert b.alive
        finally:
            b.close()
        # After close a call is a tool failure the model can read, not an exception
        # that ends the run.
        assert not b.alive
        assert "has stopped" in b.call("now", {})

    def test_a_failed_build_leaves_nothing_behind(self, broken_server):
        """The leak that actually happened: the first server starts, the second dies,
        and the first is orphaned — with its thread — on every retry. The console
        retries on every request while a server is broken."""
        threads = threading.active_count()
        for _ in range(2):
            with pytest.raises(MCPError, match="Connection closed"):
                MCPToolProvider.from_config({}, servers=(TOOLS_SERVER, broken_server))
        assert _settle(lambda: _my_servers() == 0), f"{_my_servers()} orphaned server(s)"
        assert _settle(lambda: threading.active_count() <= threads), (
            f"{threading.active_count() - threads} leaked thread(s)")


class _Closable:
    def __init__(self):
        self.tools, self.closed = [], 0

    def close(self):
        self.closed += 1


def test_the_provider_closes_its_bridges_as_a_context_manager():
    a, b = _Closable(), _Closable()
    with MCPToolProvider([a, b]):
        pass
    assert (a.closed, b.closed) == (1, 1)


def test_the_cli_closes_what_it_opens(monkeypatch):
    """`lag run --mcp` owned a provider and never closed it. Exiting the process
    happens to reap the servers (their stdin hits EOF), but only by accident."""
    from typer.testing import CliRunner

    from lag import cli

    built = []

    def fake_from_config(cls, cfg, servers=None):
        p = MCPToolProvider([_Closable()])
        built.append(p)
        return p

    monkeypatch.setattr(MCPToolProvider, "from_config", classmethod(fake_from_config))
    monkeypatch.setattr(cli.OllamaClient, "from_config",
                        classmethod(lambda c, *a, **k: type("C", (), {
                            "run": lambda s, *a, **k: "ok"})()))
    r = CliRunner().invoke(cli.app, ["run", "hi", "--mcp"])
    assert r.exit_code == 0, r.output
    assert built and built[0].bridges[0].closed == 1


def test_the_web_server_closes_its_cached_providers_on_shutdown():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lag import web

    bridge = _Closable()
    with TestClient(web.app):
        web._provider_cache["mcp"] = MCPToolProvider([bridge])
    assert bridge.closed == 1 and "mcp" not in web._provider_cache


def _server_pid(script: str) -> int:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,command="],
                         capture_output=True, text=True).stdout
    (pid,) = [int(r.split()[0]) for r in out.splitlines()
              if r.split()[1] == str(os.getpid()) and script in r]
    return pid


class TestCallsNeverRaise:
    """Over MCP a tool call used to raise on a timeout or a dead server, and the run
    ended. Every other tool failure is text the model reads; these are too."""

    def test_the_call_timeout_outlasts_every_tool(self):
        # sleep_for allows 300 s and run_shell 60 s. The tool's own limit has to fire
        # first, or a call that was still fine is abandoned.
        assert mcp_client.CALL_TIMEOUT > 300

    def test_a_slow_call_comes_back_as_text(self, monkeypatch):
        monkeypatch.setattr(mcp_client, "CALL_TIMEOUT", 0.5)
        b = MCPBridge(sys.executable, [TOOLS_SERVER])
        try:
            out = b.call("sleep_for", {"seconds": 5})
            assert "took longer than" in out
            assert b.alive                      # slow is not dead
        finally:
            b.close()

    def test_a_dead_server_is_reported_and_marks_the_bridge(self):
        import signal

        b = MCPBridge(sys.executable, [TOOLS_SERVER])
        try:
            os.kill(_server_pid("tools_server.py"), signal.SIGKILL)
            time.sleep(0.3)
            out = b.call("now", {})
            assert "failed" in out
            assert not b.alive
            provider = MCPToolProvider([b])
            assert not provider.alive           # so the console builds a new one
        finally:
            b.close()


class TestEmbedSettingsReachTheServers:
    """The RAG server searched with its built-in embed model, whatever `lag setup`
    had ingested with; a store of a different vector size then broke."""

    def test_config_becomes_server_flags(self):
        from lag.config import server_args

        cfg = {"models": {"embed": "mxbai-embed-large"}, "ollama_base_url": "http://h:1"}
        assert server_args(cfg) == ["--embed-model=mxbai-embed-large",
                                    "--ollama-host=http://h:1"]
        assert server_args({}) == []            # unset: the servers keep their defaults

    def test_every_server_is_started_with_them(self, monkeypatch):
        started = []

        class FakeBridge:
            def __init__(self, command, args):
                started.append(args)

        monkeypatch.setattr(mcp_client, "MCPBridge", FakeBridge)
        cfg = {"models": {"embed": "e:1"}, "ollama_base_url": "http://h:1"}
        MCPToolProvider.from_config(cfg)
        assert len(started) == 2
        for args in started:
            assert args[1:] == ["--embed-model=e:1", "--ollama-host=http://h:1"]

    def test_the_tools_server_accepts_them(self):
        b = MCPBridge(sys.executable, [TOOLS_SERVER, "--embed-model=e:1",
                                       "--ollama-host=http://127.0.0.1:9"])
        try:
            assert b.call("now", {}).strip()
        finally:
            b.close()
