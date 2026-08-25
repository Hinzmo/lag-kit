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
