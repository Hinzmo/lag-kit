"""Bridge tests that do not require a running MCP server.

The schema translation is the load-bearing part of the bridge: MCP hands back
`{name, description, inputSchema}` and Ollama expects
`{type, function: {name, description, parameters}}`. If that mapping drifts,
`--mcp` fails in a way that looks like a model problem rather than a plumbing
one, so it is worth pinning directly.
"""
from __future__ import annotations
import sys
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


def test_from_config_uses_venv_interpreter(monkeypatch):
    """Regression: the server was spawned with a bare "python".

    Ubuntu ships only python3, and even where `python` resolves it is the
    system interpreter, which cannot import lag. The server then dies on import
    and stdio_client waits forever on a pipe that never speaks — the historical
    "--mcp hangs" symptom.
    """
    captured = {}

    class FakeBridge:
        def __init__(self, command, args):
            captured["command"] = command
            captured["args"] = args

    monkeypatch.setattr("lag.mcp_client.MCPBridge", FakeBridge)
    MCPToolProvider.from_config({})

    assert captured["command"] == sys.executable
    assert captured["args"][0].endswith("servers/tools_server.py")


def test_from_config_server_path_is_cwd_independent(monkeypatch, tmp_path):
    """The server path must resolve to the repo, not the caller's cwd."""
    captured = {}

    class FakeBridge:
        def __init__(self, command, args):
            captured["args"] = args

    monkeypatch.setattr("lag.mcp_client.MCPBridge", FakeBridge)
    monkeypatch.chdir(tmp_path)
    MCPToolProvider.from_config({})

    from pathlib import Path

    assert Path(captured["args"][0]).is_absolute()
    assert Path(captured["args"][0]).exists()
