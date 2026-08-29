"""CLI wiring tests.

The swarm command's --mcp flag must hand every agent the MCP toolset — that's how
the orchestrator gains search_knowledge_base for the memory-offload loop that makes a
small num_ctx workable. Without the flag, from_personas falls back to in-process tools
(which have `remember` but no semantic KB search). These pin that the flag routes a
provider through to Swarm.from_personas, without spawning real servers or a model.
"""
from __future__ import annotations

from typer.testing import CliRunner

from lag.cli import app

runner = CliRunner()


def _fake_swarm(monkeypatch):
    """Replace Swarm so no model or server is touched; capture what cli passed."""
    captured = {}

    class FakeSwarm:
        @classmethod
        def from_personas(cls, names, cfg, tool_provider=None):
            captured["provider"] = tool_provider
            captured["names"] = names
            return cls()

        def run(self, task, mode="crew"):
            return "ok"

    monkeypatch.setattr("lag.swarm.Swarm", FakeSwarm)
    return captured


def test_swarm_without_mcp_falls_back_to_in_process_tools(monkeypatch):
    captured = _fake_swarm(monkeypatch)
    result = runner.invoke(app, ["swarm", "do a thing", "--mode", "pipeline"])
    assert result.exit_code == 0
    # None → Swarm.from_personas builds a LocalToolProvider itself.
    assert captured["provider"] is None


def test_swarm_with_mcp_passes_an_mcp_provider(monkeypatch):
    captured = _fake_swarm(monkeypatch)
    sentinel = object()
    monkeypatch.setattr(
        "lag.mcp_client.MCPToolProvider.from_config",
        classmethod(lambda cls, cfg: sentinel),
    )
    result = runner.invoke(app, ["swarm", "do a thing", "--mode", "pipeline", "--mcp"])
    assert result.exit_code == 0
    assert captured["provider"] is sentinel
