"""CLI wiring tests.

The pipeline command's --mcp flag must hand every stage the MCP toolset — that's how
the lead gains search_knowledge_base for the memory-offload loop that makes a small
num_ctx workable. Without the flag, from_personas falls back to in-process tools (which
have `remember` but no semantic KB search). These pin that the flag routes a provider
through to Pipeline.from_personas, without spawning real servers or a model.
"""
from __future__ import annotations

from typer.testing import CliRunner

from lag.cli import app

runner = CliRunner()


def _fake_pipeline(monkeypatch):
    """Replace Pipeline so no model or server is touched; capture what cli passed."""
    captured = {}

    class FakeStage:
        name, model = "fake_agent", "fake:1b"

    class FakePipeline:
        stages = [FakeStage()]

        @classmethod
        def from_personas(cls, names, cfg, tool_provider=None):
            captured["provider"] = tool_provider
            captured["names"] = names
            captured["routed"] = False
            return cls()

        @classmethod
        def routed(cls, task, cfg, tool_provider=None, lead_name=None, on_event=None):
            captured["provider"] = tool_provider
            captured["routed"] = True
            captured["task"] = task
            return cls()

        def loads(self):
            return 1

        def run(self, task, plan=True):
            captured["plan"] = plan
            return "ok"

    monkeypatch.setattr("lag.pipeline.Pipeline", FakePipeline)
    return captured


def test_pipeline_without_mcp_falls_back_to_in_process_tools(monkeypatch):
    captured = _fake_pipeline(monkeypatch)
    result = runner.invoke(app, ["pipeline", "do a thing", "--no-route"])
    assert result.exit_code == 0
    # None → Pipeline.from_personas builds a LocalToolProvider itself.
    assert captured["provider"] is None


def test_pipeline_with_mcp_passes_an_mcp_provider(monkeypatch):
    captured = _fake_pipeline(monkeypatch)
    sentinel = object()
    monkeypatch.setattr(
        "lag.mcp_client.MCPToolProvider.from_config",
        classmethod(lambda cls, cfg: sentinel),
    )
    result = runner.invoke(app, ["pipeline", "do a thing", "--mcp", "--no-route"])
    assert result.exit_code == 0
    assert captured["provider"] is sentinel


def test_agents_option_sets_stage_order(monkeypatch):
    """Order is meaning *and* the cost model, so the CLI must preserve it verbatim."""
    captured = _fake_pipeline(monkeypatch)
    result = runner.invoke(
        app, ["pipeline", "x", "--agents", "architect_agent, dev_agent ,tester_agent"]
    )
    assert result.exit_code == 0
    assert captured["names"] == ["architect_agent", "dev_agent", "tester_agent"]


def test_no_plan_skips_the_lead_briefing(monkeypatch):
    captured = _fake_pipeline(monkeypatch)
    assert runner.invoke(app, ["pipeline", "x", "--no-route"]).exit_code == 0
    assert captured["plan"] is True
    assert runner.invoke(app, ["pipeline", "x", "--no-plan", "--no-route"]).exit_code == 0
    assert captured["plan"] is False


def test_swarm_command_is_gone(monkeypatch):
    """`parallel` could not work under a one-resident-model cap; the name went with it."""
    assert runner.invoke(app, ["swarm", "do a thing"]).exit_code != 0



def test_routing_is_the_default(monkeypatch):
    """A fixed roster runs the same three agents whatever the task; routing is the point."""
    captured = _fake_pipeline(monkeypatch)
    assert runner.invoke(app, ["pipeline", "research three vector stores"]).exit_code == 0
    assert captured["routed"] is True
    assert captured["task"] == "research three vector stores"


def test_naming_agents_explicitly_skips_the_router(monkeypatch):
    """Clicking one agent in the console is the caller having already decided."""
    captured = _fake_pipeline(monkeypatch)
    assert runner.invoke(app, ["pipeline", "x", "--agents", "tester_agent"]).exit_code == 0
    assert captured["routed"] is False
    assert captured["names"] == ["tester_agent"]
