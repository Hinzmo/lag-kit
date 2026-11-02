"""`lag goals`: the queue from the terminal, against a faked `lag serve`.

The CLI owns no state; it must send the right request and say the right thing.
"""
from __future__ import annotations

import io

import pytest
from typer.testing import CliRunner

from lag import goals_cli as gc
from lag.cli import app


def goal(id, text, status="pending", **kw):
    return {"id": id, "text": text, "status": status, "done_when": "", "agents": [],
            "route": [], "answer": "", "error": "", "added": "", "started": "",
            "finished": "", **kw}


SNAP = {"type": "goals", "running": False, "pausing": False, "current": None, "stage": "",
        "awaiting": None, "note": "", "settings": {"plan": True, "fail": "stop", "model": ""},
        "goals": [goal("aaa111bbb222", "first", "done", route=["dev_agent"], answer="## ok"),
                  goal("aaa999ccc000", "second", "failed", error="boom"),
                  goal("ddd444eee555", "third")]}


@pytest.fixture
def server(monkeypatch):
    calls = []

    def api(method, path, body=None, opener=None):
        calls.append((method, path, body))
        snap = {**SNAP, "goals": list(SNAP["goals"])}
        if method == "POST" and path == "/api/goals":
            if "bad_agent" in body["agents"]:
                return {**snap, "error": "unknown agent(s): bad_agent"}
            snap["goals"].append(goal("fff666", body["text"]))
        return snap
    monkeypatch.setattr(gc, "api", api)
    return calls


def run(*args):
    return CliRunner().invoke(app, ["goals", *args])


def test_ls_renders_the_queue(server):
    r = run("ls")
    assert r.exit_code == 0
    assert "queue paused · plan on · on fail: stop" in r.output
    assert "✓ aaa111  done     first" in r.output
    assert "error: boom" in r.output
    assert "agents: orchestrator picks" in r.output


def test_add_sends_the_goal_and_says_the_queue_is_paused(server):
    r = run("add", "do it", "--done-when", "tests pass", "--agents", "dev_agent, tester_agent")
    assert r.exit_code == 0, r.output
    assert server[-1] == ("POST", "/api/goals", {"text": "do it", "done_when": "tests pass",
                                                  "agents": ["dev_agent", "tester_agent"]})
    assert "added fff666" in r.output and "lag goals run" in r.output


def test_add_with_empty_agents_is_an_error_not_the_default_roster(server):
    r = run("add", "do it", "--agents", " , ")
    assert r.exit_code == 1 and "names no agents" in r.output
    assert server == []


def test_server_errors_are_reported(server):
    r = run("add", "do it", "--agents", "bad_agent")
    assert r.exit_code == 1 and "unknown agent(s): bad_agent" in r.output


@pytest.mark.parametrize("ref,want", [("aaa1", "first"), ("ddd", "third"), ("3", "third")])
def test_goals_resolve_by_prefix_or_position(ref, want):
    assert gc.resolve(SNAP, ref)["text"] == want


def test_an_ambiguous_prefix_is_refused():
    import typer
    with pytest.raises(typer.Exit):
        gc.resolve(SNAP, "aaa")


def test_rm_and_retry_use_the_full_id(server):
    assert run("rm", "ddd").exit_code == 0
    assert ("DELETE", "/api/goals/ddd444eee555", None) in server
    assert run("retry", "2").exit_code == 0
    assert ("POST", "/api/goals/aaa999ccc000/retry", None) in server


def test_show_prints_the_answer(server):
    r = run("show", "1")
    assert r.exit_code == 0 and "## ok" in r.output and "agents: dev_agent" in r.output


def test_set_validates_and_sends_only_what_changed(server):
    assert run("set").exit_code == 1
    assert run("set", "--fail", "explode").exit_code == 1
    assert run("set", "--no-plan").exit_code == 0
    assert server[-1] == ("POST", "/api/goals/settings", {"plan": False, "fail": None, "model": None})


def test_no_server_is_a_clear_message(monkeypatch):
    monkeypatch.setenv("LAG_URL", "http://127.0.0.1:9")
    r = run("ls")
    assert r.exit_code == 1 and "lag serve is not running at http://127.0.0.1:9" in r.output


def test_sse_parser_skips_comments_and_junk():
    lines = [b": keepalive\n", b"data: {\"type\": \"token\", \"text\": \"hi\"}\n", b"\n",
             b"data: not json\n"]
    assert list(gc.sse_events(lines)) == [{"type": "token", "text": "hi"}]


def test_printer_streams_tokens_inline_and_reports_outcomes():
    out = io.StringIO()
    p = gc.Printer(out)
    p.event({"type": "goals", "goals": [goal("g0", "already there", "done"),
                                        goal("g1", "the goal", "pending")]})
    p.event({"type": "goals", "goals": [goal("g0", "already there", "done"),
                                        goal("g1", "the goal", "running")]})
    p.event({"type": "routed", "agents": [{"name": "analyst_agent", "model": "m"}]})
    p.event({"type": "stage", "name": "analyst_agent", "model": "m"})
    p.event({"type": "tool", "name": "read_file", "pretty": "read_file src/x.py"})
    p.event({"type": "tool_result", "name": "read_file", "content": "line one\nline two"})
    p.event({"type": "token", "text": "Hel"})
    p.event({"type": "token", "text": "lo"})
    p.event({"type": "nudge", "name": "analyst_agent", "reason": "asked a question"})
    p.event({"type": "goals", "goals": [goal("g1", "the goal", "done"), goal("g2", "next")]})
    text = out.getvalue()
    assert "routed → analyst_agent" in text
    assert "── analyst_agent (m)" in text
    assert "→ read_file src/x.py" in text and "← read_file: line one" in text
    assert "Hello\n" in text
    assert "↺ sent back to analyst_agent: asked a question" in text
    assert "✓ g1 done: the goal" in text and "+ g2 added: next" in text
    assert "▶ g1 running: the goal" in text
    assert "already there" not in text          # the opening snapshot is not news
