"""Tool-layer tests.

These deliberately avoid Ollama and MCP: the point is to pin the behaviour the
agent loop depends on — notably that tool errors come back as *strings* rather
than exceptions, since a raised exception kills the loop instead of letting the
model recover.
"""
from __future__ import annotations
import json

import pytest

from lag import tools
from lag.tools import TOOLS, LocalToolProvider


def test_all_seven_tools_registered():
    assert set(TOOLS) == {
        "read_file", "write_file", "list_dir", "run_shell",
        "system_info", "remember", "recall",
    }


def test_write_then_read_roundtrip(tmp_path):
    target = tmp_path / "nested" / "note.txt"
    msg = tools.write_file(str(target), "hello")
    assert "5 chars" in msg
    assert tools.read_file(str(target)) == "hello"


def test_write_file_creates_parent_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "c.txt"
    tools.write_file(str(target), "x")
    assert target.exists()


def test_list_dir_sorted_and_empty_marker(tmp_path):
    assert tools.list_dir(str(tmp_path)) == "(empty)"
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a.txt").write_text("")
    assert tools.list_dir(str(tmp_path)).splitlines() == ["a.txt", "b.txt"]


def test_run_shell_captures_stdout():
    assert "marker" in tools.run_shell("echo marker")


def test_run_shell_captures_stderr():
    # stderr is merged into the return value, so the model can see failures.
    assert "nope" in tools.run_shell("echo nope 1>&2")


def test_system_info_is_json_with_expected_keys():
    data = json.loads(tools.system_info())
    assert {"os", "release", "machine", "python", "cpu_count"} <= set(data)


def test_remember_recall_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_MEMORY", tmp_path / "store.json")
    tools.remember("colour", "blue")
    assert tools.recall("colour") == "blue"


def test_recall_missing_key_and_empty_store(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_MEMORY", tmp_path / "store.json")
    assert tools.recall("absent") == "(memory empty)"
    tools.remember("k", "v")
    assert "no memory for" in tools.recall("absent")


class TestLocalToolProvider:
    def test_schemas_returns_callables(self):
        # Ollama derives JSON schemas from the function objects themselves,
        # so schemas() must hand back callables, not dicts.
        schemas = LocalToolProvider().schemas()
        assert len(schemas) == 7
        assert all(callable(s) for s in schemas)

    def test_call_dispatches(self):
        out = LocalToolProvider().call("run_shell", {"command": "echo hi"})
        assert "hi" in out

    def test_unknown_tool_returns_string(self):
        # web_search is referenced by the research personas but not implemented;
        # this must degrade to a message, not an exception.
        out = LocalToolProvider().call("web_search", {"query": "x"})
        assert out == "unknown tool: web_search"

    def test_tool_error_is_returned_not_raised(self):
        out = LocalToolProvider().call("read_file", {"path": "/nonexistent/xyz"})
        assert out.startswith("error in read_file:")

    def test_bad_arguments_are_returned_not_raised(self):
        out = LocalToolProvider().call("read_file", {"wrong_kwarg": 1})
        assert out.startswith("error in read_file:")

    def test_call_with_none_args(self):
        assert json.loads(LocalToolProvider().call("system_info", None))
