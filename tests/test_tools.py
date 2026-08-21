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


def test_all_eight_tools_registered():
    assert set(TOOLS) == {
        "read_file", "write_file", "list_dir", "run_shell",
        "system_info", "remember", "recall", "web_search",
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


class TestWebSearch:
    """Parsing and failure modes. No test here touches the network."""

    PAGE = """
      <div class="result">
        <a rel="nofollow" class="result__a"
           href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs&amp;rut=x">
           The <b>Model</b> Context Protocol</a>
        <a class="result__snippet" href="#">A <b>protocol</b> for tools &amp; context.</a>
      </div>
      <div class="result">
        <a rel="nofollow" class="result__a" href="https://direct.example.org">Direct Link</a>
        <div class="result__snippet">No redirect wrapper here.</div>
      </div>
    """

    def _serve(self, monkeypatch, body=None, exc=None):
        class FakeResp:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def read(s): return (body or "").encode()
        def fake_urlopen(req, timeout=None):
            if exc: raise exc
            return FakeResp()
        monkeypatch.setattr(tools.urllib.request, "urlopen", fake_urlopen)

    def test_nested_tags_do_not_truncate_text(self, monkeypatch):
        # DuckDuckGo wraps query terms in <b>; an inner </b> must not end capture.
        self._serve(monkeypatch, self.PAGE)
        out = tools.web_search("mcp")
        assert "The Model Context Protocol" in out
        assert "A protocol for tools & context." in out

    def test_redirect_urls_are_unwrapped(self, monkeypatch):
        self._serve(monkeypatch, self.PAGE)
        out = tools.web_search("mcp")
        assert "https://example.com/docs" in out
        assert "uddg=" not in out

    def test_direct_urls_pass_through(self, monkeypatch):
        self._serve(monkeypatch, self.PAGE)
        assert "https://direct.example.org" in tools.web_search("mcp")

    def test_max_results_is_clamped(self, monkeypatch):
        self._serve(monkeypatch, self.PAGE)
        assert tools.web_search("mcp", max_results=1).count("http") == 1
        # out-of-range and junk values must not raise
        for bad in (0, -5, 99, "three", None):
            assert tools.web_search("mcp", max_results=bad)

    def test_empty_query_short_circuits(self, monkeypatch):
        self._serve(monkeypatch, self.PAGE)
        assert "empty query" in tools.web_search("   ")

    def test_no_results_is_reported(self, monkeypatch):
        self._serve(monkeypatch, "<html><body>nothing</body></html>")
        assert "no results" in tools.web_search("asdfqwerzxcv")

    def test_offline_degrades_to_a_message(self, monkeypatch):
        import urllib.error
        self._serve(monkeypatch, exc=urllib.error.URLError("Network unreachable"))
        out = tools.web_search("anything")
        assert "no network" in out and "offline" in out

    def test_http_error_degrades_to_a_message(self, monkeypatch):
        import urllib.error
        self._serve(monkeypatch, exc=urllib.error.HTTPError("u", 429, "Too Many", {}, None))
        assert "HTTP 429" in tools.web_search("anything")

    def test_unexpected_error_never_raises(self, monkeypatch):
        self._serve(monkeypatch, exc=RuntimeError("boom"))
        assert "RuntimeError" in tools.web_search("anything")


class TestRememberIndexing:
    """remember() writes to the key/value store AND the vector store."""

    def test_indexes_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "_MEMORY", tmp_path / "store.json")
        calls = []
        monkeypatch.setattr(tools, "_index_memory",
                            lambda k, v: calls.append((k, v)) or " · indexed 1 chunk(s)")
        out = tools.remember("colour", "blue")
        assert calls == [("colour", "blue")]
        assert "indexed" in out

    def test_kv_write_survives_indexing_failure(self, tmp_path, monkeypatch):
        """A broken embedder must not lose the fact."""
        monkeypatch.setattr(tools, "_MEMORY", tmp_path / "store.json")
        monkeypatch.setattr(tools, "_index_memory",
                            lambda k, v: (_ for _ in ()).throw(RuntimeError("no ollama")))
        with pytest.raises(RuntimeError):
            tools.remember("k", "v")
        assert tools.recall("k") == "v"   # written before indexing was attempted

    def test_index_helper_swallows_errors(self, monkeypatch):
        import lag.rag as rag
        monkeypatch.setattr(rag, "index_note",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        assert "not indexed" in tools._index_memory("k", "v")


class TestLocalToolProvider:
    def test_schemas_returns_callables(self):
        # Ollama derives JSON schemas from the function objects themselves,
        # so schemas() must hand back callables, not dicts.
        schemas = LocalToolProvider().schemas()
        assert len(schemas) == 8
        assert all(callable(s) for s in schemas)

    def test_call_dispatches(self):
        out = LocalToolProvider().call("run_shell", {"command": "echo hi"})
        assert "hi" in out

    def test_unknown_tool_returns_string(self):
        # A hallucinated tool name must degrade to a message, not an exception —
        # a raise here kills the loop instead of letting the model recover.
        out = LocalToolProvider().call("send_email", {"to": "x"})
        assert out == "unknown tool: send_email"

    def test_tool_error_is_returned_not_raised(self):
        out = LocalToolProvider().call("read_file", {"path": "/nonexistent/xyz"})
        assert out.startswith("error in read_file:")

    def test_bad_arguments_are_returned_not_raised(self):
        out = LocalToolProvider().call("read_file", {"wrong_kwarg": 1})
        assert out.startswith("error in read_file:")

    def test_call_with_none_args(self):
        assert json.loads(LocalToolProvider().call("system_info", None))
