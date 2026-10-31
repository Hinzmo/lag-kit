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


def test_the_whole_toolset_is_registered():
    """TOOLS is the single registry — the MCP server iterates it, so a tool missing
    here is a tool the agent silently does not have over either transport."""
    assert set(TOOLS) == {
        # original eight
        "read_file", "write_file", "list_dir", "run_shell",
        "system_info", "remember", "recall", "web_search",
        # agent mode: termination condition, clock, search, and reading a page
        "goal_status", "note_progress", "mark_done",
        "now", "sleep_for", "find_files", "grep_files", "http_fetch",
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
        assert len(schemas) == len(TOOLS)
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


class TestAgentModeTools:
    """The eight tools added for agent mode.

    The goal tools are the loop's termination condition and the search tools exist
    partly so a small model does not reach for run_shell to do routine lookups, so
    both need to behave predictably on bad input rather than raise.
    """

    def test_find_files_matches_a_glob(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.txt").write_text("x")
        out = tools.find_files("*.py", str(tmp_path))
        assert "a.py" in out and "b.txt" not in out

    def test_find_files_reports_a_bad_directory_instead_of_raising(self):
        assert "not a directory" in tools.find_files("*", "/no/such/dir")

    def test_find_files_says_so_when_nothing_matches(self, tmp_path):
        assert "no files match" in tools.find_files("*.rs", str(tmp_path))

    def test_grep_files_returns_path_line_match(self, tmp_path):
        (tmp_path / "x.py").write_text("alpha\nbeta target here\ngamma\n")
        out = tools.grep_files("target", str(tmp_path))
        assert "x.py:2:" in out and "target" in out

    def test_grep_files_reports_a_bad_regex_instead_of_raising(self, tmp_path):
        assert "bad pattern" in tools.grep_files("(unclosed", str(tmp_path))

    def test_grep_files_says_so_when_nothing_matches(self, tmp_path):
        (tmp_path / "x.py").write_text("nothing here\n")
        assert "no matches" in tools.grep_files("zzz", str(tmp_path))

    def test_http_fetch_rejects_a_non_http_url_without_touching_the_network(self):
        # file:// would otherwise be a local-file read through an "outbound" tool
        assert "must start with http" in tools.http_fetch("file:///etc/passwd")
        assert "must start with http" in tools.http_fetch("")

    def test_http_fetch_blocks_ssrf_to_non_public_addresses(self):
        """http_fetch runs on the host, so a model tricked into fetching an internal
        address would reach it. IP literals keep this hermetic — no DNS, no network."""
        assert "loopback" in tools.http_fetch("http://127.0.0.1/")
        assert "loopback" in tools.http_fetch("http://[::1]/")
        # 169.254.169.254 is the cloud metadata address; it is link-local
        assert "link-local" in tools.http_fetch("http://169.254.169.254/latest/meta-data/")
        assert "private" in tools.http_fetch("http://10.0.0.5/")
        assert "private" in tools.http_fetch("http://192.168.1.1/")
        # and Ollama on localhost by port is exactly the kind of thing we are stopping
        assert "blocked" in tools.http_fetch("http://localhost:11434/api/tags")

    def test_ssrf_guard_allows_a_public_address(self):
        """A routable public IP literal passes the guard (it still would not connect in
        this test, but the guard must not be the thing that stops it)."""
        assert tools._ssrf_block_reason("http://93.184.216.34/") is None

    def test_sleep_for_is_capped(self, monkeypatch):
        """Capped at 300 so a stop request is never far from being noticed.

        time.sleep is stubbed: the point is the clamp, and a test that genuinely
        slept for the cap would take five minutes to prove it.
        """
        slept = []
        monkeypatch.setattr("time.sleep", slept.append)
        assert tools.sleep_for(99999) == "slept 300s"
        assert slept == [300]

    def test_sleep_for_clamps_negatives(self, monkeypatch):
        slept = []
        monkeypatch.setattr("time.sleep", slept.append)
        assert tools.sleep_for(-5) == "slept 0s"
        assert slept == [0]

    def test_mark_done_sets_the_termination_flag(self):
        tools.GOAL.update({"done": False, "summary": ""})
        tools.mark_done("  finished  ")
        assert tools.GOAL["done"] is True
        assert tools.GOAL["summary"] == "finished"

    def test_note_progress_ignores_an_empty_note(self):
        tools.GOAL["notes"] = []
        tools.note_progress("   ")
        assert tools.GOAL["notes"] == []
        tools.note_progress("real note")
        assert tools.GOAL["notes"] == ["real note"]

    def test_now_reports_elapsed_once_a_goal_is_running(self):
        tools.GOAL.update({"goal": "g", "started": "2020-01-01T00:00:00+00:00"})
        assert "goal running" in tools.now()

    def test_find_files_falls_back_to_a_recursive_search(self, tmp_path):
        """A bare glob only matches direct children, and small models write "*.py"
        when they mean "everything under here" — so src/ with files in src/lag/ came
        back empty and the model concluded there were none."""
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "deep.py").write_text("x")
        out = tools.find_files("*.py", str(tmp_path))
        assert "deep.py" in out
        assert "searched recursively" in out      # and it says so, rather than lying

    def test_find_files_prefers_a_direct_match_over_the_fallback(self, tmp_path):
        (tmp_path / "top.py").write_text("x")
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "deep.py").write_text("x")
        out = tools.find_files("*.py", str(tmp_path))
        assert "top.py" in out and "deep.py" not in out
        assert "recursively" not in out


class TestHttpFetchRedirects:
    """The SSRF guard vetted only the first URL, and urlopen follows 3xx on its own.

    So a public page that answers `302 Location: http://127.0.0.1:11434/...` turned
    one vetted request into an unvetted one to the host's own services. Every hop
    has to pass the same check as the first.

    No internet needed: a local server plays both parts. Its /r path stands in for
    the attacker's public page (the guard is told to treat that one URL as public),
    and everything it redirects to faces the real guard.
    """

    @pytest.fixture
    def server(self):
        import http.server
        import threading

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                port = self.server.server_address[1]
                if self.path == "/r":                       # "evil.example"
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{port}/secret")
                    self.end_headers()
                elif self.path == "/ftp":
                    self.send_response(302)
                    self.send_header("Location", "ftp://example.com/x")
                    self.end_headers()
                elif self.path == "/public-hop":           # redirect to another "public" page
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{port}/landing")
                    self.end_headers()
                else:
                    body = b"INTERNAL SECRET" if self.path == "/secret" else b"landing page"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{srv.server_address[1]}"
        srv.shutdown()
        srv.server_close()

    def _pretend_public(self, monkeypatch, *paths):
        real = tools._ssrf_block_reason
        monkeypatch.setattr(tools, "_ssrf_block_reason",
                            lambda url: None if url.endswith(paths) else real(url))

    def test_a_redirect_to_loopback_is_refused(self, server, monkeypatch):
        self._pretend_public(monkeypatch, "/r")
        out = tools.http_fetch(f"{server}/r")
        assert "INTERNAL SECRET" not in out
        assert "blocked" in out and "redirect" in out and "loopback" in out

    def test_a_redirect_off_http_is_refused(self, server, monkeypatch):
        self._pretend_public(monkeypatch, "/ftp")
        out = tools.http_fetch(f"{server}/ftp")
        assert "blocked" in out and "ftp://" in out

    def test_a_redirect_between_public_pages_still_works(self, server, monkeypatch):
        """The guard must not break ordinary redirects (http->https, moved pages)."""
        self._pretend_public(monkeypatch, "/public-hop", "/landing")
        assert "landing page" in tools.http_fetch(f"{server}/public-hop")


class TestSSRFAddressCoverage:
    """The guard listed bad kinds of address (loopback, private, ...) and allowed the
    rest, so a range nobody listed got through: 100.64.0.0/10 is neither private nor
    global, and it is where Tailscale puts every device on a tailnet.

    Default-deny (`not is_global`) closes that — but is not enough alone either:
    Python calls the NAT64 prefix 64:ff9b::/96 *global*, though 64:ff9b::7f00:1
    carries 127.0.0.1. IPv6 forms that embed an IPv4 address are unwrapped and the
    inner address is checked too.
    """

    @pytest.mark.parametrize("host", [
        "100.64.0.1",               # CGNAT / Tailscale device
        "100.100.100.100",          # Tailscale's MagicDNS
        "198.18.0.1",               # benchmarking range
        "[::ffff:127.0.0.1]",       # IPv4-mapped loopback
        "[::ffff:192.168.1.1]",     # IPv4-mapped LAN
        "[::ffff:100.64.0.1]",      # IPv4-mapped tailnet
        "[64:ff9b::7f00:1]",        # NAT64 -> 127.0.0.1 (is_global is True!)
        "[64:ff9b::a00:5]",         # NAT64 -> 10.0.0.5
        "[64:ff9b::6440:1]",        # NAT64 -> 100.64.0.1
        "[2002:7f00:1::]",          # 6to4 -> 127.0.0.1
        "[fd7a:115c:a1e0::1]",      # Tailscale IPv6 (ULA)
        "0.0.0.0",
    ])
    def test_non_public_addresses_are_refused(self, host):
        assert tools._ssrf_block_reason(f"http://{host}/") is not None, host

    def test_cgnat_refusal_names_the_range(self):
        """A Tailscale address should be recognisable from the message."""
        assert "100.64.0.0/10" in tools._ssrf_block_reason("http://100.100.100.100/")

    def test_nat64_refusal_names_the_embedded_address(self):
        assert "127.0.0.1" in tools._ssrf_block_reason("http://[64:ff9b::7f00:1]/")

    @pytest.mark.parametrize("host", ["8.8.8.8", "1.1.1.1", "[2606:4700:4700::1111]"])
    def test_public_addresses_still_pass(self, host):
        assert tools._ssrf_block_reason(f"http://{host}/") is None, host
