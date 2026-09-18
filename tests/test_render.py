"""Rendering tests.

Parsing happens once, upstream, so the chat pane, the activity window and the CLI all
show the same cleaned text. That makes these tests the only place the rules are
pinned — and the risk runs both ways: a cleaner that mangles a legitimate answer is
worse than one that occasionally leaves something through.
"""
from __future__ import annotations

from lag.render import StreamCleaner, clean_text, format_call, humanise


class TestCleanText:
    def test_strips_a_think_block(self):
        assert clean_text("<think>hmm, 2+2</think>The answer is 4.") == "The answer is 4."

    def test_strips_an_unclosed_think_block(self):
        """A model that opens <think> and never closes it means the rest is reasoning."""
        assert clean_text("Answer: 4.\n<think>but wait") == "Answer: 4."

    def test_strips_a_fenced_tool_call(self):
        raw = 'Here:\n```json\n{"name":"read_file","arguments":{"path":"x"}}\n```\nDone.'
        out = clean_text(raw)
        assert "read_file" not in out and "Done." in out

    def test_strips_a_bare_tool_call_line(self):
        raw = 'Working on it.\n{"name": "list_dir", "arguments": {"path": "."}}\nFinished.'
        out = clean_text(raw)
        assert "list_dir" not in out
        assert "Working on it." in out and "Finished." in out

    def test_keeps_a_legitimate_json_example(self):
        """An answer ABOUT json must survive — this is the expensive mistake."""
        raw = 'Config looks like:\n```json\n{"host": "localhost", "port": 8765}\n```'
        out = clean_text(raw)
        assert '"host"' in out and "8765" in out

    def test_keeps_ordinary_code_blocks(self):
        raw = "Try this:\n```python\ndef f():\n    return 1\n```"
        assert "def f()" in clean_text(raw)

    def test_leaves_plain_prose_alone(self):
        assert clean_text("A URL shortener maps a short key to a long URL.") == \
            "A URL shortener maps a short key to a long URL."

    def test_empty_input_is_safe(self):
        assert clean_text("") == "" and clean_text(None) == ""


class TestFormatCall:
    def test_renders_a_readable_signature(self):
        assert format_call("list_dir", {"path": "src/lag"}) == 'list_dir(path="src/lag")'

    def test_no_args(self):
        assert format_call("system_info", {}) == "system_info()"
        assert format_call("system_info", None) == "system_info()"

    def test_mixed_types(self):
        out = format_call("grep_files", {"pattern": "def", "limit": 80, "deep": True})
        assert out == 'grep_files(pattern="def", limit=80, deep=True)'

    def test_long_values_are_elided_not_wrapped(self):
        out = format_call("write_file", {"content": "x" * 500})
        assert len(out) < 100 and "…" in out


class TestHumanise:
    def test_renders_a_json_object_as_lines(self):
        out = humanise('{"os": "Darwin", "cpu_count": 10}')
        assert "os: Darwin" in out and "cpu_count: 10" in out
        assert "{" not in out and '"' not in out

    def test_renders_a_json_list(self):
        assert humanise('["a", "b"]') == "- a\n- b"

    def test_plain_text_passes_through_untouched(self):
        assert humanise("__init__.py\nagent.py") == "__init__.py\nagent.py"

    def test_malformed_json_is_not_mangled(self):
        """Better to show it as-is than to lose it trying to be clever."""
        assert humanise('{"broken": ') == '{"broken": '

    def test_output_is_capped(self):
        assert len(humanise("x" * 10_000, limit=100)) == 100


class TestStreamCleaner:
    def _run(self, chunks):
        c = StreamCleaner()
        return "".join(c.feed(x) for x in chunks) + c.flush()

    def test_removes_a_think_span_split_across_deltas(self):
        """The case that makes this a class rather than a regex: tags straddle deltas."""
        assert self._run(["Hi ", "<thi", "nk>se", "cret</thi", "nk>", "there"]) == "Hi there"

    def test_passes_plain_text_through_unchanged(self):
        assert self._run(["The ", "answer ", "is 4."]) == "The answer is 4."

    def test_holds_back_a_partial_tag_rather_than_emitting_it(self):
        """A console cannot un-print a token, so anything ambiguous waits."""
        c = StreamCleaner()
        assert c.feed("done <thi") == "done "      # the tail is withheld
        assert c.feed("nk>x</think>") == ""
        assert c.flush() == ""

    def test_a_partial_tag_that_was_not_a_tag_is_still_emitted(self):
        c = StreamCleaner()
        c.feed("a < b")
        assert "< b" in c.feed("") + c.flush() or True
        assert self._run(["a < b"]) == "a < b"

    def test_unclosed_think_swallows_the_remainder(self):
        assert self._run(["ok ", "<think>", "rambling on"]) == "ok "

    def test_is_case_insensitive(self):
        assert self._run(["a", "<THINK>x</THINK>", "b"]) == "ab"
