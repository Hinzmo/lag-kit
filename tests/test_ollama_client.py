"""OllamaClient loop tests.

The loop is the one place a run can die on something outside our control — the model's
own output. These pin that it degrades instead.
"""
from __future__ import annotations

from types import SimpleNamespace

from lag.ollama_client import OllamaClient


class TestMalformedTurnRecovery:
    """A turn can fail on something the MODEL did, not something we did.

    Small models emit invalid JSON for tool arguments and ollama raises while parsing
    it. Ending the whole run for that throws away every step already taken, and the
    model usually recovers if simply told — the same "hand the error back as text"
    rule the tool layer follows.
    """

    def _client(self, monkeypatch, turns):
        c = OllamaClient(model="m", host="http://x")
        seq = iter(turns)

        def fake_turn(messages, schemas, on_event):
            item = next(seq)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(c, "_turn", fake_turn)
        return c

    def test_one_bad_turn_is_retried_not_fatal(self, monkeypatch):
        good = SimpleNamespace(content="recovered", tool_calls=None)
        c = self._client(monkeypatch, [ValueError("bad json in arguments"), good])
        assert c.run("go") == "recovered"

    def test_the_model_is_told_what_was_wrong(self, monkeypatch):
        """Retrying silently would just reproduce the same malformed reply."""
        c = OllamaClient(model="m", host="http://x")
        seen: list[list] = []
        calls = {"n": 0}

        def fake_turn(messages, schemas, on_event):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("bad json in arguments")
            seen.append([m for m in messages if isinstance(m, dict)])
            return SimpleNamespace(content="ok", tool_calls=None)

        monkeypatch.setattr(c, "_turn", fake_turn)
        assert c.run("go") == "ok"

        nudge = [m for m in seen[0] if "could not be parsed" in str(m.get("content", ""))]
        assert nudge, "the retry must tell the model what went wrong"
        assert "bad json in arguments" in nudge[0]["content"]

    def test_it_gives_up_after_repeated_failures(self, monkeypatch):
        c = self._client(monkeypatch, [ValueError("nope")] * 5)
        out = c.run("go")
        assert "malformed turns" in out

    def test_failures_are_reported_to_a_listener(self, monkeypatch):
        good = SimpleNamespace(content="ok", tool_calls=None)
        c = self._client(monkeypatch, [ValueError("bad"), good])
        events = []
        c.run("go", on_event=events.append)
        assert any(e["type"] == "turn_error" for e in events)


class TestCapabilityDetection:
    """A completion-only tag cannot be fixed by a better prompt.

    The whole loop is tool calls, and passing `tools` to a model without them fails the
    request outright. Capability comes from ollama itself, so a newly pulled model is
    classified correctly with no list to maintain — but a failed probe must never be
    read as "incapable", or one flaky status call silently strips a working model of
    every tool it has.
    """

    def _caps(self, monkeypatch, mapping):
        import lag.ollama_client as oc
        oc._CAPS.clear()

        class FakeClient:
            def __init__(self, host=None): pass
            def show(self, model):
                if model not in mapping:
                    raise RuntimeError("no such model")
                return SimpleNamespace(capabilities=mapping[model])

        monkeypatch.setattr(oc.ollama, "Client", FakeClient)
        return oc

    def test_reports_what_ollama_reports(self, monkeypatch):
        oc = self._caps(monkeypatch, {"m": ["completion", "tools", "thinking"]})
        assert oc.capabilities("h", "m") == frozenset({"completion", "tools", "thinking"})
        assert oc.supports("h", "m", "tools")
        assert oc.supports("h", "m", "thinking")

    def test_a_completion_only_model_is_known_to_lack_tools(self, monkeypatch):
        oc = self._caps(monkeypatch, {"m": ["completion"]})
        assert not oc.supports("h", "m", "tools")
        assert not oc.supports("h", "m", "thinking")

    def test_an_unreachable_probe_is_treated_as_capable(self, monkeypatch):
        """Unknown must mean capable — the alternative downgrades a good model on a
        transient failure, and the request itself would have told us anyway."""
        oc = self._caps(monkeypatch, {})
        assert oc.capabilities("h", "missing") == frozenset()
        assert oc.supports("h", "missing", "tools")

    def test_results_are_cached_per_model(self, monkeypatch):
        import lag.ollama_client as oc
        oc._CAPS.clear()
        calls = {"n": 0}

        class FakeClient:
            def __init__(self, host=None): pass
            def show(self, model):
                calls["n"] += 1
                return SimpleNamespace(capabilities=["completion", "tools"])

        monkeypatch.setattr(oc.ollama, "Client", FakeClient)
        oc.capabilities("h", "m"); oc.capabilities("h", "m"); oc.capabilities("h", "m")
        assert calls["n"] == 1


class TestToolDegradation:
    def _client(self, monkeypatch, caps, **kw):
        import lag.ollama_client as oc
        oc._CAPS.clear()

        class FakeClient:
            def __init__(self, host=None): pass
            def show(self, model): return SimpleNamespace(capabilities=caps)

        monkeypatch.setattr(oc.ollama, "Client", FakeClient)
        return OllamaClient(model="m", host="h", **kw)

    def test_tools_are_dropped_for_a_model_that_cannot_use_them(self, monkeypatch):
        c = self._client(monkeypatch, ["completion"])
        passed = {}

        def fake_turn(messages, schemas, on_event):
            passed["schemas"] = schemas
            return SimpleNamespace(content="answered anyway", tool_calls=None)

        monkeypatch.setattr(c, "_turn", fake_turn)
        provider = SimpleNamespace(schemas=lambda: [lambda: None], call=lambda n, a: "")
        assert c.run("go", tool_provider=provider, verbose=False) == "answered anyway"
        assert passed["schemas"] is None          # the run survives, without tools

    def test_tools_are_kept_for_a_model_that_can(self, monkeypatch):
        c = self._client(monkeypatch, ["completion", "tools"])
        passed = {}

        def fake_turn(messages, schemas, on_event):
            passed["schemas"] = schemas
            return SimpleNamespace(content="ok", tool_calls=None)

        monkeypatch.setattr(c, "_turn", fake_turn)
        provider = SimpleNamespace(schemas=lambda: ["a", "b"], call=lambda n, a: "")
        c.run("go", tool_provider=provider, verbose=False)
        assert passed["schemas"] == ["a", "b"]

    def test_dropping_tools_is_announced_not_silent(self, monkeypatch):
        c = self._client(monkeypatch, ["completion"])
        monkeypatch.setattr(c, "_turn",
                            lambda m, s, o: SimpleNamespace(content="x", tool_calls=None))
        events = []
        provider = SimpleNamespace(schemas=lambda: [lambda: None], call=lambda n, a: "")
        c.run("go", tool_provider=provider, on_event=events.append)
        note = [e for e in events if e["type"] == "no_tools"]
        assert note and "does not support tool calling" in note[0]["message"]

    def test_thinking_is_not_requested_from_a_model_without_it(self, monkeypatch):
        """Asking a non-thinking tag to think is a request it can reject outright."""
        c = self._client(monkeypatch, ["completion", "tools"], think=True)
        assert c.think is None


class TestOutOfTurns:
    """An agent that spends every turn on tools gets one tool-free turn to answer,
    instead of the run failing with the work it gathered thrown away."""

    class Tools:
        def schemas(self):
            return [{"type": "function", "function": {"name": "t"}}]

        def call(self, name, args):
            return "data"

    def _call(self):
        fn = SimpleNamespace(name="t", arguments={})
        return SimpleNamespace(content="", tool_calls=[SimpleNamespace(function=fn)])

    def _client(self, monkeypatch, last):
        c = OllamaClient(model="m", host="http://x")
        c.has_tools = True
        seen = []

        def fake_turn(messages, schemas, on_event):
            seen.append(schemas)
            if schemas is None:
                if isinstance(last, Exception):
                    raise last
                return SimpleNamespace(content=last, tool_calls=None)
            return self._call()
        monkeypatch.setattr(c, "_turn", fake_turn)
        return c, seen

    def test_the_last_turn_has_no_tools_and_its_answer_is_returned(self, monkeypatch):
        c, seen = self._client(monkeypatch, "here is what I found")
        events = []
        out = c.run("go", tool_provider=self.Tools(), max_iters=3, on_event=events.append)
        assert out == "here is what I found"
        assert seen[:3] != [None] * 3 and seen[-1] is None and len(seen) == 4
        assert any(e["type"] == "final_answer_nudge" for e in events)

    def test_an_empty_or_failed_last_turn_keeps_the_old_message(self, monkeypatch):
        for last in ("", RuntimeError("down")):
            c, _ = self._client(monkeypatch, last)
            out = c.run("go", tool_provider=self.Tools(), max_iters=2, verbose=False)
            assert out == "(stopped: hit max_iters without a final answer)"


class TestStreamedThinkFallback:
    """A tag that rejects `think` falls back to its default. That fallback used to
    cover only blocking calls: a streamed request is made lazily, on the first
    chunk, after _chat's try had already returned. The console, agents and goals all
    stream, so the turn failed three times and the run stopped."""

    def _client(self, monkeypatch):
        import lag.ollama_client as oc
        oc._CAPS.clear()
        calls = []

        class FakeClient:
            def __init__(self, host=None): pass
            def show(self, model):
                return SimpleNamespace(capabilities=["completion", "tools", "thinking"])

            def chat(self, **kw):
                calls.append(kw)

                def chunks():            # lazy, like ollama's own stream
                    if "think" in kw:
                        raise RuntimeError('"m" does not support thinking')
                    yield SimpleNamespace(message=SimpleNamespace(
                        content="hi", thinking=None, tool_calls=None))
                return chunks()

        monkeypatch.setattr(oc.ollama, "Client", FakeClient)
        return OllamaClient(model="m", host="h", think=True), calls

    def test_a_streamed_turn_retries_without_think(self, monkeypatch):
        c, calls = self._client(monkeypatch)
        events = []
        assert c.run("hello", on_event=events.append) == "hi"
        assert "think" in calls[0] and "think" not in calls[1]
        assert not [e for e in events if e["type"] == "turn_error"]

    def test_the_refusal_is_learned_once(self, monkeypatch):
        c, calls = self._client(monkeypatch)
        c.run("a", on_event=lambda e: None)
        c.run("b", on_event=lambda e: None)
        assert sum("think" in k for k in calls) == 1


class TestARaisingProviderDoesNotEndTheRun:
    """Providers return failures as text. One that raises anyway is reported to the
    model like any other failure, and the run carries on."""

    def test_the_error_becomes_the_tool_result(self, monkeypatch):
        c = OllamaClient(model="m", host="http://x")
        fn = SimpleNamespace(name="t", arguments={})
        turns = iter([SimpleNamespace(content="", tool_calls=[SimpleNamespace(function=fn)]),
                      SimpleNamespace(content="recovered", tool_calls=None)])
        monkeypatch.setattr(c, "_turn", lambda m, s, o: next(turns))
        monkeypatch.setattr(c, "has_tools", True)

        def boom(name, args):
            raise TimeoutError("slow")

        provider = SimpleNamespace(schemas=lambda: [{"type": "function"}], call=boom)
        events = []
        assert c.run("go", tool_provider=provider, on_event=events.append) == "recovered"
        result = [e for e in events if e["type"] == "tool_result"][0]
        assert "TimeoutError" in result["content"]
