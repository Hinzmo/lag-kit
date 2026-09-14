"""Pipeline sequencing tests.

Two properties carry the design and both are cheap to get wrong silently:

1. Cost. Ollama on the dev box holds exactly one chat model resident, so every change
   of tag down the line is a full cold reload (~5s measured). `loads()` is what makes
   that visible, and it must count tag *changes*, not stages.
2. Context. Each stage is supposed to see everything above it — that is the whole
   reason the lead can plan once and never be called back. A stage that receives only
   the previous output instead of the accumulation would still produce plausible text,
   so nothing would look broken.

No model is ever contacted here; stages are fakes recording what they were handed.
"""
from __future__ import annotations

import pytest

from lag.pipeline import Pipeline


class FakeStage:
    def __init__(self, name, model):
        self.name, self.model = name, model
        self.seen = []

    def run(self, task, on_event=None):
        self.seen.append(task)
        if on_event:                       # stand in for the client's token stream
            on_event({"type": "token", "text": f"<{self.name} output>"})
        return f"<{self.name} output>"


def _pipe(*pairs):
    return Pipeline([FakeStage(n, m) for n, m in pairs], "http://x")


class TestLoadAccounting:
    def test_counts_tag_changes_not_stages(self):
        # four stages, but only two distinct models back to back → two loads
        assert _pipe(("a", "big"), ("b", "big"), ("c", "small"), ("d", "small")).loads() == 2

    def test_alternating_tags_pay_every_time(self):
        # the shape the default roster used to have: big → small → big
        assert _pipe(("a", "big"), ("b", "small"), ("c", "big")).loads() == 3

    def test_single_model_pipeline_loads_once(self):
        assert _pipe(("a", "one"), ("b", "one"), ("c", "one")).loads() == 1

    def test_empty_pipeline_costs_nothing(self):
        assert Pipeline([], "http://x").loads() == 0


class TestContextFlow:
    def test_each_stage_sees_every_output_above_it(self):
        p = _pipe(("first", "m"), ("second", "m"), ("third", "m"))
        p.run("build a thing", plan=False)
        first, second, third = p.stages

        assert "build a thing" in first.seen[0]
        # the second stage inherits the first's output, the third inherits both
        assert "<first output>" in second.seen[0]
        assert "<first output>" in third.seen[0]
        assert "<second output>" in third.seen[0]

    def test_returns_the_last_stage_output(self):
        p = _pipe(("a", "m"), ("b", "m"))
        assert p.run("t", plan=False) == "<b output>"

    def test_empty_pipeline_is_an_error_not_an_empty_string(self):
        with pytest.raises(ValueError):
            Pipeline([], "http://x").run("t")


class TestPlanning:
    def test_lead_plans_once_and_is_not_called_back(self):
        """crew's coordinator bookended the run; that return trip cost a second load."""
        p = _pipe(("lead", "big"), ("worker", "small"))
        p.run("ship it", plan=True)
        lead, worker = p.stages

        assert len(lead.seen) == 1                 # planned, never called again
        assert "worker" in lead.seen[0]            # briefed by roster
        assert "<lead output>" in worker.seen[0]   # the plan rode down the line

    def test_no_plan_makes_the_lead_do_ordinary_work(self):
        p = _pipe(("lead", "big"), ("worker", "small"))
        p.run("ship it", plan=False)
        assert "ship it" in p.stages[0].seen[0]
        assert "instruction for each" not in p.stages[0].seen[0]

    def test_lone_stage_is_not_asked_to_brief_nobody(self):
        p = _pipe(("only", "m"))
        assert p.run("t", plan=True) == "<only output>"
        assert "t" in p.stages[0].seen[0]


class TestPerModelContext:
    """One resident model at a time means each can be sized on its own terms."""

    CFG = {
        "ollama_base_url": "http://x",
        "ollama": {"num_ctx": 8192, "num_ctx_by_model": {"wide:9b": 131072}},
    }

    def test_named_model_gets_its_own_window(self):
        from lag.ollama_client import OllamaClient
        assert OllamaClient.from_config(self.CFG, "wide:9b").num_ctx == 131072

    def test_unlisted_model_falls_back_to_the_global(self):
        from lag.ollama_client import OllamaClient
        assert OllamaClient.from_config(self.CFG, "other:8b").num_ctx == 8192

    def test_config_without_the_map_still_works(self):
        from lag.ollama_client import OllamaClient
        cfg = {"ollama_base_url": "http://x", "ollama": {"num_ctx": 4096}}
        assert OllamaClient.from_config(cfg, "any:1b").num_ctx == 4096



class TestEventStream:
    """The console renders from these events, so their shape is a contract.

    They are a side channel: adding a listener must not change what run() returns,
    or the streamed UI and the blocking API would disagree about the same run.
    """

    def _collect(self, pipe, **kw):
        events = []
        answer = pipe.run("t", on_event=events.append, **kw)
        return events, answer

    def test_every_stage_transition_is_announced_with_its_model(self):
        pipe = _pipe(("lead", "big"), ("mid", "small"), ("last", "small"))
        events, _ = self._collect(pipe, plan=False)
        stages = [(e["name"], e["model"], e["loaded"]) for e in events
                  if e["type"] == "stage"]
        assert stages == [("lead", "big", True),
                          ("mid", "small", True),
                          ("last", "small", False)]   # same tag -> no reload

    def test_loaded_flag_matches_the_load_accounting(self):
        pipe = _pipe(("a", "one"), ("b", "one"), ("c", "two"))
        events, _ = self._collect(pipe, plan=False)
        assert sum(e["loaded"] for e in events if e["type"] == "stage") == pipe.loads()

    def test_run_ends_with_the_final_answer(self):
        pipe = _pipe(("a", "m"), ("b", "m"))
        events, answer = self._collect(pipe, plan=False)
        assert events[-1] == {"type": "done", "answer": "<b output>"}
        assert answer == "<b output>"

    def test_listening_does_not_change_the_result(self):
        quiet = _pipe(("a", "m"), ("b", "m")).run("t", plan=False)
        loud = _pipe(("a", "m"), ("b", "m")).run("t", plan=False, on_event=lambda e: None)
        assert quiet == loud

    def test_the_lead_stage_is_tagged_so_the_ui_can_label_it(self):
        pipe = _pipe(("lead", "big"), ("worker", "small"))
        events, _ = self._collect(pipe, plan=True)
        roles = [(e["name"], e["role"]) for e in events if e["type"] == "stage"]
        assert roles == [("lead", "lead"), ("worker", "stage")]


class TestRouteParsing:
    """The router's reply is free text from a small local model.

    It fences JSON, prefixes prose, and invents persona names. Every one of those
    would otherwise reach load_persona and crash the run, so parsing must be tolerant
    going in and strict coming out.
    """

    VALID = {"dev_agent", "tester_agent", "research_agent"}

    def parse(self, raw):
        return Pipeline._parse_route(raw, self.VALID)

    def test_plain_object(self):
        raw = '{"agents":[{"name":"dev_agent","instruction":"write it"}]}'
        assert self.parse(raw) == [("dev_agent", "write it")]

    def test_fenced_json_with_prose_around_it(self):
        raw = ('Sure! Here is the team:\n```json\n'
               '{"agents":[{"name":"tester_agent","instruction":"test it"}]}\n```\n'
               'Let me know if you want changes.')
        assert self.parse(raw) == [("tester_agent", "test it")]

    def test_hallucinated_names_are_dropped(self):
        raw = ('{"agents":[{"name":"wizard_agent","instruction":"x"},'
               '{"name":"dev_agent","instruction":"y"}]}')
        assert self.parse(raw) == [("dev_agent", "y")]

    def test_duplicates_collapse(self):
        raw = ('{"agents":[{"name":"dev_agent","instruction":"first"},'
               '{"name":"dev_agent","instruction":"again"}]}')
        assert self.parse(raw) == [("dev_agent", "first")]

    def test_order_is_preserved(self):
        raw = ('{"agents":[{"name":"research_agent","instruction":"a"},'
               '{"name":"dev_agent","instruction":"b"},'
               '{"name":"tester_agent","instruction":"c"}]}')
        assert [n for n, _ in self.parse(raw)] == [
            "research_agent", "dev_agent", "tester_agent"]

    def test_bare_names_without_instructions(self):
        assert self.parse('{"agents":["dev_agent"]}') == [("dev_agent", "")]

    def test_unparseable_returns_empty_so_the_caller_can_fall_back(self):
        for raw in ("", "I think dev_agent should do it.", "{not json", "{}"):
            assert self.parse(raw) == []


class TestContextBudget:
    """Stages do not share a window, and the context grows with every stage.

    By the third stage the task, the plan and two full outputs are all in play, which
    is how a run reaches a stage holding 8192 with more than 8192 of prompt. That is a
    hard failure, not a soft one, so the pipeline trims rather than letting it happen.
    """

    class Stage:
        def __init__(self, window):
            self.name, self.model = "s", "m"
            self.client = type("C", (), {"num_ctx": window})()
            self.seen = []

        def run(self, task, on_event=None):
            self.seen.append(task)
            return "out"

    def _fit(self, context, window, events=None):
        pipe = Pipeline([], "http://x")
        return pipe._fit(context, self.Stage(window),
                         events.append if events is not None else None)

    def test_context_within_the_window_is_untouched(self):
        ctx = "x" * 100
        assert self._fit(ctx, 8192) == ctx

    def test_oversized_context_is_trimmed_under_budget(self):
        ctx = "x" * 200_000
        out = self._fit(ctx, 8192)
        assert len(out) < len(ctx)
        assert len(out) <= int(8192 * 4 * 0.55) + 100     # + marker

    def test_trimming_keeps_the_head_and_the_tail(self):
        ctx = "HEAD" + ("x" * 200_000) + "TAIL"
        out = self._fit(ctx, 8192)
        assert out.startswith("HEAD")      # the task and plan survive
        assert out.endswith("TAIL")        # so does what it builds on
        assert "trimmed to fit" in out     # and the gap is marked, not silent

    def test_trimming_is_announced(self):
        events = []
        self._fit("x" * 200_000, 8192, events)
        assert events and events[0]["type"] == "trimmed"
        assert events[0]["window"] == 8192
        assert events[0]["dropped_chars"] > 0

    def test_a_stage_without_a_declared_window_is_left_alone(self):
        ctx = "x" * 200_000
        assert self._fit(ctx, None) == ctx


class TestModelOverride:
    """Picking one model for a run is also how you spend zero reloads.

    Only one chat model stays resident, so a three-stage run across two tags costs
    three cold loads. Forcing every stage onto one tag makes that one.
    """

    CFG = {
        "ollama_base_url": "http://x",
        "models": {"orchestrator": "big:9b", "worker": "small:8b", "coder": "small:8b"},
        "paths": {"personas": "./personas"},
        "ollama": {"num_ctx": 8192},
    }

    def test_override_replaces_every_persona_tag(self):
        pipe = Pipeline.from_personas(
            ["architect_agent", "dev_agent", "senior_dev_agent"],
            self.CFG, tool_provider=object(), model_override="small:8b")
        assert {a.model for a in pipe.stages} == {"small:8b"}

    def test_override_collapses_the_run_to_one_load(self):
        mixed = Pipeline.from_personas(
            ["architect_agent", "dev_agent", "senior_dev_agent"],
            self.CFG, tool_provider=object())
        forced = Pipeline.from_personas(
            ["architect_agent", "dev_agent", "senior_dev_agent"],
            self.CFG, tool_provider=object(), model_override="small:8b")
        assert mixed.loads() == 3      # big -> small -> big
        assert forced.loads() == 1

    def test_no_override_keeps_each_persona_on_its_own_model(self):
        pipe = Pipeline.from_personas(
            ["architect_agent", "dev_agent"], self.CFG, tool_provider=object())
        assert pipe.stages[0].model == "big:9b"     # orchestrator role
        assert pipe.stages[1].model == "small:8b"   # coder role

    def test_the_override_reaches_the_client_not_just_the_label(self):
        pipe = Pipeline.from_personas(["architect_agent"], self.CFG,
                                      tool_provider=object(), model_override="small:8b")
        assert pipe.stages[0].client.model == "small:8b"
