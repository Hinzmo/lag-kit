"""Agent mode tests.

Agent mode is the unbounded shape — a loop with full autonomy that is meant to be
left running. Three properties decide whether that is safe to do at all, and none of
them are visible from a single successful run:

  1. It can stop. mark_done, a cycle cap, and a stop file must each end the loop.
  2. It survives its own failures. One bad cycle must not end a 24/7 run.
  3. It remembers. State must persist and resume, or a restart silently starts over.

No model is contacted; the client is a fake that scripts what each cycle does.
"""
from __future__ import annotations

import json

import pytest

from lag import agent as agent_mod
from lag import tools as tools_mod
from lag.agent import AgentLoop, AgentState, GoalSpec, stop_path


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Never touch the real ./agent directory, leak goal state, or actually wait.

    The failure path backs off 5s per consecutive error, which is right in production
    and pure dead time here — a three-failure test would sleep 30 real seconds to prove
    arithmetic. The backoff itself is asserted by watching what gets passed to sleep.
    """
    monkeypatch.setattr(agent_mod, "AGENT_DIR", tmp_path / "agent")
    tools_mod.GOAL.update({"goal": "", "started": "", "cycle": 0, "notes": [],
                           "done": False, "summary": ""})
    slept: list[float] = []
    monkeypatch.setattr(agent_mod.time, "sleep", slept.append)
    # Confinement is mandatory in agent mode, which would otherwise make every test
    # here start a real container — slow, and it would make the whole suite depend on
    # a running Docker daemon. The sandbox has its own tests, with a real container.
    monkeypatch.setattr(agent_mod, "sandbox_available", lambda: (True, ""))
    monkeypatch.setattr(agent_mod.Sandbox, "start", lambda self: None)
    monkeypatch.setattr(agent_mod.Sandbox, "stop", lambda self: None)
    yield slept


def _loop(monkeypatch, behaviour, **kw):
    """Build an AgentLoop whose every cycle runs `behaviour(cycle_number)`."""
    spec = GoalSpec(goal="do the thing", persona="dev_agent", **kw)
    monkeypatch.setattr(agent_mod, "load_persona",
                        lambda n, c: {"name": n, "model": "fake:1b",
                                      "system_prompt": "", "thinking_mode": False})

    calls = {"n": 0}

    class FakeClient:
        def run(self, prompt, tool_provider=None, on_event=None):
            calls["n"] += 1
            return behaviour(calls["n"])

    monkeypatch.setattr(agent_mod.OllamaClient, "from_config",
                        classmethod(lambda cls, *a, **k: FakeClient()))
    loop = AgentLoop(spec, {"ollama_base_url": "x"}, tool_provider=object())
    return loop, calls


class TestTermination:
    def test_mark_done_ends_the_loop(self, monkeypatch):
        def behave(n):
            if n == 2:
                tools_mod.mark_done("finished it")
            return f"cycle {n}"

        loop, calls = _loop(monkeypatch, behave)
        state = loop.run()
        assert state.done and state.cycle == 2
        assert state.summary == "finished it"
        assert calls["n"] == 2          # stopped immediately, did not run a third

    def test_cycle_cap_ends_a_loop_that_never_finishes(self, monkeypatch):
        loop, calls = _loop(monkeypatch, lambda n: "still working", max_cycles=3)
        state = loop.run()
        assert not state.done and state.cycle == 3
        assert calls["n"] == 3

    def test_stop_file_ends_the_loop(self, monkeypatch):
        def behave(n):
            stop_path("default").write_text("stop")   # asked to stop mid-cycle
            return "working"

        loop, _ = _loop(monkeypatch, behave, max_cycles=50)
        state = loop.run()
        assert state.cycle == 1 and not state.done

    def test_a_stale_stop_file_does_not_block_the_next_run(self, monkeypatch):
        loop, _ = _loop(monkeypatch, lambda n: "x", max_cycles=1)
        stop_path("default").parent.mkdir(parents=True, exist_ok=True)
        stop_path("default").write_text("stop")
        assert loop.run().cycle == 1      # cleared at start, so it still ran


class TestResilience:
    def test_one_failing_cycle_does_not_end_the_run(self, monkeypatch):
        def behave(n):
            if n == 1:
                raise RuntimeError("ollama fell over")
            tools_mod.mark_done("recovered")
            return "ok"

        loop, _ = _loop(monkeypatch, behave)
        state = loop.run()
        assert state.errors == 1
        assert state.done                      # kept going and finished
        assert state.cycle == 2

    def test_the_failure_is_handed_to_the_next_cycle(self, monkeypatch):
        seen = {}

        def behave(n):
            if n == 1:
                raise RuntimeError("boom")
            seen["last"] = loop.state.last_output
            tools_mod.mark_done("done")
            return "ok"

        loop, _ = _loop(monkeypatch, behave)
        loop.run()
        assert "boom" in seen["last"]          # the agent can see what went wrong


class TestPersistence:
    def test_state_is_written_after_every_cycle(self, monkeypatch):
        loop, _ = _loop(monkeypatch, lambda n: "x", max_cycles=2)
        state = loop.run()
        saved = json.loads(state.path.read_text())
        assert saved["cycle"] == 2
        assert saved["spec"]["goal"] == "do the thing"

    def test_an_unfinished_run_resumes_instead_of_restarting(self, monkeypatch):
        loop, _ = _loop(monkeypatch, lambda n: "x", max_cycles=2)
        loop.run()
        resumed, _ = _loop(monkeypatch, lambda n: "x", max_cycles=3)
        assert resumed.state.cycle == 2        # picked up where it left off

    def test_a_different_goal_starts_fresh(self, monkeypatch):
        loop, _ = _loop(monkeypatch, lambda n: "x", max_cycles=2)
        loop.run()
        monkeypatch.setattr(agent_mod, "load_persona",
                            lambda n, c: {"name": n, "model": "fake:1b",
                                          "system_prompt": "", "thinking_mode": False})
        monkeypatch.setattr(agent_mod.OllamaClient, "from_config",
                            classmethod(lambda cls, *a, **k: type("C", (), {
                                "run": lambda s, *a, **k: "x"})()))
        other = AgentLoop(GoalSpec(goal="a different thing", persona="dev_agent"),
                          {"ollama_base_url": "x"}, tool_provider=object())
        assert other.state.cycle == 0

    def test_a_resumed_run_gets_its_own_cycle_budget(self, monkeypatch):
        """The cap bounds ONE run. Compared against the lifetime counter, an agent
        that once reached its cap stopped before cycle 1 on every later start."""
        _loop(monkeypatch, lambda n: "x", max_cycles=2, max_idle=0)[0].run()
        resumed, calls = _loop(monkeypatch, lambda n: "x", max_cycles=2, max_idle=0)
        state = resumed.run()
        assert calls["n"] == 2                 # it actually worked this time
        assert state.cycle == 4                # and the lifetime count kept going

    def test_hitting_the_cap_says_so(self, monkeypatch):
        loop, _ = _loop(monkeypatch, lambda n: "x", max_cycles=2, max_idle=0)
        assert "cycle cap" in loop.run().stopped_reason

    def test_a_stop_request_says_so(self, monkeypatch):
        def behave(n):
            stop_path("default").write_text("stop")
            return "x"

        loop, _ = _loop(monkeypatch, behave, max_cycles=50)
        assert "stop requested" in loop.run().stopped_reason

    def test_a_resume_starts_with_fresh_breakers_and_no_stale_reason(self, monkeypatch):
        """Restarting after a breaker trip is a request to try again. Carrying the
        counters over tripped it after one cycle; carrying the reason over left a
        finished agent reporting 'done' and 'stopped: no progress' at once."""
        first = _loop(monkeypatch, lambda n: "nothing", max_cycles=50, max_idle=2)[0]
        assert "no tool call" in first.run().stopped_reason

        again, calls = _loop(monkeypatch, lambda n: "nothing", max_cycles=50, max_idle=2)
        again.run()
        assert calls["n"] == 2                 # a full breaker's worth, not one cycle

        def finish(n):
            tools_mod.mark_done("got there")
            return "ok"

        last, _ = _loop(monkeypatch, finish, max_cycles=50, max_idle=2)
        state = last.run()
        assert state.done and state.stopped_reason == ""

    def test_corrupt_state_does_not_block_a_run(self, monkeypatch):
        agent_mod.AGENT_DIR.mkdir(parents=True, exist_ok=True)
        (agent_mod.AGENT_DIR / "default.json").write_text("{not json")
        loop, _ = _loop(monkeypatch, lambda n: "x", max_cycles=1)
        assert loop.run().cycle == 1


class TestGoalPrompt:
    def test_the_goal_is_restated_every_cycle(self):
        """Cycles share no conversation memory — the brief IS the continuity."""
        spec = GoalSpec(goal="ship it", success="tests pass", constraints="no force push")
        p = spec.prompt(4, ["did a", "did b"], "last thing that happened")
        assert "ship it" in p and "tests pass" in p and "no force push" in p
        assert "cycle 4" in p
        assert "did a" in p and "did b" in p
        assert "last thing that happened" in p

    def test_progress_is_capped_so_the_brief_cannot_grow_forever(self):
        spec = GoalSpec(goal="g")
        p = spec.prompt(200, [f"note {i}" for i in range(100)])
        assert "note 99" in p and "note 0" not in p

    def test_shell_network_is_on_by_default_and_the_prompt_says_so(self):
        """Networked by default (sealing it made agent mode unworkable), and the agent
        is told the truth about its shell: public internet through a proxy, never the
        host or LAN — so it neither assumes it is offline nor tries the host."""
        assert GoalSpec(goal="g").network is True
        on = GoalSpec(goal="g").prompt(1, [])
        assert "PUBLIC internet only" in on and "proxy" in on
        assert "host machine, the LAN and private addresses are refused" in on
        assert "no network in the shell" in GoalSpec(goal="g", network=False).prompt(1, [])


class TestGoalTools:
    def test_mark_done_and_notes_flow_into_state(self, monkeypatch):
        def behave(n):
            tools_mod.note_progress("stepped forward")
            tools_mod.mark_done("all good")
            return "ok"

        loop, _ = _loop(monkeypatch, behave)
        state = loop.run()
        assert state.notes == ["stepped forward"]
        assert state.summary == "all good"

    def test_goal_status_outside_agent_mode_is_clear_not_broken(self):
        assert "no goal set" in tools_mod.goal_status()


class TestCircuitBreakers:
    """A 24/7 agent needs to fail *closed*.

    Two different failures, two different detectors. An erroring agent is loud and the
    error counter catches it. The dangerous one is silent: cycles that complete fine,
    touch nothing, and loop forever restating the plan. Neither is fixed by asking
    again, so both end the run rather than pausing it.
    """

    def test_consecutive_failures_stop_the_agent(self, monkeypatch):
        loop, calls = _loop(monkeypatch,
                            lambda n: (_ for _ in ()).throw(RuntimeError("down")),
                            max_cycles=50, max_failures=3)
        state = loop.run()
        assert state.cycle == 3
        assert state.consecutive_errors == 3
        assert "consecutive failed cycles" in state.stopped_reason

    def test_a_recovered_cycle_resets_the_failure_count(self, monkeypatch):
        """One failure at cycle 3 and another at cycle 400 is a healthy agent."""
        def behave(n):
            if n in (1, 2, 4, 5):
                raise RuntimeError("flaky")
            if n == 3:
                tools_mod.note_progress("recovered")   # real progress
                return "ok"
            tools_mod.note_progress(f"note {n}")
            return "ok"

        loop, _ = _loop(monkeypatch, behave, max_cycles=8, max_failures=3)
        state = loop.run()
        assert state.consecutive_errors < 3      # never hit three in a row
        assert state.cycle == 8                  # ran to the cap instead of tripping

    def test_cycles_that_do_nothing_stop_the_agent(self, monkeypatch):
        """No tool call and no new note is an agent talking about work, not doing it."""
        loop, _ = _loop(monkeypatch, lambda n: "I will begin shortly.",
                        max_cycles=50, max_idle=4)
        state = loop.run()
        assert state.cycle == 4
        assert "no tool call" in state.stopped_reason

    def test_noting_progress_counts_as_progress(self, monkeypatch):
        def behave(n):
            tools_mod.note_progress(f"did thing {n}")
            return "ok"

        loop, _ = _loop(monkeypatch, behave, max_cycles=6, max_idle=3)
        state = loop.run()
        assert state.idle_cycles == 0
        assert state.cycle == 6          # productive, so never tripped

    def test_calling_a_tool_counts_as_progress(self, monkeypatch):
        """A cycle can be productive without narrating it."""
        captured = {}

        def behave(n):
            captured["on_event"]({"type": "tool", "name": "read_file", "args": {}})
            return "worked silently"

        _loop(monkeypatch, behave, max_cycles=5, max_idle=2)   # installs the stubs

        class C:
            def run(self, prompt, tool_provider=None, on_event=None):
                captured["on_event"] = on_event
                return behave(0)

        monkeypatch.setattr(agent_mod.OllamaClient, "from_config",
                            classmethod(lambda cls, *a, **k: C()))
        loop2 = AgentLoop(GoalSpec(goal="g", persona="dev_agent", max_cycles=5,
                                   max_idle=2), {"ollama_base_url": "x"},
                          tool_provider=object())
        state = loop2.run()
        assert state.idle_cycles == 0
        assert state.cycle == 5

    def test_breakers_can_be_turned_off(self, monkeypatch):
        loop, _ = _loop(monkeypatch, lambda n: "nothing",
                        max_cycles=4, max_idle=0, max_failures=0)
        assert loop.run().cycle == 4      # ran to the cap, no breaker tripped

    def test_a_finished_goal_is_not_reported_as_a_breaker_stop(self, monkeypatch):
        def behave(n):
            tools_mod.mark_done("all good")
            return "ok"

        loop, _ = _loop(monkeypatch, behave, max_idle=1)
        state = loop.run()
        assert state.done and state.stopped_reason == ""


def test_the_backoff_grows_with_consecutive_failures(monkeypatch, _isolate):
    """Stubbed sleep, so this asserts the schedule rather than living through it."""
    loop, _ = _loop(monkeypatch,
                    lambda n: (_ for _ in ()).throw(RuntimeError("down")),
                    max_cycles=50, max_failures=4)
    loop.run()
    assert _isolate == [5, 10, 15, 20]        # 5s per consecutive failure


def test_the_backoff_is_capped(monkeypatch, _isolate):
    loop, _ = _loop(monkeypatch,
                    lambda n: (_ for _ in ()).throw(RuntimeError("down")),
                    max_cycles=50, max_failures=20)
    loop.run()
    assert max(_isolate) == 60                 # never longer than a minute


def test_a_crashing_agent_reports_crashing_not_idleness(monkeypatch, _isolate):
    """Both breakers would stop it; only one explains why, and that is what you read."""
    loop, _ = _loop(monkeypatch,
                    lambda n: (_ for _ in ()).throw(RuntimeError("ollama down")),
                    max_cycles=50, max_failures=5, max_idle=3)
    state = loop.run()
    assert "consecutive failed cycles" in state.stopped_reason
    assert "no tool call" not in state.stopped_reason
    assert state.cycle == 5                 # the failure breaker, not the idle one


class TestSavedDefinitions:
    """A saved agent is a thing you own: editable, toggleable, and separate from its
    history. Definition and run state are deliberately different files — editing a
    goal must not erase what the agent has already done."""

    def test_round_trips_through_yaml(self):
        spec = GoalSpec(name="auditor", goal="audit things", persona="dev_agent",
                        interval=30, tools=["read_file"], description="reads stuff")
        spec.save()
        back = GoalSpec.load("auditor")
        assert back.goal == "audit things"
        assert back.tools == ["read_file"]
        assert back.interval == 30
        assert back.description == "reads stuff"

    def test_definitions_are_human_editable_yaml(self):
        GoalSpec(name="a", goal="g").save()
        text = GoalSpec.path_for("a").read_text()
        assert "goal: g" in text          # not a pickle or an opaque blob

    def test_listing_skips_a_broken_file_instead_of_hiding_the_rest(self):
        GoalSpec(name="good", goal="g").save()
        (agent_mod.AGENT_DIR / "broken.yaml").write_text("{[not yaml")
        assert [s.name for s in GoalSpec.list_all()] == ["good"]

    def test_unknown_keys_are_ignored_so_hand_edits_do_not_brick_an_agent(self):
        GoalSpec(name="a", goal="g").save()
        p = GoalSpec.path_for("a")
        p.write_text(p.read_text() + "\nfavourite_colour: blue\n")
        assert GoalSpec.load("a").goal == "g"

    def test_disabling_keeps_the_definition(self):
        GoalSpec(name="a", goal="g").save()
        sp = GoalSpec.load("a"); sp.enabled = False; sp.save()
        assert GoalSpec.load("a").enabled is False
        assert GoalSpec.path_for("a").exists()

    def test_editing_a_goal_does_not_touch_run_history(self):
        spec = GoalSpec(name="a", goal="original"); spec.save()
        st = agent_mod.AgentState(spec={"name": "a"}, cycle=7, notes=["did a thing"])
        st.save()
        spec.goal = "revised"; spec.save()
        assert agent_mod.AgentState.load("a").cycle == 7

    def test_delete_removes_definition_state_and_stop_flag(self):
        spec = GoalSpec(name="a", goal="g"); spec.save()
        agent_mod.AgentState(spec={"name": "a"}, cycle=1).save()
        stop_path("a").write_text("stop")
        spec.delete()
        assert not GoalSpec.path_for("a").exists()
        assert not (agent_mod.AGENT_DIR / "a.state.json").exists()
        assert not stop_path("a").exists()

    def test_a_name_cannot_escape_the_directory(self):
        """Names become file paths."""
        for bad in ("../../etc/passwd", "a/b", "", ".hidden", "with space"):
            with pytest.raises(ValueError):
                GoalSpec.path_for(bad)

    def test_missing_agent_is_a_clear_error(self):
        with pytest.raises(FileNotFoundError):
            GoalSpec.load("nope")

    def test_unknown_tool_names_are_reported(self):
        spec = GoalSpec(name="a", goal="g", tools=["read_file", "nosuchtool"])
        assert spec.unknown_tools() == ["nosuchtool"]

    def test_search_knowledge_base_is_a_known_tool(self):
        """It comes from the RAG server, not TOOLS, but naming it is not a typo."""
        assert GoalSpec(name="a", goal="g",
                        tools=["search_knowledge_base"]).unknown_tools() == []


class TestToolSubset:
    """Narrowing must work identically over both transports, or an allow-list that
    looks enforced in tests silently passes everything in production."""

    class FakeInner:
        def __init__(self, schemas): self._s = schemas; self.called = []
        def schemas(self): return self._s
        def call(self, name, args): self.called.append(name); return "ran " + name

    def test_filters_callable_schemas_like_the_local_provider(self):
        def read_file(): pass
        def run_shell(): pass
        inner = self.FakeInner([read_file, run_shell])
        sub = agent_mod.ToolSubset(inner, ["read_file"])
        assert [f.__name__ for f in sub.schemas()] == ["read_file"]

    def test_filters_dict_schemas_like_the_mcp_provider(self):
        inner = self.FakeInner([
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "run_shell"}},
        ])
        sub = agent_mod.ToolSubset(inner, ["read_file"])
        assert [agent_mod._schema_name(s) for s in sub.schemas()] == ["read_file"]

    def test_a_blocked_call_returns_a_string_and_never_reaches_the_inner_provider(self):
        inner = self.FakeInner([])
        sub = agent_mod.ToolSubset(inner, ["read_file"])
        out = sub.call("run_shell", {})
        assert "blocked" in out and "read_file" in out
        assert inner.called == []

    def test_an_empty_allow_list_means_everything(self):
        def a(): pass
        inner = self.FakeInner([a])
        sub = agent_mod.ToolSubset(inner, [])
        assert sub.schemas() == [a]
        assert sub.call("anything", {}) == "ran anything"


class TestGoalToolsStayLocal:
    """The bug this exists for: an agent called mark_done on cycle 1 and ran to its
    cycle cap anyway, reporting zero notes — over MCP the tools executed in a server
    subprocess, so the record they wrote died with it."""

    class RemoteInner:
        def __init__(self): self.called = []
        def schemas(self): return []
        def call(self, name, args):
            self.called.append(name)
            return "executed in a subprocess, where nothing here can see it"

    def test_mark_done_is_never_routed_to_the_inner_provider(self):
        inner = self.RemoteInner()
        p = agent_mod.GoalToolsLocal(inner)
        tools_mod.GOAL.update({"done": False, "summary": ""})
        p.call("mark_done", {"summary": "finished"})
        assert inner.called == []                       # stayed in-process
        assert tools_mod.GOAL["done"] is True           # and actually took effect

    def test_note_progress_reaches_this_process(self):
        inner = self.RemoteInner()
        p = agent_mod.GoalToolsLocal(inner)
        tools_mod.GOAL["notes"] = []
        p.call("note_progress", {"note": "did a thing"})
        assert tools_mod.GOAL["notes"] == ["did a thing"]
        assert inner.called == []

    def test_ordinary_tools_still_go_to_the_inner_provider(self):
        inner = self.RemoteInner()
        p = agent_mod.GoalToolsLocal(inner)
        p.call("read_file", {"path": "x"})
        assert inner.called == ["read_file"]

    def test_a_bad_goal_tool_call_reports_rather_than_raising(self):
        p = agent_mod.GoalToolsLocal(self.RemoteInner())
        assert "error" in p.call("mark_done", {"wrong_arg": 1})


class TestSandboxIsMandatory:
    """Agent mode must refuse to run unconfined, and refuse BEFORE doing any work.

    Failing at the first shell call would leave the agent several tool calls deep with
    a half-finished job and a confusing error; failing at the top is a clean no.
    """

    def test_no_runtime_means_no_run(self, monkeypatch):
        loop, calls = _loop(monkeypatch, lambda n: "should never happen")
        monkeypatch.setattr(agent_mod, "sandbox_available",
                            lambda: (False, "daemon is not running"))
        state = loop.run()
        assert calls["n"] == 0                       # not one cycle executed
        assert "no sandbox" in state.stopped_reason
        assert "daemon is not running" in state.stopped_reason

    def test_a_sandbox_that_fails_to_start_also_stops_the_run(self, monkeypatch):
        loop, calls = _loop(monkeypatch, lambda n: "should never happen")
        monkeypatch.setattr(agent_mod.Sandbox, "start",
                            lambda self: "image is not present")
        state = loop.run()
        assert calls["n"] == 0
        assert "image is not present" in state.stopped_reason

    def test_the_container_is_torn_down_even_if_the_loop_raises(self, monkeypatch):
        stopped = []
        monkeypatch.setattr(agent_mod.Sandbox, "stop",
                            lambda self: stopped.append(True))

        loop, _ = _loop(monkeypatch, lambda n: "x", max_cycles=1)
        monkeypatch.setattr(loop, "_loop",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            loop.run()
        assert stopped, "a hard failure must not leak a container"

    def test_tools_are_wrapped_in_the_sandbox(self, monkeypatch):
        """The wrapper order matters: goal tools stay local, everything else confined."""
        loop, _ = _loop(monkeypatch, lambda n: "x")
        p = loop.provider
        assert isinstance(p, agent_mod.GoalToolsLocal)        # control plane outermost
        assert isinstance(p.inner, agent_mod.SandboxedProvider)


class TestToolsExcept:
    """Turning retrieval off should remove one tool, not require listing the other
    fifteen — and must work over both transports, like every other filter here."""

    class Inner:
        def __init__(self, schemas): self._s = schemas; self.called = []
        def schemas(self): return self._s
        def call(self, name, args): self.called.append(name); return "ran " + name

    def test_withholds_the_named_tool(self):
        def search_knowledge_base(): pass
        def read_file(): pass
        inner = self.Inner([search_knowledge_base, read_file])
        p = agent_mod.ToolsExcept(inner, ["search_knowledge_base"])
        assert [f.__name__ for f in p.schemas()] == ["read_file"]

    def test_withholds_it_over_mcp_schemas_too(self):
        inner = self.Inner([
            {"type": "function", "function": {"name": "search_knowledge_base"}},
            {"type": "function", "function": {"name": "read_file"}},
        ])
        p = agent_mod.ToolsExcept(inner, ["search_knowledge_base"])
        assert [agent_mod._schema_name(s) for s in p.schemas()] == ["read_file"]

    def test_a_withheld_call_never_reaches_the_inner_provider(self):
        inner = self.Inner([])
        p = agent_mod.ToolsExcept(inner, ["search_knowledge_base"])
        assert "switched off" in p.call("search_knowledge_base", {})
        assert inner.called == []

    def test_everything_else_passes_through(self):
        inner = self.Inner([])
        p = agent_mod.ToolsExcept(inner, ["search_knowledge_base"])
        assert p.call("read_file", {}) == "ran read_file"

    def test_an_empty_deny_list_changes_nothing(self):
        def a(): pass
        inner = self.Inner([a])
        p = agent_mod.ToolsExcept(inner, [])
        assert p.schemas() == [a]


class TestOneAgentAtATime:
    """One agent runs on this machine at a time, whoever starts it — the CLI, the
    console, or a second tab. Only one chat model stays resident, so two agents would
    evict each other's model every turn; two on the same name would also fight over
    one state file. The lock is an OS file lock, so it covers separate processes and
    is released by the kernel even if the holder is killed."""

    def _second(self, monkeypatch, name, calls):
        def count(n):
            calls.append(n)
            return "x"
        return _loop(monkeypatch, count, name=name, max_cycles=1)[0]

    def test_a_second_agent_cannot_start_while_one_runs(self, monkeypatch):
        seen = {}

        def first(n):
            others = []
            second = self._second(monkeypatch, "other", others)
            seen["state"] = second.run()
            seen["calls"] = list(others)
            seen["holder"] = agent_mod.running_agent()
            return "x"

        _loop(monkeypatch, first, name="first", max_cycles=1)[0].run()
        assert seen["calls"] == []                       # it never ran a cycle
        assert "already running" in seen["state"].stopped_reason
        assert "first" in seen["state"].stopped_reason   # and says who
        assert seen["holder"]["name"] == "first"
        assert not (agent_mod.AGENT_DIR / "other.state.json").exists()

    def test_a_refused_start_leaves_the_running_agents_stop_file_alone(self, monkeypatch):
        """Same name twice: the loser must not clear a stop request meant for the winner."""
        seen = {}

        def first(n):
            stop_path("same").write_text("stop")
            self._second(monkeypatch, "same", []).run()
            seen["stop_kept"] = stop_path("same").exists()
            return "x"

        _loop(monkeypatch, first, name="same", max_cycles=5)[0].run()
        assert seen["stop_kept"]

    def test_the_lock_is_released_when_a_run_ends(self, monkeypatch):
        _loop(monkeypatch, lambda n: "x", name="a", max_cycles=1)[0].run()
        assert agent_mod.running_agent() is None
        calls = []
        self._second(monkeypatch, "b", calls).run()
        assert calls == [1]

    def test_the_lock_is_released_even_if_the_run_blows_up(self, monkeypatch):
        loop = _loop(monkeypatch, lambda n: "x", name="a", max_cycles=1)[0]
        monkeypatch.setattr(loop, "_loop", lambda *a: (_ for _ in ()).throw(KeyError("x")))
        with pytest.raises(KeyError):
            loop.run()
        assert agent_mod.running_agent() is None

    def test_the_lock_spans_processes_and_survives_a_kill_9(self, tmp_path):
        """What a PID file cannot do. A real second process (standing in for
        `lag agent start` in another terminal) takes the lock; this process sees it;
        then the holder is SIGKILLed — no finally, no cleanup — and the kernel has
        released the lock anyway."""
        import signal
        import subprocess
        import sys

        holder = subprocess.Popen(
            [sys.executable, "-c",
             "import sys, time\n"
             "from pathlib import Path\n"
             "from lag import agent\n"
             "agent.AGENT_DIR = Path(sys.argv[1])     # the same dir this test uses\n"
             "lock, _ = agent._take_run_lock('cli-agent')\n"
             "print('held', flush=True)\n"
             "time.sleep(60)\n",
             str(agent_mod.AGENT_DIR)],
            stdout=subprocess.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "held"
            seen = agent_mod.running_agent()
            assert seen["name"] == "cli-agent" and seen["pid"] == holder.pid
        finally:
            holder.send_signal(signal.SIGKILL)
            holder.wait(timeout=10)
        assert agent_mod.running_agent() is None


class TestNamesStayInTheAgentsFolder:
    """A name becomes the state file, the stop file and the sandbox workspace. It was
    checked only on save, so an unsaved run named '../../..' had the home folder
    mounted read-write as its workspace."""

    @pytest.mark.parametrize("name", ["../../..", "../x", "a/b", "/abs", "", ".hidden",
                                      "x\n", "two words"])
    def test_a_spec_with_a_path_name_is_refused(self, name):
        with pytest.raises(ValueError, match="invalid agent name"):
            GoalSpec(goal="g", name=name)

    def test_ordinary_names_still_work(self):
        assert GoalSpec(goal="g", name="nightly-audit_2.v1").name == "nightly-audit_2.v1"

    def test_the_stop_file_cannot_escape(self):
        with pytest.raises(ValueError):
            stop_path("../x")

    def test_state_for_a_path_name_is_simply_absent(self):
        assert AgentState.load("../../etc/x") is None
