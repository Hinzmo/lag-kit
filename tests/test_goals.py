"""Goal pipeline tests.

The runner is driven with fake executors, so these need no Ollama, no Docker and no
MCP servers. What they pin is the queue's behaviour as you see it in the console: goals
run in order and lift, pause finishes the current goal, skip and remove take effect
mid-run, each failure policy does what its button says, and a restart never resumes
unattended work by itself.
"""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from lag import agent as agent_mod
from lag import goals as goals_mod
from lag.goals import Cancelled, GoalRunner, SandboxUnavailable


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    # The machine-wide agent lock lives under AGENT_DIR; keep it out of the repo.
    monkeypatch.setattr(agent_mod, "AGENT_DIR", tmp_path / "agents")


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    raise AssertionError("condition never became true")


def quick(answer="ok", route=("dev_agent",)):
    def ex(goal, settings, emit):
        emit({"type": "start", "stages": [{"name": n} for n in route]})
        emit({"type": "stage", "name": route[0]})
        return list(route), f"{answer}: {goal.text}"
    return ex


def blocking(release: threading.Event, started: threading.Event | None = None):
    """Runs until released, emitting as a real run streams tokens."""
    def ex(goal, settings, emit):
        emit({"type": "stage", "name": "dev_agent"})
        if started:
            started.set()
        while not release.is_set():
            emit({"type": "token", "text": "."})
            time.sleep(0.01)
        return ["dev_agent"], "finished"
    return ex


def statuses(r: GoalRunner) -> list[str]:
    return [g.status for g in r.goals]


def idle(r: GoalRunner) -> bool:
    return not r.running and r.current is None


# ── the queue ────────────────────────────────────────────────────────────────

def test_goals_run_in_order_and_record_what_ran(tmp_path):
    seen = []

    def ex(goal, settings, emit):
        seen.append(goal.text)
        return quick()(goal, settings, emit)

    r = GoalRunner(ex, tmp_path / "goals.json")
    r.add("first")
    r.add("second", done_when="tests pass")
    r.run()
    wait_for(lambda: statuses(r) == ["done", "done"])
    assert seen == ["first", "second"]
    assert r.goals[0].route == ["dev_agent"]
    assert r.goals[1].answer == "ok: second"
    assert r.goals[0].started and r.goals[0].finished


def test_done_when_reaches_the_pipeline_as_guidance(tmp_path):
    got = []
    r = GoalRunner(lambda g, s, e: (got.append(g.task()) or ([], "ok")),
                   tmp_path / "goals.json")
    r.add("add a flag", done_when="a test covers it")
    r.run()
    wait_for(lambda: statuses(r) == ["done"])
    assert "add a flag" in got[0] and "a test covers it" in got[0]


def test_an_idle_running_queue_starts_a_goal_as_soon_as_it_is_added(tmp_path):
    r = GoalRunner(quick(), tmp_path / "goals.json")
    r.run()
    time.sleep(0.05)
    assert r.running and r.current is None        # idle, waiting for work
    r.add("late arrival")
    wait_for(lambda: statuses(r) == ["done"])
    assert r.running                               # still waiting for the next one
    r.pause()
    wait_for(lambda: idle(r))


def test_empty_text_is_refused(tmp_path):
    r = GoalRunner(quick(), tmp_path / "goals.json")
    with pytest.raises(ValueError):
        r.add("   ")


def test_only_waiting_goals_move(tmp_path):
    r = GoalRunner(quick(), tmp_path / "goals.json")
    a, b, c = r.add("a"), r.add("b"), r.add("c")
    r.move(c.id, -1)
    assert [g.text for g in r.goals] == ["a", "c", "b"]
    r.move(a.id, -1)                               # already first: no-op
    assert [g.text for g in r.goals] == ["a", "c", "b"]
    r.goals[0].status = "done"
    with pytest.raises(ValueError):
        r.move(a.id, 1)
    r.move(b.id, -1)                               # swaps past c, never past done a
    assert [g.text for g in r.goals] == ["a", "b", "c"]


def test_clear_finished_keeps_what_is_still_waiting(tmp_path):
    r = GoalRunner(quick(), tmp_path / "goals.json")
    r.add("x")
    r.add("y")
    r.goals[0].status = "done"
    r.clear_finished()
    assert [g.text for g in r.goals] == ["y"]


# ── pause, skip, remove ──────────────────────────────────────────────────────

def test_pause_lets_the_current_goal_finish_then_stops(tmp_path):
    release, started = threading.Event(), threading.Event()
    r = GoalRunner(blocking(release, started), tmp_path / "goals.json")
    r.add("one")
    r.add("two")
    r.run()
    started.wait(2)
    r.pause()
    assert r.pausing and r.running
    release.set()
    wait_for(lambda: idle(r))
    assert statuses(r) == ["done", "pending"]
    assert r.note == "paused"


def test_skip_abandons_the_running_goal_and_moves_on(tmp_path):
    release, started = threading.Event(), threading.Event()
    calls = []

    def ex(goal, settings, emit):
        calls.append(goal.text)
        if goal.text == "slow":
            return blocking(release, started)(goal, settings, emit)
        return quick()(goal, settings, emit)

    r = GoalRunner(ex, tmp_path / "goals.json")
    r.add("slow")
    r.add("next")
    r.run()
    started.wait(2)
    r.skip()
    wait_for(lambda: statuses(r) == ["skipped", "done"])
    assert calls == ["slow", "next"]
    r.pause()


def test_removing_the_running_goal_drops_it(tmp_path):
    release, started = threading.Event(), threading.Event()
    r = GoalRunner(blocking(release, started), tmp_path / "goals.json")
    g = r.add("doomed")
    r.run()
    started.wait(2)
    r.remove(g.id)
    wait_for(lambda: not r.goals)
    r.pause()


def test_a_cancel_survives_code_that_catches_exception(tmp_path):
    """OllamaClient.run wraps every turn in `except Exception`. A cancel that were an
    ordinary Exception would be swallowed there and the model would keep going."""
    started = threading.Event()

    def stubborn(goal, settings, emit):
        started.set()
        for _ in range(500):
            try:  # noqa: SIM105 - spelled out: it mimics the run loop's except Exception
                emit({"type": "token", "text": "."})
            except Exception:  # noqa: BLE001 - this is the point of the test
                pass
            time.sleep(0.01)
        return [], "should never get here"

    r = GoalRunner(stubborn, tmp_path / "goals.json")
    r.add("x")
    r.run()
    started.wait(2)
    r.skip()
    wait_for(lambda: statuses(r) == ["skipped"], timeout=2)
    assert issubclass(Cancelled, BaseException) and not issubclass(Cancelled, Exception)
    r.pause()


# ── failure policies ─────────────────────────────────────────────────────────

def failing_first():
    def ex(goal, settings, emit):
        if goal.text == "bad":
            raise RuntimeError("boom")
        return quick()(goal, settings, emit)
    return ex


def test_fail_stop_pauses_the_queue(tmp_path):
    r = GoalRunner(failing_first(), tmp_path / "goals.json")
    r.configure(fail="stop")
    r.add("bad")
    r.add("good")
    r.run()
    wait_for(lambda: idle(r))
    assert statuses(r) == ["failed", "pending"]
    assert "boom" in r.goals[0].error
    assert r.note.startswith("stopped")


def test_fail_skip_carries_on(tmp_path):
    r = GoalRunner(failing_first(), tmp_path / "goals.json")
    r.configure(fail="skip")
    r.add("bad")
    r.add("good")
    r.run()
    wait_for(lambda: statuses(r) == ["failed", "done"])
    r.pause()


def test_fail_ask_waits_then_retry_runs_it_first(tmp_path):
    attempts = []

    def ex(goal, settings, emit):
        attempts.append(goal.text)
        if goal.text == "bad" and attempts.count("bad") == 1:
            raise RuntimeError("boom")
        return quick()(goal, settings, emit)

    r = GoalRunner(ex, tmp_path / "goals.json")
    r.configure(fail="ask")
    bad = r.add("bad")
    r.add("good")
    r.run()
    wait_for(lambda: idle(r))
    assert r.awaiting == bad.id and statuses(r) == ["failed", "pending"]
    r.retry(bad.id)                                # answering the question resumes
    wait_for(lambda: statuses(r) == ["done", "done"])
    assert attempts == ["bad", "bad", "good"]
    r.pause()


def test_fail_ask_skip_leaves_it_failed_and_carries_on(tmp_path):
    r = GoalRunner(failing_first(), tmp_path / "goals.json")
    r.configure(fail="ask")
    r.add("bad")
    r.add("good")
    r.run()
    wait_for(lambda: idle(r))
    r.skip()
    wait_for(lambda: statuses(r) == ["failed", "done"])
    assert r.awaiting is None
    r.pause()


def test_a_run_that_gave_up_on_malformed_turns_counts_as_failed(tmp_path):
    r = GoalRunner(lambda g, s, e: ([], "(stopped: 3 malformed turns; last was x)"),
                   tmp_path / "goals.json")
    r.add("x")
    r.run()
    wait_for(lambda: idle(r))
    assert statuses(r) == ["failed"]


def test_no_sandbox_is_the_environment_not_the_goal(tmp_path):
    def ex(goal, settings, emit):
        raise SandboxUnavailable("no sandbox: Docker is not running")

    r = GoalRunner(ex, tmp_path / "goals.json")
    r.add("x")
    r.run()
    wait_for(lambda: idle(r))
    assert statuses(r) == ["pending"]              # not marked failed
    assert "Docker is not running" in r.note


def test_unknown_fail_policy_is_refused(tmp_path):
    r = GoalRunner(quick(), tmp_path / "goals.json")
    with pytest.raises(ValueError):
        r.configure(fail="explode")


# ── one at a time, per machine ───────────────────────────────────────────────

def test_a_running_goal_holds_the_agent_lock(tmp_path):
    release, started = threading.Event(), threading.Event()
    r = GoalRunner(blocking(release, started), tmp_path / "goals.json")
    r.add("x")
    r.run()
    started.wait(2)
    assert (agent_mod.running_agent() or {}).get("name") == goals_mod.RUN_NAME
    release.set()
    wait_for(lambda: statuses(r) == ["done"])
    assert agent_mod.running_agent() is None       # released between goals
    r.pause()


def test_a_running_agent_pauses_the_queue_instead_of_fighting_it(tmp_path):
    lock, _ = agent_mod._take_run_lock("night-shift")
    try:
        r = GoalRunner(quick(), tmp_path / "goals.json")
        r.add("x")
        r.run()
        wait_for(lambda: idle(r))
        assert statuses(r) == ["pending"]
        assert "night-shift" in r.note
    finally:
        lock.close()


# ── persistence ──────────────────────────────────────────────────────────────

def test_the_list_and_settings_survive_a_restart(tmp_path):
    path = tmp_path / "goals.json"
    r = GoalRunner(quick(), path)
    r.add("keep me", done_when="saved", agents=["dev_agent"])
    r.configure(plan=False, fail="skip", model="qwen3:8b")

    again = GoalRunner(quick(), path)
    assert [g.text for g in again.goals] == ["keep me"]
    assert again.goals[0].agents == ["dev_agent"]
    assert again.settings == {"plan": False, "fail": "skip", "model": "qwen3:8b"}


def test_a_restart_never_resumes_work_by_itself(tmp_path):
    path = tmp_path / "goals.json"
    path.write_text(json.dumps({"goals": [
        {"id": "a1", "text": "interrupted", "status": "running", "started": "t"}]}))
    r = GoalRunner(quick(), path)
    assert statuses(r) == ["pending"] and r.goals[0].started == ""
    assert not r.running


def test_a_corrupt_file_starts_empty_and_says_so(tmp_path):
    path = tmp_path / "goals.json"
    path.write_text("{not json")
    r = GoalRunner(quick(), path)
    assert r.goals == [] and "could not read" in r.note


def test_subscribers_get_a_snapshot_then_live_events(tmp_path):
    r = GoalRunner(quick(), tmp_path / "goals.json")
    sub = r.subscribe()
    assert sub.get(timeout=1)["type"] == "goals"
    g = r.add("x")
    r.run()
    wait_for(lambda: statuses(r) == ["done"])
    seen = []
    while not sub.empty():
        seen.append(sub.get())
    stages = [e for e in seen if e.get("type") == "stage"]
    assert stages and stages[0]["goal"] == g.id    # live events say which goal
    r.unsubscribe(sub)
    r.pause()


# ── the real executor's wiring ───────────────────────────────────────────────

class FakePipe:
    def __init__(self, names):
        self.stages = [type("S", (), {"name": n, "model": "m"})() for n in names]

    def loads(self):
        return 1

    def run(self, task, plan=True, on_event=None):
        FakePipe.last = {"task": task, "plan": plan}
        return "answer"


@pytest.fixture
def fake_world(monkeypatch):
    import lag.pipeline as pipeline_mod
    import lag.sandbox as sandbox_mod

    calls = {}
    monkeypatch.setattr(sandbox_mod, "available", lambda: (True, ""))
    monkeypatch.setattr(sandbox_mod.Sandbox, "start", lambda self: None)
    monkeypatch.setattr(sandbox_mod.Sandbox, "stop",
                        lambda self: calls.setdefault("stopped", True))
    monkeypatch.setattr(goals_mod, "WORKSPACE", goals_mod.AGENT_DIR / "ws-test")
    # The done-when judge is one model call; scripted here so no test reaches Ollama.
    calls["verdicts"] = []
    calls["judged"] = []

    def judge_chat(cfg, model):
        def chat(prompt):
            calls["judged"].append(prompt)
            return calls["verdicts"].pop(0) if calls["verdicts"] else "MET"
        return chat
    monkeypatch.setattr(goals_mod, "judge_chat", judge_chat)

    def routed(task, cfg, provider, on_event=None, model_override=None):
        calls["routed"] = model_override
        return FakePipe(["research_agent", "dev_agent"])

    def from_personas(names, cfg, provider, model_override=None):
        calls["picked"] = names
        return FakePipe(names)

    monkeypatch.setattr(pipeline_mod.Pipeline, "routed", staticmethod(routed))
    monkeypatch.setattr(pipeline_mod.Pipeline, "from_personas", staticmethod(from_personas))
    return calls


def test_auto_goals_are_routed_by_the_orchestrator(fake_world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = goals_mod.pipeline_executor({}, lambda: object())
    g = goals_mod.Goal(id="1", text="t", done_when="d")
    route, answer, _ = ex(g, {"plan": False, "model": "qwen3:8b"}, lambda e: None)
    assert route == ["research_agent", "dev_agent"] and answer == "answer"
    assert fake_world["routed"] == "qwen3:8b"
    assert FakePipe.last == {"task": g.task(), "plan": False}
    assert fake_world["stopped"]                   # the container never outlives a goal


def test_picked_agents_skip_routing(fake_world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = goals_mod.pipeline_executor({}, lambda: object())
    g = goals_mod.Goal(id="1", text="t", agents=["dev_agent", "tester_agent"])
    route, _, _ = ex(g, {"plan": True, "model": ""}, lambda e: None)
    assert route == ["dev_agent", "tester_agent"]
    assert "routed" not in fake_world


def test_no_container_runtime_fails_closed(monkeypatch):
    import lag.sandbox as sandbox_mod

    monkeypatch.setattr(sandbox_mod, "available", lambda: (False, "Docker is not running"))
    ex = goals_mod.pipeline_executor({}, lambda: object())
    with pytest.raises(SandboxUnavailable):
        ex(goals_mod.Goal(id="1", text="t"), {}, lambda e: None)


# ── deciding instead of asking ───────────────────────────────────────────────
# The three answers below are trimmed from a real run on qwen3:8b. Two of them were
# counted done although one handed back a choice and one stopped at a plan.

ASKED = ("Pytest's `--durations` flag shows slow tests by default. Would you like me to:\n\n"
         "1. Search for pytest duration reports in the cache\n"
         "2. Suggest using `pytest --durations` directly\n\n"
         "Which approach would be most helpful?")
DEFERRED = ("4. **Test the New Flag**: Write a test to ensure it works.\n\n"
            "To proceed, we will first locate the implementation of the `lag run` command "
            "and modify it to include the `--dry-run` flag.")
FINISHED = ("Should the flag skip the network too? Yes: a dry run must not touch anything.\n\n"
            "Wrote /workspace/dry-run.patch, which adds `--dry-run` to `lag run` and a "
            "test for it.")


def test_an_answer_that_hands_the_goal_back_is_spotted():
    assert "question" in goals_mod.stopped_short(ASKED)
    assert "next step" in goals_mod.stopped_short(DEFERRED)
    assert goals_mod.stopped_short(FINISHED) == ""   # a question it answered itself is fine
    assert goals_mod.stopped_short("") == ""


def test_every_goal_is_told_to_decide_and_where_it_can_write():
    t = goals_mod.Goal(id="1", text="find slow tests", done_when="a list").task()
    assert t.startswith(goals_mod.UNATTENDED)
    assert "Goal: find slow tests" in t and "This goal is done when: a list" in t
    assert "/workspace" in goals_mod.UNATTENDED and "never with a question" in goals_mod.UNATTENDED


def test_an_answer_that_still_asks_is_not_counted_done(tmp_path):
    r = GoalRunner(lambda g, s, e: (["analyst_agent"], ASKED), tmp_path / "goals.json")
    r.configure(fail="skip")
    r.add("find slow tests")
    r.run()
    wait_for(lambda: statuses(r) == ["failed"])
    assert "not finished" in r.goals[0].error and "question" in r.goals[0].error


class AskingPipe(FakePipe):
    """Its run asks; its last stage answers the nudges from a script."""
    def __init__(self, names, replies):
        super().__init__(names)
        self.prompts = []
        last = self.stages[-1]
        last.run = lambda prompt, on_event=None: (self.prompts.append(prompt), replies.pop(0))[1]

    def run(self, task, plan=True, on_event=None):
        return ASKED

    def _fit(self, context, stage, on_event=None):
        return context


def _asking_world(fake_world, monkeypatch, replies):
    import lag.pipeline as pipeline_mod
    pipe = AskingPipe(["analyst_agent"], replies)
    monkeypatch.setattr(pipeline_mod.Pipeline, "from_personas",
                        staticmethod(lambda names, cfg, provider, model_override=None: pipe))
    return pipe


def test_the_last_stage_is_sent_back_to_decide(fake_world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipe = _asking_world(fake_world, monkeypatch, [FINISHED])
    events = []
    ex = goals_mod.pipeline_executor({}, lambda: object())
    g = goals_mod.Goal(id="1", text="find slow tests", agents=["analyst_agent"])
    _, answer, _ = ex(g, {}, events.append)
    assert answer == FINISHED
    assert len(pipe.prompts) == 1
    assert ASKED in pipe.prompts[0] and "best serves the goal" in pipe.prompts[0]
    assert [e["type"] for e in events if e["type"] in ("nudge", "stage")] == ["nudge", "stage"]


def test_nudges_are_bounded(fake_world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipe = _asking_world(fake_world, monkeypatch, [ASKED] * 5)
    ex = goals_mod.pipeline_executor({}, lambda: object())
    g = goals_mod.Goal(id="1", text="t", agents=["analyst_agent"])
    _, answer, _ = ex(g, {}, lambda e: None)
    assert len(pipe.prompts) == goals_mod.NUDGES
    assert goals_mod.stopped_short(answer)             # the runner then marks it failed



# ── done-when check ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("reply,want", [
    ("MET", ("met", "")),
    ("**MET**", ("met", "")),
    ("NOT MET: the third bullet names no function", ("not met", "the third bullet names no function")),
    ("not met - missing the list", ("not met", "missing the list")),
    ("I think it is probably fine", ("unchecked", "unreadable verdict: 'I think it is probably fine'")),
])
def test_check_done_reads_the_verdict(reply, want):
    g = goals_mod.Goal(id="1", text="t", done_when="each bullet names a function")
    assert goals_mod.check_done(lambda p: reply, g, "an answer") == want


def test_check_done_never_fails_on_a_broken_judge():
    g = goals_mod.Goal(id="1", text="t", done_when="d")

    def boom(prompt):
        raise ConnectionError("ollama down")
    verdict, reason = goals_mod.check_done(boom, g, "a")
    assert verdict == "unchecked" and "ollama down" in reason


def test_check_done_skips_goals_without_a_finish_line():
    g = goals_mod.Goal(id="1", text="t")
    assert goals_mod.check_done(lambda p: 1 / 0, g, "a")[0] == "unchecked"


def test_a_rejected_answer_is_sent_back_once_with_the_gap(fake_world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # The first answer asks (one nudge), the nudged one finishes, the check rejects it,
    # and the fix turn's answer passes.
    pipe = _asking_world(fake_world, monkeypatch, [FINISHED, FINISHED])
    fake_world["verdicts"] = ["NOT MET: no file named", "MET"]
    events = []
    ex = goals_mod.pipeline_executor({}, lambda: object())
    g = goals_mod.Goal(id="1", text="t", done_when="names the file", agents=["analyst_agent"])
    _, answer, verdict = ex(g, {}, events.append)
    assert verdict == ("met", "")
    assert len(pipe.prompts) == 2 and "no file named" in pipe.prompts[1]
    assert "names the file" in fake_world["judged"][0]
    assert [e["verdict"] for e in events if e["type"] == "check"] == ["not met", "met"]


def test_the_runner_fails_a_goal_whose_finish_line_is_not_met(tmp_path):
    r = GoalRunner(lambda g, s, e: (["a"], "an answer", ("not met", "no tests listed")),
                   tmp_path / "goals.json")
    r.add("t", done_when="lists the tests")
    r.run()
    wait_for(lambda: statuses(r) == ["failed"])
    g = r.goals[0]
    assert g.error == "done-when not met: no tests listed" and g.check == "not met: no tests listed"


def test_the_runner_records_a_met_check(tmp_path):
    r = GoalRunner(lambda g, s, e: (["a"], "an answer", ("met", "")), tmp_path / "goals.json")
    r.add("t", done_when="x")
    r.run()
    wait_for(lambda: statuses(r) == ["done"])
    assert r.goals[0].check == "met"


def test_the_runner_tracks_the_running_stages_model(tmp_path):
    seen = []

    def ex(goal, settings, emit):
        emit({"type": "stage", "name": "dev_agent", "model": "gemma4:e4b"})
        seen.append(r.model)
        return ["dev_agent"], "done"
    r = GoalRunner(ex, tmp_path / "goals.json")
    r.add("t")
    r.run()
    wait_for(lambda: statuses(r) == ["done"])
    assert seen == ["gemma4:e4b"] and r.model == ""   # cleared once the goal ends


class TestJudgeChatMatchesTheStage:
    """The done-when check runs on the model the last stage left resident. It built a
    raw client: no keep_alive (reset to Ollama's 5 minutes) and a guessed num_ctx
    (a different window reloads the model)."""

    def _sent(self, monkeypatch, cfg):
        import lag.ollama_client as oc
        oc._CAPS.clear()
        sent = {}

        class FakeClient:
            def __init__(self, host=None): pass
            def show(self, model): return SimpleNamespace(capabilities=["completion"])
            def chat(self, **kw):
                sent.update(kw)
                return SimpleNamespace(message=SimpleNamespace(content="MET"))

        monkeypatch.setattr(oc.ollama, "Client", FakeClient)
        assert goals_mod.judge_chat(cfg, "m:1")("is it done?") == "MET"
        return sent

    def test_it_keeps_the_model_resident(self, monkeypatch):
        sent = self._sent(monkeypatch, {"ollama": {"keep_alive": "30m", "num_ctx": 4096}})
        assert sent["keep_alive"] == "30m"
        assert sent["options"]["num_ctx"] == 4096
        assert sent["options"]["num_predict"] == 200 and sent["options"]["temperature"] == 0

    def test_no_window_is_invented(self, monkeypatch):
        """No num_ctx in config means no num_ctx sent — the same as the stage did."""
        sent = self._sent(monkeypatch, {})
        assert "num_ctx" not in sent["options"]
