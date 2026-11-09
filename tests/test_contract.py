"""The API contract, the recorded examples and the server must agree.

The phone app is built against contract/v1.json and replays contract/examples/. A
renamed field or an unlisted event would break it quietly, on someone's phone. These
tests make the same mistake fail here, in lag-kit, first.

What they hold together:
- every event the server code sends is in the contract, and every contract event is
  still sent (so the contract cannot go stale in either direction);
- every recorded example matches the contract, field by field;
- the live endpoints answer with the shapes the contract lists;
- docs/api-contract.md names every event and endpoint.

Re-record the examples with contract/record.py after a change to what the server sends.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "contract" / "v1.json").read_text())
EVENTS = CONTRACT["events"]
EXAMPLES = ROOT / "contract" / "examples"
DOC = ROOT / "docs" / "api-contract.md"

# `{"type": ...}` literals in the source that are not events.
NOT_EVENTS = {"object"}        # an empty JSON schema, in mcp_client._input_schema

_PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
       "array": list, "object": dict, "null": type(None)}


def _is(value, spec: str) -> bool:
    for t in spec.split("|"):
        if t in ("integer", "number") and isinstance(value, bool):
            continue                       # True is an int in Python, not in JSON
        if isinstance(value, _PY[t]):
            return True
    return False


def _fields(obj: dict, fields: dict, where: str, required: bool = True) -> list[str]:
    out = []
    for name, spec in fields.items():
        if name not in obj:
            if required:
                out.append(f"{where}: missing `{name}`")
        elif not _is(obj[name], spec):
            out.append(f"{where}: `{name}` should be {spec}, "
                       f"got {type(obj[name]).__name__} {obj[name]!r:.60}")
    return out


def check_event(ev: dict, where: str) -> list[str]:
    kind = ev.get("type")
    if kind not in EVENTS:
        return [f"{where}: event type {kind!r} is not in the contract"]
    spec = EVENTS[kind]
    out = _fields(ev, spec["fields"], f"{where} ({kind})")
    out += _fields(ev, spec.get("optional", {}), f"{where} ({kind})", required=False)
    if kind == "goals":
        for i, g in enumerate(ev["goals"]):
            out += _fields(g, CONTRACT["goal_record"]["fields"], f"{where} goals[{i}]")
            if g.get("status") not in CONTRACT["goal_record"]["status"]:
                out.append(f"{where} goals[{i}]: unknown status {g.get('status')!r}")
    return out


def _sent_by_server() -> set[str]:
    src = "".join(p.read_text() for p in (ROOT / "src" / "lag").glob("*.py"))
    return set(re.findall(r'\{"type": ?"([a-z_]+)"', src)) - NOT_EVENTS


def _recorded() -> dict[str, list[dict]]:
    return {p.stem: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            for p in sorted(EXAMPLES.glob("*.jsonl"))}


# ── the contract against the source ───────────────────────────────────────────

def test_every_event_the_server_sends_is_in_the_contract():
    missing = _sent_by_server() - set(EVENTS)
    assert not missing, (f"the server sends {sorted(missing)} but contract/v1.json does not "
                         f"list them: add them (with their fields) and to docs/api-contract.md")


def test_every_contract_event_is_still_sent():
    stale = set(EVENTS) - _sent_by_server()
    assert not stale, f"contract/v1.json lists {sorted(stale)}, which nothing sends any more"


def test_the_contract_version_is_the_servers():
    from lag import web
    assert CONTRACT["contract"] == web.CONTRACT_VERSION


# ── the recorded examples against the contract ───────────────────────────────

def test_the_examples_were_recorded():
    have = set(_recorded())
    assert {"chat-tool", "pipeline-routed", "stream-failure", "goal-run"} <= have
    assert (EXAMPLES / "responses.json").exists()


@pytest.mark.parametrize("name", sorted(p.stem for p in EXAMPLES.glob("*.jsonl")))
def test_every_recorded_event_matches_the_contract(name):
    problems = []
    for i, ev in enumerate(_recorded()[name]):
        problems += check_event(ev, f"{name}.jsonl line {i + 1}")
    assert not problems, "\n".join(problems)


def test_a_running_goals_events_name_the_goal():
    live = [e for e in _recorded()["goal-run"] if e["type"] != "goals"]
    assert live, "goal-run.jsonl recorded no live events"
    assert all(isinstance(e.get("goal"), str) for e in live)


def test_the_examples_cover_what_the_app_draws():
    seen = {e["type"] for evs in _recorded().values() for e in evs}
    core = {"start", "stage", "token", "tool", "tool_result", "done", "goals",
            "routing", "routed", "turn_error"}
    assert core <= seen, f"no recorded example of {sorted(core - seen)}"


def test_the_recorded_responses_match_the_endpoints():
    responses = json.loads((EXAMPLES / "responses.json").read_text())
    eps = CONTRACT["endpoints"]
    problems = []
    for key in ("GET /api/version", "GET /api/state", "GET /api/devices", "POST /api/pair"):
        problems += _fields(responses[key], eps[key]["response"], key)
    for key in ("GET /api/goals", "POST /api/goals", "DELETE /api/goals/{id}"):
        problems += check_event(responses[key], key)
    problems += _fields(responses["401 GET /api/state without a key"],
                        CONTRACT["errors"]["fields"], "401")
    assert not problems, "\n".join(problems)


def _model_lists(node, found):
    if isinstance(node, dict):
        if isinstance(node.get("ollama"), dict):
            o = node["ollama"]
            found |= set(o.get("available", []))
            found |= {m.get("name") for m in o.get("loaded", []) + o.get("models", [])}
        for v in node.values():
            _model_lists(v, found)
    elif isinstance(node, list):
        for v in node:
            _model_lists(v, found)
    return found


def test_the_examples_hold_nothing_personal():
    """These files are public. The first recording leaked the whole installed-model
    list (it sat one level deeper than the scrubber looked) and the scratch folder's
    path, which carries the user name. Both are checked here, by shape, so no list of
    names has to live in the repo."""
    import getpass
    import os

    text = "\n".join(p.read_text() for p in EXAMPLES.iterdir() if p.is_file())
    assert os.path.expanduser("~") not in text
    assert getpass.getuser() not in text
    assert "/private/" not in text and "/tmp/" not in text

    responses = json.loads((EXAMPLES / "responses.json").read_text())
    assert responses["POST /api/pair"]["key"].startswith("<")
    state = responses["GET /api/state"]
    roles = {state["chat_model"]} | {p.get("model") for p in state["personas"]}
    listed = _model_lists(responses, set())
    for evs in _recorded().values():
        _model_lists(evs, listed)
    assert listed <= roles, f"models beyond the roles' own: {sorted(listed - roles)}"


# ── the live endpoints against the contract ───────────────────────────────────

@pytest.fixture
def server(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lag import goals, phone, web

    monkeypatch.setattr(phone, "PHONE_DIR", tmp_path / "phone")
    monkeypatch.setattr(web, "PHONE_PORT", 8766)
    monkeypatch.setattr(web, "_ollama_status", lambda cfg: (True, ["m:1"], []))
    monkeypatch.setattr(web, "_chroma_count", lambda cfg: {"available": True, "chunks": 0})
    runner = goals.GoalRunner(lambda goal, settings, emit: ([], "", None),
                              path=tmp_path / "goals.json")
    monkeypatch.setattr(web, "_goal_runner_cache", {"runner": runner})
    return (TestClient(web.app), TestClient(web.app, base_url="http://testserver:8766"))


def test_the_live_endpoints_answer_in_the_contracts_shapes(server):
    from lag import phone

    console, handset = server
    eps = CONTRACT["endpoints"]
    problems = []
    for key in ("GET /api/version", "GET /api/state", "GET /api/devices"):
        problems += _fields(console.get(key.split()[1]).json(), eps[key]["response"], key)
    problems += check_event(console.get("/api/goals").json(), "GET /api/goals")
    added = console.post("/api/goals", json={"text": "a goal"}).json()
    problems += check_event(added, "POST /api/goals")
    code, _ = phone.new_code()
    paired = handset.post("/api/pair", json={"code": code, "name": "p"}).json()
    problems += _fields(paired, eps["POST /api/pair"]["response"], "POST /api/pair")
    refused = handset.get("/api/state")
    problems += _fields(refused.json(), CONTRACT["errors"]["fields"], "401")
    assert refused.status_code == 401
    assert not problems, "\n".join(problems)


# ── the words against the contract ────────────────────────────────────────────

def test_the_doc_names_every_event_and_endpoint():
    doc = DOC.read_text()
    missing = [f"`{e}`" for e in EVENTS if f"`{e}`" not in doc]
    missing += [ep for ep in CONTRACT["endpoints"] if ep.split()[1] not in doc]
    assert not missing, f"docs/api-contract.md does not mention: {missing}"
