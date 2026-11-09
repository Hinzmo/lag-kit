"""Record the example streams in contract/examples/ from a real lag serve.

The examples are what both sides test against: lag-kit checks they still match the
contract, and the phone app replays them through its parser and its fake server. So
they must come from the real server, not be written by hand.

Run it against a scratch server, never your own: it adds a goal and runs the queue.

    mkdir /tmp/lag-rec && cd /tmp/lag-rec
    cp ~/local-agent-kit/config.yaml . && ln -s ~/local-agent-kit/personas personas
    lag serve --port 8797 --phone-port 8796 &
    lag phone pair            # note the code
    python ~/local-agent-kit/contract/record.py --code XXXX-XXXX

Everything goes through the phone port with a device key, as the app will. What is
saved is scrubbed: the scratch server's folder becomes /Users/you/lag-kit, your user
name becomes "you", the pairing key is replaced, and every installed-model list keeps
only the models your roles use. Pass the scratch folder as --root. To scrub files
already recorded, run with --rescrub (and --root, --models).

Standard library only, so it runs in any Python 3.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "examples"
HOME = os.path.expanduser("~")
KEY_PLACEHOLDER = "<device key: 43 url-safe characters>"


class Client:
    def __init__(self, base: str, key: str = ""):
        self.base, self.key = base.rstrip("/"), key

    def _req(self, method: str, path: str, body: dict | None = None):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        data = json.dumps(body).encode() if body is not None else None
        return urllib.request.Request(self.base + path, data=data, method=method, headers=headers)

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(self._req(method, path, body), timeout=60) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def events(self, method: str, path: str, body: dict | None = None, until=None,
               timeout: float = 900):
        """Yield each `data:` event of a server-sent stream until it ends, or until
        `until(event)` is true."""
        deadline = time.time() + timeout
        with urllib.request.urlopen(self._req(method, path, body), timeout=timeout) as r:
            for raw in r:
                line = raw.decode().rstrip("\n")
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                yield ev
                if (until and until(ev)) or time.time() > deadline:
                    return


ROOT_STANDIN = "/Users/you/lag-kit"


def _models_only(node, keep: set[str]):
    """Every installed-model list, at any depth, cut to the models the roles use.
    The list sits under `ollama` in /api/state, which responses.json nests one level
    down; a check at the top level only let a whole machine's model list through."""
    if isinstance(node, dict):
        o = node.get("ollama")
        if isinstance(o, dict):
            o["available"] = [m for m in o.get("available", []) if m in keep]
            o["loaded"] = [m for m in o.get("loaded", []) if m.get("name") in keep]
            o["models"] = [m for m in o.get("models", []) if m.get("name") in keep]
        for v in node.values():
            _models_only(v, keep)
    elif isinstance(node, list):
        for v in node:
            _models_only(v, keep)
    return node


def scrub(obj, keep_models: set[str], root: str = ""):
    """Remove what is personal to this machine before anything is saved."""
    text = json.dumps(obj)
    for path in sorted({root, os.path.realpath(root)} - {""}, key=len, reverse=True):
        text = text.replace(path, ROOT_STANDIN)
    text = text.replace(HOME, "/Users/you")
    user = getpass.getuser()
    if len(user) >= 3:
        text = text.replace(user, "you")
    return _models_only(json.loads(text), keep_models)


def rescrub(root: str, keep: set[str]) -> None:
    for f in sorted(OUT.glob("*.jsonl")):
        lines = [json.dumps(scrub(json.loads(x), keep, root), ensure_ascii=False)
                 for x in f.read_text().splitlines() if x.strip()]
        f.write_text("\n".join(lines) + "\n")
    r = OUT / "responses.json"
    r.write_text(json.dumps(scrub(json.loads(r.read_text()), keep, root),
                            indent=2, ensure_ascii=False) + "\n")
    print(f"rescrubbed {OUT}")


def save_stream(name: str, events: list[dict], keep: set[str], root: str) -> None:
    lines = [json.dumps(scrub(e, keep, root), ensure_ascii=False) for e in events]
    (OUT / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    types = sorted({e.get("type", "?") for e in events})
    print(f"  {name}.jsonl: {len(events)} events · {', '.join(types)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phone", default="http://127.0.0.1:8796", help="the phone port")
    ap.add_argument("--code", help="the code `lag phone pair` printed")
    ap.add_argument("--root", default="", help="the scratch server's folder, to scrub")
    ap.add_argument("--rescrub", action="store_true", help="scrub the saved files again")
    ap.add_argument("--models", default="", help="with --rescrub: comma-separated models to keep")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.rescrub:
        rescrub(a.root, {m for m in a.models.split(",") if m})
        return
    if not a.code:
        ap.error("--code is required")

    anon = Client(a.phone)
    responses: dict = {}
    status, refused = anon.call("GET", "/api/state")
    responses["401 GET /api/state without a key"] = refused
    status, paired = anon.call("POST", "/api/pair", {"code": a.code, "name": "recorder"})
    if status != 200:
        raise SystemExit(f"pairing failed: {paired}")
    c = Client(a.phone, paired["key"])
    responses["POST /api/pair"] = {**paired, "key": KEY_PLACEHOLDER}

    _, state = c.call("GET", "/api/state")
    keep = {state.get("chat_model", "")} | {p.get("model", "") for p in state.get("personas", [])}
    for path in ("/api/version", "/api/state", "/api/goals", "/api/devices"):
        responses[f"GET {path}"] = c.call("GET", path)[1]

    print("recording (each stream runs a real model; minutes, not seconds):")
    save_stream("chat-tool", list(c.events("POST", "/api/stream", {
        "prompt": "What time is it right now? Use the now tool, then answer in one "
                  "short sentence.", "mode": "solo"})), keep, a.root)
    save_stream("pipeline-routed", list(c.events("POST", "/api/stream", {
        "prompt": "Write a Python function that returns the sum of a list of numbers, "
                  "with one pytest test for it. Keep it short.",
        "mode": "pipeline", "route": True})), keep, a.root)
    save_stream("stream-failure", list(c.events("POST", "/api/stream", {
        "prompt": "hello", "mode": "solo", "model": "no-such-model:1"})), keep, a.root)

    # A goal: watch the queue's stream while adding a goal and running the queue.
    seen: list[dict] = []
    finished = threading.Event()

    def watch():
        def over(ev):
            return ev.get("type") == "goals" and any(
                g["status"] in ("done", "failed") for g in ev.get("goals", []))
        for ev in c.events("GET", "/api/goals/events", until=over):
            seen.append(ev)
        finished.set()

    threading.Thread(target=watch, daemon=True).start()
    time.sleep(1)
    _, snap = c.call("POST", "/api/goals", {
        "text": "Write a file hello.py that prints hello, and run it once.",
        "done_when": "hello.py exists and running it prints hello"})
    responses["POST /api/goals"] = snap
    c.call("POST", "/api/goals/run")
    finished.wait(timeout=900)
    save_stream("goal-run", seen, keep, a.root)

    _, snap = c.call("GET", "/api/goals")
    gid = snap["goals"][0]["id"] if snap.get("goals") else ""
    responses["DELETE /api/goals/{id}"] = c.call("DELETE", f"/api/goals/{gid}")[1]

    (OUT / "responses.json").write_text(
        json.dumps(scrub(responses, keep, a.root), indent=2, ensure_ascii=False) + "\n")
    print(f"  responses.json: {len(responses)} endpoints")


if __name__ == "__main__":
    main()
