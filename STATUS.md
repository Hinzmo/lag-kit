# local-agent-kit v2 — status

Last updated **2026-11-08**. This is the living status doc: what works, how it works, and
what's left. Working rules for Claude Code live in `CLAUDE.md`.

---

## Where it stands

The six original phases are done, and the kit has grown well past them. **609 tests pass**
with Docker running (four real-container tests skip without it). Primary box is the **MacBook Air M4**;
the CPU-only Ubuntu machine it was built on is secondary.

| Phase | What | State |
|------:|------|-------|
| 0 | Foundation, config, packaging | ✅ |
| 1 | ReAct loop over Ollama, local tools | ✅ verified |
| 2 | MCP server (stdio) wrapping those tools | ✅ verified |
| 3 | The bridge — harness as MCP client | ✅ verified |
| 4 | Project pipeline (sequential, orchestrator-routed) | ✅ verified |
| 5 | Nine personas | ✅ all load |
| 6 | RAG server + always-on retrieval | ✅ verified |

Since Phase 6:

| Feature | Command | State |
|---------|---------|-------|
| Live web console (streaming thinking, tokens, tool calls) | `lag serve` | ✅ wired to the real harness |
| Goal pipeline — a queue of goals the orchestrator works through | console | ✅ sandboxed, runs with the tab closed |
| Agent mode — saved agents that loop on a goal | `lag agent …` | ✅ |
| Container sandbox for agents (mandatory) | — | ✅ needs Docker running |
| Study mode — ask/quiz over your own course notes | `lag study …` | ✅ |
| Readable output (`render.py`) | — | ✅ |
| SSRF guard on `http_fetch` | — | ✅ |

Repo: `github.com/Hinzmo/lag-kit`, default branch `main`.

---

## How it works

Three decoupled layers. The harness is the only piece that knows about the others.

```
Ollama (local LLM)  ──tools ↓ / tool_calls ↑──  Harness (lag)  ──stdio · JSON-RPC──  MCP servers
                                                 ReAct loop                           tools_server.py  (16 tools)
                                                 + MCP client                         rag_server.py    (1 tool)
```

### The one abstraction that matters: `tool_provider`

`OllamaClient.run` accepts anything exposing `schemas()` and `call(name, args)`.
`LocalToolProvider` (in-process) and `MCPToolProvider` (real server subprocesses) both
satisfy it, so `lag run` and `lag run --mcp` exercise identical tools over different
transports with zero change to the loop.

Everything that sits in front of the tools — allow-lists, deny-lists, keeping goal tools
in-process, the sandbox — is a **wrapper** around a provider (`providers.py`,
`sandbox.py`), so they compose and the loop never knows which are present.

### Tools

Sixteen functions in `src/lag/tools.py`, consumed unchanged by Ollama's schema generation
*and* by `servers/tools_server.py` (which imports the `TOOLS` dict):

| Group | Tools |
|-------|-------|
| Files & shell | `read_file`, `write_file`, `list_dir`, `find_files`, `grep_files`, `run_shell`, `system_info` |
| Memory | `remember`, `recall` |
| Web (leaves the machine) | `web_search`, `http_fetch` |
| Time | `now`, `sleep_for` |
| Agent goal control | `goal_status`, `note_progress`, `mark_done` |

`rag_server.py` adds `search_knowledge_base`, so `--mcp` exposes 17. Tool errors come back
to the model as strings, never raised.

### Memory is two-layered, retrieval is always on

`remember(key, value)` writes to `memory/store.json` (exact lookup via `recall`) **and**
embeds into ChromaDB, so it comes back from a plain question. Documents and remembered
facts share one collection and one ranking; every turn searches it first. Indexing degrades
safely without `chromadb` or an embedder. `src/lag/rag.py` holds the retrieval core;
`rag_server.py` is just the protocol wrapper.

### One model in memory: warm-up and sharing

- **Warm-up.** `lag serve` loads the chat (`worker`) model in a background thread at
  startup, generating one token at the chat's own `num_ctx`, so the first message skips
  the 5-7 s cold load. One token rather than an empty prompt: on the MLX engine the first
  real generation after a load also sets up the GPU kernels. `LAG_WARM=0` turns it off
  (the test suite sets it in `tests/conftest.py`).
- **Chat shares the running goal's model.** The goal runner tracks the model its current
  stage is on; while a goal runs, `/api/stream` swaps the chat onto that model and says so
  (`shared_model` event). A different model would evict the goal's and the two would swap
  at a cold load each. Chat is not blocked: same-model requests share Ollama.
- **Chat starts in ~0.2 s, even right after a start.** Every chat sends ~1,300 tokens of
  tool schemas ahead of the question; reading them costs 2.7 s on gemma4:e4b and 10 s on
  qwen3.5:9b. Ollama's cache only reuses that shared start after it has seen two prompts
  that part ways right where the question begins, so the first two different chats after
  a load used to pay in full. `_warm()` now sends two one-token chats with the real MCP
  tool list ("a", then "b") at startup, which also starts the MCP servers. Measured from
  a cold load: the first real chat dropped from 4.9 s to 0.25 s, and chat stays cached
  while goal agents with their own system prompts run in between. What is left of a
  chat's wait is the model thinking before it answers (48-265 thinking tokens on gemma).

### Live model list

`GET /api/models/watch` (SSE) fingerprints Ollama every 3s — `/api/tags` names + digests
and `/api/ps` residents, ~1.5 ms together, vs ~14 ms for a subprocess `ollama list` — and
pushes the full model block only when the fingerprint changes. Checks run only while a
client is connected; the console closes the stream when its tab is hidden. Verified
live: a terminal `ollama pull` appeared in ~2s, an `ollama rm` vanished in ~2s.

### Live streaming

`on_event` is a side channel on `OllamaClient.run` / `Pipeline.run` — same return value
with or without it. The console's `/api/stream` (POST → `text/event-stream`) forwards
`stage`, `thinking`, `token`, `tool`, `tool_result`, `done` and `error` events. `render.py`
cleans tool-call JSON and fumbled tool calls once, upstream, so chat, activity window and
CLI all show the same readable text.

### Pipeline

Sequential stages, each seeing the context above it. By default the orchestrator **routes**
— picks which personas the task needs; `--agents` / `--no-route` override. The lead plans
once and its instructions ride down. `parallel` and `crew` were removed (see *One model at
a time*).

### Goal pipeline (`goals.py`, console)

The console's pipeline panel is a **queue of goals**. You write what should get done (plus
an optional "done when"); each goal becomes one routed pipeline run, or a fixed one if you
picked its agents. Goals run **one at a time** — the one-resident-model rule again — and
the list lifts as each finishes. It replaced the six-step project wizard, which asked you
to make the orchestrator's routing decisions by hand and then only pre-filled the chat box.

- **Server-owned.** A `GoalRunner` thread in `lag serve` drains the queue, so it keeps going
  with the browser closed. The console renders snapshots from `/api/goals/events` (SSE).
  The list persists to `agents/goals.json`; a restart puts an interrupted goal back to
  pending and comes up **paused** — unattended work never resumes by itself.
- **Confined like agent mode.** Each goal runs in a fresh container sandbox with the
  egress proxy, writing to one shared workspace (`agents/workspaces/goals`) so later goals
  can build on earlier ones. No Docker → the goal goes back to pending and the queue pauses
  with the reason (it is not marked failed).
- **Decides instead of asking.** Every goal's task opens with a note that nobody will answer
  mid-run: pick the option that best serves the goal, say which, and finish with the
  result. The project is read-only at `/project`, so work lands in `/workspace`. If the
  final answer still ends on a question or an announced next step (`stopped_short()`), the
  last stage is sent back up to twice; if it still stops short, the goal fails with
  "not finished" instead of counting as done. Seen live: qwen3:8b had ended a goal on
  "Which approach would be most helpful?"; with this it ends on a conclusion.
- **Shares the agent lock.** A running goal holds `agents/.run.lock`, so agent mode is
  refused meanwhile; a running agent pauses the queue instead of fighting it. The lock is
  released between goals, never held idle.
- **Controls:** pause finishes the current goal then stops; **skip** and **stop & remove**
  take effect at the running goal's next event (measured: 3.5s mid-answer), by raising
  `Cancelled` from the event hook. `Cancelled` is a `BaseException` on purpose —
  `OllamaClient.run` catches `Exception` around every turn and would swallow it.
- **On fail:** `stop` pauses the queue, `skip` carries on, `ask` pauses with the failed goal
  in the now-working card and retry / carry-on buttons. A run that gave up on malformed
  turns (`(stopped: …`) counts as failed.
- **"Done when" is checked.** After the work, `check_done()` asks the last stage's model
  (already resident, so no reload) for a one-line verdict: `MET` or `NOT MET: <gap>`. Not
  met sends that stage back once with the gap; still not met fails the goal with
  `done-when not met: …`. An unreadable verdict or a judge error records `unchecked` and
  counts as done, so a flaky judge never sinks good work. The verdict is kept on the goal
  (`check`) and shown in the console, the activity window and `lag goals`.
- **Out of turns is not a failure.** When an agent spends all `max_iters` on tool calls,
  `OllamaClient.run` asks once more with tools withdrawn ("answer from what you have"),
  so the gathered work becomes an answer instead of `(stopped: hit max_iters …)`.
- **`lag goals` (terminal).** `add` (with `--done-when`, `--agents`), `ls`, `show`, `run`,
  `pause`, `skip`, `rm`, `retry`, `move`, `clear`, `set`, and `watch`, which streams the
  live run (stages, tool calls, tokens, nudges, the check, outcomes). It talks to the
  running `lag serve` (`LAG_URL`, default `http://127.0.0.1:8765`) instead of opening
  `agents/goals.json`, so the queue keeps one writer. Goals are addressed by id prefix or
  list position. An empty `--agents` is an error here and in `lag pipeline`, not a silent
  fall back to the default roster.

### Agent mode

One persona, one goal, a loop until `mark_done` — with `--max-cycles` and a stop file as
backstops. Each cycle is a fresh turn; the goal, the `note_progress` log and the last
observation are the continuity. Exceptions become observations, never crashes. Files live
in `agents/`: `<name>.yaml` (definition), `<name>.state.json` (run state), `<name>.stop`,
and `workspaces/`.

### Sandbox — agents only, mandatory

Agent shells run in a Docker/Podman container:

- project mounted **read-only** at `/project`, writable workspace at `/workspace`
- all capabilities dropped, 256-pid limit, 2 GB memory (swap capped at the same 2 GB),
  2 CPUs, no published ports
- **network on by default = public internet only**, through a per-agent filtering proxy
  (`egress.py`) on an `--internal` Docker network: the host, LAN and tailnet are refused,
  raw sockets have no route; `--no-network` / `network: false` seals an agent
- **fails closed** — no running container runtime, no agent run

Host-side file tools are path-translated and write-checked in Python. The command
deny-list is defence in depth, not the boundary. Supervised `lag run` / `pipeline` are
**not** sandboxed (deliberate; see *What's left*).

### Phone access (`phone.py`, `lag serve --phone-port`)

The phone app reaches `lag serve` through `tailscale serve`, which forwards only to
127.0.0.1, so by address a phone looks like the Mac's own browser. It therefore gets a
second listener (`--phone-port`, always 127.0.0.1) in the same process, told apart by the
port a request arrived on. On that port:

- **Every request needs a device key** (`Authorization: Bearer …`), except `POST /api/pair`.
  `lag phone pair` shows a QR code and an 8-character code (5 minutes, single use, 5 tries);
  the phone trades it for a 256-bit key. `phone/devices.json` keeps only the SHA-256.
- **A key is tied to the Tailscale login that paired it**, when Serve supplies one.
- **Revoking is from the Mac only** (`lag phone revoke`, or `DELETE /api/devices/{name}` on
  the console port). It takes effect on the phone's next request.
- **Chats run in the sandbox**: `/api/run` and `/api/stream` wrap the tools in one shared
  container (workspace `agents/workspaces/phone`), started on the first phone chat,
  checked once a minute, stopped at shutdown. No Docker means no phone chat (fails closed).
- **Refused requests are logged** to `phone/refused.log`, never with the key.
- The Host check is skipped there (the key is the lock); the cross-site Origin check stays.

Verified live 2026-11-08: pair over curl, key accepted, code single-use, a phone chat's
`uname -s` answered `Linux` from the container, revoke cut the key off at once, and Ctrl-C
stopped both listeners and removed the containers.

### API contract (`contract/`, `docs/api-contract.md`)

What `lag serve` promises clients, in words (`docs/api-contract.md`) and as data
(`contract/v1.json`): 26 event types with their fields, the goal record, the endpoints the
phone app calls, and the error rule (always an `error` string; 4xx for refusals, 200 for
`/api/run` and goal edits, a final `error` event in a stream). `GET /api/version` returns
`{lag, contract}`. `contract/examples/` holds streams recorded through the phone port of a
real server (`contract/record.py`), scrubbed of the home path, the key and the model list.
`tests/test_contract.py` fails if the code sends an event the contract lacks, the contract
lists one nothing sends, an example drifts from the contract, an endpoint's shape changes,
or the doc stops naming something.

### Study mode

`lag study add | ask | quiz | cards | review | status`. Each course is its own Chroma
collection; `ask` answers with citations from your notes, `quiz` grades free-text answers
strictly against the source. PDFs need the `[study]` extra (`pypdf`).

---

## Context window and model residency

Ollama **ignores a model's trained context** and allocates 4096 by default, silently
dropping the oldest tokens. `ollama.num_ctx` and `ollama.num_ctx_by_model` in
`config.yaml` are applied via `OllamaClient.from_config`.

Measured 2026-09-07 on the M4 (weights + KV):

| num_ctx | `ornith:latest` | `qwen3:8b` | note |
|--------:|----------------:|-----------:|------|
| 8192 | 5.48 GB | 6.18 GB | **qwen3:8b runs here** |
| 16384 | 6.35 GB | 7.41 GB | |
| 32768 | 6.40 GB | 9.91 GB | **ornith runs here** |
| 40960 | — | 11.16 GB | qwen3:8b's trained max |
| 65536 | 7.64 GB | — | |
| 131072 | 10.12 GB | — | tried and dropped |

**num_ctx is a memory dial, not a speed dial.** Across ornith at 8192 / 32768 / 131072,
load was 8.40s / 5.38s / 6.39s and generation 13.6 / 15.4 / 15.3 tok/s — flat. 131072 was
dropped because `Pipeline._brief` is bounded to fit the 8192 stage that reads it, so the
extra width bought nothing.

**Thinking is what costs time.** A one-stage run answering `1 2 3` emitted 140 thinking
tokens against 5 answer tokens. Persona `thinking_mode` is wired to `ollama.chat(think=…)`,
and `num_predict: 2048` caps a turn (thinking included) so a spiralling stage fails fast.

### One model at a time

Ollama here holds **one** chat model resident, and the limit is free system RAM. The old
note here called it "a hard cap, not a memory limit"; that was wrong, and it was measured
while two Ollama servers ran (see Environment traps). Re-measured 2026-10-31 on the single
app server (0.40.11), 3 rounds each way at the configured num_ctx:

| | `ornith:latest` @32768 | `qwen3:8b` @8192 |
|---|---:|---:|
| cold, nothing loaded (wall) | 7.0–8.2s (load 6.4–7.5) | 5.1–5.3s (load 4.9–5.1) |
| switching in, evicting the other | 6.1–6.6s (load 5.4–5.9) | 3.8–5.1s (load 3.6–4.9) |
| resident size | 6.42 GB | 6.19 GB |

Both stayed resident in **0 of 6** trials. Ollama's log gives the reason each time:
`predicted to exceed available memory … system_limited=true`. About 9.2 GB is free with
nothing loaded (macOS plus apps hold ~7 GB); after one model, 2.6–4.2 GB is left, while
GPU memory still shows 5.9–7.7 GB. Even a 2B model needing 3.2 GB was refused with 3.8 GB
free, so Ollama keeps a margin. More RAM, or fewer apps open, would fit two; on this box
it is one. A full A→B switch therefore costs one model's cold load, about 4–7s. That drove three decisions: `parallel` mode
deleted (agents evicting each other), `crew` mode deleted (an extra reload for the
coordinator's return trip), and every downstream role shares one tag.

---

## Models (Mac M4 — primary)

Personas name a **role**; `config.yaml → models:` is the only place tags live.

| Role | Tag | num_ctx | Personas |
|------|-----|--------:|----------|
| `orchestrator` | `ornith:latest` | 32768 | architect, senior_dev |
| `worker` | `qwen3:8b` | 8192 | analyst, biz_analyst, design, research |
| `coder` | `qwen3:8b` | 8192 | dev, tester, dev_research |
| `fast` | `qwen3:8b` | 8192 | unassigned |
| `embed` | `nomic-embed-text` | — | retrieval only (768-dim) |

The default roster (architect → dev → senior_dev) pays three loads; move `senior_dev_agent`
to `coder` for two. Tool support is **discovered** per tag via Ollama, not hardcoded; a
completion-only model runs without tools and emits a `no_tools` event. Map roles to
whatever `ollama list` shows on your machine; the console's model picker reads the same
list live from Ollama.

---

## Setup and commands (Mac)

1. `git clone git@github.com:Hinzmo/lag-kit.git && cd lag-kit`
2. `./install.sh` — builds `.venv` on **Python 3.12** (macOS 3.14 has no `chromadb` wheel;
   it offers `brew install python@3.12` if missing), installs `.[rag,web,study]`, then runs
   `lag setup`: installs Ollama if missing (asks first; brew on macOS, the official script
   on Linux), starts the server, pulls `qwen3.5:9b` (every chat role) and `nomic-embed-text`,
   writes `config.yaml` with comments intact, and indexes `docs/` if `.chroma` is absent.
   Precedence is flag → config.yaml → defaults, so a bare re-run never remaps anything.
   Switch models with `lag setup --model <tag>` / `--orchestrator <tag>`; `lag models`
   lists installed tags, their capabilities and which roles use them.
3. For development add the test extras: `.venv/bin/python -m pip install -e ".[dev]"`.
4. Start **Docker Desktop** if you want agent mode (and the 2 sandbox tests).
5. `.venv/bin/python -m pytest` — expect 609 passing with Docker running; without it, 4 real-container tests skip.
   `cd tests/js && npm install` enables the console boot test.

```bash
.venv/bin/lag run "what is this project?"
.venv/bin/lag run "same thing" --mcp
.venv/bin/lag pipeline "design a url shortener"
.venv/bin/lag serve                      # console at http://127.0.0.1
.venv/bin/lag agent new                  # then: list | start | status | stop
.venv/bin/lag study add <course> <path>
.venv/bin/python servers/tools_server.py # standalone, real tracebacks
```

Run everything from the repo root — `.chroma`, `agents/`, `study/` and `memory/` resolve
relative to cwd. All of them are gitignored and per-machine; remembered facts do not sync
between boxes, by decision.

### Dependencies

`pyproject.toml` is the single source of truth — there is no `requirements.txt`.

| Extra | Packages | Needed for |
|-------|----------|------------|
| core | `ollama`, `mcp`, `pyyaml`, `typer` | `lag run`, `pipeline`, `agent` |
| `rag` | `chromadb` | retrieval, memory indexing |
| `web` | `fastapi`, `uvicorn` (pulls `pydantic`) | `lag serve`, `tests/test_web.py` |
| `study` | `pypdf` | PDF material in `lag study` |
| `dev` | `pytest`, `httpx`, `ruff` | tests, lint |

---

## Fixed along the way

Every one of these was found by running the code, not reading it.

1. **`--mcp` hung** — server spawned as bare `"python"`; now `sys.executable`.
2. **`mcp 2.0.0` renamed `FastMCP`** → `MCPServer`, and **`Tool.inputSchema`** →
   `input_schema`. Both spellings handled.
3. **`chunk_text` never terminated** on its final chunk; every ingest would have hung.
4. **RAG server printed to stdout**, corrupting the stdio JSON-RPC channel.
5. **`.gitignore` inline comments** made `config.yaml` and `.chroma/` patterns match nothing.
6. **Chunk IDs collided** (`chunk_0..N`); now content-addressed and namespaced.
7. **`num_ctx` was never set**, running every model at 4096.
8. **UI full-screen flicker** from a keyframe overlay + blend mode over a 60fps canvas.
9. **`MCPBridge` leaked** server subprocesses and threads — *this entry used to claim a
   fix that hadn't happened; see #19.* (The console caches its provider rather than
   closing one per run, and `close()` itself only worked by accident.)
10. **Console failed to boot** on a dead `MODELS` reference — now covered by a jsdom boot test.
11. **Dead-code cleanup deleted live chat plumbing** (send path, `Tps`/`loadModel`) —
    restored, with a test that drives a full turn.
12. **`max_cycles` backstop was off**, and the deny-list's `| sh` rule fired on `| sha256sum`
    — both corrected.
13. **Agent-run output got committed** — a URL-shortener demo left `main.py` and
    `shortener/` in the repo root. Removed; agents now write only their own workspace.
14. **The deny-list matched substrings** — #12 fixed one rule, the rest kept the flaw:
    `rm -rf /workspace/build` contained `rm -rf /`, `grep shutdown log.txt` contained
    `shutdown`, while `rm -r -f /` slipped past. It now parses the command (`shlex`) and
    judges the program and its arguments; quoted text and heredoc bodies are data.
15. **Agents couldn't write a file the project also had** — a relative write fell back to
    the read-only project and was refused forever. Writes now always land in the
    workspace; reads still fall back (copy-on-write).
16. **A resumed agent past its cap never ran again** — `max_cycles` was compared to the
    lifetime cycle count, so every restart stopped before cycle 1, with no reason given.
    The cap is now per run; resumes also reset the breakers and the last stop reason.
17. **The console's agent form couldn't save** — its default `0` cycles was sent as `null`
    (422, shown as "saved as undefined"), and since saving *replaced* the definition,
    every field the form lacks reset on re-save: network back on, a disabled agent
    re-enabled, description and workspace wiped. Saves now merge; the form has a
    network checkbox; refusals say why. `tests/js/agent_form.js` drives the real form.
18. **Two agents could run at once** — the web server checked `_RUNNING` in the request
    but added to it in the worker thread, so two quick starts both passed, and the CLI
    was invisible to it entirely. Now one agent per machine: an `flock` in
    `AgentLoop.run` (cross-process, released by the kernel on any exit), a claim under a
    lock in the server (409), and greyed-out start controls in the console. Also fixed:
    Cmd+Enter both saved and started, and Enter on a focused button started an agent.
19. **MCP teardown, for real** — `close()` exited the SDK's context managers from a
    different task than entered them; anyio raised on both, `close()` swallowed it, and
    the server died only because the SDK killed it first. Nothing called `close()`
    anyway. The live leak was the failure path: when a later server failed to start,
    `from_config` orphaned the earlier one and its thread — measured 1, 2, 3 orphans over
    three retries, and the console retries per request. Now the session lives in one
    task, `from_config` is all-or-nothing, the CLI and web server close what they build,
    and errors arrive unwrapped from anyio's `ExceptionGroup`.

---

## What's left

- **Going public:** license and path cleanup are merged (#9); README refresh (#10),
  `lag setup` (#11) and `lag goals` are in review. Left: a history scrub (junk files,
  commit email, old model names), then flip the repo public.
- **Chat latency on MLX:** prompt-cache misses on the tool-schema prefix (see above).
- **The done-when judge is the working model judging its own answer.** Cheap (no reload)
  but lenient by nature; a second model or a stricter rubric would be stronger.
- **Supervised `run_shell` is unsandboxed** — `shell=True`, no path allow-list, 60s timeout.
  Accepted on a single-user box; only agent mode is confined. Revisit before anything
  multi-user.
- **cwd dependence.** `.chroma`, `agents/`, `study/` and `memory/` resolve relative to cwd,
  so running `lag` outside the repo root points at empty state. Fix: resolve against the
  repo root.
- **Two tools leave the machine**: `web_search` (DuckDuckGo HTML scrape — will break if
  their markup changes; parser pinned by captured-page tests) and `http_fetch` (SSRF-guarded,
  default-deny: only public addresses, embedded IPv4 unwrapped, every redirect re-checked;
  DNS rebinding remains a documented gap). Both keyless and
  stdlib-only.
- **Agent shells reach the public internet by default** (through the egress proxy), so
  download-and-run from the internet is possible by design. Seal an agent with
  `network: false`. Outside names do not resolve inside the container at all (measured on
  the internal network), so DNS is not an exit either; only the proxy resolves.
- **Watch `mcp` upgrades.** Venv is on mcp 2.3.0 (upgraded 2026-10-29, suite green); the SDK
  has renamed APIs the kit uses before, so re-run the suite after any bump.

---

## Environment traps (Mac)

- **Two Ollama servers on one port.** The Ollama app listens on every address and a
  Homebrew `ollama serve` on 127.0.0.1, both on 11434. `localhost` from Python reached
  the app while the `ollama` CLI reached brew, and each keeps its own model list in
  memory, so a pull through one was invisible to the other (and each loaded its own
  copy of `nomic-embed-text`). Fixed on the M4 2026-10-31 with `brew services stop
  ollama`; the kit now defaults to `127.0.0.1` (the CLI's address), and `lag setup` /
  `lag models` warn when the two addresses disagree. Residency was re-measured on the single
  server afterwards: still one chat model, and the cause is free RAM.

## Environment traps (Ubuntu box only)

- **`PIP_PREFIX=~/snap/claude-code/common`** is exported globally by snap and
  redirects every `pip install` out of any venv; `PYTHONPATH` shadows the venv. Prefix
  commands with `env -u PIP_PREFIX -u PYTHONPATH -u PIP_BREAK_SYSTEM_PACKAGES`.
- `python3-venv` isn't installed; bootstrap with `--without-pip` + `get-pip.py`.
- `ssh`, `~/.ssh`, `ollama`, `curl` and `unzip` are unreachable from the snap. Use `gh`, the
  Ollama HTTP API, and Python's stdlib.
