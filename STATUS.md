# local-agent-kit v2 — status

Last updated **2026-10-29**. This is the living status doc: what works, how it works, and
what's left. `MORNING_BRIEF*.md` and `REBUILD_COMPLETE.md` are historical — trust this file
for current state. Working rules for Claude Code live in `CLAUDE.md`.

---

## Where it stands

The six original phases are done, and the kit has grown well past them. **334 tests pass,
2 skip** (the skips need Docker Desktop running). Primary box is the **MacBook Air M4**;
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
| Agent mode — saved agents that loop on a goal | `lag agent …` | ✅ |
| Container sandbox for agents (mandatory) | — | ✅ needs Docker running |
| Study mode — ask/quiz over your own course notes | `lag study …` | ✅ |
| Readable output (`render.py`) | — | ✅ |
| SSRF guard on `http_fetch` | — | ✅ |

**In flight:** branch `feat/sandboxed-agents`, open as **PR #7** — container sandbox,
public-only egress proxy for agents, SSRF guard, console layout, and these doc refreshes.

Repo: `github.com/Hinzmo/local-agent-kit` (private), default branch `main`.

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

Ollama here holds exactly **one** chat model resident — a hard cap, not a memory limit (a
second model evicted the first at ~5.9 GB of 16 GB). Every switch is a cold reload
(`ornith` 5.7–6.4s, `qwen3:8b` 4.4–4.9s). That drove three decisions: `parallel` mode
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
completion-only model runs without tools and emits a `no_tools` event. Installed but not
mapped: `example/completion-only-8b`, `example/completion-only-mini`
(completion-only), `persona-model:latest` (baked-in persona), `example/llama-3.1-8b-…`
(rough IQ3_M quant), plus `gemma-4-e2b-aiops`, `qwen3.5:4b` and an `ornith:9b`
alias of `ornith:latest`.

---

## Setup and commands (Mac)

1. `git clone git@github.com:Hinzmo/local-agent-kit.git && cd local-agent-kit`
2. `/opt/homebrew/bin/python3.12 -m venv .venv` — **3.12, not the default `python3`**
   (macOS 3.14 has no `chromadb` wheel). `brew install python@3.12` if missing.
3. `.venv/bin/python -m pip install -e ".[dev,rag,web,study]"`
4. `cp config.example.yaml config.yaml`, then map the five roles to tags from `ollama list`.
5. `ollama pull nomic-embed-text`, then `.venv/bin/python servers/rag_server.py --ingest docs/`
6. Start **Docker Desktop** if you want agent mode (and the 2 sandbox tests).
7. `.venv/bin/python -m pytest` — expect 431 passing with Docker running; without it, 4 real-container tests skip.
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

- **Merge PR #7** (`feat/sandboxed-agents`).
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

## Environment traps (Ubuntu box only)

- **`PIP_PREFIX=/home/hinzmo/snap/claude-code/common`** is exported globally by snap and
  redirects every `pip install` out of any venv; `PYTHONPATH` shadows the venv. Prefix
  commands with `env -u PIP_PREFIX -u PYTHONPATH -u PIP_BREAK_SYSTEM_PACKAGES`.
- `python3-venv` isn't installed; bootstrap with `--without-pip` + `get-pip.py`.
- `ssh`, `~/.ssh`, `ollama`, `curl` and `unzip` are unreachable from the snap. Use `gh`, the
  Ollama HTTP API, and Python's stdlib.
