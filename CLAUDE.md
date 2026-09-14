# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`local-agent-kit` (`lag`) — a fully offline multi-agent harness over [Ollama](https://ollama.com).
No cloud, no API keys. Agents call tools exposed over the Model Context Protocol (MCP);
retrieval (Phase 6) is just one more MCP server.

This is the **v2 rebuild**. v1 faked the tool layer with a direct-dispatch HTTP/JSON-RPC
shim; v2's whole point is a *real* MCP boundary — tools live in separate server processes,
discovered at runtime via `tools/list`, reusable by any MCP client.

## Repo state

Recovered, committed, pushed, and **actually running**. All six phases are smoke-tested end
to end, including RAG.

**Primary dev box is now the MacBook Air M4** (ported here 2026-08-21 for the better specs).
The CPU-only Ubuntu machine it was built on is the secondary box; its snap-confinement traps
are real but Ubuntu-only, quarantined in the appendix at the bottom of this file. On the Mac:
231 tests pass, and `lag run` tool-calls correctly through both the direct and `--mcp` paths.

```
src/lag/     config.py  tools.py  ollama_client.py  mcp_client.py  pipeline.py  agent.py  cli.py  web.py
servers/     tools_server.py  rag_server.py
personas/    nine YAMLs
tests/       231 tests, all passing
docs/        RAG_SETUP.md  VECTORS_AND_SEARCH.md
```

Remote: `github.com/Hinzmo/local-agent-kit` (private). Branch `main`.

### History worth knowing

The tree was rebuilt after the previous machine was wiped, initially from a partial
`files.zip` (2026-06-16). A **complete** archive later turned up at
`~/Downloads/local-agent-kit-v2-complete.zip` (2026-06-18) holding the nine personas,
`rag_server.py`, and `docs/` — so nothing was lost after all. All shared source was
byte-identical between the two; the newer archive was a strict superset.

Both `MORNING_BRIEF.md` and `REBUILD_COMPLETE.md` are **June 2026 documents**: reliable on
design intent and deliberate trade-offs, stale on status. `MORNING_BRIEF.md` §5 claims the
`lag` entry point "isn't registering cleanly" — that was the broken venv below, not a code
defect. It registers fine.

### Environment (Mac — the primary box)

- **Build the venv on Python 3.12.** macOS ships **3.14** as `python3`, and `chromadb` (the
  `[rag]` extra) has no 3.14 wheels yet, so an install on 3.14 fails at the C-extension build.
  Use Homebrew's 3.12 explicitly:

  ```bash
  /opt/homebrew/bin/python3.12 -m venv .venv
  .venv/bin/python -m pip install -e ".[dev,rag]"
  ```

- **No `env -u` dance needed** — the `PIP_PREFIX` / `PYTHONPATH` / ssh traps below are
  snap-confinement artifacts specific to the Ubuntu box. On the Mac, `.venv/bin/lag`,
  `.venv/bin/python -m pytest`, `ollama`, and `ssh` all just work.
- `gh` is authenticated over HTTPS here and `ssh` reaches GitHub; either works.

### Models (Mac)

`config.yaml` (gitignored) is mapped to what is pulled on the Mac:

| Role | Tag | Personas on it |
|------|-----|----------------|
| `orchestrator` | `ornith:latest` | architect, senior_dev |
| `worker` | `qwen3:8b` | analyst, biz_analyst, design, research |
| `coder` | `qwen3:8b` | dev, tester, dev_research |
| `fast` | `qwen3:8b` | unassigned |
| `embed` | `nomic-embed-text` | retrieval only (768-dim) |

Remapped 2026-09-07 after an Ollama cleanup removed `qwen3:14b`, `qwen2.5-coder:7b`,
`llama3.2:3b` and `qwen3:4b`. Two rules drove the new shape:

- **Role models should support tools.** The loop is a tool-calling loop. This is no
  longer fatal — `ollama_client.capabilities()` asks Ollama what a tag can do, and a
  completion-only model runs *without* tools and says so (a `no_tools` event) rather
  than failing the request. But it is a real downgrade: no file access, shell,
  retrieval or memory, so replies come only from what the model already knows. On this
  box `example/completion-only-8b` and `example/completion-only-mini` are
  completion-only; `persona-model:latest` does support tools but ships a baked-in persona
  that fights the persona system prompts.

  Capability is **discovered, never hardcoded** — a newly pulled model is classified
  correctly with no list to maintain. A failed probe returns an empty set, which
  callers must read as *capable*: treating unknown as incapable would silently strip a
  working model of every tool on one flaky status call. `thinking` is resolved the same
  way, so a non-thinking tag is never asked to think.
- **A role switch costs a full model reload** (~5s, measured), because only one chat
  model stays resident. So every downstream role points at ONE tag: consecutive
  pipeline stages then reuse the resident weights for free, and `ornith` — 9B qwen3.5,
  tools + thinking, 262K trained window — spends its single load on the planning stage.

Note `senior_dev` sits on `orchestrator`, so the default three-stage roster runs
ornith → qwen3:8b → ornith and pays three loads. That is a deliberate trade (review is
the hardest reasoning); move `senior_dev_agent` to `coder` in its persona if you would
rather have the two-load run. Personas name roles, never tags, so this is all one
`config.yaml` — no persona edits.

`nomic-embed-text` is pulled, and `.chroma/` holds an ingest of `docs/` (9 chunks). Memory
and `.chroma/` are gitignored and per-machine, so the Mac grew its own from a fresh ingest.

## Architecture

Three decoupled layers; the harness is the only piece that knows about the others.

```
Ollama (local LLM)  ──tools ↓ / tool_calls ↑──  Harness (lag)  ──stdio · JSON-RPC──  MCP servers
                                                ReAct loop                            tools_server.py
                                                + MCP client                          rag_server.py (Phase 6)
```

### The central abstraction: `tool_provider`

Everything hinges on one duck-typed interface. `OllamaClient.run` accepts any object with:

```python
schemas() -> list       # whatever ollama.chat accepts as `tools`
call(name, args) -> str
```

Two implementations satisfy it, and the loop cannot tell them apart:

- `LocalToolProvider` (`tools.py`) — dispatches straight to in-process Python functions.
  `schemas()` returns the *function objects themselves*; Ollama derives JSON schemas from
  type hints + docstrings.
- `MCPToolProvider` (`mcp_client.py`) — routes through a real MCP server subprocess.
  `schemas()` translates MCP's `{name, description, inputSchema}` into Ollama's
  `{type: "function", function: {name, description, parameters}}` shape (`inputSchema` →
  `parameters`).

So `lag run "..."` and `lag run "..." --mcp` exercise the same tools through different
transports with no change to the agent loop. **Preserve this property.** If you find
yourself adding an `if mcp:` branch inside `OllamaClient`, that's the wrong layer.

### One source of truth for tools

The seven tool functions in `tools.py` (`read_file`, `write_file`, `list_dir`, `run_shell`,
`system_info`, `remember`, `recall`) are consumed *unchanged* by both Ollama's schema
generation and FastMCP's (`tools_server.py` just loops `mcp.tool()(_fn)` over them). Do not
duplicate tool logic into the server. When adding a tool: write the plain function in
`tools.py` with a typed signature and an Args-style docstring, add it to the `TOOLS` dict,
and add it to the tuple in `tools_server.py`.

Tool errors are returned as strings to the model, never raised — see `LocalToolProvider.call`.
Keep that behavior; a raised exception kills the loop instead of letting the model recover.

### The async/sync bridge

`MCPBridge` (`mcp_client.py`) reconciles the async MCP SDK with the sync agent loop: one
persistent asyncio loop on a daemon thread, calls marshalled over with
`run_coroutine_threadsafe`. The context managers (`stdio_client`, `ClientSession`) are
entered *manually* via `__aenter__` so the session and server subprocess outlive `_connect`.

This is the known rough edge, and the historical "`--mcp` hangs" symptom traced to it — but
the cause was *not* the manual `__aenter__`. The server was spawned with a bare `"python"`,
which does not exist on this box; it died on import and `stdio_client` waited forever on a
pipe that never spoke. Now fixed via `sys.executable` (see `MCPToolProvider.from_config`).

Two residual rough edges remain, both real:

- `MCPBridge` never shuts down. There's no `close()`, so the server subprocess and the daemon
  event loop leak for the process lifetime. Fine for one-shot CLI runs; a problem the moment
  something long-lived holds a bridge. The cleaner refactor is still keeping the whole run
  inside `async with` blocks.
- If the server dies at startup, the failure surfaces as `MCPError: Connection closed` with no
  hint of the child's stderr. When debugging `--mcp`, **run the server standalone first** —
  `env -u PYTHONPATH .venv/bin/python servers/tools_server.py` — where the real traceback shows.

## Commands

Mac (primary). No `env -u` prefixes here — those are Ubuntu-only, see the appendix.

```bash
.venv/bin/python -m pip install -e ".[dev,rag]"    # core + pytest + chromadb

.venv/bin/lag run "list the files here and tell me what this project is"
.venv/bin/lag run "same thing" --mcp               # routes through the MCP server
.venv/bin/lag pipeline "design a tiny URL shortener"
.venv/bin/lag pipeline "...use the knowledge base..." --mcp   # stages get the MCP toolset
.venv/bin/lag pipeline "..." --agents "architect_agent,dev_agent" --no-plan

.venv/bin/python servers/tools_server.py           # standalone; shows real tracebacks
.venv/bin/python servers/rag_server.py --ingest docs/   # populate .chroma
.venv/bin/python -m pytest                         # 231 passing

ollama pull nomic-embed-text                       # required before Phase 6
```

`cp config.example.yaml config.yaml` when starting on a new machine, then map the five roles
to models you have actually pulled (`ollama list` for exact tags).

On the M4, a single tool-calling round trip is ~30–40s including model load; a four-call
`crew` run takes longer. Background long runs rather than blocking on them. (The CPU-only
ASUS was several times slower — ~1–3 min per round trip.)

## Config

`config.py` searches `config.yaml` then falls back to `config.example.yaml`, so the kit runs
before customization. Models are per-machine (`models.orchestrator|worker|coder|fast|embed`):
the MacBook Air M4 runs `ornith:latest` as orchestrator with everything downstream on
`qwen3:8b`; the CPU-only ASUS should stay small. `ollama.num_ctx_by_model` overrides the
window per tag, falling back to `ollama.num_ctx`. **Avoid R1 distills as orchestrators** —
weak structured output, and thinking tokens destroy loop latency.

## Live streaming and per-agent tasks

`OllamaClient.run` and `Pipeline.run` take an optional `on_event` callback. It is a **side
channel**: the return value is identical with or without it, so the CLI is untouched and the
blocking `/api/run` and the streaming `/api/stream` cannot disagree about the same run. Event
types: `start`, `stage` (with `model` and a `loaded` flag), `thinking`, `token`, `tool`,
`tool_result`, `done`, `error`.

Streaming is opt-in rather than the default because reconstructing `tool_calls` from chunks
is strictly more fragile than reading them off a finished message — `_turn` streams only when
someone is listening. Thinking deltas are a separate event type on purpose: on ornith they are
most of the wait, and a console showing nothing during them looks hung.

`/api/stream` is a POST returning `text/event-stream` (not an EventSource GET, so the prompt
travels in the body rather than the URL). The synchronous loop runs on a daemon thread and
pushes events into a queue the response generator drains.

**Handing one agent a task is not a special case** — it is a one-stage pipeline with
`plan=False`. Clicking a unit in the console roster opens an inline input and posts exactly
that, so there is no separate endpoint or code path to keep in sync.

## Agent mode (`agent.py`)

The pipeline is the *bounded* shape: fixed stages, one pass, done. Agent mode is the
unbounded one — a goal, the full toolset, and a loop that runs until the goal is met or
you stop it. `lag agent start` walks a wizard for the goal (or takes `--goal`),
`lag agent status` reads the run, `lag agent stop` ends it.

Three properties make it safe to leave running, and none are visible from one good run:

- **It can stop.** `mark_done` is the termination condition — without it an autonomous
  loop burns tokens until a step cap and calls that success. `--max-cycles` and the stop
  file (`agent/<name>.stop`) are the backstops. The idle wait polls that file every second
  rather than sleeping the whole interval: an agent you cannot stop promptly is not one
  you leave on.
- **It survives its own failures.** Every cycle is wrapped; an exception becomes an
  observation the *next* cycle reads, then the loop backs off and continues. Same rule as
  the tool layer — errors come back as text the model can work with, never raised.
- **It remembers.** State is written to `agent/<name>.json` after every cycle, so status
  and a restart see the same record. A run resumes if the goal matches; a different goal
  starts fresh.

Cycles share no conversation memory — each is a fresh turn — so the goal, the progress
log and the last observation *are* the continuity. That is why `GoalSpec.prompt` restates
everything every cycle, and why `note_progress` matters: it is how a cycle talks to its
successor.

**Autonomy is unrestricted, by explicit decision.** Agent mode gets the same unsandboxed
shell and filesystem the supervised CLI has. See the `run_shell` loose end above.

## Project pipeline (`pipeline.py`)

One sequential shape, replacing the old three-mode `Swarm` (removed 2026-09-07). Stages
run in the order given, each seeing the full context accumulated above it; the lead
plans once at the top and its per-stage instructions ride that context down the line.
`Pipeline.loads()` reports the cold-load cost of an ordering, and `run` prints each
stage as LOAD or resident.

Why the other two modes went, and why not to reintroduce them:

- **`parallel` was actively harmful here.** It fired every agent at once through a
  `ThreadPoolExecutor`, but only one chat model is resident, so concurrent agents on
  different tags evict each other in a loop — slower than running them in sequence. It
  is only safe when every agent shares one tag, which is a pipeline with extra steps.
- **`crew` forced an avoidable reload.** Its coordinator bookended the run (plan →
  delegate → come back to synthesize), and that return trip cost a second cold load.
  The pipeline gets the same result by having the lead front-load its instructions and
  letting the final stage answer on whatever model is already resident.

## Phase 6 — RAG (`servers/rag_server.py`)

Verified working: `--ingest` chunks and embeds via `nomic-embed-text`, and
`search_knowledge_base` answers over the real MCP boundary.

```bash
.venv/bin/python servers/rag_server.py --ingest docs/
.venv/bin/python servers/rag_server.py     # server mode
```

Two traps already hit and fixed here, worth remembering if you write another server:

- **`chunk_text` had a non-terminating loop.** On the final chunk `start` rewound to
  `end - overlap`, still `< len(text)`, so it re-emitted the same tail until memory ran out —
  every ingest hung. Now strides forward and breaks at the end.
- **Server mode printed to stdout.** Under the stdio transport stdout *is* the JSON-RPC
  channel, so any stray `print` corrupts the stream and `initialize` fails. Diagnostics go to
  stderr. This applies to every MCP server in this repo.

**Wired into the harness.** `MCPToolProvider.from_config` spawns both `tools_server.py` and
`rag_server.py`; the provider holds a list of bridges, merges their schemas into one flat
toolset (first server wins a name collision), and routes each `call()` to the owning bridge.
The agent loop was untouched — `--mcp` now exposes all nine tools. One caveat: run `lag` from
the repo root, since the RAG server resolves its store as `./.chroma` relative to cwd.

> **`STATUS.md` is the living status doc** — current state, how it works, what's left, and Mac
> setup. Update it when behaviour changes; this file stays focused on working rules.

## Memory and retrieval

`remember(key, value)` writes to `memory/store.json` **and** embeds into the ChromaDB
collection, so stored facts return from a plain question rather than only an exact `recall`.
Indexing is the default, never a flag — a fact only `recall` can reach is one the model must
already know it stored. Remembered chunks carry `kind: "memory"` and surface tagged
`· remembered`.

`src/lag/rag.py` owns chunking, embedding, and the collection. `servers/rag_server.py` is only
the protocol wrapper and CLI — the same "one implementation, two consumers" rule as `tools.py`,
because `tools.remember` is the second consumer. Don't reintroduce retrieval logic into the
server.

Indexing must stay non-fatal: `tools._index_memory` imports `lag.rag` lazily and swallows
everything, so a core install without the `[rag]` extra still remembers.

Chunk ids are content-addressed and namespaced per writer (`chunk_*` vs `mem_*`). Never go
back to positional ids — `chunk_<i>` made a second ingest overwrite the first.

## Model roles

`models:` in `config.yaml` is the **only** place model tags live. Personas name a role
(`model: coder`) which `config.resolve_model` maps to a tag; a literal tag still passes
through. Never hardcode a tag back into a persona — that reintroduces the drift that left all
nine pinned to models the machine didn't have. Roles: `orchestrator` (architect, senior_dev),
`coder` (dev, tester, dev_research), `worker` (the other four), plus unassigned `fast` and
retrieval-only `embed`.

## Context window

Ollama ignores a model's trained context and allocates **4096** by default. The Mac's models
are trained for far more (`ornith` 262144, `qwen3:8b` 40960), so leaving `num_ctx` unset runs
them at a fraction of their window and silently drops the oldest tokens. Always construct
clients via `OllamaClient.from_config` so the window and `keep_alive` are applied;
constructing `OllamaClient(...)` directly silently reverts to 4096 and the model starts
dropping the top of its own conversation with no error.

num_ctx is a memory dial, not a speed dial (measured: gen speed ~19 tok/s and load time stay
flat as it grows; only the footprint does). *Filling* a window is the part that costs — prefill
tracks tokens actually used.

Because only one chat model is resident at a time, each is sized on its own terms via
`ollama.num_ctx_by_model` rather than being pinned to the smallest one's ceiling. Measured on
the 16 GB M4 (weights + KV):

| num_ctx | `ornith:latest` | `qwen3:8b` |
|---------|-----------------|------------|
| 8192 | 5.48 GB | 6.18 GB |
| 16384 | 6.35 GB | 7.41 GB |
| 32768 | 6.40 GB | 9.91 GB |
| 40960 | — | 11.16 GB (its trained max) |
| 131072 | 10.12 GB | — |

So `ornith` holds 16× the window for less memory than `qwen3:8b` spends on 32768.

**The window is not a speed dial in either direction.** Measured across ornith at
8192 / 32768 / 131072: load 8.40s / 5.38s / 6.39s, generation 13.6 / 15.4 / 15.3 tok/s.
Flat. Dialling num_ctx *down* does not buy throughput — it buys memory back, nothing more.

`ornith` therefore sits at **32768** (~6.4 GB) and `qwen3:8b` at **8192**. 131072 was tried
and dropped: it cost 10.12 GB for headroom the pipeline cannot use anyway, because
`Pipeline._brief` is deliberately bounded to fit the 8192 stage that reads it. A plan written
in a wide window and read by a narrow stage is truncated **silently**, not rejected — that
bound is the mitigation, and it also caps the value of a wide lead window.

If a run feels slow, num_ctx is the wrong knob. `ornith` has **thinking** enabled natively
and it dominates: a streamed one-stage run producing `1 2 3` emitted **140 thinking tokens
against 5 answer tokens**, and a two-stage run 502 against 187. See STATUS.md "One model at
a time".

## Known loose ends

These are deliberate, documented in `files/MORNING_BRIEF.md` §4 — not bugs to chase blindly:

- **Two tools now leave the machine**, not one. `web_search` (DuckDuckGo HTML endpoint,
  keyless, stdlib-only) and, as of agent mode, **`http_fetch`** — a GET of a URL the model
  chose, stripped to text. This is a deliberate widening of the original stance, decided
  when agent mode was added: search returned only snippets, so an agent could find a page
  and never read it. Both stay keyless and stdlib-only, so no dependency and no account.
  `web_search` scrapes markup and will break if DuckDuckGo changes their HTML; the parser is
  pinned by tests against a captured page, never the live network. Don't add an API-key
  search provider without asking; keyless is what keeps the rest of the stance intact.
- **`run_shell` is unsandboxed** — `shell=True`, no path allow-list, 60s timeout. Accepted
  risk on a single-user offline box. Revisit before anything multi-user. Don't silently
  "fix" this without asking; it's a deferred decision, not an oversight. **Agent mode
  inherits this unchanged, by explicit decision** — an unattended loop runs the same
  unsandboxed shell a supervised `lag run` does. The brakes are the cycle cap and the stop
  file, not a sandbox. If that ever needs to change, it is a real design task (workspace
  confinement + a deny-list), not a patch.
- ~~`thinking_mode` in personas is metadata only~~ — **wired.** It reaches
  `ollama.chat(think=...)` through `Agent`/`from_config`, with a one-shot retry without
  `think` for tags that reject the parameter. This mattered more than it looked: unwired,
  `dev_agent` (`thinking_mode: false`) emitted 140 thinking tokens to answer "1 2 3".
- **`mcp` SDK version churn — this landed.** The installed SDK is **mcp 2.0.0**, which renamed
  two things the kit depended on:

  | mcp 1.x | mcp 2.x |
  |---|---|
  | `mcp.server.fastmcp.FastMCP` | `mcp.server.mcpserver.MCPServer` |
  | `Tool.inputSchema` | `Tool.input_schema` |

  Both servers import either via a `try/except ImportError`, and `mcp_client._input_schema`
  accepts both spellings, so the kit runs on 1.x and 2.x. `.tool()` and `.run()` ergonomics
  are unchanged. Any tutorial you copy from is probably still on 1.x — check `pip show mcp`.

## Conventions

- `from __future__ import annotations` at the top of every module; PEP 604 unions (`str | None`).
- Module docstrings explain *why the design is this way*, not what the code does. Match that.
- Tool functions need Google-style `Args:` docstrings — they become the schema descriptions
  the model reads.
- Real config (`config.yaml`), `memory/`, `data/`, and `.chroma/` are gitignored. Commit
  `config.example.yaml` only. **Never put a trailing `# comment` on a `.gitignore` pattern** —
  git has no inline comments, so the comment becomes part of the pattern and the rule matches
  nothing. `config.yaml` and `.chroma/` were both silently untracked-in-name-only this way;
  verify any new rule with `git check-ignore -v <path>`.
- Privacy stance is load-bearing here ("your data never leaves the machine"). Don't introduce
  cloud calls, telemetry, or API-key-requiring dependencies.

## Appendix — Ubuntu box (secondary)

The kit was built on a CPU-only Ubuntu machine running Claude Code under **snap confinement**,
which adds traps that do **not** exist on the Mac. Kept here in case work resumes there.

- **`PIP_PREFIX=/home/hinzmo/snap/claude-code/common` is exported globally** by snap. It
  silently redirects every `pip install` *out of any venv* — installs report success, imports
  still fail. `PYTHONPATH` likewise points at snap site-packages and shadows the venv. Run
  everything with the vars cleared:

  ```bash
  env -u PIP_PREFIX -u PYTHONPATH -u PIP_BREAK_SYSTEM_PACKAGES .venv/bin/python -m pip install -e ".[dev,rag]"
  env -u PYTHONPATH .venv/bin/lag run "..."
  env -u PYTHONPATH .venv/bin/python -m pytest
  ```

- **`python3-venv` isn't installed** and `/usr/bin/python3` has no `pip`; the venv was
  bootstrapped with `python3 -m venv --without-pip` + `get-pip.py` (or `sudo apt install
  python3.12-venv`). The box has Python 3.12, so the `[rag]` extra installs there too.
- **`ssh`, `~/.ssh`, and `ollama` are unreachable from inside the snap.** Use `gh` (HTTPS) for
  GitHub, and drive Ollama over HTTP (`http://localhost:11434/api/pull`, `/api/tags`).
- Models were a smaller set: `ministral-3:8b` (orchestrator), `qwen3.5:4b` (worker),
  `num_ctx: 8192` for the 14 GB box.
