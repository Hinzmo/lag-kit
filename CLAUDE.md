# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`local-agent-kit` (`lag`) — a fully offline multi-agent harness over [Ollama](https://ollama.com).
No cloud, no API keys. Agents call tools exposed over the Model Context Protocol (MCP);
retrieval is just one more MCP server.

This is the **v2 rebuild**. v1 faked the tool layer with a direct-dispatch HTTP/JSON-RPC
shim; v2's whole point is a *real* MCP boundary — tools live in separate server processes,
discovered at runtime via `tools/list`, reusable by any MCP client.

## Repo state

All six original phases (foundation → ReAct loop → MCP server → MCP client bridge →
pipeline → personas → RAG) are done and smoke-tested. Work since then has added agent
mode, a live web console, container sandboxing for agents, and a study mode.

**Tests: 431 pass with Docker running.** Without it the four real-container tests in
`test_sandbox.py` skip (start Docker Desktop to run them). `tests/test_console_boot.py` additionally needs
node + jsdom (`cd tests/js && npm install`) and skips without them.

```
src/lag/   config  tools  ollama_client  mcp_client  providers  sandbox  pipeline  agent
           rag  render  study  web  cli  netguard  egress
servers/   tools_server.py  rag_server.py
personas/  nine YAMLs
agents/    saved agent definitions + per-machine run state (see Agent mode)
ui/        console.html — the web console served by `lag serve`
tests/     pytest suite, plus tests/js for the console boot check
docs/      RAG_SETUP.md  VECTORS_AND_SEARCH.md
```

Remote: `github.com/Hinzmo/local-agent-kit` (private). Default branch `main`; feature work
goes on branches and lands via PR.

> **`STATUS.md` is the living status doc** — update it when behaviour changes; this file
> stays focused on working rules. `MORNING_BRIEF.md` and `REBUILD_COMPLETE.md` are June 2026
> historical documents: reliable on design intent, stale on status.

### Environment (Mac — the primary box)

The primary box is a **MacBook Air M4, 16 GB** (since 2026-08-21). The CPU-only Ubuntu
machine the kit was built on is secondary; its traps are in the appendix.

- **Build the venv on Python 3.12.** macOS ships **3.14** as `python3`, and `chromadb` (the
  `[rag]` extra) has no 3.14 wheels, so an install on 3.14 fails at the C-extension build:

  ```bash
  /opt/homebrew/bin/python3.12 -m venv .venv
  .venv/bin/python -m pip install -e ".[dev,rag,web,study]"
  ```

  `web` (fastapi, uvicorn) is needed for `lag serve` **and** for `tests/test_web.py`;
  `study` (pypdf) for PDF material in `lag study`.

- No `env -u` prefixes needed on the Mac — those are Ubuntu snap artifacts.
- `gh` is authenticated over HTTPS and `ssh` reaches GitHub; either works.
- **Agent mode needs Docker Desktop running** (see Agent mode — it fails closed without it).

### Models (Mac)

`config.yaml` (gitignored) maps roles to pulled tags:

| Role | Tag | Personas on it |
|------|-----|----------------|
| `orchestrator` | `ornith:latest` | architect, senior_dev |
| `worker` | `qwen3:8b` | analyst, biz_analyst, design, research |
| `coder` | `qwen3:8b` | dev, tester, dev_research |
| `fast` | `qwen3:8b` | unassigned |
| `embed` | `nomic-embed-text` | retrieval only (768-dim) |

Two rules drove this shape:

- **Role models should support tools.** The loop is a tool-calling loop.
  `ollama_client.capabilities()` asks Ollama what a tag can do; a completion-only model runs
  *without* tools and says so (a `no_tools` event) rather than failing. That is a real
  downgrade — no files, shell, retrieval or memory. Capability is **discovered, never
  hardcoded**. A failed probe returns an empty set, which callers must read as *capable*:
  treating unknown as incapable would strip a working model of every tool on one flaky
  status call. `thinking` is resolved the same way. On this box
  `example/completion-only-8b` and `example/completion-only-mini` are
  completion-only; `persona-model:latest` supports tools but ships a baked-in persona that
  fights the persona prompts.
- **A role switch costs a full model reload** (~5s), because only one chat model stays
  resident. So every downstream role points at ONE tag, and `ornith` (9B qwen3.5, tools +
  thinking, 262K trained window) spends its single load on planning.

`senior_dev` sits on `orchestrator`, so the default three-stage roster runs
ornith → qwen3:8b → ornith and pays three loads — deliberate, since review is the hardest
reasoning. Move `senior_dev_agent` to `coder` in its persona for a two-load run. Personas
name roles, never tags.

`num_predict: 2048` caps generation per turn, thinking included — without it
`senior_dev_agent` was measured spiralling to 4534 thinking tokens and never finishing.

## Architecture

Three decoupled layers; the harness is the only piece that knows about the others.

```
Ollama (local LLM)  ──tools ↓ / tool_calls ↑──  Harness (lag)  ──stdio · JSON-RPC──  MCP servers
                                                ReAct loop                            tools_server.py
                                                + MCP client                          rag_server.py
```

### The central abstraction: `tool_provider`

Everything hinges on one duck-typed interface. `OllamaClient.run` accepts any object with:

```python
schemas() -> list       # whatever ollama.chat accepts as `tools`
call(name, args) -> str
```

Two base implementations satisfy it, and the loop cannot tell them apart:

- `LocalToolProvider` (`tools.py`) — dispatches to in-process Python functions.
  `schemas()` returns the *function objects themselves*; Ollama derives JSON schemas from
  type hints + docstrings.
- `MCPToolProvider` (`mcp_client.py`) — routes through real MCP server subprocesses.
  `schemas()` translates MCP's `{name, description, inputSchema}` into Ollama's
  `{type: "function", function: {name, description, parameters}}` shape.

So `lag run "..."` and `lag run "..." --mcp` exercise the same tools through different
transports with no change to the agent loop. **Preserve this property.** An `if mcp:`
branch inside `OllamaClient` is the wrong layer.

### Provider wrappers (`providers.py`)

Allow-lists, deny-lists, keeping goal tools in-process, and sandboxing are all **wrappers**
around a tool_provider, not replacements — they compose in any order and the loop never
knows which are present. `SandboxedProvider` lives in `sandbox.py` but follows the same
contract. **Every filter must read both schema shapes** (function objects *and*
Ollama-shaped dicts) — use `schema_name()`. A filter that only understands one looks
enforced in tests and silently passes everything over the other transport.

### One source of truth for tools

`tools.py` holds 16 tool functions in the `TOOLS` dict: `read_file`, `write_file`,
`list_dir`, `run_shell`, `system_info`, `remember`, `recall`, `web_search`, `find_files`,
`grep_files`, `http_fetch`, `now`, `sleep_for`, plus the agent-mode goal tools
`goal_status`, `note_progress`, `mark_done`. `tools_server.py` imports `TOOLS` and loops
`mcp.tool()(_fn)` over it, so the two can't drift. `rag_server.py` adds
`search_knowledge_base`, for 17 tools over `--mcp`.

To add a tool: write the plain function in `tools.py` with a typed signature and a
Google-style `Args:` docstring, and add it to `TOOLS`. Nothing goes into the server.

Tool errors are returned as strings to the model, never raised — see
`LocalToolProvider.call`. A raised exception kills the loop instead of letting the model
recover.

### The async/sync bridge

`MCPBridge` (`mcp_client.py`) reconciles the async MCP SDK with the sync loop: one asyncio
loop on a daemon thread, calls marshalled over with `run_coroutine_threadsafe`. **The
session's whole lifetime is one task** (`_live`): it enters `stdio_client` and
`ClientSession` with `async with`, signals ready, and parks; `close()` cancels it, so the
task that entered the context managers is the one that exits them. That is anyio's rule.
The old design entered them by hand (`__aenter__`) in one task and exited them in another;
anyio raised on both exits, `close()` swallowed it, and the server died only because the
SDK killed it before raising. Don't go back to manual `__aenter__`/`__aexit__`.

Ownership: **whoever builds a provider closes it.** The CLI closes in a `finally`; the web
server caches one per transport and closes them in its lifespan hook at shutdown.
`from_config` is all-or-nothing: if a later server fails, the ones already started are
closed before the error propagates — raising past them orphaned a live server and its
thread on every attempt, and the console retries on every request. Failures surface as
the real error (`MCPError: Connection closed`), unwrapped from anyio's `ExceptionGroup`.

Servers are spawned with `sys.executable`, not a bare `"python"` — the historical
"`--mcp` hangs" was a server dying on import while `stdio_client` waited on a silent pipe.
If a server dies at startup the failure surfaces only as `MCPError: Connection closed`;
**run the server standalone first** (`.venv/bin/python servers/tools_server.py`) to see the
real traceback.

## Commands

```bash
.venv/bin/python -m pip install -e ".[dev,rag,web,study]"   # everything; pyproject.toml is the source of truth, no requirements.txt

.venv/bin/lag run "list the files here and tell me what this project is"
.venv/bin/lag run "same thing" --mcp               # routes through the MCP servers
.venv/bin/lag pipeline "design a tiny URL shortener"            # orchestrator routes
.venv/bin/lag pipeline "..." --agents "architect_agent,dev_agent" --no-plan
.venv/bin/lag serve                                # web console on 127.0.0.1

.venv/bin/lag agent new | list | show | start | status | stop | enable | disable | delete
.venv/bin/lag study add | ask | quiz | cards | review | status

.venv/bin/python servers/tools_server.py           # standalone; shows real tracebacks
.venv/bin/python servers/rag_server.py --ingest docs/   # populate .chroma
.venv/bin/python -m pytest

ollama pull nomic-embed-text                       # required for retrieval
```

`cp config.example.yaml config.yaml` on a new machine, then map the five roles to tags you
have pulled (`ollama list`).

**Run `lag` from the repo root.** `rag.DB_PATH` is `./.chroma` relative to cwd, and agent,
study and memory state (`agents/`, `study/`, `memory/`) resolve the same way.

On the M4 a single tool-calling round trip is ~30–40s including model load. Background long
runs rather than blocking on them.

## Config

`config.py` searches `config.yaml` then falls back to `config.example.yaml`, so the kit runs
before customization. Model tags live **only** under `models:`; personas name a role
(`model: coder`) that `config.resolve_model` maps to a tag (a literal tag still passes
through). Never hardcode a tag into a persona — that is the drift that once left all nine
pinned to models the machine didn't have. **Avoid R1 distills as orchestrators** — weak
structured output, and thinking tokens destroy loop latency.

## Context window

Ollama ignores a model's trained context and allocates **4096** by default. Always construct
clients via `OllamaClient.from_config` so `num_ctx` and `keep_alive` are applied;
`OllamaClient(...)` directly silently reverts to 4096 and the model drops the top of its own
conversation with no error.

`ollama.num_ctx_by_model` sizes each tag on its own terms (falls back to `ollama.num_ctx`),
since only one chat model is resident. Measured on the 16 GB M4 (weights + KV):

| num_ctx | `ornith:latest` | `qwen3:8b` |
|---------|-----------------|------------|
| 8192 | 5.48 GB | 6.18 GB |
| 16384 | 6.35 GB | 7.41 GB |
| 32768 | 6.40 GB | 9.91 GB |
| 40960 | — | 11.16 GB (its trained max) |
| 131072 | 10.12 GB | — |

`ornith` sits at **32768** and `qwen3:8b` at **8192**. **num_ctx is a memory dial, not a
speed dial** — across ornith at 8192 / 32768 / 131072, load and generation speed stayed
flat. Prefill tracks tokens actually used. A wider lead window is also capped in value by
`Pipeline._brief`, which is bounded to fit the 8192 stage that reads it: a plan written wide
and read narrow is truncated **silently**.

If a run feels slow, the cause is usually **thinking**, not the window: a one-stage run
answering `1 2 3` emitted 140 thinking tokens against 5 answer tokens. Persona
`thinking_mode` is wired through to `ollama.chat(think=...)`, with a one-shot retry without
`think` for tags that reject it.

## Live streaming

`OllamaClient.run` and `Pipeline.run` take an optional `on_event` callback — a **side
channel**: the return value is identical with or without it, so blocking `/api/run` and
streaming `/api/stream` cannot disagree. Event types include `start`, `stage` (with `model`
and `loaded`), `thinking`, `token`, `tool`, `tool_result`, `no_tools`, `done`, `error`.

Streaming is opt-in because reconstructing `tool_calls` from chunks is more fragile than
reading a finished message — `_turn` streams only when someone is listening. Thinking deltas
are their own event type because on ornith they are most of the wait.

`/api/stream` is a POST returning `text/event-stream` (prompt in the body, not the URL). The
loop runs on a daemon thread feeding a queue the response drains.

**Handing one agent a task is not a special case** — it's a one-stage pipeline with
`plan=False`. The console roster posts exactly that; there is no separate code path.

**Display cleanup lives in `render.py`, once, upstream.** Tool-call arg dicts, JSON tool
output, and tool calls a model fumbled into its text are cleaned before any consumer (chat
pane, activity window, CLI) sees them. Never re-implement it in the view layer. It must
stay defensive — a cleaner that throws or mangles a real code block is worse than one that
lets something through.

## Project pipeline (`pipeline.py`)

One sequential shape. Stages run in order, each seeing the full context above it; the lead
plans once and its per-stage instructions ride that context down. By default the
orchestrator **routes** — picks which agents the task needs (`--no-route` or `--agents` to
override). `Pipeline.loads()` reports the cold-load cost of an ordering.

The old `parallel` and `crew` modes were removed (2026-09-07). Don't reintroduce them:
`parallel` made agents on different tags evict each other in a loop; `crew` forced a second
cold load for the coordinator's return trip.

## Agent mode (`agent.py`)

The unbounded shape: one persona, a goal, a loop that runs until the goal is met or you stop
it. Files are paired by name in `agents/`:

```
agents/<name>.yaml         the definition — yours to keep
agents/<name>.state.json   run state, written after every cycle — per-machine
agents/<name>.stop         a stop request
agents/workspaces/         each agent's writable workspace
agents/.run.lock           held (flock) while an agent runs — who, pid, since
```

What makes it safe to leave running:

- **It can stop.** `mark_done` is the termination condition; `--max-cycles` and the stop
  file are backstops. The cap is **per run** — `state.cycle` is the lifetime count and
  survives a resume, so the cap is measured from where this run started. Every stop
  records a `stopped_reason`. The idle wait polls the stop file every second.
- **It survives its own failures.** Every cycle is wrapped; an exception becomes an
  observation the next cycle reads, then the loop backs off and continues.
- **It remembers.** Cycles share no conversation, so the goal, the progress log
  (`note_progress`) and the last observation are the continuity — `GoalSpec.prompt`
  restates them every cycle. A run resumes if the goal matches.
- **It is confined.** See below.
- **One at a time, per machine.** Only one chat model is resident, so two agents would
  evict each other's model every turn. `AgentLoop.run` takes an OS file lock (`flock` on
  `agents/.run.lock`) before touching anything, so the CLI, the console and a second tab
  all hit the same lock — and the kernel releases it if the holder dies, even on
  `kill -9`, which a PID file can't promise. The web server also claims the run under a
  `threading.Lock` before answering (409 if busy), and the console greys out every
  start control while `/api/agent/running` names an agent.

### Sandbox (`sandbox.py`) — mandatory for agents, not configurable

An unattended loop started from a saved definition is a different risk from a supervised
`lag run`, so agent mode's shell runs in a **Docker (or Podman) container**:

- The project is mounted **read-only** at `/project`; the agent **writes only** its own
  workspace at `/workspace`. `SandboxedProvider` translates those paths for the host-side
  file tools and checks writes in Python, since those tools run outside the container.
- Capabilities dropped, `--pids-limit 256`, memory capped with `--memory-swap 2g` (so swap
  can't double it), `--cpus 2`. No published ports.
- **Network on (the default) means the public internet only, via a filtering proxy** —
  never Docker's default bridge, which NATs through the host (measured: it reached the
  host's Ollama via `host.docker.internal`). Each agent gets its own `--internal` network
  (`lag-net-<id>`, no route anywhere) plus a proxy container (`lag-egress-<id>`, `egress.py`)
  on bridge + that network. The proxy applies `netguard.py`'s public-only rule to every
  destination and connects to the exact address it vetted (no DNS rebinding). `HTTP(S)_PROXY`
  is set in both spellings; tools that ignore it have no route. `--no-network` /
  `network: false` adds `--network none` instead. Network, proxy and container are all
  created at start and removed at stop.
- **One guard, two consumers.** `netguard.py` (stdlib only) is used by `tools.http_fetch` on
  the host and by the egress proxy inside its container. Never copy the rule into either.
- **It fails closed.** No reachable container runtime → agent mode refuses to start, checked
  before the first cycle.
- The command deny-list is defence in depth for clear error messages, **not** the boundary.
  A `sandbox-exec` (seatbelt) backend was tried and abandoned. Don't reintroduce either as
  the security boundary. The list judges the **parsed** command (program + arguments, via
  `shlex`), never substrings — substring rules refused `rm -rf /workspace/build` and
  `grep shutdown log.txt` while missing `rm -r -f /`. Because it isn't the boundary, its
  failure mode to optimise is the false refusal, which costs an agent a cycle.

## Study mode (`study.py`)

`lag study` — a course brain over your own notes. `ask` answers from your material with
citations; `quiz`/`cards`/`review` drill you and grade answers strictly against the source.
Each course is its own Chroma collection so courses don't bleed into each other. Progress
and decks live in `study/<course>/` (per-machine).

## Memory and retrieval

`remember(key, value)` writes to `memory/store.json` **and** embeds into ChromaDB, so facts
come back from a plain question, not only an exact `recall`. Indexing is the default, never
a flag. Remembered chunks carry `kind: "memory"` and surface tagged `· remembered`.

`src/lag/rag.py` owns chunking, embedding and the collection; `servers/rag_server.py` is only
the protocol wrapper and CLI. Don't move retrieval logic into the server.

Indexing must stay non-fatal: `tools._index_memory` imports `lag.rag` lazily and swallows
everything, so a core install without `[rag]` still remembers.

Chunk ids are content-addressed and namespaced (`chunk_*` vs `mem_*`). Never go back to
positional ids — `chunk_<i>` made a second ingest overwrite the first.

Memory and `.chroma/` are per-machine by decision: facts a box learns are usually about that
box.

## Writing an MCP server — two traps already hit

- **Never print to stdout in server mode.** Under stdio, stdout *is* the JSON-RPC channel; a
  stray `print` corrupts it and `initialize` fails. Diagnostics go to stderr.
- **Loops must stride forward.** `chunk_text` once rewound to `end - overlap` on the last
  chunk and re-emitted the tail forever; every ingest hung.

## Known loose ends and deliberate decisions

- **Supervised `lag run` / `pipeline` shell is still unsandboxed** — `shell=True`, no path
  allow-list, 60s timeout. Accepted risk on a single-user box; only agent mode is confined.
  Don't silently change this; it's a deferred decision.
- **Two tools leave the machine:** `web_search` (DuckDuckGo HTML endpoint, keyless,
  stdlib-only) and `http_fetch` (a GET of a URL the model chose, stripped to text). Both run
  in the harness process, not the container. `http_fetch` has an **SSRF guard**
  (`_ssrf_block_reason`): **default-deny** — only globally routable addresses pass, IPv4
  embedded in IPv6 (mapped, 6to4, Teredo, NAT64) is unwrapped and checked too, and every
  redirect hop is re-checked. Keep it an allow-list; a block-list missed 100.64.0.0/10
  (Tailscale). DNS rebinding (resolve-then-reconnect) is a known, documented gap. `web_search`
  scrapes markup and will break if DuckDuckGo changes it; its parser is tested against a
  captured page, never the live network. Don't add an API-key search provider without
  asking.
- **cwd dependence** — see Commands. Resolving state paths against the repo root is an open
  follow-up.
- **mcp SDK is 2.0.0**, which renamed things the kit used:

  | mcp 1.x | mcp 2.x |
  |---|---|
  | `mcp.server.fastmcp.FastMCP` | `mcp.server.mcpserver.MCPServer` |
  | `Tool.inputSchema` | `Tool.input_schema` |

  Both servers import either via `try/except ImportError`, and `mcp_client._input_schema`
  accepts both. Tutorials are probably on 1.x — check `pip show mcp`.

## Conventions

- `from __future__ import annotations` at the top of every module; PEP 604 unions.
- Module docstrings explain *why the design is this way*, not what the code does.
- Tool functions need Google-style `Args:` docstrings — they become the schema the model
  reads.
- `config.yaml`, `memory/`, `data/`, `.chroma/`, agent run state and `study/` are
  gitignored; commit `config.example.yaml`. **Never put a trailing `# comment` on a
  `.gitignore` pattern** — git has no inline comments, so the rule matches nothing. Verify
  new rules with `git check-ignore -v <path>`.
- Privacy is load-bearing ("your data never leaves the machine"). No cloud calls, telemetry,
  or API-key dependencies.

## Appendix — Ubuntu box (secondary)

Built originally on a CPU-only Ubuntu machine running Claude Code under **snap
confinement**. These traps do **not** exist on the Mac.

- **`PIP_PREFIX=/home/hinzmo/snap/claude-code/common` is exported globally** by snap and
  silently redirects every `pip install` out of any venv; `PYTHONPATH` shadows the venv.
  Clear them:

  ```bash
  env -u PIP_PREFIX -u PYTHONPATH -u PIP_BREAK_SYSTEM_PACKAGES .venv/bin/python -m pip install -e ".[dev,rag]"
  env -u PYTHONPATH .venv/bin/lag run "..."
  env -u PYTHONPATH .venv/bin/python -m pytest
  ```

- `python3-venv` isn't installed and `/usr/bin/python3` has no `pip`; bootstrap with
  `python3 -m venv --without-pip` + `get-pip.py`. Python 3.12, so `[rag]` installs.
- `ssh`, `~/.ssh` and `ollama` are unreachable from inside the snap. Use `gh` (HTTPS) and
  Ollama's HTTP API (`http://localhost:11434/api/pull`, `/api/tags`).
- Smaller models: `ministral-3:8b` (orchestrator), `qwen3.5:4b` (worker), `num_ctx: 8192`.
