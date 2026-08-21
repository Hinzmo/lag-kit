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

Recovered, committed, pushed, and — as of 2026-08-19 — **actually running**. All six phases
are smoke-tested end to end on this machine, including RAG.

```
src/lag/     config.py  tools.py  ollama_client.py  mcp_client.py  swarm.py  cli.py
servers/     tools_server.py  rag_server.py
personas/    nine YAMLs
tests/       64 tests, all passing
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

### Environment landmines (cost hours; don't rediscover them)

- **Always run through `.venv/bin/`, and clear the snap's env vars.** Claude Code runs under
  snap confinement, which exports `PIP_PREFIX=/home/hinzmo/snap/claude-code/common` and a
  `PYTHONPATH` pointing at snap site-packages. `PIP_PREFIX` silently redirects every
  `pip install` out of the venv, so installs "succeed" and imports still fail. Use:

  ```bash
  env -u PIP_PREFIX -u PYTHONPATH -u PIP_BREAK_SYSTEM_PACKAGES .venv/bin/python -m pip install -e ".[dev]"
  env -u PYTHONPATH .venv/bin/lag run "..."
  env -u PYTHONPATH .venv/bin/python -m pytest
  ```

- **`python3-venv` is not installed** and `/usr/bin/python3` has no `pip`. The venv here was
  bootstrapped with `python3 -m venv --without-pip` + `get-pip.py`. Recreating it needs the
  same dance (or `sudo apt install python3.12-venv`).
- The original `.venv` was built for **Python 3.14** on the old machine; this box has 3.12,
  so its `site-packages` was invisible to the interpreter. It has been rebuilt against 3.12.
- **`ssh` and `~/.ssh` are unreachable** from inside the snap. Use `gh` (authenticated, HTTPS)
  for anything GitHub, or have the user run ssh commands themselves with the `!` prefix.

### Models

`config.yaml` (gitignored) is mapped to what is actually pulled here — `ministral-3:8b`
(orchestrator) and `qwen3.5:4b` (everything else). The nine personas were likewise remapped
off `qwen3:8b` / `qwen3:4b` / `qwen2.5-coder:7b`, none of which are installed. `deepseek-r1:8b`
is present but deliberately unmapped: R1 distills make poor orchestrators here (weak
structured output, thinking tokens wreck loop latency).

`nomic-embed-text` is pulled, and `.chroma/` holds an ingest of `docs/` (9 chunks).

Note: `ollama` is **not on PATH** inside the snap sandbox. Drive it over HTTP
(`http://localhost:11434/api/pull`, `/api/tags`) or have the user run it in their own shell.

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

Prefixes matter here — see the environment landmines above.

```bash
# install (note the unset vars; PIP_PREFIX will otherwise redirect out of the venv)
env -u PIP_PREFIX -u PYTHONPATH -u PIP_BREAK_SYSTEM_PACKAGES \
    .venv/bin/python -m pip install -e ".[dev]"     # core + pytest
#   ...install -e ".[rag]"                          # + chromadb, Phase 6

env -u PYTHONPATH .venv/bin/lag run "list the files here and tell me what this project is"
env -u PYTHONPATH .venv/bin/lag run "same thing" --mcp     # routes through the MCP server
env -u PYTHONPATH .venv/bin/lag swarm "design a tiny URL shortener" --mode crew

env -u PYTHONPATH .venv/bin/python servers/tools_server.py  # standalone; shows real tracebacks
env -u PYTHONPATH .venv/bin/python -m pytest

ollama pull nomic-embed-text          # required before Phase 6
```

`cp config.example.yaml config.yaml` if starting on a new machine — but note the example
names models that are not pulled here.

Runs are slow: CPU-only inference, ~1–3 min for a single tool-calling round trip, and a
four-call `crew` run takes considerably longer. Background long runs rather than waiting.

## Config

`config.py` searches `config.yaml` then falls back to `config.example.yaml`, so the kit runs
before customization. Models are per-machine (`models.orchestrator|worker|coder|fast|embed`):
the MacBook Air M4 handles `qwen3:14b` as orchestrator; the CPU-only ASUS should stay small.
**Avoid R1 distills as orchestrators** — weak structured output, and thinking tokens destroy
loop latency.

## Swarm modes (`swarm.py`)

- `pipeline` — each agent's output feeds the next.
- `parallel` — all agents run the task concurrently via `ThreadPoolExecutor`; outputs concatenated.
- `crew` — first agent coordinates: decompose → delegate to specialists → synthesize.

## Phase 6 — RAG (`servers/rag_server.py`)

Verified working: `--ingest` chunks and embeds via `nomic-embed-text`, and
`search_knowledge_base` answers over the real MCP boundary.

```bash
env -u PYTHONPATH .venv/bin/python servers/rag_server.py --ingest docs/
env -u PYTHONPATH .venv/bin/python servers/rag_server.py     # server mode
```

Two traps already hit and fixed here, worth remembering if you write another server:

- **`chunk_text` had a non-terminating loop.** On the final chunk `start` rewound to
  `end - overlap`, still `< len(text)`, so it re-emitted the same tail until memory ran out —
  every ingest hung. Now strides forward and breaks at the end.
- **Server mode printed to stdout.** Under the stdio transport stdout *is* the JSON-RPC
  channel, so any stray `print` corrupts the stream and `initialize` fails. Diagnostics go to
  stderr. This applies to every MCP server in this repo.

**Not wired into the harness yet.** `MCPToolProvider.from_config` spawns only
`tools_server.py`. Serving both means spawning both and merging their schemas — the
`schemas()`/`call()` interface already supports it; nothing in the loop needs to change.

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

Ollama ignores a model's trained context and allocates **4096** by default. `ministral-3:8b`
is trained for 262144. Always construct clients via `OllamaClient.from_config` so
`ollama.num_ctx` and `keep_alive` are applied; constructing `OllamaClient(...)` directly
silently reverts to 4096 and the model starts dropping the top of its own conversation with no
error. Budget ~130 KB/token of KV cache for an 8B model.

## Known loose ends

These are deliberate, documented in `files/MORNING_BRIEF.md` §4 — not bugs to chase blindly:

- **`web_search` is implemented** (DuckDuckGo HTML endpoint, keyless, stdlib-only). It is the
  one tool that leaves the machine — the query goes off-box. It scrapes markup, so it will
  break if DuckDuckGo changes their HTML; the parser is pinned by tests against a captured
  page, never the live network. Don't add an API-key search provider without asking; keyless
  is what keeps the privacy stance intact.
- **`run_shell` is unsandboxed** — `shell=True`, no path allow-list, 60s timeout. Accepted
  risk on a single-user offline box. Revisit before anything multi-user. Don't silently
  "fix" this without asking; it's a deferred decision, not an oversight.
- **`thinking_mode` in personas is metadata only** — parsed but never passed to
  `ollama.chat(think=...)`. Wiring it is ~2 lines in `ollama_client.py`.
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
