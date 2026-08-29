# local-agent-kit v2 — status

Last updated **2026-08-20**. This is the living status doc: what works, how it works, and
what's left. `MORNING_BRIEF*.md` and `REBUILD_COMPLETE.md` are historical — trust this file
for current state.

---

## Where it stands

All six phases run and are smoke-tested. **105 tests passing.** Primary box is now the
**MacBook Air M4** (ported 2026-08-21); the CPU-only Ubuntu machine it was built on is
secondary. On the Mac, both the direct and `--mcp` tool paths are verified end to end.

| Phase | What | State |
|------:|------|-------|
| 0 | Foundation, config, packaging | ✅ |
| 1 | ReAct loop over Ollama, 8 local tools | ✅ verified |
| 2 | MCP server (stdio) wrapping those tools | ✅ verified |
| 3 | The bridge — harness as MCP client | ✅ verified |
| 4 | Swarm: pipeline / parallel / crew | ✅ verified |
| 5 | Nine personas | ✅ all load |
| 6 | RAG server + always-on retrieval | ✅ verified |

Repo: `github.com/Hinzmo/local-agent-kit` (private), branch `main`.

---

## How it works

Three decoupled layers. The harness is the only piece that knows about the others.

```
Ollama (local LLM)  ──tools ↓ / tool_calls ↑──  Harness (lag)  ──stdio · JSON-RPC──  MCP servers
                                                 ReAct loop                           tools_server.py  (8 tools)
                                                 + MCP client                         rag_server.py    (1 tool)
```

### The one abstraction that matters: `tool_provider`

`OllamaClient.run` accepts anything exposing `schemas()` and `call(name, args)`.
`LocalToolProvider` (in-process) and `MCPToolProvider` (real server subprocess) both satisfy
it, so `lag run` and `lag run --mcp` exercise identical tools over different transports with
zero change to the loop. **Preserve this.** An `if mcp:` branch inside `OllamaClient` is the
wrong layer.

### One implementation, two consumers — twice

The same rule applies in two places now:

- **Tools.** The eight functions in `src/lag/tools.py` are consumed unchanged by Ollama's
  schema generation *and* by FastMCP in `servers/tools_server.py`. Adding a tool means: write
  the typed function with an `Args:` docstring, add it to `TOOLS`, add it to the tuple in
  `tools_server.py`. Nothing gets duplicated into the server.
- **Retrieval.** `src/lag/rag.py` holds chunking, embedding, and the ChromaDB collection.
  `servers/rag_server.py` is now just the protocol wrapper and CLI, and `tools.remember`
  imports the same core. This moved out of the server precisely because `remember` became a
  second consumer.

### Memory is now two-layered

`remember(key, value)` writes to **both**:

1. `memory/store.json` — exact-key lookup via `recall`.
2. The ChromaDB collection — embedded, so it comes back from a *plain question*.

That second path is what makes the kit accumulate knowledge. A fact only `recall` can reach
is a fact the model has to already know it stored. Indexing is the default, not a flag.
Remembered chunks carry `kind: "memory"` and surface tagged `· remembered` in search results.

Indexing degrades safely: no `chromadb` or no embedder means the key/value write still
happens and the return string says why indexing was skipped. A core install never breaks.

### Retrieval is always on

There is no "RAG mode". Every turn searches the store first. Documents and remembered facts
share one collection and one ranking.

### Context window — read this before tuning anything

Ollama **ignores a model's trained context** and allocates 4096 by default. `ministral-3:8b`
is trained for 262144, so the harness was running it at 1/64th of its window and silently
dropping the oldest tokens. The symptom is a model that "forgets" the top of its own
conversation — never an error.

`config.yaml → ollama.num_ctx` is now wired through `OllamaClient.from_config` to both the CLI
and every swarm agent. Measured on the M4: num_ctx is a **memory dial, not a speed dial** —
raising it left generation (~19 tok/s) and load time flat; only the footprint (weights + KV
pre-allocated at load) grew.

| num_ctx | qwen3:8b | qwen3:14b | note |
|--------:|---------:|----------:|------|
| 8192 | 6.5 GB | 10.3 GB | **current** on the 16 GB M4 (14b, ~5 GB free) |
| 16384 | 7.4 GB | 12.3 GB | safe for a lone 14b if little else runs |
| 32768 | 9.8 GB | 15.0 GB | 14b brushes disk swap on 16 GB — avoid |

### Single resident model on 16 GB

Two *different* chat models can't co-load on 16 GB (14b@8k 10.3 + 8b@8k 6.5 > 16), so true
multi-model swarming isn't viable locally — it belongs on bigger/cloud hardware. Instead every
chat role points at **one** model (`qwen3:14b`): the orchestrator stays resident and delegating
to a persona changes the *system prompt*, not the weights — no swaps. `ollama ps` during a
`crew` run shows just `qwen3:14b` + `nomic-embed-text` (~10.7 GB). `max_loaded_models: 2` here
means *chat model + embedder*, not two chat models — set it to 1 and every KB search would evict
and reload the 14b.

`num_ctx: 8192` keeps that safe, and the small window is covered by offloading state to memory:
`remember` writes into ChromaDB, `search_knowledge_base` reads it back. To give the orchestrator
that loop in a swarm, run with `--mcp` (`lag swarm "..." --mcp`), which backs every agent with
both MCP servers instead of only in-process tools.

---

### Models live in exactly one place

Personas name a **role**, not a model:

```yaml
# personas/dev_agent.yaml
model: coder        # resolves through config.yaml → models.coder
```

So swapping a model is one line in `config.yaml` and every agent on that role
follows — no persona edits — and each machine runs its own model set from the same
personas. A literal tag (`qwen3.5:4b`) still passes through untouched. A mistyped role
fails immediately at load with the valid roles listed, rather than as a confusing 404
from Ollama mid-run.

| Role | Personas on it |
|------|----------------|
| `orchestrator` | architect_agent, senior_dev_agent |
| `coder` | dev_agent, tester_agent, dev_research_agent |
| `worker` | analyst_agent, biz_analyst_agent, design_agent, research_agent |
| `fast` | unassigned — free for short/cheap calls |
| `embed` | retrieval only, never a chat model |

**Adding a model:** `ollama pull <tag>`, then point a role at it in `config.yaml`. That's
the whole procedure.

## Mapped models (Mac M4 — primary)

| Role | Tag | Notes |
|------|-----|-------|
| `orchestrator` | `qwen3:14b` | the M4 runs a 14B orchestrator comfortably |
| `worker` | `qwen3:8b` | specialists (ASUS used `qwen3:4b`, not pulled here) |
| `coder` | `qwen2.5-coder:7b` | dev / tester / dev_research |
| `fast` | `llama3.2:3b` | unassigned, free for cheap calls |
| `embed` | `nomic-embed-text` | 768-dim embeddings |

`deepseek-r1:*` is installed but **unmapped on purpose** — R1 distills make poor orchestrators
(weak structured output, and thinking tokens wreck loop latency). The ASUS ran a smaller set
(`ministral-3:8b` orchestrator, `qwen3.5:4b` worker) at `num_ctx: 8192`.

---

## Commands

Mac (primary). The `env -u` prefixes are Ubuntu-only — see *Environment traps*.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv          # 3.14 lacks chromadb wheels
.venv/bin/python -m pip install -e ".[dev,rag]"

.venv/bin/lag run "what is this project?"
.venv/bin/lag run "same thing" --mcp
.venv/bin/lag swarm "design a url shortener" --mode crew

.venv/bin/python servers/rag_server.py --ingest docs/
.venv/bin/python servers/tools_server.py            # standalone, real tracebacks
.venv/bin/python -m pytest                           # 105 passing
```

---

## Setting up the Mac (M4)

The repo carries everything except per-machine config, which is gitignored by design.

1. `git clone git@github.com:Hinzmo/local-agent-kit.git && cd local-agent-kit`
2. `/opt/homebrew/bin/python3.12 -m venv .venv && .venv/bin/python -m pip install -e ".[dev,rag]"`
   — **use 3.12, not the default `python3`**: macOS ships 3.14, which has no `chromadb` wheel
   yet, so the `[rag]` install fails on it. No `env -u` dance needed; the snap traps are
   Ubuntu-only. `brew install python@3.12` if it's missing.
3. `cp config.example.yaml config.yaml`
4. `ollama pull` the models you want, then **edit `config.yaml`**:
   - the M4 handles a larger orchestrator comfortably — `qwen3:14b` was the original intent
   - raise `num_ctx` to 16384–32768; unified memory makes this cheap
5. **Personas need no edits** — they name roles, so just point the five roles in
   `config.yaml` at whatever you pulled. `ollama list` gives the exact tags.
6. `ollama pull nomic-embed-text`, then `python servers/rag_server.py --ingest docs/`
7. `pytest` — expect 105 passing.

`.chroma/` and `memory/` are gitignored, so **remembered facts do not sync between machines.**
Each box accumulates its own memory. See *What's left* if you want that shared.

---

## Environment traps (Ubuntu box only)

These cost about an hour of misdiagnosis. Check `env | grep -E '^(PIP_|PYTHON)'` before
debugging any "installed but not importable" problem.

- **`PIP_PREFIX=/home/hinzmo/snap/claude-code/common` is exported globally** by snap
  confinement. It silently redirects every `pip install` *out of any venv* — installs report
  success and imports still fail.
- `PYTHONPATH` points at snap site-packages and shadows the venv.
- The original `.venv` was built for **Python 3.14** on the previous machine; this box has
  3.12, so its `site-packages` was invisible. Rebuilt against 3.12.
- `python3-venv` isn't installed and `/usr/bin/python3` has no `pip` — the venv was
  bootstrapped with `--without-pip` + `get-pip.py`.
- `ssh`, `~/.ssh`, `ollama`, `curl`, and `unzip` are unreachable from the sandbox. Use `gh`,
  the Ollama HTTP API, and Python's stdlib.

---

## Fixed along the way

Every one of these was found by running the code, not reading it.

1. **`--mcp` hung.** The server was spawned as bare `"python"`, absent on Ubuntu; it died on
   import and `stdio_client` waited forever on a silent pipe. Now `sys.executable`, with the
   server path resolved against the repo root.
2. **`mcp 2.0.0` renamed `FastMCP`** → `mcp.server.mcpserver.MCPServer`. Both servers import
   either.
3. **`mcp 2.0.0` renamed `Tool.inputSchema`** → `input_schema`, breaking the bridge's schema
   translation. Read through a helper accepting both.
4. **`chunk_text` never terminated.** On the final chunk `start` rewound to `end - overlap`,
   still below the text length, so it re-emitted the same tail until memory ran out. Every
   ingest would have hung.
5. **RAG server printed to stdout**, which *is* the JSON-RPC channel under stdio — corrupting
   the stream and breaking `initialize`. Moved to stderr.
6. **`.gitignore` was silently broken.** Git has no inline comments, so
   `config.yaml   # per-machine` was one long pattern matching nothing. `config.yaml` and
   `.chroma/` were never actually ignored.
7. **Chunk IDs collided.** Ingest used `chunk_0..N`, so a second ingest overwrote the first,
   and `remember` would have clobbered whatever shared its index. Now content-addressed
   (SHA-1 prefix) and namespaced per writer, which also makes re-adding idempotent.
8. **`num_ctx` was never set** — see above.
9. **UI: full-screen flicker.** A literal `@keyframes flick` overlay plus `mix-blend-mode`
   over a 60fps canvas plus a non-DPR-backed canvas moiréing against the scanlines.

---

## What's left

**~~Wire both MCP servers into the harness.~~ Done.** `MCPToolProvider.from_config` now spawns
both `tools_server.py` and `rag_server.py`; the provider holds a list of bridges, merges their
schemas into one flat toolset (first server wins a name collision), and routes each `call()`
to the owning bridge. The agent loop was untouched. Verified live: `lag run "...search the
knowledge base..." --mcp` spawns the RAG server and the model calls `search_knowledge_base`.
So "always-on retrieval" is now true in `lag run --mcp`, not just in direct calls.

**`lag run --mcp` must be invoked from the repo root.** The RAG server resolves its store as
`./.chroma` relative to cwd, so running the harness from elsewhere points it at a missing/empty
collection (the tools server is unaffected — its tools take explicit paths). Small follow-up:
resolve `DB_PATH` against the repo root, or pass `cwd` through `StdioServerParameters`.

**`MCPBridge` never shuts down.** No `close()`, so the subprocess and its daemon event loop
leak for the process lifetime — and it's **two** subprocesses now that both servers spawn.
Fine for one-shot CLI runs, wrong for anything long-lived; the clean fix is keeping the whole
run inside `async with` blocks.

**`thinking_mode` in personas is metadata only** — parsed, never passed to
`ollama.chat(think=...)`. Roughly two lines in `ollama_client.py`.

**`run_shell` is unsandboxed** — `shell=True`, no path allow-list, 60s timeout. Now
demonstrably live: a crew run wrote two real files into the repo root unprompted. Accepted
risk on a single-user offline box; revisit before anything multi-user.

**`web_search` leaves the machine.** It's the one tool that does — the query goes to
DuckDuckGo. Keyless and stdlib-only, so no account and no new dependency, but worth knowing
it's the single exception to "your data never leaves the machine". It also scrapes an HTML
endpoint, so it will break if DuckDuckGo changes markup; the parser is pinned by tests that
use a captured page, not the network.

**Memory is per-install, by decision.** `.chroma/` and `memory/` stay gitignored and each
machine grows its own. Facts a box learns are usually *about* that box — its hardware limits,
its model set, what's been run there — so syncing them would import claims that are false on
the other machine. Not a gap to close; if a fact genuinely belongs everywhere, it goes in a
committed doc and gets ingested on both.

**The UI is a mockup.** `ui/console.html` is scripted playback, wired to nothing. It reflects
real project data (models, personas, tool registry, context size) but no live process.
