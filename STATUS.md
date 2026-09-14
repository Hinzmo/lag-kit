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
| 4 | Project pipeline (sequential, context passed down) | ✅ verified |
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

Ollama **ignores a model's trained context** and allocates 4096 by default. `ornith:latest`
is trained for 262144, so leaving it unset runs the harness at 1/64th of its window and
silently drops the oldest tokens. The symptom is a model that "forgets" the top of its own
conversation — never an error.

`config.yaml → ollama.num_ctx` is wired through `OllamaClient.from_config` to both the CLI and
every pipeline stage, and `ollama.num_ctx_by_model` overrides it per tag. Measured on the M4:
num_ctx is a **memory dial, not a speed dial** — raising it left generation (~19 tok/s) and
load time flat; only the footprint (weights + KV pre-allocated at load) grew. *Filling* the
window is the part that costs, since prefill tracks tokens actually used.

Re-measured 2026-09-07 on the current pair:

| num_ctx | `ornith:latest` | `qwen3:8b` | note |
|--------:|----------------:|-----------:|------|
| 8192 | 5.48 GB | 6.18 GB | **qwen3:8b runs here** — it does the bulk of the work |
| 16384 | 6.35 GB | 7.41 GB | |
| 32768 | 6.40 GB | 9.91 GB | qwen3:8b's KV gets expensive fast |
| 40960 | — | 11.16 GB | qwen3:8b's trained max, and its practical ceiling |
| 65536 | 7.64 GB | — | |
| 131072 | 10.12 GB | — | **ornith runs here** — 16× the window for less than qwen3:8b@32768 |

Because only one model is resident at a time, each is sized on its own terms instead of being
pinned to the smaller one's ceiling. **Current setting: ornith 32768, qwen3:8b 8192.**

131072 was tried and dropped. It is not a speed question — measured on ornith at
8192 / 32768 / 131072, load was 8.40s / 5.38s / 6.39s and generation 13.6 / 15.4 / 15.3 tok/s,
i.e. flat, so dialling the window *down* buys memory back and nothing else. It was dropped
because 10.12 GB bought headroom the pipeline cannot use: `Pipeline._brief` is bounded to fit
the 8192 stage that reads it, since a plan too wide for the next stage is truncated
**silently**, not rejected.

**What actually costs time is thinking, not context.** ornith has it enabled natively, and
streaming makes the ratio visible: a one-stage run answering `1 2 3` emitted 140 thinking
tokens against 5 answer tokens; a two-stage run, 502 against 187. That is the lever worth
pulling if runs feel slow.

### One model at a time (measured 2026-09-07)

Ollama on this box holds exactly **one** chat model resident, and that is a hard concurrency
cap — *not* a memory ceiling. Probed at num_ctx 4096 / 6144 / 8192, loading a second model
evicted the first every time while the total sat at ~5.9 GB of 16 GB. Lowering num_ctx buys
nothing; two chat models simply cannot co-reside here.

Every switch therefore costs a full cold reload — measured `ornith` 5.7–6.4s, `qwen3:8b`
4.4–4.9s. That single fact drove three decisions:

1. **`parallel` mode was deleted.** Firing agents concurrently across two tags makes them evict
   each other in a loop — strictly slower than sequential.
2. **`crew` mode was deleted.** Its coordinator bookended the run and the return trip to
   synthesize cost a second reload for nothing.
3. **Every downstream role shares one tag** (`qwen3:8b`), so consecutive pipeline stages reuse
   the resident weights; only the lead's `ornith` costs a load.

`max_loaded_models: 2` is advisory here — the server caps at 1 regardless. Enforce for real with
`OLLAMA_MAX_LOADED_MODELS`. The narrow working window is covered by offloading state to memory:
`remember` writes into ChromaDB, `search_knowledge_base` reads it back. To give the lead that
loop, run with `--mcp` (`lag pipeline "..." --mcp`), which backs every stage with both MCP
servers instead of only in-process tools.

Verified live: `lag pipeline "..." --agents architect_agent,dev_agent` completes in ~70s
including both cold loads, and the downstream stage visibly works from the lead's plan.

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

| Role | Tag | num_ctx | Notes |
|------|-----|--------:|-------|
| `orchestrator` | `ornith:latest` | 32768 | 9B qwen3.5, tools+thinking; plans once per run |
| `worker` | `qwen3:8b` | 8192 | analyst / biz_analyst / design / research |
| `coder` | `qwen3:8b` | 8192 | dev / tester / dev_research — same tag, so no reload |
| `fast` | `qwen3:8b` | 8192 | unassigned, free for cheap calls |
| `embed` | `nomic-embed-text` | — | 768-dim embeddings, ~0.37 GB |

Remapped after an Ollama cleanup removed `qwen3:14b`, `qwen2.5-coder:7b`, `llama3.2:3b` and
`qwen3:4b`. **Role models must support tools** — the loop is a tool-calling loop, so a
completion-only model fails at run time. Still installed but unmappable for that reason:
`example/completion-only-8b` and `example/completion-only-mini` (completion only).
`persona-model:latest` does support tools but ships a baked-in persona that fights the persona
system prompts. `example/llama-3.1-8b-instruct` supports tools but is a rough
IQ3_M quant.

---

## Commands

Mac (primary). The `env -u` prefixes are Ubuntu-only — see *Environment traps*.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv          # 3.14 lacks chromadb wheels
.venv/bin/python -m pip install -e ".[dev,rag]"

.venv/bin/lag run "what is this project?"
.venv/bin/lag run "same thing" --mcp
.venv/bin/lag pipeline "design a url shortener"

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
