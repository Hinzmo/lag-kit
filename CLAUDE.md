# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`local-agent-kit` (`lag`) — a fully offline multi-agent harness over [Ollama](https://ollama.com).
No cloud, no API keys. Agents call tools exposed over the Model Context Protocol (MCP);
retrieval (Phase 6) is just one more MCP server.

This is the **v2 rebuild**. v1 faked the tool layer with a direct-dispatch HTTP/JSON-RPC
shim; v2's whole point is a *real* MCP boundary — tools live in separate server processes,
discovered at runtime via `tools/list`, reusable by any MCP client.

## Repo state — read this before touching anything

This tree is a **partial reconstruction after the previous machine was wiped**. The source
arrived as a flat `files.zip` dump and has been moved into the layout below; `files.zip` is
kept at the root as the recovery archive. Assume gaps between what the docs claim exists and
what's actually on disk — verify before trusting either.

```
src/lag/     config.py  tools.py  ollama_client.py  mcp_client.py  swarm.py  cli.py
servers/     tools_server.py
personas/    (empty)
tests/       (empty)
```

- **No commits exist yet.** Everything is staged but uncommitted. `git config user.email` is
  unset, so commits will fail until it's configured; the current branch is `master` while
  `init.defaultBranch` is `main`. See `MORNING_BRIEF.md` §1 for the intended setup.
- **`.venv/` has only pip.** No `ollama`, `mcp`, `pyyaml`, or `typer` installed.
- **`personas/` is empty — the nine persona YAMLs were lost in the wipe.** The morning brief
  lists Phase 5 as done, but they aren't in `files.zip` and aren't on disk. `lag swarm`
  cannot run until they're rewritten. `config.load_persona` expects `personas/<name>.yaml`
  with at least `name`, `model`, and `system_prompt` keys (`swarm.Agent.__init__`); the named
  crew is `architect_agent`, `dev_agent`, `senior_dev_agent`, `design_agent`.
- **`servers/rag_server.py` does not exist.** Phase 6 is unstarted.
- **Nothing has ever been run.** The code was written in an environment with no Ollama, so
  it is syntax-checked only. Treat everything as unverified until smoke-tested.
- **`MORNING_BRIEF.md` is from mid-June 2026.** Good record of intent and of the deliberate
  trade-offs; its "today's targets" are stale, not a live plan.

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

This is the known rough edge. **If `--mcp` hangs, `MCPBridge._connect` is the first
suspect.** The cleaner refactor is keeping the whole run inside `async with` blocks.

## Commands

```bash
cp config.example.yaml config.yaml    # per-machine; gitignored
pip install -e .                      # core
pip install -e ".[rag]"               # + chromadb, Phase 6
pip install -e ".[dev]"               # + pytest

ollama pull qwen3:4b                  # or whatever `models.worker` is set to

lag run "list the files here and tell me what this project is"
lag run "same thing" --mcp            # routes through the MCP server
lag swarm "design a tiny URL shortener" --mode crew   # pipeline | parallel | crew

python servers/tools_server.py                                  # run the server standalone
npx @modelcontextprotocol/inspector python servers/tools_server.py   # poke it interactively

pytest
```

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

## Known loose ends

These are deliberate, documented in `files/MORNING_BRIEF.md` §4 — not bugs to chase blindly:

- **`web_search` is referenced in persona tool lists but not implemented.** The v1 DuckDuckGo
  tool was never rebuilt; calling it returns `"unknown tool: web_search"`.
- **`run_shell` is unsandboxed** — `shell=True`, no path allow-list, 60s timeout. Accepted
  risk on a single-user offline box. Revisit before anything multi-user. Don't silently
  "fix" this without asking; it's a deferred decision, not an oversight.
- **`thinking_mode` in personas is metadata only** — parsed but never passed to
  `ollama.chat(think=...)`. Wiring it is ~2 lines in `ollama_client.py`.
- **`mcp` SDK version churn.** The official SDK ships FastMCP as its high-level interface,
  but a v2 alpha is in flight and FastMCP 3.0 shifted decorator ergonomics. Run
  `pip show mcp` and sanity-check against any tutorial before copying its API.

## Conventions

- `from __future__ import annotations` at the top of every module; PEP 604 unions (`str | None`).
- Module docstrings explain *why the design is this way*, not what the code does. Match that.
- Tool functions need Google-style `Args:` docstrings — they become the schema descriptions
  the model reads.
- Real config (`config.yaml`), `memory/`, `data/`, and `.chroma/` are gitignored. Commit
  `config.example.yaml` only.
- Privacy stance is load-bearing here ("your data never leaves the machine"). Don't introduce
  cloud calls, telemetry, or API-key-requiring dependencies.
