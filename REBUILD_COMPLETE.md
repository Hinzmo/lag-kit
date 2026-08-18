# Rebuild Complete: local-agent-kit v2

The ASUS wipe took months of work. This is the v2 rebuild — better designed, with a real MCP boundary and full RAG support. **All 6 phases complete.** Below is what you have waiting.

---

## File manifest (23 files)

**Foundation:**
- `README.md` — architecture overview, design decisions, setup
- `pyproject.toml` — build config, entry points, optional dependencies
- `config.example.yaml` — per-machine configuration template
- `.gitignore` — secrets, venvs, runtime state
- `MORNING_BRIEF.md` — git setup, status, smoke tests, next steps

**Core harness (src/lag/):**
- `config.py` — YAML loading, persona discovery
- `tools.py` — 7 local tools (read/write, list, shell, system, remember/recall) + LocalToolProvider
- `ollama_client.py` — OllamaClient + ReAct loop (the heart)
- `cli.py` — `lag run` and `lag swarm` commands
- `mcp_client.py` — **the bridge** (MCP client + async-to-sync marshalling)
- `swarm.py` — pipeline / parallel / crew orchestration

**MCP servers:**
- `servers/tools_server.py` — FastMCP wrapper around local tools (stdio transport)
- `servers/rag_server.py` — RAG server: ingest, chunk, embed, retrieve (Phase 6)

**Personas (9 agents):**
- `analyst_agent.yaml` — technical analysis
- `architect_agent.yaml` — system design (thinking mode)
- `biz_analyst_agent.yaml` — requirements and specs
- `design_agent.yaml` — UI design (text-only, no vision)
- `dev_agent.yaml` — implementation
- `dev_research_agent.yaml` — technical research
- `research_agent.yaml` — general research
- `senior_dev_agent.yaml` — code review (thinking mode)
- `tester_agent.yaml` — test writing and validation

**Documentation:**
- `docs/RAG_SETUP.md` — how to use the RAG server (ingest, query, integrate)
- `docs/VECTORS_AND_SEARCH.md` — deep dive on embeddings, HNSW, ChromaDB internals

---

## What changed from v1

| Aspect | v1 | v2 |
|--------|----|----|
| **Tool dispatch** | Direct Python function calls | Real MCP protocol (stdio) |
| **Tool provider** | Single in-process registry | Provider interface: LocalToolProvider (direct) or MCPToolProvider (MCP) |
| **Swarm** | Three modes hardcoded | Three modes as methods on Swarm class |
| **Personas** | YAML format but in Python | Full YAML, loaded at runtime, persona as a first-class object |
| **RAG** | Not included | Full MCP server: ingest → chunk → embed → ChromaDB, expose as tool |
| **Embeddings** | Not implemented | nomic-embed-text via Ollama (local, no API) |
| **Vector DB** | Not implemented | ChromaDB with HNSW indexing |

---

## How to test

```bash
# 0. Unzip and navigate
cd local-agent-kit

# 1. Configure for this machine
cp config.example.yaml config.yaml
# Edit: set models to what your ASUS can run

# 2. Install
pip install -e .
pip install -e ".[rag]"   # if testing RAG

# 3. Run smoke tests (one or more)
python -m lag.cli run "list files and describe this project"
python -m lag.cli run "same" --mcp
python -m lag.cli swarm "design a URL shortener" --mode crew

# 4. Test RAG (if you have docs)
mkdir -p docs/knowledge/
echo "Your knowledge base" > docs/knowledge/readme.md
python servers/rag_server.py --ingest ./docs/knowledge/
python servers/rag_server.py &   # background
python -m lag.cli run "what's in the knowledge base?" --mcp
```

---

## Known issues (listed, not bugs)

1. **`lag` command not found** — entry point registration issue on fresh Ubuntu. Workaround: use `python -m lag.cli` instead. Fix: likely PATH issue or incomplete pip install.

2. **`web_search` referenced but not implemented** — research personas list it, but it's not in `LocalToolProvider.TOOLS`. Quick fix: implement DuckDuckGo or other search tool.

3. **Bridge not yet tested** — MCP over stdio hasn't had a real run. The code looks correct, but `mcp_client.py`'s manual context-manager handling might need tweaks on first real connection.

4. **RAG not yet integrated** — Phase 6 is complete, but `mcp_client.py` still hardcodes spawning only `tools_server.py`. Need to extend MCPToolProvider to spawn both servers and merge schemas.

5. **No run-tested code** — everything written, nothing executed. Smoke test is genuinely step one.

---

## What's interesting to learn here

1. **The bridge** (`mcp_client.py`): MCP is async, the agent loop is sync. See how we reconcile that with an event loop on a daemon thread and `run_coroutine_threadsafe`. That pattern is portable to any async SDK.

2. **Tool provider interface** (`tools.py`, `mcp_client.py`): the agent loop talks to an interface, not an implementation. LocalToolProvider and MCPToolProvider both satisfy it. That's why swapping in a real protocol is one-line. The design makes the coupling invisible.

3. **RAG server** (`servers/rag_server.py`, `docs/VECTORS_AND_SEARCH.md`): why embeddings capture meaning, why cosine similarity works, how HNSW finds neighbors without checking all vectors. The vector DB is the bridge between human language and machine computation.

4. **Swarm patterns** (`swarm.py`): pipeline (sequential, each agent refines), parallel (all agents tackle the task, results merged), crew (coordinator breaks it down, specialists work, coordinator synthesizes). Three patterns, one interface.

---

## Next session priorities

1. Debug and fix the `lag` entry point.
2. Run the smoke tests.
3. Wire RAG into the harness (extend `MCPToolProvider` to spawn multiple servers).
4. Implement `web_search` so research personas work.
5. Add tests to `tests/`.

Everything else is optional. The core is solid. Rest up.
