# local-agent-kit (`lag`)

A fully offline multi-agent harness over [Ollama](https://ollama.com). No cloud, no
API keys — your data never leaves the machine. Agents call tools exposed over the
Model Context Protocol (MCP), and retrieval is just one more MCP server.

> **v2 rebuild.** The original kit faked the tool layer with a direct-dispatch
> HTTP/JSON-RPC shim to avoid protocol overhead. v2 replaces that with a real MCP
> boundary: tools live in separate server processes, get discovered at runtime, and
> are reusable by any MCP client (this harness, Claude Desktop, Cursor, …).

---

## Architecture

Three decoupled layers. The **harness** in the middle is the only piece that knows
about everything else — nothing else knows about each other. That's the whole point.

```
        ┌─────────────────────┐
        │  Ollama (local LLM) │  serves the model, tool-calling enabled
        └──────────┬──────────┘
                   │  tools ↓   tool_calls ↑
        ┌──────────┴──────────┐
        │  Harness  (lag)     │  ReAct loop + MCP client
        │  - drives the model │
        │  - bridges schemas  │
        └──────────┬──────────┘
                   │  stdio · JSON-RPC (MCP)
        ┌──────────┴───────────────────────────┐
        │  MCP servers (separate processes)     │
        │   • tools_server  (fs, shell, search) │
        │   • rag_server    (search_knowledge…) │──► Vector DB + embedder
        └───────────────────────────────────────┘        (Chroma · nomic-embed-text)
```

**The bridge** is the heart of the harness. Each loop iteration:

1. Connect to MCP servers, call `tools/list` to discover what they expose.
2. Translate those MCP tool schemas into Ollama's tool-calling format.
3. Send the conversation + tools to the model.
4. If the model emits `tool_calls`, route each one back via `tools/call`.
5. Append the result as a `role: "tool"` message and loop until the model is done.

**RAG is not special.** It's one MCP server exposing a single `search_knowledge_base`
tool. Internally it embeds the query, hits the vector DB, and returns chunks — but to
the agent it looks identical to any other tool call.

---

## Layout

```
local-agent-kit/
├── pyproject.toml
├── config.example.yaml      # copy → config.yaml (gitignored), edit per machine
├── README.md
├── src/lag/
│   ├── cli.py               # `lag` entry point
│   ├── config.py            # loads config.yaml
│   ├── ollama_client.py     # OllamaClient + ReAct loop   (Phase 1)
│   ├── tools.py             # the actual tool functions    (Phase 1)
│   ├── mcp_client.py        # the bridge: MCP client side  (Phase 3)
│   └── swarm.py             # pipeline / parallel / crew   (Phase 4)
├── servers/
│   ├── tools_server.py      # FastMCP server wrapping tools (Phase 2)
│   └── rag_server.py        # RAG MCP server                (Phase 6)
├── personas/                # nine agent YAMLs              (Phase 5)
└── tests/
```

---

## Build roadmap

All six phases are complete and smoke-tested end to end (105 tests passing).

| Phase | What | Status |
|------:|------|--------|
| 0 | Foundation + git | ✅ |
| 1 | Core ReAct loop over Ollama, local tools | ✅ |
| 2 | Real MCP server (stdio) | ✅ |
| 3 | Harness as MCP client — the bridge | ✅ |
| 4 | Swarm orchestrator (pipeline/parallel/crew) | ✅ |
| 5 | Nine personas | ✅ |
| 6 | RAG MCP server + always-on retrieval | ✅ |

See `STATUS.md` for current state and what's left.

---

## Setup

Primary dev box is a **MacBook Air M4**; a CPU-only Ubuntu machine is the secondary
box (its extra environment traps are documented in an appendix in `STATUS.md`).

```bash
cp config.example.yaml config.yaml       # edit models for this machine

# Build the venv on Python 3.12 — chromadb (the [rag] extra) has no 3.14 wheels yet,
# and macOS ships 3.14 as `python3`. Homebrew: `brew install python@3.12`.
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,rag]"   # core + tests + vector store
```

> **`mcp` SDK version:** this repo runs on **mcp 2.0.0**, which renamed the high-level
> server class `FastMCP` → `mcp.server.mcpserver.MCPServer` and `Tool.inputSchema` →
> `input_schema`. Both servers and the bridge import either spelling via `try/except`,
> so the kit also runs on mcp 1.x. Most tutorials are still on 1.x — check `pip show mcp`.

## Models

Set per machine in `config.yaml` (roles → tags; personas name only a role). The M4
runs a `qwen3:14b` orchestrator comfortably and `num_ctx` up to 32768 cheaply; the
CPU-only box stays on smaller models and a tighter window. Avoid R1 distills as
orchestrators — weak at structured output, and the thinking tokens kill loop latency.
