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

| Phase | What | Status |
|------:|------|--------|
| 0 | Foundation + git | ✅ this commit |
| 1 | Core ReAct loop over Ollama, hardcoded tools | ☐ |
| 2 | Real MCP server (FastMCP, stdio) | ☐ |
| 3 | Harness as MCP client — the bridge | ☐ |
| 4 | Swarm orchestrator (pipeline/parallel/crew) | ☐ |
| 5 | Nine personas | ☐ |
| 6 | RAG MCP server | ☐ |

---

## Setup

```bash
cp config.example.yaml config.yaml      # edit models for this machine
pip install -e .                         # core
pip install -e ".[rag]"                  # + vector store, when you reach Phase 6
```

> **Version note (mid-2026):** the official `mcp` Python SDK ships FastMCP as its
> high-level interface, but there's a v2 alpha in flight and FastMCP 3.0 shifted some
> decorator ergonomics. Check `pip show mcp` against any tutorial before copying it.

## Models

Set per machine in `config.yaml`. The MacBook Air M4 can run a `qwen3:14b`
orchestrator comfortably; the CPU-only ASUS should stay on smaller tool-calling
models. Avoid R1 distills as orchestrators — weak at structured output, and the
thinking tokens kill loop latency.
