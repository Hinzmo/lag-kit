# local-agent-kit (`lag`)

A fully offline multi-agent harness over [Ollama](https://ollama.com). No cloud, no API
keys, no telemetry: the models, the tools and your data all stay on the machine.

Agents call tools over a real [Model Context Protocol](https://modelcontextprotocol.io)
boundary. The tools run as separate server processes, and retrieval is just one more MCP
server. Unattended work runs in a locked-down container with a filtering egress proxy.

![The console running a goal: the orchestrator routes it to agents and their work streams in live](docs/img/console.gif)

## What it does

- **Goal queue.** Write what should get done, plus an optional "done when". The
  orchestrator reads each goal, picks which specialist agents should handle it, briefs
  them, and works through the queue one goal at a time. It runs on the server, so it keeps
  going with the browser closed. Agents decide rather than stall: an answer that ends on
  a question is sent back to choose and finish.
- **Live console** (`lag serve`). Streams every stage's thinking, tool calls and answer
  as they happen. A separate activity window keeps the raw trace.
- **Pipelines** (`lag pipeline`). Staged agents, each seeing the work above it. The
  orchestrator can choose the stages itself or you can name them.
- **Agent mode** (`lag agent`). Saved agents that loop on a long-running goal, with
  circuit breakers for repeated failures and for agents that go idle.
- **Study mode** (`lag study`). Ask questions and get quizzed on your own course notes,
  with answers cited from the source.
- **Nine personas** (architect, dev, senior dev, tester, analyst, researcher and more),
  each a YAML file that names a role, not a model.

## Design decisions

The interesting parts of this project are the constraints it was built around, most of
them measured on the target machine (a 16 GB MacBook Air M4).

**One model at a time is the real limit.** Ollama on this machine keeps exactly one chat
model resident. Loading a second evicted the first every time, even with 10 GB free, and
each switch is a 4–6 second cold load. So stages run in sequence, never in parallel (an
earlier parallel mode made concurrent agents evict each other in a loop), goals are a
queue rather than a pool, and routing and briefing happen in a single orchestrator call
instead of two. Pinning every stage to one model turns a three-load run into one.

**Thinking costs time; context size doesn't.** Generation speed was flat from 8K to 128K
context, so `num_ctx` is set for memory, not speed. Reasoning tokens were the cost: personas had a
thinking switch that never reached Ollama, and one meant to skip reasoning was spending up
to 2,600 thinking tokens. Wiring the switch through brought that to zero.

**A real MCP boundary, not a shim.** The first version faked tool calls over a direct
HTTP shim. Now tools live in their own processes, are discovered at runtime, and work
with any MCP client. The loop accepts anything with `schemas()` and `call()`, so the
in-process provider, the MCP bridge and the sandbox are wrappers that compose without
the loop knowing which are present.

**Retrieval is just a tool.** The RAG server exposes one `search_knowledge_base` tool.
To the agent it looks like any other call.

**Model capabilities are discovered, not hardcoded.** The harness asks Ollama whether a
model supports tools and thinking. A completion-only model runs without tools and says
so instead of failing.

**Unattended work is confined, and fails closed.** Agent mode and the goal queue run
shells in a Docker or Podman container:

- the project is mounted read-only, with one writable workspace
- all capabilities are dropped, and pids, memory and CPU are capped
- the only route out is a per-agent proxy on an internal network that allows the public
  internet but refuses the host, the LAN and the tailnet
- no container runtime means no run

## Architecture

```
        ┌─────────────────────┐
        │  Ollama (local LLM) │   one chat model resident at a time
        └──────────┬──────────┘
                   │  tools ↓   tool_calls ↑
        ┌──────────┴──────────┐        ┌────────────────────────────┐
        │  Harness  (lag)     │◄──────►│  Console  (lag serve)       │
        │  ReAct loop         │  SSE   │  goal queue · live stream   │
        │  pipeline · goals   │        └────────────────────────────┘
        │  agent mode         │
        └──────────┬──────────┘
                   │  stdio · JSON-RPC (MCP)       ┌──────────────────────────┐
        ┌──────────┴───────────────────────┐       │ Sandbox (unattended runs)│
        │  MCP servers (own processes)      │      │ container · ro project   │
        │   • tools_server  16 tools        │─────►│ egress proxy · fails     │
        │   • rag_server    knowledge base  │      │ closed without Docker    │
        └──────────────┬────────────────────┘      └──────────────────────────┘
                       ▼
           Chroma · nomic-embed-text
```

Each turn, the harness lists the servers' tools, translates their schemas into Ollama's
tool-calling format, sends the conversation, routes any `tool_calls` back over MCP, and
loops until the model answers.

## Quick start

Needs [Ollama](https://ollama.com) and Python 3.12. Docker Desktop is needed for agent
mode and the goal queue.

```bash
git clone https://github.com/Hinzmo/local-agent-kit.git && cd local-agent-kit
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,rag,web,study]"

cp config.example.yaml config.yaml        # map the roles to models from `ollama list`
ollama pull qwen3:8b && ollama pull nomic-embed-text
.venv/bin/python servers/rag_server.py --ingest docs/

.venv/bin/lag serve                       # console at http://127.0.0.1:8765
```

Use Python 3.12: macOS ships 3.14 as `python3`, and `chromadb` has no 3.14 wheel yet.
Run `lag` from the repo root, because its state folders are relative to the working
directory.

## Commands

```bash
lag run "what is this project?"           # one agent, local tools
lag run "same thing" --mcp                # same tools, over the MCP boundary
lag pipeline "design a url shortener"     # staged agents
lag serve                                 # web console and goal queue
lag agent new                             # then: list | start | status | stop
lag study add <course> <path>             # then: ask | quiz | cards | review
```

## Models

Bring your own. Personas name a role, and `config.yaml` maps each role to any model from
your `ollama list`. Nothing is hardcoded: the console's model picker reads your installed
models live from Ollama, and asks each one whether it supports tools and thinking.

| Role | Used by | Example |
|------|---------|---------|
| orchestrator | architect, senior dev (plans and routes) | your strongest tool-capable model |
| worker, coder, fast | the other agents | `qwen3:8b` |
| embed | retrieval only | `nomic-embed-text` |

Pick models that support tools (`ollama show <tag>`). Pointing the downstream roles at
one model means no reload between stages. The measurements above used a 9B Qwen 3.5
build as the orchestrator and `qwen3:8b` for the rest.

## Security notes

- **Supervised runs are not sandboxed.** `lag run` and `lag pipeline` use the host shell,
  by design, on a single-user machine. Only unattended work (agent mode and goals) is
  confined.
- **Two tools reach the internet:** `web_search` (keyless DuckDuckGo) and `http_fetch`.
  `http_fetch` only allows public addresses and re-checks every redirect.
- **Sandboxed agents reach the public internet by default**, through the proxy. Create
  an agent with `--no-network` to seal it.

## Tests

470 tests: the loop, the MCP bridge, the sandbox and egress proxy (with real containers
when Docker is running), the goal queue, and the console. The console is checked in jsdom.

```bash
.venv/bin/python -m pytest
cd tests/js && npm install                # enables the console tests
```

`STATUS.md` has the full current state, measurements and what's left.

## License

MIT. See [LICENSE](LICENSE).
