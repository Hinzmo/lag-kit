# RAG Setup and Usage

The RAG server is Phase 6 — one more MCP server exposing a single `search_knowledge_base` tool. To the agent, it's indistinguishable from `read_file` or `run_shell`. The magic is in the decoupling: RAG lives in its own process, gets discovered like any other server, and you can swap ChromaDB for Qdrant without touching the harness.

---

## Architecture

```
Your documents (.txt, .md)
    ↓ [ingest]
chunk → embed → store (ChromaDB)
    ↓
RAG server (search_knowledge_base tool)
    ↓ [stdio · JSON-RPC (MCP)]
Harness
    ↓
Agent calls search_knowledge_base("what is X?")
    ↓ [embed query, retrieve chunks, return as text]
Back to agent, treated like any other tool result
```

The embedding model (`nomic-embed-text` via Ollama) means no external API calls — everything stays local and offline.

---

## Step-by-step

### 1. Prepare your documents

Gather `.txt` or `.md` files into a directory:

```bash
mkdir -p docs/
# Copy or write your documents here
echo "Your knowledge base goes here" > docs/readme.md
```

### 2. Ingest (one-time)

The server runs in two modes: ingest mode (loads and processes docs) and server mode (runs as an MCP server).

```bash
# Make sure Ollama is running and has nomic-embed-text
ollama pull nomic-embed-text

# Ingest your documents (creates .chroma/ with the vector store)
python servers/rag_server.py --ingest ./docs

# The output should look like:
#   Loading documents from ./docs ...
#   loaded 3 documents
#   chunked into 47 pieces
#   embedding 47 chunks (this takes a moment) ...
#   storing in ChromaDB at ./.chroma ...
#   ✓ ingestion complete: 47 chunks in the store
```

You only need to do this once per corpus (or whenever you update the docs). The `.chroma/` directory is your index — gitignore it, back it up, move it between machines if needed.

### 3. Run the server

In a separate terminal (or background):

```bash
python servers/rag_server.py
```

This starts the MCP server on stdio. It reads from `.chroma/` and waits for queries.

### 4. Point the harness at it

When you run `lag run "..." --mcp`, the harness looks for `servers/tools_server.py` by default. To include RAG as well, you need to point at both servers. The simplest approach: update the harness to spawn *both* servers.

For now, a quick manual test:

```bash
# In a third terminal, test the search directly via MCP Inspector
npx @modelcontextprotocol/inspector python servers/rag_server.py

# Or via the harness (this is the real test):
python -m lag.cli run "What do you know about X?" --mcp
```

If the agent calls `search_knowledge_base`, it'll retrieve relevant chunks and include them in the response.

---

## How it works: the deep dive

### Chunking

Text is split into overlapping pieces (~500 tokens, ~2000 characters each, with ~100 char overlap). The overlap preserves context across chunk boundaries so retrieval doesn't lose coherence.

Why 500 tokens? Small enough that the model can reason over a single result without drowning in context, large enough that it's not fragmented. Tunable in the code.

### Embedding

Each chunk gets embedded using `nomic-embed-text` (768-dim). The query also gets embedded the same way. Then ChromaDB finds the k nearest neighbors using cosine distance.

Why nomic-embed-text? It's small (~275M), fast on CPU, and good enough for most RAG tasks. If you need semantic precision, swap to `voyage-large` (requires API), but that breaks the offline constraint.

### Storage

ChromaDB is minimal, in-process, persistent. It stores:
- `documents`: the actual text chunks
- `embeddings`: the 768-dim vectors
- `metadatas`: source document name and file path (for the agent to cite correctly)

It uses HNSW (hierarchical navigable small world) under the hood — a fast approximate nearest-neighbor search that scales to millions of chunks on a laptop.

### Query

When the agent calls `search_knowledge_base(query, k)`:
1. The query gets embedded (one Ollama call).
2. ChromaDB returns the k nearest chunks + their distance scores.
3. The chunks are formatted as text with source citations and similarity scores.
4. The text is appended to the agent's context, and it continues reasoning.

The agent *decides* when to retrieve. It's not "retrieve-then-answer" upfront; it's "if I need docs, ask for them." That's smarter than traditional RAG because the agent can reason over what it knows first, then pull docs only if necessary.

---

## Wiring RAG into the harness (next step)

Currently, `mcp_client.py` hardcodes spawning `servers/tools_server.py`. To include RAG:

```python
# In mcp_client.py, MCPToolProvider.from_config():
# Instead of one server, spawn two:

@classmethod
def from_config(cls, cfg: dict):
    # Point at both servers
    bridges = [
        MCPBridge("python", ["servers/tools_server.py"]),
        MCPBridge("python", ["servers/rag_server.py"]),
    ]
    return cls(bridges)  # extend MCPToolProvider to hold multiple bridges
```

Then merge the tool schemas from both servers. Straightforward refactor, not conceptually new.

---

## Tuning

- **Chunk size** (`chunk_text`): 500 tokens is a good start. Larger = more context per result, fewer results needed; smaller = more precise but fragmented.
- **Overlap** (`chunk_text`): 100 chars is fine. Increase if chunks feel disconnected.
- **k** (`search_knowledge_base`): 5 results is the default. The agent might not use all of them; it's a budget.
- **Embedding model**: If latency matters, `nomic-embed-text` is fast. If accuracy matters, there are larger open-source models, but they'll be slower.
- **Distance threshold**: Currently we return the top k regardless of distance. You could filter: only return results below a similarity threshold. This is a guard against retrieving irrelevant chunks.

---

## Gotchas

- **Ollama must have the embed model**: `ollama pull nomic-embed-text` before you ingest or run the server.
- **The `.chroma/` directory is your index**: if you delete it, you lose everything. Back it up.
- **Ingestion is serial, one chunk at a time**: if you have thousands of documents, it'll take a while. Not a blocker, just slow.
- **The agent doesn't know how old the knowledge is**: if you ingest static docs, the agent can confidently answer questions about them, but won't know they're from 2024. You can add doc metadata (date, version) to the chunks during ingest if precision matters.

---

## Example: a research agent with RAG

```bash
# 1. Set up a knowledge base of research papers
mkdir -p docs/research/
# ... copy your papers in as .md or .txt ...

# 2. Ingest
python servers/rag_server.py --ingest ./docs/research

# 3. Run the server (background or separate terminal)
python servers/rag_server.py &

# 4. Ask the agent
python -m lag.cli run "Based on the knowledge base, what are the key findings about X?" --mcp

# The agent will:
#   - See it doesn't know the answer
#   - Call search_knowledge_base("key findings about X")
#   - Get back relevant chunks from your papers
#   - Synthesize them into a coherent answer
```

That's the loop. RAG becomes just another tool the agent reaches for when it needs external knowledge.
