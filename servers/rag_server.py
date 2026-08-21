"""RAG MCP server: retrieval over the local vector store.

The retrieval core (chunking, embedding, the ChromaDB collection) lives in
`lag.rag` because it has two consumers now — this server, and `tools.remember`,
which indexes everything an agent stores into the same collection. This file is
just the protocol wrapper and the CLI, exactly as tools_server.py wraps
lag.tools.

Usage:
    python servers/rag_server.py --ingest docs/   # load documents
    python servers/rag_server.py                  # run the server (stdio)

Why the query happens *in* the agent loop: the model decides when to retrieve,
rather than the harness stuffing context in up front. More flexible, and it
means retrieval latency lands inside the loop where you can see it.
"""
from __future__ import annotations
import argparse
import sys

try:
    # mcp >= 2.0: FastMCP was renamed MCPServer (see servers/tools_server.py).
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from lag.rag import DB_PATH, EMBED_MODEL, OLLAMA_HOST, RAGEngine, ingest

_engine = None  # initialized on startup in server mode


def search_knowledge_base(query: str, k: int = 5) -> str:
    """Search the knowledge base for relevant documents.

    Covers both ingested documents and facts stored with the remember tool.

    Args:
        query: What you're looking for.
        k: How many results to return (default 5).

    Returns:
        Formatted text with the matching chunks and their sources.
    """
    if not _engine:
        return "(RAG engine not initialized — run with --ingest first)"

    results = _engine.search(query, k=k)
    if not results:
        return "(no matching documents)"

    out = f"Found {len(results)} relevant chunks:\n\n"
    for i, c in enumerate(results, 1):
        tag = " · remembered" if c["kind"] == "memory" else ""
        out += f"## Result {i} (source: {c['source']}{tag}, similarity: {1 - c['distance']:.2f})\n"
        out += f"{c['text']}\n\n"
    return out


mcp = _Server("lag-rag")
mcp.tool()(search_knowledge_base)


def main():
    p = argparse.ArgumentParser(description="RAG MCP server")
    p.add_argument("--ingest", help="Directory of .txt / .md files to ingest")
    p.add_argument("--db", default=DB_PATH, help="Path to ChromaDB store")
    p.add_argument("--embed-model", default=EMBED_MODEL)
    p.add_argument("--ollama-host", default=OLLAMA_HOST)
    args = p.parse_args()

    if args.ingest:
        ingest(args.ingest, db_path=args.db, model=args.embed_model, host=args.ollama_host)
        return

    global _engine
    _engine = RAGEngine(db_path=args.db, model=args.embed_model, host=args.ollama_host)
    # stderr, not stdout: stdout *is* the JSON-RPC transport under the stdio
    # transport, so anything printed there corrupts the stream and the client's
    # initialize handshake fails.
    print(f"RAG engine initialized: {args.db}", file=sys.stderr)
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
