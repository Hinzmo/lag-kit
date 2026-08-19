"""RAG MCP server: retrieval-augmented generation over local documents.

Ingest side: load documents, chunk into ~500-token pieces, embed with
nomic-embed-text (via Ollama to keep everything local), store in ChromaDB.

Query side: embed the search query, retrieve k nearest neighbors, return chunks
with similarity scores.

MCP wrapper: expose search_knowledge_base(query: str, k: int = 5) as a tool.

Usage:
    python servers/rag_server.py --ingest docs/        # load docs
    python servers/rag_server.py                        # run server (ingest must happen first)

Why this design:
  - Ollama handles the embeddings (nomic-embed-text) so it's the same local flow
    as the LLM. No separate embedding service, no cloud roundtrip.
  - ChromaDB is minimal and in-process. Qdrant works too if you swap the imports.
  - The query happens *in* the agent loop (the model decides when to retrieve).
    We don't do "retrieve-then-prompt" upfront; we let the agent pull docs on
    demand. Smarter, more flexible, but means the retrieval latency matters.
  - Chunking at ~500 tokens keeps individual results small enough that the model
    can reason over them without drowning in context. Tunable.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import chromadb
import ollama
try:
    # mcp >= 2.0: FastMCP was renamed MCPServer (see servers/tools_server.py).
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

# ────────────────────────────────────────────────────────────────────────────
# INGEST: Document loading, chunking, embedding, storage
# ────────────────────────────────────────────────────────────────────────────


def load_documents(directory: str | Path) -> list[dict]:
    """Load all .txt and .md files from a directory.
    
    Each document becomes {path, name, content}.
    """
    docs = []
    for ext in ("*.txt", "*.md"):
        for p in Path(directory).glob(f"**/{ext}"):
            docs.append({
                "path": str(p),
                "name": p.stem,
                "content": p.read_text(encoding="utf-8"),
            })
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: The document text.
        chunk_size: Target tokens per chunk (rough: 1 token ≈ 4 chars).
        overlap: How many characters of the previous chunk to keep.
    
    The overlap helps preserve context across boundaries. For 500-token chunks
    at ~4 chars/token, that's ~2000 char chunks with ~400 char overlap.
    """
    char_size = chunk_size * 4  # rough heuristic
    # The stride must advance, or the loop below never terminates.
    stride = max(char_size - overlap, 1)
    chunks = []
    start = 0
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    while start < len(text):
        end = min(start + char_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            # Final chunk. Without this, `start` rewinds to end - overlap, which
            # is still < len(text), and we re-emit the same tail chunk forever.
            break
        start += stride
    return chunks


def embed_batch(texts: list[str], model: str = "nomic-embed-text",
                host: str = "http://localhost:11434") -> list[list[float]]:
    """Embed a batch of texts using Ollama.
    
    nomic-embed-text is small (~275M) and produces 768-dim embeddings.
    We batch because it's faster than one-at-a-time.
    """
    client = ollama.Client(host=host)
    embeddings = []
    for text in texts:
        resp = client.embeddings(model=model, prompt=text)
        embeddings.append(resp["embedding"])
    return embeddings


def ingest(doc_dir: str, db_path: str = "./.chroma",
           model: str = "nomic-embed-text",
           host: str = "http://localhost:11434"):
    """Load documents, chunk, embed, and store in ChromaDB.
    
    One-time operation per corpus. Call this before running the server.
    """
    print(f"Loading documents from {doc_dir} ...")
    docs = load_documents(doc_dir)
    if not docs:
        print("  (no .txt or .md files found)")
        return

    print(f"  loaded {len(docs)} documents")

    # Chunk all documents
    all_chunks = []
    chunk_metadata = []  # track which doc each chunk came from
    for doc in docs:
        chunks = chunk_text(doc["content"])
        all_chunks.extend(chunks)
        chunk_metadata.extend([{"source": doc["name"], "path": doc["path"]}] * len(chunks))

    print(f"  chunked into {len(all_chunks)} pieces")

    # Embed
    print(f"  embedding {len(all_chunks)} chunks (this takes a moment) ...")
    embeddings = embed_batch(all_chunks, model=model, host=host)

    # Store in ChromaDB
    print(f"  storing in ChromaDB at {db_path} ...")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},  # cosine distance for embeddings
    )

    # ChromaDB expects: ids, embeddings, documents, metadatas (all same length)
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=chunk_metadata,
    )
    print(f"✓ ingestion complete: {len(all_chunks)} chunks in the store")


# ────────────────────────────────────────────────────────────────────────────
# QUERY: Embedding a search query and retrieving similar chunks
# ────────────────────────────────────────────────────────────────────────────


class RAGEngine:
    """Stateful retrieval engine over a ChromaDB collection."""

    def __init__(self, db_path: str = "./.chroma", model: str = "nomic-embed-text",
                 host: str = "http://localhost:11434"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="documents")
        self.embed_model = model
        self.ollama_host = host

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for relevant chunks.
        
        Args:
            query: The search text.
            k: Number of results to return.
        
        Returns:
            List of {text, source, distance} dicts, ranked by relevance.
        """
        # Embed the query
        client = ollama.Client(host=self.ollama_host)
        query_embed = client.embeddings(model=self.embed_model, prompt=query)["embedding"]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embed],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Unpack and format results
        # ChromaDB returns: documents (list of lists), metadatas (list of lists),
        # distances (list of lists). We flatten and pair.
        chunks = []
        if results["documents"] and results["documents"][0]:
            for text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append({
                    "text": text,
                    "source": meta.get("source", "unknown"),
                    "distance": float(dist),  # lower = more similar in cosine
                })
        return chunks


# ────────────────────────────────────────────────────────────────────────────
# MCP SERVER: Expose the RAG engine as a tool
# ────────────────────────────────────────────────────────────────────────────

_engine = None  # global, initialized on startup


def search_knowledge_base(query: str, k: int = 5) -> str:
    """Search the knowledge base for relevant documents.
    
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
    
    output = f"Found {len(results)} relevant chunks:\n\n"
    for i, chunk in enumerate(results, 1):
        output += f"## Result {i} (source: {chunk['source']}, similarity: {1 - chunk['distance']:.2f})\n"
        output += f"{chunk['text']}\n\n"
    return output


# Boot the MCP server
mcp = _Server("lag-rag")
mcp.tool()(search_knowledge_base)


# ────────────────────────────────────────────────────────────────────────────
# CLI: --ingest mode vs server mode
# ────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="RAG MCP server")
    parser.add_argument("--ingest", help="Directory of .txt / .md files to ingest")
    parser.add_argument("--db", default="./.chroma", help="Path to ChromaDB store")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    args = parser.parse_args()

    if args.ingest:
        # Ingest mode: load documents and exit
        ingest(args.ingest, db_path=args.db, model=args.embed_model,
               host=args.ollama_host)
    else:
        # Server mode: run the MCP server
        global _engine
        _engine = RAGEngine(db_path=args.db, model=args.embed_model,
                           host=args.ollama_host)
        # stderr, not stdout: stdout *is* the JSON-RPC transport under the stdio
        # transport, so anything printed there corrupts the stream and the
        # client's initialize handshake fails.
        print(f"RAG engine initialized: {args.db}", file=sys.stderr)
        mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
