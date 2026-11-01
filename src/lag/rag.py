"""Retrieval core: chunking, embedding, and the ChromaDB collection.

This lives in the package rather than in servers/rag_server.py because it now has
two consumers — the RAG MCP server, and `tools.remember`, which writes every fact
an agent stores into the same vector space so it becomes semantically
retrievable later. Same rule as tools.py: one implementation, two consumers.

Everything here is optional at import time. `chromadb` ships in the `[rag]`
extra, so `lag.tools` imports this module lazily and degrades to plain key/value
memory when it is absent — a core install must never break because retrieval
isn't installed.

Design notes carried over from the original server:
  - Ollama does the embedding (nomic-embed-text) so the flow stays local; no
    separate embedding service, no cloud roundtrip.
  - Cosine distance, because embeddings encode direction rather than magnitude.
  - ~500-token chunks keep individual results small enough for the model to
    reason over without drowning in context.
"""
from __future__ import annotations
import hashlib
import re
import time
from pathlib import Path

import chromadb
import ollama

DB_PATH = "./.chroma"
COLLECTION = "documents"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_HOST = "http://127.0.0.1:11434"


# ── chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The document text.
        chunk_size: Target tokens per chunk (rough: 1 token ≈ 4 chars).
        overlap: How many characters of the previous chunk to keep.

    The overlap preserves context across boundaries. For 500-token chunks at
    ~4 chars/token that's ~2000-char chunks with ~100 chars of carry-over.
    """
    char_size = chunk_size * 4  # rough heuristic
    # The stride must advance, or the loop below never terminates.
    stride = max(char_size - overlap, 1)
    chunks: list[str] = []
    start = 0
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    while start < len(text):
        end = min(start + char_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            # Final chunk. Without this, `start` rewinds to end - overlap, which
            # is still < len(text), and we re-emit the same tail forever.
            break
        start += stride
    return chunks


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


# ── embedding + storage ─────────────────────────────────────────────────────

def embed_batch(texts: list[str], model: str = EMBED_MODEL,
                host: str = OLLAMA_HOST) -> list[list[float]]:
    """Embed a batch of texts using Ollama.

    nomic-embed-text is small (~275M) and produces 768-dim embeddings.
    """
    client = ollama.Client(host=host)
    return [client.embeddings(model=model, prompt=t)["embedding"] for t in texts]


def get_collection(db_path: str = DB_PATH, collection: str = COLLECTION):
    """Open (or create) a collection with cosine distance.

    Collections are the isolation boundary. The harness uses one shared
    collection; `lag study` gives each course its own, so a question about
    organic chemistry can't retrieve last term's history notes.
    """
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name=collection,
        metadata={"hnsw:space": "cosine"},
    )


def list_collections(db_path: str = DB_PATH) -> list[str]:
    """Names of every collection in the store."""
    try:
        return [c.name for c in chromadb.PersistentClient(path=db_path).list_collections()]
    except Exception:  # noqa: BLE001 - a missing/locked store is not fatal here
        return []


def _ids_for(texts: list[str], prefix: str) -> list[str]:
    """Content-addressed ids.

    The original ingest used ids of the form `chunk_<i>`, which collide across
    runs: a second ingest silently overwrites the first, and anything else
    writing to the collection (now `remember`) clobbers whatever shared its
    index. Hashing the content makes re-adding the same text idempotent and
    keeps separate writers from colliding.
    """
    return [f"{prefix}_{hashlib.sha1(t.encode('utf-8')).hexdigest()[:16]}" for t in texts]


def add_texts(texts: list[str], metadatas: list[dict], *, prefix: str = "chunk",
              db_path: str = DB_PATH, collection: str = COLLECTION,
              model: str = EMBED_MODEL, host: str = OLLAMA_HOST) -> int:
    """Embed texts and upsert them into a collection. Returns the count."""
    if not texts:
        return 0
    embeddings = embed_batch(texts, model=model, host=host)
    get_collection(db_path, collection).upsert(
        ids=_ids_for(texts, prefix),
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(texts)


def index_note(key: str, value: str, *, db_path: str = DB_PATH,
               model: str = EMBED_MODEL, host: str = OLLAMA_HOST) -> int:
    """Index a remembered fact so it is retrievable by meaning, not just by key.

    `recall` needs the exact key. Anything an agent stores should also surface
    from a plain question, which means it has to live in the vector store too.
    The key is prepended to the embedded text so the label carries semantic
    weight rather than sitting only in metadata.
    """
    body = f"{key}: {value}"
    chunks = chunk_text(body)
    meta = [{
        "source": f"memory/{key}",
        "path": "memory/store.json",
        "kind": "memory",
        "key": key,
        "stored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }] * len(chunks)
    return add_texts(chunks, meta, prefix="mem", db_path=db_path, model=model, host=host)


def ingest(doc_dir: str, db_path: str = DB_PATH, model: str = EMBED_MODEL,
           host: str = OLLAMA_HOST) -> int:
    """Load documents, chunk, embed, and store. Returns chunks written."""
    print(f"Loading documents from {doc_dir} ...")
    docs = load_documents(doc_dir)
    if not docs:
        print("  (no .txt or .md files found)")
        return 0
    print(f"  loaded {len(docs)} documents")

    texts, metas = [], []
    for doc in docs:
        chunks = chunk_text(doc["content"])
        texts.extend(chunks)
        metas.extend([{"source": doc["name"], "path": doc["path"], "kind": "document"}] * len(chunks))

    print(f"  chunked into {len(texts)} pieces")
    print(f"  embedding {len(texts)} chunks (this takes a moment) ...")
    print(f"  storing in ChromaDB at {db_path} ...")
    n = add_texts(texts, metas, prefix="chunk", db_path=db_path, model=model, host=host)
    print(f"✓ ingestion complete: {n} chunks written")
    return n


# ── query ───────────────────────────────────────────────────────────────────

class RAGEngine:
    """Stateful retrieval engine over the shared collection."""

    def __init__(self, db_path: str = DB_PATH, model: str = EMBED_MODEL,
                 host: str = OLLAMA_HOST, collection: str = COLLECTION):
        self.collection = get_collection(db_path, collection)
        self.embed_model = model
        self.ollama_host = host

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for relevant chunks.

        Returns a list of {text, source, kind, distance}, ranked by relevance
        (lower cosine distance = more similar).
        """
        client = ollama.Client(host=self.ollama_host)
        q = client.embeddings(model=self.embed_model, prompt=query)["embedding"]
        n = min(k, max(self.collection.count(), 1))
        res = self.collection.query(
            query_embeddings=[q], n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        if res["documents"] and res["documents"][0]:
            # Chroma returns these as parallel arrays. strict=True because silently
            # truncating to the shortest would drop search results with no signal —
            # the failure would look like "retrieval found less", not like a bug.
            for text, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                        res["distances"][0], strict=True):
                out.append({
                    "text": text,
                    "source": (meta or {}).get("source", "unknown"),
                    "kind": (meta or {}).get("kind", "document"),
                    "distance": float(dist),
                })
        return out
