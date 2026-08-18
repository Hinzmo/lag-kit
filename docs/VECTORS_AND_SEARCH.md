# Vector Embeddings and Similarity Search: The mechanics

This is the part that looks magical until you see it. RAG works because:

1. Text → dense vector (embedding)
2. Vectors are points in high-dimensional space
3. Similar text = nearby points
4. Nearest-neighbor search finds relevant chunks in milliseconds

Here's how it actually works.

---

## What is an embedding?

An embedding is a vector of numbers that represents the *meaning* of text in high-dimensional space.

**Example (simplified):**

```
"dog" → [0.1, -0.5, 0.8, 0.2, ...]  (768 numbers for nomic-embed-text)
"cat" → [0.15, -0.48, 0.82, 0.19, ...]
"car" → [-0.8, 0.3, -0.1, 0.9, ...]
```

Words with similar meaning have similar vectors. "dog" and "cat" are closer to each other than to "car".

---

## How embeddings capture meaning

The embedding model (nomic-embed-text, trained on the internet) learns to map text to vectors such that:
- Semantic similar text → nearby vectors
- Dissimilar text → distant vectors

It does this via contrastive learning: shown pairs like ("dog", "puppy") the model learns they should have close embeddings, and pairs like ("dog", "engine") should be far apart.

The model itself is just a neural net:

```
Text input
  ↓
Tokenize (break into subword units)
  ↓
Embedding lookup (each token gets a learned vector)
  ↓
Transformer layers (attend, mix, refine)
  ↓
Pool/aggregate (combine token vectors into one)
  ↓
Normalize (make all vectors unit length, roughly)
  ↓
Output: 768-dimensional vector
```

The 768 dimensions are arbitrary and learned. You could train a model to use 256 dims or 2048 dims; 768 is a sweet spot of expressiveness vs. storage cost.

---

## Cosine similarity

When you search, you embed the query the same way. Then you need a distance metric — how "far apart" are two vectors?

We use **cosine similarity**: the angle between two vectors.

```
Given query vector q and chunk vector c:

similarity(q, c) = (q · c) / (|q| × |c|)

Where:
  · = dot product (sum of element-wise multiplication)
  |q|, |c| = magnitudes (length of vectors)
```

It returns a number between -1 and 1 (or 0 and 1 after normalization):
- 1 = vectors point in the same direction (identical meaning)
- 0 = perpendicular (unrelated)
- -1 = opposite (antonyms, roughly)

**Why cosine?** It's invariant to vector magnitude, so a short sentence and a long paragraph about the same topic still score high. And it's fast — just a few multiplications per vector.

In ChromaDB, when you configure `{"hnsw:space": "cosine"}`, it means: "use cosine distance for all neighbor searches."

---

## Vector search: the naive way

```python
def search_naive(query_vector, all_chunk_vectors, k=5):
    distances = []
    for i, chunk_vec in enumerate(all_chunk_vectors):
        dist = cosine_similarity(query_vector, chunk_vec)
        distances.append((i, dist))
    
    # Sort by distance, return top k
    distances.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in distances[:k]]
```

Simple: compute distance to every chunk, sort, return top k. Correct, but slow — O(n) where n is number of chunks. If you have a million chunks, you do a million similarity computations. Fine for <100k chunks on a laptop, but doesn't scale.

---

## Vector search: the fast way (HNSW)

ChromaDB uses **HNSW** (Hierarchical Navigable Small World) under the hood. It's an approximate nearest-neighbor algorithm that trades perfect accuracy for speed.

**The idea:**

Build a graph where each vector is a node, and edges connect nearby vectors:

```
    [vec_5]
      / \
  [vec_2]-[vec_7]
    /   \ /
[vec_0]-[vec_3]
  \
   [vec_8] ← far away
```

Then, searching for the nearest neighbor to a query:

1. Start at a random node
2. Check its neighbors; move to the closest one
3. Repeat until no neighbor is closer (local optimum)
4. Return that node + its neighbors

This is much faster than checking all vectors: you jump through the graph in ~log(n) hops instead of linear.

The "hierarchical" part: HNSW builds the graph in layers, so you can start at a coarser layer (faster, less precise) and zoom in to finer layers. Skips most of the vectors entirely.

**Trade-off:** approximate, not exact. Might miss the single best match, but gets the top 5 close neighbors correctly 99% of the time. The speedup is worth it.

---

## How ChromaDB uses HNSW

When you call `collection.add()`:

```python
collection.add(
    ids=[...],
    embeddings=[[vec1], [vec2], ...],  # your 768-dim vectors
    documents=[...],
    metadatas=[...],
)
```

ChromaDB:
1. Stores the embeddings in HNSW index (builds the graph).
2. Stores the documents and metadata in a separate table.
3. Writes both to disk in the `.chroma/` directory.

When you call `collection.query()`:

```python
results = collection.query(
    query_embeddings=[[query_vec]],
    n_results=5,
)
```

ChromaDB:
1. Runs HNSW search on the embedding index (fast — milliseconds).
2. Returns the 5 nearest-neighbor IDs.
3. Looks up the actual documents and metadata.
4. Returns them to you.

---

## Practical numbers

On a MacBook Air with ~10k documents (47k chunks):

- Ingestion: ~2 minutes (bottleneck is embedding, which is serial).
- Query latency: ~100ms (embedding the query is ~50ms, HNSW search is ~10ms, lookup is negligible).

At 100k chunks, query latency is still ~100ms (HNSW scales well). At 1M chunks, it degrades to ~500ms, but it's still usable.

---

## Why this matters for RAG

The embedding model and search algorithm are the foundation. If they're slow or inaccurate, the whole system fails:

- **Slow embedding**: every query becomes sluggish (wait 5 seconds just to embed your search).
- **Slow search**: even with a good embedding, if finding the neighbors is slow, the agent waits.
- **Inaccurate embedding**: relevant chunks score low, garbage chunks rank high, the agent gets bad context.

For local RAG, we optimize:

- **Model**: nomic-embed-text is small and fast (no 7B parameter bloat).
- **Algorithm**: HNSW is approximate but extremely fast.
- **Storage**: ChromaDB is minimal and in-process (no network latency).
- **Batching**: we could embed multiple chunks in parallel, but currently don't (quick win if latency becomes a problem).

---

## Limitations and extensions

**What breaks this design:**

1. **Domain-specific terms**: if your docs use jargon the embedding model didn't see often, similarity search struggles. (Solution: fine-tune the embedding model on your domain, or use hybrid search — vector + BM25 keyword matching.)

2. **Semantic drift**: "bank" (financial) vs. "bank" (riverbank) should be different, but a pure embedding model might conflate them. (Solution: add metadata filters, or use reranking.)

3. **Long-range dependencies**: embeddings work on individual chunks. If relevant info spans multiple non-adjacent chunks, search might miss it. (Solution: overlap chunks more, or use a reader model that can look at multiple chunks together.)

4. **Recency**: once ingested, the knowledge base is static. If your docs are a year old, the agent doesn't know. (Solution: re-ingest periodically, or use a time-based metadata filter.)

---

## Extensions you could build

1. **Hybrid search**: combine vector similarity (semantic) with BM25 keyword matching (exact match). ChromaDB and Qdrant both support this.

2. **Reranking**: retrieve top 50 via HNSW, then use a smaller but more powerful model to rerank the top 5. Improves accuracy with minimal latency cost.

3. **Metadata filtering**: `collection.query(..., where={"source": "report_2024.md"})` — search only within specific documents.

4. **Fine-tuned embeddings**: if you have labeled examples of "relevant" vs. "irrelevant" pairs, fine-tune nomic-embed-text on your domain. Better accuracy, still local.

5. **Multi-stage retrieval**: retrieve coarse chunks first, then retrieve fine-grained chunks *within* the coarse results. Speeds up search on very large corpora.

6. **Graph-based retrieval**: instead of flat chunks, index your docs as a knowledge graph (entities and relationships), and use graph search + vector search together. Much harder, but captures structure better.

---

## The bigger picture

Embeddings and vector search are the bridge between human language and machine computation. They let you ask a question in English and get relevant documents back, without explicit keyword matching or rigid rules.

This is why RAG is so powerful: you're not building hand-coded knowledge bases or fine-tuning the model. You're just showing it documents, and letting the embedding model figure out what's relevant.

The cost is that you're trusting the embedding model's notion of similarity. Garbage in, garbage out — but if you trust the model (and for most tasks, nomic-embed-text is good), RAG is fast, scalable, and works offline.
