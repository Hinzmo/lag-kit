"""Phase 6 tests that need neither ChromaDB state nor a running embedder.

chunk_text is the piece worth pinning hardest: its original loop rewound `start`
behind the end of the text on the final chunk and re-emitted the same tail
forever, so every ingest hung before it reached the embedder.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "servers"))

chromadb = pytest.importorskip("chromadb", reason="install with .[rag]")
rag_server = pytest.importorskip("rag_server")

chunk_text = rag_server.chunk_text
load_documents = rag_server.load_documents


# Every case here would hang before the termination fix.
@pytest.mark.parametrize("length", [1, 50, 500, 1999, 2000, 2001, 5000])
def test_chunking_terminates_and_covers_text(length):
    chunks = chunk_text("a" * length)
    assert chunks
    assert all(c for c in chunks)
    # nothing is dropped: every character appears in some chunk
    assert sum(len(c) for c in chunks) >= length


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_long_text_splits_into_multiple_chunks():
    assert len(chunk_text("word " * 3000)) > 1


def test_chunks_overlap_to_preserve_context():
    chunks = chunk_text("".join(f"{i:05d}" for i in range(1000)))
    assert len(chunks) > 1
    # consecutive chunks must share a boundary region
    assert chunks[0][-50:] in chunks[1]


def test_whitespace_is_normalized():
    assert chunk_text("a\n\n\tb   c") == ["a b c"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []


def test_overlap_larger_than_chunk_still_terminates():
    """Pathological config must not wedge the ingest."""
    assert chunk_text("x" * 500, chunk_size=1, overlap=10_000)


class TestLoadDocuments:
    def test_reads_txt_and_md(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
        (tmp_path / "b.md").write_text("beta", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert {d["content"] for d in docs} == {"alpha", "beta"}

    def test_ignores_other_extensions(self, tmp_path):
        (tmp_path / "keep.md").write_text("yes", encoding="utf-8")
        (tmp_path / "skip.pdf").write_bytes(b"%PDF-1.4")
        assert [d["content"] for d in load_documents(tmp_path)] == ["yes"]

    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert load_documents(tmp_path) == []


def test_search_tool_is_registered():
    names = [t.name for t in rag_server.mcp._tool_manager.list_tools()]
    assert names == ["search_knowledge_base"]
