"""Retrieval-core tests that need neither ChromaDB state nor a live embedder.

chunk_text is pinned hardest: its original loop rewound `start` behind the end
of the text on the final chunk and re-emitted the same tail forever, so every
ingest hung before it reached the embedder.
"""
from __future__ import annotations

import pytest

pytest.importorskip("chromadb", reason="install with .[rag]")
from lag import rag  # noqa: E402

chunk_text = rag.chunk_text
load_documents = rag.load_documents


# Every case here would hang before the termination fix.
@pytest.mark.parametrize("length", [1, 50, 500, 1999, 2000, 2001, 5000])
def test_chunking_terminates_and_covers_text(length):
    chunks = chunk_text("a" * length)
    assert chunks
    assert all(c for c in chunks)
    assert sum(len(c) for c in chunks) >= length


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_long_text_splits_into_multiple_chunks():
    assert len(chunk_text("word " * 3000)) > 1


def test_chunks_overlap_to_preserve_context():
    chunks = chunk_text("".join(f"{i:05d}" for i in range(1000)))
    assert len(chunks) > 1
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
        assert {d["content"] for d in load_documents(tmp_path)} == {"alpha", "beta"}

    def test_ignores_other_extensions(self, tmp_path):
        (tmp_path / "keep.md").write_text("yes", encoding="utf-8")
        (tmp_path / "skip.pdf").write_bytes(b"%PDF-1.4")
        assert [d["content"] for d in load_documents(tmp_path)] == ["yes"]

    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert load_documents(tmp_path) == []


class TestChunkIds:
    """Ids must be content-addressed.

    The original ingest numbered chunks `chunk_0..N`, so a second ingest
    silently overwrote the first, and `remember` — now writing to the same
    collection — would clobber whatever shared its index.
    """

    def test_ids_are_stable_for_identical_text(self):
        assert rag._ids_for(["hello"], "chunk") == rag._ids_for(["hello"], "chunk")

    def test_ids_differ_for_different_text(self):
        a, = rag._ids_for(["hello"], "chunk")
        b, = rag._ids_for(["goodbye"], "chunk")
        assert a != b

    def test_prefix_separates_writers(self):
        doc, = rag._ids_for(["same text"], "chunk")
        mem, = rag._ids_for(["same text"], "mem")
        assert doc != mem

    def test_no_positional_collision(self):
        ids = rag._ids_for(["one", "two", "three"], "chunk")
        assert len(set(ids)) == 3


class TestIndexNote:
    """remember() indexes by default; verify what gets embedded and tagged."""

    def test_key_and_value_are_embedded_together(self, monkeypatch):
        seen = {}

        def fake_add(texts, metas, **kw):
            seen["texts"], seen["metas"], seen["kw"] = texts, metas, kw
            return len(texts)

        monkeypatch.setattr(rag, "add_texts", fake_add)
        n = rag.index_note("editor", "PyCharm, not VS Code")

        assert n == 1
        assert "editor" in seen["texts"][0] and "PyCharm" in seen["texts"][0]
        assert seen["metas"][0]["kind"] == "memory"
        assert seen["metas"][0]["key"] == "editor"
        assert seen["kw"]["prefix"] == "mem"

    def test_long_value_is_chunked(self, monkeypatch):
        monkeypatch.setattr(rag, "add_texts", lambda t, m, **k: len(t))
        assert rag.index_note("k", "word " * 3000) > 1


def test_search_tool_is_registered():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "servers"))
    import rag_server
    assert [t.name for t in rag_server.mcp._tool_manager.list_tools()] == ["search_knowledge_base"]
