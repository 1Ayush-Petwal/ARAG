"""Unit tests for retrieval layer — no external connections required."""
import pytest
from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import (
    RRF_K,
    _rrf_score,
    build_context,
    fuse_results,
)


def _doc(content: str, chunk_id: str = "test_chunk", source: str = "test.pdf") -> Document:
    return Document(page_content=content, metadata={"chunk_id": chunk_id, "source": source})


# ── RRF scoring ───────────────────────────────────────────────────────────────

class TestRRFScoring:
    def test_score_decreases_with_rank(self):
        assert _rrf_score(0) > _rrf_score(1) > _rrf_score(10)

    def test_score_formula(self):
        assert _rrf_score(0) == pytest.approx(1.0 / (RRF_K + 0))
        assert _rrf_score(5) == pytest.approx(1.0 / (RRF_K + 5))

    def test_score_always_positive(self):
        for rank in range(20):
            assert _rrf_score(rank) > 0


# ── fuse_results ──────────────────────────────────────────────────────────────

class TestFuseResults:
    def test_top_k_respected(self):
        docs = [(_doc(f"doc {i}", chunk_id=f"doc_{i}"), 0.9 - i * 0.05) for i in range(10)]
        top_docs, _ = fuse_results(docs, [], top_k=3)
        assert len(top_docs) == 3

    def test_highest_ranked_first(self):
        doc_a = _doc("Apple revenues", chunk_id="doc_a")
        doc_b = _doc("Microsoft earnings", chunk_id="doc_b")
        # doc_a is ranked first (index 0), doc_b second
        top_docs, _ = fuse_results([(doc_a, 0.95), (doc_b, 0.70)], [], top_k=2)
        assert top_docs[0].metadata["chunk_id"] == "doc_a"

    def test_duplicate_chunk_ids_merged(self):
        """Same chunk_id from multiple queries should be deduplicated via RRF."""
        doc = _doc("shared content", chunk_id="shared")
        # Same doc appearing twice (e.g. from two queries)
        top_docs, _ = fuse_results([(doc, 0.9), (doc, 0.8)], [], top_k=5)
        ids = [d.metadata["chunk_id"] for d in top_docs]
        assert ids.count("shared") == 1

    def test_graph_facts_deduplicated(self):
        facts = ["A --[R]--> B", "A --[R]--> B", "C --[S]--> D"]
        _, unique = fuse_results([], facts, top_k=6)
        assert len(unique) == 2
        assert unique[0] == "A --[R]--> B"

    def test_empty_inputs(self):
        docs, facts = fuse_results([], [], top_k=6)
        assert docs == []
        assert facts == []


# ── build_context ─────────────────────────────────────────────────────────────

class TestBuildContext:
    def test_empty_returns_placeholder(self):
        ctx = build_context([], [])
        assert ctx == "No relevant context found."

    def test_graph_section_present(self):
        ctx = build_context([], ["Apple --[REPORTS]--> Revenue_100B"])
        assert "Knowledge Graph Facts" in ctx
        assert "Apple" in ctx

    def test_vector_section_present(self):
        ctx = build_context([_doc("Passage about earnings.")], [])
        assert "Retrieved Document Passages" in ctx
        assert "Passage about earnings." in ctx

    def test_both_sections_ordered_graph_first(self):
        ctx = build_context(
            [_doc("Vector passage", chunk_id="v1")],
            ["Entity --[REL]--> Other"],
        )
        graph_pos = ctx.index("Knowledge Graph Facts")
        vector_pos = ctx.index("Retrieved Document Passages")
        assert graph_pos < vector_pos

    def test_source_shown_in_passage(self):
        doc = _doc("Content", source="report_2023.pdf")
        ctx = build_context([doc], [])
        assert "report_2023.pdf" in ctx
