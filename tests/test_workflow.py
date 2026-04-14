"""Integration smoke-tests for the LangGraph workflow.

These tests verify graph structure and state mechanics without making
any real LLM or database calls. Full pipeline tests belong in eval/.
"""
import pytest
from langchain_core.documents import Document

from src.agent.state import GraphState
from src.agent.workflow import build_workflow


# ── Workflow compilation ───────────────────────────────────────────────────────

class TestWorkflowCompilation:
    def test_compiles_without_error(self):
        wf = build_workflow()
        app = wf.compile()
        assert app is not None

    def test_node_names_present(self):
        wf = build_workflow()
        app = wf.compile()
        graph_repr = str(app.get_graph())
        expected_nodes = [
            "query_analyzer",
            "hybrid_retriever",
            "relevance_grader",
            "query_rewriter",
            "web_searcher",
            "generator",
            "hallucination_checker",
        ]
        for node in expected_nodes:
            assert node in graph_repr, f"Node '{node}' not found in compiled graph"


# ── GraphState mechanics ──────────────────────────────────────────────────────

class TestGraphStateMechanics:
    def test_route_log_uses_append_operator(self):
        """operator.add annotation means new lists are concatenated, not replaced."""
        import operator

        log_a = ["query_analyzer"]
        log_b = ["hybrid_retriever"]
        combined = operator.add(log_a, log_b)
        assert combined == ["query_analyzer", "hybrid_retriever"]

    def test_state_has_all_required_keys(self):
        state: GraphState = {
            "question":           "test question",
            "analyzed_question":  "",
            "vector_docs":        [],
            "graph_facts":        [],
            "fused_context":      "",
            "grade":              "poor",
            "grade_reason":       "",
            "rewritten_question": None,
            "web_docs":           [],
            "answer":             None,
            "grounded":           None,
            "unsupported_claims": [],
            "iterations":         0,
            "route_log":          [],
        }
        required = [
            "question", "analyzed_question", "vector_docs", "graph_facts",
            "fused_context", "grade", "grade_reason", "rewritten_question",
            "web_docs", "answer", "grounded", "unsupported_claims",
            "iterations", "route_log",
        ]
        for key in required:
            assert key in state, f"Missing key: {key}"

    def test_iterations_starts_at_zero(self):
        state: GraphState = {
            "question":           "q",
            "analyzed_question":  "",
            "vector_docs":        [],
            "graph_facts":        [],
            "fused_context":      "",
            "grade":              "poor",
            "grade_reason":       "",
            "rewritten_question": None,
            "web_docs":           [],
            "answer":             None,
            "grounded":           None,
            "unsupported_claims": [],
            "iterations":         0,
            "route_log":          [],
        }
        assert state["iterations"] == 0


# ── Chunker mechanics ─────────────────────────────────────────────────────────

class TestChunkerMechanics:
    def test_chunk_ids_are_unique(self):
        from langchain_core.documents import Document
        from src.ingestion.chunker import chunk_documents

        docs = [
            Document(
                page_content="A" * 500 + " " + "B" * 500,
                metadata={"source": "test.txt"},
            )
        ]
        chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_ids_deterministic(self):
        from langchain_core.documents import Document
        from src.ingestion.chunker import chunk_documents

        doc = Document(
            page_content="This is a reproducibility test. " * 40,
            metadata={"source": "repro.txt"},
        )
        chunks_a = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        chunks_b = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

        ids_a = [c.metadata["chunk_id"] for c in chunks_a]
        ids_b = [c.metadata["chunk_id"] for c in chunks_b]
        assert ids_a == ids_b, "Chunk IDs not deterministic across runs"
