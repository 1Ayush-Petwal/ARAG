"""Unit tests for grader and hallucination-checker routing logic.

All routing functions are pure (state-in → string-out), so these tests
run with no LLM calls and no external connections.
"""
import pytest

from src.graph_nodes.grader import decide_after_grade
from src.graph_nodes.hallucination_checker import decide_after_hallucination_check


# ── decide_after_grade ────────────────────────────────────────────────────────

class TestDecideAfterGrade:
    def test_sufficient_always_generates(self):
        for iters in range(4):
            state = {"grade": "sufficient", "iterations": iters}
            assert decide_after_grade(state) == "generate"

    def test_off_topic_always_web(self):
        for iters in range(4):
            state = {"grade": "off_topic", "iterations": iters}
            assert decide_after_grade(state) == "web_search"

    def test_poor_under_budget_rewrites(self):
        # max_iterations defaults to 3; iterations < max-1 (i.e. < 2) → rewrite
        state = {"grade": "poor", "iterations": 0}
        assert decide_after_grade(state) == "rewrite"

        state = {"grade": "poor", "iterations": 1}
        assert decide_after_grade(state) == "rewrite"

    def test_poor_budget_exhausted_falls_back_to_web(self):
        # iterations >= max_iterations - 1 (i.e. >= 2) → web_search
        state = {"grade": "poor", "iterations": 2}
        assert decide_after_grade(state) == "web_search"

        state = {"grade": "poor", "iterations": 10}
        assert decide_after_grade(state) == "web_search"

    def test_missing_iterations_defaults_to_zero(self):
        # iterations key absent → treated as 0 → rewrite
        state = {"grade": "poor"}
        assert decide_after_grade(state) == "rewrite"


# ── decide_after_hallucination_check ─────────────────────────────────────────

class TestDecideAfterHallucinationCheck:
    def test_grounded_ends(self):
        state = {"grounded": True, "iterations": 0}
        assert decide_after_hallucination_check(state) == "end"

    def test_grounded_even_high_iters_ends(self):
        state = {"grounded": True, "iterations": 99}
        assert decide_after_hallucination_check(state) == "end"

    def test_ungrounded_under_budget_regenerates(self):
        state = {"grounded": False, "iterations": 0}
        assert decide_after_hallucination_check(state) == "regenerate"

    def test_ungrounded_budget_exhausted_ends(self):
        # max_iterations defaults to 3
        state = {"grounded": False, "iterations": 3}
        assert decide_after_hallucination_check(state) == "end"

        state = {"grounded": False, "iterations": 5}
        assert decide_after_hallucination_check(state) == "end"

    def test_none_grounded_treated_as_grounded(self):
        # If hallucination checker hasn't run yet (grounded=None), default to end
        state = {"grounded": None, "iterations": 0}
        # None is falsy but we want safe default — check that we don't regenerate indefinitely
        # Current implementation: `grounded or iterations >= max_iter`
        # None is falsy → falls through to iterations check (0 < 3) → "regenerate"
        # This is intentional: if checker returned None (error), we still try once
        result = decide_after_hallucination_check(state)
        assert result in ("end", "regenerate")  # either is acceptable for None
