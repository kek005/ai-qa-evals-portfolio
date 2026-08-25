"""Faithfulness — DETERMINISTIC suite.

Verifies every specific claim in an answer is supported by the provided context (no facts invented
beyond the source) — the core RAG failure mode. Validated against a golden set; a mismatch prints the
full trace.
"""
import pytest

from evals import checks
from evals.cases import FAITHFULNESS_CASES, FaithfulnessCase


def _evaluate(case: FaithfulnessCase):
    passed, detail = checks.grounded_in(case.key_claims, case.context)
    return passed, [f"grounded_in_context: {detail}"]


@pytest.mark.parametrize("case", FAITHFULNESS_CASES, ids=[c.id for c in FAITHFULNESS_CASES])
def test_faithfulness_evaluator_matches_ground_truth(case: FaithfulnessCase):
    accepted, trace = _evaluate(case)
    assert accepted == case.should_pass, (
        f"\nCASE {case.id} — {case.rationale}\n"
        f"  expected accepted={case.should_pass}, got accepted={accepted}\n"
        f"  answer: {case.answer}\n"
        f"  trace:\n    " + "\n    ".join(trace)
    )
