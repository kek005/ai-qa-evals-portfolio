"""Shortcut detection — DETERMINISTIC suite.

The hardest failure to catch by eye: an answer that *looks* right but skipped the required work
(citations, steps, evidence). Verifies the required work is actually shown. Validated against a golden
set — this is the exact "catch the shortcut" skill agentic-eval QA hires for.
"""
import pytest

from evals import checks
from evals.cases import SHORTCUT_CASES, ShortcutCase


def _evaluate(case: ShortcutCase):
    passed, detail = checks.no_shortcut(case.response, case.required_work)
    return passed, [f"required_work_shown: {detail}"]


@pytest.mark.parametrize("case", SHORTCUT_CASES, ids=[c.id for c in SHORTCUT_CASES])
def test_shortcut_evaluator_matches_ground_truth(case: ShortcutCase):
    accepted, trace = _evaluate(case)
    assert accepted == case.should_pass, (
        f"\nCASE {case.id} — {case.rationale}\n"
        f"  expected accepted={case.should_pass}, got accepted={accepted}\n"
        f"  response: {case.response!r}\n"
        f"  trace:\n    " + "\n    ".join(trace)
    )
