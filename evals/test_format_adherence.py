"""Format adherence — DETERMINISTIC suite.

Verifies the whole response matches the required structure, so a correct-but-chatty answer (preamble,
markdown, extra text) is caught before it breaks a downstream parser. Validated against a golden set.
"""
import pytest

from evals import checks
from evals.cases import FORMAT_CASES, FormatCase


def _evaluate(case: FormatCase):
    passed, detail = checks.matches_format(case.response, case.pattern)
    return passed, [f"format: {detail}"]


@pytest.mark.parametrize("case", FORMAT_CASES, ids=[c.id for c in FORMAT_CASES])
def test_format_evaluator_matches_ground_truth(case: FormatCase):
    accepted, trace = _evaluate(case)
    assert accepted == case.should_pass, (
        f"\nCASE {case.id} — {case.rationale}\n"
        f"  expected accepted={case.should_pass}, got accepted={accepted}\n"
        f"  response: {case.response!r}\n"
        f"  trace:\n    " + "\n    ".join(trace)
    )
