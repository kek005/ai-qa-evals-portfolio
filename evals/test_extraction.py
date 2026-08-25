"""Extraction quality — DETERMINISTIC suite.

Verifies a captured extraction is valid JSON, has every required field, correctly typed, and every
value is grounded in the source (no hallucination). The evaluator is validated against a golden set:
it MUST accept the correct response and reject each deliberately-flawed one. A test fails loudly (with
the full check trace) if the verdict disagrees with the case's known ground truth.
"""
import pytest

from evals import checks
from evals.cases import EXTRACTION_CASES, ExtractionCase


def _evaluate(case: ExtractionCase):
    """Apply the extraction acceptance checks in order; return (accepted, trace)."""
    trace = []
    ok, detail = checks.is_valid_json(case.response)
    trace.append(f"json_valid: {detail}")
    if not ok:
        return False, trace
    obj = checks.parse_json(case.response)
    for name, (passed, detail) in {
        "fields_present": checks.has_fields(obj, case.required),
        "types_ok": checks.fields_typed(obj, case.schema),
        "grounded": checks.grounded_in([obj.get(f) for f in case.required], case.source_text),
    }.items():
        trace.append(f"{name}: {detail}")
        if not passed:
            return False, trace
    return True, trace


@pytest.mark.parametrize("case", EXTRACTION_CASES, ids=[c.id for c in EXTRACTION_CASES])
def test_extraction_evaluator_matches_ground_truth(case: ExtractionCase):
    accepted, trace = _evaluate(case)
    assert accepted == case.should_pass, (
        f"\nCASE {case.id} — {case.rationale}\n"
        f"  expected accepted={case.should_pass}, got accepted={accepted}\n"
        f"  response: {case.response}\n"
        f"  trace:\n    " + "\n    ".join(trace)
    )
