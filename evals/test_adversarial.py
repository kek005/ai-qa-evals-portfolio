"""Adversarial robustness — DETERMINISTIC suite.

Verifies the evaluator flags the failures that matter for agent safety — obeying a prompt injection
embedded in user content, and answering an out-of-scope ask a scoped assistant should deflect — while
passing the responses that correctly resist/deflect. Validated against a golden set with a full trace
on any disagreement.
"""
import pytest

from evals import checks
from evals.cases import ADVERSARIAL_CASES, AdversarialCase


def _evaluate(case: AdversarialCase):
    if case.kind == "injection":
        passed, detail = checks.injection_resisted(case.response, case.injection_markers)
    elif case.kind == "scope":
        passed, detail = checks.stayed_in_scope(case.response, case.out_of_scope_markers, case.refusal_markers)
    else:
        raise ValueError(f"unknown adversarial kind: {case.kind!r}")
    return passed, [f"{case.kind}: {detail}"]


@pytest.mark.parametrize("case", ADVERSARIAL_CASES, ids=[c.id for c in ADVERSARIAL_CASES])
def test_adversarial_evaluator_matches_ground_truth(case: AdversarialCase):
    accepted, trace = _evaluate(case)
    assert accepted == case.should_pass, (
        f"\nCASE {case.id} — {case.rationale}\n"
        f"  expected accepted={case.should_pass}, got accepted={accepted}\n"
        f"  response: {case.response}\n"
        f"  trace:\n    " + "\n    ".join(trace)
    )
