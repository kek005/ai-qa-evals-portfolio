# LLM Agent Eval Suite — QA Test Plan

A reproducible evaluation suite that **stress-tests an LLM assistant and catches the ways it fails**:
wrong facts, format drift, silent shortcuts, unsafe compliance, and prompt-injection. Built the way a
QA engineer writes a test plan — clear scope, deterministic + rubric-graded assertions, adversarial
cases, and reproducible pass/fail with error traces.

> Portfolio piece for AI-QA / LLM-evaluation contract work. Stack: **Python + [DeepEval](https://github.com/confident-ai/deepeval) + pytest**.

## Why this exists
Hiring for AI QA is "show me," not "tell me." This repo demonstrates the core skills the roles ask for:
designing reward/scoring functions, probing edge cases, catching model shortcuts, and keeping eval
signals aligned and drift-free — as running, reproducible code.

## Test strategy (what each layer checks)
| Suite | What it verifies | Assertion type |
|-------|------------------|----------------|
| `test_extraction.py` | Structured-output correctness (fields present, typed, exact) | **Deterministic** |
| `test_faithfulness.py` | Answers are grounded in provided context, no hallucination | **Rubric (LLM-judge)** |
| `test_format_adherence.py` | Output obeys the required schema/format under pressure | **Deterministic** |
| `test_adversarial.py` | Refuses off-policy asks; resists prompt injection / jailbreaks | **Deterministic + rubric** |
| `test_shortcut_detection.py` | Catches "looks right but skipped the work" answers | **Rubric + trace check** |

## Design principles
1. **Every case has a written rationale** — what real failure it guards against (a test plan, not a script).
2. **Deterministic where possible, rubric only where judgment is unavoidable** — and rubric cases pin a
   scoring rubric so the signal doesn't drift between runs.
3. **Adversarial by default** — edge cases and injections are first-class, not an afterthought.
4. **Reproducible traces** — a failure prints the input, the output, and *why* it failed.

## Run it
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...   # or the model provider DeepEval is configured for
pytest -v
```

## Status
🚧 Scaffolding in progress (started 2026-08-25). Test-plan structure + first suites landing; each suite
ships with its case rationales. See `evals/`.
