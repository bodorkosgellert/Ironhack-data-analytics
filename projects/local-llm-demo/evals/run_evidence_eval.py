"""Score the durable evidence-assistant eval set against the live assistant.

Default mode is retrieval-only (no Ollama). Use --mode assistant to score the
composed answer the same way compare_models.py does for its five fixed cases.

  python evals/run_evidence_eval.py
  python evals/run_evidence_eval.py --mode assistant --model qwen2.5-coder:3b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow `python evals/run_evidence_eval.py` from the project root.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.assistant import EvidenceAssistant  # noqa: E402

DEFAULT_CASES = HERE / "evidence_assistant_eval.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "evidence_assistant_eval_results.json"


def _contains_number(text: str, expected: float, tolerance: float = 1e-9) -> bool:
    pattern = r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for match in re.finditer(pattern, text.replace(",", "")):
        try:
            if abs(float(match.group()) - expected) <= tolerance:
                return True
        except ValueError:
            continue
    return False


def _has_tract_count(text: str) -> bool:
    compact = text.replace(",", "").lower()
    return "1175" in compact


def score_case(case: dict, answer_text: str, refused: bool) -> dict:
    """Apply transparent substring / refusal checks from the eval JSON."""
    checks: dict[str, bool] = {}
    expect_refusal = bool(case.get("expect_refusal", False))
    checks["refusal"] = refused == expect_refusal

    lowered = answer_text.lower()
    for needle in case.get("must_include") or []:
        if needle == "1175" or needle == "1,175":
            checks[f"include:{needle}"] = _has_tract_count(answer_text)
        else:
            checks[f"include:{needle}"] = needle.lower() in lowered

    for needle in case.get("must_not_include") or []:
        checks[f"exclude:{needle}"] = needle.lower() not in lowered

    for value in case.get("required_values") or []:
        checks[f"value:{value}"] = _contains_number(answer_text, float(value))

    # Insight cases that mention tracts should surface the documented 1175 count.
    if case.get("id") == "v1_tract_count_insight":
        checks["include:1175"] = _has_tract_count(answer_text)

    return {
        "passed": all(checks.values()) if checks else False,
        "objective_passed": sum(checks.values()),
        "objective_total": len(checks),
        "checks": checks,
    }


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of cases in {path}")
    return data


def run(
    cases_path: Path = DEFAULT_CASES,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    mode: str = "retrieval",
    model: str | None = None,
    retrieval: str = "tfidf",
) -> dict:
    if mode not in {"retrieval", "assistant"}:
        raise ValueError("mode must be 'retrieval' or 'assistant'")

    cases = load_cases(cases_path)
    assistant = EvidenceAssistant(retrieval_mode=retrieval)
    records: list[dict] = []

    for case in cases:
        retrieval_only = mode == "retrieval"
        kwargs: dict = {"retrieval_only": retrieval_only, "top_k": 5}
        if mode == "assistant" and model:
            kwargs["model"] = model
        answer = assistant.ask(case["question"], **kwargs)
        scored = score_case(case, answer.text, answer.refused)
        records.append(
            {
                "id": case["id"],
                "category": case.get("category"),
                "question": case["question"],
                "expected_answer": case.get("expected_answer"),
                "answer": answer.text,
                "refused": answer.refused,
                "retrieval_mode": answer.retrieval_mode,
                "generation_used": answer.generation_used,
                **scored,
            }
        )

    summary = {
        "cases_path": str(cases_path),
        "mode": mode,
        "retrieval": retrieval,
        "total": len(records),
        "passed": sum(bool(row["passed"]) for row in records),
        "failed": sum(not row["passed"] for row in records),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("retrieval", "assistant"),
        default="retrieval",
        help="Score deterministic retrieval answers or full assistant composition",
    )
    parser.add_argument("--model", default=None, help="Ollama model for --mode assistant")
    parser.add_argument(
        "--retrieval",
        choices=("tfidf", "dense", "hybrid"),
        default="tfidf",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(
        args.cases,
        args.output,
        mode=args.mode,
        model=args.model,
        retrieval=args.retrieval,
    )
    print(
        f"Evidence eval ({summary['mode']}): "
        f"{summary['passed']}/{summary['total']} passed; {summary['failed']} failed"
    )
    print(f"Wrote {args.output}")
    if summary["failed"]:
        for record in summary["records"]:
            if not record["passed"]:
                failed = [key for key, ok in record["checks"].items() if not ok]
                print(f"- FAIL {record['id']}: {failed}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
