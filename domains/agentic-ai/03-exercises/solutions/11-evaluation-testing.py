"""
Day 11 -- Solutions to the easy exercises for evaluation & testing.

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/11-evaluation-testing.py
"""

from __future__ import annotations

import csv
import io
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# pylint: disable=wrong-import-position
day11 = import_module("11-evaluation-testing")
AgentTestCase = day11.AgentTestCase
AgentRun = day11.AgentRun
FakeAgent = day11.FakeAgent
MockJudgeLLM = day11.MockJudgeLLM
FinalAnswerEvaluator = day11.FinalAnswerEvaluator
TrajectoryEvaluator = day11.TrajectoryEvaluator
MetricEvaluator = day11.MetricEvaluator
EvalHarness = day11.EvalHarness
CaseResult = day11.CaseResult
build_test_cases = day11.build_test_cases


# ===========================================================================
# SOLUTION 1 -- OrderedTrajectoryEvaluator
# ===========================================================================

class OrderedTrajectoryEvaluator(TrajectoryEvaluator):
    """Like TrajectoryEvaluator but also enforces order (as a subsequence)."""

    def _is_subsequence(self, expected: list[str], actual: list[str]) -> bool:
        """Return True if expected is a subsequence (not necessarily contiguous) of actual."""
        i = 0
        for item in actual:
            if i < len(expected) and item == expected[i]:
                i += 1
        return i == len(expected)

    def evaluate(self, case: AgentTestCase, run: AgentRun) -> dict:  # type: ignore[override]
        base = super().evaluate(case, run)
        if not base["passed"] or run.error:
            return base
        if not self._is_subsequence(case.expected_tools, run.trajectory):
            return {
                "passed": False,
                "reason": (
                    f"tools out of order: expected order={case.expected_tools}, "
                    f"actual={run.trajectory}"
                ),
            }
        return base


class ReversingAgent(FakeAgent):
    """Agent that reverses every trajectory to trigger the ordered check."""

    def run(self, task: str) -> AgentRun:  # type: ignore[override]
        r = super().run(task)
        r.trajectory = list(reversed(r.trajectory))
        return r


def solution_1() -> None:
    print("\n=== Solution 1: ordered trajectory ===")
    evaluator = OrderedTrajectoryEvaluator()
    case = AgentTestCase(
        id="revenue-acme",
        task="What is the revenue of Acme in 2025?",
        expected_answer_criteria="820, euros",
        expected_tools=["search_docs", "extract_number"],
    )

    # Correct order
    good = FakeAgent(buggy=False).run(case.task)
    print("  correct order:", evaluator.evaluate(case, good))

    # Reversed order
    bad = ReversingAgent(buggy=False).run(case.task)
    print("  reversed order:", evaluator.evaluate(case, bad))

    # Extra case: competitor -> search_docs then summarize
    comp_case = AgentTestCase(
        id="competitor-ordered",
        task="Who is the main French competitor of Acme?",
        expected_answer_criteria="Artefact",
        expected_tools=["search_docs", "summarize"],
    )
    comp_good = FakeAgent(buggy=False).run(comp_case.task)
    print("  competitor good:", evaluator.evaluate(comp_case, comp_good))
    comp_bad = ReversingAgent(buggy=False).run(comp_case.task)
    print("  competitor bad:", evaluator.evaluate(comp_case, comp_bad))


# ===========================================================================
# SOLUTION 2 -- LLM judge with 3-criteria rubric
# ===========================================================================

class RubricJudge(MockJudgeLLM):
    """LLM-as-judge with a weighted rubric."""

    DEFAULT_RUBRIC = {"accuracy": 0.5, "completeness": 0.3, "conciseness": 0.2}

    def __init__(self, rubric: dict | None = None) -> None:
        super().__init__()
        self.rubric = rubric or dict(self.DEFAULT_RUBRIC)
        assert abs(sum(self.rubric.values()) - 1.0) < 1e-6, "weights must sum to 1"

    def judge_rubric(self, task: str, criteria: str, actual: str) -> dict:
        self.call_count += 1
        low = actual.lower()
        word_count = len(actual.split())
        keywords = [k.strip().lower() for k in criteria.split(",") if k.strip()]

        # Accuracy: keyword recall
        if keywords:
            hits = sum(1 for k in keywords if k in low)
            ratio = hits / len(keywords)
            if ratio == 1.0:
                accuracy = 5
            elif ratio > 0:
                accuracy = 3
            else:
                accuracy = 1
        else:
            accuracy = 3

        # Completeness: length buckets
        if word_count > 20:
            completeness = 5
        elif word_count >= 10:
            completeness = 3
        else:
            completeness = 1

        # Conciseness: short answers preferred, long penalized
        if word_count < 50:
            conciseness = 5
        elif word_count <= 100:
            conciseness = 3
        else:
            conciseness = 1

        sub_scores = {
            "accuracy": accuracy,
            "completeness": completeness,
            "conciseness": conciseness,
        }
        final = sum(sub_scores[k] * w for k, w in self.rubric.items())
        return {"score": final, "sub_scores": sub_scores}


class FinalAnswerEvaluatorV2:
    def __init__(self, judge: RubricJudge, pass_threshold: float = 3.5) -> None:
        self.judge = judge
        self.pass_threshold = pass_threshold

    def evaluate(self, case: AgentTestCase, run: AgentRun) -> dict:
        if run.error:
            return {
                "passed": False,
                "score": 0.0,
                "sub_scores": {},
                "reason": f"agent error: {run.error}",
            }
        v = self.judge.judge_rubric(case.task, case.expected_answer_criteria, run.answer)
        return {
            "passed": v["score"] >= self.pass_threshold,
            "score": v["score"],
            "sub_scores": v["sub_scores"],
            "reason": f"score={v['score']:.2f} subs={v['sub_scores']}",
        }


def solution_2() -> None:
    print("\n=== Solution 2: rubric LLM-as-judge ===")
    judge = RubricJudge()
    evaluator = FinalAnswerEvaluatorV2(judge=judge, pass_threshold=3.5)
    agent = FakeAgent(buggy=False)
    cases = build_test_cases()
    for case in cases:
        run = agent.run(case.task)
        result = evaluator.evaluate(case, run)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"  {status} {case.id:18} score={result['score']:.2f}  subs={result['sub_scores']}"
        )


# ===========================================================================
# SOLUTION 3 -- CSV export + reload
# ===========================================================================

CSV_COLUMNS = [
    "id",
    "verdict",
    "final_answer_passed",
    "final_answer_score",
    "final_answer_reason",
    "trajectory_passed",
    "trajectory_reason",
    "metrics_passed",
    "metrics_reason",
]


def export_results_to_csv(results: list[CaseResult], filepath: str) -> None:
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "id": r.id,
                    "verdict": r.verdict,
                    "final_answer_passed": r.final_answer["passed"],
                    "final_answer_score": r.final_answer.get("score", ""),
                    "final_answer_reason": r.final_answer.get("reason", ""),
                    "trajectory_passed": r.trajectory["passed"],
                    "trajectory_reason": r.trajectory["reason"],
                    "metrics_passed": r.metrics["passed"],
                    "metrics_reason": r.metrics["reason"],
                }
            )


def load_results_from_csv(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def solution_3() -> None:
    print("\n=== Solution 3: CSV export + reload ===")
    judge = MockJudgeLLM()
    harness = EvalHarness(
        cases=build_test_cases(),
        final_eval=FinalAnswerEvaluator(judge, pass_threshold=4),
        trajectory_eval=TrajectoryEvaluator(),
        metric_eval=MetricEvaluator(),
    )
    results = harness.run(FakeAgent(buggy=False))

    out = Path(__file__).parent / "_eval_results.csv"
    export_results_to_csv(results, str(out))
    loaded = load_results_from_csv(str(out))

    # Verify round-trip
    assert len(loaded) == len(results), "row count mismatch"
    for original, row in zip(results, loaded):
        assert row["id"] == original.id
        assert row["verdict"] == original.verdict

    print(f"  wrote {len(results)} rows to {out}")
    print("  sample row:", loaded[0])
    print("  round-trip OK")


if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()
