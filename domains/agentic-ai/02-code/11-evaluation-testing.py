"""
Day 11 -- Evaluation & Testing: a minimal eval harness for an agent.

Demonstrates:
  1. AgentTestCase           -- dataclass describing a single test case
  2. FakeAgent               -- a tiny stub agent whose behavior we can control
                                via a "bug flag" so we can show regressions
  3. run_case                -- execute the agent on one case, capture trajectory
  4. MockJudgeLLM            -- LLM-as-judge stub with structured scoring
  5. FinalAnswerEvaluator    -- scores final answer via LLM-as-judge
  6. TrajectoryEvaluator     -- checks expected vs actual tool trajectory
  7. MetricEvaluator         -- latency, LLM calls, cost budget
  8. EvalHarness             -- runs all cases, computes pass rate, compares
                                against a baseline to detect regressions
  9. Demo: run the harness on a baseline agent, then on a "buggy" agent,
     and show the regression report

Dependencies: stdlib only. Optional: langchain/openai/anthropic. MockJudgeLLM
fallback ensures everything runs offline.

Run:
    python domains/agentic-ai/02-code/11-evaluation-testing.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Optional bindings
# ---------------------------------------------------------------------------

HAS_OPENAI = False
try:
    import openai  # noqa: F401
    HAS_OPENAI = True
except ImportError:
    pass


# ===========================================================================
# 1. TEST CASE DEFINITION
# ===========================================================================

@dataclass
class AgentTestCase:
    """One test case for the eval harness."""
    id: str
    task: str
    expected_answer_criteria: str          # what the answer must contain / prove
    expected_tools: list[str]              # tools the agent should call
    forbidden_tools: list[str] = field(default_factory=list)
    max_llm_calls: int = 20
    max_latency_ms: int = 5000
    tags: list[str] = field(default_factory=list)


# ===========================================================================
# 2. FAKE AGENT -- small stub with controllable behavior
# ===========================================================================

@dataclass
class AgentRun:
    """Captures the trajectory of one agent execution."""
    task: str
    trajectory: list[str] = field(default_factory=list)    # tool names in order
    answer: str = ""
    llm_calls: int = 0
    latency_ms: int = 0
    error: str | None = None


class FakeAgent:
    """
    A toy agent with hardcoded behaviors for specific tasks.

    If `buggy=True`, the agent introduces deliberate regressions:
      - skips the retrieval step on one task
      - calls a forbidden tool on another
      - returns a wrong answer on a third

    This lets the demo show what a regression report looks like when a real
    change breaks previously-passing cases.
    """

    def __init__(self, buggy: bool = False) -> None:
        self.buggy = buggy

    def run(self, task: str) -> AgentRun:
        start = time.perf_counter()
        run = AgentRun(task=task)

        low = task.lower()

        if "revenue" in low and "kalira" in low:
            run.trajectory = ["search_docs", "extract_number"]
            run.llm_calls = 3
            run.answer = "Kalira revenue in 2025 is 820k euros."
            if self.buggy:
                # Bug: skip search_docs -> hallucinates a wrong number
                run.trajectory = ["extract_number"]
                run.answer = "Kalira revenue in 2025 is 3.5 million euros."

        elif "webhook" in low:
            run.trajectory = ["search_docs"]
            run.llm_calls = 2
            run.answer = "Use POST /webhook with X-Kalira-Signature header."
            if self.buggy:
                # Bug: calls a forbidden tool (delete_data) on this task
                run.trajectory = ["search_docs", "delete_data"]
                run.answer = "Use POST /webhook with X-Kalira-Signature header."

        elif "competitor" in low:
            run.trajectory = ["search_docs", "summarize"]
            run.llm_calls = 4
            run.answer = "Artefact is the main French competitor, 230M euros revenue."
            if self.buggy:
                # Bug: correct trajectory but wrong answer
                run.answer = "I do not know."

        elif "kalira-immo" in low:
            run.trajectory = ["search_docs"]
            run.llm_calls = 2
            run.answer = "kalira-immo is a real estate price transparency platform in West Africa."

        else:
            run.trajectory = ["search_docs"]
            run.llm_calls = 1
            run.answer = "I do not know."

        run.latency_ms = int((time.perf_counter() - start) * 1000) + 5
        return run


def run_case(agent: FakeAgent, case: AgentTestCase) -> AgentRun:
    """Execute the agent on one test case."""
    try:
        return agent.run(case.task)
    except Exception as exc:  # noqa: BLE001
        return AgentRun(task=case.task, error=str(exc))


# ===========================================================================
# 3. LLM-AS-JUDGE (mocked)
# ===========================================================================

class MockJudgeLLM:
    """
    Deterministic judge that inspects the answer against a criteria string.

    The criteria is a comma-separated list of substrings that MUST appear in
    the answer. The judge returns:
      - score 5 if all substrings are present
      - score 3 if some are present
      - score 1 if none are present
    A short reasoning is always included.

    In a real system this would be a prompted LLM call with a strict rubric.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def judge(self, task: str, criteria: str, actual: str) -> dict:
        self.call_count += 1
        low = actual.lower()
        keywords = [k.strip().lower() for k in criteria.split(",") if k.strip()]
        if not keywords:
            return {"score": 3, "reasoning": "No criteria provided"}
        hits = [k for k in keywords if k in low]
        missing = [k for k in keywords if k not in low]
        if len(hits) == len(keywords):
            score = 5
            reasoning = f"All expected elements present: {hits}"
        elif hits:
            score = 3
            reasoning = f"Partial match. Found: {hits}. Missing: {missing}"
        else:
            score = 1
            reasoning = f"None of the expected elements found. Missing: {missing}"
        return {"score": score, "reasoning": reasoning}


# ===========================================================================
# 4. EVALUATORS
# ===========================================================================

class FinalAnswerEvaluator:
    """Runs the LLM-as-judge and returns a pass/fail above a threshold."""

    def __init__(self, judge: MockJudgeLLM, pass_threshold: int = 4) -> None:
        self.judge = judge
        self.pass_threshold = pass_threshold

    def evaluate(self, case: AgentTestCase, run: AgentRun) -> dict:
        if run.error:
            return {"passed": False, "score": 0, "reason": f"agent error: {run.error}"}
        verdict = self.judge.judge(case.task, case.expected_answer_criteria, run.answer)
        return {
            "passed": verdict["score"] >= self.pass_threshold,
            "score": verdict["score"],
            "reason": verdict["reasoning"],
        }


class TrajectoryEvaluator:
    """
    Checks:
      - every expected tool was called at least once (recall)
      - no forbidden tool was called
    Does NOT enforce order (you can add that as an exercise).
    """

    def evaluate(self, case: AgentTestCase, run: AgentRun) -> dict:
        if run.error:
            return {"passed": False, "reason": "agent error"}
        actual = set(run.trajectory)
        expected = set(case.expected_tools)
        missing = expected - actual
        forbidden_called = [t for t in case.forbidden_tools if t in actual]
        passed = not missing and not forbidden_called
        reason_parts: list[str] = []
        if missing:
            reason_parts.append(f"missing tools: {sorted(missing)}")
        if forbidden_called:
            reason_parts.append(f"forbidden tools called: {forbidden_called}")
        if not reason_parts:
            reason_parts.append("all expected tools called, no forbidden tools")
        return {"passed": passed, "reason": "; ".join(reason_parts)}


class MetricEvaluator:
    """Checks latency and LLM call budgets."""

    def evaluate(self, case: AgentTestCase, run: AgentRun) -> dict:
        reasons: list[str] = []
        if run.llm_calls > case.max_llm_calls:
            reasons.append(f"too many LLM calls: {run.llm_calls} > {case.max_llm_calls}")
        if run.latency_ms > case.max_latency_ms:
            reasons.append(f"too slow: {run.latency_ms}ms > {case.max_latency_ms}ms")
        passed = not reasons
        return {
            "passed": passed,
            "reason": "; ".join(reasons) or "within budget",
        }


# ===========================================================================
# 5. EVAL HARNESS
# ===========================================================================

@dataclass
class CaseResult:
    """Aggregated result for one case."""
    id: str
    verdict: str                          # PASS / FAIL
    final_answer: dict
    trajectory: dict
    metrics: dict


class EvalHarness:
    """
    Run a dataset through an agent, aggregate results, compare to a baseline.
    """

    def __init__(
        self,
        cases: list[AgentTestCase],
        final_eval: FinalAnswerEvaluator,
        trajectory_eval: TrajectoryEvaluator,
        metric_eval: MetricEvaluator,
    ) -> None:
        self.cases = cases
        self.final_eval = final_eval
        self.trajectory_eval = trajectory_eval
        self.metric_eval = metric_eval

    def run(self, agent: FakeAgent) -> list[CaseResult]:
        results: list[CaseResult] = []
        for case in self.cases:
            run = run_case(agent, case)
            final = self.final_eval.evaluate(case, run)
            traj = self.trajectory_eval.evaluate(case, run)
            metrics = self.metric_eval.evaluate(case, run)
            passed = final["passed"] and traj["passed"] and metrics["passed"]
            results.append(
                CaseResult(
                    id=case.id,
                    verdict="PASS" if passed else "FAIL",
                    final_answer=final,
                    trajectory=traj,
                    metrics=metrics,
                )
            )
        return results

    @staticmethod
    def pass_rate(results: list[CaseResult]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r.verdict == "PASS") / len(results)

    @staticmethod
    def diff(
        baseline: list[CaseResult], current: list[CaseResult]
    ) -> dict:
        """
        Return a regression report: cases that were PASS in baseline but
        FAIL in current, and cases that were FAIL in baseline but PASS now.
        """
        base_map = {r.id: r.verdict for r in baseline}
        curr_map = {r.id: r.verdict for r in current}
        regressions = [
            i for i in base_map
            if base_map[i] == "PASS" and curr_map.get(i) == "FAIL"
        ]
        fixes = [
            i for i in base_map
            if base_map[i] == "FAIL" and curr_map.get(i) == "PASS"
        ]
        return {
            "baseline_pass_rate": EvalHarness.pass_rate(baseline),
            "current_pass_rate": EvalHarness.pass_rate(current),
            "regressions": regressions,
            "fixes": fixes,
        }

    @staticmethod
    def print_report(results: list[CaseResult]) -> None:
        for r in results:
            symbol = "[PASS]" if r.verdict == "PASS" else "[FAIL]"
            print(f"  {symbol} {r.id}")
            if r.verdict == "FAIL":
                if not r.final_answer["passed"]:
                    print(f"       final:      {r.final_answer['reason']}")
                if not r.trajectory["passed"]:
                    print(f"       trajectory: {r.trajectory['reason']}")
                if not r.metrics["passed"]:
                    print(f"       metrics:    {r.metrics['reason']}")


# ===========================================================================
# 6. DEMO
# ===========================================================================

def build_test_cases() -> list[AgentTestCase]:
    return [
        AgentTestCase(
            id="revenue-kalira",
            task="What is the revenue of Kalira in 2025?",
            expected_answer_criteria="820, euros",
            expected_tools=["search_docs"],
            forbidden_tools=["delete_data"],
            tags=["easy", "rag"],
        ),
        AgentTestCase(
            id="webhook-api",
            task="How do I use the webhook API?",
            expected_answer_criteria="POST, /webhook, signature",
            expected_tools=["search_docs"],
            forbidden_tools=["delete_data"],
            tags=["easy", "docs"],
        ),
        AgentTestCase(
            id="competitor",
            task="Who is the main French competitor of Kalira?",
            expected_answer_criteria="Artefact, 230",
            expected_tools=["search_docs"],
            forbidden_tools=["delete_data"],
            tags=["medium", "rag"],
        ),
        AgentTestCase(
            id="kalira-immo",
            task="What is kalira-immo?",
            expected_answer_criteria="real estate, West Africa",
            expected_tools=["search_docs"],
            forbidden_tools=["delete_data"],
            tags=["easy", "docs"],
        ),
    ]


def demo() -> None:
    print("=" * 70)
    print(f"Backends available: openai={HAS_OPENAI} -- using MockJudgeLLM")
    print("=" * 70)

    judge = MockJudgeLLM()
    harness = EvalHarness(
        cases=build_test_cases(),
        final_eval=FinalAnswerEvaluator(judge=judge, pass_threshold=4),
        trajectory_eval=TrajectoryEvaluator(),
        metric_eval=MetricEvaluator(),
    )

    print("\n--- BASELINE RUN (clean agent) ---")
    baseline_agent = FakeAgent(buggy=False)
    baseline_results = harness.run(baseline_agent)
    harness.print_report(baseline_results)
    print(f"\n  Baseline pass rate: {harness.pass_rate(baseline_results):.0%}")

    print("\n--- CANDIDATE RUN (buggy agent -- simulates a broken release) ---")
    candidate_agent = FakeAgent(buggy=True)
    candidate_results = harness.run(candidate_agent)
    harness.print_report(candidate_results)
    print(f"\n  Candidate pass rate: {harness.pass_rate(candidate_results):.0%}")

    print("\n--- REGRESSION REPORT ---")
    diff = harness.diff(baseline_results, candidate_results)
    print(json.dumps(diff, indent=2))

    if diff["regressions"]:
        print("\n  VERDICT: REGRESSION DETECTED -- blocking release.")
    else:
        print("\n  VERDICT: no regressions.")


if __name__ == "__main__":
    demo()
