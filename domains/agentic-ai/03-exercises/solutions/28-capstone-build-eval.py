"""
Day 28 -- Solutions to the capstone build & eval exercises.

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/28-capstone-build-eval.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# pylint: disable=wrong-import-position
day28 = import_module("28-capstone-build-eval")
EvalCase = day28.EvalCase
DeepOpsAgent = day28.DeepOpsAgent
DEMO_CASES = day28.DEMO_CASES
run_suite = day28.run_suite
regression_report = day28.regression_report
score = day28.score
AgentResult = day28.AgentResult
SubAgent = day28.SubAgent
VirtualFS = day28.VirtualFS
ModelRouter = day28.ModelRouter
ResearchSubAgent = day28.ResearchSubAgent
CoderSubAgent = day28.CoderSubAgent
VerifierSubAgent = day28.VerifierSubAgent


# ===========================================================================
# SOLUTION 1 -- new golden eval case + blocking regression
# ===========================================================================

def solution_1() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 1 -- golden case blocks regression")
    print("#" * 60)
    new_case = EvalCase(
        id="no-forbidden", task="fix the add bug cleanly",
        expected="SUCCESS",
        required_steps=["plan", "code", "run_tests", "verify"],
        max_steps=12, tags=["golden"])
    cases = DEMO_CASES + [new_case]

    baseline = run_suite(DeepOpsAgent(label="reliable", error_rate=0.05, seed=5), cases, k=5)
    candidate = run_suite(DeepOpsAgent(label="broken", error_rate=0.90, seed=5), cases, k=5)
    verdict = regression_report(baseline, candidate)
    assert verdict.startswith("BLOCKED"), "broken candidate must be BLOCKED on golden"
    print("  [check] broken candidate BLOCKED by golden gate -> OK")


# ===========================================================================
# SOLUTION 2 -- extra isolated sub-agent (linter)
# ===========================================================================

class LinterSubAgent(SubAgent):
    role = "lint"

    def run(self, task: str) -> str:
        repo = Path(self.fs.root) / "repo"
        src = (repo / "calc.py").read_text(encoding="utf-8")
        return "lint-ok" if "a + b" in src else "lint-fail"


class DeepOpsAgentWithLint(DeepOpsAgent):
    def solve(self, task: str) -> AgentResult:
        fs = VirtualFS()
        router = ModelRouter()
        traj = ["plan"]
        self._plan(fs, task)

        will_fix = self._rng.random() >= self.error_rate

        traj.append("research")
        ResearchSubAgent(fs, router).run(task)

        coder = CoderSubAgent(fs, router, will_fix=will_fix)
        traj.append("code")
        code_result = coder.run(task)
        traj.extend(coder.trajectory)

        # NEW: lint step shares the same fs/repo as the coder.
        traj.append("lint")
        lint = LinterSubAgent(fs, router).run(task)

        traj.append("verify")
        verdict = VerifierSubAgent(fs, router).run(code_result)
        ok = verdict == "verified-ok" and lint == "lint-ok"
        output = "SUCCESS: bug fixed, linted and verified" if ok else f"FAILURE: {code_result}/{lint}"
        return AgentResult(output=output, trajectory=traj, steps=len(traj))


def solution_2() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 2 -- linter sub-agent")
    print("#" * 60)
    agent = DeepOpsAgentWithLint(error_rate=0.0, seed=1)
    res = agent.solve("fix the add bug in calc")
    print(f"  trajectory: {res.trajectory}")
    print(f"  output    : {res.output}")
    assert "lint" in res.trajectory
    assert res.trajectory.index("lint") > res.trajectory.index("code")
    assert res.trajectory.index("lint") < res.trajectory.index("verify")
    assert "SUCCESS" in res.output
    print("  [check] lint step between code and verify, run SUCCESS -> OK")


# ===========================================================================
# SOLUTION 3 -- order-aware scoring
# ===========================================================================

def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(step in it for step in needle)


def score_ordered(result: AgentResult, case: EvalCase, ordered_steps: list[str]) -> bool:
    final_ok = case.expected.lower() in result.output.lower()
    budget_ok = result.steps <= case.max_steps
    order_ok = _is_subsequence(ordered_steps, result.trajectory)
    return final_ok and budget_ok and order_ok


def solution_3() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 3 -- order-aware scoring")
    print("#" * 60)
    case = EvalCase(id="ord", task="x", expected="SUCCESS", max_steps=12)

    good = AgentResult(output="SUCCESS", trajectory=["plan", "research", "code", "verify"], steps=4)
    assert score_ordered(good, case, ["plan", "research", "code", "verify"]) is True
    print("  ordered trajectory -> True")

    assert score_ordered(good, case, ["verify", "plan"]) is False
    print("  reversed required order -> False")

    shuffled = AgentResult(output="SUCCESS", trajectory=["verify", "plan", "code", "research"], steps=4)
    presence_ok = score(shuffled, EvalCase(id="s", task="x", expected="SUCCESS",
                                           required_steps=["plan", "code", "verify"], max_steps=12))
    ordered_ok = score_ordered(shuffled, case, ["plan", "code", "verify"])
    print(f"  shuffled: presence-score={presence_ok}, ordered-score={ordered_ok}")
    assert presence_ok is True and ordered_ok is False
    print("  [check] order-aware scoring stricter than presence -> OK")


if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()
    print("\nAll Day 28 solutions ran successfully.")
