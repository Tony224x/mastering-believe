"""
Day 26 -- Solutions to the exercises for practical benchmarking.

Run the whole file to execute every solution.

    python domains/tech/agentic-ai/03-exercises/solutions/26-benchmarking-pratique.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# pylint: disable=wrong-import-position
day26 = import_module("26-benchmarking-pratique")
EvalCase = day26.EvalCase
MockAgent = day26.MockAgent
Scorer = day26.Scorer
SuiteReport = day26.SuiteReport
run_suite = day26.run_suite
pass_k = day26.pass_k
p_hat_confidence_interval = day26.p_hat_confidence_interval
DEMO_CASES = day26.DEMO_CASES


# ===========================================================================
# SOLUTION 1 -- EvalCase with criticality + blocking regression report
# ===========================================================================

@dataclass
class PriorityCase(EvalCase):
    """EvalCase enriched with a business-criticality level.

    'critical' cases gate the deploy on their own: a single critical
    regression blocks the release even if everything else improves.
    """
    priority: str = "medium"  # "low" | "medium" | "critical"


class BrokenOnCaseAgent(MockAgent):
    """Mock agent that is reliable everywhere EXCEPT on one targeted case.

    Used to simulate a candidate that improves globally but regresses on a
    single critical scenario -- the exact situation a priority gate exists
    to catch.
    """

    def __init__(self, label: str, error_rate: float, broken_case_id: str,
                 broken_error_rate: float, seed: int = 42) -> None:
        super().__init__(label=label, error_rate=error_rate, seed=seed)
        self.broken_case_id = broken_case_id
        self.broken_error_rate = broken_error_rate

    def run(self, case: EvalCase):
        # Temporarily swap the error rate for the targeted case only.
        if case.id == self.broken_case_id:
            saved = self.error_rate
            self.error_rate = self.broken_error_rate
            try:
                return super().run(case)
            finally:
                self.error_rate = saved
        return super().run(case)


def priority_regression_report(
    baseline: SuiteReport,
    candidate: SuiteReport,
    priorities: dict[str, str],
    block_on_priority: str = "critical",
    regression_threshold: float = -0.10,
) -> str:
    """Like regression_report but blocks on any regression of a case whose
    priority == block_on_priority. Returns the verdict string.
    """
    print("\n" + "=" * 60)
    print(f"PRIORITY REGRESSION REPORT (block_on={block_on_priority})")
    print("=" * 60)

    blocking = False
    improved = 0
    for case_id in sorted({r.case_id for r in baseline.case_reports}):
        b = baseline.by_id(case_id)
        c = candidate.by_id(case_id)
        if b is None or c is None:
            continue
        delta = c.pass_k - b.pass_k
        prio = priorities.get(case_id, "medium")
        flag = ""
        if prio == block_on_priority:
            flag = " [CRITICAL -- BLOCKING]"
        if delta >= 0.10:
            improved += 1
        if delta <= regression_threshold and prio == block_on_priority:
            blocking = True
            flag += "  <== REGRESSION"
        print(f"  {case_id:<18} prio={prio:<8} {b.pass_k:.2f} -> {c.pass_k:.2f} "
              f"(delta={delta:+.2f}){flag}")

    delta_global = candidate.mean_pass_k - baseline.mean_pass_k
    print(f"\n  Global pass^k delta : {delta_global:+.3f} "
          f"({improved} case(s) improved)")
    verdict = ("BLOCKED -- critical regression" if blocking
               else "APPROVED" if delta_global > 0 else "NEUTRAL")
    print(f"  VERDICT : {verdict}")
    return verdict


def solution_1() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 1 -- Criticality gate")
    print("#" * 60)

    cases = [
        PriorityCase(
            id="CMD-cancel", input="Cancel order CMD-001",
            expected="order CMD-001 cancelled, refund initiated",
            policy={"required_tools": ["search_order", "cancel_order"], "max_steps": 5},
            priority="critical"),
        PriorityCase(
            id="CMD-status", input="Status of CMD-002?",
            expected="status=delivered",
            policy={"required_tools": ["search_order"], "max_steps": 3},
            priority="medium"),
        PriorityCase(
            id="CMD-edge", input="Cancel non-existent CMD-003",
            expected="order not found",
            policy={"required_tools": ["search_order"], "max_steps": 3},
            priority="low"),
        PriorityCase(
            id="CMD-email", input="Email user-bob",
            expected="email sent",
            policy={"required_tools": ["lookup_user", "send_email"], "max_steps": 4},
            priority="low"),
    ]
    priorities = {c.id: c.priority for c in cases}

    # Baseline: reliable everywhere (incl. the critical case).
    baseline_agent = MockAgent(label="v1.0", error_rate=0.10, seed=1)
    # Candidate: better on the rest, but badly breaks the critical case.
    candidate_agent = BrokenOnCaseAgent(
        label="v1.1", error_rate=0.03, broken_case_id="CMD-cancel",
        broken_error_rate=0.90, seed=1)

    base = run_suite(baseline_agent, cases, k=5)
    cand = run_suite(candidate_agent, cases, k=5)
    verdict = priority_regression_report(base, cand, priorities)
    assert verdict.startswith("BLOCKED"), "critical regression must block"
    print("  [check] global improvement still BLOCKED by critical case -> OK")


# ===========================================================================
# SOLUTION 2 -- Confidence interval on pass^k (delta method)
# ===========================================================================

def pass_k_confidence_interval(successes: int, n_runs: int, k: int,
                               z: float = 1.96) -> tuple[float, float]:
    """CI on pass^k = p_hat^k via the delta method (error propagation).

    Var(p_hat)   = p(1-p)/n
    Var(pass^k)  ~ (k * p^(k-1))^2 * Var(p_hat)
    """
    if n_runs == 0:
        return (0.0, 1.0)
    p = successes / n_runs
    var_p = p * (1 - p) / n_runs
    grad = k * (p ** (k - 1)) if k >= 1 else 0.0
    var_pk = (grad ** 2) * var_p
    margin = z * math.sqrt(var_pk)
    point = p ** k
    return (max(0.0, point - margin), min(1.0, point + margin))


def is_improvement_significant(base_successes: int, cand_successes: int,
                               n_runs: int, k: int, z: float = 1.96) -> bool:
    """True when the two pass^k CIs do not overlap."""
    b_lo, b_hi = pass_k_confidence_interval(base_successes, n_runs, k, z)
    c_lo, c_hi = pass_k_confidence_interval(cand_successes, n_runs, k, z)
    # Non-overlap: one interval lies entirely above the other.
    return c_lo > b_hi or b_lo > c_hi


def solution_2() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 2 -- CI on pass^k (delta method)")
    print("#" * 60)

    K = 5
    N = 5
    baseline_agent = MockAgent(label="v1.0", error_rate=0.40, seed=99)
    candidate_agent = MockAgent(label="v1.1", error_rate=0.05, seed=77)
    base = run_suite(baseline_agent, DEMO_CASES, k=K)
    cand = run_suite(candidate_agent, DEMO_CASES, k=K)

    sig_count = 0
    nonsig_count = 0
    print(f"\n{'Case':<28}{'base p^k [CI]':>22}{'cand p^k [CI]':>22}{'sig?':>6}")
    print("-" * 78)
    for case in DEMO_CASES:
        b = base.by_id(case.id)
        c = cand.by_id(case.id)
        if not (b and c):
            continue
        b_lo, b_hi = pass_k_confidence_interval(b.successes, N, K)
        c_lo, c_hi = pass_k_confidence_interval(c.successes, N, K)
        sig = is_improvement_significant(b.successes, c.successes, N, K)
        sig_count += int(sig)
        nonsig_count += int(not sig)
        print(f"{case.id:<28}"
              f"{b.pass_k:.2f}[{b_lo:.2f},{b_hi:.2f}]".rjust(22) +
              f"{c.pass_k:.2f}[{c_lo:.2f},{c_hi:.2f}]".rjust(22) +
              f"{'yes' if sig else 'no':>6}")
    print(f"\n  significant: {sig_count}, non-significant: {nonsig_count}")
    # On this dataset the strong gap should yield at least one significant case.
    assert sig_count >= 1, "expected at least one significant improvement"
    print("  [check] CI clamped to [0,1] and significance detected -> OK")


# ===========================================================================
# SOLUTION 3 -- Multi-agent leaderboard
# ===========================================================================

class SlowAgent(MockAgent):
    """Reliable per run, but pads its trajectory well past max_steps.

    Teaching point: a budget overrun only hurts if the *scorer* weights the
    trajectory enough. With the default Scorer (alpha=0.6 on the final answer,
    threshold 0.7) a correct-but-over-budget run still scores 0.6 + 0.4*0.5 =
    0.8 >= 0.7 and PASSES -- the overrun is invisible. solution_3 therefore
    evaluates with a trajectory-weighted scorer so 'slow' is actually penalized.
    """

    def _successful_run(self, case: EvalCase):
        result = super()._successful_run(case)
        result.steps *= 2
        # Pad far beyond any DEMO_CASES max_steps so the budget check fails.
        result.trajectory = result.trajectory + ["noop"] * 10
        return result


def solution_3() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 3 -- Leaderboard")
    print("#" * 60)

    K = 5
    # Trajectory-weighted scorer: here the step budget genuinely gates success,
    # so agent_conservative's over-budget runs are penalized (see SlowAgent doc).
    strict_scorer = Scorer(answer_mode="contains", alpha=0.3, threshold=0.7)
    agents = [
        SlowAgent(label="agent_conservative", error_rate=0.05, seed=3),
        MockAgent(label="agent_balanced", error_rate=0.20, seed=3),
        MockAgent(label="agent_creative", error_rate=0.35, seed=3),
        MockAgent(label="agent_broken", error_rate=0.60, seed=3),
    ]

    rows = []
    for agent in agents:
        report = run_suite(agent, DEMO_CASES, k=K, scorer=strict_scorer)
        golden = [r.pass_k for r in report.case_reports if r.is_golden]
        golden_mean = sum(golden) / len(golden) if golden else 0.0
        rows.append((agent.label, report.mean_p_hat, report.mean_pass_k, golden_mean))

    rows.sort(key=lambda r: r[2], reverse=True)

    print(f"\nLEADERBOARD (pass^{K}, k={K} runs/case)")
    print("=" * 64)
    print(f"{'Rank':<5}{'Agent':<22}{'mean_p_hat':>12}{'mean_pass^k':>14}{'Golden':>10}")
    print("-" * 64)
    for rank, (label, p_hat, pk, gp) in enumerate(rows, start=1):
        print(f"{rank:<5}{label:<22}{p_hat:>12.2f}{pk:>14.3f}{gp:>10.3f}")

    pks = [r[2] for r in rows]
    assert pks == sorted(pks, reverse=True), "leaderboard must be sorted desc"

    # Honest teaching point: the budget overrun lowers agent_conservative's score
    # but does NOT sink it, because score_trajectory AVERAGES several constraints
    # (recall, precision, budget) -- a single over-budget signal gets diluted.
    # Compare the same agent under the default vs the trajectory-weighted scorer:
    conf_strict = next(pk for label, _, pk, _ in rows if label == "agent_conservative")
    conf_default = run_suite(SlowAgent(label="c", error_rate=0.05, seed=3),
                             DEMO_CASES, k=K).mean_pass_k
    print(f"\n  agent_conservative pass^k : default scorer={conf_default:.3f} "
          f"-> trajectory-weighted={conf_strict:.3f} (budget overrun now costs it)")
    assert conf_strict < conf_default, "trajectory-weighted scorer must penalize the overrun"
    print("  [check] leaderboard sorted desc ; over-budget agent measurably penalized -> OK")


if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()
    print("\nAll Day 26 solutions ran successfully.")
