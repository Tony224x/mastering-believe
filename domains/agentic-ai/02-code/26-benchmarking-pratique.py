"""
Day 26 -- Practical Benchmarking: build and run an evaluation harness on your own agent.

Demonstrates:
  1. EvalCase       -- structured evaluation case (input, expected, policy, tags)
  2. MockAgent      -- deterministic mock agent parameterized by error rate
  3. Scorer         -- final-answer scoring (exact/contains) + trajectory scoring
  4. pass_k / pass_hat_k  -- reliability metric (all k runs succeed)
  5. passAt_k       -- capability metric (at least one success in k runs)
  6. run_suite()    -- execute the eval harness over a dataset with k runs per case
  7. regression_report()  -- compare baseline vs candidate, list improvements/regressions

All stdlib, no API key required. Agent and judge are mocked/deterministic.

Run:
    python domains/agentic-ai/02-code/26-benchmarking-pratique.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# ===========================================================================
# 1. DATA STRUCTURES
# ===========================================================================

@dataclass
class EvalCase:
    """A single evaluation case for an agent.

    Attributes:
        id:       Unique identifier used for tracking and regression diffs.
        input:    The task description sent to the agent.
        expected: Ground-truth answer (used by the scorer).
        policy:   Trajectory constraints:
                    required_tools   -- tools that MUST be called
                    forbidden_tools  -- tools that MUST NOT be called
                    tool_sequence    -- expected order (subset, not full sequence)
                    max_steps        -- budget
        tags:     Optional labels ('golden', 'regression', 'edge').
    """
    id: str
    input: str
    expected: str
    policy: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Result of one agent run on one EvalCase."""
    case_id: str
    output: str
    trajectory: list[str]   # ordered list of tool names called
    steps: int
    success: bool = False    # filled in by Scorer


@dataclass
class CaseReport:
    """Aggregated report for one EvalCase over k runs."""
    case_id: str
    k: int
    successes: int
    p_hat: float            # estimated success probability per run
    pass_k: float           # pass^k = p_hat^k   (reliability: all k succeed)
    pass_at_k: float        # pass@k = 1-(1-p)^k (capability: >=1 success)
    tags: list[str] = field(default_factory=list)

    @property
    def is_golden(self) -> bool:
        return "golden" in self.tags


@dataclass
class SuiteReport:
    """Aggregated report for a full evaluation suite."""
    agent_label: str
    k: int
    case_reports: list[CaseReport]

    @property
    def mean_pass_k(self) -> float:
        if not self.case_reports:
            return 0.0
        return sum(r.pass_k for r in self.case_reports) / len(self.case_reports)

    @property
    def mean_p_hat(self) -> float:
        if not self.case_reports:
            return 0.0
        return sum(r.p_hat for r in self.case_reports) / len(self.case_reports)

    def by_id(self, case_id: str) -> CaseReport | None:
        for r in self.case_reports:
            if r.case_id == case_id:
                return r
        return None


# ===========================================================================
# 2. MOCK AGENT
# ===========================================================================

# Simulated tool catalog used by the mock agent.
# In a real setup, these would be actual tool wrappers.
_TOOL_LOGIC: dict[str, dict[str, str]] = {
    "search_order": {
        "CMD-001": "order found: 2 items, status=pending",
        "CMD-002": "order found: 1 item, status=delivered",
        "CMD-003": "order not found",
    },
    "cancel_order": {
        "CMD-001": "order CMD-001 cancelled, refund initiated",
        "CMD-002": "cannot cancel: order already delivered",
    },
    "lookup_user": {
        "user-alice": "user Alice, tier=gold",
        "user-bob":   "user Bob, tier=standard",
    },
    "apply_discount": {
        "user-alice": "10% discount applied",
        "user-bob":   "5% discount applied",
    },
    "send_email": {
        "default": "email sent",
    },
}


class MockAgent:
    """Simulated agent parameterized by an error rate.

    error_rate: float in [0, 1].
        - 0.0 = always answers correctly
        - 1.0 = always answers incorrectly (wrong answer, wrong trajectory)

    The agent uses a lightweight rule-based system to simulate realistic
    tool calls and answers. The error_rate introduces random failures so
    that pass^k and pass@k can be meaningfully distinguished.
    """

    def __init__(self, label: str, error_rate: float = 0.0, seed: int = 42) -> None:
        self.label = label
        self.error_rate = max(0.0, min(1.0, error_rate))
        self._rng = random.Random(seed)

    def run(self, case: EvalCase) -> RunResult:
        """Execute the agent on an EvalCase and return a RunResult."""
        # Simulate stochastic failure
        if self._rng.random() < self.error_rate:
            return self._failed_run(case)
        return self._successful_run(case)

    def _successful_run(self, case: EvalCase) -> RunResult:
        """Simulate a successful execution: correct tools + correct answer."""
        trajectory: list[str] = []
        steps = 0

        # Simulate tool calls based on required_tools in the policy
        required = case.policy.get("required_tools", [])
        for tool in required:
            trajectory.append(tool)
            steps += 1

        # If a tool_sequence is specified, follow it
        seq = case.policy.get("tool_sequence", [])
        for tool in seq:
            if tool not in trajectory:
                trajectory.append(tool)
                steps += 1

        # Add a reasoning step
        steps += 1

        # Derive answer from expected (mock: agent "knows" the right answer)
        output = case.expected

        return RunResult(
            case_id=case.id,
            output=output,
            trajectory=trajectory,
            steps=steps,
        )

    def _failed_run(self, case: EvalCase) -> RunResult:
        """Simulate a failed execution: wrong answer and/or wrong trajectory."""
        # Sometimes call a forbidden tool, sometimes skip required tools
        trajectory: list[str] = []
        steps = 0

        forbidden = case.policy.get("forbidden_tools", [])
        if forbidden:
            # Accidentally call a forbidden tool
            trajectory.append(forbidden[0])
            steps += 1

        # Skip some required tools
        required = case.policy.get("required_tools", [])
        if required and len(required) > 1:
            # Only call the first required tool, skip the rest
            trajectory.append(required[0])
            steps += 1
        steps += 1  # reasoning step

        # Return a wrong answer
        output = f"[ERROR] Could not process: {case.input[:30]}..."

        return RunResult(
            case_id=case.id,
            output=output,
            trajectory=trajectory,
            steps=steps,
        )


# ===========================================================================
# 3. SCORER
# ===========================================================================

class Scorer:
    """Scores an agent RunResult against an EvalCase.

    Two dimensions:
    - final_answer: how well does the output match expected?
    - trajectory: does the tool call sequence satisfy the policy?

    Combined score: alpha * final + (1-alpha) * trajectory.
    A run is considered 'successful' when the combined score >= threshold.
    """

    def __init__(
        self,
        answer_mode: str = "contains",  # "exact" or "contains"
        alpha: float = 0.6,             # weight for final-answer vs trajectory
        threshold: float = 0.7,         # minimum combined score for success
    ) -> None:
        self.answer_mode = answer_mode
        self.alpha = alpha
        self.threshold = threshold

    def score_final_answer(self, output: str, expected: str) -> float:
        """Score the final answer. Returns float in [0, 1]."""
        if self.answer_mode == "exact":
            return 1.0 if output.strip() == expected.strip() else 0.0
        # contains mode (case-insensitive)
        return 1.0 if expected.lower() in output.lower() else 0.0

    def score_trajectory(self, trajectory: list[str], policy: dict) -> float:
        """Score the trajectory against policy constraints.

        Returns float in [0, 1]. Each violated constraint deducts points.
        """
        scores: list[float] = []

        # Check required tools (recall)
        required = policy.get("required_tools", [])
        if required:
            hit = sum(1 for t in required if t in trajectory) / len(required)
            scores.append(hit)

        # Check forbidden tools (precision)
        forbidden = policy.get("forbidden_tools", [])
        if forbidden:
            clean = all(t not in trajectory for t in forbidden)
            scores.append(1.0 if clean else 0.0)

        # Check tool sequence (order matters)
        seq = policy.get("tool_sequence", [])
        if seq:
            # Find relative order of each tool in trajectory
            positions: dict[str, int] = {}
            for i, t in enumerate(trajectory):
                if t in seq and t not in positions:
                    positions[t] = i
            in_order = all(
                positions.get(seq[i], -1) < positions.get(seq[i + 1], math.inf)
                for i in range(len(seq) - 1)
                if seq[i] in positions and seq[i + 1] in positions
            )
            scores.append(1.0 if in_order else 0.5)

        # Check step budget
        max_steps = policy.get("max_steps")
        if max_steps is not None:
            within_budget = len(trajectory) <= max_steps
            scores.append(1.0 if within_budget else 0.0)

        # No policy constraints = trajectory score is neutral (1.0)
        return sum(scores) / len(scores) if scores else 1.0

    def score(self, result: RunResult, case: EvalCase) -> RunResult:
        """Score a RunResult and mark it as success/failure. Returns the result."""
        answer_score = self.score_final_answer(result.output, case.expected)
        traj_score = self.score_trajectory(result.trajectory, case.policy)
        combined = self.alpha * answer_score + (1 - self.alpha) * traj_score
        result.success = combined >= self.threshold
        return result


# ===========================================================================
# 4. PASS^K AND PASS@K
# ===========================================================================

def pass_k(p_hat: float, k: int) -> float:
    """pass^k = p_hat^k.

    Probability that all k runs succeed (reliability metric).
    This is the production-relevant metric: a user asking the same
    task k times expects success every single time.

    Args:
        p_hat: Estimated per-run success probability (successes / k runs).
        k:     Number of runs.

    Returns:
        float in [0, 1].
    """
    return p_hat ** k


def pass_at_k(p_hat: float, k: int) -> float:
    """pass@k = 1 - (1-p_hat)^k.

    Probability that at least one of k runs succeeds (capability metric).
    This is the research metric: can the agent find the answer if given
    enough tries? Useful for benchmarks where best-of-k is acceptable.

    Args:
        p_hat: Estimated per-run success probability.
        k:     Number of runs.

    Returns:
        float in [0, 1].
    """
    return 1.0 - (1.0 - p_hat) ** k


def p_hat_confidence_interval(successes: int, k: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More accurate than the naive (p +/- z*sqrt(p(1-p)/n)) at small k.

    Returns:
        (lower, upper) bounds at the confidence level corresponding to z
        (default z=1.96 -> 95% CI).
    """
    if k == 0:
        return (0.0, 1.0)
    p = successes / k
    denominator = 1 + z ** 2 / k
    center = (p + z ** 2 / (2 * k)) / denominator
    margin = z * math.sqrt(p * (1 - p) / k + z ** 2 / (4 * k ** 2)) / denominator
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (lower, upper)


# ===========================================================================
# 5. EVAL HARNESS
# ===========================================================================

def run_suite(
    agent: MockAgent,
    cases: list[EvalCase],
    k: int = 5,
    scorer: Scorer | None = None,
    verbose: bool = False,
) -> SuiteReport:
    """Run the evaluation harness: k runs per case, score each run.

    Args:
        agent:   The agent to evaluate.
        cases:   List of EvalCase instances.
        k:       Number of runs per case (use k>=5 for meaningful pass^k).
        scorer:  Scorer instance. Defaults to Scorer() with contains mode.
        verbose: Print per-run details.

    Returns:
        SuiteReport with a CaseReport per case.
    """
    if scorer is None:
        scorer = Scorer()

    case_reports: list[CaseReport] = []

    for case in cases:
        successes = 0
        for run_idx in range(k):
            result = agent.run(case)
            result = scorer.score(result, case)
            if result.success:
                successes += 1
            if verbose:
                status = "OK" if result.success else "FAIL"
                print(f"  [{case.id}] run {run_idx + 1}/{k}: {status}")

        p = successes / k
        case_reports.append(CaseReport(
            case_id=case.id,
            k=k,
            successes=successes,
            p_hat=p,
            pass_k=pass_k(p, k),
            pass_at_k=pass_at_k(p, k),
            tags=list(case.tags),
        ))

    return SuiteReport(agent_label=agent.label, k=k, case_reports=case_reports)


# ===========================================================================
# 6. REGRESSION REPORT
# ===========================================================================

def regression_report(
    baseline: SuiteReport,
    candidate: SuiteReport,
    golden_only: bool = False,
    improvement_threshold: float = 0.10,
    regression_threshold: float = -0.10,
) -> None:
    """Print a regression report comparing baseline vs candidate.

    A case is flagged as:
    - IMPROVEMENT: candidate.pass_k - baseline.pass_k >= improvement_threshold
    - REGRESSION:  candidate.pass_k - baseline.pass_k <= regression_threshold
    - STABLE:      delta within thresholds

    Golden cases (tagged 'golden') are highlighted separately because
    any regression on a golden case is a blocking deploy gate.

    Args:
        baseline:              SuiteReport for the reference agent.
        candidate:             SuiteReport for the new agent.
        golden_only:           If True, only report on golden cases.
        improvement_threshold: Minimum delta to count as improvement.
        regression_threshold:  Maximum negative delta to count as regression.
    """
    all_ids = {r.case_id for r in baseline.case_reports} | {
        r.case_id for r in candidate.case_reports
    }

    improvements: list[tuple[str, float, float, list[str]]] = []
    regressions: list[tuple[str, float, float, list[str]]] = []
    stable: list[str] = []

    for case_id in sorted(all_ids):
        base_r = baseline.by_id(case_id)
        cand_r = candidate.by_id(case_id)

        if base_r is None or cand_r is None:
            continue
        if golden_only and "golden" not in base_r.tags:
            continue

        delta = cand_r.pass_k - base_r.pass_k

        if delta >= improvement_threshold:
            improvements.append((case_id, base_r.pass_k, cand_r.pass_k, base_r.tags))
        elif delta <= regression_threshold:
            regressions.append((case_id, base_r.pass_k, cand_r.pass_k, base_r.tags))
        else:
            stable.append(case_id)

    k = baseline.k
    print()
    print("=" * 60)
    print(f"REGRESSION REPORT (pass^{k})")
    print(f"  Baseline  : {baseline.agent_label}")
    print(f"  Candidate : {candidate.agent_label}")
    print("=" * 60)

    print(f"\nIMPROVEMENTS ({len(improvements)}) :")
    if improvements:
        for case_id, base_val, cand_val, tags in improvements:
            tag_str = " [GOLDEN]" if "golden" in tags else ""
            print(f"  [+] {case_id}{tag_str}  {base_val:.2f} -> {cand_val:.2f} (delta={cand_val - base_val:+.2f})")
    else:
        print("  (none)")

    print(f"\nREGRESSIONS ({len(regressions)}) :")
    blocking = False
    if regressions:
        for case_id, base_val, cand_val, tags in regressions:
            tag_str = ""
            if "golden" in tags:
                tag_str = " [GOLDEN -- BLOCKING]"
                blocking = True
            print(f"  [-] {case_id}{tag_str}  {base_val:.2f} -> {cand_val:.2f} (delta={cand_val - base_val:+.2f})")
    else:
        print("  (none)")

    print(f"\nSTABLE : {len(stable)} case(s)")

    print("\nSUMMARY :")
    print(f"  pass^{k} baseline  : {baseline.mean_pass_k:.3f}")
    print(f"  pass^{k} candidate : {candidate.mean_pass_k:.3f}")
    delta_global = candidate.mean_pass_k - baseline.mean_pass_k
    pct = (delta_global / baseline.mean_pass_k * 100) if baseline.mean_pass_k > 0 else float("inf")
    print(f"  Delta global       : {delta_global:+.3f} ({pct:+.1f}%)")
    print(f"  p_hat baseline     : {baseline.mean_p_hat:.3f}")
    print(f"  p_hat candidate    : {candidate.mean_p_hat:.3f}")

    if blocking:
        verdict = "BLOCKED -- golden case regression detected, do NOT deploy"
    elif regressions and len(regressions) >= len(improvements):
        verdict = "AMBIGUOUS -- improvements do not outweigh regressions, inspect manually"
    elif candidate.mean_pass_k > baseline.mean_pass_k:
        verdict = "APPROVED -- candidate is strictly better"
    else:
        verdict = "NEUTRAL -- no significant change"

    print(f"\n  VERDICT : {verdict}")
    print("=" * 60)


# ===========================================================================
# 7. DEMO DATASET
# ===========================================================================

DEMO_CASES: list[EvalCase] = [
    # --- Golden cases: must always pass ---
    EvalCase(
        id="CMD-001-cancel",
        input="Cancel order CMD-001 for user Alice",
        expected="order CMD-001 cancelled, refund initiated",
        policy={
            "required_tools": ["search_order", "cancel_order"],
            "forbidden_tools": ["delete_record"],
            "max_steps": 5,
        },
        tags=["golden"],
    ),
    EvalCase(
        id="CMD-002-status",
        input="What is the status of order CMD-002?",
        expected="status=delivered",
        policy={
            "required_tools": ["search_order"],
            "max_steps": 3,
        },
        tags=["golden"],
    ),
    EvalCase(
        id="USR-alice-discount",
        input="Apply a discount for user-alice",
        expected="10% discount applied",
        policy={
            "required_tools": ["lookup_user", "apply_discount"],
            "tool_sequence": ["lookup_user", "apply_discount"],
            "max_steps": 4,
        },
        tags=["golden"],
    ),
    # --- Regression cases: previously found bugs ---
    EvalCase(
        id="CMD-003-not-found",
        input="Cancel order CMD-003 (does not exist)",
        expected="order not found",
        policy={
            "required_tools": ["search_order"],
            "forbidden_tools": ["cancel_order"],
            "max_steps": 3,
        },
        tags=["regression", "edge"],
    ),
    EvalCase(
        id="USR-bob-email",
        input="Send a confirmation email to user-bob after applying discount",
        expected="email sent",
        policy={
            "required_tools": ["lookup_user", "apply_discount", "send_email"],
            "tool_sequence": ["lookup_user", "apply_discount", "send_email"],
            "max_steps": 6,
        },
        tags=["regression"],
    ),
    # --- Edge cases ---
    EvalCase(
        id="CMD-002-cancel-delivered",
        input="Try to cancel already-delivered order CMD-002",
        expected="cannot cancel: order already delivered",
        policy={
            "required_tools": ["search_order", "cancel_order"],
            "max_steps": 4,
        },
        tags=["edge"],
    ),
]


# ===========================================================================
# 8. MAIN DEMO
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 26 -- Practical Agent Evaluation Harness")
    print("=" * 60)

    # Baseline agent: moderate error rate (simulates an imperfect deployed agent)
    baseline_agent = MockAgent(label="agent-v1.0", error_rate=0.35, seed=99)

    # Candidate agent: lower error rate (simulates an improved version)
    candidate_agent = MockAgent(label="agent-v1.1", error_rate=0.15, seed=77)

    K = 5           # runs per case
    scorer = Scorer(answer_mode="contains", alpha=0.6, threshold=0.7)

    print(f"\nRunning evaluation harness with k={K} runs per case...")
    print(f"Dataset: {len(DEMO_CASES)} cases\n")

    # Run evaluation for both agents
    print("-- Evaluating baseline --")
    baseline_report = run_suite(baseline_agent, DEMO_CASES, k=K, scorer=scorer, verbose=True)

    print("\n-- Evaluating candidate --")
    candidate_report = run_suite(candidate_agent, DEMO_CASES, k=K, scorer=scorer, verbose=True)

    # Print individual case reports
    print("\n" + "=" * 60)
    print(f"CASE-LEVEL RESULTS (k={K})")
    print("=" * 60)
    print(f"{'Case ID':<30} {'Baseline':>10} {'Candidate':>10} {'Delta':>8}")
    print("-" * 62)
    for case in DEMO_CASES:
        b = baseline_report.by_id(case.id)
        c = candidate_report.by_id(case.id)
        if b and c:
            delta = c.pass_k - b.pass_k
            golden = " *" if "golden" in case.tags else ""
            print(
                f"{case.id + golden:<30}"
                f"  p^{K}={b.pass_k:.2f} ({b.successes}/{K})"
                f"  p^{K}={c.pass_k:.2f} ({c.successes}/{K})"
                f"  {delta:+.2f}"
            )

    print("\n* = golden case (blocking regression gate)")

    # Pass^k vs Pass@k illustration
    print("\n" + "=" * 60)
    print("PASS^K vs PASS@K ILLUSTRATION")
    print("=" * 60)
    print(f"\n{'p_hat':>8} {'pass^3':>8} {'pass^5':>8} {'pass@3':>8} {'pass@5':>8}")
    print("-" * 45)
    for p in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(
            f"{p:>8.1f}"
            f"  {pass_k(p, 3):>6.3f}"
            f"  {pass_k(p, 5):>6.3f}"
            f"  {pass_at_k(p, 3):>6.3f}"
            f"  {pass_at_k(p, 5):>6.3f}"
        )

    # Confidence interval illustration
    print("\n" + "=" * 60)
    print("CONFIDENCE INTERVALS ON P_HAT (95%)")
    print("=" * 60)
    for successes, k_val in [(3, 5), (4, 5), (8, 10), (14, 20)]:
        p = successes / k_val
        lo, hi = p_hat_confidence_interval(successes, k_val)
        print(f"  {successes}/{k_val} successes -> p_hat={p:.2f}, 95% CI=[{lo:.2f}, {hi:.2f}]")

    # Regression report
    regression_report(baseline_report, candidate_report)

    print("\nDemo complete.")
