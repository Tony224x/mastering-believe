"""
Solutions -- Day 28 (MEDIUM): Capstone build & eval extensions

Contains solutions for:
  - Medium Ex 1: pass^k vs pass@k (reliability vs coverage) and their divergence
  - Medium Ex 2: Wilson confidence interval on p_hat; only call an improvement
                 real when the CIs separate
  - Medium Ex 3: cost/quality axis -- quality-per-dollar can favour a cheaper,
                 slightly-less-reliable agent

These EXTEND the J28 capstone. To run fully offline & standalone, this file
embeds a faithful mini-DeepOpsAgent + pass^k eval harness (deterministic mock
LLM via an injectable error_rate + a seeded random.Random; the coder fix outcome
is SIMULATED, not spawned in a subprocess) -- it does NOT import
02-code/28-capstone-build-eval.py.

Run:  python 03-exercises/solutions/28-capstone-build-eval-medium.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ==========================================================================
# EMBEDDED MINI-CAPSTONE (offline stand-in for 02-code/28-capstone-build-eval.py)
# ==========================================================================
# Faithful but minimal: a planner orchestrating context-isolated sub-agents
# (research / code / verify). The coder either fixes the planted bug or "gives
# up" with probability error_rate. The fix is SIMULATED (no subprocess) so the
# run is fast, pure-python and deterministic for a given seed.


@dataclass
class ModelRouter:
    """Cost-aware weak/strong routing (mock), mirrors J24/J27."""
    threshold: int = 12
    cost_weak: float = 1.0
    cost_strong: float = 8.0
    total_cost: float = 0.0
    # force_tier lets a variant pin every call to one tier (used by Ex 3).
    force_tier: str | None = None
    routed: dict = field(default_factory=lambda: {"weak": 0, "strong": 0})

    def route(self, task: str) -> str:
        if self.force_tier in ("weak", "strong"):
            tier = self.force_tier
        else:
            tier = "strong" if len(task.split()) >= self.threshold else "weak"
        self.routed[tier] += 1
        self.total_cost += self.cost_strong if tier == "strong" else self.cost_weak
        return tier


@dataclass
class AgentResult:
    output: str
    trajectory: list[str] = field(default_factory=list)
    steps: int = 0
    cost: float = 0.0           # total router cost for this run (Ex 3)


class DeepOpsAgent:
    """Planner orchestrating isolated sub-agents to fix a planted bug, with a
    stochastic failure (error_rate) so reliability (pass^k) is measurable.

    `router_tier` pins the model tier for the whole run (None = auto): a variant
    can be made systematically expensive ('strong') or cheap ('weak').
    """

    def __init__(self, label: str = "deep-ops", error_rate: float = 0.0,
                 seed: int = 0, router_tier: str | None = None) -> None:
        self.label = label
        self.error_rate = max(0.0, min(1.0, error_rate))
        self.router_tier = router_tier
        self._rng = random.Random(seed)

    def _plan(self) -> list[str]:
        return ["research the bug", "fix the code", "verify the fix"]

    def solve(self, task: str) -> AgentResult:
        router = ModelRouter(force_tier=self.router_tier)
        traj: list[str] = ["plan"]
        self._plan()

        # The agent occasionally "loses focus" and the coder gives up (no fix).
        will_fix = self._rng.random() >= self.error_rate

        # research sub-agent (isolated context) -- routes a model call.
        router.route(task)
        traj.append("research")

        # coder sub-agent: red -> search -> (edit) -> green, all simulated.
        router.route(task)
        traj.append("code")
        traj.append("search")
        code_result = "FAILED"
        if will_fix:
            traj.append("edit")
            code_result = "fixed"
        traj.append("run_tests")

        # verifier sub-agent.
        router.route(task)
        traj.append("verify")
        verdict = "verified-ok" if code_result == "fixed" else "verify-failed"

        output = ("SUCCESS: bug fixed and verified" if verdict == "verified-ok"
                  else f"FAILURE: {code_result}")
        return AgentResult(output=output, trajectory=traj, steps=len(traj),
                           cost=router.total_cost)


@dataclass
class EvalCase:
    id: str
    task: str
    expected: str
    required_steps: list[str] = field(default_factory=list)
    max_steps: int = 12
    tags: list[str] = field(default_factory=list)


def score(result: AgentResult, case: EvalCase) -> bool:
    final_ok = case.expected.lower() in result.output.lower()
    traj_ok = all(s in result.trajectory for s in case.required_steps)
    budget_ok = result.steps <= case.max_steps
    return final_ok and traj_ok and budget_ok


@dataclass
class CaseReport:
    case_id: str
    successes: int
    k: int
    tags: list[str] = field(default_factory=list)
    mean_cost: float = 0.0      # mean router cost across the k runs (Ex 3)

    @property
    def p_hat(self) -> float:
        return self.successes / self.k if self.k else 0.0

    @property
    def pass_k(self) -> float:
        # pass^k: all k attempts succeed -> reliability.
        return self.p_hat ** self.k

    @property
    def pass_at_k(self) -> float:
        # pass@k: at least one of k attempts succeeds -> coverage / best-of-k.
        return 1.0 - (1.0 - self.p_hat) ** self.k


# Demo cases reused across exercises.
GOLDEN_CASE = EvalCase("fix-add", "fix the add bug in calc",
                       expected="SUCCESS",
                       required_steps=["plan", "research", "code", "edit",
                                       "run_tests", "verify"],
                       max_steps=12, tags=["golden"])


# ==========================================================================
# MEDIUM EXERCISE 1 -- pass^k vs pass@k
# ==========================================================================


def compare_pass_metrics(agent: DeepOpsAgent, case: EvalCase, k: int) -> CaseReport:
    successes = sum(1 for _ in range(k) if score(agent.solve(case.task), case))
    return CaseReport(case.id, successes, k, tags=list(case.tags))


def medium_ex1_pass_k_vs_at_k() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 1: pass^k (reliability) vs pass@k (coverage)")
    print("=" * 60)

    k = 5
    rows = []
    print(f"\n  k={k}   (same case, same seed, varying error_rate)")
    print(f"  {'error_rate':>10}{'p_hat':>9}{'pass^k':>9}{'pass@k':>9}{'gap':>9}")
    print("  " + "-" * 45)
    for er in (0.0, 0.2, 0.5, 0.8):
        agent = DeepOpsAgent(label=f"er={er}", error_rate=er, seed=7)
        rep = compare_pass_metrics(agent, GOLDEN_CASE, k)
        gap = rep.pass_at_k - rep.pass_k
        rows.append((er, rep.p_hat, rep.pass_k, rep.pass_at_k, gap))
        print(f"  {er:>10.2f}{rep.p_hat:>9.3f}{rep.pass_k:>9.3f}"
              f"{rep.pass_at_k:>9.3f}{gap:>9.3f}")

        # Fundamental ordering: pass^k <= p_hat <= pass@k.
        eps = 1e-9
        assert rep.pass_k <= rep.p_hat + eps, (rep.pass_k, rep.p_hat)
        assert rep.p_hat <= rep.pass_at_k + eps, (rep.p_hat, rep.pass_at_k)

    # The gap (pass@k - pass^k) widens as the agent gets less reliable, then can
    # narrow only at the extreme p_hat=0 corner. Check it grows across the
    # informative middle range (er 0.0 -> 0.5), where 0<p_hat<1.
    gaps_mid = [g for er, *_, g in rows if er <= 0.5]
    assert all(gaps_mid[i] <= gaps_mid[i + 1] + 1e-9 for i in range(len(gaps_mid) - 1)), gaps_mid

    print("\n  PASS -- ordering pass^k <= p_hat <= pass@k holds; the two metrics")
    print("         diverge as reliability drops.\n")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Wilson confidence interval on p_hat
# ==========================================================================


def wilson_interval(successes: int, k: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (stdlib math only)."""
    if k == 0:
        return (0.0, 1.0)
    p = successes / k
    z2 = z * z
    denom = 1.0 + z2 / k
    center = (p + z2 / (2 * k)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z2 / (4 * k)) / k)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def ci_of(report: CaseReport) -> tuple[float, float]:
    return wilson_interval(report.successes, report.k)


def is_real_improvement(report_a: CaseReport, report_b: CaseReport) -> bool:
    """B is a *real* improvement over A only if its CI sits entirely above A's."""
    _a_low, a_high = ci_of(report_a)
    b_low, _b_high = ci_of(report_b)
    return b_low > a_high


def medium_ex2_confidence_interval() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 2: Wilson CI on p_hat -- when is an improvement real?")
    print("=" * 60)

    # (1) CI narrows as k grows (fixed error_rate, fixed seed).
    print("\n  CI width vs k (error_rate=0.30):")
    widths = []
    for k in (5, 20, 80):
        agent = DeepOpsAgent(error_rate=0.30, seed=3)
        rep = compare_pass_metrics(agent, GOLDEN_CASE, k)
        low, high = ci_of(rep)
        widths.append(high - low)
        print(f"    k={k:>3} p_hat={rep.p_hat:.3f} CI=[{low:.3f}, {high:.3f}] "
              f"width={high - low:.3f}")
    assert widths[0] > widths[1] > widths[2], widths

    # (2) Noisy near-tie at small k -> CIs overlap -> NOT a real improvement.
    print("\n  near-tie (er 0.40 vs 0.35), small k -> prudent verdict:")
    k_small = 10
    a_close = compare_pass_metrics(DeepOpsAgent(error_rate=0.40, seed=11), GOLDEN_CASE, k_small)
    b_close = compare_pass_metrics(DeepOpsAgent(error_rate=0.35, seed=12), GOLDEN_CASE, k_small)
    print(f"    A: p_hat={a_close.p_hat:.3f} CI={tuple(round(x, 3) for x in ci_of(a_close))}")
    print(f"    B: p_hat={b_close.p_hat:.3f} CI={tuple(round(x, 3) for x in ci_of(b_close))}")
    real_close = is_real_improvement(a_close, b_close)
    print(f"    is_real_improvement = {real_close}")
    assert real_close is False, "overlapping CIs must NOT be declared an improvement"

    # (3) Clear separation -> CIs disjoint -> real improvement.
    print("\n  clear gap (er 0.60 vs 0.05), larger k -> real improvement:")
    k_big = 40
    a_far = compare_pass_metrics(DeepOpsAgent(error_rate=0.60, seed=21), GOLDEN_CASE, k_big)
    b_far = compare_pass_metrics(DeepOpsAgent(error_rate=0.05, seed=22), GOLDEN_CASE, k_big)
    print(f"    A: p_hat={a_far.p_hat:.3f} CI={tuple(round(x, 3) for x in ci_of(a_far))}")
    print(f"    B: p_hat={b_far.p_hat:.3f} CI={tuple(round(x, 3) for x in ci_of(b_far))}")
    real_far = is_real_improvement(a_far, b_far)
    print(f"    is_real_improvement = {real_far}")
    assert real_far is True, "disjoint CIs should be declared a real improvement"

    print("\n  PASS -- CI narrows with k; improvement only declared when CIs separate.\n")


# ==========================================================================
# MEDIUM EXERCISE 3 -- cost/quality axis: quality-per-dollar
# ==========================================================================


def run_suite_cost(agent: DeepOpsAgent, cases: list[EvalCase], k: int = 5) -> dict:
    reports: list[CaseReport] = []
    for case in cases:
        successes = 0
        cost_sum = 0.0
        for _ in range(k):
            res = agent.solve(case.task)
            cost_sum += res.cost
            if score(res, case):
                successes += 1
        reports.append(CaseReport(case.id, successes, k, tags=list(case.tags),
                                  mean_cost=cost_sum / k))
    mean_pass_k = sum(r.pass_k for r in reports) / len(reports) if reports else 0.0
    mean_cost = sum(r.mean_cost for r in reports) / len(reports) if reports else 0.0
    qpd = mean_pass_k / mean_cost if mean_cost else 0.0
    return {"label": agent.label, "k": k, "reports": reports,
            "mean_pass_k": mean_pass_k, "mean_cost": mean_cost,
            "quality_per_dollar": qpd}


def medium_ex3_quality_per_dollar() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 3: cost/quality axis -- quality-per-dollar")
    print("=" * 60)

    cases = [GOLDEN_CASE,
             EvalCase("fix-add-2", "repair calc.add so tests pass",
                      expected="SUCCESS",
                      required_steps=["code", "run_tests", "verify"],
                      max_steps=12, tags=["golden"])]
    k = 8

    # Reliable but always routes to the expensive 'strong' tier.
    agent_strong = DeepOpsAgent(label="strong", error_rate=0.05, seed=5,
                                router_tier="strong")
    # A bit less reliable but always routes to the cheap 'weak' tier.
    agent_cheap = DeepOpsAgent(label="cheap", error_rate=0.15, seed=5,
                               router_tier="weak")

    rs = run_suite_cost(agent_strong, cases, k=k)
    rc = run_suite_cost(agent_cheap, cases, k=k)

    print(f"\n  {'variant':>8}{'mean pass^k':>13}{'mean cost':>11}{'quality/$':>12}")
    print("  " + "-" * 44)
    for r in (rs, rc):
        print(f"  {r['label']:>8}{r['mean_pass_k']:>13.3f}"
              f"{r['mean_cost']:>11.3f}{r['quality_per_dollar']:>12.4f}")

    # Strong is at least as reliable, but cheap wins on quality-per-dollar.
    assert rs["mean_pass_k"] >= rc["mean_pass_k"], (rs["mean_pass_k"], rc["mean_pass_k"])
    assert rc["mean_cost"] < rs["mean_cost"], (rc["mean_cost"], rs["mean_cost"])
    assert rc["quality_per_dollar"] > rs["quality_per_dollar"], (
        rc["quality_per_dollar"], rs["quality_per_dollar"])

    print("\n  PASS -- cheaper agent wins on quality/$ despite a lower pass^k.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 28 MEDIUM Solutions -- Capstone build & eval extensions")
    print("#" * 60)

    medium_ex1_pass_k_vs_at_k()
    medium_ex2_confidence_interval()
    medium_ex3_quality_per_dollar()

    print("\n" + "#" * 60)
    print("  All medium solutions executed successfully.")
    print("#" * 60 + "\n")
