"""
Solutions -- Day 26 (MEDIUM): Practical Benchmarking

Contains solutions for:
  - Medium Ex 1: pass@k / pass^k estimator + bootstrap confidence interval
                 (CI narrows as the number of runs grows).
  - Medium Ex 2: Benchmark-contamination detector (train/test overlap via
                 Jaccard) -- proves the "clean" score <= raw score.
  - Medium Ex 3: A/B harness with a two-proportion z-test deciding whether
                 the difference between two agents is real or just noise.

Self-contained & offline. No network, no API key. All randomness is seeded
with random.Random(seed) so every run is reproducible. Minimal pieces of
02-code/26-benchmarking-pratique.py (pass_k / pass_at_k, MockAgent idea) are
re-embedded here -- the module is NOT imported.

Run:  python 03-exercises/solutions/26-benchmarking-pratique-medium.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# Embedded from 02-code (do not import the module):
# pass^k = reliability (all k runs succeed), pass@k = capability (>=1 success).


def pass_k(p_hat: float, k: int) -> float:
    """pass^k = p_hat ** k -- probability that ALL k runs succeed (reliability)."""
    return p_hat ** k


def pass_at_k(p_hat: float, k: int) -> float:
    """pass@k = 1 - (1 - p_hat) ** k -- probability of >=1 success (capability)."""
    return 1.0 - (1.0 - p_hat) ** k


# ==========================================================================
# MEDIUM EXERCISE 1 -- pass@k / pass^k estimator + bootstrap CI
# ==========================================================================


def simulate_runs(p_true: float, n: int, seed: int) -> list[int]:
    """Generate n binary run outcomes (1=success) for a case of true prob p_true."""
    rng = random.Random(seed)
    return [1 if rng.random() < p_true else 0 for _ in range(n)]


def estimate(runs: list[int], k: int) -> dict:
    """Estimate p_hat, pass^k and pass@k from a list of binary run outcomes."""
    n = len(runs)
    p_hat = (sum(runs) / n) if n else 0.0
    return {
        "p_hat": p_hat,
        "pass_k": pass_k(p_hat, k),
        "pass_at_k": pass_at_k(p_hat, k),
    }


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1]) on an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(
    runs: list[int],
    k: int,
    n_resamples: int,
    seed: int,
    z_level: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap CI on pass^k: resample runs WITH replacement, recompute pass^k."""
    rng = random.Random(seed)
    n = len(runs)
    stats: list[float] = []
    for _ in range(n_resamples):
        sample = [runs[rng.randrange(n)] for _ in range(n)]  # with replacement
        p = sum(sample) / n
        stats.append(pass_k(p, k))
    stats.sort()
    tail = (1.0 - z_level) / 2.0
    lo = max(0.0, _percentile(stats, tail))
    hi = min(1.0, _percentile(stats, 1.0 - tail))
    return (lo, hi)


def medium_ex1_bootstrap_ci() -> None:
    print("\n" + "=" * 64)
    print("MEDIUM EX1 -- pass@k / pass^k estimator + bootstrap CI")
    print("=" * 64)

    k = 3
    # CI must narrow as n grows: same p_true, deterministic run generation.
    p_true = 0.7
    runs_small = simulate_runs(p_true, n=10, seed=1)
    runs_large = simulate_runs(p_true, n=200, seed=1)

    ci_small = bootstrap_ci(runs_small, k, n_resamples=2000, seed=7)
    ci_large = bootstrap_ci(runs_large, k, n_resamples=2000, seed=7)
    width_small = ci_small[1] - ci_small[0]
    width_large = ci_large[1] - ci_large[0]

    print(f"\nTrue p = {p_true}, metric = pass^{k}")
    print(f"  n=10 :  CI95 = [{ci_small[0]:.3f}, {ci_small[1]:.3f}]  width={width_small:.3f}")
    print(f"  n=200:  CI95 = [{ci_large[0]:.3f}, {ci_large[1]:.3f}]  width={width_large:.3f}")

    assert width_large < width_small, "CI should narrow with more runs"

    print(f"\n{'p_true':>7} {'p_hat':>7} {'pass^3':>8} {'pass@3':>8} {'pass^3 CI95':>22}")
    print("-" * 56)
    for p in [0.5, 0.7, 0.9]:
        runs = simulate_runs(p, n=200, seed=11)
        est = estimate(runs, k)
        lo, hi = bootstrap_ci(runs, k, n_resamples=1500, seed=13)
        print(
            f"{p:>7.2f} {est['p_hat']:>7.3f} {est['pass_k']:>8.3f} "
            f"{est['pass_at_k']:>8.3f}   [{lo:.3f}, {hi:.3f}]"
        )
        # Capability >= reliability, strictly greater while p_hat in (0,1).
        assert est["pass_at_k"] >= est["pass_k"]
        if 0.0 < est["p_hat"] < 1.0:
            assert est["pass_at_k"] > est["pass_k"]
        assert 0.0 <= lo <= hi <= 1.0

    print("\n[OK] CI narrows with n; pass@k > pass^k for p<1; bounds in [0,1].")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Benchmark-contamination detector
# ==========================================================================


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip edge punctuation (stdlib only)."""
    low = text.lower()
    tokens = []
    for raw in low.split():
        tokens.append(raw.strip(".,;:!?'\"()[]{}"))
    return " ".join(t for t in tokens if t)


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity on the word sets of two normalized strings."""
    sa = set(normalize(a).split())
    sb = set(normalize(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union


def detect_contamination(
    train_set: list[str],
    test_set: list[str],
    threshold: float = 0.8,
) -> dict:
    """Flag any test item whose max Jaccard against the train set >= threshold."""
    flagged: list[str] = []
    details: dict[str, float] = {}
    for item in test_set:
        best = max((jaccard(item, t) for t in train_set), default=0.0)
        details[item] = best
        if best >= threshold:
            flagged.append(item)
    rate = len(flagged) / len(test_set) if test_set else 0.0
    return {"flagged": flagged, "contamination_rate": rate, "details": details}


def _mock_agent_score(items: list[str], leaked: set[str]) -> float:
    """Deterministic mock: agent 'aces' leaked items, ~average on fresh ones."""
    if not items:
        return 0.0
    total = 0.0
    for it in items:
        total += 1.0 if it in leaked else 0.5  # memorized vs. genuinely solved
    return total / len(items)


def medium_ex2_contamination() -> None:
    print("\n" + "=" * 64)
    print("MEDIUM EX2 -- Benchmark-contamination detector")
    print("=" * 64)

    train_set = [
        "cancel order CMD-001 for user alice",
        "what is the status of order CMD-002",
        "apply a discount for user bob",
        "send a confirmation email to the customer",
    ]
    test_set = [
        "Cancel order CMD-001 for user Alice.",          # exact leak (paraphrase)
        "What is the status of order CMD-002?",           # exact leak
        "Refund the shipping fee for order CMD-555",      # fresh, unseen
        "Update the billing country for user carol",      # fresh, unseen
    ]

    report = detect_contamination(train_set, test_set, threshold=0.8)
    leaked = set(report["flagged"])

    print(f"\nThreshold = 0.8 -> contamination_rate = {report['contamination_rate']:.2f}")
    print("Per-item max Jaccard vs train:")
    for item, sim in report["details"].items():
        flag = "  <-- LEAK" if item in leaked else ""
        print(f"  {sim:.2f}  {item[:48]!r}{flag}")

    raw_score = _mock_agent_score(test_set, leaked)
    clean_items = [it for it in test_set if it not in leaked]
    clean_score = _mock_agent_score(clean_items, leaked)

    print(f"\nRaw score   (full test) : {raw_score:.3f}")
    print(f"Clean score (no leaks)  : {clean_score:.3f}")
    print(f"Inflation removed       : {raw_score - clean_score:+.3f}")

    assert report["contamination_rate"] > 0.0, "leaks should be detected"
    assert clean_score <= raw_score, "removing leaks must not raise the score"
    assert 0.0 <= report["contamination_rate"] <= 1.0
    # Fresh items must not be flagged (no false positives).
    fresh = "Update the billing country for user carol"
    assert fresh not in leaked, "fresh item wrongly flagged as contaminated"

    print("\n[OK] Leaks flagged, clean<=raw, no false positive on fresh items.")


# ==========================================================================
# MEDIUM EXERCISE 3 -- A/B harness + two-proportion z-test
# ==========================================================================


@dataclass
class MockAgent:
    """Deterministic agent: per run, succeeds with prob (1 - error_rate)."""
    label: str
    error_rate: float = 0.0
    seed: int = 42
    _rng: random.Random = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def run_once(self) -> bool:
        return self._rng.random() >= self.error_rate


def run_ab(agent: MockAgent, n_cases: int, k: int) -> tuple[int, int]:
    """Run agent over n_cases * k runs, return (successes, total)."""
    total = n_cases * k
    successes = sum(1 for _ in range(total) if agent.run_once())
    return successes, total


def two_proportion_ztest(s_a: int, n_a: int, s_b: int, n_b: int) -> tuple[float, float]:
    """Two-proportion z-test. Returns (z, two-sided p_value) via math.erf."""
    p_a = s_a / n_a
    p_b = s_b / n_b
    p_pool = (s_a + s_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0.0:
        return (0.0, 1.0)
    z = (p_b - p_a) / se
    # two-sided p-value from the standard normal CDF: 2 * (1 - Phi(|z|))
    cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
    p_value = 2.0 * (1.0 - cdf)
    return (z, p_value)


def ab_decision(s_a: int, n_a: int, s_b: int, n_b: int, alpha: float = 0.05) -> str:
    z, p_value = two_proportion_ztest(s_a, n_a, s_b, n_b)
    prop_a, prop_b = s_a / n_a, s_b / n_b
    if p_value < alpha and prop_b > prop_a:
        return "B significativement meilleur"
    if p_value < alpha and prop_a > prop_b:
        return "A significativement meilleur"
    return "difference non significative (bruit)"


def medium_ex3_ab_test() -> None:
    print("\n" + "=" * 64)
    print("MEDIUM EX3 -- A/B harness + two-proportion z-test")
    print("=" * 64)

    n_cases, k = 40, 10  # 400 runs per agent -> enough power

    # Scenario (a): large gap -> should be SIGNIFICANT.
    a1 = MockAgent("A-baseline", error_rate=0.40, seed=101)
    b1 = MockAgent("B-candidate", error_rate=0.15, seed=202)
    sa1, na1 = run_ab(a1, n_cases, k)
    sb1, nb1 = run_ab(b1, n_cases, k)
    z1, p1 = two_proportion_ztest(sa1, na1, sb1, nb1)
    verdict1 = ab_decision(sa1, na1, sb1, nb1)
    print(f"\nScenario (a) large gap:")
    print(f"  A: {sa1}/{na1} = {sa1/na1:.3f}   B: {sb1}/{nb1} = {sb1/nb1:.3f}")
    print(f"  z={z1:+.2f}  p={p1:.4f}  -> {verdict1}")
    assert verdict1 == "B significativement meilleur", verdict1

    # Scenario (b): tiny gap -> should be NOT significant.
    a2 = MockAgent("A-baseline", error_rate=0.30, seed=303)
    b2 = MockAgent("B-candidate", error_rate=0.29, seed=404)
    sa2, na2 = run_ab(a2, n_cases, k)
    sb2, nb2 = run_ab(b2, n_cases, k)
    z2, p2 = two_proportion_ztest(sa2, na2, sb2, nb2)
    verdict2 = ab_decision(sa2, na2, sb2, nb2)
    print(f"\nScenario (b) tiny gap:")
    print(f"  A: {sa2}/{na2} = {sa2/na2:.3f}   B: {sb2}/{nb2} = {sb2/nb2:.3f}")
    print(f"  z={z2:+.2f}  p={p2:.4f}  -> {verdict2}")
    assert verdict2 == "difference non significative (bruit)", verdict2

    assert p1 < 0.05 and p2 >= 0.05
    print("\n[OK] Large gap -> significant; tiny gap -> noise.")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 64)
    print("  Day 26 MEDIUM Solutions -- Practical Benchmarking")
    print("#" * 64)

    medium_ex1_bootstrap_ci()
    medium_ex2_contamination()
    medium_ex3_ab_test()

    print("\n" + "#" * 64)
    print("  All medium solutions executed successfully.")
    print("#" * 64 + "\n")
