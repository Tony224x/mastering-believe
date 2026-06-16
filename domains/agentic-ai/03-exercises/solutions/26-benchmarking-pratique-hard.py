"""
Solutions -- Day 26 (HARD): Practical Benchmarking

Contains solutions for:
  - Hard Ex 1: Reproducible, seeded, multi-metric eval harness (quality/cost/
               latency + variance) + Goodhart drift detection (leaderboard set
               vs holdout set) proving an overfit agent collapses on holdout.
  - Hard Ex 2: Leaderboard stability AUDIT under bootstrap resampling of cases
               -- proves a tight ranking is unstable (ranks flip) while a
               clearly dominant agent is robust.

Self-contained & offline. No network, no API key. All randomness is seeded
with random.Random(seed) so every run is fully reproducible. Minimal pieces of
02-code/26-benchmarking-pratique.py (EvalCase / pass^k / run-suite idea) are
re-embedded here -- the module is NOT imported.

Run:  python 03-exercises/solutions/26-benchmarking-pratique-hard.py
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

# Embedded from 02-code (do not import): pass^k = reliability = p_hat ** k.


def pass_k(p_hat: float, k: int) -> float:
    return p_hat ** k


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ==========================================================================
# HARD EXERCISE 1 -- Reproducible multi-metric harness + Goodhart drift
# ==========================================================================


@dataclass
class EvalCase:
    """Minimal eval case (embedded from 02-code)."""
    id: str
    bucket: str  # "leaderboard" or "holdout"


@dataclass
class SeededAgent:
    """Deterministic agent with separate quality on public vs holdout cases.

    An honest agent has quality_pub == quality_holdout.
    A Goodhart agent has quality_pub >> quality_holdout (overfit the leaderboard).
    cost / latency are simulated with bounded seeded noise.
    """
    label: str
    quality_pub: float
    quality_holdout: float
    cost_per_run: float = 0.01
    latency_per_run: float = 0.50

    def _quality_for(self, case: EvalCase) -> float:
        return self.quality_pub if case.bucket == "leaderboard" else self.quality_holdout

    def run(self, case: EvalCase, rng: random.Random) -> dict:
        """One seeded run -> success bool + simulated cost/latency."""
        success = rng.random() < self._quality_for(case)
        cost = self.cost_per_run * (1.0 + 0.2 * (rng.random() - 0.5))   # +/-10%
        latency = self.latency_per_run * (1.0 + 0.4 * (rng.random() - 0.5))  # +/-20%
        return {"success": success, "cost": cost, "latency": latency}


def run_eval(agent: SeededAgent, cases: list[EvalCase], k: int, seed: int) -> dict:
    """Run k seeded runs per case. Aggregate quality/cost/latency + p_hat variance."""
    rng = random.Random(seed)  # single stream -> fully reproducible given seed
    p_hats: list[float] = []
    pass_ks: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    per_case: dict[str, dict] = {}

    for case in cases:
        succ = 0
        for _ in range(k):
            r = agent.run(case, rng)
            succ += 1 if r["success"] else 0
            costs.append(r["cost"])
            latencies.append(r["latency"])
        p = succ / k
        p_hats.append(p)
        pass_ks.append(pass_k(p, k))
        per_case[case.id] = {"p_hat": p, "pass_k": pass_k(p, k)}

    return {
        "mean_pass_k": statistics.fmean(pass_ks) if pass_ks else 0.0,
        "mean_p_hat": statistics.fmean(p_hats) if p_hats else 0.0,
        "var_p_hat": statistics.pvariance(p_hats) if len(p_hats) > 1 else 0.0,
        "mean_cost": statistics.fmean(costs) if costs else 0.0,
        "mean_latency": statistics.fmean(latencies) if latencies else 0.0,
        "per_case": per_case,
    }


def hard_ex1_goodhart() -> None:
    print("\n" + "=" * 68)
    print("HARD EX1 -- Reproducible multi-metric harness + Goodhart drift")
    print("=" * 68)

    leaderboard = [EvalCase(f"LB-{i:02d}", "leaderboard") for i in range(8)]
    holdout = [EvalCase(f"HO-{i:02d}", "holdout") for i in range(8)]
    k = 8

    honest = SeededAgent("agent-honest", quality_pub=0.80, quality_holdout=0.80)
    goodhart = SeededAgent("agent-goodhart", quality_pub=0.95, quality_holdout=0.45)

    # --- Reproducibility: same seed -> identical results ---
    r_a = run_eval(honest, leaderboard, k, seed=2024)
    r_b = run_eval(honest, leaderboard, k, seed=2024)
    assert r_a["mean_pass_k"] == r_b["mean_pass_k"], "not reproducible at fixed seed"
    assert r_a["var_p_hat"] == r_b["var_p_hat"], "variance not reproducible"
    r_c = run_eval(honest, leaderboard, k, seed=999)  # different seed may differ
    print("\nReproducibility check (agent-honest, leaderboard):")
    print(f"  seed=2024 run#1 mean_pass^{k} = {r_a['mean_pass_k']:.4f}")
    print(f"  seed=2024 run#2 mean_pass^{k} = {r_b['mean_pass_k']:.4f}  (identical)")
    print(f"  seed=999  run    mean_pass^{k} = {r_c['mean_pass_k']:.4f}  (may differ)")

    # columns show mean pass^k on the leaderboard set vs the holdout set
    print(f"\n{'agent':<16}{'lb_p^k':>8}{'ho_p^k':>8}{'drift':>8}{'cost':>8}{'lat':>7}  verdict")
    print("-" * 68)
    verdicts: dict[str, str] = {}
    drifts: dict[str, float] = {}
    for agent in (honest, goodhart):
        lb = run_eval(agent, leaderboard, k, seed=11)
        ho = run_eval(agent, holdout, k, seed=11)
        drift = lb["mean_pass_k"] - ho["mean_pass_k"]
        verdict = "OVERFIT" if drift > 0.15 else "ROBUST"
        verdicts[agent.label] = verdict
        drifts[agent.label] = drift
        print(
            f"{agent.label:<16}{lb['mean_pass_k']:>7.3f}{ho['mean_pass_k']:>8.3f}"
            f"{drift:>+8.3f}{lb['mean_cost']:>8.4f}{lb['mean_latency']:>7.3f}  {verdict}"
        )

    # Goodhart agent: large positive drift; honest agent: near zero.
    assert drifts["agent-goodhart"] > 0.15, "goodhart should drift hard"
    assert abs(drifts["agent-honest"]) < 0.15, "honest should be stable"
    assert verdicts["agent-goodhart"] == "OVERFIT"
    assert verdicts["agent-honest"] == "ROBUST"
    # Variance of p_hat is reported and non-negative.
    assert r_a["var_p_hat"] >= 0.0

    print("\n[OK] Reproducible at fixed seed; Goodhart drift detected; honest robust.")


# ==========================================================================
# HARD EXERCISE 2 -- Leaderboard stability audit under bootstrap
# ==========================================================================


def build_agent_scores(
    true_means: dict[str, float],
    n_cases: int,
    seed: int,
    noise: float = 0.25,
) -> dict[str, list[float]]:
    """For each agent, a list of per-case pass^k values jittered around its mean."""
    rng = random.Random(seed)
    scores: dict[str, list[float]] = {label: [] for label in true_means}
    for _ in range(n_cases):
        for label, mean in true_means.items():
            jitter = (rng.random() - 0.5) * 2.0 * noise  # in [-noise, +noise]
            scores[label].append(_clamp(mean + jitter))
    return scores


def nominal_ranking(agent_scores: dict[str, list[float]]) -> list[tuple[str, float]]:
    """Rank agents by mean per-case score (descending)."""
    means = {a: statistics.fmean(v) for a, v in agent_scores.items()}
    return sorted(means.items(), key=lambda kv: kv[1], reverse=True)


def bootstrap_rankings(
    agent_scores: dict[str, list[float]],
    n_boot: int,
    seed: int,
) -> dict:
    """Bootstrap-resample cases (with replacement, same indices for all agents).

    Returns rank distributions, P(rank 1), median rank, and the top-1 flip rate
    vs the nominal #1.
    """
    rng = random.Random(seed)
    agents = list(agent_scores)
    n_cases = len(next(iter(agent_scores.values())))
    nominal_top = nominal_ranking(agent_scores)[0][0]

    rank_counts: dict[str, dict[int, int]] = {a: {} for a in agents}
    rank_samples: dict[str, list[int]] = {a: [] for a in agents}
    top1_flips = 0

    for _ in range(n_boot):
        idx = [rng.randrange(n_cases) for _ in range(n_cases)]  # shared resample
        boot_means = {
            a: statistics.fmean(agent_scores[a][i] for i in idx) for a in agents
        }
        order = sorted(agents, key=lambda a: boot_means[a], reverse=True)
        for rank, a in enumerate(order, start=1):
            rank_counts[a][rank] = rank_counts[a].get(rank, 0) + 1
            rank_samples[a].append(rank)
        if order[0] != nominal_top:
            top1_flips += 1

    p_top1 = {a: rank_counts[a].get(1, 0) / n_boot for a in agents}
    median_rank = {a: statistics.median(rank_samples[a]) for a in agents}
    return {
        "p_top1": p_top1,
        "median_rank": median_rank,
        "rank_counts": rank_counts,
        "top1_flip_rate": top1_flips / n_boot,
        "nominal_top": nominal_top,
        "n_boot": n_boot,
    }


def hard_ex2_leaderboard_audit() -> None:
    print("\n" + "=" * 68)
    print("HARD EX2 -- Leaderboard stability audit under bootstrap")
    print("=" * 68)

    # --- Tight field: 4 agents with close true means ---
    tight_means = {
        "agent-W": 0.40,
        "agent-X": 0.42,
        "agent-Y": 0.44,
        "agent-Z": 0.46,
    }
    n_cases = 30
    tight_scores = build_agent_scores(tight_means, n_cases, seed=42, noise=0.25)

    nominal = nominal_ranking(tight_scores)
    print("\nNominal ranking (tight field):")
    for rank, (a, m) in enumerate(nominal, start=1):
        print(f"  #{rank}  {a}  mean pass^k = {m:.3f}")

    audit = bootstrap_rankings(tight_scores, n_boot=3000, seed=7)
    sum_p_top1 = sum(audit["p_top1"].values())

    print(f"\nBootstrap audit ({audit['n_boot']} resamples, nominal #1 = {audit['nominal_top']}):")
    print(f"{'agent':<10}{'rank_nom':>9}{'P(#1)':>9}{'median_rank':>13}")
    print("-" * 41)
    nom_rank = {a: i for i, (a, _) in enumerate(nominal, start=1)}
    for a in tight_scores:
        print(
            f"{a:<10}{nom_rank[a]:>9}{audit['p_top1'][a]:>9.3f}"
            f"{audit['median_rank'][a]:>13.1f}"
        )
    print(f"\nTop-1 flip rate (nominal #1 dethroned) : {audit['top1_flip_rate']:.3f}")
    print(f"Sum of P(#1) over agents              : {sum_p_top1:.3f}  (~1.0 expected)")

    # The tight ranking is NOT stable.
    assert abs(sum_p_top1 - 1.0) < 1e-9, "P(#1) must sum to 1"
    assert audit["top1_flip_rate"] > 0.0, "tight ranking should flip sometimes"
    assert all(p < 1.0 for p in audit["p_top1"].values()), "no agent should own #1"

    # --- Dominant agent: clear gap -> stable ---
    dom_means = dict(tight_means)
    dom_means["agent-DOMINANT"] = 0.90
    dom_scores = build_agent_scores(dom_means, n_cases, seed=42, noise=0.25)
    dom_audit = bootstrap_rankings(dom_scores, n_boot=3000, seed=7)
    p_dom = dom_audit["p_top1"]["agent-DOMINANT"]
    print(f"\nContre-exemple stable: agent-DOMINANT (mean 0.90) -> P(#1) = {p_dom:.3f}")
    assert p_dom >= 0.99, "a clearly dominant agent should be robust at #1"

    print("\n[OK] Tight ranking unstable (flips); clear gap robust (P(#1)>=0.99).")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 68)
    print("  Day 26 HARD Solutions -- Practical Benchmarking")
    print("#" * 68)

    hard_ex1_goodhart()
    hard_ex2_leaderboard_audit()

    print("\n" + "#" * 68)
    print("  All hard solutions executed successfully.")
    print("#" * 68 + "\n")
