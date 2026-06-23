"""
Solutions -- Day 24 (HARD): Inference engineering (advanced serving)

Contains solutions for:
  - Hard Ex 1: speculative decoding simulator (draft + verify) measuring
               acceptance rate and speedup vs target-only baseline, with an
               ablation when acceptance collapses
  - Hard Ex 2: inference-config optimizer sweeping batch size x quantization x
               prompt-cache, computing the cost/latency Pareto frontier, and
               returning the cheapest valid config under a latency SLO +
               quality floor

Self-contained & offline: no network, no API, no real model, no wall-clock.
All costs are in *compute units / steps* and all randomness is seeded, so the
file is fully deterministic. Building blocks of 02-code/24-inference-engineering.py
(model/cost ideas) are re-modeled inline -- nothing is imported.

Run:  python 03-exercises/solutions/24-inference-engineering-hard.py
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass


# ===========================================================================
# HARD EXERCISE 1 -- Speculative decoding simulator + ablation
# ===========================================================================
#
# Idea: a small DRAFT model proposes k tokens; the big TARGET model verifies
# them in ONE pass. Every token in the accepted prefix is committed at once; at
# the first rejection we keep the accepted prefix + 1 corrected target token.
# The speedup depends on the acceptance rate. We compare against a baseline
# where the target decodes alone (1 token/step).


def accepted_count(k: int, accept_prob: float, rng: random.Random) -> int:
    """Token-by-token acceptance: stop at the first rejection.

    Returns the number of DRAFT tokens accepted (0..k). This mirrors the
    sequential verification of speculative decoding: the first rejected token
    invalidates everything after it.
    """
    accepted = 0
    for _ in range(k):
        if rng.random() < accept_prob:
            accepted += 1
        else:
            break
    return accepted


class SpeculativeDecoder:
    def __init__(self, k: int, draft_cost: float, accept_prob: float, seed: int) -> None:
        self.k = k
        self.draft_cost = draft_cost            # compute per drafted token (<1)
        self.accept_prob = accept_prob
        self.seed = seed

    def generate(self, n_tokens: int) -> dict:
        rng = random.Random(self.seed)
        produced = 0
        target_steps = 0                        # one verification step per round
        draft_compute = 0.0
        total_proposed = 0
        total_accepted = 0

        while produced < n_tokens:
            # Draft proposes k tokens (cheap compute).
            draft_compute += self.k * self.draft_cost
            total_proposed += self.k
            acc = accepted_count(self.k, self.accept_prob, rng)
            total_accepted += acc
            # Target does ONE verification step: it commits the `acc` accepted
            # draft tokens AND emits one corrected token => acc + 1 tokens/round.
            target_steps += 1
            produced += acc + 1

        total_cost = target_steps + draft_compute
        acceptance_rate = total_accepted / total_proposed if total_proposed else 0.0
        return {
            "tokens_produced": produced,
            "target_steps": target_steps,
            "draft_compute": draft_compute,
            "total_cost": total_cost,
            "acceptance_rate": acceptance_rate,
        }


def hard_ex1_speculative() -> None:
    print("=" * 64)
    print("HARD EX1 -- speculative decoding (draft+verify) + ablation")
    print("=" * 64)

    n_tokens = 400
    k = 4
    draft_cost = 0.15           # draft token ~15% the cost of a target step

    # Baseline: target alone, 1 step per token.
    baseline_cost = n_tokens

    # accepted_count sanity: always within [0, k].
    rng = random.Random(0)
    for _ in range(200):
        a = accepted_count(k, 0.7, rng)
        assert 0 <= a <= k

    # High acceptance -> speculation pays off.
    high = SpeculativeDecoder(k=k, draft_cost=draft_cost, accept_prob=0.90, seed=7).generate(n_tokens)
    speedup_high = baseline_cost / high["total_cost"]

    # Ablation: low acceptance -> speculation barely helps (or hurts).
    low = SpeculativeDecoder(k=k, draft_cost=draft_cost, accept_prob=0.10, seed=7).generate(n_tokens)
    speedup_low = baseline_cost / low["total_cost"]

    print(f"{'accept_prob':>12}{'acc_rate':>10}{'tgt_steps':>11}"
          f"{'total_cost':>12}{'speedup':>10}")
    for label, prob, res, sp in (("high", 0.90, high, speedup_high),
                                 ("low", 0.10, low, speedup_low)):
        print(f"{prob:>12.2f}{res['acceptance_rate']:>10.2f}"
              f"{res['target_steps']:>11}{res['total_cost']:>12.1f}{sp:>10.2f}")

    # Determinism: same seed -> identical result.
    again = SpeculativeDecoder(k=k, draft_cost=draft_cost, accept_prob=0.90, seed=7).generate(n_tokens)
    assert again["total_cost"] == high["total_cost"]

    # Acceptance rate tracks accept_prob.
    assert high["acceptance_rate"] > 0.7
    assert low["acceptance_rate"] < 0.3

    # High acceptance gives a real speedup.
    assert speedup_high > 1.3, f"expected speedup>1.3, got {speedup_high:.2f}"
    # Ablation: collapsing acceptance kills the speedup.
    assert speedup_low < speedup_high
    assert speedup_low < 1.3
    print("OK -- speculation pays at high acceptance; ablation kills the speedup.")


# ===========================================================================
# HARD EXERCISE 2 -- Inference-config optimizer (Pareto under SLO)
# ===========================================================================
#
# Sweep batch_size x quantization x prompt-cache, model their effect on cost &
# latency (deterministically), filter by latency SLO + quality floor, compute
# the cost/latency Pareto frontier among valid configs, and return the cheapest
# valid config.

# quant -> (cost_factor, latency_factor, quality)
QUANT = {
    "fp16": (1.00, 1.00, 1.00),
    "int8": (0.55, 0.75, 0.97),
    "int4": (0.35, 0.55, 0.90),
}
BATCH_GRID = [1, 4, 16, 64]


@dataclass(frozen=True)
class InferenceConfig:
    batch_size: int
    quant: str          # "fp16" | "int8" | "int4"
    cache: bool

    def label(self) -> str:
        return f"bs={self.batch_size:<3} {self.quant} cache={'on ' if self.cache else 'off'}"


class InferenceOptimizer:
    # Base numbers (arbitrary but consistent units).
    BASE_COST = 100.0           # base cost per request at bs=1, fp16, no cache
    BASE_LATENCY = 20.0         # base per-request latency at bs=1, fp16, no cache
    CACHE_COST_FACTOR = 0.40
    CACHE_LATENCY_FACTOR = 0.50

    def evaluate(self, cfg: InferenceConfig) -> dict:
        cost_f, lat_f, quality = QUANT[cfg.quant]

        # Batch size amortizes per-request cost (sublinear, floored) but
        # increases per-request latency (queueing): bigger batch = more waiting.
        per_req_cost = self.BASE_COST * cost_f / cfg.batch_size
        latency = self.BASE_LATENCY * lat_f * (1 + 0.10 * (cfg.batch_size - 1))

        # Prompt cache: shared prefix served from KV cache -> cheaper & faster.
        if cfg.cache:
            per_req_cost *= self.CACHE_COST_FACTOR
            latency *= self.CACHE_LATENCY_FACTOR

        return {
            "config": cfg,
            "cost": round(per_req_cost, 4),
            "latency": round(latency, 4),
            "quality": quality,
        }

    def sweep(self) -> list[dict]:
        out = []
        for bs, q, c in itertools.product(BATCH_GRID, QUANT.keys(), (False, True)):
            out.append(self.evaluate(InferenceConfig(bs, q, c)))
        return out

    @staticmethod
    def _dominates(a: dict, b: dict) -> bool:
        # a dominates b if a is <= on both cost and latency, strictly < on one.
        return (a["cost"] <= b["cost"] and a["latency"] <= b["latency"]
                and (a["cost"] < b["cost"] or a["latency"] < b["latency"]))

    def pareto_front(self, evaluated: list[dict]) -> list[dict]:
        front = []
        for cand in evaluated:
            if not any(self._dominates(other, cand) for other in evaluated
                       if other is not cand):
                front.append(cand)
        return front

    def optimize(self, slo_latency: float, min_quality: float) -> dict | None:
        evaluated = self.sweep()
        valid = [e for e in evaluated
                 if e["latency"] <= slo_latency and e["quality"] >= min_quality]
        if not valid:
            return None
        return min(valid, key=lambda e: e["cost"])


def hard_ex2_optimizer() -> None:
    print("\n" + "=" * 64)
    print("HARD EX2 -- inference-config optimizer (Pareto under SLO)")
    print("=" * 64)

    opt = InferenceOptimizer()
    evaluated = opt.sweep()
    assert len(evaluated) == len(BATCH_GRID) * len(QUANT) * 2 == 24

    front = opt.pareto_front(evaluated)
    print("Pareto frontier (cost vs latency, all configs):")
    for e in sorted(front, key=lambda x: x["cost"]):
        print(f"  {e['config'].label()}  cost={e['cost']:>8.2f}  "
              f"lat={e['latency']:>7.2f}  q={e['quality']:.2f}")
    # No config on the frontier dominates another on the frontier.
    for a in front:
        for b in front:
            if a is not b:
                assert not opt._dominates(a, b)

    # --- Optimize under a realistic SLO + quality floor ---
    slo1 = 12.0
    minq = 0.95
    best1 = opt.optimize(slo_latency=slo1, min_quality=minq)
    assert best1 is not None
    print(f"\nSLO latency<= {slo1}, quality>= {minq}")
    print(f"  -> chosen: {best1['config'].label()}  "
          f"cost={best1['cost']:.2f} lat={best1['latency']:.2f} q={best1['quality']:.2f}")
    # Respects SLO & quality floor.
    assert best1["latency"] <= slo1 and best1["quality"] >= minq
    # Cheapest among VALID configs: nothing valid is strictly cheaper.
    valid1 = [e for e in evaluated if e["latency"] <= slo1 and e["quality"] >= minq]
    assert all(best1["cost"] <= e["cost"] for e in valid1)
    # It lies on the Pareto frontier OF THE VALID configs (the quality floor can
    # exclude cheaper-but-lower-quality configs that dominate it in the full sweep).
    valid_front = opt.pareto_front(valid1)
    assert any(best1["config"] == f["config"] for f in valid_front)

    # --- Tighten the SLO: chosen config must get FASTER (lower-or-equal latency) ---
    slo2 = 8.0
    best2 = opt.optimize(slo_latency=slo2, min_quality=minq)
    assert best2 is not None
    print(f"\nTighter SLO latency<= {slo2}, quality>= {minq}")
    print(f"  -> chosen: {best2['config'].label()}  "
          f"cost={best2['cost']:.2f} lat={best2['latency']:.2f} q={best2['quality']:.2f}")
    assert best2["latency"] <= slo2 and best2["quality"] >= minq
    assert best2["latency"] <= best1["latency"], "tighter SLO must not increase latency"

    print("\nOK -- optimizer returns cheapest valid config on the valid-config Pareto frontier.")


# ===========================================================================
# RUN ALL
# ===========================================================================

if __name__ == "__main__":
    hard_ex1_speculative()
    hard_ex2_optimizer()
    print("\n" + "=" * 64)
    print("ALL HARD SOLUTIONS PASSED (Day 24 -- inference engineering)")
    print("=" * 64)
