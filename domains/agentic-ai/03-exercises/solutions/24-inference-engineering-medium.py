"""
Solutions -- Day 24 (MEDIUM): Inference engineering (serving internals)

Contains solutions for:
  - Medium Ex 1: continuous batching vs static batching simulator
                 (throughput tokens/step + average latency, in *steps*)
  - Medium Ex 2: prefill / prefix-cache engine measuring TTFT and billed
                 input tokens vs a no-cache baseline
  - Medium Ex 3: SLO-aware router choosing the cheapest model that meets a
                 latency budget (TTFT + TPOT * output_tokens)

Self-contained & offline: no network, no API, no real model, no wall-clock.
Everything is deterministic (counts in tokens/steps; randomness is seeded).
The relevant pieces of 02-code/24-inference-engineering.py (PromptCache /
ModelSpec ideas) are re-modeled inline -- nothing is imported.

Run:  python 03-exercises/solutions/24-inference-engineering-medium.py
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from statistics import mean


# ===========================================================================
# MEDIUM EXERCISE 1 -- Continuous batching vs static batching
# ===========================================================================
#
# We model an inference engine step-by-step in *integer steps* (no wall-clock):
#   - prefill: one step where a request's KV cache is built from its prompt
#   - decode : one step produces exactly one output token per active request
#
# Static batching wastes slots: a wave runs until the LONGEST request in the
# wave finishes, so short requests idle. Continuous batching refills a freed
# slot immediately with the next pending request.


@dataclass
class Request:
    rid: int
    prompt_tokens: int      # used as a proxy; prefill is 1 step per request here
    output_tokens: int      # number of decode steps this request needs


class StaticBatchEngine:
    """Naive static batching: process fixed waves; pay the MAX output length."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def run(self, requests: list[Request]) -> dict:
        step = 0
        finish_step: dict[int, int] = {}
        total_output = 0
        # Split into fixed waves of `batch_size`.
        for i in range(0, len(requests), self.batch_size):
            wave = requests[i:i + self.batch_size]
            # 1 prefill step for the whole wave (they prefill together).
            step += 1
            # Decode for as many steps as the LONGEST request needs. Short
            # requests sit in their slot doing nothing -> wasted capacity.
            max_out = max(r.output_tokens for r in wave)
            for r in wave:
                total_output += r.output_tokens
                # This request's last token lands at: current step + its length.
                finish_step[r.rid] = step + r.output_tokens
            step += max_out
        return {
            "total_steps": step,
            "finish_step": finish_step,
            "total_output_tokens": total_output,
        }


class ContinuousBatchEngine:
    """Continuous batching: keep up to `batch_size` requests active; refill
    a freed slot immediately with the next pending request."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def run(self, requests: list[Request]) -> dict:
        pending = list(requests)            # FIFO queue of not-yet-started reqs
        # active slots: list of [request, remaining_output_tokens]
        active: list[list] = []
        finish_step: dict[int, int] = {}
        total_output = 0
        step = 0

        def fill_slots() -> None:
            # Bring active up to batch_size; each new request pays a prefill step
            # (modeled as part of entering -- it then decodes from the next step).
            while pending and len(active) < self.batch_size:
                req = pending.pop(0)
                active.append([req, req.output_tokens])

        fill_slots()
        # Each loop iteration = one decode step across all active requests.
        while active:
            step += 1
            still_active: list[list] = []
            for slot in active:
                req, remaining = slot
                remaining -= 1          # produce one token this step
                total_output += 1
                if remaining == 0:
                    finish_step[req.rid] = step
                else:
                    slot[1] = remaining
                    still_active.append(slot)
            active = still_active
            # A freed slot is refilled IMMEDIATELY (the heart of continuous batching).
            fill_slots()

        return {
            "total_steps": step,
            "finish_step": finish_step,
            "total_output_tokens": total_output,
        }


def _summary(result: dict) -> dict:
    steps = result["total_steps"]
    out = result["total_output_tokens"]
    return {
        "total_steps": steps,
        "throughput": out / steps if steps else 0.0,
        "avg_latency": mean(result["finish_step"].values()),
        "total_output_tokens": out,
    }


def medium_ex1_batching() -> None:
    print("=" * 64)
    print("MEDIUM EX1 -- continuous batching vs static batching")
    print("=" * 64)

    rng = random.Random(24)
    # Heterogeneous mix: many short requests + a few long ones.
    requests: list[Request] = []
    for rid in range(12):
        if rid % 4 == 0:
            out = rng.randint(40, 60)       # long
        else:
            out = rng.randint(2, 8)         # short
        requests.append(Request(rid=rid, prompt_tokens=rng.randint(50, 400), output_tokens=out))

    static = _summary(StaticBatchEngine(batch_size=4).run(requests))
    cont = _summary(ContinuousBatchEngine(batch_size=4).run(requests))

    print(f"{'strategy':<22}{'steps':>8}{'tok/step':>12}{'avg_lat':>12}")
    print(f"{'static batching':<22}{static['total_steps']:>8}"
          f"{static['throughput']:>12.3f}{static['avg_latency']:>12.2f}")
    print(f"{'continuous batching':<22}{cont['total_steps']:>8}"
          f"{cont['throughput']:>12.3f}{cont['avg_latency']:>12.2f}")

    # No tokens are lost: both produce the same total output.
    assert static["total_output_tokens"] == cont["total_output_tokens"]
    # Continuous batching is faster (higher throughput) and lower latency.
    assert cont["throughput"] > static["throughput"], "continuous should win throughput"
    assert cont["avg_latency"] < static["avg_latency"], "continuous should win latency"
    print("OK -- continuous batching: higher throughput, lower average latency.")


# ===========================================================================
# MEDIUM EXERCISE 2 -- Prefill / prefix cache & TTFT savings
# ===========================================================================
#
# Extends 02-code PromptCache by separating prefill (KV build) from decode and
# exposing time-to-first-token (TTFT). A prefix-cache HIT skips recomputing the
# shared prefix -> TTFT collapses and billed input tokens drop.


class PrefillCacheEngine:
    """Prefix cache modeled on Anthropic-style cache_control (read_discount)."""

    def __init__(self, read_discount: float = 0.10) -> None:
        self.read_discount = read_discount          # -90% => pay 10% on hit
        self._seen: set[str] = set()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(prefix: str) -> str:
        return hashlib.sha256(prefix.encode("utf-8")).hexdigest()

    def serve(self, prefix: str, prefix_tokens: int, suffix_tokens: int,
              output_tokens: int) -> dict:
        key = self._key(prefix)
        hit = key in self._seen
        if hit:
            self.hits += 1
            # KV of the shared prefix is reused as-is: prefill only re-encodes
            # the (short) suffix.
            prefill_steps = suffix_tokens
            billed_input = int(prefix_tokens * self.read_discount) + suffix_tokens
        else:
            self.misses += 1
            self._seen.add(key)
            # Cold call: full prefill of prefix + suffix, full billing.
            prefill_steps = prefix_tokens + suffix_tokens
            billed_input = prefix_tokens + suffix_tokens
        # TTFT: first decoded token lands one step after prefill completes.
        ttft = prefill_steps + 1
        return {
            "hit": hit,
            "prefill_steps": prefill_steps,
            "ttft": ttft,
            "billed_input_tokens": billed_input,
            "output_tokens": output_tokens,
        }


def medium_ex2_prefix_cache() -> None:
    print("\n" + "=" * 64)
    print("MEDIUM EX2 -- prefill / prefix cache & TTFT savings")
    print("=" * 64)

    engine = PrefillCacheEngine(read_discount=0.10)
    prefix = "SYSTEM PROMPT + TOOL SCHEMAS " * 320      # the static heavy context
    prefix_tokens = 8000
    suffix_tokens = 150
    output_tokens = 60
    n_calls = 50                                        # 1 cold + 49 warm

    cached_ttfts: list[int] = []
    cached_billed = 0
    cold_ttft = None
    warm_ttft = None
    for i in range(n_calls):
        r = engine.serve(prefix, prefix_tokens, suffix_tokens, output_tokens)
        cached_ttfts.append(r["ttft"])
        cached_billed += r["billed_input_tokens"]
        if i == 0:
            cold_ttft = r["ttft"]
        elif warm_ttft is None:
            warm_ttft = r["ttft"]

    # Baseline: NO cache -> every call recomputes the full prefix every time.
    baseline_ttft = (prefix_tokens + suffix_tokens) + 1
    baseline_billed = n_calls * (prefix_tokens + suffix_tokens)

    print(f"cold-miss TTFT (steps) : {cold_ttft}")
    print(f"warm-hit TTFT (steps)  : {warm_ttft}")
    print(f"hits={engine.hits}  misses={engine.misses}")
    print(f"billed input tokens (cache)   : {cached_billed:,}")
    print(f"billed input tokens (no cache): {baseline_billed:,}")
    factor = baseline_billed / cached_billed
    print(f"input-token savings factor    : {factor:.1f}x")

    # (a) a hit's TTFT is far below the cold-miss TTFT.
    assert warm_ttft < cold_ttft
    assert warm_ttft == suffix_tokens + 1               # only suffix re-encoded
    assert cold_ttft == baseline_ttft                   # cold miss == no-cache cost
    # (b) total billed input tokens with cache are >= 5x lower than baseline.
    assert factor >= 5.0, f"expected >=5x savings, got {factor:.1f}x"
    # (c) exactly 49 hits, 1 miss.
    assert engine.hits == 49 and engine.misses == 1
    print("OK -- prefix cache collapses TTFT and slashes billed input tokens.")


# ===========================================================================
# MEDIUM EXERCISE 3 -- SLO-aware router (latency budget: TTFT + TPOT)
# ===========================================================================


@dataclass
class ServingModel:
    name: str
    tier: str               # "weak" | "strong"
    cost_in: float          # $ / 1M input tokens
    cost_out: float         # $ / 1M output tokens
    ttft_steps: int         # fixed first-token latency (steps)
    tpot_steps: float       # time per output token (steps/token)
    quality: float          # 0..1


class SLORouter:
    def __init__(self, models: list[ServingModel]) -> None:
        self.models = models

    def estimate_latency(self, model: ServingModel, output_tokens: int) -> float:
        # Classic serving model: TTFT (prefill) + per-token decode * #tokens.
        return model.ttft_steps + model.tpot_steps * output_tokens

    def route(self, output_tokens: int, slo_steps: float,
              min_quality: float = 0.0) -> dict:
        eligible = [
            m for m in self.models
            if m.quality >= min_quality
            and self.estimate_latency(m, output_tokens) <= slo_steps
        ]
        if eligible:
            # Among models that meet the SLO, pick the CHEAPEST.
            best = min(eligible, key=lambda m: m.cost_in + m.cost_out)
            return {
                "model": best.name,
                "latency": self.estimate_latency(best, output_tokens),
                "slo_met": True,
                "cost": best.cost_in + best.cost_out,
                "reason": "cheapest model meeting SLO",
            }
        # No model meets the SLO: best-effort -> fastest available.
        candidates = [m for m in self.models if m.quality >= min_quality] or self.models
        fastest = min(candidates, key=lambda m: self.estimate_latency(m, output_tokens))
        return {
            "model": fastest.name,
            "latency": self.estimate_latency(fastest, output_tokens),
            "slo_met": False,
            "cost": fastest.cost_in + fastest.cost_out,
            "reason": "no model meets SLO; best-effort fastest",
        }


def medium_ex3_slo_router() -> None:
    print("\n" + "=" * 64)
    print("MEDIUM EX3 -- SLO-aware router (TTFT + TPOT)")
    print("=" * 64)

    models = [
        # cheap & slow weak model
        ServingModel("haiku-weak", "weak", cost_in=0.40, cost_out=1.60,
                     ttft_steps=5, tpot_steps=1.0, quality=0.80),
        # mid
        ServingModel("sonnet-mid", "strong", cost_in=3.00, cost_out=15.00,
                     ttft_steps=4, tpot_steps=0.6, quality=0.92),
        # expensive & fast strong model
        ServingModel("opus-fast", "strong", cost_in=15.00, cost_out=75.00,
                     ttft_steps=2, tpot_steps=0.25, quality=0.97),
    ]
    router = SLORouter(models)

    # Scenario A: generous SLO + small output -> cheapest (weak) meets it.
    a = router.route(output_tokens=20, slo_steps=100.0, min_quality=0.0)
    print(f"A generous SLO  -> {a['model']:<12} lat={a['latency']:.1f} "
          f"slo_met={a['slo_met']} cost={a['cost']:.1f}")
    assert a["model"] == "haiku-weak" and a["slo_met"] is True

    # Scenario B: tight SLO + big output -> only the fast (expensive) one fits.
    b = router.route(output_tokens=80, slo_steps=25.0, min_quality=0.0)
    print(f"B tight SLO     -> {b['model']:<12} lat={b['latency']:.1f} "
          f"slo_met={b['slo_met']} cost={b['cost']:.1f}")
    assert b["model"] == "opus-fast" and b["slo_met"] is True
    assert b["latency"] <= 25.0

    # Scenario C: impossible SLO -> best-effort, slo_met False, fastest model.
    c = router.route(output_tokens=200, slo_steps=10.0, min_quality=0.0)
    print(f"C impossible SLO-> {c['model']:<12} lat={c['latency']:.1f} "
          f"slo_met={c['slo_met']} cost={c['cost']:.1f}")
    assert c["slo_met"] is False and c["model"] == "opus-fast"  # fastest overall

    # Scenario D: min_quality filter excludes the weak model.
    d = router.route(output_tokens=20, slo_steps=100.0, min_quality=0.90)
    print(f"D quality floor -> {d['model']:<12} lat={d['latency']:.1f} "
          f"slo_met={d['slo_met']} cost={d['cost']:.1f}")
    assert d["model"] != "haiku-weak" and d["slo_met"] is True

    print("OK -- router meets latency SLO at minimum cost, honors quality floor.")


# ===========================================================================
# RUN ALL
# ===========================================================================

if __name__ == "__main__":
    medium_ex1_batching()
    medium_ex2_prefix_cache()
    medium_ex3_slo_router()
    print("\n" + "=" * 64)
    print("ALL MEDIUM SOLUTIONS PASSED (Day 24 -- inference engineering)")
    print("=" * 64)
