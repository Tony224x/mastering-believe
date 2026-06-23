"""
Day 24 -- Solutions to the exercises for inference engineering.

Run the whole file to execute every solution.

    python domains/tech/agentic-ai/03-exercises/solutions/24-inference-engineering.py
"""

from __future__ import annotations

import math
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# pylint: disable=wrong-import-position
day24 = import_module("24-inference-engineering")
ModelSpec = day24.ModelSpec
ModelRouter = day24.ModelRouter
all_strong_cost = day24.all_strong_cost
COMPLEX_KEYWORDS = day24.COMPLEX_KEYWORDS


# ===========================================================================
# SOLUTION 1 -- Weighted (sampled) enum decoder
# ===========================================================================

class WeightedEnumDecoder:
    """Samples among allowed tokens proportionally to softmax(logits).

    Out-of-grammar tokens are masked BEFORE the softmax, so they have exactly
    zero probability no matter how large their raw logit is.
    """

    def __init__(self, allowed: list[str], seed: int = 0) -> None:
        self.allowed = allowed
        self._rng = random.Random(seed)

    def _softmax(self, logits: dict[str, float]) -> dict[str, float]:
        m = max(logits.values())
        exps = {t: math.exp(v - m) for t, v in logits.items()}
        z = sum(exps.values())
        return {t: v / z for t, v in exps.items()}

    def decode(self, raw_logits: dict[str, float]) -> str:
        masked = {t: v for t, v in raw_logits.items() if t in self.allowed}
        if not masked:
            return self.allowed[0]
        probs = self._softmax(masked)
        tokens = list(probs)
        weights = [probs[t] for t in tokens]
        return self._rng.choices(tokens, weights=weights, k=1)[0]


def solution_1() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 1 -- Weighted enum decoder")
    print("#" * 60)

    allowed = ["active", "idle", "error"]
    dec = WeightedEnumDecoder(allowed, seed=42)
    # 'running' has the biggest logit but is out of grammar.
    logits = {"running": 9.0, "active": 3.0, "idle": 2.0, "error": 1.0}

    counts = {t: 0 for t in allowed}
    for _ in range(1000):
        tok = dec.decode(logits)
        assert tok in allowed, "decoder emitted an out-of-grammar token!"
        counts[tok] += 1

    total = sum(counts.values())
    print("empirical distribution over 1000 draws:")
    for t in allowed:
        print(f"  {t:<8} {counts[t]/total:.3f}")
    # 'active' has the highest allowed logit -> should dominate.
    assert counts["active"] > counts["idle"] > counts["error"]
    print("  [check] 100% in-grammar, ordering follows logits -> OK")


# ===========================================================================
# SOLUTION 2 -- Cascade routing with a verifier
# ===========================================================================

class CascadeRouter:
    """Try the weak model first; escalate to strong only if a verifier rejects."""

    def __init__(self, weak: ModelSpec, strong: ModelSpec) -> None:
        self.weak = weak
        self.strong = strong
        self.total_cost = 0.0
        self.calls = {"weak": 0, "strong": 0}

    def _cost(self, model: ModelSpec, tin: int, tout: int) -> float:
        return (tin / 1e6) * model.cost_in + (tout / 1e6) * model.cost_out

    def call(self, query: str, verifier: Callable[[str], bool],
             est_in: int, est_out: int) -> None:
        # Always pay for the weak attempt first.
        self.total_cost += self._cost(self.weak, est_in, est_out)
        self.calls["weak"] += 1
        if not verifier(query):
            # Escalate: pay for the strong model too.
            self.total_cost += self._cost(self.strong, est_in, est_out)
            self.calls["strong"] += 1


def solution_2() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 2 -- Cascade routing")
    print("#" * 60)

    weak = ModelSpec("weak-mini", "weak", 0.40, 1.60)
    strong = ModelSpec("strong", "strong", 2.00, 8.00)

    rng = random.Random(7)
    simple = ["Format date", "Classify sentiment", "Extract id", "Translate hi"]
    complex_ = ["Analyse and design architecture for X",
                "Explain step by step and prove Y",
                "Compare approaches with reasoning"]
    batch: list[tuple[str, int, int]] = []
    for _ in range(40):
        batch.append((rng.choice(simple), 200, 80))
    for _ in range(10):
        batch.append((rng.choice(complex_), 1200, 600))
    rng.shuffle(batch)

    def verifier(query: str) -> bool:
        # weak answer is "good enough" unless the query looks complex
        return not any(kw in query.lower() for kw in COMPLEX_KEYWORDS)

    # Strategy A: all-strong
    base = all_strong_cost(batch, strong)

    # Strategy B: simple binary router (J24)
    simple_router = ModelRouter(weak, strong, threshold=1.0)
    for q, tin, tout in batch:
        simple_router.call(q, tin, tout)

    # Strategy C: cascade
    cascade = CascadeRouter(weak, strong)
    for q, tin, tout in batch:
        cascade.call(q, verifier, tin, tout)

    print(f"{'strategy':<18}{'cost ($)':>12}{'weak':>8}{'strong':>8}")
    print("-" * 46)
    print(f"{'all-strong':<18}{base:>12.4f}{0:>8}{len(batch):>8}")
    print(f"{'simple router':<18}{simple_router.total_cost:>12.4f}"
          f"{simple_router.routed['weak']:>8}{simple_router.routed['strong']:>8}")
    print(f"{'cascade':<18}{cascade.total_cost:>12.4f}"
          f"{cascade.calls['weak']:>8}{cascade.calls['strong']:>8}")

    assert cascade.total_cost < base, "cascade should beat all-strong here"
    # Cascade pays weak twice on escalated cases, so >= simple router.
    assert cascade.total_cost >= simple_router.total_cost
    print("  [check] cascade < all-strong, cascade >= simple router -> OK")


# ===========================================================================
# SOLUTION 3 -- TTL + LRU prefix cache
# ===========================================================================

@dataclass
class _Stats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class TTLLRUCache:
    """Prefix cache with time-to-live expiration and LRU eviction.

    `now` is injectable so tests advance a logical clock instead of sleeping.
    """

    def __init__(self, capacity: int, ttl: float,
                 read_discount: float = 0.10,
                 now: Callable[[], float] | None = None) -> None:
        self.capacity = capacity
        self.ttl = ttl
        self.read_discount = read_discount
        self._now = now or (lambda: 0.0)
        self._store: "OrderedDict[str, float]" = OrderedDict()  # key -> inserted_at
        self.stats = _Stats()

    def call(self, prefix: str, prefix_tokens: int, suffix_tokens: int) -> bool:
        key = prefix  # in this toy version the prefix string is the key
        t = self._now()
        if key in self._store and (t - self._store[key]) <= self.ttl:
            # Fresh hit: refresh recency.
            self._store.move_to_end(key)
            self.stats.hits += 1
            return True
        # Miss (absent or expired): (re)insert, evicting LRU if full.
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = t
        self.stats.misses += 1
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)   # drop least-recently-used
            self.stats.evictions += 1
        return False


def solution_3() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 3 -- TTL + LRU cache")
    print("#" * 60)

    clock = {"t": 0.0}
    cache = TTLLRUCache(capacity=2, ttl=10.0, now=lambda: clock["t"])

    # Scenario 1: warm hit
    cache.call("A", 1000, 50)            # miss (cold)
    clock["t"] = 1.0
    hit = cache.call("A", 1000, 50)      # hit (fresh)
    assert hit, "expected a warm hit"
    print("scenario 1 (warm hit): hits=", cache.stats.hits)

    # Scenario 2: expiration
    clock["t"] = 100.0                   # well past ttl
    expired = cache.call("A", 1000, 50)  # miss (expired)
    assert not expired, "expired entry must be a miss"
    print("scenario 2 (expiration): misses=", cache.stats.misses)

    # Scenario 3: LRU eviction (capacity=2)
    cache2 = TTLLRUCache(capacity=2, ttl=1e9, now=lambda: clock["t"])
    cache2.call("X", 100, 10)            # miss
    cache2.call("Y", 100, 10)            # miss  (store: X,Y)
    cache2.call("X", 100, 10)            # hit   -> X most-recent (store: Y,X)
    cache2.call("Z", 100, 10)            # miss  -> evicts Y (LRU) (store: X,Z)
    evicted_y_is_miss = not cache2.call("Y", 100, 10)  # miss (Y was evicted)
    assert evicted_y_is_miss, "Y should have been evicted"
    assert cache2.stats.evictions >= 1
    print(f"scenario 3 (LRU): hits={cache2.stats.hits}, "
          f"misses={cache2.stats.misses}, evictions={cache2.stats.evictions}")
    print("  [check] TTL expiry + LRU eviction behave correctly -> OK")


if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()
    print("\nAll Day 24 solutions ran successfully.")
