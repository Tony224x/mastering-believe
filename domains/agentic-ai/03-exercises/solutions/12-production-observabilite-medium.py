"""
Solutions -- Day 12 (MEDIUM): Production & Observability

Contains solutions for:
  - Medium Ex 1: Span tree + self-time + latency percentiles (p50/p95/p99)
  - Medium Ex 2: Prompt-caching cost model (3 levels) + break-even analysis
  - Medium Ex 3: Multi-tier degrading fallback chain with per-attempt tracing

stdlib only, fully offline. Span/Tracer shapes mirror
02-code/12-production-observabilite.py.

Run:  python 03-exercises/solutions/12-production-observabilite-medium.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ==========================================================================
# MEDIUM EXERCISE 1 -- Span tree + self-time + percentiles
# ==========================================================================


@dataclass
class Span:
    span_id: str
    trace_id: str
    name: str
    duration_ms: float
    parent_span_id: str | None = None


def build_span_tree(spans: list[Span], trace_id: str) -> list[dict]:
    """Reconstruct the parent/child forest for one trace."""
    nodes = {s.span_id: {"span": s, "children": []} for s in spans if s.trace_id == trace_id}
    roots: list[dict] = []
    for node in nodes.values():
        parent_id = node["span"].parent_span_id
        if parent_id and parent_id in nodes:
            nodes[parent_id]["children"].append(node)
        else:
            roots.append(node)
    return roots


def self_ms(node: dict) -> float:
    """Time spent IN this span, excluding children (where the time waited)."""
    children_total = sum(c["span"].duration_ms for c in node["children"])
    return round(node["span"].duration_ms - children_total, 2)


def print_tree(nodes: list[dict], indent: int = 0) -> None:
    for node in nodes:
        s = node["span"]
        pad = "  " * indent
        print(f"  {pad}- {s.name:16s} total={s.duration_ms:6.1f}ms self={self_ms(node):6.1f}ms")
        print_tree(node["children"], indent + 1)


def latency_percentiles(durations: list[float]) -> dict:
    """p50/p95/p99 via linear interpolation (no numpy)."""
    if not durations:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    xs = sorted(durations)

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        rank = p / 100 * (len(xs) - 1)
        lo = int(rank)
        frac = rank - lo
        hi = min(lo + 1, len(xs) - 1)
        return round(xs[lo] + (xs[hi] - xs[lo]) * frac, 2)

    return {"p50": pct(50), "p95": pct(95), "p99": pct(99)}


def medium_ex1_span_tree() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 1: Span tree + self-time + percentiles")
    print("=" * 60)

    # A small trace: agent.run -> (plan, llm_step -> http_call), retrieve
    spans = [
        Span("root", "t1", "agent.run", 250.0, None),
        Span("plan", "t1", "plan_step", 30.0, "root"),
        Span("llm", "t1", "llm_step", 180.0, "root"),
        Span("http", "t1", "http_call", 150.0, "llm"),
        Span("retr", "t1", "retrieve", 20.0, "root"),
    ]
    tree = build_span_tree(spans, "t1")
    print("\n  Waterfall:")
    print_tree(tree)

    # root self = 250 - (30+180+20) = 20 ; llm self = 180 - 150 = 30
    root_node = tree[0]
    assert self_ms(root_node) == 20.0
    llm_node = [c for c in root_node["children"] if c["span"].name == "llm_step"][0]
    assert self_ms(llm_node) == 30.0

    # Percentiles with outliers.
    durations = [100.0] * 28 + [900.0, 1500.0]   # 28 normal, 2 slow outliers
    pcts = latency_percentiles(durations)
    print(f"\n  Latency percentiles (28x100ms + 2 outliers): {pcts}")
    assert pcts["p50"] == 100.0
    assert pcts["p99"] > pcts["p50"] * 5, "outliers must show in the tail"

    # Sanity check on a known dataset.
    known = latency_percentiles([10, 20, 30, 40, 50])
    assert known["p50"] == 30.0, known

    print("\n  PASS -- tree rebuilt, self-time correct, tail latency visible.\n")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Prompt-caching cost model
# ==========================================================================


class CachedCostModel:
    """4-level pricing per 1K tokens: input / output / cache_write / cache_read."""

    # (input, output) base prices per 1K tokens.
    BASE = {"claude-sonnet-4-6": (0.003, 0.015), "mock": (0.001, 0.003)}
    CACHE_WRITE_MULT = 1.25   # writing the cache costs ~25% more than input
    CACHE_READ_MULT = 0.10    # reading the cache costs ~90% less than input

    def _prices(self, model: str) -> tuple[float, float, float, float]:
        pin, pout = self.BASE.get(model, (0.0, 0.0))
        return pin, pout, pin * self.CACHE_WRITE_MULT, pin * self.CACHE_READ_MULT

    def cost_uncached(self, model: str, tokens_in: int, tokens_out: int) -> float:
        pin, pout, _, _ = self._prices(model)
        return tokens_in / 1000 * pin + tokens_out / 1000 * pout

    def cost_cached(self, model: str, cached_tokens: int, fresh_in: int,
                    tokens_out: int, is_first_call: bool) -> float:
        pin, pout, pcw, pcr = self._prices(model)
        cache_price = pcw if is_first_call else pcr
        return (cached_tokens / 1000 * cache_price
                + fresh_in / 1000 * pin
                + tokens_out / 1000 * pout)

    def break_even_calls(self, model: str, cached_tokens: int,
                         fresh_in: int = 50, tokens_out: int = 100) -> int | None:
        """Smallest N s.t. cumulative cached cost < cumulative uncached cost."""
        cum_cached = cum_uncached = 0.0
        for n in range(1, 1000):
            cum_cached += self.cost_cached(model, cached_tokens, fresh_in, tokens_out, n == 1)
            cum_uncached += self.cost_uncached(model, cached_tokens + fresh_in, tokens_out)
            if cum_cached < cum_uncached:
                return n
        return None   # never pays off within a sane horizon


def medium_ex2_caching() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 2: Prompt-caching cost model")
    print("=" * 60)

    model = CachedCostModel()
    m = "claude-sonnet-4-6"
    cached = 3000        # big reusable system prompt
    fresh, out = 50, 100

    # First call is MORE expensive with cache (cache_write), then cheaper.
    first_cached = model.cost_cached(m, cached, fresh, out, True)
    later_cached = model.cost_cached(m, cached, fresh, out, False)
    uncached = model.cost_uncached(m, cached + fresh, out)
    print(f"\n  uncached/call      = ${uncached:.5f}")
    print(f"  cached 1st call    = ${first_cached:.5f} (cache write, pricier)")
    print(f"  cached later call  = ${later_cached:.5f} (cache read, ~90% off input)")
    assert first_cached > uncached
    assert later_cached < uncached

    # Cumulative over 10 calls.
    cum_cached = cum_uncached = 0.0
    for n in range(1, 11):
        cum_cached += model.cost_cached(m, cached, fresh, out, n == 1)
        cum_uncached += model.cost_uncached(m, cached + fresh, out)
    print(f"\n  10 calls: cached=${cum_cached:.5f} vs uncached=${cum_uncached:.5f} "
          f"(save {100 * (1 - cum_cached / cum_uncached):.0f}%)")
    assert cum_cached < cum_uncached

    be = model.break_even_calls(m, cached)
    print(f"  break-even at call #{be}")
    assert be is not None and be >= 2

    # Tiny prefix: cache-write overhead may never pay off.
    be_small = model.break_even_calls(m, 200)
    print(f"\n  Small 200-token prefix break-even: {be_small} "
          f"({'never worth it' if be_small is None else f'at call #{be_small}'})")

    print("\n  PASS -- caching model captures write/read asymmetry and break-even.\n")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Multi-tier degrading fallback chain
# ==========================================================================


class TransientError(Exception):
    pass


@dataclass
class Tier:
    name: str
    fn: Callable[[str], str]
    quality_level: float


@dataclass
class DegradingChain:
    tiers: list[Tier]
    spans: list[dict] = field(default_factory=list)   # per-attempt trace
    _cache: dict[str, str] = field(default_factory=dict)

    def _attempt(self, tier: Tier, prompt: str, max_retries: int = 2) -> str | None:
        for attempt in range(max_retries):
            try:
                out = tier.fn(prompt)
                self.spans.append({"tier": tier.name, "attempt": attempt, "ok": True})
                return out
            except TransientError:
                self.spans.append({"tier": tier.name, "attempt": attempt, "ok": False})
        return None

    def __call__(self, prompt: str) -> dict:
        for tier in self.tiers:
            out = self._attempt(tier, prompt)
            if out is not None:
                return {
                    "answer": out,
                    "served_by": tier.name,
                    "quality_level": tier.quality_level,
                    "attempts": len(self.spans),
                }
        return {"answer": None, "served_by": None, "quality_level": 0.0,
                "attempts": len(self.spans)}


def medium_ex3_degrading_chain() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 3: Multi-tier degrading fallback chain")
    print("=" * 60)

    cache: dict[str, str] = {"known prompt": "[cache] cached answer"}

    def make_flaky(name: str, fails: int) -> Callable[[str], str]:
        state = {"left": fails}

        def fn(prompt: str) -> str:
            if state["left"] > 0:
                state["left"] -= 1
                raise TransientError(f"{name} 503")
            return f"[{name}] {prompt[:20]}"
        return fn

    def cache_tier(prompt: str) -> str:
        if prompt in cache:
            return cache[prompt]
        raise TransientError("cache miss")

    def build(primary_fails: int, secondary_fails: int) -> DegradingChain:
        return DegradingChain(tiers=[
            Tier("premium-opus", make_flaky("premium-opus", primary_fails), 1.0),
            Tier("cheap-mini", make_flaky("cheap-mini", secondary_fails), 0.7),
            Tier("cache", cache_tier, 0.4),
            Tier("static", lambda p: "Service temporarily degraded.", 0.1),
        ])

    print("\n  Scenario 1: everything healthy")
    r1 = build(0, 0)("some prompt")
    print(f"    served_by={r1['served_by']} quality={r1['quality_level']}")
    assert r1["served_by"] == "premium-opus" and r1["quality_level"] == 1.0

    print("\n  Scenario 2: primary + secondary down, cache hit")
    r2 = build(2, 2)("known prompt")
    print(f"    served_by={r2['served_by']} quality={r2['quality_level']} answer={r2['answer']}")
    assert r2["served_by"] == "cache" and r2["quality_level"] == 0.4

    print("\n  Scenario 3: all dynamic tiers down, cache miss -> static")
    r3 = build(2, 2)("unseen prompt")
    print(f"    served_by={r3['served_by']} quality={r3['quality_level']} answer={r3['answer']}")
    assert r3["served_by"] == "static" and r3["quality_level"] == 0.1

    print("\n  PASS -- ordered degradation, per-attempt tracing, caller sees quality.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 12 MEDIUM Solutions -- Production & Observability")
    print("#" * 60)

    medium_ex1_span_tree()
    medium_ex2_caching()
    medium_ex3_degrading_chain()

    print("\n" + "#" * 60)
    print("  All medium solutions executed successfully.")
    print("#" * 60 + "\n")
