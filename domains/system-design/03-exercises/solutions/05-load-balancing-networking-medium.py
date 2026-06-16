"""
Solutions -- Day 5 MEDIUM Exercises: Load Balancing & Networking

Worked solutions with computed numbers (and assertions). Exercise 2 actually
implements a tiny consistent-hashing ring to MEASURE the remap fraction, which
is the whole point of the exercise.

Usage:
    python3 05-load-balancing-networking-medium.py
"""

import hashlib

SEPARATOR = "=" * 60


def _h(key: str) -> int:
    """Stable 32-bit hash for the ring (md5 -> int)."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)


# =============================================================================
# MEDIUM -- Exercise 1 : Distributed rate limiter
# =============================================================================

def medium_1_rate_limiter():
    """Design a distributed rate limiter (1000 req/min, 8 instances)."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Distributed rate limiter")
    print(SEPARATOR)

    limit_per_min = 1000
    instances = 8

    print("\n  1. Algorithm choice :")
    print("     Token bucket OR sliding window counter.")
    print("     '1000/min + small bursts' -> token bucket (capacity 1000, refill")
    print("     1000/60 ~= 16.7 tokens/s) allows a burst up to the bucket size then")
    print("     smooths to the average. Sliding window counter is the alternative")
    print("     (accurate, cheap, what Cloudflare uses). Avoid fixed window (border")
    print("     effect) and sliding log (O(N) memory).")

    print("\n  2. The local-counter trap :")
    naive_max = limit_per_min * instances
    print(f"     If each of the {instances} instances counts locally, a client could")
    print(f"     do {limit_per_min} per instance = {naive_max} req/min total -> {instances}x the limit.")
    print("     Fix : keep the counter in a SHARED store (Redis). All instances")
    print("     increment the same key -> one global count per client.")
    assert naive_max == 8000

    print("\n  3. Redis structure (token bucket) :")
    print("     Key   : rl:{api_key}            (hash with fields tokens, last_refill)")
    print("     Atomic op : a Lua script that (a) refills based on elapsed time,")
    print("     (b) checks tokens >= 1, (c) decrements. Atomicity avoids the race")
    print("     where two instances both read 'tokens=1' and both pass.")
    print("     Simpler sliding-window-counter variant : INCR rl:{key}:{window} then")
    print("     EXPIRE; weight current+previous window. INCR is atomic in Redis.")

    print("\n  4. Fixed window border effect (numbers) :")
    print(f"     Limit {limit_per_min}/min. A client sends {limit_per_min} at t=00:59 (window 1)")
    print(f"     and {limit_per_min} at t=01:01 (window 2). That's {2*limit_per_min} requests in 2s,")
    print(f"     while the intended cap was {limit_per_min}/min -> 2x burst slips through.")
    print("     Sliding window counter blends the two windows by elapsed fraction,")
    print("     so the effective count near the border stays ~= the limit.")
    assert 2 * limit_per_min == 2000

    print("\n  5. Response when over limit :")
    print("     HTTP 429 Too Many Requests + headers :")
    print("       Retry-After: <seconds>")
    print("       X-RateLimit-Limit / X-RateLimit-Remaining / X-RateLimit-Reset")
    print("     -> well-behaved clients back off instead of hammering.")

    print("\n  6. Redis round-trip at 50K req/s :")
    print("     +1ms per request is significant at scale and adds a Redis hotspot.")
    print("     Hybrid : each instance pre-allocates a batch of tokens from Redis")
    print("     (e.g. 50 at a time) and counts locally; sync periodically. Slightly")
    print("     less precise at the boundary but removes a round-trip per request.")


# =============================================================================
# MEDIUM -- Exercise 2 : Consistent hashing (measured)
# =============================================================================

class Ring:
    """Minimal consistent-hashing ring with virtual nodes."""

    def __init__(self, nodes, vnodes=1):
        self.vnodes = vnodes
        self.ring = {}                       # point on ring -> node
        for n in nodes:
            self._add(n)
        self._sorted = sorted(self.ring)

    def _add(self, node):
        for v in range(self.vnodes):
            self.ring[_h(f"{node}#{v}")] = node

    def add_node(self, node):
        self._add(node)
        self._sorted = sorted(self.ring)

    def get(self, key):
        p = _h(key)
        # first ring point >= p, wrapping around
        for point in self._sorted:
            if point >= p:
                return self.ring[point]
        return self.ring[self._sorted[0]]


def medium_2_consistent_hashing():
    """Measure remap fractions: modulo vs consistent hashing."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Consistent hashing (measured)")
    print(SEPARATOR)

    n = 10
    keys = [f"user_{i}" for i in range(20_000)]

    # 1. hash % N : 10 -> 11
    moved_mod = sum(1 for k in keys
                    if (_h(k) % n) != (_h(k) % (n + 1)))
    frac_mod = moved_mod / len(keys)
    print("\n  1. hash %% N when going 10 -> 11 nodes :")
    print(f"     Remapped keys : {frac_mod*100:.1f}%  (theory ~ N/(N+1) ~ 91%)")
    print("     -> almost every key moves -> hit rate collapses right after the change")
    print("        (every moved key is a cache miss until repopulated).")
    assert frac_mod > 0.80, frac_mod

    # 2. consistent hashing : 10 -> 11 (with vnodes for realistic balance)
    nodes = [f"node{i}" for i in range(n)]
    ring = Ring(nodes, vnodes=150)
    before = {k: ring.get(k) for k in keys}
    ring.add_node(f"node{n}")
    moved_ch = sum(1 for k in keys if ring.get(k) != before[k])
    frac_ch = moved_ch / len(keys)
    print("\n  2. Consistent hashing 10 -> 11 nodes :")
    print(f"     Remapped keys : {frac_ch*100:.1f}%  (theory ~ 1/(N+1) ~ 9%)")
    print(f"     -> {frac_mod/frac_ch:.0f}x fewer keys move than hash %% N.")
    assert frac_ch < 0.20, frac_ch
    assert frac_ch < frac_mod / 3            # consistent hashing must be far better

    # 3. vnodes balance: compare load spread with 1 vnode vs 150
    def spread(vn):
        r = Ring(nodes, vnodes=vn)
        counts = {x: 0 for x in nodes}
        for k in keys:
            counts[r.get(k)] += 1
        vals = list(counts.values())
        return max(vals) / (sum(vals) / len(vals))   # max / mean

    imbalance_1 = spread(1)
    imbalance_150 = spread(150)
    print("\n  3. Virtual nodes balance the load :")
    print(f"     1 vnode/node   : worst node = {imbalance_1:.2f}x the average")
    print(f"     150 vnodes/node: worst node = {imbalance_150:.2f}x the average")
    print("     With few real nodes the ring has uneven gaps -> hot nodes. 100-200")
    print("     vnodes/node smooth it (more points -> more uniform arcs).")
    assert imbalance_150 < imbalance_1       # vnodes must improve balance

    print("\n  4. A node fails :")
    print("     Its keys (~1/N) move to the NEXT node clockwise. Global hit rate dips")
    print("     by ~1/N (those keys miss until repopulated), NOT a full collapse. With")
    print("     vnodes the failed node's keys spread across many nodes -> no single")
    print("     neighbor gets a thundering hot spot.")

    print("\n  5. Link to DynamoDB / Cassandra :")
    print("     Same primitive : both partition data on a consistent-hashing ring with")
    print("     virtual nodes (vnodes / tokens). Adding/removing a node moves only ~1/N")
    print("     of the partitions -> elastic scaling without reshuffling everything.")


# =============================================================================
# MEDIUM -- Exercise 3 : LB algorithm choice + healthchecks
# =============================================================================

def medium_3_lb_algos():
    """Pick the LB algorithm per service + healthcheck/drain design."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : LB algorithm choice + healthchecks")
    print(SEPARATOR)

    cases = [
        ("Video transcoding (10s-10min, homogeneous)", "Least connections",
         "Durations vary wildly. Round robin would pile long jobs on a server "
         "that already has one. Least connections sends work to the freest server."),
        ("Stateless price compute (~20ms, homogeneous, high traffic)", "Round robin",
         "Short uniform requests -> round robin is simplest and fair. Least "
         "connections adds stateful tracking for no benefit here."),
        ("Canary migration (5% to v2)", "Weighted round robin",
         "Set weights 95 (v1) / 5 (v2). Gradually shift the weight as v2 proves out."),
    ]
    print("\n  1-3. Algorithm per service :")
    for svc, algo, why in cases:
        print(f"     {svc}")
        print(f"       -> {algo} : {why}")

    print("\n  4. Correct healthcheck :")
    print("     Application-level : GET /healthz that checks real dependencies (DB,")
    print("     cache, downstream). A bare 'TCP connect OK' passes even when the app")
    print("     is dead-locked or its DB is down -> the LB keeps routing to a broken")
    print("     server. Frequency ~5-10s, mark unhealthy after 3 consecutive fails.")

    print("\n  5. Drain mode (connection draining) :")
    print("     The LB stops sending NEW connections to the backend but lets in-flight")
    print("     requests finish (grace period, e.g. 30-60s) before removing it. No")
    print("     dropped requests during maintenance/deploy.")

    print("\n  6. Flapping :")
    print("     Too aggressive (1s interval, 1 fail = down) makes a briefly slow server")
    print("     flip in/out of rotation repeatedly -> traffic thrash + rebalancing.")
    print("     Fix : require N consecutive fails to go down and N successes to come")
    print("     back (hysteresis), reasonable interval, and slow-start on re-add.")


def main():
    print("\n" + "=" * 60)
    print("  SOLUTIONS -- DAY 5 MEDIUM : LOAD BALANCING & NETWORKING")
    print("=" * 60)
    medium_1_rate_limiter()
    medium_2_consistent_hashing()
    medium_3_lb_algos()
    print(f"\n{'=' * 60}")
    print("  END OF MEDIUM SOLUTIONS (all assertions passed)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
