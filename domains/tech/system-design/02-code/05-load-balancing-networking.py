"""
Day 5 -- Load Balancing & Networking
Interactive demonstrations in Python.

Usage:
    python 05-load-balancing-networking.py

Simulations illustrating:
- Different load balancing algorithms (RR, least-conn, consistent hash)
- Token bucket rate limiter
- Circuit breaker with its 3 states
- Retry with exponential backoff + jitter
"""

import time
import random
import bisect
import hashlib
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : Backend -- representation of a target server
# =============================================================================


@dataclass
class Backend:
    """A backend (web server) with its metrics.

    WHY track connections? Least-connections needs to know
    how many requests are IN FLIGHT on each backend to pick
    the least loaded one. In production, the LB counts open TCP connections.
    """

    name: str
    weight: int = 1
    active_connections: int = 0
    healthy: bool = True
    total_handled: int = 0

    def handle(self):
        """Simulates handling a request."""
        self.active_connections += 1
        self.total_handled += 1
        # In real life, the time depends on the request. We simulate with a random sleep.
        time.sleep(random.uniform(0.001, 0.005))
        self.active_connections -= 1


# =============================================================================
# SECTION 2 : Round Robin and Weighted Round Robin
# =============================================================================


class RoundRobinLB:
    """Round robin: request i goes to backend i % N.

    WHY so simple? No per-backend state to track (just a global
    counter). Perfect if all requests have the same duration, bad
    if some are 100x longer.
    """

    def __init__(self, backends: list[Backend]):
        self.backends = backends
        self.index = 0
        self._lock = threading.Lock()

    def pick(self) -> Backend:
        with self._lock:
            # Look for the next healthy backend (skip the dead ones)
            for _ in range(len(self.backends)):
                b = self.backends[self.index % len(self.backends)]
                self.index += 1
                if b.healthy:
                    return b
            raise RuntimeError("No healthy backend available")


class WeightedRoundRobinLB:
    """Weighted RR: each backend has a weight.

    WHY weighted? For heterogeneous servers (16-core vs 64-core)
    or for canary deploys (10% of the traffic on V2). Naive
    implementation: we duplicate the backend in the list based on its weight.
    """

    def __init__(self, backends: list[Backend]):
        self.backends = backends
        # Expanded list: [A, A, A, B, C] for A(3), B(1), C(1)
        self.expanded: list[Backend] = []
        for b in backends:
            self.expanded.extend([b] * b.weight)
        self.index = 0
        self._lock = threading.Lock()

    def pick(self) -> Backend:
        with self._lock:
            for _ in range(len(self.expanded)):
                b = self.expanded[self.index % len(self.expanded)]
                self.index += 1
                if b.healthy:
                    return b
            raise RuntimeError("No healthy backend available")


# =============================================================================
# SECTION 3 : Least Connections
# =============================================================================


class LeastConnectionsLB:
    """Least Connections: send to the backend with the fewest active conns.

    WHY does it work? If a backend is slow (big request, blocked thread),
    its connections pile up. The LB detects it and sends elsewhere. Auto-
    adapts to requests of variable duration.
    """

    def __init__(self, backends: list[Backend]):
        self.backends = backends

    def pick(self) -> Backend:
        # Filter the healthy ones then take the one with the min conns
        healthy = [b for b in self.backends if b.healthy]
        if not healthy:
            raise RuntimeError("No healthy backend available")
        return min(healthy, key=lambda b: b.active_connections)


# =============================================================================
# SECTION 4 : Consistent Hashing with virtual nodes
# =============================================================================


class ConsistentHashLB:
    """Consistent Hashing: the ring with vnodes.

    WHY not hash(key) % N? If we change N (adding/removing a
    backend), almost all keys are reassigned. A cache would lose
    99% of its entries. With the ring, only ~1/N of the keys move.

    WHY virtual nodes? With just N points on the ring, the
    distribution is poor (one backend can grab a large zone,
    another a small one). By creating K vnodes per backend (K=100-200),
    the distribution is nearly uniform.
    """

    def __init__(self, backends: list[Backend], vnodes_per_backend: int = 100):
        self.vnodes_per_backend = vnodes_per_backend
        self.ring: list[tuple[int, Backend]] = []  # sorted by hash
        self.hashes: list[int] = []
        for b in backends:
            self._add(b)

    @staticmethod
    def _hash(key: str) -> int:
        # sha1 for a uniform distribution (not python hash() which is randomized)
        return int(hashlib.sha1(key.encode()).hexdigest(), 16)

    def _add(self, backend: Backend):
        for v in range(self.vnodes_per_backend):
            h = self._hash(f"{backend.name}-vnode-{v}")
            # bisect to keep the ring sorted (O(log N) lookups)
            idx = bisect.bisect(self.hashes, h)
            self.hashes.insert(idx, h)
            self.ring.insert(idx, (h, backend))

    def remove(self, backend: Backend):
        """WHY remove? When a backend fails its healthcheck, we remove
        its vnodes. Only the keys that targeted those vnodes are
        redistributed (the clockwise neighbors)."""
        new_ring = [(h, b) for h, b in self.ring if b is not backend]
        self.ring = new_ring
        self.hashes = [h for h, _ in new_ring]

    def pick(self, key: str) -> Backend:
        if not self.ring:
            raise RuntimeError("No backend in ring")
        h = self._hash(key)
        # Find the first vnode after h (wrap around if at the end)
        idx = bisect.bisect(self.hashes, h) % len(self.hashes)
        return self.ring[idx][1]


# =============================================================================
# SECTION 5 : Token Bucket Rate Limiter
# =============================================================================


class TokenBucket:
    """Token bucket rate limiter.

    WHY token bucket? It allows bursts (full bucket = you can send
    'capacity' requests at once) while limiting the average to the refill
    rate. It is the default choice of most APIs (Stripe,
    AWS, GitHub).

    Example: capacity=100, refill=10/s
    -> Burst: 100 req at once OK, then 10 req/s on average.
    """

    def __init__(self, capacity: int, refill_per_sec: float):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def allow(self, cost: int = 1) -> bool:
        """Tries to consume 'cost' tokens. Returns True if OK."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            # Continuous refill: every millisecond adds refill_per_sec/1000 tokens
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
            self.last_refill = now
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False


# =============================================================================
# SECTION 6 : Circuit Breaker
# =============================================================================


class CircuitBreaker:
    """Circuit breaker with the 3 classic states.

    WHY does it save lives? Without a breaker, when a downstream service goes down,
    the caller keeps trying and saturates its threads on 30s
    timeouts. The threads die, the caller goes down too -> cascade.

    The breaker detects the failures and OPENS: all subsequent requests
    fail IMMEDIATELY (no network call) with an error or a
    fallback. After a timeout, it cautiously tries 1 request (HALF_OPEN).
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 5.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.state = self.CLOSED
        self.failures = 0
        self.opened_at: Optional[float] = None
        self._lock = threading.Lock()

    def call(self, func: Callable[[], Any], fallback: Callable[[], Any] = None) -> Any:
        """Calls func through the breaker. If OPEN, returns fallback() or raises."""
        with self._lock:
            # Automatic OPEN -> HALF_OPEN transition after timeout
            if self.state == self.OPEN:
                if time.time() - self.opened_at >= self.recovery_timeout_sec:
                    self.state = self.HALF_OPEN
                    print(f"    [breaker] OPEN -> HALF_OPEN (recovery attempt)")
                else:
                    if fallback:
                        return fallback()
                    raise RuntimeError("Circuit breaker is OPEN")

        try:
            result = func()
        except Exception as e:
            self._on_failure()
            if fallback:
                return fallback()
            raise
        else:
            self._on_success()
            return result

    def _on_success(self):
        with self._lock:
            if self.state == self.HALF_OPEN:
                print(f"    [breaker] HALF_OPEN -> CLOSED (recovery OK)")
            self.state = self.CLOSED
            self.failures = 0
            self.opened_at = None

    def _on_failure(self):
        with self._lock:
            self.failures += 1
            if self.state == self.HALF_OPEN:
                # The test request failed: we go back to OPEN
                self.state = self.OPEN
                self.opened_at = time.time()
                print(f"    [breaker] HALF_OPEN -> OPEN (test request failed)")
            elif self.failures >= self.failure_threshold:
                self.state = self.OPEN
                self.opened_at = time.time()
                print(f"    [breaker] CLOSED -> OPEN (threshold {self.failure_threshold} reached)")


# =============================================================================
# SECTION 7 : Retry with exponential backoff + jitter
# =============================================================================


def retry_with_backoff(
    func: Callable[[], Any],
    max_attempts: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
) -> Any:
    """Retry with exponential backoff and jitter.

    WHY exponential? Gives the service time to recover between
    attempts (1s, 2s, 4s, 8s...).
    WHY jitter? Without randomization, all clients retry at the same
    moment -> thundering herd that overloads the service right when it
    is trying to come back up. The jitter spreads them over time.
    """
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            # delay = min(max, base * 2^attempt)
            delay = min(max_delay, base_delay * (2 ** attempt))
            # Jitter: multiply by [0.5, 1.5] to spread out
            delay = delay * random.uniform(0.5, 1.5)
            print(f"    attempt {attempt+1}/{max_attempts} failed: {e}, waiting {delay:.3f}s")
            time.sleep(delay)


# =============================================================================
# SECTION 8 : Demos
# =============================================================================


def demo_algorithms_comparison():
    """Compares the distribution of 1000 requests over 3 backends with 3 algorithms."""
    print(f"\n{SEPARATOR}\n  DEMO 1 : Comparison of LB algorithms\n{SEPARATOR}")

    def make_backends():
        return [Backend("srv-A", weight=3), Backend("srv-B", weight=1), Backend("srv-C", weight=1)]

    # Round Robin
    backends = make_backends()
    lb = RoundRobinLB(backends)
    for _ in range(100):
        lb.pick().total_handled += 1
    print(f"  Round Robin : {[(b.name, b.total_handled) for b in backends]}")

    # Weighted Round Robin
    backends = make_backends()
    lb = WeightedRoundRobinLB(backends)
    for _ in range(100):
        lb.pick().total_handled += 1
    print(f"  Weighted RR : {[(b.name, b.total_handled) for b in backends]} (weights 3:1:1)")

    # Least Connections (simulated with variable durations)
    backends = make_backends()
    lb = LeastConnectionsLB(backends)
    for i in range(30):
        b = lb.pick()
        b.active_connections += 1
        # Simulate that srv-A is slower (conns don't free up quickly)
        if b.name == "srv-A" and i % 3 != 0:
            continue
        b.active_connections = max(0, b.active_connections - 1)
    print(f"  Least Conn  : {[(b.name, b.active_connections) for b in backends]}")


def demo_consistent_hashing_stability():
    """Shows that adding/removing a backend only reshuffles a fraction of the keys."""
    print(f"\n{SEPARATOR}\n  DEMO 2 : Consistent hashing stability\n{SEPARATOR}")

    backends = [Backend(f"cache-{i}") for i in range(4)]
    lb = ConsistentHashLB(backends, vnodes_per_backend=100)

    # Assign 1000 keys -> record current owner
    keys = [f"user-{i}" for i in range(1000)]
    before = {k: lb.pick(k).name for k in keys}

    # Remove one backend (simulates failure)
    lb.remove(backends[2])
    after = {k: lb.pick(k).name for k in keys}

    moved = sum(1 for k in keys if before[k] != after[k])
    print(f"  Keys before: assigned over 4 backends")
    print(f"  Remove cache-2 : {moved}/1000 keys moved ({moved/10:.1f}%)")
    print(f"  With hash % N : 100% of the keys would have moved. With consistent,")
    print(f"  only ~25% (keys that pointed to cache-2).")


def demo_token_bucket():
    """Tests a token bucket with a burst then sustained traffic."""
    print(f"\n{SEPARATOR}\n  DEMO 3 : Token bucket rate limiter\n{SEPARATOR}")
    bucket = TokenBucket(capacity=10, refill_per_sec=5)

    # Burst: 15 requests at once
    allowed = sum(1 for _ in range(15) if bucket.allow())
    print(f"  Burst 15 req : {allowed} allowed, {15-allowed} rejected")
    print(f"  (capacity=10, so we can burst 10 at once)")

    # Wait 1 second: refill of 5 tokens
    time.sleep(1.0)
    allowed = sum(1 for _ in range(10) if bucket.allow())
    print(f"  After 1s (refill ~5 tokens) : {allowed}/10 allowed")


def demo_circuit_breaker():
    """Simulates a service that goes down and comes back up."""
    print(f"\n{SEPARATOR}\n  DEMO 4 : Circuit breaker\n{SEPARATOR}")
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=0.3)

    service_alive = [True]

    def flaky_service():
        if not service_alive[0]:
            raise ConnectionError("service down")
        return "OK"

    def fallback():
        return "CACHED_FALLBACK"

    # Service alive: everything goes through
    print("  Phase 1 : service alive")
    for i in range(3):
        r = breaker.call(flaky_service, fallback)
        print(f"    req{i+1} : {r}, state={breaker.state}")

    # Service down: the breaker will open after 3 failures
    print("  Phase 2 : service down")
    service_alive[0] = False
    for i in range(5):
        r = breaker.call(flaky_service, fallback)
        print(f"    req{i+1} : {r}, state={breaker.state}")

    # Service comes back. We wait for the recovery timeout.
    print("  Phase 3 : service back, wait for HALF_OPEN")
    service_alive[0] = True
    time.sleep(0.35)
    r = breaker.call(flaky_service, fallback)
    print(f"    req : {r}, state={breaker.state}")


def demo_retry_with_jitter():
    """Simulates a service that responds after 3 attempts."""
    print(f"\n{SEPARATOR}\n  DEMO 5 : Retry exponential backoff + jitter\n{SEPARATOR}")
    attempts = [0]

    def flaky():
        attempts[0] += 1
        if attempts[0] < 3:
            raise ConnectionError(f"transient error")
        return "success"

    result = retry_with_backoff(flaky, max_attempts=5, base_delay=0.05)
    print(f"  Result : {result} after {attempts[0]} attempts")


def main():
    random.seed(42)
    demo_algorithms_comparison()
    demo_consistent_hashing_stability()
    demo_token_bucket()
    demo_circuit_breaker()
    demo_retry_with_jitter()
    print(f"\n{SEPARATOR}\n  End of demos.\n{SEPARATOR}")


if __name__ == "__main__":
    main()
