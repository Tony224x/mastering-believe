"""
Solutions -- Day 5 Exercises: Load Balancing & Networking

This file contains the detailed solutions of the Easy exercises.

Usage:
    python 05-load-balancing-networking.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercise 1 : Choosing the LB algorithm
# =============================================================================


def easy_1_choose_lb_algorithm():
    """Solution: scenario -> algorithm mapping."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Choosing the LB algorithm")
    print(SEPARATOR)

    choices = [
        (
            "10 homogeneous stateless web servers, req ~50 ms",
            "Round Robin",
            "Homogeneous requests + identical servers = simple and fair RR. "
            "No need to track the backends' state. It's the default."
        ),
        (
            "Canary deploy 5% V2 / 95% V1",
            "Weighted Round Robin",
            "Weights 19:1 (V1:V2) = 95%/5% of the traffic. WRR is THE "
            "standard method to finely steer a progressive deploy."
        ),
        (
            "Memcached cluster of 20 nodes, avoid reshuffle on failure",
            "Consistent Hashing",
            "This is exactly the central use case of consistent hashing. "
            "When a node goes down, only ~5% of the keys (1/20) are "
            "redistributed. With hash % N, we would lose 95% of the cache."
        ),
        (
            "Analytics : reports from 100 ms to 30s",
            "Least Connections",
            "Durations are highly variable. RR would dispatch blindly, "
            "and one server could end up with 5 reports of 30s "
            "while another handles 100 fast ones. Least connections "
            "detects the real load via the count and rebalances."
        ),
        (
            "WebSocket chat, sticky session required",
            "IP Hash (or cookie-based sticky at L7)",
            "The server holds the connection state in memory. Each user "
            "MUST go back to the same server. IP Hash is simple but "
            "fragile (a user changing networks reshuffles). Prefer "
            "a cookie-based sticky session (ALB, nginx ip_hash or cookie)."
        ),
        (
            "Heterogeneous cluster : 4x16CPU + 2x64CPU",
            "Weighted Round Robin",
            "The 64-core machines have 4x the capacity of the 16-core. Weights 4:4:4:4:16:16 "
            "= the load is proportional to the capacity. Without weights, the "
            "16-core saturate while the 64-core sit idle."
        ),
    ]

    for i, (scenario, algo, reason) in enumerate(choices, 1):
        print(f"\n  {i}. {scenario}")
        print(f"     Algo : {algo}")
        print(f"     Reason : {reason}")


# =============================================================================
# EASY -- Exercise 2 : Rate limiter for a public API
# =============================================================================


def easy_2_rate_limiter_design():
    """Solution: design of a multi-tier rate limiter."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : Rate limiter for a public API")
    print(SEPARATOR)

    print("""
  1. Per-minute algorithm : TOKEN BUCKET (or sliding window counter)
     - Token bucket allows legitimate bursts (a client sending
       80 req at once then nothing for 30s stays within their quota).
     - Sliding window counter is the other good choice : ~99%
       precision and constant memory. Cloudflare uses it.
     - Avoid the fixed window : the "double at the window edge" effect
       allows exceeding the limit when straddling two windows.

  2. Per-day algorithm : FIXED WINDOW or SLIDING WINDOW
     - Over 24h, edge precision matters less (100 req more or
       less out of 50K is negligible).
     - Fixed window is the simplest to implement in Redis (INCR +
       EXPIRE). Resets every night at midnight UTC.
     - Naive implementation : one Redis counter per key, with a 24h TTL.

  3. Rate-limiting key :
     - PRIMARY : API key (the client authenticates with a token).
       -> Allows distinguishing free/pro/enterprise and billing.
     - SECONDARY (fallback) : IP address, for anonymous endpoints
       (e.g. GET /public, login page, etc.).
     - NEVER the user_id alone : the client can log out and
       create a new account to bypass it.

  4. Where to store the counters :
     - IN REDIS, centrally.
     - Why not in memory per pod : with 20 pods, the client
       could do 20x the limit by hitting each pod. Impossible
       to coordinate without a central store.
     - Why not in the LB : L7 LBs do not share state
       between instances (nginx is stateless).
     - Pattern : Redis + atomic Lua script (INCRBY + GET + EXPIRE
       in a single operation, no race condition).

  5. HTTP headers to return (RFC 9239 draft, or legacy headers) :
     - RateLimit-Limit: 100         (or X-RateLimit-Limit)
     - RateLimit-Remaining: 42       (remaining tokens)
     - RateLimit-Reset: 45           (seconds before refill)
     - Retry-After: 30               (if the limit is reached)
     Well-written clients read these headers and adapt their rate.

  6. HTTP status : 429 Too Many Requests
     - With a JSON body :
       {
         "error": "rate_limited",
         "message": "100 req/min exceeded",
         "retry_after_seconds": 45
       }
     - Never 503 Service Unavailable (it implies the service
       is down, when it is just the client misbehaving).

  7. Burst handling :
     - The token bucket NATIVELY allows bursts up to 'capacity'.
     - For a Pro 1000 req/min : capacity = 1000, refill = 1000/60 = 16.7/s.
       -> The client can send 1000 req at once, then regenerates 16.7/s.
     - If we want to limit the burst to 200 : capacity=200, refill=16.7/s.
       The bucket empties in 12s if the limit is hit, then slow refill.
     - This is the key difference with sliding window which forbids bursts.

  Bonus : different tiers via different buckets in Redis :
     key = f"rl:{api_key}:minute"  with TTL=60
     key = f"rl:{api_key}:day"     with TTL=86400
     We load the tier from the user DB the first time (cache-aside).
    """)


# =============================================================================
# EASY -- Exercise 3 : Debugging a failure cascade
# =============================================================================


def easy_3_cascading_failure():
    """Solution: root cause and remediation of a classic cascade."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : Failure cascade recommendations-api -> home-page")
    print(SEPARATOR)

    print("""
  1. Why home-page went down :
     - When recommendations-api slows down (50 ms -> 2 s), each
       home-page request blocks a thread for 2 seconds instead of 50 ms.
     - At a constant rate, the number of BUSY threads/connections
       explodes by 40x. Thread pool of 200? Saturated within seconds.
     - All subsequent requests wait for an available thread ->
       they too take 2s+. Snowball effect.
     - Retries make it worse : each failure triggers 3 retries, so
       x4 the load on recommendations-api -> it goes down harder.
     - Result : home-page is technically up, but all its
       workers are blocked in slow calls. To the users,
       it's down.

  2. Why home-page stays stuck after the recovery :
     - The existing threads are still inside in-flight calls
       with a 30s timeout. They only free up upon expiration.
     - The accumulated request queue is huge ; even after
       reco-api responds, there is a big backlog to digest.
     - If the connection pools are leaked (not closed on error),
       they stay closed even after the recovery.
     - You either have to wait, or restart home-page (which the
       ops should have done).

  3. 4+ concrete improvements :
     a) CIRCUIT BREAKER on the reco-api call :
        - failureThreshold = 10 errors in 30s
        - OPEN : immediate failure + fallback (1 ms instead of 30s)
        - HALF_OPEN after 30s to test the recovery
        -> Stops sending when things go bad, reduces the pressure on reco-api
           and prevents home-page saturation.

     b) SHORT TIMEOUT : 200-500 ms instead of 30 s
        - If reco-api normally responds in 50 ms, a 500 ms timeout
          is plenty (10x the p99).
        - Better to fail fast and show a fallback than to block
          a thread for 30 seconds.

     c) LIMIT THE RETRIES : max 1 retry with jitter, global budget
        - 3 retries = x4 the load = retry-storm.
        - 0 or 1 retry + budget (no more than 10% retries in the
          total traffic).

     d) BULKHEAD (resource isolation) :
        - Dedicated thread pool for reco-api (e.g. 50 threads), not the
          general pool. When it saturates, the other calls (DB, cache, user
          service) still work.
        - Hystrix pattern.

     e) GRACEFUL FALLBACK :
        - On failure, return a generic product list
          (top 10 sellers, stale cache, or hidden section).
        - The user sees a functional page without personalized recos
          rather than a 500 page.

     f) ASYNCHRONOUS / NON-BLOCKING CALL :
        - Ideally, the reco is loaded via AJAX after the main
          page renders. If reco-api goes down, the page is there,
          only the reco block is missing.
        - Maximum decoupling : the core path no longer depends on reco-api.

     g) MONITORING AND ALERTS :
        - Alert on reco-api p99 > 500 ms for 1 min.
        - Alert on the rate of open circuit breakers.
        - Dashboard : latency, throughput, errors per dependency.

  4. Recommended timeout value :
     - 200 ms to 500 ms, based on the normal p99 x ~10.
     - If reco-api has a p99 of 50 ms, then 500 ms = 10x headroom.
     - A 30 s timeout means 'I wait for it to respond even if
       it's catastrophic'. That's the opposite of the fail-fast philosophy.
     - As an SLA : 'Better a degraded response in 100 ms than a
       perfect response in 30 s'.

  5. Fallback for the recos :
     - Redis cache : 'top-products-fallback' refreshed every hour,
       contains the global top 20 sellers.
     - If the cache exists : display it, with a discreet 'Popular' banner.
     - If the cache is empty too : HIDE the section entirely
       (no visible error on the user side).
     - Never show a red error to the user for a reco failure.

  Summary : the problem was NOT reco-api. The problem was the
  tight coupling + long timeouts + aggressive retries on the home-page side.
  A secondary dependency must NEVER be able to kill a main
  service. That's the fundamental law of distributed architecture.
    """)


def main():
    easy_1_choose_lb_algorithm()
    easy_2_rate_limiter_design()
    easy_3_cascading_failure()
    print(f"\n{SEPARATOR}")
    print("  End of Day 5 solutions.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
