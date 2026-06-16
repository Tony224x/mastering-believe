"""
Solutions -- Day 5 HARD Exercises: Load Balancing & Networking

Worked solutions. Multi-region sizing and the thread-exhaustion / SLA math are
computed with assertions on the key numbers.

Usage:
    python3 05-load-balancing-networking-hard.py
"""

import math

SEPARATOR = "=" * 60


# =============================================================================
# HARD -- Exercise 1 : Resilient multi-region architecture
# =============================================================================

def hard_1_multi_region():
    """Design + size a global, region-failover-tolerant network layer."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Resilient multi-region architecture")
    print(SEPARATOR)

    total_rps = 2_000_000
    split = {"EU": 0.45, "US": 0.35, "APAC": 0.20}
    per_server_rps = 20_000
    region_sla = 0.9995

    print("\n  1. Global routing :")
    print("     GeoDNS (or anycast) routes each user to the nearest region. Two LB")
    print("     layers : global routing (GeoDNS/anycast) -> regional L7 LB -> pods.")
    print("     DNS TTL problem : intermediate resolvers cache the answer (30s-24h),")
    print("     so removing a dead region is slow. Mitigate with short TTL + health-")
    print("     based DNS failover, or use anycast (BGP withdraws a dead PoP fast).")

    print("\n  2. Sizing per region :")
    base = {}
    for r, frac in split.items():
        rps = total_rps * frac
        servers = math.ceil(rps / per_server_rps)
        base[r] = servers
        print(f"     {r:4} : {rps:,.0f} req/s -> {servers} servers (base)")
    assert base["EU"] == 45 and base["US"] == 35 and base["APAC"] == 20

    # Survive losing one region: surviving regions must absorb the biggest region.
    biggest = max(split.values()) * total_rps          # EU = 900K
    survivors_rps_each = biggest / 2                    # split lost load over 2 survivors
    extra_servers = math.ceil(survivors_rps_each / per_server_rps)
    print(f"     Survive losing the biggest region (EU={biggest:,.0f} req/s) :")
    print(f"       each survivor must take ~{survivors_rps_each:,.0f} extra req/s")
    print(f"       -> +{extra_servers} servers of headroom per surviving region")
    print("       (rough rule : provision each region ~50% over its base load).")
    assert extra_servers >= 20

    print("\n  3. Failover (US-East down at 14:00) :")
    print("     Sequence : regional healthcheck fails -> global DNS/anycast stops")
    print("     advertising US -> users re-resolve to EU/APAC. Time-to-serve depends")
    print("     on DNS TTL (seconds to minutes) or is near-instant with anycast.")
    us_rps = split["US"] * total_rps
    print(f"     Latency hit : US users -> EU-West adds ~80ms transatlantic.")
    print(f"     Thundering herd : {us_rps:,.0f} req/s shift at once -> protect with")
    print("     pre-warmed headroom (the +50%), autoscaling, admission control and")
    print("     rate limiting so survivors degrade gracefully instead of collapsing.")
    assert abs(us_rps - 700_000) < 1

    print("\n  4. State & consistency :")
    print("     Regional session cache : a re-routed user misses their session.")
    print("     Mitigate with globally-replicated sessions, stateless JWT, or a quick")
    print("     re-auth. Multi-region writes : eventual consistency is the pragmatic")
    print("     default (low latency, available); strong consistency costs cross-region")
    print("     round-trips and blocks on partitions (CAP). Reserve strong for money.")

    # 5. SLA math
    one_region = region_sla
    three_region_failover = 1 - (1 - region_sla) ** 3
    downtime_one = (1 - one_region) * 365 * 24 * 60          # minutes/year
    downtime_three = (1 - three_region_failover) * 365 * 24 * 60
    print("\n  5. SLA math :")
    print(f"     Single region : {one_region*100:.2f}% -> {downtime_one:.0f} min downtime/year")
    print(f"     3 regions, need >=1 up : 1-(1-{region_sla})^3 = "
          f"{three_region_failover*100:.5f}%")
    print(f"       -> {downtime_three:.2f} min downtime/year")
    print("     Multi-region multiplies availability BUT adds routing, replication,")
    print("     consistency and failover complexity -> more moving parts to get right.")
    assert three_region_failover > 0.99999
    assert downtime_three < downtime_one

    print("\n  6. Monitoring (6 metrics) + failover trigger :")
    metrics = [
        ("Regional p99 latency", "> 150ms (SLA breach)"),
        ("Regional healthcheck success", "fails > 30s -> FAILOVER"),
        ("Per-region req/s vs capacity", "> 80% capacity"),
        ("Error rate (5xx) per region", "> 1%"),
        ("DNS/anycast propagation time", "track during failover"),
        ("Cross-region replication lag", "> threshold"),
    ]
    for name, alert in metrics:
        print(f"     - {name:<32} {alert}")


# =============================================================================
# HARD -- Exercise 2 : Cascade-failure post-mortem
# =============================================================================

def hard_2_cascade_postmortem():
    """Post-mortem: slow external dependency -> thread exhaustion cascade."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Cascade-failure post-mortem")
    print(SEPARATOR)

    print("\n  1. Full causal chain :")
    chain = [
        ("EXTERNAL", "payment-gateway latency 200ms -> 30s",
         "trigger, not the root of the OUTAGE"),
        ("ARCHITECTURE", "30s threads block; shared 200-thread pool",
         "Missing guardrail : aggressive timeout + bulkhead isolation"),
        ("ARCHITECTURE", "pool saturated -> even non-payment requests rejected",
         "Missing guardrail : per-dependency thread pool (bulkhead)"),
        ("ARCHITECTURE", "client retries 3x without backoff -> traffic 3x",
         "Missing guardrail : bounded retry + backoff + jitter + budget"),
        ("ARCHITECTURE", "no circuit breaker -> keep calling a dying dependency",
         "Missing guardrail : circuit breaker with fallback"),
        ("MONITORING", "LB marks healthy servers unhealthy -> sheds them -> cascade",
         "Missing guardrail : healthcheck tuning + global health awareness"),
        ("PROCESS", "detected/acted late, no degraded mode ready",
         "Missing guardrail : pre-built degraded mode + runbook"),
    ]
    for cat, cause, guardrail in chain:
        print(f"     [{cat}] {cause}")
        print(f"       -> {guardrail}")

    # 2. Thread exhaustion math
    threads = 200
    payment_latency_s = 30
    servers = 30
    per_server_rps = threads / payment_latency_s          # payment req/s before exhaustion
    total_rps = per_server_rps * servers
    normal_payment_rps = 6_000                            # nominal traffic
    print("\n  2. Saturation math :")
    print(f"     1 server : {threads} threads / {payment_latency_s}s "
          f"= {per_server_rps:.1f} payment req/s before thread exhaustion")
    print(f"     {servers} servers : {total_rps:.0f} payment req/s total before full saturation")
    print(f"     Nominal traffic ~{normal_payment_rps:,} req/s >> {total_rps:.0f} -> the moment")
    print("     payment slows to 30s, capacity drops ~150x and the pools fill in seconds.")
    assert abs(per_server_rps - 6.666) < 0.01
    assert abs(total_rps - 200) < 0.1
    assert normal_payment_rps > total_rps * 10           # demonstrates the gap

    print("\n  3. Retry without backoff :")
    print("     3 immediate retries on timeout multiply the load on an ALREADY dying")
    print("     dependency (retry storm) -> it never recovers. Correct strategy :")
    print("     - max 2-3 attempts, exponential backoff (1s,2s,4s...) + jitter")
    print("     - retry budget <= 10% of traffic (shed beyond that)")
    print("     - SHORT timeout (2s, not 30s) so a slow call frees the thread fast")

    print("\n  4. Why a non-payment service also fell :")
    print("     SHARED thread pool : payment calls consumed all 200 threads, so")
    print("     requests that never touch payment couldn't get a thread either.")
    print("     The BULKHEAD pattern (separate bounded pool per dependency) would have")
    print("     contained payment's blast radius to payment-only requests.")

    print("\n  5. Fixed architecture :")
    print("     Circuit breaker (payment) :")
    print("       OPEN when > 50% errors OR > 20 timeouts / 10s")
    print("       OPEN -> fail fast / fallback (no thread held)")
    print("       HALF-OPEN after 30s : one probe before resuming")
    bulkhead_payment = 40
    print(f"     Bulkhead : dedicate at most {bulkhead_payment} threads to payment ->")
    print(f"       payment can never starve the other {threads - bulkhead_payment} threads.")
    assert bulkhead_payment < threads
    print("     Degraded mode : accept the booking, process payment ASYNC via a queue")
    print("       (no 500). User sees 'booking confirmed, payment processing'.")
    print("     Healthcheck : don't shed all servers at once -> use a global health")
    print("       view / minimum healthy pool + slow-start on re-add (avoid death")
    print("       spiral where shedding overloads survivors).")

    print("\n  6. Runbook (8 steps) :")
    steps = [
        "STOP the amplification : disable client retry / open the breaker, cut timeout to 2s",
        "Confirm the trigger : is the external dependency slow or down?",
        "Switch the affected path to degraded mode (async payment)",
        "Protect survivors : enable rate limiting / admission control",
        "Fix the healthcheck so healthy servers aren't shed",
        "Watch thread pool usage, p99, error rate recover",
        "Once the dependency recovers : half-open the breaker, re-enable retry gradually",
        "Post-mortem in 24h : timeline, root cause, bulkhead + breaker rollout",
    ]
    for i, s in enumerate(steps, 1):
        print(f"     {i}. {s}")


def main():
    print("\n" + "=" * 60)
    print("  SOLUTIONS -- DAY 5 HARD : LOAD BALANCING & NETWORKING")
    print("=" * 60)
    hard_1_multi_region()
    hard_2_cascade_postmortem()
    print(f"\n{'=' * 60}")
    print("  END OF HARD SOLUTIONS (all assertions passed)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
