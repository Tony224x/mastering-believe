"""
Solutions -- Day 4 HARD Exercises: Message Queues & Event-Driven

Worked solutions. The dispatch-pipeline sizing is computed with assertions
on the key numbers; the post-mortem is a structured causal analysis with a
small computed amplification model.

Usage:
    python3 04-message-queues-event-driven-hard.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# HARD -- Exercise 1 : Real-time dispatch pipeline (Uber-like)
# =============================================================================

def hard_1_dispatch_pipeline():
    """Design + size the location ingestion pipeline."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Real-time dispatch pipeline")
    print(SEPARATOR)

    drivers = 500_000
    period_s = 3                 # one position every 3 seconds
    payload_bytes = 500
    rf = 3

    # 1. Sizing
    avg_eps = drivers / period_s
    avg_mb_s = avg_eps * payload_bytes / 1e6
    peak_eps = avg_eps * 3       # peak factor 3x
    peak_mb_s = peak_eps * payload_bytes / 1e6

    print("\n  1. Sizing :")
    print(f"     Avg  : {drivers:,} / {period_s}s = {avg_eps:,.0f} events/s "
          f"= {avg_mb_s:.1f} MB/s")
    print(f"     Peak : x3 = {peak_eps:,.0f} events/s = {peak_mb_s:.1f} MB/s")
    assert abs(avg_eps - 166_666) < 5, avg_eps
    assert 80 <= avg_mb_s <= 85, avg_mb_s

    # Partitions: rule ~10 MB/s per consumer at peak, round up to power-friendly.
    per_consumer_mb = 10
    partitions_min = -(-int(peak_mb_s) // per_consumer_mb)   # ceil
    recommended_partitions = 32
    print(f"     Partitions : peak {peak_mb_s:.0f} MB/s / {per_consumer_mb} MB/s "
          f"= {partitions_min} min -> recommend {recommended_partitions}")
    print("     Partition key = driver_id (ordering per driver).")
    assert partitions_min <= recommended_partitions
    assert recommended_partitions >= 24

    sec_24h = 24 * 3600
    sec_7d = 7 * 86_400
    raw_24h_tb = avg_eps * payload_bytes * sec_24h / 1e12
    raw_7d_tb = avg_eps * payload_bytes * sec_7d / 1e12
    print(f"     Storage 24h raw : {raw_24h_tb:.1f} TB  (RF=3 : {raw_24h_tb*rf:.1f} TB)")
    print(f"     Storage 7d  raw : {raw_7d_tb:.1f} TB  (RF=3 : {raw_7d_tb*rf:.1f} TB)")
    assert 6 <= raw_24h_tb <= 8, raw_24h_tb           # ~7.2 TB
    assert 45 <= raw_7d_tb <= 55, raw_7d_tb           # ~50.4 TB

    print("\n  2. Ordering & hot partitions :")
    print("     Ordering per driver : partition key = driver_id -> all positions of a")
    print("     driver in one partition, in order.")
    print("     KEY INSIGHT : a geographic crowd (80K drivers in one city) does NOT")
    print("     create a hot partition, because the key is driver_id, not zone. The")
    print("     80K drivers hash uniformly across all partitions. (Keying by zone")
    print("     WOULD create a hot partition -> avoid it.)")

    print("\n  3. Multi-retention (matching 24h vs data lake 7d) :")
    print("     Don't pay 7d on everything. Options :")
    print("     - Separate topics : positions-hot (24h) for matching/ETA, and a")
    print("       compacted/archived stream for the lake; OR")
    print("     - Tiered storage : 24h on local broker disk, older segments offloaded")
    print("       to S3 (Kafka tiered storage / Warpstream-style) for cheap 7d replay.")

    print("\n  4. Resilience :")
    print("     A lagging consumer group (fraud, 2M behind) does NOT impact the others")
    print("     -> each group has its OWN committed offsets; they read independently.")
    print("     Broker down -> rebalance pauses consumption a few seconds. To protect")
    print("     the < 1s SLA : RF=3 + min.insync.replicas=2 (no data loss on failover),")
    print("     cooperative/incremental rebalancing (avoid stop-the-world), and enough")
    print("     headroom so a surviving broker absorbs the load.")

    print("\n  5. Delivery guarantees :")
    print("     Matching/ETA : losing one position out of many is fine -> at-most or")
    print("     at-least-once, low retry. Data lake (audit) : at-least-once + idempotence.")
    print("     Pricing INSERT idempotent via UPSERT keyed on (driver_id, ts) -> a")
    print("     replayed position overwrites instead of duplicating.")

    print("\n  6. Monitoring (6 metrics) :")
    metrics = [
        ("Consumer lag per group", "> 100K and rising"),
        ("Partition skew (msg/partition)", "> 2x the median"),
        ("Throughput in vs out", "drop > 20%"),
        ("End-to-end p99 latency", "> 1s (SLA breach)"),
        ("Under-replicated partitions", "> 0 sustained"),
        ("DLQ size", "> 0 / growing"),
    ]
    for name, alert in metrics:
        print(f"     - {name:<32} ALERT if {alert}")


# =============================================================================
# HARD -- Exercise 2 : Retry-storm post-mortem
# =============================================================================

def hard_2_retry_storm_postmortem():
    """Post-mortem of a retry storm + poison message + DLQ saturation."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Post-mortem -- the retry storm")
    print(SEPARATOR)

    print("\n  1. Full causal chain :")
    chain = [
        ("PROCESS", "inventory deploy with an unhandled 'negative stock' case",
         "Missing guardrail : handle/guard the edge case + canary the deploy"),
        ("ARCHITECTURE", "retry immediate + unlimited on 500s",
         "Missing guardrail : bounded retries + exponential backoff + jitter"),
        ("ARCHITECTURE", "no DLQ -> poison messages block offset progress",
         "Missing guardrail : DLQ after N tries to unblock head-of-line"),
        ("ARCHITECTURE", "retry loop -> 200K req/s to inventory (vs 3K)",
         "Missing guardrail : circuit breaker between consumer and inventory"),
        ("ARCHITECTURE", "consumer lag explodes to 4M; payment/email starved",
         "Missing guardrail : per-dependency isolation, retry budget"),
        ("PROCESS", "redeploy replays 4M from last commit + rebalance pause",
         "Missing guardrail : targeted offset skip, not a blind 'from last commit'"),
        ("ARCHITECTURE", "12K duplicate confirmation emails",
         "Missing guardrail : idempotent send keyed on order_id (dedup store)"),
        ("MONITORING", "detected late, no alert on lag/error rate",
         "Missing guardrail : alerts on consumer lag + downstream error rate"),
    ]
    for cat, cause, guardrail in chain:
        print(f"     [{cat}] {cause}")
        print(f"       -> {guardrail}")

    print("\n  2. Retry infinite without DLQ :")
    print("     Immediate unlimited retry turns a local bug into a global outage : a")
    print("     handful of poison messages generate a self-amplifying request flood")
    print("     (retry storm) that saturates the downstream, so even VALID messages")
    print("     start failing -> the failure spreads.")
    print("     No DLQ = head-of-line blocking : the consumer cannot commit past the")
    print("     poison message, so its whole partition stalls behind it.")

    # Small amplification model (illustrative, asserted).
    normal_rps = 3_000
    storm_rps = 200_000
    amplification = storm_rps / normal_rps
    print(f"\n     Amplification : {storm_rps:,} / {normal_rps:,} "
          f"= ~{amplification:.0f}x normal load on inventory")
    assert amplification > 50

    print("\n     Correct retry strategy (numbers) :")
    print("     - 3 immediate retries max, then stop")
    print("     - exponential backoff : 1s, 2s, 4s ... cap 30s, + jitter(0.5-1.5x)")
    print("     - retry budget : retries <= 10% of total traffic (else shed)")
    print("     - after the budget/attempts -> DLQ")

    print("\n  3. The 4M replay :")
    print("     Redeploying restarted 'from last committed offset' (09:02) and replayed")
    print("     4M messages, re-hammering inventory and triggering a rebalance pause.")
    print("     Correct move : DON'T blind-replay. Identify the poison offsets and")
    print("     SEEK past them (or route them to DLQ), resume from the current tail for")
    print("     valid traffic. Replay history only deliberately, rate-limited.")

    print("\n  4. Duplicate emails :")
    print("     at-least-once + the 4M replay re-processed many orders -> emails resent.")
    print("     Fix : idempotent send. Before sending, INSERT order_id into a")
    print("     'sent_emails' table with a UNIQUE constraint (ON CONFLICT DO NOTHING).")
    print("     If the row already exists -> skip. Dedup BEFORE the side effect, not a")
    print("     try/catch after it.")

    print("\n  5. Fixed architecture :")
    print("     Tiered retry :")
    print("       3 in-process retries (backoff) -> topic 'orders.retry.30s'")
    print("       -> topic 'orders.retry.5m' -> DLQ 'orders.dlq'")
    print("       (delay topics decouple slow retries from the live consumer)")
    print("     Circuit breaker between consumer and inventory :")
    print("       OPEN when > 50% errors over 10s -> stop calling, serve fallback/park")
    print("       HALF-OPEN after 30s with a single probe before resuming")
    print("     Poison isolation : after N failures, move the single message to DLQ so")
    print("     it never blocks its partition; keep processing the rest.")

    print("\n  6. Runbook (8 steps) :")
    steps = [
        "STOP the amplification first : pause the consumer / open the breaker",
        "Confirm the blast radius : which partitions, error rate, lag trend",
        "Isolate poison messages : route failing offsets to DLQ (do NOT blind-replay)",
        "Cap retries / enable retry budget if not already on",
        "Roll back the faulty downstream (inventory) to the stable version",
        "Resume the consumer from the current tail for valid traffic",
        "Re-process DLQ deliberately, rate-limited, once the fix is verified",
        "Post-mortem in 24h : timeline, root cause, guardrails, idempotency audit",
    ]
    for i, s in enumerate(steps, 1):
        print(f"     {i:2d}. {s}")


def main():
    print("\n" + "=" * 60)
    print("  SOLUTIONS -- DAY 4 HARD : MESSAGE QUEUES & EVENT-DRIVEN")
    print("=" * 60)
    hard_1_dispatch_pipeline()
    hard_2_retry_storm_postmortem()
    print(f"\n{'=' * 60}")
    print("  END OF HARD SOLUTIONS (all assertions passed)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
