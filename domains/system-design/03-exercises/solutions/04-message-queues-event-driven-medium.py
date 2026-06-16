"""
Solutions -- Day 4 MEDIUM Exercises: Message Queues & Event-Driven

Worked solutions with the reasoning step by step. The sizing exercises are
computed (with assertions on the key numbers) so the file is runnable and
self-checking. The design exercises print the structured reasoning expected
in a senior interview.

Usage:
    python3 04-message-queues-event-driven-medium.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# MEDIUM -- Exercise 1 : Broker choice + partition sizing
# =============================================================================

def medium_1_broker_and_partitions():
    """Choose the broker per flow and size the Kafka partitions of flow A."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Broker choice + partition sizing")
    print(SEPARATOR)

    print("\n  1. Broker per flow :")
    print("     Flow A (transactions, 4 consumers, replay 30d, ordered) -> KAFKA")
    print("       pub/sub via consumer groups : each team reads the whole stream")
    print("       independently. Long retention + replay = audit. Ordering per")
    print("       account via partition key. RabbitMQ/SQS cannot replay 30 days.")
    print("     Flow B (transactional emails, 1 consumer, moderate) -> SQS (or RabbitMQ)")
    print("       point-to-point : 1 email sent once. No replay needed. SQS = zero ops.")
    print("     Flow C (outbound webhooks, retries, per-merchant isolation) -> RabbitMQ (or SQS)")
    print("       rich routing + per-queue DLQ + retry/backoff. Isolate slow merchants.")

    # ---- Flow A partition sizing ----
    peak_events_per_sec = 90_000
    payload_bytes = 600
    # Rule of thumb: a single consumer instance comfortably handles
    # ~10 MB/s OR ~10K msg/s of business logic. We take the binding one.
    per_consumer_msgs = 10_000          # msg/s per consumer
    per_consumer_mb = 10                # MB/s per consumer

    peak_mb_per_sec = peak_events_per_sec * payload_bytes / 1e6
    partitions_by_msgs = -(-peak_events_per_sec // per_consumer_msgs)   # ceil
    partitions_by_mb = -(-int(peak_mb_per_sec) // per_consumer_mb)      # ceil
    partitions_needed = max(partitions_by_msgs, partitions_by_mb)
    # Round up to a comfortable headroom value (multiple of 12 for parallelism).
    recommended_partitions = 12

    print("\n  2. Flow A partition sizing :")
    print(f"     Peak throughput : {peak_events_per_sec:,} msg/s "
          f"= {peak_mb_per_sec:.1f} MB/s")
    print(f"     Rule : ~{per_consumer_msgs:,} msg/s OR ~{per_consumer_mb} MB/s per consumer")
    print(f"     By msgs : {peak_events_per_sec:,} / {per_consumer_msgs:,} "
          f"= {partitions_by_msgs} partitions")
    print(f"     By bytes: {peak_mb_per_sec:.0f} / {per_consumer_mb} "
          f"= {partitions_by_mb} partitions")
    print(f"     Minimum : {partitions_needed} -> recommend {recommended_partitions} "
          f"(headroom + clean parallelism)")
    print("     Partition key = account_id : all events of one account land on the")
    print("     same partition -> ordering per account guaranteed (the business need).")
    assert partitions_needed == 9, partitions_needed
    assert recommended_partitions >= partitions_needed

    print("\n  3. 50 consumers in one group vs 12 partitions :")
    print("     IMPOSSIBLE to use 50 active consumers : a partition is consumed by")
    print(f"     exactly ONE consumer of the group, so only {recommended_partitions} are active,")
    print("     the other 38 sit idle. Fix : raise the partition count (e.g. 48-64)")
    print("     BEFORE you need that parallelism (partitions can grow, not shrink,")
    print("     and growing breaks key->partition mapping, so over-provision early).")

    # ---- Flow A storage 30 days, RF=3 ----
    avg_events_per_sec = 30_000
    retention_days = 30
    rf = 3
    seconds = retention_days * 86_400
    raw_bytes = avg_events_per_sec * payload_bytes * seconds
    raw_tb = raw_bytes / 1e12
    replicated_tb = raw_tb * rf

    print("\n  4. Flow A storage (30 days, RF=3) :")
    print(f"     Avg : {avg_events_per_sec:,} msg/s * {payload_bytes} B * {seconds:,} s")
    print(f"         = {raw_bytes/1e12:.1f} TB raw")
    print(f"     With RF={rf} : {replicated_tb:.1f} TB on the cluster")
    print("     (Order of magnitude : a few dozen TB -> plan brokers + tiered storage.)")
    assert 40 <= raw_tb <= 60, raw_tb            # ~46.7 TB raw
    assert 130 <= replicated_tb <= 180, replicated_tb

    print("\n  5. Flow C noisy neighbor :")
    print("     If all merchants share one queue/partition and one merchant is slow")
    print("     (timeouts), its messages occupy the consumers and stall the head of")
    print("     the queue -> other merchants' webhooks wait behind it (head-of-line).")
    print("     Isolation options :")
    print("     - Per-merchant queue (RabbitMQ) or partition key = merchant_id")
    print("     - Per-merchant concurrency limit + circuit breaker on slow merchants")
    print("     - Bulkhead : a bounded worker pool per tenant so one cannot starve all")


# =============================================================================
# MEDIUM -- Exercise 2 : Checkout saga
# =============================================================================

def medium_2_checkout_saga():
    """Design the checkout saga with compensations."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Checkout saga")
    print(SEPARATOR)

    saga = [
        ("CreateOrder (status=PENDING)", "CancelOrder (status=CANCELLED)"),
        ("ChargeCard (idempotency key)", "RefundPayment"),
        ("ReserveStock", "ReleaseStock"),
        ("CreateShipment", "CancelShipment"),
    ]
    print("\n  1. Saga steps and compensations :")
    for i, (step, comp) in enumerate(saga, 1):
        print(f"     Step {i}: {step:<32} | compensation: {comp}")
    print("     If step 3 (ReserveStock) FAILS : run compensations in REVERSE")
    print("     of the steps already done -> RefundPayment, then CancelOrder.")
    print("     (Step 4 never ran, so nothing to compensate there.)")
    # Invariant: a saga of N steps has N compensations.
    assert all(comp for _, comp in saga)
    assert len(saga) == 4

    print("\n  2. Orchestrated vs choreographed :")
    print("     CHOSEN : orchestrated. A central orchestrator drives the 4 steps and")
    print("     triggers compensations. Why : 4-5 steps with branching failures, the")
    print("     business logic stays in one place, the saga is debuggable and the")
    print("     state machine is explicit. Choreography (pure events) would scatter")
    print("     the compensation logic across 4 services -> hard to follow at 4+ steps.")

    print("\n  3. Payment timeout (no response) :")
    print("     We DON'T know if the card was charged. Making the step safe to retry :")
    print("     send an Idempotency-Key (e.g. order_id) with the charge. On retry, the")
    print("     payment provider returns the SAME result instead of charging twice.")
    print("     The orchestrator can re-issue the charge command without double-billing.")

    print("\n  4. A compensation itself fails (refund API down) :")
    print("     Compensations must be retried until success : exponential backoff +")
    print("     jitter, persisted in a saga log (so a crash resumes), DLQ after N tries")
    print("     with a human alert. A failed refund is a money-impacting event -> it")
    print("     must never be silently dropped; it stays 'in flight' until reconciled.")

    print("\n  5. State seen by the user during the saga :")
    print("     Intermediate state : the order is PENDING / 'processing'. We assume")
    print("     EVENTUAL consistency between creation and confirmation. The UI shows")
    print("     'order received, confirming payment' and only flips to CONFIRMED once")
    print("     the saga completes (or CANCELLED if it compensates).")


# =============================================================================
# MEDIUM -- Exercise 3 : Event sourcing + CQRS wallet
# =============================================================================

def medium_3_event_sourcing_wallet():
    """Event sourcing + CQRS for a fintech wallet."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Event sourcing + CQRS wallet")
    print(SEPARATOR)

    print("\n  1. Why event sourcing here :")
    print("     The regulator requires a complete, immutable audit trail and the")
    print("     ability to reconstruct the balance at any past date (time travel).")
    print("     A plain CRUD 'balance' column loses history -> non-compliant. The")
    print("     value is the history itself, which is exactly when ES pays off.")

    # 2 + 3 : demonstrate fold + snapshot on a tiny event log.
    events = [
        ("Credited", 100),
        ("Credited", 80),
        ("Debited", -30),
        ("Credited", 50),
        ("Debited", -20),
    ]
    balance = sum(delta for _, delta in events)
    print("\n  2. Write path :")
    print("     Append IMMUTABLE events to the log (Credited/Debited/Frozen/...).")
    print("     Current balance = fold (sum) of the event deltas.")
    print(f"     Example log {[ (n, d) for n, d in events ]}")
    print(f"     -> balance = {balance}")
    assert balance == 180

    # Snapshot: store balance at T, then replay only events after T.
    snapshot_after = 3                      # snapshot taken after 3 events
    snapshot_balance = sum(d for _, d in events[:snapshot_after])
    tail = events[snapshot_after:]
    balance_from_snapshot = snapshot_balance + sum(d for _, d in tail)
    print("\n  3. Avoid replaying 2M events : SNAPSHOTTING.")
    print(f"     Snapshot balance after {snapshot_after} events = {snapshot_balance},")
    print(f"     then replay only the {len(tail)} events since the snapshot.")
    print(f"     Reconstructed balance = {balance_from_snapshot} (== full fold {balance})")
    assert balance_from_snapshot == balance        # snapshot must be exact

    print("\n  4. Two CQRS projections from the same event stream :")
    print("     - 'Current balance' projection : Redis / KV store, key=account_id,")
    print("       O(1) read for the dashboard 'solde courant'.")
    print("     - 'History' projection : SQL table indexed by (account_id, ts),")
    print("       serves 'last 50 operations' with cursor pagination.")
    print("     Each projection is rebuildable by replaying the stream.")

    print("\n  5. Fixing erroneous Debited events (immutable history) :")
    print("     Never edit the log. Append COMPENSATING events (e.g. a corrective")
    print("     Credited or a Reversed(of=event_id)) that cancel the bad deltas.")
    print("     The balance re-folds to the correct value and the audit trail shows")
    print("     both the error and the correction -> regulator-friendly.")


def main():
    print("\n" + "=" * 60)
    print("  SOLUTIONS -- DAY 4 MEDIUM : MESSAGE QUEUES & EVENT-DRIVEN")
    print("=" * 60)
    medium_1_broker_and_partitions()
    medium_2_checkout_saga()
    medium_3_event_sourcing_wallet()
    print(f"\n{'=' * 60}")
    print("  END OF MEDIUM SOLUTIONS (all assertions passed)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
