"""
Solutions -- Day 4 Exercises: Message Queues & Event-Driven

This file contains the solutions and reasoning for the Easy exercises.
Each solution is explained step by step.

Usage:
    python 04-message-queues-event-driven.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercise 1 : Queue or no queue?
# =============================================================================


def easy_1_queue_or_not():
    """Solution for identifying the need for a queue."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Queue or no queue?")
    print(SEPARATOR)

    decisions = [
        (
            "POST /signup + welcome email",
            "QUEUE -- Point-to-point -- SQS or RabbitMQ",
            "SMTP takes 2-5s, not acceptable inside the HTTP request. "
            "We push a 'SendWelcomeEmail' job into a queue, the worker "
            "processes it in the background. Point-to-point because we want ONE single "
            "email sent. SQS if on AWS, RabbitMQ otherwise (simple, reliable)."
        ),
        (
            "GET /users/:id from Postgres",
            "NO QUEUE -- Direct synchronous request",
            "A read query that must respond immediately. "
            "Adding a queue would be absurd: we cannot wait for "
            "asynchronous processing for a GET. Here we rather add "
            "a Redis cache to speed things up."
        ),
        (
            "IoT 500K events/sec to 5 teams",
            "QUEUE -- Pub/sub -- Kafka",
            "Enormous throughput (500K/sec) = Kafka mandatory (RabbitMQ "
            "saturates at ~100K). Pub/sub because 5 independent consumers "
            "(each with a consumer group). Kafka also allows replay "
            "to onboard a 6th team later without losing the history."
        ),
        (
            "Stock exchange order in < 10 us",
            "NO QUEUE -- In-memory critical path",
            "Even Kafka has 2-5 ms latency. For < 10 us (microseconds), "
            "we are in HFT territory: everything in memory, zero network between the "
            "matching engine and the order. A queue would add 1000x the "
            "allowed latency. We use in-memory ring buffers "
            "(LMAX Disruptor style)."
        ),
        (
            "Image resizing (5-30s)",
            "QUEUE -- Point-to-point -- SQS, RabbitMQ or Celery",
            "Long, CPU/IO-intensive processing. We push 'ResizeImage{url}' "
            "into a queue, a pool of workers processes in parallel. "
            "Point-to-point (each image processed once). DLQ mandatory "
            "for corrupted images. It's the textbook case for SQS/Celery."
        ),
        (
            "GET /search in < 100 ms",
            "NO QUEUE -- Direct query to Elasticsearch",
            "Another synchronous read. The search must respond within the "
            "HTTP request. We do not use a queue to serve reads. "
            "However, indexing (when a document is created) can "
            "go through a queue upstream of Elasticsearch."
        ),
    ]

    for i, (system, verdict, reason) in enumerate(decisions, 1):
        print(f"\n  {i}. {system}")
        print(f"     Verdict : {verdict}")
        print(f"     Reason  : {reason}")


# =============================================================================
# EASY -- Exercise 2 : Consumer idempotency
# =============================================================================


def easy_2_idempotent_consumer():
    """Solution: at-least-once idempotent worker via INSERT ON CONFLICT."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : Kafka consumer idempotency")
    print(SEPARATOR)

    schema = """
    CREATE TABLE payments (
        payment_id   TEXT PRIMARY KEY,          -- natural UNIQUE, provided by the producer
        user_id      TEXT NOT NULL,
        amount_cents INTEGER NOT NULL,
        status       TEXT NOT NULL,             -- 'processed', 'failed', ...
        processed_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    print("\n  1. SQL schema :")
    print(schema)

    worker_code = '''
    def process_payment_event(event: dict):
        """
        At-least-once + idempotency = effectively exactly-once.

        WHY INSERT ON CONFLICT : it is atomic on the Postgres side. If two
        processes attempt the same insert in parallel, only one wins,
        the other gets a no-op. No race condition possible.
        """
        payment_id = event["payment_id"]
        user_id = event["user_id"]
        amount = event["amount_cents"]

        # Atomic primitive: try the insert, ignore if it already exists
        result = db.execute("""
            INSERT INTO payments (payment_id, user_id, amount_cents, status)
            VALUES (%s, %s, %s, 'processed')
            ON CONFLICT (payment_id) DO NOTHING
            RETURNING payment_id
        """, (payment_id, user_id, amount))

        if result.rowcount == 0:
            # Already processed earlier: silent NO-OP
            log.info(f"payment {payment_id} already processed, skipping")
            return

        # First processing: side effect (debit the account)
        debit_account(user_id, amount)
        # The Kafka commit happens after the return (consumer loop)
    '''
    print("  2. Worker code :")
    print(worker_code)

    print("\n  3. Scenario : the message is received 3 times")
    print("""
     - Receipt 1 : INSERT succeeds (rowcount=1) -> debit_account() -> Kafka commit
     - Receipt 2 : INSERT is a no-op (conflict on payment_id) -> silent skip
     - Receipt 3 : same, silent no-op

     Result : the account is debited EXACTLY once, even though the message
     was seen 3 times by the worker.
    """)

    print("  4. Why 'SELECT then INSERT' does not work (race condition) :")
    print("""
     Sequence :
       Worker A : SELECT ... WHERE payment_id = 'X'  -> not found
       Worker B : SELECT ... WHERE payment_id = 'X'  -> not found
       Worker A : INSERT ...                          -> OK
       Worker B : INSERT ...                          -> OK (or late error)
       Worker A : debit_account(amount)
       Worker B : debit_account(amount)   <- DOUBLE DEBIT !

     The SELECT and the INSERT are two distinct operations. In between,
     another process can insert. The check is stale. The only
     correct solution is an ATOMIC primitive (INSERT ... ON CONFLICT,
     or UNIQUE constraint + try/except IntegrityError, or SELECT FOR UPDATE).
    """)

    print("  5. SQL primitives that solve the problem :")
    print("""
     - INSERT ... ON CONFLICT (payment_id) DO NOTHING      (Postgres, recommended)
     - INSERT ... ON DUPLICATE KEY UPDATE                  (MySQL)
     - MERGE INTO ...                                      (SQL Server, Oracle)
     - UNIQUE constraint + try/except IntegrityError       (portable)
     - SELECT FOR UPDATE inside a transaction              (heavier lock)
    """)


# =============================================================================
# EASY -- Exercise 3 : Kafka sizing
# =============================================================================


def easy_3_kafka_sizing():
    """Solution: capacity planning for the ride-sharing topic."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : Kafka sizing (ride-sharing)")
    print(SEPARATOR)

    # Data
    drivers = 200_000
    interval_sec = 4
    event_bytes = 400
    peak_factor = 3
    retention_days = 7
    replication = 3

    # 1. Average throughput
    events_per_sec_avg = drivers / interval_sec  # 50,000
    bytes_per_sec_avg = events_per_sec_avg * event_bytes
    mb_per_sec_avg = bytes_per_sec_avg / (1024 * 1024)

    print(f"\n  1. Average throughput :")
    print(f"     events/sec = {drivers} drivers / {interval_sec}s = {events_per_sec_avg:,.0f}")
    print(f"     bytes/sec  = {events_per_sec_avg:,.0f} * {event_bytes} B = {bytes_per_sec_avg/1e6:.1f} MB/s")
    print(f"     i.e. ~{mb_per_sec_avg:.1f} MiB/s")

    # 2. Peak throughput
    events_per_sec_peak = events_per_sec_avg * peak_factor
    mb_per_sec_peak = mb_per_sec_avg * peak_factor
    print(f"\n  2. Peak throughput (x{peak_factor}) :")
    print(f"     events/sec = {events_per_sec_peak:,.0f}")
    print(f"     MiB/s      = {mb_per_sec_peak:.1f}")

    # 3. Partitions
    # Rule: we size for the peak. A consumer comfortably handles ~10 MB/s.
    # We also target the desired parallelism (e.g. matching algorithm scales to 30 pods).
    consumer_throughput_mb = 10
    partitions_by_throughput = max(1, int(mb_per_sec_peak / consumer_throughput_mb) + 1)
    # We round up to leave some headroom
    recommended_partitions = 60
    print(f"\n  3. Partitions :")
    print(f"     Peak {mb_per_sec_peak:.1f} MiB/s / {consumer_throughput_mb} MiB/s per consumer "
          f"= {partitions_by_throughput} partitions minimum")
    print(f"     Recommended : {recommended_partitions} partitions")
    print(f"     Why round up : anticipate growth and the parallelism")
    print(f"     of the matching algorithm (30+ pods). Changing the partition count after")
    print(f"     the fact is painful (breaks per-key ordering).")

    # 4. Raw storage
    seconds_per_day = 86400
    bytes_per_day = bytes_per_sec_avg * seconds_per_day
    # We use the AVERAGE for storage (not the peak) because the peak is transient
    total_bytes_7d = bytes_per_day * retention_days
    total_gb_7d = total_bytes_7d / 1e9
    total_tb_7d = total_bytes_7d / 1e12
    print(f"\n  4. Raw storage for 7 days :")
    print(f"     bytes/day = {bytes_per_sec_avg/1e6:.1f} MB/s * 86400 = {bytes_per_day/1e9:.0f} GB/day")
    print(f"     over 7d   = {total_gb_7d:,.0f} GB = {total_tb_7d:.1f} TB")

    # 5. With replication
    replicated_tb = total_tb_7d * replication
    print(f"\n  5. Storage with replication factor {replication} :")
    print(f"     {total_tb_7d:.1f} TB x {replication} = {replicated_tb:.1f} TB (total cluster)")

    # 6. Brokers
    broker_capacity_tb = 2
    # Never fill a disk to 100%: target 60-70% max
    usable_per_broker = broker_capacity_tb * 0.65
    min_brokers = int(replicated_tb / usable_per_broker) + 1
    recommended_brokers = max(min_brokers, 6)  # minimum 6 for HA across 3 AZs
    print(f"\n  6. Number of brokers :")
    print(f"     {replicated_tb:.1f} TB / ({broker_capacity_tb} TB * 65%) = {min_brokers} brokers min")
    print(f"     Recommended : {recommended_brokers} brokers (minimum for HA, CPU/IO headroom)")
    print(f"     Spread across 3 availability zones to survive the loss of an AZ.")

    print("\n  Final remarks :")
    print("     - Use 'driver_id' as the partition key to keep per-driver ordering")
    print("     - Retention=7d allows replaying 1 week of events for debug/backfill")
    print("     - Monitor the lag of the 'matching' consumer group : lag > 10s = alert")


def main():
    easy_1_queue_or_not()
    easy_2_idempotent_consumer()
    easy_3_kafka_sizing()
    print(f"\n{SEPARATOR}")
    print("  End of Day 4 solutions.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
