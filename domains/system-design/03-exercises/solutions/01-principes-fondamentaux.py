"""
Solutions -- Day 1 Exercises: Fundamental principles

This file contains the worked solutions for the Easy, Medium, and Hard exercises.
Each solution shows the reasoning step by step.

Usage:
    python 01-principes-fondamentaux.py
"""

import math

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercise 1 : QPS estimation for a notification service
# =============================================================================

def easy_1_notifications():
    """Solution for the notification service estimation."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Push notification service")
    print(SEPARATOR)

    dau = 5_000_000
    notifs_per_user = 8
    payload_bytes = 500
    peak_factor = 4
    retention_days = 30

    # 1. Average QPS
    total_notifs_per_day = dau * notifs_per_user  # 40M notifs/day
    qps_avg = total_notifs_per_day / 86_400
    print(f"\n  1. Average QPS")
    print(f"     Total notifs/day = {dau:,} * {notifs_per_user} = {total_notifs_per_day:,}")
    print(f"     Average QPS = {total_notifs_per_day:,} / 86400 = {qps_avg:,.0f} req/s")

    # 2. Peak QPS
    qps_peak = qps_avg * peak_factor
    print(f"\n  2. Peak QPS")
    print(f"     Peak QPS = {qps_avg:,.0f} * {peak_factor} = {qps_peak:,.0f} req/s")

    # 3. Outbound bandwidth at peak
    # Careful: Mbps = megabits per second (not megabytes)
    bandwidth_bytes_per_sec = qps_peak * payload_bytes
    bandwidth_mbps = bandwidth_bytes_per_sec * 8 / 1_000_000  # bytes -> bits -> megabits
    print(f"\n  3. Outbound bandwidth (peak)")
    print(f"     = {qps_peak:,.0f} req/s * {payload_bytes} bytes * 8 bits/byte")
    print(f"     = {bandwidth_mbps:,.1f} Mbps")
    print(f"     = {bandwidth_mbps / 1000:.2f} Gbps")

    # 4. Storage for 30 days
    storage_per_day_bytes = total_notifs_per_day * payload_bytes
    storage_per_day_gb = storage_per_day_bytes / (1024**3)
    storage_total_gb = storage_per_day_gb * retention_days
    print(f"\n  4. Storage (30 days)")
    print(f"     Per day = {total_notifs_per_day:,} * {payload_bytes} = {storage_per_day_bytes / 1e9:.1f} GB")
    print(f"     30 days = {storage_total_gb:.1f} GB")


# =============================================================================
# EASY -- Exercise 2 : CP or AP
# =============================================================================

def easy_2_cp_ap():
    """Solution for the CP vs AP choice."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : CP or AP -- Pick your side")
    print(SEPARATOR)

    choices = [
        (
            "Online voting (national election)",
            "CP",
            "A double vote or a lost vote is unacceptable. "
            "Better to temporarily turn away a voter than count a vote twice."
        ),
        (
            "YouTube view counter",
            "AP",
            "A counter off by a few units is invisible to the user. "
            "An unavailable counter would block the video from displaying."
        ),
        (
            "E-commerce inventory management",
            "CP",
            "Selling an out-of-stock product = cancelled order = furious customer. "
            "Better to temporarily show 'unavailable'."
        ),
        (
            "News feed (social network)",
            "AP",
            "A post that shows up 2 seconds late is invisible. "
            "A feed that does not load = user who leaves."
        ),
        (
            "Bank money transfer",
            "CP",
            "An incorrect balance = financial loss. Transactions must be ACID. "
            "A temporary outage is preferred over a double debit."
        ),
        (
            "DNS cache",
            "AP",
            "A slightly stale DNS record (TTL) is tolerable. "
            "An unavailable DNS = internet inaccessible for the users."
        ),
    ]

    for i, (system, choice, justification) in enumerate(choices, 1):
        print(f"\n  {i}. {system}")
        print(f"     Choice : {choice}")
        print(f"     Reason : {justification}")

    print(f"\n  Note : E-commerce inventory is a nuanced case. Some systems")
    print(f"  tolerate 0.1% overselling and reconcile afterwards (AP with compensation).")
    print(f"  The answer depends on the exact business context.")


# =============================================================================
# EASY -- Exercise 3 : The nines in practice
# =============================================================================

def easy_3_nines():
    """Solution for the SLA calculations."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : The nines in practice")
    print(SEPARATOR)

    uptime_pct = 99.95
    downtime_fraction = 1 - uptime_pct / 100  # 0.0005

    # 1. Allowed downtime
    seconds_per_month = 30 * 24 * 3600  # ~2.592M seconds
    seconds_per_day = 24 * 3600  # 86400 seconds
    dt_month = downtime_fraction * seconds_per_month
    dt_day = downtime_fraction * seconds_per_day

    print(f"\n  1. Allowed downtime for SLA {uptime_pct}%")
    print(f"     Per month = {downtime_fraction} * {seconds_per_month:,} = {dt_month:.0f}s = {dt_month / 60:.1f} min")
    print(f"     Per day = {downtime_fraction} * {seconds_per_day:,} = {dt_day:.1f}s")

    # 2. Weekly 15 min maintenance
    maintenance_per_month = 15 * 4  # 4 Sundays per month, in minutes
    maintenance_seconds = maintenance_per_month * 60
    print(f"\n  2. Maintenance 15 min/Sunday = {maintenance_per_month} min/month = {maintenance_seconds}s/month")
    print(f"     Downtime budget/month = {dt_month:.0f}s = {dt_month / 60:.1f} min")
    print(f"     Maintenance alone     = {maintenance_seconds}s = {maintenance_per_month} min")
    if maintenance_seconds > dt_month:
        print(f"     INCOMPATIBLE : maintenance ({maintenance_per_month} min) > budget ({dt_month / 60:.1f} min)")
    else:
        remaining = dt_month - maintenance_seconds
        print(f"     Compatible : {remaining:.0f}s = {remaining / 60:.1f} min of budget remains")

    # 3. A 2h incident this month
    incident_seconds = 2 * 3600  # 7200s
    remaining_after_incident = dt_month - incident_seconds
    print(f"\n  3. 2h incident = {incident_seconds}s")
    print(f"     Total budget  = {dt_month:.0f}s = {dt_month / 60:.1f} min")
    print(f"     After incident = {remaining_after_incident:.0f}s = {remaining_after_incident / 60:.1f} min")
    if remaining_after_incident < 0:
        print(f"     SLA BREACH : budget exceeded by {abs(remaining_after_incident):.0f}s")
    else:
        print(f"     {remaining_after_incident:.0f}s of budget remain for the month")

    # 4. Combined SLA
    sla_interne = 99.95 / 100
    sla_ext_1 = 99.99 / 100
    sla_ext_2 = 99.99 / 100
    # The combined SLA = product of the SLAs of all components on the critical path
    sla_combined = sla_interne * sla_ext_1 * sla_ext_2
    print(f"\n  4. Combined SLA")
    print(f"     = {sla_interne} * {sla_ext_1} * {sla_ext_2}")
    print(f"     = {sla_combined * 100:.4f}%")
    print(f"     The system's SLA is capped by the weakest link.")
    print(f"     Here, the internal service at 99.95% dominates.")


# =============================================================================
# MEDIUM -- Exercise 1 : Photo storage service
# =============================================================================

def medium_1_photos():
    """Solution for the photo service estimation."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Photo storage service")
    print(SEPARATOR)

    dau = 100_000_000
    uploaders_pct = 0.20
    photos_per_uploader = 2
    readers_pct = 0.80
    photos_per_feed_load = 30
    feed_loads_per_day = 5
    photo_size_bytes = 2 * 1024 * 1024  # 2 MB
    metadata_bytes = 500
    peak_factor = 3

    # 1. Write QPS (upload)
    uploaders = dau * uploaders_pct  # 20M
    total_uploads = uploaders * photos_per_uploader  # 40M/day
    qps_write_avg = total_uploads / 86_400
    qps_write_peak = qps_write_avg * peak_factor

    print(f"\n  1. Write QPS (upload)")
    print(f"     Uploaders = {dau:,} * {uploaders_pct} = {uploaders:,.0f}")
    print(f"     Uploads/day = {uploaders:,.0f} * {photos_per_uploader} = {total_uploads:,.0f}")
    print(f"     Average QPS = {total_uploads:,.0f} / 86400 = {qps_write_avg:,.0f} req/s")
    print(f"     Peak QPS    = {qps_write_avg:,.0f} * {peak_factor} = {qps_write_peak:,.0f} req/s")

    # 2. Read QPS (feed)
    readers = dau * readers_pct  # 80M
    total_reads = readers * photos_per_feed_load * feed_loads_per_day  # 12B/day
    qps_read_avg = total_reads / 86_400
    qps_read_peak = qps_read_avg * peak_factor

    print(f"\n  2. Read QPS (feed)")
    print(f"     Readers = {dau:,} * {readers_pct} = {readers:,.0f}")
    print(f"     Photos read/day = {readers:,.0f} * {photos_per_feed_load} * {feed_loads_per_day} = {total_reads:,.0f}")
    print(f"     Average QPS = {total_reads:,.0f} / 86400 = {qps_read_avg:,.0f} req/s")
    print(f"     Peak QPS    = {qps_read_avg:,.0f} * {peak_factor} = {qps_read_peak:,.0f} req/s")

    # 3. Read/write ratio
    ratio = qps_read_avg / qps_write_avg
    print(f"\n  3. Read/write ratio = {ratio:.0f}:1")
    print(f"     Heavily read-oriented system -> CDN + cache essential")

    # 4. Bandwidth
    bw_write_peak_gbps = qps_write_peak * photo_size_bytes * 8 / 1e9
    bw_read_peak_gbps = qps_read_peak * photo_size_bytes * 8 / 1e9
    # In reality, reads are often served by a CDN, not the origin servers
    print(f"\n  4. Bandwidth (peak)")
    print(f"     Write = {qps_write_peak:,.0f} * {photo_size_bytes / 1e6:.0f} MB = {bw_write_peak_gbps:,.0f} Gbps")
    print(f"     Read  = {qps_read_peak:,.0f} * {photo_size_bytes / 1e6:.0f} MB = {bw_read_peak_gbps:,.0f} Gbps")
    print(f"     (In practice, 90%+ of the reads are served by a CDN)")

    # 5. Storage
    storage_per_day_tb = total_uploads * photo_size_bytes / (1024**4)
    storage_1y_pb = storage_per_day_tb * 365 / 1024
    storage_5y_pb = storage_1y_pb * 5

    print(f"\n  5. Storage")
    print(f"     Per day = {total_uploads:,.0f} * {photo_size_bytes / 1e6:.0f} MB = {storage_per_day_tb:.1f} TB/day")
    print(f"     1 year  = {storage_per_day_tb:.1f} * 365 = {storage_per_day_tb * 365:.0f} TB = {storage_1y_pb:.1f} PB")
    print(f"     5 years = {storage_5y_pb:.1f} PB")

    # 6. Number of servers for reads
    server_capacity_rps = 10_000
    servers_needed = math.ceil(qps_read_peak / server_capacity_rps)
    print(f"\n  6. Servers for the read peak")
    print(f"     = {qps_read_peak:,.0f} / {server_capacity_rps:,} = {servers_needed} servers")
    print(f"     (+ 30% margin for redundancy = {math.ceil(servers_needed * 1.3)} servers)")

    # Bonus: bottleneck
    print(f"\n  BONUS : Main bottleneck")
    print(f"     Outbound bandwidth ({bw_read_peak_gbps:,.0f} Gbps) is astronomical.")
    print(f"     Without a CDN, no datacenter can serve this traffic.")
    print(f"     Solution : CDN (CloudFront, Fastly) for 95%+ of the reads.")
    print(f"     Storage ({storage_5y_pb:.0f} PB over 5 years) requires an object store (S3).")


# =============================================================================
# MEDIUM -- Exercise 2 : DB choice
# =============================================================================

def medium_2_db_choice():
    """Solution for the database choice."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Tradeoff Analysis -- DB Choice")
    print(SEPARATOR)

    choices = [
        {
            "donnee": "Product catalog",
            "db": "MongoDB (or Elasticsearch for search)",
            "consistency": "Eventual consistency",
            "justification": (
                "Flexible schema: products have attributes that vary by category "
                "(clothing = size/color, electronics = technical specs). "
                "MongoDB natively handles documents with different schemas. "
                "1000:1 read/write ratio -> read-optimized with read replicas."
            ),
            "alternatives_ecartees": (
                "PostgreSQL (JSONB): possible but less performant for queries "
                "on nested fields at high volume. "
                "Cassandra: overkill for 10M products, and ad hoc queries are limited."
            ),
            "risque": (
                "MongoDB can have performance issues on complex aggregations. "
                "Mitigation: Elasticsearch as a read replica for full-text search and facets."
            ),
        },
        {
            "donnee": "Orders",
            "db": "PostgreSQL",
            "consistency": "Strong consistency",
            "justification": (
                "ACID transactions required: an order involves a stock debit, "
                "creating an order line, reserving a payment. "
                "Referential integrity between order/lines/customer/product."
            ),
            "alternatives_ecartees": (
                "MongoDB: no native multi-document transactions before 4.0, and even "
                "after, less mature than PostgreSQL for complex transactions. "
                "DynamoDB: no joins, transactions limited to 25 items."
            ),
            "risque": (
                "50K orders/day = ~0.6 req/s, that's not a scale problem. "
                "Risk: if volume rises to 5M/day, PostgreSQL sharding is complex. "
                "Mitigation: Citus (distributed PostgreSQL) or migration to an event-sourcing pattern."
            ),
        },
        {
            "donnee": "User sessions",
            "db": "Redis",
            "consistency": "Eventual consistency (acceptable here: loss tolerable)",
            "justification": (
                "Sub-millisecond access required (every HTTP request checks the session). "
                "Native 30-min TTL with automatic expiration. "
                "Tolerable loss = no need for durable persistence."
            ),
            "alternatives_ecartees": (
                "PostgreSQL: too slow for per-HTTP-request access (even with an index). "
                "Memcached: possible but no persistence at all, "
                "no advanced data structures. "
                "DynamoDB: latency > Redis, higher cost for this pattern."
            ),
            "risque": (
                "Redis is single-threaded (io-threads helps but limited): a spike can saturate it. "
                "If Redis goes down, ALL sessions are lost. "
                "Mitigation: Redis Sentinel or Redis Cluster for HA. "
                "Fallback: transparent re-authentication if the session is lost."
            ),
        },
    ]

    for choice in choices:
        print(f"\n  {'-'*56}")
        print(f"  {choice['donnee']}")
        print(f"  {'-'*56}")
        print(f"  Chosen DB      : {choice['db']}")
        print(f"  Consistency    : {choice['consistency']}")
        print(f"  Justification  : {choice['justification']}")
        print(f"  Rejected       : {choice['alternatives_ecartees']}")
        print(f"  Risk + mitig   : {choice['risque']}")


# =============================================================================
# MEDIUM -- Exercise 3 : Latency Budget
# =============================================================================

def medium_3_latency_budget():
    """Solution for the latency decomposition."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Latency Budget")
    print(SEPARATOR)

    gateway = 5
    auth = 15
    postgres = 20
    redis_cache = 2
    cache_hit_rate = 0.80
    reco = 100
    price = 30

    # 1. Without cache, sequential
    seq_no_cache = gateway + auth + postgres + reco + price
    print(f"\n  1. Sequential, no cache")
    print(f"     Gateway({gateway}) + Auth({auth}) + PG({postgres}) + Reco({reco}) + Price({price})")
    print(f"     = {seq_no_cache} ms")
    print(f"     SLO 200ms -> EXCEEDED ({seq_no_cache} > 200)")

    # 2. With cache (80% hit rate), sequential
    # p99 = worst case = cache miss (because at p99, we fall into the 20% of misses)
    # In fact, at p99, the probability of a cache miss is what matters
    # But for a SINGLE call, p99 means: in the worst 1% of cases
    # The cache miss happens 20% of the time -> at p99, it's a cache miss
    db_latency_p99 = postgres  # Cache miss = we hit the DB
    seq_with_cache_p99 = gateway + auth + db_latency_p99 + reco + price
    seq_with_cache_avg = gateway + auth + (cache_hit_rate * redis_cache + (1 - cache_hit_rate) * postgres) + reco + price

    print(f"\n  2. Sequential, with cache (80% hit rate)")
    print(f"     Average DB latency = 0.8*{redis_cache} + 0.2*{postgres} = {cache_hit_rate * redis_cache + (1 - cache_hit_rate) * postgres:.1f} ms")
    print(f"     p50 (cache hit)  = {gateway} + {auth} + {redis_cache} + {reco} + {price} = {gateway + auth + redis_cache + reco + price} ms")
    print(f"     p99 (cache miss) = {gateway} + {auth} + {postgres} + {reco} + {price} = {seq_with_cache_p99} ms")
    print(f"     SLO 200ms -> p99 EXCEEDED ({seq_with_cache_p99} > 200)")

    # 3. Proposed optimizations
    print(f"\n  3. Optimizations")
    print(f"     a) Parallelize Product lookup, Reco, and Price (independent)")
    print(f"     b) Pre-fetch the recommendations (cache the results)")
    print(f"     c) Merge Auth into the Gateway (middleware, not a network call)")

    # 4. Redesign
    print(f"\n  4. Optimized redesign")
    print(f"     Client -> Gateway+Auth (5ms) -> [Parallel: DB/Cache + Reco + Price]")
    print(f"")
    print(f"     Phase 1 (sequential) : Gateway + Auth middleware = {gateway}ms")
    print(f"     Phase 2 (parallel)   : max(DB({postgres}), Reco({reco}), Price({price})) = {max(postgres, reco, price)}ms")

    # Auth is now a middleware in the Gateway (not a separate network call)
    # We save 15ms of network latency
    optimized_p99 = gateway + max(postgres, reco, price)
    optimized_p50 = gateway + max(redis_cache, reco, price)

    print(f"\n  5. Latency after optimization")
    print(f"     p50 (cache hit)  = {gateway} + max({redis_cache}, {reco}, {price}) = {gateway} + {max(redis_cache, reco, price)} = {optimized_p50} ms")
    print(f"     p99 (cache miss) = {gateway} + max({postgres}, {reco}, {price}) = {gateway} + {max(postgres, reco, price)} = {optimized_p99} ms")
    print(f"     SLO 200ms -> p99 = {optimized_p99}ms -> MET")

    print(f"\n  Summary of the gains :")
    print(f"     Before : {seq_no_cache}ms (sequential, no cache)")
    print(f"     After  : {optimized_p99}ms (parallel, p99 cache miss)")
    print(f"     Gain   : {seq_no_cache - optimized_p99}ms ({(1 - optimized_p99/seq_no_cache)*100:.0f}% reduction)")


# =============================================================================
# HARD -- Exercise 1 : View counter (sketched solution)
# =============================================================================

def hard_1_view_counter():
    """Sketched solution for the real-time view counter."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : View counter -- Estimation & Architecture")
    print(SEPARATOR)

    # --- Estimation ---
    views_per_day = 1_000_000_000
    qps_avg = views_per_day / 86_400
    qps_peak = qps_avg * 5  # Viral video = spike
    # We assume 500M videos in total, each counter = 8 bytes (int64) + 16 bytes video_id
    num_videos = 500_000_000
    counter_size = 8 + 16  # video_id (UUID) + count (int64)
    # Analytics: 1 row per video per hour = 500M * 24 * 365 * (8+16+8) = enormous
    # We pre-aggregate hourly
    analytics_row_size = 16 + 8 + 8  # video_id + timestamp + count

    print(f"\n  ESTIMATION")
    print(f"  {'-'*50}")
    print(f"  Views/day               : {views_per_day:,}")
    print(f"  Average QPS             : {qps_avg:,.0f}")
    print(f"  Peak QPS (x5, viral)    : {qps_peak:,.0f}")
    print(f"  Counters (RAM)          : {num_videos:,} * {counter_size} B = {num_videos * counter_size / 1e9:.1f} GB")
    print(f"  Analytics/year (hourly) : {num_videos:,} * 8760h * {analytics_row_size}B = {num_videos * 8760 * analytics_row_size / 1e12:.1f} TB")
    print(f"  Bandwidth (writes)      : {qps_peak * 100 * 8 / 1e6:,.0f} Mbps (100 bytes/req)")

    # --- Architecture ---
    print(f"\n  ARCHITECTURE")
    print(f"  {'-'*50}")
    print(f"""
  Write path (increment a view) :
    Client -> API Gateway -> Kafka (topic: view-events)
    Kafka -> View Counter Service -> Redis (atomic INCR)
    Kafka -> Analytics Aggregator -> ClickHouse (batch insert hourly)

  Read path (display the counter) :
    Client -> API Gateway -> Redis (GET counter)
    Fallback on miss -> ClickHouse (SUM of the aggregations)

  Analytics path (creator dashboard) :
    Client -> API Gateway -> ClickHouse (pre-aggregated tables)
""")

    print(f"  DB CHOICE")
    print(f"  {'-'*50}")
    print(f"  Real-time counters : Redis")
    print(f"    - Atomic INCR, sub-ms, perfect for ~{num_videos * counter_size / 1e9:.0f} GB of data")
    print(f"    - Consistency : eventual (acceptable, 1-2% tolerance)")
    print(f"  Analytics : ClickHouse")
    print(f"    - Optimized for columnar aggregations (SUM, COUNT per hour/day)")
    print(f"    - Consistency : eventual (batch processing, OK for analytics)")

    print(f"\n  TRADEOFFS")
    print(f"  {'-'*50}")
    print(f"  1. Kafka buffer instead of direct Redis writes")
    print(f"     + Absorbs traffic spikes (viral video)")
    print(f"     + Decouples the write path from the counter service")
    print(f"     - Adds 10-100ms of latency between the view and the counter")
    print(f"     Accepted because : 1-2% tolerance on the displayed counter")
    print(f"")
    print(f"  2. Redis (AP) instead of PostgreSQL (CP) for the counters")
    print(f"     + Sub-ms read latency")
    print(f"     + Atomic INCR without a global lock")
    print(f"     - No guaranteed durability (possible loss on crash)")
    print(f"     Accepted because : analytics in ClickHouse = source of truth")
    print(f"")
    print(f"  3. Hourly pre-aggregation instead of raw events")
    print(f"     + Storage 8760x smaller than raw events")
    print(f"     + Fast analytics queries")
    print(f"     - Loss of granularity (no per-second view)")
    print(f"     Accepted because : creators do not need granularity < 1h")

    print(f"\n  SLIs / SLOs")
    print(f"  {'-'*50}")
    print(f"  1. Write latency p99 < 10ms (SLI: write latency into Kafka)")
    print(f"  2. Counter staleness < 30s (SLI: age of the Redis counter vs Kafka offset)")
    print(f"  3. Availability > 99.9% (SLI: successful request rate)")


# =============================================================================
# HARD -- Exercise 2 : Cascading Failures (written analysis)
# =============================================================================

def hard_2_cascading_failures():
    """Solution for the cascading failures analysis."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Cascading Failures -- Analysis")
    print(SEPARATOR)

    print(f"""
  1. FAILURE PROPAGATION
  {'-'*50}
  Step 1 : Payment Service responds in 30s instead of 200ms.
  Step 2 : The Order Service threads calling Payment are blocked for 30s.
            Each order consumes a thread for 30s instead of 200ms (150x longer).
  Step 3 : The Order Service thread pool fills up. New requests -> timeout or rejection.
  Step 4 : The API servers (API-1/2/3) waiting on Order Service get blocked too.
            Their threads fill up progressively.
  Step 5 : The API servers can no longer handle ANY request, even those
            that do NOT use the Payment Service (User, Product).
  Step 6 : The Load Balancer marks the API servers as "unhealthy" -> total downtime.

  Order of impact : Payment -> Order Svc -> API servers -> the WHOLE system

  2. IMPACT CALCULATION
  {'-'*50}
  Thread pool per API server : 200 threads
  3 API servers = 600 threads total
  With Payment at 200ms : 1 thread handles 5 orders/second
  With Payment at 30s   : 1 thread handles 1 order/30s = 0.033/s

  Threads consumed by the orders :
  - Before : at 100 orders/s, we need 100/5 = 20 threads (10% of the pool)
  - After  : at 100 orders/s, we need 100 * 30s = 3000 threads -> IMPOSSIBLE

  Saturation : The 600 threads are consumed in 600 * (1/100 orders/s) = 6 seconds
  In ~6 seconds, all threads are blocked. The system is down.

  3. PROTECTION MECHANISMS
  {'-'*50}
  a) Circuit Breaker :
     HOW : After N consecutive failures/timeouts, the circuit "opens" and short-circuits
     the calls to Payment (returns an error immediately without calling the service).
     WHY : Frees the threads instantly. Prevents the pile-up.

  b) Timeout + Retry with exponential backoff :
     HOW : 2s timeout on the Payment call. On timeout, retry after 1s, 2s, 4s.
     WHY : Limits a thread's blocking time to 2s (not 30s).
     The backoff avoids overloading a service already in trouble.

  c) Bulkhead (thread pool isolation) :
     HOW : Dedicated thread pool for Payment calls (e.g. 30 threads out of 200).
     The 170 other threads stay available for User/Product/others.
     WHY : A slow service can only consume ITS threads, not the others.

  d) Decoupling queue (Kafka) :
     HOW : Order Service publishes an "order.created" event into Kafka.
     A separate consumer processes the payment asynchronously.
     WHY : The API thread is freed immediately. The payment is processed in the background.

  e) Fallback / Degraded mode :
     HOW : If Payment is unavailable, the order is created in "pending_payment" state.
     The user is told the payment will be processed within 5 min.
     WHY : The service stays "available" even if degraded. Better than nothing.

  4. RESILIENT REDESIGN
  {'-'*50}
  Redesigned order flow :
    Client -> API -> Order Service : creates the order (status=pending) -> responds to the client
    Order Service -> Kafka : publishes "order.created"
    Payment Consumer <- Kafka : processes the payment asynchronously
    Payment Consumer -> Kafka : publishes "payment.confirmed" or "payment.failed"
    Order Service <- Kafka : updates the order -> notifies the client (websocket/push)

  Consistency model : eventual consistency between creation and confirmation.
  The order goes through the states : pending -> confirmed / failed.
  This is the Saga pattern.

  Informing the user : "Your order has been recorded. Payment confirmation
  within a few seconds." + push notification when the payment is processed.

  5. COMPOSED SLA
  {'-'*50}""")

    services = [0.9995, 0.9995, 0.9995, 0.995]  # 3 internal + Payment
    sla_sync = 1.0
    for s in services:
        sla_sync *= s
    print(f"  Synchronous SLA = 99.95% * 99.95% * 99.95% * 99.5% = {sla_sync * 100:.3f}%")
    print(f"  Downtime/year = {(1 - sla_sync) * 365.25 * 24:.1f} hours")

    print(f"""
  To reach > 99.9% despite Payment at 99.5% :
  - Decouple Payment via Kafka (async)
  - The synchronous flow no longer depends on Payment :
    Sync SLA = 99.95% * 99.95% * 99.95% = {0.9995**3 * 100:.3f}%
  - Payment failures are retried automatically by the Kafka consumer
  - With retry + dead letter queue, Payment's effective success rate rises to ~99.99%
  - Effective SLA of the full flow > 99.9%""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Runs all the solutions."""
    print("\n" + "#" * 60)
    print("#  SOLUTIONS -- DAY 1 : FUNDAMENTAL PRINCIPLES")
    print("#" * 60)

    # Easy
    easy_1_notifications()
    easy_2_cp_ap()
    easy_3_nines()

    # Medium
    medium_1_photos()
    medium_2_db_choice()
    medium_3_latency_budget()

    # Hard
    hard_1_view_counter()
    hard_2_cascading_failures()

    print("\n" + "#" * 60)
    print("#  END OF SOLUTIONS")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
