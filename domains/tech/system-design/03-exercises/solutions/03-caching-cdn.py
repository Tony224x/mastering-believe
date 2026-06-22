"""
Solutions -- Day 3 Exercises: Caching & CDN

This file contains the worked solutions for the Easy, Medium, and Hard exercises.
Each solution shows the reasoning step by step.

Usage:
    python 03-caching-cdn.py
"""

import math

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercise 1 : Which caching strategy?
# =============================================================================

def easy_1_cache_strategy():
    """Solution for the caching strategy choice per use case."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Which caching strategy?")
    print(SEPARATOR)

    choices = [
        (
            "User profiles (50K reads/sec, 10 writes/sec)",
            "Cache-Aside",
            "Read-heavy (5000:1 ratio). Cache-aside is the default choice: "
            "we only cache the profiles that are actually requested. "
            "10 writes/sec = negligible invalidation. 5-15 min TTL "
            "to tolerate occasional staleness."
        ),
        (
            "Video view counter",
            "Write-Behind (Write-Back)",
            "Write-heavy: each view = 1 increment. Write-behind accumulates "
            "the increments in memory (Redis INCR) and batch-flushes to "
            "the DB every 5-10 seconds. We tolerate losing a few "
            "views on crash (not critical). Tradeoff: weak consistency "
            "but maximum performance."
        ),
        (
            "E-commerce stock (overselling is costly)",
            "Write-Through",
            "Consistency is critical: showing stock > 0 while the "
            "real stock is 0 = overselling = financial loss. Write-through "
            "guarantees the cache and the DB are always in sync. "
            "The write latency cost is acceptable because stock updates "
            "are less frequent than reads."
        ),
        (
            "Global config for a K8s cluster",
            "Read-Through + in-process L1 cache",
            "Identical data for all pods, rarely changes. "
            "In-process L1 cache (local dict) with a 30s TTL avoids "
            "even the network round trip to Redis. Read-through because "
            "the app does not need to know the source (DB vs Redis)."
        ),
        (
            "Analytics dashboard (24h aggregation)",
            "Cache-Aside with a long TTL",
            "The aggregations are expensive (scanning millions of rows). "
            "The result is identical for all users. Cache-aside "
            "with a 5-15 min TTL. Or a materialized view in the DB + cache-aside. "
            "A few minutes of staleness is acceptable on a dashboard."
        ),
        (
            "User sessions (login/logout)",
            "Write-Through or Cache-Aside",
            "The session must be up to date (logout = immediate deletion). "
            "Write-through for writes (login creates the session in "
            "cache + DB atomically). Cache-aside for reads. "
            "TTL = session duration (30 min). Redis is the natural choice "
            "because sessions are simple key-values with native TTL."
        ),
    ]

    for i, (system, choice, justification) in enumerate(choices, 1):
        print(f"\n  {i}. {system}")
        print(f"     Strategy : {choice}")
        print(f"     Reason : {justification}")


# =============================================================================
# EASY -- Exercise 2 : Cache-Control headers
# =============================================================================

def easy_2_cache_control():
    """Solution for the Cache-Control headers per content type."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : Cache-Control headers")
    print(SEPARATOR)

    headers = [
        (
            "app.a3f2b1c.js (JS with a hash in the name)",
            "Cache-Control: public, max-age=31536000, immutable",
            "public : any cache (CDN, proxy, browser) may store it. "
            "max-age=31536000 : 1 year (practical maximum). "
            "immutable : the browser will never revalidate. "
            "The hash in the name guarantees any change = new URL. "
            "This is the standard pattern for bundled assets (webpack, vite)."
        ),
        (
            "/api/me (logged-in user profile)",
            "Cache-Control: private, no-cache",
            "private : only the browser may cache (not the CDN, since the content "
            "is personalized). no-cache : the browser must revalidate every "
            "time (If-None-Match with ETag). This saves bandwidth "
            "(304 if no change) while guaranteeing freshness."
        ),
        (
            "/api/products (catalog updated every hour)",
            "Cache-Control: public, s-maxage=3600, max-age=300, stale-while-revalidate=60",
            "public : identical for all users. "
            "s-maxage=3600 : the CDN caches for 1h (in sync with the updates). "
            "max-age=300 : the browser caches for 5 min (more frequent client-side revalidation). "
            "stale-while-revalidate=60 : serve stale for 60s while revalidating in the background."
        ),
        (
            "/login (login HTML page)",
            "Cache-Control: no-cache",
            "no-cache : the page may be cached but must be revalidated. "
            "We want the browser to revalidate because the HTML can change "
            "(new build, CSRF token). The CDN can also cache "
            "with revalidation (ETag). Not no-store because the page is "
            "not sensitive in itself."
        ),
        (
            "Invoice PDF (authenticated user)",
            "Cache-Control: private, no-store",
            "private : do not cache on the CDN (confidential document). "
            "no-store : do not store at all (even on the local disk). "
            "Invoices contain sensitive financial data. "
            "Acceptable alternative : private, no-cache if we want to allow "
            "browser caching with revalidation."
        ),
    ]

    for i, (resource, header, justification) in enumerate(headers, 1):
        print(f"\n  {i}. {resource}")
        print(f"     Header : {header}")
        print(f"     Reason : {justification}")


# =============================================================================
# EASY -- Exercise 3 : Redis memory sizing
# =============================================================================

def easy_3_redis_sizing():
    """Solution for the Redis memory sizing."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : Redis memory sizing")
    print(SEPARATOR)

    # Data
    uuid_bytes = 36
    role_bytes = 10
    token_bytes = 64
    timestamp_bytes = 8
    preferences_bytes = 200
    redis_overhead = 2.5  # Redis overhead factor
    concurrent_sessions = 2_000_000
    masters = 3
    replicas_per_master = 1

    # 1. Raw size of a session
    raw_size = uuid_bytes + role_bytes + token_bytes + timestamp_bytes + preferences_bytes
    print(f"\n  1. Raw size of a session :")
    print(f"     user_id (UUID)    : {uuid_bytes} bytes")
    print(f"     role              : {role_bytes} bytes")
    print(f"     token             : {token_bytes} bytes")
    print(f"     last_seen         : {timestamp_bytes} bytes")
    print(f"     preferences (JSON): {preferences_bytes} bytes")
    print(f"     Raw total         : {raw_size} bytes")

    # 2. Total memory with Redis overhead
    raw_total = concurrent_sessions * raw_size
    total_with_overhead = raw_total * redis_overhead
    raw_total_gb = raw_total / (1024 ** 3)
    total_gb = total_with_overhead / (1024 ** 3)
    print(f"\n  2. Total memory for {concurrent_sessions:,} sessions :")
    print(f"     Raw               : {concurrent_sessions:,} * {raw_size} = {raw_total:,} bytes = {raw_total_gb:.2f} GB")
    print(f"     With {redis_overhead}x overhead : {total_with_overhead:,.0f} bytes = {total_gb:.2f} GB")

    # 3. RAM per master node
    ram_per_master = total_gb / masters
    print(f"\n  3. RAM per master node ({masters} masters) :")
    print(f"     {total_gb:.2f} GB / {masters} = {ram_per_master:.2f} GB per master")
    print(f"     Recommendation : plan for 2x as margin -> {ram_per_master * 2:.1f} GB per master")

    # 4. Total with replicas
    total_nodes = masters * (1 + replicas_per_master)
    total_ram = total_gb * (1 + replicas_per_master)
    print(f"\n  4. With replicas ({replicas_per_master} replica per master) :")
    print(f"     Total nodes       : {masters} masters + {masters * replicas_per_master} replicas = {total_nodes}")
    print(f"     Total cluster RAM : {total_gb:.2f} * {1 + replicas_per_master} = {total_ram:.2f} GB")
    print(f"     The replicas are essential for HA (if a master goes down,")
    print(f"     its replica is promoted automatically)")

    # 5. Eviction policy
    print(f"\n  5. Recommended maxmemory-policy :")
    print(f"     volatile-ttl")
    print(f"     Reason : the sessions all have a TTL (30 min). volatile-ttl")
    print(f"     evicts the keys with the shortest TTL first, which makes")
    print(f"     sense for sessions (the oldest expire first).")
    print(f"     Alternative : volatile-lru if the sessions have identical TTLs")
    print(f"     (evicts the least recently accessed).")
    print(f"     Do NOT use allkeys-* because sessions MUST have a TTL.")


# =============================================================================
# MEDIUM -- Exercise 1 : Cache for a social feed
# =============================================================================

def medium_1_social_feed():
    """Solution for the caching layer of a social feed."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Cache for a social feed")
    print(SEPARATOR)

    users = 200_000_000
    avg_following = 200
    posts_per_min = 500_000
    feed_size = 50  # Posts per feed

    print(f"\n  1. Cache type :")
    print(f"     Redis (distributed cache) — NOT a CDN.")
    print(f"     Why not a CDN : the feed is personalized (each user sees")
    print(f"     different content). A CDN caches identical content for")
    print(f"     all users in a given region. Here, the cache must be")
    print(f"     per user -> only Redis can do that efficiently.")

    print(f"\n  2. Redis structure : Sorted Set")
    print(f"     Key : feed:{{user_id}}")
    print(f"     Score : timestamp of the post")
    print(f"     Member : post_id (reference, not the full content)")
    print(f"     Command : ZREVRANGE feed:{{user_id}} 0 49 -> the latest 50 posts")
    print(f"     The post details live in a separate Hash : post:{{post_id}}")

    print(f"\n  3. Size management :")
    print(f"     ZREMRANGEBYRANK feed:{{user_id}} 0 -501 after each ZADD")
    print(f"     -> keeps only the latest 500 posts in cache")
    print(f"     (10 pages of 50). Beyond that, pagination goes to the DB.")

    print(f"\n  4. Fanout-on-write vs Fanout-on-read :")
    print(f"     Fanout-on-write : when a user posts, we write into the cached")
    print(f"     feed of EACH follower. For a user with 200 followers,")
    print(f"     that's 200 Redis ZADDs. It's fast and the read is O(1).")
    print(f"     Fanout-on-read : when a user opens their feed, we aggregate the")
    print(f"     posts of all the accounts they follow. Fewer writes, but")
    print(f"     the read is slow (200 queries to merge 200 timelines).")

    print(f"\n     Hybrid solution (Twitter approach) :")
    print(f"     - Normal users (< 10K followers) : fanout-on-write")
    print(f"     - Celebrities (> 10K followers) : fanout-on-read")
    print(f"     When a user opens their feed : read the precomputed feed + merge")
    print(f"     the posts of the celebrities they follow.")

    print(f"\n  5. Celebrities :")
    print(f"     A celebrity has 50M followers. Fanout-on-write = 50M ZADDs per post.")
    print(f"     At {posts_per_min:,} posts/min, that's unsustainable.")
    print(f"     Solution : do NOT precompute the feed for celebrity followers.")
    print(f"     At read time, merge the precomputed feed with the celebrities' latest posts.")

    # 6. Memory estimation
    post_id_size = 20  # bytes (compact UUID or snowflake ID)
    score_size = 8     # bytes (double timestamp)
    redis_entry_overhead = 40  # bytes of overhead per entry in a sorted set
    entries_per_feed = 500
    active_users_pct = 0.3  # 30% have a cached feed (the recently active)
    active_users = int(users * active_users_pct)

    feed_size_bytes = entries_per_feed * (post_id_size + score_size + redis_entry_overhead)
    total_bytes = active_users * feed_size_bytes
    total_tb = total_bytes / (1024 ** 4)

    print(f"\n  6. Memory estimation :")
    print(f"     Size per entry : {post_id_size} + {score_size} + {redis_entry_overhead} = {post_id_size + score_size + redis_entry_overhead} bytes")
    print(f"     Size per feed ({entries_per_feed} entries) : {feed_size_bytes:,} bytes = {feed_size_bytes/1024:.1f} KB")
    print(f"     Active users with a cached feed (30%) : {active_users:,}")
    print(f"     Total : {active_users:,} * {feed_size_bytes:,} = {total_bytes:,} bytes = {total_tb:.1f} TB")
    print(f"     + the Hashes of the posts themselves (~2-5 TB more)")

    print(f"\n  7. Invalidation :")
    print(f"     Event-driven : when a post is published, a Kafka worker")
    print(f"     performs the fanout (ZADD into the followers' feeds).")
    print(f"     24h TTL on the feeds : safety net, avoids ghost")
    print(f"     feeds of inactive users. No need for a short TTL")
    print(f"     because the event-driven flow guarantees freshness.")


# =============================================================================
# MEDIUM -- Exercise 2 : Diagnosing a high cache miss rate
# =============================================================================

def medium_2_cache_diagnosis():
    """Solution for diagnosing a high cache miss rate."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Diagnosing a high cache miss rate")
    print(SEPARATOR)

    num_keys = 15_000_000
    avg_size = 500  # bytes
    redis_overhead = 2.5
    current_ram_gb = 8

    print(f"\n  1. Root cause :")
    print(f"     The cache is UNDERSIZED. 15M keys * 500 bytes * 2.5 overhead")
    print(f"     = ~17.5 GB needed, but only 8 GB available.")
    print(f"     Redis evicts constantly (12K evictions/sec) to make room.")
    print(f"     Moreover, the search: keys (25% of the cache = 3.75M keys) have a")
    print(f"     re-access rate < 5%, meaning they waste ~4.4 GB of cache")
    print(f"     on data that is rarely re-read.")

    # 2. Ideal memory calculation
    raw_total = num_keys * avg_size
    ideal_total = raw_total * redis_overhead
    ideal_gb = ideal_total / (1024 ** 3)
    print(f"\n  2. Ideal memory :")
    print(f"     {num_keys:,} * {avg_size} bytes = {raw_total:,} bytes = {raw_total/(1024**3):.1f} GB raw")
    print(f"     With Redis overhead ({redis_overhead}x) : {ideal_gb:.1f} GB")
    print(f"     Deficit : {ideal_gb:.1f} - {current_ram_gb} = {ideal_gb - current_ram_gb:.1f} GB missing")

    # 3. Actions by impact
    print(f"\n  3. Actions ranked by impact :")

    search_keys = int(num_keys * 0.25)
    search_memory = search_keys * avg_size * redis_overhead / (1024 ** 3)

    print(f"\n     Action 1 (highest impact, zero cost) : Reduce the search: keys")
    print(f"     The {search_keys:,} search: keys occupy ~{search_memory:.1f} GB")
    print(f"     with a 5% re-access rate. Options :")
    print(f"     - Reduce the TTL from 1h to 5 min")
    print(f"     - Or only cache the most frequent searches (top 10%)")
    print(f"     Estimated impact : frees ~{search_memory * 0.8:.1f} GB -> hit rate +20-25%")

    print(f"\n     Action 2 (medium impact, moderate cost) : Increase the Redis RAM")
    print(f"     Go from 8 GB to 20 GB (or 2 * 10 GB in a cluster)")
    print(f"     Cost : ~$200-400/month extra")
    print(f"     Estimated impact : hit rate +15-20% (fewer evictions)")

    print(f"\n     Action 3 (complementary impact) : Add an in-process L1 cache")
    print(f"     For the hottest keys (most accessed session: and product:)")
    print(f"     100 MB local cache per instance with a 30s TTL")
    print(f"     Estimated impact : hit rate +5-10% and a 30% reduction of the Redis load")

    # 4. Estimated improvement
    print(f"\n  4. Estimated hit rate improvement :")
    print(f"     Current : 45%")
    print(f"     After action 1 : ~70% (+25%)")
    print(f"     After actions 1+2 : ~88% (+18%)")
    print(f"     After actions 1+2+3 : ~92% (+4%)")

    # 5. Decision
    print(f"\n  5. Increase the RAM OR optimize ?")
    print(f"     FIRST optimize (action 1) : free, immediate impact.")
    print(f"     THEN increase the RAM if the hit rate stays < 85%.")
    print(f"     Justification : the search: keys waste {search_memory:.1f} GB on")
    print(f"     data accessed at 5%. Removing them frees up room for the")
    print(f"     product: and session: keys which have a much higher re-access rate.")

    print(f"\n  6. Monitoring dashboard (5 metrics) :")
    metrics = [
        ("Hit rate (%)", "> 85%", "< 70%", "The #1 indicator of cache health"),
        ("Eviction rate (/sec)", "< 100/s", "> 1K/s", "Signal that the cache is too small"),
        ("Memory usage (%)", "< 80%", "> 95%", "Margin before massive evictions"),
        ("Latency p99 (ms)", "< 2ms", "> 10ms", "Degradation = overload or network"),
        ("Key count (total)", "stable +/-10%", "sharp drop", "A drop = massive TTL or FLUSHALL"),
    ]
    for name, ok, alert, why in metrics:
        print(f"     - {name} : OK={ok}, ALERT={alert} ({why})")


# =============================================================================
# MEDIUM -- Exercise 3 : Multi-region CDN strategy
# =============================================================================

def medium_3_cdn_strategy():
    """Solution for the multi-region CDN strategy."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Multi-region CDN strategy")
    print(SEPARATOR)

    print(f"\n  1. CDN architecture :")
    print(f"     Provider : CloudFront (native S3 integration) or Cloudflare")
    print(f"     Levels : 2 levels (edge + origin shield)")
    print(f"     - Edge : POPs in each region (Paris, Frankfurt, NYC, Tokyo...)")
    print(f"     - Origin Shield : 1 intermediate cache per region (reduces origin load)")
    print(f"     - Origin : the 3 API regions (US-East, EU-West, APAC)")
    print(f"     Routes the requests to the closest origin (latency-based routing)")

    print(f"\n  2. Securing the documents :")
    print(f"     Signed URLs (CloudFront) or Signed Cookies (for a whole domain)")
    print(f"     Flow : the API generates a signed URL with an expiration (15 min)")
    print(f"     The CDN verifies the signature before serving the document.")
    print(f"     If the signature is invalid or expired -> 403 Forbidden.")
    print(f"     Advantage : the document stays in the CDN cache, but only users")
    print(f"     with a valid signed URL can access it.")

    print(f"\n  3. Document invalidation (< 5 min) :")
    print(f"     Option A : short TTL (s-maxage=300) -> the CDN revalidates every 5 min")
    print(f"     Option B : Purge API (CloudFront CreateInvalidation) triggered by the update")
    print(f"     Recommendation : 300s TTL + purge API for urgent updates.")
    print(f"     The purge API takes 30-60s on CloudFront (global propagation).")

    print(f"\n  4. Cache headers per type :")
    headers = [
        ("app.hash.js (React)", "Cache-Control: public, max-age=31536000, immutable",
         "Hash in the name = versioned. Cache for 1 year."),
        ("/api/documents/list", "Cache-Control: private, no-cache",
         "Personalized (the user's document list). Mandatory revalidation."),
        ("/documents/{id}/download", "Cache-Control: private, no-cache + Signed URL",
         "Confidential document. The CDN caches the file but requires a signed URL."),
        ("index.html", "Cache-Control: no-cache",
         "Always revalidate to point to the latest build (hash in the <script> tags)."),
    ]
    for resource, header, reason in headers:
        print(f"     {resource}")
        print(f"       {header}")
        print(f"       -> {reason}")

    print(f"\n  5. Measuring the hit rate :")
    print(f"     CloudFront provides native metrics :")
    print(f"     - Hit rate per distribution (global)")
    print(f"     - Hit rate per path pattern (/static/*, /api/*, /documents/*)")
    print(f"     - Bandwidth savings = bytes served from edge / total bytes")
    print(f"     Targets :")
    print(f"     - Static assets : > 95% hit rate")
    print(f"     - Documents : > 60% hit rate (re-downloads)")
    print(f"     - API : < 10% hit rate (personalized, normal)")
    print(f"     If the asset hit rate is < 90%, check the headers.")

    # 6. Cost estimation
    print(f"\n  6. CDN cost estimation (CloudFront) :")
    static_gb = 500       # GB/month of static assets
    docs_gb = 2000        # GB/month of PDF documents
    api_gb = 100          # GB/month of API responses
    total_gb = static_gb + docs_gb + api_gb
    cost_per_gb = 0.085   # $/GB for the first 10 TB
    requests_millions = 50  # Millions of requests/month
    request_cost_per_10k = 0.01  # $/10K HTTPS requests

    bandwidth_cost = total_gb * cost_per_gb
    request_cost = requests_millions * 1000 * request_cost_per_10k
    total_cost = bandwidth_cost + request_cost

    print(f"     Bandwidth :")
    print(f"       Static assets    : {static_gb} GB/month")
    print(f"       PDF documents    : {docs_gb} GB/month")
    print(f"       API responses    : {api_gb} GB/month")
    print(f"       Total            : {total_gb} GB/month")
    print(f"       Bandwidth cost   : {total_gb} * ${cost_per_gb} = ${bandwidth_cost:.0f}/month")
    print(f"     Requests :")
    print(f"       {requests_millions}M requests * ${request_cost_per_10k}/10K = ${request_cost:.0f}/month")
    print(f"     Total CDN          : ~${total_cost:.0f}/month")
    print(f"     (For a mid-size SaaS, $200-500/month is typical)")


# =============================================================================
# HARD -- Exercise 1 : Black Friday e-commerce cache
# =============================================================================

def hard_1_black_friday():
    """Solution for the Black Friday caching layer."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Black Friday e-commerce cache")
    print(SEPARATOR)

    products = 5_000_000
    product_size_kb = 2
    normal_reads = 50_000
    peak_reads = 1_000_000
    peak_writes = 40_000
    flash_deals = 100

    print(f"\n  1. Multi-tier architecture :")
    print(f"     L1 : In-process cache (Caffeine/dict)")
    print(f"       - Content : flash deal prices, promo config, feature flags")
    print(f"       - TTL : 5-10 seconds (prices must not be stale > 10s)")
    print(f"       - Size : ~50 MB per instance")
    print(f"       - Advantage : 0 network latency, absorbs the ultra-hot keys")
    print(f"     L2 : Redis Cluster")
    print(f"       - Content : full catalog, sessions, carts")
    print(f"       - TTL : 5 min (catalog), 30 min (sessions)")
    print(f"       - The price has a 10s TTL OR event-driven invalidation")
    print(f"     L3 : CDN (CloudFront)")
    print(f"       - Content : static assets, product images")
    print(f"       - TTL : 1 year (versioned by hash)")

    # 2. Redis sizing
    catalog_gb = (products * product_size_kb * 1024) / (1024 ** 3)
    catalog_with_overhead = catalog_gb * 2.5
    # Redis can do ~100K ops/sec per node
    ops_per_node = 100_000
    nodes_for_throughput = math.ceil(peak_reads / ops_per_node)

    print(f"\n  2. Redis sizing :")
    print(f"     Catalog memory : {products:,} * {product_size_kb} KB = {catalog_gb:.1f} GB raw")
    print(f"     With Redis overhead (2.5x) : {catalog_with_overhead:.1f} GB")
    print(f"     + sessions + carts : ~5-10 GB more")
    print(f"     Total memory : ~{catalog_with_overhead + 10:.0f} GB")
    print(f"     Nodes for {peak_reads:,} reads/sec : {peak_reads:,} / {ops_per_node:,} = {nodes_for_throughput} nodes")
    print(f"     With replicas (1 per master) : {nodes_for_throughput * 2} nodes")
    print(f"     Shard key : hash(product_id) % 16384 (standard Redis Cluster)")

    # 3. Flash deals
    print(f"\n  3. Flash deals :")
    print(f"     Problem : at 14:00 exactly, 100 products go on sale.")
    print(f"     Millions of users load the page at the same time.")
    print(f"     -> Cache stampede on 100 keys simultaneously.")
    print(f"     Solution :")
    print(f"     a) Cache warming 5 min BEFORE the deal (load the 100 products into")
    print(f"        L1 + L2 with the new price)")
    print(f"     b) stale-while-revalidate : serve the existing cache during the rebuild")
    print(f"     c) Per-key mutex for the residual misses")
    print(f"\n     Stock reservation with Redis :")
    print(f"     SET stock:product_123 500   # Initialize the stock")
    print(f"     DECR stock:product_123      # Atomic! Returns the remaining stock")
    print(f"     If DECR returns < 0 : oversell -> INCR to undo + reject the order")
    print(f"     DECR is atomic in Redis -> no race condition even with 100K requests")

    # 4. Cache warming
    print(f"\n  4. Cache warming :")
    print(f"     Data to preload :")
    print(f"     - 100 flash deals (price, description, stock)")
    print(f"     - Top 10K most viewed products (historical)")
    print(f"     - Global config (promo thresholds, feature flags)")
    print(f"     How to avoid overloading the DB :")
    print(f"     - Rate limiter : 1000 queries/sec max during the warm")
    print(f"     - Start 30 min before the peak")
    print(f"     - Use a read replica for the warm (not the master)")
    print(f"     Timing : T-30min (bulk warm) -> T-5min (refresh flash deals) -> T-0 (go)")

    # 5. Resilience
    print(f"\n  5. Resilience :")
    print(f"     If Redis goes down during Black Friday :")
    print(f"     a) Circuit breaker : detects Redis timeouts (> 5ms)")
    print(f"        After 10 consecutive failures -> open the circuit")
    print(f"     b) Fallback : serve the L1 cache data (stale but available)")
    print(f"     c) Rate limiter on the DB : max 5K queries/sec (vs 50K without cache)")
    print(f"        -> degraded but does not kill the DB")
    print(f"     d) Virtual waiting room page if the load exceeds capacity")
    print(f"     Detection : alert if hit rate < 60% for 30 seconds")

    # 6. Monitoring
    print(f"\n  6. Monitoring (8 metrics) :")
    metrics = [
        ("Redis hit rate", "< 75%", "Underperforming cache"),
        ("Redis eviction rate", "> 5K/s", "Cache too small for the workload"),
        ("Redis latency p99", "> 5ms", "Redis overloaded or network"),
        ("Redis memory usage %", "> 85%", "Risk of imminent eviction"),
        ("DB connections active", "> 160/200", "Pool nearly saturated"),
        ("DB CPU %", "> 70%", "DB overloaded (cache ineffective)"),
        ("API error rate (5xx)", "> 0.1%", "Degradation visible to users"),
        ("Stock DECR rate/sec", "> initial stock", "Potential overselling"),
    ]
    for name, threshold, meaning in metrics:
        print(f"     - {name} : alert if {threshold} ({meaning})")

    # Budget
    print(f"\n  Budget estimation :")
    redis_nodes = nodes_for_throughput * 2  # With replicas
    redis_cost_per_node = 800  # $/month for an r6g.xlarge (32 GB RAM)
    redis_cost = redis_nodes * redis_cost_per_node
    cdn_cost = 8000  # $/month for the Black Friday peak
    total = redis_cost + cdn_cost
    print(f"     Redis : {redis_nodes} nodes * ${redis_cost_per_node}/month = ${redis_cost:,}/month")
    print(f"     CDN   : ~${cdn_cost:,}/month (Black Friday peak)")
    print(f"     Total : ~${total:,}/month (within the $50K budget)")


# =============================================================================
# HARD -- Exercise 2 : Cache incident post-mortem
# =============================================================================

def hard_2_postmortem():
    """Solution for the cache incident post-mortem."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Post-mortem — The cache that broke checkout")
    print(SEPARATOR)

    print(f"\n  1. Full causal chain :")
    chain = [
        ("PROCESS", "Marketing script writes directly to the DB",
         "Missing guardrail : every write must go through the application service (API)"),
        ("ARCHITECTURE", "The cache is not invalidated because the script bypasses the service",
         "Missing guardrail : CDC (Change Data Capture) to capture direct DB writes"),
        ("ARCHITECTURE", "Stale cache for 5 min (TTL) with incorrect prices",
         "Missing guardrail : a 5 min TTL is too long for prices (should be 10-30s)"),
        ("PROCESS", "FLUSHALL as an emergency reaction",
         "Missing guardrail : incident runbook that forbids FLUSHALL in production"),
        ("ARCHITECTURE", "Massive cache stampede (15M simultaneous cache misses)",
         "Missing guardrail : anti-stampede mechanism (mutex, stale-while-revalidate)"),
        ("ARCHITECTURE", "DB saturated (100% CPU, connections exhausted)",
         "Missing guardrail : circuit breaker + rate limiter between app and DB"),
        ("MONITORING", "Cascading 500 errors, site down 35 min",
         "Missing guardrail : graceful degradation (serve stale rather than 500s)"),
    ]
    for category, cause, guardrail in chain:
        print(f"     [{category}] {cause}")
        print(f"       -> {guardrail}")

    print(f"\n  2. Was the FLUSHALL the right decision ?")
    print(f"     NO. It's the mistake that turned a minor problem")
    print(f"     (stale prices for < 5 min) into a major incident (35 min of downtime).")
    print(f"\n     Alternatives to FLUSHALL :")
    print(f"     a) Invalidate ONLY the 500 affected keys :")
    print(f"        for product_id in promo_products:")
    print(f"            redis.delete(f'product:price:{{product_id}}')")
    print(f"        Spread over 5 seconds (100 DEL/sec) to avoid the stampede")
    print(f"     b) Wait for the TTL to expire naturally (< 5 min)")
    print(f"        The stale price is a problem, but 35 min of downtime is worse")
    print(f"     c) Reduce the TTL to 10s for the price keys :")
    print(f"        redis.expire(f'product:price:{{id}}', 10)")
    print(f"        The keys expire naturally within 10s without a stampede")

    print(f"\n     If FLUSHALL was absolutely necessary :")
    print(f"     1. Enable the circuit breaker on the DB BEFORE the flush")
    print(f"     2. Limit the refill to 1000 queries/sec (rate limiter)")
    print(f"     3. Use stale-while-revalidate to serve the old cache")
    print(f"     4. Flush in chunks (SCAN + DELETE) instead of FLUSHALL")

    print(f"\n  3. Fixed architecture :")
    print(f"     a) Prevent direct DB writes :")
    print(f"        - Principle : every price change goes through the API (no direct SQL)")
    print(f"        - Enforcement : the script's DB user only has SELECT, not UPDATE")
    print(f"        - Backup : CDC (Debezium) watches the prices table in real time")
    print(f"     b) 3 mechanisms against price inconsistency :")
    print(f"        1. Short TTL (10s) on the price keys")
    print(f"        2. CDC via Debezium : captures DB writes -> invalidates the cache")
    print(f"        3. Double-check at checkout : re-read the price in the DB before creating the order")
    print(f"     c) Safe cache invalidation pattern :")
    print(f"        1. Invalidate the keys in batches (100/sec)")
    print(f"        2. Use the lock/mutex for the rebuild")
    print(f"        3. stale-while-revalidate : serve the old data during the rebuild")

    print(f"\n  4. Event-driven architecture for the prices :")
    print(f"     Option 1 : CDC with Debezium")
    print(f"       PostgreSQL WAL -> Debezium -> Kafka topic 'price-changes'")
    print(f"       -> Consumer invalidates the corresponding Redis keys")
    print(f"       Advantage : captures EVERYTHING, even direct SQL scripts")
    print(f"     Option 2 : Outbox pattern")
    print(f"       The API writes into the prices table AND into an outbox table")
    print(f"       A worker polls the outbox table and invalidates the cache")
    print(f"       Advantage : transactional (write + event in the same TX)")
    print(f"     Recommendation : CDC (Debezium) because it also captures direct scripts")

    print(f"\n  5. Resilience patterns :")
    print(f"     Circuit breaker (concrete thresholds) :")
    print(f"       - CLOSED (normal) : < 160 active DB connections")
    print(f"       - OPEN (fallback) : > 160 connections for 5 seconds")
    print(f"       - Fallback : serve the stale cache + header X-Cache-Stale: true")
    print(f"       - HALF-OPEN : after 30s, let 10% of the traffic through to test")
    print(f"     Graceful degradation :")
    print(f"       - Serve the stale prices with an 'indicative price' banner")
    print(f"       - Verify the real price at checkout (directly in the DB)")
    print(f"       - Better a stale price than a 500 error")

    print(f"\n     Runbook (10 steps) :")
    steps = [
        "Do NOT FLUSHALL (never in production without protection)",
        "Identify the impacted keys (which prefix, how many)",
        "Enable the circuit breaker if not already active",
        "Invalidate the specific keys in batches (100/sec max)",
        "Monitor the hit rate and the DB CPU in real time",
        "If the hit rate drops < 50% : enable the DB rate limiter (1K QPS max)",
        "If the DB is saturated : enable the stale-cache fallback (serve the old data)",
        "Communicate the incident internally (Slack #incidents)",
        "Once the cache is rebuilt (hit rate > 80%) : disable the protections",
        "Post-mortem within 24h : timeline, root cause, corrective actions",
    ]
    for i, step in enumerate(steps, 1):
        print(f"       {i:2d}. {step}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Runs all the solutions."""
    print("\n" + "=" * 60)
    print("  SOLUTIONS — DAY 3 : CACHING & CDN")
    print("=" * 60)

    # Easy
    easy_1_cache_strategy()
    easy_2_cache_control()
    easy_3_redis_sizing()

    # Medium
    medium_1_social_feed()
    medium_2_cache_diagnosis()
    medium_3_cdn_strategy()

    # Hard
    hard_1_black_friday()
    hard_2_postmortem()

    print(f"\n{'=' * 60}")
    print("  END OF SOLUTIONS")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
