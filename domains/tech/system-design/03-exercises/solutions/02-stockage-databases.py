"""
Solutions -- Day 2 Exercises: Storage & Databases

This file contains the worked solutions for the Easy, Medium, and Hard exercises.
Each solution shows the reasoning step by step.

Usage:
    python 02-stockage-databases.py
"""

import math

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercise 1 : SQL or NoSQL
# =============================================================================

def easy_1_sql_or_nosql():
    """Solution for the SQL vs NoSQL choice per use case."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : SQL or NoSQL — Pick your DB")
    print(SEPARATOR)

    choices = [
        (
            "Payroll management (salaries, payslips, contributions)",
            "SQL (PostgreSQL)",
            "ACID transactions required: a payslip involves computations "
            "on gross salary, contributions, net pay. Referential integrity between "
            "employee, contract, and payslip. Error = legal and financial consequence."
        ),
        (
            "Search result cache (5 min TTL)",
            "NoSQL Key-Value (Redis)",
            "Access pattern = GET/SET by search key. Native TTL. "
            "Sub-ms latency required. Tolerable loss (it's a cache). "
            "No need for joins or transactions."
        ),
        (
            "Recipe catalog (variable attributes)",
            "NoSQL Document (MongoDB)",
            "Flexible schema: each recipe has different attributes "
            "(cooking time, variable allergens, ingredients in different quantities). "
            "One JSON document per recipe contains everything. No joins needed."
        ),
        (
            "'Buyers of X also bought Y' recommendation",
            "NoSQL Graph (Neo4j)",
            "The model is a graph: nodes = users + products, edges = purchases. "
            "The 'buyers of X -> other purchases -> products' query is a traversal "
            "of depth 2-3. In SQL, that means very expensive recursive joins."
        ),
        (
            "IoT metrics (200K events/sec, 50K sensors)",
            "NoSQL Column-family (Cassandra) or TimescaleDB",
            "Massive write throughput (200K/sec). Time-series data. "
            "Cassandra: LSM-Tree, linear scaling, partition by sensor_id. "
            "TimescaleDB if SQL is needed. Tradeoff: Cassandra scales better, "
            "but TimescaleDB offers SQL queries and native aggregation functions."
        ),
        (
            "Insurance contract management back office",
            "SQL (PostgreSQL)",
            "Complex referential integrity: contract -> client -> clauses -> claims. "
            "ACID transactions for contract modifications. "
            "Frequent ad hoc queries (reporting, search). Moderate volume."
        ),
    ]

    for i, (system, choice, justification) in enumerate(choices, 1):
        print(f"\n  {i}. {system}")
        print(f"     Choice : {choice}")
        print(f"     Reason : {justification}")


# =============================================================================
# EASY -- Exercise 2 : Index or no index?
# =============================================================================

def easy_2_index_or_not():
    """Solution for the indexing decisions."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : Index or no index?")
    print(SEPARATOR)

    decisions = [
        {
            "situation": "users table (50M rows), email column, WHERE email = ? on every login",
            "decision": "YES — Unique index on email",
            "justification": (
                "50M rows and a query executed on every login = critical. "
                "Without an index: full scan of 50M rows on every login (~seconds). "
                "With a B-Tree index: lookup in O(log 50M) = ~26 comparisons (~ms). "
                "Bonus: a UNIQUE index guarantees email uniqueness."
            ),
        },
        {
            "situation": "logs table (500M rows), level column (3 values), 10K inserts/sec",
            "decision": "NO — No index on level alone",
            "justification": (
                "Cardinality = 3 (INFO, WARN, ERROR). The index only filters 33% on average. "
                "The query planner will prefer a full scan over such a low-selectivity index. "
                "Moreover, 10K inserts/sec means each insert must also "
                "update the index -> significant overhead for zero benefit. "
                "Alternative: if we only search for ERROR (1% of the total), "
                "a partial index 'WHERE level = ERROR' would be useful and small."
            ),
        },
        {
            "situation": "orders table (10M rows), WHERE customer_id = ? ORDER BY created_at DESC",
            "decision": "YES — Composite index (customer_id, created_at DESC)",
            "justification": (
                "Perfect composite index: filter by customer_id then sort by created_at. "
                "The SQL engine walks the index in clustering key order "
                "without an extra sort (index-ordered scan). "
                "An index on customer_id alone would force a sort on created_at after the filter."
            ),
        },
        {
            "situation": "config table (50 rows), key column, WHERE key = ?",
            "decision": "NO — Not necessary",
            "justification": (
                "50 rows fit in a single disk page (~8 KB). "
                "A full scan of 50 rows is faster than traversing a B-Tree. "
                "The index maintenance overhead is pointless for so little data."
            ),
        },
        {
            "situation": "events table (1B rows), insert-only, 1 nightly batch read job",
            "decision": "NO for permanent indexes — YES for a temporary index if needed",
            "justification": (
                "Insert-only + 1 read/night = 99.99% write workload. "
                "Each permanent index slows down every one of the billions of inserts. "
                "Better: create an index just before the nightly batch job, "
                "then drop it afterwards (or use date partitioning "
                "to limit the scan to the needed day)."
            ),
        },
    ]

    for i, d in enumerate(decisions, 1):
        print(f"\n  {i}. {d['situation']}")
        print(f"     Decision : {d['decision']}")
        print(f"     Reason : {d['justification']}")


# =============================================================================
# EASY -- Exercise 3 : Replication — Who reads what?
# =============================================================================

def easy_3_replication():
    """Solution for the replication lag problem."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : Replication — Who reads what?")
    print(SEPARATOR)

    print(f"""
  1. WHAT DOES THE USER SEE?
  {'-'*50}
  The user sees their OLD profile name.
  Why: they wrote to the leader, but the follower has not yet
  replicated the change. The average lag is 200ms, and the read
  arrives 100ms after the write -> the follower has ~50% chance
  of not having the up-to-date data (and at peak lag = 2s, it's certain).

  This is the classic "read-your-writes" problem.

  2. SOLUTION WITHOUT FORCING ALL READS TO THE LEADER
  {'-'*50}
  The 'read-your-writes consistency' principle:
  - When a user WRITES, we record a marker
    (timestamp of the last write) in their session or a cookie.
  - When the SAME user READS, we check the marker:
    - If the write is recent (< max lag threshold), read from the LEADER
    - Otherwise, read from a follower normally.
  - Only the reads of the user who just wrote go to the leader.
    Other users always read from the followers.

  3. PSEUDO-CODE
  {'-'*50}""")

    print("""  def handle_read(user_id, key):
      # Retrieve the timestamp of this user's last write
      last_write_ts = session.get(f"last_write:{user_id}")
      max_lag = 2000  # 2 seconds = observed peak lag

      if last_write_ts and (now() - last_write_ts) < max_lag:
          # Recent write: read from the leader to guarantee freshness
          return leader.read(key)
      else:
          # No recent write: follower is OK
          return random_follower.read(key)

  def handle_write(user_id, key, value):
      leader.write(key, value)
      # Mark the last write for this user
      session.set(f"last_write:{user_id}", now())""")

    print(f"""
  4. LOAD COST ON THE LEADER
  {'-'*50}
  If 5% of users write within a 2-second interval,
  then 5% of the reads are redirected to the leader instead of the followers.

  Concrete example:
  - 100K total read QPS
  - 5% redirected to the leader = 5K additional QPS on the leader
  - The leader already handles the writes (~2K QPS) -> total = 7K QPS
  - PostgreSQL easily handles 50K QPS -> ample headroom

  The cost is low because only a small fraction of the reads
  (those of users who just wrote) goes to the leader.
  That's much better than routing ALL read traffic to the leader.""")


# =============================================================================
# MEDIUM -- Exercise 1 : Sharding for a messaging system
# =============================================================================

def medium_1_messaging_sharding():
    """Solution for the messaging sharding scheme."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Sharding scheme — Messaging")
    print(SEPARATOR)

    # 1. Storage estimation
    users = 100_000_000
    msgs_per_user_per_day = 50
    msg_size = 500  # bytes
    retention_days = 3 * 365  # 3 years

    total_msgs_per_day = users * msgs_per_user_per_day
    storage_per_day_gb = total_msgs_per_day * msg_size / (1024 ** 3)
    storage_total_tb = storage_per_day_gb * retention_days / 1024
    storage_total_pb = storage_total_tb / 1024

    print(f"\n  1. STORAGE ESTIMATION")
    print(f"  {'-'*50}")
    print(f"  Messages/day = {users:,} * {msgs_per_user_per_day} = {total_msgs_per_day:,.0f}")
    print(f"  Storage/day = {total_msgs_per_day:,.0f} * {msg_size} B = {storage_per_day_gb:,.0f} GB")
    print(f"  Storage 3 years = {storage_per_day_gb:,.0f} GB * {retention_days} = {storage_total_tb:,.0f} TB = {storage_total_pb:,.1f} PB")

    print(f"""
  2. PARTITION KEY
  {'-'*50}
  Choice : conversation_id

  Why conversation_id :
  - The main access pattern is "last 50 messages of a conversation"
    -> ALL of the query's data is in a SINGLE partition
    -> 1 single node queried, no scatter-gather

  Why NOT user_id :
  - A message in a group of 500 people would have to be stored 500 times
    (one per user_id), or else a scatter-gather over 500 partitions.
  - 1-to-1 conversations would be split between 2 partitions.

  Why NOT message_id :
  - Each message would land on a different partition.
  - "Last 50 messages" = scatter-gather over 50 partitions -> terrible.

  3. CLUSTERING KEY
  {'-'*50}
  Clustering key : message_id (time-based, Snowflake-ID style)
  Or : created_at DESC + message_id DESC (to handle timestamp ties)

  Result : a conversation's messages are stored sorted by date.
  "SELECT * WHERE conversation_id = ? ORDER BY message_id DESC LIMIT 50"
  -> sequential read within a single partition, no sort.

  4. HOT SPOT — GROUP OF 500 PEOPLE
  {'-'*50}
  Problem : a very active conversation (1000 messages/sec) overloads
  a single partition.

  Solutions :
  a) Sub-partitioning : partition key = (conversation_id, bucket)
     Where bucket = message_id % 10 (10 sub-partitions per conversation)
     Cost : "last 50 messages" = scatter-gather over 10 buckets.

  b) Write buffering : messages are written to a buffer (Redis/Kafka)
     then batch-inserted into Cassandra every 100ms.
     Reduces the load on the partition.

  c) In practice : groups of 500+ people are rare (<0.1% of conversations).
     They can be treated as a special case (dedicated shard).

  5. KEYWORD SEARCH
  {'-'*50}
  Cassandra is NOT built for full-text search.

  Solution : Elasticsearch asynchronously.
  - Each message is indexed in Elasticsearch via an async pipeline (Kafka).
  - ES schema : conversation_id, message_text, sender_id, timestamp
  - Query : full-text search filtered by conversation_id
  - ES returns the message_ids, then we fetch them from Cassandra.

  6. TABLE SCHEMA (Cassandra CQL)
  {'-'*50}""")

    print("""  -- Main messages table
  CREATE TABLE messages (
      conversation_id UUID,
      message_id      TIMEUUID,    -- Snowflake-like, includes the timestamp
      sender_id       UUID,
      content         TEXT,
      content_type    TEXT,         -- 'text', 'image', 'file'
      metadata        MAP<TEXT, TEXT>,
      created_at      TIMESTAMP,
      PRIMARY KEY (conversation_id, message_id)
  ) WITH CLUSTERING ORDER BY (message_id DESC);

  -- Secondary index to list a user's conversations
  CREATE TABLE user_conversations (
      user_id          UUID,
      last_activity_at TIMESTAMP,
      conversation_id  UUID,
      PRIMARY KEY (user_id, last_activity_at)
  ) WITH CLUSTERING ORDER BY (last_activity_at DESC);""")


# =============================================================================
# MEDIUM -- Exercise 2 : SQL to NoSQL migration
# =============================================================================

def medium_2_migration_analysis():
    """Solution for the PostgreSQL -> DynamoDB migration analysis."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : SQL to NoSQL migration — Analysis")
    print(SEPARATOR)

    print(f"""
  1. CONCRETE ADVANTAGES OF DYNAMODB FOR THIS CASE
  {'-'*50}
  a) Native write scaling : DynamoDB auto-scales the writes.
     PostgreSQL's 95% write capacity will no longer be a problem.
  b) Nonexistent replication lag for eventually consistent reads
     (DynamoDB natively replicates across 3 AZs).
  c) Predictable latency : single-digit ms regardless of volume.
  d) No server maintenance (managed service) : no more vacuum,
     no more autovacuum blocking the writes.

  2. HIDDEN COSTS OF THE MIGRATION (at least 5)
  {'-'*50}
  a) Query rewrites : all complex SQL (joins, subqueries,
     GROUP BY) must be rewritten as DynamoDB access patterns.
     Each join = a denormalization or a GSI (Global Secondary Index).

  b) Data migration : 2 TB and 100M rows to migrate without downtime.
     Dual-write during the transition. Consistency validation.
     Estimated duration : 2-4 weeks of migration + 2 weeks of dual-run.

  c) Loss of joins : "orders JOIN order_items JOIN products"
     -> either denormalize (duplicate products into each order_item),
     or make 3 separate requests on the application side.

  d) Team training : the team knows SQL/PostgreSQL.
     DynamoDB has a different mental model (single-table design, GSI, LSI).
     1-2 month learning curve for the team.

  e) Financial cost : DynamoDB bills reads/writes (WCU/RCU).
     At 10K TPS peak, the cost can be > $10K/month.
     PostgreSQL on EC2/RDS is often cheaper for a predictable workload.

  f) AWS lock-in : DynamoDB is proprietary. No portability.

  3. WHAT YOU LOSE BY LEAVING POSTGRESQL
  {'-'*50}
  a) Multi-table ACID transactions : DynamoDB transactions = max 25 items,
     no complex cross-table transaction.
  b) Ad hoc SQL : no more "SELECT ... WHERE ... GROUP BY ... HAVING ..."
     for debugging or reporting. All access patterns must be pre-defined.
  c) Tooling ecosystem : pgAdmin, pg_dump, EXPLAIN ANALYZE, extensions
     (PostGIS, pg_trgm, pgvector). DynamoDB has limited tooling.
  d) Schema enforcement : DynamoDB has no rigid schema.
     Risk of malformed data in production.

  4. ALTERNATIVE WITHOUT LEAVING POSTGRESQL
  {'-'*50}
  The 3 identified problems :

  Problem 1 : Writes at 95% capacity
  Solution : PostgreSQL partitioning by date (monthly) + archiving old partitions.
  Reduces the active table size. Alternative : Citus for horizontal sharding.

  Problem 2 : Replication lag > 10s
  Solution : Switch to synchronous replication for 1 follower (synchronous_commit).
  Or add followers (more replicas = less load per replica).
  Or investigate the cause of the lag (vacuum, long-running queries on the followers).

  Problem 3 : Joins > 5 seconds
  Solution : Pre-computed materialized views for the frequent queries.
  Or CQRS : a denormalized read model (pre-joined table) updated via triggers/CDC.
  Or missing indexes (EXPLAIN ANALYZE to identify the full scans).

  5. DECISION FRAMEWORK
  {'-'*50}
  Migrate to DynamoDB IF AND ONLY IF :
  - The PostgreSQL alternatives have been tried and failed
  - The workload is mostly key-value (> 80% of queries are by PK)
  - The team has at least 1 person experienced with DynamoDB
  - The DynamoDB budget ($) is < 2x the current PostgreSQL cost
  - The complex joins can be eliminated through denormalization
  - The schedule allows for 2+ months of migration

  If ONE of these conditions is not met : stay on PostgreSQL
  and optimize (Citus, partitioning, CQRS, materialized views).""")


# =============================================================================
# MEDIUM -- Exercise 3 : Index strategy for an analytics dashboard
# =============================================================================

def medium_3_index_strategy():
    """Solution for the indexing strategy."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Index Strategy — Analytics Dashboard")
    print(SEPARATOR)

    print(f"""
  1. INDEX FOR EACH QUERY
  {'-'*50}

  Query 1 (90% of the traffic) :
  SELECT count(*), sum(amount) FROM events
  WHERE tenant_id = ? AND event_type = ? AND created_at BETWEEN ? AND ?

  Index : CREATE INDEX idx_events_q1
          ON events (tenant_id, event_type, created_at)
          INCLUDE (amount);

  Order : tenant_id (equality) -> event_type (equality) -> created_at (range)
  Rule : equality columns first, the range column last.
  The INCLUDE (amount) makes it a covering index (see Q2).

  Query 2 (8% of the traffic) :
  SELECT * FROM events
  WHERE tenant_id = ? AND user_id = ? ORDER BY created_at DESC LIMIT 20

  Index : CREATE INDEX idx_events_q2
          ON events (tenant_id, user_id, created_at DESC);

  No covering index possible here (SELECT *).

  Query 3 (2% of the traffic) :
  SELECT DISTINCT event_type FROM events
  WHERE tenant_id = ? AND created_at > NOW() - INTERVAL '24h'

  Index : no dedicated index necessary. The idx_events_q1 index partially
  covers it (tenant_id, event_type). Or else an index :
  CREATE INDEX idx_events_q3
  ON events (tenant_id, created_at) INCLUDE (event_type);

  But at 2% of the traffic, the cost of a dedicated index is debatable.
  Date partitioning (see Q5) will solve this problem more elegantly.

  2. COVERING INDEX
  {'-'*50}
  Yes, for Query 1 :
  Index (tenant_id, event_type, created_at) INCLUDE (amount)

  The query needs : tenant_id, event_type, created_at (filters) + amount (aggregation).
  All these columns are in the index -> index-only scan.
  The SQL engine NEVER reads the table, only the index.
  Gain : avoids random I/O to the table (heap) pages.

  3. DISK SPACE OVERHEAD
  {'-'*50}""")

    # Table size estimation
    row_size_estimate = 16 + 16 + 50 + 16 + 12 + 200 + 8  # UUID + UUID + varchar + UUID + decimal + JSONB + timestamp
    # ~318 bytes per row, but with PostgreSQL overhead (~24 bytes header) -> ~342 bytes

    msgs_per_sec = 50_000
    rows_per_day = msgs_per_sec * 86_400
    rows_per_month = rows_per_day * 30

    table_size_per_month_gb = rows_per_month * 342 / (1024 ** 3)

    print(f"  Estimated size per row : ~342 bytes (columns + PostgreSQL header)")
    print(f"  Rows/day = {msgs_per_sec:,} * 86400 = {rows_per_day:,.0f}")
    print(f"  Rows/month = {rows_per_month:,.0f}")
    print(f"  Table size/month = {table_size_per_month_gb:,.0f} GB")

    # Each index ~ 30% of the size of the indexed columns
    # idx_q1 : (UUID 16 + varchar 50 + timestamp 8 + decimal 12) = 86 bytes/entry
    idx_q1_size = rows_per_month * 86 / (1024 ** 3)
    # idx_q2 : (UUID 16 + UUID 16 + timestamp 8) = 40 bytes/entry
    idx_q2_size = rows_per_month * 40 / (1024 ** 3)

    print(f"\n  Index idx_events_q1 : ~{idx_q1_size:,.0f} GB/month")
    print(f"  Index idx_events_q2 : ~{idx_q2_size:,.0f} GB/month")
    print(f"  Total indexes       : ~{idx_q1_size + idx_q2_size:,.0f} GB/month")
    print(f"  Indexes/table ratio : ~{(idx_q1_size + idx_q2_size) / table_size_per_month_gb * 100:.0f}%")

    print(f"""
  4. IMPACT ON THE 50K INSERTS/SEC
  {'-'*50}
  Each INSERT must update :
  - The table (1 write)
  - idx_events_q1 (1 B-Tree write)
  - idx_events_q2 (1 B-Tree write)
  = 3 writes per INSERT instead of 1

  With 2 indexes : insert throughput reduced by ~30-50%
  (depends on the cache, the index sizes, and the hardware).

  If the 50K inserts/sec are critical :
  - Option 1 : batch inserts (COPY instead of INSERT)
  - Option 2 : create the indexes only on the cold partitions (see Q5)
  - Option 3 : accept the degradation (30-50% of 50K = 25-35K inserts/sec
    with indexes, which often remains sufficient)

  5. PARTITIONING STRATEGY
  {'-'*50}
  Type : Range partitioning by created_at
  Granularity : monthly (each month = one partition)

  CREATE TABLE events (...) PARTITION BY RANGE (created_at);
  CREATE TABLE events_2026_01 PARTITION OF events
      FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
  -- Repeat for each month

  Advantages :
  - Queries with WHERE created_at BETWEEN ... scan only
    the relevant partitions (partition pruning)
  - Old partitions can be archived (detach + move to cold storage)
  - Indexes are per partition (smaller, faster to maintain)
  - VACUUM is faster per partition

  6. WHEN TO MOVE TO CLICKHOUSE
  {'-'*50}
  PostgreSQL reaches its limits for analytics when :
  - The table exceeds ~1B rows (aggregations take > 10s)
  - Analytical queries (GROUP BY, SUM, COUNT) are > 50% of the traffic
  - The p99 of analytical queries exceeds the SLO (e.g. > 5s)
  - The analytical workload degrades insert performance

  ClickHouse :
  - Columnar storage : aggregations over 1B rows take 100ms-1s
  - 10-20x compression (columnar compresses better than row-based)
  - No transactions, no updates -> insert-only = perfect for events

  Transition strategy :
  - Keep PostgreSQL for the inserts (source of truth)
  - CDC (Change Data Capture) to ClickHouse for the analytics
  - The dashboard reads from ClickHouse, writes go into PostgreSQL""")


# =============================================================================
# HARD -- Exercise 1 : Storage for a payment system (sketch)
# =============================================================================

def hard_1_payment_storage():
    """Sketched solution for the multi-region payment system."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Multi-region payment system")
    print(SEPARATOR)

    # Storage estimation
    tps = 10_000
    secs_per_year = 365.25 * 24 * 3600
    txns_per_year = tps * secs_per_year  # Average, not peak
    # In reality, peak != average. Let's use a factor of 0.3 (average = 30% of the peak)
    avg_tps = tps * 0.3  # 3K average TPS
    txns_per_year_avg = avg_tps * secs_per_year

    # Size of a transaction
    # payment_id (16) + amount (12) + currency (3) + merchant_id (16) + customer_id (16)
    # + status (10) + created_at (8) + updated_at (8) + metadata (200) + indexes overhead
    txn_size = 16 + 12 + 3 + 16 + 16 + 10 + 8 + 8 + 200  # ~289 bytes
    txn_with_overhead = txn_size * 2  # PostgreSQL overhead + indexes ~ 2x

    # Audit log: each state change = 1 entry
    # On average 3 changes per transaction (pending -> confirmed -> settled)
    audit_entries_per_txn = 3
    audit_entry_size = 200  # payment_id + old_state + new_state + timestamp + actor

    print(f"\n  STORAGE ESTIMATION")
    print(f"  {'-'*50}")
    print(f"  Peak TPS : {tps:,} | Average TPS (30% of peak) : {avg_tps:,.0f}")
    print(f"  Transactions/year : {txns_per_year_avg:,.0f}")
    print(f"  Size/txn (with overhead) : {txn_with_overhead} bytes")

    storage_txn_7y_tb = txns_per_year_avg * 7 * txn_with_overhead / (1024 ** 4)
    storage_audit_7y_tb = txns_per_year_avg * 7 * audit_entries_per_txn * audit_entry_size / (1024 ** 4)

    print(f"  Transaction storage 7 years : {storage_txn_7y_tb:,.0f} TB")
    print(f"  Audit log storage 7 years   : {storage_audit_7y_tb:,.0f} TB")
    print(f"  Total                       : {storage_txn_7y_tb + storage_audit_7y_tb:,.0f} TB")

    print(f"""
  1. DB CHOICE
  {'-'*50}

  a) Payment transactions : PostgreSQL (with Citus for sharding)
     - ACID required : a payment cannot be "partially created"
     - State machine with CHECK constraints on the transitions
     - Strong consistency required
     - Citus for horizontal sharding once we outgrow 1 node

  b) Analytics dashboard : ClickHouse
     - Fast aggregations (volume/day, total amount, success rate)
     - Analytics queries must NOT impact the OLTP transactions
     - CDC from PostgreSQL -> ClickHouse (Debezium)
     - Eventual consistency acceptable (dashboard refreshed every 10s)

  c) Immutable audit log : Apache Kafka + S3/GCS
     - Kafka : append-only log, immutable, ordered, replicated
     - Each state change publishes an event into Kafka
     - Kafka -> S3 (7-year long-term archival, Parquet format)
     - Kafka retention : 30 days (for replay if needed)
     - S3 : compliance, immutability (Object Lock), low cost

  2. DATA SCHEMA
  {'-'*50}""")

    print("""  -- Main payments table
  CREATE TABLE payments (
      payment_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      merchant_id   UUID NOT NULL,
      customer_id   UUID NOT NULL,
      amount        DECIMAL(12,2) NOT NULL CHECK (amount > 0),
      currency      VARCHAR(3) NOT NULL,
      status        VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','confirmed','settled','failed','refunded')),
      created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      region        VARCHAR(2) NOT NULL  -- 'EU' or 'US' for GDPR
  );

  -- Refunds table
  CREATE TABLE refunds (
      refund_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      payment_id    UUID NOT NULL REFERENCES payments(payment_id),
      amount        DECIMAL(12,2) NOT NULL CHECK (amount > 0),
      reason        TEXT,
      status        VARCHAR(20) NOT NULL DEFAULT 'pending',
      created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );

  -- State machine : valid transitions
  -- pending -> confirmed (payment provider confirms)
  -- pending -> failed (payment provider rejects)
  -- confirmed -> settled (T+1 settlement)
  -- confirmed -> refunded (merchant initiates refund)
  -- settled -> refunded (post-settlement refund)
  -- Enforce via trigger or application-level check""")

    print(f"""
  3. SHARDING STRATEGY
  {'-'*50}
  Shard key : payment_id (UUID, uniform distribution via hash)

  Why payment_id :
  - Each payment is an independent aggregate (no cross-payment join)
  - Writes are distributed uniformly (UUID = random)
  - Lookup by payment_id is the most frequent pattern

  Problem : "list a merchant's payments"
  - payment_id as shard key = scatter-gather over ALL shards
  - Solution : Global Secondary Index (GSI) by merchant_id
    Or : denormalized table merchant_payments (merchant_id, payment_id, created_at)
    with shard key = merchant_id

  Number of shards :
  - PostgreSQL handles ~5K TPS per node (with the transactions and indexes)
  - 10K peak TPS / 5K = 2 shards minimum
  - With margin (3x) : 6 shards
  - Start with 8 shards (power of 2, headroom for growth)

  4. MULTI-REGION REPLICATION
  {'-'*50}
  GDPR constraint : EU data does not leave the EU.

  Architecture :
  - EU region : shards for the payments where region='EU'
  - US region : shards for the payments where region='US'
  - No cross-region replication of payment data
  - Routing is done by the API Gateway based on the merchant/customer region

  Consistency :
  - Intra-region : synchronous replication (1 leader + 1 sync follower)
  - Cross-region : NO replication of the transactions
  - The analytics dashboard can aggregate from both regions
    (read-only ClickHouse replicas in each region)

  Latency : no cross-region compromise for the transactions.
  Each transaction is local to its region.

  5. 99.99% AVAILABILITY
  {'-'*50}
  Identified SPOFs :
  a) PostgreSQL leader -> Automatic failover (Patroni + etcd)
  b) API Gateway -> Multi-AZ, auto-scaling
  c) Kafka cluster -> 3+ brokers, replication factor 3
  d) DNS -> Route53 health checks, multi-region failover

  Mechanisms :
  - Patroni : automatic PostgreSQL failover in < 30s
  - Health checks on every component (5s interval)
  - Circuit breaker on inter-service calls
  - Chaos engineering : Chaos Monkey, Litmus, random pod kills weekly

  6. ARCHIVAL
  {'-'*50}
  - Monthly partitioning on the payments table
  - Data > 1 year : detach partition, export to S3 (Parquet)
  - Data > 3 years : cold tier (S3 Glacier)
  - Indexes only on the recent partitions (< 1 year)
  - Queries on old data go through Athena/Presto on S3""")


# =============================================================================
# HARD -- Exercise 2 : Migration post-mortem (written analysis)
# =============================================================================

def hard_2_postmortem():
    """Solution for the post-mortem of the migration that broke production."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Post-mortem — Sessions migration PG -> Redis")
    print(SEPARATOR)

    print(f"""
  1. ROOT CAUSE ANALYSIS
  {'-'*50}

  Root cause : insufficient Redis memory sizing for 800M sessions,
  combined with the premature removal of the PostgreSQL fallback.

  Contributing factors :

  a) No capacity planning :
     800M sessions * average_size = ? GB.
     If a session averages 500 bytes : 800M * 500 = 400 GB.
     The provisioned Redis probably had 64-128 GB.
     Detection : a simple calculation before the migration would have shown the gap.

  b) No realistic load test :
     Testing with 100K sessions does not reveal a problem at 800M.
     Detection : load test with the real volume (or a significant percentage)
     on a Redis the same size as prod.

  c) Premature removal of the double-write (15:00, 1h after the start) :
     "Everything looks OK" after 1h proves nothing. Sessions accumulate
     progressively. The OOM problem hits when the RAM is full, not before.
     Detection : wait at least 24-48h before removing the fallback.
     Define an objective criterion ("when 95%+ of the session traffic is stable
     on Redis for 48h, remove the double-write").

  d) No monitoring of Redis memory usage :
     OOM at 17:00 = the memory had been climbing since 14:00 but nobody was watching.
     Detection : alert on used_memory > 80% of maxmemory.
     Alert on evicted_keys > 0.

  e) No tested rollback plan :
     At 17:10, the rollback fails because the new sessions are not in PG.
     Detection : include "rollback plan" in the migration checklist.
     Test the rollback BEFORE starting the migration.

  2. DECISION ANALYSIS
  {'-'*50}

  When to stop :
  - At 15:00 ("everything looks OK"). 1 hour is NOT enough to validate
    a migration of 800M sessions. The "looks OK" criterion is subjective.
  - Rule : no fallback removal before 48h + objective metrics OK.

  Why "everything looks OK" was dangerous :
  - The sessions fill up progressively (TTL = 30 min).
  - After 1h, only the active sessions are in Redis (~10-20% of the total).
  - The 800M sessions will be in Redis in ~30 min * renewal factor.
  - The real test is when Redis reaches steady state (24-48h).

  Why the rollback failed :
  - The double-write to PostgreSQL was removed at 15:00.
  - Any session created between 15:00 and 17:10 exists ONLY in Redis.
  - Rolling back to PG = losing 2h10 of sessions.
  - This is exactly why the fallback must never be removed
    before complete validation.

  3. FIXES
  {'-'*50}

  a) Capacity planning : calculate BEFORE the migration.
     800M sessions * 500 bytes = 400 GB minimum.
     With Redis overhead (~2x for the internal structures) = 800 GB.
     Provision 1 TB of RAM (Redis Cluster of 10+ nodes at 128 GB each).

  b) Load test : reproduce the real volume.
     Script that creates 800M sessions in a test Redis.
     Measure : time, memory, latency, throughput.

  c) Progressive migration :
     Phase 1 : double-write PG + Redis (100% of the sessions)
     Phase 2 : canary reads (5% of the traffic reads from Redis, 95% from PG)
     Phase 3 : ramp up progressively (25%, 50%, 75%, 100%)
     Phase 4 : wait 48h at 100% Redis reads + PG double-write
     Phase 5 : remove the double-write ONLY if metrics are OK for 48h

  d) Monitoring :
     - Alert : used_memory > 70% of maxmemory (warning)
     - Alert : used_memory > 85% of maxmemory (critical)
     - Alert : evicted_keys > 0 (immediate, since eviction = session loss)
     - Dashboard : number of sessions in Redis vs PostgreSQL

  4. TARGET REDIS ARCHITECTURE
  {'-'*50}

  Topology :
  - Redis Cluster (not Sentinel) : 10+ master nodes, each with 128 GB RAM
  - 1 replica per master (20 nodes total)
  - Multi-AZ for resilience

  Persistence :
  - AOF (Append Only File) : every write is logged
  - fsync = everysec (performance/durability compromise : max 1 second of loss)
  - RDB snapshot every hour (for fast recovery)

  Eviction policy :
  - maxmemory-policy = noeviction
  - WHY : sessions must NEVER be silently evicted.
  - If Redis is full, new writes fail with an error.
  - The application detects the error and falls back to PostgreSQL.
  - Alternative : allkeys-lru if the loss of old sessions is acceptable.
    BUT : a user logged out for no reason = bad UX.

  RAM sizing :""")

    sessions = 800_000_000
    session_size_bytes = 500  # average bytes per session
    redis_overhead_factor = 2.0  # Redis internal structures overhead
    total_ram_gb = sessions * session_size_bytes * redis_overhead_factor / (1024 ** 3)
    nodes_128gb = math.ceil(total_ram_gb / 128)

    print(f"  800M sessions * 500 bytes * 2x overhead = {total_ram_gb:,.0f} GB")
    print(f"  Nodes needed (128 GB/node) = {nodes_128gb}")
    print(f"  With replicas : {nodes_128gb * 2} nodes total")
    print(f"  30% margin : {math.ceil(nodes_128gb * 1.3)} masters + {math.ceil(nodes_128gb * 1.3)} replicas")

    print(f"""
  Keep PostgreSQL as a fallback :
  YES, for at least 3-6 months after the full migration.
  - Permanent double-write (the cost is low : PG easily handles the writes)
  - If Redis goes down : transparent fallback to PG (5-15ms latency instead of <1ms)
  - Monitoring : continuously compare the number of sessions in PG vs Redis

  5. GENERIC MIGRATION CHECKLIST
  {'-'*50}

  BEFORE the migration :
  [ ] Capacity planning : compute the real volume on the target
  [ ] Load test on the target with the real volume (or 50%+)
  [ ] Rollback plan documented and TESTED (dry-run of the rollback)
  [ ] Monitoring in place BEFORE starting (alerts on the target)
  [ ] Incident runbook : who to call, what to do if it breaks
  [ ] Team communication : everyone informed of the plan and the timing

  DURING the migration :
  [ ] Double-write enabled (source + target)
  [ ] Progressive canary reads (5% -> 25% -> 50% -> 100%)
  [ ] Consistency validation : compare source vs target reads
  [ ] Continuous monitoring of the key metrics (latency, errors, resources)
  [ ] Checkpoint at each step : "continue or rollback?"

  AFTER the migration :
  [ ] Dual-run for at least 48h at 100% reads on the target
  [ ] Do NOT remove the fallback before complete validation
  [ ] Objective success criteria defined and measured
  [ ] Post-mortem scheduled (even if everything goes well)
  [ ] Documentation updated (architecture, runbooks, monitoring)""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Runs all the solutions."""
    print("\n" + "#" * 60)
    print("#  SOLUTIONS -- DAY 2 : STORAGE & DATABASES")
    print("#" * 60)

    # Easy
    easy_1_sql_or_nosql()
    easy_2_index_or_not()
    easy_3_replication()

    # Medium
    medium_1_messaging_sharding()
    medium_2_migration_analysis()
    medium_3_index_strategy()

    # Hard
    hard_1_payment_storage()
    hard_2_postmortem()

    print("\n" + "#" * 60)
    print("#  END OF SOLUTIONS")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
