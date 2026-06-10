"""
Day 2 -- Storage & Databases
Interactive demonstrations in Python.

Usage:
    python 02-stockage-databases.py

Each section is independent and can be executed via the main() function.
"""

import time
import random
import hashlib
import math
import bisect
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

# =============================================================================
# SECTION 1 : B-Tree Index vs Full Scan — Speedup demo
# =============================================================================


class SimpleBTreeNode:
    """Node of a simplified B-Tree (order 4).

    A real B-Tree has complex balancing constraints.
    We simplify here to show the principle: O(log n) search
    through a sorted tree structure.
    """

    def __init__(self, order: int = 4):
        self.order = order          # Max number of keys per node
        self.keys: list = []        # Sorted keys in this node
        self.values: list = []      # Associated values (same index as keys)
        self.children: list = []    # Children (len = len(keys) + 1 if internal node)
        self.is_leaf: bool = True

    def search(self, key) -> Optional[any]:
        """Searches for a key in the B-Tree. Returns the value or None.

        Complexity: O(log n) — at each node, one branch is eliminated.
        Like a binary search, but in tree form.
        """
        # Find the position where the key should be
        i = bisect.bisect_left(self.keys, key)

        # If the key is in this node, return it
        if i < len(self.keys) and self.keys[i] == key:
            return self.values[i]

        # If this is a leaf, the key does not exist
        if self.is_leaf:
            return None

        # Otherwise, descend into the right child
        return self.children[i].search(key)


class SimpleBTree:
    """Simplified B-Tree for the demonstration.

    Supports insert and search. The split is simplified
    (we do not do a real internal-node split to keep the code readable).
    """

    def __init__(self, order: int = 100):
        self.root = SimpleBTreeNode(order)
        self.order = order
        self._comparisons = 0  # Counter to measure the work done

    def insert(self, key, value):
        """Inserts a key-value pair. Simplified: we do a sorted insert into the leaf."""
        node = self.root
        # Descend to the appropriate leaf
        while not node.is_leaf:
            i = bisect.bisect_left(node.keys, key)
            node = node.children[i]

        # Insert into the leaf (maintaining order)
        i = bisect.bisect_left(node.keys, key)
        node.keys.insert(i, key)
        node.values.insert(i, value)

        # If the node overflows, split it (simplified version)
        if len(node.keys) > self.order:
            self._split_leaf(node)

    def _split_leaf(self, node: SimpleBTreeNode):
        """Simplified split: creates a new root node with two children."""
        mid = len(node.keys) // 2

        # New right node with the upper half
        right = SimpleBTreeNode(self.order)
        right.keys = node.keys[mid:]
        right.values = node.values[mid:]

        # The current node keeps the lower half
        left_keys = node.keys[:mid]
        left_values = node.values[:mid]

        # If this is the root, create a new root
        if node is self.root:
            new_root = SimpleBTreeNode(self.order)
            new_root.is_leaf = False
            new_root.keys = [right.keys[0]]
            new_root.values = [right.values[0]]
            new_root.children = [SimpleBTreeNode(self.order), right]
            new_root.children[0].keys = left_keys
            new_root.children[0].values = left_values
            self.root = new_root

    def search(self, key) -> Optional[any]:
        """Search via the B-Tree."""
        return self.root.search(key)


def demo_btree_vs_fullscan():
    """Compares the speed of a B-Tree index vs a full table scan.

    We insert N elements, then measure the search time
    in an unsorted list (full scan) vs a B-Tree (index lookup).
    """
    print("\n" + "=" * 60)
    print("  SECTION 1 : B-Tree Index vs Full Scan")
    print("=" * 60)

    # Generate data: N records with random keys
    N = 100_000
    print(f"\n  Dataset : {N:,} records")

    # Full scan: unsorted list of (key, value) tuples
    data = [(random.randint(1, N * 10), f"record_{i}") for i in range(N)]

    # B-Tree: insert the same data into the index
    btree = SimpleBTree(order=100)
    for key, value in data:
        btree.insert(key, value)

    # Pick keys to search for (some exist, others don't)
    search_keys = [data[random.randint(0, N - 1)][0] for _ in range(1000)]

    # --- Full Scan ---
    start = time.perf_counter()
    found_scan = 0
    for key in search_keys:
        # Walk through the ENTIRE list to find the key: O(n) per search
        for k, v in data:
            if k == key:
                found_scan += 1
                break
    elapsed_scan = (time.perf_counter() - start) * 1000

    # --- B-Tree Index Lookup ---
    start = time.perf_counter()
    found_btree = 0
    for key in search_keys:
        # Walk down the tree: O(log n) per search
        result = btree.search(key)
        if result is not None:
            found_btree += 1
    elapsed_btree = (time.perf_counter() - start) * 1000

    speedup = elapsed_scan / elapsed_btree if elapsed_btree > 0 else float("inf")

    print(f"\n  {'Method':<25} {'Time (ms)':>12} {'Found':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Full Scan (O(n))':<25} {elapsed_scan:>12.1f} {found_scan:>10}")
    print(f"  {'B-Tree Index (O(log n))':<25} {elapsed_btree:>12.1f} {found_btree:>10}")
    print(f"\n  Speedup : {speedup:.1f}x")
    print(f"\n  >> Lesson: With {N:,} rows, the index is ~{speedup:.0f}x faster.")
    print(f"  >> In production (10M+ rows), the speedup is even more dramatic.")
    print(f"  >> A full scan = O(n). A B-Tree index = O(log n).")
    print(f"  >> log2({N:,}) = {math.log2(N):.0f} comparisons vs {N:,} for the scan.")


# =============================================================================
# SECTION 2 : Consistent Hashing — Implementation from scratch
# =============================================================================


class ConsistentHashRing:
    """Consistent hashing with virtual nodes.

    Principle: nodes and keys are placed on a ring
    from 0 to 2^32. Each key is assigned to the next node
    in the clockwise direction.

    Virtual nodes: each physical node has multiple positions
    on the ring for a better distribution.
    """

    def __init__(self, num_virtual_nodes: int = 150):
        self.num_virtual_nodes = num_virtual_nodes
        # The ring: sorted list of (position, node_name)
        self._ring: list[tuple[int, str]] = []
        # Sorted positions for binary search
        self._sorted_positions: list[int] = []
        # Mapping node -> its positions on the ring
        self._node_positions: dict[str, list[int]] = {}

    def _hash(self, key: str) -> int:
        """Hashes a key onto the [0, 2^32) space.

        We use MD5 for its uniform distribution.
        (In prod, we would use MurmurHash3 or xxHash for performance.)
        """
        digest = hashlib.md5(key.encode()).hexdigest()
        return int(digest[:8], 16)  # Take the first 8 hex chars = 32 bits

    def add_node(self, node_name: str):
        """Adds a physical node with its virtual nodes on the ring."""
        positions = []
        for i in range(self.num_virtual_nodes):
            # Each virtual node has a unique key: "node_name#i"
            virtual_key = f"{node_name}#vn{i}"
            pos = self._hash(virtual_key)
            self._ring.append((pos, node_name))
            positions.append(pos)

        self._node_positions[node_name] = positions
        # Re-sort the ring after the addition
        self._ring.sort(key=lambda x: x[0])
        self._sorted_positions = [pos for pos, _ in self._ring]

    def remove_node(self, node_name: str):
        """Removes a node and all of its virtual nodes from the ring."""
        self._ring = [(pos, name) for pos, name in self._ring if name != node_name]
        self._sorted_positions = [pos for pos, _ in self._ring]
        del self._node_positions[node_name]

    def get_node(self, key: str) -> str:
        """Finds the node responsible for a key.

        Method: hash the key, find the next node in the clockwise direction
        on the ring (via binary search).
        """
        if not self._ring:
            raise ValueError("Empty ring — no node available")

        key_hash = self._hash(key)
        # bisect_right finds the index of the first element > key_hash
        idx = bisect.bisect_right(self._sorted_positions, key_hash)

        # If we go past the end of the ring, wrap back to the start (it's a ring)
        if idx == len(self._ring):
            idx = 0

        return self._ring[idx][1]  # Return the node name

    def get_distribution(self, keys: list[str]) -> dict[str, int]:
        """Computes the distribution of keys per node."""
        distribution = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            distribution[node] += 1
        return dict(distribution)


def demo_consistent_hashing():
    """Demonstrates consistent hashing with node addition/removal.

    Shows that only a fraction of the keys is redistributed when a node
    is added or removed (unlike classic modulo hashing).
    """
    print("\n" + "=" * 60)
    print("  SECTION 2 : Consistent Hashing")
    print("=" * 60)

    # --- Initial setup: 4 nodes ---
    ring = ConsistentHashRing(num_virtual_nodes=150)
    initial_nodes = ["shard-A", "shard-B", "shard-C", "shard-D"]
    for node in initial_nodes:
        ring.add_node(node)

    # Generate 10,000 random keys
    keys = [f"user_{i}" for i in range(10_000)]

    # Initial distribution
    dist_before = ring.get_distribution(keys)
    print(f"\n  Initial distribution ({len(initial_nodes)} nodes, {len(keys):,} keys) :")
    for node, count in sorted(dist_before.items()):
        pct = count / len(keys) * 100
        bar = "#" * int(pct / 2)
        print(f"    {node:<12} : {count:>5} ({pct:>5.1f}%) {bar}")

    # Compute the standard deviation to measure balancing
    counts = list(dist_before.values())
    ideal = len(keys) / len(initial_nodes)
    std_dev = (sum((c - ideal) ** 2 for c in counts) / len(counts)) ** 0.5
    print(f"\n  Ideal per node  : {ideal:,.0f}")
    print(f"  Std deviation   : {std_dev:,.0f} ({std_dev / ideal * 100:.1f}% of ideal)")

    # --- Save the assignment before ---
    assignment_before = {key: ring.get_node(key) for key in keys}

    # --- Add a 5th node ---
    print(f"\n  {'='*50}")
    print(f"  Adding shard-E...")
    ring.add_node("shard-E")

    dist_after_add = ring.get_distribution(keys)
    print(f"\n  Distribution after addition ({len(initial_nodes) + 1} nodes) :")
    for node, count in sorted(dist_after_add.items()):
        pct = count / len(keys) * 100
        bar = "#" * int(pct / 2)
        print(f"    {node:<12} : {count:>5} ({pct:>5.1f}%) {bar}")

    # Count the keys that changed node
    assignment_after_add = {key: ring.get_node(key) for key in keys}
    moved = sum(1 for k in keys if assignment_before[k] != assignment_after_add[k])
    print(f"\n  Keys moved : {moved:,} / {len(keys):,} ({moved / len(keys) * 100:.1f}%)")
    print(f"  Theoretical ideal : {1 / (len(initial_nodes) + 1) * 100:.1f}% (= 1/N)")
    print(f"\n  >> With classic modulo hashing, ~80% of the keys would have moved.")
    print(f"  >> Consistent hashing moves only ~{moved / len(keys) * 100:.0f}%.")

    # --- Remove a node ---
    print(f"\n  {'='*50}")
    print(f"  Removing shard-B...")
    assignment_before_remove = {key: ring.get_node(key) for key in keys}
    ring.remove_node("shard-B")

    dist_after_remove = ring.get_distribution(keys)
    print(f"\n  Distribution after removal ({len(initial_nodes)} nodes) :")
    for node, count in sorted(dist_after_remove.items()):
        pct = count / len(keys) * 100
        bar = "#" * int(pct / 2)
        print(f"    {node:<12} : {count:>5} ({pct:>5.1f}%) {bar}")

    assignment_after_remove = {key: ring.get_node(key) for key in keys}
    moved_remove = sum(
        1 for k in keys
        if assignment_before_remove[k] != assignment_after_remove[k]
    )
    print(f"\n  Keys moved : {moved_remove:,} / {len(keys):,} ({moved_remove / len(keys) * 100:.1f}%)")

    # --- Comparison with classic modulo hashing ---
    print(f"\n  {'='*50}")
    print(f"  COMPARISON : Consistent Hashing vs Hash Modulo")
    print(f"  {'='*50}")

    def hash_modulo_distribution(keys_list, num_nodes):
        """Classic modulo hashing: hash(key) % num_nodes."""
        dist = defaultdict(int)
        for key in keys_list:
            h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
            node_idx = h % num_nodes
            dist[f"shard-{node_idx}"] = dist.get(f"shard-{node_idx}", 0) + 1
        return dist

    # With 4 nodes
    modulo_4 = hash_modulo_distribution(keys, 4)
    # Add a 5th node: how many keys change?
    modulo_5 = hash_modulo_distribution(keys, 5)
    modulo_moved = sum(
        1 for key in keys
        if int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 4
        != int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 5
    )
    modulo_pct = modulo_moved / len(keys) * 100

    print(f"\n  Hash modulo : going from 4 to 5 nodes")
    print(f"  Keys moved : {modulo_moved:,} / {len(keys):,} ({modulo_pct:.1f}%)")
    print(f"\n  Consistent hashing : {moved / len(keys) * 100:.1f}% of keys moved")
    print(f"  Hash modulo        : {modulo_pct:.1f}% of keys moved")
    print(f"\n  >> Consistent hashing moves {modulo_pct / (moved / len(keys) * 100):.0f}x fewer keys.")


# =============================================================================
# SECTION 3 : Leader-Follower Replication with Replication Lag
# =============================================================================


@dataclass
class ReplicationEvent:
    """A replication event: a write to propagate from the leader to the followers."""
    key: str
    value: str
    timestamp: float       # When the write was performed on the leader
    sequence_number: int   # Sequence number in the WAL


class LeaderNode:
    """Simulates a replication leader that accepts writes."""

    def __init__(self, name: str):
        self.name = name
        self.data: dict[str, str] = {}  # Local storage
        self.wal: list[ReplicationEvent] = []  # Write-Ahead Log
        self._seq = 0

    def write(self, key: str, value: str) -> ReplicationEvent:
        """Write on the leader: writes to the data store AND the WAL."""
        self._seq += 1
        self.data[key] = value
        event = ReplicationEvent(
            key=key, value=value,
            timestamp=time.time(), sequence_number=self._seq
        )
        self.wal.append(event)
        return event

    def read(self, key: str) -> Optional[str]:
        """Direct read on the leader: always up to date."""
        return self.data.get(key)


class FollowerNode:
    """Simulates a follower that replicates the leader's writes with a delay."""

    def __init__(self, name: str, lag_ms: float):
        self.name = name
        self.lag_ms = lag_ms        # Simulated replication delay (in ms)
        self.data: dict[str, str] = {}
        self.last_applied_seq: int = 0  # Last applied event
        self._pending: list[ReplicationEvent] = []  # Pending events

    def receive_event(self, event: ReplicationEvent):
        """Receives a replication event from the leader.

        The follower does not apply it immediately:
        it simulates a replication delay (network + apply).
        """
        self._pending.append(event)

    def apply_pending(self):
        """Applies the pending events whose replication delay has elapsed."""
        now = time.time()
        applied = 0
        remaining = []

        for event in self._pending:
            # The event is applied only if enough time has elapsed
            age_ms = (now - event.timestamp) * 1000
            if age_ms >= self.lag_ms:
                self.data[event.key] = event.value
                self.last_applied_seq = event.sequence_number
                applied += 1
            else:
                remaining.append(event)

        self._pending = remaining
        return applied

    def read(self, key: str) -> Optional[str]:
        """Read on the follower: may return an outdated value (stale read)."""
        self.apply_pending()  # Try to apply pending events
        return self.data.get(key)


def demo_replication_lag():
    """Simulates a leader-follower system and shows the impact of replication lag.

    Scenario: A leader receives writes. Two followers replicate
    with different delays. We show the stale reads.
    """
    print("\n" + "=" * 60)
    print("  SECTION 3 : Leader-Follower Replication")
    print("=" * 60)

    # Setup: 1 leader + 2 followers with different lags
    leader = LeaderNode("leader")
    follower_fast = FollowerNode("follower-fast", lag_ms=50)   # 50ms lag
    follower_slow = FollowerNode("follower-slow", lag_ms=500)  # 500ms lag

    followers = [follower_fast, follower_slow]

    print(f"\n  Setup : 1 leader + 2 followers")
    print(f"    follower-fast : lag = {follower_fast.lag_ms}ms")
    print(f"    follower-slow : lag = {follower_slow.lag_ms}ms")

    # Phase 1: Writes on the leader
    print(f"\n  Phase 1 : 10 writes on the leader...")
    events = []
    for i in range(10):
        event = leader.write(f"key_{i}", f"value_{i}_v1")
        events.append(event)
        # Propagate to all followers
        for f in followers:
            f.receive_event(event)
        time.sleep(0.01)  # 10ms between each write

    # Phase 2: Immediate reads (before the lag has elapsed)
    print(f"\n  Phase 2 : Immediate reads after the writes")
    print(f"\n  {'Key':<10} {'Leader':>15} {'Fast Follower':>15} {'Slow Follower':>15}")
    print(f"  {'-'*58}")

    stale_fast = 0
    stale_slow = 0

    for i in range(10):
        key = f"key_{i}"
        leader_val = leader.read(key)
        fast_val = follower_fast.read(key) or "(stale/missing)"
        slow_val = follower_slow.read(key) or "(stale/missing)"

        if fast_val != leader_val:
            stale_fast += 1
        if slow_val != leader_val:
            stale_slow += 1

        print(f"  {key:<10} {str(leader_val):>15} {str(fast_val):>15} {str(slow_val):>15}")

    print(f"\n  Stale reads : fast={stale_fast}/10, slow={stale_slow}/10")

    # Phase 3: Wait for the lag to pass, then read again
    print(f"\n  Phase 3 : Waiting 600ms so that all followers catch up...")
    time.sleep(0.6)

    print(f"\n  {'Key':<10} {'Leader':>15} {'Fast Follower':>15} {'Slow Follower':>15}")
    print(f"  {'-'*58}")

    for i in range(10):
        key = f"key_{i}"
        leader_val = leader.read(key)
        fast_val = follower_fast.read(key) or "(missing)"
        slow_val = follower_slow.read(key) or "(missing)"
        print(f"  {key:<10} {str(leader_val):>15} {str(fast_val):>15} {str(slow_val):>15}")

    print(f"\n  >> After the lag, all followers are up to date.")

    # Phase 4: Read-your-writes pattern
    print(f"\n  Phase 4 : The 'read-your-writes' problem")
    print(f"  {'-'*50}")

    # The user writes an update
    event = leader.write("profile_name", "Alex_Updated")
    for f in followers:
        f.receive_event(event)

    # Immediate read from a follower (simulates a load balancer)
    fast_read = follower_fast.read("profile_name") or "(not yet replicated)"
    print(f"\n  Write : profile_name = 'Alex_Updated' (on leader)")
    print(f"  Immediate read from follower-fast : '{fast_read}'")
    print(f"  The user sees their old profile even though they just updated it!")
    print(f"\n  Solution : 'read-your-writes consistency'")
    print(f"  -> After a write, force the reads of THIS user toward the leader")
    print(f"     for {follower_slow.lag_ms}ms (the duration of the worst lag).")


# =============================================================================
# SECTION 4 : DB Selection Advisor
# =============================================================================


@dataclass
class DBRequirements:
    """A system's requirements for choosing a DB."""
    name: str
    needs_acid: bool = False               # ACID transactions required
    needs_joins: bool = False              # Complex joins
    schema_flexible: bool = False          # Variable schema
    write_heavy: bool = False              # > 80% writes
    read_heavy: bool = False               # > 80% reads
    needs_full_text_search: bool = False   # Full-text search
    needs_graph_traversal: bool = False    # Deep relationships
    needs_sub_ms_latency: bool = False     # Latency < 1ms
    data_is_time_series: bool = False      # Time-series data
    expected_qps: int = 0                  # Expected QPS
    data_size_gb: int = 0                  # Data volume in GB
    ttl_required: bool = False             # Automatic expiration


@dataclass
class DBRecommendation:
    """Advisor result: a recommended DB with a justification."""
    db_name: str
    category: str
    justification: str
    warning: str = ""


def recommend_db(req: DBRequirements) -> list[DBRecommendation]:
    """Recommends one or more databases based on the requirements.

    Decision tree based on the critical constraints:
    1. ACID + joins -> SQL
    2. Sub-ms latency -> Redis
    3. Graph traversal -> Neo4j
    4. Write-heavy + time-series -> Cassandra/TimescaleDB
    5. Full-text search -> Elasticsearch (+ primary DB)
    6. Flexible schema -> MongoDB
    7. Default -> PostgreSQL
    """
    recommendations = []

    # Rule 1: Need for ACID transactions and joins -> SQL
    if req.needs_acid and req.needs_joins:
        rec = DBRecommendation(
            db_name="PostgreSQL",
            category="OLTP SQL",
            justification=(
                f"ACID transactions + complex joins required. "
                f"PostgreSQL supports up to ~50K QPS per node. "
                f"For {req.expected_qps:,} QPS, "
                f"{'1 node is enough' if req.expected_qps <= 50_000 else 'sharding will be needed (Citus)'}."
            ),
        )
        if req.data_size_gb > 5_000:
            rec.warning = (
                f"Warning: {req.data_size_gb:,} GB is large for a single PostgreSQL. "
                f"Consider Citus (distributed PostgreSQL) or native partitioning."
            )
        recommendations.append(rec)

    # Rule 2: Sub-ms latency -> Redis
    if req.needs_sub_ms_latency:
        recommendations.append(DBRecommendation(
            db_name="Redis",
            category="Key-Value (in-memory)",
            justification=(
                f"Latency < 1ms required. Redis offers ~100K-1M ops/s per node. "
                f"Ideal for {req.name} if the data fits in RAM "
                f"({req.data_size_gb} GB {'< 100 GB : OK' if req.data_size_gb < 100 else '> 100 GB : Redis Cluster required'})."
            ),
            warning="Redis loses data on crash without persistence (RDB/AOF). Not a primary DB." if not req.ttl_required else "",
        ))

    # Rule 3: Graph traversal -> Neo4j
    if req.needs_graph_traversal:
        recommendations.append(DBRecommendation(
            db_name="Neo4j",
            category="Graph",
            justification=(
                "Deep relationship traversals (friends of friends, recommendations, fraud). "
                "In SQL, a recursive join over 4+ levels is impractical. "
                "Neo4j traverses in ms what would take seconds in SQL."
            ),
        ))

    # Rule 4: Write-heavy + time-series -> Cassandra or TimescaleDB
    if req.write_heavy and req.data_is_time_series:
        recommendations.append(DBRecommendation(
            db_name="Cassandra (or TimescaleDB if SQL is needed)",
            category="Column-family / Time-series",
            justification=(
                f"Write-heavy ({req.expected_qps:,} QPS) + time-series. "
                "Cassandra: LSM-Tree optimized for massive writes, "
                "linear scaling. Partition by sensor_id + clustering by timestamp. "
                "TimescaleDB if you want to keep SQL (PostgreSQL extension)."
            ),
        ))
    elif req.write_heavy:
        recommendations.append(DBRecommendation(
            db_name="Cassandra / ScyllaDB",
            category="Column-family",
            justification=(
                f"Write-heavy ({req.expected_qps:,} QPS). "
                "Cassandra/ScyllaDB are optimized for massive writes "
                "thanks to the LSM-Tree (append-only, no random writes)."
            ),
        ))

    # Rule 5: Full-text search -> Elasticsearch
    if req.needs_full_text_search:
        recommendations.append(DBRecommendation(
            db_name="Elasticsearch (complement, not a primary DB)",
            category="Search engine",
            justification=(
                "Full-text search with scoring, fuzzy matching, facets. "
                "Elasticsearch uses an inverted index optimized for this. "
                "To be paired with a primary DB (PostgreSQL or MongoDB)."
            ),
            warning="Elasticsearch is NOT a primary DB. No transactions, possible data loss on crash.",
        ))

    # Rule 6: Flexible schema -> MongoDB
    if req.schema_flexible and not req.needs_acid:
        recommendations.append(DBRecommendation(
            db_name="MongoDB",
            category="Document store",
            justification=(
                f"Variable schema for {req.name}. Each document can have "
                "different attributes. No DDL migrations. "
                "Fast iteration. Good for catalogs, CMS, enriched profiles."
            ),
        ))

    # Default: PostgreSQL if no strong recommendation
    if not recommendations:
        recommendations.append(DBRecommendation(
            db_name="PostgreSQL",
            category="OLTP SQL (safe default)",
            justification=(
                "No strong constraint identified. PostgreSQL is the most versatile "
                "default choice: ACID, JSON(B), basic full-text search, "
                "extensions (PostGIS, TimescaleDB, pgvector). You can specialize later."
            ),
        ))

    return recommendations


def demo_db_advisor():
    """Runs the advisor on several scenarios and displays the recommendations."""
    print("\n" + "=" * 60)
    print("  SECTION 4 : DB Selection Advisor")
    print("=" * 60)

    scenarios = [
        DBRequirements(
            name="E-commerce order system",
            needs_acid=True, needs_joins=True,
            expected_qps=5_000, data_size_gb=200,
        ),
        DBRequirements(
            name="User session cache",
            needs_sub_ms_latency=True, ttl_required=True,
            expected_qps=100_000, data_size_gb=10,
        ),
        DBRequirements(
            name="IoT platform (500K sensors)",
            write_heavy=True, data_is_time_series=True,
            expected_qps=500_000, data_size_gb=50_000,
        ),
        DBRequirements(
            name="Social network — fraud detection",
            needs_graph_traversal=True,
            expected_qps=10_000, data_size_gb=500,
        ),
        DBRequirements(
            name="Marketplace product catalog",
            schema_flexible=True, needs_full_text_search=True,
            read_heavy=True,
            expected_qps=50_000, data_size_gb=100,
        ),
        DBRequirements(
            name="Simple notes app (10K users)",
            expected_qps=100, data_size_gb=1,
        ),
    ]

    for req in scenarios:
        recs = recommend_db(req)
        print(f"\n  {'='*56}")
        print(f"  Scenario : {req.name}")
        print(f"  QPS: {req.expected_qps:,} | Data: {req.data_size_gb:,} GB")
        print(f"  {'='*56}")

        for i, rec in enumerate(recs, 1):
            print(f"\n  Recommendation {i} : {rec.db_name} ({rec.category})")
            print(f"  Justification : {rec.justification}")
            if rec.warning:
                print(f"  /!\\ Warning : {rec.warning}")


# =============================================================================
# SECTION 5 : Sharding Simulator — Distribution and Hot Spots
# =============================================================================


def demo_sharding_simulator():
    """Simulates the distribution of keys across shards and shows hot spots.

    Compares range sharding (prone to hot spots) and hash sharding
    (more uniform distribution).
    """
    print("\n" + "=" * 60)
    print("  SECTION 5 : Sharding Simulator — Hot Spots")
    print("=" * 60)

    NUM_SHARDS = 4
    NUM_KEYS = 10_000

    # --- Scenario 1: Range Sharding ---
    # Recent users (high IDs) are the most active
    print(f"\n  Scenario 1 : Range Sharding")
    print(f"  {'-'*50}")

    # Simulate realistic traffic: recent users are more active
    # Zipf-like distribution: high user_id = more requests
    requests_range = []
    for _ in range(NUM_KEYS):
        # 80% of the traffic comes from the 20% most recent users (high IDs)
        if random.random() < 0.8:
            user_id = random.randint(7500, 10000)  # Recent users
        else:
            user_id = random.randint(1, 7500)       # Older users

        # Range sharding: each shard covers a range of IDs
        shard_size = 10000 // NUM_SHARDS
        shard = min(user_id // shard_size, NUM_SHARDS - 1)
        requests_range.append((user_id, f"shard-{shard}"))

    # Count the requests per shard
    range_dist = defaultdict(int)
    for _, shard in requests_range:
        range_dist[shard] += 1

    print(f"\n  Traffic distribution (range sharding, {NUM_KEYS:,} requests) :")
    max_load = max(range_dist.values())
    for shard in sorted(range_dist.keys()):
        count = range_dist[shard]
        pct = count / NUM_KEYS * 100
        bar = "#" * int(pct)
        hot = " << HOT SPOT!" if count > NUM_KEYS / NUM_SHARDS * 1.5 else ""
        print(f"    {shard:<10} : {count:>5} ({pct:>5.1f}%) {bar}{hot}")

    # --- Scenario 2: Hash Sharding ---
    print(f"\n  Scenario 2 : Hash Sharding")
    print(f"  {'-'*50}")

    hash_dist = defaultdict(int)
    for user_id, _ in requests_range:
        # Hash the user_id to distribute uniformly
        h = int(hashlib.md5(str(user_id).encode()).hexdigest()[:8], 16)
        shard = f"shard-{h % NUM_SHARDS}"
        hash_dist[shard] += 1

    print(f"\n  Traffic distribution (hash sharding, same requests) :")
    for shard in sorted(hash_dist.keys()):
        count = hash_dist[shard]
        pct = count / NUM_KEYS * 100
        bar = "#" * int(pct)
        print(f"    {shard:<10} : {count:>5} ({pct:>5.1f}%) {bar}")

    # --- Comparison ---
    print(f"\n  Comparison :")
    range_std = (sum((c - NUM_KEYS / NUM_SHARDS) ** 2 for c in range_dist.values()) / NUM_SHARDS) ** 0.5
    hash_std = (sum((c - NUM_KEYS / NUM_SHARDS) ** 2 for c in hash_dist.values()) / NUM_SHARDS) ** 0.5
    print(f"    Range sharding — std deviation : {range_std:,.0f} ({range_std / (NUM_KEYS / NUM_SHARDS) * 100:.1f}%)")
    print(f"    Hash sharding  — std deviation : {hash_std:,.0f} ({hash_std / (NUM_KEYS / NUM_SHARDS) * 100:.1f}%)")

    # --- Scenario 3: Celebrity problem (hot key) ---
    print(f"\n  Scenario 3 : Celebrity Problem (hot key)")
    print(f"  {'-'*50}")

    celebrity_dist = defaultdict(int)
    # 10,000 normal requests + 5,000 requests for a SINGLE user (viral)
    for _ in range(NUM_KEYS):
        user_id = random.randint(1, 10000)
        h = int(hashlib.md5(str(user_id).encode()).hexdigest()[:8], 16)
        shard = f"shard-{h % NUM_SHARDS}"
        celebrity_dist[shard] += 1

    # The celebrity: user_id = 42 receives 50% of the total traffic
    celebrity_hash = int(hashlib.md5(b"42").hexdigest()[:8], 16)
    celebrity_shard = f"shard-{celebrity_hash % NUM_SHARDS}"
    celebrity_dist[celebrity_shard] += NUM_KEYS  # Double the traffic on this shard

    total_with_celeb = sum(celebrity_dist.values())
    print(f"\n  Distribution with a 'celebrity user' (user_id=42, +{NUM_KEYS:,} requests) :")
    for shard in sorted(celebrity_dist.keys()):
        count = celebrity_dist[shard]
        pct = count / total_with_celeb * 100
        bar = "#" * int(pct)
        hot = " << CELEBRITY SHARD!" if shard == celebrity_shard else ""
        print(f"    {shard:<10} : {count:>5} ({pct:>5.1f}%) {bar}{hot}")

    print(f"\n  >> 'Key splitting' solution for the celebrity :")
    print(f"     Instead of hash('42'), use hash('42_0'), hash('42_1'), ..., hash('42_9')")
    print(f"     Reads are distributed across 10 keys instead of a single one.")

    # Simulate key splitting
    split_dist = defaultdict(int)
    # Redistribute the celebrity's requests across 10 keys
    for shard in sorted(celebrity_dist.keys()):
        if shard != celebrity_shard:
            split_dist[shard] = celebrity_dist[shard]
        else:
            # The celebrity's shard keeps its normal requests
            split_dist[shard] = celebrity_dist[shard] - NUM_KEYS

    # The celebrity's requests are distributed across 10 split keys
    for i in range(NUM_KEYS):
        split_key = f"42_{i % 10}"
        h = int(hashlib.md5(split_key.encode()).hexdigest()[:8], 16)
        shard = f"shard-{h % NUM_SHARDS}"
        split_dist[shard] += 1

    total_split = sum(split_dist.values())
    print(f"\n  Distribution after key splitting :")
    for shard in sorted(split_dist.keys()):
        count = split_dist[shard]
        pct = count / total_split * 100
        bar = "#" * int(pct)
        print(f"    {shard:<10} : {count:>5} ({pct:>5.1f}%) {bar}")

    print(f"\n  >> The hot spot is eliminated. The celebrity's traffic is spread uniformly.")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Runs all the demos sequentially."""
    print("\n" + "#" * 60)
    print("#  DAY 2 -- STORAGE & DATABASES")
    print("#" * 60)

    demo_btree_vs_fullscan()
    demo_consistent_hashing()
    demo_replication_lag()
    demo_db_advisor()
    demo_sharding_simulator()

    print("\n" + "#" * 60)
    print("#  DONE -- All demos have been executed.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
