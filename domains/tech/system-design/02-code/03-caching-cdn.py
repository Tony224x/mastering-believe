"""
Day 3 -- Caching & CDN
Interactive demonstrations in Python.

Usage:
    python 03-caching-cdn.py

Each section is independent and can be executed via the main() function.
"""

import time
import random
import math
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Any

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : LRU Cache — Implementation with OrderedDict
# =============================================================================


class LRUCacheOrderedDict:
    """LRU Cache using Python's OrderedDict.

    OrderedDict maintains insertion order. We use move_to_end()
    to move an accessed element to the end (= most recent).
    The element at the front (= least recent) is evicted when the cache is full.

    Complexity: O(1) for get and put thanks to the internal dict + doubly-linked list.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()  # Order = recency (front = LRU, end = MRU)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieves a value. Returns None if absent."""
        if key in self.cache:
            # Move to the end = mark as "recently used"
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> Optional[str]:
        """Inserts or updates. Returns the evicted key if an eviction happened."""
        evicted = None
        if key in self.cache:
            # Update: move to the end
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            if len(self.cache) >= self.capacity:
                # Evict the oldest (front of the OrderedDict)
                evicted, _ = self.cache.popitem(last=False)  # last=False = FIFO order = LRU
            self.cache[key] = value
        return evicted

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return f"LRUCache(size={len(self.cache)}/{self.capacity}, hit_rate={self.hit_rate:.1%})"


# =============================================================================
# SECTION 2 : LRU Cache — Implementation from scratch (Doubly-Linked List + Dict)
# =============================================================================


class DLLNode:
    """Node of a doubly-linked list.

    Each node stores a key-value pair and prev/next pointers.
    The key is stored in the node so the key can be recovered
    during eviction (we evict the node, but we must also delete from the dict).
    """

    __slots__ = ("key", "value", "prev", "next")  # Memory savings

    def __init__(self, key: str = "", value: Any = None):
        self.key = key
        self.value = value
        self.prev: Optional["DLLNode"] = None
        self.next: Optional["DLLNode"] = None


class LRUCacheFromScratch:
    """LRU Cache implemented without OrderedDict.

    Internal structure:
    - A dict (key -> DLLNode) for O(1) access by key
    - A doubly-linked list for O(1) access ordering
      - head.next = least recent node (LRU)
      - tail.prev = most recent node (MRU)
    - head and tail are sentinel (dummy) nodes to simplify the operations

    Why this is asked in interviews:
    It tests mastery of pointers, data structures,
    and the ability to combine two structures to get O(1) everywhere.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[str, DLLNode] = {}
        self.hits = 0
        self.misses = 0

        # Sentinel nodes: avoid edge cases (empty list, single element)
        self.head = DLLNode()  # Dummy head — the node after head is the LRU
        self.tail = DLLNode()  # Dummy tail — the node before tail is the MRU
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: DLLNode) -> None:
        """Removes a node from the list (O(1)).

        Before : ... <-> prev <-> node <-> next <-> ...
        After  : ... <-> prev <-> next <-> ...
        """
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_tail(self, node: DLLNode) -> None:
        """Adds a node just before the tail (= MRU position) (O(1)).

        Before : ... <-> last_real <-> tail
        After  : ... <-> last_real <-> node <-> tail
        """
        prev_node = self.tail.prev  # The former last real node
        prev_node.next = node
        node.prev = prev_node
        node.next = self.tail
        self.tail.prev = node

    def get(self, key: str) -> Optional[Any]:
        """Retrieves a value and moves the node to the MRU position."""
        if key in self.cache:
            node = self.cache[key]
            # Move to the MRU position (remove then re-add at the end)
            self._remove(node)
            self._add_to_tail(node)
            self.hits += 1
            return node.value
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> Optional[str]:
        """Inserts or updates. Returns the evicted key if an eviction happened."""
        evicted = None
        if key in self.cache:
            # Update: remove and re-add at the end
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_tail(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evict the LRU (the node just after head)
                lru_node = self.head.next
                self._remove(lru_node)
                del self.cache[lru_node.key]
                evicted = lru_node.key

            # Create a new node and add it at the end
            new_node = DLLNode(key, value)
            self.cache[key] = new_node
            self._add_to_tail(new_node)
        return evicted

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# =============================================================================
# SECTION 3 : LFU Cache — Implementation from scratch
# =============================================================================


class LFUCache:
    """Least Frequently Used Cache.

    Internal structure:
    - key_to_val: dict[key -> value]          — O(1) access to the value
    - key_to_freq: dict[key -> frequency]     — frequency counter per key
    - freq_to_keys: dict[freq -> OrderedDict] — ordered set of keys per frequency
    - min_freq: int                           — the current minimum frequency

    The per-frequency OrderedDict makes it possible to find the LRU among keys
    of the same frequency (on a tie, we evict the least recently used).

    Complexity: O(1) for get and put.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key_to_val: dict[str, Any] = {}
        self.key_to_freq: dict[str, int] = {}
        self.freq_to_keys: dict[int, OrderedDict] = defaultdict(OrderedDict)
        self.min_freq = 0  # Lowest frequency currently in the cache
        self.hits = 0
        self.misses = 0

    def _update_freq(self, key: str) -> None:
        """Increments a key's frequency and maintains the structures."""
        freq = self.key_to_freq[key]
        new_freq = freq + 1

        # Remove from the old frequency bucket
        del self.freq_to_keys[freq][key]

        # If the old bucket is empty and it was the minimum freq, increment min_freq
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq = new_freq

        # Add to the new frequency bucket
        self.freq_to_keys[new_freq][key] = None  # OrderedDict as an ordered set
        self.key_to_freq[key] = new_freq

    def get(self, key: str) -> Optional[Any]:
        """Retrieves a value and increments its frequency."""
        if key not in self.key_to_val:
            self.misses += 1
            return None
        self.hits += 1
        self._update_freq(key)
        return self.key_to_val[key]

    def put(self, key: str, value: Any) -> Optional[str]:
        """Inserts or updates. Returns the evicted key if an eviction happened."""
        if self.capacity <= 0:
            return None

        evicted = None

        if key in self.key_to_val:
            # Update of an existing key
            self.key_to_val[key] = value
            self._update_freq(key)
        else:
            if len(self.key_to_val) >= self.capacity:
                # Evict the key with the lowest frequency (and the oldest on a tie)
                # popitem(last=False) gives the oldest in the OrderedDict
                evicted_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
                if not self.freq_to_keys[self.min_freq]:
                    del self.freq_to_keys[self.min_freq]
                del self.key_to_val[evicted_key]
                del self.key_to_freq[evicted_key]
                evicted = evicted_key

            # New key: frequency = 1, min_freq = 1
            self.key_to_val[key] = value
            self.key_to_freq[key] = 1
            self.freq_to_keys[1][key] = None
            self.min_freq = 1

        return evicted

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# =============================================================================
# SECTION 4 : Cache-Aside Pattern — Simulation with metrics
# =============================================================================


class SimulatedDB:
    """Simulates a database with a configurable latency."""

    def __init__(self, latency_ms: float = 10.0):
        self.latency_ms = latency_ms
        self.data: dict[str, Any] = {}
        self.query_count = 0

    def get(self, key: str) -> Optional[Any]:
        """Simulates a SELECT with the latency."""
        time.sleep(self.latency_ms / 1000)  # Convert ms to seconds
        self.query_count += 1
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        """Simulates an INSERT/UPDATE."""
        time.sleep(self.latency_ms / 1000)
        self.data[key] = value


class SimulatedRedis:
    """Simulates a Redis cache with TTL and latency."""

    def __init__(self, latency_ms: float = 0.5):
        self.latency_ms = latency_ms
        self.data: dict[str, Any] = {}
        self.expiry: dict[str, float] = {}  # key -> expiration timestamp
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """GET with TTL check."""
        time.sleep(self.latency_ms / 1000)

        if key in self.data:
            # Check whether the key has expired
            if key in self.expiry and time.time() > self.expiry[key]:
                # Expired key -> delete and return None
                del self.data[key]
                del self.expiry[key]
                self.misses += 1
                return None
            self.hits += 1
            return self.data[key]
        self.misses += 1
        return None

    def setex(self, key: str, ttl_seconds: int, value: Any) -> None:
        """SET with TTL in seconds."""
        time.sleep(self.latency_ms / 1000)
        self.data[key] = value
        self.expiry[key] = time.time() + ttl_seconds

    def delete(self, key: str) -> None:
        """DELETE a key."""
        time.sleep(self.latency_ms / 1000)
        self.data.pop(key, None)
        self.expiry.pop(key, None)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheAsideService:
    """Service with the Cache-Aside pattern.

    READ flow:
    1. Check the cache
    2. If HIT -> return
    3. If MISS -> read the DB, write into the cache, return

    WRITE flow:
    1. Write to the DB
    2. Invalidate the cache (DELETE, not SET, to avoid race conditions)
    """

    def __init__(self, db: SimulatedDB, cache: SimulatedRedis, ttl: int = 300):
        self.db = db
        self.cache = cache
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """Cache-aside read."""
        # 1. Check the cache
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        # 2. Cache miss -> read the DB
        value = self.db.get(key)
        if value is not None:
            # 3. Write into the cache
            self.cache.setex(key, self.ttl, value)
        return value

    def update(self, key: str, value: Any) -> None:
        """Cache-aside write: DB first, then cache invalidation."""
        # 1. Write to the DB
        self.db.set(key, value)
        # 2. Invalidate the cache (DELETE, not SET!)
        self.cache.delete(key)


# =============================================================================
# SECTION 5 : Write-Through vs Write-Behind — Comparison
# =============================================================================


class WriteThroughService:
    """Write-Through: each write goes through the cache AND the DB synchronously.

    The cache is always up to date, but writes are slower
    because we wait for confirmation from both (cache + DB).
    """

    def __init__(self, db: SimulatedDB, cache: SimulatedRedis, ttl: int = 300):
        self.db = db
        self.cache = cache
        self.ttl = ttl
        self.write_count = 0

    def write(self, key: str, value: Any) -> float:
        """Synchronous write to cache + DB. Returns the total time."""
        start = time.time()
        # Synchronous write: cache first, then DB
        self.cache.setex(key, self.ttl, value)
        self.db.set(key, value)
        self.write_count += 1
        return (time.time() - start) * 1000  # ms

    def read(self, key: str) -> Optional[Any]:
        """Direct read from the cache (always up to date)."""
        return self.cache.get(key)


class WriteBehindService:
    """Write-Behind: writes go into the cache, then are flushed in batch to the DB.

    Writes are very fast (cache only), but there is a risk
    of data loss if the cache crashes before the flush.
    """

    def __init__(self, db: SimulatedDB, cache: SimulatedRedis,
                 ttl: int = 300, flush_interval: float = 1.0):
        self.db = db
        self.cache = cache
        self.ttl = ttl
        self.flush_interval = flush_interval  # Seconds between each flush
        self.write_buffer: list[tuple[str, Any]] = []  # Buffer of pending writes
        self.write_count = 0
        self.flushed_count = 0

    def write(self, key: str, value: Any) -> float:
        """Write into the cache only. The buffer will be flushed later."""
        start = time.time()
        self.cache.setex(key, self.ttl, value)
        self.write_buffer.append((key, value))  # Add to the buffer
        self.write_count += 1
        return (time.time() - start) * 1000  # ms

    def flush(self) -> int:
        """Flushes the buffer to the DB. Returns the number of writes."""
        count = 0
        for key, value in self.write_buffer:
            self.db.set(key, value)
            count += 1
        self.flushed_count += count
        self.write_buffer.clear()
        return count

    def read(self, key: str) -> Optional[Any]:
        """Read from the cache (which is ahead of the DB)."""
        return self.cache.get(key)


# =============================================================================
# SECTION 6 : Cache Stampede — Simulation and Fix (Locking)
# =============================================================================


class StampedeDemo:
    """Demonstration of the cache stampede and the locking-based solution.

    Without lock: N threads all hit a simultaneous cache miss and query the DB.
    With lock: a single thread rebuilds the cache, the others wait.
    """

    def __init__(self):
        self.db_query_count = 0
        self.lock = threading.Lock()
        self.cache: dict[str, Any] = {}
        self.db_latency_ms = 50  # The DB is slow

    def _db_query(self, key: str) -> str:
        """Simulates an expensive DB query."""
        time.sleep(self.db_latency_ms / 1000)
        self.db_query_count += 1
        return f"value_for_{key}"

    def get_without_lock(self, key: str) -> str:
        """GET without protection against the stampede.

        All the threads that arrive during a cache miss
        will query the DB in parallel -> overload.
        """
        if key in self.cache:
            return self.cache[key]

        # Cache miss -> query the DB (ALL threads do this)
        value = self._db_query(key)
        self.cache[key] = value
        return value

    def get_with_lock(self, key: str) -> str:
        """GET with a mutex to avoid the stampede.

        A single thread rebuilds the cache.
        The others wait for the lock to be released then read the cache.
        """
        if key in self.cache:
            return self.cache[key]

        # Try to acquire the lock
        with self.lock:
            # Double-check: another thread may have already rebuilt it
            if key in self.cache:
                return self.cache[key]

            # This thread (and only this one) rebuilds the cache
            value = self._db_query(key)
            self.cache[key] = value
            return value

    def reset(self):
        """Resets the counters and the cache for a new test."""
        self.db_query_count = 0
        self.cache.clear()


# =============================================================================
# SECTION 7 : TTL-based Cache with automatic expiration
# =============================================================================


class TTLCache:
    """Cache with a TTL (Time-To-Live) per entry.

    Each entry has an expiration timestamp. Expired entries
    are removed lazily (lazy deletion) on access,
    and periodically (active deletion) via cleanup().

    Lazy deletion = on every GET we check whether the entry has expired.
    Active deletion = we periodically scan to remove expired entries.
    Redis uses both strategies combined.
    """

    def __init__(self, default_ttl: int = 60):
        self.default_ttl = default_ttl
        self.data: dict[str, Any] = {}
        self.expiry: dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self.expired_count = 0

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set with an optional TTL (default = default_ttl)."""
        self.data[key] = value
        effective_ttl = ttl if ttl is not None else self.default_ttl
        self.expiry[key] = time.time() + effective_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get with lazy deletion of expired entries."""
        if key in self.data:
            if time.time() > self.expiry.get(key, float("inf")):
                # Lazy deletion: remove the expired entry
                del self.data[key]
                del self.expiry[key]
                self.expired_count += 1
                self.misses += 1
                return None
            self.hits += 1
            return self.data[key]
        self.misses += 1
        return None

    def cleanup(self) -> int:
        """Active deletion: scan and remove expired entries.

        Redis does this periodically (10 times/sec) by sampling
        20 random keys among those with a TTL.

        Returns the number of removed entries.
        """
        now = time.time()
        expired_keys = [k for k, exp in self.expiry.items() if now > exp]
        for key in expired_keys:
            del self.data[key]
            del self.expiry[key]
            self.expired_count += 1
        return len(expired_keys)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self.data)


# =============================================================================
# SECTION 8 : CDN Simulator — Edge -> Regional -> Origin
# =============================================================================


@dataclass
class CDNNode:
    """A CDN node (edge, regional, or origin).

    Each node has its own cache, its access latency,
    and a parent (upstream) node for cache misses.
    """
    name: str
    latency_ms: float           # Latency to reach this node
    cache: dict = field(default_factory=dict)
    cache_ttl: dict = field(default_factory=dict)
    upstream: Optional["CDNNode"] = None  # Parent node (for misses)
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> tuple[Any, float, list[str]]:
        """Retrieves a resource. Goes up the levels on a miss.

        Returns (value, total_latency_ms, traversed_path).
        The path shows which nodes were traversed (useful for debugging).
        """
        total_latency = self.latency_ms  # Latency to reach this node
        path = [self.name]

        # Check the local cache
        if key in self.cache:
            # Check the TTL
            if key in self.cache_ttl and time.time() > self.cache_ttl[key]:
                # Expired -> delete
                del self.cache[key]
                del self.cache_ttl[key]
            else:
                self.hits += 1
                return self.cache[key], total_latency, path

        self.misses += 1

        # Cache miss -> go up to the upstream
        if self.upstream:
            value, upstream_latency, upstream_path = self.upstream.get(key)
            total_latency += upstream_latency
            path.extend(upstream_path)

            # Store in the local cache for future accesses
            if value is not None:
                self.cache[key] = value
                self.cache_ttl[key] = time.time() + 60  # Default TTL 60s

            return value, total_latency, path

        # No upstream = we are the origin, the data does not exist
        return None, total_latency, path

    def put_origin(self, key: str, value: Any) -> None:
        """Adds a resource to the origin (no TTL for the origin)."""
        self.cache[key] = value


class CDNSimulator:
    """Simulates a 3-level CDN: Edge -> Regional -> Origin.

    Architecture:
        [Client] -> [Edge Paris/Tokyo] -> [Regional EU/APAC] -> [Origin US]

    Typical latencies:
        - Edge (same city) : 2-5 ms
        - Regional (same continent) : 15-30 ms
        - Origin (intercontinental) : 80-150 ms
    """

    def __init__(self):
        # Origin (source of truth)
        self.origin = CDNNode(name="Origin-US", latency_ms=100.0)

        # Regional caches (one per continent)
        self.regional_eu = CDNNode(name="Regional-EU", latency_ms=25.0, upstream=self.origin)
        self.regional_apac = CDNNode(name="Regional-APAC", latency_ms=30.0, upstream=self.origin)

        # Edge caches (one per city)
        self.edge_paris = CDNNode(name="Edge-Paris", latency_ms=3.0, upstream=self.regional_eu)
        self.edge_tokyo = CDNNode(name="Edge-Tokyo", latency_ms=4.0, upstream=self.regional_apac)
        self.edge_nyc = CDNNode(name="Edge-NYC", latency_ms=2.0, upstream=self.origin)

    def populate_origin(self, resources: dict[str, Any]) -> None:
        """Loads resources into the origin."""
        for key, value in resources.items():
            self.origin.put_origin(key, value)


# =============================================================================
# DEMONSTRATIONS
# =============================================================================


def demo_lru_caches():
    """Compares the two LRU implementations (OrderedDict vs from scratch)."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 1 & 2 : LRU Cache — OrderedDict vs Doubly-Linked List")
    print(SEPARATOR)

    # Test with the same operations on both implementations
    for name, cache_class in [("OrderedDict", LRUCacheOrderedDict),
                               ("From Scratch", LRUCacheFromScratch)]:
        cache = cache_class(capacity=3)
        print(f"\n  --- {name} (capacity=3) ---")

        # Insert 3 elements
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        print(f"  After put(a,b,c) : size = {len(cache.cache)}")

        # Access 'a' to mark it as recent
        cache.get("a")
        print(f"  get('a') -> 'a' is now the MRU")

        # Insert 'd' -> 'b' should be evicted (LRU)
        evicted = cache.put("d", 4)
        print(f"  put('d') -> evicted = '{evicted}' (expected: 'b')")

        # Check that 'b' is gone
        result = cache.get("b")
        print(f"  get('b') -> {result} (expected: None)")

        # Check that 'a' is still there
        result = cache.get("a")
        print(f"  get('a') -> {result} (expected: 1)")

    # Benchmark with Zipf accesses (realistic)
    print(f"\n  --- Benchmark : Zipf distribution (1000 keys, cache=100) ---")
    random.seed(42)

    # Generate accesses following a Zipf law (the first keys are more popular)
    num_keys = 1000
    num_accesses = 10000
    cache_size = 100

    # Zipf: P(key_i) proportional to 1/i
    weights = [1.0 / (i + 1) for i in range(num_keys)]
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # Generate the access sequence
    access_sequence = random.choices(range(num_keys), weights=probabilities, k=num_accesses)

    for name, cache_class in [("OrderedDict", LRUCacheOrderedDict),
                               ("From Scratch", LRUCacheFromScratch)]:
        cache = cache_class(capacity=cache_size)

        start = time.time()
        for key_idx in access_sequence:
            key = f"key_{key_idx}"
            result = cache.get(key)
            if result is None:
                cache.put(key, f"value_{key_idx}")
        elapsed = (time.time() - start) * 1000

        print(f"  {name:15s} : hit_rate={cache.hit_rate:.1%}, "
              f"time={elapsed:.1f}ms for {num_accesses} ops")


def demo_lfu_cache():
    """Demonstration of the LFU cache and comparison with LRU."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 3 : LFU Cache")
    print(SEPARATOR)

    lfu = LFUCache(capacity=3)

    # Insert 3 elements
    lfu.put("a", 1)  # freq(a) = 1
    lfu.put("b", 2)  # freq(b) = 1
    lfu.put("c", 3)  # freq(c) = 1

    # Access 'a' 3 times and 'b' 2 times
    lfu.get("a")  # freq(a) = 2
    lfu.get("a")  # freq(a) = 3
    lfu.get("b")  # freq(b) = 2

    # Insert 'd' -> 'c' is evicted (freq=1, the lowest)
    evicted = lfu.put("d", 4)
    print(f"\n  After accesses a=3x, b=2x, c=1x : put('d') evicts '{evicted}' (expected: 'c')")

    # Compare LFU vs LRU on a workload with hot keys
    print(f"\n  --- LFU vs LRU comparison with hot keys ---")
    random.seed(42)

    # Scenario: 10 hot keys accessed 80% of the time, 990 cold keys accessed 20% of the time
    hot_keys = [f"hot_{i}" for i in range(10)]
    cold_keys = [f"cold_{i}" for i in range(990)]
    num_ops = 10000
    cache_size = 50  # The cache can hold 50 entries

    for name, cache_class in [("LRU", LRUCacheOrderedDict), ("LFU", LFUCache)]:
        if name == "LRU":
            cache = cache_class(capacity=cache_size)
        else:
            cache = cache_class(capacity=cache_size)

        for _ in range(num_ops):
            if random.random() < 0.8:
                # 80% of the time: access a hot key
                key = random.choice(hot_keys)
            else:
                # 20% of the time: access a cold key
                key = random.choice(cold_keys)

            result = cache.get(key)
            if result is None:
                cache.put(key, f"value_for_{key}")

        print(f"  {name:5s} : hit_rate = {cache.hit_rate:.1%}")

    print(f"\n  LFU should have a better hit rate because it protects the hot keys")
    print(f"  LRU can evict a hot key if a burst of cold keys fills the cache")


def demo_cache_aside():
    """Demonstration of the Cache-Aside pattern with metrics."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 4 : Cache-Aside Pattern")
    print(SEPARATOR)

    db = SimulatedDB(latency_ms=10.0)
    cache = SimulatedRedis(latency_ms=0.5)
    service = CacheAsideService(db, cache, ttl=300)

    # Prepopulate the DB
    for i in range(100):
        db.data[f"user:{i}"] = {"id": i, "name": f"User_{i}"}

    # Simulate accesses (same users accessed multiple times)
    print(f"\n  Simulation : 200 accesses over 100 users (Zipf distribution)")
    random.seed(42)
    weights = [1.0 / (i + 1) for i in range(100)]
    access_keys = random.choices([f"user:{i}" for i in range(100)],
                                  weights=weights, k=200)

    start = time.time()
    for key in access_keys:
        service.get(key)
    elapsed = (time.time() - start) * 1000

    print(f"  Cache hit rate : {cache.hit_rate:.1%}")
    print(f"  DB queries     : {db.query_count} (out of 200 accesses)")
    print(f"  Total time     : {elapsed:.0f} ms")

    # Demonstration of invalidation
    print(f"\n  --- Invalidation after update ---")
    service.get("user:0")  # Make sure it is cached
    print(f"  get('user:0') -> cached (hit)")

    service.update("user:0", {"id": 0, "name": "User_0_updated"})
    print(f"  update('user:0') -> DB updated, cache invalidated")

    result = service.get("user:0")
    print(f"  get('user:0') -> {result['name']} (cache miss -> DB -> re-cache)")


def demo_write_through_vs_behind():
    """Compares Write-Through (synchronous) and Write-Behind (asynchronous)."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 5 : Write-Through vs Write-Behind")
    print(SEPARATOR)

    num_writes = 50

    # Write-Through
    db_wt = SimulatedDB(latency_ms=10.0)
    cache_wt = SimulatedRedis(latency_ms=0.5)
    wt_service = WriteThroughService(db_wt, cache_wt)

    start = time.time()
    for i in range(num_writes):
        wt_service.write(f"key:{i}", f"value_{i}")
    wt_time = (time.time() - start) * 1000

    # Write-Behind
    db_wb = SimulatedDB(latency_ms=10.0)
    cache_wb = SimulatedRedis(latency_ms=0.5)
    wb_service = WriteBehindService(db_wb, cache_wb)

    start = time.time()
    for i in range(num_writes):
        wb_service.write(f"key:{i}", f"value_{i}")
    wb_write_time = (time.time() - start) * 1000

    # Flush the Write-Behind buffer
    start = time.time()
    flushed = wb_service.flush()
    wb_flush_time = (time.time() - start) * 1000

    print(f"\n  {num_writes} writes :")
    print(f"  Write-Through : {wt_time:.0f} ms (synchronous cache + DB)")
    print(f"  Write-Behind  : {wb_write_time:.0f} ms (cache only)")
    print(f"  Write-Behind flush : {wb_flush_time:.0f} ms ({flushed} batched writes)")
    print(f"\n  Write speedup : {wt_time / wb_write_time:.1f}x")
    print(f"  Write-Behind risk : if crash before flush, {len(wb_service.write_buffer)} writes lost")

    # Consistency check
    print(f"\n  --- Consistency check ---")
    print(f"  Write-Through DB queries : {db_wt.query_count} (= {num_writes} writes)")
    print(f"  Write-Behind DB queries  : {db_wb.query_count} (= {flushed} after flush)")


def demo_cache_stampede():
    """Simulates a cache stampede and shows the effect of locking."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 6 : Cache Stampede — Without lock vs With lock")
    print(SEPARATOR)

    num_threads = 20
    key = "popular_item"

    # Test WITHOUT lock
    demo = StampedeDemo()
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=demo.get_without_lock, args=(key,))
        threads.append(t)

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    no_lock_time = (time.time() - start) * 1000

    print(f"\n  Without lock : {num_threads} simultaneous threads")
    print(f"  DB queries : {demo.db_query_count} (ideally 1, in reality {demo.db_query_count})")
    print(f"  Total time : {no_lock_time:.0f} ms")

    # Test WITH lock
    demo.reset()
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=demo.get_with_lock, args=(key,))
        threads.append(t)

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lock_time = (time.time() - start) * 1000

    print(f"\n  With lock : {num_threads} simultaneous threads")
    print(f"  DB queries : {demo.db_query_count} (exactly 1)")
    print(f"  Total time : {lock_time:.0f} ms")

    print(f"\n  Result : the lock reduces the DB queries from {num_threads}x to 1")
    print(f"  (in production, {num_threads} simultaneous queries could bring down the DB)")


def demo_ttl_cache():
    """Demonstration of the TTL cache with expiration."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 7 : TTL Cache with automatic expiration")
    print(SEPARATOR)

    cache = TTLCache(default_ttl=2)  # 2-second TTL for the demo

    # Insert data
    cache.set("fast", "expires quickly", ttl=1)
    cache.set("slow", "expires slowly", ttl=5)
    cache.set("default", "default ttl")  # 2 seconds

    print(f"\n  Data inserted : fast(1s), slow(5s), default(2s)")
    print(f"  Cache size : {len(cache)}")

    # Read immediately
    print(f"\n  Immediate read :")
    print(f"    fast    = {cache.get('fast')}")
    print(f"    slow    = {cache.get('slow')}")
    print(f"    default = {cache.get('default')}")
    print(f"    hit_rate = {cache.hit_rate:.1%}")

    # Wait 1.5 seconds
    time.sleep(1.5)
    print(f"\n  After 1.5 seconds :")
    print(f"    fast    = {cache.get('fast')} (expired after 1s)")
    print(f"    slow    = {cache.get('slow')} (still valid)")
    print(f"    default = {cache.get('default')} (still valid)")

    # Wait another second (2.5s total)
    time.sleep(1.0)
    print(f"\n  After 2.5 seconds :")
    print(f"    fast    = {cache.get('fast')} (expired)")
    print(f"    slow    = {cache.get('slow')} (still valid)")
    print(f"    default = {cache.get('default')} (expired after 2s)")

    # Active cleanup
    time.sleep(3.0)  # Wait for 'slow' to expire as well
    cleaned = cache.cleanup()
    print(f"\n  After 5.5 seconds + cleanup : {cleaned} entry(ies) removed")
    print(f"  Cache size : {len(cache)}")
    print(f"  Total expired : {cache.expired_count}")


def demo_cdn_simulator():
    """Simulates a multi-level CDN and measures the latencies."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 8 : CDN Simulator — Edge -> Regional -> Origin")
    print(SEPARATOR)

    cdn = CDNSimulator()

    # Load resources into the origin
    cdn.populate_origin({
        "/index.html": "<html>Homepage</html>",
        "/app.js": "console.log('app')",
        "/style.css": "body { margin: 0 }",
        "/api/feed": '{"posts": [...]}',
        "/images/logo.png": "[binary data]",
    })

    print(f"\n  5 resources loaded into the origin")

    # Simulate requests from different cities
    scenarios = [
        ("Paris  (1st access)", cdn.edge_paris, "/index.html"),
        ("Paris  (2nd access)", cdn.edge_paris, "/index.html"),
        ("Tokyo  (1st access)", cdn.edge_tokyo, "/index.html"),
        ("Paris  (app.js)    ", cdn.edge_paris, "/app.js"),
        ("Paris  (app.js 2nd)", cdn.edge_paris, "/app.js"),
        ("NYC    (index)     ", cdn.edge_nyc, "/index.html"),
        ("NYC    (index 2nd) ", cdn.edge_nyc, "/index.html"),
    ]

    print(f"\n  {'Scenario':<25} {'Latency':>10} {'Path'}")
    print(f"  {'-'*25} {'-'*10} {'-'*40}")

    for label, edge, resource in scenarios:
        value, latency, path = edge.get(resource)
        path_str = " -> ".join(path)
        hit_or_miss = "HIT" if len(path) == 1 else "MISS->fill"
        print(f"  {label:<25} {latency:>7.0f} ms  {path_str} ({hit_or_miss})")

    # Per-node stats
    print(f"\n  --- Per-node stats ---")
    nodes = [cdn.edge_paris, cdn.edge_tokyo, cdn.edge_nyc,
             cdn.regional_eu, cdn.regional_apac, cdn.origin]
    for node in nodes:
        total = node.hits + node.misses
        hit_rate = node.hits / total if total > 0 else 0
        if total > 0:
            print(f"  {node.name:<20} hits={node.hits:>2}, misses={node.misses:>2}, "
                  f"hit_rate={hit_rate:.0%}")

    # Compute the savings
    print(f"\n  --- Impact ---")
    total_requests = sum(n.hits + n.misses for n in [cdn.edge_paris, cdn.edge_tokyo, cdn.edge_nyc])
    origin_requests = cdn.origin.hits + cdn.origin.misses
    print(f"  Total requests to the edges : {total_requests}")
    print(f"  Requests reaching the origin : {origin_requests}")
    print(f"  Traffic saved on the origin : {(1 - origin_requests/total_requests)*100:.0f}%")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Runs all the demonstrations."""
    print("\n" + "=" * 70)
    print("  DAY 3 — CACHING & CDN : INTERACTIVE DEMONSTRATIONS")
    print("=" * 70)

    demo_lru_caches()
    demo_lfu_cache()
    demo_cache_aside()
    demo_write_through_vs_behind()
    demo_cache_stampede()
    demo_ttl_cache()
    demo_cdn_simulator()

    print(f"\n{'=' * 70}")
    print("  END OF DEMONSTRATIONS")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
