"""
Day 7 -- Classic designs
Interactive demonstrations in Python.

Usage:
    python 07-design-classiques.py

Mocks and tools for the 3 classic designs:
- Capacity estimation helpers (orders of magnitude)
- URL shortener: base62 encoder, in-memory sharded KV store
- Twitter timeline: fanout-on-write with an in-memory sorted set
- Chat system: mini WebSocket-like pub/sub with a connection registry

Everything is in-memory, runnable without external dependencies.
"""

import time
import uuid
import random
from bisect import insort
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : Capacity estimation helpers
# =============================================================================


def si(n: float) -> str:
    """Formats a number with suffixes (K, M, B)."""
    for unit, threshold in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= threshold:
            return f"{n/threshold:.1f}{unit}"
    return f"{n:.0f}"


def bytes_human(n: float) -> str:
    for unit, thr in [("PB", 1e15), ("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]:
        if n >= thr:
            return f"{n/thr:.1f} {unit}"
    return f"{n:.0f} B"


def estimate_qps(dau: int, actions_per_day: int, peak_factor: float = 3.0):
    """Estimates the average and peak QPS for a system.

    WHY peak_factor=3? In general, the hourly peak is 2-4x the average
    (non-uniform distribution over 24h). 3x is a reasonable order of
    magnitude for interview calculations.
    """
    daily_events = dau * actions_per_day
    avg_qps = daily_events / 86400
    peak_qps = avg_qps * peak_factor
    return avg_qps, peak_qps, daily_events


def estimate_storage(events_per_day: int, bytes_per_event: int, years: float = 1):
    """Computes the total storage."""
    daily = events_per_day * bytes_per_event
    total = daily * 365 * years
    return daily, total


# =============================================================================
# SECTION 2 : URL Shortener
# =============================================================================


BASE62_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def base62_encode(num: int) -> str:
    """Encodes an int in base62.

    WHY base62? 62^7 = 3.5 trillion possible codes in 7 chars.
    More than enough for a URL shortener. Shorter than base10 or
    base16. Accepted in URLs without encoding.
    """
    if num == 0:
        return BASE62_ALPHABET[0]
    parts = []
    while num > 0:
        parts.append(BASE62_ALPHABET[num % 62])
        num //= 62
    return "".join(reversed(parts))


def base62_decode(code: str) -> int:
    """Decodes a base62 string to an int."""
    num = 0
    for char in code:
        num = num * 62 + BASE62_ALPHABET.index(char)
    return num


class ShardedKVStore:
    """Mock of a sharded KV store (Cassandra / DynamoDB style).

    WHY sharded? To demonstrate the idea: each key is routed to
    a shard via hash(key) % N. This is what distributed DBs do
    internally.
    """

    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
        self.shards: list[dict] = [{} for _ in range(num_shards)]

    def _shard_for(self, key: str) -> int:
        return hash(key) % self.num_shards

    def put(self, key: str, value: Any):
        self.shards[self._shard_for(key)][key] = value

    def get(self, key: str) -> Optional[Any]:
        return self.shards[self._shard_for(key)].get(key)

    def size(self) -> int:
        return sum(len(s) for s in self.shards)


class URLShortenerService:
    """Complete service with a (simplified) distributed counter and a cache.

    WHY counter + cache? "Global counter" approach: we allocate unique
    ids in sequence, encoded in base62. The Redis cache means that 95%
    of the redirects are served without touching the KV store.
    """

    def __init__(self, base_url: str = "https://tiny.ly/"):
        self.base_url = base_url
        self.kv = ShardedKVStore(num_shards=4)
        self.cache: dict[str, str] = {}  # LRU simplification
        # Distributed counter: we start from an offset to get codes >= 3 chars
        self.counter = 100000
        self.stats = {"shortens": 0, "cache_hits": 0, "cache_misses": 0}

    def shorten(self, long_url: str) -> str:
        """Creates a short code for a URL."""
        self.counter += 1
        code = base62_encode(self.counter)
        self.kv.put(code, long_url)
        self.stats["shortens"] += 1
        return self.base_url + code

    def expand(self, short_url: str) -> Optional[str]:
        """Resolves a short URL to its long version."""
        code = short_url.replace(self.base_url, "")
        # Check cache first
        if code in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[code]
        # Cache miss -> go to KV
        self.stats["cache_misses"] += 1
        long_url = self.kv.get(code)
        if long_url:
            self.cache[code] = long_url  # populate cache
        return long_url


# =============================================================================
# SECTION 3 : Twitter-like timeline with fanout-on-write
# =============================================================================


@dataclass
class Tweet:
    tweet_id: str
    user_id: str
    content: str
    created_at: float = field(default_factory=time.time)


class TwitterTimeline:
    """Timeline precomputed via fanout-on-write.

    WHY fanout-on-write? Read = very frequent (100x write). We trade
    expensive writes for ultra-fast reads. The timeline is just
    an in-memory sorted set, LRANGE = O(log N).

    Twitter's real hybrid: fanout-on-write for normal users, fanout-
    on-read for celebrities with 100M followers. Here we simplify
    with fanout-on-write only.
    """

    def __init__(self, max_timeline_size: int = 100):
        self.max_timeline_size = max_timeline_size
        self.tweets: dict[str, Tweet] = {}  # tweet_id -> Tweet
        self.followers: dict[str, set[str]] = defaultdict(set)  # user -> set of followers
        # timeline[user] = list of tweet_ids sorted in reverse chronological order
        self.timelines: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=max_timeline_size))
        # Stats for the demo
        self.fanout_writes = 0

    def follow(self, follower: str, followed: str):
        self.followers[followed].add(follower)

    def post_tweet(self, user_id: str, content: str) -> str:
        """Posts a tweet and fans it out to all the followers' timelines.

        WHY synchronous fanout here? In prod, it is an asynchronous job
        (Kafka -> fanout workers). Here we simplify for the demo.
        """
        tweet = Tweet(tweet_id=str(uuid.uuid4())[:8], user_id=user_id, content=content)
        self.tweets[tweet.tweet_id] = tweet

        # Add to the author's timeline
        self.timelines[user_id].appendleft(tweet.tweet_id)

        # FANOUT: write into each follower's timeline
        for follower in self.followers[user_id]:
            self.timelines[follower].appendleft(tweet.tweet_id)
            self.fanout_writes += 1
        return tweet.tweet_id

    def get_home_timeline(self, user_id: str, limit: int = 20) -> list[Tweet]:
        """Returns a user's timeline: direct lookup in the sorted set."""
        tweet_ids = list(self.timelines[user_id])[:limit]
        return [self.tweets[tid] for tid in tweet_ids if tid in self.tweets]


# =============================================================================
# SECTION 4 : Chat system -- WebSocket-like pub/sub mock
# =============================================================================


@dataclass
class ChatMessage:
    message_id: str
    conversation_id: str
    sender_id: str
    content: str
    created_at: float = field(default_factory=time.time)


class ConnectionRegistry:
    """Registry that tracks which user is connected to which chat server.

    WHY? When Alice sends a message to Bob, we need to know which
    WebSocket server holds Bob's connection in order to route the
    message to him. In prod: Redis hash 'user -> chat_server'.
    """

    def __init__(self):
        self.online: dict[str, str] = {}  # user_id -> server_id

    def register(self, user_id: str, server_id: str):
        self.online[user_id] = server_id

    def unregister(self, user_id: str):
        self.online.pop(user_id, None)

    def is_online(self, user_id: str) -> bool:
        return user_id in self.online

    def server_for(self, user_id: str) -> Optional[str]:
        return self.online.get(user_id)


class ChatService:
    """Mini chat service with persist + delivery via the registry.

    WHY separate persist and deliver? Persist must ALWAYS succeed
    (otherwise the message is lost). Delivery is best-effort (if the
    recipient is offline, we send a push notification). In prod:
    write to Cassandra first, then Kafka for the async delivery.
    """

    def __init__(self, server_id: str, registry: ConnectionRegistry):
        self.server_id = server_id
        self.registry = registry
        self.storage: dict[str, list[ChatMessage]] = defaultdict(list)
        self.inbox: dict[str, list[ChatMessage]] = defaultdict(list)  # delivered messages per user
        self.push_queue: list[tuple[str, ChatMessage]] = []  # offline users

    def send(self, sender: str, recipient: str, content: str):
        """Sends a 1-to-1 message."""
        # Conversation id = deterministic hash of the sorted pair
        convo_id = "-".join(sorted([sender, recipient]))
        msg = ChatMessage(
            message_id=str(uuid.uuid4())[:8],
            conversation_id=convo_id,
            sender_id=sender,
            content=content,
        )
        # 1. Persist (ALWAYS first)
        self.storage[convo_id].append(msg)
        # 2. Route via the registry
        if self.registry.is_online(recipient):
            self.inbox[recipient].append(msg)
            print(f"    [{self.server_id}] delivered '{content}' to {recipient} (online)")
        else:
            # Offline -> push notification
            self.push_queue.append((recipient, msg))
            print(f"    [{self.server_id}] queued push for {recipient} (offline)")

    def history(self, user_a: str, user_b: str, limit: int = 50) -> list[ChatMessage]:
        convo_id = "-".join(sorted([user_a, user_b]))
        return self.storage[convo_id][-limit:]


# =============================================================================
# SECTION 5 : Demos
# =============================================================================


def demo_capacity_estimation():
    print(f"\n{SEPARATOR}\n  DEMO 1 : Capacity estimation helpers\n{SEPARATOR}")

    # Twitter
    print("  Twitter (200M DAU, 2 tweets/day, 50 reads/day) :")
    write_avg, write_peak, total_writes = estimate_qps(200_000_000, 2)
    read_avg, read_peak, total_reads = estimate_qps(200_000_000, 50)
    print(f"    Writes : {si(write_avg)}/s avg, {si(write_peak)}/s peak "
          f"({si(total_writes)} tweets/day)")
    print(f"    Reads  : {si(read_avg)}/s avg, {si(read_peak)}/s peak "
          f"({si(total_reads)} reads/day)")
    _, storage_year = estimate_storage(int(total_writes), 300, years=1)
    print(f"    Storage: {bytes_human(storage_year)} per year (300 B/tweet)")

    # URL shortener
    print("\n  URL Shortener (10M DAU, 5 shortens/day, 100 clicks/day) :")
    s_avg, s_peak, _ = estimate_qps(10_000_000, 5)
    c_avg, c_peak, _ = estimate_qps(10_000_000, 100)
    print(f"    Shortens : {si(s_avg)}/s avg, {si(s_peak)}/s peak")
    print(f"    Clicks   : {si(c_avg)}/s avg, {si(c_peak)}/s peak")

    # Chat
    print("\n  Chat (100M DAU, 10 messages/day) :")
    m_avg, m_peak, total_msgs = estimate_qps(100_000_000, 10)
    print(f"    Messages : {si(m_avg)}/s avg, {si(m_peak)}/s peak")
    _, storage_chat = estimate_storage(int(total_msgs), 300, years=1)
    print(f"    Storage  : {bytes_human(storage_chat)} per year (300 B/msg)")


def demo_url_shortener():
    print(f"\n{SEPARATOR}\n  DEMO 2 : URL Shortener\n{SEPARATOR}")
    svc = URLShortenerService()
    urls = [
        "https://example.com/very/long/path/to/some/article?ref=123",
        "https://another.com/blog/2026/04/the-best-post",
        "https://acme-corp.example/products/acme-base",
    ]
    shorts = []
    for u in urls:
        s = svc.shorten(u)
        shorts.append(s)
        print(f"  Shortened : {u[:40]}... -> {s}")

    print(f"\n  Resolving the 3 URLs twice (first = miss, second = hit) :")
    for s in shorts:
        svc.expand(s)
    for s in shorts:
        svc.expand(s)
    print(f"  Stats : {svc.stats}")
    print(f"  KV shards sizes : {[len(shard) for shard in svc.kv.shards]}")
    # Note: with only 3 URLs, the shard # depends on the hash. OK for the demo.


def demo_twitter_fanout():
    print(f"\n{SEPARATOR}\n  DEMO 3 : Twitter fanout-on-write\n{SEPARATOR}")
    tw = TwitterTimeline(max_timeline_size=50)

    # Alice has 5 followers, Bob has 2
    for i in range(5):
        tw.follow(f"user-{i}", "alice")
    tw.follow("alice", "bob")
    tw.follow("charlie", "bob")

    # Alice tweets 3 times
    tw.post_tweet("alice", "hello world")
    tw.post_tweet("alice", "system design is fun")
    tw.post_tweet("alice", "check out my new blog post")

    # Bob tweets once
    tw.post_tweet("bob", "good morning")

    print(f"  Fanout writes performed : {tw.fanout_writes}")
    print(f"  (3 alice tweets * 5 followers + 1 bob tweet * 2 followers = 17)")

    # Each follower sees alice in their timeline
    for user in ["user-0", "user-1", "alice", "charlie"]:
        tl = tw.get_home_timeline(user, limit=10)
        print(f"  Timeline of {user} : {[t.content for t in tl]}")


def demo_chat_system():
    print(f"\n{SEPARATOR}\n  DEMO 4 : Chat system\n{SEPARATOR}")
    registry = ConnectionRegistry()
    # 2 chat servers to simulate the partitioning
    server_1 = ChatService("ws-1", registry)
    server_2 = ChatService("ws-2", registry)

    # Alice is connected on ws-1, Bob on ws-2, Charlie offline
    registry.register("alice", "ws-1")
    registry.register("bob", "ws-2")
    print(f"  Online users : alice (ws-1), bob (ws-2)")
    print(f"  Offline users : charlie")

    # Alice sends a message to Bob (via her server ws-1)
    print(f"\n  Alice -> Bob :")
    server_1.send("alice", "bob", "hey bob")

    # Alice sends a message to Charlie (offline)
    print(f"\n  Alice -> Charlie (offline) :")
    server_1.send("alice", "charlie", "call me back")

    # Bob replies to Alice
    print(f"\n  Bob -> Alice :")
    server_2.send("bob", "alice", "hi alice, how are you?")

    # History
    print(f"\n  History alice-bob : {[m.content for m in server_1.history('alice', 'bob')]}")
    print(f"  History alice-charlie : {[m.content for m in server_1.history('alice', 'charlie')]}")
    print(f"  Push queue ws-1 : {[(u, m.content) for u, m in server_1.push_queue]}")


def main():
    random.seed(42)
    demo_capacity_estimation()
    demo_url_shortener()
    demo_twitter_fanout()
    demo_chat_system()
    print(f"\n{SEPARATOR}\n  End of demos.\n{SEPARATOR}")


if __name__ == "__main__":
    main()
