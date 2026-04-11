"""
Jour 7 -- Design classiques
Demonstrations interactives en Python.

Usage:
    python 07-design-classiques.py

Mocks et outils pour les 3 designs classiques :
- Capacity estimation helpers (ordres de grandeur)
- URL shortener : base62 encoder, in-memory sharded KV store
- Twitter timeline : fanout-on-write avec sorted set en memoire
- Chat system : mini pub/sub WebSocket-like avec registry de connexions

Tout est in-memory, runnable sans dependance externe.
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
    """Format un nombre avec des suffixes (K, M, B)."""
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
    """Estime les QPS moyen et peak pour un systeme.

    WHY peak_factor=3 ? En general, le peak horaire est 2-4x la moyenne
    (distribution non uniforme sur 24h). 3x est un ordre de grandeur
    raisonnable pour les calculs d'entretien.
    """
    daily_events = dau * actions_per_day
    avg_qps = daily_events / 86400
    peak_qps = avg_qps * peak_factor
    return avg_qps, peak_qps, daily_events


def estimate_storage(events_per_day: int, bytes_per_event: int, years: float = 1):
    """Calcule le stockage total."""
    daily = events_per_day * bytes_per_event
    total = daily * 365 * years
    return daily, total


# =============================================================================
# SECTION 2 : URL Shortener
# =============================================================================


BASE62_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def base62_encode(num: int) -> str:
    """Encode un int en base62.

    WHY base62 ? 62^7 = 3.5 trillions de codes possibles en 7 chars.
    Largement assez pour une url shortener. Plus court que base10 ou
    base16. Accepte dans les URLs sans encoding.
    """
    if num == 0:
        return BASE62_ALPHABET[0]
    parts = []
    while num > 0:
        parts.append(BASE62_ALPHABET[num % 62])
        num //= 62
    return "".join(reversed(parts))


def base62_decode(code: str) -> int:
    """Decode une chaine base62 vers un int."""
    num = 0
    for char in code:
        num = num * 62 + BASE62_ALPHABET.index(char)
    return num


class ShardedKVStore:
    """Mock d'un KV store shardé (type Cassandra / DynamoDB).

    WHY sharded ? Pour demontrer l'idee : chaque cle est routee vers
    un shard selon hash(key) % N. C'est ce que font les DBs distribuees
    en interne.
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
    """Service complet avec counter distribue (simplified) et cache.

    WHY counter + cache ? Approche "compteur global" : on alloue des ids
    uniques en sequence, encodes en base62. Le cache Redis fait que 95%
    des redirects sont servies sans toucher au KV store.
    """

    def __init__(self, base_url: str = "https://tiny.ly/"):
        self.base_url = base_url
        self.kv = ShardedKVStore(num_shards=4)
        self.cache: dict[str, str] = {}  # LRU simplification
        # Compteur distribue : on part d'un offset pour avoir des codes >= 3 chars
        self.counter = 100000
        self.stats = {"shortens": 0, "cache_hits": 0, "cache_misses": 0}

    def shorten(self, long_url: str) -> str:
        """Cree un short code pour une URL."""
        self.counter += 1
        code = base62_encode(self.counter)
        self.kv.put(code, long_url)
        self.stats["shortens"] += 1
        return self.base_url + code

    def expand(self, short_url: str) -> Optional[str]:
        """Resolve un short URL vers sa version longue."""
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
    """Timeline precalculee via fanout-on-write.

    WHY fanout-on-write ? Read = tres frequent (100x write). On trade
    writes couteuses contre reads ultra-rapides. La timeline est juste
    un sorted set en memoire, LRANGE = O(log N).

    Hybride reel Twitter : fanout-on-write pour users normaux, fanout-
    on-read pour les celebrites avec 100M followers. Ici on simplifie
    avec seulement fanout-on-write.
    """

    def __init__(self, max_timeline_size: int = 100):
        self.max_timeline_size = max_timeline_size
        self.tweets: dict[str, Tweet] = {}  # tweet_id -> Tweet
        self.followers: dict[str, set[str]] = defaultdict(set)  # user -> set of followers
        # timeline[user] = list de tweet_ids tries chronologiquement inverse
        self.timelines: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=max_timeline_size))
        # Stats pour la demo
        self.fanout_writes = 0

    def follow(self, follower: str, followed: str):
        self.followers[followed].add(follower)

    def post_tweet(self, user_id: str, content: str) -> str:
        """Poste un tweet et le fanout vers toutes les timelines des followers.

        WHY fanout synchrone ici ? En prod, c'est un job asynchrone
        (Kafka -> fanout workers). Ici on simplifie pour la demo.
        """
        tweet = Tweet(tweet_id=str(uuid.uuid4())[:8], user_id=user_id, content=content)
        self.tweets[tweet.tweet_id] = tweet

        # Ajoute a la timeline de l'auteur
        self.timelines[user_id].appendleft(tweet.tweet_id)

        # FANOUT : ecriture dans la timeline de chaque follower
        for follower in self.followers[user_id]:
            self.timelines[follower].appendleft(tweet.tweet_id)
            self.fanout_writes += 1
        return tweet.tweet_id

    def get_home_timeline(self, user_id: str, limit: int = 20) -> list[Tweet]:
        """Retourne la timeline d'un user : lookup direct dans le sorted set."""
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
    """Registry qui track quel user est connecte a quel serveur chat.

    WHY ? Quand Alice envoie un message a Bob, on doit savoir quel
    serveur WebSocket tient la connexion de Bob pour lui router le
    message. En prod : Redis hash 'user -> chat_server'.
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
    """Mini chat service avec persist + livraison via registry.

    WHY separer persist et deliver ? Persist doit TOUJOURS reussir
    (sinon le message est perdu). Delivery est best-effort (si le
    destinataire est offline, on envoie une push notif). En prod :
    ecrit d'abord Cassandra, puis Kafka pour le delivery async.
    """

    def __init__(self, server_id: str, registry: ConnectionRegistry):
        self.server_id = server_id
        self.registry = registry
        self.storage: dict[str, list[ChatMessage]] = defaultdict(list)
        self.inbox: dict[str, list[ChatMessage]] = defaultdict(list)  # delivered messages per user
        self.push_queue: list[tuple[str, ChatMessage]] = []  # offline users

    def send(self, sender: str, recipient: str, content: str):
        """Envoie un message 1-to-1."""
        # Conversation id = deterministic hash of the sorted pair
        convo_id = "-".join(sorted([sender, recipient]))
        msg = ChatMessage(
            message_id=str(uuid.uuid4())[:8],
            conversation_id=convo_id,
            sender_id=sender,
            content=content,
        )
        # 1. Persist (TOUJOURS en premier)
        self.storage[convo_id].append(msg)
        # 2. Route via le registry
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
        "https://kalira-ia.com/products/kalira-base",
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
    # Note : avec seulement 3 URLs, le shard # depend du hash. OK pour la demo.


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

    print(f"  Fanout writes effectues : {tw.fanout_writes}")
    print(f"  (3 tweets alice * 5 followers + 1 tweet bob * 2 followers = 17)")

    # Chaque follower voit alice dans sa timeline
    for user in ["user-0", "user-1", "alice", "charlie"]:
        tl = tw.get_home_timeline(user, limit=10)
        print(f"  Timeline de {user} : {[t.content for t in tl]}")


def demo_chat_system():
    print(f"\n{SEPARATOR}\n  DEMO 4 : Chat system\n{SEPARATOR}")
    registry = ConnectionRegistry()
    # 2 chat servers pour simuler le partitioning
    server_1 = ChatService("ws-1", registry)
    server_2 = ChatService("ws-2", registry)

    # Alice est connectee sur ws-1, Bob sur ws-2, Charlie offline
    registry.register("alice", "ws-1")
    registry.register("bob", "ws-2")
    print(f"  Online users : alice (ws-1), bob (ws-2)")
    print(f"  Offline users : charlie")

    # Alice envoie un message a Bob (via son serveur ws-1)
    print(f"\n  Alice -> Bob :")
    server_1.send("alice", "bob", "hey bob")

    # Alice envoie un message a Charlie (offline)
    print(f"\n  Alice -> Charlie (offline) :")
    server_1.send("alice", "charlie", "call me back")

    # Bob repond a Alice
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
    print(f"\n{SEPARATOR}\n  Fin des demos.\n{SEPARATOR}")


if __name__ == "__main__":
    main()
