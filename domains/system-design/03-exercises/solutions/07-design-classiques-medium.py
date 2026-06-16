"""
Solutions -- Day 7 MEDIUM Exercises: Design classiques (interviews)

Worked solutions. The fanout / encoding / WS-fleet sizing numbers are computed
with assertions so the file is self-checking.

Usage:
    python3 07-design-classiques-medium.py
"""

import math

SEPARATOR = "=" * 70


# =============================================================================
# MEDIUM -- Exercise 1 : Size the fanout of a Twitter timeline
# =============================================================================

def medium_1_twitter_fanout():
    """Quantify fanout-on-write vs fanout-on-read and justify the hybrid."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Size the fanout of a Twitter timeline")
    print(SEPARATOR)

    median_followers = 200
    tweets_per_day = 2
    mega_followers = 100_000_000

    # 1. fanout-on-write writes per day
    median_writes_day = median_followers * tweets_per_day        # 400
    mega_writes_day = mega_followers * tweets_per_day            # 200M
    print(f"\n  1. Fanout-on-write writes/day :")
    print(f"     median user : {median_followers} followers * {tweets_per_day} tweets = {median_writes_day} writes/day")
    print(f"     mega account: {mega_followers:,} * {tweets_per_day} = {mega_writes_day:,} writes/day")

    # 2. writes per tweet for a mega account
    mega_writes_per_tweet = mega_followers                       # 100M
    print(f"\n  2. Why fanout-on-write is unsustainable for mega accounts :")
    print(f"     {mega_writes_per_tweet:,} writes for ONE tweet -> latency + cost explode.")
    print(f"     A single celebrity tweet would touch 100M timeline caches.")

    # 3. fanout-on-read merges
    follows = 500
    home_opens_day = 50
    merges_day = follows * home_opens_day
    print(f"\n  3. Pure fanout-on-read :")
    print(f"     a user following {follows} accounts merges {follows} timelines per open,")
    print(f"     * {home_opens_day} opens/day = {merges_day:,} merges/day -- terrible default for")
    print(f"     a 200M-DAU READ-heavy product (read cost dominates).")

    print(f"\n  4. The hybrid :")
    print(f"     - normal accounts (< ~10K followers) : fanout-on-WRITE (precomputed home)")
    print(f"     - mega accounts                       : fanout-on-READ (pulled at read time)")
    print(f"     Home of a user following 499 normal + 1 mega = LRANGE the precomputed")
    print(f"     timeline + live-pull the 1 mega account, then merge sorted by time.")

    # 5. Redis cache size
    tweets_kept = 800
    bytes_per_entry = 40
    users = 200_000_000
    cache_bytes = tweets_kept * bytes_per_entry * users
    cache_tb = cache_bytes / 1e12                        # decimal TB
    print(f"\n  5. Redis timeline cache size :")
    print(f"     {tweets_kept} tweets * {bytes_per_entry} B * {users:,} users = {cache_tb:.2f} TB")
    print(f"     -> Redis SHARDING is mandatory (no single instance holds 6+ TB).")

    # ---- assertions ----
    assert median_writes_day == 400, median_writes_day
    assert mega_writes_day == 200_000_000, mega_writes_day
    assert mega_writes_per_tweet == 100_000_000
    assert merges_day == 25_000, merges_day
    assert 6.0 < cache_tb < 6.5, cache_tb               # ~6.4 TB
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 2 : URL shortener encoding -- collision & capacity
# =============================================================================

def medium_2_url_encoding():
    """Reason about code space, the birthday paradox, hash vs counter."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : URL shortener encoding -- collision & capacity")
    print(SEPARATOR)

    urls_per_day = 50_000_000

    # 1. code space
    space6 = 62 ** 6
    space7 = 62 ** 7
    print(f"\n  1. Code space :")
    print(f"     62^6 = {space6:,} (~57 billion)")
    print(f"     62^7 = {space7:,} (~3.5 trillion)")

    # 2. years to exhaust k=7
    years = space7 / (urls_per_day * 365)
    print(f"\n  2. Years to exhaust k=7 at {urls_per_day:,}/day :")
    print(f"     {space7:,} / ({urls_per_day:,} * 365) = ~{years:.0f} years")

    # 3. birthday paradox
    birthday = math.sqrt(space7)
    print(f"\n  3. Birthday paradox on k=7 :")
    print(f"     sqrt(62^7) = ~{birthday:,.0f} (~1.9M)")
    print(f"     That's < one day of traffic ({urls_per_day:,}) -> a TRUNCATED HASH forces")
    print(f"     collisions early : every write needs a check + retry-with-salt (extra KV read).")

    print(f"\n  4. Range allocation (counter) :")
    print(f"     Each server reserves a batch of 100K ids in ONE round-trip to the")
    print(f"     coordinator -> 1 coordinator call per 100K URLs (cheap). A crash mid-batch")
    print(f"     loses at most one batch = benign GAPS (no duplicates, uniqueness preserved).")

    print(f"\n  5. Guessable sequential codes :")
    print(f"     1,2,3 -> aa,ab,ac lets anyone ENUMERATE links and infer your volume.")
    print(f"     Mitigate WITHOUT going back to hashing : permute / offset the counter")
    print(f"     (e.g. multiply by a coprime mod N, or base62-shuffle) -> still unique,")
    print(f"     no collisions, but not enumerable.")

    # ---- assertions ----
    assert space6 == 56_800_235_584, space6
    assert space7 == 3_521_614_606_208, space7
    assert 180 < years < 210, years                    # ~193 years
    assert 1_800_000 < birthday < 1_900_000, birthday  # ~1.88M
    assert birthday < urls_per_day                     # collision before a day of traffic
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 3 : Chat -- connection registry & ordering
# =============================================================================

def medium_3_chat_registry():
    """Route a 1-to-1 message across stateful WS servers and order messages."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Chat -- connection registry & ordering")
    print(SEPARATOR)

    total_conns = 100_000_000
    per_server = 150_000

    # 1. WS server count
    base_servers = math.ceil(total_conns / per_server)
    with_headroom = math.ceil(base_servers * 1.3)
    print(f"\n  1. WebSocket server count :")
    print(f"     {total_conns:,} / {per_server:,} = {base_servers} servers (base)")
    print(f"     + ~30% headroom -> ~{with_headroom} servers")

    print(f"\n  2. Flow Alice (WS-A) -> Bob (WS-B), 6 steps :")
    steps = [
        "Alice -> WS-A over her persistent WebSocket",
        "WS-A persists the message in Cassandra (conversation_id partition)",
        "WS-A looks up the registry (Redis) : user_id=Bob -> WS-B",
        "WS-A publishes the message on WS-B's channel (Redis pub/sub or Kafka)",
        "WS-B receives it and pushes to Bob over his WebSocket",
        "If Bob is offline : fall back to FCM/APNs + store for catch-up",
    ]
    for i, s in enumerate(steps, 1):
        print(f"     {i}. {s}")

    print(f"\n  3. Registry writes :")
    print(f"     Written on every connect/disconnect. When Bob reconnects on WS-C, the")
    print(f"     registry is updated to user_id=Bob -> WS-C (so routing follows him).")

    print(f"\n  4. Ordering despite out-of-order network :")
    print(f"     The SERVER-side receive timestamp determines final order; message_id is")
    print(f"     a TIMEUUID (timestamp + random -> unique even at the same ms); the")
    print(f"     Cassandra partition (conversation_id) keeps clustering order.")

    print(f"\n  5. Offline + resync :")
    print(f"     Offline -> FCM/APNs push + persisted message. On reconnect, Bob sends his")
    print(f"     last_message_id and pulls everything newer (catch-up sync).")

    # ---- assertions ----
    assert base_servers == 667, base_servers
    assert with_headroom == 868, with_headroom
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 7 MEDIUM : DESIGN CLASSIQUES")
    print(SEPARATOR)
    medium_1_twitter_fanout()
    medium_2_url_encoding()
    medium_3_chat_registry()
    print(f"\n{SEPARATOR}")
    print("  END OF MEDIUM SOLUTIONS (all assertions passed)")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
