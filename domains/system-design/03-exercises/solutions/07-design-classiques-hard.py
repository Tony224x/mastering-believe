"""
Solutions -- Day 7 HARD Exercises: Design classiques (entretiens)

Worked solutions. The Twitter capacity estimates and the fanout post-mortem math
are computed and pinned with assertions on the numbers the exercise calls out
(~4.6K tweets/s, ~115K reads/s, ~44 TB/year tweets, ~15 PB/year media, 240M
fanout writes, 1 celebrity tweet ~= 400K median tweets). The architecture and
runbooks are printed and pinned by checkable facts.

Usage:
    python3 07-design-classiques-hard.py
"""

import math

SEPARATOR = "=" * 60

SEC_PER_DAY = 86_400
DAYS_PER_YEAR = 365


# =============================================================================
# HARD -- Exercise 1 : Design Twitter @ 200M DAU (45-min senior interview)
# =============================================================================

def hard_1_design_twitter():
    """6-step framework, capacity math, hybrid fanout, sharding, bottlenecks."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Design Twitter (200M DAU)")
    print(SEPARATOR)

    dau = 200_000_000
    tweets_per_user_day = 2
    opens_per_user_day = 50

    print("\n  1. Clarify (5 questions) + estimate :")
    for q in [
        "Chronological vs ranked home timeline? (chrono here)",
        "Read-heavy ratio? (50 opens vs 2 tweets/user/day -> ~25:1 read:write)",
        "Eventual consistency OK on the timeline? (yes, 1-2 s)",
        "Media size & share? (10% of tweets, ~1 MB avg)",
        "Follower distribution? (long tail + a few 100M-follower mega-accounts)",
    ]:
        print(f"     - {q}")

    # Tweets/s : peak ~= 3x average is the usual interview rule of thumb.
    tweets_per_day = dau * tweets_per_user_day
    tweets_per_s_avg = tweets_per_day / SEC_PER_DAY
    tweets_per_s_peak = tweets_per_s_avg * 3
    reads_per_day = dau * opens_per_user_day
    reads_per_s_avg = reads_per_day / SEC_PER_DAY
    reads_per_s_peak = reads_per_s_avg * 3
    print("\n     Throughput :")
    print(f"       tweets/s : ~{tweets_per_s_avg:,.0f} avg, ~{tweets_per_s_peak:,.0f} peak")
    print(f"       reads/s  : ~{reads_per_s_avg:,.0f} avg, ~{reads_per_s_peak:,.0f} peak")
    # Pin the orders of magnitude the exercise expects.
    assert abs(tweets_per_s_avg - 4_630) < 50      # ~4.6K tweets/s
    assert abs(reads_per_s_avg - 115_740) < 500    # ~115K reads/s
    assert abs(reads_per_s_peak - 347_000) < 2000  # ~350K peak

    # Storage : tweet ~300 B, media 10% * 1 MB.
    tweet_bytes = 300
    tweets_storage_year = tweets_per_day * DAYS_PER_YEAR * tweet_bytes
    media_bytes = 1_000_000
    media_storage_year = tweets_per_day * 0.10 * DAYS_PER_YEAR * media_bytes
    print("\n     Storage :")
    print(f"       tweets : ~{tweets_storage_year / 1e12:,.0f} TB/year")
    print(f"       media  : ~{media_storage_year / 1e15:,.1f} PB/year")
    assert abs(tweets_storage_year / 1e12 - 44) < 3     # ~44 TB/year
    assert abs(media_storage_year / 1e15 - 15) < 2      # ~15 PB/year

    print("\n  2. High-level (write path / read path) :")
    print("     WRITE: client -> LB -> API -> Tweet svc -> Cassandra (tweets)")
    print("            -> Kafka (fanout) -> fanout workers -> Redis timeline cache")
    print("            media -> S3 (+ CDN)")
    print("     READ : client -> LB -> API -> Timeline svc -> Redis cache (hit)")
    print("            -> hydrate tweets from Cassandra -> media via CDN")

    print("\n  3. Deep dive -- hybrid fanout :")
    print("     fanout-on-write : push tweet into each follower's cached timeline")
    print("       -> reads are O(1) (just read your list), writes are heavy.")
    print("     fanout-on-read  : pull-merge followees' tweets at read time")
    print("       -> writes are O(1), reads are heavy.")
    print("     HYBRID : fanout-on-WRITE for normal accounts (< ~10K followers),")
    print("       fanout-on-READ for mega-accounts; at read time MERGE the pushed")
    print("       timeline with a pull of the few mega-accounts you follow.")
    print("     Redis timeline : sorted set, key=timeline:{user}, score=ts,")
    print("       member=tweet_id, capped at ~800-1000 entries.")
    print("     Cassandra tweets : PRIMARY KEY ((user_id), created_at DESC).")

    print("\n  4. Sharding & hot partitions :")
    print("     Shard tweets by user_id (partition key); timelines by owner user.")
    celebrity_followers = 100_000_000
    print(f"     Celebrity problem : a {celebrity_followers:,}-follower account would")
    print(f"       trigger {celebrity_followers:,} timeline writes PER TWEET on")
    print("       fanout-on-write -> infeasible -> serve these accounts via read.")
    assert celebrity_followers == 100_000_000

    print("\n  5. Bottlenecks :")
    read_write_ratio = reads_per_s_avg / tweets_per_s_avg
    print(f"     Read amplification ~{read_write_ratio:.0f}:1 -> absorb with an")
    print("       aggressive Redis cache (90%+ hit on home timelines).")
    print(f"     Media at ~{media_storage_year / 1e15:.0f} PB/year -> object store (S3)")
    print("       + CDN edge caching (never serve media from origin).")
    assert read_write_ratio > 20

    print("\n  6. Extensions (3 credible) :")
    for ext, tech in [
        ("Search", "Elasticsearch (inverted index on tweet text)"),
        ("ML ranking", "feature store + ranking model, re-rank merged timeline"),
        ("Trending", "streaming aggregation (Flink/Spark) over a sliding window"),
    ]:
        print(f"     - {ext:<11}: {tech}")


# =============================================================================
# HARD -- Exercise 2 : Post-mortem -- the home timeline that blew up
# =============================================================================

def hard_2_fanout_postmortem():
    """Post-mortem: pure fanout-on-write + a mega-account = global meltdown."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Fanout / hot-partition post-mortem")
    print(SEPARATOR)

    print("\n  1. Root cause analysis (cascade) :")
    cascade = [
        ("ARCHITECTURE", "mega-account (80M followers) posts -> 80M fanout writes",
         "Missing guardrail : hybrid (read for mega-accounts)"),
        ("ARCHITECTURE", "single shared Kafka queue/workers monopolised by the burst",
         "Missing guardrail : dedicated queue/workers for big accounts (bulkhead)"),
        ("ARCHITECTURE", "single Redis cluster, hot key not sharded -> write saturated",
         "Missing guardrail : shard/replicate the hot timeline key"),
        ("PROCESS", "no follower threshold to switch an account to read",
         "Missing guardrail : auto-classify accounts by follower count"),
        ("MONITORING", "no alert on Kafka lag / Redis p99 / timeline freshness",
         "Missing guardrail : alert on queue depth + p99 + freshness"),
    ]
    for cat, cause, guardrail in cascade:
        print(f"     [{cat}] {cause}")
        print(f"       -> {guardrail}")
    print("     Why ONE account degraded EVERYONE : the queue, workers and Redis")
    print("     are SHARED -> the mega-account starved every normal user behind it.")

    print("\n  2. The numbers :")
    celeb_followers = 80_000_000
    n_tweets = 3
    total_fanout = celeb_followers * n_tweets
    median_followers = 200
    equiv_median_tweets = total_fanout / median_followers
    one_tweet_equiv = celeb_followers / median_followers
    print(f"     {n_tweets} tweets x {celeb_followers:,} = {total_fanout:,} fanout writes")
    print(f"     Median user has {median_followers} followers -> 1 celebrity tweet")
    print(f"       ~= {one_tweet_equiv:,.0f} median tweets in load")
    print(f"       (the 3 tweets ~= {equiv_median_tweets:,.0f} median tweets total)")
    assert total_fanout == 240_000_000                 # 3 * 80M
    assert abs(one_tweet_equiv - 400_000) < 1          # ~400K median tweets

    print("\n  3. Why metrics were green :")
    print("     The API still returns 200 -- just slowly and with STALE data. The")
    print("     failure isn't a 5xx, so 5xx/latency dashboards look fine. The")
    print("     metrics that SHOULD have alerted : Kafka consumer lag, Redis p99,")
    print("     and timeline FRESHNESS (age of newest delivered tweet).")

    print("\n  4. Would the hybrid have avoided it? :")
    print("     Yes : a mega-account served via fanout-on-READ does ZERO timeline")
    print("     writes on post -> no 80M-write burst, no queue/Redis saturation.")
    print("     Their followers pull-merge the mega-account at read time instead.")
    threshold = 100_000
    print(f"     Threshold to flip to read : ~{threshold:,} followers (tune by cost;")
    print("       credible range ~10K-1M depending on write budget).")
    assert 10_000 <= threshold <= 1_000_000

    print("\n  5. Isolation (bulkhead) :")
    print("     A big account must NEVER share the queue/workers of normal accounts")
    print("     (same J5 bulkhead lesson) : a dedicated lane bounds its blast radius")
    print("     so a celebrity burst can't starve everyone else.")

    print("\n  6. Runbook (7 steps) -- mega-account saturates fanout :")
    runbook = [
        "Identify the hot account (top fanout producer in the queue)",
        "Stop its fanout-on-write : flip the account to fanout-on-read NOW",
        "Drain / prioritise the shared queue so normal tweets flow again",
        "Watch Kafka lag, Redis p99 and timeline freshness recover",
        "Add the missing alerts (queue depth, Redis p99, freshness)",
        "Roll out the follower threshold + dedicated big-account lane (bulkhead)",
        "Post-mortem in 24h : cascade, missing guardrails, action items",
    ]
    for i, s in enumerate(runbook, 1):
        print(f"     {i}. {s}")


def main():
    print("\n" + "=" * 60)
    print("  SOLUTIONS -- DAY 7 HARD : DESIGN CLASSIQUES")
    print("=" * 60)
    hard_1_design_twitter()
    hard_2_fanout_postmortem()
    print(f"\n{'=' * 60}")
    print("  END OF HARD SOLUTIONS (all assertions passed)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
