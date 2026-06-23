"""
Solutions -- Day 7 Exercises: Classic designs

This file contains the pedagogical walkthroughs of the 3 Easy exercises.
Each solution follows the 6-step framework: clarify, estimate,
high-level, deep dive, bottlenecks, extensions.

Usage:
    python 07-design-classiques.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercise 1 : Design Pastebin
# =============================================================================


def easy_1_pastebin():
    """Solution: Pastebin design."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Design Pastebin")
    print(SEPARATOR)

    print("""
  ----- CLARIFICATION -----
  Questions to ask :
  1. Max size per paste? (1-10 MB per the prompt)
  2. Min/max expiration? (1h to never)
  3. Password protection expected?
  4. Server-side or client-side syntax highlighting?
  5. Paste editable after creation? If so, versioning?
  6. Search over the content? (impacts Elasticsearch)
  7. Authenticated or anonymous users? Account required?
  8. Public API in addition to the UI?
  9. Which market? France only, global, GDPR?

  ----- CAPACITY ESTIMATION -----
  1M DAU, reads:writes = 20:1, so users view ~20 pastes per day.
  Writes : 1M * 1 paste/day = 1M pastes/day = 11.6/s avg, ~35/s peak
  Reads  : 1M * 20 reads/day = 20M/day = 230/s avg, ~700/s peak

  Average size : say 50 KB (between a short note and a 10 MB dump).
  Storage/day : 1M * 50 KB = 50 GB/day = ~18 TB/year (before compression).
  After gzip (3-5x compression on text) : ~4-6 TB/year.

  CDN bandwidth : 20M reads * 50 KB = 1 TB/day out. CDN essential.

  ----- HIGH-LEVEL ARCHITECTURE -----
  Client --> CDN --> API Gateway --> App Service --> Metadata KV (Cassandra)
                                            |              |
                                            v              v
                                   Object Storage (S3)   Redis Cache
                                   for the body
  (Separate workers : expiration cleanup, thumbnail/preview, abuse detection)

  ----- DEEP DIVE : STORAGE -----
  Two levels :
  1. Metadata (Cassandra or DynamoDB) :
     pastes (
       paste_id      TEXT PRIMARY KEY,    -- base62, ~8 chars
       user_id       TEXT,                -- nullable (anonymous)
       title         TEXT,
       language      TEXT,                -- 'python', 'js', ...
       s3_key        TEXT,                -- pointer to the body in S3
       created_at    TIMESTAMPTZ,
       expires_at    TIMESTAMPTZ,         -- NULL = never
       password_hash TEXT,
       size_bytes    INT,
       views         INT
     )
     Partition key = paste_id : O(1) lookup, uniform.

  2. Body in Object Storage (S3 / GCS) :
     The average 50 KB are stored outside the KV. Why?
     - Cheaper per GB ($0.023/GB/month S3 vs $0.25+ DynamoDB)
     - No size limit per row
     - Direct access from the CDN via presigned URL

  ----- CACHE STRATEGY -----
  - Redis cache for the hot pastes (just created + trending)
  - Key : paste:{id} -> JSON metadata + body (if < 100 KB)
  - TTL : 5-15 min initial, extended on each read (TTL-based LRU)
  - Viral hot paste : potentially tens of K reads/sec.
    The CDN in front of the cache absorbs that since we can cache the
    full HTTP response (headers Cache-Control: public, max-age=60).

  Target hit rate : 90%+ thanks to the CDN and Redis.

  ----- BOTTLENECKS -----
  1. Viral hot paste :
     Problem : a paste link posted on HackerNews = 50K reads/s on 1 paste.
     Solution : CDN HTTP cache (max-age=60). The CDN absorbs 99% of the traffic.
     Redis as a backup. The storage is untouched.

  2. Expiration at scale :
     Problem : purging the expired pastes. We can't just 'DELETE
     WHERE expires_at < NOW()' on a 500M-row table.
     Solution : native Cassandra TTL (automatic DELETE). OR a worker
     that scans by day bucket and deletes in batches.

  3. Abuse (malware storage, dead drops) :
     Solution : scan the pastes (regex / ML), rate limit per IP,
     captcha for large pastes, user reporting.

  ----- EXTENSIONS -----
  - Paste versioning
  - Syntax highlighting via Prism.js client-side
  - Search over public pastes (Elasticsearch)
  - API keys for developer integrations
  - Iframe embedding on other sites
    """)


# =============================================================================
# EASY -- Exercise 2 : Design Instagram feed
# =============================================================================


def easy_2_instagram_feed():
    """Solution: Instagram feed design."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : Design Instagram feed")
    print(SEPARATOR)

    print("""
  ----- CLARIFICATION -----
  1. Chronological or algorithmic feed? (prompt = chrono)
  2. Stories too or just the posts?
  3. Videos supported or photos only?
  4. Multi-region (global)?
  5. Likes / comments included in the feed?
  6. Max size of a follow graph (max number of followees per user)?
  7. Photo expiration?

  ----- CAPACITY ESTIMATION -----
  500M DAU, 1 post/day, 20 feed views/day.

  Writes : 500M / 86400 = ~5800 posts/sec avg, ~17K peak
  Feed reads : 500M * 20 = 10B/day = 116K reads/sec avg, ~350K peak

  Photos : 2 MB * 500M = 1 PB/day of media -> 365 PB/year
  -> S3 + CDN mandatory. Storage is the main COST driver.

  Bandwidth out : 10B reads * 2 MB/photo * 0.1 photos shown per feed = 2 PB/day
  -> CDN with a very high cache hit rate (95%+) mandatory.

  ----- HIGH-LEVEL ARCHITECTURE -----
  Upload path :
    Client -> CDN -> API Gateway -> Upload service -> S3
                                           |
                                           v
                                     Kafka (PhotoUploaded)
                                           |
                              +------------+-------------+
                              v            v             v
                        Image processor  Fanout worker   Analytics
                        (thumbnails)    (pre-compute     (ingest)
                                         feeds)

  Read path :
    Client -> CDN -> API Gateway -> Feed service -> Redis (precomputed feed)
                                                          |
                                                          v (miss or tail)
                                                     Cassandra (posts)

  ----- DEEP DIVE : FANOUT ON WRITE vs FANOUT ON READ -----
  Fanout on write (push) :
    When Alice posts, we pre-write post_id into the Redis feed of each
    of her followers.
    - Read : O(1), RANGE on a Redis sorted set.
    - Write : O(N) where N = number of followers.
    - Bad for celebrities (Kylie Jenner 400M followers = 400M
      writes per post, server dead).

  Fanout on read (pull) :
    We just store the post. On each feed open, we fetch the posts
    of every followee and merge.
    - Read : O(M * log P) where M = followees and P = posts/followee.
    - Write : O(1).
    - Bad for users following many accounts (scanning 500 timelines).

  HYBRID SOLUTION (real Instagram) :
    - Normal users (< 100K followers) : fanout on write.
    - Celebrities (> 100K followers) : fanout on read.
    - When a user opens their feed :
      1. Fetch their precomputed feed (normal users)
      2. Pull the recent posts of the celebrities they follow (< 10 typically)
      3. Merge and sort chronologically.
    - The compromise is tunable : the 100K threshold can be adapted.

  Calculation :
    Without hybrid : 500M users * 500 average followers = 250B fanout writes/day.
    With hybrid (excluding the 1000 celebrities) : ~10% reduction on
    the total but eliminates the catastrophic spikes of 400M/post.
    The celebrities (a few thousand) do not impact the system.

  ----- PHOTO STORAGE -----
  - Direct upload to S3 via presigned URLs (no proxy through the API).
  - Async worker : generates thumbnails (150x150, 320x320, 1080x1080),
    formats (jpg, webp, avif), and optional watermark.
  - CDN in front of S3 with a very long cache (immutable, max-age=1y)
    because the photos are immutable once uploaded.
  - Hot regions : S3 multi-region replication, CDN PoPs globally.

  ----- BOTTLENECKS -----
  1. Celebrity problem : solved by the hybrid described above.

  2. Storage cost (365 PB/year) :
     - Aggressive compression (AVIF replaces JPEG, -40% size)
     - Tiering : photos older than 90 days -> S3 Infrequent Access
     - Deduplication via content hash (not critical for Instagram
       since photos are unique, but useful against reposts)

  3. Cold start for new users :
     - No precomputed feed : we fall back on fanout-on-read of the
       users they follow. The first feed loads more slowly (~500 ms).

  ----- EXTENSIONS -----
  - Algorithmic feed (ML ranking : engagement, relationship, timing)
  - Stories (24h retention, separate flow)
  - Explore tab (recommendation engine)
  - Live video (RTMP streaming -> HLS)
  - Direct messages
  - Insights / analytics for creators
    """)


# =============================================================================
# EASY -- Exercise 3 : Notification System
# =============================================================================


def easy_3_notification_system():
    """Solution: Notification System design."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : Notification System")
    print(SEPARATOR)

    print("""
  ----- CLARIFICATION -----
  1. Who triggers the events? (other internal services)
  2. Which channels? (push, email, SMS, in-app, webhook)
  3. Templates : maintained by whom, in which language?
  4. I18n required? (EN, FR, ES...)
  5. Preference granularity? (per notif type? per channel?)
  6. Latency SLA : real time (< 1s) or batch (< 5 min)?
  7. Which third parties in prod? (SendGrid, Twilio, FCM, APNs already negotiated?)
  8. Compliance : GDPR (opt-out, unsubscribe), CAN-SPAM?

  ----- CAPACITY ESTIMATION -----
  100M users * 10 notifs/day = 1B notifs/day = 11.6K/sec avg, 35K peak
  Incoming events : often fewer than notifs since 1 event -> N channels
    -> Say 500M events/day = 5.8K events/sec avg, 17K peak
  Log storage : 500 bytes/notif * 1B = 500 GB/day for the audit (30d retention)

  ----- HIGH-LEVEL ARCHITECTURE -----
  Event Producers --> Kafka (topic: notifications-raw)
                           |
                           v
                  [Notification Router]
                           |
                  +--------+---------+--------+
                  v        v         v        v
             Pref Svc  Template  Dedup   Rate Limit
                  |        |         |        |
                  +--------+---------+--------+
                           |
                           v
                     Kafka (topic: notifications-ready)
                           |
              +------------+------------+------------+
              v            v            v            v
         Push Worker  Email Worker  SMS Worker   In-App Worker
              |            |            |            |
              v            v            v            v
            FCM/APNs   SendGrid      Twilio      WebSocket
              |            |            |            |
              v            v            v            v
                    Delivery DB + DLQ (on failure)

  ----- FLOW OF A 'new_follower' EVENT -----
  1. UserService detects that Alice started following Bob.
  2. UserService publishes into Kafka 'notifications-raw' :
     {
       event_id: 'evt_xyz789',
       event_type: 'new_follower',
       actor_id: 'alice',
       recipient_id: 'bob',
       created_at: ...
     }
  3. The Notification Router consumes the event :
     3a. Dedup check : has evt_xyz789 already been processed? (Redis SET)
     3b. Fetch Bob's preferences : Bob has opted in to follower notifs
         via push and email, not via SMS.
     3c. Rate limit check : has Bob already received 5 pushes this hour? (token bucket)
     3d. Render template : load the 'new_follower' template in FR
         (Bob's language), inject 'Alice te suit maintenant'.
     3e. Publish into 'notifications-ready' with 1 msg per channel.
  4. The Push Worker consumes :
     4a. Look up Bob's device tokens (user_devices table).
     4b. Send to FCM / APNs in parallel.
     4c. On success : log in the DB (delivery_log).
     4d. On failure : retry 3x with backoff, then DLQ.

  ----- DEDUPLICATION -----
  Each event has a unique 'event_id' generated by the producer.
  At the start of the Notification Router :
    if redis.set(f'notif:dedup:{event_id}', '1', nx=True, ex=3600):
        process()
    else:
        skip()  # already seen

  So even if Kafka delivers the same message twice (at-least-once), we
  only process it once. 1h TTL = protection against short-lived duplicates.
  For longer protection, store in the DB with a UNIQUE constraint
  on event_id (table 'processed_events').

  ----- RATE LIMITING -----
  Granularity : per (user_id, channel, notification_type).
  Example : Bob can receive at most 20 pushes/hour for 'new_follower'.
  Algo : token bucket in Redis, key = f'rl:{user}:{channel}:{type}'.
  If exceeded : we skip + log (metric 'rate_limited_count').
  Alert if > X% of the notifs are skipped (signal : a producer in a loop).

  ----- RETRY STRATEGY -----
  Case : SendGrid returns 503 for 1 specific email.
  Strategy :
  - Exponential backoff : 1s, 2s, 4s, 8s, 16s up to 5 attempts.
  - Jitter to avoid the thundering herd.
  - After 5 failures : send to the Kafka DLQ.
  - A 'dlq-handler' worker inspects the DLQ :
    - If the email is definitively bad (invalid format) : log and drop.
    - If it's transient (503, 502) : republish into 'notifications-ready'
      for a new attempt 1h later.
  - Ops alert if DLQ > 1000 items (something systemic).

  Circuit breaker on the SendGrid call :
  - After 10 consecutive failures -> OPEN, we stop calling for 30s.
  - During that time, the messages are put in a local queue.
  - HALF_OPEN after 30s : we test with 1 request.
  - If OK -> CLOSED, we flush the queue.

  ----- DISCUSSED TRADEOFFS -----
  1. Push vs Poll :
     Push (Kafka) = real time, low latency, higher complexity.
     Poll = simple, batching possible, 1-5 min latency.
     For 35K peak/s, push is mandatory.

  2. Sync vs Async :
     Events are processed async via Kafka. The event producer must
     NEVER block on sending the notif. It publishes and forgets.
     Final user latency : 1-5 seconds for non-critical notifs.

  3. Centralized vs embedded template service :
     Centralized (dedicated microservice) = product-managed content,
     i18n, A/B testing. Embedded in the router = simpler but not
     scalable team-wise.
     Choice : centralized for an app with 100M users.

  ----- EXTENSIONS -----
  - A/B testing on the templates
  - Daily digest (batch the non-critical notifs into one email)
  - Quiet hours (do not send between 10pm and 8am local time)
  - Analytics : delivery rate, open rate, click rate, unsubscribe rate
  - Webhook callback to the producer for confirmation
    """)


def main():
    easy_1_pastebin()
    easy_2_instagram_feed()
    easy_3_notification_system()
    print(f"\n{SEPARATOR}")
    print("  End of Day 7 solutions.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
