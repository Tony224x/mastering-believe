"""
Solutions -- Day 14 MEDIUM Exercises: Capstone extensions

Worked solutions with the reasoning step by step. Assertions lock the
key calculations so the file is self-checking.

These EXTEND the two J14 reference designs (Dropbox + LLM Support Assistant);
they do not redesign them.

Usage:
    python3 14-capstone-medium.py
"""

SEPARATOR = "=" * 70


# =============================================================================
# MEDIUM -- Exercise 1 : Add a semantic cache tier to the LLM Support Assistant
# =============================================================================

def medium_1_semantic_cache_tier():
    """Insert a semantic cache in the J14 support design and quantify impact."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Add a semantic cache tier to the LLM Support Assistant")
    print(SEPARATOR)

    daily_convs = 500_000
    turns = 5
    daily_calls = daily_convs * turns
    hit_rate = 0.40

    print(f"\n  1. Placement in the flow :")
    print(f"     AFTER input guardrails (PII scrub) and BEFORE the RAG/LLM. We scrub")
    print(f"     first (so we never cache raw PII), then short-circuit the EXPENSIVE")
    print(f"     part (retrieval + reranker + generation) on a hit.")

    print(f"\n  2. Realistic hit rate (J11) :")
    print(f"     Support/FAQ assistant : 40-60% on GENERIC questions ('how to reset")
    print(f"     password'). Personalized questions ('status of MY order') : ~0% and")
    print(f"     must NOT be cached globally.")

    # 3. Savings at 40% hit
    saved_calls = int(daily_calls * hit_rate)
    print(f"\n  3. Savings at {hit_rate:.0%} hit :")
    print(f"     Daily LLM calls : {daily_convs:,} * {turns} = {daily_calls:,}")
    print(f"     Calls saved/day : {saved_calls:,} ({hit_rate:.0%})")
    print(f"     LLM cost reduction : ~{hit_rate:.0%} (a hit costs ~0 in LLM tokens).")

    print(f"\n  4. Threshold & scope :")
    print(f"     High threshold (0.92-0.97) to avoid serving a wrong answer.")
    print(f"     Scope excludes any request containing personal data (skip the cache).")

    print(f"\n  5. Support-specific pitfall :")
    print(f"     'status of MY order 12345' is personalized -- a GLOBAL cache would")
    print(f"     leak one user's order to another. Never cache personalized queries")
    print(f"     globally (this was incident I2 in the J11 cache exercise).")

    print(f"\n  6. Measuring quality (not just hit rate) :")
    print(f"     Sample served hits and run an LLM-as-a-judge to verify the cached")
    print(f"     answer fits the new query; track a false-positive rate alongside hits.")

    # ---- assertions ----
    assert daily_calls == 2_500_000, daily_calls
    assert saved_calls == 1_000_000, saved_calls       # 40% of 2.5M
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 2 : Size the Dropbox-like for 10x traffic
# =============================================================================

def medium_2_dropbox_10x():
    """Re-run the J14 Dropbox estimates at 10x and find the first bottleneck."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Size the Dropbox-like for 10x traffic")
    print(SEPARATOR)

    users = 500_000_000          # 10x of 50M
    uploads_per_user = 2
    downloads_per_user = 10
    file_mb = 2
    replication = 3

    # 1. Recompute at 10x
    uploads_day = users * uploads_per_user
    downloads_day = users * downloads_per_user
    uploads_s = uploads_day / 86_400
    downloads_s = downloads_day / 86_400
    # bandwidth download : downloads/s * file size * 8 bits
    dl_bandwidth_gbps = downloads_s * file_mb * 1024 * 1024 * 8 / 1e9
    storage_net_day_tb = uploads_day * file_mb / (1024 * 1024)   # MB -> TB
    storage_net_year_pb = storage_net_day_tb * 365 / 1024        # TB -> PB
    storage_replicated_year_pb = storage_net_year_pb * replication

    print(f"\n  1. At 10x ({users:,} users) :")
    print(f"     uploads/s        : {uploads_s:,.0f}")
    print(f"     downloads/s      : {downloads_s:,.0f}")
    print(f"     download BW      : {dl_bandwidth_gbps:,.0f} Gbps")
    print(f"     storage net/day  : {storage_net_day_tb:,.0f} TB")
    print(f"     storage net/year : {storage_net_year_pb:,.1f} PB")
    print(f"     with replication x{replication} : {storage_replicated_year_pb:,.0f} PB/year")

    print(f"\n  2. First component to break :")
    print(f"     Download BANDWIDTH (~{dl_bandwidth_gbps:,.0f} Gbps) and the WebSocket")
    print(f"     connection count. Storage grows linearly (manageable with tiering);")
    print(f"     bandwidth and live connections hit hard limits first.")

    print(f"\n  3. Storage strategy :")
    print(f"     Tiering hot (S3) -> warm (IA) -> cold (Glacier) after 90 days inactive,")
    print(f"     PLUS block dedup (the J14 design saves ~30-40% of storage).")

    print(f"\n  4. CDN at 10x :")
    print(f"     Popular files are served from edge -> origin bandwidth grows much")
    print(f"     slower than user-facing bandwidth. A high CDN hit rate is what makes")
    print(f"     10x affordable; without it, origin egress explodes.")

    print(f"\n  5. Metadata DB at 10x :")
    print(f"     Postgres partitioned by user_id : add read replicas for list_folder,")
    print(f"     sub-partition hot users, cache the folder tree in Redis.")

    # 6. WebSocket connections
    concurrent_pct = 0.10   # ~10% of daily-active online at once
    ws_connections = int(users * concurrent_pct)
    print(f"\n  6. WebSocket connections at 10x :")
    print(f"     ~{concurrent_pct:.0%} of {users:,} online at once = {ws_connections:,} concurrent")
    print(f"     connections. Architecture : a POOL of WS servers (sticky sessions) +")
    print(f"     Redis pub/sub for fan-out; no single server holds them all.")

    # ---- assertions ----
    assert abs(downloads_s - 57_870) < 200, downloads_s
    assert abs(dl_bandwidth_gbps - 971) < 30, dl_bandwidth_gbps    # ~930-970 Gbps
    assert 1900 < storage_net_day_tb < 2100, storage_net_day_tb    # ~1907 TB/day
    assert ws_connections == 50_000_000, ws_connections
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 3 : Apply the 6-step framework to a multilingual extension
# =============================================================================

def medium_3_multilingual():
    """Use the J14 framework to design a multilingual extension of the support bot."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Multilingual extension (6-step framework)")
    print(SEPARATOR)

    articles = 100_000
    languages = 12

    print(f"\n  1. Clarify :")
    print(f"     Functional : answer in the user's language; cite localized sources.")
    print(f"     Non-functional : HOMOGENEOUS quality across languages (no second-class")
    print(f"     language), PII detection per language, same latency budget.")

    # 2. Estimate
    chunks_per_article = 5
    base_chunks = articles * chunks_per_article
    approach_b_chunks = base_chunks * languages
    print(f"\n  2. Estimate (index impact) :")
    print(f"     Base index : {articles:,} articles * {chunks_per_article} = {base_chunks:,} chunks.")
    print(f"     Approach B (translate + index per language) : ~{languages}x = {approach_b_chunks:,} chunks.")
    print(f"     Multilingual embeddings (approach A') : ~1x chunks (one shared index).")

    print(f"\n  3. Design -- two approaches :")
    print(f"     (A) translate query -> EN -> RAG in EN -> translate answer back :")
    print(f"         cheap on storage (1 index) but adds 2 translation hops (latency,")
    print(f"         quality loss on jargon).")
    print(f"     (B) translate + index the KB in every language : best per-language")
    print(f"         quality but ~{languages}x storage & embedding cost, and {languages}x re-indexing.")
    print(f"     MODERN choice : multilingual EMBEDDINGS -> ONE index, cross-lingual")
    print(f"     retrieval, no translation hop.")

    print(f"\n  4. Deep dive -- cross-lingual retrieval :")
    print(f"     Use a multilingual embedding model (e.g. BGE-m3, Cohere multilingual)")
    print(f"     so a query in language X retrieves chunks regardless of their language,")
    print(f"     plus a multilingual reranker.")

    print(f"\n  5. Bottlenecks :")
    print(f"     Recall is typically lower on rare languages; PII detection must be")
    print(f"     per-language (regex/NER differ); generation quality varies by language.")

    print(f"\n  6. Extension -- language detection & routing :")
    print(f"     A lightweight language classifier as the FIRST step routes the request")
    print(f"     and selects the right guardrails/prompt variant.")

    # ---- assertions ----
    assert base_chunks == 500_000, base_chunks
    assert approach_b_chunks == 6_000_000, approach_b_chunks   # 12x
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 14 MEDIUM : CAPSTONE EXTENSIONS")
    print(SEPARATOR)
    medium_1_semantic_cache_tier()
    medium_2_dropbox_10x()
    medium_3_multilingual()
    print(f"\n{SEPARATOR}")
    print("  END OF MEDIUM SOLUTIONS")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
