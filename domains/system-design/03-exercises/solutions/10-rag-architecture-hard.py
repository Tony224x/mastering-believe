"""
Solutions -- Day 10 HARD Exercises: RAG Architecture

Worked solutions with the reasoning step by step. Assertions lock the
key calculations so the file is self-checking.

Usage:
    python3 10-rag-architecture-hard.py
"""

import math

SEPARATOR = "=" * 70


# =============================================================================
# HARD -- Exercise 1 : Multi-tenant enterprise RAG
# =============================================================================

def hard_1_multitenant():
    """Design a resilient multi-tenant RAG-as-a-Service."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Multi-tenant enterprise RAG")
    print(SEPARATOR)

    tenants = 500
    total_chunks = 200_000_000
    dims = 1536
    bytes_per_dim = 4
    daily_requests = 5_000_000
    daily_change_pct = 0.02

    print(f"\n  1. Multi-tenant isolation :")
    print(f"     a) Collection per tenant : strongest isolation, but 500 collections")
    print(f"        = ops overhead, poor packing for tiny tenants (10K docs each).")
    print(f"     b) Shared namespace + tenant_id filter : best packing, cheapest, but")
    print(f"        isolation depends entirely on the filter being correct.")
    print(f"     c) Dedicated cluster per big tenant : strong isolation + no noisy")
    print(f"        neighbor, but expensive; only justified for the largest tenants.")
    print(f"     RECOMMENDATION : tiered.")
    print(f"     - Tiny/medium tenants -> shared namespace + MANDATORY tenant_id filter")
    print(f"     - Top ~20% tenants (80% of traffic) -> dedicated collection or cluster")
    print(f"     Anti-leak guarantee : push tenant_id as a PRE-FILTER in the vector DB")
    print(f"     query (not a post-filter), partition keys by tenant_id, AND run an")
    print(f"     automated isolation test in CI (tenant A query must never return B).")

    print(f"\n  2. Incremental indexing pipeline :")
    print(f"     Detect changes : CDC on the source (Confluence/Drive webhooks) OR a")
    print(f"     change feed; never full re-embed. Push changed doc_ids to a queue.")
    print(f"     Freshness < 5 min : a worker pool consumes the queue -> chunk -> embed")
    print(f"     -> upsert into the index. Size the pool for the steady-state change")
    print(f"     rate (see below) plus burst headroom.")
    print(f"     Deletion < 1h : on delete event, tombstone the chunks immediately and")
    print(f"     hard-delete in a background sweep; the tombstone makes them")
    print(f"     non-retrievable instantly.")

    changed_chunks_per_day = total_chunks * daily_change_pct
    changed_chunks_per_sec = changed_chunks_per_day / 86_400
    print(f"     Change volume : {total_chunks:,} * {daily_change_pct:.0%} = "
          f"{changed_chunks_per_day:,.0f} chunks/day = {changed_chunks_per_sec:.0f} chunks/s steady-state")

    # 3. Sizing
    raw_bytes = total_chunks * dims * bytes_per_dim
    raw_tb = raw_bytes / (1024 ** 4)
    print(f"\n  3. Sizing & placement :")
    print(f"     Raw dense index : {total_chunks:,} * {dims} * {bytes_per_dim}B = {raw_tb:.2f} TB")
    print(f"     With HNSW overhead (~1.5x) : ~{raw_tb*1.5:.2f} TB")
    print(f"     Nodes : with ~64 GB usable RAM/node for vectors, you need")
    nodes = math.ceil((raw_tb * 1.5 * 1024) / 64)
    print(f"             ceil({raw_tb*1.5:.2f}TB*1024 / 64GB) = ~{nodes} nodes (hot tier).")
    print(f"     Noisy neighbor : per-tenant quotas (QPS + memory), isolate big tenants")
    print(f"     on their own shards so a heavy tenant can't evict a small one.")
    print(f"     Hot/cold : keep active tenants' indexes in RAM (HNSW); park dormant")
    print(f"     tenants on disk/blob (IVF or memory-mapped), warm on first query.")

    # 4. Latency budget
    print(f"\n  4. Latency under constraint (p95 target 2000ms) :")
    budget = [
        ("auth + tenant resolution", 30),
        ("hybrid retrieval (pre-filtered)", 150),
        ("rerank top-50", 220),
        ("generation (main LLM)", 1500),
    ]
    total_ms = sum(ms for _, ms in budget)
    for name, ms in budget:
        print(f"     {name:<35} {ms:>5} ms")
    print(f"     {'TOTAL':<35} {total_ms:>5} ms (fits 2000ms)")
    print(f"     Cache : a per-tenant semantic cache on the FAQ-like queries.")
    print(f"     Pitfall : a GLOBAL semantic cache is forbidden (cross-tenant leak);")
    print(f"     always key the cache by tenant_id, and never cache PII answers.")

    # 5. Cost
    print(f"\n  5. Cost & pricing :")
    print(f"     3 big cost posts : (1) vector storage/RAM, (2) embedding (indexing),")
    print(f"     (3) generation tokens (the largest at runtime).")
    print(f"     Fair pricing : meter per tenant on docs indexed (storage) + queries")
    print(f"     served (generation). A 5M-doc tenant pays far more than a 10K one,")
    print(f"     matching their actual footprint.")

    # 6. Failure modes
    print(f"\n  6. Failure modes :")
    print(f"     Indexing backlog : if the queue lags, serve STALE results + flag")
    print(f"     'index updating', alert when freshness SLA (5 min) is breached, and")
    print(f"     autoscale the worker pool. Never block reads on indexing.")
    print(f"     Cross-tenant load : per-tenant rate limits + bulkheads so one tenant's")
    print(f"     query storm can't saturate another's shard.")

    # ---- assertions ----
    assert abs(raw_tb - 1.146) < 0.05, raw_tb                # ~1.15 TB
    assert nodes >= 25, nodes                                # need many nodes
    assert abs(changed_chunks_per_sec - 46.3) < 2, changed_chunks_per_sec
    assert total_ms <= 2000, "p95 budget must fit SLA"
    print("\n  [assertions OK]")


# =============================================================================
# HARD -- Exercise 2 : Post-mortem -- RAG data leak + hallucination
# =============================================================================

def hard_2_postmortem():
    """Post-mortem of a composite RAG incident (leak + quality regression)."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Post-mortem -- the RAG that leaked data and hallucinated")
    print(SEPARATOR)

    print(f"\n  1. Full causal chain :")
    chain = [
        ("PROCESS", "Partial re-embedding : only public docs re-embedded to v3, "
         "personal docs kept v2",
         "Missing guardrail : an embedding migration must be all-or-nothing"),
        ("ARCHITECTURE", "v2 and v3 vectors live in the same index : incompatible "
         "spaces -> meaningless cosine scores",
         "Missing guardrail : index must be versioned by embedding model"),
        ("ARCHITECTURE", "ACL filter moved from vector-DB pre-filter to app-side "
         "post-filter",
         "Missing guardrail : access control must be enforced at the data layer (pre-filter)"),
        ("SECURITY", "ACL filter bug : int user_id compared to list of strings -> "
         "filter never matches -> everything passes",
         "Missing guardrail : typed comparison + isolation test in CI"),
        ("QUALITY", "Aberrant scores surface other employees' payslips in top-k",
         "Missing guardrail : confidentiality circuit breaker on aberrant scores"),
        ("QUALITY", "No gold-set re-eval after the model upgrade",
         "Missing guardrail : CI gate requiring re-eval before deploy"),
        ("SECURITY", "LLM copies other employees' salary data into the answer",
         "Missing guardrail : output PII / out-of-scope check before serving"),
    ]
    for cat, cause, guard in chain:
        print(f"     [{cat}] {cause}")
        print(f"        -> {guard}")

    print(f"\n  2. The embedding mismatch :")
    print(f"     v2 and v3 embed text into DIFFERENT geometric spaces. A cosine")
    print(f"     distance between a v2 vector and a v3 vector is NOISE -- it has no")
    print(f"     semantic meaning. Mixing them makes retrieval scores nonsensical,")
    print(f"     which is exactly how off-ACL payslips bubbled to the top.")
    print(f"     Correct migration (zero downtime, zero mismatch) :")
    print(f"     a) Re-embed the ENTIRE corpus with v3 into a NEW index (blue/green)")
    print(f"     b) Validate the new index on the gold set")
    print(f"     c) Flip the read path atomically from old to new index")
    print(f"     d) Keep the old index for instant rollback, then retire it")

    print(f"\n  3. The ACL filter :")
    print(f"     Post-retrieval filtering is dangerous : the vector DB already")
    print(f"     RETURNED off-ACL chunks; a single bug downstream leaks them.")
    print(f"     Defense-in-depth (independent layers) :")
    print(f"     1. PRE-FILTER : push acl/tenant constraints INTO the vector DB query")
    print(f"     2. POST-CHECK : re-verify each returned chunk's ACL before use")
    print(f"     3. OUTPUT GUARD : scan the final answer for out-of-scope PII")
    print(f"     4. AUDIT LOG : record which chunks were served to whom")
    print(f"     Test : 'user A can NEVER retrieve a doc owned only by user B' as a")
    print(f"     CI test -- it would have caught the int-vs-string type bug instantly.")

    print(f"\n  4. The quality regression :")
    print(f"     No gold-set re-eval after swapping the embedding model meant the")
    print(f"     broken retrieval (mixed spaces) shipped silently. A CI/CD gate must")
    print(f"     require : (a) full re-embed, (b) gold-set recall >= baseline,")
    print(f"     (c) isolation test green -- all mandatory before promotion.")

    print(f"\n  5. Resilience & response :")
    print(f"     Confidentiality circuit breaker : if retrieval scores are aberrant")
    print(f"     (e.g. all near-equal, or below a sanity floor) OR any returned chunk")
    print(f"     fails the post-check ACL, REFUSE to answer rather than risk a leak.")
    print(f"     Runbook (8 steps) :")
    steps = [
        "Cut the assistant immediately (kill switch), serve a maintenance message",
        "Freeze the index (no further reads/writes)",
        "Scope the blast radius : which users saw which off-ACL data (audit log)",
        "Roll back to the last known-good index (single embedding version + pre-filter)",
        "Trigger the legal/compliance path (GDPR breach notification clock starts)",
        "Add the isolation test + embedding-version gate to CI before any redeploy",
        "Re-embed the whole corpus consistently, validate on the gold set",
        "Post-mortem within 24h : timeline, root causes, corrective actions, owners",
    ]
    for i, s in enumerate(steps, 1):
        print(f"       {i}. {s}")

    # ---- assertions ----
    categories = {c[0] for c in chain}
    assert {"PROCESS", "ARCHITECTURE", "SECURITY", "QUALITY"} <= categories, categories
    assert len(chain) == 7, len(chain)
    assert len(steps) == 8, len(steps)
    assert steps[0].lower().startswith("cut"), "runbook must start by cutting the assistant"
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 10 HARD : RAG ARCHITECTURE")
    print(SEPARATOR)
    hard_1_multitenant()
    hard_2_postmortem()
    print(f"\n{SEPARATOR}")
    print("  END OF HARD SOLUTIONS")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
