"""
Solutions -- Day 14 HARD Exercises: Capstone extensions

Worked solutions with the reasoning step by step. Assertions lock the
key calculations / structural checks so the file is self-checking.

These EXTEND the two J14 reference designs (Dropbox + LLM Support Assistant);
they do not redesign them.

Usage:
    python3 14-capstone-hard.py
"""

SEPARATOR = "=" * 70


# =============================================================================
# HARD -- Exercise 1 : Zero-downtime migration + rollback of the support bot
# =============================================================================

def hard_1_migration_rollback():
    """Industrial deployment strategy for the J14 support assistant."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Zero-downtime migration + rollback (support assistant)")
    print(SEPARATOR)

    print(f"\n  1. Deployment strategy :")
    print(f"     Pipeline per change : shadow -> canary (1% -> 10% -> 50% -> 100%) -> promote.")
    print(f"     ISOLATE the 3 changes (model, prompt, index) -- one at a time. If you")
    print(f"     ship them together and quality regresses, you cannot tell WHICH broke it.")
    print(f"     Re-indexing avoids the J10 embedding mismatch : re-embed the ENTIRE")
    print(f"     corpus into a NEW index (blue/green), validate, then flip atomically.")

    print(f"\n  2. Rollback (instant, < 1 min) :")
    print(f"     - Model : feature flag selecting the model version -> flip back instantly.")
    print(f"     - Prompt : prompt registry versioned -> point back to the old version.")
    print(f"     - Index : keep the OLD index live in parallel; flip the read pointer back.")
    print(f"     Keep old model + old prompt + old index ACTIVE until the new ones are")
    print(f"     validated. Prompt and index both need versioning; the index also needs")
    print(f"     a consistent embedding version (no mixing).")

    print(f"\n  3. Quality gates (J13) :")
    gates = [
        "gold-set recall & faithfulness >= baseline",
        "cost per conversation <= $0.10 target",
        "latency p95 within budget",
    ]
    for g in gates:
        print(f"     - {g}")
    print(f"     Detect canary regression early : score the canary slice continuously")
    print(f"     (LLM-as-a-judge faithfulness) and compare to baseline; auto-rollback")
    print(f"     if faithfulness or a guardrail metric degrades.")

    print(f"\n  4. Cost & portability (J11) :")
    print(f"     New model price : recompute $/conversation on the canary's real token")
    print(f"     usage; block promotion if > $0.10. A prompt tuned for the old model may")
    print(f"     misbehave on the new one -> re-run the gold set on the new model.")

    print(f"\n  5. A/B testing (J13) :")
    print(f"     Split by user_id hash, primary = business metric (resolution rate /")
    print(f"     CSAT), guardrails = latency/cost/faithfulness, run 2-4 weeks. 'Better")
    print(f"     offline' does NOT prove better in production (novelty, distribution).")

    print(f"\n  6. Failure modes :")
    print(f"     - 8h re-indexing, old+new index coexist : route by index VERSION so a")
    print(f"       request never mixes v_old and v_new vectors (J10 mismatch).")
    print(f"     - Canary faithfulness drops to 84% (< 90%) : auto-rollback, stop promote.")
    print(f"     - Rollback target deprecated by provider : keep a pinned old version OR")
    print(f"       an alternative fallback model already validated; never have a single")
    print(f"       irreversible path.")

    # ---- assertions ----
    assert len(gates) == 3, len(gates)
    assert any("faithfulness" in g for g in gates), "faithfulness gate required"
    print("\n  [assertions OK]")


# =============================================================================
# HARD -- Exercise 2 : Merge the two designs (Dropbox + AI document assistant)
# =============================================================================

def hard_2_merge_designs():
    """Combine the two J14 designs into one coherent system."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Merge the two designs (Dropbox + AI document assistant)")
    print(SEPARATOR)

    users = 50_000_000
    docs_per_user = 50
    chunks_per_doc = 20
    adoption = 0.05    # only 5% use the assistant

    print(f"\n  1. Combined architecture :")
    print(f"     Reuse from Dropbox-like : block store (S3), metadata DB (Postgres),")
    print(f"     CDN. Reuse from Support Assistant : LLM Gateway (router/cache/")
    print(f"     guardrails/fallback) + RAG engine (chunker, hybrid retriever, reranker).")
    print(f"     Indexing : LAZY (index a user's files on their FIRST question) is best")
    print(f"     given low adoption; index-at-upload wastes embedding on files never queried.")

    print(f"\n  2. Multi-user isolation (critical) :")
    print(f"     Each user must NEVER retrieve another user's chunk. Push the user_id")
    print(f"     filter as a PRE-FILTER INSIDE the vector DB query (not post-filter),")
    print(f"     plus an isolation test in CI. This is the exact J10 post-mortem lesson.")

    # 3. Indexing scale
    theoretical_chunks = users * docs_per_user * chunks_per_doc
    print(f"\n  3. Indexing scale :")
    print(f"     Theoretical : {users:,} * {docs_per_user} * {chunks_per_doc} = "
          f"{theoretical_chunks:,} chunks ({theoretical_chunks/1e9:.0f}B).")
    print(f"     That's why you DON'T index everyone upfront -> index on demand.")

    # 4. Cost of lazy indexing
    lazy_chunks = int(theoretical_chunks * adoption)
    saving = 1 - adoption
    print(f"\n  4. Cost -- lazy vs eager :")
    print(f"     Only {adoption:.0%} of users use the assistant -> indexing eagerly wastes")
    print(f"     ~{saving:.0%} of the embedding spend. Lazy indexing indexes only")
    print(f"     {lazy_chunks:,} chunks ({lazy_chunks/1e9:.1f}B) -> ~{saving:.0%} embedding cost saved.")

    print(f"\n  5. Reused patterns (J10-J13) :")
    patterns = [
        "hybrid search (dense + BM25) + RRF + cross-encoder reranker (J10)",
        "semantic cache PER USER (J11)",
        "input/output guardrails incl. PII (J11)",
        "tracing + cost spans (J13)",
        "drift monitoring on query topics (J13)",
    ]
    for p in patterns:
        print(f"     - {p}")
    print(f"     NON-NEGOTIABLE output guardrail : GROUNDEDNESS check -- every claim must")
    print(f"     map to a retrieved chunk of the USER'S OWN file (no hallucinated docs).")

    print(f"\n  6. Failure modes :")
    print(f"     - Cites a just-deleted file : on delete, invalidate that file's chunks")
    print(f"       in the index (tombstone) so it's no longer retrievable/citable.")
    print(f"     - Big user (1M files) drowns small users : per-user quotas + isolate")
    print(f"       big users on their own shard (noisy-neighbor control).")
    print(f"     - Revoked share / no read right : keep the ACL pre-filter current ->")
    print(f"       the chunk is no longer retrieved, so it can't appear in an answer.")

    # ---- assertions ----
    assert theoretical_chunks == 50_000_000_000, theoretical_chunks   # 50B
    assert lazy_chunks == 2_500_000_000, lazy_chunks                  # 5% = 2.5B
    assert abs(saving - 0.95) < 1e-9, saving
    assert len(patterns) == 5, len(patterns)
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 14 HARD : CAPSTONE EXTENSIONS")
    print(SEPARATOR)
    hard_1_migration_rollback()
    hard_2_merge_designs()
    print(f"\n{SEPARATOR}")
    print("  END OF HARD SOLUTIONS")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
