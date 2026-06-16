"""
Solutions -- Day 10 MEDIUM Exercises: RAG Architecture

Worked solutions with the reasoning step by step. Assertions lock the
key calculations so the file is self-checking.

Usage:
    python3 10-rag-architecture-medium.py
"""

SEPARATOR = "=" * 70


# =============================================================================
# MEDIUM -- Exercise 1 : Size and budget a documentation RAG
# =============================================================================

def medium_1_sizing():
    """Sizing + cost of a documentation RAG pipeline."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Size and budget a documentation RAG")
    print(SEPARATOR)

    pages = 2_000_000
    words_per_page = 600
    tokens_per_word = 1 / 0.75          # 1 token ~ 0.75 word -> ~1.33 tokens/word
    chunk_tokens = 500
    overlap_pct = 0.15
    dims = 1024
    bytes_per_dim = 4
    hnsw_overhead = 1.5
    daily_requests = 200_000
    peak_factor = 4
    embed_price_per_1m = 0.02           # $/1M tokens

    # 1. Number of chunks (overlap inflates the effective corpus by ~15%)
    total_tokens = pages * words_per_page * tokens_per_word
    effective_tokens = total_tokens * (1 + overlap_pct)   # overlap = re-embedded text
    n_chunks = round(effective_tokens / chunk_tokens)
    print(f"\n  1. Chunks :")
    print(f"     Corpus tokens (no overlap) : {total_tokens:,.0f}")
    print(f"     With {overlap_pct:.0%} overlap        : {effective_tokens:,.0f}")
    print(f"     Chunks ({chunk_tokens} tok each)    : {n_chunks:,}")
    print(f"     -> 1 vector per chunk = {n_chunks:,} vectors to index")

    # 2. Dense index size
    raw_index_bytes = n_chunks * dims * bytes_per_dim
    raw_index_gb = raw_index_bytes / (1024 ** 3)
    hnsw_gb = raw_index_gb * hnsw_overhead
    print(f"\n  2. Dense index size :")
    print(f"     Raw vectors : {n_chunks:,} * {dims} * {bytes_per_dim}B = {raw_index_gb:.1f} GB")
    print(f"     With HNSW overhead ({hnsw_overhead}x) : {hnsw_gb:.1f} GB")
    print(f"     -> fits comfortably in RAM on a few nodes")

    # 3. QPS
    avg_qps = daily_requests / 86_400
    peak_qps = avg_qps * peak_factor
    print(f"\n  3. QPS :")
    print(f"     Avg  : {daily_requests:,} / 86400 = {avg_qps:.2f} req/s")
    print(f"     Peak : {avg_qps:.2f} * {peak_factor} = {peak_qps:.1f} req/s")
    print(f"     Low QPS -> the cost driver is latency/quality, not throughput")

    # 4. Latency budget (p95)
    rewrite_ms = 150       # small LLM call
    retrieval_ms = 120     # HNSW dense + BM25 in parallel
    rerank_ms = 250        # cross-encoder on top-50
    generation_ms = 1800   # main LLM streaming, dominates
    total_ms = rewrite_ms + retrieval_ms + rerank_ms + generation_ms
    print(f"\n  4. Latency budget (p95, target < 2500ms) :")
    budget = [
        ("query rewrite (nano LLM)", rewrite_ms),
        ("hybrid retrieval (dense||BM25)", retrieval_ms),
        ("rerank cross-encoder (top-50)", rerank_ms),
        ("generation (main LLM)", generation_ms),
    ]
    for name, ms in budget:
        print(f"     {name:<35} {ms:>5} ms ({ms/total_ms:.0%})")
    print(f"     {'TOTAL':<35} {total_ms:>5} ms")
    print(f"     -> GENERATION dominates ({generation_ms/total_ms:.0%}). Optimize there first")
    print(f"        (smaller model, streaming TTFT, prompt compression).")

    # 5. HNSW vs IVF
    print(f"\n  5. HNSW or IVF ?")
    print(f"     HNSW. Volume is {n_chunks/1e6:.0f}M vectors (< 100M) and SLA is tight.")
    print(f"     HNSW gives 95%+ recall with single-digit ms latency and fits in RAM.")
    print(f"     IVF would save memory but costs recall and needs tuning of nprobe;")
    print(f"     only worth it at 100M+ vectors or under hard memory constraints.")

    # 6. Weekly re-embedding cost
    weekly_embed_cost = effective_tokens * embed_price_per_1m / 1_000_000
    monthly_embed_cost = weekly_embed_cost * (52 / 12)
    print(f"\n  6. Weekly full re-indexing embedding cost :")
    print(f"     Tokens to embed : {effective_tokens:,.0f}")
    print(f"     Cost / week  : {effective_tokens:,.0f} * ${embed_price_per_1m}/1M = ${weekly_embed_cost:,.0f}")
    print(f"     Cost / month : ~${monthly_embed_cost:,.0f}")
    print(f"     -> a strong argument for INCREMENTAL re-indexing instead of full weekly.")

    # ---- assertions on the key numbers ----
    assert 1_500_000_000 < total_tokens < 1_700_000_000, total_tokens   # ~1.6B tokens
    assert 3_400_000 < n_chunks < 3_900_000, n_chunks                   # ~3.68M chunks
    assert 12 < raw_index_gb < 16, raw_index_gb                         # ~14 GB raw
    assert 18 < hnsw_gb < 24, hnsw_gb                                   # ~21 GB
    assert abs(avg_qps - 2.31) < 0.1, avg_qps
    assert abs(peak_qps - 9.26) < 0.5, peak_qps
    assert generation_ms / total_ms > 0.7, "generation must dominate the budget"
    assert total_ms < 2500, "p95 budget must fit the SLA"
    assert 30 < weekly_embed_cost < 45, weekly_embed_cost              # ~$37/week
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 2 : Diagnose a RAG that answers off-target
# =============================================================================

def medium_2_diagnosis():
    """Diagnose a low-quality RAG and prioritize fixes."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Diagnose a RAG that answers off-target")
    print(SEPARATOR)

    recall_dense = 0.62
    recall_hybrid = 0.71
    faithfulness = 0.74
    context_precision = 0.38

    print(f"\n  1. Which stage is the weak link ?")
    print(f"     RETRIEVAL. recall@5 is only {recall_hybrid:.0%} even with hybrid, and")
    print(f"     context precision is {context_precision:.0%} (~3 of 5 chunks are noise).")
    print(f"     The generator can only be as good as the context it receives.")

    print(f"\n  2. Is 74% faithfulness a generation or a retrieval problem ?")
    print(f"     Mostly a CONSEQUENCE of weak retrieval. If the right passage is")
    print(f"     missing from the top-5 ({1-recall_hybrid:.0%} of the time), the model either")
    print(f"     guesses (hallucination) or answers from noise. Faithfulness is")
    print(f"     CAPPED by recall : you cannot be faithful to context you never got.")

    print(f"\n  3. Why do exact product codes fail ?")
    print(f"     Dense embeddings smear exact tokens (REF-4471-B looks like REF-4471-C).")
    print(f"     The fix is the SPARSE side : BM25 matches the exact string. Either")
    print(f"     BM25 is under-weighted in the fusion, or there is no reranker to")
    print(f"     promote the exact-match chunk. Hybrid + reranker is the answer.")

    print(f"\n  4. The 1500-token, 0-overlap chunk :")
    print(f"     A huge chunk dilutes relevance : 1 useful sentence buried in 1500")
    print(f"     tokens of noise -> low context precision. 0 overlap also cuts facts")
    print(f"     across boundaries. New config : ~500 tokens, 10-20% overlap.")

    print(f"\n  5. Actions ranked by impact/cost :")
    actions = [
        ("Add cross-encoder reranker on top-50",
         "+12-18% recall@5", "moderate (latency + API)", "#1"),
        ("Re-chunk to 500 tok + 15% overlap",
         "+5-10% recall@5, +precision", "one-off re-indexing", "#2"),
        ("Up-weight BM25 in RRF / tune fusion",
         "+3-6% on exact-code queries", "cheap (config)", "#3"),
    ]
    for action, gain, cost, rank in actions:
        print(f"     [{rank}] {action}")
        print(f"          gain={gain}, cost={cost}")
    expected_recall_after = recall_hybrid + 0.15 + 0.07
    print(f"     Expected recall@5 after #1+#2 : ~{min(expected_recall_after, 0.93):.0%}")

    print(f"\n  6. Risk of top-k=20 without vs with reranker :")
    print(f"     Without reranker : more noise in the context -> faithfulness DROPS")
    print(f"     and cost rises (more input tokens). Without a way to re-sort, the")
    print(f"     LLM sees 20 chunks of which only ~2 are useful.")
    print(f"     With reranker : retrieve top-20, rerank, keep top-5. You get the")
    print(f"     recall benefit of a wider net WITHOUT feeding noise to the LLM.")

    # ---- assertions ----
    assert recall_hybrid > recall_dense, "hybrid should beat dense"
    assert context_precision < 0.5, "context precision is the smoking gun"
    assert faithfulness <= recall_hybrid + 0.05, "faithfulness is capped by recall"
    assert min(expected_recall_after, 0.93) >= 0.90, expected_recall_after
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 3 : Choose GraphRAG vs Agentic RAG vs hybrid
# =============================================================================

def medium_3_architecture_choice():
    """Map 3 need profiles to the right advanced RAG architecture."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : GraphRAG vs Agentic RAG vs hybrid classique")
    print(SEPARATOR)

    print(f"\n  1. Recommendations :")
    recs = [
        ("A - Legal (multi-hop, entities, cross-refs)", "GraphRAG (+ ColBERT for rare clause terms)",
         "Info is spread across docs; a knowledge graph traverses relations and "
         "ColBERT's token-level match catches rare legal jargon."),
        ("B - Support (factual, direct FAQ)", "Hybrid classique (dense + BM25 + RRF)",
         "Stable, single-hop questions. The canonical pipeline already hits 90%+; "
         "anything more is over-engineering."),
        ("C - Research (open, exploratory, verify sources)", "Agentic RAG (CRAG / Self-RAG)",
         "Open-ended queries need dynamic search, query reformulation, and source "
         "checking. An evaluator routes ambiguous results to a web search."),
    ]
    for team, arch, why in recs:
        print(f"     {team}")
        print(f"       -> {arch}")
        print(f"          {why}")

    print(f"\n  2. Team A : why does flat dense retrieval fail on multi-hop ?")
    print(f"     The answer ('non-compete clauses in contracts with subsidiaries of X')")
    print(f"     requires JOINING facts: which companies are subsidiaries of X, then")
    print(f"     which contracts they signed, then which contain the clause. A flat")
    print(f"     dense search returns chunks similar to the QUERY, not chunks that")
    print(f"     chain together. The graph encodes the relations explicitly.")

    # 3. Cost of agentic RAG on a FAQ
    base_calls = 1
    agentic_multiplier_low, agentic_multiplier_high = 2, 4
    print(f"\n  3. Team B : why NOT agentic RAG ?")
    print(f"     Agentic RAG costs {agentic_multiplier_low}-{agentic_multiplier_high}x the LLM calls per query")
    print(f"     (evaluate, reformulate, re-retrieve). For 'how do I reset my password',")
    print(f"     that's {agentic_multiplier_low}-{agentic_multiplier_high}x the cost and latency for ZERO quality gain on a")
    print(f"     single-hop factual question. The juice is not worth the squeeze.")

    print(f"\n  4. Team C : which agentic pattern ?")
    print(f"     CRAG. A lightweight evaluator labels retrieved results as")
    print(f"     correct / ambiguous / incorrect. On 'ambiguous' it triggers a web")
    print(f"     search; on 'incorrect' it reformulates. For untrusted sources :")
    print(f"     score source credibility, require a verifiable quote per claim,")
    print(f"     cross-check across >=2 independent sources before asserting a fact.")

    print(f"\n  5. Risks + mitigations :")
    risks = [
        ("A (GraphRAG)", "High indexing cost (LLM extracts entities/relations); graph rebuild on corpus change",
         "Incremental graph updates; batch entity extraction off-peak"),
        ("B (hybrid)", "Plateaus on rare exact terms",
         "Add reranker; up-weight BM25; only escalate if the gold set shows a real gap"),
        ("C (agentic)", "Latency + cost spikes, possible loops",
         "Hard budget (max steps/tokens), step timeout, stop criteria"),
    ]
    for team, risk, mit in risks:
        print(f"     {team}: risk={risk}")
        print(f"             mitigation={mit}")

    print(f"\n  6. 'The best RAG possible, no budget limit' -- good approach ?")
    print(f"     No. Start with a gold set and the simplest pipeline that passes it.")
    print(f"     Stacking GraphRAG + agentic + ColBERT blindly adds cost, latency and")
    print(f"     failure modes. The simplest system that hits the target wins.")
    print(f"     Measure on YOUR data, then add complexity only where it pays.")

    # ---- assertions ----
    assert agentic_multiplier_high == 4
    assert agentic_multiplier_low * base_calls >= 2, "agentic must be >= 2x"
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 10 MEDIUM : RAG ARCHITECTURE")
    print(SEPARATOR)
    medium_1_sizing()
    medium_2_diagnosis()
    medium_3_architecture_choice()
    print(f"\n{SEPARATOR}")
    print("  END OF MEDIUM SOLUTIONS")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
