"""
Solutions -- Day 10 : RAG Architecture
"""


def solution_exercice_1() -> None:
    """
    Exercise 1 -- Chunking strategies.

    +--------------------------+-------------------+---------+---------+
    | Corpus                   | Strategy          | Size    | Overlap |
    +--------------------------+-------------------+---------+---------+
    | SaaS Markdown docs       | document-aware    | 400-800 | 10 %    |
    | (split by H1/H2/H3)      |                   | tokens  |         |
    +--------------------------+-------------------+---------+---------+
    | Support transcripts      | recursive         | 300-500 | 15 %    |
    +--------------------------+-------------------+---------+---------+
    | Python code              | per function /    | 200-500 | 0 %     |
    |                          | class (tree-sit)  |         |         |
    +--------------------------+-------------------+---------+---------+
    | Legal contracts          | document-aware    | 1000-   | 20 %    |
    |                          | (per section /    | 1500    |         |
    |                          | clause)           | tokens  |         |
    +--------------------------+-------------------+---------+---------+
    | HTML blog (500-3000 words)| recursive + tag- | 400-600 | 10 %    |
    |                          | aware (h2, p)     |         |         |
    +--------------------------+-------------------+---------+---------+
    | Tweets / short-form      | 1 post = 1 chunk  | ~50     | 0       |
    +--------------------------+-------------------+---------+---------+

    Justifications (excerpts) :
    - Markdown with headers : the semantic structure is already there. It
      would be absurd to cut it while ignoring the H2/H3. We gain
      "section" granularity.
    - Code : we chunk per function because a function is a complete logical
      block. A model understands a whole function better than a piece
      from the middle.
    - Contracts : a chunk cut at the wrong place can omit a
      decisive clause. We prefer more context and more overlap.
    - Tweets : the natural size is < 300 chars. No splitting.

    Tradeoff reminder : smaller = more precise but risks losing
    context. Larger = more context but more noise and more expensive
    queries for the LLM.
    """


def solution_exercice_2() -> None:
    """
    Exercise 2 -- Debugging a RAG.

    Symptoms : 40% incorrect answers, recall@10=72%, chunks with the right
    keywords but the wrong context. GPT-4, fixed 512 chunks, dense-only,
    no reranker.

    Root cause hypotheses + experiments :

    H1. **Missing reranker** (most probable, quick win)
        - Exp : add a cross-encoder (BGE-reranker or Cohere Rerank)
          on the top 20 dense candidates, keep only the top 5.
        - Expected : recall@5 after rerank > 90%, faithfulness rises.
        - Cost : ~50 ms of extra latency, that's acceptable.
        - PRIORITY 1 : 1 line of code, immediate result.

    H2. **Missing hybrid retrieval (no BM25)**
        - Exp : index a BM25 in parallel and do RRF.
        - Expected : recall@10 goes from 72% to ~85%, especially on
          queries with proper nouns or identifiers.
        - PRIORITY 2 : medium effort, 10-20% gain.

    H3. **Fixed-size chunking breaks the context**
        - Exp : switch to recursive chunking (paragraph / sentence
          separators) or semantic chunking. Test with the same size.
        - Expected : the chunks contain complete units of meaning,
          faithfulness increases.
        - PRIORITY 3 : requires reindexing (a few hours-days).

    H4. **Weak generation prompt**
        - Exp : add explicit instructions "use ONLY
          the provided passages, if the info is not there say 'I don't
          know'", force [N] citations.
        - Expected : faithfulness rises (fewer hallucinations).
        - PRIORITY 2 : trivial, test first.

    H5. **Embedding model too weak for the domain**
        - text-embedding-3-small is 1536 dims and general-purpose.
          On a specialized domain (medical, legal, finance),
          domain-specific models (BGE, Voyage, Cohere) perform
          better.
        - Exp : swap the embedder on a subset of the corpus and
          compare recall@10.
        - PRIORITY 4 : very impactful but implies reindexing everything.

    H6. **No query rewriting / expansion**
        - User queries are often short and ambiguous. An
          LLM can reformulate them before retrieval.
        - Exp : HyDE (generate a hypothetical answer and embed
          that) or classic query expansion.
        - PRIORITY 3 : 5-15% gain.

    Prioritization : H1 -> H4 -> H2 -> H3 -> H6 -> H5.
    (Quick wins first, structural changes afterwards.)

    Metrics to measure after each fix :
    - Recall@10 (pure retrieval) on the gold set
    - MRR
    - Faithfulness (LLM-as-a-judge)
    - Answer relevance
    - p99 latency (verify we are not degrading)
    - Cost per query

    Rule : one change at a time. Otherwise it's impossible to attribute
    the metric variations.
    """


def solution_exercice_3() -> None:
    """
    Exercise 3 -- Sizing the infra for a legal RAG.

    Assumptions :
      500K docs * 15 pages * 500 tokens/page = 3.75 B tokens
      Chunks of 800 tokens, 15% overlap = effective size ~680 tokens
      => 3.75B / 680 = ~5.5 M chunks

    Q1 -- Chunks : ~5.5 M

    Q2 -- Vector index memory :
      5.5M * 3072 dims * 4 bytes (float32) = 67.6 GB
      With HNSW overhead (~30%) : ~87 GB.
      With int8 quantization : 67.6/4 = 17 GB -- a serious option.
      With binary quantization : ~2 GB -- but noisier.

    Q3 -- DB choice :
      - pgvector : too slow / memory-hungry for 5.5M * 3072 dims. No.
      - Pinecone : ~$70 * 5.5 = $385/month minimum. Simple, zero ops.
      - Qdrant Cloud : memory-based billing, ~$200-400/month for 17 GB
        quantized. Good compromise.
      - Qdrant self-hosted : a 32 GB RAM VM ~$150/month. The cheapest.
      Recommendation : **Qdrant self-hosted** if there is a DevOps team,
      otherwise Qdrant Cloud. Pinecone if you want zero friction.

    Q4 -- Initial indexing cost :
      Tokens to embed : 3.75B input tokens.
      text-embedding-3-large : $0.13 / 1M tokens
      => 3750 * 0.13 = **$487.5** (one-shot, not recurring).
      Tip : with text-embedding-3-small ($0.02 / 1M), it's
      $75. Test the quality of small first before paying for large.

    Q5 -- Monthly runtime cost :
      Queries : 50 users * 5 q/min * 60 min * 8 h peak * 22 days
             = 2,640,000 queries / month.
      (aggressive assumption -- in practice much fewer at the real peak)

      Per query :
        - Embed query : ~50 tokens -> 50 * $0.13/1M = $0.0000065
        - Retrieve : covered by the DB subscription
        - LLM generation : ~3000 input tokens (5 chunks * 600 tokens
          context) + 400 output tokens
          GPT-4o-mini : 3000*0.15/1M + 400*0.60/1M = $0.00045 + $0.00024
                     = ~$0.00069 / query
      Total per query : ~$0.00070
      Monthly : 2.64M * 0.00070 = **~$1848 / month**.
      We are almost at the budget! And this scenario is pessimistic.

      DB cost : $200 (Qdrant Cloud) or $150 (self-hosted).
      => Total : ~$2000 - $2100 / month. It is VERY tight.

    Q6 -- Optimizations to stay under $2000 :
      1. **Switch to text-embedding-3-small** : quality often sufficient,
         6x lower indexing cost, dimension 1536 so the index is 2x
         smaller.
      2. **Vector quantization** (int8 in Qdrant) : memory /4,
         DB cost goes down.
      3. **Semantic caching** on the frequent queries : 20-40% of the
         queries are variations of the same question -> a cache hit
         avoids an LLM call.
      4. **Reduce the top-k** passed to the LLM : from 5 to 3 chunks -> input
         tokens drop by 40%.
      5. **Reranker** to win on the top-k : 3 good chunks are worth
         more than 5 mediocre ones.
      6. **LLM distillation** to a smaller model (Haiku,
         self-hosted Llama-3-8B) for the specific use case.
      7. **Request batching** if the SLA allows it.
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)
