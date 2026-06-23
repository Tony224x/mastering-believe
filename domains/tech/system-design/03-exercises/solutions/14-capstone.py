"""
Solutions -- Day 14 : Capstone

The 3 capstone exercises are long. The solutions are provided as
complete walkthroughs in docstrings. Each solution follows the
6-step framework : clarify -> estimate -> HL design -> deep dive
-> bottlenecks -> extensions.
"""


def solution_exercice_1() -> None:
    """
    Exercise 1 -- Spotify Recommendations.

    === CLARIFY ===
    Functional :
      - Personalized home feed (mix of tracks + playlists)
      - Discover Weekly : 30 new tracks every Monday
      - Personalized radios from a track / artist
      - Feedback (like/dislike/skip) feeds the recos
      - Cold start for new users

    Non-functional :
      - 500M MAU, 200M DAU, peak 50-80M concurrent
      - Home feed latency < 500 ms p95
      - Recos updated in "near real-time" (after a like, the effect
        must be visible within minutes)
      - Cost efficient : 50 recos/user/day = 10 G recos/day

    === CAPACITY ===
      Home requests : 200M users * 10 visits / day * 50 items = 100 G items/day
      Peak factor 3x -> 3.5M items/sec. But the items are batched
      server-side -> we rather count "home requests" :
      200M * 10 = 2 G home requests / day = 23K rps avg, ~70K peak.
      Materialized reco storage : 200M users * 50 reco_slots * 20 B (ids)
      = 200 GB. Nothing.
      Feature store : 200M users * ~50 features = 10 G cells.
      Track vector embeddings : 100M * 256 dims * 4 bytes = 100 GB.
      Event ingestion (plays, skips, likes) : 200M users * 50 events/day
      = 10 G events/day, stored for retraining.

    === HIGH LEVEL DESIGN ===
                                          +-----------+
        client app ---> LB ---> API ---+->| Recos svc |---+
                                       |  +-----------+   |
                                       v                  v
                                 +-----------+      +----------+
                                 | Home      |      | Radios   |
                                 | Composer  |      | svc      |
                                 +-----+-----+      +----------+
                                       |
                          fetch cached     fetch online scores
                          recos (Redis)    (feature store online)
                                       |
                     +-----------------+------------------+
                     |                 |                  |
                     v                 v                  v
              +------------+    +-------------+    +-------------+
              | Candidate  |    | Ranker      |    | Diversifier |
              | generator  |    | (XGBoost /  |    | + freshness |
              |            |    |  2-tower NN)|    |             |
              +------------+    +-------------+    +-------------+
                     ^
                     |
              +------+------+
              | Feature     |
              | store       |
              | (offline +  |
              |  online)    |
              +-------------+
                     ^
                     |
              +-------------+      +-------------+
              | Batch jobs  |<-----| Event stream|
              | Spark/Flink |      | Kafka       |
              | (Discover,  |      | (plays,     |
              |  playlists) |      |  skips...)  |
              +-------------+      +-------------+

    === DEEP DIVE ===

    1) Feature store
      - Offline (BigQuery / Parquet) : per-user histories (avg skip rate,
        diversity index, genre preferences 30d, listening time by time-of-day)
      - Online (Redis / Dynamo) : latest materialized features, key =
        user_id, value = pack of features needed by the ranker
      - Materialization : each batch/stream feature has its own pipeline

    2) Discover Weekly batch
      - Weekly Spark job that :
        a) joins the user embeddings (learned by 2-tower) with the
           track embeddings
        b) computes the top-500 candidates per user via ANN in the
           embedding space
        c) applies filters (not already listened, diversity, freshness)
        d) materializes 30 tracks per user into Redis with a 1-week TTL
      - Volume : 200M users * 30 = 6G writes -> doable in a few
        hours on a properly sized Spark cluster

    3) Real-time home scoring
      - Candidates : mix of 3 sources
        a) Precomputed Discover Weekly (40% weight)
        b) Batch collaborative filtering (30% weight)
        c) Context-aware online (30% weight) : time of day,
           device, latest listen
      - Online ranker : XGBoost or a light 2-tower, features read from the
        online store, latency < 30 ms
      - Final re-ranking : diversity constraint (not 3 tracks by the
        same artist in the top 5), freshness boost, A/B test variants
      - All of this runs in < 200 ms. With a Redis cache for returning
        users, < 50 ms on a cache hit.

    === BOTTLENECKS ===
      B1. Online feature store : hot key (top users) -> local cache
          on the ranker pods + replicas sharded by user_id.
      B2. Ranker CPU : the peak QPS saturates the ranker pods -> HPA
          autoscale on CPU + queue depth, ranker batched in micro-batches of 10.
      B3. Event ingestion : Kafka -> Flink -> feature store, lag on
          like bursts -> buffer + retries + lag monitoring.

    === EXTENSIONS ===
      - Real-time personalization + updates : after each skip/like,
        adjust the online features within 1 s.
      - LLM-generated descriptions ("Because you listen to
        X, here is Y") : the LLM is used ONLY for the human-facing text,
        never for scoring.
      - Cold-start for new users : onboarding with a few
        explicit preferences + content-based on the first
        session.
    """


def solution_exercice_2() -> None:
    """
    Exercise 2 -- GitHub Copilot backend.

    === CLARIFY ===
    Functional :
      - Inline code completion while the user types
      - Multi-line suggestions
      - Context-aware : knows what is in the current file + neighboring
        files
      - Streaming : tokens arrive one by one
      - Cancellation : a new user keystroke -> cancels the in-flight completion
      - Multi-language (TS, Py, Go, Rust...)

    Non-functional :
      - 10M devs / day
      - p95 latency < 300 ms end-to-end (TTFT)
      - Uptime 99.9%+
      - Privacy : enterprise can require "don't send code out of VPC"
      - Cost : extremely tight budget ($0.001 - $0.003 per completion)

    === CAPACITY ===
      Completions / day : 10M * 30/hr * 8 hr = 2.4 G completions / day
      -> ~28K rps avg, peak 80K rps.
      Tokens per call : ~2500 in + 80 out
      -> 6T tokens in / day. Gigantic.
      Estimated cost (hosted code-specialized model) : $0.0015 / completion
      -> $3.6M / day of raw cost.
      -> obviously not sustainable without routing + caching + smaller models.
      Margin : each dev pays $10-20/month -> $100M/month revenue.
      Infra target : < 20% of revenue = $20M/month infra budget.

    === HIGH LEVEL DESIGN ===

        IDE client ----> edge router (geo) ----+
                              |                |
                      cancellation              v
                      handler          +----------------+
                                       | Context engine |
                                       | - open buffer  |
                                       | - nearby files |
                                       | - symbol index |
                                       +--------+-------+
                                                |
                                                v
                                     +----------------------+
                                     | Prefix cache /       |
                                     | KV sharing           |
                                     +-----------+----------+
                                                 |
                                                 v
                                     +----------------------+
                                     | Routing layer        |
                                     | - small model 70% req|
                                     | - big model 25% req  |
                                     | - giant model  5% req|
                                     +-----------+----------+
                                                 |
                         +-----------+-----------+------------+
                         |           |           |            |
                         v           v           v            v
                    +---------+ +---------+ +---------+  +--------+
                    | Small   | | Medium  | | Frontier|  | Enterp.|
                    | LLM     | | LLM     | | (rare)  |  | isolated
                    | (A100)  | | (H100)  | | (H100)  |  | deploy |
                    +---------+ +---------+ +---------+  +--------+

    === DEEP DIVE ===

    1) How to hold < 300 ms p95 ?
      - Main model quantized (int8) running on aggressively batched
        GPUs (vLLM continuous batching)
      - Streaming : return the first tokens from ~100 ms (TTFT)
      - Prefix cache : if the prompt starts with the same code as the
        previous request, the KV cache is reused (2-3x gain)
      - Routing : 70% of the requests go to the small model (~100 ms
        inference) and are only escalated if needed

    2) Cache strategy
      - Completion-level cache : key = hash(buffer prefix). If the same
        completion was already served, return it directly. Real hit
        rate : 5-10% (lots of variation) but free.
      - KV prefix cache on the model : if the first 2000 tokens of the
        prompt are the same, the KV is reused (below the model,
        handled by vLLM).
      - Embedding-based cache : for common patterns ("def
        process_" "return ..."), a semantic cache can catch them.

    3) Routing by tier
      - Small (quantized code-llama-7b, hosted) : 70% of the requests
        (simple code, 1-line completion)
      - Medium (starcoder-15b or dedicated) : 25% (multi-line, logic)
      - Frontier (GPT-5.4) : 5% (ambiguous cases, or "explain-this")
      - Enterprise : deployed in the client's VPC, dedicated model

    === SPECIFIC CHALLENGES ===

    Token-by-token streaming :
      - The API must be SSE (Server-Sent Events) or HTTP streaming
      - The IDE reads as it goes and updates the preview
      - No buffering on the CDN side !

    Cancellation :
      - Each request has a request_id
      - If the user types a new key, the IDE sends
        POST /cancel/{request_id}
      - On the serving side : the in-flight job is interrupted, the GPU moves on
        to the next one (vLLM supports it with abort_request)
      - Critical to avoid wasting GPU on abandoned requests

    Enterprise privacy :
      - Two distinct deployments :
        a) Public : multi-tenant, centralized model
        b) Enterprise : VPC isolated, either self-hosted or peered
           with the client VPC, encrypted logs, no data retained
      - Enterprise pays more (2-3x) for this comfort

    === BOTTLENECKS ===
      - GPU capacity : aggressive autoscale + reservation guarantee for
        enterprise
      - Context engine : file indexing must be fast
        (tree-sitter + light embedding on the client side)
      - Network latency : multi-region deployment (US, EU, Asia)

    === EXTENSIONS ===
      - Copilot Chat : same infra but with a long context and tools
      - Multi-file editing : agentic mode that edits several files
      - Inline tests : generate a test next to the function
      - Tenant fine-tuning : adapt the model to a client's own code
      - Telemetry to fine-tune future models (with consent)
    """


def solution_exercice_3() -> None:
    """
    Exercise 3 -- Autonomous Research Agent.

    === CLARIFY ===
    Functional :
      - Input : a topic ("state of the art of batteries 2026")
      - Output : 10-20 page markdown report, verifiable citations
      - Real-time progress tracking
      - Interrupt / resume

    Non-functional :
      - Run : 5-30 minutes, async
      - Budget : $5-20 per run in LLM cost
      - 100-500 search queries max
      - Robust to sites being down

    Definition of "done" :
      - The critic agent validates that the report covers all the sub-topics
      - Budget not exceeded
      - Completeness score > 80%
      - OR the user interrupts

    === MULTI-AGENT ARCHITECTURE (hierarchical) ===

                              +-----------------+
                              | Orchestrator    |
                              | (master plan)   |
                              +--------+--------+
                                       |
              +------------------------+-----------------------+
              |                        |                       |
              v                        v                       v
        +------------+           +-------------+         +------------+
        | Planner    |           | Executor    |         | Reviewer   |
        | - decompose|           | supervisor  |         | - critic   |
        | - subquest |           +------+------+         | - rewrite  |
        +------------+                  |                +------------+
                                        |
                        +---------------+---------------+
                        |               |               |
                        v               v               v
                 +-----------+   +-----------+   +-----------+
                 | Searcher  |   | Reader    |   | Summarizer|
                 | agent     |   | agent     |   | agent     |
                 +-----------+   +-----------+   +-----------+
                        |               |
                        v               v
                 +-----------+   +-----------+
                 | Web search|   | Content   |
                 | tool      |   | extractor |
                 +-----------+   +-----------+

    === DEEP DIVE ===

    1) Planning
      - The planner uses a strong LLM to decompose the question into
        5-10 covering sub-questions (coverage check)
      - Each sub-question has : objective, priority, estimated budget
      - The plan is versioned : it can be modified during the run

    2) Source evaluation
      - Each source receives a credibility score :
        - Domain reputation (scientific whitelist, low-quality blacklist)
        - Publication date
        - Citations present
        - LLM-as-a-judge : is it a primary source ?
      - Score < threshold -> reject
      - Diversity : avoid using 5 sources from the same site

    3) Citation verification
      - Every claim in the report is linked to a source chunk
      - Before writing the report, the writer must provide the chunks
      - After generation, an exact-match check verifies that the quotes
        exist in the sources. If not : alert and regenerate.
      - The URL + the exact quote are stored to let the user
        verify.

    4) Memory management
      - Working memory (current context) : current plan + top 5
        active sources + summary of the recent searches
      - Long-term memory : a vector store indexes all the sources
        read during this run (allows retrieving them later)
      - When the context approaches the max, summarize the oldest
        sources and archive them in the long-term store

    === FAILURE MODES ===

    F1. Infinite loop (the agent searches for the same things)
      - Detection : if the same sub-goal is retried > 3 times
        without progress -> skip + note
      - Hard cap : max_iterations = 50

    F2. Budget explosion
      - Every LLM call and search is counted against the budget
      - Warning at 70%, hard stop at 100%
      - The report can be "partial" with a disclaimer

    F3. Source hallucination
      - Groundedness check before writing : every citation is verified
      - An independent critic (different model) re-reads and flags the
        dubious citations
      - If a hallucination is detected -> regenerate the section OR flag
        "unverified" in the report

    === OBSERVABILITY ===

    - Langfuse trace per run, with sub-spans per agent and per step
    - Live progress : a "run_id -> status" table is updated at each
      step. The frontend polls or uses SSE to display "30/100
      searches done, 5/10 sub-questions covered"
    - Budget tracker visible live
    - A "debug" mode shows the prompt and the response of each
      sub-agent

    === EXTENSIONS ===

    - Human interrupt : the user can pause, redirect
      ("focus on the chemistry, not the manufacturers"),
      resume
    - Collaboration between agents : an agent can ask another
      agent for help when it is stuck
    - Reuse : already-performed searches are cacheable. If
      another user asks the same question, we reuse them.
    - Automatic verification : before delivering the report, a
      "fact-checker" agent repeats part of the queries to verify
      that the facts still hold.
    - Multiple output formats : markdown, PDF, slides
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)
