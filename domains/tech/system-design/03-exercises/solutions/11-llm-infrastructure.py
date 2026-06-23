"""
Solutions -- Day 11 : LLM Infrastructure
"""


def solution_exercice_1() -> None:
    """
    Exercise 1 -- Routing policy for an email assistant.

    +---------------+---------+-------------------------------------------+
    | Task          | Primary | Escalation condition                      |
    +---------------+---------+-------------------------------------------+
    | Classify      | nano    | If confidence < 0.8 -> mini               |
    |               |         | If spam + contentious -> std (false       |
    |               |         | positives are costly)                     |
    +---------------+---------+-------------------------------------------+
    | Summarize     | mini    | If thread > 10 emails or > 5K tokens ->   |
    |               |         | std                                       |
    +---------------+---------+-------------------------------------------+
    | Draft reply   | mini    | If a professional email with a senior     |
    |               |         | actor or a delicate tone (detected by a   |
    |               |         | nano classifier) -> std. If a complex     |
    |               |         | negotiation -> frontier                   |
    +---------------+---------+-------------------------------------------+
    | Extract       | nano    | If the nano classifier is unsure or the   |
    |               |         | strict JSON schema is invalid -> mini ->  |
    |               |         | std                                       |
    +---------------+---------+-------------------------------------------+
    | Answer        | std     | If the question requires deep reasoning   |
    | question      |         | (multi-step, calculations) -> frontier    |
    +---------------+---------+-------------------------------------------+

    Q2 -- Comparative cost (relative units per call : nano=1, mini=4, std=15, frontier=80) :

    Volume distribution :
      classify : 50% * 2M = 1M
      extract  : 25% * 2M = 500K
      summarize: 15% * 2M = 300K
      draft    :  8% * 2M = 160K
      answer   :  2% * 2M =  40K

    Proposed policy (with ~15% escalation) :
      classify  : 1M * (0.85*1 + 0.15*4)  = 1M * 1.45 = 1.45M units
      extract   : 500K * (0.9*1 + 0.1*4)  = 500K * 1.3 = 0.65M
      summarize : 300K * (0.8*4 + 0.2*15) = 300K * 6.2 = 1.86M
      draft     : 160K * (0.8*4 + 0.18*15 + 0.02*80)
                = 160K * (3.2+2.7+1.6) = 160K * 7.5 = 1.20M
      answer    : 40K * (0.7*15 + 0.3*80)
                = 40K * 34.5 = 1.38M
      TOTAL     ~ 6.54M units

    Naive "frontier for everything" policy :
      2M * 80 = 160M units.

    => Savings factor ~= **24x**. Well above the target.

    Q3 -- When to accept frontier :
    - Draft reply for detected sensitive emails (negotiation, HR,
      client conflict). A worthless mistake = business risk.
    - Answer question when the query requires chaining (multiple
      sources, calculation, reasoning over a contract).
    - Never for classify or extract : a classifier must be fast
      and cheap.
    """


def solution_exercice_2() -> None:
    """
    Exercise 2 -- Guardrails per product.

    1) MEDICAL CHATBOT
       Input :
         - PII detection (name, SSN) + masking before sending to the LLM
         - Prompt injection detection
         - Rate limit per user
         - Obvious suicide / emergency content -> immediate escalation
       Output :
         - Automatic disclaimer "I am not a substitute for medical advice"
         - Validation that the output contains no definitive diagnosis
         - Toxicity check (Llama Guard)
         - Human escalation if the question contains critical keywords
       Failure : HUMAN ESCALATION. Never a "degraded answer".

    2) RESUME GENERATOR
       Input :
         - PII collected voluntarily (authorized) but encrypted
         - Prompt injection detection
         - No defamatory content
       Output :
         - Verification that every cited experience is in the user
           input (no hallucinated experience)
         - JSON validation against the schema (sections, dates, etc.)
         - Retry with constraints if the format is invalid
       Failure : retry + default template fallback.

    3) E-COMMERCE SUPPORT WITH ORDER ACCESS
       Input :
         - Strong authentication (session, non-LLM)
         - Prompt enrichment with user data ON THE BACKEND SIDE,
           never "trusted" from the user prompt
         - Prompt injection detection
       Output :
         - Validation that the order numbers cited in the response
           really belong to the logged-in user (post-generation
           authz check)
         - Masking of other customers' data
         - JSON validation if the API returns an action (e.g. "refund")
       Failure : block + error message without a leak. Log the incident.

    4) LEGAL ASSISTANT (clause extraction)
       Input :
         - PDF doc parsing + controlled chunking
         - Prompt injection detection within the document itself
           (adversarial documents)
       Output :
         - Groundedness check : every cited clause must exist in
           the source doc (string match or LLM-as-a-judge)
         - JSON validation (clause_id, text, commentary)
         - Disclaimer "this is not legal advice"
       Failure : retry with a stricter prompt, then bypass with the output
       "I did not find reliable information in the document".

    Cross-cutting note : all these systems must have a trace ID
    visible in the output so that human support can find
    the conversation in case of a complaint.
    """


def solution_exercice_3() -> None:
    """
    Exercise 3 -- Semantic cache tuning.

    State : threshold=0.90, hit rate=45%, 3 false positives on
    questions that are close but semantically different.

    Q1 -- Root cause :
      Cosine on BoW / generic embeddings does not capture well the
      small words that change the whole meaning ("deploy" vs "update
      a deploy"). The model sees a massive overlap (k8s, deploy, service)
      and scores high. DILEMMA : raising the threshold loses hit
      rate. Lowering it increases the false positives.

    Q2 -- Techniques to reduce the false positives without sacrificing the
    hit rate :

      1. **Adaptive threshold per query type** : "how to" questions
         tolerate less approximation than "what is" questions.
         We learn the right threshold per category on a
         validation dataset.

      2. **Boosting verbs and action keywords** : give a higher
         weight to the main verbs (deploy, update, delete,
         rollback). An imperfect match on the verb = mismatch.

      3. **LLM-as-a-judge pre-serve** : before returning a hit, a
         small nano model verifies that the cached question and the
         current question really ask the same thing. Cost : one
         extra nano call, but a cache hit is still cheaper than a
         std call.

      4. **Cache segmentation by topic** : a nano classifier assigns
         a topic ("k8s", "python-debug", "security"), and the cache
         only searches within the same topic. Reduces cross-topic
         collisions.

      5. **Lightweight cross-encoder** as the final verifier (more precise
         than BoW cosine).

      6. **Shortened TTL** for fast-evolving topics (security,
         deprecations, library versions).

    Q3 -- Measurement plan :

      a) Validation dataset : 500 pairs (query_a, query_b) of which
         half are "equivalent" and the other half "different". Human labels.
      b) Metrics before / after the change :
         - False positive rate (incorrect cache hit) on the dataset
         - Hit rate in production
         - User feedback (thumbs down on cached vs fresh answers)
         - Average latency
      c) Deploy the changes in shadow mode : the cache hit is
         computed but the real answer is still generated and
         compared. If the gap > threshold, log it.

    Q4 -- When to disable the cache :
      If the false positive rate > 1% AND the false positives are
      visible to the user (not just different phrasings).
      For this use case (devs, critical technical trust), it is better
      to pay the LLM cost than to risk losing trust. Disable
      the cache entirely OR restrict it to the simplest queries
      (definitions, glossary).
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)
