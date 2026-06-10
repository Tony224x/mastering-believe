"""
Solutions -- Day 12 : Agent Systems Architecture
"""


def solution_exercice_1() -> None:
    """
    Exercise 1 -- Single-agent or multi-agent ?

    +-----------------------------+---------------+------------------------------+
    | Product                     | Pattern       | Justification                |
    +-----------------------------+---------------+------------------------------+
    | Coding assistant (Cursor)   | Single-agent  | Homogeneous tasks (code), low|
    |                             |               | latency critical, < 20 tools.|
    |                             |               | A good ReAct loop suffices.  |
    +-----------------------------+---------------+------------------------------+
    | Financial research          | Supervisor    | Heterogeneous tasks (read,   |
    | + memo                      | multi-agent   | compare, write), long context|
    |                             |               | , specialists needed.        |
    +-----------------------------+---------------+------------------------------+
    | Internal docs Slack bot     | Single-agent  | It's a RAG + 2-3 tools.      |
    |                             |               | Multi-agent would be over-   |
    |                             |               | engineered.                  |
    +-----------------------------+---------------+------------------------------+
    | Marketing automation        | Supervisor    | Segment + write + send +     |
    |                             | multi-agent   | analyze. Specialists         |
    |                             |               | allow optimizing each        |
    |                             |               | step independently.          |
    +-----------------------------+---------------+------------------------------+
    | Deep research report        | Hierarchical  | Phases (planning / research /|
    |                             |               | drafting / review) with sub- |
    |                             |               | agents per phase. High       |
    |                             |               | budget, non-critical latency.|
    +-----------------------------+---------------+------------------------------+
    | IDE grammar checker         | Single-agent  | Latency < 100 ms mandatory.  |
    |                             | (local !)     | No tools, just the model.    |
    |                             |               | No room for a multi setup.   |
    +-----------------------------+---------------+------------------------------+

    Underlying principle : **start with the simplest thing that works**.
    Start single-agent. Move to multi when you have proof that it is
    necessary (context too large, distinct specializations, or a
    clear bottleneck).

    Latency mentioned for :
    - Coding assistant : must respond < 2 s -> single
    - Grammar checker : < 100 ms -> local single
    - Slack bot : < 5 s acceptable -> single
    - Deep research : several minutes OK -> hierarchical multi-agent
    """


def solution_exercice_2() -> None:
    """
    Exercise 2 -- State and stop conditions of a travel assistant.

    1) Minimum state (pseudo-dataclass) :

        @dataclass
        class TravelState:
            # user input
            user_id: str
            session_id: str
            user_request: str
            destination: Optional[str]
            date_start: Optional[str]
            date_end: Optional[str]
            budget_eur: Optional[float]
            preferences: dict        # {airline, hotel_class, vegetarian...}

            # agent working memory
            candidate_flights: list[dict]
            candidate_hotels: list[dict]
            itinerary: Optional[dict]
            validation_status: Literal["pending", "approved", "rejected"]

            # conversation
            messages: list[dict]      # chronological dialog

            # control flow
            steps_used: int
            max_steps: int
            last_agent: Optional[str]
            waiting_for_user: bool
            done: bool
            error: Optional[str]

    2) Stop conditions :
       a) `done == True AND validation_status == "approved"` : task complete
       b) `steps_used >= max_steps` : budget exhausted
       c) `error is not None` : unrecoverable error (API down, auth KO)
       d) `waiting_for_user == True` : control handed back to the user, we
          exit the loop until wake-up
       e) Loop detection (same action 3 times in a row) : defensive
          stop

    3) If the budget is exceeded before completion :
       - The agent must return a *checkpoint* to the user : "here is what
         I found so far, continue or stop ?"
       - Do NOT crash. Do NOT hallucinate to finish.
       - Record the partial results in the state.

    4) Human-in-the-loop without a loop :
       - The agent sets `waiting_for_user = True` and returns a
         structured message ("I propose itinerary X, do you approve ?").
       - The state is *persisted* (DB checkpoint).
       - The agent is no longer running. When the user answers, we reload the
         state and resume at the next point.
       - Frameworks : LangGraph `interrupt()` + `Command(resume=...)`.
         CrewAI also has this pattern.

    5) Persistence between sessions :
       - **Structured state** : relational DB (PostgreSQL) for the
         critical data (remaining budget, bookings, auth). Key =
         session_id.
       - **Agent checkpoint** : key-value store (Redis, DynamoDB)
         serialized as JSON, indexed by session_id.
       - **Long-term preferences** : separate store per user_id. Can
         be a vector store + a DB (e.g. "likes 4-star hotels
         with a pool, no flights after 10pm").
       - Always respect the GDPR : TTL on the PII, explicit consent.
    """


def solution_exercice_3() -> None:
    """
    Exercise 3 -- A hallucinating multi-agent system.

    Hypotheses, tests, mitigations :

    H1 -- **Handoff messages too poor**
      - Test : inspect the messages passed from one agent to another. If they
        just contain "continue" or "do the thing", we found it.
      - Mitigation : formalize a Handoff schema {context, done,
        remaining, success_criteria, budget}. Reject the handoffs that
        do not follow the schema.
      - Impact/effort : HIGH / LOW -> **PRIORITY 1**

    H2 -- **Context bleed / reuse without verification**
      - The writer cites the search results without verifying that they
        actually exist in memory.
      - Test : inspect the writer's citations and compare to the
        search_agent's log.
      - Mitigation : automatic groundedness check : every citation
        by the writer must point to a chunk_id that exists in
        state.results. If not found -> flag invalid.
      - Impact/effort : HIGH / MEDIUM -> **PRIORITY 2**

    H3 -- **Stale memory reused**
      - Data from a previous step is reused even though it is
        no longer valid (e.g. a price snapshot 30 min later).
      - Test : timestamp all the state.results entries and
        check the time difference at the moment of use.
      - Mitigation : TTL on the memory entries, and explicit
        invalidation per step.
      - Impact/effort : MEDIUM / MEDIUM -> PRIORITY 4

    H4 -- **Non-independent critic (same model as the agents)**
      - If the critic uses the same LLM as the executors, it validates
        the very hallucinations it is supposed to detect.
      - Test : run the critic over 50 examples with known errors
        and measure the recall.
      - Mitigation : use a different model for the critic
        (another provider, or a stronger one, or stricter via prompt).
        Ideally combined with deterministic checks
        (schema, groundedness) BEFORE invoking the LLM critic.
      - Impact/effort : HIGH / MEDIUM -> **PRIORITY 3**

    H5 -- **No explicit stopping criteria / overly permissive
          supervisor**
      - The supervisor declares "done" while part of the plan is
        not done because it has no checklist.
      - Test : compare the initial plan with the steps actually
        executed.
      - Mitigation : the supervisor maintains a checklist, and may
        only declare "done" after verifying that every item
        is checked (with proof -> each item is linked to a result in
        the state).
      - Impact/effort : HIGH / LOW -> **PRIORITY 1bis**

    Bonus H6 -- No observability : without LangSmith / Langfuse
    traces, debugging is impossible. Install as priority zero.

    Prioritization plan (impact/effort) :
      1. Structured handoff messages (H1)
      2. Mandatory supervisor checklist (H5)
      3. Automatic groundedness check (H2)
      4. Independent critic model (H4)
      5. Timestamped and invalidatable memory (H3)

    Cross-cutting : global metric = hallucination rate measured on
    a gold set of 50 tasks. Track after each mitigation.
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)
