"""
Solutions -- Day 12 MEDIUM Exercises: Agent Systems Architecture

Worked solutions with the reasoning step by step. Assertions lock the
key calculations so the file is self-checking.

Usage:
    python3 12-agent-systems-architecture-medium.py
"""

import json

SEPARATOR = "=" * 70


# =============================================================================
# MEDIUM -- Exercise 1 : Choose the orchestration pattern
# =============================================================================

def medium_1_pattern_choice():
    """Map systems to single / supervisor / hierarchical and resist over-arch."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Choose the orchestration pattern")
    print(SEPARATOR)

    print(f"\n  1. Recommendations :")
    recs = [
        ("S1 Code assistant", "single ReAct",
         "Homogeneous domain (code), one model has the context. Simple, debuggable."),
        ("S2 Ops assistant", "supervisor",
         "Heterogeneous (web + code + email + analysis). Each specialist gets a focused prompt."),
        ("S3 Deep research", "hierarchical",
         "Multi-phase (plan -> research -> verify -> write). Phases map to sub-supervisors."),
        ("S4 FAQ bot", "single (or a simple chain)",
         "2 tools, single-hop. Anything more is over-engineering."),
    ]
    for sys_, pat, why in recs:
        print(f"     {sys_:<22} -> {pat}")
        print(f"        {why}")

    print(f"\n  2. S2 : why supervisor beats a single agent with 12 tools ?")
    print(f"     With 12 heterogeneous tools, a single agent's context explodes and the")
    print(f"     tool-choice quality degrades. A supervisor dispatches to specialists,")
    print(f"     each with a SHORT, focused context and only its own tools.")

    # 3. Over-engineering cost for S4
    single_latency = (1, 3)        # seconds
    supervisor_latency = (5, 15)
    cost_mult = (2, 5)
    print(f"\n  3. S4 : why multi-agent is a mistake :")
    print(f"     Latency : single {single_latency[0]}-{single_latency[1]}s vs supervisor")
    print(f"     {supervisor_latency[0]}-{supervisor_latency[1]}s. Cost : {cost_mult[0]}-{cost_mult[1]}x.")
    print(f"     For a single-hop FAQ, that's {cost_mult[0]}-{cost_mult[1]}x cost and ~5x latency")
    print(f"     for ZERO quality gain. Unjustified.")

    print(f"\n  4. Orders of magnitude (from the course tradeoff table) :")
    table = [
        ("single", "1x", "1-3s", "easy"),
        ("supervisor", "2-5x", "5-15s", "medium"),
        ("hierarchical", "10x+", "30s+", "hard"),
    ]
    print(f"     {'pattern':<14}{'cost':<8}{'latency':<10}{'debug'}")
    for pat, cost, lat, dbg in table:
        print(f"     {pat:<14}{cost:<8}{lat:<10}{dbg}")

    print(f"\n  5. The 6-agent proposal for S1 :")
    print(f"     Alarm signal : 6 agents for what is really ~3 tools on ONE agent.")
    print(f"     Answer : start single-agent; go multi-agent only with PROOF it's")
    print(f"     needed. The simplest system that works beats the sophisticated one.")

    print(f"\n  6. S3 hierarchy depth :")
    max_levels = 3
    print(f"     At most {max_levels} levels. Each level adds latency and cost; beyond 2-3")
    print(f"     the round-trips dominate and debugging becomes intractable.")

    # ---- assertions ----
    assert supervisor_latency[0] > single_latency[1], "supervisor must be slower than single"
    assert cost_mult[1] == 5
    assert max_levels <= 3, "hierarchy should stay shallow"
    assert recs[0][1] == "single ReAct" and recs[3][1].startswith("single")
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 2 : Conversational agent memory architecture
# =============================================================================

def medium_2_memory():
    """Design short + long term memory and size the context budget."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Conversational agent memory architecture")
    print(SEPARATOR)

    ctx_window = 128_000
    sys_tools = 4_000
    safety = 0.60
    tok_per_msg = 80
    msgs_per_session = 60

    # 1. Real available budget
    usable = ctx_window * safety - sys_tools
    print(f"\n  1. Real available tokens :")
    print(f"     {ctx_window:,} * {safety:.0%} - {sys_tools:,} = {usable:,.0f} tokens for history + retrieved memory")

    # 2. Session size
    session_tokens = msgs_per_session * tok_per_msg
    print(f"\n  2. A {msgs_per_session}-message session :")
    print(f"     {msgs_per_session} * {tok_per_msg} = {session_tokens:,} tokens -> fits easily in {usable:,.0f}")

    # 3. Overflow point
    overflow_msgs = int(usable / tok_per_msg)
    print(f"\n  3. Overflow point :")
    print(f"     {usable:,.0f} / {tok_per_msg} = {overflow_msgs:,} messages before the budget is full.")
    print(f"     Mechanism to trigger : intermediate SUMMARIZATION of older messages.")

    print(f"\n  4. 3-type memory architecture :")
    mem = [
        ("Episodic", "vector store", "'the user said they are vegetarian on 2026-03-01'"),
        ("Semantic", "KV / relational store", "'user name = Alex, plan = pro'"),
        ("Procedural", "rules / templates", "'JSON format expected by the booking tool'"),
    ]
    for kind, store, ex in mem:
        print(f"     {kind:<12} -> {store:<22} e.g. {ex}")

    print(f"\n  5. Why retrieve-on-demand (not all-in-prompt) :")
    print(f"     Stuffing all long-term memory into the prompt causes context overflow")
    print(f"     and cost blow-up. Instead, RETRIEVE only the relevant memories per turn")
    print(f"     -- this is RAG over your own memory (the J10 pattern).")

    print(f"\n  6. Short-term summarization strategy :")
    keep_verbatim = 10
    print(f"     Keep the last ~{keep_verbatim} messages verbatim; when exceeded, summarize")
    print(f"     the oldest into a compact note prepended as a system message. Trigger")
    print(f"     well before the {overflow_msgs:,}-message hard limit (e.g. every 20 messages).")

    # ---- assertions ----
    assert abs(usable - 72_800) < 1, usable          # 0.6*128k - 4k = 72800
    assert session_tokens < usable, "a normal session must fit"
    assert overflow_msgs == 910, overflow_msgs       # 72800/80
    assert len(mem) == 3, "must define 3 memory types"
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 3 : Robust handoff + stop conditions
# =============================================================================

def medium_3_handoff():
    """Design a complete handoff protocol and stop conditions."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Robust handoff + stop conditions")
    print(SEPARATOR)

    print(f"\n  1. The 5 mandatory handoff fields + example :")
    handoff = {
        "from": "supervisor",
        "to": "code_agent",
        "context": "Refactor the OrderService class to async/await.",
        "done": ["file read", "current sync version analyzed"],
        "remaining": ["generate async version", "update call sites"],
        "success_criteria": "class uses async/await and existing tests still pass",
        "budget": {"max_steps": 8, "max_tokens": 30000},
    }
    print(json.dumps(handoff, indent=6))
    fields = {"context", "done", "remaining", "success_criteria", "budget"}

    print(f"\n  2. Why 'continue' is a code smell :")
    print(f"     The specialist receives no context, no done/remaining, no success")
    print(f"     criterion -> it re-does work, ignores prior results, or guesses the")
    print(f"     goal. Concrete failure : duplicated work and divergence from the task.")

    print(f"\n  3. Four independent stop conditions :")
    stops = [
        ("task done", "success_criteria met (tests pass + async)"),
        ("step budget", "steps_used >= max_steps"),
        ("token budget", "tokens_used >= max_tokens"),
        ("no-progress / human needed", "N turns with no new state -> escalate"),
    ]
    for name, trig in stops:
        print(f"     - {name:<28} trigger: {trig}")

    print(f"\n  4. Breaking the code<->test loop :")
    print(f"     Maintain a no-progress counter : if the same actions repeat or the")
    print(f"     state hash doesn't change for K turns, stop and escalate. Cap the")
    print(f"     number of code<->test round-trips explicitly.")

    print(f"\n  5. Budget propagation :")
    print(f"     The handoff carries the REMAINING budget (budget_steps = max - used).")
    print(f"     Each specialist decrements it per step; a child can never exceed the")
    print(f"     budget granted by its parent.")

    print(f"\n  6. Budget exhausted, task unfinished :")
    print(f"     Graceful : return the PARTIAL result + a clear 'budget exhausted,")
    print(f"     escalate to human' status. Never raise an unhandled exception or loop")
    print(f"     forever -- a stop condition must always produce a clean output.")

    # ---- assertions ----
    assert fields <= set(handoff.keys()), "handoff must contain the 5 fields"
    assert len(stops) == 4, "must define 4 independent stop conditions"
    assert any("no-progress" in s[0] for s in stops), "need a no-progress condition"
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 12 MEDIUM : AGENT SYSTEMS ARCHITECTURE")
    print(SEPARATOR)
    medium_1_pattern_choice()
    medium_2_memory()
    medium_3_handoff()
    print(f"\n{SEPARATOR}")
    print("  END OF MEDIUM SOLUTIONS")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
