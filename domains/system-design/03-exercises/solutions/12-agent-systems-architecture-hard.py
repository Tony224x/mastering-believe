"""
Solutions -- Day 12 HARD Exercises: Agent Systems Architecture

Worked solutions with the reasoning step by step. Assertions lock the
key structural checks so the file is self-checking.

Usage:
    python3 12-agent-systems-architecture-hard.py
"""

import json

SEPARATOR = "=" * 70


# =============================================================================
# HARD -- Exercise 1 : Autonomous incident-resolution agent (SRE copilot)
# =============================================================================

def hard_1_sre_copilot():
    """Design an SRE copilot agent with safe actions and human-in-the-loop."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : SRE copilot agent")
    print(SEPARATOR)

    n_tools = 25

    print(f"\n  1. Orchestration pattern :")
    print(f"     SINGLE-AGENT with tool routing is defensible : the SRE domain is")
    print(f"     coherent and latency matters during an incident (no inter-agent")
    print(f"     round-trips). A LIGHT supervisor (diagnose vs remediate) is also fine.")
    print(f"     Avoid deep hierarchy : it adds latency exactly when MTTR matters.")

    print(f"\n  2. Tool routing ({n_tools} tools) :")
    print(f"     Namespacing into modules : metrics.*, logs.*, traces.*, deploy.*, k8s.*.")
    print(f"     The agent picks a MODULE first, then a tool inside it. Per step, filter")
    print(f"     to the 5-8 most relevant tools (embedding of the task vs tool descs).")

    print(f"\n  3. Action safety (human-in-the-loop) :")
    actions = {
        "read-only": ("read dashboards/logs/traces", "auto-allowed"),
        "reversible": ("scale up, clear cache", "auto with audit log"),
        "destructive": ("restart pod, rollback deploy", "REQUIRES human approval"),
    }
    for cls, (ex, policy) in actions.items():
        print(f"     {cls:<12} e.g. {ex:<28} -> {policy}")
    print(f"     Enforcement : destructive tools are gated behind an approval step in")
    print(f"     the runtime (the tool refuses to execute without an approval token),")
    print(f"     NOT merely 'the prompt says ask first'. Defense at the tool layer.")

    print(f"\n  4. Loop, stop & budget :")
    stops = [
        "hypothesis validated by evidence",
        "remediation applied AND verified (metric recovered)",
        "budget exceeded (steps/tokens/time)",
        "human takeover",
    ]
    for s in stops:
        print(f"     - stop: {s}")
    print(f"     No-progress detector breaks re-diagnosis loops (same evidence, no new")
    print(f"     hypothesis -> escalate). On UNCERTAINTY : never take a destructive")
    print(f"     action at random -> present options to the on-call and wait.")

    print(f"\n  5. Memory & context :")
    print(f"     Inject : current incident (alert, recent metrics/logs window),")
    print(f"     relevant runbooks, and similar PAST incidents (long-term store).")
    print(f"     Reuse past post-mortems via retrieval (RAG over incident history).")

    print(f"\n  6. Observability & failure modes :")
    print(f"     Log every step : decision + reasoning + tool called + result + who")
    print(f"     approved what. Fully auditable timeline.")
    failure_modes = [
        ("Remediation makes it worse",
         "prefer reversible actions; verify metric after each action; auto-revert if worse"),
        ("Agent stuck / looping",
         "no-progress detector + budget + escalate to human"),
        ("A tool lies (stale data)",
         "cross-check across sources (metrics vs logs vs traces); flag staleness; don't act on a single signal"),
    ]
    for fm, mit in failure_modes:
        print(f"     - {fm}")
        print(f"         -> {mit}")

    # ---- assertions ----
    assert "destructive" in actions and "human approval" in actions["destructive"][1]
    assert len(stops) == 4, "must define 4 stop conditions"
    assert len(failure_modes) == 3, "must cover 3 failure modes"
    print("\n  [assertions OK]")


# =============================================================================
# HARD -- Exercise 2 : Post-mortem -- multi-agent budget burn + loop
# =============================================================================

def hard_2_postmortem():
    """Post-mortem of a multi-agent research run gone wrong."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Post-mortem -- multi-agent loop + budget burn")
    print(SEPARATOR)

    print(f"\n  1. Full causal chain :")
    chain = [
        ("HANDOFF", "Niche topic with few reliable sources",
         "Missing : a 'low-evidence -> stop / ask user' branch"),
        ("HANDOFF", "Handoff is just 'continue with the research' (no context)",
         "Missing : structured handoff listing what was already searched"),
        ("HANDOFF", "Searcher re-runs the SAME searches (doesn't know what's done)",
         "Missing : shared search history in state"),
        ("STOP", "No no-progress detector -> searcher->reader->supervisor loops",
         "Missing : no-progress detector + max round-trips"),
        ("STOP", "No global budget cap (~2000 web searches)",
         "Missing : budget on searches/steps/tokens/time"),
        ("CONTEXT", "Aggregated context piles up all sources -> overflow",
         "Missing : summarization / retrieve-on-demand instead of stuffing"),
        ("CONTEXT", "Supervisor can't plan with an overflowing context -> incoherent plans",
         "Missing : bounded working memory for the supervisor"),
        ("QUALITY", "Writer fabricates sources for a low-evidence topic",
         "Missing : groundedness check + completeness gate"),
    ]
    for cat, cause, guard in chain:
        print(f"     [{cat}] {cause}")
        print(f"        -> {guard}")

    print(f"\n  2. The faulty handoff :")
    print(f"     'continue with the research' carries no state, so the searcher cannot")
    print(f"     know which queries already ran -> it duplicates them. Correct handoff :")
    handoff = {
        "from": "supervisor",
        "to": "searcher",
        "context": "Research niche topic X; reliable sources are scarce.",
        "done": ["12 sub-questions planned", "8 queries already run (see queries_done)"],
        "remaining": ["find sources for sub-questions 9-12 ONLY"],
        "success_criteria": ">=2 independent reliable sources per sub-question, or mark as 'insufficient'",
        "budget": {"max_searches": 20, "max_steps": 5},
        "queries_done": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"],
    }
    print(json.dumps(handoff, indent=6))

    print(f"\n  3. Loop & stop :")
    print(f"     'writer produced a report' is insufficient : it can fire on a garbage")
    print(f"     report after an infinite loop. Add :")
    extra_stops = [
        "no-progress detector : no new reliable source for K turns -> stop",
        "budget exceeded : searches / steps / tokens / wall-clock",
        "completeness/coverage threshold reached (enough evidence) ",
    ]
    for s in extra_stops:
        print(f"     - {s}")
    print(f"     Budget is bounded AND propagated : each handoff carries the remaining")
    print(f"     search/step/token budget, decremented as work proceeds.")

    print(f"\n  4. Context overflow :")
    print(f"     Piling every source into the aggregated context blew the supervisor's")
    print(f"     window -> incoherent plans. Fix : summarize findings into compact notes")
    print(f"     and retrieve sources on demand (RAG over the run's own findings),")
    print(f"     never stuff the full corpus into the planning prompt.")

    print(f"\n  5. Quality & resilience :")
    print(f"     The writer hallucinated because no GROUNDEDNESS check (J10) forced every")
    print(f"     claim to map to a retrieved source. Completeness check : if evidence")
    print(f"     coverage is below a threshold, the agent must answer 'insufficient")
    print(f"     reliable information' rather than invent. Runbook (7 steps) :")
    steps = [
        "Kill the runaway run immediately",
        "Identify the loop (duplicate searches, no new state) from the trace",
        "Add a no-progress detector + max round-trips",
        "Cap and propagate budget (searches/steps/tokens/time) via handoffs",
        "Replace 'continue' handoffs with structured ones (incl. queries_done)",
        "Add groundedness + completeness gates before the writer emits a report",
        "Post-mortem within 24h : timeline, root causes, guardrails, owners",
    ]
    for i, s in enumerate(steps, 1):
        print(f"       {i}. {s}")

    # ---- assertions ----
    categories = {c[0] for c in chain}
    assert {"HANDOFF", "STOP", "CONTEXT", "QUALITY"} <= categories, categories
    assert "queries_done" in handoff, "corrected handoff must carry search history"
    assert len(steps) == 7, len(steps)
    assert steps[0].lower().startswith("kill"), "runbook must start by killing the run"
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 12 HARD : AGENT SYSTEMS ARCHITECTURE")
    print(SEPARATOR)
    hard_1_sre_copilot()
    hard_2_postmortem()
    print(f"\n{SEPARATOR}")
    print("  END OF HARD SOLUTIONS")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
