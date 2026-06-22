"""
Solutions -- Day 11 HARD Exercises: LLM Infrastructure

Worked solutions with the reasoning step by step. Assertions lock the
key calculations so the file is self-checking.

Usage:
    python3 11-llm-infrastructure-hard.py
"""

import math

SEPARATOR = "=" * 70


# =============================================================================
# HARD -- Exercise 1 : LLM Gateway for 50M requests/day
# =============================================================================

def hard_1_gateway():
    """Design a multi-provider, multi-tenant LLM Gateway at scale."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : LLM Gateway for 50M requests/day")
    print(SEPARATOR)

    daily_req = 50_000_000
    peak_factor = 5
    instance_capacity = 2000   # req/s per gateway instance

    print(f"\n  1. Architecture (request flow) :")
    print(f"     client -> [ Gateway ]")
    print(f"       CRITICAL PATH (sync, low latency) :")
    print(f"         auth/tenant -> quota check -> input guardrails -> cache lookup")
    print(f"         -> router -> provider call (with fallback) -> output guardrails")
    print(f"       ASYNC (off critical path) :")
    print(f"         tracing/spans, cost accounting, scoring, billing, log scrubbing")
    print(f"     The Gateway is STATELESS : quotas, cache and circuit-breaker state")
    print(f"     live in a shared store (Redis). Run N instances behind a LB -> no SPOF.")

    # 2. Routing & cost + QPS sizing
    avg_qps = daily_req / 86_400
    peak_qps = avg_qps * peak_factor
    instances = math.ceil(peak_qps / instance_capacity)
    instances_ha = instances + math.ceil(instances * 0.3)  # +30% headroom for HA
    print(f"\n  2. Routing & cost + sizing :")
    print(f"     Avg QPS  : {avg_qps:,.0f} req/s")
    print(f"     Peak QPS : {peak_qps:,.0f} req/s (x{peak_factor})")
    print(f"     Instances : ceil({peak_qps:,.0f} / {instance_capacity}) = {instances}")
    print(f"                 + 30% HA headroom -> ~{instances_ha} instances")
    print(f"     Per-tenant budget : token bucket per team (QPS) + monthly $ quota.")
    print(f"     One team cannot burn another's budget : isolated buckets + hard caps;")
    print(f"     when a team hits its cap, its requests are throttled/queued, not others.")

    # 3. Cache
    print(f"\n  3. Cache :")
    print(f"     Semantic cache PER TENANT (never global -> PII). ROI = LLM $ saved by")
    print(f"     hits MINUS cache infra cost. Only enable per tenant where hit rate")
    print(f"     justifies it (FAQ-like teams 40%+, code teams ~5% -> not worth it).")

    # 4. Reliability
    print(f"\n  4. Reliability (SLA 99.95%) :")
    sla = 0.9995
    minutes_month = 30 * 24 * 60
    budget_min = (1 - sla) * minutes_month
    print(f"     Budget : {budget_min:.1f} min downtime/month.")
    print(f"     Chain : provider1 -> provider2 -> self-hosted, aggressive timeouts +")
    print(f"     circuit breaker. With 3 ~99% independent providers, composed")
    print(f"     availability >> 99.95%.")
    print(f"     All external providers degraded at once : serve from cache where")
    print(f"     possible, queue non-urgent (batch) requests, return a clear 'degraded'")
    print(f"     message for the rest. Provider rate limits : client-side token bucket")
    print(f"     per provider + spread load across providers.")

    # 5. Security & governance
    print(f"\n  5. Security & governance :")
    print(f"     PII : scrub BEFORE any logging/tracing (regex + NER); store a hash +")
    print(f"     sample, never the raw prompt. Chargeback : every call emits a span with")
    print(f"     gen_ai.usage.* + cost, aggregated by tenant -> monthly showback.")
    print(f"     Prompts : versioned in a registry, deployed via the same CI gate as code.")

    # 6. Observability & failure modes
    print(f"\n  6. Observability (8 metrics) :")
    metrics = [
        ("Requests/s per tenant", "spike > 3x baseline"),
        ("Cost/hour per tenant", "> daily budget / 12"),
        ("Fallback rate", "> 5%"),
        ("Cache hit rate", "< target per tenant"),
        ("Latency p99", "> SLA"),
        ("Error rate (5xx)", "> 0.1%"),
        ("Guardrail block rate", "sudden change"),
        ("Provider availability", "any provider < 99%"),
    ]
    for name, alert in metrics:
        print(f"     - {name:<28} alert if {alert}")
    print(f"     Self-hosted GPU down at peak : circuit-break it, shift load to external")
    print(f"     providers (more expensive but available), alert on cost rise.")
    print(f"     Runaway cost runbook (5 steps) :")
    steps = [
        "Identify the offending tenant/agent from per-session cost spans",
        "Apply the hard cap / kill switch on THAT tenant (not the whole gateway)",
        "Verify cost/hour drops; keep other tenants serving",
        "Inspect the looping session (missing stop condition?) and patch",
        "Post-mortem + add a per-session hard cap if it was missing",
    ]
    for i, s in enumerate(steps, 1):
        print(f"       {i}. {s}")

    # ---- assertions ----
    assert abs(avg_qps - 578.7) < 2, avg_qps
    assert abs(peak_qps - 2893.5) < 10, peak_qps
    assert instances == 2, instances          # ceil(2893/2000) = 2
    assert budget_min < 25, budget_min        # ~21.9 min/month
    assert len(metrics) == 8, len(metrics)
    assert len(steps) == 5, len(steps)
    print("\n  [assertions OK]")


# =============================================================================
# HARD -- Exercise 2 : Post-mortem -- $80K LLM bill overnight
# =============================================================================

def hard_2_postmortem():
    """Post-mortem of a runaway LLM cost incident."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Post-mortem -- the $80K LLM bill overnight")
    print(SEPARATOR)

    print(f"\n  1. Full causal chain :")
    chain = [
        ("AGENT", "Ambiguous, contradictory ticket the agent cannot resolve",
         "Missing : a 'cannot resolve -> escalate to human' branch"),
        ("AGENT", "No stop condition on number of turns -> ~400 turns per ticket",
         "Missing : max_steps / no-progress detector"),
        ("COST", "Each turn re-sends a 6000-token system prompt with NO prompt caching",
         "Missing : native prompt caching on the stable system prompt"),
        ("COST", "Context grows to 80K+ tokens per session (full history re-sent)",
         "Missing : context summarization / sliding window"),
        ("COST", "Frontier tier ($15/$75 per 1M) used for a ticket auto-resolver",
         "Missing : tier routing (a mini model handles most tickets)"),
        ("BUDGET", "No per-session budget cap",
         "Missing : soft cap + hard cap per session"),
        ("OBSERVABILITY", "Logs lack cost-per-session ; no cost alert",
         "Missing : per-session cost spans + cost/hour alert"),
        ("PROCESS", "Runs unsupervised overnight, only a global kill switch",
         "Missing : granular kill switch + on-call cost alerting"),
    ]
    for cat, cause, guard in chain:
        print(f"     [{cat}] {cause}")
        print(f"        -> {guard}")

    print(f"\n  2. Cost multipliers :")
    # frontier vs nano per-1M-input ratio
    frontier_in, nano_in = 15.0, 0.05
    tier_mult = frontier_in / nano_in
    caching_mult = 10      # paying 100% vs 10% on the repeated system prompt
    loop_turns = 400
    print(f"     a) Tier : frontier vs nano input = {frontier_in}/{nano_in} = {tier_mult:.0f}x")
    print(f"     b) No prompt caching : ~{caching_mult}x on the repeated 6000-token sys prompt")
    print(f"     c) Loop : {loop_turns} turns instead of ~5 -> ~{loop_turns//5*1}x more calls")
    print(f"     d) Growing context : input tokens grow each turn (quadratic-ish total)")
    print(f"     These stack MULTIPLICATIVELY -> a normal $300/day becomes $80K/8h.")

    print(f"\n  3. Budget guardrails :")
    print(f"     Per-session budget with two caps :")
    print(f"     - SOFT cap (e.g. $0.50/session) : warn, force context summarization")
    print(f"     - HARD cap (e.g. $2.00/session) : stop the loop, escalate to human")
    print(f"     Middleware tracks a running cost per session (each call adds its cost")
    print(f"     delta) and cuts when the hard cap is hit. A per-USER daily budget would")
    print(f"     NOT help : a single runaway session burns the budget in minutes, long")
    print(f"     before any daily quota resets.")

    print(f"\n  4. Agent architecture :")
    print(f"     Missing stop conditions : max_steps, max_tokens, wall-clock timeout,")
    print(f"     and a no-progress detector (same tool calls / no new info -> stop).")
    print(f"     Prompt caching : reading the 6000-token sys prompt at ~10% instead of")
    print(f"     100% across 400 turns saves ~90% of that prompt's cost -> ~10x on it.")
    print(f"     Frontier is overkill : a mini model resolves most tickets; reserve")
    print(f"     frontier for the few genuinely hard cases (and cap them).")

    print(f"\n  5. Observability & response :")
    print(f"     4 attributes that enable a <15-min alert :")
    print(f"     - cost/hour per tenant, - tokens/session (running), - turns/session,")
    print(f"     - fallback/loop indicator. Alert when any crosses a threshold.")
    print(f"     Granular kill switch : disable a single tenant/agent, not the platform.")
    print(f"     Runaway-cost runbook (7 steps) :")
    steps = [
        "Identify the offending tenant/agent/session from cost spans",
        "Apply the granular kill switch on that agent (keep others running)",
        "Confirm cost/hour returns to baseline",
        "Inspect the looping sessions -> find the missing stop condition",
        "Enable per-session hard cap + max_steps immediately",
        "Enable prompt caching + downgrade the tier for this workload",
        "Post-mortem within 24h : timeline, multipliers, guardrails, owners",
    ]
    for i, s in enumerate(steps, 1):
        print(f"       {i}. {s}")

    # ---- assertions ----
    categories = {c[0] for c in chain}
    assert {"AGENT", "COST", "BUDGET", "OBSERVABILITY", "PROCESS"} <= categories, categories
    assert tier_mult == 300, tier_mult
    assert len(steps) == 7, len(steps)
    assert "kill switch" in steps[1], "step 2 must apply the granular kill switch"
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 11 HARD : LLM INFRASTRUCTURE")
    print(SEPARATOR)
    hard_1_gateway()
    hard_2_postmortem()
    print(f"\n{SEPARATOR}")
    print("  END OF HARD SOLUTIONS")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
