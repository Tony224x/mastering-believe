"""
Solutions -- Day 13 HARD Exercises: Observabilite & MLOps

Worked solutions with the reasoning step by step. Assertions lock the
key structural checks so the file is self-checking.

Usage:
    python3 13-observabilite-mlops-hard.py
"""

SEPARATOR = "=" * 70


# =============================================================================
# HARD -- Exercise 1 : Observability + MLOps for an ML/LLM platform
# =============================================================================

def hard_1_platform():
    """Design observability + MLOps for a mixed ML/LLM platform."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Observability + MLOps for an ML/LLM platform")
    print(SEPARATOR)

    ml_preds = 100_000_000
    llm_reqs = 20_000_000

    print(f"\n  Scope : {ml_preds:,} ML preds/day + {llm_reqs:,} LLM reqs/day.")

    print(f"\n  1. Observability architecture (3 pillars) :")
    print(f"     TRACING : ML -> prediction id, model version, feature snapshot;")
    print(f"               LLM -> spans (llm.call, tool.call), tokens, cost, latency.")
    print(f"     METRICS : ML -> PSI, prediction distribution, perf-on-labels;")
    print(f"               LLM -> tokens/min, cost/1K, cache hit, fallback rate,")
    print(f"               faithfulness score.")
    print(f"     LOGS    : raw events; hash+sample prompts; scrub PII; aggressive TTL.")
    print(f"     UNIFICATION : one stack via OpenTelemetry (gen_ai.* semantics for LLM)")
    print(f"     so ML and LLM share traces/metrics/logs -- no two parallel infras.")

    print(f"\n  2. Drift & data quality :")
    print(f"     ML (5 models) : PSI / KL on numeric features; chi-square for categoricals.")
    print(f"     LLM (3 products) : drift on prompts/responses (length, topic, sentiment).")
    print(f"     < 1h detection : frequent job (sample) + alert on PSI > 0.25 or a drop")
    print(f"     in faithfulness. Data QUALITY (separate from drift) : nulls, types,")
    print(f"     ranges, freshness via Great Expectations / Soda.")

    print(f"\n  3. Silent failures :")
    print(f"     Fraud concept drift : monitor perf on delayed labels + drift on the")
    print(f"     input->output relation, not just inputs; alert on recall drop.")
    print(f"     LLM hallucination on a new user segment : LLM-as-a-judge faithfulness")
    print(f"     sampling + drift on incoming topics (PSI on intents) -- no labels needed.")

    print(f"\n  4. A/B testing & deployment (CI/CD gates) :")
    gates = [
        "CI tests + data version pinned",
        "offline eval >= baseline (else reject)",
        "register model -> staging",
        "shadow deploy (no user impact)",
        "canary 1% -> 10% -> 50% -> 100% with auto-rollback",
        "manual approval gate + audit trail for regulated models (fraud/pricing)",
    ]
    for g in gates:
        print(f"     - {g}")
    print(f"     Auto-rollback triggers : guardrail metric degraded (latency/error) OR")
    print(f"     business metric drop OR drift/perf alert during canary.")

    print(f"\n  5. Cost & governance :")
    print(f"     Keep observability cheap : SAMPLE traces (e.g. 1-10%), aggressive TTL")
    print(f"     (30-90 days), hash+sample prompts instead of storing all of them.")
    print(f"     Regulated PII : scrub before logging; access restricted by team.")

    print(f"\n  6. Failure modes :")
    print(f"     Alert fatigue (50 false alerts/day) : recalibrate thresholds per")
    print(f"     feature, dedup, add severities, alert on SUSTAINED drift not single spikes.")
    print(f"     Model passes offline but breaks in prod : the missing gate was SHADOW")
    print(f"     + CANARY (offline eval alone is insufficient).")

    # ---- assertions ----
    assert len(gates) == 6, len(gates)
    assert any("audit trail" in g for g in gates), "regulated models need audit trail"
    assert any("canary" in g for g in gates), "need canary gate"
    print("\n  [assertions OK]")


# =============================================================================
# HARD -- Exercise 2 : Post-mortem -- silent model drift for 3 weeks
# =============================================================================

def hard_2_postmortem():
    """Post-mortem of an undetected silent drift incident."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Post-mortem -- the model that drifted silently for 3 weeks")
    print(SEPARATOR)

    print(f"\n  1. Full causal chain :")
    chain = [
        ("DETECTION", "New fraud pattern emerges (concept drift: relation changes)",
         "Missing : concept-drift monitoring (perf on labels + relation drift)"),
        ("DETECTION", "Model misses the new pattern -> false negatives rise",
         "Missing : output-score distribution monitoring (blocked-rate drop)"),
        ("MONITORING", "Technical metrics stay green (latency/errors/throughput OK)",
         "Missing : awareness that a wrong model returns 200, not 500"),
        ("MONITORING", "Transaction amount distribution shifts (data drift) -- not monitored",
         "Missing : PSI on input features (daily)"),
        ("PROCESS", "No continuous eval (model evaluated once at deploy)",
         "Missing : continuous eval as labels arrive"),
        ("PROCESS", "Labels delayed 3 weeks -> detection only via chargebacks",
         "Missing : label-free proxy signals for early alerting"),
    ]
    for cat, cause, guard in chain:
        print(f"     [{cat}] {cause}")
        print(f"        -> {guard}")

    print(f"\n  2. The green-metrics trap :")
    print(f"     Latency/error/throughput describe the SERVICE, not the model's")
    print(f"     correctness. A classic API failure throws a 500; a model failure")
    print(f"     returns a confident 200 with a WRONG answer -- invisible to infra")
    print(f"     metrics. That's the core J13 lesson : ML failures are silent.")

    print(f"\n  3. Delayed labels -> detect WITHOUT labels :")
    print(f"     Recall needs labels (3 weeks late). Proxy signals that fire in week 0-1 :")
    proxies = [
        "PSI / KL drift on input features (amounts, geo)",
        "Drift on the OUTPUT score distribution (scores shift lower)",
        "Drop in the blocked-transaction rate (model flags far fewer)",
    ]
    for p in proxies:
        print(f"     - {p}")

    print(f"\n  4. Concept drift vs data drift here :")
    print(f"     CONCEPT drift (new fraud pattern) is the PRIMARY cause of the losses")
    print(f"     -> requires RE-TRAINING. DATA drift (amount distribution) is SECONDARY")
    print(f"     -> could be handled by recalibration, but doesn't fix the missed pattern.")

    print(f"\n  5. Corrected system :")
    print(f"     Monitoring that catches it in < 1 day : daily PSI on inputs + daily")
    print(f"     monitoring of the output-score distribution and blocked-rate, with")
    print(f"     alerts. Continuous eval that computes recall/precision as labels land")
    print(f"     (rolling), not just at deploy. Runbook (7 steps) :")
    steps = [
        "Contain risk NOW : tighten the score threshold / add manual review for high-value tx",
        "Quantify the blast radius (false-negative rate, losses) from available labels",
        "Confirm drift type : input PSI (data) + relation/perf (concept)",
        "Stand up daily drift + output-distribution monitoring immediately",
        "Re-train on recent labelled fraud (concept drift) and validate offline",
        "Canary the retrained model with auto-rollback, then promote",
        "Post-mortem within 24h : timeline, root causes, guardrails, owners",
    ]
    for i, s in enumerate(steps, 1):
        print(f"       {i}. {s}")

    # ---- assertions ----
    categories = {c[0] for c in chain}
    assert {"DETECTION", "MONITORING", "PROCESS"} <= categories, categories
    assert len(proxies) == 3, "need 3 label-free proxies"
    assert len(steps) == 7, len(steps)
    assert "contain risk" in steps[0].lower(), "runbook must start by containing risk"
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 13 HARD : OBSERVABILITE & MLOPS")
    print(SEPARATOR)
    hard_1_platform()
    hard_2_postmortem()
    print(f"\n{SEPARATOR}")
    print("  END OF HARD SOLUTIONS")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
