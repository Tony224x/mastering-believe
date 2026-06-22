"""
Solutions -- Day 13 MEDIUM Exercises: Observabilite & MLOps

Worked solutions with the reasoning step by step. Assertions lock the
key calculations so the file is self-checking.

Usage:
    python3 13-observabilite-mlops-medium.py
"""

import math

SEPARATOR = "=" * 70


# =============================================================================
# MEDIUM -- Exercise 1 : Compute and interpret PSI
# =============================================================================

def medium_1_psi():
    """Compute PSI by hand and decide the action."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Compute and interpret PSI")
    print(SEPARATOR)

    # bins : (baseline %, current %)
    bins = [(0.40, 0.25), (0.30, 0.30), (0.20, 0.25), (0.10, 0.20)]

    print(f"\n  1. PSI bin by bin :")
    print(f"     {'bin':<5}{'base':>8}{'curr':>8}{'contribution':>16}")
    psi = 0.0
    for i, (pb, pc) in enumerate(bins, 1):
        contrib = (pb - pc) * math.log(pb / pc)
        psi += contrib
        print(f"     {i:<5}{pb:>8.2f}{pc:>8.2f}{contrib:>16.5f}")
    print(f"     PSI total = {psi:.4f}")

    print(f"\n  2. Interpretation (thresholds 0.1 / 0.25) :")
    if psi < 0.1:
        verdict = "no drift"
    elif psi < 0.25:
        verdict = "MODERATE drift (watch)"
    else:
        verdict = "significant drift (act)"
    print(f"     PSI = {psi:.3f} -> {verdict}")

    print(f"\n  3. Data drift or concept drift ?")
    print(f"     PSI measures DATA drift : it compares the distribution of the INPUTS.")
    print(f"     It cannot see concept drift (a change in the input->output relation),")
    print(f"     because it never looks at the labels/outcomes.")

    print(f"\n  4. Drift present but AUC stable :")
    print(f"     The inputs shifted but the model still performs. WATCH, don't retrain")
    print(f"     reflexively. Drift is a warning, not a verdict -- performance is.")

    print(f"\n  5. Perf drops WITHOUT detectable input drift :")
    print(f"     That's CONCEPT drift : the input->output relation changed (e.g. new")
    print(f"     fraud patterns). Action : RE-TRAIN, not just recalibrate.")

    print(f"\n  6. Frequency & sample :")
    print(f"     Run daily on a sample (~1000 prod events). Cheap, stable, catches")
    print(f"     drift early without scanning the whole stream.")

    # ---- assertions ----
    assert 0.10 <= psi <= 0.16, psi          # ~0.13
    assert 0.1 <= psi < 0.25, "must be moderate drift"
    assert verdict.startswith("MODERATE")
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 2 : Design an ML A/B test that yields a real decision
# =============================================================================

def medium_2_ab_test():
    """Design a correct ML A/B test."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Design an ML A/B test")
    print(SEPARATOR)

    base_ctr = 0.08
    rel_uplift = 0.02
    target_ctr = base_ctr * (1 + rel_uplift)

    print(f"\n  1. Primary metric :")
    print(f"     A BUSINESS metric : CTR (or downstream engagement), chosen BEFORE")
    print(f"     the test. Offline NDCG does not prove business impact -- a better")
    print(f"     ranking can still fail to move clicks. Guardrails : latency p99,")
    print(f"     error rate, cost (must not degrade).")

    print(f"\n  2. Split by user_id hash :")
    print(f"     Per-request random split would show the SAME user both V1 and V2,")
    print(f"     polluting the comparison (and confusing the user). Hashing user_id")
    print(f"     gives each user a stable, consistent bucket.")

    print(f"\n  3. Novelty effect :")
    print(f"     Users react to a model just because it's NEW, not better. This inflates")
    print(f"     early results, so you must run 2-4 weeks for the effect to wash out.")

    print(f"\n  4. 20 secondary metrics, 1 'significant' :")
    print(f"     Multiple testing : with 20 metrics, ~1 looks significant by chance.")
    print(f"     Mitigation : define the primary metric a priori, and apply a")
    print(f"     Bonferroni correction to secondaries (never decide on a secondary).")

    print(f"\n  5. Positive signal on day 2 :")
    print(f"     Don't conclude : novelty effect + the signal isn't stable/significant")
    print(f"     yet. Early wins routinely regress to the mean.")

    print(f"\n  6. +2% CTR but +40% p99 latency :")
    print(f"     Latency is a GUARDRAIL. A 40% p99 regression blocks the rollout despite")
    print(f"     the CTR gain (or you optimize latency first). Guardrails exist exactly")
    print(f"     to stop a 'win' that quietly degrades the product.")

    # ---- assertions ----
    assert abs(target_ctr - 0.0816) < 1e-6, target_ctr
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 3 : Agent tracing + per-session cost budget
# =============================================================================

def medium_3_tracing_cost():
    """Design span structure and a per-session cost middleware."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Agent tracing + per-session cost budget")
    print(SEPARATOR)

    # prices $/1M tokens
    mini = (0.15, 0.60)
    std = (2.50, 10.00)

    def cost(price, t_in, t_out):
        return t_in * price[0] / 1_000_000 + t_out * price[1] / 1_000_000

    print(f"\n  1. Span structure + tree :")
    print(f"     Span fields : name, start/end ts (latency), parent_id, attributes")
    print(f"     (model, tokens_in, tokens_out, cost_usd), status, events.")
    print(f"     Tree for this request :")
    print(f"       agent.run")
    print(f"        +- retrieval")
    print(f"        +- llm.call #1 (rewrite, mini)")
    print(f"        +- rerank")
    print(f"        +- tool.call")
    print(f"        +- llm.call #2 (final, std)")

    # 2. Total LLM cost
    c1 = cost(mini, 500, 120)
    c2 = cost(std, 1200, 350)
    total = c1 + c2
    print(f"\n  2. Total LLM cost for this request :")
    print(f"     llm #1 (mini) : 500 in + 120 out = ${c1:.6f}")
    print(f"     llm #2 (std)  : 1200 in + 350 out = ${c2:.6f}")
    print(f"     Total = ${total:.6f}")

    print(f"\n  3. Why tree tracing beats line-by-line logs :")
    print(f"     An agent makes a chained, non-deterministic set of decisions.")
    print(f"     Flat logs lose the parent-child structure, per-step cost attribution,")
    print(f"     and the ability to see WHICH branch was slow/expensive.")

    print(f"\n  4. Per-session cost middleware :")
    print(f"     On EACH LLM call : compute its cost delta and ADD it to a running")
    print(f"     total keyed by session_id (each call emits a span with gen_ai.cost).")

    # 5. Requests before hard cap
    hard_cap = 0.50
    n_requests = int(hard_cap / total)
    print(f"\n  5. Hard cap ${hard_cap}/session :")
    print(f"     {hard_cap} / {total:.6f} = ~{n_requests} typical requests before the cap.")
    print(f"     At the cap : stop / force a new thread. Warn the user earlier at a")
    print(f"     soft cap (e.g. propose to summarize the conversation to shrink it).")

    print(f"\n  6. Minimal trace metadata :")
    print(f"     user_id, session_id, prompt_version / tags -- so you can filter and")
    print(f"     group in the observability UI (per user, per session, per prompt).")

    # ---- assertions ----
    assert abs(c1 - 0.000147) < 1e-6, c1
    assert abs(c2 - 0.0065) < 1e-6, c2
    assert abs(total - 0.006647) < 1e-5, total
    assert n_requests == 75, n_requests       # 0.50 / 0.006647 = ~75
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 13 MEDIUM : OBSERVABILITE & MLOPS")
    print(SEPARATOR)
    medium_1_psi()
    medium_2_ab_test()
    medium_3_tracing_cost()
    print(f"\n{SEPARATOR}")
    print("  END OF MEDIUM SOLUTIONS")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
