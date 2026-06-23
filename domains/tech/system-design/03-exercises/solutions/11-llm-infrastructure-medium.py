"""
Solutions -- Day 11 MEDIUM Exercises: LLM Infrastructure

Worked solutions with the reasoning step by step. Assertions lock the
key calculations so the file is self-checking.

Usage:
    python3 11-llm-infrastructure-medium.py
"""

SEPARATOR = "=" * 70


# =============================================================================
# MEDIUM -- Exercise 1 : Cut the LLM bill via routing + cache
# =============================================================================

def medium_1_cost_optimization():
    """Quantify routing + semantic cache + prompt caching savings."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Cut the LLM bill via routing + cache")
    print(SEPARATOR)

    daily_req = 3_000_000
    tok_in, tok_out = 2500, 400
    sys_prompt_tok = 1200

    # prices : $/1M tokens (in, out)
    std = (2.50, 10.00)
    nano = (0.05, 0.40)
    mini = (0.15, 0.60)

    def req_cost(price, t_in=tok_in, t_out=tok_out):
        return t_in * price[0] / 1_000_000 + t_out * price[1] / 1_000_000

    # 1. Current cost (all standard)
    cur_day = daily_req * req_cost(std)
    cur_month = cur_day * 30
    print(f"\n  1. Current cost (all standard) :")
    print(f"     Per request : ${req_cost(std):.5f}")
    print(f"     Per day     : ${cur_day:,.0f}")
    print(f"     Per month   : ${cur_month:,.0f}")

    # 2. After routing by tier
    mix = {"nano": (0.50, nano), "mini": (0.30, mini), "std": (0.20, std)}
    routed_day = sum(daily_req * frac * req_cost(price) for frac, price in mix.values())
    routed_month = routed_day * 30
    print(f"\n  2. After routing by tier (50% nano, 30% mini, 20% std) :")
    for name, (frac, price) in mix.items():
        seg = daily_req * frac * req_cost(price) * 30
        print(f"     {name:<5} {frac:.0%} : ${seg:,.0f}/month")
    print(f"     Total : ${routed_month:,.0f}/month  (vs ${cur_month:,.0f})")

    # 3. + semantic cache 25% hit (hits are free)
    cache_hit = 0.25
    cached_month = routed_month * (1 - cache_hit)
    print(f"\n  3. + semantic cache ({cache_hit:.0%} hit, hits cost 0) :")
    print(f"     ${routed_month:,.0f} * (1 - {cache_hit}) = ${cached_month:,.0f}/month")

    # 4. + prompt caching on the repeated system prompt
    # The 1200 sys-prompt tokens are part of tok_in. Prompt caching makes their
    # read cost 10% of input price -> we save 90% of the sys-prompt input cost.
    # Estimate the system-prompt share of the input cost across the routed mix.
    sys_cost_saved_day = 0.0
    for frac, price in mix.values():
        # 90% saving on sys_prompt input tokens, only on the (1-cache_hit) live requests
        sys_cost_saved_day += daily_req * frac * (1 - cache_hit) * sys_prompt_tok * price[0] / 1_000_000 * 0.90
    sys_cost_saved_month = sys_cost_saved_day * 30
    final_month = cached_month - sys_cost_saved_month
    print(f"\n  4. + native prompt caching on the {sys_prompt_tok}-token system prompt :")
    print(f"     Saving ~90% of repeated sys-prompt input cost : ${sys_cost_saved_month:,.0f}/month")
    print(f"     New total : ${final_month:,.0f}/month")

    # 5. Total reduction factor
    factor = cur_month / final_month
    print(f"\n  5. Final cost : ${final_month:,.0f}/month")
    print(f"     Reduction vs start : {factor:.1f}x  (${cur_month:,.0f} -> ${final_month:,.0f})")

    # 6. Dominant cost post at the end
    print(f"\n  6. Dominant post at the end :")
    print(f"     The std tier (20% of traffic) now dominates because its per-request")
    print(f"     cost is ~50-200x the nano/mini tiers, and output tokens ($10/1M) cost")
    print(f"     4x input. Next move : push more 'std' work to 'mini' via better")
    print(f"     classification, and trim output length (structured/short answers).")

    # ---- assertions ----
    assert abs(req_cost(std) - 0.01025) < 1e-5, req_cost(std)
    assert 900_000 < cur_month < 950_000, cur_month            # ~$922K/month
    assert routed_month < cur_month / 3, "routing must cut cost > 3x"
    assert cached_month < routed_month, "cache must reduce further"
    assert final_month < cached_month, "prompt caching must reduce further"
    assert factor > 4, f"total reduction should be > 4x, got {factor:.1f}x"
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 2 : Fallback chain + circuit breaker for an SLA
# =============================================================================

def medium_2_fallback_sla():
    """Compose provider availabilities to hit a 99.9% SLA."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Fallback chain + circuit breaker for an SLA")
    print(SEPARATOR)

    a, b, c = 0.995, 0.990, 0.980
    sla = 0.999
    minutes_month = 30 * 24 * 60

    def downtime(avail):
        return (1 - avail) * minutes_month

    # 1. A alone
    print(f"\n  1. Provider A alone (99.5%) :")
    print(f"     Downtime : {downtime(a):.0f} min/month vs SLA budget {downtime(sla):.0f} min.")
    print(f"     {downtime(a):.0f} > {downtime(sla):.0f} -> SLA 99.9% NOT achievable with A alone.")

    # 2. A -> B
    avail_ab = 1 - (1 - a) * (1 - b)
    print(f"\n  2. Chain A -> B (at least one up) :")
    print(f"     1 - (1-{a})*(1-{b}) = {avail_ab:.5f} = {avail_ab*100:.3f}%")
    print(f"     Downtime : {downtime(avail_ab):.1f} min/month -> SLA TENU (< {downtime(sla):.0f} min).")

    # 3. A -> B -> C
    avail_abc = 1 - (1 - a) * (1 - b) * (1 - c)
    print(f"\n  3. Chain A -> B -> C :")
    print(f"     1 - (1-{a})*(1-{b})*(1-{c}) = {avail_abc:.6f} = {avail_abc*100:.4f}%")
    print(f"     Downtime : {downtime(avail_abc):.2f} min/month -> large margin.")

    # 4. Timeout
    print(f"\n  4. Why an 8s timeout is critical :")
    print(f"     The fallback only 'counts' as availability if you fail FAST. With an")
    print(f"     8s timeout, a dead primary costs 8s before switching to B -> the user")
    print(f"     still gets an answer. With a 60s timeout, the user waits a minute or")
    print(f"     gives up : that's experienced downtime even though B was available.")

    # 5. Circuit breaker
    print(f"\n  5. Role of the circuit breaker :")
    print(f"     After N consecutive failures, bypass the dead provider for M seconds.")
    print(f"     Without it : EVERY request hits the dead A, waits the 8s timeout, THEN")
    print(f"     falls back -> every user eats +8s latency during the whole outage.")
    print(f"     With it : after 5 fails, A is skipped instantly -> B serves directly.")

    # 6. Portability pitfalls
    print(f"\n  6. Portability pitfalls A -> B :")
    print(f"     a) A prompt tuned for A may underperform on B (different instruction")
    print(f"        following) -> test prompts on both, keep provider-specific variants.")
    print(f"     b) Output format differences (JSON strictness, system role handling)")
    print(f"        -> re-validate output guardrails per provider.")

    # ---- assertions ----
    assert downtime(a) > downtime(sla), "A alone must miss the SLA"
    assert avail_ab >= sla, "A->B must meet the SLA"
    assert abs(avail_ab - 0.99995) < 1e-5, avail_ab
    assert avail_abc > avail_ab, "adding C must improve availability"
    assert downtime(avail_abc) < 1, "A->B->C downtime should be < 1 min/month"
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 3 : Tune the semantic cache (threshold/scope/TTL)
# =============================================================================

def medium_3_cache_tuning():
    """Diagnose and fix semantic cache misconfigurations."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Tune the semantic cache")
    print(SEPARATOR)

    print(f"\n  1. I1 (cancel vs activate subscription) :")
    print(f"     Cause : threshold too low (0.85) -> semantically NEAR but OPPOSITE")
    print(f"     queries collide. Fix : raise to 0.92-0.97. Inverse risk : push it too")
    print(f"     high and the hit rate collapses (almost nothing matches).")

    print(f"\n  2. I2 (user B sees user A's order number) :")
    print(f"     Cause : GLOBAL scope leaks personalized answers across users.")
    print(f"     Policy : NEVER cache requests containing personal data globally.")
    print(f"     Either skip the cache for personalized queries, or scope per-user")
    print(f"     (which yields ~0% hit, so usually just skip).")

    print(f"\n  3. I3 (stale prices, 2 days old) :")
    print(f"     Cause : TTL of 24h is far too long for time-sensitive prices.")
    print(f"     Fix : short TTL (minutes) or event-driven invalidation. A UNIFORM TTL")
    print(f"     is wrong : freshness needs differ by data type.")

    print(f"\n  4. Differentiated cache policy :")
    policy = [
        ("Generic stable Q&A", "threshold 0.95", "global scope", "TTL hours-days"),
        ("Personal data", "no cache (or per-user)", "per-user only", "n/a"),
        ("Time-sensitive (prices, status)", "threshold 0.97", "global ok", "TTL minutes + invalidation"),
    ]
    for cat, thr, scope, ttl in policy:
        print(f"     {cat:<32} | {thr:<22} | {scope:<16} | {ttl}")

    print(f"\n  5. Realistic hit rates :")
    rates = [
        ("Generic stable Q&A", "30-60%"),
        ("Personal data", "~0% (and that's correct)"),
        ("Time-sensitive", "variable, capped by the short TTL"),
    ]
    for cat, r in rates:
        print(f"     {cat:<32} -> {r}")

    print(f"\n  6. Measuring cache QUALITY (not just hit rate) :")
    print(f"     Sample served cache hits and run an LLM-as-a-judge to check the cached")
    print(f"     answer actually fits the new query. Track a 'false positive rate'")
    print(f"     alongside hit rate. A high hit rate with rising false positives means")
    print(f"     the threshold is too low.")

    # ---- assertions (qualitative checks encoded as structure) ----
    assert len(policy) == 3, "policy must cover 3 categories"
    assert any("no cache" in p[1] for p in policy), "personal data must skip cache"
    assert any("minutes" in p[3] for p in policy), "time-sensitive needs short TTL"
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 11 MEDIUM : LLM INFRASTRUCTURE")
    print(SEPARATOR)
    medium_1_cost_optimization()
    medium_2_fallback_sla()
    medium_3_cache_tuning()
    print(f"\n{SEPARATOR}")
    print("  END OF MEDIUM SOLUTIONS")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
