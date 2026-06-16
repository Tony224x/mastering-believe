"""
Solutions -- Day 8 MEDIUM Exercises: ML System Design Intro

Worked solutions. Point-in-time leakage, batch-vs-real-time cost, and the
shadow/canary rollout numbers are computed with assertions.

Usage:
    python3 08-ml-system-design-intro-medium.py
"""

SEPARATOR = "=" * 70


# =============================================================================
# MEDIUM -- Exercise 1 : Point-in-time correctness -- spot the data leakage
# =============================================================================

def medium_1_point_in_time():
    """Show how a naive join leaks the future and inflates offline AUC."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Point-in-time correctness -- spot the data leakage")
    print(SEPARATOR)

    event_time = "2026-01-10"
    # purchases as (date, count)
    purchases = [("2026-01-05", 1), ("2026-01-20", 3), ("2026-02-01", 2)]

    naive = sum(c for _, c in purchases)                       # today's state = 6
    point_in_time = sum(c for d, c in purchases if d < event_time)   # only before event = 1

    print(f"\n  1. Naive join (today's state) for the {event_time} example :")
    print(f"     count(*) over all purchases = {naive} (includes purchases AFTER the click)")

    print(f"\n  2. Correct point-in-time value :")
    print(f"     only purchases strictly before {event_time} = {point_in_time}")

    print(f"\n  3. Why the naive value leaks the future :")
    print(f"     it counts purchases made AFTER event_time -- info the model could never")
    print(f"     have at prediction time in production.")

    print(f"\n  4. Excellent offline AUC, collapse in prod :")
    print(f"     the model 'cheats' offline by correlating with future purchases; that")
    print(f"     signal is absent at serving time -> the model degrades in production.")

    print(f"\n  5. How a feature store fixes it :")
    print(f"     it does a point-in-time (as-of) join; the offline store keeps the")
    print(f"     TIME-VERSIONED history of each feature so you can read its value AS OF")
    print(f"     event_time, not today.")

    print(f"\n  6. CI test that catches the bug :")
    print(f"     recompute the feature AS OF event_time and assert == expected ({point_in_time}).")
    print(f"     If it returns {naive}, the join is leaking -> fail the build.")

    # ---- assertions ----
    assert naive == 6, naive
    assert point_in_time == 1, point_in_time
    assert naive != point_in_time
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 2 : Batch vs real-time with the cost computed
# =============================================================================

def medium_2_batch_vs_realtime():
    """Arbitrate batch / real-time / micro-batch with actual cost numbers."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Batch vs real-time with the cost computed")
    print(SEPARATOR)

    users = 50_000_000

    # batch
    runs_per_day = 4                # every 6h
    cost_per_run = 200
    batch_day = runs_per_day * cost_per_run

    # real-time
    home_views = 5
    cost_per_scoring = 0.0002
    realtime_day = users * home_views * cost_per_scoring

    print(f"\n  1. Daily cost :")
    print(f"     batch     : {runs_per_day} runs * ${cost_per_run} = ${batch_day:,}/day")
    print(f"     real-time : {users:,} * {home_views} * ${cost_per_scoring} = ${realtime_day:,.0f}/day")
    ratio = realtime_day / batch_day
    print(f"     -> batch is ~{ratio:.0f}x cheaper")

    print(f"\n  2. Freshness :")
    print(f"     batch : up to 6h stale. real-time : immediate.")

    print(f"\n  3. After a 'I like running shoes' click :")
    print(f"     batch reflects it after up to 6h; real-time on the very next home view.")

    print(f"\n  4. Hybrid :")
    print(f"     candidate generation in BATCH (heavy, stable) + online RE-RANKING with")
    print(f"     fresh context (device, time, last click). Best cost/freshness ratio :")
    print(f"     the expensive part is amortized in batch, the cheap part stays fresh.")

    print(f"\n  5. Micro-batch :")
    print(f"     use it when per-request cost is prohibitive but you still want a few-")
    print(f"     seconds latency -- group inputs every N seconds and score the mini-batch.")

    # ---- assertions ----
    assert batch_day == 800, batch_day
    assert abs(realtime_day - 50_000) < 1, realtime_day
    assert ratio > 50, ratio              # batch ~62x cheaper
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 3 : Shadow then canary -- size the safe rollout
# =============================================================================

def medium_3_shadow_canary():
    """Lay out shadow -> canary -> 100% and quantify each stage."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Shadow then canary -- size the safe rollout")
    print(SEPARATOR)

    daily_predictions = 100_000_000

    print(f"\n  1. Shadow deployment :")
    print(f"     V2 receives the REAL traffic, but its response is NOT sent to the user.")
    print(f"     V2 predictions are logged and compared to V1.")

    print(f"\n  2. Three pre-quality signals :")
    print(f"     - it doesn't crash on real traffic")
    print(f"     - it holds latency / resource budget")
    print(f"     - prediction distribution is sane (disagreement rate vs V1 reasonable)")

    disagreement = 0.18
    print(f"\n  3. {disagreement:.0%} disagreement with V1 :")
    print(f"     not alarming by itself -- you must check WHO is right on those cases")
    print(f"     (evaluate on labels). V2 may simply be better there.")

    canary_pct = 0.01
    canary_served = int(daily_predictions * canary_pct)
    print(f"\n  4. Canary at 1% :")
    print(f"     1% of {daily_predictions:,} = {canary_served:,} predictions/day actually")
    print(f"     served by V2. Start low to limit the blast radius if V2 is bad.")

    print(f"\n  5. Auto-rollback triggers :")
    print(f"     guardrail regressions : p99 latency up, error rate up, business metric")
    print(f"     (CTR/conversion) down -> roll back automatically.")

    print(f"\n  6. Why offline AUC isn't enough :")
    print(f"     offline metrics don't perfectly correlate with business impact -> shadow")
    print(f"     + canary + online measurement are required before promotion.")

    # ---- assertions ----
    assert canary_served == 1_000_000, canary_served
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 8 MEDIUM : ML SYSTEM DESIGN INTRO")
    print(SEPARATOR)
    medium_1_point_in_time()
    medium_2_batch_vs_realtime()
    medium_3_shadow_canary()
    print(f"\n{SEPARATOR}")
    print("  END OF MEDIUM SOLUTIONS (all assertions passed)")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
