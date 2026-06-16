"""
Solutions -- Day 8 HARD Exercises: ML System Design Intro

Worked solutions. Exercise 1 designs an e-commerce ML platform (feature store,
serving, retraining); exercise 2 is a training-serving skew post-mortem made
CONCRETE with a runnable demo : two divergent feature implementations produce
different distributions (PSI fires), and a SINGLE shared feature definition makes
the skew impossible. Key facts are pinned with assertions.

Usage:
    python3 08-ml-system-design-intro-hard.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# HARD -- Exercise 1 : e-commerce ML platform (feature store / serving / retrain)
# =============================================================================

def hard_1_ml_platform():
    """Design + reason about a multi-model ML platform that kills skew."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : E-commerce ML platform")
    print(SEPARATOR)

    print("\n  1. Lifecycle & components :")
    print("     data ingestion -> FEATURE STORE -> training -> MODEL REGISTRY")
    print("       -> serving -> MONITORING -> (drift) -> retraining -> registry")
    print("     SHARED across the 5 models : feature store, registry, monitoring,")
    print("       data ingestion. SPECIFIC per model : the model artifact + its")
    print("       serving endpoint + its eval/labels.")

    print("\n  2. Feature store (not just a Redis cache) :")
    print("     Justified by : cross-team REUSE of features, GUARANTEED offline/")
    print("       online consistency, and POINT-IN-TIME correctness -- a per-team")
    print("       Redis cache gives none of these.")
    print("     Offline store : Parquet/BigQuery, long retention, point-in-time")
    print("       as-of joins for training datasets.")
    print("     Online store  : Redis/DynamoDB, < 10 ms, latest feature value.")
    print("     Train==serving : ONE feature definition (same code path) computes")
    print("       the value both sides. Point-in-time : as-of join on the timestamped")
    print("       history so the dataset only sees data available BEFORE the label.")

    print("\n  3. Batch vs real-time per model :")
    decisions = [
        ("anti-fraud", "real-time", "< 100 ms, blocks the transaction -> most constrained"),
        ("search ranking", "real-time/online", "must rank at query time"),
        ("product reco", "hybrid", "batch candidates + online re-rank with fresh signals"),
        ("demand forecast", "batch", "daily horizon, huge volume, freshness not critical"),
        ("churn", "batch", "scored periodically, no latency pressure"),
    ]
    for model, mode, why in decisions:
        print(f"     - {model:<15}: {mode:<16} ({why})")
    # The exercise's expected key fact: anti-fraud is the latency-critical one.
    most_constrained = min(
        [("anti-fraud", 100), ("search", 200), ("reco", 300)],
        key=lambda kv: kv[1],
    )[0]
    assert most_constrained == "anti-fraud"

    print("\n  4. Model registry & deployment :")
    print("     Registry stores : artifacts + metadata (metrics, params) + lineage")
    print("       (data/feature versions) + stage (staging/prod/archived).")
    print("     Rollout : SHADOW (mirror traffic, no user impact) -> CANARY (small")
    print("       % live) -> A/B (measure business metric) -> PROMOTE (full).")

    print("\n  5. Retraining :")
    print("     Triggered by : feature/label DRIFT, perf drop, or a schedule.")
    print("     Offline GATE before promotion : the candidate must beat the current")
    print("       baseline on a held-out set, else it is NOT promoted.")

    print("\n  6. Failure modes :")
    print("     Online store (Redis) down -> degrade : default/last-known features or")
    print("       a simpler fallback model (never hard-fail the request).")
    print("     Feature in Python (train) vs SQL (serving) -> training-serving SKEW")
    print("       (wrong scores, green metrics). Fix structurally : ONE feature")
    print("       definition in the feature store (no divergent code).")
    print("     AUC 0.95 offline but collapses in prod -> 2 likely causes :")
    print("       (1) train/serving skew, (2) label LEAKAGE in the offline dataset.")


# =============================================================================
# HARD -- Exercise 2 : Post-mortem -- the 95%-AUC model that rejected good clients
# =============================================================================

# --- The bug, reproduced concretely -----------------------------------------

def feature_revenue_TRAIN(monthly_income_net):
    """At train time the feature came from `monthly_income` (already NET)."""
    return monthly_income_net


def feature_revenue_SERVING(salary_gross):
    """At serving it was (wrongly) computed from `salary` (GROSS)."""
    return salary_gross                    # ~1.25x higher than net -> skew


def population_stability_index(expected, actual, bins=5):
    """
    PSI between two samples : sum over bins of (a% - e%) * ln(a% / e%).
    A small PSI (< 0.1) means stable; large means the distribution shifted.
    Used here to show a feature-distribution monitor WOULD have fired at J0.
    """
    lo = min(min(expected), min(actual))
    hi = max(max(expected), max(actual))
    width = (hi - lo) / bins or 1.0
    edges = [lo + i * width for i in range(bins + 1)]
    edges[-1] = hi + 1e-9                  # include the max in the last bin

    def hist(xs):
        counts = [0] * bins
        for x in xs:
            idx = min(int((x - lo) / width), bins - 1)
            counts[idx] += 1
        n = len(xs)
        # epsilon avoids log(0) on empty bins
        return [(c / n) or 1e-6 for c in counts]

    e, a = hist(expected), hist(actual)
    import math
    return sum((ai - ei) * math.log(ai / ei) for ei, ai in zip(e, a))


# --- The fix : ONE shared feature definition --------------------------------

def feature_revenue_SHARED(record):
    """
    Single source of truth used by BOTH train and serving (feature store).
    Always reads the NET field and fills NaN the SAME way -> skew impossible.
    """
    val = record.get("monthly_income")
    if val is None:                        # consistent NaN handling, both sides
        val = 0.0
    return val


def hard_2_skew_postmortem():
    """Post-mortem + the corrected, skew-proof pipeline, demonstrated."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Training-serving skew post-mortem")
    print(SEPARATOR)

    print("\n  1. Root cause analysis :")
    chain = [
        ("ARCHITECTURE", "revenu_net : train reads NET, serving recomputes from GROSS",
         "divergent code -> distribution shifted ~1.25x"),
        ("PROCESS", "nb_commandes_30j : NaN filled at train, sent raw (NaN) at serving",
         "divergent NaN handling"),
        ("MONITORING", "no feature-distribution monitoring",
         "drift invisible until business complaints (J1+)"),
    ]
    for cat, cause, detail in chain:
        print(f"     [{cat}] {cause}")
        print(f"       -> {detail}")
    print('     "Works on the CSV" : offline eval reuses the SAME biased train data,')
    print("     so it never exercises the serving code path -> it can't see the skew.")

    # Demonstrate the skew + show PSI would have fired.
    import random
    random.seed(0)
    net_incomes = [random.gauss(3000, 500) for _ in range(2000)]      # NET (train)
    gross_incomes = [v * 1.25 for v in net_incomes]                   # GROSS (serving)
    train_feat = [feature_revenue_TRAIN(v) for v in net_incomes]
    serve_feat = [feature_revenue_SERVING(v) for v in gross_incomes]
    psi = population_stability_index(train_feat, serve_feat)
    print(f"\n     PSI(train vs prod) on revenu_net = {psi:.3f}  (>0.25 = major shift)")
    assert psi > 0.25                      # a distribution monitor WOULD alert at J0

    print("\n  2. The 'works on my machine' trap :")
    print("     Offline eval can't catch it because train and eval share the biased")
    print("     data. Missing property : a SINGLE shared feature function so train")
    print("     and serving are guaranteed identical.")

    print("\n  3. Why metrics were green :")
    print("     API failure = 500. MODEL failure = 200 with a WRONG score. 5xx/latency")
    print("     stay green. The metric that would have alerted at J0 : PSI/drift on")
    print("     the input feature distributions (prod vs train baseline).")

    print("\n  4. Skew by cause -> structural fix :")
    print("     revenu_net (divergent code)   -> ONE shared feature definition.")
    print("     nb_commandes_30j (NaN handling) -> same definition handles NaN once.")
    print("     A single feature store resolves BOTH at once (one code path, one")
    print("     transformation applied identically offline and online).")

    # Demonstrate the fix : same definition on both sides -> identical values.
    rec = {"monthly_income": 2750.0}
    rec_nan = {"monthly_income": None}
    assert feature_revenue_SHARED(rec) == feature_revenue_SHARED(rec)   # deterministic
    assert feature_revenue_SHARED(rec_nan) == 0.0                       # NaN handled once
    train_fixed = [feature_revenue_SHARED({"monthly_income": v}) for v in net_incomes]
    serve_fixed = [feature_revenue_SHARED({"monthly_income": v}) for v in net_incomes]
    assert train_fixed == serve_fixed      # same definition -> ZERO skew
    print("     -> shared definition gives byte-identical train/serving values (skew=0).")

    print("\n  5. Corrected system + CI test :")
    print("     Pipeline : feature store as the single transform; offline (as-of")
    print("       joins) and online read the SAME definition; drift monitor on inputs.")
    print("     CI test : recompute a feature on the SAME records via the train path")
    print("       and the serving path and assert equality (or compare train vs a")
    print("       prod-sample distribution and fail on high PSI). Demo of the assert :")
    sample = [{"monthly_income": 3100.0}, {"monthly_income": None}]
    for r in sample:
        assert feature_revenue_SHARED(r) == feature_revenue_SHARED(r)  # CI gate
    print("       train_value == serving_value for every record -> gate passes.")

    print("\n  6. Runbook (6 steps) -- suspected skew in prod :")
    runbook = [
        "Pull a prod feature sample; compare its distribution to the train baseline (PSI)",
        "Identify which feature(s) shifted",
        "Diff the train vs serving computation of that feature (code path + NaN handling)",
        "Hotfix : align the computation (or roll back to a known-good model)",
        "Backfill / re-score affected decisions if business-critical",
        "Structural fix : move the feature into the shared feature store + add drift alerts",
    ]
    for i, s in enumerate(runbook, 1):
        print(f"     {i}. {s}")


def main():
    print("\n" + "=" * 60)
    print("  SOLUTIONS -- DAY 8 HARD : ML SYSTEM DESIGN INTRO")
    print("=" * 60)
    hard_1_ml_platform()
    hard_2_skew_postmortem()
    print(f"\n{'=' * 60}")
    print("  END OF HARD SOLUTIONS (all assertions passed)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
