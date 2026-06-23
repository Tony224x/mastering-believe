"""
Solutions -- Day 13 : Observability & MLOps
"""


def solution_exercice_1() -> None:
    """
    Exercise 1 -- LLM Dashboard.

    DASHBOARD SECTIONS

    1) PERFORMANCE
       - Latency p50, p99 (ms)                threshold : p99 > 3s -> warn, > 5s -> page
       - TTFT (time to first token, ms)       threshold : > 1500ms -> warn
       - Throughput (req/min)                 info / capacity
       - Batcher queue depth                  threshold : > 50 sustained -> leading

    2) COSTS
       - Cost per request ($)                 threshold : 2x baseline -> warn
       - Tokens in / out (avg)                info
       - Daily burn ($)                       threshold : > 120% forecast -> warn
       - Cache hit rate (%)                   threshold : sharp drop > 20% -> investigate

    3) QUALITY
       - User feedback score (thumbs up/down ratio)   threshold : < 0.85 -> warn
       - Faithfulness score (LLM-as-judge)            threshold : < 0.8 -> act
       - JSON validation rate                         threshold : < 0.98 -> investigate
       - Guardrail rejections rate                    info + trend

    4) RELIABILITY
       - Error rate per provider                      threshold : > 2% -> warn, > 5% -> page
       - Fallback rate                                threshold : spike > 10% -> leading indicator
       - Circuit breaker state                        info
       - Drift score (PSI) on prompt length / topic   threshold : > 0.25 -> act

    ALERTS
    -------
    a) p99 latency > 5 s for 5 consecutive min -> PAGE on call
    b) a provider's error rate > 5% for 2 min -> PAGE (fallback triggered)
    c) fallback rate > 10% for 5 min -> WARN Slack + investigate
    d) PSI drift on any key feature > 0.25 for 24h -> WARN +
       open a quality incident

    LEADING vs LAGGING
    ------------------
    Leading (detected before the user impact) :
      - Batcher queue depth
      - Fallback rate
      - Drift score
      - Cost per request (explosion = a context inflation bug)
      - Cache hit rate drop (someone broke the cache)

    Lagging (records the impact) :
      - User feedback thumbs down
      - Customer complaints / support tickets
      - Churn
      - p99 latency (by the time it slows down, the users already feel it)

    Principle : invest first in the leading indicators. The
    lagging ones confirm, the leading ones prevent.
    """


def solution_exercice_2() -> None:
    """
    Exercise 2 -- Reading PSI for credit scoring.

    Month 6 :
      monthly_income       : 0.08  -> no drift (watch)
      debt_ratio           : 0.22  -> moderate drift (borderline "act")
      age                  : 0.01  -> stable (normal : demographics move slowly)
      num_late_payments    : 0.35  -> **significant drift, act NOW**
      employment_type      : 0.11  -> moderately stable (~low threshold)

    Q2 -- The most worrying : **num_late_payments**.
      Why :
        - PSI > 0.25 = alert threshold crossed
        - INCREASING TREND since month 2 (0.05 -> 0.35), it is not
          noise : it is an underlying trend.
        - It is probably a key feature of the credit scoring -> direct
          impact on the predictions.

    Q3 -- Immediate actions :
      1. **Investigate the root cause** before blindly retraining :
         - External change ? (economic crisis, regulation)
         - Pipeline bug ? (different calculation, new data source,
           code divergence between training and serving -- cf. D8)
         - User base change ? (new geographic market)
      2. Compare the current distribution to the baseline graphically.
      3. Measure the business impact : have the decision quality metrics
         (approval rate, default rate) moved ?
      4. If the impact is confirmed : retrain on recent data + redeploy
         via shadow -> canary.
      5. If the impact is not confirmed : document and monitor weekly.

    Q4 -- employment_type is "borderline stable" at ~0.11.
      It is ABOVE 0.1 but not moving. Two readings :
        - There is a constant divergence between training and prod (an old
          pipeline bug) -> to investigate.
        - Or the categorical encoding naturally has a higher base
          PSI -> re-baseline after investigation.
      No urgent action but it is a good candidate for an
      offline investigation.

    Q5 -- Rollout plan for a new model :

      Step 1 : Train on recent data (last 3 months)
      Step 2 : Offline eval gate :
                 - AUC >= baseline + 0.01
                 - Test on hold-outs per segment (no regression
                   on a key segment)
                 - No unfair bias (parity check per protected group)
      Step 3 : Register in staging (MLflow)
      Step 4 : Shadow deploy : 100% of the traffic runs V2 in parallel,
                log the predictions. Monitor for 48h. Check :
                  - disagreement_rate with V1 < 15%
                  - V2 score distribution coherent
      Step 5 : Canary : 1% -> 10% -> 25% -> 50% -> 100% over 7 days.
                Check after each step :
                  - accuracy / approval rate
                  - business metrics (default rate, volume)
                  - latency p99
                If degraded : immediate auto ROLLBACK (feature flag).
      Step 6 : Archive V1, mark V2 as production.

      Rollback strategy :
        - A feature flag controls the version (Redis config, no redeploy)
        - Real-time monitoring of the key metrics
        - Auto-rollback if prod accuracy < threshold for N min
        - 1-click manual rollback via the dashboard
    """


def solution_exercice_3() -> None:
    """
    Exercise 3 -- CI/CD for a sentiment classifier.

    OVERALL ARCHITECTURE

        +--------+   +-----+   +-------+   +-------+   +----------+
        | commit |-->| CI  |-->| Data  |-->| Train |-->| Offline  |
        +--------+   |tests|   | valid.|   |       |   | eval gate|
                     +-----+   +-------+   +-------+   +-----+----+
                                                             |
                                                             v
                                                       +----------+
                                                       | Register |
                                                       |  MLflow  |
                                                       +-----+----+
                                                             |
                                                             v
                                                       +----------+
                                                       |  Staging |
                                                       +-----+----+
                                                             |
                                                             v
                                                       +----------+
                                                       |  Shadow  |
                                                       +-----+----+
                                                             |
                                                             v
                                                    +---------------+
                                                    | Canary region1|
                                                    +-------+-------+
                                                            |
                                                            v
                                                    +---------------+
                                                    | Canary region2|
                                                    +-------+-------+
                                                            |
                                                            v
                                                    +---------------+
                                                    | Full prod 4   |
                                                    | regions       |
                                                    +---------------+

    TOOLING
      - Data versioning : DVC or LakeFS
      - Orchestration   : Airflow, Prefect or Kubeflow
      - Training        : PyTorch / sklearn depending on the model
      - Eval            : custom + Great Expectations for data quality
      - Registry        : MLflow Model Registry
      - Deployment      : KServe or Seldon on EKS, or ECS
      - Feature flags   : LaunchDarkly or Redis config
      - Monitoring      : Evidently (drift) + Prometheus + Grafana +
                          Langfuse (if an LLM is upstream)

    GATES (blocking conditions)
      "Data validation" step :
        - Schema match (no missing column, coherent types)
        - Null rate < 5%
        - No duplicates on the IDs
      "Train" step :
        - Convergence (stable loss)
        - No overfitting (val_loss < 1.1 * train_loss)
      "Offline eval" step :
        - Test accuracy >= baseline + 0.02
        - Per-class F1 >= 0.80
        - Latency benchmark p99 < 200ms on 1000 inputs
      "Shadow" step :
        - 24h with disagreement_rate < 15%
        - No errors
      "Canary" step :
        - Prod accuracy > 84%
        - p99 < 200 ms
        - User feedback stable

    MULTI-REGION DEPLOYMENT (4 regions)
      Step 1 : Region1 with 5% canary traffic
      Step 2 : Region1 100% + Region2 canary 5%
      Step 3 : Regions 1 + 2 at 100%, Region3 canary
      Step 4 : All at 100%
      Delays between steps : 4-24h to let the metrics stabilize.
      NEVER : deploy all 4 regions at the same time. One bug = global impact.

    ROLLBACK (if accuracy degrades at +72h)
      Automatic trigger :
        - prod accuracy < 83% for 30 min
        - OR a spike of user complaints
        - OR PSI drift > 0.3 on a key feature
      Action :
        - Feature flag flip : load_model("previous_version")
        - Alert the team
        - Open an incident
      Rollback = 1 command or 1 click, not a Kubernetes redeploy.

    FORCED OUT-OF-SCHEDULE RETRAIN
      Triggers :
        - PSI drift > 0.25 on 2+ main features
        - Prod accuracy < 84% for 4 consecutive hours
        - User feedback score drops > 10 points in 24h
        - New market / language
        - Upstream change (a critical product changes its structure)
      A human validates before launching the off-cycle retrain.

    Principle : **the pipeline itself is a product**. It must have
    tests, versioning, monitoring. It is the biggest long-term
    investment of a serious ML team.
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)
