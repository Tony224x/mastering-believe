"""
Solutions -- Jour 13 : Observabilite & MLOps
"""


def solution_exercice_1() -> None:
    """
    Exercice 1 -- Dashboard LLM.

    SECTIONS DU DASHBOARD

    1) PERFORMANCE
       - Latency p50, p99 (ms)                seuil : p99 > 3s -> warn, > 5s -> page
       - TTFT (time to first token, ms)       seuil : > 1500ms -> warn
       - Throughput (req/min)                 info / capacite
       - Queue depth du batcher               seuil : > 50 en continu -> leading

    2) COUTS
       - Cost per request ($)                 seuil : x2 baseline -> warn
       - Tokens in / out (avg)                info
       - Daily burn ($)                       seuil : > 120% forecast -> warn
       - Cache hit rate (%)                   seuil : chute brutale > 20% -> investigate

    3) QUALITE
       - User feedback score (thumbs up/down ratio)   seuil : < 0.85 -> warn
       - Faithfulness score (LLM-as-judge)            seuil : < 0.8 -> act
       - JSON validation rate                         seuil : < 0.98 -> investigate
       - Guardrail rejections rate                    info + trend

    4) FIABILITE
       - Error rate par provider                      seuil : > 2% -> warn, > 5% -> page
       - Fallback rate                                seuil : spike > 10% -> leading indicator
       - Circuit breaker state                        info
       - Drift score (PSI) sur prompt length / topic  seuil : > 0.25 -> act

    ALERTES
    -------
    a) p99 latency > 5 s pendant 5 min consecutives -> PAGE on call
    b) error rate d'un provider > 5% pendant 2 min -> PAGE (fallback declenche)
    c) fallback rate > 10% pendant 5 min -> WARN Slack + investigate
    d) drift PSI sur any key feature > 0.25 pendant 24h -> WARN +
       ouverture d'un incident de quality

    LEADING vs LAGGING
    ------------------
    Leading (detecte avant l'impact user) :
      - Queue depth du batcher
      - Fallback rate
      - Drift score
      - Cost per request (explosion = un bug d'inflation du contexte)
      - Cache hit rate chute (quelqu'un a casse le cache)

    Lagging (constate l'impact) :
      - User feedback thumbs down
      - Customer complaints / tickets support
      - Churn
      - p99 latency (quand ca ralentit, les users le sentent deja)

    Principe : investir prioritairement dans les leading indicators. Les
    lagging confirment, les leading previennent.
    """


def solution_exercice_2() -> None:
    """
    Exercice 2 -- Lecture de PSI credit scoring.

    Mois 6 :
      monthly_income       : 0.08  -> no drift (watch)
      debt_ratio           : 0.22  -> moderate drift (borderline "act")
      age                  : 0.01  -> stable (normal : demographic slow)
      num_late_payments    : 0.35  -> **significant drift, act NOW**
      employment_type      : 0.11  -> moderate stable (~seuil bas)

    Q2 -- La plus inquietante : **num_late_payments**.
      Pourquoi :
        - PSI > 0.25 = seuil d'alerte franchi
        - TREND CROISSANT depuis le mois 2 (0.05 -> 0.35), ce n'est pas un
          bruit : c'est une tendance de fond.
        - C'est probablement une feature cle du credit scoring -> impact
          direct sur les predictions.

    Q3 -- Actions immediates :
      1. **Investiguer la cause root** avant de retrain aveuglement :
         - Changement externe ? (crise economique, regulation)
         - Bug pipeline ? (calcul different, nouvelle source de donnees,
           code divergent entre training et serving -- cf J8)
         - Changement user base ? (nouveau marche geographique)
      2. Comparer la distribution actuelle a la baseline graphiquement.
      3. Mesurer l'impact business : les metriques de qualite de decision
         (approval rate, default rate) ont-elles bouge ?
      4. Si impact confirme : retrain sur donnees recentes + re-deployer
         via shadow -> canary.
      5. Si impact non confirme : documenter et surveiller chaque semaine.

    Q4 -- employment_type est "borderline stable" a ~0.11.
      Il est AU-DESSUS de 0.1 mais ne bouge pas. Deux lectures :
        - Il y a une divergence constante entre training et prod (un bug
          de pipeline ancien) -> a investiguer.
        - Ou bien l'encodage categorique a naturellement un PSI de base
          plus eleve -> baseliner apres investigation.
      Pas d'action urgente mais c'est un bon candidat pour une
      investigation offline.

    Q5 -- Plan de rollout d'un nouveau modele :

      Etape 1 : Train sur donnees recentes (3 derniers mois)
      Etape 2 : Offline eval gate :
                 - AUC >= baseline + 0.01
                 - Test sur des hold-outs par segment (pas de regression
                   sur un segment clé)
                 - No unfair bias (parity check par groupe protege)
      Etape 3 : Register en staging (MLflow)
      Etape 4 : Shadow deploy : 100% du trafic fait tourner V2 en paralelle,
                log les predictions. Monitor pendant 48h. Check :
                  - disagreement_rate avec V1 < 15%
                  - distribution des scores V2 coherente
      Etape 5 : Canary : 1% -> 10% -> 25% -> 50% -> 100% sur 7 jours.
                Check apres chaque etape :
                  - accuracy / approval rate
                  - business metrics (default rate, volume)
                  - latency p99
                Si degrade : ROLLBACK auto immediat (feature flag).
      Etape 6 : Archive V1, marquer V2 comme production.

      Strategie de rollback :
        - Feature flag controle la version (config Redis, pas redeploy)
        - Monitoring temps reel des metriques cles
        - Auto-rollback si accuracy prod < threshold pendant N min
        - Rollback manuel 1-click via le dashboard
    """


def solution_exercice_3() -> None:
    """
    Exercice 3 -- CI/CD pour un classifieur de sentiment.

    ARCHITECTURE GLOBALE

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

    OUTILS
      - Data versioning : DVC ou LakeFS
      - Orchestration   : Airflow, Prefect ou Kubeflow
      - Training        : PyTorch / sklearn selon le modele
      - Eval            : custom + Great Expectations pour data quality
      - Registry        : MLflow Model Registry
      - Deployment      : KServe ou Seldon sur EKS, ou ECS
      - Feature flags   : LaunchDarkly ou config Redis
      - Monitoring      : Evidently (drift) + Prometheus + Grafana +
                          Langfuse (si LLM en amont)

    GATES (conditions bloquantes)
      Etape "Data validation" :
        - Schema match (pas de colonne manquante, types coherents)
        - Null rate < 5%
        - No duplicate sur les IDs
      Etape "Train" :
        - Convergence (loss stable)
        - Pas d'overfitting (val_loss < 1.1 * train_loss)
      Etape "Offline eval" :
        - Accuracy test >= baseline + 0.02
        - F1 per class >= 0.80
        - Benchmark latence p99 < 200ms sur 1000 inputs
      Etape "Shadow" :
        - 24h avec disagreement_rate < 15%
        - Pas d'erreur
      Etape "Canary" :
        - Accuracy prod > 84%
        - p99 < 200 ms
        - User feedback stable

    DEPLOIEMENT MULTI-REGION (4 regions)
      Etape 1 : Region1 avec 5% de trafic canary
      Etape 2 : Region1 100% + Region2 canary 5%
      Etape 3 : Region1 + 2 a 100%, Region3 canary
      Etape 4 : Toutes a 100%
      Delais entre etapes : 4-24h pour stabiliser metriques.
      NEVER : deployer les 4 regions en meme temps. Un bug = impact global.

    ROLLBACK (si accuracy degrade a +72h)
      Trigger automatique :
        - accuracy prod < 83% pendant 30 min
        - OU spike de user complaints
        - OU drift PSI > 0.3 sur une feature cle
      Action :
        - Feature flag flip : load_model("previous_version")
        - Alerter l'equipe
        - Ouvrir un incident
      Rollback = 1 commande ou 1 click, pas un redeploy Kubernetes.

    RETRAIN FORCE HORS SCHEDULE
      Triggers :
        - Drift PSI > 0.25 sur 2+ features principales
        - Accuracy prod < 84% pendant 4h consecutives
        - User feedback score chute > 10 points en 24h
        - Nouveau marche / langue
        - Changement upstream (un produit critique change de structure)
      Un humain valide avant de lancer le retrain sauvage.

    Principe : **le pipeline lui-meme est un produit**. Il doit avoir des
    tests, du versioning, du monitoring. C'est le plus gros investissement
    long terme d'une equipe ML serieuse.
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)
