# Jour 8 — ML System Design : Introduction

## Pourquoi le ML en production n'a presque rien a voir avec le ML en notebook

**Exemple d'abord** : Tu entraines un modele de scoring de credit qui atteint 92% d'AUC sur ton jeu de test. Tu es content. Tu le deploies. Une semaine plus tard, les commerciaux se plaignent : le modele refuse 40% des bons clients. Pourquoi ? Parce qu'en production, la feature `revenu_net_mensuel` est calculee a partir du champ `salary`, alors qu'en entrainement tu utilisais `monthly_income` (apres un `groupby().mean()`). Resultat : des inputs legerement decales, une distribution differente, un modele qui explose.

C'est ce qu'on appelle le **training-serving skew**. Et c'est, avec les problemes de donnees, la premiere cause d'echec des projets ML en production. Le code du modele represente < 5% du systeme. Le reste : pipelines de donnees, feature stores, registries, orchestration, monitoring, retraining, A/B tests.

**Key takeaway** : Un systeme ML n'est pas un modele. C'est un systeme distribue complexe dont le modele n'est qu'un composant. Le design de ce systeme determine la reussite du projet autant ou plus que la qualite du modele.

---

## Le cycle de vie complet d'un systeme ML

```
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Data    │───>│  Train   │───>│  Eval    │───>│  Deploy  │───>│ Monitor  │
    │ ingestion│    │          │    │ offline  │    │          │    │          │
    └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │               │               │
         │               │               │               │               │
         v               v               v               v               v
    Feature         Experiment      Offline        Shadow /         Drift /
    store           tracking        metrics        canary /         performance /
    (Feast,         (MLflow,        + offline      A-B test         data quality
    Tecton)         Weights&Biases) holdout                         (Evidently,
                                                                    Arize)
                                        ^                               │
                                        │                               │
                                        └───────────────────────────────┘
                                         Retraining triggered by drift
```

Ce cycle n'est pas lineaire : c'est une boucle. Un bon systeme ML retraine automatiquement quand les donnees drift, re-evalue le nouveau modele offline, puis re-deploie progressivement (shadow -> canary -> 100%).

---

## Feature Store : pourquoi c'est le composant le plus important

Un **feature store** est une base de donnees specialisee qui stocke les features utilisees par les modeles. Il resout trois problemes majeurs :

1. **Reutilisation** : une feature calculee par l'equipe A (ex: `user_avg_order_value_30d`) peut etre consommee par les modeles des equipes B, C, D. Sans feature store, chaque equipe refait le calcul dans son coin, avec des bugs differents.
2. **Consistance offline/online** : la meme feature doit etre calculee de maniere identique a l'entrainement (batch, sur des fenetres historiques) et a l'inference (online, sur les donnees du moment).
3. **Point-in-time correctness** : quand on genere un dataset d'entrainement, chaque exemple doit voir les features telles qu'elles etaient a l'instant de l'evenement, pas telles qu'elles sont aujourd'hui. Sinon -> fuite de donnees (data leakage).

### Architecture d'un feature store

```
                            ┌───────────────────────┐
                            │   Feature Definitions  │
                            │  (SQL, Python, Spark)  │
                            └────────┬──────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    v                v                v
            ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
            │ Batch compute │ │ Stream       │ │ On-demand    │
            │ (Spark, DBT)  │ │ (Flink, KSQL)│ │ (real-time)  │
            └───────┬───────┘ └──────┬───────┘ └──────┬───────┘
                    │                │                │
                    v                v                v
            ┌───────────────────────────────────────────┐
            │          Offline store (training)          │
            │     Parquet / BigQuery / Snowflake          │
            │     Long retention, point-in-time joins     │
            └───────────────┬───────────────────────────┘
                            │ materialize
                            v
            ┌───────────────────────────────────────────┐
            │           Online store (serving)            │
            │       Redis / DynamoDB / Cassandra           │
            │       Latence < 10 ms, last value only       │
            └────────────────────────────────────────────┘
```

**Outils populaires** : Feast (open-source), Tecton (SaaS, cree par les fondateurs de Feast), Hopsworks, AWS SageMaker Feature Store, Databricks Feature Store.

---

## Model Registry : versioning sans chaos

Un **model registry** est un catalogue versionne des modeles entraines. Pour chaque modele tu stockes :

- Les **artifacts** (poids, config, tokenizer, preprocessor pipeline)
- Les **metadata** (dataset utilise, hyperparametres, metriques offline, date, auteur)
- Le **lineage** (quelle version de code, quel commit git, quels features)
- Le **stage** (dev, staging, production, archived)

```
Model: credit-scoring-v1
├── version 1.0.0  [archived]   AUC=0.87  2025-01-10
├── version 1.1.0  [archived]   AUC=0.89  2025-02-15
├── version 1.2.0  [production] AUC=0.91  2025-03-20  <- live
└── version 1.3.0  [staging]    AUC=0.92  2025-04-05  <- candidate
```

**Outils** : MLflow Model Registry (le plus repandu), Weights & Biases, Vertex AI Model Registry, SageMaker Model Registry.

**Pourquoi c'est critique** : quand un modele en prod commence a mal se comporter, tu dois pouvoir rollback en un clic. Sans registry, tu cherches les artifacts dans des buckets S3 et tu pries pour que le preprocessor soit compatible.

---

## Training-serving skew : le killer silencieux

Le skew survient quand la distribution des features vues au serving est differente de celle vue a l'entrainement. Causes frequentes :

| Cause | Exemple | Mitigation |
|---|---|---|
| **Code divergent** | Feature calculee en Python au training et en SQL au serving | Feature store unique |
| **Donnees manquantes differentes** | Training : NaN fillna(0). Serving : NaN laisse brut | Pipelines partages |
| **Timezone / formats** | Training utilise UTC, serving utilise local | Conventions documentees |
| **Features leakees** | Feature calculee apres coup (ex: refund rate sur 30j pour un order de 5j) | Point-in-time correctness |
| **Feature drift** | La distribution des users change (ex: COVID) | Monitoring + retraining |

**Regle d'or** : la fonction `compute_features(event)` doit etre **exactement la meme** en training et en serving. C'est pour ca que les feature stores existent.

---

## Batch vs real-time inference

Deux architectures fondamentalement differentes.

### Batch inference

Tu calcules les predictions **en avance**, de maniere periodique, et tu stockes les resultats dans une DB. L'application lit ces resultats.

```
  cron toutes les 6h
      │
      v
  ┌─────────────────────┐
  │ Spark / Airflow job │ -- lit tous les users -- applique le modele --
  └──────────┬──────────┘                                              │
             │                                                         v
             v                                             ┌──────────────────┐
     ┌──────────────┐                                      │  Predictions DB  │
     │  Data source │                                      │  (PostgreSQL,    │
     └──────────────┘                                      │   DynamoDB)      │
                                                           └────────┬─────────┘
                                                                    │
                                                                    v
                                                               App lit ici
```

**Avantages** : simple, pas de contraintes de latence, pas de GPU online, debuggable.
**Inconvenients** : predictions pas fraiches (6h de retard), scale quadratique si l'input change souvent.

**Use cases** : recommandations quotidiennes, credit scoring, prevision de demande.

### Real-time (online) inference

Le modele est un service (API) qui repond a chaque requete. L'input arrive, le modele predit, la reponse repart.

```
  client ─── request ──> API ──> feature store (online) ──> model serving ──> response
                                          ^                        │
                                          └──── features ──────────┘
```

**Avantages** : predictions toujours a jour, possible sur des donnees non vues avant, necessaire pour des use cases comme la detection de fraude.
**Inconvenients** : latence critique (p99 < 100ms), infra plus complexe, plus couteux.

**Use cases** : fraud detection, recommandations temps reel (feed), pricing dynamique, LLM serving.

### Le compromis : micro-batch

Entre les deux : traiter les inputs en mini-lots toutes les N secondes. Utilise pour les cas ou le cout d'un modele par requete est prohibitif mais ou on veut une latence raisonnable.

---

## Offline evaluation vs online evaluation

**Offline** : tu tests ton modele sur un jeu de donnees historique (holdout, train/val/test). Tu mesures AUC, precision, recall, RMSE, etc.

**Probleme** : les metriques offline ne correlent pas parfaitement avec les metriques business. Un modele avec AUC=0.92 n'est pas forcement meilleur qu'un AUC=0.90 si le premier fait des erreurs plus couteuses dans des cas rares.

**Online** : tu deploies le modele (partiellement) et tu mesures son impact reel sur les metriques business (CTR, revenue, conversion rate).

**Regle** : avant de promote un modele en production, il doit passer les deux evaluations.

---

## Shadow deployment : tester sans risque

Le **shadow deployment** consiste a envoyer les requetes reelles au nouveau modele **en parallele** de l'ancien, mais a ne renvoyer que la reponse de l'ancien a l'utilisateur. Les predictions du nouveau sont loggees et comparees a celles de l'ancien.

```
                  ┌────────────────┐
   request ──────>│  Old model V1  │──────> response to user
          │       └────────────────┘
          │
          └──────>┌────────────────┐
                  │  New model V2  │──────> log + metrics (not sent to user)
                  └────────────────┘
```

**Utilite** :
- Valider que le nouveau modele ne crash pas sur du trafic reel
- Mesurer la latence et la consommation de ressources en conditions reelles
- Comparer les predictions (distribution, disagreement rate)
- **Zero risque pour l'utilisateur**

Apres validation en shadow : canary (1% -> 10% -> 50% -> 100%) ou A/B test (50/50 avec metriques business).

---

## Tradeoffs recapitulatifs

| Dimension | Choix A | Choix B | Quand A ? |
|---|---|---|---|
| Inference | Batch | Real-time | Predictions peuvent attendre, volume stable |
| Feature store | Build | Buy (Tecton) | Petite equipe, peu de features -> build. Grande scale -> buy |
| Model registry | MLflow (OSS) | SageMaker | OSS si on veut le controle, managed si on veut la vitesse |
| Deploiement | Rolling update | Blue/green | Rolling si stateless, B/G si switch atomique requis |
| Validation | Shadow | A/B test | Shadow en premier, A/B pour mesurer l'impact business |

---

## Exemples reels

- **Netflix** : feature store interne, ~10K features, retraining quotidien, A/B test systematique (chaque change passe par un test)
- **Uber Michelangelo** : feature store + registry + deployment unifies, utilise par des centaines d'equipes
- **Airbnb Zipline** : feature store pensé pour le point-in-time correctness
- **LinkedIn** : shadow deployment systematique avant toute promotion en prod

---

## Flash cards

**Q: Qu'est-ce que le training-serving skew ?**
R: Quand la distribution ou le calcul des features differe entre l'entrainement et l'inference. Cause principale d'echec en prod. Mitigation : feature store unique.

**Q: Pourquoi un feature store au lieu d'un simple cache Redis ?**
R: Parce qu'il faut resoudre le point-in-time correctness (pas de data leakage), la consistance offline/online, et la reutilisation entre equipes.

**Q: Batch ou real-time inference pour un systeme de recommandations Netflix ?**
R: Hybride. Batch pour la majorite des users (refresh quotidien). Real-time pour ajuster selon le contexte (device, heure, dernier film regarde).

**Q: Difference entre shadow deployment et canary ?**
R: Shadow = le nouveau modele recoit les requetes mais ses reponses ne sont pas envoyees au user. Canary = le nouveau modele sert reellement un petit % des users.

**Q: Que stocke un model registry ?**
R: Les artifacts (poids), les metadata (metriques, dataset, hyperparams), le lineage (git commit), et le stage (dev/staging/prod).

---

## Key takeaways

- Un systeme ML = data pipeline + feature store + training + registry + serving + monitoring. Le modele est < 5%.
- Le training-serving skew est le killer silencieux. Feature store = prevention.
- Batch vs real-time = choix structurant. Commence toujours par batch si possible.
- Shadow deployment avant canary avant A/B test. Ne jamais deployer directement en prod.
- Un bon systeme ML retraine automatiquement. Le monitoring doit declencher le retraining.
