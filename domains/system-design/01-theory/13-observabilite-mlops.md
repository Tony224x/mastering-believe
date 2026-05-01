# Jour 13 — Observabilite & MLOps

## Pourquoi l'observability est la difference entre prototype et production

**Exemple d'abord** : Tu deploies un agent LLM en prod. Les users font 10K requetes par jour. Un matin, tu recois un message Slack : "le bot repond n'importe quoi depuis hier soir". Tu regardes tes logs : rien d'anormal, les appels OpenAI retournent des 200. Tu tentes de reproduire : impossible. 4 heures de debug perdu. La cause ? Un changement silencieux dans la distribution des inputs (users d'un nouveau marche) qui a fait derailler le prompt. Sans drift detection, sans tracing de tokens, sans scoring de faithfulness, tu es aveugle.

Les ML/LLM systems sont **non-deterministes** et **silencieux** dans leurs modes d'echec. Une API classique qui plante envoie un 500. Un modele qui hallucine envoie un 200. Sans observability dediee, les pannes sont invisibles jusqu'a ce qu'un user se plaigne.

**Key takeaway** : En 2026, tu ne peux pas faire de ML en prod sans tracing + drift monitoring + CI/CD modele. C'est aussi basique qu'un serveur HTTP sans logs.

---

## Les 3 piliers de l'observability LLM

```
  ┌────────────────────────────────────────────────────────────┐
  │                       OBSERVABILITY                          │
  │                                                              │
  │    ┌──────────┐      ┌──────────┐      ┌──────────┐         │
  │    │ TRACING  │      │ METRICS  │      │   LOGS   │         │
  │    │          │      │          │      │          │         │
  │    │ spans,   │      │ latency, │      │ raw      │         │
  │    │ LLM calls│      │ errors,  │      │ events,  │         │
  │    │ tool use,│      │ tokens,  │      │ prompts, │         │
  │    │ agent    │      │ cost,    │      │ responses│         │
  │    │ steps    │      │ feedback │      │          │         │
  │    └──────────┘      └──────────┘      └──────────┘         │
  └────────────────────────────────────────────────────────────┘
```

### Tracing (le plus important pour LLM/agents)

Un **trace** est un graphe de **spans** qui representent le chemin d'une requete a travers le systeme. Pour un agent :

```
  trace: "user asks about pricing"
   ├── span: supervisor_plan             250ms  tokens=120  $0.0003
   ├── span: search_agent                800ms
   │    ├── span: llm_call (rewrite)     200ms  tokens=80   $0.0001
   │    └── span: vector_search          100ms  hits=5
   ├── span: writer_agent                600ms
   │    └── span: llm_call               500ms  tokens=800  $0.004
   └── total                            1850ms            $0.0044
```

Chaque span a :
- **name** (identifiant logique)
- **start / end time** (latence)
- **parent** (arbre)
- **attributes** (tokens, cost, model, user_id, feedback)
- **events / logs** attaches

**Outils** : Langfuse, LangSmith, Arize Phoenix, Helicone, Weights & Biases Traces, Braintrust. Tous sont compatibles avec OpenTelemetry.

### Metrics (dashboards + alerts)

Metriques classiques (latence, error rate, throughput) + metriques specifiques LLM :

- **Tokens in/out** par minute
- **Cost per 1K requests**
- **Cache hit rate**
- **Fallback rate**
- **User feedback score** (thumbs up/down)
- **Faithfulness score** (evaluation automatique)
- **Drift score** (PSI / KL)

### Logs (pour le debug approfondi)

Log la conversation complete, le prompt utilise, la reponse brute. Piege : les logs LLM sont **gros** et contiennent souvent du PII. Bonnes pratiques :

- Logger le prompt avec un hash + un sample (pas le prompt complet systematiquement)
- Chiffrer / masquer les PII avant de logger
- TTL agressif (30-90 jours)
- Cote RGPD : log accesibles par equipe uniquement

---

## Drift detection

Le drift = la distribution des donnees change entre l'entrainement (ou la derniere release) et la production. Deux types :

### 1. Data drift (covariate shift)

Les **inputs** changent. Exemples :
- Le produit ouvre un nouveau marche (users parlent une autre langue)
- Un concurrent fait la une, les users posent plus de questions compar
- COVID : plus personne n'achete de voitures

Le modele n'a pas forcement tort, mais il est applique a des donnees qu'il n'a pas vues.

### 2. Concept drift

La **relation** input -> output change. Exemples :
- Fraud patterns evoluent : ce qui etait suspicious en 2023 ne l'est plus
- Style d'ecriture change, le modele de style devient date
- Politique de prix change, le modele de recommandation est desaligne

Le modele doit etre re-entraine, pas juste re-calibre.

### Comment detecter

Pour des **features numeriques** :

**PSI (Population Stability Index)** :

```
PSI = sum (baseline_pct - current_pct) * ln(baseline_pct / current_pct)
     over all bins
```

Valeurs d'alerte :
- PSI < 0.1 : pas de drift
- PSI 0.1 - 0.25 : drift modere, a surveiller
- PSI > 0.25 : drift significatif, action requise

**KL divergence** : mesure alternative, plus sensible aux petites probabilites.

**KS test** (Kolmogorov-Smirnov) : test statistique non parametrique sur les distributions continues.

Pour des **features categoriques** : chi-square test, Jensen-Shannon divergence.

Pour des **embeddings / textes** : mesurer le mean embedding par batch et tracker la distance au baseline.

### Setup practique

```
  daily job
     │
     v
  sample 1000 prod events  ──>  compute feature distributions  ──>  compare to baseline
                                                                            │
                                                                            v
                                                                 alert if PSI > 0.25
```

Outils : **Evidently AI**, **WhyLabs**, **Arize**, **NannyML**. Pour les LLMs, mesurer plutot le drift sur les prompts et les responses (shift en longueur, en topic, en sentiment).

---

## Data quality monitoring

Drift != data quality. Data quality = valeurs invalides, nulls inattendus, types qui changent, duplications.

### Checks communs

- **Completeness** : % de NaN par feature
- **Uniqueness** : nombre de duplicates
- **Validity** : valeurs dans les enums attendus, ranges numeriques
- **Consistency** : les relations entre features sont respectees (ex: `date_end > date_start`)
- **Freshness** : la derniere mise a jour est recente

**Outils** : Great Expectations, Soda, dbt tests.

---

## A/B testing pour ML

Ton modele V2 est meilleur que V1 en offline. Mais est-il meilleur en **business metric** ? Un A/B test repond a cette question.

### Setup

```
           ┌────────────────┐
  request ─┤  splitter (50/50) │
           └──────┬──────┬───┘
                  v      v
             model V1  model V2
                  │      │
                  v      v
              metrics  metrics
                  │      │
                  └──┬───┘
                     v
              statistical test
```

### Pieges classiques

1. **Pas de signal clair** : la difference n'est pas significative statistiquement. Solution : plus de trafic, ou metriques plus sensibles.
2. **Novelty effect** : les nouveaux modeles performent mieux au debut juste parce qu'ils sont differents. Solution : laisser tourner le test 2-4 semaines.
3. **Selection bias** : le split n'est pas vraiment aleatoire (ex: sticky sessions). Solution : split par user_id hash.
4. **Multiple testing** : tu regardes 20 metriques, 1 sort significative. Solution : Bonferroni correction ou definir a priori la metrique cle.

### Metriques

- **Primary metric** (choisie AVANT le test, business)
- **Guardrail metrics** (latence, erreur, cost) : ne doivent pas degrader
- **Secondary metrics** : curiosite, mais jamais pour decider

---

## CI/CD pour les modeles ML

Un modele en prod doit etre **promue automatiquement ou pas du tout**. Processus type :

```
 commit code ──> CI tests ──> train ──> eval offline ──> register ──> staging ──> shadow ──> canary ──> promote
                                                              │
                                                              v
                                                    manual approval gate
                                                        (optional)
```

### Etapes cles

1. **Version the data** : DVC, MLflow, LakeFS. Ton dataset doit etre reproductible.
2. **Run training** : orchestration via Airflow, Prefect, Kubeflow, Metaflow.
3. **Offline eval** : compare aux baselines. Gate : si AUC < 0.02 au-dessus de la baseline, reject.
4. **Register** : MLflow Model Registry, promeut en "staging".
5. **Shadow deploy** : cf J8.
6. **Canary** : 1% -> 10% -> 50% -> 100%, avec rollback automatique si metriques degradent.
7. **Promote to production**, archive de l'ancienne version.

### Feature flags

Pour switcher rapidement entre modeles sans re-deploy :

```python
if feature_flag("use_model_v2"):
    model = load_model("v2")
else:
    model = load_model("v1")
```

Outils : LaunchDarkly, Unleash, ou simple config dans Redis/DynamoDB.

---

## Observability specifique LLM : Langfuse et LangSmith

### Langfuse (open source, self-hostable)

- Tracing natif pour LLM, integrations SDK (Python, TS)
- Scoring (faithfulness, user feedback)
- Cost tracking par modele
- Datasets et evaluations
- Dashboards par project / user / session

### LangSmith (SaaS, par LangChain)

- Meme features
- Plus integre avec LangChain / LangGraph
- Dataset et regression testing natifs
- SaaS, moins de setup

### Patterns courants

1. **Decorateur @observe** : instrumente une fonction, les sous-calls sont automatiquement traces
2. **Capture des metadata** : user_id, session_id, tags pour filtrer
3. **Scoring post-hoc** : apres chaque reponse, un LLM-as-a-judge note la reponse sur faithfulness / relevance
4. **Session grouping** : toutes les requetes d'une meme conversation sont groupees
5. **Prompt management** : versioner les prompts dans l'outil, tester l'impact

---

## Tradeoffs recapitulatifs

| Dimension | Peu d'observability | Full observability | Quand |
|---|---|---|---|
| Tracing | Logs texte | Langfuse / LangSmith | Des le jour 1 en prod |
| Drift | None | PSI mensuel | Des que la distribution compte |
| A/B | Deploy + cross fingers | Feature flags + stats | Des qu'il y a du trafic serieux |
| CI/CD | Manuel | Pipeline automatise | > 1 modele en prod |

---

## Exemples reels

- **Netflix** : metaflow + MLflow, A/B tests systematiques, feature flags natif
- **Uber Michelangelo** : data quality + drift integre au feature store
- **Airbnb** : monitoring drift sur embeddings, alerting auto
- **Anthropic / OpenAI** : tracing propre, evaluations continues sur datasets internes
- **Langfuse** : utilise par des milliers de startups IA en 2026

---

## Flash cards

**Q: Qu'est-ce que le PSI et comment l'interpreter ?**
R: Population Stability Index. < 0.1 = pas de drift. 0.1-0.25 = drift modere. > 0.25 = drift significatif, action requise.

**Q: Difference entre data drift et concept drift ?**
R: Data drift = les inputs changent. Concept drift = la relation input-output change. Le premier peut etre gere par recalibration, le second exige un re-entrainement.

**Q: Pourquoi les logs classiques ne suffisent pas pour un agent LLM ?**
R: Un agent fait une chaine de decisions non-deterministes. Un log ligne-par-ligne perd la structure d'arbre des spans, l'attribution des couts, et le lien parent-enfant.

**Q: Qu'est-ce qu'un novelty effect dans un A/B test ML ?**
R: Les users reagissent plus positivement a un nouveau modele parce qu'il est nouveau, pas parce qu'il est meilleur. Solution : attendre 2-4 semaines.

**Q: Quelles sont les 3-5 metriques a surveiller en permanence sur un LLM en prod ?**
R: Latence (p50/p99), tokens in/out, cost per request, fallback rate, cache hit rate, user feedback score.

---

## Key takeaways

- Les ML systems ont des pannes silencieuses. Sans observability, tu es aveugle jusqu'au ticket user.
- Tracing + metrics + logs = trinite. Pour les LLM/agents, le tracing est le plus critique.
- Drift detection avec PSI / KL est cheap et devrait tourner en continu.
- A/B tests : definir la metrique primary AVANT, attendre 2-4 semaines, surveiller les guardrails.
- CI/CD : ton modele doit etre une unite de deploiement comme du code. Feature flags + canary = standard.
