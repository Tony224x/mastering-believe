# Jour 21 — Evaluations & observability LLM en production

> **Temps estime** : 4h | **Prerequis** : J10 (fine-tuning), J15 (reasoning), J20 (distillation)

---

## 1. Pourquoi ce chapitre est le plus important de la semaine

Un LLM product sans **eval** est un avion sans instruments. Tu peux voler, peut-etre arriver, mais tu ne sauras jamais **pourquoi** les crashes arrivent, ni comment ameliorer. En 2023-2024, beaucoup d'equipes ont brule des millions en iterant sur des prompts **sans metrique objective**. En 2025-2026, les equipes serieuses ont une eval pipeline des le jour 1.

### L'equation qui resume tout

```
Prod quality = model_quality × prompt_quality × context_quality × retry_quality
```

Sans eval, tu optimises un facteur au hasard. Avec eval, tu sais lequel tirer.

---

## 2. Les 5 niveaux d'evaluation que tu dois avoir

### Niveau 0 — Spot checks (debutant)

Tu regardes les outputs a la main quand quelqu'un se plaint. Aucune metrique. A eviter, meme en MVP.

### Niveau 1 — Eval set golden

10-200 paires `(input, expected_output)` representatives. Metriques :
- **Exact match** : trop strict pour du texte libre
- **Semantic similarity** : embedding cosine entre output et expected
- **Rule-based** : assertions metier ("doit contenir X", "ne doit pas contenir Y")

Toujours versionne, isole du training data, grandit au fil des bug reports ("regression tests").

### Niveau 2 — LLM-as-judge

Un LLM (souvent plus gros que celui en prod) juge la qualite sur des criteres structures :
```
Criteres : groundedness (0-5), relevance (0-5), fluency (0-5), safety (0-1)
Prompt   : "Voici la query, la reponse, le contexte. Note selon les criteres..."
```

Les papers 2023-2024 (MT-Bench, AlpacaEval) ont montre que **GPT-4 judge ≈ humain** sur 80-90% des cas pour les taches subjectives. En 2026, utiliser Claude 4.5 Opus ou GPT-5 comme juge est le standard.

**Pieges du LLM-as-judge** :
- **Position bias** : le juge prefere la reponse A si elle est premiere. Solution : inverser A/B et moyenner.
- **Length bias** : le juge prefere les reponses longues. Solution : normaliser par longueur ou prompt "ne juge pas sur la longueur".
- **Self-preference** : un juge prefere les reponses de son propre modele. Solution : utiliser un modele DIFFERENT pour judge.
- **Faithfulness** : un juge peut noter correctement mais citer des raisons fausses. Auditer sur un sample.

### Niveau 3 — Pairwise & arenas

Au lieu de noter en absolu, comparer deux outputs :
> Lequel des deux est meilleur : A ou B ?

Plus fiable que le scoring absolu. **Chatbot Arena** (LMSys, 2023+) et **Scale's SEAL Arena** (2025) sont les arenes humaines de reference. En interne : construire un "mini-arena" avec tes vrais utilisateurs ou un juge LLM en comparaisons pairwise.

Calcul d'Elo ou Bradley-Terry pour classer N modeles/prompts.

### Niveau 4 — Online metrics (production)

L'eval ultime : **l'utilisateur**. Metriques 2026 standard :
- **Thumbs up/down** par message
- **Edit rate** : fraction de reponses editees avant d'etre utilisees
- **Abandon rate** : users qui quittent sans finir
- **Task success rate** (si le produit a une tache claire)
- **Regeneration rate** : combien de fois l'user demande a regenerer
- **Click-through sur citations** (pour les produits RAG)

Correler ces metriques online avec les scores offline eval → validation de ta pipeline.

---

## 3. LLM-as-judge : comment bien faire

### La structure d'un prompt judge solide

```
# Context
You are evaluating the output of an LLM on a support chatbot task.

# Input
User query: {query}
Retrieved context: {context}

# Output to evaluate
{output}

# Criteria
Score each 0-5 :
1. groundedness - chaque affirmation est-elle supportee par le contexte ?
2. relevance - la reponse repond-elle a la question ?
3. completeness - y a-t-il des elements manquants ?
4. tone - le ton est-il professionnel et utile ?

# Format
Output only JSON : {"groundedness": N, "relevance": N, ...}
```

### Calibrer ton judge contre des humains

Avant de deployer un judge en eval regression, **calibrer** :
1. 50 exemples note par 3 humains (expert)
2. Le LLM judge note les memes 50
3. Inter-rater agreement (kappa) entre judge et humains
4. Si kappa > 0.6 → judge utilisable. Si < 0.4 → refaire le prompt ou changer de judge.

### Coût du LLM judge

Un judge consomme ~2-5x les tokens d'une prediction. Pour une eval suite de 500 examples avec 5 criteres chacun, environ 50 000 tokens judge / run. Sur Claude Opus : ~$4 par run. Ok pour les CI reguliers. Trop cher pour chaque PR → Haiku/GPT-5-mini.

---

## 4. L'observability stack 2026

### Le minimum vital pour un produit LLM

```
APPLICATION
    │
    ▼
OTEL trace par requete
    │
    ▼
Store (Langfuse / Braintrust / LangSmith / Helicone / interne)
    │
    ▼
Dashboards + alertes
```

### Par requete, tu dois logger

- Trace ID (lie aux autres services)
- Model name + version
- System prompt hash (pour detecter regressions)
- Input tokens (frais + cached)
- Output tokens
- Latency (TTFT + total)
- Cost ($)
- Rating (si feedback user)
- Tool calls (pour agents)
- Retrieval hits (pour RAG)
- Error / retry count

### Les 4 pannes classiques a surveiller

1. **Cache hit rate drop** : quelqu'un a change le system prompt, break du cache → explosion de cout.
2. **Latency spike** : upstream provider outage, hotspot sur une route.
3. **Hallucination increase** : changement dans les donnees retrievees ou prompt drift. Detecter via sample LLM-as-judge continu.
4. **Cost per query creep** : plus de tool calls par session, prompt grandit avec le temps.

### Alertes utiles

- TTFT p95 > 5s pendant > 5 min
- Cache hit rate < 40% pendant > 10 min
- Error rate > 2%
- Cost per 1000 queries > X (fixe selon budget)
- Judge score mean < seuil (drift detection)

---

## 5. Evaluations pour les agents

Les agents sont plus durs a evaluer car multi-step. Criteres specifiques :

### Success @ task

Pour chaque tache, un oracle dit "done/not done". SWE-Bench, OSWorld, WebArena ont tous cette structure. Construire ton propre oracle pour tes taches produit.

### Trace-level eval

- Nombre de tool calls (moins = mieux, a qualite egale)
- Nombre de redondances (meme tool call repete sans progres)
- Recovery rate : apres une erreur, l'agent recupere-t-il ?

### Cost per task

Budget en dollars ou tokens par tache. Tu veux une distribution, pas une moyenne.

### Benchmark suites 2026

- **SWE-Bench Verified** (code fixes) — Claude 4.5 Opus : ~75%, GPT-5 : ~72%
- **OSWorld** (computer use) — Claude Computer Use 2026 : ~60%, humain ~90%
- **TAC-Bench** (tool use tracing) — nouveau 2025
- **AgentBench** — tasks generales
- **Browse / Web agents** : WebArena, Mind2Web-Live

---

## 6. Piege : eval overfit et Goodhart

> "When a measure becomes a target, it ceases to be a good measure." — Goodhart's law

Ton eval set est un **proxy** pour la qualite reelle. Optimiser trop dessus :
- Les exemples "faciles" disparaissent (modele apprend a les resoudre)
- Les prompts s'adaptent specifiquement a ces cas
- La performance en prod stagne pendant que l'eval monte

### Solutions

1. **Eval set rotatif** : remplacer 10% par mois
2. **Eval set cache** : un set secret visible seulement au CI, jamais au dev
3. **Online metrics** comme verite terrain (edit rate, thumbs down)
4. **Red teaming** : 1 jour/semaine, un ingenieur casse le modele creative, les cas trouves entrent dans l'eval

---

## 7. Comparer deux modeles / deux prompts proprement

### La methode scientifique

1. **Fixer tout sauf le facteur teste** (modele OU prompt, pas les deux)
2. Utiliser le MEME eval set
3. Sampler > 100 exemples (mean stable)
4. Calculer des intervalles de confiance (bootstrap)
5. Si la difference est dans le bruit, ne pas switcher

### Anti-pattern : "ca a l'air mieux"

Voir 5 exemples, trouver que le nouveau prompt est meilleur, shipper. 3 jours plus tard, plainte user → rollback. Cout : une semaine perdue.

### Anti-pattern : swap providers sans bench

"Claude est moins cher, on switch". Sans eval, tu decouvres dans 2 semaines que ton JSON parsing rate a chute de 12% car le modele repond differemment. Toujours re-run l'eval complet apres un swap.

---

## 8. Metriques de RAG specifiques

Pour un systeme RAG, layer d'eval additionnel :

### Retrieval

- **Recall@k** : est-ce que le chunk golden est dans le top-k ?
- **nDCG@k** : ordre correct parmi les top-k ?
- **MRR** : rank moyen du premier chunk pertinent

### Generation groundedness

- **Faithfulness** : chaque fait dans la reponse vient-il du contexte ?
- **Citation accuracy** : les citations sont-elles correctes ?
- **Answer completeness** : couvre tous les aspects de la question ?

### End-to-end

- **Answer correctness** vs golden answer (juge LLM)
- **User satisfaction** (thumbs, edit rate)

Benchmarks 2026 : RAGAS (open-source), TruLens, Arize Phoenix. Tous integrent les trois layers.

---

## 9. Red teaming et safety evals

Un LLM en prod doit resister a :
- **Prompt injection** (via inputs users, documents retrieves, outils)
- **Data exfiltration** via PII leaks
- **Jailbreaks** (convaincre le modele de briser ses regles)
- **Tool misuse** (agents qui executent des actions nuisibles)

Suite d'eval safety :
- **HarmBench** (2024+) : requetes adversaires
- **AgentHarm** (2024) : taches agent malicieuses
- **Custom adversarial set** : 50-200 inputs specifiques a ton produit

Toujours **rouge-teamer a chaque release**. Un modele qui refuse "comment faire une bombe" peut echouer sur ton cas d'usage produit ("envoie un email a tous mes contacts disant qu'ils ont perdu leur job").

---

## 10. Workflow complet d'une amelioration de prompt

```
1. Observation : LLM judge ou user feedback signale un probleme
2. Collecter 10-30 exemples concrets du probleme
3. Hypothese : "le prompt ne specifie pas X"
4. Nouveau prompt, tester sur les 10-30 exemples (dev loop rapide)
5. Si ca marche → run EVAL COMPLET (200+ exemples)
6. Comparer metrique-par-metrique avec baseline
7. Verifier qu'on n'a pas regresse sur d'autres axes
8. Si OK → shadow deploy (A/B sur 5% traffic)
9. Monitor online metrics (edit rate, thumbs) pendant 24-48h
10. Full rollout OU rollback
```

Sans eval : tu coupes toutes les etapes intermediaires et tu pries. Avec eval : tu as un process repetable qui scale a une equipe.

---

## Key takeaways (flashcards)

**Q1** — Quels sont les 5 niveaux d'evaluation d'un LLM produit ?
> (0) Spot checks, (1) Eval set golden + metriques rule-based/semantic, (2) LLM-as-judge, (3) Pairwise/arena, (4) Online metrics (edit rate, thumbs, task success).

**Q2** — Quels sont les 4 biais principaux du LLM-as-judge ?
> Position bias (prefere A si premiere), length bias (prefere long), self-preference (prefere son propre modele), faithfulness (raisonnement post-hoc).

**Q3** — Que mesure l'edit rate et pourquoi c'est utile ?
> Fraction d'outputs edites par l'user avant usage. Mesure de qualite online tres robuste, peu biaisee par le type d'utilisateur.

**Q4** — Qu'est-ce que Goodhart's law dans le contexte eval ?
> "When a measure becomes a target, it ceases to be a good measure". Trop optimiser sur un eval set fait stagner la qualite reelle. Solution : rotation, eval caches, red teaming.

**Q5** — Pourquoi inclure un juge calibre contre des humains ?
> Pour valider que ton LLM judge n'est pas systematiquement biaise. Kappa > 0.6 avec humains = utilisable. En dessous, changer le prompt ou le modele.

**Q6** — Quelles sont les 3 layers d'eval d'un systeme RAG ?
> (1) Retrieval (recall@k, nDCG@k), (2) Generation groundedness (faithfulness, citations), (3) End-to-end correctness (answer vs golden, user satisfaction).
