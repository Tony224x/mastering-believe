# J11 — Evaluation & Testing : comment mesurer si ton agent marche

> **Temps estime** : 3h | **Prerequis** : J1-J10
> **Objectif** : comprendre les differents niveaux d'evaluation d'un agent, maitriser le LLM-as-judge, ecrire des regression tests, et savoir quels benchmarks utiliser.

---

## 1. Pourquoi evaluer un agent est plus dur qu'evaluer un LLM

Evaluer un LLM "pur" (un chatbot qui repond a des questions), c'est deja difficile : qualite subjective, pas de ground truth unique, biais. Mais evaluer un **agent**, c'est un cran plus dur parce qu'un agent fait **plusieurs etapes**.

Un LLM prend une question, produit une reponse. Tu compares la reponse a ce que tu attendais.

Un agent prend une tache, fait 15 etapes (rechercher, appeler des outils, raisonner, choisir, decider), puis produit un output. Si l'output est mauvais, **quelle etape a casse** ?

### Agent evals vs LLM evals — la grande difference

| Aspect | LLM eval | Agent eval |
|--------|----------|------------|
| Input | 1 question | 1 tache (souvent longue) |
| Output | 1 reponse textuelle | 1 reponse + trajectoire multi-etapes |
| Ce qu'on mesure | Qualite de la reponse | Qualite de la reponse **et** qualite du processus |
| Ground truth | Parfois | Souvent impossible (trop de chemins valides) |
| Reproductibilite | Faible (LLM stochastique) | Encore plus faible (stochastique x N etapes) |
| Cout | Modere | Eleve (le test execute l'agent complet) |
| Debug en cas d'echec | "La reponse est fausse" | "L'etape 7 a pris la mauvaise decision parce que l'etape 5 a ecrit une mauvaise info en memoire" |

**Implication** : evaluer un agent demande plusieurs niveaux de mesure.

---

## 2. Les niveaux d'evaluation

### 2.1 Niveau 1 : final answer evaluation

**Principe** : on compare la reponse finale de l'agent a un "ground truth" ou a un critere de qualite.

```
Tache : "Quel est le CA de Kalira en 2025 ?"
Reponse attendue : "820 000 euros"

Agent run →  "Kalira a realise un chiffre d'affaires de 820k euros en 2025."
Comparaison → OK (contient "820", "euros", "2025")
```

**Methodes de comparaison** :

| Methode | Comment | Quand |
|---------|---------|-------|
| Exact match | `output == expected` | Reponses tres structurees (chiffre, date, oui/non) |
| String contains | `expected in output` | Reponses ou un mot cle doit apparaitre |
| Regex | `re.search(pattern, output)` | Patterns attendus |
| Semantic similarity | embedding cosine | Reponses en prose ou le sens compte |
| LLM-as-judge | Un LLM juge la qualite | Reponses longues, nuancees |

**Avantage** : simple, rapide, facile a mettre en place.
**Limite** : si la reponse est fausse, tu ne sais pas pourquoi.

### 2.2 Niveau 2 : trajectory evaluation

**Principe** : on regarde la **trajectoire** (liste des etapes, outils utilises, decisions prises) et on juge si elle est correcte.

```
Tache : "Envoie un email de bienvenue au nouvel utilisateur"

Trajectoire attendue :
  1. lookup_user(email) → recupere les infos
  2. load_template("welcome") → charge le template
  3. render_template(user_info) → personnalise
  4. send_email(to, body) → envoie

Trajectoire reelle :
  1. lookup_user ← OK
  2. send_email ← HORS-ORDRE (n'a pas charge le template)
  3. ...
```

**Ce qu'on evalue** :
- **Etapes necessaires faites ?** (recall)
- **Etapes parasites ?** (precision)
- **Ordre correct ?**
- **Arguments corrects pour chaque outil ?**
- **Detection d'erreurs** : l'agent a-t-il vu et recupere une erreur ?

**Methodes** :
- **Expected tool list** : la trajectoire attendue est une liste de tools. On check `set(actual) == set(expected)`.
- **Expected trace** : on annote la trajectoire attendue pas-a-pas. On compare.
- **LLM-as-judge over traces** : on donne la trajectoire au LLM, on lui demande "cette trajectoire accomplit-elle la tache ? y a-t-il des etapes inutiles ?"

**Avantage** : tu vois **ou** l'agent a casse.
**Limite** : definir une trajectoire "correcte" est souvent ambigu — il y a plusieurs bons chemins.

### 2.3 Niveau 3 : step-wise evaluation

**Principe** : on juge **chaque etape individuellement**. Est-ce que l'agent a fait le bon appel d'outil, avec les bons arguments, a cette etape ?

```
Step 3 context: [user info loaded]
Step 3 action: render_template(template="welcome", user={...})
Step 3 judgment: CORRECT (the template is loaded and ready, rendering is the logical next step)
```

**Avantage** : granularite maximum. On peut pinpointer exactement ou l'agent s'est trompe.
**Limite** : cher a produire (un appel LLM par step) et subjectif.

### 2.4 Niveau 4 : metric-based evaluation

**Principe** : on mesure des metriques quantitatives sur l'execution.

- **Latence** : combien de temps l'agent a mis
- **Cout** : combien de tokens / dollars
- **Nombre d'appels LLM** : doit etre < N
- **Nombre d'appels d'outils** : doit etre < N
- **Taux d'echec** : % de runs qui echouent avec une erreur
- **Taux de hallucination** : % de runs ou l'agent invente

Ces metriques s'ajoutent aux autres levels.

### 2.5 En pratique : une eval multi-niveaux

```python
eval_result = {
    "final_answer_score": 0.9,         # LLM-as-judge on output
    "trajectory_valid": True,           # Expected tools were called
    "extra_tools_called": ["slack_dm"], # parasites
    "latency_ms": 12_450,
    "total_tokens": 8_200,
    "cost_usd": 0.032,
    "errors": [],
    "verdict": "PASS",
}
```

Chaque niveau a son role. On n'est pas oblige de tous les utiliser — on choisit selon les contraintes.

---

## 3. LLM-as-judge — le cheval de trait de l'eval moderne

### 3.1 Le principe

On utilise un LLM pour juger la qualite d'une reponse (ou d'une trajectoire). On lui donne :
- Le contexte de la tache
- La reponse attendue (ou les criteres de qualite)
- La reponse reelle de l'agent

Et on lui demande un score + justification.

```
System : You are a strict evaluator.
User : Task: "Find the revenue of Kalira in 2025"
       Expected: The answer should mention a number close to 820k euros.
       Actual : "Kalira made about 820k euros in revenue in 2025."

       Rate the answer from 1 to 5 and justify.

Response : 5 -- The answer is correct and precise. It mentions 820k euros
           which matches the expected value, includes the year, and uses
           the correct currency.
```

### 3.2 Pourquoi ca marche

- Les LLMs modernes (GPT-5.4, Claude 4.6) sont **tres bons** en jugement qualitatif
- Beaucoup plus scalable que du label humain
- Peut evaluer des choses subjectives (clarte, ton, completude) qu'un regex ne peut pas

### 3.3 Pieges classiques

| Piege | Symptome | Defense |
|-------|----------|---------|
| **Biais positif** | Le juge est trop permissif, donne 5/5 trop souvent | Prompt strict, "be harsh", demander > 3 criteres de verification |
| **Biais de longueur** | Le juge prefere les reponses longues | Penaliser explicitement la verbosite |
| **Biais d'auto-preference** | Le juge prefere les reponses d'un LLM de la meme famille | Utiliser un juge d'une autre famille |
| **Incoherence** | Score different a chaque run | Baisser la temperature a 0, ou moyenne de N runs |
| **Gaming du prompt** | Le LLM evalue apprend a donner 5 meme sur des reponses vides | Varier les prompts, pieger avec des mauvaises reponses connues |

### 3.4 Template de prompt LLM-as-judge

Un template robuste contient :

```
[SYSTEM]
You are a strict evaluator. You give a score from 1 to 5.
- 1 = completely wrong, hallucinated, unhelpful
- 2 = mostly wrong, missing key elements
- 3 = partially correct, some issues
- 4 = correct with minor issues
- 5 = correct, precise, complete

Before giving a score, verify:
1. Does the answer directly address the task?
2. Is every fact mentioned accurate?
3. Are there hallucinations or inventions?
4. Is the level of detail appropriate?

[USER]
Task: {task}
Expected answer criteria: {criteria}
Actual answer: {actual}

Output a JSON object: {"score": <1-5>, "reasoning": "<your reasoning>"}
```

**Pourquoi les 4 criteres de verification** : force le LLM a penser avant de noter. Sans ca, il tend vers une note moyenne.

### 3.5 Variantes avancees

- **Pairwise comparison** : au lieu d'un score absolu, demander "lequel est meilleur, A ou B ?". Plus fiable car relative.
- **Multi-judge averaging** : plusieurs juges (meme prompt) → moyenne. Reduit la variance.
- **Rubric-based** : un rubrique detaille avec criteres ponderes (accuracy 0.5, clarity 0.3, concision 0.2).
- **Structured output** : forcer un JSON schema (Pydantic) pour parser le verdict sans ambiguite.

---

## 4. Regression testing pour agents

### 4.1 Pourquoi

Tu modifies un prompt, un outil, un model. Comment savoir que tu n'as pas casse quelque chose qui marchait ?

**Regression testing** : un ensemble fixe de cas de test que l'agent doit toujours reussir. Tu les lances a chaque modification.

### 4.2 Construire un dataset de regression

Un bon dataset de regression contient :
- **Cas simples** : tache facile, agent doit reussir a 100%
- **Cas moyens** : tache normale, agent doit reussir > 80%
- **Cas difficiles** : tache limite, agent doit reussir > 50%
- **Cas pieges** : questions ambigues, tool errors attendues, prompts injections connus

**Taille** : commence petit (20-50 cas), grandis au fur et a mesure. Tu n'as pas besoin de 1000 cas pour commencer.

### 4.3 Structure d'un cas de test

```python
@dataclass
class AgentTestCase:
    id: str                       # unique identifier
    task: str                     # the input task
    expected_answer_criteria: str # what the answer should contain
    expected_tools: list[str]     # tools the agent should call
    forbidden_tools: list[str]    # tools the agent should NOT call
    max_llm_calls: int            # budget
    max_latency_ms: int           # SLA
    tags: list[str]               # ["easy", "prompt_injection", "rag"]
```

### 4.4 Comparaison avec un baseline

Le point cle du regression testing : **comparaison**. On ne dit pas "mon agent a 85% de reussite". On dit "mon agent est passe de 85% a 82% apres la modif — il y a une regression sur 3 cas."

```python
baseline_results = load("baseline.json")  # Results of the previous run
current_results = run_all_tests(agent)

regressions = []
for case_id in baseline_results:
    if baseline_results[case_id] == "PASS" and current_results[case_id] == "FAIL":
        regressions.append(case_id)

if regressions:
    print(f"REGRESSION: {len(regressions)} cases broke: {regressions}")
```

**Principe** : tant qu'on ameliore, on accepte. Des qu'on regresse, on bloque (ou au moins on alerte).

---

## 5. Benchmarks publics pour agents

### 5.1 AgentBench

**URL** : https://github.com/THUDM/AgentBench
**Ce que c'est** : une collection de taches dans 8 environnements (OS, DB, code, jeux, lateral thinking, etc.).
**Ce que ca mesure** : capacite generale d'un agent a raisonner et agir dans des environnements varies.
**Quand l'utiliser** : pour comparer differents LLMs comme backbone d'agent.

### 5.2 GAIA

**Nom** : General AI Assistants
**Ce que c'est** : 466 questions complexes necessitant plusieurs outils (web, code, lecture de fichiers) pour etre resolues. Simple pour un humain, dur pour un agent.
**Ce que ca mesure** : capacite d'un agent a raisonner et utiliser des outils pour repondre a des questions pratiques.
**Score reference** : GPT-4 atteint ~30%, un humain bien informe > 90%. Beaucoup de marge.

### 5.3 SWE-bench

**URL** : https://www.swebench.com/
**Ce que c'est** : des vrais bugs sur des vrais repos open-source Python. L'agent doit proposer un patch qui passe les tests.
**Ce que ca mesure** : capacite d'un agent de codage a resoudre des bugs reels de production.
**Score reference** : les meilleurs agents (Claude, Devin) sont autour de 50-60% sur SWE-bench Verified.

### 5.4 TAU-bench (tool use agent)

**URL** : https://github.com/sierra-research/tau-bench
**Ce que c'est** : simulation d'un customer service. L'agent doit utiliser des outils (API) pour resoudre des demandes client. Il y a un simulateur du client qui peut poser des questions de suivi.
**Ce que ca mesure** : qualite d'un agent dans un workflow tool-heavy avec interaction multi-tour.

### 5.5 WebArena

**URL** : https://webarena.dev
**Ce que c'est** : un environnement avec 4 sites web realistes (e-commerce, forum, CMS, etc.) ou l'agent doit completer des taches.
**Ce que ca mesure** : capacite d'un agent a naviguer et interagir avec des sites web.

### 5.6 Faut-il utiliser ces benchmarks ?

**Oui si** : tu construis un framework / agent generique et tu veux comparer des alternatives.
**Non si** : tu construis un agent specialise (ex: assistant juridique pour un cabinet) — ils ne mesurent pas ce qui t'interesse. Construis ton propre dataset.

---

## 6. Pipeline d'eval complet

```
┌───────────────────────────────────────────────────┐
│  1. DEV : developpeur ecrit des tests unitaires   │
│      (agent.test.py -- rapide, deterministe)      │
└─────────────────┬─────────────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────────────┐
│  2. CI : dataset de regression roulé a chaque PR  │
│      (20-50 cas, ~5 min, bloque la merge si fail) │
└─────────────────┬─────────────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────────────┐
│  3. NIGHTLY : eval etendue (200-500 cas)          │
│      (rapport de qualite, alerte si regression)   │
└─────────────────┬─────────────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────────────┐
│  4. PROD : monitoring continu (tracing + scoring) │
│      (evaluation online sur les vraies queries)   │
└───────────────────────────────────────────────────┘
```

**Point cle** : les niveaux 1-2 sont **deterministes ou pseudo-deterministes**. Le niveau 4 est **en live** sur le traffic reel — on utilise LLM-as-judge pour scorer les vraies queries.

---

## 7. Outils pour faire de l'eval

| Outil | Ce qu'il fait | Quand l'utiliser |
|-------|---------------|-----------------|
| **Langfuse** | Tracing + scoring + dataset eval | Open-source, self-hostable |
| **LangSmith** | Tracing + eval + dataset management | Si tu es deja LangChain |
| **Braintrust** | Eval framework avec UI | Focus sur l'eval plutot que le tracing |
| **OpenAI Evals** | Framework d'eval open-source | Si tu utilises OpenAI |
| **promptfoo** | Eval CLI avec YAML configs | Pour l'equipe devops |
| **pytest + assertions** | Tests classiques | Pour les cas simples |

**Recommandation pragmatique** : commence par du **pytest + LLM-as-judge fait maison**. Ajoute Langfuse quand tu passes en prod pour le tracing et les datasets.

### 7.1 Frameworks eval 2025 — la nouvelle generation

Plusieurs frameworks dedies a l'evaluation LLM/agent ont emerge en 2024-2025 et complementent (ou remplacent) le pytest maison pour les cas specifiques :

- **Ragas** — specialise **RAG eval**. Metriques natives : `faithfulness` (la reponse est-elle supportee par les docs retrieves ?), `answer_relevance` (la reponse repond-elle a la question ?), `context_precision` / `context_recall` (les bons chunks ont-ils ete retrieves ?). Pythonic, integration facile avec LangChain/LlamaIndex. A utiliser **des que tu as un RAG** en prod.

- **DeepEval** — framework general, pytest-like. Modele conceptuel proche de unittest : tu ecris des `LLMTestCase` avec des metriques (LLM-as-judge, hallucination, toxicity, summarization). Fournit des templates de judge pretes a l'emploi. Bon pour ceux qui veulent une DX pytest familiere.

- **Prometheus** (KAIST) — un LLM **open-source fine-tune specifiquement pour l'evaluation**. Au lieu d'utiliser GPT-4 comme judge (cher, biais), tu utilises Prometheus 2 (13B parametres). Avantages : cheap, reproductible (pas de drift du provider), on peut le self-host. A utiliser en CI/CD pour les evals a haut volume.

- **LangSmith / Langfuse / Arize Phoenix** — au-dela du tracing, ces plateformes offrent de l'**eval en production** avec datasets versionnes, comparisons baselines, dashboards. C'est la ou on fait le continuous evaluation sur les vraies queries prod.

**Table de decision rapide** :

| Tu veux... | Utilise |
|------------|---------|
| Eval de RAG (faithfulness, context precision) | **Ragas** |
| Pytest-like pour LLM/agent | **DeepEval** |
| Judge cheap, reproductible, self-hosted | **Prometheus** |
| Eval en prod avec datasets + baselines | **LangSmith / Langfuse / Phoenix** |
| Rien de specifique, tests simples | **pytest + LLM-as-judge maison** |

**Retour d'experience** : chez Kalira, on utilise **Ragas** pour tous les RAG agentiques (les metriques natives couvrent 90% des besoins), **DeepEval** pour les agents generaux, et **Phoenix** pour le monitoring continuous en prod. Le pytest maison est garde uniquement pour les smoke tests ultra-rapides en dev.

---

## 8. Flash Cards — Test de comprehension

**Q1 : Quelle est la difference fondamentale entre evaluer un LLM et evaluer un agent ?**
> R : Un LLM produit une reponse a partir d'une question — on evalue la reponse. Un agent produit une reponse **et** une trajectoire multi-etapes — on doit evaluer a la fois l'output final ET le processus (ordre des etapes, outils appeles, arguments, erreurs gerees). En cas d'echec, l'agent necessite souvent de tracer ou le processus a casse, pas juste de dire "la reponse est fausse".

**Q2 : Quels sont les 4 niveaux d'evaluation d'un agent ?**
> R : (1) **Final answer** : comparer la reponse finale (exact match, regex, LLM-as-judge). (2) **Trajectory** : verifier que la trajectoire contient les bonnes etapes dans le bon ordre avec les bons arguments. (3) **Step-wise** : juger chaque etape individuellement. (4) **Metric-based** : latence, cout, tokens, nombre d'appels LLM, taux d'echec.

**Q3 : Qu'est-ce que le LLM-as-judge et quels sont ses principaux pieges ?**
> R : Utiliser un LLM pour juger la qualite d'une reponse (ou d'une trajectoire) avec un prompt d'evaluation. Pieges : biais positif (trop permissif), biais de longueur (prefere les reponses longues), biais d'auto-preference (prefere les reponses de sa propre famille de modeles), incoherence (score variable), gaming du prompt. Defense : prompt strict, temperature 0, juges de families differentes, pairwise comparison, multi-judge averaging.

**Q4 : Quel est le but du regression testing pour un agent et comment on le structure ?**
> R : S'assurer qu'une modification (prompt, outil, model) n'a pas casse des cas qui marchaient. On maintient un dataset fixe de 20-500 cas (simples, moyens, difficiles, pieges), on compare les resultats de la version actuelle a une baseline, et on alerte (ou bloque) si un cas passe de PASS a FAIL. Le focus est la **comparaison relative**, pas la note absolue.

**Q5 : Pourquoi les benchmarks publics (GAIA, SWE-bench, AgentBench) sont utiles pour certains projets et inutiles pour d'autres ?**
> R : Utiles quand on construit un agent generique (framework, comparaison de backbones) — ils fournissent des baselines publiques et des cas standardises. Inutiles quand on construit un agent specialise (ex: assistant juridique pour un cabinet) — ils ne mesurent pas les competences domain-specific. Dans ce cas il faut construire son propre dataset d'evaluation aligne avec les cas d'usage reels.

---

## Points cles a retenir

- Evaluer un agent est plus dur qu'evaluer un LLM parce qu'il faut evaluer **le processus ET l'output**
- 4 niveaux d'eval : final answer, trajectory, step-wise, metric-based — on les combine selon les contraintes
- **LLM-as-judge** est le cheval de trait moderne : scalable, qualitatif, bon sur les taches nuancees
- Pieges LLM-as-judge : biais positif, biais longueur, auto-preference, incoherence — defense via prompts stricts et temperature 0
- **Regression testing** : dataset fixe, comparaison relative a une baseline, bloque les modifs qui cassent
- Benchmarks publics (GAIA, SWE-bench, AgentBench, TAU-bench, WebArena) : utiles pour les agents generiques, peu pertinents pour les agents specialises
- Pipeline d'eval type : tests unitaires (dev) -> regression (CI) -> eval etendue (nightly) -> monitoring (prod)
- Outils : pytest + LLM-as-judge pour commencer, Langfuse/LangSmith/Braintrust pour la prod
- Commence petit : 20-50 cas de test suffisent pour attraper 80% des regressions
