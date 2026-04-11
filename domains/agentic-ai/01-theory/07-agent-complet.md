# J7 — Build : un agent de recherche+analyse complet (capstone semaine 1)

> **Temps estime** : 4h | **Prerequis** : J1 a J6 (tout ce qu'on a vu)
> **Objectif** : assembler tout ce qu'on a appris en un agent production-credible. On construit pas a pas un agent de recherche qui prend une question complexe, planifie, execute des outils, maintient une memoire, gere les erreurs, et synthetise une reponse finale.

---

## 1. Le cahier des charges

On veut construire un agent qui repond a ce genre de questions :

> "Quelle est la superficie totale du continent africain et combien d'habitants y vivent, et quel en est le ratio ?"

L'agent doit :
1. **Decomposer** la question en sous-questions
2. **Chercher** les infos via des outils (web search, doc reader, etc.)
3. **Memoriser** ce qu'il apprend pour ne pas re-chercher
4. **Analyser** / resumer les resultats bruts
5. **Synthetiser** la reponse finale
6. **Gerer les erreurs** (tool qui echoue, info manquante)
7. **Demander a l'utilisateur** si vraiment coincé

C'est un **agent de recherche multi-etapes**. C'est 70% des agents qu'on construit en vrai.

---

## 2. Architecture de haut niveau

```
┌──────────────────────────────────────────────────────────────┐
│                       RESEARCH AGENT                          │
│                                                                │
│   START → planner → executor → analyzer → synthesizer → END  │
│              ^          │                                      │
│              │          │                                      │
│              └──────────┘                                      │
│          (replan if stuck)                                     │
│                                                                │
│   State:                                                       │
│   - question: str                                              │
│   - plan: list[str]          (the planner's steps)             │
│   - short_term_memory: dict  (scratchpad for current task)     │
│   - long_term_knowledge: list (persistent facts)               │
│   - findings: list           (raw tool outputs)                │
│   - final_answer: str                                          │
│                                                                │
│   Tools:                                                       │
│   - mock_web_search(query)                                     │
│   - read_doc(doc_name)                                         │
│   - summarize(text)                                            │
└──────────────────────────────────────────────────────────────┘
```

Les nodes correspondent directement aux patterns de J4 (plan-and-execute) et J5 (LangGraph).

---

## 3. Etape 1 — Les outils

On se donne 3 outils minimalistes. Chacun simule un vrai outil qu'on brancherait en prod (Serper, Tavily, un PDF reader, un LLM summarizer).

### 3.1 `mock_web_search(query) -> str`

Un moteur de recherche fake qui a un petit index pre-rempli. Il retourne des "snippets" quand la query matche un keyword.

```python
SEARCH_INDEX = {
    "africa area": "The African continent covers approximately 30.37 million km2.",
    "africa population": "Africa has ~1.46 billion inhabitants as of 2024.",
    "africa density": "Average density in Africa is ~48 inhabitants per km2.",
}
```

**Ce qu'on simule** : un vrai search engine retourne des snippets bruyants, souvent 3-5 resultats, certains hors sujet. Notre mock est plus propre mais le principe est le meme.

### 3.2 `read_doc(doc_name) -> str`

Un "PDF reader" fake qui retourne le contenu d'un doc nomme.

```python
DOCS = {
    "africa_report_2024.pdf": "FULL TEXT: The African continent ... ~1.46B habitants ...",
    "paris_stats.pdf": "FULL TEXT: Paris has 2.16M inhabitants in 105 km2.",
}
```

**Ce qu'on simule** : en prod, on brancherait PyPDF ou unstructured. Meme API : doc_name → text.

### 3.3 `summarize(text) -> str`

Un resumeur qui prend un long texte et en extrait le(s) chiffre(s) cle(s).

```python
def summarize(text: str) -> str:
    # Extract the first numeric value with unit
    m = re.search(r"(\d[\d,.]*)\s*(km2|billion|inhabitants|%)", text)
    if m:
        return f"Key fact: {m.group(0)}"
    return f"No key numeric fact found in: {text[:50]}..."
```

**Ce qu'on simule** : en prod, on ferait un appel LLM avec un prompt de resume. Ici, regex pour rester deterministe.

---

## 4. Etape 2 — La memoire

L'agent a **deux memoires** :

### 4.1 Short-term memory (scratchpad)

Un dict qui vit le temps d'une execution. Il contient les resultats intermediaires :

```python
short_term = {
    "area_km2": 30370000,
    "population": 1460000000,
    "density_computed": 48.07,
    "sources_used": ["mock_web_search", "read_doc"],
}
```

Le planner et l'executor y ecrivent. Le synthesizer y lit pour construire la reponse.

### 4.2 Long-term knowledge store

Une liste de faits (dicts) qui persistent entre les executions. L'agent y va chercher avant de lancer un nouveau tool call :

```python
long_term = [
    {"fact": "Africa has ~30.37M km2", "source": "web_search", "confidence": 0.9},
    {"fact": "Africa has ~1.46B inhabitants", "source": "doc", "confidence": 0.95},
]
```

Avant chaque tool call, on verifie : **est-ce que le fait existe deja dans la long-term memory ?** Si oui, on skip le tool call (cache hit). C'est l'optimisation la plus importante d'un agent multi-query.

---

## 5. Etape 3 — Le planner

Le planner prend la question et produit un **plan** en etapes.

```
Question: "What is Africa's area, population, and density ratio?"

Plan:
  1. Search for 'africa area km2'
  2. Search for 'africa population'
  3. Compute density = population / area
  4. Format the answer
```

### 5.1 Strategie de planning

Deux options :

**Option A : planning statique** — le planner produit tout le plan en une passe, l'executor suit a la lettre.
- (+) Simple, rapide
- (-) Aveugle aux resultats intermediaires
- Bon quand : la tache est bien structuree

**Option B : planning dynamique (replan)** — apres chaque etape, le planner regarde les resultats et decide de replaner ou continuer.
- (+) Flexible, peut s'adapter
- (-) Plus cher en LLM calls
- Bon quand : la tache a des incertitudes

Pour notre agent, on commence **statique**, avec un fallback vers replan si l'executor echoue.

### 5.2 Exemple de planner simple

```python
def planner_node(state: AgentState) -> dict:
    """Decompose the question into a list of steps."""
    question = state["question"]
    # In a real agent, this is an LLM call. Here we hardcode for determinism.
    plan = decompose(question)
    return {"plan": plan}
```

---

## 6. Etape 4 — L'executor

L'executor prend un step du plan et l'execute. C'est la ou on branche les outils, la memoire, et le error handling.

### 6.1 Pseudocode

```
def executor_node(state):
    next_step = pick_next_step(state.plan, state.completed_steps)

    # 1. Check long-term memory first (cache)
    cached = search_long_term(next_step, state.long_term_knowledge)
    if cached:
        update_short_term(state, cached)
        return

    # 2. Decide which tool to use
    tool = select_tool(next_step)

    # 3. Call the tool, handle errors
    try:
        result = tool(args)
    except ToolError as e:
        return handle_error(e, state)

    # 4. Update memories
    update_short_term(state, result)
    if is_durable_fact(result):
        store_long_term(state, result)

    return state
```

### 6.2 Gestion d'erreurs

Un bon executor ne crashe pas sur une erreur d'outil. Il :
1. **Retry** avec backoff (2-3 fois max)
2. **Fallback** : essayer un autre outil qui peut repondre a la meme question
3. **Escalate** : demander au user si vraiment coincé
4. **Skip** : marquer l'etape comme "failed" et continuer si possible

---

## 7. Etape 5 — L'analyzer

L'analyzer regarde les findings bruts et extrait les **faits cles**. C'est le pont entre "donnees" et "reponse".

```python
def analyzer_node(state):
    findings = state["findings"]
    # Extract numeric values, entities, key phrases
    for finding in findings:
        fact = extract_fact(finding)
        if fact:
            state.short_term_memory[fact.name] = fact.value
    return state
```

**Pourquoi une etape separee ?** Parce que les findings bruts sont bruyants. Un tool peut retourner 500 mots alors qu'on veut un seul chiffre. L'analyzer fait le **tri**.

---

## 8. Etape 6 — Le synthesizer

Le dernier node. Il prend tous les faits collectes dans la short-term memory et construit la **reponse finale**.

```python
def synthesizer_node(state):
    facts = state.short_term_memory
    answer = format_answer(state.question, facts)
    return {"final_answer": answer}
```

**Best practice** : le synthesizer doit **citer ses sources**. Quand c'est possible, il indique d'ou vient chaque fait ("selon le web search... selon le doc...") pour que l'utilisateur puisse verifier.

---

## 9. Etape 7 — Le "when to ask the user"

Un bon agent sait demander de l'aide. **Quand demander ?**

| Situation | Agir ou demander ? |
|-----------|-------------------|
| Info manquante apres 3 tools | Demander |
| Plusieurs interpretations possibles | Demander |
| Action destructive (delete, send) | Demander (toujours) |
| Info evidente / bien connue | Ne pas demander |
| Petite ambiguite (on peut choisir) | Ne pas demander, annoncer la decision |

**Pattern** : injecter un node `ask_user` dans le graph, declenche par un conditional edge quand l'executor signale un blocage.

```python
def should_ask_user(state) -> str:
    if state.get("stuck"):
        return "ask_user"
    if state.get("plan_complete"):
        return "synthesizer"
    return "executor"
```

En prod avec LangGraph, on combine ca avec les **interrupts** vus a J6 : l'agent fait pause, on reprend apres l'input humain.

---

## 10. Le flow complet

```
User question: "Quelle est la densite de population de l'Afrique ?"

[START]
  ↓
[PLANNER]
  → plan = [
      "1. Find Africa's total area in km2",
      "2. Find Africa's population",
      "3. Compute density = population / area",
      "4. Format the answer",
    ]
  ↓
[EXECUTOR] step 1
  → check long_term memory: empty
  → tool = mock_web_search("africa area km2")
  → result = "Africa: 30.37M km2"
  → short_term.area_km2 = 30370000
  → long_term.append({"fact": "Africa area ~30.37M km2", ...})
  ↓
[EXECUTOR] step 2
  → check long_term: empty for population
  → tool = mock_web_search("africa population")
  → result = "Africa: 1.46B inhabitants"
  → short_term.population = 1460000000
  → long_term.append({...})
  ↓
[EXECUTOR] step 3 (computation)
  → density = 1460000000 / 30370000 = 48.07
  → short_term.density = 48.07
  ↓
[ANALYZER]
  → extracts facts, verifies plausibility
  ↓
[SYNTHESIZER]
  → "The population density of Africa is approximately 48 inhabitants/km2
     (1.46B habitants / 30.37M km2)."
  ↓
[END]

Final output:
"The population density of Africa is approximately 48 inhabitants/km2,
 calculated from a total area of 30.37 million km2 and a population of
 approximately 1.46 billion inhabitants."
```

---

## 11. Ce que l'agent fait PAS (volontairement)

Cet agent est simple pour etre pedagogique. En prod on ajouterait :

| Feature | Pourquoi pas ici | Comment l'ajouter |
|---------|------------------|-------------------|
| Vraie LLM call | Zero API key requise | Remplacer MockLLM par make_llm() de J4 |
| Vector store | Surcharge de complexite | Brancher Chroma sur long_term_knowledge |
| Checkpointer | Stub deja ~300 lignes | Utiliser SqliteSaver de LangGraph |
| Parallel tools | Non necessaire pour 3 steps | Fan-out via Send API (J6) |
| Streaming UI | Demo CLI | Ajouter `app.stream()` et flush sur stdout |
| Retry avec backoff | Pour rester court | Wrapper les tools avec tenacity |

> **Opinion** : quand tu construis un agent en vrai, **commence toujours minimaliste**, fais tourner, puis ajoute les features une par une en verifiant chaque fois que rien ne casse. 90% des devs tentent de tout implementer d'un coup et se retrouvent avec un agent qui ne tourne pas.

---

## 12. Points cles — ce qu'on a assemble

L'agent de J7 combine **tout** le contenu de la semaine 1 :

| Concept de la semaine | Utilise dans J7 |
|-----------------------|----------------|
| J1 — Boucle agent (perceive-decide-act-observe) | Le flow global |
| J2 — Tool use | mock_web_search, read_doc, summarize |
| J3 — Memory & state | short_term + long_term memory |
| J4 — Planning & reasoning | planner + executor + synthesizer |
| J5 — LangGraph fondamentaux | StateGraph + nodes + edges |
| J6 — LangGraph avance | subgraphs (analyzer), streaming (demo) |

Si tu comprends cet agent de bout en bout, tu comprends 80% des agents en production.

---

## 13. Flash Cards — Test de comprehension

**Q1 : Pourquoi separer short-term et long-term memory dans un agent de recherche ?**
> R : **Short-term** contient les variables intermediaires de la tache courante (area, population, density) qui ne servent plus apres. **Long-term** contient les faits durables (Africa has 30.37M km2) qui peuvent etre reutilises dans d'autres questions. Sans cette separation, l'agent perd tout entre executions (pas de cache) ou accumule du bruit indefiniment.

**Q2 : Quel est le role du node `analyzer` entre `executor` et `synthesizer` ?**
> R : L'analyzer **extrait les faits cles** des findings bruts. Les outils retournent souvent du texte bruyant (500 mots, chiffres noyes). L'analyzer tri, parse les valeurs numeriques, verifie la plausibilite, et met a jour la short-term memory avec les facts propres. Ca evite de pousser du bruit dans le synthesizer.

**Q3 : Quelles sont les 4 strategies d'error handling dans l'executor ?**
> R : (1) **Retry** avec backoff (2-3 tentatives si l'erreur est temporaire). (2) **Fallback** : essayer un autre outil qui peut repondre a la meme question. (3) **Escalate** : demander l'aide de l'utilisateur si vraiment bloque. (4) **Skip** : marquer l'etape comme failed et continuer si le plan peut survivre sans.

**Q4 : Quand l'agent doit-il demander l'aide de l'utilisateur et quand doit-il decider seul ?**
> R : **Demander** : info critique manquante apres plusieurs tools, plusieurs interpretations non-distinguables, action destructive (delete/send). **Decider seul** : info evidente ou bien connue, petite ambiguite sur un detail (annoncer la decision prise). La regle d'or : demander quand l'erreur coute cher et qu'on n'a pas assez d'info pour trancher.

**Q5 : Quel est l'avantage concret de verifier la long-term memory avant chaque tool call ?**
> R : Eviter les **appels redondants** aux outils. Si l'agent a deja appris "Africa has 1.46B inhabitants" lors d'une execution precedente, il peut skip l'appel au web search. Economie de temps, d'argent, et amelioration de la fiabilite (les chiffres ne changent pas entre les executions). Dans un agent multi-query, ca peut diviser le cout par 10.

---

## Points cles a retenir

- Un agent de recherche typique : planner → executor → analyzer → synthesizer
- Deux memoires : short-term (scratchpad de la tache) + long-term (facts durables)
- Toujours **consulter la memoire avant d'appeler un outil** (cache-first)
- Error handling : retry, fallback, escalate, skip — dans cet ordre
- L'analyzer est le pont entre findings bruts et facts propres
- Le synthesizer cite ses sources quand c'est possible
- Demander a l'user : info critique manquante, action destructive, ambiguite grave
- Commencer minimaliste, ajouter les features une par une (surtout pas "big bang")
- Cet agent combine tout le contenu de la semaine 1 — si tu le comprends, tu es pret pour la semaine 2
