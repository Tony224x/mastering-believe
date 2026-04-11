# J1 — Anatomie d'un Agent IA

> **Temps estime** : 3h | **Prerequis** : appels API LLM, Python async basique
> **Objectif** : comprendre ce qu'est reellement un agent IA, quand l'utiliser, et maitriser la boucle ReAct from scratch.

---

## 1. Qu'est-ce qu'un agent IA ? Definition precise

Un agent IA n'est **pas** un chatbot avec des plugins. C'est un systeme qui :

1. **Recoit un objectif** (pas juste une question)
2. **Decide de maniere autonome** quelles actions effectuer
3. **Execute ces actions** dans un environnement (APIs, fichiers, bases de donnees...)
4. **Observe les resultats** et adapte sa strategie
5. **Itere** jusqu'a ce que l'objectif soit atteint ou qu'il abandonne

> **Definition operationnelle** : Un agent est un LLM qui controle son propre flux d'execution. Il choisit quand appeler des outils, lesquels, dans quel ordre, et quand s'arreter.

La difference fondamentale avec un pipeline classique : **le LLM est dans la boucle de controle**, pas juste un noeud dans un DAG predetermine.

```
Pipeline classique :  Input → Etape 1 → Etape 2 → Etape 3 → Output  (chemin fixe)
Agent :               Input → LLM decide → Action → Observation → LLM decide → ... → Output  (chemin dynamique)
```

---

## 2. Agent vs Chatbot vs Pipeline — les 3 niveaux

| Critere | Pipeline | Chatbot | Agent |
|---------|----------|---------|-------|
| **Flux de controle** | Predetermine (DAG fixe) | Tour par tour, humain guide | Autonome, LLM decide |
| **Outils** | Codes en dur dans le pipeline | Aucun ou pre-routes | Selection dynamique |
| **Boucle** | Lineaire, une passe | Request-response | Iteratif jusqu'a completion |
| **Memoire** | Pas de memoire inter-etapes | Historique de conversation | Working memory + long-term |
| **Quand utiliser** | Tache deterministe, toujours le meme chemin | Q&A, assistance conversationnelle | Tache ouverte, multi-etapes, decouverte |
| **Cout** | Bas, predictible | Moyen | Eleve, variable |
| **Risque** | Quasi nul | Faible | Hallucination, boucles infinies, tool misuse |
| **Exemple** | ETL, summarization pipeline | ChatGPT vanilla | Devin, Claude Code, AutoGPT |

### Lecon pratique

La plupart des cas d'usage en entreprise **ne necessitent pas** un agent. Un bon pipeline RAG avec du prompt engineering solide couvre 80% des besoins. L'agent intervient quand :
- Le nombre d'etapes est **inconnu a l'avance**
- Les outils necessaires **dependent des resultats intermediaires**
- La tache requiert de la **decouverte** (chercher, essayer, corriger)

---

## 3. La boucle fondamentale : Perceive → Think → Act

Tout agent, qu'il soit un robot physique ou un systeme LLM, suit la meme boucle :

```
┌─────────────────────────────────────┐
│                                     │
│   ┌──────────┐                      │
│   │ PERCEIVE │ ← Observations,      │
│   └────┬─────┘   resultats outils,  │
│        │         contexte            │
│        ▼                             │
│   ┌──────────┐                      │
│   │  THINK   │ ← Raisonnement,      │
│   └────┬─────┘   planification       │
│        │                             │
│        ▼                             │
│   ┌──────────┐                      │
│   │   ACT    │ → Appel d'outil,      │
│   └────┬─────┘   reponse finale      │
│        │                             │
│        └──────── boucle ─────────────┘
```

**Pourquoi c'est la cle de tout** : chaque framework d'agents (LangGraph, CrewAI, AutoGen, Anthropic Agent SDK) est une implementation de cette boucle. Si tu comprends cette boucle, tu peux :
- Debugger n'importe quel agent en identifiant **ou** dans la boucle ca deraille
- Construire un agent from scratch sans framework
- Migrer un agent d'un framework a un autre

### Les 3 points de defaillance

| Phase | Symptome | Cause typique |
|-------|----------|---------------|
| Perceive | L'agent ignore une observation importante | Trop de contexte, info noyee dans le bruit |
| Think | L'agent prend une mauvaise decision | Prompt faible, manque de few-shot, modele trop petit |
| Act | L'agent appelle le mauvais outil ou avec les mauvais params | Schema d'outil ambigu, descriptions floues |

---

## 4. Le pattern ReAct : Reasoning + Acting

**ReAct** (Yao et al., 2022) est le pattern fondateur des agents LLM modernes. L'idee est simple mais puissante : forcer le LLM a **raisonner avant d'agir**, et a **raisonner sur les observations** apres chaque action.

### Format d'une trace ReAct

```
Question: Quel est le PIB par habitant de la France en 2024, et comment se compare-t-il a l'Allemagne ?

Thought 1: Je dois trouver le PIB par habitant de la France en 2024. Je vais chercher cette info.
Action 1: search("PIB par habitant France 2024")
Observation 1: Le PIB par habitant de la France en 2024 est d'environ 44 400 USD (FMI).

Thought 2: Maintenant je dois trouver celui de l'Allemagne pour comparer.
Action 2: search("PIB par habitant Allemagne 2024")
Observation 2: Le PIB par habitant de l'Allemagne en 2024 est d'environ 52 800 USD (FMI).

Thought 3: J'ai les deux chiffres. La France est a 44 400 et l'Allemagne a 52 800, soit un ecart d'environ 16%. Je peux repondre.
Action 3: finish("Le PIB par habitant de la France (44 400 USD) est inferieur de ~16% a celui de l'Allemagne (52 800 USD) en 2024.")
```

### Pourquoi ca marche

1. **Trace de raisonnement explicite** → le LLM "montre son travail", plus facile a debugger
2. **Grounding par les observations** → chaque pensee est ancree dans des faits reels (sorties d'outils)
3. **Decomposition naturelle** → le LLM decompose un probleme complexe en sous-etapes
4. **Self-correction** → si une observation contredit l'hypothese, le LLM peut changer de strategie

### Implementation concrete

En pratique, ReAct se traduit par un **system prompt** qui definit le format, un **parsing** de la sortie du LLM, et une **boucle** qui execute les actions :

```python
# Pseudo-code de la boucle ReAct
while True:
    response = llm(prompt + history)      # Think
    action = parse_action(response)        # Extract action from LLM output
    if action.name == "finish":
        return action.input                # Done
    observation = execute_tool(action)      # Act
    history.append(action, observation)     # Perceive (feed back)
```

> **Opinion** : ReAct est au coeur de 90% des agents en production. Meme les architectures complexes (plan-and-execute, reflexion) sont des variations de ReAct. Maitrise ce pattern, et tout le reste n'est que de la specialisation.

---

## 5. Les 4 composants d'un agent

### 5.1 LLM — Le cerveau

Le LLM est le moteur de raisonnement. Il recoit le contexte (historique, observations, outils disponibles) et decide quoi faire.

**Ce qui compte en pratique** :
- **Capacite de function calling** : les modeles recents (Claude 4.x, GPT-5.x) gerent nativement les schemas d'outils
- **Taille de contexte** : plus le contexte est grand, plus l'agent peut gerer de complexite (mais attention au "lost in the middle")
- **Cout** : un agent peut faire 5-50 appels LLM par tache. Un modele a $15/Mtok coute 10x plus que $1.5/Mtok. Utiliser le bon modele pour le bon usage (petit modele pour le routing, gros pour le raisonnement)
- **Latence** : chaque appel = 1-5s. 20 etapes = 20-100s. Le streaming aide l'UX mais pas la latence totale

### 5.2 Tools — Les bras

Les outils sont les actions que l'agent peut effectuer. Chaque outil a :
- Un **nom** clair et non ambigu
- Une **description** que le LLM utilise pour decider quand l'utiliser
- Un **schema de parametres** (JSON Schema) qui definit les inputs
- Une **fonction d'execution** qui fait le travail reel

```python
tools = [
    {
        "name": "search_web",
        "description": "Search the web for current information. Use when you need facts you don't know.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    }
]
```

**Regles de design d'outils** (retour d'experience) :
1. **Noms explicites** : `search_web` > `search` > `s`
2. **Descriptions qui disent QUAND utiliser** : pas juste ce que l'outil fait, mais quand le choisir
3. **Peu d'outils** : 3-7 outils est le sweet spot. Au-dela, le LLM se perd
4. **Schemas stricts** : valider les inputs, retourner des erreurs claires
5. **Idempotents si possible** : un outil appele 2 fois avec les memes params doit donner le meme resultat

### 5.3 Memory — La memoire

| Type | Duree | Implementation | Usage |
|------|-------|----------------|-------|
| **Short-term** | Pendant l'execution | Messages dans le prompt | Historique de la conversation en cours |
| **Working memory** | Pendant une tache | Scratchpad, variables | Notes intermediaires de l'agent |
| **Long-term** | Persistant | Vector store, BDD | Connaissances passees, preferences utilisateur |
| **Episodic** | Persistant | Logs structures | "La derniere fois que j'ai fait X, j'ai appris que..." |

**En pratique** : la short-term memory (= le context window) est de loin la plus utilisee. La long-term memory (RAG) est utile mais complexe. La memoire episodique est le graal (cf. MemGPT) mais encore experimentale.

### 5.4 Planning — La strategie

Le planning, c'est la capacite de l'agent a decomposer un objectif en sous-etapes avant d'agir.

**3 niveaux de planning** :
1. **Aucun** (ReAct pur) : l'agent decide etape par etape, sans plan global. Fonctionne pour les taches simples.
2. **Plan-then-execute** : l'agent cree un plan d'abord, puis l'execute etape par etape. Mieux pour les taches complexes.
3. **Adaptive planning** : l'agent cree un plan, l'execute, et re-planifie si necessaire. Le plus robuste mais le plus couteux.

> **Opinion** : commence sans planning explicite (ReAct pur). Ajoute du planning seulement si tu observes que l'agent se perd dans les taches longues. Le planning premature est une source de complexite inutile.

---

## 6. Taxonomie des agents

### 6.1 Agent reactif (Reactive)

- **Decide au coup par coup** : pas de plan, pas de modele du monde
- Reagit directement aux observations
- **Exemple** : un agent ReAct basique qui repond a chaque observation sans strategie globale
- **Quand l'utiliser** : taches courtes (1-5 etapes), bien definies
- **Avantage** : simple, rapide, previsible
- **Limite** : se perd dans les taches longues ou ambigues

### 6.2 Agent deliberatif (Deliberative)

- **Planifie avant d'agir** : cree un modele mental de la tache
- Maintient un plan explicite, le met a jour
- **Exemple** : un agent plan-and-execute qui decompose "Analyse ce repo et propose des refactors" en sous-taches
- **Quand l'utiliser** : taches longues (10+ etapes), objectif complexe
- **Avantage** : meilleure coherence sur la duree
- **Limite** : planning couteux, re-planning complexe, plan peut devenir obsolete

### 6.3 Agent hybride (Hybrid)

- **Combine les deux** : planification de haut niveau + execution reactive
- Le plus courant en production
- **Exemple** : Claude Code — planifie une approche globale, puis execute de maniere reactive a chaque etape, re-planifie si ca deraille
- **Quand l'utiliser** : presque toujours en production
- **Avantage** : robuste, adaptatif
- **Limite** : plus complexe a implementer et debugger

---

## 7. Decision framework : Agent vs Simple prompt

```
Ta tache necessite-t-elle...

  Des actions dans le monde reel (API, fichiers, BDD) ?
  ├── NON → Simple prompt / pipeline suffit
  └── OUI
       │
       Le nombre d'etapes est-il connu a l'avance ?
       ├── OUI → Pipeline deterministe (pas besoin d'agent)
       └── NON
            │
            La tache necessite-t-elle de la decouverte ?
            (chercher, essayer, corriger, adapter)
            ├── NON → Chain LLM + tools fixes
            └── OUI → AGENT ✓
```

### Exemples concrets

| Tache | Solution | Pourquoi |
|-------|----------|----------|
| Resumer un document | Simple prompt | Une passe, pas d'outil |
| Extraire des donnees d'un PDF | Pipeline | Etapes fixes : parse → extract → format |
| Repondre a une question avec recherche web | Chain (1 outil) | search → answer, toujours 2 etapes |
| "Investigue ce bug et propose un fix" | **Agent** | Nombre d'etapes inconnu, decouverte, iteration |
| "Analyse ce repo et cree une PR" | **Agent** | Exploration, decisions multiples, actions variees |
| "Planifie mon voyage a Tokyo" | **Agent** | Multi-outils, preferences, iteration |

> **Regle d'or** : si tu peux ecrire un `for` loop qui fait le job, tu n'as pas besoin d'un agent. L'agent n'a de sens que quand le LLM doit prendre des **decisions de routing** a chaque etape.

---

## 8. Limites actuelles des agents

### 8.1 Hallucination d'outils

Le LLM invente des outils qui n'existent pas, ou passe des parametres incorrects.

**Mitigation** :
- Schema strict avec validation
- Peu d'outils (3-7 max)
- Descriptions precises avec exemples dans le prompt
- Retry avec feedback d'erreur

### 8.2 Boucles infinies

L'agent repete la meme action sans fin, ou alterne entre deux etats.

**Mitigation** :
- **Max iterations** : toujours mettre une limite (10-30 typiquement)
- **Detection de repetition** : si les 3 dernieres actions sont identiques, forcer un arret
- **Timeout** : limite de temps en plus de la limite d'iterations

### 8.3 Explosion de couts

Un agent peut faire 50+ appels LLM, chacun avec un contexte croissant.

**Mitigation** :
- **Budget tokens** : plafond par execution
- **Modele adaptatif** : petit modele pour les decisions simples, gros modele pour le raisonnement
- **Summarization** : resumer l'historique au lieu de tout garder
- **Early exit** : si la reponse est evidente, arreter tot

### 8.4 Tool misuse

L'agent utilise un outil a mauvais escient (ex: supprimer un fichier au lieu de le lire).

**Mitigation** :
- **Sandbox** : limiter les permissions des outils
- **Human-in-the-loop** : demander confirmation pour les actions destructives
- **Dry-run mode** : montrer ce que l'agent ferait sans executer
- **Outils read-only par defaut** : les actions d'ecriture doivent etre explicitement autorisees

### 8.5 Context window overflow

Apres beaucoup d'etapes, le contexte deborde.

**Mitigation** :
- **Summarization** : condenser l'historique ancien
- **Sliding window** : ne garder que les N dernieres etapes
- **Working memory** : extraire les infos cles dans un scratchpad compact

---

## 9. Flash Cards — Test de comprehension

**Q1 : Quelle est la difference fondamentale entre un agent et un pipeline ?**
> R : Dans un pipeline, le flux est predetermine (DAG fixe). Dans un agent, le LLM controle le flux d'execution — il decide dynamiquement quelles actions effectuer et quand s'arreter.

**Q2 : Quelles sont les 3 phases de la boucle d'un agent ?**
> R : Perceive (observer le resultat), Think (raisonner, decider), Act (executer une action ou repondre). L'agent boucle jusqu'a completion.

**Q3 : Pourquoi le pattern ReAct force-t-il le LLM a generer un "Thought" avant chaque "Action" ?**
> R : Pour ancrer le raisonnement dans les observations, faciliter le debugging (trace explicite), et permettre la self-correction si une observation contredit l'hypothese.

**Q4 : Tu dois construire un systeme qui extrait des donnees de factures PDF et les insere en BDD. Agent ou pipeline ?**
> R : Pipeline. Les etapes sont fixes (parse PDF → extraire champs → valider → inserer BDD), pas de decouverte ni de decision dynamique. Un agent serait du over-engineering.

**Q5 : Cite 3 mitigations contre les boucles infinies d'un agent.**
> R : (1) Max iterations (hard limit), (2) detection de repetition (3 actions identiques = stop), (3) timeout global. Bonus : budget tokens comme safety net.

---

## Points cles a retenir

- Un agent = LLM dans la boucle de controle (flux dynamique, pas un DAG fixe)
- La boucle Perceive → Think → Act est universelle — tout framework l'implemente
- ReAct est le pattern fondateur : raisonner avant d'agir, observer apres
- 4 composants : LLM (cerveau), Tools (bras), Memory (memoire), Planning (strategie)
- La majorite des cas d'usage ne necessitent PAS un agent — utiliser le decision framework
- Toujours mettre des guardrails : max iterations, budget, timeout, sandbox
