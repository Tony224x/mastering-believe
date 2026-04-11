# Jour 12 — Agent Systems Architecture

## Pourquoi les agents sont difficiles a architecturer

**Exemple d'abord** : Tu veux construire un assistant de recherche qui : (1) lit un email, (2) cherche des infos sur le web, (3) consulte le CRM, (4) ecrit une reponse, (5) prend rendez-vous dans le calendrier. Tu codes une chaine lineaire : email -> web -> CRM -> draft -> calendar. Trois mois plus tard, tu realises que : (a) parfois il ne faut pas chercher sur le web, (b) parfois le CRM renvoie des infos insuffisantes et il faut demander au user, (c) parfois le calendrier a un conflit et il faut reproposer. Ta chaine lineaire devient un arbre de `if/else` illisible.

C'est exactement a ce moment que tu dois passer a un **pattern d'agent**. Un agent gere le controle-flow dynamiquement en fonction de l'etat et du resultat des tools. Un multi-agent gere la decomposition de la tache entre specialistes.

**Key takeaway** : Un agent n'est pas "un LLM qui appelle des fonctions". C'est un systeme de controle decisionnel qui choisit quelle action faire ensuite, en boucle, jusqu'a atteindre un objectif. L'architecture determine la robustesse plus que le modele.

---

## Anatomy d'un agent minimal

```
  ┌────────────────────────────────────────────────────────────┐
  │                         AGENT LOOP                          │
  │                                                              │
  │    ┌──────────┐    ┌──────────┐    ┌─────────────┐          │
  │    │   STATE  │───>│  PLAN    │───>│  ACT        │          │
  │    │ (context,│    │ (what to │    │ (call tool, │          │
  │    │  memory) │    │  do)     │    │  LLM call)  │          │
  │    └─────▲────┘    └──────────┘    └──────┬──────┘          │
  │          │                                │                  │
  │          │          ┌──────────┐          │                  │
  │          └──────────│ OBSERVE  │<─────────┘                  │
  │                     │ (result) │                             │
  │                     └──────────┘                             │
  │                                                              │
  │    Stop when: task done, budget exceeded, human needed       │
  └────────────────────────────────────────────────────────────┘
```

Les 4 composants essentiels :
- **State** : le contexte, la memoire, les outputs partiels, le budget restant
- **Plan** : decider la prochaine action (LLM decision ou rule)
- **Act** : executer l'action (tool call, LLM call, API)
- **Observe** : lire le resultat et mettre a jour l'etat

Un agent mal concu est un agent qui ne sait pas s'arreter. **Toujours** avoir une condition de sortie explicite.

---

## Les patterns d'orchestration

### 1. Single agent (ReAct loop)

Un seul agent, plusieurs tools. Le LLM decide a chaque step quel tool appeler et quand s'arreter.

```
  user ──> agent ──> LLM ──> tool_call ──> result
                       │                      │
                       └──────<───────────────┘
                          (loop until done)
```

**Quand l'utiliser** :
- Taches homogenes sur un domaine unique
- < 5 tools
- Quand un seul modele a toute la connaissance necessaire

**Avantages** : simple, debuggable, un seul contexte a raisonner.
**Inconvenients** : le contexte grandit vite, les erreurs du LLM sont concentrees sur une seule entite.

### 2. Supervisor pattern (hub and spoke)

Un superviseur (LLM fort) dispatche les taches a des specialistes (LLM par domaine ou tools lourds).

```
                 ┌────────────┐
                 │ Supervisor │
                 └─────┬──────┘
            ┌──────────┼──────────┬──────────┐
            v          v          v          v
       ┌─────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐
       │ Coder   │ │ Writer │ │ Searcher│ │ Analyst  │
       └─────────┘ └────────┘ └─────────┘ └──────────┘
```

Le superviseur :
1. Recoit la requete utilisateur
2. Decompose en sous-taches
3. Envoie chaque sous-tache a l'agent approprie
4. Aggrege les resultats
5. Repond a l'utilisateur

**Quand l'utiliser** :
- Taches heterogenes (code + ecriture + search)
- Plusieurs modeles specialises
- Quand tu veux limiter le contexte de chaque agent

**Avantages** : chaque specialiste a un prompt focus, contexte court, performances specialisees.
**Inconvenients** : latence (plusieurs appels en serie), cout, complexite.

### 3. Hierarchical (supervisors of supervisors)

```
                      ┌───────────────┐
                      │  Master Plan  │
                      └───────┬───────┘
              ┌───────────────┼───────────────┐
              v               v               v
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Research │   │ Execution│   │ Review   │
        │ Super    │   │ Super    │   │ Super    │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
         [sub-agents]   [sub-agents]    [sub-agents]
```

**Quand l'utiliser** : projets multi-phases (recherche profonde, development, enterprise workflows).
**Attention** : chaque niveau ajoute latence et cout. Ne pas abuser.

### 4. Swarm / peer-to-peer

Pas de superviseur central. Les agents se passent la main directement via **handoff**.

```
  Agent A ───handoff──> Agent B ───handoff──> Agent C ───return──> user
```

Implemente par OpenAI Swarm, CrewAI "sequential". Chaque agent sait a qui passer la main selon le contexte.

**Avantages** : simple, flexible, pas de single point of failure (superviseur).
**Inconvenients** : plus dur a debugger, risque de boucles entre agents.

### 5. Debate / adversarial

Plusieurs agents discutent entre eux avant de produire une reponse, chacun defendant un point de vue ou un role (critic, optimist, skeptic).

**Quand** : prises de decision complexes, ecriture technique, raisonnement ou la diversite des perspectives ameliore le resultat.
**Cout** : eleve (N appels LLM pour une seule reponse).

---

## Tool routing strategies

Un agent avec 30 tools ne peut pas tous les voir a chaque step -> contexte explose, decisions degradent. Solutions :

### 1. Tool filtering

Avant chaque step, selectionner les 5-10 tools les plus pertinents pour la requete courante. Peut etre fait via un embedding de la description du tool vs l'embedding de la tache.

### 2. Tool namespacing / hierarchies

Regrouper les tools en "modules" (ex: `github.*`, `calendar.*`, `email.*`). Le LLM choisit d'abord un module, puis un tool dans ce module.

### 3. Tool registry

Les tools sont stockes dans une DB. L'agent "search_tools(query)" comme un first step. Pattern utilise quand le catalogue de tools change souvent.

### 4. MCP (Model Context Protocol)

Standard Anthropic pour exposer des tools externes comme des MCP servers. L'agent se connecte et decouvre les tools dynamiquement. Permet de composer facilement des capacites.

---

## Memoire : court terme vs long terme

### Short-term memory (working memory)

- Le historique de la conversation actuelle
- Les resultats des tools intermediaires
- Le plan courant
- Tient dans le contexte du LLM

**Probleme** : le contexte grandit vite, couts LLM montent, latence aussi. Solution : **summarization intermediaire** (toutes les 10 messages, remplacer par un resume).

### Long-term memory

Persiste entre les sessions. Types :

1. **Episodic memory** : evenements passes ("l'utilisateur m'a dit qu'il est vegetarien")
2. **Semantic memory** : faits stables ("le user s'appelle Anthony")
3. **Procedural memory** : "comment faire" ("format JSON pour ce tool")

**Architecture** : un vector store (episodic) + une DB structuree (semantic) + des regles pre-apprises (procedural).

**Framework** : LangGraph a des checkpointers et un Store pour ca. Letta (ex MemGPT) est dedie a ce probleme.

### Piege : le context overflow

Ne pas mettre TOUTE la memoire dans le prompt. Toujours retrieve on demand, comme un RAG sur ta propre memoire.

---

## Handoff protocols (multi-agent)

Quand un agent passe la main a un autre, il doit :

1. Expliquer le contexte (pas juste "continue")
2. Preciser ce qui est deja fait
3. Preciser ce qui reste a faire
4. Specifier le critere de succes
5. Fournir une budget (N steps, M minutes)

**Exemple de handoff message** :
```json
{
  "from": "supervisor",
  "to": "code_agent",
  "context": "User wants to refactor this Python class to async.",
  "done": ["file read", "current version analyzed"],
  "remaining": ["generate async version", "update tests"],
  "success_criteria": "tests pass + class uses async/await",
  "budget": {"max_steps": 10, "max_tokens": 50000}
}
```

---

## Quand single-agent bat multi-agent

La mode est aux systemes multi-agents, mais ils ne sont pas toujours justifies. Single-agent est meilleur quand :

- Le domaine est homogene
- Le contexte est court
- La tache est lineaire
- Tu veux de la latence basse (pas de round-trips inter-agents)
- Tu veux un debug simple

**Signal d'alarme multi-agent** : tu definis 5 agents pour une tache qui pourrait etre 3 tools sur un seul agent. Tu ajoutes de la complexite pour satisfaire l'architecte, pas pour resoudre un probleme.

**Regle de Rich Sutton / Karpathy** : le systeme le plus simple qui marche bat toujours le systeme le plus sophistique qui marche egalement.

---

## Observability des agents

Les agents sont particulierement difficiles a debugger parce que leurs decisions sont non deterministes et chainees. Ce qu'il faut logger :

- **Chaque step** : quel agent, quel LLM call, quel tool call, quel resultat
- **La decision du plan** : pourquoi ce choix ? (reasoning chain, si disponible)
- **Les erreurs** et les retries
- **Le budget consomme** (tokens, steps, temps)
- **Les messages de handoff** (si multi-agent)

Outils : **LangSmith**, **Langfuse**, **Arize Phoenix**, **Helicone**. Vu demain (J13).

---

## Tradeoffs recapitulatifs

| Dimension | Single-agent | Multi-agent supervisor | Hierarchical |
|---|---|---|---|
| Complexite | Basse | Moyenne | Haute |
| Latence | 1-3 s | 5-15 s | 30+ s |
| Cout | 1x | 2-5x | 10x+ |
| Debuggabilite | Facile | Moyen | Difficile |
| Quand | Taches homogenes | Taches heterogenes | Workflows multi-phases |

---

## Exemples reels

- **Cursor / Devin** : agents de coding, architecture principalement single-agent avec tools riches
- **Claude Code / Cursor Composer** : single-agent avec sub-agents specialises a la demande
- **AutoGPT historique** : single-agent avec memoire, mauvaise robustesse
- **CrewAI / AutoGen / OpenAI Swarm** : frameworks multi-agent avec handoff
- **LangGraph** : le framework le plus utilise pour construire des agents complexes (graph-based)

---

## Flash cards

**Q: Quels sont les 4 composants essentiels d'un agent ?**
R: State, Plan, Act, Observe. En boucle, avec une condition de sortie explicite.

**Q: Quand utiliser un supervisor pattern au lieu d'un single-agent ?**
R: Quand les sous-taches sont heterogenes (code + search + writing) et que tu veux limiter le contexte de chaque specialiste.

**Q: Quel est le probleme principal du single-agent quand il y a > 20 tools ?**
R: Le contexte explose, la decision se degrade. Solution : tool filtering ou namespacing.

**Q: Qu'est-ce qu'un handoff message bien construit ?**
R: Contexte + done + remaining + success criteria + budget. Un agent qui dit juste "continue" est mal concu.

**Q: Quand single-agent bat multi-agent ?**
R: Quand le domaine est homogene, le contexte court, la latence critique et que le debug doit rester simple.

---

## Key takeaways

- Un agent = state + plan + act + observe + condition d'arret. Pas un LLM qui "pense".
- Commence TOUJOURS single-agent. Passe a multi-agent quand tu as une preuve que c'est necessaire.
- Le supervisor pattern est le multi-agent le plus utilise. Hierarchical pour les workflows complexes seulement.
- Memoire : short-term dans le contexte avec summarization, long-term dans un store externe (vector + relational).
- Observability est non negociable. Les agents sans tracing = impossibilite de debugger = projet mort.
