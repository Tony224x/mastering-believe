# J6 — LangGraph avance : subgraphs, parallel, streaming, persistence, time-travel

> **Temps estime** : 3h | **Prerequis** : J5 (LangGraph fondamentaux)
> **Objectif** : maitriser les patterns avances de LangGraph (subgraphs, parallel execution, streaming, persistence, time-travel debugging) pour construire des agents production-grade.

---

## 1. Pourquoi on a besoin d'aller plus loin

Le graph de J5 est lineaire : un node, un autre, un conditional edge, END. Ca couvre 60% des cas. Pour les 40% restants :

| Besoin | Solution LangGraph |
|--------|-------------------|
| Un workflow reutilisable dans plusieurs agents | **Subgraphs** |
| Executer plusieurs branches en parallele (fan-out) | **Send API / parallel execution** |
| Voir en temps reel les events du graph (UI) | **Streaming** (values / updates / messages) |
| Reprendre apres un crash | **Persistence / checkpointer** |
| Debugger en revenant en arriere | **Time-travel** |
| Explorer "et si l'agent avait fait X ?" | **Branching from past state** |

Ces patterns sont ce qui separe un prototype d'un agent de production.

---

## 2. Subgraphs — composition de graphs

### 2.1 Le principe

Un subgraph, c'est un graph **reutilisable** qu'on peut inclure comme un node dans un graph parent. C'est l'equivalent des fonctions en programmation : au lieu de copier-coller un workflow, on le nomme et on l'appelle.

```
Parent graph:
  START -> preprocess -> [RESEARCH SUBGRAPH] -> synthesize -> END

Research subgraph (reutilisable) :
  START -> search -> filter -> rank -> END
```

### 2.2 Quand creer un subgraph

| Situation | Subgraph ? |
|-----------|------------|
| Le meme workflow est utilise dans 3+ agents | Oui |
| Le workflow fait plus de 5 nodes et a sa propre logique | Oui |
| Le workflow est une "phase" conceptuelle (research, planning, ...) | Oui |
| Workflow unique a un agent, 2-3 nodes | Non, inline |

### 2.3 Comment un subgraph partage son state avec le parent

Il y a deux strategies :

**Shared state** : le subgraph et le parent partagent le meme schema de state. Les updates du subgraph sont directement visibles dans le parent.

```python
# Parent and subgraph both use ResearchState
parent_graph.add_node("research", research_subgraph)
# research_subgraph writes directly to parent's state
```

**Transformed state** : le subgraph a son propre schema. On passe un sous-ensemble du parent en entree, et on extrait un sous-ensemble en sortie.

```python
def research_wrapper(parent_state: ParentState) -> dict:
    sub_input = {"query": parent_state["query"]}
    sub_output = research_subgraph.invoke(sub_input)
    return {"research_results": sub_output["results"]}
```

**Regle** : partager le state quand le subgraph est couple au parent. Transformer quand on veut isoler (meilleure reusabilite).

---

## 3. Parallel execution — Send API

### 3.1 Le probleme

Imagine un agent qui doit chercher dans 5 sources differentes en parallele. Avec des edges sequentiels, il les traiterait une par une — 5x plus lent.

```
Sequentiel (naif) :
  agent -> source1 -> source2 -> source3 -> source4 -> source5 -> synth
  Total = 5 * temps_par_source

Parallele :
  agent ┬-> source1 -┐
        ├-> source2 -┤
        ├-> source3 -┼-> synth
        ├-> source4 -┤
        └-> source5 -┘
  Total = max(temps_par_source)
```

### 3.2 Le pattern Send

LangGraph offre l'API `Send` : au lieu de retourner des updates, un node retourne une **liste de Send** qui indique "va executer ce node avec ces donnees, et ca ce node avec ces autres donnees".

```python
from langgraph.constants import Send

def fan_out_node(state: AgentState) -> list[Send]:
    """Launch parallel searches, one per source."""
    sources = ["source1", "source2", "source3", "source4", "source5"]
    return [Send("search_one", {"source": s, "query": state["query"]})
            for s in sources]
```

LangGraph execute les 5 `search_one` en parallele, et chacun ecrit dans le state. Le reducer (ex : `add`) permet de collecter tous les resultats dans la meme liste.

### 3.3 Fan-in : collecter les resultats

Apres le fan-out, on a besoin d'un node qui attend que tout soit fini et qui collecte. Avec LangGraph, le **reducer** fait ce travail automatiquement : chaque branche ecrit dans le meme champ, et le reducer les merge.

```python
class State(TypedDict):
    results: Annotated[list, add]   # tous les resultats s'accumulent ici

def search_one(state) -> dict:
    result = do_search(state["source"], state["query"])
    return {"results": [result]}
```

Le node suivant (synth) ne s'execute qu'une fois que **tous les parallel ont fini**. C'est le "rendez-vous" naturel dans un graph DAG.

### 3.4 Limites de la parallelisation

- **Coherence** : les branches paralleles ne peuvent pas se voir mutuellement
- **Cout** : K branches en parallele = K appels LLM (cout mais pas latence)
- **Debug** : les events paralleles sont entrelacees dans les logs
- **Non-determinisme** : l'ordre d'arrivee des resultats varie

> **Opinion** : la parallelisation est sous-utilisee en prod. Des que tu as 3+ appels independants, le gain en latence est enorme pour un cout implementation minimal. Faites-le.

---

## 4. Streaming — events vs values vs messages

LangGraph a 3 modes de streaming principaux, chacun pour un usage different.

### 4.1 `stream_mode="values"` — state complet

Apres chaque step, on emet le **state complet**. Utile pour les dashboards qui affichent l'etat courant.

```python
for state in app.stream(initial_state, stream_mode="values"):
    # state est le dict complet du state apres chaque step
    ui.update(state)
```

**Avantage** : simple, tu as toujours l'etat courant.
**Inconvenient** : beaucoup de donnees emises a chaque step, meme si peu change.

### 4.2 `stream_mode="updates"` — uniquement les deltas

Apres chaque step, on emet **uniquement ce qui a change** (l'update du node qui vient de s'executer).

```python
for event in app.stream(initial_state, stream_mode="updates"):
    # event = {node_name: updates_retournees}
    for node, updates in event.items():
        ui.append(f"[{node}] {updates}")
```

**Avantage** : leger, ideal pour l'UI.
**Inconvenient** : ne connait que l'update, pas le state complet.

### 4.3 `stream_mode="messages"` — token-level streaming

Emet les **tokens du LLM au fur et a mesure** qu'ils arrivent. C'est le mode "machine a ecrire" qu'on voit dans ChatGPT.

```python
for chunk, metadata in app.stream(initial_state, stream_mode="messages"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

**Avantage** : experience utilisateur optimale.
**Inconvenient** : le plus complexe a gerer (tokens partiels, chunks a concatener).

### 4.4 Quel mode pour quoi ?

| Usage | Mode recommande |
|-------|-----------------|
| Dashboard de monitoring | `values` |
| UI chat avec liste de steps | `updates` |
| UI chat type-writer | `messages` |
| Debug / tracing | `debug` (tout) |
| Scripts automatises | Pas de stream, utiliser `invoke` |

---

## 5. Persistence — checkpointer

### 5.1 Le principe

Un **checkpointer** est un backend qui sauvegarde le state apres chaque step. LangGraph fournit plusieurs implementations :

- **MemorySaver** : in-memory, utile pour les tests et demos
- **SqliteSaver** : SQLite, leger, mono-processus
- **PostgresSaver** : Postgres, production multi-instances
- **Custom** : implementer `BaseCheckpointSaver` pour Redis, Mongo, etc.

### 5.2 Utilisation de base

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(checkpointer=checkpointer)

# Pour identifier une "conversation" on passe un thread_id
config = {"configurable": {"thread_id": "user_123"}}

# Premier tour
state = app.invoke({"messages": [{"role": "user", "content": "Hi"}]}, config)

# Plus tard, on reprend ou on s'est arrete
state = app.invoke({"messages": [{"role": "user", "content": "And then?"}]}, config)
# La seconde invocation voit l'historique du premier tour automatiquement
```

### 5.3 Thread ID et multi-conversation

Le `thread_id` est la cle qui identifie une **conversation / execution**. Tu peux avoir des milliers de threads en parallele, et LangGraph les gere separement.

```
thread_id = user_123_convo_456
thread_id = user_123_convo_789    <-- meme user, autre conversation
thread_id = user_456              <-- autre user
```

### 5.4 Pourquoi le checkpointer est critique en prod

Sans checkpointer :
- Un crash = perte totale de l'etat
- Impossible de reprendre une conversation
- Impossible de debugger a froid (on n'a que les logs)

Avec checkpointer :
- **Resume apres crash** : reprendre exactement la ou on s'est arrete
- **Multi-tour naturel** : chaque tour voit l'historique automatiquement
- **Audit trail** : inspecter chaque etape de chaque execution
- **Time-travel** : revenir en arriere (voir section suivante)

**Regle** : en production, toujours un checkpointer. Le surcout est minime.

---

## 6. Time-travel debugging

### 6.1 L'idee

Avec un checkpointer, chaque step produit un state persiste. Tu peux donc **revenir a n'importe quel point passe** et :
- Inspecter l'etat a ce moment-la
- Modifier l'etat (override)
- Re-executer depuis ce point (branching)

C'est le superpouvoir du pattern graph-based : on peut reconstruire l'execution comme un repository git.

### 6.2 Lister l'historique

```python
# Recuperer toute l'historique d'un thread
history = list(app.get_state_history(config))
for snapshot in history:
    print(f"Step {snapshot.metadata['step']}: next_node={snapshot.next}")
    print(f"  state: {snapshot.values}")
```

### 6.3 Revenir a un point passe

```python
# Identifier un checkpoint specifique (par exemple 5 steps en arriere)
target = history[5]
target_config = target.config

# Reprendre a partir de ce checkpoint
result = app.invoke(None, target_config)
```

### 6.4 Brancher depuis un point passe (with update)

Le vrai pouvoir : tu peux **modifier l'etat** a un point passe et re-executer.

```python
# "Et si le user avait dit X au lieu de Y ?"
target = history[3]
new_state_values = {**target.values, "messages": [...]}  # on change le dernier message

# Creer un nouveau fork a partir de ce point
new_config = app.update_state(target.config, new_state_values)
result = app.invoke(None, new_config)
```

Ca cree une **nouvelle branche** dans l'historique, comme un git branch. Le thread original reste intact.

### 6.5 Cas d'usage

| Cas | Pourquoi le time-travel aide |
|-----|------------------------------|
| Reproduction de bug | Charger le state au moment du bug, inspecter |
| "Et si..." | Tester des scenarios alternatifs sans refaire tout |
| Demo interactive | Revenir en arriere pour comparer des options |
| Regression testing | Rejouer exactement le meme flow apres un fix |
| Audit | Retracer une decision : "pourquoi l'agent a fait X au step 7 ?" |

---

## 7. Patterns combines : l'agent production complet

Voici ce qu'un agent LangGraph production ressemble typiquement :

```
┌────────────────────────────────────────────────────────┐
│                    MAIN GRAPH                           │
│                                                         │
│   START → preprocess ─┐                                 │
│                       │                                 │
│                       v                                 │
│                  [RESEARCH SUBGRAPH] (fan-out 5 sources)│
│                       │                                 │
│                       v                                 │
│                   analyze                                │
│                       │                                 │
│                       v                                 │
│              should_iterate ? ──yes─┐                   │
│                       │              │                   │
│                       no             │                   │
│                       │              v                   │
│                       v          [RETRY SUBGRAPH]         │
│                    synthesize                            │
│                       │                                  │
│                       v                                  │
│                     END                                  │
│                                                          │
│  Checkpointer: PostgresSaver                            │
│  Streaming: updates (pour UI)                           │
│  Interrupts: before synthesize (human validation)       │
└────────────────────────────────────────────────────────┘
```

**Composants** :
- **Subgraph Research** : reutilisable, fait 5 recherches paralleles via Send API
- **Conditional loop** : retry si la qualite est insuffisante
- **Checkpointer** : Postgres, pour la resilience multi-instance
- **Streaming updates** : pour l'UI temps reel
- **Interrupt before synth** : l'humain valide le plan final avant l'envoi
- **Time-travel** : disponible pour le debugging

---

## 8. Anti-patterns avances

### 8.1 Partager trop de state entre subgraphs

Si 3 subgraphs partagent 10 champs du state, ils sont **couples**. Changer un subgraph casse les autres. Preferer les subgraphs avec un sous-ensemble explicite du state.

### 8.2 Sur-paralleliser

Si tu paralleiss 50 branches, tu vas :
- Exploser le cout LLM
- Bloquer sur le rate limit
- Rendre les logs illisibles

**Regle** : pareliser 3-10 branches, rarement plus. Au-dela, batch les en groupes.

### 8.3 Checkpointer en RAM en production

`MemorySaver` est parfait pour les tests et les demos, **pas pour la prod**. Un crash = tout perdu. Utilise SqliteSaver pour les petits projets, PostgresSaver pour les gros.

### 8.4 Time-travel sans audit

Le time-travel est un superpouvoir, mais il peut aussi etre utilise pour **reecrire l'histoire**. En production, logger clairement quand un time-travel a lieu pour l'audit.

---

## 9. Flash Cards — Test de comprehension

**Q1 : Quand creer un subgraph plutot que d'inliner les nodes ?**
> R : Quand le workflow est (1) reutilise dans plusieurs agents, (2) fait plus de 5 nodes et represente une phase conceptuelle, ou (3) meriterait sa propre documentation/tests. Si c'est un workflow unique a un agent et < 5 nodes, inline est plus simple.

**Q2 : Comment fonctionne le fan-out parallele via l'API Send ?**
> R : Un node retourne une **liste de `Send(node_name, state_update)`** au lieu d'un dict d'updates. LangGraph execute chaque `Send` en parallele, chaque branche ecrit dans le meme state, et les **reducers** (comme `operator.add` sur une liste) collectent automatiquement les resultats.

**Q3 : Quelle difference entre `stream_mode="values"` et `stream_mode="updates"` ?**
> R : `values` emet le **state complet** apres chaque step (lourd mais simple). `updates` emet uniquement les **deltas** retournes par le node courant (leger, ideal pour UI). `messages` emet les tokens du LLM un par un (pour l'effet type-writer).

**Q4 : Pourquoi un checkpointer est indispensable en production ?**
> R : (1) Resilience aux crashes : l'execution peut reprendre la ou elle s'est arretee. (2) Conversations multi-tours : chaque tour voit l'historique automatiquement via le `thread_id`. (3) Audit trail : inspecter chaque step de chaque execution. (4) Time-travel debugging : revenir en arriere pour comprendre un bug ou tester des scenarios alternatifs.

**Q5 : Qu'est-ce que le "branching" depuis un past state, et a quoi ca sert ?**
> R : C'est la capacite de **forker l'execution** depuis un point passe en modifiant l'etat. Tu charges un checkpoint, tu modifies certains champs, puis tu re-executes : le graph suit une nouvelle branche a partir de ce point. Utile pour tester "et si le user avait dit X ?", reproduire un bug avec des variantes, ou explorer des scenarios alternatifs sans toucher le thread original.

---

## Points cles a retenir

- Subgraphs = fonctions pour graphs : reutilisables, composables, encapsulables
- Send API = parallelisation native, avec reducers pour le fan-in automatique
- 3 modes de streaming : values (state complet), updates (deltas), messages (tokens)
- Checkpointer = indispensable en production, MemorySaver pour les tests
- Thread ID = identifiant d'une conversation/execution, permet le multi-tour
- Time-travel = inspecter ou re-executer depuis n'importe quel point passe
- Branching = forker l'execution en modifiant un past state — git branch pour agents
- Anti-patterns : sur-partage de state entre subgraphs, sur-parallelisation, MemorySaver en prod, time-travel sans audit

---

## Pour aller plus loin

Ressources canoniques sur le sujet :

- **LangGraph how-to guides** (LangChain) — recettes officielles pour subgraphs, multi-agent (supervisor, swarm), persistence, streaming, time-travel. https://langchain-ai.github.io/langgraph/
- **DeepLearning.AI — "AI Agents in LangGraph"** (Harrison Chase + Rotem Weiss) — short course pratique : agents from scratch, persistence, human-in-the-loop, search agent. https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/
