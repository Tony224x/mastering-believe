# J5 — LangGraph fondamentaux : StateGraph, nodes, edges, conditional routing

> **Temps estime** : 3h | **Prerequis** : J1 (Agent anatomy), J2 (Tool use), J3 (Memory), J4 (Planning)
> **Objectif** : comprendre le modele mental de LangGraph (StateGraph + nodes + edges + state), savoir construire un graph minimal from scratch, et maitriser invoke/stream + human-in-the-loop.

---

## 1. Pourquoi LangGraph existe

Jusqu'a J4, tous les agents qu'on a ecrits sont des **boucles imperatives** :

```python
while not done:
    response = llm(messages)
    if needs_tool:
        result = execute_tool(response)
        messages.append(result)
    else:
        done = True
```

Ca marche pour des agents simples. Mais ca casse des qu'on veut :

- **Plusieurs chemins d'execution** (si X alors A, sinon B)
- **Parallelisation** (executer 3 tool calls en meme temps)
- **Checkpointing automatique** (sauvegarder l'etat a chaque etape)
- **Human-in-the-loop** (interrompre pour demander validation)
- **Debugging** (voir exactement quel noeud a ete execute dans quel ordre)
- **Streaming d'evenements** (voir en temps reel ce que l'agent fait)

Implementer ca a la main donne un plat de spaghettis. LangGraph **factorise ces patterns** dans une API de graph.

> **Analogie** : c'est comme Redux pour les agents. Au lieu d'une boucle imperative avec des mutations partout, tu definis des **nodes purs** qui prennent un state et retournent des updates. Le framework gere le reste.

---

## 2. Le modele mental : un graph de fonctions

LangGraph modelise un agent comme un **graph dirige** ou :

- Les **nodes** sont des fonctions qui prennent un `state` et retournent des **updates**
- Les **edges** relient les nodes — soit inconditionnels, soit conditionnels
- Le **state** est un dict (typiquement un `TypedDict` ou Pydantic) qui circule entre les nodes
- **START** et **END** sont des nodes speciaux qui marquent le debut et la fin

```
                  ┌─────────┐
                  │  START  │
                  └────┬────┘
                       │
                       v
                  ┌─────────┐
              ┌──>│  agent  │  (node)
              │   └────┬────┘
              │        │ (conditional edge)
              │  needs_tool ?
              │        │
              │    yes │ │ no
              │        v v
              │   ┌─────────┐   ┌────────┐
              └───│  tools  │   │  END   │
                  └─────────┘   └────────┘
```

A chaque step, LangGraph :
1. Regarde ou on est dans le graph
2. Execute le node courant, recoit des updates
3. Merge les updates dans le state global
4. Suit l'edge (ou les edges conditionnels) vers le prochain node
5. Recommence jusqu'a atteindre `END`

---

## 3. Definir le state — TypedDict et reducers

### 3.1 Le pattern TypedDict

En LangGraph, le state est typiquement un `TypedDict` :

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]       # Les messages s'ajoutent
    query: str                             # Remplacement direct
    step_count: int                        # Remplacement direct
```

**Points cles** :
- Chaque champ a un type Python standard
- Le `Annotated[list, add]` indique que pour ce champ, les updates doivent etre **concatenees** (non pas remplacees)
- Les champs sans annotation sont remplaces par defaut

### 3.2 Reducers : comment merger les updates

Un **reducer** est une fonction `(ancien_state, update) -> nouveau_state`. LangGraph en fournit des standards :

| Reducer | Comportement | Usage |
|---------|--------------|-------|
| Par defaut | Replace — le nouveau ecrase l'ancien | Champs scalaires (int, str, bool) |
| `operator.add` | Concatene les listes | `messages`, `history` |
| `add_messages` (LangGraph) | Plus intelligent : deduplique par ID, concatene | Messages de chat (LC messages) |
| Custom | Ta propre fonction (a, b) -> c | Logique metier specifique |

```python
# Exemple custom reducer
def merge_dicts(existing: dict, new: dict) -> dict:
    """Merge deep-ish: new values win on key conflicts."""
    return {**existing, **new}

class MyState(TypedDict):
    config: Annotated[dict, merge_dicts]  # Les updates de config se mergent
    messages: Annotated[list, add]         # Les messages se concatenent
```

### 3.3 Why reducers matter

Sans reducers, un node qui retourne `{"messages": [new_msg]}` **ecraserait** tous les messages existants. Avec le reducer `add`, les messages s'**accumulent** correctement.

C'est un point ou beaucoup de debutants se trompent : si tu oublies l'annotation, tu perds des donnees silencieusement.

---

## 4. Nodes — les fonctions qui font le travail

### 4.1 Signature d'un node

Un node est juste une fonction Python :

```python
def my_node(state: AgentState) -> dict:
    """
    Takes the current state, does some work, returns updates as a dict.
    The returned dict is merged into the state via reducers.
    """
    # Read from state
    query = state["query"]
    messages = state["messages"]

    # Do work (LLM call, tool call, computation, ...)
    response = llm(query)

    # Return UPDATES only, not the full state
    return {
        "messages": [{"role": "assistant", "content": response}],
        "step_count": state["step_count"] + 1,
    }
```

**Points cles** :
- Le node recoit l'**ensemble du state** en lecture
- Il retourne **uniquement les champs a mettre a jour**
- Les champs non retournes restent inchanges
- Le dict retourne est merge via les reducers definis sur le state

### 4.2 Types de nodes courants

| Type | Role | Exemple |
|------|------|---------|
| **LLM node** | Appelle un LLM, met a jour `messages` | `agent_node` |
| **Tool node** | Execute des outils sur la base du dernier message | `tool_node` |
| **Router node** | Decide de la direction sans mutation | `should_continue` |
| **Human node** | Interrompt pour demander input a l'humain | `human_approval` |
| **Data node** | Transforme des donnees internes | `parse_output` |

### 4.3 Nodes purs vs nodes avec side effects

Idealement, un node est **pur** : memes inputs → memes outputs. Pas d'IO cache, pas d'etat global modifie.

En pratique, les nodes LLM ont des side effects (appel API, cout, non-determinisme). On vit avec.

**Regle** : minimise les side effects. Un node qui ecrit dans un fichier + appelle une API + modifie une variable globale est imbouffable a debugger.

---

## 5. Edges — le routage entre nodes

### 5.1 Edges inconditionnels

Un edge inconditionnel force le passage d'un node a un autre :

```python
graph.add_edge("agent", "tools")   # apres "agent", va toujours a "tools"
graph.add_edge(START, "agent")      # START -> agent
```

### 5.2 Conditional edges — le routing dynamique

Un conditional edge choisit la destination **a partir du state**. Tu definis une fonction qui retourne le nom du prochain node :

```python
def should_continue(state: AgentState) -> str:
    """Return the name of the next node."""
    last_message = state["messages"][-1]
    if last_message.get("tool_calls"):
        return "tools"       # il y a des outils a executer
    return "end"              # sinon, on termine

graph.add_conditional_edges(
    "agent",                  # apres ce node...
    should_continue,          # ...appelle cette fonction...
    {                          # ...pour choisir parmi ces destinations
        "tools": "tools",
        "end": END,
    },
)
```

**Pattern classique** : le node `agent` appelle le LLM, et le conditional edge regarde la reponse pour decider s'il faut executer des outils ou terminer. C'est le **routing de base** d'un agent ReAct.

### 5.3 Edges multiples (fan-out)

Tu peux definir plusieurs edges depuis un meme node — ils seront **tous executes** (en parallele si possible). On verra ce pattern en detail a J6.

---

## 6. START, END, compile, invoke

### 6.1 Les nodes speciaux

LangGraph reserve deux nodes :
- `START` : point d'entree du graph (equivalent a "main")
- `END` : point de sortie — quand on atteint END, l'execution s'arrete

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")     # apres tools, retour a l'agent
```

### 6.2 Compile

Avant d'utiliser le graph, il faut le **compiler**. Ca valide la structure et produit un objet executable :

```python
app = graph.compile()
```

Le compile :
- Verifie que tous les nodes sont connectes
- Verifie qu'il existe un chemin de START a END
- Optimise la structure pour l'execution
- Peut attacher un **checkpointer** (voir J6)

### 6.3 Invoke — execution synchrone

```python
initial_state = {"messages": [], "query": "Quelle est la densite de Paris ?", "step_count": 0}
final_state = app.invoke(initial_state)

print(final_state["messages"])
print(final_state["step_count"])
```

`invoke` execute le graph jusqu'a atteindre END, et retourne le **state final**.

### 6.4 Stream — execution avec evenements

```python
for event in app.stream(initial_state):
    # event est un dict {node_name: updates_returned_by_that_node}
    print(event)
```

`stream` execute le graph node par node et **yield un evenement a chaque step**. Ca permet :
- De voir en temps reel ce que fait l'agent (UI)
- De logger / debugger pas a pas
- D'interrompre l'execution si besoin

Il existe plusieurs modes de stream :

| Mode | Ce qui est emis | Quand l'utiliser |
|------|-----------------|------------------|
| `values` | Le state complet apres chaque step | Debug, monitoring |
| `updates` | Uniquement les updates de chaque step | Optimal pour UI |
| `messages` | Les tokens du LLM un par un | Streaming "type-writer" |
| `debug` | Tout : inputs, outputs, timings | Debug profond |

---

## 7. Human-in-the-loop avec interrupts

### 7.1 Pourquoi interrompre

Certains agents doivent demander validation avant des actions critiques :
- "Tu veux bien que je supprime ce fichier ?"
- "Je vais envoyer cet email, tu confirmes ?"
- "Le diagnostic est X, c'est bon ?"

Pour ca, LangGraph supporte les **interrupts** : on peut faire pause avant ou apres un node, attendre un input humain, puis reprendre.

### 7.2 Interrupts statiques

```python
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["sensitive_action"],   # pause avant ce node
)

# 1. Premier invoke : s'arrete avant "sensitive_action"
config = {"configurable": {"thread_id": "user_123"}}
state = app.invoke(initial_state, config)
print("Paused before sensitive action")
print("State so far:", state)

# 2. L'humain decide -- on peut meme modifier le state
# Si OK, on reprend sans modifier :
state = app.invoke(None, config)    # None = continue la ou on s'est arrete
```

### 7.3 Ce que ca change

Sans interrupts, un agent est un script qui tourne d'un bout a l'autre. Avec interrupts, c'est une **conversation a 2** : l'agent fait son travail, l'humain valide les etapes sensibles, et l'agent reprend.

C'est ce qui permet de mettre des agents en production sur des actions critiques : on garde l'humain dans la boucle sans tout refaire a la main.

---

## 8. Pattern complet : un mini-agent avec outils

Voici le squelette d'un agent LangGraph complet :

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: Annotated[list, add]
    query: str

def agent_node(state: AgentState) -> dict:
    """Call the LLM. It may return a tool_call or a final answer."""
    response = call_llm(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState) -> dict:
    """Execute the last tool_call from the last message."""
    last = state["messages"][-1]
    result = run_tool(last["tool_call"])
    return {"messages": [{"role": "tool", "content": result}]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if last.get("tool_call") else "end"

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

app = graph.compile()
result = app.invoke({"messages": [{"role": "user", "content": "..."}], "query": "..."})
```

**Lis ce pattern plusieurs fois** — c'est le squelette de 90% des agents LangGraph. Si tu comprends ca, tu comprends LangGraph.

---

## 9. Anti-patterns et pieges

### 9.1 Oublier les reducers

```python
# BUG : on ecrase messages a chaque node
class State(TypedDict):
    messages: list      # pas de reducer !

# Les nodes qui retournent {"messages": [new]} ecrasent au lieu d'ajouter
# Fix : Annotated[list, add]
```

### 9.2 Nodes qui mutent le state

```python
def bad_node(state):
    state["messages"].append("x")   # MUTE le state !
    return {}
```

Les nodes doivent **retourner** les updates, pas muter le state. La mutation casse le checkpointing et le streaming.

### 9.3 Conditional edges qui retournent un mauvais nom

```python
def should_continue(state):
    return "tool"      # BUG : le dict dit "tools" pas "tool"

graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
# KeyError au runtime
```

Le mapping entre la string retournee et les noms de nodes doit etre exact. Preferer une enum ou des constantes.

### 9.4 Graphs trop complexes trop tot

Commence simple : 2-3 nodes. Si tu te retrouves avec 12 nodes et 25 edges dans ton premier agent, tu as pris un mauvais virage. Decoupe en subgraphs (J6).

---

## 10. Flash Cards — Test de comprehension

**Q1 : Quels sont les 4 concepts fondamentaux d'un graph LangGraph ?**
> R : (1) **State** : un TypedDict qui circule entre les nodes. (2) **Nodes** : des fonctions qui prennent le state et retournent des updates. (3) **Edges** : les liens entre nodes, inconditionnels ou conditionnels. (4) **START/END** : les nodes speciaux qui marquent le debut et la fin du graph.

**Q2 : Pourquoi un state typique utilise-t-il `Annotated[list, add]` pour les messages ?**
> R : Parce qu'un node qui retourne `{"messages": [new_msg]}` ecraserait toute la liste existante par defaut (le reducer par defaut = replace). Avec `Annotated[list, add]`, LangGraph utilise `operator.add` pour **concatener** la nouvelle liste a l'ancienne, ce qui preserve l'historique.

**Q3 : Quelle est la difference entre `invoke` et `stream` ?**
> R : `invoke` execute le graph jusqu'a END et retourne le state final en une seule fois. `stream` execute le graph step par step et yield un evenement apres chaque node (ou meme chaque token en mode `messages`). `stream` est utile pour l'UI temps reel, le debug, et l'interruption.

**Q4 : A quoi servent les interrupts dans LangGraph ?**
> R : Ils permettent de faire **pause** l'execution avant ou apres un node, pour demander validation humaine (human-in-the-loop). L'agent sauve son state, attend qu'on lui dise de reprendre (avec ou sans modification), puis continue. Essentiel pour les actions sensibles (envois d'email, suppressions, paiements).

**Q5 : Quel est le pattern de routage classique pour un agent ReAct en LangGraph ?**
> R : Un node `agent` (LLM) et un node `tools` (execution). Apres `agent`, un **conditional edge** regarde le dernier message : si le LLM a demande un outil, on va a `tools` ; sinon, on va a `END`. Apres `tools`, un edge inconditionnel retourne a `agent`. Ce pattern minimal tourne en boucle jusqu'a ce que le LLM decide d'arreter.

---

## Points cles a retenir

- LangGraph = modele graph-based pour les agents : state + nodes + edges
- State est un TypedDict avec des **reducers** (Annotated) pour controler comment les updates se mergent
- Nodes sont des fonctions pures : prennent state, retournent updates
- Edges : inconditionnels (A → B) ou conditionnels (fonction qui retourne la cle d'un dict)
- START et END sont des nodes speciaux qui marquent le debut et la fin
- Compile valide le graph et produit l'executable
- `invoke` = execution sync, `stream` = execution avec evenements
- Interrupts = human-in-the-loop : pause avant/apres un node, attend validation
- Pattern minimal : agent + tools + conditional edge. C'est le squelette de 90% des agents
- Anti-patterns : oublier les reducers, muter le state dans les nodes, conditional edges avec mauvais mapping

---

## Pour aller plus loin

Ressources canoniques sur le sujet :

- **LangGraph official docs** (LangChain) — concepts, API reference, tutoriels et how-to officiels. https://langchain-ai.github.io/langgraph/
- **LangChain Academy — "Introduction to LangGraph"** — MOOC gratuit (55 lecons, 6h) : StateGraph, ReAct, memory, human-in-the-loop. https://academy.langchain.com/courses/intro-to-langgraph
