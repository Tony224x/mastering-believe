# Projet 01 — Agent Fleet Coordinator

## Contexte metier

Point de depart : une **flotte d'AGV** (~30 robots) qui doit executer un Work Order Plan emis par l'OCC. Dans FleetSim aujourd'hui, AutonomyAI SDK gere ca avec un arbre de regles. On veut savoir ce qu'un agent LLM apporterait en plus : comprehension de Work Orders en langage naturel, adaptation a des contextes non prevus, explicabilite des decisions ("pourquoi avoir renvoye la flotte au dock ?").

C'est la premiere brique indispensable avant d'attaquer le multi-agent.

## Objectif technique

Construire un agent LangGraph single-node qui :
1. Recoit un **Work Order Plan** en texte libre ("Couvrir la zone B-12 du quai 4 jusqu'a 0800, report toutes les 15 min, priorite colis fragiles")
2. Maintient un **state** : position, batterie, capacite de charge, dernier ordre, evenement bloquant
3. A acces a des **tools** : `move_to`, `hold_position`, `pickup`, `report`, `return_to_dock`
4. Boucle ReAct : observation -> reasoning -> tool call -> observation -> ...
5. Stop sur condition (shift complete, batterie critique, capacite saturee)

## Consigne

Architecture :

```
[Work Order Plan + initial state]
        |
        v
  +------------+
  | Agent node |<---+
  +-----+------+    |
        |           | (observation mise a jour apres tool)
        v           |
  +------------+    |
  | Tool node  |----+
  +------------+
        |
        v (sur STOP condition)
     [final]
```

State (TypedDict) :
```python
class FleetState(TypedDict):
    messages: list[BaseMessage]
    position: tuple[float, float]
    battery: float
    payload: float
    last_order: str
    blocking_event: dict | None
    steps_remaining: int
```

Tools (decores avec `@tool`) :
- `move_to(x: float, y: float) -> str` — deplace la flotte, retourne la nouvelle position
- `hold_position() -> str` — stabilise sur place (hold-and-scan)
- `pickup(parcel_id: str) -> str` — embarque un colis
- `report(summary: str) -> str` — envoie un report d'etat au coordinator OCC
- `return_to_dock(direction: str) -> str` — retour au dock

## Etapes guidees

1. **Tools avec side-effects sur l'etat** — LangChain `@tool` ne modifie pas directement le state LangGraph. Utilise `Command(update={...})` pour mettre a jour le state depuis un tool.
2. **System prompt** — decris le role (Fleet Coordinator), les contraintes (SOP du site, objectif), le style attendu (decisions breves, vocabulaire operationnel).
3. **Boucle** — `create_react_agent` from `langgraph.prebuilt` fait le plus gros du boulot pour un single-agent. Mais comprends ce que ca fait avant de l'utiliser.
4. **Stop conditions** — via le state : si `steps_remaining == 0` ou `battery < 0.2`, force la sortie.
5. **Trace** — imprime chaque tool call + argument + resultat pour comprendre ce que l'agent fait.

## Piege du stub LLM

Pour tester sans cle API, on utilise un `_StubLLM` qui renvoie des tool calls scriptes dans un ordre fixe. **Attention** : ce stub doit etre instancie **une seule fois** (singleton module-level), pas a chaque appel d'`agent_node`. Sinon le compteur `_steps` repart a 0 a chaque node et l'agent boucle sur le premier step.

Dans la solution, c'est gere par `_STUB_LLM_SINGLETON` declare au niveau module. Si tu refais le projet from scratch, c'est le premier bug que tu vas rencontrer — il vient du fait que LangGraph recree la closure du node a chaque invocation.

## Criteres de reussite

- L'agent execute un scenario simple de 10 steps sans planter
- Toutes les decisions sont justifiees par le LLM (visible dans les messages)
- La stop condition fonctionne (batterie basse -> return_to_dock automatique)
- Le script tourne sans cle API (stub mode) ET avec (mode live)

## Solution

Voir `solution/fleet_agent.py`.

## Questions de reflexion

- Que se passe-t-il si le LLM hallucine un `parcel_id` inexistant ? Comment garantir que `pickup(parcel_id)` ne fasse rien si le colis n'est pas dans la zone observee ?
- Pourquoi limiter les tools a 5 ? Que se passe-t-il avec 50 tools ?
- L'agent peut-il etre **non-deterministe** et passer en certification client (auditabilite) ?

## Pour aller plus loin

- Ajouter un tool `request_support(type="agv_extra" | "drone_scan", zone=...)` qui fait apparaitre la notion de coordination avec d'autres flottes (prelude au projet 02)
- Persister le state dans un checkpointer pour reprendre apres un crash
- Streamer les tokens pour afficher en temps reel dans une UI OCC
