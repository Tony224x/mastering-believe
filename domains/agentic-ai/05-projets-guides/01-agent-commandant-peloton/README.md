# Projet 01 — Agent commandant de peloton

## Contexte metier

Point de depart : un **peloton d'infanterie** (30 hommes) qui doit executer une mission donnee par le commandant de compagnie. Dans SWORD aujourd'hui, Direct AI gere ca avec un arbre de regles. On veut savoir ce qu'un agent LLM apporterait en plus : comprehension d'ordres en langage naturel, adaptation a des contextes non prevus, explicabilite des decisions ("pourquoi as-tu battu en retraite ?").

C'est le premier brique indispensable avant d'attaquer le multi-agent.

## Objectif technique

Construire un agent LangGraph single-node qui :
1. Recoit un **OPORD** en texte libre ("Tenir la ligne de crete 4521 jusqu'a 0800, rapport toutes les 15 min, ROE defensif")
2. Maintient un **state** : position, sante, munitions, dernier ordre, contact ennemi
3. A acces a des **tools** : `move_to`, `take_cover`, `engage`, `report`, `withdraw`
4. Boucle ReAct : observation -> reasoning -> tool call -> observation -> ...
5. Stop sur condition (mission accomplie, out of ammo, sante critique)

## Consigne

Architecture :

```
[OPORD + initial state]
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
class PlatoonState(TypedDict):
    messages: list[BaseMessage]
    position: tuple[float, float]
    health: float
    ammo: float
    last_order: str
    enemy_contact: dict | None
    steps_remaining: int
```

Tools (decorated avec `@tool`) :
- `move_to(x: float, y: float) -> str` — deplace le peloton, retourne la nouvelle position
- `take_cover() -> str` — passe en defense
- `engage(target_id: str) -> str` — tire sur une cible
- `report(summary: str) -> str` — envoie un sitrep au commandant
- `withdraw(direction: str) -> str` — battue en retraite

## Etapes guidees

1. **Tools avec side-effects sur l'etat** — LangChain `@tool` ne modifie pas directement le state LangGraph. Utilise `Command(update={...})` pour mettre a jour le state depuis un tool.
2. **System prompt** — decris le role (commandant de peloton), les contraintes (ROE, objectif), le style attendu (decisions breves, citations ROE).
3. **Boucle** — `create_react_agent` from `langgraph.prebuilt` fait le plus gros du boulot pour un single-agent. Mais comprends ce que ca fait avant de l'utiliser.
4. **Stop conditions** — via le state : si `steps_remaining == 0` ou `health < 0.2`, force la sortie.
5. **Trace** — imprime chaque tool call + argument + resultat pour comprendre ce que l'agent fait.

## Piege du stub LLM

Pour tester sans cle API, on utilise un `_StubLLM` qui renvoie des tool calls scriptes dans un ordre fixe. **Attention** : ce stub doit etre instancie **une seule fois** (singleton module-level), pas a chaque appel d'`agent_node`. Sinon le compteur `_steps` repart a 0 a chaque node et l'agent boucle sur le premier step.

Dans la solution, c'est geree par `_STUB_LLM_SINGLETON` declare au niveau module. Si tu refais le projet from scratch, c'est le premier bug que tu vas rencontrer — il vient du fait que LangGraph recree la closure du node a chaque invocation.

## Criteres de reussite

- L'agent execute un scenario simple de 10 steps sans planter
- Toutes les decisions sont justifiees par le LLM (visible dans les messages)
- La stop condition fonctionne (low health -> withdraw automatique)
- Le script tourne sans cle API (stub mode) ET avec (mode live)

## Solution

Voir `solution/platoon_agent.py`.

## Questions de reflexion

- Que se passe-t-il si le LLM hallucine une position de cible ? Comment garantir que `engage(target_id)` ne fasse rien si la cible n'est pas dans `enemy_contact` ?
- Pourquoi limiter les tools a 5 ? Que se passe-t-il avec 50 tools ?
- L'agent peut-il etre **non-deterministe** et passer en certification MASA ?

## Pour aller plus loin

- Ajouter un tool `request_support(type="artillery" | "air", coords=...)` qui fait apparaitre la notion de coordination avec d'autres unites (prelude au projet 02)
- Persister le state dans un checkpointer pour reprendre apres un crash
- Streamer les tokens pour afficher en temps reel dans une UI formateur
