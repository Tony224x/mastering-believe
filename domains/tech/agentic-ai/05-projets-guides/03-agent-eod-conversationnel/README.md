# Projet 03 — Agent EOD conversationnel

## Contexte metier

Complement du projet LLM 03 (generation d'EOD automatique) : **au lieu de generer un rapport one-shot, on transforme l'EOD en agent conversationnel**. L'OCC pose des questions en langage naturel sur un shift passe, et l'agent repond avec des citations vers les events sources.

Exemples de questions typiques :
- "Pourquoi l'AGV Alpha-2 a-t-il essaye un pickup avant le feu vert WMS ?"
- "Quelles sont les flottes qui ont perdu plus de 30% de leur capacite (panne / batterie) ?"
- "Fais-moi un resume des 15 dernieres minutes du shift"
- "Compare la reactivite des flottes Alpha et Bravo face au pic d'arrivee de 11h"

C'est typiquement ou les agents brillent : la question est ouverte, la reponse demande plusieurs tools (search events, aggregate, compare), l'utilisateur veut **dialoguer** plutot que lire un rapport fige.

## Objectif technique

Agent LangGraph ReAct avec des tools qui requetent les events d'un shift. Le RAG est **structure** (pas des embeddings) : on query le log d'events par filtre (t, unit_id, kind).

## Consigne

Tools :
- `search_events(filter: dict) -> list[dict]` — filtre sur les events (by unit, kind, time window)
- `get_unit_timeline(unit_id: str, t_start: float, t_end: float) -> list[dict]` — timeline d'une unite
- `aggregate_stats(filter: dict, metric: str) -> dict` — stats agregees (pickups, dropoffs, faults, collisions)
- `list_units(fleet: str) -> list[str]` — liste les unites d'une flotte

Architecture :
```
   User question -> agent node -> tool call -> tool result ->
   agent node -> (loop) -> final answer with citations
```

Contraintes :
- Le modele DOIT citer les events sources (format `[ev:id]`)
- Si la question est ambigue (ex: "Alpha c'etait quoi ?"), l'agent pose une question de clarification
- Si les events ne contiennent pas l'info, l'agent dit "non documente" et ne fabrique pas

## Etapes guidees

1. **Charger le log d'events** — a partir d'un JSON fixture (fourni dans `solution/demo_events.json`, a generer)
2. **Implementer les tools** — pures functions sur le log
3. **System prompt** — role "analyste EOD logistique", contraintes de citation
4. **Boucle agent** — `create_react_agent` de langgraph.prebuilt suffit ici
5. **Tester** — 5 questions types, verifier que toutes ont des citations et qu'aucune hallucination

## Criteres de reussite

- L'agent repond correctement a 5 questions de test sur un log fictif
- Chaque reponse contient au moins une citation `[ev:id]`
- Une question hors-contexte ("quel temps fait-il ?") est refusee proprement
- Une question sur des donnees non presentes retourne "non documente"
- Stub mode fonctionne (reponses scriptees pour 2-3 questions)

## Solution

Voir `solution/eod_agent.py`. Le demo log est genere dynamiquement par le script.

## Pour aller plus loin

- **Embeddings** en plus : pour les questions qualitatives ("trouve des moments stressants"), ajouter un index vectoriel sur les descriptions d'events
- **Multi-shift** : l'agent peut comparer 2 shifts ("Alpha a-t-il ete meilleur lundi ou mardi ?")
- **Persistence** : checkpointer pour conserver la conversation, permettre les questions de suivi ("et pour Bravo maintenant ?")
- **Export** : generer un document PDF en fin de conversation avec les Q/A retenues
