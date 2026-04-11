# Projet 03 — Agent AAR conversationnel

## Contexte metier

Complement du projet LLM 03 (generation d'AAR automatique) : **au lieu de generer un rapport one-shot, on transforme l'AAR en agent conversationnel**. Le formateur pose des questions en langage naturel sur un exercice passe, et l'agent repond avec des citations vers les events.

Exemples de questions typiques :
- "Pourquoi le peloton Alpha-2 a ouvert le feu avant l'ordre ?"
- "Quelles sont les unites qui ont perdu plus de 30% de leurs effectifs ?"
- "Fais-moi un resume des 15 dernieres minutes de l'exercice"
- "Compare la reactivite d'Alpha et Bravo face a la menace blindee"

C'est typiquement ou les agents brillent : la question est ouverte, la reponse demande plusieurs tools (search events, aggregate, compare), le formateur veut **dialoguer** plutot que lire un rapport fige.

## Objectif technique

Agent LangGraph ReAct avec des tools qui requetent les events d'un exercice. Le RAG est **structure** (pas des embeddings) : on query le log d'events par filtre (t, unit_id, kind).

## Consigne

Tools :
- `search_events(filter: dict) -> list[dict]` — filtre sur les events (by unit, kind, time window)
- `get_unit_timeline(unit_id: str, t_start: float, t_end: float) -> list[dict]` — timeline d'une unite
- `aggregate_stats(filter: dict, metric: str) -> dict` — stats agregees (kills, losses, moves)
- `list_units(side: str) -> list[str]` — liste les unites d'un camp

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
3. **System prompt** — role "assistant d'analyse tactique", contraintes de citation
4. **Boucle agent** — `create_react_agent` de langgraph.prebuilt suffit ici
5. **Tester** — 5 questions types, verifier que toutes ont des citations et qu'aucune hallucination

## Criteres de reussite

- L'agent repond correctement a 5 questions de test sur un log fictif
- Chaque reponse contient au moins une citation `[ev:id]`
- Une question hors-contexte ("quel temps fait-il ?") est refusee proprement
- Une question sur des donnees non presentes retourne "non documente"
- Stub mode fonctionne (reponses scriptees pour 2-3 questions)

## Solution

Voir `solution/aar_agent.py`. Le demo log est genere dynamiquement par le script.

## Pour aller plus loin

- **Embeddings** en plus : pour les questions qualitatives ("trouve des moments stressants"), ajouter un index vectoriel sur les descriptions d'events
- **Multi-exercice** : l'agent peut comparer 2 exercices ("Alpha a-t-il ete meilleur lundi ou mardi ?")
- **Persistence** : checkpointer pour conserver la conversation, permettre les questions de suivi ("et pour Bravo maintenant ?")
- **Export** : generer un document PDF en fin de conversation avec les Q/A retenues
