# J15 — Exercice facile : le mini-pipeline INGEST → REPORT

## Objectif

Construire le squelette du toolkit : charger une flotte d'agents dans un registry, valider les 4 piliers, detecter les orphelins, et produire un mini-rapport de couverture de gouvernance. C'est l'ossature `ingest → report` sans encore l'enforcement ni l'audit.

## Consigne

1. Definis une dataclass `GovernedAgent` avec : `agent_id`, `owner` (peut etre `None`), `scopes` (tuple), `risk_tier`. Ajoute une methode `is_fully_governed()` qui renvoie `(bool, list_des_piliers_manquants)` — un agent est pleinement gouverne s'il a un `agent_id`, un `owner` non vide et au moins un scope.
2. Ecris une fonction `ingest(raw_fleet)` qui prend une liste de `dict` (export brut, potentiellement avec des cles manquantes) et renvoie une liste de `GovernedAgent`. Tolere les cles absentes (`raw.get(...)`), mais ignore un agent **sans `agent_id` du tout**.
3. Ecris `orphans(agents)` qui renvoie la liste des agents **sans owner** (les shadow agents).
4. Ecris `coverage(agents)` qui renvoie `(nb_gouvernes, total, pourcentage)`.
5. Ecris `render_report(org, agents)` qui imprime : le nombre d'agents, la couverture en %, et la liste des orphelins par `agent_id`.
6. Lance sur une flotte d'au moins 3 agents dont **un orphelin** (owner `None`), et verifie que le rapport l'affiche bien.

## Criteres de reussite

- [ ] `GovernedAgent.is_fully_governed()` renvoie `False` et liste `"owner"` pour un agent sans owner.
- [ ] `ingest` tolere un `dict` sans la cle `scopes` ou `risk_tier` sans lever d'erreur.
- [ ] `ingest` ignore un agent dont `agent_id` est vide ou absent.
- [ ] `orphans` retourne exactement les agents sans owner.
- [ ] `coverage` calcule le bon pourcentage (ex. 2 gouvernes / 3 total → 66 %).
- [ ] Le script s'execute sans erreur (`python`, stdlib uniquement) et passe `python -m py_compile`.
