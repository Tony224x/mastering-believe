# Exercice (facile) — Recenser une flotte d'agents

## Objectif

Construire un mini-recensement (*census*) d'une flotte d'agents et répondre, par le code, à la question fondatrice du Jour 1 : **« Combien d'agents tournent, et combien sont sans propriétaire ? »**

## Consigne

1. Crée un fichier `01-pourquoi-gouverner.py` dans ton `workspace/` (ou édite-le là où tu travailles).
2. Représente chaque agent par un `dict` avec au minimum les clés `agent_id` (str), `owner` (str ou `None`) et `tools` (liste de str).
3. Code une liste d'au moins **5 agents**, dont **au moins 2 sans owner** (`owner=None` ou chaîne vide).
4. Écris une fonction `count_orphans(agents) -> int` qui renvoie le nombre d'agents *orphelins* (sans propriétaire). Attention : `None` **et** la chaîne vide `""` comptent tous deux comme « sans owner ».
5. Dans un bloc `if __name__ == "__main__":`, affiche trois lignes lisibles : le nombre total d'agents, le nombre d'orphelins, et la liste de leurs `agent_id`.
6. Le script doit tourner avec `python 01-pourquoi-gouverner.py` (stdlib uniquement, aucune dépendance).

## Critères de réussite

- [ ] Le fichier passe `python -m py_compile 01-pourquoi-gouverner.py` sans erreur.
- [ ] La liste contient au moins 5 agents, dont au moins 2 orphelins.
- [ ] `count_orphans` compte correctement `None` **et** `""` comme orphelins.
- [ ] La sortie affiche le total, le nombre d'orphelins, et les `agent_id` concernés.
- [ ] Aucune dépendance externe n'est importée (stdlib seule).
