# Exercice (facile) — RACI : trouver l'agent orphelin de responsabilité

## Objectif

Vérifier la règle d'or du RACI sur une petite flotte : **exactement un Accountable par agent**. Tu écris une fonction qui, pour chaque agent, dit s'il a un responsable clair ou s'il est « orphelin de responsabilité » (zéro ou plusieurs A).

## Consigne

1. Crée un fichier `solution.py` dans ton `workspace/`.
2. Représente chaque agent par un dictionnaire `agent_id -> {actor_id: role}` où `role` est l'une des chaînes `"R"`, `"A"`, `"C"`, `"I"`. Utilise le jeu d'exemple suivant :
   - `"daily-report"` : `{"sam": "R", "lea": "A", "kim": "I"}`
   - `"price-watcher"` : `{"sam": "R", "kim": "C"}`  *(aucun A)*
   - `"shipping-bot"` : `{"lea": "A", "kim": "A"}`  *(deux A)*
3. Écris `count_accountables(raci)` qui renvoie le nombre de rôles `"A"` dans le RACI d'un agent.
4. Écris `is_clearly_owned(raci)` qui renvoie `True` si et seulement si il y a **exactement un** `"A"`.
5. Parcours la flotte et imprime, pour chaque agent, une ligne : `<agent_id> : OK` ou `<agent_id> : ORPHELIN (<n> accountables)`.
6. Termine en imprimant le nombre total d'agents « clairement possédés » sur le total.

## Criteres de reussite

- [ ] `count_accountables` renvoie bien `1`, `0`, `2` pour les trois agents d'exemple.
- [ ] `is_clearly_owned` renvoie `True` uniquement pour `"daily-report"`.
- [ ] La sortie liste chaque agent avec son verdict (OK / ORPHELIN + nombre).
- [ ] Une ligne de synthèse affiche le ratio « possédés / total » (ici `1/3`).
- [ ] Le script tourne en **stdlib seule** avec `python solution.py` (exit 0).
