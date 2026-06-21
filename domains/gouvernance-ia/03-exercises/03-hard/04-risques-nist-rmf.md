# J4 — Exercice difficile : un risk register board-ready avec couverture RMF

## Objectif

Construire un **risk register** complet pour une flotte d'agents : ingestion de risques tagues par taxonomie et fonction RMF, scoring avec modulateurs, tri par criticite, et **controles de gouvernance** (couverture des 4 fonctions RMF, detection des trous). Produire un rapport lisible par un comite.

## Consigne

En Python 3.11+ (stdlib uniquement) :

1. Modelisez un `Risk` (dataclass) portant au minimum : `risk_id`, `agent_id`, `title`, les **3 axes causaux** (entite/intention/timing), un `domain`, une `rmf_function` (parmi GOVERN/MAP/MEASURE/MANAGE), `likelihood`, `impact`, et les flags `irreversible`, `autonomous`. Validez les scores `1..5`.
2. Implementez une classe `RiskRegister` avec :
   - `add(risk)` qui **refuse un `risk_id` duplique** (`ValueError`) ;
   - `scored_sorted()` qui renvoie les risques scores (modulateurs appliques) **tries par criticite decroissante** ;
   - `coverage_by_function()` qui compte les risques par fonction RMF ;
   - `missing_functions()` qui renvoie la **liste des fonctions RMF sans aucun risque** (un trou de gouvernance a signaler).
3. Chargez **au moins 4 risques** couvrant des situations variees (au moins un irreversible+autonome, au moins un human-in-the-loop reversible).
4. Produisez un `render_report()` qui affiche : le tableau trie (risk_id, agent, L, I, criticite, decision, fonction), le **detail du risque le plus critique** (coordonnees causales + modulateurs appliques), la **couverture par fonction**, et un **avertissement** si une fonction RMF n'a aucun risque.
5. Ajoutez un seuil de traitement assume (`TREAT >= 12`, `MONITOR 6..11`, `ACCEPT < 6`) et un tally global des decisions.

## Criteres de reussite

- [ ] Le script tourne avec `python <fichier>` sans erreur (stdlib seule) et passe `python -m py_compile`.
- [ ] Ajouter deux fois le meme `risk_id` leve `ValueError`.
- [ ] `scored_sorted()` est bien trie par criticite decroissante (le pire en tete).
- [ ] `coverage_by_function()` renvoie un compte par fonction, et `missing_functions()` detecte correctement une fonction non couverte.
- [ ] Le rapport affiche le detail du risque le plus critique avec ses 3 coordonnees causales et les modulateurs qui se sont appliques.
- [ ] Un risque irreversible + autonome est correctement re-score (impact et vraisemblance majores, plafonnes a 5).
- [ ] Un seuil de traitement est applique et un tally TREAT/MONITOR/ACCEPT est affiche.
