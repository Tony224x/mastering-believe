# Exercice (difficile) — Rapport d'oversight board-ready avec score de gouvernance

## Objectif

Assembler les briques (RACI + Three Lines) en un **rapport de gouvernance organisationnelle** qu'un comité IA pourrait lire. Tu calcules un **score de couverture de redevabilité** sur toute la flotte, tu classes les agents par criticité de leurs trous, et tu produis un résumé chiffré du type de ceux qu'attend un board ([McKinsey, 2025] : fixer le mandat et **vérifier qu'il est tenu**).

## Consigne

1. Crée `solution.py` dans ton `workspace/`. Réutilise (ou réécris) le résolveur RACI + Three Lines de l'exercice moyen.
2. Pour chaque agent, calcule une liste de **gaps** avec une **sévérité** :
   - `critical` : zéro Accountable, **ou** plusieurs Accountables, **ou** un `actor_id` du RACI absent de l'org chart (responsabilité fantôme) ;
   - `warning` : pas de 1re ligne ; ou agent `high` sans 3e ligne.
3. Écris `governance_score(fleet, actors)` qui renvoie un entier de **0 à 100** :
   - base = part d'agents ayant **exactement un Accountable connu** (en %) ;
   - applique une **pénalité** : `-10` par gap `critical` et `-3` par gap `warning`, plancher à `0`.
4. Écris `rank_agents(fleet, actors)` qui trie les agents par criticité **décroissante** (d'abord ceux qui ont des gaps `critical`, puis `warning`, puis les sains), pour produire une « pile de remédiation ».
5. Produis un rapport texte (ou markdown) avec : le **score global**, le **nombre d'agents sans Accountable clair**, le **total de gaps par sévérité**, et la **liste ordonnée** des agents à remédier avec leurs gaps.
6. Inclus dans ton jeu de test au moins : 1 agent sain, 1 avec responsabilité fantôme (acteur inconnu), 1 avec double Accountable, 1 high sans 3e ligne.

## Criteres de reussite

- [ ] `governance_score` renvoie un entier borné dans `[0, 100]` (jamais négatif).
- [ ] Un agent référençant un `actor_id` inexistant génère un gap `critical` « responsabilité fantôme ».
- [ ] `rank_agents` place tous les agents à gap `critical` **avant** ceux à `warning`, et les sains en dernier.
- [ ] Le rapport affiche : score global, agents sans Accountable, totaux gaps `critical`/`warning`, pile de remédiation ordonnée.
- [ ] Une **probe adverse** : une flotte vide (`[]`) ne plante pas et renvoie un score défini (convention : `100`, rien à reprocher — documente ton choix).
- [ ] Le script tourne en **stdlib seule** avec `python solution.py` (exit 0).
