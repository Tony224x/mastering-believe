# Exercice (moyen) — Mapper la flotte sur le Three Lines Model

## Objectif

Étendre le RACI avec la dimension **Three Lines Model** [IIA, 2020] : chaque acteur appartient à une ligne (1 = opérationnel, 2 = risque/conformité, 3 = audit interne, ou board). Tu construis un résolveur qui, pour chaque agent, indique quelles lignes sont couvertes et signale deux trous classiques : **pas de 1re ligne** (aucun opérateur) et **agent à haut risque sans 3e ligne** (pas d'assurance indépendante).

## Consigne

1. Crée `solution.py` dans ton `workspace/`.
2. Modélise les acteurs dans un dictionnaire `actor_id -> line`, où `line ∈ {"1", "2", "3", "board"}`. Exemple :
   - `"sam": "1"`, `"lea": "1"`, `"kim": "2"`, `"max": "3"`, `"board": "board"`.
3. Modélise chaque agent par `{"risk_tier": ..., "raci": {actor_id: role}}` avec `risk_tier ∈ {"minimal", "limited", "high"}`. Prévois au moins :
   - un agent **high** correctement couvert (1re et 3e lignes présentes) ;
   - un agent **high** **sans** 3e ligne ;
   - un agent **minimal** sans 1re ligne.
4. Écris `lines_covered(agent, actors)` qui renvoie l'ensemble des lignes présentes dans le RACI de l'agent (en ignorant les acteurs inconnus).
5. Écris `find_line_gaps(agent, actors)` qui renvoie une liste de messages :
   - si la ligne `"1"` est absente → `"no first line"` ;
   - si `risk_tier == "high"` et ligne `"3"` absente → `"high-risk without third line"`.
6. Imprime, par agent : les lignes couvertes (triées) puis ses trous (ou `aucun`).

## Criteres de reussite

- [ ] `lines_covered` ignore les `actor_id` qui ne sont pas dans le dictionnaire des acteurs.
- [ ] `find_line_gaps` ne déclenche le trou « third line » **que** pour les agents `high`.
- [ ] L'agent high bien couvert ne remonte **aucun** trou.
- [ ] L'agent high sans 3e ligne remonte exactement `"high-risk without third line"`.
- [ ] L'agent minimal sans 1re ligne remonte `"no first line"` (et **pas** de trou « third line »).
- [ ] Le script tourne en **stdlib seule** avec `python solution.py` (exit 0).
