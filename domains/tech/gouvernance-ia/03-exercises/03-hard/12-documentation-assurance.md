# Exercice (hard) — Gate d'assurance : refuser le déploiement sur preuve insuffisante

## Objectif

Transformer la documentation d'assurance en **contrôle de gouvernance actif** : un *gate* qui décide `APPROVE` / `BLOCK` le déploiement d'un agent en croisant la complétude de son agent card et la solidité de son safety case. C'est le pont entre la preuve statique (J12) et l'enforcement (J14) : on **bloque** ce qui n'est pas défendable.

## Consigne

Vous disposez (ou recodez) d'une `agent card` et d'un `safety case` (cf. easy + medium). Construisez un **assurance gate** calibré sur le risque.

1. Écrivez `assess_card(agent) -> tuple[float, list[str]]` : renvoie un score de complétude (0–1) et la liste des champs manquants (owner, permissions, base légale si données perso, éval, limites).
2. Écrivez `assess_case(case) -> dict` renvoyant au minimum : `coverage` (fraction de claims étayés), `n_open_gaps`, et `relies_on_weak_argument` (booléen : vrai si **au moins un** claim repose sur `TRUSTWORTHINESS` ou `DEFERENCE` **sans aucune evidence**).
3. Écrivez `assurance_gate(agent, case, risk_tier) -> dict` qui rend une décision `APPROVE` / `APPROVE_WITH_CONDITIONS` / `BLOCK` selon des seuils **dépendant du tier de risque** :
   - tier `high` : exiger complétude card ≥ 0.8 **et** couverture safety case = 1.0 **et** aucun claim fragile non étayé → sinon `BLOCK`.
   - tier `limited` : complétude ≥ 0.6 et couverture ≥ 0.7 → sinon `APPROVE_WITH_CONDITIONS` (et lister les conditions = champs/claims à corriger).
   - tier `minimal` : `APPROVE` tant que l'owner existe ; sinon `BLOCK`.
4. La décision doit inclure un champ `reasons` (liste lisible) expliquant **pourquoi** — un gate qui bloque sans dire pourquoi est inutilisable en audit.
5. Démontrez le gate sur **trois cas** : (a) un agent high-risk bien documenté → `APPROVE` ; (b) le même agent high-risk mais avec un claim critique sans evidence → `BLOCK` ; (c) un agent limited à moitié documenté → `APPROVE_WITH_CONDITIONS` avec la liste des conditions.

## Critères de réussite

- [ ] `assurance_gate` renvoie `BLOCK` pour un agent **high** dont un claim d'inability/control critique n'a **aucune** evidence, et `APPROVE` quand tout est étayé.
- [ ] Les seuils dépendent réellement du `risk_tier` (un même dossier peut passer en `minimal` et être bloqué en `high`).
- [ ] La décision contient des `reasons` explicites et, pour `APPROVE_WITH_CONDITIONS`, la liste des conditions à lever.
- [ ] `relies_on_weak_argument` détecte un claim `TRUSTWORTHINESS`/`DEFERENCE` sans evidence.
- [ ] Les trois scénarios (a/b/c) sont démontrés et impriment des décisions cohérentes.
- [ ] Le script tourne en **stdlib pure** (`python <fichier>`), sans dépendance externe.
