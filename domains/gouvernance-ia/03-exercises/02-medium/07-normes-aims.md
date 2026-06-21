# Exercice (medium) — Construire un mini-crosswalk et calculer la couverture

## Objectif

Implementer un **crosswalk** simple : mapper des controles internes vers des exigences de plusieurs referentiels, puis calculer la **couverture** par referentiel et lister les **trous** (exigences non couvertes). C'est le mecanisme central du jour.

## Consigne

1. Dans `workspace/solution.py`, definis un catalogue d'au moins **6 exigences** reparties sur 3 referentiels : EU AI Act (obligatoire), NIST AI RMF (volontaire), ISO/IEC 42001 (volontaire). Chaque exigence a une cle stable `framework::ref`.
2. Definis au moins **2 controles internes**, chacun couvrant un **set** de cles d'exigences. Laisse volontairement **au moins une exigence non couverte**.
3. Ecris `coverage_by_framework(controls) -> dict[str, tuple[int, int]]` renvoyant, par referentiel, `(nb_couvertes, nb_total)`.
4. Ecris `gaps(controls) -> list[...]` renvoyant les exigences couvertes par **aucun** controle.
5. Affiche un rapport lisible : pour chaque referentiel le ratio `couvertes/total` et le pourcentage ; puis la liste des trous avec leur statut (obligatoire/volontaire).
6. Verifie que `python solution.py` tourne sans erreur (stdlib seule).

## Criteres de reussite

- [ ] Au moins 6 exigences sur 3 referentiels, dont l'EU AI Act marque `mandatory=True`.
- [ ] `coverage_by_framework` renvoie des ratios coherents avec les controles definis.
- [ ] `gaps` renvoie exactement les exigences non couvertes (ni plus, ni moins).
- [ ] Le rapport affiche un pourcentage de couverture par referentiel.
- [ ] Les trous sont marques obligatoire ou volontaire.
- [ ] Le script s'execute sans erreur en stdlib pure.
