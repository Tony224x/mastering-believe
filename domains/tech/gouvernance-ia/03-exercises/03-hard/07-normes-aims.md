# Exercice (hard) — Crosswalk avec verdict de conformite et PDCA "Act"

## Objectif

Etendre le crosswalk en un mini-moteur de conformite qui : (a) distingue les trous **obligatoires** des trous **volontaires**, (b) emet un **verdict** legal, et (c) propose la prochaine etape **Act** du PDCA (quel controle implementer en priorite). C'est ce que produit un AIMS a l'etape *Check -> Act*.

## Consigne

1. Repars du crosswalk du niveau medium (catalogue d'exigences + controles + couverture).
2. Ajoute a chaque exigence un attribut `mandatory: bool` et un poids de priorite (ex. obligatoire = 3, volontaire = 1).
3. Ecris `verdict(controls) -> str` qui renvoie :
   - une **non-conformite legale** s'il existe au moins un trou `mandatory=True` (en indiquant le nombre) ;
   - sinon, un statut conforme-au-legal precisant que les trous restants sont volontaires.
4. Ecris `next_action(controls) -> dict` qui renvoie la **prochaine etape Act** recommandee : le referentiel/exigence prioritaire a couvrir (trous obligatoires d'abord, puis volontaires par poids), avec un champ `reason`.
5. Ajoute une **probe adversariale** : injecte dynamiquement une nouvelle exigence obligatoire **non couverte**, recalcule, et montre que le verdict bascule en non-conformite et que `next_action` la cible.
6. Affiche un rapport board-ready complet (crosswalk + couverture par referentiel + trous + verdict + next_action).
7. Verifie que `python solution.py` tourne sans erreur (stdlib seule).

## Criteres de reussite

- [ ] `verdict` distingue correctement non-conformite legale (>=1 trou obligatoire) et conforme-au-legal.
- [ ] `next_action` priorise les trous **obligatoires** avant les volontaires et inclut un `reason`.
- [ ] La probe adversariale (exigence obligatoire injectee non couverte) fait basculer le verdict en non-conformite.
- [ ] Le rapport est deterministe (sortie identique d'un run a l'autre, tri stable).
- [ ] Aucune dependance externe ; `python solution.py` s'execute sans erreur en stdlib pure.
