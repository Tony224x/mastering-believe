# 05 — Projets guides (Algorithmie Python)

> Voir `shared/logistics-context.md` pour le contexte metier de LogiSim.

Cette section applique les patterns algorithmiques vus en theorie a des problemes reels du moteur de simulation FleetSim. Chaque projet est un cas simplifie mais realiste de ce qu'un AI Engineer chez LogiSim peut rencontrer.

## Projets

| # | Projet | Concepts couverts | Difficulte |
|---|---|---|---|
| 01 | **Pathfinding entrepot A*** | graphes, heuristique, priority queue, couts variables d'allee | medium |
| 02 | **Couverture sensorielle (Bresenham)** | algo de ligne, grille 2D, early-exit, profiling | medium |
| 03 | **Operations event queue** | heap, simulation discrete-event, invariants | hard |

## Methodologie

Chaque projet suit :
1. Lire le **contexte metier** (pourquoi LogiSim a ce probleme)
2. Implementer une **v0 naive** sans regarder la correction
3. Confronter la v0 a la **correction commentee**
4. Mesurer l'ecart de performance avec le benchmark fourni
5. Appliquer les **extensions** pour se rapprocher du niveau production

## Contrainte rouge

Un moteur de simulation logistique doit etre **deterministe** : meme seed et memes entrees = memes sorties, sinon le rejeu pour EOD Review et la reconstitution d'incident sont casses. Chaque projet rappelle comment preserver ce determinisme (tri stable, iteration ordonnee des dict, pas de set non ordonne pour les tie-breakers).
