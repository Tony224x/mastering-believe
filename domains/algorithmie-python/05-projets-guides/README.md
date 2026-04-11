# 05 — Projets guides (Algorithmie Python)

> Voir `shared/masa-context.md` pour le contexte metier de MASA Group.

Cette section applique les patterns algorithmiques vus en theorie a des problemes reels du moteur de simulation SWORD. Chaque projet est un cas simplifie mais realiste de ce qu'un AI Engineer chez MASA peut rencontrer.

## Projets

| # | Projet | Concepts couverts | Difficulte |
|---|---|---|---|
| 01 | **Pathfinding tactique A*** | graphes, heuristique, priority queue, couts variables terrain | medium |
| 02 | **Line-of-sight Bresenham** | algo de ligne, grille 2D, early-exit, profiling | medium |
| 03 | **Combat event queue** | heap, simulation discrete-event, invariants | hard |

## Methodologie

Chaque projet suit :
1. Lire le **contexte metier** (pourquoi MASA a ce probleme)
2. Implementer une **v0 naive** sans regarder la correction
3. Confronter la v0 a la **correction commentee**
4. Mesurer l'ecart de performance avec le benchmark fourni
5. Appliquer les **extensions** pour se rapprocher du niveau production

## Contrainte rouge

Un moteur de simulation tactique doit etre **deterministique** : meme seed et memes entrees = memes sorties, sinon le rejeu pour AAR est casse. Chaque projet rappelle comment preserver ce determinisme (tri stable, iteration ordonnee des dict, pas de set non ordonne pour les tie-breakers).
