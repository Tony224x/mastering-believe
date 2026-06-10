# Exercice facile — J19 VLA introduction

## Objectif

Manipuler la **discrétisation d'action** utilisée par les VLAs RT-1 / RT-2 / OpenVLA, pour intérioriser le passage continu ↔ tokens.

## Consigne

Tu disposes d'une plage continue `[-0.05, +0.05]` (mètres) que tu veux discrétiser en `N` bins uniformes. C'est exactement le mécanisme que RT-1 utilise sur 256 bins par dimension d'action.

1. Écris une fonction `discretize(value: float, n_bins: int, lo: float, hi: float) -> int` qui mappe une valeur continue clipée dans `[lo, hi]` vers un id de bin entier dans `[0, n_bins-1]`.
2. Écris la fonction inverse `undiscretize(bin_id: int, n_bins: int, lo: float, hi: float) -> float`.
3. Vérifie sur 5 valeurs de ton choix (dont les bornes `lo` et `hi`) que `undiscretize(discretize(v)) ≈ v` à la précision d'un bin près. Affiche l'erreur d'arrondi (quantization error) maximale.
4. Calcule la **résolution** du discretizer pour `N=256` et plage `[-0.05, 0.05]` : combien de millimètres représente un bin ? Compare avec `N=16`.

## Criteres de reussite

- Les fonctions sont pures (pas de side effect), pas de dépendance hors stdlib + numpy/torch.
- Asserts qui passent : `discretize(lo) == 0`, `discretize(hi) == n_bins - 1`, et `abs(undiscretize(discretize(v)) - v) <= (hi - lo) / (2 * (n_bins - 1))` pour 1000 valeurs aléatoires dans `[lo, hi]`.
- Le script imprime la résolution en millimètres pour `N=256` (~0.39 mm/bin sur `[-0.05, 0.05]`) et pour `N=16` (~6.7 mm/bin).
- En commentaire de fin de script, tu as écrit en 1-2 phrases pourquoi `N=256` suffit pour le pick-and-place mais pas pour la chirurgie, en t'appuyant sur tes résolutions calculées (≈ 0.4 mm vs la précision sub-0.1 mm requise en chirurgie).

## Indices

- L'erreur de quantization maximale d'un binning uniforme à N bins sur `[lo, hi]` vaut `(hi - lo) / (2*(N-1))`.
- Les bornes incluses : `discretize(lo) == 0` et `discretize(hi) == N-1`.
