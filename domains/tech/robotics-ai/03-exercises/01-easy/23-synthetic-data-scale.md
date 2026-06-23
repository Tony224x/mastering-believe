# J23 — Exercice EASY : implémenter une augmentation "lighting" + une "background" jouet

## Objectif

Vérifier que tu as compris :
- ce qu'**augmente** une augmentation "lighting" (gain + temperature) sur une image,
- ce qu'**augmente** une augmentation "background" (swap d'arrière-plan préservant l'objet),
- pourquoi ces deux augmentations sont les **moins chères** du pipeline GR00T (REFERENCES.md #15).

Tu n'as pas besoin de PyTorch — NumPy suffit.

## Consigne

Implémente deux fonctions pures (pas de side-effects) sur une image RGB `img : np.ndarray` de shape `(H, W, 3)` valeurs dans `[0, 1]` :

1. `lighting_augment(img, gain, temperature, rng=None) -> np.ndarray`
   - applique `out = clip(img * gain + temperature, 0, 1)`,
   - `gain` : scalaire, ex. `0.7` ou `1.3`,
   - `temperature` : array `(3,)` à ajouter par canal, ex. `[0.05, -0.02, 0.03]`,
   - retourne une **nouvelle** image.

2. `background_swap(img, new_color, gray_threshold=1e-3) -> np.ndarray`
   - identifie les pixels "background" : ceux dont les 3 canaux sont à `0.5 ± gray_threshold`,
   - remplace ces pixels par `new_color` (array `(3,)`),
   - **préserve** les autres pixels (objets, robot, distractors) tels quels,
   - retourne une nouvelle image.

Puis écris une mini-démo qui :
1. Génère une image jouet 8x8 RGB en gris uniforme `0.5`, sauf un pixel rouge `[1, 0, 0]` au centre.
2. Applique `lighting_augment(img, gain=0.6, temperature=[0.1, 0.0, 0.0])` → vérifie que le pixel central reste plus rouge que le fond.
3. Applique `background_swap(img, new_color=[0.2, 0.2, 0.8])` → vérifie que le fond est devenu bleu et que le pixel central reste rouge.

## Étapes suggérées

1. Écrire les 2 fonctions dans un fichier `solution_easy.py` (ou en interactif).
2. Construire l'image jouet : `img = np.full((8, 8, 3), 0.5); img[4, 4] = [1, 0, 0]`.
3. Tester `lighting_augment` avec différents gains et observer que la dynamique est compressée (gain < 1) ou dilatée (gain > 1).
4. Tester `background_swap` et compter le nombre de pixels modifiés (`np.sum((before != after).any(axis=-1))`).

## Critères de réussite

- [ ] `lighting_augment` retourne une image strictement entre `0` et `1` (pas de NaN, pas de dépassement).
- [ ] Avec `gain=0.6, temperature=[0.1, 0, 0]`, le canal R du pixel central reste **strictement supérieur** au canal R du fond (l'augmentation préserve la séparabilité objet/fond).
- [ ] `background_swap` modifie **exactement** les pixels gris (pas le pixel central rouge).
- [ ] Tu peux expliquer en 1 phrase pourquoi le **lighting** est l'augmentation préférée des labos quand le compute est limité.

## Indices

- `np.clip(x, 0, 1)` pour borner les valeurs.
- Pour identifier les pixels gris : `mask = np.all(np.abs(img - 0.5) < threshold, axis=-1, keepdims=True)`.
- `np.where(mask, new_color, img)` fait le swap conditionnel élégamment.
- Le `rng` argument est optionnel ici — on passe `gain` et `temperature` explicitement pour tester déterministiquement.
