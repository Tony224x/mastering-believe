# Exercice moyen — J19 VLA introduction

## Objectif

Implémenter une variante **multi-étape** du tiny-VLA du module : au lieu de prédire un step `(dx, dy)`, prédire une **séquence d'actions** (chunking) — mécanisme central chez Octo et Diffusion Policy.

## Consigne

Reprendre le code du module (`02-code/19-vla-introduction.py`) et l'étendre pour que :

1. Le dataset génère des trajectoires de **K=4** steps. Pour chaque échantillon, l'instruction reste un mot dans `{up, down, left, right}` et l'arrow doit faire `K=4` pas dans cette direction (en restant dans la grille — clipper si nécessaire).
2. La policy émet **K × 2 = 8 tokens d'action** (un par dim × steps) au lieu de 2.
3. L'action head devient une `nn.Linear(D_MODEL, N_BINS * K * N_ACTION_DIMS)` reshape en `(K, N_ACTION_DIMS, N_BINS)`, ou bien une liste de `K * N_ACTION_DIMS` heads.
4. La loss est la moyenne des cross-entropies sur les `K * 2` tokens.
5. Au moment du décodage, afficher la trajectoire prédite vs vraie pour 1-2 échantillons (par ex. `[(0,-1), (0,-1), (0,-1), (0,-1)]` pour "up").

## Critères de réussite

- Le code passe `python -m py_compile`.
- Après ~5 epochs, la **per-token accuracy** sur la val depasse 0.85.
- Tu sais expliquer pourquoi le chunking (Octo prédit 4 actions, Diffusion Policy 8-16) améliore la stabilité par rapport à une prédiction step-par-step (indice : moins de re-planification, plus de cohérence temporelle, distribution shift atténuée).

## Indices

- Pour le clipping en bord de grille : si l'arrow est en `(0, 0)` et l'instruction est `up`, les pas suivants restent à `(0, 0)` → action `(0, 0)` à discrétiser au bin central.
- Garder `N_BINS=16` pour rester rapide sur CPU.
- Tu peux utiliser un seul `nn.Linear(D_MODEL, K * N_ACTION_DIMS * N_BINS)` puis reshape, ou une `nn.ModuleList` de K*N_ACTION_DIMS heads — choix au goût.
