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

## Criteres de reussite

- Le code passe `python -m py_compile`.
- Après ~5 epochs, la **per-token accuracy** sur la val depasse 0.85 (imprimée à chaque epoch).
- Le décodage affiché pour 2 échantillons de val montre une séquence de K=4 actions **cohérente** : les 4 pas vont dans la même direction (hors clipping en bord de grille), et correspondent à l'instruction.
- La loss imprimée est bien la moyenne de `K * 2 = 8` cross-entropies (vérifiable : shape des logits `(batch, K, N_ACTION_DIMS, N_BINS)` ou équivalent, assert dans le code).
- En commentaire de fin de script, tu as écrit en 2 phrases pourquoi le chunking améliore la stabilité, en citant au moins 2 des 3 mécanismes : moins de re-planification, cohérence temporelle, distribution shift atténué.

## Indices

- Pour le clipping en bord de grille : si l'arrow est en `(0, 0)` et l'instruction est `up`, les pas suivants restent à `(0, 0)` → action `(0, 0)` à discrétiser au bin central.
- Garder `N_BINS=16` pour rester rapide sur CPU.
- Tu peux utiliser un seul `nn.Linear(D_MODEL, K * N_ACTION_DIMS * N_BINS)` puis reshape, ou une `nn.ModuleList` de K*N_ACTION_DIMS heads — choix au goût.
