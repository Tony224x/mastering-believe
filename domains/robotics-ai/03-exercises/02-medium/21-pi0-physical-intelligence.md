# J21 — Exercice MEDIUM : multi-embodiment padding + masked loss

## Objectif

Reproduire un mécanisme clé de π0 : le **padding multi-embodiment**. Tu vas implémenter et entraîner un mini policy MSE (sans flow matching pour l'instant — on isole le concept) sur 3 "robots" aux espaces d'action de **dimensions différentes**, en utilisant la stratégie de padding et de masking de loss.

## Consigne

1. Définis 3 embodiments :
   - Robot A : `dim_action = 2` (ex. base XY).
   - Robot B : `dim_action = 4` (ex. arm 4 DoF).
   - Robot C : `dim_action = 7` (ex. arm 7 DoF UR5e).

2. Choisis `dim_max = 7`. Toutes les actions ground-truth sont **paddées à 7** : les dimensions inutiles sont mises à `0` ET un mask `m ∈ {0, 1}^7` indique quelles dimensions sont valides pour l'embodiment courant.

3. Génère un mini-dataset toy : pour chaque embodiment, l'action ground-truth est une fonction simple (ex. fonction sinus paramétrée par un id). Bash ce qui compte c'est la **shape** et le **mask**, pas la sémantique.

4. Construis un MLP simple `policy(context, embodiment_id) → action ∈ R^7`.

5. Entraîne avec une **loss MSE masquée** :
   ```
   L = mean( m * (pred - target)^2 )
   ```
   et **non** `mean((pred - target)^2)` brute (sinon le réseau apprend à prédire 0 sur les dimensions inutiles, ce qui n'est pas ce qu'on veut côté généralisation).

6. À l'évaluation, vérifie pour chaque embodiment que la MSE *sur les dimensions valides* est faible et que les **dimensions paddées sont ignorées**.

## Étapes suggérées

1. Écrire la fonction `make_targets_and_masks(embodiment_id, batch_size, dim_max)` → `(targets, masks)` shape `(B, dim_max)`.
2. Construire le MLP : `nn.Linear(context_dim + n_emb, 64) → ReLU → Linear(64, dim_max)`.
3. Boucle d'entraînement classique avec la masked loss.
4. Tableau d'évaluation : embodiment | MSE-valid-dims | MSE-padded-dims.

## Critères de réussite

- [ ] Le code tourne sans crash, fait au moins 1000 étapes d'entraînement.
- [ ] Pour chaque embodiment, la MSE sur les dimensions valides converge à `< 0.05`.
- [ ] La MSE sur les dimensions paddées peut être *quelconque* (le modèle a le droit d'y produire n'importe quoi puisqu'elles sont masquées) — mais la masked loss à la fin doit être ≈ 0.
- [ ] Tu peux expliquer en 2 phrases pourquoi cette approche permet à π0 d'entraîner un seul checkpoint sur 7 robots avec des espaces d'action différents.

## Pièges classiques

- **Oublier le mask dans la loss** → le réseau "apprend" à prédire 0 sur les dimensions inutiles et ses prédictions deviennent moins riches (saturation des gradients sur les dimensions paddées).
- **Mettre les targets paddés à une grande valeur** (`-1` ou `inf`) → ça pollue le réseau si le mask est mal appliqué. Conventionellement les targets paddés sont à 0 (mais peu importe puisque masquées).
- **Confondre `mean` et `sum`** sur la masked loss : préférer `(m * sq).sum() / m.sum().clamp_min(1)` pour avoir une moyenne propre par batch.
