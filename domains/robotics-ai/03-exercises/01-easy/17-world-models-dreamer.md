# J17 — Exercice facile : symlog et lecture du RSSM

## Objectif

Vérifier ta compréhension du transform `symlog` (DreamerV3) et de la séparation prior/posterior dans la cellule RSSM.

## Consigne

1. Implémente une fonction `symlog(x)` et `symexp(x)` en pur PyTorch (sans regarder le code de J17), puis vérifie que `symexp(symlog(x)) ≈ x` pour `x ∈ [-1000, 1000]`.
2. Affiche la valeur de `symlog` pour `x = -1000, -1, 0, 1, 1000`. Confirme à voix haute :
   - le signe est conservé,
   - autour de 0 le transform est ≈ identité (pourquoi `log(|x|+1)` et pas `log|x|` ?),
   - les grandes magnitudes sont écrasées de façon lisse.
3. Dans le code `02-code/17-world-models-dreamer.py`, identifie *par numéro de ligne approximatif* les 3 endroits suivants et écris une phrase pour chacun :
   - où le **prior** est calculé,
   - où le **posterior** est calculé,
   - où le **KL balancing** est appliqué (avec ses deux termes asymétriques).

## Critères de réussite

- `symexp(symlog(x))` retourne x à 1e-5 près sur le range [-1000, 1000].
- Tu peux expliquer en une phrase pourquoi `log(|x|+1)` est préféré à `log|x|` (réponse attendue : éviter la singularité en 0 et garder le transform identité au voisinage).
- Tu identifies sans hésiter prior, posterior, KL balancing dans le code.
