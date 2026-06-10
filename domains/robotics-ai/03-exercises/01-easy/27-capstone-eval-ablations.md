# J27 — Exercice EASY : success rate à la main + variance sur K seeds

## Objectif

Construire l'intuition du **bruit statistique** d'un success rate :
- comprendre que `success_rate` est un estimateur d'une vraie probabilité,
- savoir calculer un mean et un std *across seeds* à la main,
- distinguer "variance entre rollouts d'un seul seed" et "variance entre seeds".

Tu n'as PAS besoin d'entraîner un modèle. Tu n'as PAS besoin de PyTorch. Juste numpy.

## Consigne

On te donne ci-dessous les résultats bruts de 3 seeds × 20 rollouts pour une policy fictive. Chaque case = 1 si le rollout a réussi, 0 sinon.

```python
results = [
    # seed 0 (20 rollouts)
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    # seed 1
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    # seed 2
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
]
```

Tu dois écrire un petit script Python (ou directement dans un REPL) qui :

1. Calcule `success_rate_per_seed` = liste des 3 taux de succès, un par seed.
2. Calcule `mean_across_seeds` et `std_across_seeds` (la quantité qu'on REPORT dans un papier).
3. Calcule aussi le `mean_across_all_rollouts` (en aplatissant les 3 listes en une seule de 60 booléens) et compare-le à `mean_across_seeds`. Sont-ils égaux ? Sinon pourquoi ?
4. Écris à la main la formule de l'erreur standard binomiale `σ_binom = sqrt(p * (1-p) / N)` avec `p = mean_across_seeds` et `N = total_rollouts = 60`. Compare-la à `std_across_seeds`. Laquelle est plus grande ? Que représente chacune ?

## Criteres de reussite

- [ ] Ton script imprime `success_rate = 0.73 +/- 0.02` (mean ± std across seeds, `ddof=0`) — les taux par seed attendus sont `[0.75, 0.70, 0.75]`.
- [ ] Assert dans le script : `mean_across_seeds == mean_across_all_rollouts` (à 1e-12 près), suivi d'1 phrase en commentaire expliquant pourquoi l'égalité tient quand chaque seed a le même nombre de rollouts (moyenne de moyennes à poids égaux).
- [ ] Tu imprimes `σ_binom` (≈ 0.057 avec p=0.733, N=60) à côté de `std_across_seeds`, avec 1-2 phrases en commentaire sur ce que chacune mesure (variance binomiale par-rollout vs variance systémique entre seeds : initialisation, ordre des batches, etc.).
- [ ] Bonus : le script imprime le verdict du test "vraie success rate = 50%" en comparant `|p - 0.5|` à `1.96 * σ_binom`, et conclut explicitement (rejeté ou non à 95%).

## Indices

- `np.mean(boolean_list)` retourne directement le success rate.
- `np.std(seed_rates, ddof=0)` pour l'écart-type biaisé (suffisant ici).
- Si tu hésites entre `ddof=0` et `ddof=1`, dans un capstone on prend `ddof=0` parce qu'on a très peu de seeds (3) et l'estimateur biaisé est plus stable. Le papier Diffusion Policy ne précise pas — c'est un détail mineur.

## Pour aller plus loin (5 min)

Que se passe-t-il si tu remplaces les 3 seeds × 20 par 1 seed × 60 ? Quelles métriques peux-tu encore reporter ? Lesquelles ne tiennent plus debout ?
