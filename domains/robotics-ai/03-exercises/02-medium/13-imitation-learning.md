# J13 - Exercice MEDIUM : DAgger from scratch + ablation rollouts/iter

## Objectif

Coder DAgger sans copier le code du cours, et mesurer l'impact du parametre `rollouts_per_iter` (nombre de roll-outs student avant chaque retraining).

## Consigne

Sur `CartPole-v1`, partant de **2 demos expert** seulement (volontairement faible) :

1. Implementer la boucle DAgger (cf. cours, section 3.1) :
   - roll-out la policy student courante,
   - relabel chaque etat visite par l'expert,
   - merger avec le dataset cumule,
   - retrain (warm-start sur les poids actuels, Adam).
2. Comparer 3 configurations DAgger (5 iterations chacune) :
   - `rollouts_per_iter = 1`
   - `rollouts_per_iter = 5`
   - `rollouts_per_iter = 20`
3. Pour chacune, logger a chaque iter : taille du dataset, return moyen sur 20 episodes d'eval (seeds disjointes).
4. Tracer / afficher les 3 courbes `iteration -> return moyen`.
5. Repondre : quelle config converge le plus vite **en nombre d'iterations** ? **en transitions cumulees** ?

## Criteres de reussite

- Boucle DAgger fonctionnelle, code lisible (separer "collect", "relabel", "train").
- Distinction claire entre l'**action prise** par le student (qui determine la transition) et l'**action loggee** dans le dataset (celle de l'expert) — c'est le coeur de DAgger.
- Tableau / graphique comparatif lisible.
- Discussion : `rollouts_per_iter` plus grand = plus de couverture par iter mais moins de feedback loop ; trade-off classique on-policy.

## Indices

- Garder une seed differente par iteration et par rollout pour eviter l'overfitting trivial sur les memes etats.
- Le warm-start (continuer l'entrainement plutot que reinitialiser) accelere DAgger sans changer la garantie theorique.
- Si tu observes que toutes les configs saturent a 500 tres vite, reduis `n_demos` initiales a 1 ou bruite l'expert.

## Pour aller plus loin (optionnel)

Implementer **HG-DAgger** (`[Kelly et al., 2019]`) : l'expert n'intervient que si la divergence entre student et expert depasse un seuil. Compare le nombre total de queries expert vs DAgger vanilla.
