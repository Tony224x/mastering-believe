# J17 — Exercice medium : faire varier l'imagination horizon et tracer la qualité

## Objectif

Mesurer empiriquement comment la qualité du *latent imagination rollout* dépend de l'horizon `H`. Tu dois sentir physiquement que plus on rêve loin, plus le modèle dérive.

## Consigne

1. Pars du fichier `02-code/17-world-models-dreamer.py`. Entraîne le world model 60 itérations comme d'habitude, en notant les checkpoints intermédiaires.
2. Écris une fonction `imagination_drift(wm, env, H, n_runs=20)` qui :
   - reset l'env, fait rouler une politique aléatoire pendant `H` steps en collectant les vrais latents (via `observe_step`).
   - Reset l'env, fait rouler la **même séquence d'actions** mais cette fois en imagination pure (`imagine_step`) après avoir encodé seulement la première observation.
   - Compare les deux trajectoires de latents step par step en distance L2 entre les `h_t` (déterministe).
   - Retourne un array shape (H,) avec la dérive moyenne sur `n_runs` runs.
3. Trace la dérive pour `H ∈ {1, 2, 5, 10, 20}` et commente. Question pédagogique : à partir de quel H la dérive explose-t-elle ?
4. Bonus : refais l'expérience avec `kl_weight=0.1` puis `kl_weight=5.0` dans la `Config` et compare. Que voit-on ?

## Criteres de reussite

- `imagination_drift(wm, env, H, n_runs=20)` retourne un array de shape exactement `(H,)`, sans NaN, avec `drift[0] >= 0`.
- Monotonie vérifiée numériquement : pour `H = 20`, `drift[-1] > drift[0]` et la dérive moyenne sur les 5 derniers steps est strictement supérieure à celle des 5 premiers (sinon, ton modèle imagine ou observe mal — debug d'abord).
- Le plot montre les 5 valeurs de `H ∈ {1, 2, 5, 10, 20}` et tu as noté par écrit le `H` à partir duquel la dérive dépasse 2× la dérive à `H=1`.
- Bonus quantifié : avec `kl_weight=0.1`, la dérive à `H=20` est strictement supérieure à celle du baseline (`kl_weight` par défaut) — et ta note d'1-2 phrases l'explique avec les mots « prior mal aligné sur le posterior » et fait le lien avec la théorie §6 (« model bias »).
