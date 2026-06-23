# J21 — Exercice HARD : reproduire un mini π0 + benchmarker FM vs DDPM

## Objectif

Combiner les concepts du jour pour produire ta propre **mini foundation policy** style π0 :
- VLM mock (mini-MLP),
- multi-embodiment avec padding,
- flow matching action head **et** DDPM action head pour comparaison rigoureuse,
- évaluation : courbe **MSE-vs-inference-steps** et **walltime-vs-steps**.

C'est l'exercice qui matérialise réellement la valeur de π0.

## Consigne

1. Construis 3 embodiments dont au moins 2 ont des `dim_action` différentes (réutilise le pattern de l'exercice MEDIUM). Padding à `dim_max`.

2. Implémente :
   - une `MockVLM(MLP)` qui prend `(embodiment_id, instruction_id)` (deux one-hot) → `h ∈ R^32`.
   - un `ActionExpert(MLP)` qui prend `(A_t, t_emb, h)` → `A_pred ∈ R^{H × dim_max}`.

3. Crée deux policies partageant le même backbone `ActionExpert` :
   - `FlowMatchingPolicy` : loss MSE sur `(A_star - A_0)`, sample par Euler N steps.
   - `DDPMPolicy` : loss MSE sur `eps`, schedule linéaire β ∈ [1e-4, 0.02], T=100, sampling ancestral DDPM.

4. **N'oublie pas le mask multi-embodiment dans les deux losses.**

5. Entraîne les deux séparément avec budget identique (mêmes epochs, même lr, même batch).

6. Évalue : pour chaque policy, fais varier `n_steps_inference ∈ {2, 5, 10, 20, 50, 100}` (DDPM ne supporte que ≤ T=100) et trace :
   - MSE sur dimensions valides vs n_steps,
   - walltime moyen par appel `sample()` vs n_steps.

7. Présente le résultat dans un tableau ASCII et **commente** dans la console quel régime favorise chaque méthode.

## Étapes suggérées

1. Pars du fichier `02-code/21-pi0-physical-intelligence.py` du module : il contient déjà la structure FM + DDPM, **sans le multi-embodiment masking**.
2. Ajoute le masking : modifie `make_target_chunk` pour produire des chunks (H, dim_max) selon l'embodiment, et un `mask` (H, dim_max). Modifie les deux losses pour devenir `(mask * sq).mean() / mask.mean()`.
3. Lance `python ton_fichier.py` avec budget réduit (`train_steps=800`) pour itérer vite, puis budget complet (`train_steps=3000`) une fois propre.
4. Génère un tableau Markdown des résultats (ou au minimum un print structuré).

## Critères de réussite

- [ ] Le script tourne end-to-end en < 5 min sur CPU.
- [ ] FM atteint une MSE comparable à DDPM en **5 à 10× moins** de steps d'inférence.
- [ ] La masked loss converge correctement pour les 3 embodiments (pas un seul oublié).
- [ ] Tu produis un tableau de comparaison `(method, steps, MSE, walltime)` lisible.
- [ ] Tu produis un commentaire d'au moins 3 phrases qui interprète :
  1. À partir de combien de steps DDPM "rattrape" FM en qualité ?
  2. Quel est le ratio walltime DDPM(50 steps) / FM(5 steps) ?
  3. Dans quel régime DDPM resterait préférable à FM (s'il y en a un) ?

## Pour aller plus loin (bonus)

- Ajouter un **DDIM-style strided sampling** pour DDPM (échantillonner ts non-uniformes) → DDPM @ 10 steps devrait s'approcher de FM @ 10 steps.
- Comparer les **trajectoires intermédiaires** (snapshot de `A` à τ ∈ {0.0, 0.25, 0.5, 0.75, 1.0} pour FM ; à t/T ∈ similar pour DDPM) et plot 2D : on doit voir FM décrire une **trajectoire droite** entre `A_0` et `A_star` (interpolation linéaire), et DDPM une **trajectoire plus courbée et bruitée**.

## Pièges classiques

- **Bug subtil** : si tu réutilises *le même* `ActionExpert` pour FM et DDPM en partageant les poids, tu auras des résultats foireux. Les deux modèles doivent avoir leurs **propres poids** (deux instances) puisqu'ils prédisent des cibles différentes (vélocité vs bruit).
- **Échelle des t_emb** : pour FM, `tau ∈ [0, 1]` ; pour DDPM, `t ∈ {0..T-1}`. Si tu balances directement à `time_embedding`, ça marche mais les deux modèles voient des distributions très différentes — c'est OK puisque chacun s'entraîne sur sa propre.
- **Mask dim normalization** : `(mask * sq).mean()` divise par le total `B*H*dim_max`, pas par `mask.sum()`. Corrige avec `.sum() / mask.sum().clamp_min(1)` pour une moyenne juste.
