# Exercice J16 — Hard : DDIM sampler + receding horizon + ablation action chunking

## Objectif

Implémenter trois optimisations clés du repo `real-stanford/diffusion_policy` (REFERENCES.md #19) et reproduire deux ablations du papier (§6) :

1. **DDIM sampler** déterministe (Song 2020) pour réduire 100 → 16 steps en inférence.
2. **Receding horizon execution** (T_p=16, T_a=8) sur un environnement jouet.
3. **Ablation action chunking** : montrer que `T_p=1` dégrade les performances même avec diffusion.

## Consigne

### Partie 1 — DDIM sampler

1. Implémente `ddim_sample(model, encoder, obs, sched, n_inference_steps=16)` :
   - Sous-échantillonne uniformément `n_inference_steps` indices parmi `[0, K-1]`.
   - Applique la formule DDIM déterministe (η=0) : `a^{k_prev} = √(ᾱ_{k_prev}) · â^0 + √(1-ᾱ_{k_prev}) · ε_θ`, avec `â^0 = (a^k - √(1-ᾱ_k)·ε_θ) / √(ᾱ_k)`.
2. Compare visuellement les samples DDIM (16 steps) vs DDPM (100 steps) sur le dataset bimodal — ils doivent être qualitativement identiques.
3. Mesure la latence d'un appel `ddim_sample` (batch 1) vs `ddpm_sample` sur ton CPU/GPU. Reporte le facteur d'accélération.

### Partie 2 — Receding horizon sur un env jouet

1. Construis un mini-environnement 2D où l'objectif est de faire passer un point de `(0, 0)` à `(1, 1)` en évitant un obstacle circulaire au centre. L'observation est la position courante `(x, y)`.
2. Génère 200 démos "expertes" multimodales : moitié contournent par le haut, moitié par le bas.
3. Entraîne Diffusion Policy avec `T_p = 16` (predict horizon).
4. À l'exécution, implémente la boucle :
   ```
   pendant que not_done :
       obs = env.get_obs()
       chunk = ddim_sample(model, encoder, obs)  # [T_p, 2]
       pour i in range(T_a) :  # T_a=8
           env.step(chunk[i])
   ```
5. Mesure le success rate sur 50 rollouts.

### Partie 3 — Ablation action chunking

1. Réentraîne le **même** modèle Diffusion Policy avec `T_p = 1` (i.e. on prédit une action à la fois).
2. Évalue sur 50 rollouts (à T_p=1, T_a=1 forcément).
3. Compare le success rate à la version T_p=16, T_a=8.

## Critères de réussite

- DDIM 16 steps produit visuellement les **deux modes** comme DDPM 100 steps, avec un facteur de speedup d'environ 5-7×.
- L'agent receding horizon résout l'env jouet avec >70% de success rate.
- L'ablation `T_p=1` montre une dégradation **mesurable** (paper Chi 2023 §6 : -20 à -30 points). Si tu observes -10 à -30 points, c'est cohérent.
- Tu peux expliquer pourquoi `T_p=1` dégrade : sans chunking, chaque sample peut switcher de mode entre deux pas (incohérence temporelle, voir théorie §3.1).
