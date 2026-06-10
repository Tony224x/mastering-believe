# Exercice J16 — Easy : Forward noising et schedule

## Objectif

Comprendre concrètement le **forward process** de DDPM et l'effet du schedule (linear vs squared cosine) sur la décroissance du signal — la première brique de Diffusion Policy (Chi 2023, REFERENCES.md #19).

## Consigne

1. Génère une trajectoire d'actions synthétique `a^0` de forme `[T=16, dim=2]` qui ressemble à une demi-cardioïde (ex : `a^0[t] = (sin(πt/T), t/T)`).
2. Implémente une fonction `forward_noise(a0, k, alphas_cumprod)` qui calcule `a^k = √(ᾱ_k) · a^0 + √(1-ᾱ_k) · ε` avec `ε ~ N(0, I)`.
3. Implémente deux schedules sur `K = 100` steps :
   - **Linear** : `β_k` linéairement de `1e-4` à `0.02`.
   - **Squared cosine** (Nichol & Dhariwal 2021) : formule du papier, déjà donnée dans `02-code/16-diffusion-policy.py`.
4. Pour chaque schedule, plot (matplotlib) la trajectoire bruitée à `k ∈ {0, 25, 50, 75, 99}` (5 sous-plots côte à côte par schedule).
5. Compare visuellement : à `k = 50`, quel schedule conserve le mieux la structure de la trajectoire ?

## Criteres de reussite

- Sanity-check numérique des extrêmes (asserts dans le script) : à `k=0`, `np.abs(a_k - a0).max() < 0.05` pour les deux schedules ; à `k=K-1`, `sqrt(ᾱ_{K-1}) < 0.05` pour la **squared cosine** (signal résiduel < 5%).
- Piège découvert et documenté : pour la **linear** telle que donnée (`1e-4 → 0.02`, plage calibrée pour T=1000), tu imprimes `sqrt(ᾱ_{K-1})` et tu observes `≈ 0.6` — il reste ~60% de signal à la fin, ce n'est PAS du bruit pur. Note d'1 phrase dans le script : quand on réduit K, il faut rescaler `β` (≈ ×10 ici).
- Après rescaling de la linear (`1e-3 → 0.2`), tu vérifies numériquement `sqrt(ᾱ_50)_linear < sqrt(ᾱ_50)_cosine` : la linear détruit la structure plus tôt (visible sur tes sous-plots vers `k ≈ 30-40`), la squared cosine la préserve plus longtemps (`k ≈ 50-60`).
- En 2 phrases écrites en commentaire de fin de script, tu expliques pourquoi Chi 2023 préfère squared cosine pour des séquences d'actions courtes (T=16), en t'appuyant sur tes valeurs de `sqrt(ᾱ_k)` (la linear, calibrée pour des images 64×64, écrase le SNR trop tôt pour un signal aussi peu dimensionnel).
