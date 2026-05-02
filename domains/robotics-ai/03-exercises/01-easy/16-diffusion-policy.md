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

## Critères de réussite

- Aux deux extrêmes (`k=0` et `k=K-1`), les courbes correspondent à `a^0` propre et à du pur bruit gaussien (sanity-check visuel).
- `linear` détruit la structure plus tôt (vers `k ≈ 30-40`) ; `squared cosine` la préserve plus longtemps (jusqu'à `k ≈ 50-60`).
- Tu peux expliquer en 2 phrases pourquoi Chi 2023 préfère squared cosine pour des **séquences d'actions courtes** (T=16) plutôt qu'une schedule linéaire conçue pour des images de taille 64×64.
