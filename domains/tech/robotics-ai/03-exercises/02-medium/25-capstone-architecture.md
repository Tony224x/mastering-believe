# Exercice medium J25 — DDIM vs DDPM et comparaison qualite/vitesse

## Objectif

Implementer un **DDIM scheduler** (deterministe) et le comparer au **DDPM scheduler** fourni : qualite du denoising et nombre d'etapes necessaires a l'inference.

DDIM (Song et al. 2020, "Denoising Diffusion Implicit Models") reformule le reverse process pour le rendre **non-markovien** et **deterministe** (eta=0), ce qui permet de sauter des etapes. Diffusion Policy l'utilise typiquement pour passer de 100 a 16 etapes a l'inference.

## Consigne

1. Creer `DDIMScheduler` dans un nouveau fichier `25-medium-ddim.py`. Reutiliser les memes betas/alphas que `DDPMScheduler`.
2. Implementer `step_ddim(eps_pred, x_t, t, t_prev)` selon la formule :

    ```
    x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
    x_{t_prev} = sqrt(alpha_bar_{t_prev}) * x0_hat
                 + sqrt(1 - alpha_bar_{t_prev}) * eps_pred       (eta=0)
    ```

3. Construire une **denoising trajectory** fictive : on entraine pas la policy, mais on peut tester sur des actions synthetiques. Procedure :
   - Echantillonner `a_0 ~ N(0, I)` de shape `(1, T_act, action_dim)`.
   - Forward jusqu'a `a_T` via `q_sample`.
   - Reverse en utilisant **(a)** DDPM (100 etapes) puis **(b)** DDIM avec 10, 25, 50 etapes.
   - Pour chaque cas, mesurer `MSE(a_0_predit, a_0)` et le temps wall-clock.
4. Tracer un tableau :

   | Scheduler | Steps | MSE vs a0 | Temps (s) |
   |-----------|-------|-----------|-----------|
   | DDPM      | 100   |    ?      |    ?      |
   | DDIM      |  50   |    ?      |    ?      |
   | DDIM      |  25   |    ?      |    ?      |
   | DDIM      |  10   |    ?      |    ?      |

5. Repondre : a partir de combien d'etapes DDIM la qualite se degrade significativement ?

## Criteres de reussite

- DDIM implemente sans bug (pas de NaN, shapes coherentes).
- Le tableau est rempli, le temps DDIM 25 steps est `~4x` plus rapide que DDPM 100 steps.
- Conclusion ecrite (3-5 phrases) explique le **trade-off vitesse/qualite** observe et **pourquoi** la formule DDIM economise du calcul (deterministe, pas de bruit re-injecte, sous-grille temporelle).

## Bonus

- Tester `eta` intermediaire (0.0 / 0.5 / 1.0) : `eta=1` doit reproduire DDPM.
- Verifier sur des **actions reelles** issues du dataset J24 (si dispo) plutot que des fakes.
