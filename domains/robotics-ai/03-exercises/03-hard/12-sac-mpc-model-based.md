# Exercice J12 (hard) — Mini-Dyna : SAC + modele appris pour booster la sample efficiency

## Objectif

Implementer une version simplifiee de Dyna : entrainer en parallele (1) une policy par TD-learning sur l'env reel et (2) un modele appris `f_hat(s, a) -> s'` qui sert a generer des rollouts imagines pour des updates supplementaires. C'est l'idee qui sous-tend MBPO (Janner 2019) et Dreamer.

## Consigne

Sur `Pendulum-v1` (Gymnasium) ou un env continu equivalent :

1. **Modele dynamique appris**. Ecris une petite MLP (PyTorch ou numpy + autograd manuel) qui prend `(s, a)` et predit `s' - s` (delta state, plus stable que predire `s'` direct). Loss MSE.

2. **Boucle Dyna-style**. A chaque step :
   - Step env reel, ajouter `(s, a, r, s', done)` au replay buffer.
   - Update policy via SGD/TD sur batch reel (utilise SAC de SB3 OU une mini-implementation actor-critic, peu importe).
   - Update modele `f_hat` sur batch du buffer reel.
   - **Step Dyna** : sampler `K` paires `(s_i, a_i)` du buffer, predire `s'_i = s_i + f_hat(s_i, a_i)` et `r_i` (donne par formule connue ou aussi appris), faire `K` updates policy supplementaires sur ces transitions imaginees.

3. Compare deux configurations :
   - **Baseline** : SAC pur, 5000 steps reels.
   - **Dyna-SAC** : SAC + modele appris, 5000 steps reels mais `K=5` updates imagines par step (donc 6× plus d'updates policy).

4. Trace la courbe d'apprentissage (reward moyen par episode) pour les deux. Quel mode atteint -200 (seuil "ca marche" sur Pendulum) le plus vite en steps reels ?

5. **Ablation**. Que se passe-t-il si tu fais `K=20` (rollouts longs) au lieu de `K=5` ? Et si l'horizon imagine fait 1 step ou 10 steps ?

Reponds aux questions :
- Pourquoi l'horizon imagine doit rester court (cf. theorie MBPO) ?
- Quel cout payes-tu en compute en echange du gain en sample efficiency ?
- Si tu avais un robot reel ou chaque rollout coute 30s, lequel des deux choisirais-tu ? Justifie.

## Criteres de reussite

- Script Python complet, reproductible, qui trace les deux courbes.
- Mesure quantitative : "Dyna-SAC atteint reward = X en N steps reels, vs M steps pour SAC baseline".
- Tu peux relier ton experience au papier MBPO (`[Janner et al., 2019]`) et expliquer ce que les **ensembles** apporteraient en plus.
- Bonus : ajouter un ensemble de 5 modeles, prendre la variance des predictions comme bonus d'incertitude. Affecte-t-il la stabilite ?

## Indices

- Predire `delta_s = s' - s` plutot que `s'` reduit drastiquement la difficulte (le delta est centre proche de 0).
- La reward de Pendulum-v1 est analytique : `-(theta**2 + 0.1*theta_dot**2 + 0.001*u**2)` avec `theta = angle_normalize(...)`. Pas besoin de l'apprendre.
- `K=5` rollouts d'horizon 1 = 5 updates supplementaires par step reel. Suffisant pour observer le speedup.
- Si SAC SB3 te facilite la vie, tu peux acceder a `model.policy.actor`, `model.policy.critic`, `model.replay_buffer` directement.
