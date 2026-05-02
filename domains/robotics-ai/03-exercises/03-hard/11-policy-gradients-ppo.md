# J11 — Exercice difficile : PPO continu sur HalfCheetah-v4

## Objectif

Adapter le PPO discret du cours (`02-code/11-policy-gradients-ppo.py`) au cas **action continue** et le faire tourner sur **HalfCheetah-v4** (MuJoCo). Tu dois atteindre un retour moyen > 1500 en moins de 1M steps.

## Consigne

1. **Adapte la policy au continu** (action `dim = 6` pour HalfCheetah) :
   - Le head policy retourne un vecteur `mu` (taille = action_dim).
   - Cree un parametre libre `log_sigma` de shape `(action_dim,)`, **independent de `s`** (state-independent std, comme dans CleanRL).
   - Distribution : `Normal(mu, log_sigma.exp())`. Pour avoir une joint distribution sur les `action_dim` dimensions independantes, somme les log-probs et les entropies sur la derniere dim.

2. **Vec envs** : utilise `gymnasium.vector.SyncVectorEnv` avec `n_envs = 4` ou `8`. Le rollout devient `(n_steps, n_envs)` et tu flattens en `(n_steps * n_envs,)` avant le SGD.

3. **Hyperparametres MuJoCo standard** :
   - `n_steps = 2048`, `n_epochs = 10`, `minibatch_size = 64`
   - `gamma = 0.99`, `gae_lambda = 0.95`, `clip_eps = 0.2`
   - `lr = 3e-4`, decroit lineairement vers 0
   - `total_timesteps = 1_000_000` (ou stoppe plus tot si retour > 2500)
   - Network MLP `[obs_dim -> 64 -> 64 -> action_dim]` avec `tanh`.

4. **Observability** :
   - Log toutes les 10 updates : `mean_return`, `value_loss`, `policy_loss`, `entropy`, `clip_fraction` (proportion de samples ou `r_t` est hors `[1-eps, 1+eps]`), `approx_kl` (KL approxime entre nouvelle et ancienne policy).
   - Plot `mean_return` vs `step`.

5. **Question bonus** : que se passe-t-il si tu fais `n_epochs = 1` ? `n_epochs = 50` ? Explique avec ce que tu sais du clip ratio.

## Criteres de reussite

- Le script tourne en moins de 30 minutes sur CPU 8 cores (ou GPU si dispo).
- Le retour moyen depasse 1500 sur HalfCheetah-v4 a 1M steps.
- Le `clip_fraction` reste typiquement entre 0.1 et 0.3 — sinon le clip eps est trop large ou trop strict.
- Le `approx_kl` reste sous 0.02 — sinon les updates sont trop violents.

## Indices

- Pour calculer `approx_kl` (cheap, sans calculer la vraie KL) :
  ```python
  approx_kl = ((logratio.exp() - 1) - logratio).mean()
  ```
  ([CleanRL] convention).

- **Action clipping** : MuJoCo attend des actions dans `[-1, 1]`. Soit tu wrappes avec `gym.wrappers.ClipAction`, soit tu ne tronques **pas** la distribution Normal mais tu clippes l'action *apres* sample pour le step (et tu calcules la log-prob avec l'action **non-clippee**, sinon tu biaises le gradient).

- **Observation normalization** : `gym.wrappers.NormalizeObservation` + `gym.wrappers.NormalizeReward` ameliorent dramatiquement la convergence sur MuJoCo. Garde les stats running mean/std synchronisees entre rollout et evaluation.

- Reference : [CleanRL ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) — single-file, ~500 lignes, exactement ce qu'on construit ici.

## Sources

- [Schulman et al., 2017, PPO]
- [OpenAI Spinning Up, PPO]
- [CleanRL]
