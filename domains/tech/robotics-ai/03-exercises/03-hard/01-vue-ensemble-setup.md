# Exercice J1 — Hard : custom Gymnasium env minimal + comparaison

## Objectif
Construire un environnement Gymnasium minimal **from scratch** qui respecte le contrat moderne (5-tuple step, distinction terminated/truncated, espaces typés), puis comparer les rollouts random sur ton env vs `Pendulum-v1`.

## Consigne
Écris `hard.py` contenant :

1. Une classe `OneDPointEnv(gymnasium.Env)` modélisant un point sur une ligne `[-10, 10]` :
   - **Observation space** : `Box(low=-10, high=10, shape=(1,), dtype=np.float32)` — la position `x`.
   - **Action space** : `Box(low=-1, high=1, shape=(1,), dtype=np.float32)` — la vitesse appliquée.
   - **Dynamique** : `x_{t+1} = x_t + a * 0.1`.
   - **Reward** : `-x_{t+1}**2` (encourage à rester proche de 0).
   - **terminated** : `True` quand `|x| > 10` (sortie de l'arène).
   - **truncated** : `True` quand `t >= 200`.
   - Implémente `reset(seed=None, options=None)` et `step(action)` conformes à `gymnasium.Env`.
2. Une fonction `random_rollout(env, seed)` qui retourne `(steps, total_reward, terminated, truncated)`.
3. Un main qui lance 10 rollouts random sur `OneDPointEnv` et 10 sur `Pendulum-v1`, et imprime un tableau comparatif `mean_reward ± std` pour les deux.
4. Bonus : registre ton env via `gymnasium.register("OneDPoint-v0", entry_point=...)` et instancie-le ensuite via `gym.make("OneDPoint-v0")`.

## Critères de réussite
- `gymnasium.utils.env_checker.check_env(OneDPointEnv())` passe sans warning fatal.
- Le main tourne, imprime les deux moyennes ± std.
- Tu peux justifier pourquoi `dtype=np.float32` est important pour la compatibilité avec PyTorch en aval (J11+).
- `gym.make("OneDPoint-v0")` fonctionne après `register`.
- Tu peux expliquer pourquoi cet env minimal sera trivial à apprendre pour PPO (réponse attendue : reward dense quadratique, dim faible, dynamique linéaire — exactement le cas où PG converge en quelques minutes).
