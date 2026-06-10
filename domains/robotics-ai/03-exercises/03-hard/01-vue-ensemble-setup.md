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

## Criteres de reussite
- `gymnasium.utils.env_checker.check_env(OneDPointEnv())` passe sans warning fatal.
- Le main tourne, imprime les deux moyennes ± std.
- Le script contient `assert env.reset()[0].dtype == np.float32` (et l'assertion passe), avec un commentaire d'1 phrase sur la compatibilité PyTorch (dtype par défaut des tensors, pas de cast en J11+).
- `gym.make("OneDPoint-v0")` fonctionne après `register`.
- Sur 10 rollouts random de `OneDPointEnv` : `total_reward < 0` pour chaque rollout (la reward est une pénalité quadratique, jamais positive) et `steps <= 200` partout (le TimeLimit coupe) — sinon ta dynamique ou ton compteur est faux.
- En commentaire de fin de script, tu as noté en 1-2 phrases pourquoi cet env sera trivial pour PPO, en citant au moins 2 des 3 raisons : reward dense quadratique, dimension faible, dynamique linéaire.
