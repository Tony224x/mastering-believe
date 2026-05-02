# J13 - Exercice EASY : Behavior Cloning sur expert deterministe

## Objectif

Implementer une boucle Behavior Cloning minimale et identifier *empiriquement* l'effet du nombre de demos sur la performance du student.

## Consigne

Sur `CartPole-v1` (Gymnasium) :

1. Reprendre l'expert heuristique du cours :
   ```python
   def expert_action(obs):
       _, _, theta, theta_dot = obs
       return 1 if (theta + 0.5 * theta_dot) > 0 else 0
   ```
2. Generer trois datasets de demos : `n = 1`, `n = 5`, `n = 50` episodes.
3. Pour chaque dataset, entrainer un MLP `4 -> 64 -> 2` par cross-entropy (Adam, lr=1e-3, ~30 epochs).
4. Evaluer chaque policy sur 20 episodes (seeds disjointes du train), reporter le return moyen.
5. Tracer (terminal `print` ou matplotlib) la courbe `n_demos -> return moyen`.

## Criteres de reussite

- `n=1`  : return moyen typiquement < 200 (BC fragile).
- `n=5`  : return moyen autour de 200-450 selon seed.
- `n=50` : return moyen ≈ 500 (CartPole resolu).
- Tu commentes en 2-3 phrases pourquoi la courbe monte avec `n_demos` et fais le lien avec la borne O(T² ε) du cours (`[Ross & Bagnell, 2010]`).

## Indices

- Reutilise un MLP simple, pas besoin de regularisation.
- Penser a fixer les seeds (`torch`, `numpy`, `env.reset(seed=...)`) pour avoir des chiffres reproductibles.
- `env.spec.max_episode_steps` = 500 pour CartPole-v1.

## Pour aller plus loin (optionnel)

Refais l'experience en **bruitant l'expert** : avec proba 0.1, action aleatoire. Surprise : la performance BC peut s'**ameliorer** avec peu de demos. Pourquoi ? (reponse dans `[Zare et al., 2024]`).
