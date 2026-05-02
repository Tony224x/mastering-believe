# J27 — Exercice MEDIUM : implémenter une nouvelle métrique (trajectory deviation from expert)

## Objectif

Aller au-delà des 3 métriques canoniques (success rate, ep_len, smoothness) en construisant **ta propre** métrique d'évaluation. C'est ce qu'un eval rigoureux te demande quand tes baselines plateauent à success rate identique mais que tu sens que les trajectoires ne se valent pas.

Tu vas implémenter la **trajectory deviation** : pour chaque rollout, on mesure à quel point la trajectoire de l'agent dévie de la trajectoire qu'un *expert* (oracle) aurait suivie depuis le même état initial.

## Consigne

1. Réutilise (ou re-implémente light) l'environnement toy `ToyPushT` du module `02-code/27-capstone-eval-ablations.py`.

2. Implémente une fonction `trajectory_deviation(env_seed, policy, expert_fn, horizon)` qui :
   - reset `env` avec `env_seed`,
   - simule en parallèle **deux** rollouts depuis le même état initial :
     - rollout A : la `policy` qu'on évalue,
     - rollout B : `expert_fn(obs)` qui calcule l'action experte (oracle),
   - à chaque step, enregistre les positions agent_xy de A et B,
   - retourne la **DTW (dynamic time warping)** *ou* la simple distance moyenne pas-à-pas entre les deux trajectoires.

   Pour cet exercice on fait la version simple : `mean_t || pos_A(t) - pos_B(t) ||`. Si les deux rollouts ont des longueurs différentes (l'un termine plus tôt), on aligne sur le plus court.

3. Compare cette métrique sur 3 policies (n'importe lesquelles, tu peux faire des "fake" policies) :
   - `expert_policy(obs) = expert_action(obs)` (la copie parfaite — devrait avoir deviation ≈ 0),
   - `random_policy(obs) = np.random.uniform(-1, 1, 2)` (très haut),
   - `lazy_policy(obs) = np.zeros(2)` (le robot reste sur place — moyen, dépend de l'init).

4. Discute en 3-5 phrases :
   - quel rang devrait-on observer entre les 3 ? (Spoiler : expert < lazy < random ? ou random < lazy ?)
   - cette métrique pénalise-t-elle un *raccourci* trouvé par la policy (genre la policy fait mieux que l'expert) ? Si oui, est-ce un défaut ou une feature ?
   - cette métrique est-elle invariante à la *vitesse* de la trajectoire ?

## Étapes suggérées

1. Réimporte `ToyPushT` et `expert_action` depuis le module `02-code/27-capstone-eval-ablations.py` (en relatif si tu travailles dans un script local, ou copie-les si plus simple).
2. Crée 2 instances d'env avec **le même `np.random.default_rng(seed)`** pour garantir le même état initial.
3. Roll les deux env en parallèle, enregistre les `agent_xy` à chaque step.
4. Aligne par troncation : `T = min(len(traj_A), len(traj_B))`, calcule `mean(np.linalg.norm(traj_A[:T] - traj_B[:T], axis=1))`.
5. Aggrège sur N=15 rollouts, reporte `mean ± std`.

## Critères de réussite

- [ ] La fonction `trajectory_deviation` retourne un float positif.
- [ ] `deviation(expert_policy)` est sensiblement plus faible que `deviation(random_policy)`.
- [ ] Tu peux nommer 1 cas où cette métrique trompe (ex : policy qui prend un meilleur chemin que l'expert sera punie alors qu'elle est meilleure).
- [ ] Tu produis un mini-tableau Markdown avec les 3 policies × `deviation_mean ± std`.
- [ ] Bonus : tu ajoutes la corrélation entre `deviation` et `success_rate` sur les 3 policies. Est-elle parfaite ? Pourquoi pas ?

## Indices

- Pour ré-instancier deux envs synchronisés : crée `env_A`, fais `env_A.reset(seed=...)`, puis `env_B = ToyPushT(...) ; env_B.rng = np.random.default_rng(...)` avec le même seed pour produire le même init.
- Numpy: `np.linalg.norm(diff, axis=1)` pour distance par-step, puis `.mean()`.
- Évite la DTW pour cet exercice : c'est utile mais ajoute beaucoup de complexité. La distance pas-à-pas suffit pour 90% des cas.

## Pièges classiques

- **Init désynchronisé** : si tu fais `env_A.reset()` puis `env_B.reset()` sans seed identique, tu compares deux trajectoires depuis deux états initiaux différents. Le test n'a alors aucun sens.
- **Longueurs différentes** : si la policy termine en 50 steps et l'expert en 80 (succès plus rapide), tu compares 50 steps contre 50 steps de l'expert qui n'a pas encore fini de pousser. Penser à ce cas avant de comparer naïvement.
- **Action vs position** : on compare les *positions* de l'agent, pas les *actions*. L'action est un signal de contrôle (delta vélocité), la position est l'état effectif. C'est la position qui mesure la déviation visible.
