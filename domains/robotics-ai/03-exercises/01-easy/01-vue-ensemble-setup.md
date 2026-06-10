# Exercice J1 — Easy : faire tourner un env et inspecter le step

## Objectif
Confirmer que la stack `gymnasium[mujoco] + mujoco + torch` est installée et comprendre concrètement ce que renvoie `env.step()`.

## Consigne
Écris un script `easy.py` qui :

1. Crée un env `Pendulum-v1` (classic-control, pas besoin de MuJoCo pour celui-ci).
2. Reset avec `seed=42`.
3. Exécute exactement 10 steps avec une action aléatoire (`env.action_space.sample()`).
4. À chaque step, imprime sur une ligne : `t=<i> reward=<r:.3f> terminated=<bool> truncated=<bool>`.
5. À la fin, imprime la **shape de `obs`** et la **shape de `action_space.sample()`**.
6. Ferme l'env proprement (`env.close()`).

## Criteres de reussite
- Le script tourne sans crash sur Python 3.10+.
- 10 lignes de log step + 2 lignes de shape.
- Le script contient `assert obs.shape == (3,)` et `assert env.action_space.sample().shape == (1,)`, et les deux assertions passent.
- En commentaire de fin de script, tu as écrit en 1 phrase la différence `terminated` / `truncated`, en utilisant les deux notions « état terminal du MDP » et « limite de temps (TimeLimit) ».
