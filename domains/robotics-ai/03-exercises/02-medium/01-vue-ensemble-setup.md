# Exercice J1 — Medium : rollout instrumenté HalfCheetah avec stats

## Objectif
Faire tourner un env MuJoCo (`HalfCheetah-v4`) avec une policy aléatoire, collecter des stats par épisode, et calculer la **reward moyenne** sur 5 épisodes — référence à laquelle on comparera plus tard les algos appris.

## Consigne
Écris un script `medium.py` qui :

1. Crée `HalfCheetah-v4` (nécessite `gymnasium[mujoco]` et `mujoco`).
2. Lance 5 épisodes complets, chacun jusqu'à `terminated` ou `truncated`, avec `seed=i` pour le i-ème épisode (`i = 0..4`).
3. Pour chaque épisode collecte : nombre de steps, reward totale, raison de fin (`terminated` / `truncated`).
4. Imprime un tableau récapitulatif aligné (header + une ligne par épisode).
5. Imprime ensuite : `mean_reward = ..., std_reward = ..., mean_length = ...` calculés via numpy.
6. Précondition : si `gymnasium` ou `mujoco` n'est pas importable, le script doit imprimer un message clair avec la commande `pip install` à exécuter et sortir avec `sys.exit(1)`.

## Critères de réussite
- 5 épisodes tournent sans crash sur HalfCheetah-v4.
- La reward moyenne d'une policy random est **négative** (typiquement ~-300 à -100 sur HalfCheetah). Si tu vois +1000, tu mesures autre chose.
- Le script échoue proprement (avec message d'install) sur une machine sans MuJoCo.
- Tu peux expliquer pourquoi `truncated=True` est attendu plus souvent que `terminated=True` sur HalfCheetah (indice : TimeLimit wrapper, 1000 steps par défaut, et le robot ne tombe pas vraiment dans un état terminal MDP).
