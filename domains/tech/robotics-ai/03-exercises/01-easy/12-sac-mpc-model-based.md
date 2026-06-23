# Exercice J12 (easy) — Comprendre le receding horizon

## Objectif

Saisir intuitivement pourquoi MPC re-planifie a chaque step au lieu d'appliquer toute la sequence planifiee d'un coup.

## Consigne

Dans `02-code/12-sac-mpc-model-based.py`, on planifie `H=20` actions a chaque step et on n'applique que la premiere. Modifie temporairement le fichier (en local, ne commit pas) ou ecris un petit script qui :

1. Reprend la fonction `cem_plan` et l'env pendule.
2. Lance UNE SEULE planification au step 0, recupere `plan` (longueur 20).
3. Applique **les 20 actions a la suite** (open-loop, sans re-planifier) en utilisant `pendulum_step` directement.
4. Compare le reward cumule open-loop vs le reward cumule MPC closed-loop sur les **20 premiers steps** de `run_mpc(planner_name="cem", n_steps=20)`.

Reponds aux questions :
- Quel mode obtient un meilleur reward sur 20 steps ?
- Que se passe-t-il si tu ajoutes du bruit (`+ 0.05 * np.random.randn(2)`) a `state` apres chaque `pendulum_step` en mode open-loop ?

## Criteres de reussite

- Tu as un script qui lance les deux modes et imprime les deux rewards.
- Tu sais expliquer en 2 phrases pourquoi le closed-loop est plus robuste au bruit / aux erreurs de modele.
- Tu peux citer une situation reelle ou open-loop suffit (ex : trajectoire pre-calculee sur banc d'essai sans perturbation).

## Indice

L'open-loop ne corrige jamais une derive : si au step 5 la position reelle s'eloigne de la prediction, les 15 actions restantes ont ete optimisees pour une trajectoire imaginaire qui n'existe plus.
