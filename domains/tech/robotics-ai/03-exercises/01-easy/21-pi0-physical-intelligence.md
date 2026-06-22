# J21 — Exercice EASY : flow matching basics

## Objectif

Vérifier que tu as compris :
- la **différence de loss** entre flow matching et DDPM,
- la **boucle d'inférence Euler** d'une ODE,
- pourquoi π0 préfère ODE à SDE pour l'action head.

Tu n'auras pas besoin d'entraîner un modèle — uniquement des manipulations numériques.

## Consigne

Implémente une fonction `flow_matching_step(A_tau, tau, A_star)` qui :

1. Prend en entrée :
   - `A_tau` : un tenseur représentant l'état courant le long de la trajectoire d'interpolation, shape `(B, H, d)`.
   - `tau` : un scalaire ou un vecteur dans `[0, 1]`, shape `(B,)`.
   - `A_star` : la cible "propre" (action experte), shape `(B, H, d)`.

2. Retourne **la velocity-cible** que la loss flow matching demande de régresser, c'est-à-dire `(A_star - A_0)` où `A_0` est implicite : on remonte par `A_0 = (A_tau - tau * A_star) / (1 - tau)` (en évitant la division par 0 — clipper `tau` à `1 - 1e-3` au max).

Puis implémente une fonction `euler_integrate(velocity_fn, A_init, n_steps)` qui :
- part de `A_init`,
- applique `n_steps` pas d'Euler `A ← A + dt · velocity_fn(A, tau)` avec `dt = 1 / n_steps`,
- retourne le `A` final.

Tu testes avec un `velocity_fn` factice qui retourne **toujours** `(target - A)` où `target` est une constante (ex. tenseur de 1.0). Tu vérifies que partir de bruit gaussien et intégrer 10 steps fait bien converger vers `target` à epsilon près.

## Étapes suggérées

1. Écrire les deux fonctions dans un fichier `solution_easy.py` (ou simplement essayer mentalement).
2. Échantillonner `A_star = ones((4, 16, 2))`, `A_init = randn((4, 16, 2))`.
3. Lancer `euler_integrate(lambda A, tau: A_star - A, A_init, n_steps=10)` et vérifier que la sortie est ≈ `A_star`.
4. Refaire avec `n_steps=2` et observer que la convergence est moins bonne.

## Critères de réussite

- [ ] `flow_matching_step` retourne bien un tenseur de la même shape que `A_tau`.
- [ ] L'intégration Euler converge vers `target` à `MSE < 0.01` avec `n_steps=10`.
- [ ] L'intégration Euler avec `n_steps=2` donne un MSE *plus grand* que `n_steps=10` (illustration du compromis vitesse/qualité).
- [ ] Tu peux expliquer en 1 phrase pourquoi le flow matching "ODE" ne nécessite **pas** d'injecter du bruit à chaque pas, alors que DDPM "SDE" oui.

## Indices

- Pas besoin de PyTorch — NumPy suffit pour ce niveau easy.
- `A_0` n'est PAS la sortie attendue : c'est juste la formule mathématique qu'on inverse pour reconstituer le bruit initial à partir de `A_tau`. La velocity-cible reste `A_star - A_0`.
- Pour la boucle Euler, n'oublie pas de passer le `tau` courant à `velocity_fn` (même si dans cet exercice il n'est pas utilisé).
