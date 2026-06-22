# Exercice J9 — Hard : Modified Policy Iteration + benchmark VI vs PI vs MPI

## Objectif

Implementer **Modified Policy Iteration (MPI)** — l'algorithme qui interpole entre VI et PI — et **benchmarker** les trois sur des MDP de tailles croissantes pour comprendre le compromis vitesse / nombre d'iterations.

## Consigne

### Partie A — Implementer MPI

Modified Policy Iteration (Sutton & Barto, ch. 4.6) generalise PI :
- au lieu de resoudre exactement la phase d'evaluation jusqu'a convergence,
- on fait `m` etapes de Bellman *expected* (m fixe, par exemple 5 ou 10),
- puis une amelioration de politique,
- puis on recommence.

Cas particuliers :
- `m = 1` -> MPI degenere en VI (avec une politique gloutonne extraite)
- `m -> infty` -> MPI redonne PI

**Implementer** :
```python
def modified_policy_iteration(P, R, gamma, m, eps, max_iter):
    """Retourne V, pi, n_outer_iter, n_total_bellman_updates."""
```

Compter `n_total_bellman_updates` = nombre total d'applications d'une operation de Bellman (chaque iter d'evaluation = 1 update, chaque iter externe = 1 update pour l'amelioration).

### Partie B — Generer des MDP aleatoires

Ecris une fonction `random_mdp(n_states, n_actions, gamma, seed)` qui genere :
- `P[s, a, :]` echantillonne d'une Dirichlet(1, ..., 1) (transitions aleatoires)
- `R[s, a, s']` echantillonne d'une `N(0, 1)` puis fixe
- pas d'etats terminaux pour cette etude (MDP "infinite-horizon discounted")

### Partie C — Benchmark

Pour `n_states ∈ {10, 50, 200}`, `n_actions = 4`, `gamma = 0.95`, `eps = 1e-8` :

1. Faire tourner :
   - VI
   - PI (evaluation iterative jusqu'a `eps_inner = 1e-10`)
   - MPI avec `m ∈ {1, 5, 10, 50}`

2. Mesurer pour chacun :
   - nombre d'iterations externes
   - nombre total d'updates Bellman
   - temps mural (timeit avec 5 reps)

3. Verifier que **toutes** les V* convergent vers la meme valeur (ecart `< 1e-6`).

4. Tracer (matplotlib ou tableau ASCII) updates Bellman vs taille du MDP.

### Questions a repondre

- Q1 : confirme-tu empiriquement que PI fait moins d'iterations externes que VI ?
- Q2 : pour `n_states = 200`, quelle valeur de `m` minimise le temps mural ? Pourquoi ?
- Q3 : comment ces resultats se generaliseraient-ils a un MDP issu d'un robot reel avec `|S| = 10^6` (configuration discretisee d'un bras 6-DOF) ?

## Criteres de reussite

- MPI est correctement implemente et reproduit VI quand `m = 1` (a l'extraction de politique pres) et PI quand `m` grand.
- Les trois algorithmes convergent vers la meme V* (preuve par numpy `assert_allclose`).
- Tu obtiens un ordre de grandeur clair : PI a typiquement ~5-10 iterations externes la ou VI en a 100-500 pour `gamma = 0.95`.
- Discussion claire sur le compromis : MPI avec `m = 5..10` est souvent un bon defaut en pratique.
- Bonus : tu identifies que pour les vrais robots, ni VI ni PI tabulaires ne marchent (curse of dimensionality) -- d'ou le passage au RL approxime (J10+).
