# Exercices Easy — Dynamic Programming

---

## Exercice 1 : Linear DP — Climbing Stairs

### Objectif

Comprendre la recurrence Fibonacci et maitriser la tabulation 1D avec optimisation d'espace O(1).

### Consigne

Tu montes un escalier de `n` marches. A chaque pas, tu peux monter 1 ou 2 marches. Combien de facons distinctes d'atteindre la marche n ?

```python
def climb_stairs(n: int) -> int:
    """
    Return the number of distinct ways to climb n stairs
    taking 1 or 2 steps at a time.
    """
    pass
```

### Tests

```python
assert climb_stairs(1) == 1
assert climb_stairs(2) == 2
assert climb_stairs(3) == 3
assert climb_stairs(4) == 5
assert climb_stairs(5) == 8
assert climb_stairs(10) == 89
assert climb_stairs(45) == 1836311903
```

### Criteres de reussite

- [ ] Utilise une approche iterative (pas de recursion naive)
- [ ] Complexite O(n) temps
- [ ] Complexite O(1) espace (deux variables suffisent)
- [ ] Comprend pourquoi la recurrence est `dp[n] = dp[n-1] + dp[n-2]`
- [ ] Tous les tests passent, y compris n=45

---

## Exercice 2 : Coin Change — Minimum pieces

### Objectif

Maitriser le template "min operations to reach target" avec memoization ou tabulation.

### Consigne

Etant donne une liste de denominations `coins` et un `amount`, retourne le **nombre minimum** de pieces necessaires pour atteindre `amount`. Retourne `-1` si c'est impossible. Tu peux utiliser chaque denomination autant de fois que tu veux.

```python
def coin_change(coins: list[int], amount: int) -> int:
    """
    Return the minimum number of coins needed to make `amount`,
    or -1 if it cannot be made.
    """
    pass
```

### Tests

```python
assert coin_change([1, 2, 5], 11) == 3   # 5 + 5 + 1
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0
assert coin_change([1, 2, 5], 0) == 0
assert coin_change([1], 2) == 2
assert coin_change([2, 5, 10, 1], 27) == 4   # 10 + 10 + 5 + 2
assert coin_change([186, 419, 83, 408], 6249) == 20
```

### Criteres de reussite

- [ ] Tabulation 1D avec `dp[0] = 0` et le reste initialise a +inf
- [ ] Retourne -1 si `dp[amount]` reste a +inf
- [ ] Complexite O(amount * len(coins)) temps, O(amount) espace
- [ ] Tous les tests passent, y compris le cas impossible `[2], 3`

---

## Exercice 3 : Grid DP — Unique Paths

### Objectif

Pratiquer la DP 2D sur une grille avec la recurrence `dp[i][j] = dp[i-1][j] + dp[i][j-1]`.

### Consigne

Un robot est place dans le coin haut-gauche d'une grille `m x n`. Il ne peut se deplacer que vers la droite ou vers le bas. Combien de chemins distincts pour atteindre le coin bas-droit ?

```python
def unique_paths(m: int, n: int) -> int:
    """
    Return the number of unique paths from (0,0) to (m-1, n-1)
    moving only right or down.
    """
    pass
```

### Tests

```python
assert unique_paths(3, 7) == 28
assert unique_paths(3, 2) == 3
assert unique_paths(1, 1) == 1
assert unique_paths(1, 10) == 1
assert unique_paths(10, 1) == 1
assert unique_paths(7, 3) == 28
assert unique_paths(10, 10) == 48620
```

### Criteres de reussite

- [ ] DP 2D avec initialisation de la premiere ligne et premiere colonne a 1
- [ ] Recurrence `dp[i][j] = dp[i-1][j] + dp[i][j-1]`
- [ ] Complexite O(m * n) temps et espace (O(n) avec rolling row bonus)
- [ ] Tous les tests passent
