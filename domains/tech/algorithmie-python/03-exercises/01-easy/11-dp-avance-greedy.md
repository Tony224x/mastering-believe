# Exercices Easy — DP Avance & Greedy

---

## Exercice 1 : State Machine — Best Time to Buy and Sell Stock

### Objectif

Maitriser le template state machine avec deux etats (cash / hold) et transitions en O(1) espace.

### Consigne

Etant donne un tableau `prices` ou `prices[i]` est le prix d'une action au jour `i`, calcule le profit maximum. Tu peux effectuer autant de transactions que tu veux (acheter + vendre = 1 transaction), mais tu ne peux detenir qu'une seule action a la fois (il faut vendre avant de racheter).

```python
def max_profit(prices: list[int]) -> int:
    """
    Return the maximum profit from unlimited buy/sell transactions.
    You must sell before buying again.
    """
    pass
```

### Tests

```python
assert max_profit([7, 1, 5, 3, 6, 4]) == 7   # buy 1 sell 5, buy 3 sell 6
assert max_profit([1, 2, 3, 4, 5]) == 4       # buy 1 sell 5
assert max_profit([7, 6, 4, 3, 1]) == 0       # decreasing, no profit
assert max_profit([]) == 0
assert max_profit([5]) == 0
assert max_profit([2, 4, 1, 7]) == 8          # (4-2) + (7-1) = 8
```

### Criteres de reussite

- [ ] Utilise deux variables (cash, hold) et PAS de tableau
- [ ] Complexite O(n) temps, O(1) espace
- [ ] Comprend la transition : `cash = max(cash, hold + price)` et `hold = max(hold, cash - price)`
- [ ] Tous les tests passent

---

## Exercice 2 : Greedy — Interval Scheduling

### Objectif

Maitriser le pattern "tri par end + scan lineaire" pour max non-overlapping intervals.

### Consigne

Etant donne une liste d'intervalles `intervals`, retourne le **nombre minimum** d'intervalles a retirer pour que les intervalles restants soient non chevauchants (deux intervalles qui partagent juste un point final ne sont pas consideres comme chevauchants).

```python
def erase_overlap_intervals(intervals: list[list[int]]) -> int:
    """
    Return the minimum number of intervals to remove to make the rest
    non-overlapping. Touching endpoints are NOT considered overlapping.
    """
    pass
```

### Tests

```python
assert erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
assert erase_overlap_intervals([[1, 2], [1, 2], [1, 2]]) == 2
assert erase_overlap_intervals([[1, 2], [2, 3]]) == 0
assert erase_overlap_intervals([]) == 0
assert erase_overlap_intervals([[0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]) == 2
```

### Criteres de reussite

- [ ] Tri par **end** (pas par start)
- [ ] Scan lineaire avec `last_end` track
- [ ] Complexite O(n log n) temps, O(1) espace (en place)
- [ ] Tous les tests passent

---

## Exercice 3 : Greedy — Jump Game I

### Objectif

Maitriser le greedy one-pass avec `max_reach` pour tester l'atteignabilite.

### Consigne

Etant donne un tableau d'entiers non negatifs `nums`, tu commences a l'index 0. Chaque element represente le saut maximum depuis cette position. Determine si tu peux atteindre le dernier index.

```python
def can_jump(nums: list[int]) -> bool:
    """
    Return True if you can reach the last index starting from index 0.
    """
    pass
```

### Tests

```python
assert can_jump([2, 3, 1, 1, 4]) == True
assert can_jump([3, 2, 1, 0, 4]) == False
assert can_jump([0]) == True
assert can_jump([1]) == True
assert can_jump([0, 1]) == False
assert can_jump([1, 0, 1, 0]) == False
assert can_jump([2, 0, 0]) == True
```

### Criteres de reussite

- [ ] Une seule passe avec `max_reach`
- [ ] Retourne `False` des qu'un index n'est pas atteignable (short-circuit)
- [ ] Complexite O(n) temps, O(1) espace
- [ ] Tous les tests passent, y compris les cas avec 0 bloquant
