# Exercices Hard — Arrays & Strings

---

## Exercice 7 : Sliding Window — Sliding Window Maximum

### Objectif

Combiner le sliding window avec une structure de donnees monotone (deque) pour maintenir le maximum d'une fenetre glissante en O(1) amorti.

### Consigne

Etant donne un tableau d'entiers `nums` et un entier `k`, retourne un tableau contenant le **maximum de chaque fenetre** de taille `k` qui glisse de gauche a droite.

```python
def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """
    Return array of maximums for each window of size k.
    len(result) = len(nums) - k + 1
    """
    pass
```

**Approche attendue** : utiliser une **deque monotone decroissante** qui stocke les **indices** (pas les valeurs). A chaque pas :
1. Retirer de la deque les indices hors de la fenetre (gauche)
2. Retirer de la deque les indices dont la valeur est inferieure a l'element courant (droite) — ils ne seront jamais le max
3. Ajouter l'index courant
4. Le front de la deque est toujours le max de la fenetre courante

### Tests

```python
assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
assert max_sliding_window([1], 1) == [1]
assert max_sliding_window([1, -1], 1) == [1, -1]
assert max_sliding_window([9, 11], 2) == [11]
assert max_sliding_window([4, 3, 2, 1], 3) == [4, 3]
assert max_sliding_window([1, 2, 3, 4], 3) == [3, 4]

# Edge case: all same values
assert max_sliding_window([5, 5, 5, 5], 2) == [5, 5, 5]

# Large decreasing then increasing
assert max_sliding_window([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11], 3) == [10, 9, 8, 7, 6, 5, 4, 3, 11]
```

### Criteres de reussite

- [ ] Utilise une deque monotone decroissante (pas un heap, pas un recalcul max() a chaque fenetre)
- [ ] Complexite O(n) temps — chaque element est ajoute et retire de la deque au plus une fois
- [ ] Complexite O(k) espace pour la deque
- [ ] Comprend pourquoi stocker des indices (et non des valeurs) dans la deque
- [ ] La brute force O(n*k) est presentee d'abord pour comparaison
- [ ] Tous les tests passent

---

## Exercice 8 : Two Pointers + Prefix — Trapping Rain Water with Follow-ups

### Objectif

Maitriser Trapping Rain Water et ses variantes — probleme classique qui combine prefix arrays et two pointers, frequemment pose en entretien FAANG.

### Consigne

**Partie A** : Implemente les trois approches de Trapping Rain Water :
1. Brute force O(n^2)
2. Prefix/suffix max arrays O(n) temps, O(n) espace
3. Two pointers O(n) temps, O(1) espace

```python
def trap_brute(height: list[int]) -> int:
    pass

def trap_prefix(height: list[int]) -> int:
    pass

def trap_two_pointers(height: list[int]) -> int:
    pass
```

**Partie B — Follow-up** : Etant donne la meme elevation map, calcule le **volume total d'eau piege** mais cette fois en retournant aussi la **liste des quantites d'eau piege a chaque position**.

```python
def trap_detailed(height: list[int]) -> tuple[int, list[int]]:
    """
    Return (total_water, water_per_position) where water_per_position[i]
    is the amount of water trapped above bar i.
    """
    pass
```

**Partie C — Follow-up 2** : Combien de "piscines" (pools) separees y a-t-il ? Une piscine est un groupe contigu de positions avec de l'eau.

```python
def count_pools(height: list[int]) -> int:
    """
    Return the number of separate pools of trapped water.
    A pool is a contiguous group of positions with water > 0.
    """
    pass
```

### Tests

```python
# Partie A
height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
assert trap_brute(height) == 6
assert trap_prefix(height) == 6
assert trap_two_pointers(height) == 6

assert trap_two_pointers([4, 2, 0, 3, 2, 5]) == 9
assert trap_two_pointers([]) == 0
assert trap_two_pointers([1]) == 0
assert trap_two_pointers([1, 2]) == 0
assert trap_two_pointers([2, 0, 2]) == 2
assert trap_two_pointers([3, 0, 0, 0, 3]) == 9

# Partie B
total, water = trap_detailed([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
assert total == 6
assert water == [0, 0, 1, 0, 1, 2, 1, 0, 0, 1, 0, 0]

# Partie C
assert count_pools([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 2  # Pool at index 2, pool at indices 4-6 and 9
# Actually: water = [0,0,1,0,1,2,1,0,0,1,0,0] → pools at [2], [4,5,6], [9] = 3 pools
assert count_pools([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 3
assert count_pools([2, 0, 2]) == 1
assert count_pools([3, 0, 0, 0, 3]) == 1         # One continuous pool
assert count_pools([1, 2, 3]) == 0                # No water trapped
assert count_pools([3, 1, 2, 1, 3]) == 1          # Continuous pool under the bridge
```

### Criteres de reussite

- [ ] Les trois approches de Partie A sont correctes et ont la complexite attendue
- [ ] La progression brute → prefix → two pointers est expliquee dans les commentaires
- [ ] Partie B : les quantites par position sont correctes
- [ ] Partie C : le comptage de piscines est correct (groupes contigus de water > 0)
- [ ] Edge cases geres : tableau vide, tableau monotone, deux barres, vallee unique
- [ ] Commentaires expliquant l'invariant du two pointers (pourquoi on peut faire confiance au cote oppose)
