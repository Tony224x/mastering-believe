# Exercices Medium — Sorting & Searching

---

## Exercice 4 : Binary Search Bounds — Find First and Last Position

### Objectif

Maitriser les variantes **lower bound / upper bound** du binary search. C'est la faiblesse n°1 sur ce sujet : tout le monde sait trouver UN index, peu savent trouver le PREMIER et le DERNIER sans boucle lineaire.

### Consigne

Etant donne un tableau `nums` trie par ordre croissant (avec doublons possibles) et une valeur `target`, retourne `[first, last]` : les indices de la **premiere** et de la **derniere** occurrence de `target`. Si `target` est absent, retourne `[-1, -1]`.

**Contrainte : O(log n) temps.** Trouver un index puis etendre lineairement a gauche/droite est O(n) sur `[5, 5, 5, ..., 5]` et ne valide pas.

```python
def search_range(nums: list[int], target: int) -> list[int]:
    """
    Return [first_index, last_index] of target, or [-1, -1] if absent.
    Must run in O(log n).
    """
    pass
```

**Indice** : ecris un helper `lower_bound(nums, x)` qui retourne le premier index avec `nums[i] >= x`. Alors `first = lower_bound(target)` et `last = lower_bound(target + 1) - 1`.

### Tests

```python
assert search_range([5, 7, 7, 8, 8, 10], 8) == [3, 4]
assert search_range([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
assert search_range([], 0) == [-1, -1]
assert search_range([1], 1) == [0, 0]
assert search_range([2, 2], 2) == [0, 1]
assert search_range([5, 5, 5, 5, 5], 5) == [0, 4]       # The O(n) trap input
assert search_range([1, 2, 3], 0) == [-1, -1]            # Smaller than everything
assert search_range([1, 2, 3], 4) == [-1, -1]            # Larger than everything
```

### Criteres de reussite

- [ ] Deux recherches binaires (ou un helper lower_bound appele deux fois) — pas d'extension lineaire
- [ ] Le pattern `lo < hi` avec `hi = mid` / `lo = mid + 1` ne boucle pas a l'infini
- [ ] Les bornes hors tableau sont gerees (`target` absent, plus petit/grand que tout)
- [ ] O(log n) temps, O(1) espace
- [ ] Verification optionnelle avec `bisect.bisect_left` / `bisect_right` comme oracle

---

## Exercice 5 : Custom Comparator — Largest Number

### Objectif

Utiliser `functools.cmp_to_key` pour un ordre qui ne se reduit PAS a une cle simple — le cas exact ou `key=` ne suffit plus.

### Consigne

Etant donne une liste d'entiers non negatifs `nums`, arrange-les pour former le **plus grand nombre possible** et retourne-le sous forme de string.

```python
def largest_number(nums: list[int]) -> str:
    """
    Arrange numbers to form the largest possible number (as a string).
    """
    pass
```

**Indice** : comparer `a` et `b` ne suffit pas (`3` vs `30` : lequel d'abord ?). Le bon critere : `a` avant `b` si la concatenation `str(a) + str(b) > str(b) + str(a)`.

**Piege final** : `[0, 0]` doit donner `"0"`, pas `"00"`.

### Tests

```python
assert largest_number([10, 2]) == "210"
assert largest_number([3, 30, 34, 5, 9]) == "9534330"
assert largest_number([0, 0]) == "0"                     # NOT "00"
assert largest_number([0]) == "0"
assert largest_number([1]) == "1"
assert largest_number([432, 43243]) == "43243432"        # Tricky prefix case
assert largest_number([121, 12]) == "12121"              # 12|121 > 121|12
assert largest_number([8308, 830]) == "8308830"
```

### Criteres de reussite

- [ ] Utilise `cmp_to_key` avec la comparaison de concatenations (pas un tri lexicographique simple)
- [ ] Tu peux expliquer pourquoi `key=str` est FAUX (contre-exemple : `[3, 30]` → "303" au lieu de "330")
- [ ] Le cas "que des zeros" retourne `"0"`
- [ ] Complexite O(n log n * k) ou k = longueur moyenne des nombres
- [ ] Tous les tests passent

---

## Exercice 6 : Binary Search sans tableau trie — Find Peak Element

### Objectif

Comprendre que le binary search ne requiert pas un tableau trie : il requiert un **invariant qui permet d'eliminer une moitie**. C'est le declic conceptuel qui debloque toute la famille "binary search on answer".

### Consigne

Un element pic est un element strictement plus grand que ses voisins. Etant donne un tableau `nums` ou `nums[i] != nums[i+1]` pour tout i (pas de plateaux), retourne l'index de **n'importe quel pic**. On considere que `nums[-1]` et `nums[n]` valent `-infini` (les bords peuvent etre des pics).

**Contrainte : O(log n) temps** — le scan lineaire ne valide pas.

```python
def find_peak_element(nums: list[int]) -> int:
    """
    Return the index of any peak element in O(log n).
    """
    pass
```

**Indice** : compare `nums[mid]` a `nums[mid + 1]`. Si `nums[mid] < nums[mid + 1]`, la pente monte → il existe forcement un pic a droite. Sinon, il en existe un a gauche (mid inclus).

### Tests

```python
assert find_peak_element([1, 2, 3, 1]) == 2
assert find_peak_element([1, 2, 1, 3, 5, 6, 4]) in (1, 5)   # Two valid peaks
assert find_peak_element([1]) == 0
assert find_peak_element([1, 2]) == 1                        # Peak at right edge
assert find_peak_element([2, 1]) == 0                        # Peak at left edge
assert find_peak_element([1, 2, 3, 4, 5]) == 4               # Strictly increasing
assert find_peak_element([5, 4, 3, 2, 1]) == 0               # Strictly decreasing

# Property check: result must actually be a peak
import random
for _ in range(100):
    n = random.randint(1, 50)
    arr = random.sample(range(1000), n)      # sample → no equal neighbors
    i = find_peak_element(arr)
    left = arr[i - 1] if i > 0 else float("-inf")
    right = arr[i + 1] if i < n - 1 else float("-inf")
    assert left < arr[i] > right
```

### Criteres de reussite

- [ ] Binary search avec l'invariant "un pic existe toujours dans la moitie gardee"
- [ ] Tu peux enoncer POURQUOI l'invariant tient (la pente montante mene forcement a un pic ou au bord)
- [ ] Pas d'acces hors bornes (`mid + 1` toujours valide grace a `lo < hi`)
- [ ] O(log n) temps, O(1) espace
- [ ] Le property check sur inputs aleatoires passe
