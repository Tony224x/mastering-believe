# Exercices Easy — Arrays & Strings

---

## Exercice 1 : Two Pointers — Remove Duplicates from Sorted Array

### Objectif

Maitriser le pattern two pointers "meme direction" pour modifier un tableau in-place.

### Consigne

Etant donne un tableau d'entiers **trie en ordre croissant**, supprime les doublons **in-place** de sorte que chaque element unique n'apparaisse qu'une seule fois. Retourne le nombre d'elements uniques `k`.

Les k premiers elements du tableau doivent contenir les elements uniques dans leur ordre original. Le contenu au-dela de k n'a pas d'importance.

```python
def remove_duplicates(nums: list[int]) -> int:
    """
    Modify nums in-place and return k (number of unique elements).
    The first k elements of nums should contain the unique values.
    """
    pass
```

### Tests

```python
nums = [1, 1, 2]
k = remove_duplicates(nums)
assert k == 2
assert nums[:k] == [1, 2]

nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
k = remove_duplicates(nums)
assert k == 5
assert nums[:k] == [0, 1, 2, 3, 4]

nums = [1]
k = remove_duplicates(nums)
assert k == 1
assert nums[:k] == [1]

nums = []
k = remove_duplicates(nums)
assert k == 0
```

### Criteres de reussite

- [ ] L'algorithme utilise le pattern two pointers (slow/fast)
- [ ] Modification in-place, pas de tableau auxiliaire → O(1) espace
- [ ] Complexite temps O(n) — une seule passe
- [ ] Tous les tests passent, y compris les edge cases (vide, un seul element)

---

## Exercice 2 : Sliding Window — Maximum Sum of Subarray of Size K

### Objectif

Maitriser le sliding window a taille fixe pour calculer un aggregat sur une fenetre glissante.

### Consigne

Etant donne un tableau d'entiers et un entier `k`, trouve la **somme maximale** d'un sous-tableau contigu de taille exactement `k`.

```python
def max_subarray_sum(nums: list[int], k: int) -> int:
    """
    Return the maximum sum of any contiguous subarray of size k.
    Assume 1 <= k <= len(nums).
    """
    pass
```

### Tests

```python
assert max_subarray_sum([2, 1, 5, 1, 3, 2], 3) == 9        # [5, 1, 3]
assert max_subarray_sum([2, 3, 4, 1, 5], 2) == 7            # [3, 4]
assert max_subarray_sum([1, -1, 5, -2, 3], 3) == 6          # [5, -2, 3]
assert max_subarray_sum([10], 1) == 10                       # Single element
assert max_subarray_sum([-1, -2, -3, -4], 2) == -3           # [-1, -2] — all negative
assert max_subarray_sum([1, 2, 3, 4, 5], 5) == 15           # Entire array
```

### Criteres de reussite

- [ ] Utilise le sliding window (pas de recalcul `sum()` a chaque position)
- [ ] Complexite O(n) temps, O(1) espace
- [ ] Gere les tableaux avec des nombres negatifs
- [ ] Tous les tests passent

---

## Exercice 3 : Prefix Sum — Running Sum of 1D Array

### Objectif

Comprendre la construction d'un prefix sum et l'utiliser pour repondre a des requetes.

### Consigne

**Partie A** : Etant donne un tableau d'entiers, retourne le tableau de sommes cumulees (running sum).

```python
def running_sum(nums: list[int]) -> list[int]:
    """
    Return array where result[i] = sum(nums[0..i]).
    """
    pass
```

**Partie B** : Utilise un prefix sum pour implementer une fonction `range_sum_query` qui repond a Q requetes de somme de plage en O(1) chacune.

```python
def range_sum_query(nums: list[int], queries: list[tuple[int, int]]) -> list[int]:
    """
    For each (left, right) query, return sum(nums[left..right]).
    Build prefix sum once, then answer each query in O(1).
    """
    pass
```

### Tests

```python
# Partie A
assert running_sum([1, 2, 3, 4]) == [1, 3, 6, 10]
assert running_sum([1, 1, 1, 1, 1]) == [1, 2, 3, 4, 5]
assert running_sum([3, 1, 2, 10, 1]) == [3, 4, 6, 16, 17]
assert running_sum([5]) == [5]

# Partie B
nums = [1, 2, 3, 4, 5]
assert range_sum_query(nums, [(0, 2)]) == [6]           # 1+2+3
assert range_sum_query(nums, [(1, 3)]) == [9]           # 2+3+4
assert range_sum_query(nums, [(0, 4)]) == [15]          # All
assert range_sum_query(nums, [(2, 2)]) == [3]           # Single element
assert range_sum_query(nums, [(0, 0), (4, 4), (1, 3)]) == [1, 5, 9]
```

### Criteres de reussite

- [ ] Partie A : O(n) temps, modification en une seule passe
- [ ] Partie B : Construction du prefix sum en O(n), chaque requete en O(1)
- [ ] Le prefix sum a un element supplementaire (index 0 = 0) pour gerer le edge case `left=0`
- [ ] Tous les tests passent
