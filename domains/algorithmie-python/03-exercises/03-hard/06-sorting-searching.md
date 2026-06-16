# Exercices Hard — Sorting & Searching

---

## Exercice 7 : Binary search croise — Median of Two Sorted Arrays

### Objectif

Trouver la mediane de deux tableaux tries en **O(log(min(m, n)))** via une binary search sur la **partition** — l'un des problemes hard les plus celebres et discriminants en entretien.

### Consigne

Etant donne deux tableaux tries `nums1` et `nums2` de tailles `m` et `n`, retourne la mediane de l'union des deux tableaux. La complexite doit etre **O(log(m + n))** (vise O(log(min(m, n)))).

```python
def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    """
    Return the median of the two sorted arrays in O(log(min(m, n))).
    """
    pass
```

**Approche attendue** :
1. Binary search sur le plus PETIT tableau : on coupe `nums1` en `i` elements a gauche, `nums2` en `j = half - i`
2. La partition est valide si `left1 <= right2` ET `left2 <= right1` (en utilisant +/- inf aux bords)
3. Si valide : la mediane se calcule depuis `max(left1, left2)` et `min(right1, right2)`

### Tests

```python
assert find_median_sorted_arrays([1, 3], [2]) == 2.0
assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5
assert find_median_sorted_arrays([], [1]) == 1.0
assert find_median_sorted_arrays([2], []) == 2.0
assert find_median_sorted_arrays([1, 3], [2, 7]) == 2.5
assert find_median_sorted_arrays([1, 2, 3, 4, 5], [6, 7, 8]) == 4.5
assert find_median_sorted_arrays([1, 1, 1], [1, 1, 1]) == 1.0
assert find_median_sorted_arrays([0, 0], [0, 0]) == 0.0
assert find_median_sorted_arrays([1, 3, 5, 7], [2, 4, 6]) == 4.0
```

### Criteres de reussite

- [ ] Binary search sur le PLUS PETIT des deux tableaux (garantit O(log(min(m, n))))
- [ ] Utilise les sentinelles `-inf` / `+inf` pour les bords de partition
- [ ] Gere les longueurs paires et impaires de l'union
- [ ] Gere un tableau vide
- [ ] Complexite O(log(min(m, n))) temps, O(1) espace
- [ ] Tous les tests passent

---

## Exercice 8 : Merge sort instrumente — Count of Smaller Numbers After Self

### Objectif

Compter, pour chaque element, le nombre d'elements plus petits a sa droite — en O(n log n) via un **merge sort instrumente** (ou un BIT/Fenwick). Probleme hard qui revele la profondeur du divide & conquer.

### Consigne

Etant donne un tableau `nums`, retourne un tableau `counts` ou `counts[i]` est le nombre d'elements a droite de `nums[i]` qui lui sont strictement inferieurs.

```python
def count_smaller(nums: list[int]) -> list[int]:
    """
    counts[i] = number of elements to the right of nums[i] that are smaller.
    Target O(n log n).
    """
    pass
```

**Approche attendue (merge sort sur indices)** :
- Trie les **indices** par valeur via merge sort
- Pendant le merge, quand on prend un element de la moitie droite avant un element de la moitie gauche, cet element droit est plus petit que tous les elements gauches restants : incremente leurs compteurs

### Tests

```python
assert count_smaller([5, 2, 6, 1]) == [2, 1, 1, 0]
assert count_smaller([-1]) == [0]
assert count_smaller([-1, -1]) == [0, 0]
assert count_smaller([]) == []
assert count_smaller([2, 0, 1]) == [2, 0, 0]
assert count_smaller([1, 2, 3, 4]) == [0, 0, 0, 0]      # Already sorted
assert count_smaller([4, 3, 2, 1]) == [3, 2, 1, 0]      # Reverse sorted
assert count_smaller([5, 5, 5]) == [0, 0, 0]            # Strict: equals don't count
```

### Criteres de reussite

- [ ] Utilise merge sort instrumente OU un Fenwick tree / BIT (pas O(n^2))
- [ ] Complexite O(n log n) temps, O(n) espace
- [ ] Strictement inferieur (les egalites ne comptent pas)
- [ ] Gere tableau vide, tableau trie, tableau inverse, doublons
- [ ] Tous les tests passent

---

## Exercice 9 : Binary search on answer 2D — Split Array Largest Sum

### Objectif

Generaliser le pattern "binary search on answer" a un probleme d'allocation : minimiser la plus grande somme parmi `k` sous-tableaux contigus. Combine binary search + greedy de faisabilite.

### Consigne

Etant donne un tableau d'entiers non negatifs `nums` et un entier `k`, decoupe `nums` en `k` sous-tableaux **contigus non vides** de facon a minimiser la **plus grande somme** parmi ces sous-tableaux. Retourne cette plus grande somme minimisee.

```python
def split_array(nums: list[int], k: int) -> int:
    """
    Split nums into k contiguous subarrays minimizing the largest subarray sum.
    Return that minimized largest sum.
    """
    pass
```

**Indice** : la reponse est entre `max(nums)` (chaque element seul ne peut etre depasse) et `sum(nums)` (un seul groupe). Pour une borne candidate `cap`, un greedy compte combien de groupes sont necessaires si aucun groupe ne depasse `cap`. Binary search la plus petite `cap` realisable avec <= `k` groupes.

### Tests

```python
assert split_array([7, 2, 5, 10, 8], 2) == 18
assert split_array([1, 2, 3, 4, 5], 2) == 9
assert split_array([1, 4, 4], 3) == 4
assert split_array([1], 1) == 1
assert split_array([1, 2, 3, 4, 5], 1) == 15      # One group = total sum
assert split_array([1, 2, 3, 4, 5], 5) == 5       # k == n: max element
assert split_array([2, 3, 1, 1, 1, 1, 1], 5) == 3
```

### Criteres de reussite

- [ ] Espace de recherche `[max(nums), sum(nums)]`
- [ ] Predicat greedy : nombre de groupes necessaires pour un plafond donne
- [ ] Binary search la plus petite somme max faisable
- [ ] Complexite O(n log(sum(nums))) temps, O(1) espace
- [ ] Gere `k = 1`, `k = n`, un seul element
- [ ] Tous les tests passent
