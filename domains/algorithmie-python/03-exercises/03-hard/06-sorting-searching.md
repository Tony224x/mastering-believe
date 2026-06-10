# Exercices Hard — Sorting & Searching

---

## Exercice 7 : Binary Search Partition — Median of Two Sorted Arrays

### Objectif

LE probleme hard de binary search par excellence : trouver la mediane de deux tableaux tries en **O(log(min(m, n)))**, en cherchant une partition au lieu d'une valeur.

### Consigne

Etant donne deux tableaux tries `nums1` et `nums2`, retourne la **mediane** des deux tableaux combines.

**Contrainte : O(log(min(m, n))) temps.** Les approches a analyser d'abord (mais qui ne valident pas) :
- Merger les deux tableaux : O(m + n) temps et espace.
- Two pointers jusqu'au milieu : O(m + n) temps, O(1) espace.

```python
def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    """
    Return the median of the two sorted arrays in O(log(min(m, n))).
    """
    pass
```

**Idee de la partition** : couper `nums1` en `i` elements a gauche et `nums2` en `j` elements a gauche avec `i + j = (m + n + 1) // 2`. La partition est correcte si `nums1[i-1] <= nums2[j]` et `nums2[j-1] <= nums1[i]`. Binary search sur `i` uniquement (le plus petit tableau), `j` s'en deduit.

**Pieges** : les bords de partition (i = 0 ou i = m → utiliser ±infini), tableau vide, parite de la somme des longueurs.

### Tests

```python
assert find_median_sorted_arrays([1, 3], [2]) == 2.0
assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5
assert find_median_sorted_arrays([], [1]) == 1.0
assert find_median_sorted_arrays([2], []) == 2.0
assert find_median_sorted_arrays([1, 1], [1, 1]) == 1.0
assert find_median_sorted_arrays([1, 2, 3, 4, 5], [6, 7, 8, 9]) == 5.0   # Disjoint ranges
assert find_median_sorted_arrays([6, 7, 8, 9], [1, 2, 3, 4, 5]) == 5.0   # Swapped
assert find_median_sorted_arrays([1, 5], [2, 3, 4]) == 3.0

# Oracle check against full merge on random inputs
import random, statistics
for _ in range(200):
    a = sorted(random.choices(range(50), k=random.randint(0, 20)))
    b = sorted(random.choices(range(50), k=random.randint(0, 20)))
    if not a and not b:
        continue
    assert abs(find_median_sorted_arrays(a, b) - statistics.median(a + b)) < 1e-9
```

### Criteres de reussite

- [ ] Binary search sur la partition du **plus petit** tableau uniquement
- [ ] Les bords (partition vide d'un cote) sont geres avec ±infini
- [ ] Parite geree : max des gauches si impair, moyenne des deux milieux si pair
- [ ] O(log(min(m, n))) temps, O(1) espace
- [ ] Oracle `statistics.median` sur 200 inputs aleatoires passe
- [ ] Tu peux expliquer pourquoi la condition de partition garantit la mediane

---

## Exercice 8 : Merge Sort Augmente — Count Inversions

### Objectif

Modifier un algorithme de tri pour qu'il **compte en meme temps qu'il trie** — le pattern "augmented merge sort" qui resout toute une famille de problemes hard (count inversions, count smaller after self, reverse pairs).

### Consigne

Une **inversion** dans un tableau `arr` est une paire d'indices `(i, j)` telle que `i < j` et `arr[i] > arr[j]`. C'est une mesure du "desordre" du tableau.

Ecris une fonction qui compte le nombre total d'inversions.

**Contraintes** :
- O(n log n) temps impose. La brute force O(n^2) doit etre ecrite comme oracle, mais ne valide pas.
- Benchmark obligatoire : sur un tableau **trie a l'envers** de taille n = 2000, 4000, 8000, la version merge sort doit rester sous la seconde la ou la brute force explose en x4 a chaque doublement.

```python
def count_inversions_brute(arr: list[int]) -> int:
    """O(n^2) oracle — check every pair."""
    pass

def count_inversions(arr: list[int]) -> int:
    """O(n log n) — augmented merge sort. Must not modify the input."""
    pass
```

**Indice** : pendant le merge de `left` et `right`, quand on prend `right[j]` alors que `left[i:]` n'est pas vide, chaque element restant de `left` forme une inversion avec `right[j]` → `count += len(left) - i`. C'est la SEULE ligne a ajouter au merge sort.

### Tests

```python
assert count_inversions([1, 2, 3, 4]) == 0                  # Sorted: zero inversions
assert count_inversions([4, 3, 2, 1]) == 6                  # Reversed: n(n-1)/2
assert count_inversions([2, 4, 1, 3, 5]) == 3               # (2,1), (4,1), (4,3)
assert count_inversions([]) == 0
assert count_inversions([1]) == 0
assert count_inversions([3, 3, 3]) == 0                     # Equal pairs are NOT inversions
assert count_inversions([2, 1, 2, 1]) == 3                  # With duplicates

# Oracle on random inputs
import random
for _ in range(100):
    arr = [random.randint(0, 30) for _ in range(random.randint(0, 40))]
    assert count_inversions(arr) == count_inversions_brute(arr)
```

### Criteres de reussite

- [ ] Le comptage est fait au moment du merge (`count += len(left) - i`), pas par double boucle
- [ ] Les valeurs **egales** ne comptent pas comme inversions (utiliser `<=` dans le merge)
- [ ] L'input n'est pas modifie (travailler sur une copie)
- [ ] Oracle brute force valide sur 100 inputs aleatoires
- [ ] Benchmark sur tableau inverse : scaling ~n log n confirme, brute force ~n^2
- [ ] Tu sais relier ce pattern a "Count of Smaller Numbers After Self" (LeetCode 315)
