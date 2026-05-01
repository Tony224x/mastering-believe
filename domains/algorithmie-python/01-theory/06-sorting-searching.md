# Jour 6 — Sorting & Searching : Binary Search, Quickselect & Custom Comparators

> **Temps estime** : 60 min de lecture active | **Objectif** : maitriser le built-in sort Python, les variantes de binary search (lower_bound, rotated), et quickselect pour des solutions plus elegantes que le tri complet.

---

## 1. Pourquoi ce jour est le plus rentable en temps investi/resultat

Tu sais deja utiliser `sorted(arr)` et `arr.sort()`. Ca suffit a resoudre ~30% des problemes Easy directement. Ce qui fait la difference en entretien :

1. **Binary search** dans ses 4 variantes (exact, lower_bound, upper_bound, rotated) — resout les "trouver X dans O(log n)" et decuple les patterns du sliding window
2. **Custom comparators** avec `key=lambda` — pour trier par des criteres non triviaux
3. **Quickselect** — pour trouver le K-ieme element en O(n) moyen au lieu de O(n log n) avec un tri complet
4. **Stabilite** — savoir quand le tri preserve l'ordre des elements egaux (CRITIQUE pour les tris composes)

La regle : si tu peux trier, trier est presque toujours la meilleure premiere approche. Mais sache aussi quand NE PAS trier (ex: quickselect, heap, counting sort).

---

## 2. Le built-in sort de Python (Timsort)

### Les API

```python
# Hors place — retourne une nouvelle liste, ne modifie pas l'original
arr = [3, 1, 4, 1, 5, 9]
sorted_arr = sorted(arr)              # [1, 1, 3, 4, 5, 9]
arr_unchanged = arr                    # Toujours [3, 1, 4, 1, 5, 9]

# En place — modifie la liste, retourne None
arr.sort()                             # arr devient [1, 1, 3, 4, 5, 9]

# Decroissant
sorted(arr, reverse=True)              # [9, 5, 4, 3, 1, 1]
```

### Timsort : les proprietes a connaitre

```python
# Complexite :
#   - O(n log n) worst case
#   - O(n) si l'input est deja trie ou presque trie (adaptive)
#   - Space O(n) worst case pour le merge temporaire

# Stable : les elements egaux gardent leur ordre relatif d'entree
# C'est CRUCIAL pour trier par cles composees (voir section 4)
```

**Pourquoi Python est rapide** : Timsort est un merge sort hybride qui detecte les "runs" (sequences deja triees) dans l'input et fusionne intelligemment. Pour un tableau reverse-sorted, il reverse puis merge — O(n).

---

## 3. Custom comparators — le pouvoir de `key=`

### L'argument `key`

```python
# Trier par longueur
words = ["banana", "apple", "cherry", "fig"]
sorted(words, key=len)                            # ['fig', 'apple', 'banana', 'cherry']

# Trier par 2eme caractere
sorted(words, key=lambda s: s[1])                 # ['banana', 'cherry', 'fig', 'apple']

# Trier des tuples par 2eme element
pairs = [(1, 'c'), (2, 'a'), (3, 'b')]
sorted(pairs, key=lambda x: x[1])                 # [(2, 'a'), (3, 'b'), (1, 'c')]
```

`key` est une **fonction** appelee une fois par element. Python construit un tableau de cles decorees, trie selon ces cles, puis retourne les elements originaux dans le nouvel ordre (decorate-sort-undecorate).

### Tri a plusieurs cles

```python
# Trier d'abord par longueur croissante, puis par ordre alphabetique
sorted(words, key=lambda s: (len(s), s))
# [('fig'), 'apple', 'banana', 'cherry'] — les tuples se comparent lexicographiquement

# Tri mixte : longueur croissante, nom decroissant
# Astuce : inverser l'ordre en niant la cle numerique
sorted(words, key=lambda s: (len(s), -ord(s[0])))
```

**Regle** : pour un tri compose stable, emballe tes criteres dans un tuple. Python compare les tuples element par element, et s'arrete au premier ecart.

### L'exploitation de la stabilite

```python
# On veut trier d'abord par grade DECROISSANT, puis par nom CROISSANT
students = [("Alice", 85), ("Bob", 92), ("Charlie", 85), ("Dave", 92)]

# METHODE 1 : tri compose en une passe (recommande)
sorted(students, key=lambda s: (-s[1], s[0]))
# [('Bob', 92), ('Dave', 92), ('Alice', 85), ('Charlie', 85)]

# METHODE 2 : exploiter la stabilite en triant CRITERE LE MOINS IMPORTANT EN PREMIER
step1 = sorted(students, key=lambda s: s[0])        # Trie par nom
step2 = sorted(step1, key=lambda s: -s[1])          # Puis par grade decroissant (stable)
# Meme resultat — parce que Timsort est stable
```

Les deux methodes sont correctes. La methode 1 est plus idiomatique ; la methode 2 est utile quand tu ne peux pas combiner les cles (ex: cles non comparables).

### `functools.cmp_to_key` pour les comparaisons complexes

```python
from functools import cmp_to_key

# Cas : ordre custom qui ne se reduit pas a une cle
# Ex: Largest Number — trier pour que la concatenation soit maximale
def compare(a, b):
    # Si a+b > b+a en tant que strings, alors a doit venir avant b
    if a + b > b + a:
        return -1
    elif a + b < b + a:
        return 1
    return 0

nums = [10, 2, 9, 39, 17]
str_nums = [str(n) for n in nums]
result = ''.join(sorted(str_nums, key=cmp_to_key(compare)))
# "939210..." etc.
```

**Quand l'utiliser** : quand l'ordre ne peut pas etre exprime comme une cle fixe. Moins rapide que `key=` (constante plus elevee).

---

## 4. Binary Search — les 4 variantes a maitriser

### Variante 1 — Exact match (le classique)

```python
def binary_search(arr, target):
    """Return index of target in arr, or -1 if not found. arr must be sorted."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
# Time: O(log n), Space: O(1)
```

**Piege** : `mid = (lo + hi) // 2` — en Python pas d'overflow, mais en C/Java il faut `lo + (hi - lo) // 2`.

### Variante 2 — Lower bound (premier index `>= target`)

```python
def lower_bound(arr, target):
    """Return the first index i such that arr[i] >= target (or len(arr) if none)."""
    lo, hi = 0, len(arr)           # Note: hi = len (pas len - 1)
    while lo < hi:                 # Note: strict <
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1           # arr[mid] trop petit, chercher a droite
        else:
            hi = mid               # arr[mid] >= target, candidat valide, chercher a gauche
    return lo
```

### Variante 3 — Upper bound (premier index `> target`)

```python
def upper_bound(arr, target):
    """Return the first index i such that arr[i] > target (or len(arr) if none)."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

**L'utilite de lower/upper bound** : compter les occurrences de `target` en O(log n) :
```python
count = upper_bound(arr, target) - lower_bound(arr, target)
```

Python a tout ca dans `bisect` :
```python
from bisect import bisect_left, bisect_right

bisect_left(arr, target)    # lower_bound
bisect_right(arr, target)   # upper_bound
```

**En entretien** : tu peux utiliser `bisect`, mais sache ecrire la version manuelle — on te le demandera.

### Variante 4 — Binary search dans un tableau rotated

```python
def search_rotated(nums, target):
    """Search target in a rotated sorted array. Returns index or -1."""
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid

        # Key insight: at least ONE half is sorted. Identify which one.
        if nums[lo] <= nums[mid]:
            # Left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1         # Target is in the sorted left half
            else:
                lo = mid + 1         # Target is in the unsorted right half
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1         # Target is in the sorted right half
            else:
                hi = mid - 1         # Target is in the unsorted left half
    return -1
# Time: O(log n), Space: O(1)
```

**Insight cle** : dans un rotated sorted array, au moins une des deux moities (gauche ou droite de `mid`) est triee. On identifie laquelle en comparant `nums[lo]` et `nums[mid]`. Puis on verifie si `target` est dans cette moitie triee et on ajuste les bounds.

---

## 5. Quickselect — K-th element en O(n) moyen

### Concept

On veut le K-ieme plus petit element d'un tableau. Le tri complet est O(n log n), mais on n'a pas besoin de tout trier — on a juste besoin du K-ieme. Quickselect adapte le partition de quicksort pour recurser UNIQUEMENT dans la moitie qui contient le K-ieme.

```python
import random

def quickselect(arr, k):
    """Return the k-th smallest element (0-indexed) in arr. Mutates arr."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        pivot_idx = partition(arr, lo, hi)
        if pivot_idx == k:
            return arr[pivot_idx]
        elif pivot_idx < k:
            lo = pivot_idx + 1              # K-ieme est a droite du pivot
        else:
            hi = pivot_idx - 1              # K-ieme est a gauche du pivot


def partition(arr, lo, hi):
    """Lomuto partition. Returns the final index of the pivot."""
    # RANDOMIZE pour eviter le worst case O(n^2) sur les tableaux tries
    pivot_idx = random.randint(lo, hi)
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]

    pivot = arr[hi]
    i = lo                                   # Next position for elements < pivot
    for j in range(lo, hi):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]        # Pivot goes to its final position
    return i
# Time: O(n) moyen, O(n^2) worst case (tres rare avec randomisation)
# Space: O(1) iteratif
```

**Quand l'utiliser** :
- "K-ieme plus petit/plus grand" — quickselect ou heap
- "Top K elements" — heap est souvent plus simple
- Si K est petit, un min-heap de taille K suffit : O(n log K)

**Alternative heap** :
```python
import heapq

def kth_largest(nums, k):
    """Return the k-th LARGEST element."""
    # Maintain a min-heap of size k
    heap = nums[:k]
    heapq.heapify(heap)
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    return heap[0]
# Time: O(n log k), Space: O(k)
```

---

## 6. Binary Search sur une MATRICE 2D

### Cas 1 — Matrice avec lignes triees ET colonnes triees (LeetCode 74)

Si la matrice est **entierement triee** (fin de la ligne `i` < debut de la ligne `i+1`), on peut la traiter comme un tableau 1D :

```python
def search_matrix_74(matrix, target):
    """Search in a matrix that is sorted row-wise AND end-of-row < start-of-next-row."""
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    lo, hi = 0, rows * cols - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        val = matrix[mid // cols][mid % cols]
        if val == target:
            return True
        elif val < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False
# Time: O(log(m*n)), Space: O(1)
```

### Cas 2 — Matrice avec lignes triees ET colonnes triees separement (LeetCode 240)

Si chaque ligne est triee et chaque colonne est triee mais **pas necessairement globalement**, on utilise le **staircase search** :

```python
def search_matrix_240(matrix, target):
    """Search where rows AND columns are sorted independently."""
    if not matrix or not matrix[0]:
        return False
    # Start at top-right corner
    r, c = 0, len(matrix[0]) - 1
    while r < len(matrix) and c >= 0:
        if matrix[r][c] == target:
            return True
        elif matrix[r][c] > target:
            c -= 1                  # Move left: all values below are larger
        else:
            r += 1                  # Move down: all values to the left are smaller
    return False
# Time: O(m + n), Space: O(1)
```

**Pourquoi commencer en haut a droite ?** Parce que de cet angle, on a deux directions orthogonales : aller a gauche diminue, aller en bas augmente. A chaque step, on elimine une ligne OU une colonne entiere.

---

## 7. Decision Tree — Quel pattern utiliser ?

```
Le probleme implique des donnees TRIEES ou qu'il faut trier ?
|
├── Donnees deja triees ?
│   ├── Chercher un element ou sa position ?
│   │   └── BINARY SEARCH (exact, lower_bound, upper_bound, rotated)
│   ├── Matrice triee globalement ?
│   │   └── BINARY SEARCH 2D (treat as 1D with div/mod)
│   └── Matrice triee ligne/colonne independamment ?
│       └── STAIRCASE SEARCH (top-right start)
|
├── Donnees non triees, mais tu veux un element PARTICULIER ?
│   ├── "K-ieme plus petit/grand" ?
│   │   ├── K petit fixe (<= 100) → HEAP O(n log k)
│   │   └── K arbitraire → QUICKSELECT O(n) moyen
│   └── "Top K frequent/smallest/largest" ?
│       └── HEAP ou BUCKET sort
|
├── Donnees non triees, tu veux les trier par un critere ?
│   ├── Critere simple (int, str) → sorted(arr, key=...)
│   ├── Critere compose (N champs) → sorted(arr, key=lambda x: (a, b, c))
│   └── Critere non exprimable en cle → cmp_to_key
|
└── Tu veux explorer l'espace des REPONSES POSSIBLES ?
    └── BINARY SEARCH ON ANSWER
        Ex: Koko eating bananas, split array largest sum, capacity to ship
```

**Raccourcis mentaux** :

| Signal dans l'enonce | Pattern |
|---------------------|---------|
| "sorted array" / "sorted matrix" | Binary search |
| "rotated" | Rotated binary search |
| "find the K-th..." | Quickselect ou heap |
| "top K frequent" | Heap ou bucket sort |
| "smallest/largest such that..." | Binary search on answer |
| "custom order" | cmp_to_key |

---

## 8. Complexites et pieges

### Complexites de reference

| Algorithme | Temps | Space | Stable ? |
|-----------|-------|-------|----------|
| Timsort (Python) | O(n log n) | O(n) | Oui |
| Quicksort | O(n log n) moyen, O(n^2) worst | O(log n) | Non |
| Mergesort | O(n log n) | O(n) | Oui |
| Heapsort | O(n log n) | O(1) | Non |
| Counting sort | O(n + k) | O(k) | Oui |
| Binary search | O(log n) | O(1) | N/A |
| Quickselect | O(n) moyen | O(1) | N/A |

### Pieges courants

**Piege 1 — `sort()` vs `sorted()`**
```python
arr = [3, 1, 2]
# arr.sort() modifie en place et retourne None
result = arr.sort()
print(result)     # None — erreur classique !
print(arr)        # [1, 2, 3]

# sorted() retourne une nouvelle liste
result = sorted(arr)
print(result)     # [1, 2, 3]
```

**Piege 2 — Off-by-one dans binary search**
```python
# Les TROIS styles incompatibles :
# 1. lo <= hi, hi = len-1, hi = mid-1   — pour exact match
# 2. lo < hi, hi = len, hi = mid        — pour lower/upper bound
# 3. lo + 1 < hi                        — pour "trouver la frontiere entre 2 conditions"

# Choisir UN style et s'y tenir. Melanger cree des off-by-one infernaux.
```

**Piege 3 — Custom comparator qui ne retourne pas un int**
```python
# En Python 3, cmp a ete supprime. Utiliser cmp_to_key pour les anciennes fonctions.
# MAUVAIS
sorted(arr, cmp=lambda a, b: a < b)   # TypeError en Python 3

# BON
from functools import cmp_to_key
sorted(arr, key=cmp_to_key(lambda a, b: -1 if a < b else 1 if a > b else 0))
```

**Piege 4 — Quickselect non randomise**
```python
# Si on choisit toujours le dernier element comme pivot, le worst case est O(n^2)
# sur les tableaux deja tries. TOUJOURS randomiser le choix du pivot.
```

---

## 9. Flash Cards — Revision espacee

**Q1** : Quelle est la complexite de Timsort dans le meilleur cas ?
> **R1** : O(n) quand l'input est deja trie ou quasi trie. Timsort detecte les "runs" et ne les touche pas. Worst case O(n log n), stable, avec O(n) memoire auxiliaire.

**Q2** : Explique en une phrase la difference entre `lower_bound` et `upper_bound`.
> **R2** : `lower_bound(arr, x)` retourne le premier index `i` tel que `arr[i] >= x`. `upper_bound(arr, x)` retourne le premier index `i` tel que `arr[i] > x`. La difference des deux donne le nombre d'occurrences de x en O(log n).

**Q3** : Dans un binary search sur un rotated sorted array, comment identifies-tu quelle moitie est triee ?
> **R3** : En comparant `nums[lo]` et `nums[mid]`. Si `nums[lo] <= nums[mid]`, la moitie gauche est triee. Sinon, la moitie droite est triee. Au moins une des deux l'est toujours, parce que la rotation casse l'ordre en au plus un point.

**Q4** : Pourquoi `staircase search` sur une matrice commence en haut a droite et pas en haut a gauche ?
> **R4** : Parce qu'en haut a droite, les deux directions (gauche = decroissant, bas = croissant) sont orthogonales et permettent d'eliminer une ligne ou une colonne entiere a chaque step. En haut a gauche, aller a droite ET aller en bas augmentent tous deux — impossible de decider dans quelle direction aller.

**Q5** : Quelle est la complexite moyenne et worst case de quickselect ? Comment reduire la probabilite du worst case ?
> **R5** : O(n) moyen (chaque partition reduit l'espace de recherche d'une fraction constante), O(n^2) worst case. On reduit la probabilite du worst case en randomisant le choix du pivot avant chaque partition — avec randomisation, la probabilite du O(n^2) est essentiellement nulle sur des inputs adverses.

---

## Resume — Key Takeaways

1. **`sorted(arr, key=...)` est ton ami** — maitrise les tuples comme cles composees
2. **Timsort est stable et adaptatif** — exploite la stabilite pour les tris multi-criteres
3. **Binary search = 4 variantes** : exact, lower_bound, upper_bound, rotated
4. **`bisect` module** donne lower_bound et upper_bound gratuits
5. **Quickselect** trouve le K-ieme en O(n) moyen — mieux que tri complet
6. **Rotated search** : au moins une moitie est triee a chaque step
7. **Staircase search** sur matrice 2D : O(m+n), partir d'un coin strategique
8. **Binary search on answer** : pattern meta pour optimisation (Koko, ship capacity)
