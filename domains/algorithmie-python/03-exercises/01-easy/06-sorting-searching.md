# Exercices Easy — Sorting & Searching

---

## Exercice 1 : Binary Search — Exact Match

### Objectif

Ecrire un binary search iteratif from scratch, sans `bisect`. C'est la base — si tu n'arrives pas a l'ecrire en 2 minutes sans erreur, aucune variante plus avancee ne te sauvera.

### Consigne

Etant donne un tableau `nums` trie en ordre croissant et un entier `target`, retourne l'index de `target` s'il est present, sinon retourne `-1`.

Tu dois ecrire un algorithme en **O(log n)** temps et **O(1)** espace. **Ne pas utiliser `bisect` ni `index`.**

```python
def binary_search(nums: list[int], target: int) -> int:
    """
    Return the index of target in the sorted array, or -1 if not found.
    Must run in O(log n) time with O(1) extra space.
    """
    pass
```

### Tests

```python
assert binary_search([-1, 0, 3, 5, 9, 12], 9) == 4
assert binary_search([-1, 0, 3, 5, 9, 12], 2) == -1
assert binary_search([5], 5) == 0
assert binary_search([5], -5) == -1
assert binary_search([], 0) == -1
assert binary_search([1, 2, 3, 4, 5], 1) == 0
assert binary_search([1, 2, 3, 4, 5], 5) == 4
```

### Criteres de reussite

- [ ] Utilise `lo` et `hi` comme bounds
- [ ] Condition de boucle `lo <= hi` (inclusif des deux cotes)
- [ ] `mid = (lo + hi) // 2`
- [ ] Met a jour `lo = mid + 1` et `hi = mid - 1` (jamais `mid` seul)
- [ ] Gere le edge case tableau vide
- [ ] Complexite O(log n), O(1) espace

---

## Exercice 2 : Custom Sort — Sort Strings by Frequency

### Objectif

Maitriser le tri avec `key=` et un critere compose. Tres frequent en entretien pour tout probleme de "top K" ou "groupement par frequence".

### Consigne

Etant donne une chaine `s`, trie ses caracteres par frequence **decroissante**. Si plusieurs caracteres ont la meme frequence, ils peuvent apparaitre dans n'importe quel ordre entre eux. Retourne la chaine triee.

Exemple : `"tree"` → `"eert"` ou `"eetr"` (e apparait 2 fois, t et r une fois chacun)

```python
def frequency_sort(s: str) -> str:
    """
    Return s with characters sorted by frequency descending.
    Characters with the same frequency may appear in any order.
    """
    pass
```

### Tests

```python
def check(s, expected_set):
    result = frequency_sort(s)
    assert len(result) == len(s)
    assert result in expected_set, f"Got {result}"

check("tree", {"eert", "eetr"})
check("cccaaa", {"cccaaa", "aaaccc"})
check("Aabb", {"bbAa", "bbaA"})
check("", {""})
check("a", {"a"})
```

### Criteres de reussite

- [ ] Utilise `collections.Counter` pour compter les frequences
- [ ] Utilise `sorted` avec `key=` pour trier par frequence decroissante
- [ ] Reconstruit la chaine en multipliant chaque caractere par sa frequence
- [ ] Complexite O(n log k) ou k = nombre de caracteres uniques
- [ ] Gere le edge case chaine vide

---

## Exercice 3 : Quickselect / Heap — Kth Largest Element

### Objectif

Resoudre le classique "K-ieme plus grand element" sans utiliser un tri complet. Tu peux choisir entre quickselect et heap — les deux sont acceptables en entretien.

### Consigne

Etant donne un tableau `nums` et un entier `k`, retourne le **k-ieme plus grand element** du tableau (k est 1-indexed, donc k=1 est le max, k=2 est le second max, etc.).

Note : il s'agit du k-ieme plus grand dans l'ordre TRIE, pas du k-ieme element distinct. Donc dans `[3, 2, 1, 5, 6, 4]` avec k=2, le resultat est `5`.

```python
def kth_largest(nums: list[int], k: int) -> int:
    """
    Return the k-th largest element (1-indexed) in nums.
    """
    pass
```

### Tests

```python
assert kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
assert kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
assert kth_largest([1], 1) == 1
assert kth_largest([1, 2], 1) == 2
assert kth_largest([1, 2], 2) == 1
assert kth_largest([7, 7, 7, 7], 2) == 7
```

### Criteres de reussite

- [ ] Utilise un min-heap de taille k OU quickselect (pas `sorted(nums)[-k]`)
- [ ] Si heap : utilise `heapq.heappush` et `heapq.heappop` (ou `heapify` + `heapreplace`)
- [ ] Si quickselect : randomise le pivot pour eviter le worst case O(n^2)
- [ ] Heap : complexite O(n log k) temps, O(k) espace
- [ ] Quickselect : O(n) moyen temps, O(1) espace
- [ ] Gere les doublons correctement (k=2 dans `[7,7,7,7]` retourne 7, pas une exception)
