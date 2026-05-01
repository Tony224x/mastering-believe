# Exercices Medium — Hash Maps & Sets

---

## Exercice 4 : Grouping — Group Anagrams

### Objectif

Maitriser le pattern de grouping avec defaultdict(list) en utilisant une cle de hash calculee — pattern reutilisable dans des dizaines de variantes.

### Consigne

Etant donne un tableau de strings `strs`, regroupe les anagrammes ensemble. Tu peux retourner les groupes dans n'importe quel ordre.

Deux strings sont des anagrammes si elles contiennent exactement les memes caracteres avec les memes frequences.

```python
def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Group strings that are anagrams of each other.
    Return a list of groups (each group is a list of anagram strings).
    """
    pass
```

**Indice** : toutes les anagrammes d'un mot produisent la meme "signature". Cette signature peut etre le mot trie (`tuple(sorted(s))`) ou un tuple de frequences de caracteres.

### Tests

```python
result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
# Expected groups (order doesn't matter):
# [["eat","tea","ate"], ["tan","nat"], ["bat"]]
normalized = sorted([sorted(g) for g in result])
assert normalized == sorted([["ate", "eat", "tea"], ["nat", "tan"], ["bat"]])

result = group_anagrams([""])
assert result == [[""]]

result = group_anagrams(["a"])
assert result == [["a"]]

# All same anagrams
result = group_anagrams(["abc", "bca", "cab"])
assert len(result) == 1
assert sorted(result[0]) == ["abc", "bca", "cab"]

# No anagrams at all
result = group_anagrams(["abc", "def", "ghi"])
assert len(result) == 3
```

### Criteres de reussite

- [ ] Utilise defaultdict(list) avec une cle de signature (sorted ou frequency tuple)
- [ ] Complexite O(n * k log k) avec tri ou O(n * k) avec frequency tuple (ou k = longueur moyenne des strings)
- [ ] La cle est un type hashable (tuple, pas list)
- [ ] Gere les edge cases : string vide, un seul element
- [ ] Tous les tests passent

---

## Exercice 5 : Frequency + Set — Intersection of Two Arrays II

### Objectif

Combiner frequency counting et iteration pour calculer l'intersection avec doublons — un probleme classique qui teste la maitrise fine de Counter.

### Consigne

Etant donne deux tableaux d'entiers `nums1` et `nums2`, retourne un tableau contenant leur intersection. Chaque element du resultat doit apparaitre autant de fois qu'il apparait dans les deux tableaux. Le resultat peut etre dans n'importe quel ordre.

```python
def intersect(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    Return the intersection of two arrays, including duplicates.
    Each element appears min(count_in_nums1, count_in_nums2) times.
    """
    pass
```

**Follow-up** : 
- Et si `nums1` etait beaucoup plus petit que `nums2` ? Quelle optimisation ?
- Et si les elements de `nums2` etaient stockes sur disque et ne pouvaient pas tous etre charges en memoire ?

### Tests

```python
assert sorted(intersect([1, 2, 2, 1], [2, 2])) == [2, 2]
assert sorted(intersect([4, 9, 5], [9, 4, 9, 8, 4])) == [4, 9]
assert intersect([1, 2, 3], [4, 5, 6]) == []
assert sorted(intersect([1, 1, 1], [1, 1])) == [1, 1]
assert intersect([], [1, 2, 3]) == []
assert intersect([1, 2, 3], []) == []
assert sorted(intersect([3, 1, 2], [1, 1])) == [1]
```

### Criteres de reussite

- [ ] Utilise Counter ou dict de frequences (pas de tri + merge)
- [ ] Complexite O(n + m) temps, O(min(n, m)) espace (compteur sur le plus petit tableau)
- [ ] Chaque element apparait le bon nombre de fois (min des deux frequences)
- [ ] Comprend le follow-up : construire le Counter sur le plus petit tableau pour economiser la memoire
- [ ] Tous les tests passent

---

## Exercice 6 : Complement Lookup — Subarray Sum Equals K

### Objectif

Combiner prefix sum + hash map — la technique hybride la plus puissante pour les sous-tableaux avec contrainte de somme. (Revue depuis J2, mais ici on insiste sur le hash map.)

### Consigne

Etant donne un tableau d'entiers `nums` et un entier `k`, retourne le **nombre de sous-tableaux contigus** dont la somme est egale a `k`.

Le tableau peut contenir des nombres negatifs (le sliding window classique ne fonctionne PAS).

```python
def subarray_sum(nums: list[int], k: int) -> int:
    """
    Return the count of contiguous subarrays that sum to k.
    Array may contain negative numbers.
    """
    pass
```

**Indice** : utilise un running prefix sum. Si `prefix[j] - prefix[i] == k`, alors le sous-tableau `nums[i..j-1]` somme a k. Un hashmap stocke les frequences de chaque prefix sum deja vu.

### Tests

```python
assert subarray_sum([1, 1, 1], 2) == 2           # [1,1] at 0-1 and 1-2
assert subarray_sum([1, 2, 3], 3) == 2           # [1,2] and [3]
assert subarray_sum([1, -1, 0], 0) == 3          # [1,-1], [-1,0], [1,-1,0]
assert subarray_sum([0, 0, 0], 0) == 6           # Every subarray
assert subarray_sum([3, 4, 7, 2, -3, 1, 4, 2], 7) == 4
assert subarray_sum([-1, -1, 1], 0) == 1         # [-1, 1] at indices 1-2
assert subarray_sum([1], 0) == 0
assert subarray_sum([1], 1) == 1
```

### Criteres de reussite

- [ ] Utilise prefix sum + hash map (pas de double boucle O(n^2))
- [ ] Le hash map est initialise avec `{0: 1}` pour gerer les sous-tableaux partant de l'index 0
- [ ] Complexite O(n) temps, O(n) espace
- [ ] Comprend pourquoi le sliding window ne fonctionne pas avec des nombres negatifs
- [ ] Tous les tests passent
