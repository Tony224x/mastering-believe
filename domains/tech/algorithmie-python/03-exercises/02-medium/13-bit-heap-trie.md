# Exercices Medium — Bit Manipulation, Heaps & Tries

> `heapq` est un min-heap : pour un max-heap, pousser `-x`. Trie = arbre prefix avec `children` dict + `is_end`.

---

## Exercice 4 : Heap — Top K Frequent Elements

### Objectif

Combiner un `Counter` et un heap de taille k pour extraire les k elements les plus frequents en O(n log k). Pattern omnipresent (top trends, top words, leaderboard) qui evite le tri complet O(n log n).

### Consigne

Etant donne un tableau `nums` et un entier `k`, retourne les `k` elements les plus frequents. L'ordre des elements retournes n'a pas d'importance. Tu peux supposer que la reponse est unique.

```python
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """
    Return the k most frequent elements (order does not matter).
    Target O(n log k).
    """
    pass
```

**Indice** : compte les frequences avec `collections.Counter`. Maintiens un **min-heap de taille k** de tuples `(freq, value)` : pour chaque `(value, freq)`, push, et si le heap depasse k, pop le plus petit. A la fin, le heap contient les k plus frequents. (`heapq.nlargest(k, ...)` est aussi accepte si tu sais le recoder.)

### Tests

```python
assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
assert top_k_frequent([1], 1) == [1]
assert sorted(top_k_frequent([4, 4, 4, 5, 5, 6], 2)) == [4, 5]
assert sorted(top_k_frequent([1, 2, 3, 4], 4)) == [1, 2, 3, 4]
assert top_k_frequent([7, 7, 7], 1) == [7]
assert sorted(top_k_frequent([-1, -1, -2, -2, -2, 3], 2)) == [-2, -1]
```

### Criteres de reussite

- [ ] Compte les frequences avec `Counter`
- [ ] Min-heap de taille k (ou `nlargest`), pas de tri complet
- [ ] Le heap reste de taille k (pop quand il deborde)
- [ ] Gere k = nombre d'elements distincts, valeurs negatives
- [ ] Complexite O(n log k) temps, O(n) espace

---

## Exercice 5 : Trie — Implement Trie (Prefix Tree)

### Objectif

Implementer un trie complet (`insert`, `search`, `starts_with`). C'est la structure de base de tout le pattern prefix : autocomplete, dictionnaires, word search II. Sans `is_end`, impossible de distinguer un mot complet d'un simple prefixe.

### Consigne

Implemente la classe `Trie` avec trois methodes :
- `insert(word)` : ajoute `word` au trie
- `search(word)` : retourne `True` si `word` a ete insere exactement
- `starts_with(prefix)` : retourne `True` si au moins un mot insere commence par `prefix`

```python
class Trie:
    def __init__(self):
        pass

    def insert(self, word: str) -> None:
        pass

    def search(self, word: str) -> bool:
        pass

    def starts_with(self, prefix: str) -> bool:
        pass
```

**Indice** : chaque noeud a un `children` (dict char→node) et un flag `is_end`. `insert` descend en creant les noeuds manquants, puis marque `is_end = True`. `search` descend et verifie `is_end` a la fin. `starts_with` descend sans verifier `is_end` (factorise les deux via un helper `_walk`).

### Tests

```python
trie = Trie()
trie.insert("apple")
assert trie.search("apple") is True
assert trie.search("app") is False           # Prefix, not a complete word
assert trie.starts_with("app") is True
trie.insert("app")
assert trie.search("app") is True            # Now it is a word

t2 = Trie()
assert t2.search("anything") is False        # Empty trie
assert t2.starts_with("") is True            # Empty prefix matches root
t2.insert("")
assert t2.search("") is True                 # Empty word can be inserted/searched

t3 = Trie()
for w in ["a", "ab", "abc"]:
    t3.insert(w)
assert all(t3.search(w) for w in ["a", "ab", "abc"])
assert t3.starts_with("ab") is True
assert t3.search("abcd") is False
```

### Criteres de reussite

- [ ] Noeud avec `children` (dict) + flag `is_end`
- [ ] `insert` cree les noeuds manquants et marque `is_end`
- [ ] `search` distingue mot complet (`is_end`) de prefixe
- [ ] `starts_with` ne verifie pas `is_end`
- [ ] Gere le mot vide et le trie vide
- [ ] `insert`/`search`/`starts_with` en O(L) ou L = longueur du mot

---

## Exercice 6 : Bit Manipulation — Counting Bits

### Objectif

Calculer le nombre de bits a 1 pour chaque entier de 0 a n en O(n) via une recurrence DP sur les bits (`dp[i] = dp[i >> 1] + (i & 1)`). C'est le pont entre bit manipulation et programmation dynamique.

### Consigne

Etant donne un entier `n`, retourne un tableau `ans` de longueur `n + 1` ou `ans[i]` est le nombre de bits a 1 dans la representation binaire de `i`.

Vise une solution en **O(n)** (une seule passe), pas O(n log n) en comptant chaque nombre independamment.

```python
def count_bits(n: int) -> list[int]:
    """
    ans[i] = number of set bits in i, for i in 0..n. Target O(n).
    """
    pass
```

**Indice** : `i >> 1` est `i` sans son dernier bit ; `i & 1` est ce dernier bit. Donc `dp[i] = dp[i >> 1] + (i & 1)`. Variante : `dp[i] = dp[i & (i - 1)] + 1` (Brian Kernighan : `i & (i-1)` efface le plus petit bit a 1).

### Tests

```python
assert count_bits(0) == [0]
assert count_bits(2) == [0, 1, 1]
assert count_bits(5) == [0, 1, 1, 2, 1, 2]
# Cross-check against the naive bin().count("1") for a range of n
for n in range(0, 50):
    assert count_bits(n) == [bin(i).count("1") for i in range(n + 1)]
```

### Criteres de reussite

- [ ] DP en une passe O(n) (pas `bin().count` par nombre)
- [ ] Recurrence correcte (`dp[i >> 1] + (i & 1)` ou Brian Kernighan)
- [ ] Tableau de longueur exacte `n + 1`
- [ ] Gere `n = 0` → `[0]`
- [ ] Cross-check avec la version naive valide
