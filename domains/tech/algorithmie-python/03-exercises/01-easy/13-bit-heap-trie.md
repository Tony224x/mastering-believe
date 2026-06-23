# Exercices Easy — Bit Manipulation, Heaps & Tries

---

## Exercice 1 : Bit Manipulation — Single Number

### Objectif

Maitriser XOR comme outil pour trouver l'element unique dans un tableau ou les autres apparaissent en paires, en O(n) temps et O(1) espace.

### Consigne

Etant donne un tableau d'entiers `nums` ou chaque element apparait **exactement deux fois** sauf un qui apparait **une seule fois**, retourne cet element.

Contraintes : solution en O(n) temps et O(1) espace.

```python
def single_number(nums: list[int]) -> int:
    """
    Find the element that appears only once; all others appear twice.
    Must be O(n) time and O(1) extra space.
    """
    pass
```

### Tests

```python
assert single_number([2, 2, 1]) == 1
assert single_number([4, 1, 2, 1, 2]) == 4
assert single_number([1]) == 1
assert single_number([-1, -1, -2]) == -2
assert single_number([0, 1, 0]) == 1
assert single_number([7, 7, 3, 5, 3]) == 5
```

### Criteres de reussite

- [ ] Solution utilisant XOR (pas de dict, pas de set, pas de tri)
- [ ] Complexite O(n) temps, O(1) espace
- [ ] Fonctionne avec des nombres negatifs
- [ ] Tous les tests passent

---

## Exercice 2 : Heap — Kth Largest Element

### Objectif

Utiliser un min-heap de taille k pour trouver le k-ieme plus grand element en O(n log k).

### Consigne

Etant donne un tableau d'entiers `nums` et un entier `k`, retourne le k-ieme plus grand element (dans l'ordre trie, pas le k-ieme distinct).

Contraintes : solution en O(n log k).

```python
import heapq

def find_kth_largest(nums: list[int], k: int) -> int:
    """
    Return the kth largest element (1-indexed).
    Must be O(n log k).
    """
    pass
```

### Tests

```python
assert find_kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
assert find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
assert find_kth_largest([1], 1) == 1
assert find_kth_largest([1, 2], 1) == 2
assert find_kth_largest([1, 2], 2) == 1
assert find_kth_largest([7, 6, 5, 4, 3, 2, 1], 3) == 5
```

### Criteres de reussite

- [ ] Utilise `heapq` avec un heap de **taille k** (pas n)
- [ ] Complexite O(n log k) temps, O(k) espace
- [ ] Ne fait PAS `sorted(nums)[-k]` (O(n log n))
- [ ] Tous les tests passent

---

## Exercice 3 : Trie — Implement Prefix Tree

### Objectif

Implementer un Trie avec insertion, recherche exacte et recherche par prefixe.

### Consigne

Implemente une classe `Trie` avec trois methodes :
- `insert(word)` : ajoute un mot au trie
- `search(word)` : retourne `True` si le mot est dans le trie (mot complet)
- `starts_with(prefix)` : retourne `True` si un mot du trie commence par ce prefixe

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

### Tests

```python
trie = Trie()
trie.insert("apple")
assert trie.search("apple") == True
assert trie.search("app") == False
assert trie.starts_with("app") == True
trie.insert("app")
assert trie.search("app") == True

trie2 = Trie()
trie2.insert("hello")
trie2.insert("help")
trie2.insert("helicopter")
assert trie2.starts_with("hel") == True
assert trie2.starts_with("helix") == False
assert trie2.search("hell") == False
assert trie2.search("help") == True

trie3 = Trie()
assert trie3.search("") == False  # Empty trie, nothing to search
assert trie3.starts_with("") == True  # Empty prefix matches anything (including nothing)
```

### Criteres de reussite

- [ ] Utilise une classe `TrieNode` interne avec un dict `children` et un flag `is_end`
- [ ] `insert`, `search`, `starts_with` sont tous en O(L) ou L = longueur du mot
- [ ] Distingue bien un mot complet ("app") d'un prefixe ("ap")
- [ ] Tous les tests passent
