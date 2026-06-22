# Exercices Hard — Hash Maps & Sets

---

## Exercice 7 : Frequency + Grouping — Minimum Window Substring (revisited with hash map focus)

### Objectif

Combiner sliding window + deux hash maps (need/have) pour resoudre le probleme de fenetre minimale — le probleme hard le plus classique qui teste la maitrise des hash maps en entretien.

### Consigne

Etant donne deux strings `s` et `t`, retourne la **plus petite sous-chaine** de `s` qui contient **tous les caracteres** de `t` (y compris les doublons).

S'il n'existe pas de telle sous-chaine, retourne `""`.

```python
def min_window(s: str, t: str) -> str:
    """
    Return the minimum window substring of s containing all characters of t.
    If no such window exists, return "".
    """
    pass
```

**Approche attendue** :
1. Construire un Counter `need` a partir de `t`
2. Maintenir un Counter `window` pour la fenetre courante
3. Tracker `have` = nombre de caracteres uniques satisfaits (count dans window >= count dans need)
4. Quand `have == len(need)` : tenter de reduire la fenetre depuis la gauche
5. Maintenir le meilleur (plus court) resultat

### Tests

```python
assert min_window("ADOBECODEBANC", "ABC") == "BANC"
assert min_window("a", "a") == "a"
assert min_window("a", "aa") == ""             # Can't form "aa" from single "a"
assert min_window("", "abc") == ""
assert min_window("abc", "") == ""
assert min_window("aa", "aa") == "aa"
assert min_window("bba", "ab") == "ba"
assert min_window("aaflslflsldkalskaaa", "aaa") == "aaa"

# Stress test: t contains chars not in s
assert min_window("abcdef", "z") == ""

# All same characters
assert min_window("aaaa", "a") == "a"
assert min_window("aaaa", "aa") == "aa"
```

### Criteres de reussite

- [ ] Utilise deux hash maps (need et window) avec un compteur `have` pour eviter de comparer les maps entiers a chaque etape
- [ ] Complexite O(n + m) temps ou n = len(s), m = len(t)
- [ ] Complexite O(m + alphabet) espace
- [ ] Comprend l'optimisation : incrementer/decrementer `have` uniquement quand un seuil est franchi
- [ ] La fenetre est reduite correctement depuis la gauche (pas d'erreur off-by-one)
- [ ] Tous les tests passent, y compris les edge cases (s vide, t vide, pas de solution)

---

## Exercice 8 : Index Mapping + Set — LRU Cache

### Objectif

Implementer un LRU Cache en combinant un hash map (acces O(1) par cle) et une structure ordonnee (suivi de l'ordre d'utilisation) — un design classique qui teste la maitrise profonde des hash maps.

### Consigne

Implemente une structure `LRUCache` qui supporte les operations suivantes en **O(1)** chacune :
- `get(key)` : retourne la valeur associee a la cle, ou -1 si la cle n'existe pas. Marque la cle comme recemment utilisee.
- `put(key, value)` : insere ou met a jour la paire (key, value). Si la capacite est depassee apres l'insertion, retire la cle **la moins recemment utilisee** (Least Recently Used).

```python
class LRUCache:
    def __init__(self, capacity: int):
        """Initialize LRU cache with positive capacity."""
        pass

    def get(self, key: int) -> int:
        """Return value for key if it exists, else -1. Marks key as recently used."""
        pass

    def put(self, key: int, value: int) -> None:
        """Insert or update key-value pair. Evict LRU if over capacity."""
        pass
```

**Approche attendue** : utiliser `collections.OrderedDict` (Python) qui combine hash map + doubly-linked list. Ou implementer manuellement avec un dict + une doubly-linked list (bonus pour l'entretien).

**Hint OrderedDict** :
- `move_to_end(key)` : marque comme recemment utilise en O(1)
- `popitem(last=False)` : retire le plus ancien (LRU) en O(1)

### Tests

```python
# Example from LeetCode 146
cache = LRUCache(2)

cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1        # Returns 1, marks key 1 as recently used

cache.put(3, 3)                 # Evicts key 2 (LRU: 2 was used before 1)
assert cache.get(2) == -1       # Key 2 was evicted

cache.put(4, 4)                 # Evicts key 1 (LRU: 1 was used before 3)
assert cache.get(1) == -1       # Key 1 was evicted
assert cache.get(3) == 3        # Key 3 still exists
assert cache.get(4) == 4        # Key 4 still exists

# Capacity 1
cache = LRUCache(1)
cache.put(1, 1)
cache.put(2, 2)                 # Evicts key 1
assert cache.get(1) == -1
assert cache.get(2) == 2

# Update existing key (doesn't increase size)
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
cache.put(1, 10)                # Update key 1's value, marks it as recently used
assert cache.get(1) == 10
cache.put(3, 3)                 # Evicts key 2 (LRU, not key 1 which was just updated)
assert cache.get(2) == -1
assert cache.get(1) == 10

# Large sequence
cache = LRUCache(3)
for i in range(10):
    cache.put(i, i * 10)
# Only keys 7, 8, 9 should remain
assert cache.get(6) == -1
assert cache.get(7) == 70
assert cache.get(8) == 80
assert cache.get(9) == 90
```

### Criteres de reussite

- [ ] `get` et `put` sont tous les deux **O(1)** (pas O(n))
- [ ] L'eviction supprime bien le **moins recemment utilise** (pas le premier insere si un get l'a "rafraichi")
- [ ] La mise a jour d'une cle existante ne change pas la capacite mais rafraichit la position LRU
- [ ] Implementee avec OrderedDict (solution Pythonic) ou dict + doubly-linked list (solution manuelle)
- [ ] Tous les tests passent, y compris capacite 1 et sequences longues
