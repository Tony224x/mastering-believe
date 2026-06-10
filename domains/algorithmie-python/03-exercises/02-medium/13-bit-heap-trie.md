# Exercices Medium — Bit Manipulation, Heaps & Tries

---

## Exercice 4 : Bit Counting — Single Number II

### Objectif

Generaliser le XOR trick : quand chaque element apparait **3 fois** (sauf un), XOR ne suffit plus. Il faut compter les bits modulo 3 — et gerer le piege Python des entiers negatifs (pas de 32 bits natifs).

### Consigne

Etant donne un tableau `nums` ou chaque element apparait exactement **trois fois** sauf un qui apparait **une fois**, retourne cet element unique.

**Contraintes : O(n) temps, O(1) espace.** (Le `Counter` est O(n) espace — ecris-le comme oracle mais il ne valide pas.)

```python
def single_number_ii(nums: list[int]) -> int:
    """
    Element appearing once when all others appear three times.
    O(n) time, O(1) space — bit counting mod 3.
    """
    pass
```

**Indice** : pour chaque position de bit i (0..31), compte combien de nombres ont ce bit a 1. Ce compte mod 3 donne le bit i de la reponse.

**Piege Python** : les ints Python sont en precision arbitraire. Pour les nombres negatifs, travaille en complement a deux 32 bits (`num & 0xFFFFFFFF` ou masque par bit) et si le bit 31 du resultat est a 1, retranche `2**32`.

### Tests

```python
assert single_number_ii([2, 2, 3, 2]) == 3
assert single_number_ii([0, 1, 0, 1, 0, 1, 99]) == 99
assert single_number_ii([7]) == 7
assert single_number_ii([-2, -2, 1, -2]) == 1
assert single_number_ii([-4, -4, -4, -5]) == -5         # Negative answer — the Python trap
assert single_number_ii([1, 1, 1, 0]) == 0

# Oracle check on random inputs
import random
for _ in range(50):
    triples = random.sample(range(-50, 50), 6)
    single = random.choice([x for x in range(-50, 50) if x not in triples])
    arr = triples * 3 + [single]
    random.shuffle(arr)
    assert single_number_ii(arr) == single
```

### Criteres de reussite

- [ ] Comptage par position de bit, modulo 3 — pas de dict/Counter dans la version finale
- [ ] Les negatifs sont geres (complement a deux 32 bits, correction `- 2**32`)
- [ ] O(n * 32) = O(n) temps, O(1) espace
- [ ] Oracle Counter ecrit et confronte sur des inputs aleatoires
- [ ] Tous les tests passent

---

## Exercice 5 : Heap de taille k — K Closest Points to Origin

### Objectif

Le pattern "Top K avec heap borne" : garder un **max-heap de taille k** (via negation) pour rester en O(n log k), au lieu de trier tout (O(n log n)) ou de heapifier tout (O(n + k log n)).

### Consigne

Etant donne une liste de points `points[i] = [x, y]` et un entier `k`, retourne les `k` points les plus proches de l'origine (distance euclidienne). L'ordre du resultat n'importe pas.

**Contrainte : O(n log k) temps, O(k) espace** — le heap ne doit jamais depasser k elements.

```python
def k_closest(points: list[list[int]], k: int) -> list[list[int]]:
    """
    The k points closest to the origin, in O(n log k) time and O(k) space.
    """
    pass
```

**Indices** :
- Pas besoin de `sqrt` : comparer les distances **au carre** suffit (la racine est monotone).
- Python n'a qu'un min-heap : pousse `(-dist_carre, x, y)` pour simuler un max-heap, et ejecte le plus lointain quand la taille depasse k.

### Tests

```python
def normalize(pts):
    return sorted(map(tuple, pts))

assert normalize(k_closest([[1, 3], [-2, 2]], 1)) == [(-2, 2)]
assert normalize(k_closest([[3, 3], [5, -1], [-2, 4]], 2)) == [(-2, 4), (3, 3)]
assert normalize(k_closest([[0, 1], [1, 0]], 2)) == [(0, 1), (1, 0)]
assert normalize(k_closest([[5, 5]], 1)) == [(5, 5)]
assert normalize(k_closest([[1, 1], [2, 2], [3, 3]], 3)) == [(1, 1), (2, 2), (3, 3)]
assert normalize(k_closest([[0, 0], [10, 10], [1, 1]], 2)) == [(0, 0), (1, 1)]
```

### Criteres de reussite

- [ ] Max-heap simule par negation, taille bornee a k en permanence
- [ ] `heapq.heappushpop` (ou push puis pop) utilise quand le heap est plein
- [ ] Distances au carre — aucun appel a `sqrt`
- [ ] Tu sais comparer les trois approches : tri O(n log n), heap-k O(n log k), `nsmallest` O(n log k)
- [ ] Tous les tests passent

---

## Exercice 6 : DP sur les bits — Counting Bits

### Objectif

Relier bits et DP : calculer le popcount de 0 a n en O(n) TOTAL grace a une recurrence sur les bits, au lieu de O(n log n) avec `bin(i).count("1")` par nombre.

### Consigne

Etant donne un entier `n`, retourne un tableau `ans` de longueur `n + 1` ou `ans[i]` est le nombre de bits a 1 dans `i`.

**Contrainte : O(n) temps total** — une passe, chaque valeur calculee en O(1) a partir d'une valeur precedente.

```python
def count_bits(n: int) -> list[int]:
    """
    ans[i] = number of 1-bits in i, computed in O(n) total.
    """
    pass
```

**Deux recurrences au choix (implemente les deux et verifie qu'elles concordent)** :
1. `ans[i] = ans[i >> 1] + (i & 1)` — retirer le dernier bit.
2. `ans[i] = ans[i & (i - 1)] + 1` — Brian Kernighan : `i & (i-1)` efface le plus petit bit a 1.

### Tests

```python
assert count_bits(2) == [0, 1, 1]
assert count_bits(5) == [0, 1, 1, 2, 1, 2]
assert count_bits(0) == [0]
assert count_bits(1) == [0, 1]
assert count_bits(16)[16] == 1                       # Power of two: single bit
assert count_bits(15)[15] == 4                       # 0b1111
assert count_bits(255)[255] == 8

# Cross-check against bin().count on a larger range
assert count_bits(1000) == [bin(i).count("1") for i in range(1001)]
```

### Criteres de reussite

- [ ] Les deux recurrences implementees et concordantes
- [ ] Tu sais expliquer POURQUOI `i & (i - 1)` efface le plus petit bit a 1
- [ ] O(n) temps total, pas de `bin()` ni de boucle interne sur les bits
- [ ] O(n) espace (le tableau resultat lui-meme)
- [ ] Tous les tests passent
