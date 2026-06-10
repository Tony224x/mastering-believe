# Exercices Medium — Mock Interviews (fresh problems)

**Format** : trois mocks de 25-30 minutes chacun, a faire dans les conditions reelles (chrono, a voix haute, sans IDE avance). Applique le process en 6 etapes du cours : clarifier → exemples → brute force → optimiser → coder en parlant → tester.

---

## Mock 4 : Spiral Matrix (Medium)

### Objectif

Un probleme de pure simulation : zero algorithme savant, 100% gestion rigoureuse des frontieres. C'est un test de communication — l'intervieweur regarde si tu poses les invariants AVANT de coder, sinon c'est le festival des off-by-one.

### Consigne

Etant donne une matrice `m x n`, retourne tous ses elements en **ordre spirale** (droite → bas → gauche → haut, en spiralant vers l'interieur).

```python
def spiral_order(matrix: list[list[int]]) -> list[int]:
    """
    Return all elements of the matrix in spiral order.
    """
    pass
```

**Approche attendue** : quatre frontieres (`top`, `bottom`, `left`, `right`) resserrees apres chaque cote consomme. Les deux gardes du milieu (`if top <= bottom`, `if left <= right`) avant les passes "gauche" et "haut" sont LE point que teste ce probleme (matrices a une seule ligne/colonne).

### Tests

```python
assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
assert spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
assert spiral_order([[1]]) == [1]
assert spiral_order([[1, 2, 3]]) == [1, 2, 3]            # Single row — the guard trap
assert spiral_order([[1], [2], [3]]) == [1, 2, 3]        # Single column — the other guard trap
assert spiral_order([]) == []
assert spiral_order([[1, 2], [3, 4]]) == [1, 2, 4, 3]
assert spiral_order([[1, 2], [4, 5], [7, 8]]) == [1, 2, 5, 8, 7, 4]
```

### Criteres de reussite

- [ ] Quatre frontieres explicites, resserrees apres chaque cote
- [ ] Les gardes anti-double-passage sont presentes (les tests ligne/colonne uniques passent)
- [ ] Aucune case visitee deux fois, aucune oubliee — O(m * n) temps, O(1) espace auxiliaire
- [ ] Resolu en moins de 25 minutes en verbalisant les invariants
- [ ] Tous les tests passent

---

## Mock 5 : Insert Interval (Medium)

### Objectif

La variante "intervalle a inserer" de Merge Intervals : trois phases distinctes (avant / fusion / apres) au lieu d'un tri. Teste si tu sais EXPLOITER une precondition (la liste est deja triee et disjointe) au lieu de re-trier betement.

### Consigne

On te donne `intervals`, une liste d'intervalles **disjoints et tries** par debut, et un intervalle `new_interval`. Insere `new_interval` en fusionnant si necessaire, et retourne une liste toujours triee et disjointe.

**Contrainte : O(n) temps** — re-trier (O(n log n)) gaspille la precondition et ne valide pas.

```python
def insert_interval(intervals: list[list[int]], new_interval: list[int]) -> list[list[int]]:
    """
    Insert new_interval into sorted disjoint intervals, merging overlaps.
    O(n) single pass.
    """
    pass
```

**Les trois phases** :
1. Copier tous les intervalles qui finissent **avant** le debut de `new_interval`.
2. Fusionner tous ceux qui chevauchent (`start <= new_end` et `end >= new_start`) dans `new_interval`.
3. Copier le reste tel quel.

### Tests

```python
assert insert_interval([[1, 3], [6, 9]], [2, 5]) == [[1, 5], [6, 9]]
assert insert_interval([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]) == [[1, 2], [3, 10], [12, 16]]
assert insert_interval([], [5, 7]) == [[5, 7]]
assert insert_interval([[1, 5]], [2, 3]) == [[1, 5]]     # Fully contained
assert insert_interval([[1, 5]], [6, 8]) == [[1, 5], [6, 8]]    # After, no overlap
assert insert_interval([[6, 8]], [1, 5]) == [[1, 5], [6, 8]]    # Before, no overlap
assert insert_interval([[1, 5]], [5, 7]) == [[1, 7]]     # Touching bounds merge
assert insert_interval([[3, 4]], [1, 2]) == [[1, 2], [3, 4]]
```

### Criteres de reussite

- [ ] Une seule passe, trois phases visibles dans le code
- [ ] Pas de tri — la precondition "trie et disjoint" est exploitee
- [ ] Bornes qui se touchent fusionnees (`<=`, pas `<`)
- [ ] O(n) temps, O(n) espace (la liste resultat)
- [ ] Resolu en moins de 25 minutes
- [ ] Tous les tests passent

---

## Mock 6 : Insert Delete GetRandom O(1) (Medium — design)

### Objectif

Un mock de **design de structure de donnees** : chaque operation en O(1) force a combiner deux structures (list + dict) et a inventer le swap-with-last pour la suppression. C'est exactement le type de question qui revele si tu connais les complexites REELLES des structures Python.

### Consigne

Implemente `RandomizedSet` :
- `insert(val) -> bool` : insere `val` si absent. Retourne `True` si insere.
- `remove(val) -> bool` : retire `val` si present. Retourne `True` si retire.
- `get_random() -> int` : retourne un element existant **uniformement au hasard**.

**Contrainte : chaque operation en O(1) moyen.**

```python
import random

class RandomizedSet:
    def __init__(self):
        pass

    def insert(self, val: int) -> bool:
        pass

    def remove(self, val: int) -> bool:
        pass

    def get_random(self) -> int:
        pass
```

**Pourquoi pas un set seul ?** `random.choice` exige un indexable — un set ne l'est pas (le convertir en list est O(n)). **Pourquoi pas une list seule ?** `remove` par valeur est O(n).

**Le trick** : une list `values` + un dict `index_of`. Pour supprimer : **swap** l'element avec le dernier de la list, puis `pop()` — la suppression au milieu devient une suppression en queue, O(1).

### Tests

```python
rs = RandomizedSet()
assert rs.insert(1) == True
assert rs.insert(1) == False             # Already present
assert rs.remove(2) == False             # Absent
assert rs.insert(2) == True
assert rs.get_random() in (1, 2)
assert rs.remove(1) == True
assert rs.get_random() == 2
assert rs.remove(2) == True
assert rs.insert(2) == True              # Re-insert after removal

# Remove the LAST element (the swap edge case: element swaps with itself)
rs2 = RandomizedSet()
rs2.insert(10); rs2.insert(20); rs2.insert(30)
assert rs2.remove(30) == True
assert rs2.remove(10) == True
assert rs2.get_random() == 20

# Uniformity smoke test
rs3 = RandomizedSet()
for v in range(3):
    rs3.insert(v)
counts = {0: 0, 1: 0, 2: 0}
for _ in range(3000):
    counts[rs3.get_random()] += 1
assert all(c > 700 for c in counts.values())     # Roughly uniform (expected ~1000)
```

### Criteres de reussite

- [ ] list + dict combines ; `get_random` = `random.choice` sur la list
- [ ] `remove` utilise le swap-with-last puis `pop()` — pas de `list.remove` (O(n))
- [ ] Le dict est mis a jour pour l'element swappe (le bug classique du probleme)
- [ ] Le edge case "supprimer le dernier element" marche (swap avec soi-meme)
- [ ] Les trois operations sont O(1) moyen et tu sais le justifier
- [ ] Tous les tests passent, y compris le test d'uniformite
