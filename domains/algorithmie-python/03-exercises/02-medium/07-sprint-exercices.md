# Exercices Medium — Sprint Jour 7

**But** : 3 problemes medium chronometres, dans la continuite du sprint P1-P10. Applique le processus en 6 etapes du cours (reformuler, clarifier, exemple, brute force, coder en parlant, tester). Note pour chaque probleme : pattern utilise, temps pris, bugs rencontres.

---

## Exercice 4 : Sprint 10 min — Product of Array Except Self

### Objectif

Resoudre en moins de 10 minutes un classique des entretiens qui interdit la solution evidente (la division) — teste le reflexe "prefix/suffix".

### Consigne

Etant donne un tableau d'entiers `nums`, retourne un tableau `answer` tel que `answer[i]` est le **produit de tous les elements sauf** `nums[i]`.

**Contraintes** :
- Interdiction d'utiliser la division (le cas zero la casse de toute facon).
- O(n) temps. Bonus : O(1) espace auxiliaire (le tableau resultat ne compte pas).

```python
def product_except_self(nums: list[int]) -> list[int]:
    """
    Return the products of all elements except self, without division.
    """
    pass
```

### Tests

```python
assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
assert product_except_self([2, 3]) == [3, 2]
assert product_except_self([0, 0]) == [0, 0]            # Two zeros: everything is 0
assert product_except_self([5, 0]) == [0, 5]            # One zero
assert product_except_self([1, 1, 1, 1]) == [1, 1, 1, 1]
```

### Criteres de reussite

- [ ] Resolu en moins de 10 minutes, processus en 6 etapes respecte
- [ ] Aucune division — passe prefix (produit a gauche) puis passe suffix (produit a droite)
- [ ] Les cas avec un et deux zeros passent sans traitement special
- [ ] O(n) temps ; bonus : suffix accumule dans une variable (O(1) espace auxiliaire)

---

## Exercice 5 : Sprint 12 min — Longest Consecutive Sequence

### Objectif

Detecter en moins de 12 minutes que le tri (O(n log n)) n'est PAS la solution attendue, et basculer sur un set avec demarrage intelligent.

### Consigne

Etant donne un tableau d'entiers `nums` non trie, retourne la longueur de la **plus longue suite de nombres consecutifs** (les elements n'ont pas besoin d'etre adjacents dans le tableau).

**Contrainte : O(n) temps.** La solution par tri ne valide pas le sprint.

```python
def longest_consecutive(nums: list[int]) -> int:
    """
    Return the length of the longest run of consecutive integers.
    Must run in O(n).
    """
    pass
```

**Indice** : mets tout dans un set. Ne demarre un comptage que depuis un **debut de suite** (`num - 1` absent du set) — sinon le pire cas redevient O(n^2).

### Tests

```python
assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4         # 1, 2, 3, 4
assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
assert longest_consecutive([]) == 0
assert longest_consecutive([1]) == 1
assert longest_consecutive([1, 1, 1]) == 1                       # Duplicates
assert longest_consecutive([5, 3, 1]) == 1                       # No consecutive pair
assert longest_consecutive([-2, -1, 0, 1]) == 4                  # Negatives
```

### Criteres de reussite

- [ ] Resolu en moins de 12 minutes
- [ ] Set utilise pour les lookups O(1) — pas de tri
- [ ] Le comptage ne demarre que sur les debuts de suite (`num - 1 not in seen`)
- [ ] Tu peux expliquer pourquoi c'est O(n) malgre la boucle imbriquee (chaque nombre visite au plus 2 fois)
- [ ] Tous les tests passent, y compris doublons et negatifs

---

## Exercice 6 : Sprint 12 min — Merge Intervals

### Objectif

Le pattern "trier puis balayer" sur des intervalles — present dans une question d'entretien sur cinq. A resoudre en moins de 12 minutes sans bug d'off-by-one sur les bornes.

### Consigne

Etant donne un tableau d'intervalles `intervals` ou `intervals[i] = [start_i, end_i]`, fusionne tous les intervalles qui se **chevauchent** et retourne le tableau d'intervalles disjoints couvrant exactement les memes points.

Deux intervalles `[a, b]` et `[c, d]` avec `c <= b` se chevauchent (les bornes qui se touchent comptent : `[1, 4]` et `[4, 5]` fusionnent en `[1, 5]`).

```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """
    Merge all overlapping intervals. Touching bounds count as overlapping.
    """
    pass
```

### Tests

```python
assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]            # Touching bounds merge
assert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]            # Full containment
assert merge_intervals([]) == []
assert merge_intervals([[1, 4]]) == [[1, 4]]
assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]    # Unsorted input
assert merge_intervals([[1, 4], [0, 4]]) == [[0, 4]]
assert merge_intervals([[2, 2], [2, 2], [2, 2]]) == [[2, 2]]    # Degenerate points
```

### Criteres de reussite

- [ ] Resolu en moins de 12 minutes
- [ ] Tri par borne de depart d'abord — sans le tri, l'algo est faux
- [ ] La fusion etend la fin avec `max(end, current_end)` (le cas "contenu dedans" ne retrecit pas l'intervalle)
- [ ] Bornes qui se touchent fusionnees (`<=`, pas `<`)
- [ ] O(n log n) temps (le tri domine), O(n) espace resultat
