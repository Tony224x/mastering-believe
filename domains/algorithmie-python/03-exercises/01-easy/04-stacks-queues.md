# Exercices Easy — Stacks & Queues

---

## Exercice 1 : Stack — Valid Parentheses

### Objectif

Maitriser la stack comme outil de **matching de paires**. C'est le probleme d'entretien le plus classique avec une stack.

### Consigne

Etant donne une chaine `s` contenant uniquement les caracteres `()[]{}`, determine si la chaine est valide.

Une chaine est valide si :
1. Les parentheses ouvrantes sont fermees par le meme type de parenthese.
2. Les parentheses ouvrantes sont fermees dans le bon ordre.
3. Chaque parenthese fermante a une parenthese ouvrante correspondante.

```python
def is_valid(s: str) -> bool:
    """
    Return True if s is a well-formed parentheses string.
    """
    pass
```

### Tests

```python
assert is_valid("()") == True
assert is_valid("()[]{}") == True
assert is_valid("(]") == False
assert is_valid("([)]") == False
assert is_valid("{[]}") == True
assert is_valid("") == True
assert is_valid("(((") == False
assert is_valid(")(") == False
assert is_valid("((()))") == True
```

### Criteres de reussite

- [ ] Utilise une stack (list Python) pour tracker les openers non matches
- [ ] Short-circuit : retourne `False` des qu'un mismatch est detecte
- [ ] Gere le edge case stack vide au moment d'un closer
- [ ] Verifie que la stack est vide a la fin (sinon openers orphelins)
- [ ] Complexite O(n) temps, O(n) espace

---

## Exercice 2 : Queue — Number of Islands (BFS)

### Objectif

Utiliser une queue (deque) pour faire du BFS sur une grille. C'est le pattern fondamental pour tous les problemes de "connected components" ou "shortest path" sur grille.

### Consigne

Etant donne une grille 2D de `"1"` (terre) et `"0"` (eau), retourne le nombre d'iles. Une ile est un groupe de `"1"` connectes horizontalement ou verticalement (pas en diagonale), entoure d'eau ou des bords de la grille.

Tu dois modifier la grille en place (ou utiliser un set `visited`) pour ne pas recompter la meme ile.

```python
def num_islands(grid: list[list[str]]) -> int:
    """
    Return the number of distinct islands in the grid.
    """
    pass
```

### Tests

```python
grid1 = [
    ["1","1","1","1","0"],
    ["1","1","0","1","0"],
    ["1","1","0","0","0"],
    ["0","0","0","0","0"],
]
assert num_islands([row[:] for row in grid1]) == 1

grid2 = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"],
]
assert num_islands([row[:] for row in grid2]) == 3

assert num_islands([["0"]]) == 0
assert num_islands([["1"]]) == 1
assert num_islands([]) == 0
```

### Criteres de reussite

- [ ] Utilise `collections.deque` pour le BFS (pas `list.pop(0)`)
- [ ] Marque visited au moment de l'enqueue (pas du dequeue)
- [ ] Parcourt les 4 voisins (haut, bas, gauche, droite) — PAS les diagonales
- [ ] Complexite O(rows * cols) temps et espace
- [ ] Gere le edge case grille vide

---

## Exercice 3 : Monotonic Stack — Next Greater Element I

### Objectif

Decouvrir le monotonic stack sur un probleme plus facile que "daily temperatures". Le but : sentir l'invariant de la stack decroissante.

### Consigne

On te donne deux tableaux `nums1` et `nums2` ou `nums1` est un **sous-ensemble** de `nums2`.

Pour chaque element `nums1[i]`, trouve son **next greater element** dans `nums2` : le premier element a droite de `nums1[i]` dans `nums2` qui est strictement plus grand. Si un tel element n'existe pas, retourne `-1` pour cette position.

Le resultat est un tableau de la meme taille que `nums1`.

```python
def next_greater_element(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    For each element of nums1, find its next greater element in nums2.
    Return -1 when none exists.
    """
    pass
```

### Tests

```python
assert next_greater_element([4, 1, 2], [1, 3, 4, 2]) == [-1, 3, -1]
assert next_greater_element([2, 4], [1, 2, 3, 4]) == [3, -1]
assert next_greater_element([1], [1]) == [-1]
assert next_greater_element([1, 3, 5, 2, 4], [6, 5, 4, 3, 2, 1, 7]) == [7, 7, 7, 7, 7]
assert next_greater_element([5], [4, 5, 6]) == [6]
```

### Criteres de reussite

- [ ] Utilise un monotonic stack (decroissant) sur `nums2`
- [ ] Construit un dict `element → next_greater` pour lookup O(1) pendant la seconde passe sur `nums1`
- [ ] Complexite O(n + m) temps ou n = len(nums1), m = len(nums2)
- [ ] Comprend pourquoi la stack contient des VALEURS (pas des indices) ici — car on n'a pas besoin des distances
