# Exercices Easy — Backtracking & Recursion

---

## Exercice 1 : Subsets (Powerset)

### Objectif

Maitriser le template backtracking le plus simple (subsets) et comprendre pourquoi chaque etat partiel est une solution.

### Consigne

Etant donne un tableau d'entiers distincts `nums`, retourne tous les sous-ensembles possibles (l'ensemble puissance). L'ordre des sous-ensembles et l'ordre interne n'ont pas d'importance, mais chaque sous-ensemble doit etre unique.

```python
def subsets(nums: list[int]) -> list[list[int]]:
    """
    Return all possible subsets of nums (the power set).
    """
    pass
```

### Tests

```python
def sorted_subsets(result):
    return sorted([sorted(s) for s in result])

assert sorted_subsets(subsets([])) == [[]]
assert sorted_subsets(subsets([0])) == [[], [0]]
assert sorted_subsets(subsets([1, 2, 3])) == [
    [], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]
]
# Should produce 2^n subsets
assert len(subsets([1, 2, 3, 4])) == 16
assert len(subsets([1, 2, 3, 4, 5])) == 32
```

### Criteres de reussite

- [ ] Utilise backtracking avec un `start` index
- [ ] Ajoute une copie du path (`path[:]`) au resultat, pas une reference
- [ ] Retourne exactement 2^n sous-ensembles
- [ ] Tous les tests passent

---

## Exercice 2 : Permutations

### Objectif

Maitriser le backtracking avec `used[]` pour generer les n! permutations.

### Consigne

Etant donne un tableau d'entiers distincts `nums`, retourne toutes les permutations possibles. Chaque permutation utilise chaque element exactement une fois.

```python
def permute(nums: list[int]) -> list[list[int]]:
    """
    Return all permutations of nums.
    """
    pass
```

### Tests

```python
def sorted_perms(result):
    return sorted([tuple(p) for p in result])

assert sorted_perms(permute([1])) == [(1,)]
assert sorted_perms(permute([1, 2])) == [(1, 2), (2, 1)]
assert sorted_perms(permute([1, 2, 3])) == [
    (1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)
]
assert len(permute([1, 2, 3, 4])) == 24
assert len(permute([1, 2, 3, 4, 5])) == 120
```

### Criteres de reussite

- [ ] Utilise un tableau `used[]` pour tracker les elements deja pris
- [ ] Copie le path (`path[:]`) avant de l'ajouter au resultat
- [ ] Retourne exactement n! permutations
- [ ] Tous les tests passent

---

## Exercice 3 : Generate Parentheses

### Objectif

Appliquer le backtracking avec invariants simples (open_count < n, close_count < open_count) pour generer uniquement des sequences valides.

### Consigne

Etant donne un entier `n`, retourne toutes les combinaisons distinctes de `n` paires de parentheses bien formees.

```python
def generate_parentheses(n: int) -> list[str]:
    """
    Return all combinations of n pairs of well-formed parentheses.
    """
    pass
```

### Tests

```python
assert sorted(generate_parentheses(1)) == ["()"]
assert sorted(generate_parentheses(2)) == ["(())", "()()"]
assert sorted(generate_parentheses(3)) == [
    "((()))", "(()())", "(())()", "()(())", "()()()"
]
# Count is the nth Catalan number
assert len(generate_parentheses(4)) == 14
assert len(generate_parentheses(5)) == 42
```

### Criteres de reussite

- [ ] Utilise deux compteurs (open_count, close_count) passes en parametres
- [ ] Ajoute `(` uniquement si `open_count < n`
- [ ] Ajoute `)` uniquement si `close_count < open_count`
- [ ] Produit EXACTEMENT les Catalan(n) sequences (pas de doublons, pas de tri necessaire)
- [ ] Tous les tests passent
