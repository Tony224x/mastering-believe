# Exercices Medium — Backtracking & Recursion

> Template universel : **choose → explore → unchoose**. Toujours copier l'etat (`path[:]`) au moment de l'enregistrer.

---

## Exercice 4 : Combination Sum — Reutilisation autorisee

### Objectif

Maitriser le backtracking avec **reutilisation** d'un meme element (on passe `i` et non `i+1` a la recursion) et le **pruning** par tri. C'est LE probleme qui differencie "subsets" (start = i+1) de "combination sum" (start = i).

### Consigne

Etant donne une liste d'entiers distincts `candidates` et un entier `target`, retourne **toutes** les combinaisons uniques de candidats dont la somme egale `target`. Le **meme** candidat peut etre utilise un nombre illimite de fois. Deux combinaisons sont differentes si la multiplicite d'au moins un nombre differe (l'ordre n'importe pas).

```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    """
    Return all unique combinations summing to target. Each candidate may be
    reused unlimited times. Order within and across combinations does not matter.
    """
    pass
```

**Indice** : trie `candidates`. Backtrack avec `(start, path, remaining)`. Pour chaque `i >= start`, si `candidates[i] > remaining` : `break` (pruning, le reste est plus grand car trie). Sinon, ajoute, recurse avec `i` (PAS `i+1`, pour autoriser la reutilisation), puis `pop`.

### Tests

```python
def sort_combos(combos):
    return sorted(sorted(c) for c in combos)

assert sort_combos(combination_sum([2, 3, 6, 7], 7)) == sort_combos([[2, 2, 3], [7]])
assert sort_combos(combination_sum([2, 3, 5], 8)) == sort_combos([[2, 2, 2, 2], [2, 3, 3], [3, 5]])
assert combination_sum([2], 1) == []                  # No combination
assert sort_combos(combination_sum([1], 2)) == [[1, 1]]
assert combination_sum([3, 5], 1) == []
assert sort_combos(combination_sum([2, 4], 6)) == sort_combos([[2, 2, 2], [2, 4]])
```

### Criteres de reussite

- [ ] Template choose → explore → unchoose avec copie `path[:]`
- [ ] Recursion avec `i` (pas `i+1`) pour autoriser la reutilisation
- [ ] Pruning par tri : `break` des que `candidates[i] > remaining`
- [ ] Gere "aucune combinaison" → `[]`, un seul candidat reutilise
- [ ] Pas de doublons dans la sortie

---

## Exercice 5 : Subsets II — Powerset avec doublons

### Objectif

Generer le powerset SANS doublons quand l'entree contient des elements repetes. La subtilite : trier puis **skipper les duplicates au meme niveau** (`if i > start and nums[i] == nums[i-1]: continue`).

### Consigne

Etant donne une liste `nums` pouvant contenir des doublons, retourne tous les sous-ensembles **uniques** (le powerset). La solution ne doit contenir aucun sous-ensemble en double.

```python
def subsets_with_dup(nums: list[int]) -> list[list[int]]:
    """
    Return all unique subsets of nums (which may contain duplicates).
    """
    pass
```

**Indice** : trie `nums`. Backtrack avec `(start, path)`, en enregistrant `path[:]` a CHAQUE noeud (chaque etat partiel est un sous-ensemble valide). Pour eviter les doublons : a un niveau donne, si `i > start` et `nums[i] == nums[i-1]`, skip — cette valeur a deja ete exploree a ce niveau.

### Tests

```python
def sort_subsets(subs):
    return sorted(sorted(s) for s in subs)

assert sort_subsets(subsets_with_dup([1, 2, 2])) == sort_subsets(
    [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]])
assert sort_subsets(subsets_with_dup([0])) == [[], [0]]
assert sort_subsets(subsets_with_dup([])) == [[]]
assert len(subsets_with_dup([1, 1, 1])) == 4          # [], [1], [1,1], [1,1,1]
assert sort_subsets(subsets_with_dup([4, 4, 4, 1, 4])) == sort_subsets(
    [[], [1], [1, 4], [1, 4, 4], [1, 4, 4, 4], [1, 4, 4, 4, 4],
     [4], [4, 4], [4, 4, 4], [4, 4, 4, 4]])
```

### Criteres de reussite

- [ ] Trie l'entree pour regrouper les doublons
- [ ] Enregistre chaque etat partiel (`path[:]`) — pas seulement les feuilles
- [ ] Skip des duplicates au meme niveau (`i > start and nums[i] == nums[i-1]`)
- [ ] Gere l'entree vide → `[[]]`, tous identiques
- [ ] Aucun sous-ensemble en double dans la sortie

---

## Exercice 6 : Generate Parentheses — Generation contrainte

### Objectif

Generer toutes les sequences valides de `n` paires de parentheses via deux contraintes simples (`open < n`, `close < open`). C'est l'exemple canonique ou le pruning rend la generation directe et evite de filtrer a posteriori.

### Consigne

Etant donne `n`, retourne toutes les combinaisons **bien formees** de `n` paires de parentheses.

```python
def generate_parentheses(n: int) -> list[str]:
    """
    Return all well-formed strings of n pairs of parentheses.
    """
    pass
```

**Indice** : backtrack en suivant `(path, open_count, close_count)`. Tu peux ajouter `(` tant que `open_count < n`. Tu peux ajouter `)` tant que `close_count < open_count` (on ne ferme jamais plus qu'on n'a ouvert). Quand `len(path) == 2 * n`, la sequence est complete et forcement valide.

### Tests

```python
assert sorted(generate_parentheses(1)) == ["()"]
assert sorted(generate_parentheses(2)) == ["(())", "()()"]
assert sorted(generate_parentheses(3)) == sorted(
    ["((()))", "(()())", "(())()", "()(())", "()()()"])
assert generate_parentheses(0) == [""]
# Count = Catalan number: C(0)=1, C(1)=1, C(2)=2, C(3)=5, C(4)=14
assert len(generate_parentheses(4)) == 14
```

### Criteres de reussite

- [ ] Deux contraintes : `open_count < n` pour `(`, `close_count < open_count` pour `)`
- [ ] Aucune sequence invalide generee (pas de filtrage a posteriori)
- [ ] Gere `n = 0` → `[""]`
- [ ] Le nombre de resultats suit la suite de Catalan
- [ ] Complexite O(4^n / sqrt(n))
