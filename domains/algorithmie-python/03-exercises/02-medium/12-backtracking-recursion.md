# Exercices Medium — Backtracking & Recursion

---

## Exercice 4 : Start Index — Combination Sum

### Objectif

Maitriser le parametre `start` : c'est lui qui evite les doublons (`[2,3]` et `[3,2]`) tout en autorisant la **reutilisation** du meme element. La nuance `i` vs `i+1` dans l'appel recursif est exactement ce que teste ce probleme.

### Consigne

Etant donne un tableau `candidates` d'entiers **distincts** et une cible `target`, retourne toutes les combinaisons uniques dont la somme vaut `target`. Le meme nombre peut etre choisi un nombre **illimite** de fois.

```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    """
    All unique combinations summing to target; each candidate reusable.
    """
    pass
```

**Indice** : dans la recursion, repars de `i` (pas `i + 1`) pour autoriser la reutilisation — mais jamais de `i - 1` ou 0, sinon les permutations d'une meme combinaison apparaissent.

### Tests

```python
def normalize(result):
    return sorted([sorted(c) for c in result])

assert normalize(combination_sum([2, 3, 6, 7], 7)) == [[2, 2, 3], [7]]
assert normalize(combination_sum([2, 3, 5], 8)) == [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
assert combination_sum([2], 1) == []
assert normalize(combination_sum([1], 2)) == [[1, 1]]
assert combination_sum([3, 5], 2) == []
assert normalize(combination_sum([7], 7)) == [[7]]
```

### Criteres de reussite

- [ ] Le template universel est respecte : choisir → recurser → annuler (`pop`)
- [ ] Reutilisation via `i` dans l'appel recursif, anti-doublons via `start`
- [ ] Pruning : arret de la branche des que `remaining < 0` (ou tri + `break`)
- [ ] La copie `path[:]` est faite au moment d'ajouter au resultat (piege de la reference partagee)
- [ ] Tous les tests passent

---

## Exercice 5 : Pruning de doublons — Permutations II

### Objectif

La regle de skip la plus mal comprise du backtracking : `if nums[i] == nums[i-1] and not used[i-1]: continue`. L'exercice force a l'appliquer ET a l'expliquer.

### Consigne

Etant donne un tableau `nums` qui peut contenir des **doublons**, retourne toutes les permutations **uniques**, sans utiliser de set de deduplication a posteriori.

**Contrainte** : la deduplication doit etre faite par **pruning pendant la recursion** (tri prealable + regle de skip). Dedupliquer a la fin avec un set sur des tuples genere n! branches inutiles et ne valide pas.

```python
def permute_unique(nums: list[int]) -> list[list[int]]:
    """
    All unique permutations. Dedup must happen DURING recursion (pruning),
    not by filtering the final result.
    """
    pass
```

**La regle a expliquer en commentaire** : apres tri, parmi des copies identiques, on ne peut choisir la copie `i` que si la copie `i-1` est deja utilisee dans le chemin courant. Cela force un ordre canonique entre copies → chaque permutation generee une seule fois.

### Tests

```python
def normalize(result):
    return sorted([tuple(p) for p in result])

assert normalize(permute_unique([1, 1, 2])) == [(1, 1, 2), (1, 2, 1), (2, 1, 1)]
assert normalize(permute_unique([1, 2, 3])) == [
    (1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)
]
assert normalize(permute_unique([1, 1])) == [(1, 1)]
assert normalize(permute_unique([1])) == [(1,)]
assert len(permute_unique([1, 1, 2, 2])) == 6        # 4! / (2! * 2!)
assert len(permute_unique([2, 2, 2])) == 1
```

### Criteres de reussite

- [ ] Tri prealable du tableau (precondition de la regle de skip)
- [ ] Regle `nums[i] == nums[i-1] and not used[i-1]` appliquee et expliquee en commentaire
- [ ] Aucun set de deduplication sur le resultat final
- [ ] Le nombre de feuilles explorees = nombre de permutations uniques (pas n!)
- [ ] Tous les tests passent, y compris `[1,1,2,2]` → 6 permutations

---

## Exercice 6 : Backtracking sur grille — Word Search

### Objectif

Le backtracking spatial : marquer une case pendant l'exploration et la **restaurer** au retour. Oublier la restauration est LE bug classique — une lettre devient inutilisable pour les chemins suivants.

### Consigne

Etant donne une grille `board` de caracteres et un mot `word`, retourne `True` si le mot peut etre construit en suivant des cases **adjacentes** (4 directions), chaque case utilisee **au plus une fois** par chemin.

```python
def exist(board: list[list[str]], word: str) -> bool:
    """
    True if word can be traced through adjacent cells, each cell used
    at most once per path.
    """
    pass
```

**Indice** : marque la case courante (`board[r][c] = "#"`) avant de recurser sur les 4 voisins, puis **restaure** la lettre apres. Pas besoin de set `visited` separe.

### Tests

```python
board = [
    ["A", "B", "C", "E"],
    ["S", "F", "C", "S"],
    ["A", "D", "E", "E"],
]
assert exist([row[:] for row in board], "ABCCED") == True
assert exist([row[:] for row in board], "SEE") == True
assert exist([row[:] for row in board], "ABCB") == False    # B reused — must fail
assert exist([row[:] for row in board], "") == True
assert exist([["A"]], "A") == True
assert exist([["A"]], "AA") == False                         # Single cell reused
assert exist([["A", "A"]], "AAA") == False

# The board must be intact after the search (restoration check)
b = [row[:] for row in board]
exist(b, "ABCCED")
assert b == board
```

### Criteres de reussite

- [ ] Marquage in-place + restauration apres la recursion (le test d'integrite de la grille passe)
- [ ] Le test `"ABCB"` passe — une case ne sert qu'une fois par chemin
- [ ] DFS lance depuis chaque case qui matche `word[0]`
- [ ] Complexite O(R * C * 3^L) — tu sais expliquer le 3 (on ne revient pas sur ses pas)
- [ ] Tous les tests passent
