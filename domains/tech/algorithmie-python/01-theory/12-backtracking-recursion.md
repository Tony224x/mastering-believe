# Jour 12 — Backtracking & Recursion : Permutations, Subsets, N-Queens

> **Temps estime** : 60-75 min de lecture active | **Objectif** : maitriser le template universel du backtracking et generer permutations, combinations, subsets, puis resoudre N-queens, sudoku, word search

---

## 1. Qu'est-ce que le backtracking ?

**Backtracking = DFS sur l'arbre des choix, avec annulation (undo) a chaque etape.**

L'idee : explorer toutes les solutions possibles en construisant une solution pas a pas. A chaque etape, on fait un **choix**, on **recurse**, puis on **defait** le choix pour explorer une autre branche.

```
Exemple : generer toutes les permutations de [1, 2, 3]

                        []
                /       |       \
              [1]     [2]     [3]
             /   \   /   \   /   \
          [1,2][1,3][2,1][2,3][3,1][3,2]
            |    |   |    |   |     |
         [1,2,3]...                etc.
```

**La structure universelle** : un arbre d'exploration. Chaque noeud est un etat partiel, chaque enfant est un choix possible.

---

## 2. Le template universel

```python
def backtrack(state, choices):
    # Cas de base : solution complete trouvee
    if is_complete(state):
        results.append(state[:])   # Copie !
        return

    # Iterer sur tous les choix possibles
    for choice in choices:
        if is_valid(state, choice):
            # 1. Choisir
            state.append(choice)
            # 2. Recurser avec le nouvel etat
            backtrack(state, next_choices(choices, choice))
            # 3. Defaire le choix (backtrack)
            state.pop()
```

**Les 3 phases** :
1. **Choose** : ajouter le choix courant a la solution en construction
2. **Explore** : recursion sur les sous-problemes restants
3. **Unchoose** : retirer le choix pour explorer d'autres branches

> **Cle** : le `state[:]` (ou `list(state)`) est CRUCIAL. Si tu appends `state` directement, tous tes resultats pointeront vers le meme objet et seront vides a la fin (car on les `pop()` apres).

---

## 3. Pattern 1 — Subsets (ensembles)

**Probleme** : generer tous les sous-ensembles de `nums` (le powerset).

Il y a `2^n` sous-ensembles pour un tableau de taille n, donc la complexite est au moins O(2^n).

```python
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path[:])     # Chaque etat PARTIEL est une solution !
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path) # i + 1 pour eviter les duplicates
            path.pop()
    backtrack(0, [])
    return result
# Time: O(2^n * n), Space: O(n) recursion
```

**Cle** : `start` garantit qu'on ne reutilise pas un element deja ajoute, et qu'on n'obtient pas deux fois le meme sous-ensemble sous des ordres differents (`[1,2]` et `[2,1]`).

### Iterative (pour reference)

```python
def subsets_iter(nums):
    result = [[]]
    for num in nums:
        result += [curr + [num] for curr in result]
    return result
```

---

## 4. Pattern 2 — Permutations

**Probleme** : generer toutes les `n!` permutations.

```python
def permutations(nums):
    result = []
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False] * len(nums))
    return result
# Time: O(n! * n), Space: O(n)
```

### Variante — Permutations II (avec duplicates)

```python
def permute_unique(nums):
    result = []
    nums.sort()                        # Trier pour regrouper les duplicates
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            # Skip duplicates : si nums[i] == nums[i-1] et i-1 pas utilise,
            # on a deja genere cette branche
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False] * len(nums))
    return result
```

---

## 5. Pattern 3 — Combinations

**Probleme** : generer toutes les combinaisons de taille k parmi [1, n].

```python
def combinations(n, k):
    result = []
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        # Pruning : si meme en prenant tous les elements restants
        # on n'atteint pas k, on arrete
        for i in range(start, n + 1 - (k - len(path)) + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    backtrack(1, [])
    return result
# Time: O(C(n, k) * k)
```

### Combination Sum (repetitions autorisees)

```python
def combination_sum(candidates, target):
    result = []
    candidates.sort()
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break                  # Pruning : tout le reste est plus grand
            path.append(candidates[i])
            # On passe `i` (pas `i+1`) pour autoriser la reutilisation
            backtrack(i, path, remaining - candidates[i])
            path.pop()
    backtrack(0, [], target)
    return result
```

---

## 6. Pattern 4 — N-Queens

**Probleme** : placer N reines sur un echiquier N x N sans qu'elles s'attaquent.

```python
def solve_n_queens(n):
    result = []
    cols = set()                       # Colonnes deja occupees
    diag1 = set()                      # Diagonales "\": row - col
    diag2 = set()                      # Diagonales "/": row + col
    board = [["."] * n for _ in range(n)]

    def backtrack(row):
        if row == n:
            result.append(["".join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            # Choose
            board[row][col] = "Q"
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            # Explore
            backtrack(row + 1)
            # Unchoose
            board[row][col] = "."
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result
# Time: O(N!) — super fort elague grace aux sets
```

**Cle des diagonales** : toutes les cellules sur la meme diagonale "\" ont la meme valeur `row - col`. Toutes les cellules sur la meme diagonale "/" ont la meme valeur `row + col`.

---

## 7. Pattern 5 — Word Search dans une grille

**Probleme** : verifier si un mot existe dans une grille en se deplacant horizontalement/verticalement (pas de reutilisation de cellule).

```python
def word_search(board, word):
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        if (r < 0 or r >= rows or c < 0 or c >= cols or
                board[r][c] != word[idx]):
            return False
        # Choose : marquer la cellule comme utilisee
        tmp = board[r][c]
        board[r][c] = "#"
        # Explore
        found = (backtrack(r + 1, c, idx + 1) or
                 backtrack(r - 1, c, idx + 1) or
                 backtrack(r, c + 1, idx + 1) or
                 backtrack(r, c - 1, idx + 1))
        # Unchoose : restaurer la cellule
        board[r][c] = tmp
        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False
# Time: O(R * C * 4^L) ou L = len(word), Space: O(L) recursion
```

**Astuce** : marquer la cellule `"#"` a la place d'un set visited. Restaure apres la recursion.

---

## 8. Pattern 6 — Sudoku Solver

```python
def solve_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    # Remplir les sets avec les valeurs deja presentes
    for r in range(9):
        for c in range(9):
            if board[r][c] != ".":
                val = board[r][c]
                rows[r].add(val)
                cols[c].add(val)
                boxes[(r // 3) * 3 + c // 3].add(val)

    def backtrack(idx):
        if idx == 81:
            return True
        r, c = divmod(idx, 9)
        if board[r][c] != ".":
            return backtrack(idx + 1)
        box = (r // 3) * 3 + c // 3
        for digit in "123456789":
            if digit not in rows[r] and digit not in cols[c] and digit not in boxes[box]:
                # Choose
                board[r][c] = digit
                rows[r].add(digit); cols[c].add(digit); boxes[box].add(digit)
                # Explore
                if backtrack(idx + 1):
                    return True
                # Unchoose
                board[r][c] = "."
                rows[r].remove(digit); cols[c].remove(digit); boxes[box].remove(digit)
        return False

    backtrack(0)
```

---

## 9. Pattern 7 — Generate Parentheses

**Probleme** : generer toutes les combinaisons valides de `n` paires de parentheses.

```python
def generate_parentheses(n):
    result = []
    def backtrack(path, open_count, close_count):
        if len(path) == 2 * n:
            result.append("".join(path))
            return
        if open_count < n:
            path.append("(")
            backtrack(path, open_count + 1, close_count)
            path.pop()
        if close_count < open_count:   # Ne peut pas fermer plus qu'on n'a ouvert
            path.append(")")
            backtrack(path, open_count, close_count + 1)
            path.pop()
    backtrack([], 0, 0)
    return result
# Time: O(4^n / sqrt(n)) — nombre de Catalan
```

**Cle** : deux contraintes simples (`open < n` et `close < open`) suffisent a garantir que toutes les combinaisons generees sont valides.

---

## 10. Decision Tree — Quel pattern ?

```
Le probleme demande de generer ou trouver toutes les solutions ?
|
├── SOUS-ENSEMBLES d'un tableau ?
│   └── SUBSETS (start index pour eviter les doublons)
|
├── ORDRE compte (permutations) ?
│   └── PERMUTATIONS (used[] array)
|
├── TAILLE FIXE k parmi n elements ?
│   └── COMBINATIONS (start index + pruning)
|
├── CONTRAINTES COMPLEXES sur une grille 2D ?
│   ├── Recherche d'un chemin/mot → WORD SEARCH (marquer + demarquer)
│   ├── Placement sans conflit → N-QUEENS / SUDOKU (sets pour check O(1))
|
├── GENERATION de structures combinatoires valides ?
│   └── GENERATE PARENTHESES / SEQUENCES VALIDES
|
└── Recursion simple sans annulation ?
    └── Pas vraiment du backtracking — c'est juste du DFS (voir Days 8/9)
```

**Signaux dans l'enonce** :

| Signal | Pattern |
|--------|---------|
| "all permutations" | Permutations |
| "all subsets" | Subsets / powerset |
| "all combinations of size k" | Combinations |
| "find all valid X" | Backtracking avec contraintes |
| "solve the puzzle" | Sudoku / N-queens |
| "word exists in zone" | Word search (DFS + undo) |

---

## 11. Complexites

| Probleme | Nombre de solutions | Temps |
|----------|---------------------|-------|
| Subsets | 2^n | O(2^n * n) |
| Permutations | n! | O(n! * n) |
| Combinations C(n, k) | C(n, k) | O(C(n, k) * k) |
| N-queens | depend de n | O(n!) avec pruning |
| Word search | - | O(R * C * 4^L) |
| Generate parentheses | Catalan(n) | O(4^n / sqrt(n)) |
| Sudoku | - | O(9^(cases vides)) |

**L'espace** : typiquement O(n) pour la recursion (profondeur max).

---

## 12. Pieges courants

**Piege 1 — Oublier `path[:]` au moment d'ajouter au resultat**
```python
result.append(path)      # FAUX : tous les results pointent vers le meme path
result.append(path[:])   # OK : copie
```

**Piege 2 — Oublier le "unchoose"**
Sans `path.pop()`, on accumule les choix sans jamais revenir en arriere. Les resultats seront faux.

**Piege 3 — `start` index confusion dans subsets/combinations**
`i + 1` = pas de repetition. `i` = repetition autorisee (combination sum). A bien choisir selon le probleme.

**Piege 4 — Word search sans restaurer la cellule**
Si on oublie `board[r][c] = tmp`, la cellule reste marquee et bloque d'autres branches.

**Piege 5 — Permutations avec duplicates : mauvais skip**
La condition `if i > 0 and nums[i] == nums[i-1] and not used[i-1]` est subtile. Sans le `not used[i-1]`, on skip des permutations valides.

---

## 13. Flash Cards — Revision espacee

**Q1** : Quelles sont les 3 phases du template backtracking universel ?
> **R1** : **Choose** (ajouter le choix courant a l'etat), **Explore** (recurse), **Unchoose** (defaire le choix pour tester les autres branches). Sans le Unchoose, on ne peut pas explorer d'autres solutions.

**Q2** : Pourquoi doit-on faire `result.append(path[:])` et non `result.append(path)` ?
> **R2** : `path` est modifie en place pendant toute la recursion. Si on ajoute la reference, tous les elements de `result` pointeront vers le meme objet `path`, qui sera vide (ou dans un etat arbitraire) a la fin car on aura fait tous les `pop()`. Il faut copier.

**Q3** : Comment verifier en O(1) si une reine attaque une autre dans N-Queens ?
> **R3** : Utiliser trois sets : `cols`, `diag1` (row - col), `diag2` (row + col). Toutes les cellules de la meme diagonale "\" partagent `row - col`, et celles de "/" partagent `row + col`. Lookup en O(1).

**Q4** : Dans word search, pourquoi marquer la cellule visitee avec `"#"` puis la restaurer ?
> **R4** : C'est le pattern backtracking : on "choose" la cellule (la bloque pour ne pas la reutiliser dans le meme chemin), on "explore" dans 4 directions, puis on "unchoose" (restaure la valeur originale) pour que d'autres branches puissent la reutiliser.

**Q5** : Quelle est la complexite pour generer toutes les permutations d'un tableau de taille n ?
> **R5** : O(n! * n). Il y a n! permutations, et on passe O(n) temps pour copier chaque solution dans le resultat. La recursion a une profondeur de n, donc l'espace est O(n) (hors resultat).

---

## Resume — Key Takeaways

1. **Template universel** : choose → explore → unchoose
2. **Toujours copier** l'etat avant de l'ajouter au resultat (`path[:]`)
3. **Subsets** : chaque etat PARTIEL est une solution valide (on append a chaque noeud)
4. **Permutations** : `used[]` array, on pop a la fin
5. **Combinations** : `start` index + pruning pour eviter C(n, k) explosion
6. **N-queens** : sets pour cols + diag1 + diag2 (check O(1))
7. **Word search** : marque avec `"#"` et restaure
8. **Pruning = performance** : plus on coupe tot, moins on explore

---

## Pour aller plus loin

Ressources canoniques sur le backtracking :

- **NeetCode — Backtracking roadmap** — 9 problemes phares (Subsets, Combinations, Permutations, Combination Sum, Word Search, Palindrome Partitioning, N-Queens) avec template universel "choose / explore / unchoose". https://neetcode.io/roadmap
- **The Algorithm Design Manual** (Skiena, 3rd ed 2020) — Ch 7 (Backtracking) : la presentation la plus pedagogique avec generation systematique de subsets, permutations, combinations + war story sur pruning. https://www.algorist.com/
- **Cracking the Coding Interview** (Gayle Laakmann McDowell, 6th ed) — Ch 8 (Recursion and Dynamic Programming) : 12 problemes types entretien dont Permutations, Power Set, N-Queens. https://www.crackingthecodinginterview.com/
- **CLRS — Introduction to Algorithms** (4th ed, MIT Press 2022) — Ch 34 (NP-Completeness) explique pourquoi le backtracking est souvent la meilleure option pratique pour les problemes NP-hard. https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
