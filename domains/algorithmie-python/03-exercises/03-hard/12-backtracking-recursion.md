# Exercices Hard — Backtracking & Recursion

---

## Exercice 7 : Pruning par sets — N-Queens

### Objectif

Le backtracking avec **elagage O(1)** : trois sets (colonnes, deux familles de diagonales) rendent le test de conflit constant. Sans eux, chaque placement coute O(n) et la solution est disqualifiee en entretien.

### Consigne

Place `n` reines sur un echiquier `n x n` sans qu'aucune ne puisse en attaquer une autre. Retourne **toutes les solutions distinctes**, chaque solution etant une liste de strings (`"."` = vide, `"Q"` = reine).

**Contraintes** :
- Test de conflit en **O(1)** via trois sets : `cols`, `diag1` (r - c), `diag2` (r + c).
- Verifie les comptes connus : n=4 → 2 solutions, n=6 → 4, n=8 → 92.

```python
def solve_n_queens(n: int) -> list[list[str]]:
    """
    All distinct n-queens placements, each as a list of strings.
    Conflict check must be O(1) via sets.
    """
    pass
```

**Pourquoi r - c et r + c ?** Sur une diagonale "descendante", `r - c` est constant ; sur une "montante", `r + c` est constant. Deux reines partagent une diagonale ssi l'une de ces quantites coincide.

### Tests

```python
result = solve_n_queens(4)
assert len(result) == 2
expected = {
    (".Q..", "...Q", "Q...", "..Q."),
    ("..Q.", "Q...", "...Q", ".Q.."),
}
assert {tuple(sol) for sol in result} == expected

assert len(solve_n_queens(1)) == 1
assert solve_n_queens(2) == []                   # No solution exists
assert solve_n_queens(3) == []
assert len(solve_n_queens(6)) == 4
assert len(solve_n_queens(8)) == 92              # The classic count

# Every returned board must actually be valid
for sol in solve_n_queens(6):
    queens = [(r, row.index("Q")) for r, row in enumerate(sol)]
    assert len({c for _, c in queens}) == 6                      # Distinct columns
    assert len({r - c for r, c in queens}) == 6                  # Distinct diag1
    assert len({r + c for r, c in queens}) == 6                  # Distinct diag2
```

### Criteres de reussite

- [ ] Une reine par ligne (la recursion avance ligne par ligne — pas de boucle sur les lignes dans le test de conflit)
- [ ] Trois sets `cols`, `diag1`, `diag2` ajoutes/retires symetriquement autour de la recursion
- [ ] Les comptes 2 / 0 / 0 / 2→4 / 92 sont corrects (n = 4, 2, 3, 6, 8)
- [ ] La construction des strings est faite au moment de la solution (pas une grille mutee en permanence)
- [ ] n=8 se resout en bien moins d'une seconde
- [ ] Tous les tests passent

---

## Exercice 8 : Contraintes multiples — Sudoku Solver

### Objectif

Le backtracking sous triple contrainte (ligne, colonne, boite 3x3) avec un choix d'implementation qui change tout : precalculer les sets de contraintes plutot que re-scanner la grille a chaque placement.

### Consigne

Resous une grille de Sudoku 9x9 **en place**. `board[r][c]` est un caractere `"1"`-`"9"` ou `"."` pour une case vide. La grille d'entree a une solution unique.

**Contraintes d'implementation** :
- Pre-calcule 27 sets (9 lignes, 9 colonnes, 9 boites) en une passe initiale. Le test "puis-je poser d soit ici ?" doit etre O(1) — pas de re-scan de la ligne/colonne/boite.
- Index de boite : `(r // 3) * 3 + c // 3`.
- Benchmark : resous la grille "difficile" des tests en moins de 5 secondes.

```python
def solve_sudoku(board: list[list[str]]) -> None:
    """
    Solve the sudoku in place. Precomputed constraint sets required:
    each placement test must be O(1).
    """
    pass
```

**Bonus (pruning avance)** : au lieu de remplir les cases dans l'ordre de lecture, choisis a chaque etape la case vide avec le **moins de candidats** (MRV — Minimum Remaining Values). Compare le nombre d'appels recursifs avec et sans.

### Tests

```python
puzzle = [
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"],
]
solve_sudoku(puzzle)

def is_solved(board):
    full = set("123456789")
    for i in range(9):
        if {board[i][j] for j in range(9)} != full:      # Row i
            return False
        if {board[j][i] for j in range(9)} != full:      # Column i
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = {board[br + dr][bc + dc] for dr in range(3) for dc in range(3)}
            if box != full:
                return False
    return True

assert is_solved(puzzle)
assert puzzle[0] == ["5", "3", "4", "6", "7", "8", "9", "1", "2"]   # Known unique solution

# Hard puzzle (Arto Inkala's "world's hardest sudoku") — must still solve fast
hard = [
    ["8", ".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", "3", "6", ".", ".", ".", ".", "."],
    [".", "7", ".", ".", "9", ".", "2", ".", "."],
    [".", "5", ".", ".", ".", "7", ".", ".", "."],
    [".", ".", ".", ".", "4", "5", "7", ".", "."],
    [".", ".", ".", "1", ".", ".", ".", "3", "."],
    [".", ".", "1", ".", ".", ".", ".", "6", "8"],
    [".", ".", "8", "5", ".", ".", ".", "1", "."],
    [".", "9", ".", ".", ".", ".", "4", ".", "."],
]
solve_sudoku(hard)
assert is_solved(hard)
```

### Criteres de reussite

- [ ] 27 sets de contraintes pre-calcules en une seule passe initiale
- [ ] Placement/retrait symetriques dans les trois sets autour de la recursion
- [ ] Test de candidat O(1) — aucune fonction `is_valid` qui re-scanne ligne/colonne/boite
- [ ] La grille difficile (21 indices, Inkala) est resolue en moins de 5 secondes
- [ ] La fonction modifie la grille en place et la laisse entierement resolue
- [ ] Tous les tests passent (`is_solved` + premiere ligne de la solution connue)
