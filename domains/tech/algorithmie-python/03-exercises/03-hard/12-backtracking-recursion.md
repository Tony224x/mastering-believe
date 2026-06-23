# Exercices Hard — Backtracking & Recursion

> Backtracking sur grille (marquer/restaurer) et placement sous contraintes (sets pour des checks O(1)). Pruning = performance.

---

## Exercice 7 : Word Search — Chemin dans une grille

### Objectif

Backtracking sur grille 2D avec marquage/restauration de cellule. C'est le pattern "DFS + undo" applique a la recherche d'un mot : on bloque la cellule courante, on explore les 4 voisins, on restaure en remontant.

### Consigne

Etant donne une grille `board` de caracteres et un mot `word`, retourne `True` si `word` existe dans la grille. Le mot se forme par cellules **adjacentes** (horizontalement ou verticalement), et une meme cellule ne peut **pas** etre reutilisee dans un meme chemin.

```python
def exist(board: list[list[str]], word: str) -> bool:
    """
    Return True if word can be traced through adjacent cells without reuse.
    """
    pass
```

**Indice** : pour chaque cellule de depart, lance un backtracking `(r, c, idx)`. Si `board[r][c] != word[idx]`, echec sur cette branche. Marque la cellule (`board[r][c] = "#"`), explore les 4 directions pour `idx+1`, puis **restaure** la cellule. Succes quand `idx == len(word)`.

### Tests

```python
board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
assert exist(board, "ABCCED") is True
assert exist(board, "SEE") is True
assert exist(board, "ABCB") is False         # 'B' would need to be reused
assert exist([["A"]], "A") is True
assert exist([["A"]], "B") is False
assert exist([["A", "B"], ["C", "D"]], "ACDB") is True
assert exist([["A", "A"]], "AAA") is False   # Only 2 cells for 3 letters
```

### Criteres de reussite

- [ ] Backtracking avec marquage de cellule (`"#"`) puis **restauration**
- [ ] Verifie les bornes et le match `board[r][c] == word[idx]`
- [ ] Pas de reutilisation d'une cellule dans un meme chemin
- [ ] Gere grille 1x1, mot plus long que le nombre de cellules
- [ ] La grille est restauree a son etat initial apres l'appel
- [ ] Complexite O(R * C * 4^L), L = len(word)

---

## Exercice 8 : N-Queens — Placement sous contraintes

### Objectif

Le probleme de backtracking canonique : placer N reines sans conflit. La cle est de verifier les attaques en **O(1)** via trois sets (colonnes, diagonales `row-col`, diagonales `row+col`).

### Consigne

Retourne le **nombre** de facons distinctes de placer `n` reines sur un echiquier `n x n` de sorte qu'aucune ne puisse en attaquer une autre (pas deux reines sur la meme ligne, colonne, ou diagonale). C'est la version "Total N-Queens".

```python
def total_n_queens(n: int) -> int:
    """
    Return the number of distinct solutions to the n-queens puzzle.
    """
    pass
```

**Indice** : place une reine par ligne (donc pas de conflit de ligne par construction). Pour la colonne `col` a la ligne `row`, le placement est valide si `col not in cols`, `(row - col) not in diag1`, `(row + col) not in diag2`. Ajoute aux sets (choose), recurse `row+1` (explore), retire des sets (unchoose). Compte quand `row == n`.

### Tests

```python
assert total_n_queens(1) == 1
assert total_n_queens(2) == 0                # No solution
assert total_n_queens(3) == 0                # No solution
assert total_n_queens(4) == 2
assert total_n_queens(5) == 10
assert total_n_queens(6) == 4
assert total_n_queens(8) == 92               # The classic 8-queens count
```

### Criteres de reussite

- [ ] Une reine par ligne (la recursion descend ligne par ligne)
- [ ] Check O(1) via 3 sets : `cols`, `diag1` (row-col), `diag2` (row+col)
- [ ] choose → explore → unchoose sur les trois sets
- [ ] Gere n=1, et les n sans solution (2, 3) → 0
- [ ] Retourne 92 pour n=8
- [ ] Complexite O(n!) avec pruning

---

## Exercice 9 : Palindrome Partitioning — Partition + contrainte

### Objectif

Combiner backtracking et test de contrainte (palindrome) a chaque coupe. Probleme hard ou chaque "choix" est une sous-chaine prefixe et la contrainte filtre les branches : on ne recurse que si le prefixe est un palindrome.

### Consigne

Etant donne une string `s`, partitionne-la de sorte que **chaque** sous-chaine de la partition soit un palindrome. Retourne **toutes** les partitions possibles.

```python
def partition(s: str) -> list[list[str]]:
    """
    Return all partitions of s where every substring is a palindrome.
    """
    pass
```

**Indice** : backtrack avec `start`. Pour chaque `end` de `start+1` a `len(s)`, considere le prefixe `s[start:end]`. S'il est palindrome, ajoute-le au chemin, recurse depuis `end`, puis pop. Quand `start == len(s)`, le chemin est une partition complete : enregistre `path[:]`.

### Tests

```python
def sort_parts(parts):
    return sorted(parts)

assert sort_parts(partition("aab")) == sort_parts([["a", "a", "b"], ["aa", "b"]])
assert sort_parts(partition("a")) == [["a"]]
assert sort_parts(partition("")) == [[]]
assert sort_parts(partition("aba")) == sort_parts([["a", "b", "a"], ["aba"]])
assert sort_parts(partition("abc")) == sort_parts([["a", "b", "c"]])
assert sort_parts(partition("aaa")) == sort_parts(
    [["a", "a", "a"], ["a", "aa"], ["aa", "a"], ["aaa"]])
```

### Criteres de reussite

- [ ] Backtracking par prefixe (`start`/`end`), recursion depuis `end`
- [ ] Ne recurse que sur les prefixes qui sont des palindromes
- [ ] Enregistre `path[:]` quand `start == len(s)`
- [ ] Gere string vide → `[[]]`, string a un caractere
- [ ] Toutes les partitions valides, aucune invalide
