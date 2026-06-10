# Exercices Hard — Stacks & Queues

---

## Exercice 7 : Monotonic Stack — Largest Rectangle in Histogram

### Objectif

Le boss final du monotonic stack : trouver le plus grand rectangle dans un histogramme en O(n), la ou l'approche naive est O(n^2). C'est LE probleme hard de stack le plus pose en entretien.

### Consigne

Etant donne un tableau `heights` representant un histogramme (chaque barre a une largeur de 1), retourne l'aire du **plus grand rectangle** inscriptible dans l'histogramme.

**Contrainte de complexite : O(n) temps.** Une solution O(n^2) ne valide pas l'exercice.

```python
def largest_rectangle_area(heights: list[int]) -> int:
    """
    Return the area of the largest rectangle in the histogram.
    Must run in O(n) time.
    """
    pass
```

**Etapes** :
1. Ecris d'abord la brute force O(n^2) (pour chaque barre, etendre a gauche et a droite tant que les barres sont >= a elle). Garde-la comme oracle de test.
2. Ecris la version O(n) avec un monotonic stack **croissant** d'indices : quand une barre plus basse arrive, chaque barre poppee connait sa frontiere droite (l'indice courant) et sa frontiere gauche (le nouveau sommet de stack).
3. Benchmark : compare les deux versions sur n = 1000, 2000, 4000, 8000 et verifie que la version stack scale lineairement (temps ~x2 quand n double) alors que la brute force scale en ~x4.

**Pieges** : la barre sentinelle (ajouter un 0 a la fin ou vider la stack apres la boucle), le calcul de la largeur quand la stack devient vide (`width = i`, pas `i - stack[-1] - 1`).

### Tests

```python
assert largest_rectangle_area([2, 1, 5, 6, 2, 3]) == 10     # Rectangle on bars 5,6 (height 5, width 2)
assert largest_rectangle_area([2, 4]) == 4
assert largest_rectangle_area([1]) == 1
assert largest_rectangle_area([]) == 0
assert largest_rectangle_area([0, 0, 0]) == 0
assert largest_rectangle_area([5, 5, 5, 5]) == 20            # Full width
assert largest_rectangle_area([1, 2, 3, 4, 5]) == 9          # Increasing: 3*3 = 9
assert largest_rectangle_area([5, 4, 3, 2, 1]) == 9          # Decreasing: symmetric
assert largest_rectangle_area([2, 1, 2]) == 3                # Valley: height 1 across width 3

# Cross-check against the brute force oracle on random inputs
import random
for _ in range(50):
    arr = [random.randint(0, 20) for _ in range(random.randint(0, 30))]
    assert largest_rectangle_area(arr) == largest_rectangle_brute(arr)
```

### Criteres de reussite

- [ ] Brute force O(n^2) ecrite et utilisee comme oracle sur des inputs aleatoires
- [ ] Version monotonic stack en O(n) : chaque indice est push/pop au plus une fois
- [ ] Le cas "stack vide apres pop" donne la bonne largeur (le rectangle s'etend jusqu'au bord gauche)
- [ ] La fin de tableau est geree (sentinelle 0 ou drain de la stack)
- [ ] Benchmark : la version stack montre un scaling lineaire, la brute force quadratique
- [ ] Tous les tests passent, y compris tableau vide et hauteurs nulles

---

## Exercice 8 : Multi-source BFS — Rotting Oranges

### Objectif

Generaliser le BFS a **plusieurs sources simultanees** et compter les "vagues" (niveaux) — le pattern exact des problemes de propagation (feu, infection, distance a la sortie la plus proche).

### Consigne

Une grille contient : `0` = case vide, `1` = orange fraiche, `2` = orange pourrie.

Chaque minute, toute orange fraiche **adjacente** (4 directions) a une orange pourrie devient pourrie a son tour.

Retourne le **nombre minimum de minutes** pour que toutes les oranges soient pourries. Si c'est impossible, retourne `-1`.

**Contraintes** :
- O(rows * cols) temps et espace — une seule passe BFS, pas de re-scan de la grille a chaque minute.
- Le BFS doit demarrer avec **toutes les oranges pourries en meme temps** dans la queue (multi-source), pas une par une.

```python
def oranges_rotting(grid: list[list[int]]) -> int:
    """
    Return the minimum number of minutes until no fresh orange remains,
    or -1 if some orange can never rot.
    """
    pass
```

**Pieges** :
- Une grille sans orange fraiche du tout doit retourner `0` (pas `-1`, pas `1`).
- Le compteur de minutes ne doit pas etre incremente pour la "vague" initiale.
- Compter les fraiches au depart et decrementer : a la fin, `fresh > 0` → `-1`.

### Tests

```python
assert oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
assert oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1   # Bottom-left orange unreachable
assert oranges_rotting([[0, 2]]) == 0                              # No fresh orange: 0 minutes
assert oranges_rotting([[0]]) == 0
assert oranges_rotting([[1]]) == -1                                # Fresh orange, no source
assert oranges_rotting([[2]]) == 0
assert oranges_rotting([[2, 2], [1, 1], [0, 0], [2, 0]]) == 1     # Two sources rot in parallel
```

### Criteres de reussite

- [ ] Toutes les sources (oranges pourries initiales) sont enqueues AVANT de demarrer le BFS
- [ ] Le BFS traite la queue **niveau par niveau** (taille de queue figee a chaque minute) ou stocke `(r, c, minute)` dans la queue
- [ ] Une seule passe : O(rows * cols) temps, pas de re-scan de la grille a chaque minute
- [ ] Les compteurs sont justes : grille sans fraiche → 0, fraiche inaccessible → -1
- [ ] Marquage visited au moment de l'enqueue (mutation `1 → 2` ou set)
- [ ] Tous les tests passent
