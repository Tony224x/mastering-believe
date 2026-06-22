# Exercices Medium — Trees & BST

> Toutes les exercices utilisent la structure `TreeNode(val, left, right)` du module. Les solutions fournissent une fonction `build(values)` (BFS level-order avec `None` markers) pour construire les arbres de test.

---

## Exercice 4 : BFS zigzag — Binary Tree Zigzag Level Order Traversal

### Objectif

Maitriser le BFS par niveaux avec un twist : alterner le sens de lecture a chaque niveau (gauche→droite, puis droite→gauche). C'est le test classique pour verifier que tu maitrises le pattern `level_size = len(queue)` et que tu sais le moduler.

### Consigne

Etant donne la racine `root` d'un arbre binaire, retourne le parcours **zigzag** par niveaux : le niveau 0 de gauche a droite, le niveau 1 de droite a gauche, le niveau 2 de gauche a droite, etc.

```python
def zigzag_level_order(root: "TreeNode | None") -> list[list[int]]:
    """
    Return the zigzag level-order traversal: level 0 left-to-right,
    level 1 right-to-left, alternating.
    """
    pass
```

**Indice** : fais un BFS level order standard avec `level_size = len(queue)`, mais maintiens un booleen `left_to_right`. Quand il est faux, inverse la liste du niveau avant de l'ajouter (ou utilise un `deque` et `appendleft`). Ne change PAS l'ordre d'enfilement des enfants — seule la lecture du niveau s'inverse.

### Tests

```python
#         3
#        / \
#       9  20
#         /  \
#        15   7
assert zigzag_level_order(build([3, 9, 20, None, None, 15, 7])) == [[3], [20, 9], [15, 7]]
assert zigzag_level_order(build([1])) == [[1]]
assert zigzag_level_order(build([])) == []
assert zigzag_level_order(build([1, 2, 3, 4, 5, 6, 7])) == [[1], [3, 2], [4, 5, 6, 7]]
```

### Criteres de reussite

- [ ] BFS avec `level_size = len(queue)` (snapshot avant d'ajouter les enfants)
- [ ] Alternance correcte du sens a chaque niveau
- [ ] L'ordre d'enfilement des enfants reste gauche→droite (seule la lecture s'inverse)
- [ ] Gere l'arbre vide et l'arbre a un seul noeud
- [ ] Complexite O(n) temps, O(n) espace

---

## Exercice 5 : BST kth smallest — Kth Smallest Element in a BST

### Objectif

Exploiter la propriete maitresse du BST : **l'inorder donne les valeurs triees**. Trouver le k-ieme plus petit revient a s'arreter au k-ieme element de l'inorder — sans materialiser tout le tableau.

### Consigne

Etant donne la racine `root` d'un BST et un entier `k` (1-indexed), retourne la **k-ieme plus petite valeur** de l'arbre. Tu peux supposer `1 <= k <= n` (n = nombre de noeuds).

Vise une solution qui s'arrete des qu'elle a trouve le k-ieme element (inorder iteratif avec stack), pas un inorder complet suivi d'une indexation.

```python
def kth_smallest(root: "TreeNode", k: int) -> int:
    """
    Return the k-th smallest value in the BST (1-indexed).
    Stop early — do not materialize the full inorder.
    """
    pass
```

**Indice** : utilise l'inorder **iteratif** vu dans le cours (descendre a gauche a fond, pop, visiter, pivoter a droite). Decremente un compteur a chaque pop ; quand il atteint 0, tu es sur le k-ieme.

### Tests

```python
#     3
#    / \
#   1   4
#    \
#     2
assert kth_smallest(build([3, 1, 4, None, 2]), 1) == 1
assert kth_smallest(build([3, 1, 4, None, 2]), 2) == 2
#         5
#        / \
#       3   6
#      / \
#     2   4
#    /
#   1
assert kth_smallest(build([5, 3, 6, 2, 4, None, None, 1]), 3) == 3
assert kth_smallest(build([1]), 1) == 1
assert kth_smallest(build([2, 1, 3]), 3) == 3
```

### Criteres de reussite

- [ ] Utilise l'inorder (recursif ou iteratif) — l'inorder d'un BST est trie
- [ ] S'arrete au k-ieme element (early exit), pas d'inorder complet inutile
- [ ] Complexite O(h + k) temps avec l'inorder iteratif, O(h) espace
- [ ] Gere k=1, k=n, arbre a un seul noeud
- [ ] Tous les tests passent

---

## Exercice 6 : Path Sum II — Tous les chemins racine→feuille de somme donnee

### Objectif

Combiner DFS avec un accumulateur de chemin ET du backtracking (undo). C'est le pont entre le module Trees et le module Backtracking : on construit un chemin en descendant, on l'enregistre aux feuilles, on le defait en remontant.

### Consigne

Etant donne la racine `root` d'un arbre binaire et un entier `target_sum`, retourne **tous** les chemins racine→feuille dont la somme des valeurs egale `target_sum`. Chaque chemin est la liste des valeurs des noeuds, de la racine a la feuille.

```python
def path_sum(root: "TreeNode | None", target_sum: int) -> list[list[int]]:
    """
    Return all root-to-leaf paths whose node values sum to target_sum.
    """
    pass
```

**Indice** : DFS avec un `path` mutable. A chaque noeud : `path.append(node.val)`. Si c'est une **feuille** (pas d'enfant) et que la somme courante == target, copie `path[:]` dans le resultat. Recurse a gauche/droite, puis `path.pop()` en remontant (backtracking). Attention : les valeurs peuvent etre negatives — n'elague pas sur le signe.

### Tests

```python
#         5
#        / \
#       4   8
#      /   / \
#     11  13  4
#    / \      / \
#   7   2    5   1
tree = build([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1])
assert path_sum(tree, 22) == [[5, 4, 11, 2], [5, 8, 4, 5]]
assert path_sum(build([1, 2, 3]), 5) == []
assert path_sum(build([]), 0) == []
assert path_sum(build([1, 2]), 0) == []          # No root-to-leaf path sums to 0 here
assert path_sum(build([-2, None, -3]), -5) == [[-2, -3]]   # Negative values
```

### Criteres de reussite

- [ ] DFS avec accumulateur de chemin (`path` mutable)
- [ ] On enregistre UNIQUEMENT aux feuilles (left == right == None)
- [ ] Backtracking : `path.pop()` en remontant, copie `path[:]` au moment d'enregistrer
- [ ] Gere les valeurs negatives (pas d'elagage sur le signe)
- [ ] Gere l'arbre vide
- [ ] Complexite O(n) noeuds visites (O(n^2) au pire pour les copies de chemins)
