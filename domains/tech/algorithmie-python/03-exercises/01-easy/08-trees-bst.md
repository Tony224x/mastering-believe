# Exercices Easy — Trees & BST

---

## Exercice 1 : DFS — Max Depth of Binary Tree

### Objectif

Maitriser la recursion de base sur un arbre : retourner la profondeur maximale en combinant les resultats des sous-arbres gauche et droit.

### Consigne

Etant donne la racine d'un arbre binaire, retourne sa profondeur maximale (le nombre de noeuds sur le chemin le plus long entre la racine et une feuille).

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth(root: "TreeNode | None") -> int:
    """
    Return the maximum depth of a binary tree.
    Depth of an empty tree is 0. Depth of a single-node tree is 1.
    """
    pass
```

### Tests

```python
assert max_depth(None) == 0
assert max_depth(TreeNode(1)) == 1
# [3, 9, 20, null, null, 15, 7]
tree = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
assert max_depth(tree) == 3
# Left-only chain
chain = TreeNode(1, TreeNode(2, TreeNode(3, TreeNode(4))))
assert max_depth(chain) == 4
```

### Criteres de reussite

- [ ] Solution recursive en 2-3 lignes
- [ ] Cas de base explicite : `if not root: return 0`
- [ ] Complexite O(n) temps, O(h) espace (h = hauteur)
- [ ] Tous les tests passent

---

## Exercice 2 : BFS — Level Order Traversal

### Objectif

Maitriser le pattern BFS par niveaux avec `deque` et `level_size`, qui est la base de tous les problemes "par niveau".

### Consigne

Etant donne la racine d'un arbre binaire, retourne la liste des valeurs groupees par niveau (du haut vers le bas).

```python
from collections import deque

def level_order(root: "TreeNode | None") -> list[list[int]]:
    """
    Return node values grouped by level, top to bottom, left to right.
    """
    pass
```

### Tests

```python
assert level_order(None) == []
assert level_order(TreeNode(1)) == [[1]]
# [3, 9, 20, null, null, 15, 7]
tree = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
assert level_order(tree) == [[3], [9, 20], [15, 7]]
# Left-skewed
chain = TreeNode(1, TreeNode(2, TreeNode(3)))
assert level_order(chain) == [[1], [2], [3]]
```

### Criteres de reussite

- [ ] Utilise `collections.deque` (pas une list avec pop(0))
- [ ] Capture `level_size = len(queue)` avant la boucle interne
- [ ] Complexite O(n) temps, O(w) espace (w = largeur max)
- [ ] Tous les tests passent, y compris l'arbre vide

---

## Exercice 3 : BST — Validate Binary Search Tree

### Objectif

Comprendre pourquoi une validation locale (`left.val < node.val`) est insuffisante et maitriser la validation avec bornes propagees.

### Consigne

Etant donne la racine d'un arbre binaire, retourne `True` si c'est un BST valide. Un BST valide respecte :
- Toutes les valeurs du sous-arbre gauche sont STRICTEMENT inferieures a la valeur du noeud
- Toutes les valeurs du sous-arbre droit sont STRICTEMENT superieures
- Les deux sous-arbres sont eux-memes des BST valides

```python
def is_valid_bst(root: "TreeNode | None") -> bool:
    """
    Return True if the tree is a valid Binary Search Tree.
    """
    pass
```

### Tests

```python
assert is_valid_bst(None) == True
assert is_valid_bst(TreeNode(1)) == True
# Valid BST: [2, 1, 3]
assert is_valid_bst(TreeNode(2, TreeNode(1), TreeNode(3))) == True
# Invalid: [5, 1, 4, null, null, 3, 6] — 3 is in right subtree of 5 but < 5
bad = TreeNode(5, TreeNode(1), TreeNode(4, TreeNode(3), TreeNode(6)))
assert is_valid_bst(bad) == False
# Edge case: duplicate value is NOT allowed (strict inequality)
assert is_valid_bst(TreeNode(1, TreeNode(1))) == False
```

### Criteres de reussite

- [ ] Propage des bornes min/max, PAS juste une comparaison locale
- [ ] Gere le cas d'inegalite stricte (pas de doublons)
- [ ] Complexite O(n) temps, O(h) espace
- [ ] Comprend pourquoi l'approche naive echoue sur le test `bad`
