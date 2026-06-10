# Exercices Medium — Trees & BST

---

## Exercice 4 : Recursion post-order — Lowest Common Ancestor of a Binary Tree

### Objectif

Maitriser le pattern "la reponse remonte depuis les feuilles" : le LCA dans un arbre binaire **quelconque** (pas un BST), ou aucune propriete d'ordre ne guide la descente.

### Consigne

Etant donne la racine d'un arbre binaire et deux noeuds `p` et `q` (garantis presents et distincts), retourne leur **plus proche ancetre commun** (LCA). Un noeud peut etre son propre ancetre.

```python
def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Return the lowest common ancestor of nodes p and q.
    """
    pass
```

**Indice** : la fonction retourne `root` si `root` est `p`, `q` ou `None`. Sinon, cherche dans les deux sous-arbres : si les DEUX retournent un noeud, `root` est le LCA ; sinon le LCA est le cote non-None.

**Piege** : compare les noeuds par **identite** (`is`), pas par valeur — deux noeuds peuvent porter la meme valeur.

### Tests

```python
#         3
#       /   \
#      5     1
#     / \   / \
#    6   2 0   8
#       / \
#      7   4
n7, n4 = TreeNode(7), TreeNode(4)
n6, n2 = TreeNode(6), TreeNode(2, n7, n4)
n0, n8 = TreeNode(0), TreeNode(8)
n5, n1 = TreeNode(5, n6, n2), TreeNode(1, n0, n8)
root = TreeNode(3, n5, n1)

assert lowest_common_ancestor(root, n5, n1) is root     # Different subtrees
assert lowest_common_ancestor(root, n5, n4) is n5       # p is ancestor of q
assert lowest_common_ancestor(root, n6, n4) is n5
assert lowest_common_ancestor(root, n7, n4) is n2
assert lowest_common_ancestor(root, n7, n8) is root
```

### Criteres de reussite

- [ ] Recursion post-order : la reponse est construite en remontant
- [ ] Le cas "p est ancetre de q" marche sans traitement special (on s'arrete des qu'on trouve p)
- [ ] Comparaison par identite de noeud, pas par valeur
- [ ] O(n) temps (chaque noeud visite une fois), O(h) espace de pile
- [ ] Tous les tests passent

---

## Exercice 5 : BFS variante — Binary Tree Right Side View

### Objectif

Adapter le template BFS par niveaux (exercice easy 2) a une variante : ne garder que le **dernier noeud de chaque niveau**. Teste la comprehension du template, pas sa memorisation.

### Consigne

Etant donne la racine d'un arbre binaire, retourne la liste des valeurs visibles **depuis la droite** : pour chaque niveau, la valeur du noeud le plus a droite.

```python
def right_side_view(root: TreeNode | None) -> list[int]:
    """
    Return the values visible from the right side, top to bottom.
    """
    pass
```

**Piege** : ce n'est PAS "toujours descendre a droite" — un noeud profond a gauche est visible si le cote droit est plus court (voir test 2).

### Tests

```python
# Test 1:    1            view: [1, 3, 4]
#           / \
#          2   3
#           \   \
#            5   4
root = TreeNode(1, TreeNode(2, None, TreeNode(5)), TreeNode(3, None, TreeNode(4)))
assert right_side_view(root) == [1, 3, 4]

# Test 2:    1            view: [1, 3, 4] — 4 is on the LEFT branch!
#           / \
#          2   3
#         /
#        4
root = TreeNode(1, TreeNode(2, TreeNode(4)), TreeNode(3))
assert right_side_view(root) == [1, 3, 4]

assert right_side_view(None) == []
assert right_side_view(TreeNode(1)) == [1]

# Left-only chain: every node is visible
chain = TreeNode(1, TreeNode(2, TreeNode(3)))
assert right_side_view(chain) == [1, 2, 3]
```

### Criteres de reussite

- [ ] BFS par niveaux avec taille de queue figee (ou DFS root-right-left avec profondeur)
- [ ] Le dernier noeud de CHAQUE niveau est capture, meme s'il est dans le sous-arbre gauche
- [ ] O(n) temps, O(w) espace (largeur max de l'arbre)
- [ ] Le test 2 (noeud visible a gauche) passe
- [ ] Tous les tests passent

---

## Exercice 6 : BST in-order — Kth Smallest Element in a BST

### Objectif

Exploiter LA propriete fondamentale du BST : le parcours in-order produit les valeurs **triees**. Avec arret anticipe — parcourir tout l'arbre puis indexer est un anti-pattern.

### Consigne

Etant donne la racine d'un BST et un entier `k` (1-indexe), retourne la `k`-ieme **plus petite** valeur de l'arbre.

**Contrainte** : arret des que le k-ieme element est atteint — O(h + k) temps, pas O(n) systematique. Utilise un parcours in-order **iteratif** avec une stack explicite.

```python
def kth_smallest(root: TreeNode, k: int) -> int:
    """
    Return the kth smallest value in the BST (1-indexed).
    Stop traversing as soon as the answer is found.
    """
    pass
```

### Tests

```python
#       3
#      / \
#     1   4
#      \
#       2
root = TreeNode(3, TreeNode(1, None, TreeNode(2)), TreeNode(4))
assert kth_smallest(root, 1) == 1
assert kth_smallest(root, 2) == 2
assert kth_smallest(root, 3) == 3
assert kth_smallest(root, 4) == 4

#       5
#      / \
#     3   6
#    / \
#   2   4
#  /
# 1
root = TreeNode(5, TreeNode(3, TreeNode(2, TreeNode(1)), TreeNode(4)), TreeNode(6))
assert kth_smallest(root, 3) == 3
assert kth_smallest(root, 6) == 6
assert kth_smallest(TreeNode(1), 1) == 1
```

### Criteres de reussite

- [ ] Parcours in-order ITERATIF avec stack explicite (pas de generation de la liste complete)
- [ ] Arret anticipe au k-ieme pop — O(h + k) temps
- [ ] O(h) espace pour la stack
- [ ] Tu sais expliquer pourquoi in-order sur un BST donne l'ordre trie (gauche < racine < droite recursivement)
- [ ] Tous les tests passent
