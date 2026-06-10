# Exercices Hard — Trees & BST

---

## Exercice 7 : Design — Serialize and Deserialize Binary Tree

### Objectif

Concevoir un format de serialisation **sans ambiguite** pour un arbre binaire quelconque. Le piege conceptuel : un seul parcours (pre-order ou level-order) ne suffit PAS sans marqueurs de noeuds null.

### Consigne

Implemente une classe `Codec` avec deux methodes :
- `serialize(root) -> str` : encode l'arbre en une string.
- `deserialize(data) -> TreeNode | None` : reconstruit l'arbre **exactement identique** depuis la string.

**Contraintes** :
- Round-trip parfait : `deserialize(serialize(t))` doit reproduire la structure ET les valeurs de `t`, y compris les arbres degeneres (chaines gauche/droite) et les valeurs negatives.
- O(n) temps dans les deux sens. La deserialisation ne doit PAS re-scanner la string a chaque noeud (pas de `split` repete, pas de `pop(0)` sur une liste — utiliser un iterateur ou un index).

```python
class Codec:
    def serialize(self, root: TreeNode | None) -> str:
        pass

    def deserialize(self, data: str) -> TreeNode | None:
        pass
```

**Approche recommandee** : pre-order avec marqueur null (`"#"`) et separateur (`","`). La position des marqueurs rend la reconstruction deterministe : la recursion consomme les tokens dans l'ordre exact ou ils ont ete produits.

### Tests

```python
def same_tree(a, b):
    if not a and not b:
        return True
    if not a or not b or a.val != b.val:
        return False
    return same_tree(a.left, b.left) and same_tree(a.right, b.right)

codec = Codec()

# Round-trip on a normal tree
t = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
assert same_tree(codec.deserialize(codec.serialize(t)), t)

# Empty tree
assert codec.deserialize(codec.serialize(None)) is None

# Single node, negative values
t = TreeNode(-42)
assert same_tree(codec.deserialize(codec.serialize(t)), t)

# Degenerate chains (left-only and right-only) — same VALUES, different SHAPES
left_chain = TreeNode(1, TreeNode(2, TreeNode(3)))
right_chain = TreeNode(1, None, TreeNode(2, None, TreeNode(3)))
assert same_tree(codec.deserialize(codec.serialize(left_chain)), left_chain)
assert same_tree(codec.deserialize(codec.serialize(right_chain)), right_chain)
assert not same_tree(codec.deserialize(codec.serialize(left_chain)), right_chain)

# Duplicate values everywhere — structure must still be preserved
t = TreeNode(7, TreeNode(7, TreeNode(7)), TreeNode(7))
assert same_tree(codec.deserialize(codec.serialize(t)), t)
```

### Criteres de reussite

- [ ] Les marqueurs null sont presents dans le format — tu peux montrer deux arbres differents qui seraient confondus sans eux
- [ ] Deserialisation en O(n) via un iterateur sur les tokens (pas de `pop(0)` O(n))
- [ ] Round-trip parfait sur : arbre vide, noeud unique, chaines degenerees, valeurs negatives, doublons
- [ ] Les valeurs multi-chiffres et negatives survivent (split sur separateur, pas par caractere)
- [ ] O(n) temps et espace dans les deux sens

---

## Exercice 8 : Recursion globale vs locale — Binary Tree Maximum Path Sum

### Objectif

Le hard d'arbre le plus discriminant en entretien : il force a distinguer la valeur **retournee a son parent** (chemin descendant simple) de la valeur **candidate globale** (chemin en "tente" passant par le noeud). Confondre les deux est l'erreur n°1.

### Consigne

Un **chemin** est une suite de noeuds relies par des aretes, ou chaque noeud apparait au plus une fois. Le chemin n'a pas besoin de passer par la racine. Retourne la **somme maximale** d'un chemin de l'arbre (au moins un noeud).

```python
def max_path_sum(root: TreeNode) -> int:
    """
    Return the maximum path sum of any path in the tree.
    The path needs at least one node and does not need to pass the root.
    """
    pass
```

**La distinction cle** :
- Ce que le noeud **retourne a son parent** : `node.val + max(gain_gauche, gain_droit, 0)` — un chemin qui continue vers le parent ne peut descendre que d'UN cote.
- Ce que le noeud **propose comme candidat global** : `node.val + gain_gauche_clampe + gain_droit_clampe` — le chemin en tente utilise les DEUX cotes mais s'arrete la.

**Pieges** :
- Valeurs toutes negatives : la reponse est le max des noeuds, pas 0 (clamp les gains des enfants a 0, jamais la valeur du noeud).
- Le candidat global doit etre mis a jour a CHAQUE noeud, pas seulement a la racine.

### Tests

```python
#    1
#   / \
#  2   3
root = TreeNode(1, TreeNode(2), TreeNode(3))
assert max_path_sum(root) == 6              # 2 -> 1 -> 3

#    -10
#    /  \
#   9    20
#       /  \
#      15   7
root = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
assert max_path_sum(root) == 42             # 15 -> 20 -> 7 (skips the root!)

assert max_path_sum(TreeNode(-3)) == -3     # Single negative node
assert max_path_sum(TreeNode(-2, TreeNode(-1))) == -1   # All negative: best single node
assert max_path_sum(TreeNode(2, TreeNode(-1))) == 2     # Negative child is dropped

#     5
#    / \
#   4   8
#  /   / \
# 11  13  4
# /\        \
#7  2        1
n11 = TreeNode(11, TreeNode(7), TreeNode(2))
root = TreeNode(5, TreeNode(4, n11), TreeNode(8, TreeNode(13), TreeNode(4, None, TreeNode(1))))
assert max_path_sum(root) == 48             # 7 -> 11 -> 4 -> 5 -> 8 -> 13
```

### Criteres de reussite

- [ ] Deux quantites distinctes dans le code : gain retourne au parent (1 branche) vs candidat global (2 branches)
- [ ] Les gains des enfants sont clampes a 0 (`max(gain, 0)`), la valeur du noeud jamais
- [ ] Le cas tout-negatif retourne le meilleur noeud seul (test `-2/-1` → `-1`)
- [ ] Le test 2 passe : le meilleur chemin ignore la racine
- [ ] O(n) temps, O(h) espace de pile
- [ ] Tous les tests passent
