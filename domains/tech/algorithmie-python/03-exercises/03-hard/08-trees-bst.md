# Exercices Hard — Trees & BST

> Toutes les exercices utilisent la structure `TreeNode(val, left, right)`. Les solutions fournissent `build(values)` (BFS level-order) et `to_list(root)` (BFS level-order avec `None` markers) pour comparer les arbres.

---

## Exercice 7 : Serialize / Deserialize Binary Tree

### Objectif

Implementer un codec arbre↔string robuste. C'est LE probleme hard canonique sur les arbres : il valide que tu maitrises a la fois la serialisation pre-order avec markers et la reconstruction par consommation d'iterateur. Le round-trip doit etre parfait sur n'importe quelle forme d'arbre (y compris non-BST, valeurs negatives, arbres degenere).

### Consigne

Implemente une classe `Codec` avec deux methodes :
- `serialize(root)` : convertit l'arbre en string
- `deserialize(data)` : reconstruit l'arbre exact a partir de la string

Le round-trip `deserialize(serialize(root))` doit produire un arbre **structurellement identique** (memes valeurs, meme forme). Le format interne est libre tant que le round-trip est exact.

```python
class Codec:
    def serialize(self, root: "TreeNode | None") -> str:
        """Encode a tree to a single string."""
        pass

    def deserialize(self, data: str) -> "TreeNode | None":
        """Decode the string back to the exact same tree."""
        pass
```

**Indice** : pre-order avec markers `#` pour les `None` (voir le cours). A la deserialisation, consomme les tokens via un `iter(...)` : lis une valeur, construis le noeud, puis recurse pour `left` puis `right` (le pre-order garantit cet ordre). Gere les valeurs negatives (ne split pas sur le `-`).

### Tests

```python
codec = Codec()

def roundtrip(values):
    tree = build(values)
    return to_list(codec.deserialize(codec.serialize(tree)))

assert roundtrip([1, 2, 3, None, None, 4, 5]) == [1, 2, 3, None, None, 4, 5]
assert roundtrip([]) == []
assert roundtrip([1]) == [1]
assert roundtrip([-1, -2, -3]) == [-1, -2, -3]        # Negative values
assert roundtrip([1, 2, None, 3, None, 4]) == [1, 2, None, 3, None, 4]   # Left-skewed
assert roundtrip([5, 3, 8, 1, 4, 7, 9]) == [5, 3, 8, 1, 4, 7, 9]
```

### Criteres de reussite

- [ ] `serialize` : parcours systematique avec markers explicites pour les `None`
- [ ] `deserialize` : reconstruit dans le meme ordre que la serialisation (iterateur de tokens)
- [ ] Round-trip exact (valeurs ET forme) sur arbre vide, degenere, valeurs negatives
- [ ] Complexite O(n) temps et O(n) espace pour chaque sens
- [ ] Tous les tests passent

---

## Exercice 8 : Binary Tree Maximum Path Sum

### Objectif

Le probleme hard ou la recursion "retourne une chose, met a jour une autre" atteint son sommet. Un chemin peut commencer et finir n'importe ou (pas forcement racine ni feuille) et passer par un noeud commun. C'est le pattern `diameter` pousse a l'extreme, avec gestion des valeurs negatives.

### Consigne

Etant donne la racine `root` d'un arbre binaire (valeurs possiblement negatives), retourne la **somme maximale** d'un chemin. Un chemin est une sequence de noeuds ou chaque paire consecutive est reliee par une arete, **chaque noeud apparait au plus une fois**, et le chemin ne descend/remonte qu'une fois (forme en "^" autour d'un noeud sommet). Le chemin doit contenir au moins un noeud.

```python
def max_path_sum(root: "TreeNode") -> int:
    """
    Return the maximum path sum. A path need not pass through the root
    and need not start/end at a leaf. At least one node.
    """
    pass
```

**Approche attendue** :
1. Une fonction recursive `gain(node)` retourne la **contribution maximale descendante** d'un noeud (la valeur du noeud + au mieux UN de ses cotes, jamais les deux).
2. Un cote dont le gain est negatif est ignore : on prend `max(gain(child), 0)`.
3. A chaque noeud, le chemin qui passe PAR ce noeud (sommet du "^") vaut `node.val + left_gain + right_gain` : on met a jour un maximum global avec cette valeur.

### Tests

```python
assert max_path_sum(build([1, 2, 3])) == 6
assert max_path_sum(build([-10, 9, 20, None, None, 15, 7])) == 42   # 15 + 20 + 7
assert max_path_sum(build([-3])) == -3                # Single negative node
assert max_path_sum(build([2, -1])) == 2              # Drop the negative child
assert max_path_sum(build([-2, -1])) == -1            # Best is a single node
assert max_path_sum(build([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1])) == 48
```

### Criteres de reussite

- [ ] Fonction recursive qui RETOURNE le gain descendant (un seul cote) et MET A JOUR un max global (deux cotes)
- [ ] Les contributions negatives des enfants sont coupees a 0 (`max(gain, 0)`)
- [ ] Gere un noeud unique negatif, un arbre tout-negatif
- [ ] Complexite O(n) temps, O(h) espace
- [ ] Tous les tests passent

---

## Exercice 9 : Construct Binary Tree from Preorder and Inorder

### Objectif

Reconstruire un arbre depuis ses parcours preorder + inorder. Probleme hard de divide & conquer : il faut comprendre que le premier element du preorder est la racine, et que sa position dans l'inorder separe les sous-arbres gauche/droit. Une map `valeur→index inorder` evite le O(n^2).

### Consigne

Etant donne deux listes `preorder` et `inorder` representant les parcours d'un arbre binaire **sans valeurs dupliquees**, reconstruis et retourne la racine de l'arbre.

```python
def build_tree(preorder: list[int], inorder: list[int]) -> "TreeNode | None":
    """
    Reconstruct the binary tree from its preorder and inorder traversals.
    Values are unique.
    """
    pass
```

**Approche attendue** :
- `preorder[0]` est la racine.
- Trouve sa position `i` dans `inorder` : tout ce qui est a gauche de `i` est le sous-arbre gauche, tout ce qui est a droite est le sous-arbre droit.
- Le nombre d'elements a gauche dans l'inorder = nombre d'elements du sous-arbre gauche dans le preorder. Recurse sur les sous-tranches.
- Optimise avec une `dict {valeur: index_inorder}` (lookup O(1)) et un pointeur global sur le preorder pour eviter de slicer.

### Tests

```python
def roundtrip_pre_in(values):
    """Build a tree, extract its preorder+inorder, rebuild, compare shapes."""
    tree = build(values)
    pre, ino = preorder_vals(tree), inorder_vals(tree)
    return to_list(build_tree(pre, ino))

assert to_list(build_tree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])) == [3, 9, 20, None, None, 15, 7]
assert to_list(build_tree([-1], [-1])) == [-1]
assert to_list(build_tree([], [])) == []
assert roundtrip_pre_in([1, 2, 3, 4, 5, 6, 7]) == [1, 2, 3, 4, 5, 6, 7]
assert roundtrip_pre_in([1, 2, None, 3]) == [1, 2, None, 3]      # Left-skewed
assert roundtrip_pre_in([1, None, 2, None, 3]) == [1, None, 2, None, 3]   # Right-skewed
```

### Criteres de reussite

- [ ] `preorder[0]` identifie comme racine, position dans l'inorder pour separer les sous-arbres
- [ ] Map `valeur→index inorder` pour un lookup O(1) (pas de `.index()` en O(n) dans la recursion)
- [ ] Recursion sur les bonnes tranches (tailles gauche/droite coherentes)
- [ ] Gere l'arbre vide, l'arbre degenere (gauche/droite skew), un seul noeud
- [ ] Complexite O(n) temps, O(n) espace
- [ ] Tous les tests passent
