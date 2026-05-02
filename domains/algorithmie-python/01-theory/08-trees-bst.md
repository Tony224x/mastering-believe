# Jour 8 — Trees & BST : DFS, BFS, BST, LCA, Serialize

> **Temps estime** : 60 min de lecture active | **Objectif** : maitriser les arbres binaires et les BST, les 3 parcours DFS et le BFS par niveaux, et 5 patterns recurrents en entretien

---

## 1. Pourquoi les arbres sont incontournables

Apres les arrays/strings et les hash maps, **les arbres binaires** sont la structure la plus frequente en entretien. Pourquoi ?

- Ils modelisent naturellement des donnees hierarchiques : filesystem, DOM, organigramme, expression mathematique
- Ils sont le terrain d'entrainement parfait pour la **recursion**
- Les BST permettent de comprendre l'intuition des structures O(log n) (`heap`, `TreeMap`, `sorted container`)
- 80% des problemes tombent dans 5 patterns : parcours DFS, parcours BFS, BST properties, LCA, construction/serialisation

**Si tu maitrises parfaitement DFS et BFS sur un arbre binaire, tu peux resoudre la majorite des questions d'entretien medium sur les arbres.**

---

## 2. La structure de base — TreeNode

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

Simple, mais lourd de sens : un noeud = une valeur + deux pointeurs. L'arbre lui-meme est juste **une reference vers la racine**.

### Construction rapide

```python
#         1
#        / \
#       2   3
#      / \
#     4   5
root = TreeNode(1,
    TreeNode(2, TreeNode(4), TreeNode(5)),
    TreeNode(3))
```

### Vocabulaire a connaitre

| Terme | Definition |
|-------|------------|
| **root** | Le noeud au sommet |
| **leaf** | Noeud sans enfants (left == right == None) |
| **height** | Longueur du chemin le plus long de la racine a une feuille |
| **depth** | Distance de la racine a un noeud donne |
| **balanced** | La difference de hauteur entre sous-arbres gauche/droit est au plus 1 partout |
| **complete** | Tous les niveaux sont pleins sauf peut-etre le dernier (rempli de gauche a droite) |
| **full** | Chaque noeud a 0 ou 2 enfants |
| **BST** | Binary Search Tree : `left < node < right` partout |

---

## 3. Pattern 1 — DFS : les 3 parcours

Depth-First Search = "descendre le plus loin possible avant de remonter". 3 variantes selon **quand on visite le noeud par rapport a ses enfants** :

| Parcours | Ordre | Usage typique |
|----------|-------|---------------|
| **Preorder** | Node → Left → Right | Copier un arbre, serialiser |
| **Inorder** | Left → Node → Right | **BST** : donne les valeurs triees |
| **Postorder** | Left → Right → Node | Supprimer un arbre, evaluer une expression |

### Version recursive (la plus simple)

```python
def preorder(node):
    if not node:
        return
    print(node.val)      # Visit
    preorder(node.left)
    preorder(node.right)

def inorder(node):
    if not node:
        return
    inorder(node.left)
    print(node.val)      # Visit
    inorder(node.right)

def postorder(node):
    if not node:
        return
    postorder(node.left)
    postorder(node.right)
    print(node.val)      # Visit
```

Toutes en **O(n) temps, O(h) espace** (h = hauteur = stack de recursion).

### Version iterative avec stack (a connaitre pour entretien)

```python
def preorder_iterative(root):
    if not root:
        return []
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push right FIRST so left is processed first (LIFO)
        if node.right: stack.append(node.right)
        if node.left: stack.append(node.left)
    return result

def inorder_iterative(root):
    stack = []
    result = []
    node = root
    while node or stack:
        # Descendre a gauche a fond
        while node:
            stack.append(node)
            node = node.left
        # Visiter le noeud, puis passer a droite
        node = stack.pop()
        result.append(node.val)
        node = node.right
    return result
```

> **Piege entretien** : l'inorder iteratif est le plus souvent demande. Memorise-le par coeur.

---

## 4. Pattern 2 — BFS par niveaux

Breadth-First Search = parcours par "couches". Indispensable pour :
- Trouver le noeud le plus proche de la racine avec une propriete
- Niveaux de l'arbre (level order, zigzag, right side view)
- Plus court chemin dans un arbre

### Template BFS

```python
from collections import deque

def level_order(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)     # CRUCIAL : snapshot de la taille du niveau
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
# Time: O(n), Space: O(w) ou w = largeur max (au pire O(n/2) ~ O(n))
```

> **Cle** : `level_size = len(queue)` capture combien de noeuds sont a ce niveau AVANT qu'on ajoute les enfants. C'est ce qui permet de grouper les valeurs par niveau.

### Variantes classiques

```python
# Right Side View : le dernier noeud de chaque niveau
def right_side_view(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:   # Dernier du niveau
                result.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
    return result
```

---

## 5. Pattern 3 — Calculs recursifs sur l'arbre

Template : "je decris ce que la fonction retourne pour un sous-arbre, puis je combine les resultats gauche/droit".

### Max Depth

```python
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
# Time: O(n), Space: O(h)
```

### Diameter (le plus long chemin entre deux feuilles)

```python
def diameter(root):
    best = [0]
    def depth(node):
        if not node:
            return 0
        l = depth(node.left)
        r = depth(node.right)
        best[0] = max(best[0], l + r)   # Chemin passant par ce noeud
        return 1 + max(l, r)            # Hauteur du sous-arbre
    depth(root)
    return best[0]
```

> **Pattern mental** : la fonction recursive retourne la hauteur, mais on met a jour un etat global (`best`) avec le chemin qui passe par le noeud courant.

### Balanced Binary Tree

```python
def is_balanced(root):
    def check(node):
        if not node:
            return 0                      # Hauteur 0
        l = check(node.left)
        if l == -1: return -1             # Deja desequilibre
        r = check(node.right)
        if r == -1: return -1
        if abs(l - r) > 1: return -1      # Desequilibre ici
        return 1 + max(l, r)
    return check(root) != -1
```

---

## 6. Pattern 4 — BST Properties

Un BST garantit **`left.val < node.val < right.val`** partout. Consequences :

1. **Inorder traversal donne les valeurs triees** → LE test clef pour valider un BST
2. **Recherche en O(h)** (pas O(n)) : on compare et on descend a gauche ou a droite
3. **Insertion et suppression en O(h)** avec la meme technique

### Validate BST

```python
# PIEGE : verifier juste left.val < node.val < right.val est FAUX
# Il faut propager des bornes min/max le long de la descente

def is_valid_bst(root, low=float('-inf'), high=float('inf')):
    if not root:
        return True
    if not (low < root.val < high):
        return False
    return (is_valid_bst(root.left, low, root.val) and
            is_valid_bst(root.right, root.val, high))
```

### Search in BST

```python
def search_bst(root, target):
    while root:
        if root.val == target:
            return root
        root = root.left if target < root.val else root.right
    return None
# Time: O(h) — O(log n) balanced, O(n) worst case
```

---

## 7. Pattern 5 — Lowest Common Ancestor (LCA)

**LCA(p, q)** = le noeud le plus profond qui a p ET q dans son sous-arbre.

### LCA dans un arbre binaire quelconque

```python
def lca(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    if left and right:
        return root              # p et q sont de chaque cote → LCA = root
    return left or right         # Un seul cote → c'est la
# Time: O(n), Space: O(h)
```

### LCA dans un BST (plus efficace)

```python
def lca_bst(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root          # Split point = LCA
# Time: O(h)
```

---

## 8. Serialize / Deserialize

Convertir un arbre en string (et vice versa). Utile pour transmettre des arbres, les mettre en cache, les comparer.

```python
# Pre-order with null markers
def serialize(root):
    result = []
    def dfs(node):
        if not node:
            result.append("#")
            return
        result.append(str(node.val))
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ",".join(result)

def deserialize(data):
    values = iter(data.split(","))
    def build():
        val = next(values)
        if val == "#":
            return None
        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node
    return build()
```

> **Cle** : les markers `#` pour les None permettent de reconstruire l'arbre sans ambiguite. Sans eux, on ne pourrait pas differencier les sous-arbres.

---

## 9. Decision Tree — Quel pattern utiliser ?

```
Le probleme est sur un arbre binaire ?
|
├── On cherche a PARCOURIR tous les noeuds dans un ordre ?
│   ├── Par PROFONDEUR → DFS (pre/in/post order)
│   └── Par NIVEAU → BFS avec deque + level_size
|
├── On calcule une PROPRIETE recursive (depth, diameter, balanced) ?
│   └── Recursion bottom-up : retourner la valeur du sous-arbre,
│       mettre a jour un etat global si besoin
|
├── C'est un BST et il faut CHERCHER / VALIDER / INSERER ?
│   └── BST pattern : exploiter left < node < right,
│       propager des bornes pour valider
|
├── On cherche un ANCETRE commun de 2 noeuds ?
│   └── LCA pattern (recursion ou iterative pour BST)
|
├── On doit CONVERTIR un arbre en string ou vice versa ?
│   └── Serialize/Deserialize avec markers pour les null
|
└── On doit RECONSTRUIRE un arbre depuis preorder + inorder ?
    └── Diviser avec les indices inorder (voir Day 9/10)
```

**Signaux dans l'enonce** :

| Signal | Pattern |
|--------|---------|
| "level", "zigzag", "right side" | BFS par niveaux |
| "depth", "height", "diameter" | DFS recursif |
| "validate BST", "inorder" | BST properties |
| "lowest common ancestor" | LCA |
| "path sum", "root to leaf" | DFS avec accumulator |
| "serialize", "encode tree" | Serialize/Deserialize |

---

## 10. Complexites

| Operation | Arbre equilibre | Arbre quelconque |
|-----------|-----------------|-------------------|
| DFS (pre/in/post) | O(n) | O(n) |
| BFS level order | O(n) | O(n) |
| Max depth | O(n) | O(n) |
| Search BST | O(log n) | O(n) |
| Insert BST | O(log n) | O(n) |
| Validate BST | O(n) | O(n) |
| LCA | O(n) | O(n) |
| LCA BST | O(log n) | O(n) |
| Serialize | O(n) | O(n) |

**Espace** : la plupart des operations recursives utilisent O(h) pour la stack, avec h = hauteur. Donc O(log n) pour un arbre equilibre, O(n) pour un arbre degenere (chaine).

---

## 11. Pieges courants

**Piege 1 — Confondre hauteur et profondeur**
- Profondeur : du haut vers le bas (root = 0)
- Hauteur : d'une feuille vers le haut (feuilles = 0)

**Piege 2 — Validate BST naif**
```python
# FAUX : ne verifie pas les contraintes globales
def is_bst_wrong(node):
    if not node: return True
    if node.left and node.left.val >= node.val: return False
    if node.right and node.right.val <= node.val: return False
    return is_bst_wrong(node.left) and is_bst_wrong(node.right)
# Contre-exemple : root=5, left=3, left.right=6 → le 6 viole la contrainte
```

**Piege 3 — DFS recursif sur arbre profond → RecursionError**
Python a une limite de recursion (~1000 par defaut). Pour un arbre degenere, passer en version iterative avec stack, ou `sys.setrecursionlimit(10000)`.

**Piege 4 — BFS sans level_size**
Oublier le `level_size = len(queue)` melange tous les niveaux dans un meme groupe.

**Piege 5 — Modifier la structure pendant l'iteration**
Comme pour les dicts, ne pas muter l'arbre pendant qu'on le parcourt.

---

## 12. Flash Cards — Revision espacee

**Q1** : Quel parcours donne les valeurs d'un BST dans l'ordre croissant ?
> **R1** : L'inorder (Left → Node → Right). C'est la maniere la plus simple de tester qu'un arbre est un BST valide : faire un inorder et verifier que la sequence est strictement croissante.

**Q2** : Dans un BFS level order, a quoi sert `level_size = len(queue)` ?
> **R2** : A capturer le nombre de noeuds du niveau courant AVANT d'ajouter leurs enfants. Sans ce snapshot, les niveaux se melangent car on ajoute de nouveaux noeuds a la queue pendant qu'on la vide.

**Q3** : Pourquoi valider un BST en comparant juste `left.val < node.val` est-il faux ?
> **R3** : Parce que la contrainte est globale : TOUS les descendants gauches doivent etre < node.val, pas juste l'enfant direct. Il faut propager des bornes min/max pendant la descente.

**Q4** : Quelle est la complexite d'une recherche dans un BST ? Pourquoi ce n'est pas toujours O(log n) ?
> **R4** : O(h) ou h = hauteur. Pour un BST equilibre, h = O(log n). Mais pour un BST degenere (valeurs inserees dans l'ordre), h = O(n) — l'arbre devient une liste chainee.

**Q5** : Dans le LCA, pourquoi `return left or right` fonctionne-t-il quand un seul cote retourne un resultat ?
> **R5** : Si seul un cote retourne non-None, c'est que p et q sont tous les deux de ce cote → le LCA est deja trouve plus bas. On propage donc ce resultat vers le haut. Si les deux cotes sont non-None, on est au split point, donc `root` est le LCA.

---

## Resume — Key Takeaways

1. **TreeNode(val, left, right)** — la structure de base, a connaitre par coeur
2. **3 parcours DFS** : pre (copie/serialize), in (BST trie), post (suppression/eval)
3. **BFS avec deque + level_size** pour les problemes par niveaux
4. **Validate BST** avec bornes min/max propagees, PAS avec comparaisons locales
5. **LCA** : recursion simple retournant root, left, ou right selon la position
6. **Serialize** : pre-order avec markers `#` pour les None
7. **Complexites** : O(n) temps pour tout parcours complet, O(h) espace recursion

---

## Pour aller plus loin

Ressources canoniques sur les arbres et BST :

- **MIT 6.006 — Introduction to Algorithms** (Erik Demaine, MIT OCW Spring 2020) — Lec. 5-7 (Binary Trees, BST, AVL) : derivation pedagogique des invariants BST et de l'auto-balancement. https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/
- **CLRS — Introduction to Algorithms** (4th ed, MIT Press 2022) — Ch 12 (Binary Search Trees) et Ch 13 (Red-Black Trees) : l'analyse formelle complete avec preuves de hauteur O(log n). https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
- **NeetCode — Trees roadmap** — 15 problemes phares (Invert, Max Depth, Same Tree, Subtree, LCA, Validate BST, Serialize) avec template recursif Python. https://neetcode.io/roadmap
- **Princeton Algorithms Part 1** (Sedgewick & Wayne, Coursera, gratuit) — Week 5 (Balanced Search Trees) : 2-3 trees + red-black BSTs avec visualisations. https://www.coursera.org/learn/algorithms-part1
