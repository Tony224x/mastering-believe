# Exercices Hard — Linked Lists

> Les solutions reutilisent la classe `ListNode` et les helpers `build` / `to_list` du fichier medium.

---

## Exercice 7 : Reversal par blocs — Reverse Nodes in k-Group

### Objectif

Inverser la liste **par groupes de k** noeuds, en laissant le dernier groupe incomplet intact. C'est le boss final du pattern reversal : il combine comptage, reversal local et re-couture des blocs. Probleme hard tres frequent en entretien senior.

### Consigne

Etant donne la tete d'une liste et un entier `k`, inverse les noeuds k par k et retourne la liste modifiee. Si le nombre de noeuds n'est pas un multiple de `k`, les noeuds restants a la fin restent dans leur ordre original.

Tu ne dois **PAS** modifier les valeurs des noeuds, seulement les pointeurs.

```python
def reverse_k_group(head, k: int):
    """
    Reverse the list in groups of k nodes. Leftover tail (< k) stays as-is.
    Only pointer manipulation — do not change node values.
    """
    pass
```

**Approche attendue** :
1. Verifie qu'il reste au moins `k` noeuds (sinon, retourne tel quel)
2. Inverse les `k` premiers noeuds
3. Connecte recursivement (ou iterativement) le bloc inverse au resultat du reste
4. Utilise un dummy head pour la version iterative

### Tests

```python
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 2)) == [2, 1, 4, 3, 5]
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 3)) == [3, 2, 1, 4, 5]
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 1)) == [1, 2, 3, 4, 5]   # k=1, no change
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 5)) == [5, 4, 3, 2, 1]
assert to_list(reverse_k_group(build([1, 2, 3, 4]), 2)) == [2, 1, 4, 3]
assert to_list(reverse_k_group(build([1]), 2)) == [1]                            # fewer than k
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5, 6, 7]), 3)) == [3, 2, 1, 6, 5, 4, 7]
```

### Criteres de reussite

- [ ] Verifie qu'il reste >= k noeuds avant d'inverser un groupe
- [ ] Le dernier groupe incomplet reste dans l'ordre original
- [ ] Ne modifie QUE les pointeurs (pas les valeurs)
- [ ] Complexite O(n) temps, O(1) espace (version iterative) ou O(n/k) pile (recursive)
- [ ] Gere `k = 1`, liste plus courte que k, et longueur multiple de k
- [ ] Tous les tests passent

---

## Exercice 8 : Heap merge — Merge k Sorted Lists

### Objectif

Fusionner `k` listes triees en une seule, de maniere optimale via un **min-heap**. Ce probleme combine linked lists + structure de tas et teste le choix de la bonne strategie de merge (O(N log k) vs O(N k)).

### Consigne

On te donne un tableau de `k` listes chainees, chacune triee en ordre croissant. Fusionne-les en une seule liste triee et retourne sa tete.

```python
def merge_k_lists(lists):
    """
    Merge k sorted linked lists into one sorted list. Return its head.
    lists: list of ListNode heads (some may be None).
    """
    pass
```

**Approche attendue (min-heap)** :
- Pousse la tete de chaque liste non vide dans un `heapq`
- Le tas contient des tuples `(val, index_tiebreaker, node)` — l'index evite de comparer des `ListNode` (non comparables)
- Pop le plus petit, attache-le au resultat, pousse son successeur

> Alternative acceptee : divide & conquer (merge par paires), egalement O(N log k).

### Tests

```python
r = merge_k_lists([build([1, 4, 5]), build([1, 3, 4]), build([2, 6])])
assert to_list(r) == [1, 1, 2, 3, 4, 4, 5, 6]

assert to_list(merge_k_lists([])) == []
assert to_list(merge_k_lists([build([])])) == []
assert to_list(merge_k_lists([build([]), build([1]), build([])])) == [1]
assert to_list(merge_k_lists([build([1, 2, 3])])) == [1, 2, 3]
assert to_list(merge_k_lists([build([5]), build([4]), build([3]), build([2]), build([1])])) == [1, 2, 3, 4, 5]

# Stress: 10 lists, each [i, i+10, i+20]
lists = [build([i, i + 10, i + 20]) for i in range(10)]
assert to_list(merge_k_lists(lists)) == sorted([i for i in range(30)])
```

### Criteres de reussite

- [ ] Utilise un min-heap (`heapq`) OU un divide & conquer par paires
- [ ] Avec le heap : utilise un tie-breaker (compteur/index) pour eviter de comparer des `ListNode`
- [ ] Complexite **O(N log k)** ou N = nombre total de noeuds, k = nombre de listes
- [ ] Gere les listes vides dans l'entree et l'entree vide
- [ ] Reutilise les noeuds existants (pas de copie de valeurs dans de nouveaux noeuds)
- [ ] Tous les tests passent

---

## Exercice 9 : Deep copy — Copy List with Random Pointer

### Objectif

Cloner une liste dont chaque noeud possede, en plus de `next`, un pointeur `random` qui peut pointer vers n'importe quel noeud (ou `None`). Ce probleme teste la gestion d'un mapping ancien→nouveau, avec une variante elegante O(1) espace par interleaving.

### Consigne

Chaque noeud a la structure suivante :

```python
class Node:
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random
```

Construis une **copie profonde** de la liste : tous les noeuds sont neufs, et les pointeurs `next`/`random` de la copie pointent vers des noeuds de la copie (jamais vers l'originale).

```python
def copy_random_list(head):
    """
    Return a deep copy of a linked list where each node has a random pointer.
    """
    pass
```

**Approche attendue** :
- Solution simple : un dict `{ancien_noeud: nouveau_noeud}`, deux passes (O(n) temps, O(n) espace)
- Solution avancee (bonus) : interleaving — inserer chaque copie juste apres son original, fixer les `random`, puis desentrelacer. O(n) temps, O(1) espace auxiliaire.

### Tests

```python
class Node:
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def build_with_random(arr):
    # arr = list of (val, random_index_or_None)
    nodes = [Node(v) for v, _ in arr]
    for i, (_, r) in enumerate(arr):
        nodes[i].next = nodes[i + 1] if i + 1 < len(nodes) else None
        nodes[i].random = nodes[r] if r is not None else None
    return nodes[0] if nodes else None

def serialize(head):
    # Map node -> index, then emit (val, random_index)
    idx, n, out = {}, head, []
    i = 0
    while n:
        idx[n] = i; n = n.next; i += 1
    n = head
    while n:
        out.append((n.val, idx[n.random] if n.random else None))
        n = n.next
    return out

def assert_deep_copy(original, copy):
    # Copy must serialize identically AND share no node with the original
    assert serialize(original) == serialize(copy)
    orig_ids, n = set(), original
    while n:
        orig_ids.add(id(n)); n = n.next
    n = copy
    while n:
        assert id(n) not in orig_ids   # All nodes are brand new
        n = n.next

cases = [
    [(7, None), (13, 0), (11, 4), (10, 2), (1, 0)],
    [(1, 1), (2, 1)],
    [(3, None), (3, 0), (3, None)],
    [],
    [(1, None)],
]
for case in cases:
    src = build_with_random(case)
    cpy = copy_random_list(src)
    assert_deep_copy(src, cpy)
```

### Criteres de reussite

- [ ] La copie ne partage AUCUN noeud avec l'originale (deep copy reelle)
- [ ] Les pointeurs `random` de la copie sont corrects (y compris `None` et auto-reference)
- [ ] Complexite O(n) temps ; O(n) espace (dict) ou O(1) auxiliaire (interleaving)
- [ ] Gere la liste vide et la liste a un seul noeud
- [ ] Tous les tests passent
