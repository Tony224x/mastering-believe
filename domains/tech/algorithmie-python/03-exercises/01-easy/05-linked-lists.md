# Exercices Easy — Linked Lists

---

## Exercice 1 : Reversal — Reverse Linked List

### Objectif

Maitriser la manipulation a trois pointeurs (`prev`, `curr`, `next`). C'est l'operation fondamentale sur laquelle reposent de nombreux problemes plus complexes (k-group reversal, palindrome check).

### Consigne

Etant donne la tete d'une linked list, inverse la liste et retourne la nouvelle tete.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_list(head):
    """
    Reverse a singly linked list and return the new head.
    """
    pass
```

### Tests

```python
def build(values):
    dummy = ListNode(0)
    t = dummy
    for v in values:
        t.next = ListNode(v)
        t = t.next
    return dummy.next

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

assert to_list(reverse_list(build([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
assert to_list(reverse_list(build([1, 2]))) == [2, 1]
assert to_list(reverse_list(build([1]))) == [1]
assert reverse_list(None) is None
```

### Criteres de reussite

- [ ] Utilise trois pointeurs : `prev`, `curr`, `nxt`
- [ ] Sauvegarde `curr.next` AVANT de modifier `curr.next = prev`
- [ ] Retourne `prev` a la fin (pas `head`)
- [ ] Complexite O(n) temps et O(1) espace (version iterative)
- [ ] Gere le edge case `head is None`

---

## Exercice 2 : Fast & Slow — Middle of the Linked List

### Objectif

Decouvrir le pattern fast/slow pointers en une passe, sans aucune memoire auxiliaire. C'est la base de nombreux problemes (cycle detection, palindrome, nth from end).

### Consigne

Etant donne la tete d'une linked list non vide, retourne le noeud du milieu. Si la liste a un nombre pair d'elements, retourne le SECOND milieu.

Exemple : `[1, 2, 3, 4, 5]` → retourne le noeud `3`
Exemple : `[1, 2, 3, 4, 5, 6]` → retourne le noeud `4`

```python
def middle_node(head):
    """
    Return the middle node of the linked list.
    If even length, return the second middle.
    """
    pass
```

### Tests

```python
assert middle_node(build([1, 2, 3, 4, 5])).val == 3
assert middle_node(build([1, 2, 3, 4, 5, 6])).val == 4
assert middle_node(build([1])).val == 1
assert middle_node(build([1, 2])).val == 2
```

### Criteres de reussite

- [ ] Utilise deux pointeurs `slow` et `fast`
- [ ] `fast` avance de 2, `slow` avance de 1 a chaque step
- [ ] Condition de boucle `while fast and fast.next` (pas juste `while fast`)
- [ ] Complexite O(n) temps et O(1) espace
- [ ] Retourne `slow` a la fin (et pas fast)

---

## Exercice 3 : Merge — Merge Two Sorted Lists

### Objectif

Maitriser le pattern merge avec dummy head. C'est la brique de base du merge sort sur linked list.

### Consigne

Etant donne les tetes de deux linked lists triees `list1` et `list2`, fusionne-les en une seule liste triee (croissante). Retourne la tete de la liste fusionnee.

Tu dois **reutiliser les noeuds existants** — ne cree PAS de nouveaux noeuds.

```python
def merge_two_lists(list1, list2):
    """
    Merge two sorted linked lists and return the merged sorted list.
    Reuse existing nodes — do not allocate new ones.
    """
    pass
```

### Tests

```python
assert to_list(merge_two_lists(build([1, 2, 4]), build([1, 3, 4]))) == [1, 1, 2, 3, 4, 4]
assert to_list(merge_two_lists(build([]), build([]))) == []
assert to_list(merge_two_lists(build([]), build([0]))) == [0]
assert to_list(merge_two_lists(build([1, 2, 3]), build([]))) == [1, 2, 3]
assert to_list(merge_two_lists(build([5]), build([1, 2, 3]))) == [1, 2, 3, 5]
```

### Criteres de reussite

- [ ] Utilise un dummy head pour simplifier le code
- [ ] Maintient un pointeur `tail` qui est toujours le dernier noeud de la liste fusionnee
- [ ] A la fin de la boucle, attache le reste de la liste non vide (`tail.next = l1 or l2`)
- [ ] Retourne `dummy.next` (pas `dummy`)
- [ ] Complexite O(n + m) temps et O(1) espace
- [ ] NE cree PAS de nouveaux ListNode
