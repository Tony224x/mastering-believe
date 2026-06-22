# Jour 5 — Linked Lists : Fast/Slow Pointers, Reversal & Merge

> **Temps estime** : 60 min de lecture active | **Objectif** : maitriser les linked lists et les trois patterns (fast/slow, reversal, merge) qui couvrent 95% des problemes d'entretien.

---

## 1. Pourquoi les linked lists sont un sujet special en entretien

**Observation** : en production, on n'utilise quasi jamais de linked list. Les `list` Python, les `ArrayList` Java, les `Vec` Rust sont presque toujours le bon choix.

**Alors pourquoi en entretien ?** Parce que les linked lists testent trois competences cruciales :

1. **Manipulation de pointeurs** — tu dois raisonner sur ce que `prev.next` et `curr.next` designent a chaque etape
2. **Raisonnement sur invariants** — les bugs viennent de pointeurs mal updates, il faut ecrire du code dont chaque ligne preserve un invariant clair
3. **Efficacite memoire** — les patterns fast/slow font TOUT en O(1) espace, ce qui est elegant

**Regle d'or** : en entretien, un probleme sur linked list est TOUJOURS resolvable en O(n) temps et O(1) espace. Si ta premiere idee utilise un array ou un set, c'est souvent bon mais pas optimal — cherche un pattern a deux pointeurs.

---

## 2. Definitions et implementation de base

### Singly Linked List

```python
class ListNode:
    """A node in a singly linked list."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Building: 1 -> 2 -> 3 -> None
head = ListNode(1, ListNode(2, ListNode(3)))
```

Chaque noeud a une valeur et un pointeur vers le suivant. Le dernier noeud pointe vers `None`.

**Operations de base et leur cout :**

| Operation | Cost | Commentaire |
|-----------|------|-------------|
| Acces au i-eme element | O(n) | Il faut walker |
| Insert en tete | O(1) | `new_node.next = head; head = new_node` |
| Insert en queue | O(n) | Sans pointeur de queue |
| Delete en tete | O(1) | `head = head.next` |
| Delete d'un noeud donne | O(1) | Si on a deja le noeud ET son predecesseur |
| Recherche | O(n) | Pas de hash |

### Doubly Linked List

```python
class DListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next
```

Plus flexible (on peut traverser dans les deux sens) mais double la memoire. Utile pour LRU cache, deque implementation.

### Technique du Dummy Head (sentinel)

```python
# Le dummy head evite de gerer specialement le cas "head is None"
dummy = ListNode(0)         # Sentinel: pas une vraie valeur
dummy.next = head           # On rattache la vraie liste
curr = dummy
# ... operations ...
return dummy.next           # La vraie tete potentiellement modifiee
```

**Pourquoi c'est important** : quand on construit ou modifie le debut d'une liste, on a souvent besoin de raisonner sur "le predecesseur de la tete". Un dummy head donne un predecesseur gratuit et elimine les cas speciaux.

---

## 3. Pattern 1 — Fast & Slow Pointers (Floyd's Cycle Detection)

### Concept

Deux pointeurs avancent dans la liste a des vitesses differentes :
- **slow** avance de 1 a chaque step
- **fast** avance de 2 a chaque step

Ce decalage permet de resoudre plusieurs problemes en une seule passe sans memoire auxiliaire.

```python
# Template generique
def fast_slow_pattern(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next          # 1 step
        fast = fast.next.next     # 2 steps
        # ... check condition ...
```

### Cas 1 — Trouver le milieu en O(n) une seule passe

```python
def find_middle(head):
    """Return the middle node (in the case of even length, the second middle)."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
# When fast reaches the end, slow is exactly at the middle
# Time: O(n), Space: O(1)
```

**Pourquoi ca marche ?** Si fast avance 2x plus vite, quand fast a parcouru n pas, slow a parcouru n/2. Quand fast est a la fin, slow est au milieu.

### Cas 2 — Detection de cycle (Floyd's algorithm)

```python
def has_cycle(head):
    """Return True if the linked list contains a cycle."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:          # Pointers met -> cycle exists
            return True
    return False
# Time: O(n), Space: O(1)
```

**Insight mathematique** : si la liste a un cycle, les deux pointeurs se rencontreront FORCEMENT dans le cycle. Pourquoi ? Parce que la difference entre fast et slow diminue de 1 a chaque step a l'interieur du cycle, donc elle atteindra zero en au plus `cycle_length` iterations.

### Cas 3 — Trouver le debut du cycle

```python
def detect_cycle(head):
    """Return the node where the cycle begins, or None."""
    slow = fast = head
    # Phase 1: detect meeting point
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            break
    else:
        return None               # No cycle (fast hit None)

    # Phase 2: restart slow from head, advance both at speed 1
    slow = head
    while slow is not fast:
        slow = slow.next
        fast = fast.next
    return slow                   # The node where the cycle starts
# Time: O(n), Space: O(1)
```

**Demonstration rapide** : soit `a` la distance head -> debut du cycle, `b` la distance debut du cycle -> point de rencontre, `c` le reste du cycle. Quand ils se rencontrent : slow a marche `a + b`, fast a marche `2(a + b)` = `a + b + k*cycle` pour un entier k. Donc `a + b = k * cycle = k * (b + c)`, d'ou `a = (k-1)*cycle + c`. En redemarrant slow depuis head a vitesse 1 et en avancant fast depuis le meeting point a vitesse 1, ils se rejoindront exactement au debut du cycle apres `a` steps.

**Ne memorise pas la preuve — memorise le code (3 lignes).**

---

## 4. Pattern 2 — Reversal

### Concept

Inverser une linked list (ou un segment) en place, en maintenant trois pointeurs : `prev`, `curr`, `next`.

```python
# Template iteratif
def reverse(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next         # 1. Sauvegarde le suivant AVANT de casser le lien
        curr.next = prev        # 2. Inverse le pointeur
        prev = curr             # 3. Avance prev
        curr = nxt              # 4. Avance curr
    return prev                 # Nouveau head = ancien tail
# Time: O(n), Space: O(1)
```

**Regle mnemotechnique** : dans l'ordre — sauvegarde next, inverse, avance prev, avance curr. Si tu oublies "sauvegarde next" en premier, tu perds le reste de la liste.

### Version recursive

```python
def reverse_recursive(head):
    """Reverse a linked list recursively."""
    # Base case: empty list or single node
    if head is None or head.next is None:
        return head

    # Reverse the rest first
    new_head = reverse_recursive(head.next)

    # Fix the pointer: head.next.next should point back to head
    head.next.next = head
    head.next = None            # Break the forward link to avoid cycles

    return new_head
# Time: O(n), Space: O(n) due to recursion depth
```

**Attention** : la version recursive utilise O(n) espace (pile d'appels). Prefere l'iteratif en entretien sauf si on te demande explicitement du recursif.

### Application — Reverse Between m and n

```python
def reverse_between(head, left, right):
    """Reverse nodes from position left to right (1-indexed)."""
    dummy = ListNode(0, head)         # Dummy head to handle left=1
    prev = dummy
    for _ in range(left - 1):
        prev = prev.next              # prev is now just before position `left`

    # curr is the first node to reverse
    curr = prev.next
    for _ in range(right - left):
        # Classic "insert at front of sublist" trick
        nxt = curr.next
        curr.next = nxt.next
        nxt.next = prev.next
        prev.next = nxt

    return dummy.next
# Time: O(n), Space: O(1)
```

---

## 5. Pattern 3 — Merge

### Concept

Fusionner deux linked lists triees en une seule liste triee. Classique, mais c'est la base de mergesort sur linked list.

```python
def merge_two_sorted(l1, l2):
    """Merge two sorted linked lists into one sorted list."""
    dummy = ListNode(0)         # Dummy simplifies the edge cases
    tail = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next

    # At most one of l1, l2 is non-empty — attach the rest
    tail.next = l1 or l2

    return dummy.next
# Time: O(n + m), Space: O(1) — we reuse the existing nodes
```

**Cle** : le dummy head nous donne un `tail` qu'on peut toujours updater sans cas special pour la premiere iteration.

### Application — Sort List (Merge Sort)

```python
def sort_list(head):
    """Sort a linked list in O(n log n) time and O(1) extra space (excl. recursion)."""
    if not head or not head.next:
        return head

    # 1. Split in two with fast/slow (O(n))
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None                  # Break the list in two

    # 2. Recurse on both halves
    left = sort_list(head)
    right = sort_list(mid)

    # 3. Merge
    return merge_two_sorted(left, right)
# Time: O(n log n), Space: O(log n) for recursion
```

---

## 6. Pattern 4 — Two Pointers with Gap (nth from end, intersection)

### Concept

Deux pointeurs demarrent au meme endroit (ou a des endroits differents), avec un **decalage fixe**. Quand le pointeur de tete atteint la fin, le pointeur de queue est a la position voulue.

### Application 1 — Remove Nth From End

```python
def remove_nth_from_end(head, n):
    """Remove the nth node from the end in a single pass."""
    dummy = ListNode(0, head)
    fast = slow = dummy

    # Advance fast by n+1 steps, so the gap between fast and slow is n+1
    for _ in range(n + 1):
        fast = fast.next

    # Move both at the same pace until fast hits None
    while fast:
        fast = fast.next
        slow = slow.next

    # slow is now one before the node to remove
    slow.next = slow.next.next

    return dummy.next
# Time: O(n), Space: O(1) — single pass
```

**Pourquoi `n+1` et pas `n` ?** Parce qu'on veut que `slow` s'arrete UN NOEUD AVANT le noeud a supprimer (pour pouvoir modifier `slow.next`). Le gap doit etre de `n+1` pour ca.

### Application 2 — Intersection of Two Linked Lists

```python
def get_intersection(headA, headB):
    """Return the node where two linked lists intersect, or None."""
    if not headA or not headB:
        return None

    a, b = headA, headB
    # Both pointers traverse A then B (and vice versa)
    # After at most len(A) + len(B) steps, they meet at the intersection (or at None)
    while a is not b:
        a = a.next if a else headB
        b = b.next if b else headA
    return a                          # Either the intersection node or None
# Time: O(n + m), Space: O(1)
```

**Magie** : en faisant en sorte que chaque pointeur parcoure `A + B`, les deux pointeurs parcourent la meme distance totale. Ils arrivent donc au point d'intersection EN MEME TEMPS. Si pas d'intersection, les deux arrivent a `None` en meme temps.

---

## 7. Decision Tree — Quel pattern utiliser ?

```
Le probleme implique une linked list ?
|
├── On cherche une POSITION particuliere (milieu, nth from end, k-th) ?
│   └── FAST & SLOW pointers
│       Ex: middle node, nth from end, palindrome check
|
├── On detecte un CYCLE ou on cherche son debut ?
│   └── FLOYD'S algorithm (fast & slow)
│       Ex: has cycle, cycle start, happy number
|
├── On doit INVERSER la liste (entiere ou un segment) ?
│   └── REVERSAL avec prev/curr/next
│       Ex: reverse list, reverse between, reverse k-group
|
├── On doit FUSIONNER des listes triees ?
│   └── MERGE avec dummy head
│       Ex: merge two, merge k lists, sort list
|
├── On cherche une INTERSECTION ou un element commun ?
│   └── TWO POINTERS with gap
│       Ex: intersection of two lists, remove nth from end
|
└── On doit supprimer/dedupliquer ?
    └── Single pass avec pointeur precedent (dummy head)
        Ex: remove duplicates, delete node
```

**Raccourcis mentaux** :

| Signal dans l'enonce | Pattern |
|---------------------|---------|
| "middle of the linked list" | Fast & Slow |
| "cycle" | Floyd's algorithm |
| "reverse" | Reversal iteratif |
| "merge" / "sorted" | Merge avec dummy |
| "nth from end" | Two pointers with gap |
| "intersection" | Two pointers avec swap |
| "palindrome" | Reverse second half + compare |

---

## 8. Complexites et pieges

### Complexites standard pour linked list

| Operation | Singly | Doubly | Array (comparaison) |
|-----------|--------|--------|---------------------|
| Acces index | O(n) | O(n) | O(1) |
| Recherche | O(n) | O(n) | O(n) |
| Insert en tete | O(1) | O(1) | O(n) |
| Insert en queue | O(n) sans pointer | O(1) | O(1) amorti |
| Delete noeud connu | O(1) | O(1) | O(n) |

### Pieges courants

**Piege 1 — Oublier le dummy head**
```python
# MAUVAIS — gestion de cas speciaux
def remove_head_if_X(head):
    if head and head.val == X:
        head = head.next
    # ... puis traiter le reste ...

# BON — dummy head elimine le cas special
def remove_head_if_X(head):
    dummy = ListNode(0, head)
    # ... traiter uniformement ...
    return dummy.next
```

**Piege 2 — Perdre le reste de la liste lors d'un reversal**
```python
# MAUVAIS
curr.next = prev     # On ecrase le lien vers le reste !
curr = curr.next     # curr = prev -> bug

# BON — sauvegarder next AVANT de modifier
nxt = curr.next      # Sauvegarde
curr.next = prev     # Safe
curr = nxt           # Safe
```

**Piege 3 — Not checking fast.next before fast.next.next**
```python
# MAUVAIS — crash si fast.next est None
while fast:
    slow = slow.next
    fast = fast.next.next    # NoneType error

# BON — verifie les deux conditions
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next    # Safe
```

**Piege 4 — Creer un cycle accidentellement dans le reversal recursif**
```python
# MAUVAIS — head.next n'est pas reinitialise
def reverse(head):
    if not head or not head.next:
        return head
    new_head = reverse(head.next)
    head.next.next = head
    # Oublie: head.next = None -> cycle !
    return new_head

# BON
def reverse(head):
    if not head or not head.next:
        return head
    new_head = reverse(head.next)
    head.next.next = head
    head.next = None             # Break the forward link
    return new_head
```

---

## 9. Flash Cards — Revision espacee

**Q1** : Dans le pattern fast/slow, pourquoi la condition `while fast and fast.next` et pas juste `while fast` ?
> **R1** : Parce qu'on fait `fast = fast.next.next`. Si `fast.next` est `None`, alors `fast.next.next` leve une AttributeError. Les deux conditions garantissent qu'on peut avancer de 2 steps sans crash.

**Q2** : Apres un reversal iteratif, que retourne-t-on ? `head` ou `prev` ? Pourquoi ?
> **R2** : On retourne `prev`. A la fin de la boucle, `curr` est `None` (on a parcouru toute la liste) et `prev` pointe vers l'ancien dernier noeud, qui est maintenant le nouveau premier. `head` pointe toujours vers l'ancien premier noeud, qui est maintenant le dernier.

**Q3** : Pourquoi utiliser un dummy head (sentinel) ?
> **R3** : Pour eliminer les cas speciaux du traitement du premier noeud. Avec un dummy, il y a toujours un "noeud precedent" disponible, donc on peut ecrire un code uniforme. Tres utile pour merge, remove, insert au debut.

**Q4** : Dans Floyd's cycle detection, apres avoir trouve le meeting point, pourquoi redemarrer slow depuis head a vitesse 1 ?
> **R4** : Mathematique : soit `a` = distance head -> debut cycle, `b` = distance debut cycle -> meeting point. On prouve que `a = k*cycle - b` = distance depuis meeting point jusqu'au debut du cycle (modulo cycle). Donc si slow repart de head et fast du meeting point, tous deux a vitesse 1, ils se rencontrent au debut du cycle apres `a` steps.

**Q5** : Pourquoi "merge two sorted lists" est-il O(1) en espace alors qu'on construit une liste fusionnee ?
> **R5** : On ne cree PAS de nouveaux noeuds. On re-tisse les pointeurs existants. Le dummy head est UN seul noeud supplementaire, donc O(1) espace. La liste resultante reutilise les memes noeuds que les inputs.

---

## Resume — Key Takeaways

1. **3 patterns couvrent 95%** : fast/slow, reversal, merge
2. **Dummy head** = l'astuce qui simplifie TOUT (merge, remove, insert)
3. **Fast/slow** = milieu, cycle, nth from end — TOUT en O(1) espace
4. **Reversal** = sauvegarde next avant d'inverser, retourne `prev` a la fin
5. **Merge** = dummy head + tail pointer + `or` pour attacher le reste
6. **Floyd's algorithm** = detection de cycle en O(n) temps, O(1) espace
7. **Eviter les listes en production, mais les maitriser en entretien** — c'est un test de rigueur sur les pointeurs

---

## Pour aller plus loin

Ressources canoniques sur les listes chainees :

- **NeetCode — Linked List roadmap** — 11 problemes phares (Reverse, Merge Two Sorted, Reorder, Detect Cycle, LRU Cache) avec videos qui mettent en scene les pointeurs. https://neetcode.io/roadmap
- **Cracking the Coding Interview** (Gayle Laakmann McDowell, 6th ed) — Ch 2 : 8 problemes types entretien sur les linked lists avec discussion fast/slow runner et dummy node. https://www.crackingthecodinginterview.com/
- **CLRS — Introduction to Algorithms** (4th ed, MIT Press 2022) — Ch 10.2 : implementation rigoureuse des doubly linked lists avec sentinelles, base pour comprendre OrderedDict / LRU. https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
- **MIT 6.006 — Introduction to Algorithms** (Erik Demaine, MIT OCW Spring 2020) — Lec. 2 (Data Structures, Dynamic Arrays) : compare arrays vs linked lists en amortise. https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/
