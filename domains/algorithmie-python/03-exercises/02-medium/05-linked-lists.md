# Exercices Medium — Linked Lists

---

## Exercice 4 : Fast & Slow — Linked List Cycle II (debut du cycle)

### Objectif

Aller au-dela de la simple detection de cycle : trouver le **noeud ou le cycle commence**, en O(1) espace. C'est la phase 2 de l'algorithme de Floyd, que beaucoup de candidats connaissent de nom sans savoir la justifier.

### Consigne

Etant donne la tete d'une linked list, retourne le **noeud ou le cycle commence**. S'il n'y a pas de cycle, retourne `None`.

**Contrainte : O(1) espace.** La solution avec un `set` de noeuds visites (O(n) espace) ne valide pas l'exercice — ecris-la quand meme en premier comme oracle.

```python
def detect_cycle(head: ListNode | None) -> ListNode | None:
    """
    Return the node where the cycle begins, or None if there is no cycle.
    Must use O(1) extra space.
    """
    pass
```

**Indice (phase 2 de Floyd)** : apres la rencontre fast/slow, remets un pointeur a `head` et avance les deux **a la meme vitesse**. Ils se rencontrent au debut du cycle. Sache expliquer pourquoi : si `a` = distance head→debut du cycle et `b` = distance debut du cycle→point de rencontre, alors fast a parcouru `a + b + k*L` et slow `a + b`, donc `a ≡ L - b (mod L)`.

### Tests

```python
# Build: 3 -> 2 -> 0 -> -4 -+
#             ^-------------+
n1, n2, n3, n4 = ListNode(3), ListNode(2), ListNode(0), ListNode(-4)
n1.next, n2.next, n3.next, n4.next = n2, n3, n4, n2
assert detect_cycle(n1) is n2

# Self-loop: 1 -> 2 -+
#                 ^--+
a, b = ListNode(1), ListNode(2)
a.next, b.next = b, b
assert detect_cycle(a) is b

# No cycle
assert detect_cycle(build([1, 2, 3])) is None
assert detect_cycle(None) is None
assert detect_cycle(ListNode(1)) is None

# Cycle on the head itself
c = ListNode(1)
c.next = c
assert detect_cycle(c) is c
```

### Criteres de reussite

- [ ] Version oracle avec `set` ecrite d'abord (O(n) espace) et comparee a Floyd
- [ ] Version Floyd : phase 1 (rencontre) puis phase 2 (reset a head, meme vitesse)
- [ ] O(n) temps, O(1) espace — aucun set, aucun dict
- [ ] Comparaison par **identite** de noeud (`is`), pas par valeur
- [ ] Tu peux expliquer la preuve mathematique de la phase 2
- [ ] Tous les tests passent (self-loop, cycle sur head, pas de cycle)

---

## Exercice 5 : Two Pointers with Gap — Remove Nth Node From End

### Objectif

Supprimer le n-ieme noeud depuis la fin **en une seule passe**, avec la technique du gap et un dummy head. Le piege cible : la suppression de la tete elle-meme.

### Consigne

Etant donne la tete d'une linked list, supprime le `n`-ieme noeud **en partant de la fin** et retourne la tete. `n` est garanti valide (1 <= n <= longueur).

**Contrainte : une seule passe** (pas le droit de compter la longueur d'abord).

```python
def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
    """
    Remove the nth node from the end in ONE pass. Return the new head.
    """
    pass
```

**Indice** : avance un pointeur `lead` de `n + 1` pas depuis un **dummy head**, puis avance `lead` et `lag` ensemble jusqu'a ce que `lead` soit `None`. `lag` pointe alors sur le predecesseur du noeud a supprimer.

### Tests

```python
assert to_list(remove_nth_from_end(build([1, 2, 3, 4, 5]), 2)) == [1, 2, 3, 5]
assert to_list(remove_nth_from_end(build([1]), 1)) == []          # Delete the only node
assert to_list(remove_nth_from_end(build([1, 2]), 2)) == [2]      # Delete the HEAD
assert to_list(remove_nth_from_end(build([1, 2]), 1)) == [1]      # Delete the tail
assert to_list(remove_nth_from_end(build([1, 2, 3]), 3)) == [2, 3]
```

### Criteres de reussite

- [ ] Une seule passe sur la liste (pas de calcul de longueur prealable)
- [ ] Dummy head utilise — le cas "supprimer la tete" ne demande aucun code special
- [ ] Le gap entre les deux pointeurs est exact (n+1 depuis le dummy) — pas d'off-by-one
- [ ] O(n) temps, O(1) espace
- [ ] Tous les tests passent, en particulier la suppression de la tete

---

## Exercice 6 : Combo — Reorder List

### Objectif

Combiner les TROIS patterns du jour en un seul probleme : trouver le milieu (fast/slow), inverser la seconde moitie (reversal), puis fusionner en alternance (merge). C'est le test ultime de la maitrise des pointeurs.

### Consigne

Etant donne une liste `L0 -> L1 -> ... -> Ln-1 -> Ln`, reordonne-la **en place** en :

```
L0 -> Ln -> L1 -> Ln-1 -> L2 -> Ln-2 -> ...
```

Tu ne peux pas modifier les valeurs des noeuds — seulement les pointeurs `next`.

**Contrainte : O(1) espace** (pas de copie des noeuds dans une liste Python).

```python
def reorder_list(head: ListNode | None) -> None:
    """
    Reorder the list in place: first, last, second, second-to-last, ...
    Modifies the list, returns nothing.
    """
    pass
```

**Plan d'attaque** :
1. Fast/slow pour trouver le milieu.
2. Couper la liste en deux et inverser la seconde moitie.
3. Fusionner les deux moities en alternance.

### Tests

```python
lst = build([1, 2, 3, 4])
reorder_list(lst)
assert to_list(lst) == [1, 4, 2, 3]

lst = build([1, 2, 3, 4, 5])
reorder_list(lst)
assert to_list(lst) == [1, 5, 2, 4, 3]

lst = build([1])
reorder_list(lst)
assert to_list(lst) == [1]

lst = build([1, 2])
reorder_list(lst)
assert to_list(lst) == [1, 2]

lst = build([1, 2, 3])
reorder_list(lst)
assert to_list(lst) == [1, 3, 2]
```

### Criteres de reussite

- [ ] Les 3 etapes sont identifiables dans le code (milieu, reverse, merge alterne)
- [ ] La premiere moitie est bien **terminee par None** avant le merge (sinon cycle infini dans `to_list`)
- [ ] O(n) temps, O(1) espace — aucune structure auxiliaire
- [ ] Le cas longueur impaire place le noeud du milieu en derniere position
- [ ] Tous les tests passent
