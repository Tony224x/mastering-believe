# Exercices Medium — Linked Lists

> Toutes les solutions reutilisent la meme classe et les memes helpers :
>
> ```python
> class ListNode:
>     def __init__(self, val=0, next=None):
>         self.val = val
>         self.next = next
>
> def build(values):
>     dummy = ListNode(0)
>     t = dummy
>     for v in values:
>         t.next = ListNode(v)
>         t = t.next
>     return dummy.next
>
> def to_list(head):
>     out = []
>     while head:
>         out.append(head.val)
>         head = head.next
>     return out
> ```

---

## Exercice 4 : Reversal cible — Reverse Linked List II

### Objectif

Inverser un **segment** `[left, right]` (1-indexe) au milieu d'une liste, sans inverser le reste. Ce probleme combine dummy head + reversal local et teste la precision de la manipulation de pointeurs.

### Consigne

Etant donne la tete d'une liste et deux entiers `left` et `right` (1-indexes, `1 <= left <= right <= n`), inverse les noeuds des positions `left` a `right` incluses, et retourne la nouvelle tete.

```python
def reverse_between(head, left: int, right: int):
    """
    Reverse the sublist from position left to right (1-indexed). Return new head.
    """
    pass
```

**Indice** : place-toi sur le noeud juste AVANT `left` (utilise un dummy head pour gerer `left = 1`). Puis applique le "head-insertion trick" : a chaque iteration, deplace le noeud suivant juste apres `prev`. Une seule passe, O(1) espace.

### Tests

```python
assert to_list(reverse_between(build([1, 2, 3, 4, 5]), 2, 4)) == [1, 4, 3, 2, 5]
assert to_list(reverse_between(build([5]), 1, 1)) == [5]
assert to_list(reverse_between(build([1, 2]), 1, 2)) == [2, 1]
assert to_list(reverse_between(build([1, 2, 3, 4, 5]), 1, 5)) == [5, 4, 3, 2, 1]
assert to_list(reverse_between(build([1, 2, 3, 4, 5]), 3, 3)) == [1, 2, 3, 4, 5]  # left==right
assert to_list(reverse_between(build([3, 5]), 1, 1)) == [3, 5]
```

### Criteres de reussite

- [ ] Utilise un dummy head pour gerer le cas `left = 1`
- [ ] Une seule passe, O(n) temps, O(1) espace
- [ ] Gere `left == right` (aucune modification)
- [ ] Gere l'inversion totale (`left = 1`, `right = n`)
- [ ] Tous les tests passent

---

## Exercice 5 : Arithmetique sur listes — Add Two Numbers

### Objectif

Simuler une addition chiffre par chiffre avec gestion de la retenue (carry). C'est un probleme de "merge avec etat" : on parcourt deux listes simultanement en maintenant un invariant (le carry).

### Consigne

On te donne deux listes non vides representant deux entiers positifs. Les chiffres sont stockes en **ordre inverse** (le chiffre des unites en premier) et chaque noeud contient un seul chiffre. Additionne les deux nombres et retourne la somme sous forme de liste chainee, egalement en ordre inverse.

```python
def add_two_numbers(l1, l2):
    """
    l1 and l2 store a non-negative integer in reverse order (units first).
    Return their sum as a linked list, also in reverse order.
    """
    pass
```

**Indice** : utilise un dummy head et un `carry`. A chaque etape, additionne `(l1.val + l2.val + carry)`, le nouveau chiffre est `total % 10`, le nouveau carry est `total // 10`. Continue tant qu'il reste un noeud OU un carry.

### Tests

```python
assert to_list(add_two_numbers(build([2, 4, 3]), build([5, 6, 4]))) == [7, 0, 8]   # 342 + 465 = 807
assert to_list(add_two_numbers(build([0]), build([0]))) == [0]
assert to_list(add_two_numbers(build([9, 9, 9, 9, 9, 9, 9]), build([9, 9, 9, 9]))) == [8, 9, 9, 9, 0, 0, 0, 1]
assert to_list(add_two_numbers(build([5]), build([5]))) == [0, 1]                   # 5 + 5 = 10
assert to_list(add_two_numbers(build([1, 8]), build([0]))) == [1, 8]
assert to_list(add_two_numbers(build([9, 9]), build([1]))) == [0, 0, 1]             # 99 + 1 = 100
```

### Criteres de reussite

- [ ] Utilise un dummy head + une variable `carry`
- [ ] La boucle continue tant que `l1` OU `l2` OU `carry` est non nul
- [ ] Gere les longueurs differentes des deux listes
- [ ] Gere le carry final qui cree un nouveau noeud de tete
- [ ] Complexite O(max(n, m)) temps, O(max(n, m)) espace (liste resultat)
- [ ] Tous les tests passent

---

## Exercice 6 : Reorganisation — Odd Even Linked List

### Objectif

Reorganiser une liste en place en separant les noeuds d'indices impairs et pairs, puis en les rattachant. Ce probleme teste la gestion simultanee de deux sous-listes via leurs queues.

### Consigne

Etant donne la tete d'une liste, regroupe d'abord tous les noeuds d'index **impair** (1er, 3e, 5e... en 1-indexation) puis tous les noeuds d'index **pair**, en preservant l'ordre relatif dans chaque groupe. Fais-le en O(1) espace et O(n) temps.

> Note : on parle de la **position** du noeud dans la liste (1, 2, 3...), pas de la valeur.

```python
def odd_even_list(head):
    """
    Group all odd-indexed nodes first, then even-indexed nodes (1-indexed by
    position). In-place, O(1) space.
    """
    pass
```

**Indice** : maintiens deux pointeurs `odd` et `even`, plus une reference `even_head` vers le debut de la sous-liste paire. Avance par sauts de deux. A la fin, rattache `odd.next = even_head`.

### Tests

```python
assert to_list(odd_even_list(build([1, 2, 3, 4, 5]))) == [1, 3, 5, 2, 4]
assert to_list(odd_even_list(build([2, 1, 3, 5, 6, 4, 7]))) == [2, 3, 6, 7, 1, 5, 4]
assert to_list(odd_even_list(build([1]))) == [1]
assert to_list(odd_even_list(build([1, 2]))) == [1, 2]
assert to_list(odd_even_list(build([]))) == []
assert to_list(odd_even_list(build([1, 2, 3, 4]))) == [1, 3, 2, 4]
```

### Criteres de reussite

- [ ] Maintient `odd`, `even` et `even_head`
- [ ] Rattache la queue impaire au debut de la sous-liste paire a la fin
- [ ] Complexite O(n) temps, O(1) espace (aucune allocation de noeud)
- [ ] Gere les listes de taille 0, 1 et 2
- [ ] Tous les tests passent
