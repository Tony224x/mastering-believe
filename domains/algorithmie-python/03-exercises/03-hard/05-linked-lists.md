# Exercices Hard — Linked Lists

---

## Exercice 7 : Heap + Merge — Merge K Sorted Lists

### Objectif

Generaliser le merge de 2 listes a k listes avec un min-heap, et comprendre pourquoi la complexite passe de O(N*k) (merge naif un par un) a O(N log k).

### Consigne

Etant donne un tableau de `k` linked lists, chacune triee par ordre croissant, fusionne-les en **une seule liste triee** et retourne sa tete.

**Contrainte de complexite : O(N log k)** ou N = nombre total de noeuds. Les approches suivantes sont a analyser mais ne valident pas :
- Concatener tout + `sorted()` : O(N log N)
- Merger les listes une par une : O(N * k)

```python
def merge_k_lists(lists: list[ListNode | None]) -> ListNode | None:
    """
    Merge k sorted linked lists into one sorted list.
    Must run in O(N log k).
    """
    pass
```

**Pieges** :
- `heapq` ne sait pas comparer deux `ListNode` : pousse des tuples `(val, tie_breaker, node)` avec un compteur croissant comme tie-breaker (sinon `TypeError` sur valeurs egales).
- Les listes vides (`None`) dans le tableau d'entree doivent etre ignorees.

**Bonus** : implemente aussi la version "divide & conquer" (merger les listes par paires, comme un tournoi) — meme complexite O(N log k), zero heap.

### Tests

```python
result = merge_k_lists([build([1, 4, 5]), build([1, 3, 4]), build([2, 6])])
assert to_list(result) == [1, 1, 2, 3, 4, 4, 5, 6]

assert merge_k_lists([]) is None
assert merge_k_lists([None]) is None
assert to_list(merge_k_lists([None, build([1]), None])) == [1]
assert to_list(merge_k_lists([build([5]), build([1]), build([3])])) == [1, 3, 5]

# Duplicate values across lists — the tie-breaker must prevent a TypeError
result = merge_k_lists([build([2, 2]), build([2, 2]), build([2])])
assert to_list(result) == [2, 2, 2, 2, 2]
```

### Criteres de reussite

- [ ] Min-heap de taille au plus k : push la tete de chaque liste, pop le min, push son successeur
- [ ] Tie-breaker dans les tuples du heap — pas de `TypeError` sur valeurs dupliquees
- [ ] O(N log k) temps, O(k) espace pour le heap
- [ ] Les `None` dans l'input sont geres (liste vide, tableau vide)
- [ ] Tu peux expliquer pourquoi le merge un-par-un est O(N*k) : les premiers noeuds sont re-parcourus a chaque merge
- [ ] Tous les tests passent

---

## Exercice 8 : Merge Sort — Sort List en O(n log n) et O(1) espace auxiliaire

### Objectif

Trier une linked list SANS la convertir en tableau : merge sort bottom-up ou top-down sur les pointeurs. Cible la faiblesse classique : savoir trier un tableau mais etre perdu des qu'il n'y a plus d'indices.

### Consigne

Etant donne la tete d'une linked list, retourne la liste **triee par ordre croissant**.

**Contraintes** :
- O(n log n) temps garanti (pas de quicksort sur liste — pivot degenere).
- Interdiction de copier les valeurs dans une liste Python pour la trier (`sorted()` interdit sur les valeurs). On manipule **uniquement les pointeurs**.
- Version top-down acceptee : O(log n) espace de pile. Bonus : version bottom-up iterative en O(1) espace.

```python
def sort_list(head: ListNode | None) -> ListNode | None:
    """
    Sort the linked list in O(n log n) time by manipulating pointers only.
    """
    pass
```

**Plan top-down** :
1. Couper la liste en deux au milieu (fast/slow — attention : `fast = head.next` pour que slow s'arrete AVANT le milieu, sinon recursion infinie sur 2 elements).
2. Trier recursivement chaque moitie.
3. Merger les deux moities triees (exercice easy 3 reutilise tel quel).

**Benchmark impose** : genere des listes aleatoires de n = 1000, 2000, 4000, 8000 noeuds, mesure le temps, et verifie que le ratio temps(2n)/temps(n) reste proche de ~2.2 (scaling n log n), pas de ~4 (quadratique).

### Tests

```python
assert to_list(sort_list(build([4, 2, 1, 3]))) == [1, 2, 3, 4]
assert to_list(sort_list(build([-1, 5, 3, 4, 0]))) == [-1, 0, 3, 4, 5]
assert to_list(sort_list(build([]))) == []
assert to_list(sort_list(build([1]))) == [1]
assert to_list(sort_list(build([2, 1]))) == [1, 2]
assert to_list(sort_list(build([1, 1, 1]))) == [1, 1, 1]
assert to_list(sort_list(build([5, 4, 3, 2, 1]))) == [1, 2, 3, 4, 5]   # Reverse sorted

# Random cross-check against sorted() as oracle
import random
for _ in range(20):
    values = [random.randint(-100, 100) for _ in range(random.randint(0, 50))]
    assert to_list(sort_list(build(values))) == sorted(values)
```

### Criteres de reussite

- [ ] Split au milieu avec fast/slow — le cas 2 elements ne boucle pas a l'infini
- [ ] Le merge reutilise le pattern de l'exercice easy 3 (dummy head)
- [ ] Aucune conversion en tableau — uniquement des manipulations de pointeurs
- [ ] O(n log n) temps confirme par le benchmark (ratio ~2.2x quand n double)
- [ ] Oracle `sorted()` valide la correction sur des inputs aleatoires
- [ ] Tous les tests passent (liste vide, doublons, deja triee a l'envers)
