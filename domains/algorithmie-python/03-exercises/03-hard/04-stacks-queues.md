# Exercices Hard — Stacks & Queues

---

## Exercice 7 : Monotonic Deque — Sliding Window Maximum

### Objectif

Maitriser le **monotonic deque** (file double a invariant d'ordre) pour calculer le maximum d'une fenetre glissante en O(n). C'est l'extension hard du monotonic stack — l'un des problemes les plus discriminants en entretien senior.

### Consigne

Etant donne un tableau `nums` et un entier `k`, une fenetre de taille `k` glisse de la gauche vers la droite, un cran a la fois. Retourne la liste des maximums de chaque position de la fenetre.

```python
def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """
    Return the maximum of each sliding window of size k.
    """
    pass
```

**Approche attendue** :
1. Maintiens un `deque` d'**indices** dont les valeurs sont en ordre **decroissant**
2. Avant d'ajouter `i`, pop par la droite tous les indices dont la valeur est <= `nums[i]` (ils ne pourront jamais etre le max)
3. Pop par la gauche l'indice qui sort de la fenetre (`<= i - k`)
4. Le max de la fenetre est toujours `nums[deque[0]]`

### Tests

```python
assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
assert max_sliding_window([1], 1) == [1]
assert max_sliding_window([9, 8, 7, 6], 2) == [9, 8, 7]      # Decreasing array
assert max_sliding_window([1, 2, 3, 4], 2) == [2, 3, 4]      # Increasing array
assert max_sliding_window([4, 4, 4, 4], 2) == [4, 4, 4]      # All equal
assert max_sliding_window([1, 3, 1, 2, 0, 5], 3) == [3, 3, 2, 5]
assert max_sliding_window([7, 2, 4], 2) == [7, 4]
assert max_sliding_window([1, -1], 1) == [1, -1]             # k = 1
```

### Criteres de reussite

- [ ] Utilise un `collections.deque` d'indices avec invariant decroissant
- [ ] Pop par la gauche les indices hors fenetre, par la droite les indices domines
- [ ] Complexite **O(n) temps** (chaque indice entre/sort du deque une fois), O(k) espace
- [ ] N'utilise PAS `max(window)` a chaque cran (O(n*k), trop lent)
- [ ] Gere `k = 1` et les tableaux monotones
- [ ] Tous les tests passent

---

## Exercice 8 : Two Stacks Design — Basic Calculator (avec + - parentheses)

### Objectif

Combiner stack de parsing + gestion du signe pour evaluer une expression arithmetique avec parentheses, sans utiliser `eval`. Ce probleme hard teste la rigueur du parsing manuel.

### Consigne

Implemente un calculateur de base qui evalue une expression string `s` valide contenant : des entiers non negatifs, `+`, `-`, `(`, `)` et des espaces. La division et la multiplication ne sont **pas** au programme de cet exercice.

Tu ne dois **PAS** utiliser `eval` ni `exec`.

```python
def calculate(s: str) -> int:
    """
    Evaluate a math expression with +, -, ( ), spaces and non-negative integers.
    No eval/exec allowed.
    """
    pass
```

**Indice** : parcours la string en gardant un `result`, un `number` en cours, un `sign` (+1/-1) et une **stack**. A `(`, empile `(result, sign)` et reinitialise. A `)`, depile et combine. C'est le pattern "stack de contexte" applique aux signes.

### Tests

```python
assert calculate("1 + 1") == 2
assert calculate(" 2-1 + 2 ") == 3
assert calculate("(1+(4+5+2)-3)+(6+8)") == 23
assert calculate("- (3 + (4 + 5))") == -12
assert calculate("2147483647") == 2147483647     # Single large number
assert calculate("1-(     -2)") == 3              # Negative inside parens
assert calculate("(1)") == 1
assert calculate("10 - (2 + 3) - 1") == 4
assert calculate("0") == 0
```

### Criteres de reussite

- [ ] N'utilise NI `eval` NI `exec`
- [ ] Gere les nombres a plusieurs chiffres et les espaces
- [ ] Gere les parentheses imbriquees et les signes negatifs unaires (`-(...)`)
- [ ] Complexite O(n) temps, O(n) espace (profondeur des parentheses)
- [ ] Utilise une stack pour sauvegarder `(result, sign)` a chaque `(`
- [ ] Tous les tests passent

---

## Exercice 9 : Monotonic Stack — Trapping Rain Water

### Objectif

Resoudre le probleme "trapping rain water" avec un monotonic stack (variante de l'approche two-pointers). On apprend a accumuler l'eau "par couches horizontales" via la stack — un raisonnement geometrique exigeant.

### Consigne

Etant donne un tableau `height` de hauteurs (largeur de chaque barre = 1), calcule la quantite d'eau de pluie piegee entre les barres apres la pluie.

```python
def trap(height: list[int]) -> int:
    """
    Return the total amount of trapped rain water.
    """
    pass
```

**Approche attendue (monotonic stack)** :
- Maintiens un monotonic stack **decroissant** d'indices
- Quand `height[i]` depasse le sommet, on a un "creux" : pop le fond (`bottom`), le `left` devient le nouveau sommet, le `right` est `i`
- L'eau ajoutee = `(min(height[left], height[i]) - height[bottom]) * (i - left - 1)`

### Tests

```python
assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap([4, 2, 0, 3, 2, 5]) == 9
assert trap([]) == 0
assert trap([1, 2, 3]) == 0           # Monotone increasing — no trapping
assert trap([3, 2, 1]) == 0           # Monotone decreasing — no trapping
assert trap([5]) == 0
assert trap([2, 0, 2]) == 2
assert trap([5, 4, 1, 2]) == 1
```

### Criteres de reussite

- [ ] Utilise un monotonic stack d'indices (ou justifie une approche two-pointers O(1) espace)
- [ ] Calcule l'eau par couches horizontales (largeur * hauteur bornee)
- [ ] Complexite O(n) temps, O(n) espace (stack) — ou O(1) espace si two-pointers
- [ ] Gere les tableaux monotones (resultat 0) et les tableaux courts
- [ ] Tous les tests passent
