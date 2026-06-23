# Jour 4 — Stacks & Queues : LIFO, FIFO, Monotonic Stack & BFS Foundations

> **Temps estime** : 60 min de lecture active | **Objectif** : maitriser stack et queue comme structures de controle pour resoudre parsing, expressions imbriquees, next-greater, et poser les bases du BFS.

---

## 1. Pourquoi stacks et queues meritent un jour complet

Tu connais deja `list` en Python. Tu pourrais te dire "stack et queue = juste des listes". C'est une erreur classique d'entretien.

**Ce qui compte, ce n'est pas la structure de donnees — c'est le PATTERN de controle qu'elle te debloque :**

| Pattern | Structure | Utilite |
|---------|-----------|---------|
| **LIFO** (Last In First Out) | Stack | Backtracking, expressions imbriquees, parsing, fonctions recursives converties en iteratif |
| **FIFO** (First In First Out) | Queue | BFS, traitement par niveau, scheduling, streaming |
| **Monotonic** (pile avec invariant d'ordre) | Stack | Next/previous greater/smaller, histogramme, sliding window max |
| **Double-ended** (Deque) | Deque | Sliding window, palindromes, LRU, BFS 0-1 |

**La regle d'or** : si un probleme implique du **matching de paires** (ouvrant/fermant, push/pop, encode/decode), c'est presque toujours une stack. Si un probleme implique **parcourir un graphe/arbre niveau par niveau**, c'est presque toujours une queue.

---

## 2. Stack — Implementations en Python

### Option 1 : `list` (recommande par defaut)

```python
stack = []
stack.append(1)     # push — O(1) amorti
stack.append(2)
top = stack[-1]     # peek — O(1)
x = stack.pop()     # pop — O(1) amorti
is_empty = not stack
size = len(stack)
```

**Pourquoi `list` plutot qu'une classe Stack custom ?** Parce que CPython optimise `list.append` et `list.pop` (fin de liste) en O(1) amorti. En entretien, ecrire `stack = []` est idiomatique, rapide, et l'interviewer apprecie.

**Attention** : `list.pop(0)` (retirer en TETE) est **O(n)** — ne jamais utiliser pour une queue.

### Option 2 : `collections.deque` (pour stack + queue)

```python
from collections import deque
stack = deque()
stack.append(1)     # push right — O(1)
stack.pop()         # pop right — O(1)
```

`deque` est double-ended : tu peux l'utiliser en stack OU en queue. Mais pour une stack pure, `list` suffit et evite l'import.

### Exemple canonique — Valid Parentheses

```python
def is_valid(s: str) -> bool:
    """Check if brackets are properly nested."""
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for c in s:
        if c in "([{":
            stack.append(c)                    # Push opener
        elif c in ")]}":
            if not stack or stack.pop() != pairs[c]:
                return False                   # Mismatch or empty stack
    return not stack                           # All openers must be matched
# Time: O(n), Space: O(n)
```

**Pourquoi une stack ici ?** Parce que le dernier opener vu doit etre le **premier** ferme — c'est exactement LIFO.

---

## 3. Queue — Implementations en Python

### Option 1 : `collections.deque` (recommande)

```python
from collections import deque
queue = deque()
queue.append(1)       # enqueue right — O(1)
queue.append(2)
front = queue[0]      # peek left — O(1)
x = queue.popleft()   # dequeue left — O(1)
```

**NE JAMAIS utiliser `list.pop(0)` comme queue** — c'est O(n) parce que Python doit decaler tous les elements. Sur 10^5 elements, ca passe de 0.01s a 5s.

### Option 2 : `queue.Queue` (uniquement pour le multi-threading)

```python
from queue import Queue
q = Queue()
q.put(1)
q.get()
# N'utilise JAMAIS en entretien algorithmique — c'est thread-safe donc LENT.
```

### Exemple canonique — BFS sur une grille

```python
from collections import deque

def bfs_shortest_path(zone, start, target):
    """Find shortest path from start to target in a 0/1 zone (0 = free, 1 = wall)."""
    rows, cols = len(zone), len(zone[0])
    queue = deque([(start, 0)])           # (position, distance)
    visited = {start}

    while queue:
        (r, c), dist = queue.popleft()    # FIFO: always explore oldest first
        if (r, c) == target:
            return dist

        # Explore 4 neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and zone[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))

    return -1
# Time: O(rows * cols), Space: O(rows * cols)
```

**Pourquoi une queue ici ?** Parce que le BFS explore par niveau (distance croissante). La premiere fois qu'on visite la cible, c'est forcement le plus court chemin — **garanti par la propriete FIFO**.

---

## 4. Pattern 1 — Monotonic Stack

### Concept

Une **monotonic stack** est une stack qui maintient un ordre (croissant ou decroissant) parmi ses elements. Quand un nouvel element viole l'ordre, on **pop** jusqu'a restaurer l'invariant.

```python
# Template : monotonic DECREASING stack (pour "next greater")
def next_greater(arr):
    n = len(arr)
    result = [-1] * n            # Default: no greater element
    stack = []                    # Stores INDICES, not values

    for i, val in enumerate(arr):
        # While current value BREAKS the decreasing invariant
        while stack and arr[stack[-1]] < val:
            j = stack.pop()       # j's next greater is val
            result[j] = val
        stack.append(i)

    return result
```

### Quand l'utiliser

- Le probleme demande le "**next/previous greater/smaller** element"
- On cherche la **plus grande surface sous un histogramme**
- On doit calculer pour chaque element la **distance jusqu'au prochain X**
- Les mots "next greater", "daily temperatures", "largest rectangle" apparaissent

### Exemple 1 — Daily Temperatures

```python
def daily_temperatures(temps):
    """For each day, how many days until a warmer temperature?"""
    n = len(temps)
    result = [0] * n
    stack = []     # indices of days waiting for a warmer day

    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            result[j] = i - j      # Distance in days
        stack.append(i)

    return result
# Time: O(n) — each index is pushed and popped at most once
# Space: O(n)
```

**Pourquoi O(n) malgre la boucle while imbriquee ?** Parce que chaque element est pousse une fois et pope une fois sur toute la duree de l'algo. Le total de travail est donc borne a 2n operations.

### Exemple 2 — Largest Rectangle in Histogram

```python
def largest_rectangle(heights):
    """Find the largest rectangle that fits under the histogram."""
    stack = []       # Indices with INCREASING heights
    best = 0
    heights = heights + [0]    # Sentinel to flush the stack at the end

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            top = stack.pop()
            # Width extends from previous smaller (or 0) to current index
            left = stack[-1] if stack else -1
            width = i - left - 1
            best = max(best, heights[top] * width)
        stack.append(i)

    return best
# Time: O(n), Space: O(n)
```

**Insight cle** : pour chaque barre, on veut le **plus grand rectangle qui a cette barre comme plus petite hauteur**. La monotonic stack nous donne en O(1) les bords gauche et droit ou une barre plus petite apparait.

---

## 5. Pattern 2 — Parentheses & Matching

### Concept

Tout probleme de **matching de paires** (parentheses, balises HTML, call stack) se resout avec une stack. L'invariant : un symbole fermant doit correspondre au dernier ouvrant non ferme.

### Exemple 1 — Valid Parentheses (revisite)

Voir section 2. Template : push les ouvrants, pop et verifie a chaque fermant.

### Exemple 2 — Minimum Remove to Make Valid Parentheses

```python
def min_remove_to_make_valid(s: str) -> str:
    """Remove minimum parens to make s valid. Keep letters."""
    stack = []                    # Indices of unmatched '('
    to_remove = set()

    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                stack.pop()       # Matched
            else:
                to_remove.add(i)  # Unmatched ')'

    # After the loop, any remaining '(' in stack is unmatched
    to_remove.update(stack)

    return ''.join(c for i, c in enumerate(s) if i not in to_remove)
# Time: O(n), Space: O(n)
```

### Exemple 3 — Evaluate Reverse Polish Notation

```python
def eval_rpn(tokens):
    """Evaluate an expression in Reverse Polish Notation."""
    stack = []
    ops = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b),   # Truncate toward zero (NOT floor)
    }

    for tok in tokens:
        if tok in ops:
            b = stack.pop()              # Careful: order matters
            a = stack.pop()
            stack.append(ops[tok](a, b))
        else:
            stack.append(int(tok))

    return stack[0]
# Time: O(n), Space: O(n)
```

**Piege classique** : la division en Python utilise `//` qui **floor** (arrondi vers le bas, `-7 // 2 == -4`). Pour RPN, on veut la **troncature vers zero** (`int(-7 / 2) == -3`).

---

## 6. Pattern 3 — Queue with Two Stacks (et vice-versa)

### Concept

Classique d'entretien : on te donne UNE structure et on te demande de simuler l'autre. C'est un test de comprehension profonde des invariants.

### Implementer une Queue avec deux Stacks

```python
class MyQueue:
    """FIFO queue implemented with two LIFO stacks."""
    def __init__(self):
        self.in_stack = []     # Push goes here
        self.out_stack = []    # Pop/peek come from here

    def push(self, x):
        self.in_stack.append(x)               # O(1)

    def pop(self):
        self._ensure_out()
        return self.out_stack.pop()           # Amortized O(1)

    def peek(self):
        self._ensure_out()
        return self.out_stack[-1]

    def empty(self):
        return not self.in_stack and not self.out_stack

    def _ensure_out(self):
        if not self.out_stack:
            # Transfer all: order is REVERSED, so in_stack's bottom becomes out_stack's top
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
```

**Complexite amortie** : chaque element est pousse une fois sur `in_stack`, transfere une fois a `out_stack`, et pope une fois. Donc O(1) amorti par operation, meme si un pop individuel peut couter O(n) si `out_stack` est vide.

---

## 7. Pattern 4 — BFS Foundations

### Concept

BFS = **Breadth First Search** = on explore tous les voisins a distance 1, puis tous a distance 2, etc. C'est la seule facon de garantir le **plus court chemin en nombre d'aretes** dans un graphe non pondere.

```python
# Template BFS generique
from collections import deque

def bfs(start, is_goal, neighbors):
    """
    Generic BFS:
    - start: initial node
    - is_goal: function(node) -> bool
    - neighbors: function(node) -> iterable of neighbors
    """
    queue = deque([(start, 0)])          # (node, distance)
    visited = {start}

    while queue:
        node, dist = queue.popleft()
        if is_goal(node):
            return dist

        for nxt in neighbors(node):
            if nxt not in visited:
                visited.add(nxt)          # Mark AS SOON AS enqueued, not when dequeued
                queue.append((nxt, dist + 1))

    return -1                              # No path found
```

**Regle critique** : marquer `visited` **au moment de l'enqueue**, pas au moment du dequeue. Sinon tu peux enfiler plusieurs fois le meme noeud et exploser la complexite.

### Exemple — Rotting Oranges (BFS multi-source)

```python
def oranges_rotting(zone):
    """Return min minutes until all fresh oranges rot. -1 if impossible."""
    rows, cols = len(zone), len(zone[0])
    queue = deque()
    fresh = 0

    # Phase 1: enqueue ALL initial rotten oranges (multi-source BFS)
    for r in range(rows):
        for c in range(cols):
            if zone[r][c] == 2:
                queue.append((r, c, 0))
            elif zone[r][c] == 1:
                fresh += 1

    # Phase 2: BFS
    max_time = 0
    while queue:
        r, c, t = queue.popleft()
        max_time = max(max_time, t)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and zone[nr][nc] == 1:
                zone[nr][nc] = 2               # Mark as rotten
                fresh -= 1
                queue.append((nr, nc, t + 1))

    return max_time if fresh == 0 else -1
# Time: O(rows * cols), Space: O(rows * cols)
```

**Insight** : BFS **multi-source** initialise la queue avec TOUS les points de depart simultanement. La distance retournee est la distance minimale depuis N'IMPORTE lequel des points de depart.

---

## 8. Decision Tree — Quel pattern utiliser ?

```
Le probleme implique du matching, parsing, ou parcours par niveau ?
|
├── Matching paire (ouvrant/fermant, pattern bien imbrique) ?
│   └── STACK de parentheses / matching
│       Ex: valid parentheses, min remove to valid, decode string
|
├── Parsing d'expression (RPN, calculator, basic calc) ?
│   └── STACK d'operandes / operateurs
│       Ex: eval RPN, basic calculator, simplify path
|
├── "Next/previous greater/smaller" element ?
│   └── MONOTONIC STACK
│       Ex: daily temperatures, next greater, largest rectangle
|
├── Parcours d'arbre/graphe par niveau OU plus court chemin non pondere ?
│   └── BFS avec QUEUE (deque)
│       Ex: level order traversal, shortest path in zone, rotting oranges
|
├── On te demande de convertir une recursion en iteratif ?
│   └── STACK explicite (simule le call stack)
│       Ex: DFS iteratif, traversal in-order iteratif
|
└── On te demande de simuler une structure avec une autre ?
    └── Queue via 2 stacks / Stack via 2 queues
        Ex: MyQueue, MyStack
```

**Raccourcis mentaux** :

| Signal dans l'enonce | Pattern |
|---------------------|---------|
| "valid parentheses", "balanced" | Stack matching |
| "next greater", "next warmer day" | Monotonic stack |
| "largest rectangle", "trapping rain water" | Monotonic stack |
| "level order", "by level" | BFS avec queue |
| "shortest path" (non pondere) | BFS avec queue |
| "evaluate expression" | Stack (RPN ou operateur) |
| "decode string" (imbrique) | Stack |

---

## 9. Complexites et pieges

### Complexites

| Operation | list (stack) | deque (queue) | Commentaire |
|-----------|--------------|---------------|-------------|
| Push/append a droite | O(1) amorti | O(1) | |
| Pop a droite | O(1) amorti | O(1) | |
| Pop a gauche | **O(n)** | **O(1)** | Ne jamais utiliser `list.pop(0)` |
| Peek | O(1) | O(1) | `s[-1]` ou `q[0]` |
| `in` (membership) | O(n) | O(n) | Les deux sont lineaires |
| Acces par index | O(1) | O(n) | deque n'est PAS array-like |

### Pieges courants

**Piege 1 — Utiliser `list.pop(0)` comme queue**
```python
# MAUVAIS — O(n) par operation
queue = []
queue.append(1)
queue.pop(0)    # O(n) !

# BON — O(1) par operation
from collections import deque
queue = deque()
queue.append(1)
queue.popleft()  # O(1)
```

**Piege 2 — Oublier de marquer visited a l'enqueue en BFS**
```python
# MAUVAIS — un noeud peut etre enqueue plusieurs fois
while queue:
    node = queue.popleft()
    if node in visited: continue
    visited.add(node)   # <-- trop tard
    for nxt in neighbors(node):
        queue.append(nxt)

# BON — marque visited AVANT l'enqueue
visited = {start}
while queue:
    node = queue.popleft()
    for nxt in neighbors(node):
        if nxt not in visited:
            visited.add(nxt)     # <-- avant l'enqueue
            queue.append(nxt)
```

**Piege 3 — Monotonic stack : stocker indices vs valeurs**
```python
# En general, STOCKE LES INDICES. Ca permet de calculer des distances
# et de retrouver la valeur avec arr[i].

stack = []   # indices
for i, v in enumerate(arr):
    while stack and arr[stack[-1]] < v:   # <- comparer par valeur
        j = stack.pop()
        # ... utiliser j (indice) pour distance, et arr[j] pour valeur
    stack.append(i)
```

**Piege 4 — Division en RPN**
```python
# Le probleme RPN veut la TRONCATURE VERS ZERO, pas le FLOOR
print(-7 // 2)       # -4  (floor — Python default)
print(int(-7 / 2))   # -3  (truncation toward zero — ce qu'on veut)
```

---

## 10. Flash Cards — Revision espacee

> **Methode** : couvrir la reponse, repondre a voix haute, puis verifier. Revenir dans 1 jour, 3 jours, 7 jours.

**Q1** : Pourquoi utiliser `collections.deque` et pas `list` pour une queue ?
> **R1** : `list.pop(0)` est O(n) parce que Python doit decaler tous les elements vers la gauche. `deque.popleft()` est O(1) parce que deque est une doubly linked list de blocks. Sur 10^5 elements, la difference est 500x.

**Q2** : Explique pourquoi le monotonic stack donne O(n) malgre la boucle while imbriquee.
> **R2** : Argument amorti. Chaque element est pousse exactement une fois et pope au plus une fois sur toute la duree de l'algo. Le cout total du while cumule sur toutes les iterations du for est donc borne a n. Total = 2n operations = O(n).

**Q3** : Dans un BFS, pourquoi marquer `visited` au moment de l'enqueue et pas au moment du dequeue ?
> **R3** : Sinon un meme noeud peut etre ajoute plusieurs fois a la queue avant d'etre traite, ce qui peut exploser la memoire et augmenter la complexite a O(n^2). En marquant a l'enqueue, on garantit qu'un noeud n'est ajoute qu'une seule fois.

**Q4** : Pour "next greater element", quel type de monotonic stack (croissante ou decroissante) ?
> **R4** : Decroissante. Tant que le top de la stack est **plus petit** que l'element courant, on pop et on dit "l'element courant est ton next greater". Donc les elements encore dans la stack sont en ordre decroissant depuis le bas.

**Q5** : Comment implementer une queue avec deux stacks en O(1) amorti ?
> **R5** : Une stack pour les pushs (`in_stack`), une pour les pops (`out_stack`). Push = append sur in_stack. Pop = si out_stack vide, transferer tout in_stack en le vidant (ca inverse l'ordre), puis pop out_stack. Chaque element est transfere au plus une fois, donc O(1) amorti.

---

## Resume — Key Takeaways

1. **Stack = LIFO**, utilise pour matching, parsing, backtracking, conversion de recursion en iteratif
2. **Queue = FIFO**, utilise pour BFS, traitement par niveau, plus court chemin non pondere
3. **`list` pour stack**, **`deque` pour queue** — JAMAIS `list.pop(0)` (O(n))
4. **Monotonic stack** : resout tous les problemes "next greater/smaller" en O(n)
5. **BFS** : marque visited a l'enqueue, pas au dequeue
6. **Matching de paires** → stack ; **parcours par niveau** → queue
7. **Pieges** : division tronquee en RPN, indices vs valeurs en monotonic stack, visited mal place

---

## Pour aller plus loin

Ressources canoniques sur les piles et files :

- **CLRS — Introduction to Algorithms** (4th ed, MIT Press 2022) — Ch 10 (Elementary Data Structures) : implementation rigoureuse de stacks, queues, deques avec invariants. https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
- **Princeton Algorithms Part 1** (Sedgewick & Wayne, Coursera, gratuit) — Week 2 (Stacks and Queues) : implementations linked-list et resizing array, analyse amortie. https://www.coursera.org/learn/algorithms-part1
- **NeetCode — Stack roadmap** — 7 problemes phares (Valid Parentheses, Min Stack, Daily Temperatures, Largest Rectangle) avec patterns monotonic stack. https://neetcode.io/roadmap
- **Algorithms 4th ed companion site** (Sedgewick & Wayne) — Ch 1.3 : code Java reference + exercices pour stacks/queues. https://algs4.cs.princeton.edu/
