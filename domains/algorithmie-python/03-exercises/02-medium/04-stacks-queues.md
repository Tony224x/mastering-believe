# Exercices Medium — Stacks & Queues

---

## Exercice 4 : Monotonic Stack — Daily Temperatures

### Objectif

Passer du monotonic stack de VALEURS (Next Greater Element, exercice easy) au monotonic stack d'INDICES — necessaire des qu'on doit retourner une distance et non une valeur.

### Consigne

Etant donne un tableau `temperatures` de temperatures journalieres, retourne un tableau `answer` tel que `answer[i]` est le **nombre de jours a attendre** apres le jour `i` pour avoir une temperature plus chaude. S'il n'y a aucun jour futur plus chaud, `answer[i] = 0`.

```python
def daily_temperatures(temperatures: list[int]) -> list[int]:
    """
    Return, for each day, the number of days until a warmer temperature.
    0 if no warmer day exists.
    """
    pass
```

**Indice** : la stack doit contenir des **indices** (pas des temperatures) pour pouvoir calculer la distance `i - stack[-1]` au moment du pop.

### Tests

```python
assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
assert daily_temperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
assert daily_temperatures([30, 60, 90]) == [1, 1, 0]
assert daily_temperatures([90, 60, 30]) == [0, 0, 0]    # Strictly decreasing: never warmer
assert daily_temperatures([70]) == [0]
assert daily_temperatures([70, 70, 70]) == [0, 0, 0]    # Equal is NOT warmer
```

### Criteres de reussite

- [ ] Utilise un monotonic stack decroissant d'**indices**
- [ ] Chaque indice est push une fois et pop au plus une fois → O(n) temps (pas O(n^2))
- [ ] Les temperatures egales ne declenchent PAS de pop (strictement plus chaud)
- [ ] Complexite O(n) temps, O(n) espace
- [ ] Tous les tests passent

---

## Exercice 5 : Stack Evaluation — Evaluate Reverse Polish Notation

### Objectif

Utiliser une stack pour evaluer une expression postfixee, et reperer le piege classique de la division entiere Python (`//` arrondit vers le bas, l'enonce demande une troncature vers zero).

### Consigne

Evalue une expression en notation polonaise inversee (RPN). Les tokens sont soit des entiers, soit un operateur parmi `+`, `-`, `*`, `/`.

Regles :
- Chaque operateur s'applique aux **deux derniers operandes** empiles.
- La division **tronque vers zero** : `6 / -132 = 0`, `-7 / 2 = -3` (pas `-4` !).
- L'expression est toujours valide (pas de division par zero).

```python
def eval_rpn(tokens: list[str]) -> int:
    """
    Evaluate a Reverse Polish Notation expression.
    Division truncates toward zero.
    """
    pass
```

**Piege** : en Python, `-7 // 2 == -4` (floor division). Utilise `int(a / b)` pour tronquer vers zero. Attention aussi a l'**ordre des operandes** : le premier pop est l'operande de DROITE.

### Tests

```python
assert eval_rpn(["2", "1", "+", "3", "*"]) == 9         # (2 + 1) * 3
assert eval_rpn(["4", "13", "5", "/", "+"]) == 6         # 4 + (13 / 5)
assert eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
assert eval_rpn(["42"]) == 42                            # Single number
assert eval_rpn(["-7", "2", "/"]) == -3                  # Truncation toward zero, NOT floor
assert eval_rpn(["7", "-2", "/"]) == -3                  # Same trap, negative divisor
assert eval_rpn(["3", "4", "-"]) == -1                   # Operand order: 3 - 4, not 4 - 3
```

### Criteres de reussite

- [ ] Utilise une stack : push les nombres, pop deux operandes par operateur
- [ ] L'ordre des operandes est correct (`b op a` ou `a` est le premier pop... a verifier !)
- [ ] La division tronque vers zero avec `int(a / b)` (pas `//`)
- [ ] Gere les nombres negatifs dans les tokens (`"-11"` est un nombre, pas un operateur)
- [ ] Complexite O(n) temps, O(n) espace

---

## Exercice 6 : Design — Implement Queue using Two Stacks

### Objectif

Comprendre l'analyse **amortie** : chaque operation individuelle peut etre O(n), mais la moyenne sur une sequence est O(1). C'est une question de design tres frequente en entretien.

### Consigne

Implemente une queue FIFO en utilisant **uniquement deux stacks** (listes Python avec seulement `append` et `pop` autorises — pas de `pop(0)`, pas de `insert`, pas de `deque`).

```python
class MyQueue:
    def __init__(self):
        pass

    def push(self, x: int) -> None:
        """Add element to the back of the queue."""
        pass

    def pop(self) -> int:
        """Remove and return the front element."""
        pass

    def peek(self) -> int:
        """Return the front element without removing it."""
        pass

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        pass
```

**Indice** : une stack `inbox` recoit les push. Une stack `outbox` sert les pop/peek. On ne transvase `inbox → outbox` que quand `outbox` est vide. Chaque element est ainsi deplace au plus une fois.

### Tests

```python
q = MyQueue()
q.push(1)
q.push(2)
assert q.peek() == 1            # FIFO: 1 entered first
assert q.pop() == 1
assert q.empty() == False
assert q.pop() == 2
assert q.empty() == True

# Interleaved push/pop — the transfer must not lose ordering
q2 = MyQueue()
q2.push(1)
q2.push(2)
assert q2.pop() == 1            # outbox = [2]
q2.push(3)                      # inbox = [3], outbox = [2]
assert q2.pop() == 2            # Must serve outbox FIRST
assert q2.pop() == 3
assert q2.empty() == True
```

### Criteres de reussite

- [ ] N'utilise que `append` et `pop` (fin de liste) sur les deux stacks
- [ ] Le transvasement `inbox → outbox` n'a lieu que si `outbox` est vide (sinon l'ordre casse)
- [ ] `push` est O(1), `pop`/`peek` sont O(1) **amorti** — et tu sais expliquer pourquoi (chaque element est transvase au plus une fois dans sa vie)
- [ ] Le test interleave (push apres pop) passe — c'est lui qui detecte les transvasements faux
- [ ] Tous les tests passent
