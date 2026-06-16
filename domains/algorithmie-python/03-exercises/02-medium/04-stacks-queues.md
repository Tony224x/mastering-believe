# Exercices Medium — Stacks & Queues

---

## Exercice 4 : Monotonic Stack — Daily Temperatures

### Objectif

Maitriser le monotonic stack qui stocke des **indices** (pas des valeurs) pour calculer une distance jusqu'au prochain element plus grand. C'est LE pattern monotonic stack de reference en entretien.

### Consigne

Etant donne un tableau `temperatures` representant les temperatures quotidiennes, retourne un tableau `answer` ou `answer[i]` est le **nombre de jours** a attendre apres le jour `i` pour avoir une temperature plus chaude. S'il n'y a aucun jour plus chaud dans le futur, mets `0`.

```python
def daily_temperatures(temperatures: list[int]) -> list[int]:
    """
    For each day, return how many days to wait for a warmer temperature.
    0 if no warmer day exists.
    """
    pass
```

**Indice** : maintiens un monotonic stack **decroissant** d'indices. Quand la temperature courante depasse celle au sommet de la stack, tu as trouve le "next warmer day" de cet indice — pop et calcule la distance `i - j`.

### Tests

```python
assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
assert daily_temperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
assert daily_temperatures([60, 50, 40, 30]) == [0, 0, 0, 0]
assert daily_temperatures([30]) == [0]
assert daily_temperatures([30, 30, 30]) == [0, 0, 0]   # Strictly warmer required
assert daily_temperatures([89, 62, 70, 58, 47, 47, 46, 76, 100, 70]) == [8, 1, 5, 4, 3, 2, 1, 1, 0, 0]
```

### Criteres de reussite

- [ ] Utilise un monotonic stack d'**indices** (pas de valeurs) — necessaire pour la distance
- [ ] La stack reste decroissante en temperature de bas en haut
- [ ] Complexite O(n) temps (chaque indice pousse/pope au plus une fois), O(n) espace
- [ ] Sait expliquer pourquoi le while imbrique reste O(n) amorti
- [ ] Gere les egalites (un jour de meme temperature n'est PAS plus chaud)

---

## Exercice 5 : Stack de parsing — Decode String

### Objectif

Utiliser une stack pour parser une structure **imbriquee** (encodage de type `k[contenu]`). C'est le pattern "stack de contexte" : on empile l'etat avant d'entrer dans un bloc, on depile en sortant.

### Consigne

Etant donne une string encodee, retourne sa version decodee. La regle d'encodage est `k[encoded_string]` ou `encoded_string` est repete exactement `k` fois. `k` est un entier positif. Les encodages peuvent etre imbriques.

L'entree ne contient que des lettres minuscules, des chiffres et des crochets ; elle est toujours valide.

```python
def decode_string(s: str) -> str:
    """
    Decode a string encoded as k[encoded_string] (possibly nested).
    Example: "3[a2[c]]" -> "accaccacc"
    """
    pass
```

**Indice** : utilise deux stacks (ou une stack de tuples). Quand tu vois `[`, empile le `(string_courante, multiplicateur)` et reinitialise. Quand tu vois `]`, depile et reconstruis : `prev_string + k * current_string`.

### Tests

```python
assert decode_string("3[a]2[bc]") == "aaabcbc"
assert decode_string("3[a2[c]]") == "accaccacc"
assert decode_string("2[abc]3[cd]ef") == "abcabccdcdcdef"
assert decode_string("abc") == "abc"
assert decode_string("10[a]") == "aaaaaaaaaa"          # k a plusieurs chiffres
assert decode_string("2[3[a]b]") == "aaabaaab"
assert decode_string("") == ""
```

### Criteres de reussite

- [ ] Utilise une (ou deux) stack(s) pour sauvegarder le contexte avant chaque `[`
- [ ] Gere correctement les `k` a plusieurs chiffres (ex: `10[a]`)
- [ ] Gere l'imbrication (un bloc dans un bloc)
- [ ] Complexite O(N) ou N = longueur de la sortie decodee
- [ ] Tous les tests passent, y compris string sans encodage et string vide

---

## Exercice 6 : Design — Min Stack

### Objectif

Concevoir une structure qui maintient un **invariant auxiliaire** (le minimum courant) en O(1) a chaque operation. Ce probleme de design teste si tu sais maintenir une information derivee sans tout recalculer.

### Consigne

Implemente une `MinStack` qui supporte `push`, `pop`, `top` et `get_min`, **toutes en O(1)**.

- `push(val)` : empile `val`
- `pop()` : depile le sommet
- `top()` : retourne le sommet sans le retirer
- `get_min()` : retourne le minimum present dans la stack

```python
class MinStack:
    def __init__(self):
        """Initialize an empty stack."""
        pass

    def push(self, val: int) -> None:
        pass

    def pop(self) -> None:
        pass

    def top(self) -> int:
        pass

    def get_min(self) -> int:
        pass
```

**Indice** : maintiens une **seconde stack** qui suit le minimum courant a chaque niveau. A chaque push, empile `min(val, min_actuel)`. A chaque pop, depile les deux stacks ensemble.

### Tests

```python
ms = MinStack()
ms.push(-2)
ms.push(0)
ms.push(-3)
assert ms.get_min() == -3
ms.pop()
assert ms.top() == 0
assert ms.get_min() == -2

ms = MinStack()
ms.push(5)
assert ms.get_min() == 5
assert ms.top() == 5
ms.push(5)                  # Duplicate of the minimum
assert ms.get_min() == 5
ms.pop()
assert ms.get_min() == 5    # Still 5 because of the duplicate

ms = MinStack()
ms.push(2)
ms.push(0)
ms.push(3)
ms.push(0)
assert ms.get_min() == 0
ms.pop()
assert ms.get_min() == 0
ms.pop()
assert ms.get_min() == 0
ms.pop()
assert ms.get_min() == 2
```

### Criteres de reussite

- [ ] `push`, `pop`, `top` et `get_min` sont **toutes O(1)**
- [ ] Gere correctement les doublons du minimum (push de la meme valeur plusieurs fois)
- [ ] Complexite O(n) espace (au plus deux stacks de taille n)
- [ ] N'utilise pas `min(stack)` (O(n)) dans `get_min`
- [ ] Tous les tests passent
