# Exercices Easy — Complexite & Big-O

---

## Exercice 1 : Identifier la complexite

### Objectif

Savoir lire un code et determiner sa complexite temps et espace sans hesiter.

### Consigne

Pour chacune des fonctions suivantes, determine la complexite **temps** et **espace** en Big-O. Justifie en une phrase.

```python
# Fonction A
def func_a(arr):
    total = 0
    for x in arr:
        total += x
    return total

# Fonction B
def func_b(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i] == arr[j] and i != j:
                return True
    return False

# Fonction C
def func_c(n):
    i = 1
    while i < n:
        i *= 2
    return i

# Fonction D
def func_d(arr):
    return sorted(set(arr))

# Fonction E
def func_e(s):
    result = ""
    for char in s:
        result += char
    return result
```

### Criteres de reussite

- [ ] Les 5 complexites temps sont correctes
- [ ] Les 5 complexites espace sont correctes
- [ ] Chaque reponse est justifiee (pas juste "O(n)" sans explication)
- [ ] Le piege de la Fonction E (concatenation string) est identifie

---

## Exercice 2 : Classer par complexite

### Objectif

Developper l'intuition pour comparer les ordres de grandeur.

### Consigne

Classe les expressions suivantes de la plus petite a la plus grande pour n → ∞ :

```
n^2, 100n, n log n, 2^n, log n, n!, 1, n^3, n, sqrt(n)
```

Ensuite, pour chaque paire consecutive dans ton classement, donne un ratio approximatif quand n = 1000 (ex: "100x plus grand").

### Criteres de reussite

- [ ] L'ordre est parfaitement correct (10 expressions)
- [ ] Au moins 3 ratios sont calcules correctement
- [ ] `sqrt(n)` est place au bon endroit (piege classique)

---

## Exercice 3 : Optimise le lookup

### Objectif

Appliquer le reflexe set/dict pour eliminer les lookups O(n).

### Consigne

La fonction suivante est O(n * m). Reecris-la pour qu'elle soit O(n + m).

```python
def common_elements(list1, list2):
    """Return elements present in both lists."""
    result = []
    for x in list1:          # n iterations
        if x in list2:       # O(m) lookup per iteration!
            result.append(x)
    return result
```

Ecris la version optimisee et explique dans un commentaire pourquoi elle est O(n + m).

### Criteres de reussite

- [ ] La solution utilise un `set` pour les lookups
- [ ] La complexite est bien O(n + m) temps, O(min(n, m)) espace
- [ ] La solution gere le cas des doublons correctement (pas de doublon dans le resultat)
- [ ] Un commentaire explique le changement de complexite
