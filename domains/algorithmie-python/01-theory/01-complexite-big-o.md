# Jour 1 — Complexite & Big-O

> **Temps estime** : 45 min de lecture active | **Objectif** : analyser la complexite de n'importe quel code Python en < 30 secondes

---

## 1. Pourquoi la complexite compte (contexte entretien)

En entretien live coding, ton code doit **fonctionner** ET **scaler**. Un candidat qui resout le probleme en O(n^2) alors qu'une solution O(n) existe sera recale, meme si son code est correct.

```python
# Exemple concret : trouver les doublons dans une liste de 10M elements
# Approche naive O(n^2) : 10^14 operations → ~28 heures
# Approche set O(n)     : 10^7 operations  → ~0.01 seconde
```

**Ce qu'on attend de toi en entretien** :
1. Proposer d'abord une solution brute-force (montrer que tu comprends le probleme)
2. Identifier sa complexite
3. Optimiser en expliquant quel facteur tu elimines

---

## 2. Big-O — Definition simplifiee

Big-O decrit **comment le temps d'execution grandit quand l'input grandit**.

```python
def find_max(arr):        # n = len(arr)
    max_val = arr[0]      # 1 operation
    for x in arr:         # n iterations
        if x > max_val:   # 1 comparaison par iteration
            max_val = x   # 1 affectation (parfois)
    return max_val        # 1 operation
# Total : 1 + n + n + 1 = 2n + 2
# Big-O : O(n) — on garde le terme dominant, on vire les constantes
```

**Regles de simplification** :
- Garder uniquement le **terme dominant** : O(n^2 + n) → O(n^2)
- Ignorer les **constantes multiplicatives** : O(3n) → O(n)
- Big-O est un **worst case** par defaut (sauf mention contraire)

---

## 3. Les 7 complexites a connaitre

Du plus rapide au plus lent, avec des exemples concrets :

### O(1) — Constant

```python
def get_first(arr):
    return arr[0]  # Toujours 1 operation, meme avec 1 milliard d'elements

d = {"key": "value"}
d["key"]  # Acces dict = O(1) en moyenne (hash table)
```

**Quand** : acces par index, acces dict/set, operations mathematiques.

### O(log n) — Logarithmique

```python
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:           # A chaque iteration, on divise l'espace par 2
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
# 1 milliard d'elements → ~30 iterations seulement (log2(10^9) ≈ 30)
```

**Quand** : on divise le probleme par 2 (ou k) a chaque etape. Binary search, arbres equilibres.

### O(n) — Lineaire

```python
def sum_all(arr):
    total = 0
    for x in arr:    # On touche chaque element exactement une fois
        total += x
    return total
```

**Quand** : un seul parcours de l'input. C'est souvent le minimum possible (il faut au moins lire l'input).

### O(n log n) — Linearithmique

```python
arr = [3, 1, 4, 1, 5, 9, 2, 6]
arr.sort()  # Timsort en Python = O(n log n) en moyenne et worst case
```

**Quand** : tri optimal (merge sort, timsort), certains algorithmes divide & conquer. C'est la **barriere du tri** — si tu dois trier, tu ne feras pas mieux que O(n log n) sur des comparaisons.

### O(n^2) — Quadratique

```python
def has_duplicate_naive(arr):
    for i in range(len(arr)):         # n iterations
        for j in range(i + 1, len(arr)):  # n-1, n-2, ... iterations
            if arr[i] == arr[j]:
                return True
    return False
# n*(n-1)/2 comparaisons = O(n^2)
```

**Quand** : boucles imbriquees sur le meme input. **Signal d'alarme en entretien** — chercher une optimisation.

### O(2^n) — Exponentiel

```python
def fibonacci_naive(n):
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)
# Chaque appel genere 2 sous-appels → arbre binaire de profondeur n
# fib(40) = ~1 milliard d'appels
```

**Quand** : recursion sans memoization, exploration exhaustive de sous-ensembles. **Inutilisable** pour n > 25-30.

### O(n!) — Factoriel

```python
def permutations(arr):
    if len(arr) <= 1:
        return [arr]
    result = []
    for i in range(len(arr)):
        rest = arr[:i] + arr[i+1:]
        for p in permutations(rest):
            result.append([arr[i]] + p)
    return result
# 10! = 3.6M, 15! = 1.3 trillion
```

**Quand** : generer toutes les permutations, TSP brute force. **Limite pratique** : n ≈ 10-12.

### Tableau recapitulatif

| Big-O | Nom | n=10 | n=100 | n=10^6 | Viable ? |
|-------|-----|------|-------|--------|----------|
| O(1) | Constant | 1 | 1 | 1 | Toujours |
| O(log n) | Logarithmique | 3 | 7 | 20 | Toujours |
| O(n) | Lineaire | 10 | 100 | 10^6 | Toujours |
| O(n log n) | Linearithmique | 33 | 664 | 2*10^7 | Oui |
| O(n^2) | Quadratique | 100 | 10^4 | 10^12 | n < 10^4 |
| O(2^n) | Exponentiel | 1024 | 10^30 | Mort | n < 25 |
| O(n!) | Factoriel | 3.6M | 10^158 | Mort | n < 12 |

> **Regle de pouce pour les entretiens** : avec les contraintes typiques (n ≈ 10^5 a 10^6), viser O(n) ou O(n log n). Si n ≤ 20, DP/backtracking O(2^n) est acceptable.

---

## 4. Comment analyser la complexite d'un code

### Regle 1 : Boucles simples

```python
for i in range(n):    # O(n)
    ...
```

### Regle 2 : Boucles imbriquees — multiplier

```python
for i in range(n):        # O(n)
    for j in range(n):    #   * O(n)
        ...               # = O(n^2)
```

### Regle 3 : Boucles consecutives — additionner

```python
for i in range(n):    # O(n)
    ...
for j in range(m):    # O(m)
    ...
# Total : O(n + m) — on garde les deux si n et m sont independants
```

### Regle 4 : Division par 2 a chaque iteration → O(log n)

```python
i = n
while i > 0:
    i //= 2    # O(log n) iterations
```

### Regle 5 : Recursion — dessiner l'arbre d'appels

```python
def solve(n):
    if n <= 0:
        return
    solve(n - 1)  # 1 branche → O(n) (lineaire)

def solve(n):
    if n <= 0:
        return
    solve(n - 1)  # 2 branches → O(2^n) (exponentiel)
    solve(n - 1)
```

### Regle 6 : Appels a des methodes Python — connaitre leur cout

| Operation | list | dict/set | str |
|-----------|------|----------|-----|
| `x in ...` | O(n) | O(1) | O(n) |
| `.append(x)` | O(1) amorti | — | — |
| `.insert(0, x)` | O(n) | — | — |
| `[i]` | O(1) | O(1) | O(1) |
| `.sort()` | O(n log n) | — | — |
| `.pop()` | O(1) | — | — |
| `.pop(0)` | O(n) | — | — |
| `len(...)` | O(1) | O(1) | O(1) |
| slice `[a:b]` | O(b-a) | — | O(b-a) |
| `+` (concat) | O(n+m) | — | O(n+m) |

---

## 5. Complexite espace vs temps

La complexite **espace** mesure la memoire supplementaire utilisee (hors input).

```python
# Espace O(1) — tri en place
arr.sort()

# Espace O(n) — on cree un set de meme taille que l'input
seen = set()
for x in arr:
    if x in seen:
        return True
    seen.add(x)

# Espace O(n^2) — matrice n*n
matrix = [[0] * n for _ in range(n)]
```

**En entretien** : on mentionne TOUJOURS la complexite temps ET espace. Format attendu : "Time O(n), Space O(n)".

> **Trade-off classique** : on echange souvent de l'espace pour du temps. Exemple : hash map O(n) espace pour eviter une boucle imbriquee O(n^2) temps.

---

## 6. Complexite amortie (Amortized Complexity)

Certaines operations sont couteuses *parfois* mais pas *en moyenne*.

### Exemple : `list.append()` en Python

```python
arr = []
for i in range(n):
    arr.append(i)  # Chaque append est O(1)... vraiment ?
```

**Sous le capot** : Python alloue un tableau avec de la capacite en reserve. Quand le tableau est plein, il realloue un nouveau tableau ~1.125x plus grand et copie tout.

- 99% des appends : O(1) (il y a de la place)
- 1% des appends : O(n) (reallocation + copie)
- **En moyenne sur n appends** : O(1) par append → complexite **amortie O(1)**

Autre exemple : l'union-find avec path compression + union by rank → O(alpha(n)) amorti ≈ O(1) en pratique.

---

## 7. Pieges courants en Python

### Piege 1 : `in` sur une list vs un set

```python
# LENT — O(n) par lookup, O(n^2) total
items = [1, 2, 3, ..., n]
for x in data:
    if x in items:  # Parcours lineaire a chaque fois !
        ...

# RAPIDE — O(1) par lookup, O(n) total
items_set = set(items)  # O(n) pour construire
for x in data:
    if x in items_set:  # Hash lookup O(1)
        ...
```

### Piege 2 : Concatenation de strings en boucle

```python
# LENT — O(n^2) car chaque += cree une NOUVELLE string
result = ""
for s in strings:
    result += s  # Copie tout le contenu existant a chaque iteration

# RAPIDE — O(n) total
result = "".join(strings)  # Une seule allocation
```

### Piege 3 : `list.insert(0, x)` et `list.pop(0)`

```python
# LENT — O(n) car decalage de tous les elements
for x in data:
    arr.insert(0, x)  # O(n) * n iterations = O(n^2)

# RAPIDE — utiliser collections.deque
from collections import deque
dq = deque()
for x in data:
    dq.appendleft(x)  # O(1)
```

### Piege 4 : Slicing cache

```python
# Attention : le slicing cree une COPIE
def process(arr):
    return process(arr[1:])  # Copie de taille n-1 a chaque appel
    # Espace total : O(n^2) ! Utiliser un index a la place
```

---

## 8. Flash Cards — Revision espacee

> **Methode** : couvrir la reponse, repondre a voix haute, puis verifier. Revenir dans 1 jour, 3 jours, 7 jours.

**Q1** : Quelle est la complexite de `x in my_list` vs `x in my_set` ?
> **R1** : `list` → O(n), `set` → O(1) en moyenne. Toujours convertir en set si on fait des lookups repetes.

**Q2** : Pourquoi `list.append()` est O(1) alors qu'il doit parfois reallouer ?
> **R2** : Complexite amortie. La reallocation double (environ) la capacite, donc le cout de copie est "dilue" sur les prochains appends. En moyenne : O(1) par operation.

**Q3** : Tu vois deux boucles `for` imbriquees sur un tableau de taille n. Complexite ?
> **R3** : O(n^2). Multiplier les bornes des boucles imbriquees. Attention : si la boucle interne est sur un autre tableau de taille m, c'est O(n*m).

**Q4** : Quelle est la barriere theorique pour le tri par comparaison ?
> **R4** : O(n log n). Aucun tri base sur des comparaisons ne peut faire mieux. (Counting sort et radix sort contournent cette limite en n'utilisant pas de comparaisons.)

**Q5** : En entretien, les contraintes disent n ≤ 10^5. Quelle complexite viser ?
> **R5** : O(n) ou O(n log n). Un O(n^2) donnerait ~10^10 operations = trop lent (~10 secondes+). Regle : viser ~10^7-10^8 operations max.

---

## Resume — Key Takeaways

1. **Big-O = croissance, pas vitesse absolue** — on s'en fiche des constantes
2. **7 classes a connaitre** : O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n) < O(n!)
3. **Boucles imbriquees = multiplier, consecutives = additionner**
4. **Connaitre les couts des operations Python** (le piege `in list` tombe en entretien)
5. **Toujours annoncer temps ET espace** en entretien
6. **Trade-off espace/temps** : la plupart des optimisations utilisent une structure auxiliaire

---

## Pour aller plus loin

Ressources canoniques sur l'analyse de complexite :

- **MIT 6.006 — Introduction to Algorithms** (Erik Demaine & Srini Devadas, MIT OCW Spring 2020) — Lec. 1 (Algorithmic Thinking, Peak Finding) introduit l'analyse asymptotique sur un probleme concret. https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/
- **CLRS — Introduction to Algorithms** (Cormen, Leiserson, Rivest, Stein, 4th ed, MIT Press 2022) — Ch 2-3 : analyse, notation asymptotique (O, Theta, Omega), recurrences. La reference academique. https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
- **The Algorithm Design Manual** (Steven Skiena, 3rd ed 2020) — Ch 1-2 : pedagogie tres accessible avec war stories sur l'impact concret de la complexite. https://www.algorist.com/
- **NeetCode — Big-O notation** (roadmap structuree) — videos courtes pour calibrer le pattern matching en entretien. https://neetcode.io/roadmap
