# Jour 10 — Dynamic Programming : Memoization, Tabulation, Classiques

> **Temps estime** : 75 min de lecture active | **Objectif** : reconnaitre un probleme DP en < 60 secondes, choisir memoization vs tabulation, et coder les 6 patterns classiques

---

## 1. Ce qu'est vraiment le DP (et ce qu'il n'est pas)

**Dynamic Programming = recursion + memoization**. Ni plus, ni moins.

Un probleme est resoluble en DP si et seulement si il verifie **deux proprietes** :

1. **Optimal substructure** : la solution optimale au probleme peut etre construite a partir des solutions optimales de sous-problemes plus petits.
2. **Overlapping subproblems** : les memes sous-problemes apparaissent plusieurs fois dans la recursion naive — si on les cache, on evite un travail exponentiel.

### Exemple — Fibonacci naif

```python
def fib(n):
    if n < 2: return n
    return fib(n - 1) + fib(n - 2)
# fib(5) recalcule fib(2) 5 fois, fib(3) 3 fois, etc.
# Complexite : O(2^n) — exponentielle !
```

Avec memoization :

```python
def fib(n, memo=None):
    if memo is None: memo = {}
    if n in memo: return memo[n]
    if n < 2: return n
    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]
# Chaque fib(i) calcule UNE fois. Complexite : O(n).
```

**Gain** : O(2^n) → O(n) juste en cachant les resultats intermediaires. C'est la magie du DP.

---

## 2. Les deux approches — Memoization vs Tabulation

| Aspect | Memoization (top-down) | Tabulation (bottom-up) |
|--------|------------------------|-------------------------|
| **Direction** | Recursif, part du but | Iteratif, part des cas de base |
| **Stockage** | Dict ou array | Array |
| **Code** | Proche de la formulation recursive | Proche d'une boucle |
| **Overhead** | Appels de fonction (plus lent en Python) | Boucles (plus rapide) |
| **Espace** | Stack + cache | Juste le tableau (parfois O(1) possible) |
| **Sous-problemes** | Calcule uniquement les necessaires | Calcule tout, meme inutile |

> **En entretien** : commence TOUJOURS par la version recursive + memoization (plus facile a ecrire), puis dis "je peux optimiser en tabulation bottom-up". L'interviewer appreciera.

### Template memoization

```python
def solve(state):
    memo = {}
    def dp(s):
        if s in memo: return memo[s]
        if base_case(s): return base_value(s)
        memo[s] = combine(dp(subproblem_1), dp(subproblem_2), ...)
        return memo[s]
    return dp(state)
```

### Template tabulation

```python
def solve(n):
    dp = [0] * (n + 1)
    dp[0] = base_value
    for i in range(1, n + 1):
        dp[i] = combine(dp[i - 1], dp[i - 2], ...)
    return dp[n]
```

---

## 3. La recette en 5 etapes pour resoudre un probleme DP

1. **Definir l'etat** : quelles variables capturent entierement un sous-probleme ? Exemple : `dp[i]` = "meilleure reponse en considerant les i premiers elements".
2. **Ecrire la recurrence** : comment `dp[i]` depend-il des etats plus petits ? C'est la partie la plus difficile.
3. **Identifier les cas de base** : les etats qui n'ont pas besoin de recurrence (`dp[0]`, `dp[1]`).
4. **Choisir l'ordre** : memoization (naturellement recursif) ou tabulation (de `0` a `n`).
5. **Optimiser l'espace** si possible : si `dp[i]` ne depend que de `dp[i-1]` et `dp[i-2]`, on peut garder juste 2 variables au lieu d'un tableau.

---

## 4. Pattern 1 — Linear DP (Climbing Stairs, House Robber)

### Climbing Stairs

Tu peux monter 1 ou 2 marches a la fois. Combien de facons d'atteindre la marche n ?

```python
# Etat : dp[i] = nombre de facons d'atteindre la marche i
# Recurrence : dp[i] = dp[i-1] + dp[i-2]
# (venant de i-1 avec un pas de 1, ou de i-2 avec un pas de 2)
# Base : dp[0] = 1, dp[1] = 1

def climb_stairs(n):
    if n < 2: return 1
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
# C'est... Fibonacci !

# Optimisation espace : O(1)
def climb_stairs_opt(n):
    if n < 2: return 1
    a, b = 1, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

### House Robber

Tu ne peux pas cambrioler deux maisons adjacentes. Max total vole ?

```python
# Etat : dp[i] = max gold a la maison i (inclus ou non)
# Recurrence : dp[i] = max(dp[i-1], dp[i-2] + nums[i])
#              (skip la maison i)  (prendre nums[i] + meilleur a i-2)
# Base : dp[0] = nums[0], dp[1] = max(nums[0], nums[1])

def rob(nums):
    if not nums: return 0
    if len(nums) == 1: return nums[0]
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        prev2, prev1 = prev1, max(prev1, prev2 + nums[i])
    return prev1
# Time: O(n), Space: O(1)
```

---

## 5. Pattern 2 — Knapsack 0/1

**Probleme** : tu as n objets avec (poids, valeur) et un sac de capacite W. Maximise la valeur totale sans depasser W. Chaque objet est pris 0 ou 1 fois.

```python
# Etat : dp[i][w] = max valeur avec les i premiers objets et capacite w
# Recurrence :
#   Si weight[i] > w : dp[i][w] = dp[i-1][w]  (on ne peut pas prendre i)
#   Sinon : dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight[i]] + value[i])
#           (skip i, ou prendre i)
# Base : dp[0][w] = 0 pour tout w

def knapsack_01(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(
                    dp[i - 1][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]
                )
    return dp[n][W]
# Time: O(n * W), Space: O(n * W)

# Optimisation espace : 1D
def knapsack_01_1d(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        # IMPORTANT: iterer de droite a gauche pour ne pas reutiliser l'objet i
        for w in range(W, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]
# Time: O(n * W), Space: O(W)
```

> **Piege classique** : pour knapsack **unbounded** (objets illimites), iterer de **gauche a droite**. Pour knapsack **0/1**, iterer de **droite a gauche** pour empecher de reutiliser un objet dans la meme passe.

---

## 6. Pattern 3 — Coin Change

**Probleme** : quel est le nombre minimum de pieces pour atteindre un montant ? Tu as des pieces de denominations fixes, en quantite illimitee.

```python
# Etat : dp[amount] = min pieces pour faire "amount"
# Recurrence : dp[a] = min(dp[a - coin] + 1) pour chaque coin
# Base : dp[0] = 0, dp[a] = inf pour a > 0

def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for coin in coins:
            if coin <= a:
                dp[a] = min(dp[a], dp[a - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
# Time: O(amount * len(coins)), Space: O(amount)
```

### Variante — Nombre de combinaisons (Coin Change II)

```python
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1                          # Une facon de faire 0 : ne rien prendre
    for coin in coins:                 # BOUCLE EXT = coin (evite les doublons)
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]
    return dp[amount]
```

> **Piege** : inverser les boucles change le probleme ! `for coin: for a:` compte les combinaisons (ordre indifferent). `for a: for coin:` compte les permutations (ordre compte).

---

## 7. Pattern 4 — Longest Common Subsequence (LCS)

**Probleme** : longueur de la plus longue sous-sequence commune a deux strings. Une sous-sequence n'est pas contigue (contrairement a une substring).

```python
# Etat : dp[i][j] = longueur LCS de s1[:i] et s2[:j]
# Recurrence :
#   Si s1[i-1] == s2[j-1] : dp[i][j] = dp[i-1][j-1] + 1
#   Sinon : dp[i][j] = max(dp[i-1][j], dp[i][j-1])
# Base : dp[0][j] = dp[i][0] = 0

def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
# Time: O(m * n), Space: O(m * n)
```

### Variante — Edit Distance (Levenshtein)

```python
# Etat : dp[i][j] = nombre min d'operations pour transformer s1[:i] en s2[:j]
# Recurrence :
#   Si s1[i-1] == s2[j-1] : dp[i][j] = dp[i-1][j-1]
#   Sinon : dp[i][j] = 1 + min(dp[i-1][j],     # delete
#                              dp[i][j-1],     # insert
#                              dp[i-1][j-1])   # replace
```

---

## 8. Pattern 5 — Longest Increasing Subsequence (LIS)

**Probleme** : longueur de la plus longue sous-sequence strictement croissante.

```python
# Etat : dp[i] = longueur LIS se terminant a i
# Recurrence : dp[i] = 1 + max(dp[j] pour j < i si nums[j] < nums[i])
# Base : dp[i] = 1 pour tout i (l'element seul est deja une LIS)

def lis(nums):
    if not nums: return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
# Time: O(n^2), Space: O(n)
```

### Version O(n log n) avec patience sorting

```python
import bisect

def lis_fast(nums):
    # tails[i] = plus petite valeur terminant une LIS de longueur i+1
    tails = []
    for num in nums:
        idx = bisect.bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num
    return len(tails)
# Time: O(n log n), Space: O(n)
```

---

## 9. Pattern 6 — Grid DP (Unique Paths)

**Probleme** : combien de chemins distincts d'un coin a l'autre d'une grille m x n en ne se deplacant que vers la droite ou le bas ?

```python
# Etat : dp[i][j] = nombre de chemins pour atteindre (i, j)
# Recurrence : dp[i][j] = dp[i-1][j] + dp[i][j-1]
# Base : dp[0][j] = dp[i][0] = 1

def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]
# Time: O(m * n), Space: O(m * n) (peut etre O(n) avec rolling row)
```

---

## 10. Decision Tree — Comment reconnaitre un probleme DP

```
Signaux d'un probleme DP :
|
├── Le probleme demande un OPTIMUM (min, max, count) ?
│   └── Compte / Min / Max de quelque chose
|
├── Le probleme parle de CHOIX a faire a chaque etape ?
│   └── Take / not take, choose / skip, include / exclude
|
├── Tu peux DECOMPOSER le probleme en sous-problemes independants
│   qui se RECOUVRENT ?
│   └── Fib, Climb stairs, Knapsack, LCS
|
├── Une brute force recursive EXPONENTIELLE et tu sens qu'il y a
│   de la redondance dans les appels ?
│   └── MEMOIZE !
|
└── Le probleme parle de subsequence / substring / partition ?
    └── DP 2D tres probablement (LCS, edit distance, partition DP)
```

**Signaux dans l'enonce** :

| Signal | Pattern probable |
|--------|------------------|
| "how many ways", "number of ways" | Count DP (climb stairs, coin change II) |
| "minimum / maximum ... to do X" | Min/Max DP (coin change, min path sum) |
| "longest / shortest subsequence" | LCS / LIS family |
| "can you split / partition" | Partition DP |
| "unique paths in a zone" | Grid DP 2D |
| "knapsack", "capacity", "fit items" | 0/1 or unbounded knapsack |

---

## 11. Complexites

| Probleme | Temps | Espace |
|----------|-------|--------|
| Fibonacci | O(n) | O(1) |
| Climb Stairs | O(n) | O(1) |
| House Robber | O(n) | O(1) |
| Knapsack 0/1 | O(n * W) | O(W) apres optimisation |
| Coin Change (min) | O(A * C) | O(A) |
| LCS | O(m * n) | O(m * n) ou O(min(m, n)) |
| LIS DP | O(n^2) | O(n) |
| LIS patience | O(n log n) | O(n) |
| Unique Paths | O(m * n) | O(n) apres optimisation |
| Edit Distance | O(m * n) | O(m * n) |

---

## 12. Pieges courants

**Piege 1 — Oublier la memoization**
Ecrire une recursion "propre" sans cache te donne du O(2^n). Toujours memoiser.

**Piege 2 — Mauvais ordre de boucles en knapsack**
0/1 : droite→gauche. Unbounded : gauche→droite. Inverser casse tout.

**Piege 3 — Confondre subsequence et substring**
Subsequence : non contigue (LCS, LIS). Substring : contigue (longest common substring, palindromic substring).

**Piege 4 — Off-by-one sur dp indices**
Quand `dp[i][j]` represente "les i premiers caracteres de s1", attention que `s1[i-1]` est le dernier caractere, pas `s1[i]`.

**Piege 5 — Retourner dp[0] au lieu de dp[n]**
Toujours verifier : tu retournes le cas final, pas le cas de base.

---

## 13. Flash Cards — Revision espacee

**Q1** : Quelles sont les deux proprietes necessaires pour qu'un probleme soit resoluble en DP ?
> **R1** : **Optimal substructure** (la solution optimale se construit a partir des solutions optimales des sous-problemes) ET **overlapping subproblems** (les memes sous-problemes apparaissent plusieurs fois, rendant le cache utile).

**Q2** : Quelle est la difference entre memoization et tabulation ?
> **R2** : Memoization = top-down, recursif, cache les resultats, calcule uniquement les sous-problemes necessaires. Tabulation = bottom-up, iteratif, remplit une table des cas de base vers l'objectif, parfois calcule des sous-problemes inutiles mais plus rapide car pas d'overhead d'appels de fonction.

**Q3** : Pour le knapsack 0/1 en 1D, dans quel sens iterer la capacite w et pourquoi ?
> **R3** : De droite a gauche (W vers weights[i]). Sinon, on utilise la valeur deja mise a jour dans la meme iteration, ce qui permet de prendre l'objet i plusieurs fois — on passe en knapsack unbounded.

**Q4** : Pourquoi dp[0] = 1 dans coin change II (comptage de combinaisons) ?
> **R4** : Il y a exactement UNE facon de faire le montant 0 : ne prendre aucune piece. C'est l'identite neutre pour l'addition, et elle amorce correctement la recurrence `dp[a] += dp[a - coin]`.

**Q5** : Quelle optimisation passe LIS de O(n^2) a O(n log n) ?
> **R5** : Patience sorting avec `bisect_left`. On maintient un tableau `tails` ou `tails[i]` = la plus petite valeur terminant une LIS de longueur i+1. Pour chaque num, on binary-search sa place et on remplace. La longueur finale du tableau est la LIS.

---

## Resume — Key Takeaways

1. **DP = recursion + cache** — pas plus complique
2. **5 etapes** : etat, recurrence, base, ordre, optimisation espace
3. **Memoization d'abord** (facile), **tabulation ensuite** (rapide)
4. **6 patterns classiques** : linear, knapsack, coin change, LCS, LIS, zone
5. **Pieges** : mauvais sens d'iteration en 1D, confusion subsequence/substring
6. **Signaux** : "how many ways", "min/max to do X", "longest subsequence", "partition"
