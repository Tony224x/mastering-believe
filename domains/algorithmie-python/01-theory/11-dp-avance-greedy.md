# Jour 11 — DP Avance & Greedy : State Machines, Interval DP, Quand Greedy Marche

> **Temps estime** : 60-75 min de lecture active | **Objectif** : maitriser les DP non triviaux (state machine, interval, partition) et reconnaitre quand un algorithme glouton est correct (ou pas)

---

## 1. Pourquoi ce jour est charniere

Apres le DP classique (Day 10), les problemes d'entretien montent d'un cran : etats multiples, decisions temporelles, fusions d'intervalles. Et en parallele, certains problemes qui **ressemblent** a du DP se resolvent en fait en **glouton** (greedy) en O(n).

**Le piege classique** : sortir un DP O(n^2) alors que l'interviewer attendait un greedy O(n). Savoir distinguer les deux est ce qui separe un candidat "competent" d'un candidat "senior".

---

## 2. Pattern 1 — State Machine DP

**Concept** : a chaque etape, le probleme a un **etat discret** (holding stock / not holding, cooldown / active, etc.). La DP devient un automate fini : on calcule le max/min pour chaque etat a chaque etape.

### Best Time to Buy and Sell Stock II (transactions illimitees)

```python
# Etats : hold (j'ai une action), cash (je n'en ai pas)
# Transitions :
#   cash[i] = max(cash[i-1], hold[i-1] + prices[i])    # rester, ou vendre
#   hold[i] = max(hold[i-1], cash[i-1] - prices[i])    # rester, ou acheter

def max_profit(prices):
    cash, hold = 0, float('-inf')
    for p in prices:
        cash, hold = max(cash, hold + p), max(hold, cash - p)
    return cash
# Time: O(n), Space: O(1)
```

### Best Time to Buy and Sell with Cooldown

```python
# 3 etats : hold, cash (vient d'etre vendu), rest (en cooldown)
# Transitions :
#   hold[i] = max(hold[i-1], rest[i-1] - prices[i])   # achat apres cooldown
#   cash[i] = hold[i-1] + prices[i]                    # vente
#   rest[i] = max(rest[i-1], cash[i-1])                # continuer a rien faire

def max_profit_cooldown(prices):
    if not prices: return 0
    hold, cash, rest = -prices[0], 0, 0
    for p in prices[1:]:
        prev_cash = cash
        cash = hold + p
        hold = max(hold, rest - p)
        rest = max(rest, prev_cash)
    return max(cash, rest)
```

### Template general

```python
def state_machine_dp(events):
    # Initialiser chaque etat avec sa valeur de depart
    state_a = initial_value_a
    state_b = initial_value_b
    for event in events:
        new_a = transition_a(state_a, state_b, event)
        new_b = transition_b(state_a, state_b, event)
        state_a, state_b = new_a, new_b
    return best_state
```

> **Cle** : ecrire les transitions sur papier AVANT de coder. Dessiner l'automate si besoin.

---

## 3. Pattern 2 — Interval DP

**Concept** : dp[i][j] = reponse optimale pour l'intervalle `[i, j]`. On construit les reponses pour les petits intervalles d'abord, puis les plus grands.

### Palindrome Partitioning — Minimum Cuts

```python
# Etape 1 : table des palindromes is_palin[i][j] = s[i..j] est-il palindrome ?
# Etape 2 : dp[i] = min cuts pour partitionner s[:i+1] en palindromes

def min_cut(s):
    n = len(s)
    is_palin = [[False] * n for _ in range(n)]
    for i in range(n):
        is_palin[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2 or is_palin[i + 1][j - 1]:
                    is_palin[i][j] = True

    dp = [0] * n
    for i in range(n):
        if is_palin[0][i]:
            dp[i] = 0
            continue
        dp[i] = i   # Upper bound: cut before every char
        for j in range(1, i + 1):
            if is_palin[j][i]:
                dp[i] = min(dp[i], dp[j - 1] + 1)
    return dp[n - 1]
# Time: O(n^2), Space: O(n^2)
```

### Longest Palindromic Substring

```python
def longest_palindrome(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1
    for i in range(n):
        dp[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and (length == 2 or dp[i + 1][j - 1]):
                dp[i][j] = True
                if length > max_len:
                    start, max_len = i, length
    return s[start:start + max_len]
```

> **Ordre de remplissage** : interval DP se remplit par **longueur d'intervalle croissante**. D'abord les length=1, puis length=2, etc. C'est la condition pour que `dp[i+1][j-1]` soit deja calcule.

---

## 4. Pattern 3 — Partition DP

**Concept** : on decoupe un tableau en segments. `dp[i]` = meilleure reponse pour les i premiers elements, en considerant tous les derniers segments possibles `[j, i]`.

### Word Break

```python
# dp[i] = True si s[:i] peut etre decoupe en mots du dictionnaire
# dp[0] = True (chaine vide)
# dp[i] = OR sur j : dp[j] ET s[j:i] in words

def word_break(s, word_dict):
    word_set = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]
# Time: O(n^2), Space: O(n)
```

### Perfect Squares

```python
# dp[n] = min carres parfaits dont la somme est n
# dp[n] = min(dp[n - i*i] + 1) pour tout i ou i*i <= n

def num_squares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
    return dp[n]
```

---

## 5. Algorithmes gloutons — Quand ca marche

**Un greedy est correct SSI** : a chaque etape, faire le **choix localement optimal** conduit a l'optimum global.

**Comment le prouver** : exchange argument. "Si on n'avait pas fait ce choix, on pourrait remplacer par le choix glouton sans empirer le resultat."

**Quand ca echoue** : contre-exemple. Coin change avec `[1, 3, 4]` et target 6 : greedy prend 4+1+1 (3 pieces), mais 3+3 (2 pieces) est optimal. Le greedy est faux ici.

---

## 6. Pattern 4 — Interval Scheduling

**Probleme** : tu as un ensemble d'intervalles, maximise le nombre que tu peux faire sans chevauchement.

**Greedy** : trier par **fin croissante**, prendre chaque intervalle compatible avec le dernier pris.

```python
def max_non_overlapping(intervals):
    if not intervals: return 0
    intervals.sort(key=lambda x: x[1])     # Trier par end
    count = 1
    last_end = intervals[0][1]
    for start, end in intervals[1:]:
        if start >= last_end:              # Pas de chevauchement
            count += 1
            last_end = end
    return count
# Time: O(n log n), Space: O(1)
```

**Pourquoi trier par end et pas par start ?** Parce que l'intervalle qui se termine le plus tot **laisse le plus de place** pour les suivants. Intuition : "liberer le terrain au plus vite".

### Variante — Minimum Arrows to Burst Balloons

Meme technique, mais au lieu de compter les intervalles selectionnes, on compte les "coups" (arrows) necessaires pour couvrir tous les intervalles.

---

## 7. Pattern 5 — Jump Game

### Jump Game I — Peut-on atteindre la fin ?

```python
# Greedy : garder la plus loin position atteignable (max_reach)
# A chaque index i, si i > max_reach → impossible
# Sinon, max_reach = max(max_reach, i + nums[i])

def can_jump(nums):
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
    return True
# Time: O(n), Space: O(1)
```

### Jump Game II — Nombre minimum de sauts

```python
# Greedy : BFS implicite
# On maintient (current_end, farthest) : la frontiere actuelle et la prochaine
# Quand i atteint current_end, on fait un "saut" et on etend la frontiere

def jump(nums):
    jumps, current_end, farthest = 0, 0, 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:               # Fin de la frontiere actuelle
            jumps += 1
            current_end = farthest
    return jumps
```

---

## 8. Pattern 6 — Gas Station

**Probleme** : `gas[i]` = essence a la station i, `cost[i]` = essence pour aller a i+1. Existe-t-il un point de depart ou tu peux faire le tour complet ? Retourne l'index ou -1.

```python
# Greedy en O(n) base sur deux observations :
# 1. Si total_gas < total_cost, impossible -> return -1
# 2. Sinon, il existe exactement un point de depart.
#    Si depuis start on epuise l'essence a index i, aucun point entre
#    start et i ne peut etre le depart (ils ont tous moins d'essence),
#    donc on essaie i+1.

def can_complete_circuit(gas, cost):
    total_tank, curr_tank, start = 0, 0, 0
    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        curr_tank += gas[i] - cost[i]
        if curr_tank < 0:
            start = i + 1              # Aucun depart valide dans [old_start, i]
            curr_tank = 0
    return start if total_tank >= 0 else -1
# Time: O(n), Space: O(1)
```

> **Pourquoi ce greedy est-il correct ?** Exchange argument : si curr_tank devient negatif a l'index i en partant de start, alors aucun index j dans [start, i] ne peut etre un depart valide (car partir de j donne moins d'essence accumulee que de partir de start). Donc on peut safely sauter a i+1.

---

## 9. Quand Greedy echoue — Contre-exemples classiques

| Probleme | Greedy naif | Pourquoi ca echoue |
|----------|-------------|---------------------|
| Coin change [1,3,4] target=6 | Toujours prendre la plus grande | 4+1+1 (3 pieces) vs 3+3 (2 pieces) |
| Knapsack 0/1 | Prendre l'objet avec meilleur ratio val/poids | Peut rater une combinaison meilleure |
| Travelling Salesman | Aller a la ville la plus proche | NP-hard, pas de greedy exact |
| LIS | "Greedy on value" | Ne capture pas la structure de sous-sequence |

**Regle** : si tu n'arrives pas a **prouver** que le greedy marche (par exchange ou induction), utilise du DP.

---

## 10. Decision Tree — DP avance ou Greedy ?

```
Le probleme peut-il se decomposer en sous-problemes avec etats ?
|
├── Chaque etape a plusieurs ETATS DISCRETS (hold/cash, cooldown, ...) ?
│   └── STATE MACHINE DP
|
├── La reponse depend d'un INTERVALLE [i, j] ?
│   └── INTERVAL DP — remplir par longueur croissante
|
├── Le probleme parle de PARTITIONNER / DECOUPER une sequence ?
│   └── PARTITION DP — dp[i] depend de tous les dp[j] pour j < i
|
├── Le choix optimal peut etre fait LOCALEMENT sans regret ?
│   └── GREEDY (prouve l'exchange argument d'abord)
|
└── Brute force recursive avec sous-problemes qui se recouvrent ?
    └── DP classique (Day 10)
```

**Signaux pour greedy** :
- "Maximum number of X without overlap" → interval scheduling
- "Minimum number of jumps / coins / steps" → souvent greedy ou BFS
- "Can you reach / complete" → souvent greedy en one-pass
- "Schedule / arrange / sort" → trier et faire un choix local

---

## 11. Complexites

| Algo | Temps | Espace |
|------|-------|--------|
| Stock Buy/Sell (unlimited) | O(n) | O(1) |
| Stock with Cooldown | O(n) | O(1) |
| Min Palindrome Cuts | O(n^2) | O(n^2) |
| Longest Palindromic Substring | O(n^2) | O(n^2) |
| Word Break | O(n^2) | O(n) |
| Interval Scheduling | O(n log n) | O(1) |
| Jump Game I/II | O(n) | O(1) |
| Gas Station | O(n) | O(1) |

---

## 12. Pieges courants

**Piege 1 — Greedy incorrect sur coin change**
`[1, 3, 4]` et target 6 → greedy donne 3 pieces (4+1+1), optimal est 2 (3+3). Quand les denominations ne sont pas "canoniques", il FAUT du DP.

**Piege 2 — Tri par start au lieu de end (interval scheduling)**
Tri par start peut donner plus de collisions. Toujours trier par end pour max non-overlapping.

**Piege 3 — Oublier de reinitialiser curr_tank dans Gas Station**
Si tu ne remets pas `curr_tank = 0` quand tu changes de start, tu accumules des dettes qui ne t'appartiennent pas.

**Piege 4 — Interval DP sans ordre par longueur**
Remplir `dp[i][j]` avant `dp[i+1][j-1]` donne des valeurs incorrectes. L'ordre de remplissage est CRUCIAL.

**Piege 5 — Oublier de retourner le meilleur etat a la fin**
State machine DP : il faut retourner le max des etats finaux, pas seulement le dernier etat calcule.

---

## 13. Flash Cards — Revision espacee

**Q1** : Dans un probleme de stock avec transactions illimitees, quelles sont les deux transitions de la state machine ?
> **R1** : `cash[i] = max(cash[i-1], hold[i-1] + prices[i])` (rester en cash ou vendre) et `hold[i] = max(hold[i-1], cash[i-1] - prices[i])` (continuer a detenir ou acheter). A la fin, retourner `cash`.

**Q2** : Pourquoi l'interval scheduling greedy fonctionne avec un tri par end et pas par start ?
> **R2** : Parce que l'intervalle qui se termine le plus tot laisse le maximum de "terrain libre" pour les suivants. Exchange argument : si un autre intervalle est meilleur que celui qui finit en premier, on peut le remplacer sans perte. Trier par start peut forcer a choisir un intervalle long qui bloque plusieurs autres.

**Q3** : Comment prouver qu'un algorithme glouton est correct ?
> **R3** : Exchange argument. On montre que pour toute solution optimale, on peut remplacer son premier choix par le choix glouton sans empirer le resultat. Par induction, le greedy produit donc une solution optimale.

**Q4** : Dans Gas Station, pourquoi peut-on sauter directement a l'index `i+1` quand `curr_tank < 0` ?
> **R4** : Parce qu'aucun index j entre l'ancien start et i ne peut etre un depart valide. Partir de j donne MOINS d'essence accumulee que de partir de start (qui a accumule positif jusqu'a j). Donc si start echoue a i, tous les j < i+1 echouent aussi au plus tard a i.

**Q5** : Quel est l'ordre de remplissage correct pour un interval DP ?
> **R5** : Par **longueur d'intervalle croissante**. D'abord length=1 (cas de base), puis length=2, 3, ..., n. C'est la seule maniere de garantir que `dp[i+1][j-1]` est calcule avant `dp[i][j]`.

---

## Resume — Key Takeaways

1. **State machine DP** : identifier les etats discrets, ecrire les transitions, optimiser en O(1) espace
2. **Interval DP** : remplir par longueur croissante, dp[i][j] depend de sous-intervalles plus petits
3. **Partition DP** : dp[i] combine toutes les facons de decouper s[:i] en segments
4. **Greedy correct SSI** exchange argument valide — sinon, DP
5. **Interval scheduling** : trier par **end**, pas par start
6. **Jump Game / Gas Station** : greedy one-pass en O(n), a connaitre par coeur
7. **Le piege** : les problemes gloutons qui ressemblent a du DP — savoir les reconnaitre
