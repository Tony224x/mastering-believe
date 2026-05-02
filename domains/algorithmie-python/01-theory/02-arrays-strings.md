# Jour 2 — Arrays & Strings : Two Pointers, Sliding Window, Prefix Sum

> **Temps estime** : 60 min de lecture active | **Objectif** : reconnaitre et appliquer les 3 patterns fondamentaux sur arrays/strings en < 2 min

---

## 1. Pourquoi arrays & strings dominent les entretiens

**>40% des questions d'entretien** portent sur des arrays ou des strings. Pourquoi ?

- Structures universelles : tout le monde les connait, pas d'avantage "librairie"
- Testent l'optimisation : la brute force est triviale, le pattern optimal ne l'est pas
- Combinent plusieurs concepts : complexite, espace, manipulation d'index
- Faciles a tester : un input, un output, pas d'ambiguite

**En entretien, 3 patterns couvrent ~80% des problemes arrays/strings :**

| Pattern | Type de probleme | Complexite typique |
|---------|------------------|--------------------|
| Two Pointers | Paires, sous-sequences, partitionnement | O(n) |
| Sliding Window | Sous-tableaux/sous-chaines contigus | O(n) |
| Prefix Sum | Sommes de plages, cumuls, requetes | O(n) build + O(1) query |

---

## 2. Rappels Python essentiels

### list vs array

```python
# Python list = tableau dynamique heterogene (comme ArrayList en Java)
arr = [1, "two", 3.0]    # O(1) acces par index, O(1) amorti append

# array.array = tableau type (rarement utilise en entretien)
import array
arr = array.array('i', [1, 2, 3])  # Que des ints, plus compact en memoire

# En entretien : TOUJOURS utiliser list. Personne n'attend array.array.
```

### String immutability — le piege #1

```python
s = "hello"
# s[0] = "H"  # TypeError! Les strings sont IMMUTABLES en Python

# Chaque concatenation cree une NOUVELLE string → O(n) par operation
result = ""
for c in s:
    result += c  # O(1+2+3+...+n) = O(n^2) total !

# Solution : liste + join
chars = list(s)           # O(n) — conversion en liste mutable
chars[0] = "H"            # O(1) — modification in place
result = "".join(chars)   # O(n) — une seule allocation
```

### Couts des operations courantes

| Operation | Cout | Piege ? |
|-----------|------|---------|
| `arr[i]` | O(1) | Non |
| `arr.append(x)` | O(1) amorti | Non |
| `arr.pop()` | O(1) | Non |
| `arr.pop(0)` | **O(n)** | Oui — decalage |
| `arr[a:b]` | **O(b-a)** | Oui — copie |
| `x in arr` | **O(n)** | Oui — scan lineaire |
| `s[a:b]` | **O(b-a)** | Oui — copie (string immutable) |
| `s1 + s2` | **O(len(s1) + len(s2))** | Oui — nouvelle string |
| `"".join(lst)` | O(total length) | Non — optimal |

> **Regle d'or** : en entretien, si tu fais `s += ...` dans une boucle, c'est un red flag de O(n^2).

---

## 3. Pattern 1 — Two Pointers

### Concept

Utiliser **deux index** (pointeurs) qui se deplacent dans le tableau selon une strategie. Deux variantes principales :

**Variante A — Convergence (left/right)** : un pointeur part du debut, l'autre de la fin, ils convergent vers le centre.

```
[1, 2, 3, 4, 5, 6, 7]
 ^                  ^
 L                  R
     →          ←       ils se rapprochent
```

**Variante B — Meme direction (slow/fast)** : les deux partent du debut, le rapide avance en eclaireur.

```
[1, 2, 3, 2, 4, 2, 5]
 ^  ^
 S  F
 →  →→→   fast avance, slow avance sous condition
```

### Quand l'utiliser

- Le tableau est **trie** (ou peut l'etre) → convergence left/right
- On cherche une **paire** qui satisfait une condition (somme, difference)
- On doit **partitionner** ou **compacter** le tableau in-place
- Le probleme parle de **palindrome** ou de **symetrie**

### Template de code — Convergence

```python
def two_pointer_converge(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        # Evaluate condition using arr[left] and arr[right]
        if condition_met(arr[left], arr[right]):
            # Process result
            return result
        elif need_bigger:
            left += 1      # Move left pointer right to increase value
        else:
            right -= 1     # Move right pointer left to decrease value
    return default_result
# Time: O(n) — each pointer moves at most n times total
# Space: O(1) — only two variables
```

### Template de code — Meme direction

```python
def two_pointer_same_direction(arr):
    slow = 0
    for fast in range(len(arr)):
        if should_keep(arr[fast]):
            arr[slow] = arr[fast]  # Place element at slow position
            slow += 1
    return slow  # New length of processed array
# Time: O(n) — fast visits each element once
# Space: O(1) — in-place modification
```

### Exemple 1 — Two Sum (sorted array)

> **Probleme** : tableau trie, trouver deux nombres dont la somme = target. Retourner leurs indices.

```python
# Brute force: O(n^2) — tester chaque paire
def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

# Two pointers: O(n) — exploiter le fait que le tableau est trie
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1       # Sum too small → need bigger number → move left right
        else:
            right -= 1      # Sum too big → need smaller number → move right left
    return []
# WHY it works: sorted order means moving left increases sum, moving right decreases it
# Time: O(n), Space: O(1)
```

### Exemple 2 — Container With Most Water

> **Probleme** : tableau de hauteurs, trouver deux barres qui forment le conteneur avec le plus d'eau.

```python
# Brute force: O(n^2)
def max_area_brute(height):
    max_water = 0
    for i in range(len(height)):
        for j in range(i + 1, len(height)):
            water = min(height[i], height[j]) * (j - i)
            max_water = max(max_water, water)
    return max_water

# Two pointers: O(n)
def max_area(height):
    left, right = 0, len(height) - 1
    max_water = 0
    while left < right:
        # Area = width * min(height_left, height_right)
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)
        # KEY INSIGHT: move the SHORTER bar inward
        # Moving the taller bar can only decrease or maintain the area
        # Moving the shorter bar MIGHT find a taller bar → potential increase
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_water
# Time: O(n), Space: O(1)
```

### Exemple 3 — Valid Palindrome

> **Probleme** : verifier si une string est un palindrome (en ignorant les non-alphanumeriques et la casse).

```python
# Pythonic but O(n) space (creates a new string)
def is_palindrome_simple(s):
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]  # Slicing creates ANOTHER copy

# Two pointers: O(n) time, O(1) space
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        # Compare (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True
# Time: O(n), Space: O(1) — no extra string created
```

---

## 4. Pattern 2 — Sliding Window

### Concept

Maintenir une **fenetre** (sous-tableau contigu) qui glisse sur le tableau. Au lieu de recalculer tout le contenu de la fenetre a chaque position, on **ajoute** l'element entrant et on **retire** l'element sortant.

```
Array:  [2, 1, 5, 1, 3, 2]
         ------             window size 3, sum=8
            ------          slide: remove 2, add 1, sum=7
               ------       slide: remove 1, add 3, sum=9
                  ------    slide: remove 5, add 2, sum=6
```

### Deux sous-types

| Type | Taille de fenetre | Quand l'utiliser |
|------|-------------------|------------------|
| **Fixed size** | Constante (donnee) | "sous-tableau de taille k" |
| **Variable size** | Dynamique (grandit/retrecit) | "plus long/court sous-tableau qui..." |

### Quand l'utiliser

- Le probleme parle de **sous-tableau contigu** ou **sous-chaine**
- On cherche un **max/min** sur un sous-ensemble contigu
- Les mots "consecutive", "contiguous", "substring", "subarray" apparaissent
- On peut decrire l'etat de la fenetre avec un petit nombre de variables

### Template — Fixed Size Window

```python
def fixed_window(arr, k):
    """Compute something over every window of size k."""
    # Step 1: compute the first window
    window_value = compute(arr[:k])   # Initial window [0..k-1]
    best = window_value

    # Step 2: slide — remove leftmost, add rightmost
    for i in range(k, len(arr)):
        window_value += arr[i]        # Add new element entering the window
        window_value -= arr[i - k]    # Remove element leaving the window
        best = max(best, window_value)

    return best
# Time: O(n) — each element is added once and removed once
# Space: O(1) — just tracking the window value
```

### Template — Variable Size Window

```python
def variable_window(arr):
    """Find the longest/shortest window satisfying a condition."""
    left = 0
    window_state = {}  # Track what's in the window (Counter, set, sum, etc.)
    best = 0

    for right in range(len(arr)):
        # EXPAND: add arr[right] to window state
        update_state_add(window_state, arr[right])

        # SHRINK: while window is invalid, remove from left
        while not is_valid(window_state):
            update_state_remove(window_state, arr[left])
            left += 1

        # UPDATE: current window [left..right] is valid
        best = max(best, right - left + 1)

    return best
# Time: O(n) — left and right each move at most n times
# Space: O(k) — where k = distinct elements in window
```

### Exemple 1 — Maximum Average Subarray (fixed window)

> **Probleme** : trouver le sous-tableau de taille k avec la plus grande moyenne.

```python
# Brute force: O(n*k) — recalculer la somme a chaque position
def max_avg_brute(nums, k):
    max_sum = float('-inf')
    for i in range(len(nums) - k + 1):
        current_sum = sum(nums[i:i+k])  # O(k) per position
        max_sum = max(max_sum, current_sum)
    return max_sum / k

# Sliding window: O(n)
def max_avg(nums, k):
    # Compute first window
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide: add right, remove left
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]  # O(1) update instead of O(k)
        max_sum = max(max_sum, window_sum)

    return max_sum / k
# Time: O(n), Space: O(1)
```

### Exemple 2 — Longest Substring Without Repeating Characters (variable window)

> **Probleme** : trouver la longueur de la plus longue sous-chaine sans caractere repete.

```python
# Brute force: O(n^3) — check every substring, verify no repeats
def length_of_longest_brute(s):
    best = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            substring = s[i:j+1]
            if len(set(substring)) == len(substring):  # No duplicates
                best = max(best, j - i + 1)
    return best

# Sliding window + hashmap: O(n)
def length_of_longest(s):
    char_index = {}   # char → its last seen index
    left = 0
    best = 0

    for right in range(len(s)):
        # If char already in window, shrink window past the previous occurrence
        if s[right] in char_index and char_index[s[right]] >= left:
            left = char_index[s[right]] + 1  # Jump left past the duplicate

        char_index[s[right]] = right  # Update last seen position
        best = max(best, right - left + 1)

    return best
# WHY char_index[s[right]] >= left: the char might exist in map but OUTSIDE current window
# Time: O(n) — right moves n times, left moves at most n times total
# Space: O(min(n, alphabet_size)) — at most 26 for lowercase English
```

### Exemple 3 — Minimum Window Substring (variable window, the classic hard)

> **Probleme** : trouver la plus petite sous-chaine de `s` contenant tous les caracteres de `t`.

```python
from collections import Counter

def min_window(s, t):
    if not t or not s:
        return ""

    # What we need: frequency count of t
    need = Counter(t)
    need_count = len(need)  # Number of UNIQUE chars we still need to satisfy

    # Window state
    window = {}
    have = 0       # Number of unique chars in window with sufficient count
    left = 0
    best = (float('inf'), 0, 0)  # (length, left, right)

    for right in range(len(s)):
        # EXPAND: add s[right]
        char = s[right]
        window[char] = window.get(char, 0) + 1

        # Check if this char's requirement is now met
        if char in need and window[char] == need[char]:
            have += 1

        # SHRINK: while all requirements are met, try to minimize window
        while have == need_count:
            # Update best
            window_len = right - left + 1
            if window_len < best[0]:
                best = (window_len, left, right)

            # Remove s[left] from window
            left_char = s[left]
            window[left_char] -= 1
            if left_char in need and window[left_char] < need[left_char]:
                have -= 1  # Lost a required char
            left += 1

    length, lo, hi = best
    return s[lo:hi+1] if length != float('inf') else ""
# Time: O(n) — left and right each advance at most n times
# Space: O(|t| + |alphabet|) — counters for need and window
```

---

## 5. Pattern 3 — Prefix Sum

### Concept

Precalculer un tableau de **sommes cumulees** pour pouvoir repondre a n'importe quelle "somme d'un sous-tableau" en O(1).

```python
# Original:    [2, 4, 1, 3, 5]
# Prefix sum:  [0, 2, 6, 7, 10, 15]
#                  ^  ^           ^
#              prefix[0]=0  prefix[2]=2+4=6  prefix[5]=sum of all=15

# Sum of arr[i..j] = prefix[j+1] - prefix[i]
# Example: sum(arr[1..3]) = arr[1]+arr[2]+arr[3] = 4+1+3 = 8
#        = prefix[4] - prefix[1] = 10 - 2 = 8  ✓
```

### Construction

```python
def build_prefix_sum(arr):
    """Build prefix sum array. prefix[i] = sum(arr[0..i-1])."""
    prefix = [0] * (len(arr) + 1)    # One extra element (prefix[0] = 0)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix
# Time: O(n), Space: O(n)

# Range query: sum(arr[i..j]) = prefix[j+1] - prefix[i]
def range_sum(prefix, i, j):
    return prefix[j + 1] - prefix[i]  # O(1) !
```

### Quand l'utiliser

- Le probleme demande la **somme d'une plage** (range sum query)
- On fait **plusieurs requetes** de somme sur le meme tableau
- On cherche un **sous-tableau dont la somme = K**
- Le mot "cumulative", "running total", ou "prefix" apparait
- On doit calculer un **produit** de tous sauf un (prefix product)

### Intro Prefix Sum 2D (pour aller plus loin)

```python
# Pour une matrice m x n, prefix[i][j] = somme du rectangle (0,0) -> (i-1, j-1)
# Construction: O(m*n)
# Query rectangle (r1,c1) -> (r2,c2): O(1) avec inclusion-exclusion
#   sum = prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1]
# Utile pour les problemes de matrices (ex: "maximal square", "range sum query 2D")
```

### Exemple 1 — Subarray Sum Equals K

> **Probleme** : compter le nombre de sous-tableaux contigus dont la somme = k.

```python
# Brute force: O(n^2) — try every subarray
def subarray_sum_brute(nums, k):
    count = 0
    for i in range(len(nums)):
        current_sum = 0
        for j in range(i, len(nums)):
            current_sum += nums[j]
            if current_sum == k:
                count += 1
    return count

# Prefix sum + hashmap: O(n)
def subarray_sum(nums, k):
    count = 0
    prefix = 0              # Running prefix sum
    seen = {0: 1}           # prefix_sum → number of times we've seen it
    # seen[0] = 1 because an empty prefix (before index 0) has sum 0

    for num in nums:
        prefix += num
        # If (prefix - k) was seen before, there's a subarray summing to k
        # Because: prefix[j] - prefix[i] = k  ⟹  prefix[i] = prefix[j] - k
        if prefix - k in seen:
            count += seen[prefix - k]
        seen[prefix] = seen.get(prefix, 0) + 1

    return count
# KEY INSIGHT: we don't need the actual prefix array, just a running sum + hashmap
# Time: O(n), Space: O(n)
```

### Exemple 2 — Product of Array Except Self

> **Probleme** : pour chaque element, calculer le produit de tous les autres elements, SANS division.

```python
# Brute force: O(n^2)
def product_except_self_brute(nums):
    n = len(nums)
    result = [1] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                result[i] *= nums[j]
    return result

# Prefix/suffix product: O(n) time, O(1) extra space (output doesn't count)
def product_except_self(nums):
    n = len(nums)
    result = [1] * n

    # Pass 1: left-to-right prefix product
    # result[i] = product of all elements to the LEFT of i
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]

    # Pass 2: right-to-left suffix product
    # Multiply result[i] by product of all elements to the RIGHT of i
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]

    return result
# result[i] = (product of left) * (product of right) = product of all except self
# Time: O(n), Space: O(1) extra (result array is the output)
```

---

## 6. Decision Tree — Quel pattern utiliser ?

```
Le probleme porte sur un array/string contigu ?
│
├── NON → Probablement un autre pattern (hash map, graph, DP...)
│
└── OUI → On cherche quoi ?
    │
    ├── Une PAIRE d'elements (somme, difference, distance) ?
    │   └── Le tableau est trie ou peut l'etre ?
    │       ├── OUI → TWO POINTERS (convergence)
    │       └── NON → Hash map (Two Sum classique)
    │
    ├── Un SOUS-TABLEAU/SOUS-CHAINE optimal (max, min, exact) ?
    │   ├── Taille fixe donnee → SLIDING WINDOW (fixed)
    │   ├── Taille variable → SLIDING WINDOW (variable)
    │   └── Somme = K, ou somme dans une plage → PREFIX SUM + hash map
    │
    ├── Partitionner / compacter in-place ?
    │   └── TWO POINTERS (meme direction)
    │
    ├── Palindrome ou symetrie ?
    │   └── TWO POINTERS (convergence)
    │
    └── Somme de plage / requetes multiples ?
        └── PREFIX SUM
```

**Raccourcis mentaux** :

| Signal dans l'enonce | Pattern |
|---------------------|---------|
| "sorted array" + "pair" | Two Pointers |
| "in-place", "remove duplicates" | Two Pointers (same direction) |
| "palindrome" | Two Pointers (converge) |
| "subarray of size k" | Sliding Window (fixed) |
| "longest/shortest substring" | Sliding Window (variable) |
| "subarray sum equals K" | Prefix Sum + hashmap |
| "product except self" | Prefix/Suffix product |
| "range sum query" | Prefix Sum |

---

## 7. Complexites typiques

| Pattern | Temps | Espace | Commentaire |
|---------|-------|--------|-------------|
| Two Pointers | O(n) | O(1) | Pas de structure auxiliaire |
| Sliding Window (fixed) | O(n) | O(1) ou O(k) | k = taille de fenetre |
| Sliding Window (variable) | O(n) | O(k) | k = taille alphabet/elements distincts |
| Prefix Sum (build) | O(n) | O(n) | Le tableau prefix |
| Prefix Sum (query) | O(1) | — | Apres construction |
| Prefix Sum + hashmap | O(n) | O(n) | Pour "subarray sum = K" |

> **Point cle** : les trois patterns transforment des problemes O(n^2) ou O(n^3) brute-force en O(n). C'est exactement ce qu'on attend en entretien — montrer la brute force, puis optimiser avec le bon pattern.

---

## 8. Flash Cards — Revision espacee

> **Methode** : couvrir la reponse, repondre a voix haute, puis verifier. Revenir dans 1 jour, 3 jours, 7 jours.

**Q1** : Tu as un tableau trie et tu cherches une paire dont la somme = target. Quel pattern, quelle complexite ?
> **R1** : Two Pointers (convergence). O(n) temps, O(1) espace. Left part de 0, right de n-1. Si somme trop petite, left++, sinon right--.

**Q2** : Quelle est la difference entre un sliding window fixed et variable ?
> **R2** : Fixed = taille de fenetre constante (donnee par l'enonce, ex: "sous-tableau de taille k"). Variable = la fenetre grandit/retrecit selon une condition (ex: "plus longue sous-chaine sans doublon"). Les deux sont O(n).

**Q3** : Comment calculer la somme de arr[i..j] en O(1) apres un preprocessing O(n) ?
> **R3** : Prefix Sum. Construire prefix[] ou prefix[k] = sum(arr[0..k-1]). Puis sum(arr[i..j]) = prefix[j+1] - prefix[i]. Le prefix[0] = 0 est crucial pour gerer le cas i=0.

**Q4** : Pourquoi le sliding window variable est-il O(n) alors qu'il a deux pointeurs qui bougent ?
> **R4** : Chaque pointeur (left et right) avance **au plus n fois** au total (jamais de retour en arriere). Donc le nombre total d'operations est au plus 2n = O(n). Ce n'est PAS un O(n^2) malgre la boucle while interne.

**Q5** : Dans "Subarray Sum Equals K", pourquoi ne peut-on pas utiliser un simple sliding window ?
> **R5** : Le sliding window variable suppose qu'agrandir la fenetre augmente toujours la "mesure" (ou la retrecir la diminue). Avec des nombres negatifs, agrandir peut diminuer la somme. On utilise donc prefix sum + hashmap : si prefix[j] - prefix[i] = k, alors le sous-tableau i..j-1 somme a k.

---

## Resume — Key Takeaways

1. **Two Pointers** : ideal pour les tableaux tries, paires, palindromes, partitionnement in-place → O(n) temps, O(1) espace
2. **Sliding Window** : ideal pour les sous-tableaux/sous-chaines contigus avec une contrainte → O(n) temps
3. **Prefix Sum** : ideal pour les sommes de plages et les requetes multiples → O(n) build, O(1) query
4. **Brute force d'abord** : toujours presenter la solution naive, puis optimiser avec le pattern
5. **Decision tree** : apprendre les signaux de l'enonce pour choisir le bon pattern en < 30 secondes
6. **String en Python** : immutable ! Utiliser list + join, jamais += dans une boucle

---

## Pour aller plus loin

Ressources canoniques sur les patterns arrays/strings :

- **NeetCode — Arrays & Hashing roadmap** (Navi, ex-Google) — section dediee avec videos commentees sur Two Pointers, Sliding Window, Prefix Sum. La meilleure ressource visuelle pour ces patterns. https://neetcode.io/roadmap
- **Cracking the Coding Interview** (Gayle Laakmann McDowell, 6th ed) — Ch 1 : 9 problemes arrays/strings types entretien avec solutions detaillees. https://www.crackingthecodinginterview.com/
- **CLRS — Introduction to Algorithms** (4th ed, MIT Press 2022) — Ch 32 (String Matching) pour les fondements theoriques sur la manipulation de chaines. https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
- **Elements of Programming Interviews in Python** (Aziz, Lee, Prakash) — chapitres "Arrays" et "Strings" : 60+ problemes denses, version Python.
