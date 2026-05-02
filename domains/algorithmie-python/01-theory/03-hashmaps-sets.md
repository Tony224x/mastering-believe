# Jour 3 — Hash Maps & Sets : Frequency Counting, Grouping, Two-Sum Patterns

> **Temps estime** : 60 min de lecture active | **Objectif** : utiliser les hash maps comme arme principale pour resoudre 80% des problemes d'entretien en O(n)

---

## 1. Pourquoi les hash maps sont la structure la plus importante en entretien

**Si tu ne devais maitriser qu'UNE seule structure de donnees pour les entretiens, ce serait le hash map.**

Pourquoi ? Parce que le hash map est l'outil universel de l'optimisation :

| Probleme brute force | Avec hash map | Gain |
|----------------------|---------------|------|
| Chercher un element dans une liste → O(n) | Lookup dans un dict → O(1) | n → 1 |
| Trouver une paire avec somme cible → O(n^2) | Stocker les complements → O(n) | n^2 → n |
| Compter les frequences → O(n^2) | Counter/defaultdict → O(n) | n^2 → n |
| Grouper des elements par propriete → O(n^2) | defaultdict(list) → O(n) | n^2 → n |
| Detecter un doublon → O(n^2) ou O(n log n) | Set → O(n) | n^2 → n |

**Le principe fondamental** : on **echange de l'espace contre du temps** (trade space for time). Un hash map utilise O(n) memoire supplementaire pour transformer des lookups O(n) en O(1).

> **En entretien** : quand tu es bloque sur une brute force O(n^2), demande-toi "est-ce que je peux stocker quelque chose dans un dict pour eviter le scan interieur ?"

---

## 2. Internals Python — Comment dict et set fonctionnent sous le capot

### Hash function

```python
# Chaque objet Python a un hash (sauf les mutables: list, dict, set)
hash(42)          # → 42 (int se hash a lui-meme)
hash("hello")     # → un entier deterministe pour la session
hash((1, 2, 3))   # → ok, les tuples sont hashables
# hash([1, 2])    # TypeError: unhashable type: 'list'

# La hash function mappe un objet a un INDEX dans le tableau interne
# index = hash(key) % table_size
```

### Collision handling

Python utilise le **open addressing** (probing) et non le chainage :
- Si la case est occupee, Python cherche la prochaine case libre avec une formule de probing
- Ce probing est pseudo-aleatoire (pas lineaire) pour eviter le clustering
- Resultat : O(1) **amorti** pour insert/lookup/delete, mais O(n) worst case si tous les hash collisionnent

### Load factor et redimensionnement

```python
# Python redimensionne le dict quand il est rempli a 2/3 (load factor = 0.66)
# Le redimensionnement double (environ) la taille du tableau
# C'est pourquoi insert est O(1) AMORTI : la plupart des inserts sont O(1),
# mais de temps en temps un insert declenche un resize O(n)

# En pratique : ne JAMAIS mentionner le worst case O(n) en entretien
# sauf si l'interviewer demande. Dire "O(1) lookup amorti" est suffisant.
```

### Pourquoi l'ordre est preserve depuis Python 3.7

```python
# Depuis Python 3.7 (CPython 3.6 en implementation), les dicts preservent
# l'ordre d'insertion. C'est garanti par le langage.
d = {"b": 2, "a": 1, "c": 3}
list(d.keys())   # → ['b', 'a', 'c']  — ordre d'insertion, PAS alphabetique

# IMPORTANT en entretien : tu peux compter sur cet ordre.
# Mais si on te demande un "ordered dict" avec d'autres proprietes
# (ex: move_to_end, popitem(last=True)), utilise collections.OrderedDict
```

### dict vs set

```python
# dict = {key: value}  — association cle/valeur
# set  = {key}         — collection de cles uniques (sans valeurs)
# Sous le capot : MEME structure (hash table), le set est un dict sans valeurs

# Quand utiliser quoi :
# - Tu as besoin de STOCKER une info par cle → dict
# - Tu as juste besoin de SAVOIR si un element existe → set

# Couts identiques :
# | Operation      | dict       | set        |
# |----------------|------------|------------|
# | Lookup (in)    | O(1)       | O(1)       |
# | Insert         | O(1)       | O(1)       |
# | Delete         | O(1)       | O(1)       |
# | Iteration      | O(n)       | O(n)       |
```

### Les outils Python indispensables

```python
from collections import Counter, defaultdict

# Counter — compteur de frequences optimise
freq = Counter("abracadabra")
# Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})
freq.most_common(2)    # [('a', 5), ('b', 2)]
freq['z']              # 0 — pas de KeyError !

# defaultdict — dict avec valeur par defaut
dd = defaultdict(int)
dd['new_key'] += 1     # Pas de KeyError, cree 0 puis incremente → 1

dd = defaultdict(list)
dd['group'].append(42) # Pas de KeyError, cree [] puis append

# dict.get — alternative sans import
d = {}
d['key'] = d.get('key', 0) + 1  # Equivalent a defaultdict(int)

# dict.setdefault — cree la cle si absente et retourne la valeur
d = {}
d.setdefault('key', []).append(42)  # Equivalent a defaultdict(list)
```

---

## 3. Pattern 1 — Frequency Counting

### Concept

Compter combien de fois chaque element apparait. C'est la base de dizaines de problemes d'entretien.

```python
# Template generique
from collections import Counter

def solve_with_frequency(arr):
    freq = Counter(arr)   # O(n) — une seule passe
    # Utiliser freq pour repondre a la question
    # freq[x] = nombre d'occurrences de x
```

### Quand l'utiliser

- Le probleme parle d'**anagrammes**, de **permutations**, de **frequences**
- On cherche l'element le **plus/moins frequent**
- On doit verifier si deux collections ont la **meme composition**
- Les mots "frequency", "count", "occurrence", "how many" apparaissent

### Exemple 1 — Valid Anagram

```python
# Deux strings sont des anagrammes si elles ont les memes frequences de caracteres
def is_anagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)
# Time: O(n), Space: O(n) — n = len(s) + len(t)

# Alternative sans import :
def is_anagram_manual(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    for c in t:
        freq[c] = freq.get(c, 0) - 1
        if freq[c] < 0:
            return False
    return True
```

### Exemple 2 — Top K Frequent Elements

```python
def top_k_frequent(nums, k):
    """Return the k most frequent elements."""
    freq = Counter(nums)
    return [x for x, _ in freq.most_common(k)]
# Time: O(n + k log n) — Counter is O(n), most_common uses a heap
# Space: O(n)

# Optimal O(n) with bucket sort :
def top_k_frequent_bucket(nums, k):
    freq = Counter(nums)
    # Bucket: index = frequency, value = list of elements with that frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)
    # Collect from highest frequency buckets
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    return result
# Time: O(n), Space: O(n)
```

### Exemple 3 — First Unique Character

```python
def first_unique_char(s: str) -> int:
    """Return index of first non-repeating character, or -1."""
    freq = Counter(s)               # O(n) — count all chars
    for i, c in enumerate(s):       # O(n) — find the first with count 1
        if freq[c] == 1:
            return i
    return -1
# Time: O(n), Space: O(1) — au plus 26 caracteres pour des lowercase letters
```

---

## 4. Pattern 2 — Grouping / Bucketing

### Concept

Regrouper des elements par une **propriete commune** calculee via une cle de hash.

```python
# Template generique
from collections import defaultdict

def group_by_property(items):
    groups = defaultdict(list)
    for item in items:
        key = compute_key(item)  # La "signature" de groupage
        groups[key].append(item)
    return list(groups.values())  # ou dict si besoin des cles
```

### Quand l'utiliser

- Le probleme demande de **grouper** ou **classer** des elements
- On doit trouver des elements avec une **propriete partagee** (anagrammes, meme somme, meme hash)
- Les mots "group", "classify", "bucket", "partition by" apparaissent

### Exemple 1 — Group Anagrams

```python
def group_anagrams(strs: list[str]) -> list[list[str]]:
    """Group strings that are anagrams of each other."""
    groups = defaultdict(list)
    for s in strs:
        # KEY INSIGHT: all anagrams produce the same sorted string
        key = tuple(sorted(s))  # "eat" → ('a','e','t'), "tea" → ('a','e','t')
        groups[key].append(s)
    return list(groups.values())
# Time: O(n * k log k) — n strings of max length k (sorting each string)
# Space: O(n * k)

# Optimization: count-based key (avoids sorting) — O(n * k)
def group_anagrams_fast(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    for s in strs:
        # Use character frequency as key instead of sorting
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        key = tuple(count)  # (1,0,0,...,1,...,1) for "eat"
        groups[key].append(s)
    return list(groups.values())
# Time: O(n * k) — no sorting needed
```

### Exemple 2 — Group by Digit Sum

```python
def group_by_digit_sum(nums: list[int]) -> dict[int, list[int]]:
    """Group numbers that have the same digit sum."""
    groups = defaultdict(list)
    for num in nums:
        digit_sum = sum(int(d) for d in str(num))
        groups[digit_sum].append(num)
    return dict(groups)
# Example: [18, 36, 27, 45, 99] → {9: [18, 36, 27, 45], 18: [99]}
```

---

## 5. Pattern 3 — Two-Sum & Complement Lookup

### Concept

Utiliser un hash map comme **table de lookup** pour chercher le complement d'un element courant. Au lieu de tester chaque paire (O(n^2)), on stocke les elements vus et on cherche le complement en O(1).

```python
# Template generique
def find_pair(arr, target):
    seen = {}  # element → index (ou element → True)
    for i, num in enumerate(arr):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]  # Found the pair!
        seen[num] = i
    return []
```

### Quand l'utiliser

- On cherche une **paire** d'elements avec une relation cible (somme, difference, rapport)
- Le tableau n'est **PAS trie** (sinon utiliser two pointers)
- On peut **generaliser** : 3Sum = fixer un element + Two Sum, 4Sum = fixer deux + Two Sum

### Exemple 1 — Two Sum (le plus celebre)

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices of two numbers that sum to target."""
    seen = {}   # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i    # Store AFTER checking (avoid using same element twice)
    return []
# Time: O(n), Space: O(n)

# IMPORTANT: on stocke APRES le check pour eviter de trouver le meme element
# Exemple : nums=[3,3], target=6
#   i=0: complement=3, not in seen → store seen[3]=0
#   i=1: complement=3, 3 in seen → return [0, 1] ✓
```

### Exemple 2 — 4Sum II (multi-array)

```python
def four_sum_count(nums1, nums2, nums3, nums4):
    """Count tuples (i,j,k,l) where nums1[i]+nums2[j]+nums3[k]+nums4[l]=0."""
    # Split into two groups: (nums1+nums2) and (nums3+nums4)
    ab_sums = Counter()
    for a in nums1:
        for b in nums2:
            ab_sums[a + b] += 1

    count = 0
    for c in nums3:
        for d in nums4:
            complement = -(c + d)
            count += ab_sums[complement]   # 0 if not found (Counter behavior)

    return count
# Time: O(n^2) — much better than brute force O(n^4)
# Space: O(n^2) — storing all AB sums
```

### Exemple 3 — Count Pairs with Difference K

```python
def count_pairs_with_diff(nums: list[int], k: int) -> int:
    """Count pairs (i, j) where |nums[i] - nums[j]| == k and i < j."""
    freq = Counter(nums)
    count = 0

    for num in freq:
        # For each number, check if num + k exists
        if k > 0:
            if num + k in freq:
                count += freq[num] * freq[num + k]
        elif k == 0:
            # Pairs within the same number: C(n,2) = n*(n-1)/2
            count += freq[num] * (freq[num] - 1) // 2

    return count
# Time: O(n), Space: O(n)
# Note: k < 0 is handled by taking abs(k)
```

---

## 6. Pattern 4 — Seen Set / Visited

### Concept

Utiliser un set pour tracker les elements **deja vus** et detecter des conditions comme :
- Doublons
- Cycles
- Elements uniques dans un stream

```python
# Template generique
def detect_with_set(arr):
    seen = set()
    for item in arr:
        if item in seen:
            # Doublon detecte !
            return True
        seen.add(item)
    return False
```

### Quand l'utiliser

- Detection de **doublons** avec une contrainte (dans une fenetre, premier doublon, etc.)
- **Cycle detection** (Floyd's algorithm est O(1) espace, mais un set est plus simple)
- Elements **uniques** dans un flux de donnees
- Les mots "duplicate", "seen before", "unique", "already visited" apparaissent

### Exemple 1 — Contains Duplicate Within K

```python
def contains_nearby_duplicate(nums: list[int], k: int) -> bool:
    """Check if there are two distinct indices i,j with nums[i]==nums[j] and |i-j| <= k."""
    window = set()   # Elements in the current window of size k
    for i, num in enumerate(nums):
        if num in window:
            return True     # Duplicate found within window
        window.add(num)
        # Maintain window size at most k
        if len(window) > k:
            window.remove(nums[i - k])  # Remove the element leaving the window
    return False
# Time: O(n), Space: O(k) — sliding window as a set
```

### Exemple 2 — Longest Consecutive Sequence

```python
def longest_consecutive(nums: list[int]) -> int:
    """Find the length of the longest consecutive elements sequence. Must be O(n)."""
    num_set = set(nums)   # O(n) — for O(1) lookups
    best = 0

    for num in num_set:
        # Only start counting from the BEGINNING of a sequence
        # num-1 NOT in set means 'num' is the start of a new sequence
        if num - 1 not in num_set:
            current = num
            length = 1
            while current + 1 in num_set:
                current += 1
                length += 1
            best = max(best, length)

    return best
# Time: O(n) — each number is visited at most twice (once in outer loop, once in while)
# Space: O(n) — the set
# KEY INSIGHT: the "if num - 1 not in num_set" check ensures we only start
# counting from the beginning of each sequence, making the total work O(n)
```

---

## 7. Pattern 5 — Index Mapping

### Concept

Stocker des **indices** (pas juste des valeurs) dans le hash map pour des lookups ulterieurs : premier/dernier index d'apparition, distance entre occurrences, etc.

```python
# Template generique
def index_mapping(arr):
    first_seen = {}   # element → first index
    for i, val in enumerate(arr):
        if val not in first_seen:
            first_seen[val] = i
    return first_seen
```

### Quand l'utiliser

- On doit retrouver la **position** d'un element deja vu
- Le probleme parle de "first occurrence", "last occurrence", "distance between occurrences"
- On combine avec d'autres patterns (ex: Two Sum retourne des indices)

### Exemple — Shortest Subarray with All Values

```python
def shortest_subarray_with_target_values(nums, target_values):
    """Find the shortest contiguous subarray containing all target values."""
    need = set(target_values)
    last_pos = {}     # value → last seen index
    best = float('inf')

    for i, num in enumerate(nums):
        if num in need:
            last_pos[num] = i
        # Check if we have all target values
        if len(last_pos) == len(need):
            # Shortest subarray = from earliest last_pos to current
            earliest = min(last_pos.values())
            best = min(best, i - earliest + 1)

    return best if best != float('inf') else -1
```

---

## 8. Decision Tree — Quel pattern utiliser ?

```
Le probleme implique des LOOKUPS ou du COMPTAGE ?
|
├── NON → Probablement arrays/strings (two pointers, sliding window, prefix sum)
|
└── OUI → Quel type de probleme ?
    |
    ├── Compter les FREQUENCES / OCCURRENCES ?
    │   └── FREQUENCY COUNTING (Counter, defaultdict(int))
    │       Ex: anagram check, top-K frequent, first unique
    |
    ├── GROUPER des elements par une propriete ?
    │   └── GROUPING (defaultdict(list))
    │       Ex: group anagrams, group by digit sum, bucket sort
    |
    ├── Trouver une PAIRE avec une relation cible ?
    │   ├── Tableau TRIE → Two Pointers (Day 2)
    │   └── Tableau NON trie → TWO-SUM PATTERN (dict as lookup)
    │       Ex: two sum, pair with diff K, 4sum II
    |
    ├── Detecter des DOUBLONS ou elements DEJA VUS ?
    │   └── SEEN SET (set)
    │       Ex: contains duplicate, longest consecutive, cycle detection
    |
    └── Stocker/retrouver des POSITIONS ?
        └── INDEX MAPPING (dict: value → index)
            Ex: two sum (return indices), first/last occurrence
```

**Raccourcis mentaux** :

| Signal dans l'enonce | Pattern |
|---------------------|---------|
| "anagram", "permutation" | Frequency Counting |
| "frequency", "most common", "top K" | Frequency Counting |
| "group by", "classify", "bucket" | Grouping |
| "two sum", "pair with sum/diff" | Two-Sum / Complement |
| "duplicate", "seen before" | Seen Set |
| "first/last occurrence", "index of" | Index Mapping |
| "consecutive sequence" | Seen Set |
| "subarray sum = K" | Prefix Sum + hashmap (Day 2) |

---

## 9. Complexites et pieges

### Complexites

| Operation | dict/set | Commentaire |
|-----------|----------|-------------|
| Lookup (`x in d`) | O(1) amorti | O(n) worst case (hash collisions) |
| Insert (`d[k] = v`) | O(1) amorti | Resize toutes les ~n/3 inserts |
| Delete (`del d[k]`) | O(1) amorti | |
| Iteration | O(n) | Parcourir toutes les cles |
| `Counter(arr)` | O(n) | Une seule passe |
| `most_common(k)` | O(n + k log n) | Heap interne |

### Pieges courants

**Piege 1 — Cles mutables**
```python
# Les listes ne sont PAS hashables → ne peuvent pas etre des cles
d = {}
# d[[1, 2]] = "value"  # TypeError: unhashable type: 'list'
d[tuple([1, 2])] = "value"  # OK — convertir en tuple
d[frozenset([1, 2])] = "value"  # OK pour les ensembles comme cles
```

**Piege 2 — defaultdict vs get**
```python
# defaultdict CREE la cle si elle n'existe pas
from collections import defaultdict
dd = defaultdict(int)
dd['x']  # Cree la cle 'x' avec valeur 0 !
len(dd)  # 1 — la cle existe maintenant

# dict.get ne cree PAS la cle
d = {}
d.get('x', 0)  # Retourne 0, mais 'x' n'est PAS dans d
len(d)  # 0
```

**Piege 3 — Modifier un dict pendant l'iteration**
```python
d = {'a': 1, 'b': 2, 'c': 3}
# for k in d:
#     if d[k] < 2:
#         del d[k]  # RuntimeError: dictionary changed size during iteration

# Solution : iterer sur une copie des cles
for k in list(d.keys()):
    if d[k] < 2:
        del d[k]  # OK — on itere sur la copie
```

**Piege 4 — Confondre Counter et dict pour la valeur par defaut**
```python
from collections import Counter
c = Counter()
c['missing']  # → 0 (PAS KeyError — Counter retourne 0 par defaut)

d = {}
# d['missing']  # KeyError !
d.get('missing', 0)  # → 0 (methode safe)
```

---

## 10. Flash Cards — Revision espacee

> **Methode** : couvrir la reponse, repondre a voix haute, puis verifier. Revenir dans 1 jour, 3 jours, 7 jours.

**Q1** : Quelle est la complexite de lookup dans un dict Python ? Quel est le worst case et pourquoi ?
> **R1** : O(1) amorti. Worst case O(n) si toutes les cles ont le meme hash (collision totale). En pratique, CPython utilise un probing pseudo-aleatoire qui rend ce cas quasi impossible.

**Q2** : Comment verifier que deux strings sont des anagrammes en O(n) ?
> **R2** : Comparer Counter(s) == Counter(t). Ou : construire un dict de frequences sur s, puis decrementer pour t — si une frequence passe en negatif, ce n'est pas un anagramme.

**Q3** : Dans le Two Sum classique (tableau non trie), pourquoi stocker la valeur dans le dict APRES le check du complement ?
> **R3** : Pour eviter d'utiliser le meme element deux fois. Si on stocke d'abord et cherche ensuite, nums=[3], target=6 trouverait une paire (0,0) invalide. En stockant apres, on ne regarde que les elements precedents.

**Q4** : Comment Longest Consecutive Sequence atteint O(n) malgre la boucle while imbriquee ?
> **R4** : Le check `if num - 1 not in num_set` garantit qu'on ne demarre une sequence que depuis son debut. Chaque nombre est visite au plus 2 fois (une fois dans le for, une fois dans un while). Donc total = O(n).

**Q5** : Quelle est la difference entre `defaultdict(int)`, `Counter()`, et `dict.get()` pour compter des frequences ?
> **R5** : Les trois permettent de compter, mais : `defaultdict(int)` cree la cle avec 0 si absente, `Counter` fait pareil ET offre `most_common()` + supporte les operations ensemblistes, `dict.get(k, 0)` ne cree PAS la cle (lecture seule). En entretien, `Counter` est le plus idiomatique pour les frequences.

---

## Resume — Key Takeaways

1. **Hash map = O(1) lookup** : c'est l'outil qui transforme les brute force O(n^2) en O(n)
2. **5 patterns** : Frequency Counting, Grouping, Two-Sum/Complement, Seen Set, Index Mapping
3. **Counter et defaultdict** sont tes meilleurs amis en Python — utilise-les, l'interviewer apprecie le code idiomatique
4. **Trade space for time** : la quasi-totalite des optimisations avec hash map ajoutent O(n) espace
5. **Decision rapide** : anagramme/frequence → Counter, grouper → defaultdict(list), paire → dict lookup, doublon → set
6. **Pieges** : cles immutables uniquement, ne pas modifier pendant l'iteration, defaultdict cree les cles

---

## Pour aller plus loin

Ressources canoniques sur les hash tables :

- **CLRS — Introduction to Algorithms** (4th ed, MIT Press 2022) — Ch 11 (Hash Tables) : fonctions de hachage, collision resolution, analyse amortie de l'open addressing. La theorie complete. https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
- **NeetCode — Arrays & Hashing roadmap** — la majorite des problemes "hash map" entretien sont dans cette section, avec template Python (Counter, defaultdict). https://neetcode.io/roadmap
- **MIT 6.006 — Introduction to Algorithms** (Erik Demaine, MIT OCW Spring 2020) — Lec. 8-10 (Hashing, Open Addressing, Cryptographic Hashing) : derivation pedagogique du O(1) amorti. https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/
- **The Algorithm Design Manual** (Skiena, 3rd ed 2020) — Ch 3 : compromis entre BST et hash table avec exemples reels.
