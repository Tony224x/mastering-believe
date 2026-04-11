# Jour 13 — Bit Manipulation, Heaps & Tries

> **Temps estime** : 60-75 min de lecture active | **Objectif** : maitriser 3 structures/techniques de niche mais ultra-puissantes — les bits pour des astuces O(1), les heaps pour le top-K, les tries pour le prefixe

---

## 1. Pourquoi ces trois sujets ensemble

Ils partagent un point commun : **ce ne sont pas des "must-know" absolus**, mais chacun debloque une classe de problemes autrement inaccessibles en O(n log n) ou pire.

- **Bits** : operations O(1) qui remplacent des boucles entieres (flags, sets compacts)
- **Heaps** : top-K, merge K sorted, priority queues — en O(k log n) ou O(log n) par op
- **Tries** : recherche par prefixe, autocomplete, dictionnaires — O(L) independant de la taille du dico

En entretien, ces sujets apparaissent moins souvent, mais quand ils apparaissent, **ils sont la cle de la solution optimale**. Ne pas les connaitre = stuck.

---

## 2. Bit Manipulation — Les bases

### Operateurs et leur semantique

```python
# AND: bit a 1 si les deux bits sont 1
0b1100 & 0b1010  # = 0b1000 = 8

# OR: bit a 1 si au moins un des deux bits est 1
0b1100 | 0b1010  # = 0b1110 = 14

# XOR: bit a 1 si les bits different
0b1100 ^ 0b1010  # = 0b0110 = 6

# NOT: inverse tous les bits (en Python, donne un nombre negatif a cause de la representation)
~0b1100          # = -13 (complement a 2)

# Shift left: multiplication par 2^n
5 << 2           # 5 * 4 = 20

# Shift right: division par 2^n (arrondi vers le bas)
20 >> 2          # 20 / 4 = 5
```

### Les 5 tricks incontournables

```python
# 1. Verifier si le bit i est a 1
bit_i_is_set = (num >> i) & 1

# 2. Mettre le bit i a 1
num |= (1 << i)

# 3. Mettre le bit i a 0
num &= ~(1 << i)

# 4. Toggler le bit i
num ^= (1 << i)

# 5. Isoler le plus petit bit a 1 (lowest set bit)
lowest = num & (-num)

# 6. Supprimer le plus petit bit a 1 (Brian Kernighan)
num &= (num - 1)

# 7. Verifier si num est une puissance de 2
is_power_of_2 = num > 0 and (num & (num - 1)) == 0
```

### Les proprietes magiques de XOR

```python
a ^ a = 0           # XOR avec soi-meme = 0
a ^ 0 = a           # XOR avec 0 = identite
a ^ b ^ a = b       # Commutatif et associatif
```

C'est pourquoi XOR est la technique incontournable pour **trouver l'element unique** dans un tableau ou tous les autres apparaissent en paires.

---

## 3. Pattern 1 — Single Number

**Probleme** : dans un tableau ou chaque element apparait **deux fois** sauf un, trouve le celui qui apparait une seule fois. Solution en O(n) temps, O(1) espace.

```python
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num          # Les paires s'annulent, le unique reste
    return result
```

**Pourquoi ca marche** : XOR est commutatif et associatif. Donc `a ^ b ^ a ^ b ^ c = (a^a) ^ (b^b) ^ c = 0 ^ 0 ^ c = c`.

### Variante — Single Number II (chaque apparait 3 fois sauf un)

```python
def single_number_ii(nums):
    # Compter les bits position par position modulo 3
    result = 0
    for i in range(32):
        bit_count = sum((num >> i) & 1 for num in nums)
        if bit_count % 3 != 0:
            result |= (1 << i)
    return result if result < 2**31 else result - 2**32
```

---

## 4. Pattern 2 — Count Bits

**Probleme** : pour chaque nombre de 0 a n, compter le nombre de bits a 1.

### Solution O(n log n) naive

```python
def count_bits_naive(n):
    return [bin(i).count("1") for i in range(n + 1)]
```

### Solution O(n) avec DP

```python
def count_bits(n):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        # dp[i] = dp[i >> 1] + (i & 1)
        # Car i >> 1 retire le dernier bit, et (i & 1) le recupere
        dp[i] = dp[i >> 1] + (i & 1)
    return dp
```

### Autre formulation elegante

```python
def count_bits_brian(n):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        # Brian Kernighan : dp[i] = dp[i & (i-1)] + 1
        dp[i] = dp[i & (i - 1)] + 1
    return dp
```

---

## 5. Heaps — La structure de priorite

Un **heap** (tas) est un arbre binaire complet qui maintient l'invariant : le parent est toujours <= ses enfants (min-heap) ou >= (max-heap). Operations :

- `push(x)` : O(log n)
- `pop()` (extract min/max) : O(log n)
- `peek()` : O(1)
- `heapify(list)` : O(n)

Python fournit `heapq` (min-heap uniquement). Pour un max-heap, **on pousse -x et on oublie pas de negater en output**.

```python
import heapq

nums = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(nums)            # O(n) — transforme la liste en heap en place
heapq.heappush(nums, 7)        # O(log n)
smallest = heapq.heappop(nums) # O(log n), retourne le plus petit
peek = nums[0]                 # Plus petit sans le retirer

# Max-heap : pousser les negatifs
max_heap = []
for x in [3, 1, 4, 1, 5]:
    heapq.heappush(max_heap, -x)
largest = -heapq.heappop(max_heap)  # Negate pour retrouver la vraie valeur
```

---

## 6. Pattern 3 — Top K Elements

**Probleme** : trouve les k plus grands elements d'un tableau en O(n log k) (au lieu de O(n log n) en triant).

```python
import heapq

def top_k_largest(nums, k):
    # Maintenir un min-heap de taille k
    # Le plus petit des k plus grands est au sommet
    heap = []
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:
            heapq.heapreplace(heap, num)   # pop + push en une seule op
    return heap
# Time: O(n log k), Space: O(k)

# Ou plus simplement avec nlargest
def top_k_largest_simple(nums, k):
    return heapq.nlargest(k, nums)
```

**Cle** : on garde un heap de **taille k**, pas de taille n. Si un nouveau element est plus grand que le minimum du heap, on remplace. Sinon, on skip.

---

## 7. Pattern 4 — Merge K Sorted Lists

**Probleme** : fusionner k listes triees en une seule liste triee.

```python
import heapq

def merge_k_sorted(lists):
    heap = []
    # Initialiser avec le premier element de chaque liste
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))   # (value, list_idx, elem_idx)

    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        # Ajouter le prochain element de la meme liste
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result
# Time: O(N log k) ou N = nombre total d'elements
```

**Pourquoi (val, list_idx, elem_idx)** ? Si deux valeurs sont egales, heapq compare les elements suivants du tuple. Sans le `list_idx`, on risque un `TypeError` si les objets ne sont pas comparables.

---

## 8. Tries — L'arbre prefix

Un **trie** (prefix tree) stocke des strings en les decomposant caractere par caractere. Chaque noeud a un dict d'enfants + un flag `is_end`. Operations :

- `insert(word)` : O(L) ou L = longueur du mot
- `search(word)` : O(L)
- `starts_with(prefix)` : O(L)

**Gain par rapport a un set** : avec un set, `starts_with` necessite de scanner toutes les entrees O(n * L). Avec un trie, c'est O(L) independamment du nombre de mots.

### Implementation

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word):
        node = self._walk(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._walk(prefix) is not None

    def _walk(self, s):
        node = self.root
        for c in s:
            if c not in node.children:
                return None
            node = node.children[c]
        return node
```

### Usage typique

```python
trie = Trie()
for word in ["apple", "app", "apricot"]:
    trie.insert(word)

trie.search("apple")     # True
trie.search("app")       # True
trie.search("ap")        # False (pas de is_end sur "ap")
trie.starts_with("ap")   # True
```

---

## 9. Pattern 5 — Word Search II (trie + DFS)

**Probleme** : etant donne une grille et un ensemble de mots, retourne tous les mots qu'on peut former par chemin dans la grille (word search pour chaque mot, optimise par un trie).

```python
def find_words(board, words):
    # 1. Construire le trie avec tous les mots
    root = TrieNode()
    for word in words:
        node = root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True
        node.word = word  # On stocke le mot a la fin

    rows, cols = len(board), len(board[0])
    result = []

    def dfs(r, c, node):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        char = board[r][c]
        if char == "#" or char not in node.children:
            return
        node = node.children[char]
        if node.is_end:
            result.append(node.word)
            node.is_end = False   # Eviter les doublons
        board[r][c] = "#"
        dfs(r + 1, c, node)
        dfs(r - 1, c, node)
        dfs(r, c + 1, node)
        dfs(r, c - 1, node)
        board[r][c] = char

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
    return result
```

**Cle** : au lieu de lancer une DFS par mot (O(words * R * C * 4^L)), on lance une seule DFS par cellule et on suit le trie. Si le prefixe courant n'est pas dans le trie, on coupe immediatement.

---

## 10. Decision Tree

```
Type de probleme ?
|
├── Trouver l'unique / detecter un bit / puissance de 2 ?
│   └── BIT MANIPULATION (XOR, & avec -1, shift)
|
├── Top-K / K plus petits / plus grands / streaming ?
│   └── HEAP (size k)
|
├── Merge K sorted / scheduling avec priorite ?
│   └── HEAP (priority queue)
|
├── Autocomplete / prefix search / dictionnaire ?
│   └── TRIE
|
├── Plusieurs mots a chercher dans une grille ?
│   └── TRIE + DFS (word search II)
|
└── Simple comptage / frequency ?
    └── Day 3 — hash maps
```

**Signaux dans l'enonce** :

| Signal | Structure |
|--------|-----------|
| "single number", "appears once" | XOR |
| "power of 2", "count set bits" | Bit ops |
| "top K", "k largest/smallest" | Heap |
| "merge k sorted" | Heap |
| "priority queue" | Heap |
| "starts with", "prefix", "autocomplete" | Trie |
| "word dictionary in a grid" | Trie + DFS |

---

## 11. Complexites

| Operation | Temps | Espace |
|-----------|-------|--------|
| XOR tous les nums | O(n) | O(1) |
| Count bits 0..n | O(n) | O(n) |
| Heap push/pop | O(log n) | - |
| Heap heapify | O(n) | - |
| Top K | O(n log k) | O(k) |
| Merge K sorted | O(N log k) | O(k) |
| Trie insert/search | O(L) | O(sum of word lengths) |
| Word Search II | O(R*C*4^L) | O(sum) |

---

## 12. Pieges courants

**Piege 1 — heapq est min-heap uniquement**
Pour un max-heap, negater : `heapq.heappush(heap, -x)` et `-heapq.heappop(heap)`. Oublier le `-` donne des resultats faux.

**Piege 2 — Tuples dans heapq et valeurs egales**
`heapq.heappush(heap, (val, obj))` peut crasher si `obj` n'est pas comparable. Toujours mettre un tie-breaker unique (index, counter) avant l'objet.

**Piege 3 — Trie sans `is_end`**
Sans ce flag, on ne peut pas distinguer "appl" (prefixe) et "apple" (mot complet). Ne pas oublier `is_end = True` a la fin de l'insertion.

**Piege 4 — Oublier de restaurer la cellule dans Word Search II**
Meme piege que Day 12 : si on ne restaure pas `board[r][c]`, la DFS sur les cellules adjacentes echouera.

**Piege 5 — XOR sur des ints Python negatifs**
Python utilise une representation en complement a deux infinie. Si tu travailles sur des bits fixes (ex: 32-bit), il faut masquer avec `& 0xFFFFFFFF`.

---

## 13. Flash Cards — Revision espacee

**Q1** : Quelle propriete de XOR permet de trouver l'element unique dans un tableau ou les autres apparaissent en paires ?
> **R1** : XOR est commutatif (`a ^ b = b ^ a`), associatif (`(a^b)^c = a^(b^c)`), et `a ^ a = 0`. Donc en faisant XOR de tous les elements, les paires s'annulent et il ne reste que l'element unique.

**Q2** : Comment realiser Top K en O(n log k) avec un heap ?
> **R2** : On maintient un **min-heap de taille k**. Pour chaque nouvel element, si le heap contient moins de k elements, on l'ajoute. Sinon, si l'element est plus grand que le minimum du heap, on remplace le minimum par lui. A la fin, le heap contient les k plus grands.

**Q3** : Pourquoi Python n'a-t-il qu'un min-heap et comment simuler un max-heap ?
> **R3** : `heapq` ne fournit qu'un min-heap par choix de design. Pour simuler un max-heap, on inverse les valeurs : `heappush(heap, -x)` et on negate le pop : `-heappop(heap)`. C'est leger et correct pour des nombres.

**Q4** : Quel est le gain d'un Trie par rapport a un set pour chercher par prefixe ?
> **R4** : Un set necessite O(n * L) pour verifier quels mots commencent par un prefixe (scanner tous les mots). Un Trie fait ca en **O(L)** independamment du nombre de mots dans le dictionnaire, en descendant simplement dans l'arbre caractere par caractere.

**Q5** : Dans Word Search II, pourquoi utiliser un trie plutot que de lancer une DFS par mot ?
> **R5** : Avec n mots et une grille R*C, lancer une DFS par mot est O(n * R * C * 4^L). Avec un trie, on fait **une seule DFS par cellule** qui descend dans le trie : si le prefixe courant n'existe pas dans le trie, on coupe immediatement. Complexite : O(R * C * 4^L) independamment de n.

---

## Resume — Key Takeaways

1. **Bits** : XOR pour single number, `num & (num - 1)` pour le bit trick, shift pour *2^n
2. **Heaps** : `heapq` min-heap, simulate max avec `-x`, top K en O(n log k)
3. **Merge K sorted** : tuple `(val, idx, pos)` dans le heap pour eviter les collisions
4. **Tries** : O(L) par op, indispensable pour prefix queries et word search II
5. **Trie + DFS** : combinaison puissante pour rechercher plusieurs mots dans une grille
6. **Ces sujets sont situationnels** mais incontournables quand ils apparaissent
