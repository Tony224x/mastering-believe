# Sprint Exercices — Jour 7

**But** : resoudre les 10 problemes ci-dessous en **moins de 15 minutes chacun**, soit un maximum de 2h30 au total. Chronometre-toi strictement pour chaque probleme.

**Regles** :
- Ne pas regarder les solutions avant d'avoir tente chaque probleme.
- Si tu bloques a 12 min, annonce a voix haute ton brute force et ecris-le.
- Apres chaque probleme, note dans un cahier : pattern utilise, temps pris, bugs rencontres.
- A la fin, compare avec les solutions de `solutions/07-sprint-exercices.py`.

---

## P1 — Two Sum

### Objectif (Easy — 5 min)

Etant donne un tableau d'entiers `nums` non trie et un entier `target`, retourne les indices des deux elements tels que leur somme soit egale a `target`. Il y a exactement une solution et tu ne peux pas utiliser le meme element deux fois.

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    pass
```

### Tests

```python
assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([3, 2, 4], 6) == [1, 2]
assert two_sum([3, 3], 6) == [0, 1]
```

---

## P2 — Valid Anagram

### Objectif (Easy — 3 min)

Etant donne deux strings `s` et `t`, retourne `True` si `t` est un anagramme de `s`, `False` sinon.

```python
def is_anagram(s: str, t: str) -> bool:
    pass
```

### Tests

```python
assert is_anagram("anagram", "nagaram") == True
assert is_anagram("rat", "car") == False
assert is_anagram("", "") == True
```

---

## P3 — Valid Parentheses

### Objectif (Easy — 5 min)

Etant donne une string `s` contenant uniquement `()[]{}`, retourne `True` si elle est bien formee.

```python
def is_valid_parens(s: str) -> bool:
    pass
```

### Tests

```python
assert is_valid_parens("()[]{}") == True
assert is_valid_parens("([)]") == False
assert is_valid_parens("") == True
assert is_valid_parens("((") == False
```

---

## P4 — Best Time to Buy and Sell Stock

### Objectif (Easy — 5 min)

Etant donne un tableau `prices` ou `prices[i]` est le prix d'une action le jour `i`, retourne le profit maximum que tu peux realiser en achetant UN jour et en vendant UN jour ulterieur. Si aucun profit n'est possible, retourne `0`.

```python
def max_profit(prices: list[int]) -> int:
    pass
```

### Tests

```python
assert max_profit([7, 1, 5, 3, 6, 4]) == 5
assert max_profit([7, 6, 4, 3, 1]) == 0
assert max_profit([]) == 0
assert max_profit([1]) == 0
```

---

## P5 — Reverse Linked List

### Objectif (Easy — 5 min)

Inverse une linked list singly et retourne la nouvelle tete.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_list(head):
    pass
```

### Tests

```python
def build(vals):
    dummy = ListNode(0)
    t = dummy
    for v in vals:
        t.next = ListNode(v)
        t = t.next
    return dummy.next

def to_list(h):
    out = []
    while h:
        out.append(h.val)
        h = h.next
    return out

assert to_list(reverse_list(build([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
assert to_list(reverse_list(build([1]))) == [1]
assert reverse_list(None) is None
```

---

## P6 — Longest Substring Without Repeating Characters

### Objectif (Medium — 10 min)

Etant donne une string `s`, trouve la longueur de la plus longue substring sans caractere repete.

```python
def length_of_longest_substring(s: str) -> int:
    pass
```

### Tests

```python
assert length_of_longest_substring("abcabcbb") == 3
assert length_of_longest_substring("bbbbb") == 1
assert length_of_longest_substring("pwwkew") == 3
assert length_of_longest_substring("") == 0
```

---

## P7 — Group Anagrams

### Objectif (Medium — 10 min)

Etant donne une liste de strings `strs`, groupe ensemble celles qui sont des anagrammes les unes des autres. Retourne une liste de groupes (l'ordre des groupes et l'ordre dans chaque groupe peuvent varier).

```python
def group_anagrams(strs: list[str]) -> list[list[str]]:
    pass
```

### Tests

```python
result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
# Normalize for comparison
normalized = sorted([sorted(g) for g in result])
expected = sorted([sorted(g) for g in [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]])
assert normalized == expected
```

---

## P8 — Top K Frequent Elements

### Objectif (Medium — 10 min)

Etant donne un tableau `nums` d'entiers et un entier `k`, retourne les `k` elements les plus frequents. Tu peux retourner le resultat dans n'importe quel ordre.

```python
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    pass
```

### Tests

```python
assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
assert top_k_frequent([1], 1) == [1]
```

---

## P9 — Search in Rotated Sorted Array

### Objectif (Medium — 12 min)

On te donne un tableau `nums` trie en ordre croissant qui a ensuite ete **rote** une fois autour d'un pivot inconnu. Ainsi, `[0,1,2,4,5,6,7]` peut devenir `[4,5,6,7,0,1,2]`.

Etant donne `nums` et un entier `target`, retourne l'index de `target` s'il existe, sinon `-1`. Tu dois ecrire un algorithme en **O(log n)**.

```python
def search_rotated(nums: list[int], target: int) -> int:
    pass
```

### Tests

```python
assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
assert search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1
assert search_rotated([1], 0) == -1
assert search_rotated([1], 1) == 0
```

---

## P10 — Number of Islands

### Objectif (Medium — 12 min)

Etant donne une grille 2D de caracteres `'1'` (terre) et `'0'` (eau), compte le nombre d'iles. Une ile est entouree d'eau et est formee de terres connectees **horizontalement ou verticalement** (pas en diagonale). Tu peux supposer que les 4 bords de la grille sont entoures d'eau.

```python
def num_islands(zone: list[list[str]]) -> int:
    pass
```

### Tests

```python
grid1 = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"],
]
assert num_islands([row[:] for row in grid1]) == 3
assert num_islands([["0"]]) == 0
assert num_islands([["1"]]) == 1
```

---

## Debriefing (a remplir apres les 10 problemes)

- **Score** : _____ / 10 en moins de 15 min chacun
- **Temps total** : _____ min
- **Pattern le plus difficile** : _____
- **Bug le plus frequent** : _____
- **Edge case que j'ai oublie** : _____
- **Action pour demain** : _____
