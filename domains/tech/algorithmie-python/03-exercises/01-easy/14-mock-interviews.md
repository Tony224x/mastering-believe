# Exercices Easy — Mock Interviews (fresh problems)

> **Instructions** : chacun des 3 problemes est un mock d'entretien reel. Mets 30-45 min au chrono, fais le process en 6 etapes (clarifier → exemples → brute force → optimiser → coder → tester) **a voix haute**. Ne regarde PAS les solutions avant d'avoir essaye, meme si tu galeres.

---

## Mock 1 : Move Zeroes (Easy)

### Objectif

S'entrainer a un probleme array in-place avec contrainte d'espace O(1). Tester ton reflexe "two pointers" sans avoir besoin de reflechir.

### Consigne

Etant donne un tableau d'entiers `nums`, deplace tous les 0 vers la fin du tableau tout en maintenant l'ordre relatif des elements non nuls. Tu dois faire ca **en place**, sans allouer de nouveau tableau.

```python
def move_zeroes(nums: list[int]) -> None:
    """
    Move all zeros to the end of nums in-place, preserving the order
    of non-zero elements. Do not return anything, modify nums.
    """
    pass
```

### Tests

```python
def check(nums, expected):
    move_zeroes(nums)
    assert nums == expected, f"Got {nums}, expected {expected}"

check([0, 1, 0, 3, 12], [1, 3, 12, 0, 0])
check([0], [0])
check([1, 2, 3], [1, 2, 3])
check([0, 0, 0], [0, 0, 0])
check([4, 2, 4, 0, 0, 3, 0, 5, 1, 0], [4, 2, 4, 3, 5, 1, 0, 0, 0, 0])
check([], [])
```

### Criteres de reussite

- [ ] Solution in-place (O(1) espace supplementaire)
- [ ] Complexite O(n) temps
- [ ] Maintient l'ordre relatif des elements non nuls
- [ ] Process entretien fait **a voix haute** (meme seul)

---

## Mock 2 : Valid Palindrome (Easy)

### Objectif

S'entrainer a un probleme string avec edge cases (majuscules, caracteres speciaux, string vide). Tester ton reflexe "clarifier les contraintes d'input".

### Consigne

Une string est un palindrome valide si, apres avoir converti toutes les majuscules en minuscules et supprime tous les caracteres non alphanumeriques, elle se lit de la meme maniere dans les deux sens.

Retourne `True` si `s` est un palindrome valide, `False` sinon.

```python
def is_palindrome(s: str) -> bool:
    """
    Return True if s is a valid palindrome after lowercasing and
    removing non-alphanumeric characters.
    """
    pass
```

### Tests

```python
assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("race a car") == False
assert is_palindrome(" ") == True         # Empty after cleanup
assert is_palindrome("") == True
assert is_palindrome("a.") == True        # Single alphanumeric char
assert is_palindrome("0P") == False       # '0' != 'p'
assert is_palindrome("ab_a") == True
```

### Criteres de reussite

- [ ] Utilise two pointers (pas de `s[::-1] == s` sur une string nettoyee, meme si ca marche)
- [ ] Complexite O(n) temps, O(1) espace (avec two pointers)
- [ ] Gere les caracteres speciaux et la casse correctement
- [ ] Le test `"0P"` → `False` ne t'a pas surpris (majuscule de 'p' != '0')

---

## Mock 3 : Group Anagrams (Medium)

### Objectif

S'entrainer a un probleme de grouping avec hash map. Tester ton reflexe "choisir une bonne cle de hash".

### Consigne

Etant donne un tableau de strings `strs`, groupe les anagrammes ensemble. Tu peux retourner la reponse dans n'importe quel ordre. Deux strings sont anagrammes si elles contiennent exactement les memes caracteres (avec les memes frequences) mais potentiellement dans un ordre different.

```python
def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Group anagrams together. Return groups in any order, and strings
    within each group in any order.
    """
    pass
```

### Tests

```python
def sort_groups(result):
    return sorted([sorted(group) for group in result])

assert sort_groups(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])) == [
    ["ate", "eat", "tea"],
    ["bat"],
    ["nat", "tan"],
]
assert sort_groups(group_anagrams([""])) == [[""]]
assert sort_groups(group_anagrams(["a"])) == [["a"]]
assert sort_groups(group_anagrams([])) == []
assert sort_groups(group_anagrams(["abc", "bca", "cab", "xyz"])) == [
    ["abc", "bca", "cab"],
    ["xyz"],
]
```

### Criteres de reussite

- [ ] Utilise `defaultdict(list)` avec une cle de hash par signature
- [ ] Choisit une cle qui identifie uniquement chaque classe d'anagramme (tuple trie OU tuple de frequences)
- [ ] Complexite O(n * k log k) ou O(n * k) selon la strategie de cle (k = longueur max)
- [ ] Tous les tests passent, y compris les cas vides
