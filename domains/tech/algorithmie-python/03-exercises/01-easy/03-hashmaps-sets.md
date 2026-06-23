# Exercices Easy — Hash Maps & Sets

---

## Exercice 1 : Frequency Counting — Ransom Note

### Objectif

Maitriser le frequency counting pour verifier si une collection de caracteres est un sous-ensemble d'une autre.

### Consigne

Etant donne deux strings `ransom_note` et `magazine`, retourne `True` si `ransom_note` peut etre construite en utilisant les lettres de `magazine`, chaque lettre ne pouvant etre utilisee qu'une seule fois.

```python
def can_construct(ransom_note: str, magazine: str) -> bool:
    """
    Return True if ransom_note can be constructed from magazine letters.
    Each letter in magazine can only be used once.
    """
    pass
```

### Tests

```python
assert can_construct("a", "b") == False
assert can_construct("aa", "ab") == False
assert can_construct("aa", "aab") == True
assert can_construct("", "anything") == True
assert can_construct("abc", "abc") == True
assert can_construct("abc", "cba") == True
assert can_construct("aab", "baa") == True
assert can_construct("fihjjjjei", "hjibagacbhadfaefdjaeaebgi") == False
```

### Criteres de reussite

- [ ] Utilise Counter ou un dict de frequences (pas de sorted + comparaison)
- [ ] Complexite O(n + m) temps ou n = len(ransom_note), m = len(magazine)
- [ ] Complexite O(1) espace (borne par la taille de l'alphabet)
- [ ] Tous les tests passent, y compris le edge case string vide

---

## Exercice 2 : Seen Set — Contains Duplicate

### Objectif

Utiliser un set comme structure de detection de doublons pour transformer une verification O(n^2) en O(n).

### Consigne

Etant donne un tableau d'entiers `nums`, retourne `True` s'il contient au moins un doublon (une valeur qui apparait au moins deux fois).

```python
def contains_duplicate(nums: list[int]) -> bool:
    """
    Return True if any value appears at least twice in the array.
    """
    pass
```

### Tests

```python
assert contains_duplicate([1, 2, 3, 1]) == True
assert contains_duplicate([1, 2, 3, 4]) == False
assert contains_duplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]) == True
assert contains_duplicate([]) == False
assert contains_duplicate([1]) == False
assert contains_duplicate([1, 1]) == True
```

### Criteres de reussite

- [ ] Utilise un set pour la detection de doublons (pas de tri, pas de double boucle)
- [ ] Complexite O(n) temps, O(n) espace
- [ ] Short-circuit : retourne `True` des le premier doublon trouve (ne parcourt pas tout le tableau inutilement)
- [ ] Tous les tests passent

---

## Exercice 3 : Two-Sum Pattern — Two Sum

### Objectif

Maitriser le pattern Two Sum avec hash map — LE probleme le plus important a connaitre par coeur pour les entretiens.

### Consigne

Etant donne un tableau d'entiers `nums` et un entier `target`, retourne les **indices** des deux elements dont la somme est egale a `target`.

Tu peux supposer qu'il y a exactement une solution et que tu ne peux pas utiliser le meme element deux fois.

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Return indices [i, j] where nums[i] + nums[j] == target.
    Exactly one solution exists. i < j.
    """
    pass
```

### Tests

```python
assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([3, 2, 4], 6) == [1, 2]
assert two_sum([3, 3], 6) == [0, 1]
assert two_sum([1, 5, 8, 3], 4) == [0, 3]
assert two_sum([-1, -2, -3, -4, -5], -8) == [2, 4]
```

### Criteres de reussite

- [ ] Utilise un dict comme table de lookup (valeur → index)
- [ ] Stocke l'element APRES avoir verifie le complement (pour eviter le self-match)
- [ ] Complexite O(n) temps, O(n) espace
- [ ] Comprend pourquoi cette approche est meilleure que le tri + two pointers (on retourne des indices, pas des valeurs)
- [ ] Tous les tests passent
