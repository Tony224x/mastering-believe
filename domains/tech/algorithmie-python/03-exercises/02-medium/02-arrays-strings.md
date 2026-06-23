# Exercices Medium — Arrays & Strings

---

## Exercice 4 : Two Pointers — 3Sum

### Objectif

Combiner le tri + two pointers pour resoudre un probleme a trois elements — la progression naturelle apres Two Sum.

### Consigne

Etant donne un tableau d'entiers `nums`, trouve tous les triplets uniques `[nums[i], nums[j], nums[k]]` tels que `i != j != k` et `nums[i] + nums[j] + nums[k] == 0`.

La solution ne doit pas contenir de triplets en double.

```python
def three_sum(nums: list[int]) -> list[list[int]]:
    """
    Return all unique triplets that sum to zero.
    """
    pass
```

**Indice** : trier d'abord, puis pour chaque element `nums[i]`, utiliser two pointers sur le reste du tableau pour trouver la paire complementaire `-nums[i]`.

### Tests

```python
result = three_sum([-1, 0, 1, 2, -1, -4])
assert sorted([sorted(t) for t in result]) == sorted([[-1, -1, 2], [-1, 0, 1]])

result = three_sum([0, 1, 1])
assert result == []

result = three_sum([0, 0, 0])
assert result == [[0, 0, 0]]

result = three_sum([0, 0, 0, 0])
assert result == [[0, 0, 0]]  # Only one triplet, not duplicated

result = three_sum([-2, 0, 1, 1, 2])
assert sorted([sorted(t) for t in result]) == sorted([[-2, 0, 2], [-2, 1, 1]])

result = three_sum([])
assert result == []

result = three_sum([1])
assert result == []
```

### Criteres de reussite

- [ ] Utilise le tri O(n log n) + two pointers O(n) pour chaque element fixe → O(n^2) total
- [ ] Les doublons sont geres correctement (pas de triplets en double)
- [ ] Complexite O(n^2) temps, O(1) espace (hors le tri et le resultat)
- [ ] Tous les tests passent, y compris les edge cases (tableau vide, zeros multiples)

---

## Exercice 5 : Sliding Window — Longest Repeating Character Replacement

### Objectif

Maitriser le sliding window variable avec un compteur de frequences pour un probleme de type "plus long sous-tableau avec contrainte".

### Consigne

Etant donne une string `s` composee uniquement de majuscules et un entier `k`, tu peux choisir n'importe quel caractere de la string et le remplacer par une autre majuscule. Tu peux effectuer cette operation **au plus k fois**.

Retourne la longueur de la **plus longue sous-chaine** contenant la meme lettre apres avoir effectue les operations.

```python
def character_replacement(s: str, k: int) -> int:
    """
    Return the length of the longest substring with same letter
    after at most k replacements.
    """
    pass
```

**Indice** : dans une fenetre de taille `window_size`, si le caractere le plus frequent apparait `max_freq` fois, alors on a besoin de `window_size - max_freq` remplacements. Si ce nombre <= k, la fenetre est valide.

### Tests

```python
assert character_replacement("ABAB", 2) == 4        # Replace 2 A's → "BBBB" or 2 B's → "AAAA"
assert character_replacement("AABABBA", 1) == 4      # Replace 1 B at index 3 → "AAAAABA" → "AAAA"
assert character_replacement("AAAA", 0) == 4         # Already all same
assert character_replacement("ABCD", 3) == 4         # Replace 3 chars → all same
assert character_replacement("A", 0) == 1            # Single char
assert character_replacement("ABAB", 0) == 1         # No replacements allowed
assert character_replacement("AAAB", 0) == 3         # "AAA" without any replacement
```

### Criteres de reussite

- [ ] Utilise le sliding window variable avec un Counter/dict de frequences
- [ ] Complexite O(n) temps (ou O(26n) = O(n) si on parcourt le compteur a chaque pas)
- [ ] Comprend pourquoi `max_freq` n'a pas besoin d'etre decremente quand on retrecit la fenetre (astuce d'optimisation)
- [ ] Tous les tests passent

---

## Exercice 6 : Prefix Sum — Contiguous Array (0s and 1s)

### Objectif

Appliquer le pattern prefix sum + hashmap a un probleme non-trivial qui requiert une transformation intelligente de l'input.

### Consigne

Etant donne un tableau binaire `nums` (contenant uniquement des 0 et des 1), trouve la longueur du **plus long sous-tableau contigu** contenant un nombre egal de 0 et de 1.

```python
def find_max_length(nums: list[int]) -> int:
    """
    Return the length of the longest subarray with equal 0s and 1s.
    """
    pass
```

**Indice** : remplace chaque 0 par -1. Le probleme devient : trouver le plus long sous-tableau de somme 0. Utilise le prefix sum + hashmap : si prefix[i] == prefix[j], alors le sous-tableau i..j-1 a une somme de 0.

### Tests

```python
assert find_max_length([0, 1]) == 2
assert find_max_length([0, 1, 0]) == 2              # [0, 1] or [1, 0]
assert find_max_length([0, 0, 1, 0, 0, 0, 1, 1]) == 6  # [0, 1, 0, 0, 1, 1] at indices 2-7
assert find_max_length([0, 0, 0, 1, 1, 1]) == 6     # Entire array
assert find_max_length([1, 1, 1, 1]) == 0            # No equal count possible
assert find_max_length([0]) == 0                     # Single element
assert find_max_length([]) == 0
assert find_max_length([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]) == 12  # Entire array
```

### Criteres de reussite

- [ ] Transformation 0 → -1 appliquee pour ramener le probleme a "sous-tableau de somme 0"
- [ ] Utilise prefix sum + hashmap (prefix_sum → premier index ou cette somme est apparue)
- [ ] Complexite O(n) temps, O(n) espace
- [ ] Comprend pourquoi on stocke le PREMIER index (pas le dernier) dans le hashmap
- [ ] Tous les tests passent
