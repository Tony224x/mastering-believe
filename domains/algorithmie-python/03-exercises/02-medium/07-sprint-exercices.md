# Sprint Complementaire Medium — Jour 7

> Le fichier `01-easy/07-sprint-exercices.md` couvre deja 10 problemes (P1-P10). Ce sprint complementaire ajoute **3 nouveaux problemes medium** sur les memes patterns J1-J6, **sans aucun doublon** avec P1-P10. Applique le meme protocole : chronometre-toi, repete l'enonce, annonce la brute force, code en parlant, teste les edge cases.

---

## Exercice 4 : Sliding window — Permutation in String

### Objectif (Medium — 12 min)

Combiner sliding window de taille fixe + comparaison de frequences. C'est la variante "fenetre fixe" du sliding window (vs la fenetre variable du P6).

### Consigne

Etant donne deux strings `s1` et `s2`, retourne `True` si `s2` contient une **permutation** de `s1`, c'est-a-dire si une des sous-chaines contigues de `s2` est un anagramme de `s1`.

```python
def check_inclusion(s1: str, s2: str) -> bool:
    """
    Return True if s2 contains a contiguous substring that is a permutation of s1.
    """
    pass
```

**Indice** : fenetre glissante de taille `len(s1)` sur `s2`. Maintiens un compteur de frequences de la fenetre et compare-le au compteur de `s1`. Mets a jour le compteur en O(1) a chaque glissement (ajoute le nouveau, retire l'ancien).

### Tests

```python
assert check_inclusion("ab", "eidbaooo") == True       # "ba" is a permutation of "ab"
assert check_inclusion("ab", "eidboaoo") == False
assert check_inclusion("a", "a") == True
assert check_inclusion("abc", "ab") == False           # s1 longer than s2
assert check_inclusion("adc", "dcda") == True
assert check_inclusion("hello", "ooolleoooleh") == False
assert check_inclusion("ab", "ab") == True
```

### Criteres de reussite

- [ ] Sliding window de taille FIXE = `len(s1)`
- [ ] Compteurs maintenus en O(1) par glissement (pas de recomptage complet)
- [ ] Court-circuite si `len(s1) > len(s2)`
- [ ] Complexite O(len(s2)) temps, O(26) = O(1) espace
- [ ] Resolu en moins de 12 minutes chrono

---

## Exercice 5 : Two pointers + tri — 3Sum Closest

### Objectif (Medium — 12 min)

Variante de 3Sum (tri + two pointers) ou on minimise une distance plutot que de chercher une egalite exacte. Teste la gestion d'un "meilleur courant".

### Consigne

Etant donne un tableau d'entiers `nums` et un entier `target`, trouve trois entiers de `nums` dont la somme est **la plus proche** de `target`. Retourne cette somme. Une seule solution est garantie.

```python
def three_sum_closest(nums: list[int], target: int) -> int:
    """
    Return the sum of the three integers closest to target.
    """
    pass
```

**Indice** : trie le tableau. Pour chaque `i`, fais un two-pointers sur le reste. Garde la somme dont l'ecart absolu avec `target` est minimal. Avance les pointeurs selon le signe de `current_sum - target`.

### Tests

```python
assert three_sum_closest([-1, 2, 1, -4], 1) == 2       # -1 + 2 + 1 = 2
assert three_sum_closest([0, 0, 0], 1) == 0
assert three_sum_closest([1, 1, 1, 0], -100) == 2
assert three_sum_closest([1, 2, 3], 6) == 6            # Exact match
assert three_sum_closest([-3, -2, -5, 3, -4], -1) == -2
assert three_sum_closest([0, 1, 2], 3) == 3
```

### Criteres de reussite

- [ ] Trie d'abord, puis two pointers pour chaque element fixe
- [ ] Maintient la meilleure somme par ecart absolu minimal
- [ ] Complexite O(n^2) temps, O(1) espace (hors tri)
- [ ] Gere le cas de l'egalite exacte (peut court-circuiter)
- [ ] Resolu en moins de 12 minutes chrono

---

## Exercice 6 : Prefix product — Product of Array Except Self

### Objectif (Medium — 12 min)

Maitriser le pattern prefix/suffix products SANS division et en O(1) espace auxiliaire (hors output). Un classique qui piege ceux qui pensent "diviser par l'element".

### Consigne

Etant donne un tableau `nums`, retourne un tableau `answer` ou `answer[i]` est le produit de tous les elements de `nums` SAUF `nums[i]`. Tu dois resoudre **sans division** et en O(n).

```python
def product_except_self(nums: list[int]) -> list[int]:
    """
    answer[i] = product of all nums except nums[i]. No division, O(n).
    """
    pass
```

**Indice** : `answer[i] = (produit des elements a gauche de i) * (produit des elements a droite de i)`. Premiere passe : produits prefixes dans `answer`. Deuxieme passe (de droite a gauche) : multiplie par les produits suffixes accumules dans une seule variable.

### Tests

```python
assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
assert product_except_self([2, 3]) == [3, 2]
assert product_except_self([0, 0]) == [0, 0]           # Two zeros
assert product_except_self([5]) == [1]
assert product_except_self([1, 0]) == [0, 1]
```

### Criteres de reussite

- [ ] N'utilise PAS la division (gere les zeros naturellement)
- [ ] Deux passes (prefix puis suffix), O(1) espace auxiliaire hors output
- [ ] Complexite O(n) temps
- [ ] Gere un seul element, des zeros (un ou plusieurs)
- [ ] Resolu en moins de 12 minutes chrono
