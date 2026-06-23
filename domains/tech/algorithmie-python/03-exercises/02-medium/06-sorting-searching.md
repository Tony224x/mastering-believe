# Exercices Medium — Sorting & Searching

---

## Exercice 4 : Binary search variante — Find First and Last Position

### Objectif

Maitriser `lower_bound` et `upper_bound` (les deux variantes "frontiere" du binary search) pour localiser un intervalle d'occurrences en O(log n). C'est la competence qui debloque tout le reste du binary search avance.

### Consigne

Etant donne un tableau `nums` trie en ordre croissant (avec doublons possibles) et un `target`, retourne `[premier_index, dernier_index]` des occurrences de `target`. Si `target` est absent, retourne `[-1, -1]`.

Complexite **O(log n)** obligatoire (deux binary searches).

```python
def search_range(nums: list[int], target: int) -> list[int]:
    """
    Return [first, last] index of target in the sorted array, or [-1, -1].
    Must run in O(log n).
    """
    pass
```

**Indice** : ecris `lower_bound` (premier index `>= target`) et `upper_bound` (premier index `> target`). Le premier index est `lower_bound(target)`, le dernier est `upper_bound(target) - 1`. Verifie que l'index est valide et que `nums[index] == target`.

### Tests

```python
assert search_range([5, 7, 7, 8, 8, 10], 8) == [3, 4]
assert search_range([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
assert search_range([], 0) == [-1, -1]
assert search_range([1], 1) == [0, 0]
assert search_range([2, 2, 2, 2], 2) == [0, 3]
assert search_range([1, 2, 3], 3) == [2, 2]
assert search_range([1, 2, 3], 1) == [0, 0]
assert search_range([1, 4], 2) == [-1, -1]
```

### Criteres de reussite

- [ ] Implemente `lower_bound` ET `upper_bound` (sans `bisect`, ou avec mais en sachant les recoder)
- [ ] Complexite O(log n) — pas de scan lineaire pour trouver les bornes
- [ ] Gere `target` absent, tableau vide, toutes valeurs egales
- [ ] Pas d'erreur d'index quand `target` n'existe pas
- [ ] Tous les tests passent

---

## Exercice 5 : Binary search on answer — Koko Eating Bananas

### Objectif

Decouvrir le pattern **"binary search on answer"** : on ne cherche pas dans le tableau d'entree mais dans l'espace des reponses possibles, en testant la faisabilite a chaque candidat. Un pattern meta tres puissant.

### Consigne

Koko a `piles[i]` bananes dans la pile `i` et un gardien revient dans `h` heures. Chaque heure, Koko choisit une pile et mange `k` bananes ; si la pile en a moins de `k`, elle mange tout et s'arrete pour cette heure. Trouve la **plus petite vitesse `k`** (entier) qui lui permet de tout manger en `h` heures.

```python
def min_eating_speed(piles: list[int], h: int) -> int:
    """
    Return the minimum integer eating speed to finish all piles within h hours.
    """
    pass
```

**Indice** : la reponse est entre `1` et `max(piles)`. Pour une vitesse `k`, le temps total est `sum(ceil(pile / k) for pile in piles)`. C'est une fonction **monotone decroissante** en `k` : binary search le plus petit `k` faisable.

### Tests

```python
assert min_eating_speed([3, 6, 7, 11], 8) == 4
assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23
assert min_eating_speed([1], 1) == 1
assert min_eating_speed([1000000000], 2) == 500000000
assert min_eating_speed([3, 6, 7, 11], 4) == 11      # h == len(piles): max speed
assert min_eating_speed([312884470], 312884469) == 2
```

### Criteres de reussite

- [ ] Identifie l'espace de recherche `[1, max(piles)]`
- [ ] Definit un predicat de faisabilite monotone (temps total <= h)
- [ ] Binary search la plus petite vitesse faisable (lower_bound sur le predicat)
- [ ] Complexite O(n log(max(piles))) temps, O(1) espace
- [ ] Gere `h == len(piles)` (vitesse = max), une seule pile, grandes valeurs
- [ ] Tous les tests passent

---

## Exercice 6 : Custom comparator — Largest Number

### Objectif

Maitriser `functools.cmp_to_key` pour un ordre qui ne se reduit PAS a une cle fixe : il faut comparer deux elements par leur concatenation. Un piege classique en entretien.

### Consigne

Etant donne une liste d'entiers non negatifs `nums`, organise-les pour former le **plus grand nombre possible** par concatenation. Retourne le resultat sous forme de string (il peut etre tres grand).

```python
def largest_number(nums: list[int]) -> str:
    """
    Arrange the numbers to form the largest possible concatenated number.
    Return it as a string.
    """
    pass
```

**Indice** : trie les nombres (en strings) avec un comparateur custom : `a` vient avant `b` si `a + b > b + a` (comparaison de strings). Attention au cas `[0, 0]` qui doit donner `"0"`, pas `"00"`.

### Tests

```python
assert largest_number([10, 2]) == "210"
assert largest_number([3, 30, 34, 5, 9]) == "9534330"
assert largest_number([1]) == "1"
assert largest_number([10]) == "10"
assert largest_number([0, 0]) == "0"          # Leading-zeros edge case
assert largest_number([0]) == "0"
assert largest_number([34, 3, 32]) == "34332"
assert largest_number([121, 12]) == "12121"
```

### Criteres de reussite

- [ ] Utilise `functools.cmp_to_key` (ou justifie une cle equivalente)
- [ ] Comparateur base sur `a + b` vs `b + a` (concatenation de strings)
- [ ] Gere le edge case "tout zero" → `"0"` (pas `"000"`)
- [ ] Complexite O(n log n * k) ou k = longueur max d'un nombre
- [ ] Tous les tests passent
