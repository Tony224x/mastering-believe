# Exercices Medium — Dynamic Programming

---

## Exercice 4 : Linear DP circulaire — House Robber II

### Objectif

Apprendre a **reduire un probleme nouveau a un probleme connu** : la contrainte circulaire se decompose en deux sous-problemes lineaires. C'est le reflexe DP le plus utile en entretien.

### Consigne

Comme House Robber (theorie du jour), mais les maisons sont disposees **en cercle** : la premiere et la derniere maison sont adjacentes. Tu ne peux pas voler deux maisons adjacentes. Retourne le butin maximum.

```python
def rob_circular(nums: list[int]) -> int:
    """
    Max loot when houses form a circle (first and last are adjacent).
    """
    pass
```

**Indice** : la maison 0 et la maison n-1 ne peuvent pas etre volees toutes les deux. Donc la reponse est `max(rob_lineaire(nums[:-1]), rob_lineaire(nums[1:]))`. Ecris d'abord le helper lineaire en O(1) espace.

**Piege** : le cas `len(nums) == 1` casse la decomposition (les deux slices sont vides).

### Tests

```python
assert rob_circular([2, 3, 2]) == 3          # Can't take both 2s (adjacent in circle)
assert rob_circular([1, 2, 3, 1]) == 4       # 1 + 3
assert rob_circular([1, 2, 3]) == 3
assert rob_circular([5]) == 5                # Single house — decomposition trap
assert rob_circular([]) == 0
assert rob_circular([1, 2]) == 2
assert rob_circular([200, 3, 140, 20, 10]) == 340
```

### Criteres de reussite

- [ ] Helper `rob_line` lineaire en O(n) temps, O(1) espace (deux variables, pas de tableau)
- [ ] Decomposition en deux appels : sans la derniere maison / sans la premiere
- [ ] Le cas une seule maison est gere explicitement
- [ ] Tu sais justifier la decomposition (les deux extremites sont mutuellement exclusives)
- [ ] Tous les tests passent

---

## Exercice 5 : DP + Binary Search — Longest Increasing Subsequence

### Objectif

Implementer LIS deux fois : la DP quadratique standard, puis la version "patience sorting" en O(n log n) avec `bisect` — et comprendre pourquoi le tableau `tails` n'est PAS la sous-suite elle-meme.

### Consigne

Etant donne un tableau d'entiers `nums`, retourne la longueur de la **plus longue sous-suite strictement croissante** (les elements gardent leur ordre relatif mais ne sont pas forcement contigus).

Implemente DEUX versions :
1. `lis_dp(nums)` — DP O(n^2) : `dp[i]` = longueur de la meilleure LIS finissant en `i`.
2. `lis_fast(nums)` — O(n log n) : tableau `tails` ou `tails[k]` = plus petite fin possible d'une sous-suite croissante de longueur k+1, mis a jour avec `bisect_left`.

```python
def lis_dp(nums: list[int]) -> int:
    """O(n^2) DP version."""
    pass

def lis_fast(nums: list[int]) -> int:
    """O(n log n) patience-sorting version using bisect."""
    pass
```

**Piege** : strictement croissante → `bisect_left` (remplace les egaux), pas `bisect_right`.

### Tests

```python
for lis in (lis_dp, lis_fast):
    assert lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4     # 2, 3, 7, 101 (or 18)
    assert lis([0, 1, 0, 3, 2, 3]) == 4
    assert lis([7, 7, 7, 7]) == 1                      # Strictly increasing!
    assert lis([]) == 0
    assert lis([5]) == 1
    assert lis([1, 2, 3, 4]) == 4
    assert lis([4, 3, 2, 1]) == 1

# The two versions must agree on random inputs
import random
for _ in range(100):
    arr = [random.randint(0, 20) for _ in range(random.randint(0, 30))]
    assert lis_dp(arr) == lis_fast(arr)
```

### Criteres de reussite

- [ ] `lis_dp` : double boucle, `dp[i] = max(dp[j] + 1)` pour `j < i` et `nums[j] < nums[i]`
- [ ] `lis_fast` : `bisect_left` sur `tails`, remplacement ou append
- [ ] Le cas doublons (`[7,7,7,7]` → 1) passe dans les deux versions
- [ ] Tu sais expliquer l'invariant de `tails` et pourquoi sa longueur = longueur de la LIS
- [ ] Les deux versions concordent sur 100 inputs aleatoires

---

## Exercice 6 : Ordre des boucles — Coin Change II (combinaisons)

### Objectif

LE piege le plus subtil du DP unidimensionnel : l'ordre des boucles decide si on compte des **combinaisons** ou des **permutations**. Quasi personne ne sait le justifier en entretien — toi si.

### Consigne

Etant donne des pieces `coins` (utilisables a volonte) et un montant `amount`, retourne le **nombre de combinaisons** distinctes qui forment ce montant. Deux suites utilisant les memes pieces dans un ordre different comptent pour UNE seule combinaison.

```python
def change(amount: int, coins: list[int]) -> int:
    """
    Number of DISTINCT combinations (order doesn't matter) summing to amount.
    """
    pass
```

**La question cle** : pourquoi `for coin: for a:` compte des combinaisons alors que `for a: for coin:` compte des permutations ? Reponds en commentaire dans ton code. Ecris aussi `change_permutations` (l'autre ordre) et verifie sur `amount=3, coins=[1,2]` que les resultats different (2 combinaisons vs 3 permutations).

### Tests

```python
assert change(5, [1, 2, 5]) == 4         # 5 / 2+2+1 / 2+1+1+1 / 1*5
assert change(3, [2]) == 0
assert change(0, [7]) == 1               # Empty combination
assert change(0, []) == 1
assert change(10, [10]) == 1
assert change(3, [1, 2]) == 2            # 1+1+1, 1+2
assert change_permutations(3, [1, 2]) == 3   # 1+1+1, 1+2, 2+1 — order matters
assert change(500, [3, 5, 7, 8, 9, 10, 11]) == 35502874
```

### Criteres de reussite

- [ ] `dp[0] = 1` (la combinaison vide) — pas 0
- [ ] Boucle externe sur les pieces : chaque combinaison est construite dans un ordre canonique unique
- [ ] `change_permutations` ecrite avec l'autre ordre de boucles, et le contre-exemple verifie
- [ ] L'explication de la difference est dans les commentaires (en anglais)
- [ ] O(amount * len(coins)) temps, O(amount) espace
