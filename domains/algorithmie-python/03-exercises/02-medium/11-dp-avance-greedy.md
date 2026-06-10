# Exercices Medium — DP Avance & Greedy

---

## Exercice 4 : State Machine DP — Best Time to Buy and Sell Stock with Cooldown

### Objectif

Concevoir une machine a etats a TROIS etats (hold / juste vendu / repos) et ses transitions — le passage oblige avant les problemes de stock generalises.

### Consigne

Tu peux faire autant de transactions que tu veux (acheter puis vendre), mais apres une **vente**, tu dois observer **un jour de cooldown** avant de racheter. Tu ne peux detenir qu'une action a la fois. Retourne le profit maximum.

```python
def max_profit_cooldown(prices: list[int]) -> int:
    """
    Max profit with unlimited transactions and a 1-day cooldown after each sell.
    """
    pass
```

**Les 3 etats** (valeurs = meilleur cash possible dans cet etat au jour i) :
- `hold` : je detiens une action.
- `sold` : je viens de vendre AUJOURD'HUI (demain = cooldown).
- `rest` : je ne detiens rien et je suis libre d'acheter.

**Transitions** :
- `hold[i] = max(hold[i-1], rest[i-1] - prices[i])` — on ne peut acheter que depuis `rest`.
- `sold[i] = hold[i-1] + prices[i]`
- `rest[i] = max(rest[i-1], sold[i-1])` — le cooldown est encode ici.

### Tests

```python
assert max_profit_cooldown([1, 2, 3, 0, 2]) == 3     # buy,sell,cooldown,buy,sell
assert max_profit_cooldown([1]) == 0
assert max_profit_cooldown([]) == 0
assert max_profit_cooldown([1, 2]) == 1
assert max_profit_cooldown([2, 1]) == 0              # Never trade at a loss
assert max_profit_cooldown([1, 2, 4]) == 3           # No cooldown needed
assert max_profit_cooldown([2, 1, 4]) == 3
assert max_profit_cooldown([6, 1, 3, 2, 4, 7]) == 6  # Single trade 1->7 beats trading twice with cooldown
```

### Criteres de reussite

- [ ] Trois etats explicites avec initialisations correctes (`hold = -prices[0]`, pas 0)
- [ ] Le cooldown est encode par la transition `rest ← sold` (un jour de decalage)
- [ ] O(n) temps, O(1) espace (trois variables, pas de tableaux)
- [ ] La reponse finale est `max(sold, rest)` — jamais `hold` (detenir une action a la fin est sous-optimal)
- [ ] Tous les tests passent

---

## Exercice 5 : Partition DP — Word Break

### Objectif

Le partition DP canonique : `dp[i]` depend de TOUS les `j < i`, gate par un lookup O(1) dans un set. Cible le piege de performance : chercher dans une **list** de mots rend l'algo quadratique en pratique.

### Consigne

Etant donne une string `s` et un dictionnaire `word_dict`, retourne `True` si `s` peut etre segmentee en une suite de mots du dictionnaire (chaque mot reutilisable a volonte).

```python
def word_break(s: str, word_dict: list[str]) -> bool:
    """
    Return True if s can be segmented into dictionary words.
    """
    pass
```

**Recurrence** : `dp[i] = OR sur j` de `dp[j] AND s[j:i] in words`. `dp[0] = True` (prefixe vide).

**Optimisation attendue** : borne la boucle interne par la longueur du plus long mot du dictionnaire (inutile de tester des `s[j:i]` plus longs que ca).

### Tests

```python
assert word_break("leetcode", ["leet", "code"]) == True
assert word_break("applepenapple", ["apple", "pen"]) == True       # Reuse allowed
assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) == False
assert word_break("", ["a"]) == True                                # Empty string
assert word_break("a", []) == False
assert word_break("aaaaaaa", ["aaaa", "aaa"]) == True               # 4+3
assert word_break("aaaaaaab", ["aaaa", "aaa"]) == False
assert word_break("cars", ["car", "ca", "rs"]) == True              # Needs backtrack on greedy "car"
```

### Criteres de reussite

- [ ] `dp[0] = True` et la reponse est `dp[len(s)]`
- [ ] Le dictionnaire est converti en `set` (lookup O(1), pas O(taille du dict))
- [ ] La boucle interne est bornee par la longueur max des mots
- [ ] Le test `"cars"` passe (le greedy "prendre le plus long mot" est faux — c'est pour ca qu'on fait du DP)
- [ ] O(n^2) temps au pire (ou O(n * maxlen)), O(n) espace

---

## Exercice 6 : Greedy — Jump Game II (minimum de sauts)

### Objectif

Passer de "peut-on atteindre la fin ?" (easy 3) a "en combien de sauts minimum ?" — le greedy par fenetres qui est en realite un BFS deguise sur les indices.

### Consigne

Etant donne `nums` ou `nums[i]` est la portee de saut maximale depuis `i`, retourne le **nombre minimum de sauts** pour atteindre le dernier index. On garantit que c'est toujours possible.

**Contrainte : O(n) temps.** Le DP O(n^2) ne valide pas.

```python
def jump(nums: list[int]) -> int:
    """
    Minimum number of jumps to reach the last index. O(n) greedy.
    """
    pass
```

**Indice (fenetres BFS)** : maintiens `current_end` (limite de la fenetre atteignable avec `jumps` sauts) et `farthest` (meilleure portee vue dans la fenetre). Quand `i` atteint `current_end`, on "saute" : `jumps += 1`, `current_end = farthest`.

**Piege** : la boucle s'arrete a `len(nums) - 1` EXCLUS — arriver sur le dernier index ne doit pas declencher un saut de plus.

### Tests

```python
assert jump([2, 3, 1, 1, 4]) == 2            # 0 -> 1 -> 4
assert jump([2, 3, 0, 1, 4]) == 2
assert jump([0]) == 0                         # Already there
assert jump([1, 2]) == 1
assert jump([1, 1, 1, 1]) == 3
assert jump([5, 1, 1, 1, 1]) == 1             # One big jump
assert jump([1, 2, 1, 1, 1]) == 3
assert jump([4, 1, 1, 3, 1, 1, 1]) == 2       # 0 -> 3 -> end
```

### Criteres de reussite

- [ ] Greedy par fenetres : `jumps`, `current_end`, `farthest`
- [ ] La boucle exclut le dernier index (pas de saut fantome a l'arrivee)
- [ ] Tu sais expliquer l'analogie BFS : chaque fenetre = un niveau, `jumps` = la profondeur
- [ ] O(n) temps, O(1) espace
- [ ] Tous les tests passent
