# Exercices Hard — DP Avance & Greedy

---

## Exercice 7 : Interval DP — Palindrome Partitioning II (minimum de coupes)

### Objectif

Combiner DEUX tables DP qui se nourrissent l'une l'autre : la table des palindromes `is_palin[i][j]` puis la table des coupes minimales. La version naive (test de palindrome par slicing dans la boucle) est O(n^3) — l'exercice impose O(n^2).

### Consigne

Etant donne une string `s`, retourne le **nombre minimum de coupes** pour partitionner `s` en sous-chaines qui sont toutes des palindromes.

**Contrainte de complexite : O(n^2) temps, O(n^2) espace.**

```python
def min_cut(s: str) -> int:
    """
    Minimum cuts to partition s into palindromic substrings.
    Must be O(n^2) — no substring slicing inside the DP loops.
    """
    pass
```

**Etapes** :
1. Table `is_palin[i][j]` remplie par longueur croissante : `s[i] == s[j] AND (longueur <= 2 OR is_palin[i+1][j-1])`. Aucun slicing.
2. `cuts[i]` = min coupes pour `s[:i+1]` : si `s[:i+1]` est un palindrome → 0 ; sinon `min(cuts[j] + 1)` pour tout `j` tel que `s[j+1:i]` est palindrome (via la table, pas de slice).
3. Benchmark : sur `"ab" * 600` (n=1200), la version O(n^2) doit repondre en moins de quelques secondes ; estime ce que ferait la version O(n^3) (chaque test de palindrome re-scanne la sous-chaine).

### Tests

```python
assert min_cut("aab") == 1               # "aa" | "b"
assert min_cut("a") == 0
assert min_cut("ab") == 1
assert min_cut("aaaa") == 0              # Already a palindrome
assert min_cut("abba") == 0
assert min_cut("abcba") == 0
assert min_cut("abcde") == 4             # Worst case: every char alone
assert min_cut("aabaa") == 0
assert min_cut("aabba") == 1             # "aabba"? no — "a|abba" → 1
assert min_cut("cabababcbc") == 3
```

### Criteres de reussite

- [ ] Table des palindromes remplie par longueur croissante, sans aucun slicing de string
- [ ] `cuts[i] = 0` court-circuite quand le prefixe entier est un palindrome
- [ ] O(n^2) temps et espace — verifie sur n=1200 (reponse quasi instantanee)
- [ ] Tu sais expliquer pourquoi la version slicing est O(n^3)
- [ ] Tous les tests passent

---

## Exercice 8 : State Machine generalisee — Best Time to Buy and Sell Stock IV (k transactions)

### Objectif

Generaliser la state machine a 2k etats (`hold[t]` / `cash[t]` pour chaque transaction t) et reperer l'optimisation indispensable : quand k est grand, le probleme degenere en "transactions illimitees".

### Consigne

Retourne le profit maximum avec **au plus k transactions** (1 transaction = 1 achat + 1 vente). Tu ne peux detenir qu'une action a la fois.

**Contraintes** :
- O(n * k) temps, O(k) espace.
- **Shortcut obligatoire** : si `k >= n // 2`, le nombre de transactions n'est plus une contrainte (une transaction prend au moins 2 jours) → bascule sur l'algo greedy "somme des deltas positifs" en O(n). Sans ce shortcut, un test avec k enorme ferait exploser le temps.

```python
def max_profit_k(k: int, prices: list[int]) -> int:
    """
    Max profit with at most k buy+sell transactions.
    O(n*k) time, O(k) space, with the k >= n//2 shortcut.
    """
    pass
```

**Etats** : `hold[t]` = meilleur cash en detenant une action dans la transaction t ; `cash[t]` = meilleur cash apres avoir clos t transactions.

### Tests

```python
assert max_profit_k(2, [2, 4, 1]) == 2
assert max_profit_k(2, [3, 2, 6, 5, 0, 3]) == 7          # (2->6) + (0->3)
assert max_profit_k(1, [3, 2, 6, 5, 0, 3]) == 4          # Only (2->6)
assert max_profit_k(0, [1, 5]) == 0                       # Zero transactions
assert max_profit_k(2, []) == 0
assert max_profit_k(2, [5, 4, 3, 2, 1]) == 0              # Decreasing: no trade
assert max_profit_k(100, [1, 2, 3, 4, 5]) == 4            # Shortcut path: k huge
assert max_profit_k(1000000, list(range(1000)) * 2) == 1998  # Must be fast (shortcut!)
assert max_profit_k(3, [1, 5, 2, 8, 3, 10]) == 17         # All three transactions
```

### Criteres de reussite

- [ ] Le shortcut `k >= n // 2` est present et le test avec k = 1 000 000 repond instantanement
- [ ] State machine : `hold[t] = max(hold[t], cash[t-1] - price)`, `cash[t] = max(cash[t], hold[t] + price)`
- [ ] Initialisation `hold = -inf` (on ne detient rien avant le premier achat)
- [ ] O(n * k) temps, O(k) espace — pas de table n x k
- [ ] Tu sais retrouver Stock I (k=1) et Stock II (k=inf) comme cas particuliers
- [ ] Tous les tests passent
