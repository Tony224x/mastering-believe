# Exercices Hard — Dynamic Programming

---

## Exercice 7 : 2D DP — Edit Distance (Levenshtein)

### Objectif

Le 2D DP de reference sur les strings : trois transitions concurrentes, des cas de base souvent rates, et une optimisation espace 2D → 1D a maitriser pour le niveau senior.

### Consigne

Etant donne deux strings `word1` et `word2`, retourne le **nombre minimum d'operations** pour transformer `word1` en `word2`. Operations autorisees : inserer un caractere, supprimer un caractere, remplacer un caractere.

**Etapes imposees** :
1. Version 2D : `dp[i][j]` = distance entre `word1[:i]` et `word2[:j]`. Soigne les cas de base (`dp[i][0] = i`, `dp[0][j] = j` — transformer vers/depuis la chaine vide).
2. Version optimisee en espace O(min(m, n)) : une seule ligne + une variable `diagonal` (le `dp[i-1][j-1]` qu'on vient d'ecraser).
3. Les deux versions doivent concorder sur des inputs aleatoires.

```python
def min_distance(word1: str, word2: str) -> int:
    """2D DP version."""
    pass

def min_distance_1d(word1: str, word2: str) -> int:
    """Space-optimized version: O(min(m, n)) memory."""
    pass
```

**La recurrence** :
- Si `word1[i-1] == word2[j-1]` : `dp[i][j] = dp[i-1][j-1]` (rien a payer).
- Sinon : `1 + min(dp[i-1][j-1] (replace), dp[i-1][j] (delete), dp[i][j-1] (insert))`.

### Tests

```python
for f in (min_distance, min_distance_1d):
    assert f("horse", "ros") == 3
    assert f("intention", "execution") == 5
    assert f("", "") == 0
    assert f("", "abc") == 3             # 3 inserts
    assert f("abc", "") == 3             # 3 deletes
    assert f("abc", "abc") == 0
    assert f("a", "b") == 1              # 1 replace
    assert f("ab", "ba") == 2

# Cross-check both versions on random inputs
import random, string
for _ in range(100):
    w1 = "".join(random.choices("ab", k=random.randint(0, 10)))
    w2 = "".join(random.choices("ab", k=random.randint(0, 10)))
    assert min_distance(w1, w2) == min_distance_1d(w1, w2)
```

### Criteres de reussite

- [ ] Cas de base corrects : ligne 0 et colonne 0 remplies avec i et j
- [ ] Les trois transitions sont commentees avec l'operation qu'elles representent
- [ ] Version 1D : la diagonale est sauvegardee AVANT d'ecraser la case
- [ ] La version 1D itere sur la plus courte des deux strings (O(min(m, n)) espace)
- [ ] Les deux versions concordent sur 100 paires aleatoires
- [ ] O(m * n) temps dans les deux cas

---

## Exercice 8 : Knapsack 0/1 complet — valeur max + reconstruction + optimisation espace

### Objectif

Traiter le knapsack 0/1 comme en entretien senior : pas seulement la valeur optimale, mais aussi **quels objets** sont choisis (reconstruction), et la version 1D avec son piege de boucle inversee.

### Consigne

Etant donne `weights`, `values` (listes paralleles) et une capacite `capacity`, chaque objet etant utilisable **au plus une fois** :

1. `knapsack_2d(weights, values, capacity)` → `(best_value, chosen_indices)` : table 2D + backtracking dans la table pour retrouver les indices choisis (ordre croissant).
2. `knapsack_1d(weights, values, capacity)` → `best_value` : tableau 1D, **boucle capacite decroissante**.
3. En commentaire : pourquoi la boucle 1D croissante est fausse pour le 0/1 (elle reutiliserait l'objet courant = unbounded knapsack). Demontre-le avec un contre-exemple dans les tests.

```python
def knapsack_2d(weights: list[int], values: list[int], capacity: int) -> tuple[int, list[int]]:
    pass

def knapsack_1d(weights: list[int], values: list[int], capacity: int) -> int:
    pass
```

**Reconstruction** : depuis `dp[n][capacity]`, si `dp[i][w] != dp[i-1][w]`, l'objet i-1 a ete pris → ajoute-le et retire son poids.

### Tests

```python
best, chosen = knapsack_2d([1, 3, 4, 5], [1, 4, 5, 7], 7)
assert best == 9
assert sorted(chosen) == [1, 2]                 # Items of weight 3 and 4
assert sum(1 for i in chosen if chosen.count(i) > 1) == 0   # No item twice

best, chosen = knapsack_2d([2, 2, 2], [3, 3, 3], 4)
assert best == 6 and len(chosen) == 2           # Only TWO copies fit (0/1!)

assert knapsack_2d([], [], 10) == (0, [])
assert knapsack_2d([5], [10], 4) == (0, [])     # Item doesn't fit

assert knapsack_1d([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9
assert knapsack_1d([2, 2, 2], [3, 3, 3], 4) == 6
assert knapsack_1d([3], [5], 9) == 5            # The 0/1 counterexample:
                                                 # ascending loop would give 15 (item reused 3x)

# 2D and 1D agree on random instances
import random
for _ in range(50):
    n = random.randint(0, 10)
    ws = [random.randint(1, 10) for _ in range(n)]
    vs = [random.randint(1, 20) for _ in range(n)]
    cap = random.randint(0, 30)
    best2d, chosen = knapsack_2d(ws, vs, cap)
    assert best2d == knapsack_1d(ws, vs, cap)
    # The reconstruction must be consistent: feasible and worth best_value
    assert sum(ws[i] for i in chosen) <= cap
    assert sum(vs[i] for i in chosen) == best2d
```

### Criteres de reussite

- [ ] Table 2D correcte : `dp[i][w] = max(skip, take)` avec garde sur le poids
- [ ] Reconstruction par backtracking dans la table — les indices retournes sont coherents (poids total <= capacite, somme des valeurs == best)
- [ ] Version 1D avec boucle capacite **decroissante** ; le contre-exemple `[3],[5],cap=9` reste a 5
- [ ] Le commentaire explique le bug de la boucle croissante (0/1 → unbounded)
- [ ] O(n * W) temps ; 2D : O(n * W) espace, 1D : O(W) espace
- [ ] Les versions concordent sur 50 instances aleatoires
