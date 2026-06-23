# Exercices Hard — DP Avance & Greedy

---

## Exercice 7 : State Machine DP — Best Time to Buy and Sell Stock IV (k transactions)

### Objectif

Generaliser la state machine DP a un nombre **borne** `k` de transactions. C'est la version la plus exigeante de la famille "stock", avec une optimisation cruciale quand `k >= n/2` (transactions illimitees).

### Consigne

Etant donne un entier `k` et un tableau `prices`, maximise ton profit avec **au plus `k` transactions** (une transaction = un achat suivi d'une vente). Tu ne peux pas detenir plus d'une action a la fois.

```python
def max_profit_k(k: int, prices: list[int]) -> int:
    """
    Maximize profit with AT MOST k transactions (buy then sell), holding at
    most one share at a time.
    """
    pass
```

**Approche attendue** :
- Si `k >= len(prices) // 2`, c'est equivalent a transactions illimitees -> greedy en O(n) (sommer toutes les montees).
- Sinon, DP : pour chaque transaction `t` de 1 a k, maintenir `buy[t]` (meilleur profit en ayant achete) et `sell[t]` (meilleur profit en ayant vendu).
  - `buy[t] = max(buy[t], sell[t-1] - price)`
  - `sell[t] = max(sell[t], buy[t] + price)`

### Tests

```python
assert max_profit_k(2, [2, 4, 1]) == 2
assert max_profit_k(2, [3, 2, 6, 5, 0, 3]) == 7
assert max_profit_k(0, [1, 3, 5]) == 0           # 0 transaction autorisee
assert max_profit_k(2, []) == 0
assert max_profit_k(1, [1, 2, 3, 4, 5]) == 4     # 1 transaction : acheter 1 vendre 5
assert max_profit_k(100, [1, 2, 3, 4, 5]) == 4   # k enorme = illimite
assert max_profit_k(2, [1, 2, 4, 2, 5, 7, 2, 4, 9, 0]) == 13
assert max_profit_k(3, [5, 4, 3, 2, 1]) == 0     # decroissant
```

### Criteres de reussite

- [ ] Optimise le cas `k >= n/2` en greedy O(n) (sinon TLE / memoire excessive)
- [ ] DP correcte avec `buy[t]` / `sell[t]` pour `t` de 1 a k
- [ ] Gere `k = 0`, prices vide, tableau decroissant (profit 0)
- [ ] Complexite O(n * k) temps (ou O(n) dans le cas illimite), O(k) espace
- [ ] Tous les tests passent

---

## Exercice 8 : Interval DP — Burst Balloons

### Objectif

Maitriser l'interval DP "le dernier a eclater" : au lieu de penser au premier choix, on raisonne sur le **dernier** ballon eclate dans un intervalle. C'est l'un des interval DP les plus contre-intuitifs et discriminants en entretien.

### Consigne

On a `n` ballons indexes de 0 a n-1, chacun avec un nombre `nums[i]` peint dessus. Quand tu eclates le ballon `i`, tu gagnes `nums[left] * nums[i] * nums[right]` ou `left` et `right` sont les voisins **adjacents apres les eclatements precedents**. Si un voisin n'existe pas (bord), on le traite comme `1`. Maximise les pieces gagnees.

```python
def max_coins(nums: list[int]) -> int:
    """
    Burst all balloons to maximize coins. Bursting balloon i yields
    nums[left] * nums[i] * nums[right] with current adjacent neighbors.
    Out-of-bounds neighbors count as 1.
    """
    pass
```

**Approche attendue** :
- Ajoute des bornes virtuelles `1` aux deux extremites : `arr = [1] + nums + [1]`.
- `dp[i][j]` = max pieces en eclatant tous les ballons strictement entre `i` et `j` (exclusifs).
- Pour le **dernier** ballon `m` eclate dans `(i, j)` :
  `dp[i][j] = max(dp[i][m] + arr[i]*arr[m]*arr[j] + dp[m][j])` pour `i < m < j`.
- Remplir par longueur d'intervalle croissante.

### Tests

```python
assert max_coins([3, 1, 5, 8]) == 167
assert max_coins([1, 5]) == 10
assert max_coins([]) == 0
assert max_coins([5]) == 5
assert max_coins([7, 9, 8, 0, 7, 1, 3, 5, 5, 2, 3]) == 1654
assert max_coins([1]) == 1
assert max_coins([9, 76, 64, 21]) == 116718
```

### Criteres de reussite

- [ ] Ajoute les bornes virtuelles `1` aux extremites
- [ ] Raisonne sur le **dernier** ballon eclate de l'intervalle (pas le premier)
- [ ] Remplit la table par longueur d'intervalle croissante
- [ ] Complexite O(n^3) temps, O(n^2) espace
- [ ] Gere tableau vide / un seul element
- [ ] Tous les tests passent

---

## Exercice 9 : Greedy avec heap — Task Scheduler

### Objectif

Resoudre un probleme d'ordonnancement greedy avec une formule fermee (le pattern "remplir les idle slots"). Tu apprends a raisonner sur le caractere le plus frequent pour borner le temps total.

### Consigne

Etant donne un tableau `tasks` de caracteres (chaque caractere est une tache) et un entier `n` (cooldown), retourne le **nombre minimum d'unites de temps** pour executer toutes les taches. Deux taches **identiques** doivent etre separees par au moins `n` unites de temps ; le CPU peut rester idle.

```python
def least_interval(tasks: list[str], n: int) -> int:
    """
    Return the minimum number of time units to finish all tasks, where two
    identical tasks must be at least n units apart (CPU may stay idle).
    """
    pass
```

**Approche attendue (formule greedy)** :
- Soit `f_max` la frequence maximale et `n_max` le nombre de taches ayant cette frequence.
- On organise `(f_max - 1)` blocs de taille `(n + 1)`, puis on ajoute `n_max` a la fin :
  `frame = (f_max - 1) * (n + 1) + n_max`.
- Le resultat est `max(frame, len(tasks))` (si beaucoup de taches distinctes, aucun idle n'est necessaire).

### Tests

```python
assert least_interval(["A", "A", "A", "B", "B", "B"], 2) == 8
assert least_interval(["A", "A", "A", "B", "B", "B"], 0) == 6
assert least_interval(["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"], 4) == 10
assert least_interval(["A"], 2) == 1
assert least_interval(["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"], 2) == 16
assert least_interval(["A", "A", "A", "B", "B", "B"], 3) == 10
assert least_interval(["A", "B", "A", "B"], 2) == 4
```

### Criteres de reussite

- [ ] Calcule `f_max` et `n_max` (nombre de taches a frequence maximale)
- [ ] Applique la formule `(f_max - 1) * (n + 1) + n_max` puis `max(..., len(tasks))`
- [ ] Sait expliquer pourquoi `max(frame, len(tasks))` capture les deux regimes
- [ ] Complexite O(T) temps (T = nombre de taches), O(1) espace (26 lettres)
- [ ] Tous les tests passent (y compris n = 0, une seule tache)
