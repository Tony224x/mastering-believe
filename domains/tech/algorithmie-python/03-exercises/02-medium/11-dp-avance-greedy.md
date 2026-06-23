# Exercices Medium — DP Avance & Greedy

---

## Exercice 4 : State Machine DP — Best Time to Buy and Sell Stock with Cooldown

### Objectif

Maitriser le DP "machine a etats" a 3 etats (hold / sold / rest) avec une contrainte temporelle (cooldown apres une vente). Tu apprends a ecrire les transitions AVANT de coder.

### Consigne

Etant donne un tableau `prices` ou `prices[i]` est le prix d'une action au jour `i`, maximise ton profit. Tu peux faire autant de transactions que tu veux, mais : tu ne peux pas detenir deux actions a la fois, et apres avoir **vendu**, tu dois attendre **un jour de cooldown** avant de pouvoir racheter.

```python
def max_profit_cooldown(prices: list[int]) -> int:
    """
    Maximize profit with unlimited transactions and a 1-day cooldown after each sell.
    """
    pass
```

**Indice** : trois etats par jour.
- `hold` : je detiens une action (max profit dans cet etat)
- `sold` : je viens de vendre aujourd'hui (cooldown demain)
- `rest` : je ne detiens rien et je ne suis pas en cooldown

Transitions : `hold = max(hold, rest - p)`, `sold = hold + p`, `rest = max(rest, sold)`.

### Tests

```python
assert max_profit_cooldown([1, 2, 3, 0, 2]) == 3      # buy, sell, cooldown, buy, sell
assert max_profit_cooldown([1]) == 0
assert max_profit_cooldown([]) == 0
assert max_profit_cooldown([1, 2, 4]) == 3            # buy at 1, sell at 4
assert max_profit_cooldown([5, 4, 3, 2, 1]) == 0      # decreasing, never buy
assert max_profit_cooldown([2, 1, 4]) == 3            # buy at 1, sell at 4
assert max_profit_cooldown([6, 1, 3, 2, 4, 7]) == 6
```

### Criteres de reussite

- [ ] Utilise les 3 etats (hold, sold, rest) avec leurs transitions
- [ ] Respecte le cooldown : on ne peut racheter qu'apres etre passe par `rest`
- [ ] Complexite O(n) temps, O(1) espace (3 variables roulantes)
- [ ] Gere les tableaux vide / taille 1 / decroissant (profit 0)
- [ ] Tous les tests passent

---

## Exercice 5 : Greedy interval — Non-overlapping Intervals

### Objectif

Maitriser le greedy "trier par fin croissante" pour minimiser les suppressions d'intervalles. Tu dois justifier pourquoi trier par end (et pas par start) est optimal (exchange argument).

### Consigne

Etant donne un ensemble d'intervalles `intervals` ou `intervals[i] = [start, end]`, retourne le **nombre minimum d'intervalles a supprimer** pour que les restants ne se chevauchent pas.

Convention : deux intervalles qui se touchent seulement aux bornes (ex: `[1,2]` et `[2,3]`) ne se chevauchent PAS.

```python
def erase_overlap_intervals(intervals: list[list[int]]) -> int:
    """
    Return the minimum number of intervals to remove so the rest don't overlap.
    Touching endpoints ([1,2] and [2,3]) do NOT count as overlapping.
    """
    pass
```

**Indice** : trie par **end croissant**. Garde le dernier `end` accepte. Pour chaque intervalle, s'il commence avant ce `end`, c'est un chevauchement -> on le supprime (on le compte). Sinon, on l'accepte et on met a jour `end`.

### Tests

```python
assert erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
assert erase_overlap_intervals([[1, 2], [1, 2], [1, 2]]) == 2
assert erase_overlap_intervals([[1, 2], [2, 3]]) == 0       # Touching, no overlap
assert erase_overlap_intervals([]) == 0
assert erase_overlap_intervals([[1, 100], [11, 22], [1, 11], [2, 12]]) == 2
assert erase_overlap_intervals([[0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]) == 2
assert erase_overlap_intervals([[1, 2]]) == 0
```

### Criteres de reussite

- [ ] Trie par **end croissant** (justifie pourquoi pas par start)
- [ ] Compte les suppressions (chevauchements) en un seul passage
- [ ] Gere correctement le cas "touching endpoints" (pas de chevauchement)
- [ ] Complexite O(n log n) temps (tri), O(1) espace additionnel
- [ ] Tous les tests passent

---

## Exercice 6 : Greedy one-pass — Gas Station

### Objectif

Maitriser le greedy one-pass de Gas Station avec ses deux invariants (faisabilite globale + unicite du point de depart). C'est un greedy non evident dont la preuve (exchange argument) est attendue en entretien.

### Consigne

Sur un circuit, il y a `n` stations-service en cercle. `gas[i]` est l'essence disponible a la station `i`, et `cost[i]` est l'essence necessaire pour aller de `i` a `i+1`. Tu pars avec un reservoir vide. Retourne l'index de la station de **depart** qui te permet de faire le tour complet une fois, ou `-1` si c'est impossible. La solution est garantie unique si elle existe.

```python
def can_complete_circuit(gas: list[int], cost: list[int]) -> int:
    """
    Return the starting gas station index to complete the circular route once,
    or -1 if impossible. The answer is guaranteed unique when it exists.
    """
    pass
```

**Indice** : deux observations.
1. Si `sum(gas) < sum(cost)`, c'est impossible -> `-1`.
2. Sinon, balaye en une passe : si le reservoir courant devient negatif a l'index `i`, aucun point entre l'ancien depart et `i` ne peut etre valide -> recommence a `i+1`.

### Tests

```python
assert can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3
assert can_complete_circuit([2, 3, 4], [3, 4, 3]) == -1
assert can_complete_circuit([5], [4]) == 0
assert can_complete_circuit([2], [2]) == 0
assert can_complete_circuit([3, 1, 1], [1, 2, 2]) == 0
assert can_complete_circuit([1, 1, 1], [2, 1, 1]) == -1     # total gas < total cost
assert can_complete_circuit([4, 5, 2, 6, 5, 3], [3, 2, 7, 3, 2, 9]) == -1
```

### Criteres de reussite

- [ ] Detecte l'impossibilite via `sum(gas) < sum(cost)`
- [ ] Reinitialise le reservoir courant et le candidat depart quand le tank passe negatif
- [ ] Sait expliquer l'exchange argument (pourquoi sauter directement a `i+1`)
- [ ] Complexite O(n) temps, O(1) espace
- [ ] Tous les tests passent
