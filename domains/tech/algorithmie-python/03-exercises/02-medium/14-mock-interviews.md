# Mocks Medium — Entretiens Chronometres

> **Format** : ces exercices ne sont PAS de simples problemes — ce sont des **mocks chronometres**. Pour chacun : lance un timer, applique les 6 etapes du process (clarifier → exemples → brute force → optimiser → coder → tester) **a voix haute**, puis auto-evalue-toi avec la grille. Ne regarde la solution qu'APRES.

> **Process en 6 etapes (rappel)** : 1. Clarifier · 2. Exemples & edge cases · 3. Brute force · 4. Optimiser · 5. Coder en parlant · 6. Tester + complexite.

---

## Exercice 4 : Mock chrono 30 min — Group Anagrams

### Objectif

Simuler un entretien medium complet sur un probleme de hashing/grouping. L'enjeu n'est pas seulement de resoudre, mais de **derouler le process** et de **verbaliser** ta demarche dans le temps imparti.

### Consigne

**Chronometre 30 minutes.** Enonce a traiter comme en entretien :

> Etant donne une liste de strings `strs`, regroupe les **anagrammes** ensemble. Retourne la liste des groupes (l'ordre des groupes et l'ordre dans chaque groupe n'importent pas).

```python
def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Group strings that are anagrams of each other.
    """
    pass
```

**Deroule attendu** (a faire a voix haute) :
1. **Clarifier** : casse sensible ? caracteres uniquement a-z ? strings vides possibles ? doublons ?
2. **Exemples** : `["eat","tea","tan","ate","nat","bat"]` → `[["eat","tea","ate"],["tan","nat"],["bat"]]`. Edge cases : `[]`, `[""]`, un seul mot.
3. **Brute force** : comparer chaque paire en triant → O(n^2 * k log k). Mentionne-la.
4. **Optimiser** : cle = signature de l'anagramme (mot trie, ou tuple des 26 comptes). `dict signature → liste`. O(n * k log k) ou O(n * k).
5. **Coder** en nommant tes variables.
6. **Tester** : derouler l'exemple, annoncer complexite temps **et** espace.

### Tests (a lancer APRES ton mock)

```python
def normalize(groups):
    return sorted(sorted(g) for g in groups)

assert normalize(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])) == \
    normalize([["bat"], ["nat", "tan"], ["ate", "eat", "tea"]])
assert group_anagrams([""]) == [[""]]
assert group_anagrams(["a"]) == [["a"]]
assert normalize(group_anagrams([])) == []
assert normalize(group_anagrams(["abc", "bca", "cab", "xyz"])) == \
    normalize([["abc", "bca", "cab"], ["xyz"]])
```

### Criteres de reussite — Grille d'auto-evaluation (note /10)

- [ ] **Clarification (1 pt)** : tu as pose au moins 2 questions avant de coder
- [ ] **Exemples & edge cases (1 pt)** : tu as cite `[]`, `[""]` spontanement
- [ ] **Brute force enoncee (1 pt)** : O(n^2 * k log k) verbalisee
- [ ] **Optimisation expliquee AVANT de coder (2 pts)** : signature → dict
- [ ] **Code correct (2 pts)** : passe tous les tests du premier coup
- [ ] **Communication continue (1 pt)** : pas de silence > 30 s
- [ ] **Complexite annoncee (1 pt)** : temps O(n*k log k) ET espace O(n*k)
- [ ] **Dans les temps (1 pt)** : code termine et teste avant 30 min
- **Score < 6/10** : refais un mock similaire demain. **>= 8/10** : passe au suivant.

---

## Exercice 5 : Mock chrono 35 min — Product of Array Except Self

### Objectif

Mock medium sur un probleme a contrainte forte (pas de division, O(n), O(1) espace extra). L'entretien teste ta capacite a **reagir a la contrainte** : la solution naive est interdite, tu dois pivoter.

### Consigne

**Chronometre 35 minutes.**

> Etant donne un tableau d'entiers `nums`, retourne un tableau `answer` ou `answer[i]` est le produit de **tous** les elements de `nums` **sauf** `nums[i]`. Contrainte : **pas de division**, et vise **O(n)** temps. Bonus : O(1) espace extra (hors output).

```python
def product_except_self(nums: list[int]) -> list[int]:
    """
    answer[i] = product of all nums except nums[i], without division, O(n).
    """
    pass
```

**Deroule attendu** :
1. **Clarifier** : il peut y avoir des zeros ? (oui, c'est le piege de la division). Overflow ? (en Python non). Taille mini ?
2. **Exemples** : `[1,2,3,4]` → `[24,12,8,6]`. Avec un zero : `[0,1,2]` → `[2,0,0]`. Deux zeros : `[0,0,3]` → `[0,0,0]`.
3. **Brute force** : double boucle O(n^2). Mentionne-la, puis explique pourquoi la division (O(n)) est interdite/piegeuse avec les zeros.
4. **Optimiser** : prefix products puis suffix products en deux passes. O(n) temps, O(1) extra (l'output sert d'accumulateur).
5. **Coder**.
6. **Tester** + complexite.

### Tests (a lancer APRES ton mock)

```python
assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
assert product_except_self([0, 1, 2]) == [2, 0, 0]      # One zero
assert product_except_self([0, 0, 3]) == [0, 0, 0]      # Two zeros
assert product_except_self([2, 3]) == [3, 2]
assert product_except_self([5]) == [1]                  # Single element
```

### Criteres de reussite — Grille (note /10)

- [ ] **Clarification (1 pt)** : tu as demande "y a-t-il des zeros ?" (cle ici)
- [ ] **Exemples avec zeros (1 pt)** : tu as teste le cas 1 zero ET 2 zeros mentalement
- [ ] **Brute force + pourquoi pas la division (2 pts)** : zeros cassent la division
- [ ] **Prefix/suffix explique AVANT de coder (2 pts)**
- [ ] **Code correct (2 pts)** : passe tous les tests
- [ ] **Espace O(1) extra atteint (1 pt)** : output reutilise comme accumulateur
- [ ] **Complexite annoncee (1 pt)** : O(n) temps, O(1) extra
- **Score < 6/10** : revois le module 2 (arrays) puis refais. **>= 8/10** : continue.

---

## Exercice 6 : Mock chrono 35 min — Coin Change

### Objectif

Mock medium sur un probleme de DP. L'entretien teste si tu sais **reconnaitre un probleme DP** et formuler la recurrence proprement, sans paniquer devant l'explosion combinatoire de la brute force.

### Consigne

**Chronometre 35 minutes.**

> Etant donne une liste `coins` de valeurs de pieces et un entier `amount`, retourne le **nombre minimal** de pieces necessaires pour atteindre `amount`. Si c'est impossible, retourne `-1`. Tu disposes d'un nombre illimite de chaque piece.

```python
def coin_change(coins: list[int], amount: int) -> int:
    """
    Return the minimum number of coins to make `amount`, or -1 if impossible.
    """
    pass
```

**Deroule attendu** :
1. **Clarifier** : valeurs positives ? `amount` peut etre 0 ? (oui → 0 pieces). Pieces dupliquees ?
2. **Exemples** : `coins=[1,2,5], amount=11` → 3 (5+5+1). `coins=[2], amount=3` → -1.
3. **Brute force** : recursion exponentielle (essayer chaque piece) → trop lent.
4. **Optimiser** : DP bottom-up. `dp[x] = 1 + min(dp[x - c] for c in coins if c <= x)`, `dp[0] = 0`, init a l'infini. O(amount * len(coins)).
5. **Coder**.
6. **Tester** + complexite, et bien gerer le cas impossible (`-1`).

### Tests (a lancer APRES ton mock)

```python
assert coin_change([1, 2, 5], 11) == 3       # 5 + 5 + 1
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0              # Zero amount = zero coins
assert coin_change([1], 2) == 2
assert coin_change([2, 5, 10, 1], 27) == 4   # 10 + 10 + 5 + 2
assert coin_change([186, 419, 83, 408], 6249) == 20
assert coin_change([5], 3) == -1
```

### Criteres de reussite — Grille (note /10)

- [ ] **Clarification (1 pt)** : tu as demande "amount peut etre 0 ?"
- [ ] **Reconnaissance DP (2 pts)** : tu as identifie la sous-structure optimale
- [ ] **Recurrence formulee AVANT de coder (2 pts)** : `dp[x] = 1 + min(...)`
- [ ] **Init correcte (1 pt)** : `dp[0] = 0`, le reste a l'infini
- [ ] **Code correct (2 pts)** : passe tous les tests, gere le `-1`
- [ ] **Complexite annoncee (1 pt)** : O(amount * len(coins))
- [ ] **Dans les temps (1 pt)** : termine avant 35 min
- **Score < 6/10** : revois le module 10 (DP). **>= 8/10** : tu es pret pour les mocks hard.
