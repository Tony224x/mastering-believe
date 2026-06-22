# Exercices Hard — Dynamic Programming

---

## Exercice 7 : DP 2D — Edit Distance (Levenshtein)

### Objectif

Coder le DP 2D le plus exigeant en entretien : la distance d'edition. Trois transitions (insert / delete / replace) qui doivent etre parfaitement justifiees, plus une optimisation espace O(min(m, n)).

### Consigne

Etant donne deux strings `word1` et `word2`, retourne le **nombre minimum d'operations** pour transformer `word1` en `word2`. Les operations autorisees sont : inserer un caractere, supprimer un caractere, remplacer un caractere.

```python
def min_distance(word1: str, word2: str) -> int:
    """
    Return the minimum number of insert/delete/replace operations to
    transform word1 into word2 (Levenshtein distance).
    """
    pass
```

**Approche attendue** :
- Etat : `dp[i][j]` = distance entre `word1[:i]` et `word2[:j]`
- Base : `dp[i][0] = i` (supprimer i caracteres), `dp[0][j] = j` (inserer j caracteres)
- Si `word1[i-1] == word2[j-1]` : `dp[i][j] = dp[i-1][j-1]` (rien a faire)
- Sinon : `dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])` (delete / insert / replace)

### Tests

```python
assert min_distance("horse", "ros") == 3
assert min_distance("intention", "execution") == 5
assert min_distance("", "") == 0
assert min_distance("abc", "") == 3            # 3 suppressions
assert min_distance("", "abc") == 3            # 3 insertions
assert min_distance("abc", "abc") == 0
assert min_distance("a", "b") == 1             # 1 remplacement
assert min_distance("sunday", "saturday") == 3
assert min_distance("kitten", "sitting") == 3
```

### Criteres de reussite

- [ ] Initialise correctement la premiere ligne et la premiere colonne (i suppressions / j insertions)
- [ ] Les trois transitions (insert, delete, replace) sont presentes et justifiees
- [ ] Complexite O(m * n) temps ; espace O(m * n) ou optimise en O(min(m, n))
- [ ] Gere les deux strings vides et le cas egal (distance 0)
- [ ] Tous les tests passent

---

## Exercice 8 : DP partition — Word Break II (reconstruction)

### Objectif

Passer du DP "decision" (peut-on decouper ?) au DP "construction" (lister TOUS les decoupages). Tu apprends a combiner memoization et generation, en evitant l'explosion exponentielle par memoisation par index.

### Consigne

Etant donne une string `s` et un dictionnaire de mots `word_dict`, ajoute des espaces dans `s` pour construire **toutes les phrases** ou chaque mot est dans le dictionnaire. Retourne toutes ces phrases (ordre indifferent).

```python
def word_break_all(s: str, word_dict: list[str]) -> list[str]:
    """
    Return every sentence formed by inserting spaces in s such that every
    word belongs to word_dict. Each word in the dictionary may be reused.
    """
    pass
```

**Approche attendue** :
- Memoise par **index de depart** : `dfs(start)` retourne la liste de toutes les phrases formables a partir de `s[start:]`.
- Pour chaque prefixe `s[start:end]` present dans le dictionnaire, prefixe chaque phrase de `dfs(end)`.
- Cache `dfs(start)` pour eviter de recalculer les memes suffixes.

### Tests

```python
res = word_break_all("catsanddog", ["cat", "cats", "and", "sand", "dog"])
assert sorted(res) == sorted(["cats and dog", "cat sand dog"])

res = word_break_all("pineapplepenapple",
                     ["apple", "pen", "applepen", "pine", "pineapple"])
assert sorted(res) == sorted([
    "pine apple pen apple",
    "pineapple pen apple",
    "pine applepen apple",
])

# No valid segmentation
assert word_break_all("catsandog", ["cats", "dog", "sand", "and", "cat"]) == []

# Empty string -> one empty sentence
assert word_break_all("", ["a"]) == [""]

# Single word
assert word_break_all("a", ["a"]) == ["a"]
```

### Criteres de reussite

- [ ] Memoise par index de depart (chaque suffixe `dfs(start)` calcule une seule fois)
- [ ] Reutilise les mots du dictionnaire (un mot peut apparaitre plusieurs fois)
- [ ] Retourne `[]` quand aucun decoupage n'existe et `[""]` pour la chaine vide
- [ ] N'a pas de complexite catastrophique sur les cas "many segmentations" grace au cache
- [ ] Tous les tests passent

---

## Exercice 9 : DP avec wildcards — Regular Expression Matching ('.' et '*')

### Objectif

Resoudre le matching regex avec `.` et `*` en DP 2D — l'un des problemes hard les plus discriminants en entretien senior. La difficulte est la gestion du `*` (zero ou plusieurs occurrences du caractere precedent).

### Consigne

Implemente un matching d'expression reguliere ou :
- `.` correspond a **n'importe quel** caractere unique,
- `*` correspond a **zero ou plusieurs** occurrences de l'element **precedent**.

Le matching doit couvrir la string `s` **entiere** (pas un sous-segment). Tu ne dois **PAS** utiliser le module `re`.

```python
def is_match(s: str, p: str) -> bool:
    """
    Return True if pattern p matches the ENTIRE string s.
    '.' matches any single char; '*' matches zero or more of the preceding element.
    No use of the `re` module allowed.
    """
    pass
```

**Approche attendue (DP 2D)** :
- `dp[i][j]` = `s[:i]` matche-t-il `p[:j]` ?
- Base : `dp[0][0] = True`. Pour `p[j-1] == '*'`, `dp[0][j] = dp[0][j-2]` (le `x*` peut matcher vide).
- Si `p[j-1]` matche `s[i-1]` (egal ou `.`) : `dp[i][j] = dp[i-1][j-1]`.
- Si `p[j-1] == '*'` : `dp[i][j] = dp[i][j-2]` (zero occurrence) OU (`p[j-2]` matche `s[i-1]` ET `dp[i-1][j]`).

### Tests

```python
assert is_match("aa", "a") is False
assert is_match("aa", "a*") is True
assert is_match("ab", ".*") is True
assert is_match("aab", "c*a*b") is True
assert is_match("misissipi", "mis*is*p*.") is False
assert is_match("", "") is True
assert is_match("", "a*") is True
assert is_match("", ".*") is True
assert is_match("abc", "") is False
assert is_match("aaa", "a*a") is True
assert is_match("aaa", "ab*a*c*a") is True
assert is_match("a", ".*..a*") is False
```

### Criteres de reussite

- [ ] N'utilise PAS le module `re`
- [ ] Gere `'*'` correctement : zero occurrence (`dp[i][j-2]`) ou une de plus (`dp[i-1][j]`)
- [ ] Initialise la ligne `dp[0][*]` pour les patterns `x*` qui matchent la chaine vide
- [ ] Le matching couvre la string entiere (resultat = `dp[len(s)][len(p)]`)
- [ ] Complexite O(m * n) temps et espace
- [ ] Tous les tests passent, y compris chaines/patterns vides
