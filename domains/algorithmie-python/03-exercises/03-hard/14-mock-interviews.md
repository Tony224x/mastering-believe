# Exercices Hard — Mock Interviews (fresh problems)

**Format** : deux mocks hard de 35-40 minutes, conditions reelles. Sur du hard, l'intervieweur evalue surtout la demarche : brute force annoncee, optimisation raisonnee, invariants verbalises, tests systematiques.

---

## Mock 7 : Longest Valid Parentheses (Hard)

### Objectif

Le hard de stack le plus pose en mock final : il a l'air d'une variante de Valid Parentheses (jour 4) mais demande un changement de representation — empiler des **indices** et raisonner en "base de fenetre valide". Deux approches valides existent (stack et DP) : tu dois en maitriser une et savoir esquisser l'autre.

### Consigne

Etant donne une string `s` composee uniquement de `'('` et `')'`, retourne la **longueur de la plus longue sous-chaine contigue** de parentheses bien formees.

**Contrainte : O(n) temps, une seule passe** (l'approche O(n^2) qui teste chaque sous-chaine ne valide pas).

```python
def longest_valid_parentheses(s: str) -> int:
    """
    Length of the longest contiguous well-formed parentheses substring.
    O(n) single pass.
    """
    pass
```

**Approche stack (recommandee)** :
- La stack contient des **indices**. Initialise-la avec `-1` (la "base" : dernier index invalide).
- `'('` : push l'index.
- `')'` : pop. Si la stack devient vide, ce `')'` est une nouvelle base → push son index. Sinon, `longueur = i - stack[-1]` (distance a la derniere base/parenthese non fermee).

**Approche DP (a esquisser)** : `dp[i]` = longueur de la plus longue sous-chaine valide **finissant** en i ; transition selon que `s[i - dp[i-1] - 1]` est un `'('`.

### Tests

```python
assert longest_valid_parentheses("(()") == 2
assert longest_valid_parentheses(")()())") == 4
assert longest_valid_parentheses("") == 0
assert longest_valid_parentheses("(") == 0
assert longest_valid_parentheses(")") == 0
assert longest_valid_parentheses("()(()") == 2           # Broken middle resets the base
assert longest_valid_parentheses("()(())") == 6          # Nested + adjacent combine
assert longest_valid_parentheses("(()())") == 6
assert longest_valid_parentheses("())((())") == 4
assert longest_valid_parentheses("()()") == 4            # Adjacent pairs chain
```

### Criteres de reussite

- [ ] Stack d'**indices** avec base initiale `-1` (pas une stack de caracteres)
- [ ] Le cas "stack vide apres pop" pousse le nouvel index base
- [ ] Les sous-chaines adjacentes se combinent (`"()()"` → 4, pas 2)
- [ ] O(n) temps, O(n) espace
- [ ] L'approche DP alternative est esquissee (en commentaire ou a l'oral)
- [ ] Tous les tests passent

---

## Mock 8 : Basic Calculator (Hard)

### Objectif

Le mock de parsing par excellence : evaluer une expression avec `+`, `-` et **parentheses imbriquees** sans `eval`. Teste la capacite a derouler un etat (resultat courant + signe) et a le sauvegarder/restaurer sur une stack a chaque parenthese — le meme mecanisme qu'une pile d'appels.

### Consigne

Etant donne une string `s` representant une expression valide avec des entiers non negatifs, `+`, `-`, `(`, `)` et des espaces, retourne sa valeur. **`eval` et equivalents interdits.**

Le `-` peut etre **unaire** apres une parenthese ouvrante : `"-(3 + 4)"` ou `"(-2)"` doivent marcher.

**Contrainte : O(n) temps, une seule passe caractere par caractere.**

```python
def calculate(s: str) -> int:
    """
    Evaluate +, -, parentheses, spaces. No eval(). O(n) single pass.
    """
    pass
```

**L'etat a maintenir** :
- `result` : somme accumulee du niveau courant.
- `sign` : +1 ou -1, signe du prochain operande.
- `num` : l'entier multi-chiffres en cours de lecture (attention : `"123"` = trois caracteres, UN nombre).

**Aux parentheses** : `'('` → push `(result, sign)` et reset ; `')'` → finalise le niveau, pop et combine : `result = saved_result + saved_sign * result`.

### Tests

```python
assert calculate("1 + 1") == 2
assert calculate(" 2-1 + 2 ") == 3
assert calculate("(1+(4+5+2)-3)+(6+8)") == 23
assert calculate("123") == 123                   # Multi-digit number
assert calculate("0") == 0
assert calculate("-(3 + 4)") == -7               # Unary minus
assert calculate("(-2)") == -2
assert calculate("2-(5-6)") == 3                 # Sign distribution over parens
assert calculate("- (3 + (4 + 5))") == -12
assert calculate("1-(-7)") == 8                  # Double negative
assert calculate("10 - (2 + 3) + (4 - 1)") == 8
```

### Criteres de reussite

- [ ] Nombres multi-chiffres accumules correctement (`num = num * 10 + digit`)
- [ ] Le nombre en cours est flush au bon moment (operateur, parenthese, fin de string)
- [ ] `(result, sign)` sauvegardes sur la stack a chaque `'('`, combines au `')'`
- [ ] Le moins unaire marche (`"-(...)"`, `"(-2)"`) — `num` vaut 0 par defaut
- [ ] Aucun `eval`/`exec`/`ast.literal_eval`
- [ ] O(n) temps, O(p) espace ou p = profondeur de parenthesage
- [ ] Tous les tests passent, y compris le dernier flush en fin de string
