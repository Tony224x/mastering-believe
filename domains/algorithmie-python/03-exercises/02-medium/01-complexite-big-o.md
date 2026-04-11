# Exercices Medium — Complexite & Big-O

---

## Exercice 4 : Analyse de complexite recursive

### Objectif

Savoir dessiner l'arbre d'appels et en deduire la complexite d'une fonction recursive.

### Consigne

Analyse la complexite temps et espace de chacune de ces fonctions recursives. Pour chacune, dessine (sur papier ou en commentaire) l'arbre d'appels pour n=4.

```python
# Fonction F
def func_f(n):
    if n <= 0:
        return 0
    return func_f(n - 1) + func_f(n - 1)

# Fonction G
def func_g(n):
    if n <= 0:
        return 0
    return func_g(n - 1) + func_g(n - 2)

# Fonction H
def func_h(arr, lo=0, hi=None):
    if hi is None:
        hi = len(arr) - 1
    if lo >= hi:
        return
    mid = (lo + hi) // 2
    func_h(arr, lo, mid)
    func_h(arr, mid + 1, hi)
    # (imagine a merge step here)
```

Pour la Fonction G, explique la difference avec la Fonction F en termes de complexite. Sont-elles dans la meme classe Big-O ?

### Criteres de reussite

- [ ] Fonction F : complexite temps et espace correctes, arbre dessine
- [ ] Fonction G : complexite temps et espace correctes, comparaison avec F expliquee
- [ ] Fonction H : complexite identifiee comme un merge sort (O(n log n) temps, O(n) espace avec la pile + merge)
- [ ] La profondeur de pile (espace de la recursion) est mentionnee pour chaque fonction

---

## Exercice 5 : Profiler et optimiser

### Objectif

Utiliser le profiling pour confirmer la complexite theorique, puis optimiser.

### Consigne

1. Ecris une fonction `find_duplicates(arr)` qui retourne la liste de tous les elements qui apparaissent plus d'une fois. Commence par une version brute-force O(n^2).

2. Mesure le temps d'execution pour n = 1000, 5000, 10000, 20000. Verifie que le temps quadruple quand n double (confirmation empirique de O(n^2)).

3. Ecris une version optimisee en O(n) avec un dictionnaire de comptage.

4. Compare les temps des deux versions dans un tableau.

### Criteres de reussite

- [ ] Version brute-force correcte et effectivement O(n^2)
- [ ] Mesures montrent bien un facteur ~4x quand n double
- [ ] Version optimisee en O(n) avec dict/Counter
- [ ] Tableau comparatif avec au moins 4 tailles differentes
- [ ] Explication du trade-off espace/temps

---

## Exercice 6 : Piege des operations cachees

### Objectif

Detecter les operations couteuses cachees dans du code Python apparemment simple.

### Consigne

Le code suivant a l'air O(n), mais il est en realite O(n^2) ou pire. Identifie TOUS les problemes de performance et reecris une version optimale.

```python
def process_logs(logs: list[str]) -> dict:
    """
    logs = ["user1:login", "user2:logout", "user1:click", ...]
    Return {user: [actions]} for users with more than 1 action.
    """
    result = {}
    all_users = []

    for log in logs:
        parts = log.split(":")
        user, action = parts[0], parts[1]

        if user not in all_users:        # Problem 1: ???
            all_users.append(user)

        if user not in result:
            result[user] = ""
        result[user] = result[user] + action + ","  # Problem 2: ???

    # Filter users with more than 1 action
    final = {}
    for user in all_users:
        actions = result[user].split(",")
        actions = [a for a in actions if a]  # Remove empty strings
        if len(actions) > 1:
            final[user] = actions

    return final
```

### Criteres de reussite

- [ ] Probleme 1 identifie : `in` sur une list → O(n) par lookup
- [ ] Probleme 2 identifie : concatenation de string → O(k^2) pour k actions par user
- [ ] Version optimisee utilisant set + list (ou defaultdict(list))
- [ ] Complexite de la version optimisee : O(n) temps, O(n) espace
- [ ] Explication de pourquoi le code original est O(n^2) dans le pire cas
