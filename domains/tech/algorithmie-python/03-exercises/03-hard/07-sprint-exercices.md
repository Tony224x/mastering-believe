# Sprint Complementaire Hard — Jour 7

> Prolongement du sprint J7 avec **3 problemes hard** combinant plusieurs patterns J1-J6, sans doublon avec P1-P10 du fichier easy. Ces problemes simulent les questions de fin d'entretien senior (15-20 min chacun). Garde le protocole : reformule, annonce la complexite cible, code en parlant.

---

## Exercice 7 : Stack + design — Min Stack avec getMin O(1) puis Evaluate Expression imbriquee

### Objectif (Hard — 15 min)

Combiner stack de parsing + arbre d'operateurs implicite pour evaluer une expression entierement parenthesee avec `+`, `-`, `*`, `/` et priorite des operateurs. Un cran au-dessus du "Basic Calculator" : ici la priorite `* /` > `+ -` doit etre geree.

### Consigne

Implemente `evaluate(s)` qui evalue une expression arithmetique valide contenant des entiers non negatifs, les operateurs `+ - * /`, des espaces et **sans** parentheses. La priorite standard s'applique : `* /` avant `+ -`. La division est une division entiere tronquee vers zero. N'utilise NI `eval` NI `exec`.

```python
def evaluate(s: str) -> int:
    """
    Evaluate an expression with + - * / and operator precedence (no parens).
    Integer division truncates toward zero. No eval/exec.
    """
    pass
```

**Indice** : parcours en gardant une stack. Pour `+`/`-`, empile le nombre (signe). Pour `*`/`/`, depile le sommet, applique l'operation avec le nombre courant, et rempile. A la fin, la somme de la stack est le resultat. La stack encode la priorite.

### Tests

```python
assert evaluate("3+2*2") == 7
assert evaluate(" 3/2 ") == 1
assert evaluate(" 3+5 / 2 ") == 5
assert evaluate("2*3+4") == 10
assert evaluate("14-3/2") == 13
assert evaluate("1") == 1
assert evaluate("2*2*2*2") == 16
assert evaluate("100/3/3") == 11           # (100/3)=33, 33/3=11
assert evaluate("0") == 0
```

### Criteres de reussite

- [ ] N'utilise NI `eval` NI `exec`
- [ ] Gere correctement la priorite `* /` > `+ -` via la stack
- [ ] Division entiere tronquee vers zero (pas floor)
- [ ] Gere les espaces et les nombres a plusieurs chiffres
- [ ] Complexite O(n) temps, O(n) espace
- [ ] Resolu en moins de 15 minutes chrono

---

## Exercice 8 : Heap + design — Find Median from Data Stream

### Objectif (Hard — 18 min)

Maintenir la mediane d'un flux de donnees en O(log n) par insertion grace a **deux heaps equilibrees** (max-heap pour la moitie basse, min-heap pour la moitie haute). Un probleme de design hard tres classique.

### Consigne

Implemente `MedianFinder` :
- `add_num(num)` : ajoute un entier au flux
- `find_median()` : retourne la mediane de tous les elements ajoutes jusqu'ici (float)

```python
class MedianFinder:
    def __init__(self):
        pass

    def add_num(self, num: int) -> None:
        pass

    def find_median(self) -> float:
        pass
```

**Approche attendue** :
- `low` : max-heap (negation des valeurs) contenant la moitie inferieure
- `high` : min-heap contenant la moitie superieure
- Invariant : `len(low) == len(high)` ou `len(low) == len(high) + 1`
- Mediane : `low[0]` si tailles inegales, sinon moyenne des deux sommets

### Tests

```python
mf = MedianFinder()
mf.add_num(1)
mf.add_num(2)
assert mf.find_median() == 1.5
mf.add_num(3)
assert mf.find_median() == 2.0

mf = MedianFinder()
mf.add_num(5)
assert mf.find_median() == 5.0

mf = MedianFinder()
for x in [6, 10, 2, 6, 5, 0]:
    mf.add_num(x)
assert mf.find_median() == 5.5             # sorted: [0,2,5,6,6,10] -> (5+6)/2

mf = MedianFinder()
for x in [-1, -2, -3]:
    mf.add_num(x)
assert mf.find_median() == -2.0
```

### Criteres de reussite

- [ ] Utilise deux heaps (`heapq`) avec l'invariant d'equilibrage
- [ ] `add_num` en O(log n), `find_median` en O(1)
- [ ] Re-equilibre apres chaque insertion
- [ ] Gere un seul element, nombre pair/impair d'elements, valeurs negatives
- [ ] Resolu en moins de 18 minutes chrono

---

## Exercice 9 : Binary search + greedy — Longest Increasing Subsequence (O(n log n))

### Objectif (Hard — 15 min)

Resoudre LIS en O(n log n) avec la technique "patience sorting" : maintenir un tableau de fins de sous-suites et binary search la position d'insertion. Combine binary search (J6) et un invariant greedy subtil.

### Consigne

Etant donne un tableau `nums`, retourne la longueur de la **plus longue sous-suite strictement croissante** (les elements ne sont pas forcement contigus). Complexite **O(n log n)** visee.

```python
def length_of_lis(nums: list[int]) -> int:
    """
    Return the length of the longest strictly increasing subsequence. O(n log n).
    """
    pass
```

**Indice** : maintiens un tableau `tails` ou `tails[i]` est la plus petite valeur de fin d'une sous-suite croissante de longueur `i+1`. Pour chaque `num`, trouve via `bisect_left` la position d'insertion : si elle est en fin, on agrandit ; sinon on remplace (greedy). La longueur de `tails` est la reponse.

### Tests

```python
assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4    # [2,3,7,101]
assert length_of_lis([0, 1, 0, 3, 2, 3]) == 4
assert length_of_lis([7, 7, 7, 7]) == 1                     # Strict: all equal -> 1
assert length_of_lis([1]) == 1
assert length_of_lis([]) == 0
assert length_of_lis([4, 10, 4, 3, 8, 9]) == 3             # [4,8,9]
assert length_of_lis([1, 3, 6, 7, 9, 4, 10, 5, 6]) == 6
```

### Criteres de reussite

- [ ] Maintient le tableau `tails` + `bisect_left` (ou binary search manuel)
- [ ] Strictement croissant (les egalites remplacent, n'agrandissent pas)
- [ ] Complexite O(n log n) temps, O(n) espace
- [ ] Gere tableau vide, tableau a un element, tous egaux
- [ ] Resolu en moins de 15 minutes chrono
