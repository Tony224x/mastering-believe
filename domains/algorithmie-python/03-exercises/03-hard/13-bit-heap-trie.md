# Exercices Hard — Bit Manipulation, Heaps & Tries

---

## Exercice 7 : Trie + DFS — Word Search II

### Objectif

LE probleme qui justifie l'existence des tries en entretien : chercher PLUSIEURS mots dans une grille en partageant les prefixes. Lancer Word Search (exercice du jour 12) mot par mot est O(W * R * C * 4^L) — le trie mutualise tout.

### Consigne

Etant donne une grille `board` de lettres et une liste `words`, retourne tous les mots de la liste qui peuvent etre traces dans la grille par cases adjacentes (4 directions), chaque case utilisee au plus une fois par mot.

**Contraintes** :
- UN SEUL parcours DFS de la grille guide par un trie construit depuis `words` (pas un DFS par mot).
- Chaque mot ne doit apparaitre qu'une fois dans le resultat, meme s'il est tracable par plusieurs chemins.
- **Pruning impose** : marquer le mot comme trouve (effacer `node.word`) au moment de la collecte, pour ne pas le re-collecter.
- **Bonus pruning** : supprimer les feuilles mortes du trie (`node` sans enfants apres collecte) pour accelerer les grilles denses.

```python
def find_words(board: list[list[str]], words: list[str]) -> list[str]:
    """
    All words from the list traceable in the board.
    One trie-guided DFS, not one DFS per word.
    """
    pass
```

### Tests

```python
board = [
    ["o", "a", "a", "n"],
    ["e", "t", "a", "e"],
    ["i", "h", "k", "r"],
    ["i", "f", "l", "v"],
]
assert sorted(find_words([row[:] for row in board], ["oath", "pea", "eat", "rain"])) == ["eat", "oath"]

assert find_words([["a", "b"], ["c", "d"]], ["abcb"]) == []
assert sorted(find_words([["a", "a"]], ["a", "aa", "aaa"])) == ["a", "aa"]   # "aaa" needs 3 cells
assert find_words([["a"]], []) == []

# A word that is a PREFIX of another — both must be found
board2 = [["a", "b", "c"]]
assert sorted(find_words([row[:] for row in board2], ["ab", "abc"])) == ["ab", "abc"]

# Duplicate paths must not duplicate the word
board3 = [["a", "a"], ["a", "a"]]
assert find_words([row[:] for row in board3], ["aa"]) == ["aa"]
```

### Criteres de reussite

- [ ] Trie construit une fois, mot stocke sur son noeud terminal (`node.word`)
- [ ] Un seul DFS par case de depart, descente simultanee grille/trie
- [ ] Mark/restore des cases (comme Word Search) pendant le DFS
- [ ] `node.word = None` a la collecte — aucune deduplication a posteriori
- [ ] Le test prefixe (`"ab"` et `"abc"`) passe : la collecte ne stoppe pas la descente
- [ ] Tous les tests passent

---

## Exercice 8 : Bit Trie — Maximum XOR of Two Numbers in an Array

### Objectif

Fusionner les deux sujets du jour : un **trie binaire** (bits de poids fort en premier) qui repond a "quel nombre deja insere maximise le XOR avec x ?" en O(32). La solution brute O(n^2) doit etre battue par un benchmark.

### Consigne

Etant donne un tableau `nums` d'entiers non negatifs, retourne la valeur maximale de `nums[i] XOR nums[j]` pour `0 <= i <= j < n`.

**Contraintes** :
- O(n * 32) temps avec un trie de bits. La brute force O(n^2) sert d'oracle.
- Benchmark : sur n = 2000, 4000, 8000 valeurs aleatoires < 2^31, le trie doit scaler lineairement, la brute force quadratiquement.

```python
def find_maximum_xor_brute(nums: list[int]) -> int:
    """O(n^2) oracle."""
    pass

def find_maximum_xor(nums: list[int]) -> int:
    """O(n * 32) — binary trie, greedy bit by bit."""
    pass
```

**Le greedy bit a bit** : pour chaque nombre `x`, descends le trie en essayant a chaque niveau de prendre la branche du **bit oppose** (ca met un 1 dans le XOR a cette position). Si elle n'existe pas, prends l'autre. Comme on traite les bits de poids fort d'abord, le choix glouton est optimal (un 1 en position i vaut plus que tous les bits suivants reunis).

**Pieges** : inserer chaque nombre AVANT ou APRES la requete (gerer le premier element) ; nombre fixe de niveaux (32) meme pour les petits nombres — sinon les prefixes se melangent.

### Tests

```python
for f in (find_maximum_xor_brute, find_maximum_xor):
    assert f([3, 10, 5, 25, 2, 8]) == 28        # 5 XOR 25
    assert f([0]) == 0
    assert f([2, 4]) == 6
    assert f([8, 10, 2]) == 10
    assert f([14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70]) == 127
    assert f([0, 0, 0]) == 0
    assert f([1]) == 0                           # Single element: x XOR x = 0

# Oracle on random inputs
import random
for _ in range(50):
    arr = [random.randint(0, 1 << 20) for _ in range(random.randint(1, 60))]
    assert find_maximum_xor(arr) == find_maximum_xor_brute(arr)
```

### Criteres de reussite

- [ ] Trie binaire a exactement 32 niveaux (ou bit_length du max, constant pour tous)
- [ ] Greedy : branche du bit oppose preferee a chaque niveau, avec justification en commentaire
- [ ] Le cas un seul element retourne 0 (x XOR x)
- [ ] Oracle brute force concordant sur 50 inputs aleatoires
- [ ] Benchmark : scaling lineaire pour le trie vs quadratique pour la brute force
- [ ] O(n * 32) temps, O(n * 32) espace
