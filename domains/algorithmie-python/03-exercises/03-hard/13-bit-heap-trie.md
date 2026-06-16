# Exercices Hard — Bit Manipulation, Heaps & Tries

> Heaps pour les flux et les fenetres glissantes, tries pour la recherche avec wildcard, et combinaison trie + DFS pour les grilles.

---

## Exercice 7 : Heap — Find Median from Data Stream

### Objectif

Maintenir la mediane d'un flux en O(log n) par insertion via la technique des **deux heaps** : un max-heap pour la moitie basse, un min-heap pour la moitie haute, equilibres. Un grand classique des structures de donnees en entretien.

### Consigne

Implemente la classe `MedianFinder` :
- `add_num(num)` : ajoute `num` au flux
- `find_median()` : retourne la mediane de tous les elements ajoutes (float)

```python
class MedianFinder:
    def __init__(self):
        pass

    def add_num(self, num: int) -> None:
        pass

    def find_median(self) -> float:
        pass
```

**Indice** : `low` = max-heap (negation) pour la moitie inferieure, `high` = min-heap pour la superieure. Invariants : tout element de `low` <= tout element de `high`, et `len(low) - len(high) in {0, 1}`. Pour ajouter : push dans `low`, deplace le max de `low` vers `high`, puis re-equilibre si `high` est plus gros. Mediane : si tailles egales, moyenne des deux sommets ; sinon le sommet de `low`.

### Tests

```python
mf = MedianFinder()
mf.add_num(1)
mf.add_num(2)
assert mf.find_median() == 1.5
mf.add_num(3)
assert mf.find_median() == 2.0

mf2 = MedianFinder()
mf2.add_num(5)
assert mf2.find_median() == 5.0           # Single element

import statistics
import random
mf3 = MedianFinder()
rng = random.Random(7)
seen = []
for _ in range(200):
    x = rng.randint(-50, 50)
    mf3.add_num(x)
    seen.append(x)
    assert mf3.find_median() == statistics.median(seen)   # Cross-check
```

### Criteres de reussite

- [ ] Deux heaps : max-heap (low) + min-heap (high)
- [ ] Invariant de partition (low <= high) et d'equilibrage des tailles
- [ ] `add_num` en O(log n), `find_median` en O(1)
- [ ] Gere un seul element, nombres negatifs, longue sequence
- [ ] Cross-check avec `statistics.median` valide

---

## Exercice 8 : Trie — Word Search II (trie + DFS)

### Objectif

Optimiser la recherche de PLUSIEURS mots dans une grille en construisant un trie : une seule DFS par cellule suit le trie et coupe des qu'un prefixe n'existe pas. Combinaison trie + backtracking, le sommet du module.

### Consigne

Etant donne une grille `board` de caracteres et une liste de mots `words`, retourne tous les mots de `words` qui peuvent etre formes par un chemin de cellules **adjacentes** (4-directions), sans reutiliser une cellule dans un meme mot.

```python
def find_words(board: list[list[str]], words: list[str]) -> list[str]:
    """
    Return all words from `words` that exist in the grid (trie + DFS).
    Order of returned words does not matter.
    """
    pass
```

**Indice** : construis un trie avec tous les mots (stocke le mot complet sur le noeud `is_end`). DFS par cellule en descendant le trie : si la lettre n'est pas un enfant du noeud courant, coupe. Quand tu atteins un noeud `is_end`, ajoute le mot et **demarque-le** (`is_end = False`) pour eviter les doublons. Marque/restaure la cellule comme dans word search.

### Tests

```python
board = [
    ["o", "a", "a", "n"],
    ["e", "t", "a", "e"],
    ["i", "h", "k", "r"],
    ["i", "f", "l", "v"],
]
assert sorted(find_words(board, ["oath", "pea", "eat", "rain"])) == ["eat", "oath"]
assert find_words([["a"]], ["b"]) == []
assert sorted(find_words([["a", "b"]], ["a", "b", "ab", "ba"])) == ["a", "ab", "b", "ba"]
assert find_words([["a", "a"]], ["aaa"]) == []         # Not enough cells
assert sorted(find_words([["a"]], ["a"])) == ["a"]
```

### Criteres de reussite

- [ ] Trie construit depuis `words`, mot stocke sur le noeud final
- [ ] Une seule DFS par cellule qui descend le trie (coupe si prefixe absent)
- [ ] Demarquage (`is_end = False`) pour eviter les doublons
- [ ] Marquage/restauration de cellule (pas de reutilisation)
- [ ] Gere grille 1x1, mot trop long, aucun match
- [ ] Complexite O(R * C * 4^L) independante du nombre de mots

---

## Exercice 9 : Trie + DFS wildcard — Add and Search Word

### Objectif

Etendre le trie a la recherche avec **wildcard** `.` (matche n'importe quel caractere). Le `.` force un DFS qui essaie tous les enfants du noeud courant — le pont entre trie et backtracking pour le pattern matching.

### Consigne

Implemente la classe `WordDictionary` :
- `add_word(word)` : ajoute `word` au dictionnaire
- `search(word)` : retourne `True` si un mot ajoute matche `word`. `word` peut contenir des `.` ou chaque `.` matche **n'importe quel** caractere.

```python
class WordDictionary:
    def __init__(self):
        pass

    def add_word(self, word: str) -> None:
        pass

    def search(self, word: str) -> bool:
        pass
```

**Indice** : `add_word` est un insert de trie standard. `search` est un DFS `(node, idx)` : pour un caractere normal, descends dans l'enfant correspondant (ou echoue) ; pour un `.`, essaie **tous** les enfants du noeud (recursion). Succes quand `idx == len(word)` ET le noeud courant a `is_end`.

### Tests

```python
wd = WordDictionary()
wd.add_word("bad")
wd.add_word("dad")
wd.add_word("mad")
assert wd.search("pad") is False
assert wd.search("bad") is True
assert wd.search(".ad") is True              # Matches bad, dad, mad
assert wd.search("b..") is True              # Matches bad
assert wd.search("...") is True
assert wd.search("....") is False            # No 4-letter word
assert wd.search("..") is False             # No 2-letter word

wd2 = WordDictionary()
assert wd2.search("a") is False              # Empty dictionary
assert wd2.search(".") is False
```

### Criteres de reussite

- [ ] `add_word` insere dans le trie comme d'habitude
- [ ] `search` gere le `.` par DFS sur tous les enfants du noeud
- [ ] Succes uniquement si la longueur matche ET `is_end` au noeud final
- [ ] Gere dictionnaire vide, wildcard total (`...`), longueur non matchee
- [ ] Complexite `search` O(26^d) au pire (d = nombre de `.`), sinon O(L)
