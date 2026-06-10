# Exercices Hard — Graphs

---

## Exercice 7 : BFS sur graphe implicite — Word Ladder

### Objectif

Faire du BFS sur un graphe qui n'existe pas en memoire : les noeuds sont des mots, les aretes "different d'une lettre". Le piege de performance : construire les aretes naivement est O(N^2 * L) — il faut les buckets wildcard.

### Consigne

Etant donne deux mots `begin_word` et `end_word` et un dictionnaire `word_list`, retourne la **longueur de la plus courte sequence de transformation** de `begin_word` vers `end_word` (nombre de mots, extremites incluses), telle que :
- Chaque mot adjacent differe d'exactement UNE lettre.
- Chaque mot intermediaire (et `end_word`) appartient a `word_list`.

Retourne `0` si aucune transformation n'existe.

**Contrainte de complexite : O(N * L^2)** ou N = nombre de mots, L = longueur des mots. La comparaison de toutes les paires de mots (O(N^2 * L)) ne valide pas pour de grands N.

```python
def ladder_length(begin_word: str, end_word: str, word_list: list[str]) -> int:
    """
    Return the length of the shortest transformation sequence, or 0.
    """
    pass
```

**Indice (buckets wildcard)** : pre-calcule un dict `pattern → [mots]` ou chaque mot de longueur L genere L patterns (`"hot"` → `"*ot"`, `"h*t"`, `"ho*"`). Deux mots sont voisins ssi ils partagent un pattern. Le BFS explore via ces buckets.

**Pieges** : `end_word` absent de `word_list` → 0 immediat ; marquer visited au moment de l'enqueue ; vider un bucket apres usage (ou set visited) pour eviter les re-parcours.

### Tests

```python
assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0     # cog not in list
assert ladder_length("a", "c", ["a", "b", "c"]) == 2
assert ladder_length("hot", "dog", ["hot", "dog"]) == 0    # No 1-letter bridge
assert ladder_length("hit", "hit", ["hit"]) == 1            # Already there
assert ladder_length("ab", "cd", ["ad", "cd"]) == 3         # ab -> ad -> cd
```

### Criteres de reussite

- [ ] Buckets wildcard pre-calcules — pas de comparaison de toutes les paires
- [ ] BFS niveau par niveau (la longueur de sequence = niveau + 1)
- [ ] `end_word` absent → 0 sans lancer le BFS
- [ ] Visited marque a l'enqueue ; buckets vides apres usage pour rester O(N * L^2)
- [ ] Tu sais expliquer pourquoi BFS (et pas DFS) garantit le plus court chemin ici
- [ ] Tous les tests passent

---

## Exercice 8 : Topological Sort — Alien Dictionary

### Objectif

Le hard de topological sort le plus selectif : il faut d'abord **construire le graphe** (extraire les contraintes d'ordre des paires de mots), puis le trier — et detecter DEUX types d'invalidite differents.

### Consigne

Une langue alien utilise l'alphabet latin mais dans un ordre inconnu. On te donne `words`, une liste de mots **triee selon l'ordre lexicographique de cette langue**.

Retourne **une** string contenant les lettres uniques de la langue, triees selon l'ordre alien. S'il n'existe aucun ordre valide, retourne `""`. S'il existe plusieurs ordres valides, retourne n'importe lequel.

```python
def alien_order(words: list[str]) -> str:
    """
    Return one valid letter ordering for the alien language, or "".
    """
    pass
```

**Construction du graphe** : pour chaque paire de mots **adjacents**, trouve le premier caractere qui differe : `c1 != c2` → arete `c1 → c2`. Une seule arete par paire (les caracteres suivants ne disent rien).

**Les deux cas invalides a distinguer** :
1. **Prefixe inverse** : `["abc", "ab"]` — un mot suivi de son propre prefixe est impossible dans un ordre lexicographique → `""` (a detecter PENDANT la construction, ce n'est pas un cycle).
2. **Cycle** : contraintes contradictoires (`a < b` et `b < a`) → Kahn ne consomme pas toutes les lettres → `""`.

### Tests

```python
assert alien_order(["wrt", "wrf", "er", "ett", "rftt"]) == "wertf"
assert alien_order(["z", "x"]) == "zx"
assert alien_order(["z", "x", "z"]) == ""                # Cycle: z < x < z
assert alien_order(["abc", "ab"]) == ""                  # Inverted prefix — NOT a cycle
assert alien_order(["ab", "abc"]) != ""                  # Valid prefix order

# Single word: any permutation of its unique letters is valid
result = alien_order(["zyx"])
assert sorted(result) == ["x", "y", "z"]

# Letters with no constraints must still appear in the output
result = alien_order(["ac", "ab", "zc", "zb"])
assert set(result) == {"a", "b", "c", "z"}
assert result.index("c") < result.index("b")             # Only constraint: c < b
```

### Criteres de reussite

- [ ] Une arete par paire de mots adjacents (premier caractere different uniquement)
- [ ] Le cas "prefixe inverse" retourne `""` des la construction du graphe
- [ ] Kahn's algorithm (BFS + in-degree) ; cycle detecte par `len(result) < nombre de lettres`
- [ ] Toutes les lettres presentes dans `words` apparaissent dans le resultat, meme sans contrainte
- [ ] Pas d'aretes dupliquees (sinon les in-degrees sont faux) — utiliser des sets
- [ ] O(C) temps ou C = somme des longueurs des mots
- [ ] Tous les tests passent
