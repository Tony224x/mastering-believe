# Mocks Hard — Entretiens Chronometres (45 min, niveau senior)

> **Format** : ces exercices sont des **mocks chronometres niveau hard** (rounds onsite FAANG / senior). Pour chacun : lance un timer **45 min**, deroule les 6 etapes du process (clarifier → exemples → brute force → optimiser → coder → tester) **a voix haute**, puis auto-evalue-toi avec la grille. Ne regarde la solution qu'APRES. Chaque mock se termine par une **variante "design lite"** : la question de relance qu'un interviewer senior pose une fois l'algo resolu.

> **Process en 6 etapes (rappel)** : 1. Clarifier · 2. Exemples & edge cases · 3. Brute force · 4. Optimiser · 5. Coder en parlant · 6. Tester + complexite.

> **Reglage du timer hard** : 5 min clarif + plan · 25 min code · 10 min test/complexite · 5 min variante design. Si a 25 min tu n'as pas de code qui tourne, code la brute force et annonce-le.

---

## Exercice 7 : Mock chrono 45 min — Merge Intervals

### Objectif

Simuler un entretien hard sur un probleme d'intervalles. L'enjeu : reconnaitre que **trier debloque le probleme** (les chevauchements deviennent forcement consecutifs), gerer proprement les chevauchements aux bornes (endpoint touch), et enchainer sur une relance "design lite" sur le streaming.

### Consigne

**Chronometre 45 minutes.** Enonce a traiter comme en entretien :

> Etant donne un tableau d'intervalles `intervals` ou `intervals[i] = [start_i, end_i]`, fusionne tous les intervalles qui se chevauchent et retourne la liste des intervalles non chevauchants couvrant toutes les entrees (ordre croissant de `start`).

```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """
    Merge all overlapping intervals; return non-overlapping intervals sorted by start.
    """
    pass
```

**Deroule attendu** (a voix haute) :
1. **Clarifier** : `[1,4]` et `[4,5]` se chevauchent-ils (endpoint touch) ? (conventionnellement OUI, on fusionne). Intervalles deja tries ? Single-point `[2,2]` possible ? Tableau vide ?
2. **Exemples** : `[[1,3],[2,6],[8,10],[15,18]]` → `[[1,6],[8,10],[15,18]]`. Piege : `[[1,4],[2,3]]` → `[[1,4]]` (le 2e est inclus dans le 1er, on garde le max des `end`).
3. **Brute force** : comparer chaque paire et fusionner en boucle jusqu'a stabilite → O(n^2) voire pire. Mentionne-la.
4. **Optimiser** : **trier par `start`** O(n log n), puis une passe : pour chaque intervalle, soit il chevauche le dernier du resultat (`start <= last_end` → `last_end = max(last_end, end)`), soit non (append).
5. **Coder** en nommant clairement (`result`, `last`, `cur_start`, `cur_end`).
6. **Tester** + complexite (O(n log n) temps, O(n) espace).

### Tests (a lancer APRES ton mock)

```python
def norm(out):
    return sorted([list(x) for x in out])

assert norm(merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])) == [[1, 6], [8, 10], [15, 18]]
assert norm(merge_intervals([[1, 4], [4, 5]])) == [[1, 5]]        # Endpoint touch fusionne
assert norm(merge_intervals([[1, 4], [2, 3]])) == [[1, 4]]        # Inclusion totale
assert norm(merge_intervals([[1, 4], [0, 4]])) == [[0, 4]]
assert merge_intervals([]) == []                                  # Vide
assert merge_intervals([[5, 7]]) == [[5, 7]]                      # Singleton
assert norm(merge_intervals([[1, 4], [0, 0]])) == [[0, 0], [1, 4]]  # Disjoints
```

### Variante "design lite" (5 dernieres min)

> "Les intervalles arrivent en **streaming** (un flux infini), tu dois pouvoir interroger a tout moment l'ensemble fusionne. Comment fais-tu ?"
> Reponse attendue : structure ordonnee (arbre balance / `sortedcontainers`, ou skip-list) keyee par `start` ; a chaque insertion, fusionne avec les voisins gauche/droite en O(log n + k) ou k = intervalles absorbes. Trade-off : insertion log vs. re-tri complet a chaque fois.

### Criteres de reussite — Grille d'auto-evaluation (note /10)

- [ ] **Clarification (1 pt)** : tu as demande "endpoint touch fusionne ?" avant de coder
- [ ] **Exemples & edge cases (1 pt)** : tu as cite l'inclusion totale `[[1,4],[2,3]]` et le vide
- [ ] **Insight du tri verbalise AVANT de coder (2 pts)** : "trier rend les chevauchements consecutifs"
- [ ] **Code correct (2 pts)** : passe tous les tests du premier coup, `max` sur les `end`
- [ ] **Communication continue (1 pt)** : pas de silence > 30 s
- [ ] **Complexite annoncee (1 pt)** : O(n log n) temps, O(n) espace
- [ ] **Variante design abordee (1 pt)** : tu as propose une structure ordonnee pour le streaming
- [ ] **Dans les temps (1 pt)** : code teste avant 35 min
- **Score < 6/10** : revois le module 6 (sorting) puis refais. **>= 8/10** : passe au suivant.

---

## Exercice 8 : Mock chrono 45 min — LRU Cache (design)

### Objectif

Mock hard de **design de structure de donnees** : combiner hashmap + liste doublement chainee pour atteindre O(1) sur `get` ET `put`. C'est LE probleme de design le plus pose en entretien senior — il teste si tu sais composer deux structures pour eliminer le goulot.

### Consigne

**Chronometre 45 minutes.**

> Implemente un cache LRU (Least Recently Used) de capacite fixe :
> - `get(key)` : retourne la valeur si presente, sinon `-1`. Un acces compte comme une utilisation recente.
> - `put(key, value)` : insere/met a jour. Si la capacite est depassee, **evince l'element le moins recemment utilise**.
> Les deux operations doivent etre **O(1)** en moyenne.

```python
class LRUCache:
    def __init__(self, capacity: int):
        pass

    def get(self, key: int) -> int:
        pass

    def put(self, key: int, value: int) -> None:
        pass
```

**Deroule attendu** :
1. **Clarifier** : capacite >= 1 ? `put` d'une cle existante = mise a jour + refresh recency ? Valeurs negatives autorisees (sinon `-1` ambigu) ? (oui, `-1` = absence, c'est le contrat).
2. **Exemples** : capacite 2, `put(1,1) put(2,2) get(1)=1 put(3,3)` evince 2, `get(2)=-1`.
3. **Brute force** : liste + scan lineaire pour trouver le LRU → O(n) par operation. Inacceptable.
4. **Optimiser** : **hashmap `key -> node`** + **liste doublement chainee** (most-recent en tete, least-recent en queue). `get`/`put` : O(1) pour localiser via le map, O(1) pour deplacer le noeud en tete. Eviction = supprimer la queue. **Astuce** : sentinelles `head`/`tail` pour eviter les `if None`. (Alternative acceptee a mentionner : `collections.OrderedDict` + `move_to_end`.)
5. **Coder** : `_remove(node)`, `_add_front(node)`, puis `get`/`put` qui les composent.
6. **Tester** + complexite (O(1) amorti, O(capacity) espace).

### Tests (a lancer APRES ton mock)

```python
c = LRUCache(2)
c.put(1, 1)
c.put(2, 2)
assert c.get(1) == 1            # 1 devient le plus recent
c.put(3, 3)                     # capacite depassee -> evince 2 (LRU)
assert c.get(2) == -1
c.put(4, 4)                     # evince 1
assert c.get(1) == -1
assert c.get(3) == 3
assert c.get(4) == 4

c2 = LRUCache(2)
c2.put(2, 1)
c2.put(2, 2)                    # update, pas d'eviction
assert c2.get(2) == 2
c2.put(1, 1)
c2.put(4, 1)                    # evince 2 (1 vient d'etre touche par get? non: 2 est LRU)
assert c2.get(2) == -1

c3 = LRUCache(1)               # capacite 1 (edge)
c3.put(1, 10)
c3.put(2, 20)                  # evince 1
assert c3.get(1) == -1
assert c3.get(2) == 20
```

### Variante "design lite" (5 dernieres min)

> "Ton cache doit etre **thread-safe** et partage entre process. Comment l'adaptes-tu ?"
> Reponse attendue : (1) en mono-process multi-thread : un lock global ou un lock par shard (sharding par hash(key)) pour reduire la contention ; (2) en multi-process / distribue : Redis avec politique d'eviction `allkeys-lru` (le travail O(1) est delegue), TTL, et invalidation. Trade-off : latence reseau vs. capacite, coherence eventuelle.

### Criteres de reussite — Grille (note /10)

- [ ] **Clarification (1 pt)** : tu as confirme le contrat `get` absent = `-1` et le refresh sur `put` existant
- [ ] **Brute force enoncee (1 pt)** : liste + scan O(n), pourquoi inacceptable
- [ ] **Choix hashmap + DLL explique AVANT de coder (2 pts)** : O(1) localisation + O(1) deplacement
- [ ] **Sentinelles ou OrderedDict (1 pt)** : tu evites les cas `None` a la main
- [ ] **Code correct (2 pts)** : passe tous les tests, eviction du vrai LRU
- [ ] **Complexite annoncee (1 pt)** : O(1) amorti get/put, O(capacity) espace
- [ ] **Variante design abordee (1 pt)** : thread-safety / Redis distribue
- [ ] **Dans les temps (1 pt)** : code teste avant 40 min
- **Score < 6/10** : revois les modules 3 (hashmaps) et 5 (linked lists). **>= 8/10** : continue.

---

## Exercice 9 : Mock chrono 45 min — Trapping Rain Water

### Objectif

Mock hard sur un probleme d'optimisation d'espace : passer d'une solution O(n) memoire (prefix/suffix max) a une solution **O(1) espace via two pointers**. Le test : savoir pivoter quand l'interviewer dit "peux-tu faire en O(1) espace ?".

### Consigne

**Chronometre 45 minutes.**

> Etant donne `height`, une liste d'entiers >= 0 representant des barres de largeur 1, calcule la quantite d'eau de pluie piegee apres la pluie.

```python
def trap(height: list[int]) -> int:
    """
    Total trapped rainwater. Water above index i = min(max_left, max_right) - height[i].
    """
    pass
```

**Deroule attendu** :
1. **Clarifier** : hauteurs negatives ? (non, >= 0). Liste vide / 1-2 barres ? (0 eau). Doit-on retourner le volume total, pas la forme ?
2. **Exemples** : `[0,1,0,2,1,0,1,3,2,1,2,1]` → 6. Insight : l'eau au-dessus de l'index `i` vaut `min(max_a_gauche, max_a_droite) - height[i]` (si positif).
3. **Brute force** : pour chaque `i`, scanner gauche et droite pour les max → O(n^2).
4. **Optimiser (palier 1)** : precalculer `prefix_max` et `suffix_max` → O(n) temps, **O(n) espace**.
5. **Optimiser (palier 2, attendu en hard)** : **two pointers** `left`/`right` avec `left_max`/`right_max`. On avance le cote dont le max est le plus petit : ce cote est le facteur limitant, donc on peut comptabiliser son eau immediatement. → O(n) temps, **O(1) espace**.
6. **Tester** + complexite. Annonce explicitement le tradeoff entre les deux paliers.

### Tests (a lancer APRES ton mock)

```python
assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap([4, 2, 0, 3, 2, 5]) == 9
assert trap([]) == 0
assert trap([1]) == 0
assert trap([2, 2]) == 0
assert trap([5, 4, 3, 2, 1]) == 0       # Strictement decroissant -> rien
assert trap([1, 2, 3, 4, 5]) == 0       # Strictement croissant -> rien
assert trap([3, 0, 3]) == 3
assert trap([0, 0, 0]) == 0
```

### Variante "design lite" (5 dernieres min)

> "Les hauteurs sont un **flux** (tu ne peux pas tout garder en memoire), ou bien la version 2D (carte de hauteurs). Comment generalises-tu ?"
> Reponse attendue : (1) en streaming pur, le two-pointer ne marche plus (besoin du max droit) → fenetre/segment tree, ou on accepte deux passes ; (2) version 2D ("Trapping Rain Water II") = **BFS/Dijkstra avec un min-heap** depuis le bord vers l'interieur, l'eau d'une cellule = max(niveau frontiere courant) - hauteur. Mentionne juste l'idee (heap sur la bordure), pas l'implem complete.

### Criteres de reussite — Grille (note /10)

- [ ] **Clarification (1 pt)** : tu as demande "volume total ?" et confirme hauteurs >= 0
- [ ] **Insight `min(maxL, maxR) - h` verbalise (2 pts)** : la formule cle avant de coder
- [ ] **Palier O(n) espace propose (1 pt)** : prefix/suffix max
- [ ] **Pivot vers two pointers O(1) (2 pts)** : tu as su repondre a "peux-tu faire en O(1) espace ?"
- [ ] **Code correct (2 pts)** : passe tous les tests (croissant, decroissant, vide)
- [ ] **Complexite annoncee (1 pt)** : O(n) temps, O(1) espace pour la version finale
- [ ] **Variante 2D / streaming abordee (1 pt)** : heap sur la bordure mentionne
- **Score < 6/10** : revois le module 2 (two pointers) puis refais. **>= 8/10** : tu es pret pour les onsites hard. Felicitations, tu as boucle les 14 modules.
