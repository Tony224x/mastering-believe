# Plan figé domaine `algorithmie-python` (14 modules / 2 semaines)

> Curriculum figé, dérivé du [`README.md`](README.md). Source de vérité pour la structure du domaine.
> **Double palier** : Semaine 1 (J1→J7, fondations & patterns, autonome) + Semaine 2 (J8→J14, patterns avancés & mise en situation). Le **module J7 est un sprint de consolidation** de la S1 ; le **J14** est le capstone de mise en situation.
> **Convention de slug** : pour un module N, `01-theory/NN-x.md`, `02-code/NN-x.py`, `01-theory-qd/NN-x.qd` et les exercices/solutions (`03-exercises/**/NN-x.*`) partagent le même slug numéroté.
> **Couverture exercices** : `01-easy/` couvre les 14 modules ; `02-medium/` et `03-hard/` se concentrent sur les modules 01-03 (fondamentaux les plus rentables). Les corrigés sont dans `03-exercises/solutions/`.
> **Parcours** : voir le `README.md` (section « Parcours express vs complet ») pour la priorisation core/avancé. Cœur (« 20% qui couvre 80% ») : modules 01, 02, 03, 08, 10.
> **REFERENCES.md** : non encore produit (phase 2). Les ressources canoniques sont listées dans le `README.md` (section « Ressources externes » : Neetcode 150, Grokking the Coding Interview, Competitive Programmer's Handbook).

---

## Semaine 1 — Fondations & Patterns (parcours autonome)

## J1 — Complexité & Big-O

- **Concepts clés** :
  - Big-O comme croissance (pas vitesse absolue), 7 classes O(1)→O(n!)
  - Boucles imbriquées (multiplier) vs consécutives (additionner)
  - Complexité espace, trade-off espace/temps
  - Complexité amortie (`list.append` ~1.125x, dict resize x2/x4, union-find)
  - Coûts des opérations Python (`in` list vs set, concat strings, `insert(0)`)
- **Acquis fin de module** : analyser temps ET espace d'un algo, repérer les bottlenecks, viser la bonne complexité selon les contraintes (n ≤ 10^5 → O(n log n)).
- **Stack** : Python stdlib
- **Slug** : `01-complexite-big-o`
- **Temps** : ~3h | **Core**

## J2 — Arrays & Strings

- **Concepts clés** :
  - Two pointers (convergents, lents/rapides)
  - Sliding window (taille fixe et variable)
  - Prefix sum, manipulation in-place
- **Acquis fin de module** : reconnaître et appliquer two pointers / sliding window / prefix sum, résoudre les problèmes d'array/string medium classiques.
- **Stack** : Python stdlib
- **Slug** : `02-arrays-strings`
- **Temps** : ~3h | **Core**

## J3 — Hash Maps & Sets

- **Concepts clés** :
  - Frequency counting, grouping, two-sum patterns
  - Fonctionnement interne (hash, open addressing, load factor, resize x2/x4)
  - Ordre d'insertion préservé (Python 3.7+), `Counter`, `defaultdict`
- **Acquis fin de module** : transformer un O(n^2) en O(n) avec une hash map, maîtriser les patterns de comptage/regroupement.
- **Stack** : Python stdlib
- **Slug** : `03-hashmaps-sets`
- **Temps** : ~3h | **Core**

## J4 — Stacks & Queues

- **Concepts clés** :
  - LIFO / FIFO, `deque` vs `list`
  - Monotonic stack (next greater element, etc.)
  - Fondations BFS, parsing (parenthèses, expressions)
- **Acquis fin de module** : utiliser une pile/file à bon escient, appliquer le monotonic stack, poser les bases de BFS.
- **Stack** : Python stdlib
- **Slug** : `04-stacks-queues`
- **Temps** : ~3h

## J5 — Linked Lists

- **Concepts clés** :
  - Fast/slow pointers (cycle detection, milieu)
  - Reversal in-place, merge de listes triées
  - Dummy head, manipulation de pointeurs sans perte
- **Acquis fin de module** : manipuler des listes chaînées sans bug de pointeur, appliquer fast/slow et la reversal in-place.
- **Stack** : Python stdlib
- **Slug** : `05-linked-lists`
- **Temps** : ~3h

## J6 — Sorting & Searching

- **Concepts clés** :
  - Variations de binary search (borne gauche/droite, sur réponse)
  - Custom comparators (`key`, `functools.cmp_to_key`)
  - Quickselect (k-ième élément), barrière O(n log n) du tri par comparaison
- **Acquis fin de module** : écrire une binary search correcte (off-by-one maîtrisé), trier avec un comparateur custom, appliquer quickselect.
- **Stack** : Python stdlib
- **Slug** : `06-sorting-searching`
- **Temps** : ~3h

## J7 — Sprint exercices (consolidation Semaine 1)

- **Concepts clés** :
  - Méthodologie : attaquer un problème en moins de 15 min (clarifier → exemple → pattern → code → test)
  - 10 problèmes chronométrés (mix easy/medium) couvrant les modules 01-06
- **Acquis fin de module** : enchaîner reconnaissance de pattern + implémentation propre sous contrainte de temps.
- **Stack** : Python stdlib
- **Slug** : `07-sprint-exercices`
- **Temps** : ~4h

---

## Semaine 2 — Patterns avancés & Mise en situation

## J8 — Trees & BST (le pont vers le raisonnement récursif)

- **Concepts clés** :
  - TreeNode, 3 parcours DFS (pré/in/post-ordre), BFS par niveaux
  - Calculs récursifs sur l'arbre (hauteur, somme, diamètre)
  - BST properties, Lowest Common Ancestor (LCA), serialize/deserialize
- **Acquis fin de module** : maîtriser DFS + BFS sur arbre binaire, appliquer les 5 patterns récurrents.
- **Rôle** : module-charnière — premier raisonnement récursif sur structure après 7 modules surtout itératifs (un encadré « pont » rappelle les prérequis : récursion, lien arbres↔listes/piles).
- **Stack** : Python stdlib
- **Slug** : `08-trees-bst`
- **Temps** : ~3h | **Core**

## J9 — Graphs

- **Concepts clés** :
  - Représentations (liste d'adjacence, matrice)
  - DFS / BFS sur graphe, détection de cycle
  - Topological sort (Kahn / DFS), Dijkstra, Union-Find
- **Acquis fin de module** : modéliser un problème en graphe, choisir le bon parcours, appliquer tri topologique / plus court chemin / union-find.
- **Stack** : Python stdlib
- **Slug** : `09-graphs`
- **Temps** : ~3h

## J10 — Dynamic Programming

- **Concepts clés** :
  - Memoization (top-down) vs tabulation (bottom-up)
  - Patterns : Fibonacci/escaliers, House Robber, Knapsack 0/1 (1D), Coin Change
  - LCS (2D + variante rolling O(min(m,n))), LIS (O(n^2) et patience O(n log n)), Grid DP
  - Decision tree pour reconnaître un problème DP
- **Acquis fin de module** : reconnaître un problème DP, choisir memoization/tabulation, appliquer les 6 patterns et optimiser l'espace.
- **Stack** : Python stdlib
- **Slug** : `10-dynamic-programming`
- **Temps** : ~4h | **Core**

## J11 — DP avancé & Greedy

- **Concepts clés** :
  - State machines (stock buy/sell), interval DP
  - Interval scheduling, quand greedy est correct (et quand il échoue)
  - Échange argument (exchange argument) pour prouver un greedy
- **Acquis fin de module** : distinguer DP vs greedy, prouver la validité d'un choix glouton, traiter les DP à états.
- **Stack** : Python stdlib
- **Slug** : `11-dp-avance-greedy`
- **Temps** : ~3h | **Avancé**

## J12 — Backtracking & Recursion

- **Concepts clés** :
  - Squelette du backtracking (choix → explorer → annuler)
  - Permutations, combinaisons, subsets
  - Constraint satisfaction (N-Queens), élagage (pruning)
- **Acquis fin de module** : écrire un backtracking propre, énumérer permutations/combinaisons/subsets, élaguer l'espace de recherche.
- **Stack** : Python stdlib
- **Slug** : `12-backtracking-recursion`
- **Temps** : ~3h | **Avancé**

## J13 — Bit manipulation, Heaps & Tries

- **Concepts clés** :
  - Opérations bit à bit (masques, XOR, comptage de bits)
  - Heaps (`heapq`), top-K, two heaps (médiane glissante)
  - Tries (préfixes, autocomplétion)
- **Acquis fin de module** : appliquer les techniques de niche à haut rendement (bit tricks, top-K avec heap, trie de préfixes).
- **Stack** : Python stdlib
- **Slug** : `13-bit-heap-trie`
- **Temps** : ~3h | **Avancé**

## J14 — Mock interviews (capstone / mise en situation)

- **Concepts clés** :
  - 3 sessions chronométrées format FAANG (45 min chacune)
  - Thinking out loud, gestion des hints, communication
  - Postures d'entretien en 2026 (LLM interdit / autorisé / take-home), transitions vers le system design lite (IC4+)
- **Acquis fin de module** : conduire un entretien complet (clarifier, coder, tester, expliquer la complexité) en conditions réelles.
- **Stack** : Python stdlib
- **Slug** : `14-mock-interviews`
- **Temps** : ~4h

---

## Projets guidés (contexte logistique FleetSim, transversal)

Appliquent les patterns vus en théorie à des cas réels du moteur de simulation FleetSim (voir [`../../shared/logistics-context.md`](../../../shared/logistics-context.md)). Tous **déterministes** (rejouabilité).

| # | Projet | Concepts | Difficulté |
|---|--------|----------|------------|
| 01 | Pathfinding entrepôt A* | graphes, heuristique, priority queue, coûts variables | medium |
| 02 | Couverture sensorielle (Bresenham) | algo de ligne, grille 2D, early-exit, profiling | medium |
| 03 | Operations event queue | heap, simulation discrete-event, invariants | hard |

## Patterns clés (les 15 qui couvrent 90% des problèmes)

1. Two Pointers · 2. Sliding Window · 3. Fast & Slow Pointers · 4. Merge Intervals · 5. Cyclic Sort · 6. In-place Reversal (Linked List) · 7. Tree BFS / DFS · 8. Two Heaps · 9. Subsets (Backtracking) · 10. Modified Binary Search · 11. Top-K Elements (Heap) · 12. K-way Merge · 13. Topological Sort · 14. Dynamic Programming (5 sous-patterns) · 15. Monotonic Stack
