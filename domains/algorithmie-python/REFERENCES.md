# REFERENCES — `algorithmie-python` (14 jours)

> Sources de tier-1 par module pour le domaine **Algorithmie & Data Structures — Live Coding Python**.
> Source de vérité pour les modules J1..J14. Titres / éditions / années vérifiés via WebSearch/WebFetch (juin 2026).
> Convention de citation : `Auteurs (année). *Titre* (chapitre/section si pertinent). — note 1 ligne`.
> Les manuels génériques (CLRS, Sedgewick, Skiena, CP Handbook) et les plateformes (LeetCode, NeetCode) sont regroupés en bas dans **Ressources transversales** ; chaque module pointe vers les chapitres pertinents.

---

## Module 01 — Complexité & Big-O

- Cormen, Leiserson, Rivest, Stein (2022). *Introduction to Algorithms*, 4e éd. (MIT Press), ch. 2-3 « Getting Started » & « Growth of Functions / Asymptotic notation ». — Définition rigoureuse de O / Θ / Ω, l'analyse de référence pour Big-O.
- Python Software Foundation. *TimeComplexity* — Python Wiki. https://wiki.python.org/moin/TimeComplexity — Tableau de référence des coûts amortis des opérations `list`/`dict`/`set` en CPython (page en cours d'archivage mais toujours canonique).
- Skiena (2020). *The Algorithm Design Manual*, 3e éd. (Springer), ch. 2 « Algorithm Analysis ». — Approche pragmatique et intuitive de l'analyse asymptotique, orientée résolution de problèmes.

## Module 02 — Arrays & Strings (Two Pointers, Sliding Window, Prefix Sum)

- Sedgewick & Wayne (2011). *Algorithms*, 4e éd. (Addison-Wesley), ch. 1.3-1.4 « Bags, Queues, and Stacks / Analysis of Algorithms » + booksite. https://algs4.cs.princeton.edu/home/ — Fondations sur les tableaux et l'analyse, complément avec code exécutable.
- Laaksonen (2018). *Competitive Programmer's Handbook* (draft libre), ch. 8-9 « Amortized analysis » & « Range queries ». https://cses.fi/book/book.pdf — Two pointers, sliding window et prefix sums présentés comme techniques de base du CP.
- LeetCode. *Two Pointers* & *Sliding Window* (problèmes de référence : #1 Two Sum, #167 Two Sum II, #3 Longest Substring Without Repeating Characters, #560 Subarray Sum Equals K). https://leetcode.com/ — Problèmes canoniques pour ancrer chaque pattern.

## Module 03 — Hash Maps & Sets (Frequency Counting, Grouping, Two-Sum)

- Cormen, Leiserson, Rivest, Stein (2022). *Introduction to Algorithms*, 4e éd. (MIT Press), ch. 11 « Hash Tables ». — Tables de hachage, fonctions de hachage, résolution de collisions, analyse en moyenne O(1).
- Hettinger (2017). *Modern Python Dictionaries: A confluence of a dozen great ideas*, PyCon 2017. https://www.youtube.com/watch?v=npw4s1QTmPg — Internals du `dict` compact-et-ordonné de CPython (key-sharing, compaction) par le core dev qui l'a conçu.
- Python Software Foundation. *collections — Container datatypes* (`Counter`, `defaultdict`). https://docs.python.org/3/library/collections.html — Docs officielles des outils Python privilégiés pour le frequency counting et le grouping.

## Module 04 — Stacks & Queues (LIFO, FIFO, Monotonic Stack, BFS Foundations)

- Sedgewick & Wayne (2011). *Algorithms*, 4e éd. (Addison-Wesley), ch. 1.3 « Bags, Queues, and Stacks ». https://algs4.cs.princeton.edu/13stacks/ — Implémentations et usages canoniques des stacks/queues.
- Python Software Foundation. *collections.deque*. https://docs.python.org/3/library/collections.html#collections.deque — File à deux bouts O(1) aux deux extrémités : la structure correcte pour BFS et monotonic deque (vs `list.pop(0)` en O(n)).
- LeetCode. Problèmes de référence : #20 Valid Parentheses, #739 Daily Temperatures (monotonic stack), #84 Largest Rectangle in Histogram, #239 Sliding Window Maximum (monotonic deque). https://leetcode.com/ — Ancrage des patterns LIFO/FIFO/monotonic.

## Module 05 — Linked Lists (Fast/Slow Pointers, Reversal, Merge)

- Cormen, Leiserson, Rivest, Stein (2022). *Introduction to Algorithms*, 4e éd. (MIT Press), ch. 10.2 « Linked lists ». — Représentation et opérations de base sur les listes chaînées.
- Floyd (cycle detection) — « Floyd's tortoise and hare » / patience de l'algorithme fast-slow. Voir CLRS ch. 10 + problème exposé. https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_tortoise_and_hare — Algorithme canonique de détection de cycle en O(1) espace (à vérifier : Floyd n'a pas de publication primaire isolée ; attribution traditionnelle).
- LeetCode. Problèmes de référence : #206 Reverse Linked List, #21 Merge Two Sorted Lists, #141/#142 Linked List Cycle (I/II), #876 Middle of the Linked List. https://leetcode.com/ — Couvre reversal, merge et fast/slow.

## Module 06 — Sorting & Searching (Binary Search, Quickselect, Custom Comparators)

- Peters, T. *listsort.txt* — description de Timsort dans la source CPython. https://github.com/python/cpython/blob/main/Objects/listsort.txt — Document de référence du concepteur de Timsort (runs, galloping, binary insertion sort < 64 éléments) ; tri stable O(n log n) de Python.
- Cormen, Leiserson, Rivest, Stein (2022). *Introduction to Algorithms*, 4e éd. (MIT Press), ch. 7 « Quicksort » & ch. 9 « Medians and Order Statistics » (quickselect / selection en O(n) moyen). — Base théorique de quickselect et de l'analyse du tri.
- Python Software Foundation. *bisect — Array bisection algorithm* & *Sorting HOW TO* (`key=`, stabilité). https://docs.python.org/3/library/bisect.html — `bisect_left`/`bisect_right` pour lower_bound/upper_bound, et le contrat de stabilité des `sorted`/`list.sort`.
- LeetCode. Problèmes de référence : #704 Binary Search, #33 Search in Rotated Sorted Array, #215 Kth Largest Element (quickselect/heap). https://leetcode.com/ — Variantes de binary search et quickselect.

## Module 07 — Sprint méthodologie (attaquer un problème en < 15 min)

- McDowell (2015). *Cracking the Coding Interview*, 6e éd. (CareerCup). https://www.crackingthecodinginterview.com/ — Référence sur le processus d'entretien (BUD, approche par étapes, communication) ; appui de la méthode en 6 étapes du module (à vérifier : 6e éd. 2015 est la dernière courante).
- NeetCode. *Roadmap* & *How to use NeetCode effectively*. https://neetcode.io/roadmap — Ordre de pratique par patterns ; cadre la sélection des 10 problèmes chronométrés du sprint (modules 01-06).
- Laaksonen (2018). *Competitive Programmer's Handbook* (draft libre), ch. 1 « Introduction ». https://cses.fi/book/book.pdf — Mindset de résolution rapide et arsenal de techniques transversales.

## Module 08 — Trees & BST (DFS, BFS, BST, LCA, Serialize)

- Cormen, Leiserson, Rivest, Stein (2022). *Introduction to Algorithms*, 4e éd. (MIT Press), ch. 12 « Binary Search Trees » + ch. 10.4 « Representing rooted trees ». — Propriétés des BST, parcours (inorder/preorder/postorder), opérations.
- Sedgewick & Wayne (2011). *Algorithms*, 4e éd. (Addison-Wesley), ch. 3.2 « Binary Search Trees ». https://algs4.cs.princeton.edu/32bst/ — Implémentation BST commentée avec code exécutable et visualisations.
- LeetCode. Problèmes de référence : #104 Maximum Depth, #102 Level Order Traversal (BFS), #98 Validate BST, #236 Lowest Common Ancestor, #297 Serialize and Deserialize Binary Tree. https://leetcode.com/ — Couvre les 5 patterns du module.

## Module 09 — Graphs (DFS, BFS, Topological Sort, Dijkstra, Union-Find)

- Cormen, Leiserson, Rivest, Stein (2022). *Introduction to Algorithms*, 4e éd. (MIT Press), Part VI « Graph Algorithms » (ch. 20 BFS/DFS/topological sort, ch. 22 Dijkstra, ch. 19 Disjoint sets / union-find). — Référence canonique pour tous les algorithmes de graphes du module.
- Sedgewick & Wayne (2011). *Algorithms*, 4e éd. (Addison-Wesley), ch. 4 « Graphs » (4.1 Undirected, 4.2 Directed, 4.4 Shortest Paths) + ch. 1.5 « Union-Find ». https://algs4.cs.princeton.edu/40graphs/ — Représentations, BFS/DFS, topo sort, Dijkstra et union-find avec code.
- Laaksonen (2018). *Competitive Programmer's Handbook* (draft libre), ch. 11-13 (Basics of graphs, Graph traversal, Shortest paths). https://cses.fi/book/book.pdf — Présentation orientée CP, concise et directement codable.

## Module 10 — Dynamic Programming (Memoization, Tabulation, Classiques)

- Cormen, Leiserson, Rivest, Stein (2022). *Introduction to Algorithms*, 4e éd. (MIT Press), ch. 14 « Dynamic Programming » (rod cutting, LCS, optimal substructure / overlapping subproblems). — Fondations rigoureuses du DP et des deux propriétés clés.
- Skiena (2020). *The Algorithm Design Manual*, 3e éd. (Springer), ch. 10 « Dynamic Programming ». — Méthode pratique « comment trouver l'état DP », nombreux cas (edit distance, knapsack).
- Python Software Foundation. *functools.lru_cache / functools.cache*. https://docs.python.org/3/library/functools.html#functools.cache — Memoization idiomatique en Python (décorateur), utilisée dans le module pour le top-down.
- LeetCode. Problèmes de référence : #70 Climbing Stairs, #322 Coin Change, #1143 Longest Common Subsequence, #300 Longest Increasing Subsequence, #62 Unique Paths. https://leetcode.com/ — Les 6 patterns classiques du module.

## Module 11 — DP avancé & Greedy (State Machine, Interval DP, quand greedy marche)

- Cormen, Leiserson, Rivest, Stein (2022). *Introduction to Algorithms*, 4e éd. (MIT Press), ch. 15 « Greedy Algorithms » (activity selection, matroïdes, propriété du choix glouton). — Critère formel de correction d'un algorithme glouton.
- Aldous & Diaconis (1999). *Longest increasing subsequences: from patience sorting to the Baik-Deift-Johansson theorem*, Bull. Amer. Math. Soc. 36(4), 413-432. https://www.stat.berkeley.edu/~aldous/Papers/me86.pdf — Article canonique reliant patience sorting et LIS (justifie l'algorithme LIS en O(n log n) via piles/binary search).
- LeetCode. Problèmes de référence : #122 Best Time to Buy and Sell Stock II (state machine), #309 with Cooldown, #1235 Maximum Profit in Job Scheduling (interval DP), #55 Jump Game, #435 Non-overlapping Intervals (greedy). https://leetcode.com/ — Distingue les cas DP des cas greedy.

## Module 12 — Backtracking & Recursion (Permutations, Subsets, N-Queens)

- Cormen, Leiserson, Rivest, Stein (2022). *Introduction to Algorithms*, 4e éd. (MIT Press), ch. 2.3 (récursivité / diviser-pour-régner) — base du raisonnement récursif. — Cadre formel pour récursion et arbre d'appels.
- Skiena (2020). *The Algorithm Design Manual*, 3e éd. (Springer), ch. 9 « Combinatorial Search » (backtracking, pruning, N-Queens, sudoku). — Template universel du backtracking et techniques d'élagage — l'appui direct du module.
- LeetCode. Problèmes de référence : #46 Permutations, #78 Subsets, #77 Combinations, #51 N-Queens, #37 Sudoku Solver, #79 Word Search. https://leetcode.com/ — Couvre génération combinatoire et CSP.

## Module 13 — Bit Manipulation, Heaps & Tries

- Cormen, Leiserson, Rivest, Stein (2022). *Introduction to Algorithms*, 4e éd. (MIT Press), ch. 6 « Heapsort » & 6.5 « Priority Queues » + ch. B.1 (notation binaire). — Heaps binaires, priority queues, opérations O(log n).
- Python Software Foundation. *heapq — Heap queue algorithm*. https://docs.python.org/3/library/heapq.html — Min-heap stdlib (`heappush`/`heappop`/`nlargest`/`nsmallest`) : l'outil top-K et K-way merge du module.
- Sedgewick & Wayne (2011). *Algorithms*, 4e éd. (Addison-Wesley), ch. 5.2 « Tries ». https://algs4.cs.princeton.edu/52trie/ — Tries (R-way, TST), recherche par préfixe en O(L) — appui de la partie trie.
- LeetCode. Problèmes de référence : #136 Single Number (XOR), #338 Counting Bits, #347 Top K Frequent Elements (heap), #208 Implement Trie, #212 Word Search II (trie). https://leetcode.com/ — Bits, heaps et tries.

## Module 14 — Mock Interviews (simuler un vrai entretien tech FAANG)

- McDowell (2015). *Cracking the Coding Interview*, 6e éd. (CareerCup). https://www.crackingthecodinginterview.com/ — Structure d'un entretien FAANG, gestion du temps, lecture des hints, erreurs de communication (à vérifier : 6e éd. dernière publiée).
- NeetCode. *How to Prepare for Coding Interviews*. https://blog.neetcode.io/p/prepare-coding-interviews — Stratégie d'entraînement en conditions réelles et choix des problèmes pour les sessions chronométrées.
- Pramp / interviewing.io (plateformes de mock peer-to-peer). https://www.pramp.com/ — https://interviewing.io/ — Outils pour simuler un entretien live avec un pair (cités comme support de pratique).

---

## Ressources transversales

### Manuels de référence (couvrent plusieurs modules)

- Cormen, Leiserson, Rivest, Stein (2022). *Introduction to Algorithms*, 4e éd. (MIT Press). ISBN 978-0262046305. https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/ — « CLRS » : la référence rigoureuse, source primaire pour la complexité, le tri, les graphes, le DP, le greedy, les heaps.
- Sedgewick & Wayne (2011). *Algorithms*, 4e éd. (Addison-Wesley). ISBN 978-0321573513. Booksite libre : https://algs4.cs.princeton.edu/home/ — Implémentations exécutables, visualisations et MOOC ; excellent complément concret à CLRS.
- Skiena (2020). *The Algorithm Design Manual*, 3e éd. (Springer). ISBN 978-3030542559. https://link.springer.com/book/10.1007/978-3-030-54256-6 — Approche « catalogue de techniques » orientée résolution ; le « War Stories » et le « Hitchhiker's Guide » sont précieux pour reconnaître un pattern.
- Laaksonen (2018). *Competitive Programmer's Handbook* (draft libre, PDF officiel CSES). https://cses.fi/book/book.pdf — Référence concise et codable pour le competitive programming ; aussi publié sous *Guide to Competitive Programming* (Springer).
- McDowell (2015). *Cracking the Coding Interview*, 6e éd. (CareerCup). https://www.crackingthecodinginterview.com/ — Référence sur le *processus* d'entretien (utile pour J7 et J14).

### Plateformes de pratique

- **LeetCode** — https://leetcode.com/ — Banque de problèmes principale, tagués par structure/pattern ; cible des critères de réussite (mediums < 20 min, hards < 35 min).
- **NeetCode 150 / Roadmap** — https://neetcode.io/practice/practice/neetcode150 — Sélection curée (Blind 75 + 75) organisée par pattern, avec solutions vidéo ; suit l'ordre pédagogique du domaine.
- **CSES Problem Set** — https://cses.fi/problemset/ — 300+ problèmes classés par thème (compagnon du CP Handbook), idéal pour la profondeur algorithmique au-delà des entretiens.
- **Pramp / interviewing.io** — https://www.pramp.com/ — https://interviewing.io/ — Mock interviews live entre pairs (appui de J14).

### Documentation Python officielle (internals & stdlib)

- *TimeComplexity* — Python Wiki. https://wiki.python.org/moin/TimeComplexity — Coûts amortis des opérations `list`/`dict`/`set` (J1).
- *listsort.txt* (Tim Peters, source CPython). https://github.com/python/cpython/blob/main/Objects/listsort.txt — Spécification de Timsort (J6).
- *collections* (`Counter`, `defaultdict`, `deque`). https://docs.python.org/3/library/collections.html — Structures idiomatiques (J3, J4).
- *bisect* & *Sorting HOW TO*. https://docs.python.org/3/library/bisect.html — Binary search prête à l'emploi et contrat de stabilité (J6).
- *heapq*. https://docs.python.org/3/library/heapq.html — Min-heap stdlib (J13).
- *functools.cache / lru_cache*. https://docs.python.org/3/library/functools.html#functools.cache — Memoization (J10).

---

## Notes

- **Stack confirmé** : Python stdlib uniquement (cf CLAUDE.md « Algorithmie & System Design : stdlib seulement »). Aucune dépendance externe ; les références « docs Python » ci-dessus suffisent pour tout le code.
- **CLRS = source primaire** pour la théorie (complexité, tri, graphes, DP, greedy, heaps) ; **Sedgewick booksite** = code exécutable + visuels ; **Skiena** = reconnaissance de pattern ; **LeetCode/NeetCode** = pratique.
- **Sources marquées « (à vérifier) »** : (1) attribution de l'algorithme de Floyd (tortoise & hare) — pas de publication primaire isolée, attribution traditionnelle ; (2) édition courante de *Cracking the Coding Interview* (6e éd. 2015 retenue comme dernière publiée). Le reste a été vérifié WebSearch (titres / éditions / années / ISBN) en juin 2026.
