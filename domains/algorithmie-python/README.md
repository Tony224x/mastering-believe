# Algorithmie & Data Structures — Live Coding Python

> ### Carte d'entree — "Peux-tu commencer ?"
>
> Reponds honnetement a ces 3 questions avant de te lancer :
>
> 1. **Peux-tu lire et ecrire une fonction Python avec boucles, conditions et une `list`/`dict` sans chercher la syntaxe ?** (sinon : revois les bases de Python d'abord)
> 2. **Sais-tu ce que veut dire O(n) vs O(n^2) intuitivement** (« deux boucles imbriquees = plus lent ») ? (sinon : pas grave, le module 01 part de zero la-dessus)
> 3. **As-tu un objectif concret** — entretien tech, competition, ou solidifier tes fondations ? (ca determine ton parcours, voir ci-dessous)
>
> **Prerequis reels** : Python courant (fonctions, classes simples, list/dict comprehensions). Aucune connaissance algo prealable requise — le module 01 reprend la complexite depuis le debut.
>
> **Temps** : ~3-4h par module. Parcours complet ~45h sur 2 semaines (planning ci-dessous), ou etale a ton rythme. Un module seul = une session de 3h autonome.

## Scope

Maitriser les structures de donnees et algorithmes necessaires pour exceller en live coding (entretiens tech, competitions). Objectif : resoudre un probleme medium LeetCode en < 15 min, un hard en < 30 min, en Python, avec une complexite optimale et un code propre.

## Prerequisites

- Python courant (fonctions, classes, list comprehensions)
- Notions de base en complexite (O(n), O(log n))

## Planning (2 semaines)

### Semaine 1 — Fondations & Patterns

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J1 | Complexite & Big-O | Analyser, comparer, identifier les bottlenecks | 3h |
| J2 | Arrays & Strings | Two pointers, sliding window, prefix sum | 3h |
| J3 | Hash Maps & Sets | Frequency counting, grouping, two-sum patterns | 3h |
| J4 | Stacks & Queues | Monotonic stack, BFS foundations, parsing | 3h |
| J5 | Linked Lists | Fast/slow pointers, reversal, merge | 3h |
| J6 | Sorting & Searching | Binary search variations, custom sorts, quickselect | 3h |
| J7 | **Sprint exercices** | 10 problemes chronomentres (easy/medium mix) | 4h |

### Semaine 2 — Patterns avances & Competition

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J8 | Trees & BST | DFS/BFS, traversals, path problems, BST properties | 3h |
| J9 | Graphs | DFS/BFS, topological sort, shortest path, union-find | 3h |
| J10 | Dynamic Programming | Memoization, tabulation, patterns (knapsack, LIS, grid) | 4h |
| J11 | DP avance + Greedy | Interval scheduling, state machines, optimisation | 3h |
| J12 | Backtracking & Recursion | Permutations, combinations, constraint satisfaction | 3h |
| J13 | Bit manipulation, Heaps, Tries | Techniques de niche a haut rendement | 3h |
| J14 | **Mock interviews** | 3 sessions chronometrees format FAANG (45 min chacune) | 4h |

## Parcours express vs complet

Tu n'as pas toujours 2 semaines. Voici comment prioriser selon ton temps :

| Parcours | Modules | Pour qui |
|----------|---------|----------|
| **Express (~5 modules core)** | 01 Complexite · 02 Arrays/Strings · 03 Hash Maps · 08 Trees · 10 DP | Tu as peu de temps avant un entretien, ou tu veux le **20% qui couvre 80%** des questions. Ces 5 patterns reviennent dans la grande majorite des problemes medium. |
| **Solide (+4 modules)** | + 04 Stacks/Queues · 05 Linked Lists · 06 Sorting/Searching · 09 Graphs | Tu vises des entretiens FAANG/scale-up serieusement : il te faut aussi BFS/DFS, binary search et les graphes. |
| **Complet (14 modules)** | Tout, dans l'ordre | Maitrise complete, competition, ou tu construis une vraie memoire musculaire. Les modules 11-13 (DP avance, backtracking, bit/heap/trie) sont les patterns de niche a haut rendement ; le 14 est la mise en situation finale. |

> **Regle simple** : commence toujours par 01 (la complexite est le langage commun de tous les autres modules), puis fais 02-03 (les structures les plus utilisees), puis branche-toi sur 08 et 10 si le temps presse. Les modules de patterns avances supposent les fondations en place.

## Structure du contenu

- `01-theory/` — 1 module theorique par jour (source-of-truth)
- `02-code/` — exemples executables alignes sur la theorie
- `03-exercises/` — easy couvre tous les modules ; medium et hard se concentrent sur les modules 01-03 (fondamentaux)
- `04-projects/` — mini-projets libres (vide pour l'instant)
- `05-projets-guides/` — 3 projets appliques au contexte LogiSim (voir [`shared/logistics-context.md`](../../shared/logistics-context.md))

## Criteres de reussite

- [ ] Resoudre 80% des mediums LeetCode du premier coup en < 20 min
- [ ] Resoudre 50% des hards en < 35 min
- [ ] Identifier le pattern correct en < 2 min apres lecture du probleme
- [ ] Expliquer la complexite temps/espace de chaque solution sans hesiter
- [ ] Code propre : nommage clair, pas de variables jetables, edge cases geres

## Patterns cles (les 15 qui couvrent 90% des problemes)

1. Two Pointers
2. Sliding Window
3. Fast & Slow Pointers
4. Merge Intervals
5. Cyclic Sort
6. In-place Reversal (Linked List)
7. Tree BFS / DFS
8. Two Heaps
9. Subsets (Backtracking)
10. Modified Binary Search
11. Top-K Elements (Heap)
12. K-way Merge
13. Topological Sort
14. Dynamic Programming (5 sous-patterns)
15. Monotonic Stack

## Ressources externes

- **Neetcode 150** — selection curee des problemes LeetCode par pattern
- **Grokking the Coding Interview** — patterns systematises
- **Competitive Programmer's Handbook** (Laaksonen) — PDF gratuit, reference algorithmique
