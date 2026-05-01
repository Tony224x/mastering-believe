# Jour 7 — Sprint Methodology : Attaquer un Probleme en Moins de 15 Minutes

> **Temps estime** : 60 min de lecture + 2h30 de sprint | **Objectif** : internaliser un processus reproductible pour attaquer un probleme inconnu en <15 min, puis s'entrainer sur 10 problemes chronometres qui couvrent les patterns des jours 1 a 6.

---

## 1. Pourquoi la methodologie compte plus que la "ruse"

Au niveau senior, ce qui differencie un candidat embauche d'un candidat refuse n'est presque **jamais** le QI ou la connaissance d'astuces exotiques. C'est :

1. **La capacite a attaquer methodiquement** un probleme jamais vu
2. **La vitesse de pattern matching** — reconnaitre en <60 secondes quel pattern s'applique
3. **La discipline sur les edge cases** — ceux qui oublient les inputs vides, negatifs, de taille 1
4. **La communication** — expliquer son raisonnement a voix haute sans bloquer l'interviewer

Ce jour a deux objectifs :
- T'apprendre un **processus en 6 etapes** que tu peux executer sur pilote automatique
- Te faire vivre un **sprint de 10 problemes** pour stresser-tester ton pattern matching

---

## 2. Le processus en 6 etapes (a repeter jusqu'a l'avoir dans les doigts)

### Etape 1 — Repeter l'enonce avec ses propres mots (30 sec)

**Ne JAMAIS** commencer a coder sans avoir reformule. Ca fait deux choses :
- Te force a verifier que tu as compris
- Donne a l'interviewer l'occasion de corriger une mauvaise interpretation avant que tu perdes 10 min

Template oral :
> "Donc si je comprends bien, on me donne X, je dois retourner Y, et la contrainte est Z. Est-ce que c'est correct ?"

### Etape 2 — Poser 2-3 questions de clarification (30 sec)

Les questions standard qui debloquent 90% des ambiguites :
- **Input vide ?** `[]`, `""`, `None`
- **Un seul element ?** `[5]`, `"a"`
- **Doublons autorises ?** Dans le tableau, dans les paires, dans le resultat
- **Taille max de l'input ?** Determine si O(n^2) passe ou pas
- **Valeurs negatives ?** Changent completement certains algos (sliding window, binary search)
- **Tri garanti en entree ?** Change tout pour chercher une paire

### Etape 3 — Donner un exemple simple et le resoudre a la main (2 min)

Prends un exemple de taille 3-5, et resous-le manuellement au tableau. Ca fait deux choses :
- Verifie ta comprehension une seconde fois
- Te donne des INDICES sur la structure de donnees a utiliser

**Si a la fin de cette etape tu ne "sens" pas de pattern, tu es en danger.** Passe directement a un brute force explicite.

### Etape 4 — Annoncer la brute force, sa complexite, puis chercher l'optimisation (3 min)

Template oral :
> "Une approche naive serait de faire X, ce qui serait O(n^2) en temps. Je pense qu'on peut faire mieux en utilisant Y..."

**Regle** : l'interviewer veut voir que tu es conscient de la complexite ET que tu cherches a l'ameliorer. Annoncer la brute force en premier est toujours valorise.

### Etape 5 — Ecrire le code en parlant (5-7 min)

- Ecris des noms de variables explicites : `seen_chars`, pas `s`
- Commente les invariants importants : `# at this point, left is the start of the valid window`
- **Annonce** ce que tu ecris ("je vais maintenant gerer le cas ou...")
- Si tu as un bug, ne panique pas — relis ligne par ligne a voix haute

### Etape 6 — Tester mentalement sur l'exemple et un edge case (2 min)

Apres avoir ecrit le code, execute-le a la main sur :
1. L'exemple donne par l'enonce
2. UN edge case : input vide, taille 1, ou cas extreme

Template oral :
> "Testons sur l'exemple [1, 2, 3]... [dry run]... ok ca retourne 5 comme attendu. Maintenant testons sur le cas vide [] — ma boucle ne s'execute pas, je retourne 0, c'est correct."

---

## 3. Checklist de pattern recognition (memoriser par coeur)

Quand tu lis un enonce, passe-le dans cette grille. **Plus de 80% des problemes sont deja la.**

| Signal dans l'enonce | Pattern probable | Structure |
|---------------------|------------------|-----------|
| "sorted array" + "pair/triplet" | Two pointers | Pointers |
| "substring" + "longest/shortest" | Sliding window | Window + hash |
| "subarray" + "sum = K" | Prefix sum + hashmap | Prefix sum |
| "anagram" / "frequency" | Frequency counting | Counter |
| "group by..." | Grouping | defaultdict(list) |
| "two sum" / "pair with diff" | Complement lookup | dict |
| "duplicate" / "seen before" | Seen set | set |
| "valid parentheses" / "balanced" | Stack matching | list as stack |
| "next greater" / "daily temperatures" | Monotonic stack | list as stack |
| "level order" / "shortest path unweighted" | BFS | deque |
| "connected components" / "islands" | BFS ou DFS | deque/set |
| "middle of list" / "cycle" | Fast/slow pointers | Pointers |
| "reverse the list" | 3-pointer reversal | Pointers |
| "merge sorted lists" | Merge + dummy head | Dummy sentinel |
| "rotated sorted" | Modified binary search | Pointers |
| "K-th largest/smallest" | Quickselect ou heap | heapq |
| "shortest path weighted" | Dijkstra | heapq |

---

## 4. Time-boxing : repartir 45 minutes d'entretien

Un entretien algo standard dure 45 minutes. Voici la repartition cible :

| Phase | Duree | Ce que tu fais |
|-------|-------|----------------|
| Intro + lecture de l'enonce | 2 min | Ecoute, repete, clarifie |
| Exemple + pattern matching | 3 min | Dessine un exemple, identifie le pattern |
| Brute force + optimisation | 3 min | Annonce brute, propose mieux, valide avec l'interviewer |
| Codage | 15-20 min | Ecris en parlant, commente les invariants |
| Tests mentaux + edge cases | 5 min | Dry run sur l'exemple + 1 edge case |
| Complexite + discussion | 3 min | Annonce T(n) et S(n), discute des trade-offs |
| Questions a l'interviewer | 5 min | Team, culture, challenges techniques |

**Regle de panique** : si a 10 min tu n'as toujours rien ecrit, AVOUE et demande un indice. Les interviewers preferent un candidat qui sait demander de l'aide a un candidat qui bloque en silence.

---

## 5. Recap des patterns J1-J6

Avant le sprint, relis mentalement ce recap — c'est le "cheat sheet" que tu aurais dans ta tete en entretien :

### J1 — Complexite et Arrays

- O(1), O(log n), O(n), O(n log n), O(n^2), O(2^n)
- `list`, `tuple`, slicing, `enumerate`, `zip`
- Prefix sum pour les sommes de subarray en O(1)

### J2 — Strings, Two Pointers, Sliding Window

- Deux pointeurs : triee et cherche une paire → two pointers
- Sliding window : "plus longue/courte" substring avec contrainte
- Anatomie du window : expand right, shrink left

### J3 — Hash Maps & Sets

- Counter, defaultdict(int), defaultdict(list)
- Two-sum pattern : stocker complement dans un dict
- Seen set pour detecter doublons et cycles

### J4 — Stacks & Queues

- Stack (list) pour matching et parsing
- Monotonic stack pour next greater/smaller (O(n) amorti)
- Queue (deque) pour BFS — JAMAIS `list.pop(0)`
- Marquer visited a l'enqueue, pas au dequeue

### J5 — Linked Lists

- Fast/slow pour milieu, cycle, nth from end
- Reversal : prev/curr/next, sauvegarder next en premier
- Merge avec dummy head et tail pointer

### J6 — Sorting & Searching

- `sorted(arr, key=...)` avec tuples pour criteres composes
- Binary search 4 variantes : exact, lower_bound, upper_bound, rotated
- Quickselect O(n) moyen pour K-ieme element

---

## 6. Le Sprint — 10 problemes en 2h30

Regles du sprint :
- **15 minutes MAX par probleme.** Chronometre-toi strictement.
- Si tu bloques a 12 minutes, annonce a voix haute ton brute force et ecris-le.
- Apres chaque probleme, reviens ici et coche la checklist mentale : "quel pattern ? ai-je parle en codant ? ai-je teste les edge cases ?"
- A la fin des 10 problemes, compare avec les solutions de `07-sprint-exercices.py` et fais un debriefing de 20 min.

### Les 10 problemes

**P1 (Easy) — Two Sum**
Tableau `nums` non trie et `target`. Retourne les INDICES de deux elements dont la somme vaut `target`. Solution unique garantie. → *Pattern J3, hash map lookup.*

**P2 (Easy) — Valid Anagram**
Deux strings `s` et `t`. Retourne `True` si `t` est un anagramme de `s`. → *Pattern J3, frequency counting.*

**P3 (Easy) — Valid Parentheses**
String `s` contenant `()[]{}`. Retourne `True` si bien formee. → *Pattern J4, stack matching.*

**P4 (Easy) — Best Time to Buy and Sell Stock**
Tableau `prices` ou `prices[i]` est le prix au jour `i`. Retourne le profit max d'une seule transaction (achat puis vente). → *Pattern J1, single pass avec min running.*

**P5 (Easy) — Reverse Linked List**
Inverser une linked list singly. Retourner la nouvelle tete. → *Pattern J5, 3-pointer reversal.*

**P6 (Medium) — Longest Substring Without Repeating Characters**
String `s`. Retourne la longueur de la plus longue substring sans caractere repete. → *Pattern J2, sliding window + hash set.*

**P7 (Medium) — Group Anagrams**
Liste de strings. Groupe ensemble celles qui sont des anagrammes. → *Pattern J3, grouping avec defaultdict.*

**P8 (Medium) — Top K Frequent Elements**
Tableau `nums` et entier `k`. Retourne les `k` elements les plus frequents. → *Pattern J3 + J6, Counter + heap ou bucket sort.*

**P9 (Medium) — Search in Rotated Sorted Array**
Tableau trie puis rote une fois. Cherche `target`, retourne son index ou -1. → *Pattern J6, rotated binary search.*

**P10 (Medium) — Number of Islands**
Grille 2D de `'1'` et `'0'`. Compte les iles (composantes connexes de `'1'`). → *Pattern J4, BFS avec queue.*

---

## 7. Debriefing post-sprint (a faire apres les 10 problemes)

Repondre honnetement a ces questions apres le sprint. L'objectif est d'identifier tes zones de faiblesse pour les travailler demain.

1. **Combien ai-je resolu en moins de 15 min ?** Score cible : 7+/10 apres 2 semaines d'entrainement.
2. **Sur quels problemes ai-je bloque ?** Quel pattern m'a echappe ? Pourquoi ?
3. **Ai-je REPETE l'enonce avant de coder ?** (Oui/Non)
4. **Ai-je ANNONCE la brute force avant l'optimisation ?** (Oui/Non)
5. **Ai-je teste les edge cases (vide, taille 1, negatif) ?** (Oui/Non)
6. **Combien de bugs ai-je fait et en combien de temps les ai-je corriges ?**
7. **Quel pattern dois-je re-pratiquer en premier demain ?**

**Regle d'or** : le sprint est un diagnostic, pas une competition. Un score de 4/10 avec une bonne analyse est plus utile qu'un score de 8/10 en trichant avec les solutions.

---

## 8. Flash Cards — Revision espacee

**Q1** : Quelles sont les 6 etapes du processus pour attaquer un probleme ?
> **R1** : 1. Repeter l'enonce, 2. Poser 2-3 questions de clarification, 3. Exemple simple a la main, 4. Brute force + complexite, 5. Coder en parlant, 6. Tester mentalement avec un edge case.

**Q2** : A 10 min sans avoir ecrit une ligne, que fais-tu ?
> **R2** : J'annonce a voix haute mon brute force, meme s'il est O(n^2). Je commence a l'ecrire, et je cherche a l'optimiser pendant l'ecriture. Les interviewers preferent un code sous-optimal mais correct a un silence.

**Q3** : Quels sont les 3 edge cases a tester systematiquement ?
> **R3** : Input vide (`[]`, `""`), input de taille 1 (`[5]`, `"a"`), et un cas extreme lie au probleme (negatif si sommes, doublon si unique, etc.).

**Q4** : Quand "sorted array" apparait dans un enonce, quels sont les 2 patterns a considerer immediatement ?
> **R4** : Two pointers (pour les paires/triplets) et binary search (pour la recherche ou l'optimisation). Si la structure est 1D et qu'on cherche une paire : two pointers. Si on cherche un element : binary search.

**Q5** : Pourquoi repeter l'enonce avec ses propres mots avant de coder ?
> **R5** : Trois raisons : (1) ca te force a verifier ta comprehension, (2) ca donne a l'interviewer l'occasion de corriger une mauvaise interpretation AVANT que tu perdes 10 min, (3) ca montre de la discipline et de la communication claire.

---

## Resume — Key Takeaways

1. **Le processus bat le QI** : 6 etapes systematiques > astuce de genie
2. **Pattern matching en <60 sec** : la checklist J1-J6 couvre 80% des problemes
3. **Time-box strict** : 15 min par probleme, brute force explicite si bloque
4. **Parle en codant** : les interviewers embauchent pour la communication autant que pour le code
5. **Teste TOUJOURS** les edge cases : vide, taille 1, extreme
6. **Sprint = diagnostic** : fais-le honnetement, ton score n'a pas d'importance — ce qui compte c'est l'analyse post-sprint
7. **Patterns cles a memoriser** : two pointers, sliding window, hash map, stack/queue, linked list, binary search
