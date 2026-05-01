# Exercices Hard — Complexite & Big-O

---

## Exercice 7 : Master Theorem et recursions complexes

### Objectif

Analyser des recursions non triviales en utilisant le Master Theorem et le raisonnement sur les arbres d'appels.

### Consigne

Determine la complexite temps de chacune des relations de recurrence suivantes. Justifie avec le Master Theorem (T(n) = aT(n/b) + O(n^d)) ou par analyse directe de l'arbre.

```
1. T(n) = 2T(n/2) + O(n)        — (merge sort)
2. T(n) = 2T(n/2) + O(1)        — (binary tree traversal)
3. T(n) = T(n/2) + O(n)         — (???)
4. T(n) = 3T(n/3) + O(n)        — (???)
5. T(n) = 2T(n/2) + O(n^2)      — (???)
```

Ensuite, implemente en Python une fonction qui correspond a chaque recurrence et mesure empiriquement que ton analyse theorique est correcte (le temps doit scaler comme prevu quand n double).

### Criteres de reussite

- [ ] Les 5 complexites sont correctes avec justification (Master Theorem ou arbre)
- [ ] Master Theorem applique correctement : identifier a, b, d, puis comparer log_b(a) vs d
- [ ] Au moins 3 fonctions Python implementees et mesurees
- [ ] Les mesures empiriques confirment l'analyse theorique (ratio ~2x pour O(n), ~2.5x pour O(n log n), etc.)
- [ ] Cas 3 et 5 correctement identifies comme des cas ou le travail a chaque niveau domine

---

## Exercice 8 : Conception sous contrainte de complexite

### Objectif

Resoudre un probleme reel en respectant une contrainte de complexite stricte, comme en entretien technique FAANG.

### Consigne

**Probleme** : Etant donne un flux de nombres entiers (stream), implemente une classe `MedianFinder` qui supporte :
- `add_num(num: int)` — ajoute un nombre au flux
- `find_median() -> float` — retourne la mediane de tous les nombres ajoutes

**Contraintes** :
- `add_num` doit etre O(log n)
- `find_median` doit etre O(1)
- L'espace total doit etre O(n)

**Etapes** :
1. Explique d'abord pourquoi une approche naive (tri a chaque appel) serait O(n log n) par `add_num`.
2. Explique pourquoi garder une liste triee avec `bisect.insort` serait O(n) par `add_num` (a cause du decalage).
3. Implemente la solution optimale avec deux heaps (max-heap pour la moitie inferieure, min-heap pour la moitie superieure).
4. Prouve que ta solution respecte les contraintes avec des mesures de temps.
5. Gere les edge cases : premier element, nombre pair/impair d'elements, doublons.

### Criteres de reussite

- [ ] Les deux approches sous-optimales sont expliquees avec leur complexite
- [ ] Solution avec deux heaps implementee correctement
- [ ] `add_num` est bien O(log n) — mesure empirique avec n croissant
- [ ] `find_median` est bien O(1) — temps constant quel que soit n
- [ ] Edge cases geres : stream vide (erreur propre), un seul element, doublons
- [ ] Code commente expliquant l'invariant des deux heaps (tailles equilibrees, max_heap[0] ≤ min_heap[0])
- [ ] Tests avec au moins 10 valeurs verifiant la mediane a chaque etape
