# Exercices Hard — Sprint Jour 7

**But** : 2 problemes hard chronometres. C'est le format reel d'un entretien onsite : annonce ta brute force a voix haute avant d'optimiser, et verbalise chaque invariant pendant que tu codes.

---

## Exercice 7 : Sprint 15 min — First Missing Positive

### Objectif

Le hard d'arrays le plus demande : trouver le premier entier positif manquant en **O(n) temps et O(1) espace**. Cible la technique "le tableau est son propre hash map" (index as hash).

### Consigne

Etant donne un tableau d'entiers `nums` non trie (valeurs quelconques : negatives, zeros, doublons, grands nombres), retourne le **plus petit entier strictement positif absent** du tableau.

**Contraintes imposees** :
- O(n) temps.
- O(1) espace auxiliaire — pas de set, pas de dict, pas de tableau de presence. Tu peux modifier `nums` en place.

```python
def first_missing_positive(nums: list[int]) -> int:
    """
    Return the smallest missing positive integer.
    O(n) time, O(1) extra space (in-place mutation allowed).
    """
    pass
```

**Etapes** :
1. Annonce d'abord les solutions sous-optimales : set O(n) espace, tri O(n log n).
2. Observation cle : la reponse est dans `[1, n+1]` (n = taille du tableau). Tout le reste est du bruit.
3. Cyclic sort : place chaque valeur `v` dans `[1, n]` a l'index `v - 1` par swaps successifs.
4. Re-scan : le premier index `i` ou `nums[i] != i + 1` donne la reponse `i + 1`.

**Piege** : la boucle de swap doit verifier `nums[i] != nums[nums[i] - 1]` (pas seulement `nums[i] != i + 1`), sinon les doublons creent une boucle infinie.

### Tests

```python
assert first_missing_positive([1, 2, 0]) == 3
assert first_missing_positive([3, 4, -1, 1]) == 2
assert first_missing_positive([7, 8, 9, 11, 12]) == 1
assert first_missing_positive([]) == 1
assert first_missing_positive([1]) == 2
assert first_missing_positive([2]) == 1
assert first_missing_positive([1, 1]) == 2                  # Duplicates: infinite-loop trap
assert first_missing_positive([2, 2, 2, 2]) == 1
assert first_missing_positive([1, 2, 3, 4, 5]) == 6         # Complete sequence
assert first_missing_positive([-1, -2, -3]) == 1            # All negative
```

### Criteres de reussite

- [ ] Resolu en moins de 15 minutes, brute force annoncee d'abord
- [ ] Cyclic sort en place : chaque valeur valide finit a l'index `v - 1`
- [ ] La condition de swap gere les doublons (pas de boucle infinie sur `[1, 1]`)
- [ ] O(n) temps — chaque element est swappe au plus une fois vers sa place finale
- [ ] O(1) espace auxiliaire — aucune structure de hachage
- [ ] Tous les tests passent

---

## Exercice 8 : Sprint 20 min — Reverse Nodes in k-Group

### Objectif

Le hard de linked list par excellence : inverser par blocs de k, **sans toucher au dernier bloc incomplet**. Teste la gestion rigoureuse des pointeurs sous pression du chrono.

### Consigne

Etant donne la tete d'une linked list et un entier `k`, inverse les noeuds **k par k** et retourne la nouvelle tete. Si le nombre de noeuds restants en fin de liste est inferieur a `k`, ces noeuds restent **dans leur ordre d'origine**.

Tu ne peux pas modifier les valeurs des noeuds — uniquement les pointeurs.

```python
def reverse_k_group(head: ListNode | None, k: int) -> ListNode | None:
    """
    Reverse the list k nodes at a time. A final group smaller than k
    is left untouched. Pointers only, no value swaps.
    """
    pass
```

**Plan d'attaque** :
1. Depuis un dummy head, garde un pointeur `group_prev` sur le noeud avant le bloc courant.
2. Compte k noeuds devant : s'il en manque, stop.
3. Inverse le bloc (reverse standard sur k noeuds), puis recolle : `group_prev.next` → nouvelle tete du bloc, ancienne tete du bloc → noeud suivant le bloc.
4. Avance `group_prev` sur l'ancienne tete du bloc (devenue la queue).

### Tests

```python
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 2)) == [2, 1, 4, 3, 5]
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 3)) == [3, 2, 1, 4, 5]
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 1)) == [1, 2, 3, 4, 5]   # k=1: no-op
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 5)) == [5, 4, 3, 2, 1]   # Whole list
assert to_list(reverse_k_group(build([1, 2]), 3)) == [1, 2]                      # Group too small
assert to_list(reverse_k_group(build([]), 2)) == []
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5, 6]), 2)) == [2, 1, 4, 3, 6, 5]
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5, 6, 7]), 3)) == [3, 2, 1, 6, 5, 4, 7]
```

### Criteres de reussite

- [ ] Resolu en moins de 20 minutes
- [ ] Dummy head utilise — le premier bloc ne demande aucun cas special
- [ ] Le comptage prealable garantit que le dernier bloc incomplet n'est jamais inverse
- [ ] Les recollages avant/apres bloc sont corrects (aucun noeud perdu, pas de cycle)
- [ ] O(n) temps (chaque noeud manipule un nombre constant de fois), O(1) espace
- [ ] Tous les tests passent, y compris k = 1 et k = longueur
