# Solutions — Module 04 : Difficultés désirables

---

## Exercice 1 — Comparer blocked vs interleaved sur papier

### Corrigé modèle (exemple avec trois patterns algorithmiques : sliding window, two pointers, binary search)

**Plan bloqué :**
| Jour | Contenu |
|------|---------|
| Lundi | Sliding window — 6 problèmes |
| Mardi | Two pointers — 6 problèmes |
| Mercredi | Binary search — 6 problèmes |
| Jeudi | Repos |
| Vendredi | Révision rapide de chaque bloc |

**Plan entrelacé :**
| Jour | Bloc 1 (20 min) | Bloc 2 (20 min) | Bloc 3 (20 min) |
|------|----------------|----------------|----------------|
| Lundi | Sliding window (2 pb) | Binary search (2 pb) | Two pointers (2 pb) |
| Mardi | Binary search (2 pb) | Two pointers (2 pb) | Sliding window (2 pb) |
| Mercredi | Two pointers (2 pb) | Sliding window (2 pb) | Binary search (2 pb) |
| Jeudi | Entrelacé — les 3, dans un ordre nouveau |
| Vendredi | Test blanc : 6 problèmes mélangés, sans indication du type |

**Analyse attendue :**

- *Quelle version semble plus confortable ?* Le plan bloqué — parce qu'on voit les progrès immédiats dans chaque bloc, les erreurs diminuent au fil des répétitions, et on n'a pas à « basculer » de contexte mental.

- *Quelle version est plus efficace sur un test deux semaines plus tard ?* Le plan entrelacé — parce qu'il force la discrimination (quel type de problème est-ce ?) plutôt que l'exécution automatique. Les tests différés montrent 72 % vs 38 % (Rohrer, Dedrick & Stershic, 2015).

- *Que faire quand on se trompe plus en mode entrelacé ?* Ne pas interpréter l'erreur comme un échec — c'est la « difficulté désirable » au travail. Chaque erreur de classification est une occasion d'ancrer la distinction. Se corriger activement, pas juste regarder la solution.

**Point clé :** La facilité pendant l'entraînement est un mauvais indicateur d'apprentissage réel. La difficulté pendant l'entraînement est souvent un bon signal.

---

## Exercice 2 — Concevoir une session d'étude entrelacée

### Corrigé modèle (exemple avec le domaine algorithmie-python du repo)

**Cinq types distincts :**
1. Sliding window (sous-tableaux/sous-chaînes)
2. Two pointers (paires, partitionnement)
3. Binary search (recherche dans un espace ordonné)
4. DFS/BFS sur graphes (exploration)
5. Dynamic programming 1D (sous-problèmes)

**Session de 90 minutes :**

| Heure | Bloc | Activité | Retrieval final |
|-------|------|----------|-----------------|
| 0:00–0:15 | Two pointers | 1 problème medium | Explique le schéma de résolution sans regarder |
| 0:15–0:30 | Dynamic programming | 1 problème easy | Blank-page : redéplie la récurrence |
| 0:30–0:45 | Sliding window | 1 problème medium | 1 flashcard Q/R sur le pattern |
| 0:45–1:00 | Binary search | 1 problème easy | Explique oralement la condition d'arrêt |
| 1:00–1:15 | DFS/BFS | 1 problème easy | Dessine le graphe de parcours de mémoire |
| 1:15–1:30 | Revue entrelacée | 5 questions mélangées (1 par type) — choisir le bon pattern avant de résoudre | Auto-correction |

**Mesure de suivi :**
Le lendemain, avant toute révision, faire 5 problèmes mélangés à livre fermé (pas de notes, pas d'indication de type) et noter le score. Si < 60 %, relancer un cycle entrelacé. Si > 80 %, passer au niveau de difficulté supérieur.

---

## Exercice 3 — Détecter la pseudoscience sur les styles d'apprentissage

### Corrigé modèle

**1. La claim vérifiable :**
Les personnes identifiées comme « apprenants visuels » apprennent mieux (obtiennent de meilleurs résultats mesurables) quand les informations sont présentées sous forme visuelle plutôt que sous forme auditive ou textuelle.

**2. Le type de preuve nécessaire :**
Un protocole randomisé avec **interaction croisée style × méthode** : il faut au moins quatre groupes — (visuel + méthode visuelle), (visuel + méthode auditive), (auditif + méthode visuelle), (auditif + méthode auditive) — et observer que le groupe « visuel/méthode visuelle » surperforme par rapport au groupe « visuel/méthode auditive », ET que le groupe « auditif/méthode auditive » surperforme par rapport au groupe « auditif/méthode visuelle ». Sans ce patron d'interaction croisée, l'hypothèse n'est pas validée.

**3. Ce que dit la littérature :**
Pashler, McDaniel, Rohrer & Bjork (2008, *Psychological Science in the Public Interest*, 9(3), 105–119) ont passé en revue la littérature sur les styles d'apprentissage. Leur verdict : quasiment aucune étude ne teste l'interaction croisée correctement, et celles qui le font **ne la trouvent pas** ou **la contredisent**. Il n'existe pas de base de preuve suffisante pour adapter l'enseignement au « style » de l'apprenant.

Dekker et al. (2012, *Frontiers in Psychology*) ajoutent que ~93–96 % des enseignants croient pourtant au mythe VAK — l'un des neuromythes les plus répandus.

**4. Le conseil alternatif :**
Utilise les techniques à preuve élevée indépendamment de ton « style » :
- **Retrieval practice** : teste-toi plutôt que de relire, quel que soit le format.
- **Espacement** : révise à intervalles croissants.
- **Interleaving** : mélange les types de contenus.
- Adapte la méthode au **contenu**, pas à une préférence subjective : la géographie s'apprend avec des cartes (tout le monde), la prononciation d'une langue s'apprend en écoutant (tout le monde).
