# Solutions — Module 04 : Difficultes desirables

---

## Exercice 1 — Comparer blocked vs interleaved sur papier

### Corrige modele (exemple avec trois patterns algorithmiques : sliding window, two pointers, binary search)

**Plan bloque :**
| Jour | Contenu |
|------|---------|
| Lundi | Sliding window — 6 problemes |
| Mardi | Two pointers — 6 problemes |
| Mercredi | Binary search — 6 problemes |
| Jeudi | Repos |
| Vendredi | Revision rapide de chaque bloc |

**Plan interleave :**
| Jour | Bloc 1 (20 min) | Bloc 2 (20 min) | Bloc 3 (20 min) |
|------|----------------|----------------|----------------|
| Lundi | Sliding window (2 pb) | Binary search (2 pb) | Two pointers (2 pb) |
| Mardi | Binary search (2 pb) | Two pointers (2 pb) | Sliding window (2 pb) |
| Mercredi | Two pointers (2 pb) | Sliding window (2 pb) | Binary search (2 pb) |
| Jeudi | Entrelace les 3, dans un ordre nouveau |
| Vendredi | Test blanc : 6 problemes melanges, sans indication du type |

**Analyse attendue :**

- *Quelle version semble plus confortable ?* Le plan bloque — parce qu'on voit les progres imediats dans chaque bloc, les erreurs diminuent au fil des repetitions, et on n'a pas a "basculer" de contexte mental.

- *Quelle version est plus efficace sur un test deux semaines plus tard ?* Le plan interleave — parce qu'il force la discrimination (quel type de probleme est-ce ?) plutot que l'execution automatique. Les tests differees montrent 72 % vs 38 % (Rohrer et al. 2015).

- *Que faire quand on se trompe plus en mode interleave ?* Ne pas interpreter l'erreur comme un echec — c'est la "difficulte desirable" au travail. Chaque erreur de classification est une occasion d'ancrer la distinction. Se corriger activement, pas juste regarder la solution.

**Point cle :** La facilite pendant l'entrainement est un mauvais indicateur d'apprentissage reel. La difficulte pendant l'entrainement est souvent un bon signal.

---

## Exercice 2 — Concevoir une session d'etude entrelacee

### Corrige modele (exemple avec le domaine algorithmie-python du repo)

**Cinq types distincts :**
1. Sliding window (sous-tableaux/sous-chaines)
2. Two pointers (paires, partitionnement)
3. Binary search (recherche dans un espace ordonne)
4. DFS/BFS sur graphes (exploration)
5. Dynamic programming 1D (sous-problemes)

**Session de 90 minutes :**

| Heure | Bloc | Activite | Retrieval final |
|-------|------|----------|-----------------|
| 0:00–0:15 | Two pointers | 1 probleme medium | Explique le schema de resolution sans regarder |
| 0:15–0:30 | Dynamic programming | 1 probleme easy | Blank-page : redeplie la recurrence |
| 0:30–0:45 | Sliding window | 1 probleme medium | 1 flashcard Q/R sur le pattern |
| 0:45–1:00 | Binary search | 1 probleme easy | Explique oralement la condition d'arret |
| 1:00–1:15 | DFS/BFS | 1 probleme easy | Dessine le graphe de parcours de memoire |
| 1:15–1:30 | Revue entrelacee | 5 questions melanges (1 par type), a choisir le bon pattern avant de resoudre | Auto-correction |

**Mesure de suivi :**
Le lendemain, avant toute revision, faire 5 problemes melanges a livre ferme (pas de notes, pas d'indication de type) et noter le score. Si < 60 %, relancer un cycle interleave. Si > 80 %, passer au niveau de difficulte superieur.

---

## Exercice 3 — Detecter la pseudoscience sur les styles d'apprentissage

### Corrige modele

**1. La claim verifiable :**
Les personnes identifiees comme "apprenants visuels" apprennent mieux (obtiennent de meilleurs resultats mesurables) quand les informations sont presentees sous forme visuelle plutot que sous forme auditive ou textuelle.

**2. Le type de preuve necessaire :**
Un protocole randomise avec **interaction croisee style x methode** : il faut au moins quatre groupes — (visuel + methode visuelle), (visuel + methode auditive), (auditif + methode visuelle), (auditif + methode auditive) — et observer que le groupe "visuel/methode visuelle" surperforme par rapport au groupe "visuel/methode auditive", ET que le groupe "auditif/methode auditive" surperforme par rapport au groupe "auditif/methode visuelle". Sans ce patron d'interaction croisee, l'hypothese n'est pas validee.

**3. Ce que dit la litterature :**
Pashler, McDaniel, Rohrer & Bjork (2008, *Psychological Science in the Public Interest*, 9(3), 105-119) ont passe en revue la litterature sur les styles d'apprentissage. Leur verdict : quasiment aucune etude ne teste l'interaction croisee correctement, et celles qui le font **ne la trouvent pas** ou **la contredisent**. Il n'existe pas de base de preuve suffisante pour adapter l'enseignement au "style" de l'apprenant.

Dekker et al. (2012, *Frontiers in Psychology*) ajoutent que ~93-96 % des enseignants croient pourtant au mythe VAK — l'un des neuromythes les plus repandus.

**4. Le conseil alternatif :**
Utilise les techniques a preuve elevee independamment de ton "style" :
- **Retrieval practice** : teste-toi plutot que de relire, quel que soit le format.
- **Espacement** : revise a intervalles croissants.
- **Interleaving** : melange les types de contenus.
- Adapte la methode au **contenu**, pas a une preference subjective : la geographie s'apprend avec des cartes (tout le monde), la prononciation d'une langue s'apprend en ecoutant (tout le monde).
