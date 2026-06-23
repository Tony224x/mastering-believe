# Solutions — Module 06 : Attention, charge cognitive & deep work

> Ces solutions sont des **exemples commentés**, pas des réponses uniques. Les exercices 1, 2 et 3 reposent sur ton vécu personnel — l'essentiel est que les concepts soient correctement appliqués à ta situation réelle.

---

## Solution Exercice 1 — Autopsie d'une session d'étude

### Exemple de réponse commentée

**Session reconstituée :**
Durée : 2 h. Matière : statistiques (test du chi-deux). Environnement : chambre, ordinateur portable, Spotify en fond, téléphone à portée de main, deux onglets Twitter ouverts « juste pour vérifier ».

**Interruptions identifiées :**

| # | Interruption | Type | Temps de retour estimé |
|---|---|---|---|
| 1 | Notification Instagram (×4 en 2 h) | Externe | ~15-20 min × 4 |
| 2 | Passage à la messagerie « juste pour voir » | Pseudo-productivité | ~15 min × 3 |
| 3 | Pensée parasite : « il faudrait que je note ça pour mon autre cours » | Interne | ~5-10 min × 2 |

**Calcul d'impact** : si ces interruptions sont cumulées, le temps de reconcentration dépasse la durée totale de la session — ce qui signifie que le régime profond n'a probablement jamais été atteint.

**Sources de charge extrinsèque identifiées :**
- Musique avec paroles (compète pour la mémoire de travail phonologique)
- Termes statistiques (« ddl », « p-value ») non préparés → allers-retours fréquents vers Google
- PDF mal structuré nécessitant de naviguer entre sections non liées

**Charge extrinsèque vs pertinente :**
- La complexité intrinsèque du test du chi-deux (comparer des fréquences observées/attendues) est la charge pertinente à conserver.
- La musique, les notifications et la navigation dans un PDF mal structuré sont de la charge extrinsèque pure — consommée sans apprentissage.

**Trois modifications concrètes pour la prochaine session :**
1. Téléphone en mode avion dans une autre pièce (pas juste en silence).
2. Préparer un lexique des termes statistiques clés *avant* la session (2 min de préparation → charge extrinsèque réduite pendant la session).
3. Musique instrumentale ou silence (pas de paroles).

---

## Solution Exercice 2 — Chunking en action

### Exemple de réponse commentée

**Domaine débutant choisi : allemand (niveau A2)**
**Domaine expert choisi : Python (3 ans de pratique)**

**Partie A — Comparaison**

*Tâche en domaine débutant :* lire et comprendre un paragraphe d'un article allemand simple.
- Éléments distincts en mémoire de travail : déchiffrage de chaque mot (sens + genre + cas), règle de conjugaison, structure de la phrase (verbe en fin de proposition subordonnée), intonation imaginée. Facilement 5-8 éléments distincts → surcharge quasi certaine sur 4 slots.
- Résultat : beaucoup d'effort pour comprendre la surface ; peu de ressources restantes pour saisir le sens général ou mémoriser les idées.

*Tâche en domaine expert :* déboguer une KeyError dans un dict Python.
- Éléments distincts : le message d'erreur (1 chunk = « la clé n'existe pas »), la ligne incriminée (1 chunk reconnu : accès dict sans `.get()`), la correction canonique (1 chunk : `.get(key, default)` ou `if key in d`).
- Chunks transparents : syntaxe Python de base, les types courants d'erreurs dict, les patterns de correction — tout ça est automatique.
- Résultat : la mémoire de travail est quasi libre pour réfléchir à la logique métier, pas à la syntaxe.

**Ce que ça change pour l'attention :** en Python, les ressources cognitives disponibles vont vers la résolution du problème. En allemand, elles sont consommées par le déchiffrage élémentaire — impossible de se concentrer sur la compréhension globale.

**Partie B — Former un chunk en allemand**

Cible : le cas accusatif (changement d'article masculin : *der* → *den*).

1. **Comprendre la structure** (pas mémoriser mécaniquement) : l'accusatif marque l'objet direct ; en allemand, seul l'article masculin change. Comprendre *pourquoi* (marquer le rôle syntaxique) aide à ne pas confondre avec le datif.
2. **Pratiquer en contexte varié** : écrire 5 phrases avec des verbes différents (*sehen*, *kaufen*, *nehmen*…) — pas répéter la même phrase. L'interleaving (Module 04) s'applique ici.
3. **Retrieval sans support** : fermer ses notes, produire 3 phrases avec l'accusatif, vérifier. Répéter à J+2, J+7.

Après quelques semaines de pratique variée, *den/einen* deviennent un chunk automatique — la mémoire de travail n'a plus besoin de les traiter consciemment.

---

## Solution Exercice 3 — Concevoir et mener une session de deep work

### Exemple de plan pré-session et d'analyse post-session commentés

**Objectif formulé :**
*« À la fin de cette session, je saurai expliquer la différence entre charge cognitive intrinsèque, extrinsèque et pertinente avec un exemple personnel pour chacune. »*

— Correct : formulation en résultat précis et testable (je *saurai expliquer*), pas en activité (*je lirai le Module 06*).

**Préparation (avant) :**
- Téléphone mode avion, posé hors du bureau.
- Navigateur : tous les onglets fermés sauf le PDF du module.
- Durée choisie : 50 min (justification : matière conceptuelle familière après avoir lu J1-J5 — les blocs longs conviennent mieux une fois l'habitude installée).
- Carnet ouvert à côté pour noter les pensées parasites.

**Extrait de carnet de capture pendant la session :**
- 14h08 : envie de vérifier un mail → noté, non cédé.
- 14h19 : pensée « faut que je rappelle X » → noté.
- 14h32 : envie de chercher l'article original de Sweller → noté (noté comme tâche post-session).
- 14h47 : distraction sonore (voiture dehors) → récupération rapide, noté.

**Recall à chaud (rédigé sans notes, 4 min) :**
> *« La mémoire de travail tient ~4 chunks. Trois types de charge : intrinsèque (le sujet lui-même), extrinsèque (la présentation, le bruit — à réduire), pertinente (ce qui encode vraiment). Le chunking libère des slots en transformant plusieurs éléments en un. Deep work : objectif clair + distractions éliminées + blocs 25-50 min + recall à chaud + planification révision. »*

— Ce recall est valide : imparfait mais substantiel. Les lacunes identifiées (ex. : incapable de définir la charge pertinente avec un exemple précis) indiquent exactement ce qu'il faudra travailler à la révision suivante.

**Analyse du carnet :**
- Interruptions évitées : 4
- Interruptions cédées : 0
- Source dominante : interne (pensées parasites), pas externe → indique que l'environnement était bien préparé ; l'attention errante est le levier suivant à travailler.

**À améliorer :** formuler l'objectif encore plus précisément avant de commencer — *« savoir donner un exemple personnel pour chaque type de charge »* plutôt que *« comprendre les trois types de charge »*.

**Date de révision planifiée :** J+2 (récupération courte) puis J+7 (révision espacée, Module 03).

---

## Grille d'auto-évaluation transversale

| Critère | Exercice 1 | Exercice 2 | Exercice 3 |
|---|---|---|---|
| Concepts correctement identifiés | Charge extrinsèque vs pertinente | Chunks transparents en domaine expert | Types de charge + structure avant/pendant/après |
| Ancrage dans son vécu personnel | Oui — session réelle reconstituée | Oui — deux domaines personnels précis | Oui — session réelle planifiée et menée |
| Propositions concrètes et actionnables | 3 modifications immédiates | Plan de chunking en 3 étapes | Carnet de capture + recall + date de révision |
| Erreur fréquente à éviter | Confondre charge intrinsèque et extrinsèque | Décrire ce qu'est un chunk sans expliquer pourquoi ça aide | Formuler l'objectif en activité (*« lire »*) plutôt qu'en résultat |
