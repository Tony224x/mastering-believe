# Module 03 — Spaced repetition : espacer pour ancrer

> **Temps estime** : 45 min | **Prerequis** : Modules 01-02
>
> **Objectif** : Comprendre pourquoi espacer les revisions multiplie la retention, implementer un calendrier d'espacement manuel, et comprendre le fonctionnement de l'algorithme SM-2 qui pilote Anki.

---

## 1. Le probleme concret

Tu as fait tes flashcards du module 02. Tu les as testees hier, tu avais 90 % de bonnes reponses. Quand les revoir ?

La plupart des gens repondent : "Demain, pour consolider." C'est intuitif. C'est faux.

**Voici l'experience :** Cepeda et al. (2008) ont teste >1350 sujets avec des intervalles de revision varies sur du vocabulaire. Resultat : pour un test final dans 10 jours, l'intervalle optimal est environ **1 jour**. Pour un test dans 35 jours, c'est environ **1 semaine**. Pour un test dans 350 jours (1 an), c'est environ **3-4 semaines**.

La regle empirique : **l'intervalle optimal ≈ 10-20 % du delai avant le prochain test.**

Si tu revois demain ce que tu as appris hier, tu revois trop tot — quand la memoire est encore fraiche et que la revision coute peu. La retention ne gagne presque rien. Il faut attendre que la memoire ait *commence* a s'effacer pour que la revision soit vraiment efficace.

---

## 2. La courbe d'oubli et l'espacement optimal

### 2.1 Ebbinghaus (1885) et la courbe d'oubli

Hermann Ebbinghaus a mesure sa propre memoire sur des syllabes sans sens. Sa courbe montre :

- Apres 20 minutes : ~58 % oublie
- Apres 1 heure : ~44 % oublie
- Apres 1 jour : ~33 % retenu
- Apres 1 semaine : ~21 % retenu (sans revision)

Chaque revision *resette* partiellement la courbe et la rend plus douce. Avec 3-4 revisions espacees, on peut maintenir ~90 % de retention indefiniment.

### 2.2 Distributed practice vs massed practice

- **Massed practice** (bachotage) : tout apprendre en une seule session. Efficace pour un examen demain. Memoire de 2 semaines : proche de zero.
- **Distributed practice** (pratique espacee) : revisions reparties dans le temps. Moins efficace *sur le moment*. Retention a 1 mois : bien superieure.

Cepeda et al. (2006) ont synthetise 839 mesures dans 317 experiences : **la pratique espacee bat systematiquement le bachotage pour la retention long terme.**

> **A retenir :** Le bachotage fonctionne pour l'examen de vendredi. Il ne fonctionnera pas pour te souvenir de ce contenu en mars.

---

## 3. L'algorithme SM-2 : comment Anki calcule tes intervalles

Anki (et SuperMemo avant lui) operationnalise l'espacement grace a un algorithme simple : SM-2 (Wozniak, 1987-1990).

### 3.1 Principe

A chaque carte, tu notes ta qualite de rappel de 0 a 5 :

| Note | Signification |
|------|---------------|
| 0 | Echec total |
| 1 | Echec, mais la reponse semble familiere |
| 2 | Echec, mais la bonne reponse etait facile a reconnaitre |
| 3 | Succes avec effort significatif |
| 4 | Succes avec legere hesitation |
| 5 | Succes parfait, immediat |

### 3.2 Calcul de l'intervalle

Chaque carte a un **facteur d'aisance (easiness factor, EF)**, initialement a 2.5.

- Si la note >= 3 (succes) : `prochain_intervalle = intervalle_precedent × EF`
- Si la note < 3 (echec) : on recommence a l'intervalle 1 jour
- EF se met a jour a chaque revision : `EF = EF + (0.1 - (5 - note) × (0.08 + (5 - note) × 0.02))`
- EF ne descend jamais sous 1.3

Premiers intervalles fixes :
- Premiere revision : 1 jour
- Deuxieme revision (si succes) : 6 jours
- Ensuite : intervalle × EF (≈ 2.5 par defaut → intervalles 1, 6, 15, 37 jours...)

### 3.3 Ce que ca signifie en pratique

Une carte apprise aujourd'hui avec un EF de 2.5 sera revue :
- J+1
- J+6
- J+15
- J+37
- J+93
- ...

Une carte difficile (EF qui baisse vers 1.3) sera revue beaucoup plus souvent. Une carte facile (EF monte) sera revue de moins en moins souvent — ce qui est exactement ce qu'on veut.

> **A retenir :** SM-2 ne fait pas de magie. Il applique mecaniquement la recherche de Cepeda : revoir au bon moment — ni trop tot (gaspille), ni trop tard (oublie).

Le script `02-code/03-spaced-repetition.py` de ce domaine implemente SM-2 from scratch en Python stdlib pur. Lance-le pour voir les intervalles calculer en direct.

---

## 4. Mettre en place sa pratique espacee

### 4.1 Avec Anki (recommande)

1. Telecharger Anki : https://apps.ankiweb.net/ (gratuit, open-source)
2. Creer un deck par domaine d'etude
3. Ajouter des cartes apres chaque session d'etude (pas apres — *pendant* ou juste apres pendant que c'est frais)
4. Faire ses revisions quotidiennes (5-15 min/jour suffit)

**Piege classique :** creer 200 cartes d'un coup et se retrouver avec 200 revisions le lendemain. Solution : limiter les nouvelles cartes a 10-20 par jour.

### 4.2 Sans outil numerique

Un calendrier de revision manuel :

1. Apres la premiere etude : revision J+1
2. Si succes : J+7
3. Si succes : J+30
4. Si succes : J+90
5. Si echec a n'importe quelle etape : retour a J+1

C'est une version simplifiee a 3-4 intervalles. Moins optimal qu'Anki mais infiniment mieux que le bachotage.

### 4.3 Quand faire ses revisions

Le moment optimal : juste *avant* d'oublier. En pratique, en debut de session d'etude — quand la memoire est fraiche du sommeil et qu'on n'est pas encore fatigue.

**Sommeil et memoire :** la consolidation memorielle se produit principalement pendant le sommeil (phases REM et ondes lentes). Etudier puis dormir ancre mieux qu'etudier le matin d'un examen tardif. (Voir LHTL, Oakley & Sejnowski.)

---

## Flash-cards

**Q1** : Quelle est la "regle chiffree" de Cepeda et al. (2008) pour l'intervalle de revision optimal ?
> **R :** L'intervalle optimal est approximativement 10-20 % du delai avant le prochain test (ex. si le test est dans 10 jours, l'intervalle est d'environ 1 jour ; dans 1 an, environ 3-4 semaines).

**Q2** : Qu'est-ce que le "facteur d'aisance" (easiness factor, EF) dans l'algorithme SM-2 ?
> **R :** Un coefficient (defaut 2.5, minimum 1.3) qui multiplie l'intervalle precedent pour calculer le prochain. Il monte pour les cartes faciles (revision plus espacee) et descend pour les cartes difficiles (revision plus frequente).

**Q3** : Que se passe-t-il dans SM-2 quand une carte recoit une note < 3 ?
> **R :** L'intervalle est remis a 1 jour (echec = recommencer le cycle), et l'EF diminue.

**Q4** : Pourquoi le bachotage (massed practice) est-il inefficace pour la retention long terme ?
> **R :** Il maximise la performance immediate mais ne benifice pas de l'effet d'espacement. Sans revisions ulterieures, la courbe d'oubli efface rapidement le contenu.

**Q5** : Quelle est la difference entre distributed practice et massed practice ?
> **R :** La pratique distribuee (espacee) repartit les revisions dans le temps, produisant une bien meilleure retention long terme. La pratique massee concentre l'etude en une session (bachotage), efficace a court terme, tres inefficace a long terme.

---

## Points cles a retenir

- **L'espacement optimal ≈ 10-20 % du delai avant le prochain test** (Cepeda 2008) — pas "le plus souvent possible".
- La courbe d'oubli (Ebbinghaus) montre qu'on oublie ~80 % en 1 semaine sans revision.
- **SM-2** : algorithme fondateur d'Anki — intervalles croissants modulees par la qualite de rappel.
- Bachotage : efficace pour vendredi, oublie en mars.
- **Pratique espacee + retrieval practice = la combinaison la plus puissante** de toutes les techniques d'etude (Dunlosky 2013).

---

## Pour aller plus loin

- Cepeda, N. J. et al. (2006). "Distributed Practice in Verbal Recall Tasks". *Psychological Bulletin*, 132(3), 354–380. https://augmentingcognition.com/assets/Cepeda2006.pdf
- Cepeda, N. J. et al. (2008). "Spacing Effects in Learning". *Psychological Science*, 19(11), 1095–1102. https://laplab.ucsd.edu/articles/Cepeda%20et%20al%202008_psychsci.pdf
- Wozniak, P. (1990). SuperMemo / SM-2. https://www.supermemo.com/en/supermemo-method
- Anki (open-source SRS) : https://apps.ankiweb.net/
- Code : `02-code/03-spaced-repetition.py` — implemention SM-2 stdlib pur

**Module precedent :** [02 — Retrieval practice](02-retrieval-practice.md) | **Prochain module :** [04 — Difficultes desirables](04-difficultes-desirables.md)
