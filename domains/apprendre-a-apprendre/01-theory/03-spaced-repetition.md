# Module 03 — Spaced repetition & Anki : espacer pour ancrer

> **Temps estimé** : 45 min | **Prérequis** : Modules 01-02
>
> **Objectif** : Comprendre pourquoi espacer les révisions multiplie la rétention, comprendre la courbe d'oubli, implémenter un calendrier d'espacement manuel, et maîtriser le fonctionnement de l'algorithme SM-2 qui pilote Anki.

---

## 1. Le problème concret

Tu as fait tes flashcards du module 02. Tu les as testées hier, tu avais 90 % de bonnes réponses. Quand les revoir ?

La plupart des gens répondent : "Demain, pour consolider." C'est intuitif. C'est faux.

**L'expérience :** Cepeda et al. (2008) ont testé >1350 sujets avec des intervalles de révision variés sur du vocabulaire. Résultat :

| Test final dans… | Intervalle optimal |
|------------------|--------------------|
| 10 jours | ~1 jour |
| 35 jours | ~1 semaine |
| 350 jours (1 an) | ~3-4 semaines |

La règle empirique : **l'intervalle optimal ≈ 10-20 % du délai avant le prochain test.**

Si tu revois demain ce que tu as appris hier, tu revois trop tôt — quand la mémoire est encore fraîche et que la révision coûte peu d'effort. La rétention ne gagne presque rien. Il faut attendre que la mémoire ait *commencé* à s'effacer pour que la révision soit vraiment efficace.

> **À retenir** : Réviser trop tôt est presque aussi inutile qu'oublier. L'effort de récupération — juste au bon moment — est ce qui ancre.

---

## 2. La courbe d'oubli et l'espacement optimal

### 2.1 Ebbinghaus (1885) et la courbe d'oubli

Hermann Ebbinghaus a mesuré sa propre mémoire sur des syllabes sans sens. Sa courbe montre qu'on oublie une grande partie du contenu très rapidement sans révision :

- Après 20 minutes : ~58 % oublié
- Après 1 heure : ~44 % oublié
- Après 1 jour : ~67 % oublié
- Après 1 semaine : ~79 % oublié (sans révision)

Chaque révision *réinitialise* partiellement la courbe et la rend plus douce. Avec 3-4 révisions espacées, on peut maintenir ~90 % de rétention indéfiniment.

### 2.2 Distributed practice vs massed practice

- **Massed practice** (bachotage) : tout apprendre en une seule session. Efficace pour un examen demain. Mémoire à 2 semaines : proche de zéro.
- **Distributed practice** (pratique espacée) : révisions réparties dans le temps. Moins efficace *sur le moment*. Rétention à 1 mois : bien supérieure.

Cepeda et al. (2006) ont synthétisé 839 mesures dans 317 expériences : **la pratique espacée bat systématiquement le bachotage pour la rétention long terme.**

> **À retenir** : Le bachotage fonctionne pour l'examen de vendredi. Il ne fonctionnera pas pour te souvenir de ce contenu en mars.

---

## 3. L'algorithme SM-2 : comment Anki calcule tes intervalles

Anki (et SuperMemo avant lui) opérationnalise l'espacement grâce à un algorithme simple : SM-2 (Wozniak, 1987-1990).

### 3.1 Principe

À chaque carte, tu notes ta qualité de rappel de 0 à 5 :

| Note | Signification |
|------|---------------|
| 0 | Échec total |
| 1 | Échec, mais la réponse semble familière |
| 2 | Échec, mais la bonne réponse était facile à reconnaître |
| 3 | Succès avec effort significatif |
| 4 | Succès avec légère hésitation |
| 5 | Succès parfait, immédiat |

### 3.2 Calcul de l'intervalle

Chaque carte a un **facteur d'aisance (easiness factor, EF)**, initialement à 2.5.

**Mise à jour de l'EF** (formule SM-2 officielle) :
```
EF_new = EF + (0.1 - (5 - note) × (0.08 + (5 - note) × 0.02))
EF_new = max(1.3, EF_new)   # plancher à 1.3
```

**Calcul du prochain intervalle** :
- Si note >= 3 (succès) :
  - 1re révision réussie → intervalle = **1 jour**
  - 2e révision réussie → intervalle = **6 jours**
  - Suivantes → `intervalle_new = round(intervalle_précédent × EF)`
- Si note < 3 (échec) : intervalle remis à **1 jour**, compteur de succès remis à 0

### 3.3 Ce que ça signifie en pratique

Une carte apprise aujourd'hui avec un EF de 2.5 sera revue :

| Révision | Intervalle | Date approximative |
|----------|------------|--------------------|
| 1 | 1 jour | J+1 |
| 2 | 6 jours | J+7 |
| 3 | ~15 jours | J+22 |
| 4 | ~37 jours | J+59 |
| 5 | ~93 jours | J+152 |

Une carte difficile (EF qui baisse vers 1.3) sera revue beaucoup plus souvent. Une carte facile (EF monte) sera revue de moins en moins souvent — ce qui est exactement ce qu'on veut.

> **À retenir** : SM-2 ne fait pas de magie. Il applique mécaniquement la recherche de Cepeda : revoir au bon moment — ni trop tôt (gaspillé), ni trop tard (oublié).

Le script `02-code/03-spaced-repetition.py` de ce domaine implémente SM-2 from scratch en Python stdlib pur. Lance-le pour voir les intervalles calculer en direct.

---

## 4. Mettre en place sa pratique espacée

### 4.1 Avec Anki (recommandé)

1. Télécharger Anki : https://apps.ankiweb.net/ (gratuit, open-source)
2. Créer un deck par domaine d'étude
3. Ajouter des cartes après chaque session d'étude (pas après — *pendant* ou juste après pendant que c'est frais)
4. Faire ses révisions quotidiennes (5-15 min/jour suffit)

**Piège classique :** créer 200 cartes d'un coup et se retrouver avec 200 révisions le lendemain. Solution : limiter les nouvelles cartes à 10-20 par jour.

**Comment formuler une bonne carte :**
- ❌ "Qu'est-ce que SM-2 ?" (trop large, mémoire de liste)
- ✅ "Dans SM-2, que se passe-t-il si la note est < 3 ?" (question précise, rappel ciblé)
- ✅ "Formule EF dans SM-2 ?" → `EF + (0.1 - (5-note)(0.08 + (5-note)×0.02))` (fait tester la formule)

### 4.2 Sans outil numérique

Un calendrier de révision manuel simplifié :

1. Après la première étude : révision J+1
2. Si succès : J+7
3. Si succès : J+30
4. Si succès : J+90
5. Si échec à n'importe quelle étape : retour à J+1

C'est une version simplifiée à 3-4 intervalles. Moins optimal qu'Anki mais infiniment mieux que le bachotage.

### 4.3 Quand faire ses révisions

Le moment optimal : juste *avant* d'oublier. En pratique, en début de session d'étude — quand la mémoire est fraîche du sommeil et qu'on n'est pas encore fatigué.

**Sommeil et mémoire :** la consolidation mémorielle se produit principalement pendant le sommeil (phases REM et ondes lentes). Étudier puis dormir ancre mieux qu'étudier le matin d'un examen tardif. (Voir LHTL, Oakley & Sejnowski.)

---

## Flash-cards

**Q1** : Quelle est la règle chiffrée de Cepeda et al. (2008) pour l'intervalle de révision optimal ?
> **R :** L'intervalle optimal ≈ 10-20 % du délai avant le prochain test. Ex. : test dans 10 jours → intervalle ~1 jour ; test dans 1 an → intervalle ~3-4 semaines.

**Q2** : Qu'est-ce que le "facteur d'aisance" (EF) dans SM-2, et quelles sont ses valeurs typiques ?
> **R :** Coefficient (défaut 2.5, minimum 1.3) qui multiplie l'intervalle précédent pour calculer le prochain. Monte pour les cartes faciles (révision plus espacée), descend pour les cartes difficiles (révision plus fréquente).

**Q3** : Que se passe-t-il dans SM-2 quand une carte reçoit une note < 3 ?
> **R :** L'intervalle est remis à 1 jour (pas 0 — l'espacement minimal d'1 jour reste bénéfique), et le compteur de révisions réussies est remis à 0.

**Q4** : Pourquoi le bachotage (massed practice) est-il inefficace pour la rétention long terme ?
> **R :** Il maximise la performance immédiate mais ne bénéficie pas de l'effet d'espacement. Sans révisions ultérieures, la courbe d'oubli efface rapidement le contenu (Cepeda 2006 : 839 mesures, 317 expériences).

**Q5** : Quels sont les 3 premiers intervalles d'une carte SM-2 réussie à chaque fois (EF = 2.5) ?
> **R :** 1 jour, 6 jours, puis round(6 × 2.5) = 15 jours. Ensuite les intervalles continuent de croître par facteur ~2.5.

---

## Points clés à retenir

- **L'espacement optimal ≈ 10-20 % du délai avant le prochain test** (Cepeda 2008) — pas "le plus souvent possible".
- La courbe d'oubli (Ebbinghaus 1885) : sans révision, on oublie ~80 % en 1 semaine.
- **SM-2** : algorithme fondateur d'Anki — intervalles croissants modulés par la qualité de rappel (EF défaut 2.5, plancher 1.3).
- Bachotage : efficace pour vendredi, oublié en mars.
- **Pratique espacée + retrieval practice = la combinaison la plus puissante** de toutes les techniques d'étude (Dunlosky 2013, utilité "élevée" pour les deux).

---

## Pour aller plus loin

- Cepeda, N. J. et al. (2006). "Distributed Practice in Verbal Recall Tasks: A Review and Quantitative Synthesis". *Psychological Bulletin*, 132(3), 354–380. https://augmentingcognition.com/assets/Cepeda2006.pdf
- Cepeda, N. J. et al. (2008). "Spacing Effects in Learning: A Temporal Ridgeline of Optimal Retention". *Psychological Science*, 19(11), 1095–1102. https://laplab.ucsd.edu/articles/Cepeda%20et%20al%202008_psychsci.pdf
- Wozniak, P. (1990). SuperMemo / SM-2. https://www.supermemo.com/en/supermemo-method
- Dunlosky, J. et al. (2013). "Improving Students' Learning With Effective Learning Techniques". *Psychological Science in the Public Interest*, 14(1), 4–58. https://journals.sagepub.com/doi/10.1177/1529100612453266
- Anki (open-source SRS) : https://apps.ankiweb.net/
- Code : `02-code/03-spaced-repetition.py` — implémentation SM-2 stdlib pur

**Module précédent :** [02 — Retrieval practice](02-retrieval-practice.md) | **Prochain module :** [04 — Difficultés désirables](04-difficultes-desirables.md)
