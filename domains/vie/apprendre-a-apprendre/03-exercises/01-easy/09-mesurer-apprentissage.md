# Exercices — Module 09 : Mesurer son apprentissage

---

## Exercice 1 — Calculer et interpréter un delta pré/post

### Objectif
Appliquer la métrique delta pré/post sur un vrai contenu, distinguer le gain immédiat de la rétention à long terme, et identifier la limite de cet indicateur seul.

### Consigne

**Contexte** : tu viens de terminer le Module 03 (Spaced Repetition). Voici les résultats de deux apprenants après un test pré/post à 10 questions sur les concepts de l'algorithme SM-2 :

| Apprenant | Score pré | Score post | Score J+7 |
|-----------|-----------|------------|-----------|
| Alice     | 2/10      | 9/10       | 8/10      |
| Bob       | 2/10      | 9/10       | 3/10      |

**Étape 1 — Calcul :**
Pour chaque apprenant, calcule :
- Le delta pré/post en nombre de questions et en points de pourcentage (pp).
- Le taux de rétention à J+7 (score J+7 ÷ total × 100).
- La chute de rétention entre post-test et J+7 (post_pct − J7_pct).

**Étape 2 — Interprétation :**
Alice et Bob ont le même delta pré/post. Pourtant, leurs profils d'apprentissage sont très différents.
- Que révèle le score J+7 sur la qualité de l'apprentissage de chacun ?
- Que peut-on inférer sur la stratégie d'étude de Bob entre le post-test et J+7 ?
- Quel est l'enseignement principal sur la limite du delta pré/post comme seule métrique ?

**Étape 3 — Conseil :**
Propose une recommandation concrète à Bob (une action, avec une date) pour améliorer sa rétention à J+7 lors du prochain module.

### Critères de réussite
- [ ] Les deltas pré/post d'Alice et Bob sont calculés correctement (chiffres et pp)
- [ ] Les taux de rétention J+7 sont calculés correctement pour les deux
- [ ] L'interprétation mentionne explicitement que le delta pré/post ne prédit pas la rétention à long terme
- [ ] La recommandation à Bob cite au moins une technique du domaine (révision espacée, retrieval practice) avec une date ou un délai précis

---

## Exercice 2 — Construire et calibrer son journal de métriques

### Objectif
Mettre en place un journal de suivi personnel avec pré-test, post-test et mesure de calibration, puis analyser l'écart entre prédit et réel.

### Consigne

**Étape 1 — Ton pré-test (avant de rouvrir le module) :**
Sans relire le Module 09, réponds par écrit aux cinq questions suivantes (note tes réponses) :

1. Quelle est la différence entre un taux de rappel et un score de fluency ?
2. Que mesure précisément le delta pré/post ?
3. Cite les trois éléments d'un bon feedback formatif (selon Black & Wiliam 1998).
4. Comment calcule-t-on un score de calibration simple ?
5. Qu'est-ce que la Goodhart's Law appliquée aux métriques d'apprentissage ?

**Avant de vérifier tes réponses**, note combien tu penses en avoir eu correct (sur 5) : \_\_\_/5.

**Étape 2 — Vérification et post-test :**
Consulte le Module 09 et corrige chacune de tes réponses. Compte le nombre de réponses correctes.

**Étape 3 — Remplir ton journal :**
Complète ce tableau pour cette session :

```
Contenu      : Module 09 — Mesurer son apprentissage
Date         : _______________
Pré-test     : ___/5   (___  %)
Score prédit : ___/5   (___  %)
Post-test    : ___/5   (___  %)
Delta        : +___ questions  (+___ pp)
Calibration  : |prédit_pct − pré_pct| = ___ pp
```

**Étape 4 — Planification J+7 :**
Planifie dans ton agenda un test de rappel dans 7 jours sur le même contenu. Note dès maintenant ta prédiction pour ce rappel à J+7 : \_\_\_ %.

### Critères de réussite
- [ ] Le pré-test est complété sans avoir relu le module au préalable
- [ ] La prédiction est notée AVANT la vérification (pas après — c'est la règle de la calibration)
- [ ] Le tableau de journal est rempli avec tous les champs
- [ ] La calibration est calculée correctement (|prédit − réel| en pp)
- [ ] Un rappel J+7 est planifié avec une date et une prédiction de taux

---

## Exercice 3 — Analyser une courbe d'oubli et concevoir un plan de révision

### Objectif
Lire des données de rétention mesurées dans le temps, identifier le point de chute critique, comparer avec la courbe théorique d'Ebbinghaus, et concevoir un plan de révision espacée fondé sur les métriques.

### Consigne

**Contexte** : un apprenant a mesuré son taux de rappel sur le Module 02 (Retrieval Practice) à plusieurs intervalles après l'apprentissage initial, **sans aucune révision intermédiaire** :

| Jour | Taux de rappel mesuré |
|------|-----------------------|
| J+0  | 85 %                  |
| J+1  | 72 %                  |
| J+3  | 58 %                  |
| J+7  | 41 %                  |
| J+14 | 30 %                  |
| J+30 | 21 %                  |

**Étape 1 — Analyse :**
- Entre quels deux points de mesure se produit la chute la plus forte (en pp absolu) ?
- Le taux de rappel à J+7 (41 %) dépasse-t-il le seuil de 61 % mesuré par Roediger & Karpicke (2006) pour le groupe "test" ? Que suggère cet écart ?
- Ébbinghaus prédit une rétention théorique d'environ 58 % à J+1 et 33 % à J+7 (sans révision). Cet apprenant est-il au-dessus ou en dessous de la courbe théorique ? Que peut-on inférer ?

**Étape 2 — Conception du plan de révision :**
L'apprenant veut maintenir une rétention d'au moins 70 % jusqu'à J+30. En te basant sur les données et sur les principes de l'espacement (Module 03), propose un plan de révision qui spécifie :
- Les dates ou intervalles des révisions (J+1, J+3, J+7…).
- Le nombre de révisions estimé.
- La métrique que l'apprenant devra mesurer après chaque révision pour valider que la rétention est bien au-dessus de 70 %.

**Étape 3 — Limite :**
Ce plan est fondé sur des données agrégées. Cite un facteur personnel qui pourrait faire que tes propres intervalles optimaux soient différents de ceux recommandés par la courbe théorique.

### Critères de réussite
- [ ] La chute la plus forte est correctement identifiée (avec les valeurs en pp)
- [ ] La comparaison avec Roediger & Karpicke 2006 (61 %) est faite explicitement
- [ ] Le plan de révision comporte au moins 3 jalons avec des intervalles précis
- [ ] La métrique à mesurer après chaque révision est spécifiée (pas juste "vérifier" — préciser comment)
- [ ] Au moins un facteur personnel de variabilité est cité (contenu, sleep, niveau initial, etc.)
