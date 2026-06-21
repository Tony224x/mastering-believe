# Module 09 — Mesurer son apprentissage

> **Temps estimé** : 45 min | **Prérequis** : Modules 01-08
> **Objectif** : savoir si tu apprends vraiment — construire des métriques objectives (delta pré/post, taux de rappel, rétention à J+7) plutôt que te fier à l'impression subjective ; poser les bases du tableau de bord que le capstone J14 va exiger.

---

## 1. Le problème : tu ne sais pas si tu as appris

Tu as passé deux heures sur un chapitre. Tu te sens à l'aise. Tu peux relire et suivre le raisonnement. Est-ce que tu as appris ?

Pas nécessairement. Ce que tu ressens, c'est la **fluency illusion** (Module 01) — la facilité de reconnaissance au moment de la lecture, pas la capacité à restituer sans aide 48 heures plus tard. Les études montrent régulièrement que les apprenants surestiment leur rétention et sous-estiment à quel point ils ont oublié (Bjork, Dunlosky & Kornell, 2013).

La solution : **remplacer le sentiment subjectif par des indicateurs mesurables**. C'est ce que ce module t'apprend à faire.

---

## 2. Trois métriques fondamentales

### 2.1 Le delta pré/post

**Principe** : avant de commencer une session d'étude, tu te testes sur le contenu que tu vas apprendre. Tu notes ton score (ou ta qualité de rappel). Après la session, tu te retestes avec les mêmes questions. La différence = le delta pré/post.

**Pourquoi c'est utile :**
- Il rend l'apprentissage **visible** — tu vois concrètement ce que la session a apporté.
- Il calibre tes prédictions (Module 08) — tu compares ce que tu pensais savoir avant avec ce que tu savais réellement.
- Il te force à faire du retrieval practice (Module 02) dès le début, ce qui renforce l'apprentissage.

**Format minimal** :

```
Avant : score pré-test  = 3/10
Après : score post-test = 8/10
Delta = +5 points (+50 pp)
```

**Nuance importante** : un delta élevé juste après la session ne garantit pas la rétention à long terme. Il mesure l'apprentissage immédiat, pas la mémoire consolidée. C'est pourquoi il faut coupler cette métrique aux deux suivantes.

### 2.2 Le taux de rappel

**Principe** : lors d'une session de révision (J+1, J+7, J+30…), tu te retestes sur le même contenu et tu calcules le **pourcentage de réponses correctes sans aide**.

```
Taux de rappel J+7 = (réponses correctes ÷ total de questions) × 100
```

Ce taux est l'indicateur central d'Anki et de tout système SM-2 : il mesure directement ce que tu retiens effectivement dans le temps, pas ce que tu reconnais en lisant.

**Référence** : Roediger & Karpicke (2006) ont mesuré un taux de rappel de 61 % après une semaine pour le groupe "test" vs 40 % pour le groupe "relecture". L'écart visible sur cette métrique est ce qui rend l'effet test démontrable — pas seulement croyable.

### 2.3 La rétention mesurée dans le temps (courbe d'oubli réelle)

**Principe** : tu mesures ton taux de rappel à plusieurs intervalles après l'apprentissage initial — par exemple J+1, J+3, J+7, J+14 — et tu traces ta courbe d'oubli personnelle.

La courbe d'Ebbinghaus (1885) est une courbe théorique moyenne. Ta courbe réelle sera différente selon le contenu, ton niveau initial, ton sleep, tes révisions intermédiaires. Mesurer ta propre courbe te donne des données personnalisées sur :
- **Ton point de chute critique** : à quel intervalle tu perds la majorité de la rétention.
- **L'effet de tes révisions** : est-ce que chaque révision remonte vraiment la courbe ?

---

## 3. Le feedback formatif

Le feedback formatif est un concept issu de la recherche en éducation (Black & Wiliam, 1998) : c'est un retour **continu et intégré à l'apprentissage**, par opposition au feedback sommatif (la note finale de l'examen).

**Caractéristiques d'un bon feedback formatif :**
1. **Rapide** : idéalement dans la même session, ou le jour suivant.
2. **Spécifique** : il indique précisément ce qui est faux et pourquoi, pas seulement "incorrect".
3. **Actionnable** : il pointe vers une action de correction concrète.
4. **Fréquent** : pas une fois par mois — à chaque session.

**Dans la pratique** :
- Quand tu fais une flashcard Anki et que tu te trompes → feedback immédiat sur ce que tu n'as pas retenu.
- Quand tu fais un exercice de code et que les tests échouent → feedback sur la logique précise.
- Quand tu expliques un concept à voix haute (Feynman, Module 08) et que tu butes → feedback sur les lacunes.

Le feedback formatif transforme l'erreur en donnée. Au lieu de subir l'échec, tu le quantifies : "J'ai eu 4 erreurs sur le module spaced repetition, toutes sur les formules SM-2" → tu sais exactement quoi travailler.

---

## 4. La calibration comme métrique de métacognition

En Module 08, tu as vu que la calibration mesure l'écart entre ce que tu penses savoir et ce que tu sais réellement. Ici, on la rend opérationnelle comme métrique :

**Méthode** :
Avant chaque test ou exercice, note ta prédiction : "Je pense que je vais avoir X/10." Après, note ton résultat réel. La calibration = |prédit - réel|.

- **Calibration parfaite** : |prédit - réel| = 0. Tu sais exactement ce que tu sais.
- **Surestimation** : prédit > réel. Tu penses savoir, mais tu as oublié ou tu te confonds. C'est l'erreur classique (fluency illusion).
- **Sous-estimation** : prédit < réel. Tu doutes de toi-même plus que nécessaire. Souvent chez les apprenants anxieux.

**Score de Brier simplifié** : pour une prédiction de probabilité (0 à 1) sur une réponse oui/non :
```
Brier = (p_prédit - p_réel)²   (proche de 0 = bon calibrateur)
```

L'objectif n'est pas d'avoir un score parfait immédiatement — c'est de voir la tendance s'améliorer dans le temps.

---

## 5. Définir tes métriques personnelles

Un bon système de métriques d'apprentissage répond à ces quatre questions :

| Question | Métrique |
|----------|----------|
| Est-ce que j'ai appris quelque chose aujourd'hui ? | Delta pré/post de la session |
| Est-ce que je retiens à court terme ? | Taux de rappel J+1 |
| Est-ce que je retiens à moyen terme ? | Taux de rappel J+7 |
| Est-ce que je me connais bien ? | Écart calibration (prédit vs réel) |

**Règle pratique** : commence simple. Un tableau de 4 colonnes dans un fichier texte suffit. Le danger, c'est de passer plus de temps à mesurer qu'à apprendre.

```
Date | Contenu | Score pré | Score post | Rappel J+7 | Prédit J+7 | Calibration
```

---

> **Pseudoscience ?**
>
> **"Plus tu mesures, mieux tu apprends"** — ce n'est pas automatiquement vrai.
>
> Il existe un risque de **Goodhart's Law** appliqué à l'apprentissage : quand une mesure devient un objectif, elle cesse d'être une bonne mesure. Si tu optimises ton score de taux de rappel en révisant uniquement les questions faciles, tu as un bon score et une compréhension superficielle.
>
> La mesure est un outil de feedback, pas une fin. Les métriques ci-dessus fonctionnent **si** elles restent couplées à des pratiques d'apprentissage robustes (retrieval, espacement, interleaving). Une métrique seule ne produit pas d'apprentissage.

---

## 6. Lien avec le capstone J14

Le capstone (Module 14) demande de construire un système d'apprentissage complet sur un sujet réel et d'y intégrer des **métriques de suivi**. Ce module est le prérequis direct : tu ne peux pas concevoir les métriques du capstone sans comprendre delta pré/post, taux de rappel et calibration.

Concrètement, le capstone exigera :
- Un test pré/post sur le sujet choisi.
- Un plan de révisions espacées avec des jalons de rappel mesurables.
- Un journal de calibration (prédit vs réel) sur au moins deux sessions.

---

> **À retenir :**
> - **Delta pré/post** = ce que la session a apporté. Ne suffit pas seul.
> - **Taux de rappel** = ce que tu retiens sans aide à J+1, J+7, J+30.
> - **Courbe d'oubli personnelle** = tes propres données, pas la moyenne d'Ebbinghaus.
> - **Feedback formatif** = retour rapide, spécifique, actionnable — transforme l'erreur en donnée.
> - **Calibration** = écart prédit/réel. Un bon apprenant se connaît avec précision.
> - Mesurer n'apprend pas. Mesurer guide l'effort vers ce qui est réellement oublié.

---

## Flash-cards

**Q1.** Qu'est-ce que le delta pré/post mesure, et pourquoi ne suffit-il pas à garantir la rétention à long terme ?
**R.** Il mesure l'apprentissage immédiat lors d'une session. Il ne suffit pas car l'oubli survient après la session — un bon score post-test peut chuter à J+7 sans révision espacée.

**Q2.** Quelle est la différence entre un taux de rappel et un score de fluency ?
**R.** Le taux de rappel = restituer sans aide (test objectif). La fluency = facilité à suivre un contenu qu'on relit (subjectif, trompeuse).

**Q3.** Qu'est-ce qu'un bon feedback formatif a de plus qu'un feedback sommatif ?
**R.** Il est continu, rapide, spécifique et actionnable — intégré à l'apprentissage. Le feedback sommatif est une note finale : trop tardif pour corriger le tir.

**Q4.** Comment calcule-t-on un score de calibration simple ?
**R.** |score prédit - score réel|. Plus c'est proche de 0, meilleure est la calibration. La surestimation (prédit > réel) est le biais le plus courant.

**Q5.** Que risque-t-on si on optimise ses métriques plutôt que son apprentissage ?
**R.** La loi de Goodhart : la métrique cesse d'être un bon indicateur dès qu'elle devient un objectif. On peut avoir un score parfait avec un apprentissage superficiel.

---

## Points clés à retenir

1. Le sentiment subjectif "je comprends" est un indicateur peu fiable — remplace-le par des métriques objectives (tests, taux de rappel, calibration).
2. Le delta pré/post rend l'apprentissage visible, mais mesure l'acquisition immédiate, pas la rétention consolidée.
3. Le taux de rappel à J+7 est l'indicateur central de ce qui est vraiment ancré en mémoire à moyen terme.
4. La calibration mesure ta connaissance de toi-même en tant qu'apprenant — un prérequis pour ajuster ta stratégie efficacement.
5. Moins de métriques bien suivies vaut mieux que beaucoup de métriques mal utilisées.

---

## Pour aller plus loin

- **Black, P., & Wiliam, D. (1998).** "Inside the Black Box: Raising Standards Through Classroom Assessment." *Phi Delta Kappan*, 80(2), 139-148. — L'étude fondatrice sur le feedback formatif.
- **Koriat, A. (1997).** "Monitoring one's own knowledge during study: A cue-utilization approach to judgments of learning." *Journal of Experimental Psychology: General*, 126(4), 349–370. — Pourquoi nos jugements d'apprentissage sont biaisés.
- **Dunlosky, J., & Metcalfe, J. (2009).** *Metacognition*. SAGE Publications. — Vue complète sur le monitoring et le contrôle de l'apprentissage.
- **Roediger, H. L., & Karpicke, J. D. (2006).** *Psychological Science*, 17(3), 249–255. — La mesure du taux de rappel comme preuve de l'effet test.
- **Brier, G. W. (1950).** "Verification of Forecasts Expressed in Terms of Probability." *Monthly Weather Review*, 78(1), 1–3. — Référence fondatrice du score de Brier utilisé pour la calibration (§4).
