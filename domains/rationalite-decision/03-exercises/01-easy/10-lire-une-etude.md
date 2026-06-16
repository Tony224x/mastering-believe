# Exercices — Module 10 : Lire une étude & stats trompeuses

> **Niveau** : easy → medium → hard (progression dans les 3 exercices)
> **Prérequis** : avoir étudié `01-theory/10-lire-une-etude.md`
> **Format** : exercices de réflexion et d'analyse — pas de code requis.

---

## Exercice 1 — Repérer la p-value insuffisante (easy)

### Objectif
Distinguer significativité statistique et importance pratique, et savoir lire une taille d'effet.

### Consigne

Un logisticien évalue deux logiciels de planification d'itinéraires (A et B) sur la durée moyenne des trajets (en minutes). Il teste chacun sur un grand groupe de tournées et obtient les résultats suivants :

| Comparaison | Durée moyenne A | Durée moyenne B | p-value | d de Cohen |
|-------------|-----------------|-----------------|---------|------------|
| Étude 1     | 58,2 min        | 57,9 min        | 0,003   | 0,04       |
| Étude 2     | 62,0 min        | 55,0 min        | 0,11    | 0,72       |

**Questions :**

1. L'étude 1 est-elle statistiquement significative ? Quelle est la taille d'effet ? Que conseillez-vous au logisticien ?
2. L'étude 2 n'est pas significative. Faut-il en conclure que les deux logiciels sont équivalents ? Pourquoi ?
3. Quelle information manque probablement dans l'étude 2 pour conclure ?

### Critères de réussite
- [ ] L'étude 1 est identifiée comme significative mais avec un effet négligeable (d = 0,04 << 0,20).
- [ ] La recommandation tient compte de l'importance pratique, pas seulement de la p-value.
- [ ] L'étude 2 est interprétée comme non concluante (pas de preuve d'équivalence), pas comme « pas d'effet ».
- [ ] La taille d'échantillon insuffisante est mentionnée comme explication possible du p non significatif en étude 2.

---

## Exercice 2 — Identifier les pièges dans un rapport (medium)

### Objectif
Appliquer la grille de lecture rapide (graphiques tronqués, % vs points de %, dénominateur manquant, moyenne vs médiane) à un cas concret.

### Consigne

Le responsable qualité d'un centre de tri colis rédige le rapport mensuel suivant. Lisez-le attentivement et identifiez **au moins 3 pièges statistiques**, en nommant chaque piège et en expliquant pourquoi il induit en erreur.

---

*Extrait du rapport du mois de mai :*

> « Notre taux de livraisons dans les délais a progressé de **25 % en un mois**, passant de 92 % en avril à 93 % en mai — une performance record. »
>
> « Le nombre de réclamations clients a atteint **47 incidents** ce mois-ci, un chiffre qui doit alerter. »
>
> « Le graphique ci-dessous montre l'évolution de nos délais moyens de traitement (axe Y de 18 min à 22 min) : la courbe en forte hausse confirme une dégradation majeure. »
>
> *[Graphique : axe Y de 18 à 22 min ; la courbe passe de 19 min en semaine 1 à 21 min en semaine 4]*
>
> « Le temps de traitement moyen est de 21 min. Il est donc urgent d'agir pour protéger les opérateurs les plus lents. »
>
> *[Note : le détail des temps montre 4, 5, 6, 7, 8, 9, 10, 90 min pour 8 opérateurs.]*

---

**Pour chaque piège identifié, répondre :**
- Quel est le nom du piège ?
- Quelle est l'information réelle derrière la formulation trompeuse ?
- Comment reformuler correctement ?

### Critères de réussite
- [ ] Piège 1 (% relatifs vs points de %) identifié : +25 % relatif = +1 point de %, l'amélioration absolue est minime.
- [ ] Piège 2 (dénominateur manquant) identifié : 47 incidents sur combien de livraisons ? Sans le dénominateur, on ne peut pas évaluer la gravité.
- [ ] Piège 3 (graphique tronqué) identifié : l'axe Y commence à 18 min, faisant paraître +2 min comme une chute dramatique ; en réalité l'augmentation est de ~10 %.
- [ ] Piège 4 optionnel (moyenne vs médiane) identifié : la moyenne de 21 min est tirée par la valeur aberrante à 90 min ; la médiane est ~7,5 min, bien plus représentative de l'expérience typique.
- [ ] Chaque piège est correctement reformulé avec les chiffres exacts.

---

## Exercice 3 — Analyser un cas de p-hacking (hard)

### Objectif
Raisonner sur les researcher degrees of freedom, calculer un taux de faux positifs attendu et proposer un plan d'analyse pré-enregistré.

### Consigne

Une équipe de recherche en ergonomie teste l'effet d'un nouvel éclairage LED sur la productivité dans un entrepôt. Elle mesure **8 métriques** différentes : temps de picking, taux d'erreurs, nombre de colis/heure, satisfaction au travail, niveau de fatigue déclaré, absentéisme, accidents bénins, et retards de quai.

Elle effectue un test statistique sur chacune des 8 métriques, avec α = 0,05. Résultat : une seule métrique ressort significative (p = 0,038) : le « nombre de colis/heure ».

**Questions :**

1. Si l'éclairage LED n'avait aucun effet réel sur aucune des 8 métriques, quelle est la probabilité d'obtenir au moins un résultat « significatif » parmi les 8 tests ? (Formule : P(≥1 FP) = 1 − (1 − α)^k, avec k = 8)

2. La probabilité que le résultat observé (p = 0,038 sur « colis/heure ») soit un faux positif est-elle de 3,8 % ? Pourquoi ou pourquoi pas ?

3. L'équipe décide de publier uniquement le résultat sur « colis/heure ». Identifiez les deux principaux problèmes de cette démarche.

4. Proposez un protocole de recherche pré-enregistré qui aurait permis d'éviter ces problèmes. Votre protocole doit comporter au moins 3 éléments concrets.

### Critères de réussite
- [ ] Q1 : P(≥1 FP) = 1 − 0,95^8 ≈ 33,7 % correctement calculé.
- [ ] Q2 : Non — la p-value n'est pas la probabilité que le résultat soit un faux positif ; c'est P(données | H₀). La probabilité réelle dépend aussi du prior (vraisemblance que l'éclairage ait un effet).
- [ ] Q3 : publication selective (biais de publication) + absence de correction pour tests multiples.
- [ ] Q4 : protocole contient au minimum : (a) déclaration des métriques primaires/secondaires avant collecte, (b) correction pour tests multiples (Bonferroni ou FDR), (c) taille d'échantillon pré-calculée (analyse de puissance).
