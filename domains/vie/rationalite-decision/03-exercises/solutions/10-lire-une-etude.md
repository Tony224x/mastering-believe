# Solutions — Module 10 : Lire une étude & stats trompeuses

> Corrigé chiffré modèle. Consulter après avoir tenté les exercices.

---

## Solution — Exercice 1 : p-value insuffisante

### Question 1 — Étude 1

L'étude 1 est **statistiquement significative** (p = 0,003 < 0,05). Mais la taille d'effet est **d = 0,04**, soit un effet quasi nul (les seuils de Cohen : petit ≥ 0,20, moyen ≥ 0,50, grand ≥ 0,80).

Différence concrète : 58,2 − 57,9 = **0,3 minute** (18 secondes) par trajet. Avec un très grand nombre de tournées, même 18 secondes devient statistiquement détectable — mais 18 secondes est sans intérêt opérationnel pour un logisticien.

**Recommandation** : ne pas changer de logiciel sur la base de cette étude. La différence est statistiquement réelle mais pratiquement négligeable. Évaluer plutôt le coût de migration, la fiabilité, et l'ergonomie.

### Question 2 — Étude 2

**Non.** p = 0,11 ne signifie pas « pas d'effet » — cela signifie « données insuffisantes pour rejeter H₀ ». La différence observée est de 7 minutes (d = 0,72, taille d'effet moyenne à grande). L'étude n'a simplement **pas assez de puissance statistique** pour détecter cet effet avec certitude.

Règle : **absence de preuve ≠ preuve d'absence**.

### Question 3 — Ce qui manque

La **taille d'échantillon** (N) n'est pas mentionnée. Avec un d = 0,72, une étude bien dimensionnée (ex. N ≈ 25 par groupe pour puissance 0,80) aurait probablement donné p < 0,05. L'étude 2 est probablement sous-alimentée.

Information manquante : analyse de puissance (power analysis) ou indication du N effectif.

---

## Solution — Exercice 2 : pièges dans un rapport

### Piège 1 — % relatifs vs points de pourcentage

**Formulation trompeuse** : « progressé de 25 % en un mois »

**Réalité** : le taux passe de 92 % à 93 %. La variation absolue est de **+1 point de pourcentage**. Le « 25 % » est un changement relatif calculé ainsi : (93 − 92) / 92 × 100 ≈ 1,1 %, *pas* 25 %. La formulation du rapport est incohérente (le calcul de 25 % ne correspond à rien).

**Reformulation correcte** : « Le taux de livraisons dans les délais a progressé de 92 % à 93 %, soit +1 point de pourcentage. »

### Piège 2 — Dénominateur manquant

**Formulation trompeuse** : « 47 incidents ce mois-ci, un chiffre qui doit alerter »

**Réalité** : 47 incidents sur combien de livraisons ? Si le centre traite 50 000 colis/mois, le taux d'incidents est 47/50 000 = **0,094 %** — excellent. Si le centre traite 500 colis/mois, c'est **9,4 %** — problématique.

**Reformulation correcte** : « 47 incidents sur X livraisons, soit un taux de Y %. »

### Piège 3 — Graphique tronqué

**Formulation trompeuse** : « courbe en forte hausse confirme une dégradation majeure » (axe Y de 18 à 22 min)

**Réalité** : l'axe commence à 18 min. La variation réelle est de 19 min → 21 min = +2 minutes, soit **+10,5 %**. Sur un axe commençant à 0, la courbe serait quasi plate. La représentation visuelle grossit l'effet d'un facteur ~5.

**Reformulation correcte** : « Le temps de traitement est passé de 19 à 21 min (+2 min, +10,5 %). » + graphique avec axe Y à 0.

### Piège 4 (bonus) — Moyenne vs médiane

**Formulation trompeuse** : « temps de traitement moyen de 21 min — protéger les opérateurs les plus lents »

**Réalité** : les 8 valeurs sont : 4, 5, 6, 7, 8, 9, 10, 90 min.

```
Moyenne  = (4+5+6+7+8+9+10+90) / 8 = 139 / 8 = 17,4 min  ← tirée par l'outlier
Médiane  = (7+8) / 2 = 7,5 min  ← représente mieux l'opérateur typique
```

(Note : le rapport cite 21 min — probablement issu d'un autre calcul, mais l'argument de l'outlier reste valide.) L'opérateur à 90 minutes est probablement un cas exceptionnel (incident, formation, panne). La médiane de 7,5 min montre que 7 opérateurs sur 8 sont rapides. La « dégradation majeure » suggérée par la moyenne n'existe pas pour la majorité de l'équipe.

**Reformulation correcte** : « La médiane des temps de traitement est de 7,5 min. Un opérateur présente un temps atypique de 90 min (probablement lié à un incident ponctuel) qui tire la moyenne à la hausse. »

---

## Solution — Exercice 3 : p-hacking

### Question 1 — Probabilité d'au moins un faux positif

```
P(≥1 FP) = 1 − (1 − α)^k
           = 1 − (1 − 0,05)^8
           = 1 − 0,95^8
           = 1 − 0,6634
           ≈ 33,7 %
```

Autrement dit : même si l'éclairage LED n'avait **aucun effet** sur aucune des 8 métriques, il y a une chance sur trois d'obtenir au moins un résultat « significatif » par hasard.

### Question 2 — La p-value = probabilité d'un faux positif ?

**Non.** La p-value est P(données aussi extrêmes | H₀ vraie). Ce n'est pas P(H₀ vraie | données observées), qui est ce que nous voulons réellement savoir.

Pour estimer la probabilité que le résultat soit un vrai positif, il faudrait appliquer le théorème de Bayes en tenant compte du **prior** : quelle est la vraisemblance *a priori* que l'éclairage LED affecte le nombre de colis/heure ? Si ce prior est faible (peu de mécanisme plausible), la valeur prédictive positive du test est elle-même faible — même à p < 0,05.

Ioannidis (2005) montre formellement que dans des domaines avec de nombreuses hypothèses testées et des effets attendus faibles, la majorité des résultats significatifs publiés sont de faux positifs.

### Question 3 — Problèmes de la publication sélective

1. **Biais de publication** (publication bias) : seul le résultat « positif » est publié ; les 7 tests non significatifs disparaissent. Un méta-analyste futur ne verra que le résultat favorable et en surestimera l'effet.

2. **Absence de correction pour tests multiples** : avec 8 tests à α = 0,05, le seuil effectif pour chaque test devrait être ajusté. Par correction de Bonferroni, le seuil par test serait 0,05 / 8 = **0,00625**. Avec p = 0,038, le résultat ne serait **pas** significatif après correction.

### Question 4 — Protocole pré-enregistré

Un protocole robuste contiendrait au minimum :

**(a) Déclaration des métriques primaires avant collecte**
Choisir *une* métrique principale (ex. nombre de colis/heure) sur laquelle repose la conclusion principale, définie *avant* de voir les données. Les 7 autres métriques deviennent secondaires ou exploratoires, clairement labellisées comme telles.

**(b) Correction pour tests multiples**
Si plusieurs tests sont planifiés, appliquer une correction a priori : Bonferroni (seuil par test = α/k), ou FDR (taux de fausses découvertes) selon le contexte. Le seuil ajusté doit être déclaré dans le protocole.

**(c) Analyse de puissance pré-collecte**
Calculer la taille d'échantillon nécessaire pour détecter l'effet minimal jugé pratiquement utile, avec une puissance ≥ 0,80. Cela évite d'accumuler des données jusqu'à ce que p < 0,05 (arrêt séquentiel non planifié = une forme de p-hacking).

**(d) Dépôt du protocole** (optionnel mais recommandé)
Enregistrer le protocole sur un registre public (ex. OSF — Open Science Framework) avant le début de l'étude, avec date. Cela rend toute déviation visible.
