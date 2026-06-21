# Module 10 — Lire une étude & stats trompeuses

> **Temps estimé** : 45 min | **Prérequis** : Modules 01–09
> **Objectif** : Comprendre pourquoi un résultat publié peut être faux (p-hacking, taille d'effet, crise de réplication), puis repérer les pièges statistiques classiques dans les médias et les rapports.

---

## 1. Pourquoi un résultat publié peut être faux

### 1.1 Le problème avec la p-value

Une étude conclut qu'un effet est « statistiquement significatif » si la p-value < 0,05. Traduction littérale : *si l'effet n'existait pas du tout (H₀ vraie), la probabilité d'observer une différence aussi grande par hasard est inférieure à 5 %.*

Ce que la p-value **ne dit pas** :
- Elle ne dit pas que l'effet est grand ou important.
- Elle ne dit pas que H₀ est vraie avec 95 % de probabilité.
- Elle ne dit pas que l'étude se répliquerait.

> **À retenir** : p < 0,05 signifie « résultat rare si H₀ vraie » — pas « résultat vrai » ni « résultat utile ».

### 1.2 Taille d'effet > p-value

Imaginons deux études sur l'effet d'un entraînement de mémorisation sur le score à un test (sur 100) :

| Étude | N | Différence moyenne | p-value | Taille d'effet (d de Cohen) |
|-------|---|-------------------|---------|---------------------------|
| A     | 30 | +8 points | 0,12 (non sig.) | d = 0,60 (moyen) |
| B     | 10 000 | +0,3 point | 0,004 (sig.) | d = 0,04 (négligeable) |

L'étude B est statistiquement significative, mais sa taille d'effet est quasiment nulle — l'entraînement n'apporte rien en pratique. L'étude A, non significative, montre un effet de taille moyenne... avec trop peu de participants pour conclure.

**Règle de lecture** : toujours chercher la taille d'effet (d de Cohen, η², odds ratio…) en plus de la p-value.

### 1.3 Le p-hacking : fabriquer un résultat par flexibilité analytique

Simmons, Nelson & Simonsohn (2011) ont montré par simulation qu'un chercheur disposant d'une flexibilité dans ses choix d'analyse (les *researcher degrees of freedom*) peut rendre presque n'importe quel résultat « significatif ».

**Exemple concret** : on mesure l'effet d'une intervention sur le temps de réaction. Le chercheur peut :
1. Décider *après* avoir vu les données d'exclure les participants « aberrants ».
2. Tester 6 variables dépendantes et ne rapporter que celle qui donne p < 0,05.
3. Continuer à recruter des participants jusqu'à ce que p < 0,05 soit atteint.
4. Tester l'interaction avec l'âge, le sexe, la condition, et seulement rapporter ce qui « sort ».

Chacun de ces choix est en apparence raisonnable — mais combinés, ils gonflent le taux de faux positifs bien au-delà de 5 %.

**Simulation** : si on teste 20 hypothèses aléatoires (pur bruit), on s'attend à trouver environ 1 résultat avec p < 0,05 par hasard. Si on ne publie que celui-là, le lecteur voit « un résultat significatif » — sans savoir que 19 tests ont échoué.

> **À retenir** : le p-hacking n'est pas toujours intentionnel. La flexibilité analytique non déclarée suffit à produire des faux positifs à la chaîne.

### 1.4 La crise de réplication

L'Open Science Collaboration (Nosek et al., 2015) a tenté de répliquer 100 études de psychologie publiées dans des revues de premier rang. Résultats :
- Seulement **~36 %** des réplications atteignaient la significativité.
- Les tailles d'effet répliquées étaient en moyenne **deux fois plus petites** que dans l'original.

Ce n'est pas propre à la psychologie : des phénomènes similaires ont été observés en médecine, en économie comportementale et en biologie moléculaire.

**Ce que ça ne veut pas dire** : que la science est inutile ou que tout est faux. Les outils statistiques fonctionnent — mais ils demandent rigueur et transparence.

**Ce que ça veut dire** : un seul résultat publié est une hypothèse, pas une conclusion définitive. La convergence de plusieurs études indépendantes bien menées est bien plus solide.

> **Mention méta-analyse** : la méta-analyse agrège les résultats de plusieurs études pour estimer un effet moyen plus fiable. Elle reste le niveau de preuve le plus élevé, à condition que les études incluses soient comparables et sans biais de publication majeur.

---

## 2. Lire les stats dans les médias et les rapports

Les pièges suivants n'impliquent pas forcément une mauvaise foi — souvent ils viennent d'une simplification maladroite ou d'une méconnaissance.

### 2.1 Graphiques tronqués

Un axe Y qui ne commence pas à zéro peut faire paraître une différence minime comme dramatique.

**Exemple** : taux de colis livrés à temps dans un entrepôt logistique.

```
Axe tronqué (Y de 96 % à 100 %) :    Axe complet (Y de 0 % à 100 %) :
100 % |   ████                         100 % | ████ ████
99 %  | ████ ████                        50 % |
98 %  |                                   0 % | ████ ████
      Mois A  Mois B                          Mois A  Mois B
```

Le graphique de gauche suggère une chute de 50 % de la performance — le graphique de droite montre que la différence est de 1 point de pourcentage.

**Réflexe** : toujours regarder l'origine de l'axe Y avant de lire une variation.

### 2.2 Pourcentage vs points de pourcentage

Un résultat est présenté comme « une amélioration de 50 % ». Mais de quoi ?

| Formulation | Taux de départ | Taux d'arrivée | Différence absolue |
|-------------|---------------|---------------|--------------------|
| « +50 % relatif » | 2 % | 3 % | +1 point de % |
| « +50 % relatif » | 40 % | 60 % | +20 points de % |

Les deux titres sont factuellement vrais mais donnent une impression très différente de l'ampleur de l'effet.

**Réflexe** : toujours chercher les valeurs absolues (les chiffres bruts ou les points de pourcentage) derrière un changement relatif.

### 2.3 Biais de survie

On analyse les caractéristiques d'un groupe *parce qu'il a réussi*, sans voir ceux qui ont échoué mais avaient les mêmes caractéristiques.

**Exemple neutre** : une étude sur les 20 meilleurs athlètes d'une compétition montre qu'ils s'entraînent tous plus de 6 h/jour. Conclusion tirée : « s'entraîner 6 h/jour mène au succès ». Mais combien d'athlètes s'entraînaient 6 h/jour et n'ont pas atteint ce niveau ? Les données sur ceux-là sont absentes.

**Réflexe** : demander « où sont les non-sélectionnés ? ».

### 2.4 Dénominateur manquant

Un titre annonce « 500 pannes signalées sur le nouveau modèle de scanner logistique ». Alarmant ? Ça dépend : 500 pannes sur 500 unités vendues (taux de 100 %) ou sur 500 000 unités (taux de 0,1 %) ?

**Réflexe** : toujours demander « sur combien ? ».

### 2.5 Moyenne vs médiane

La moyenne est sensible aux valeurs extrêmes. La médiane (valeur centrale) est souvent plus représentative pour des distributions asymétriques.

**Exemple** : temps de traitement des commandes dans un entrepôt (en minutes).

```
Commandes : 3, 4, 4, 5, 5, 6, 6, 7, 8, 120
Moyenne   : (3+4+4+5+5+6+6+7+8+120) / 10 = 16,8 min
Médiane   : (5+6)/2 = 5,5 min
```

Un incident exceptionnel (commande à 120 min) fait tripler la moyenne apparente. La médiane reflète mieux l'expérience typique.

**Réflexe** : en présence d'une distribution asymétrique ou d'outliers potentiels, préférer la médiane. Toujours demander quelle mesure de tendance centrale est utilisée.

---

## 3. Synthèse : grille de lecture rapide

Quand vous lisez un résultat d'étude ou une statistique médiatique, posez-vous ces questions :

| Question | Piège évité |
|----------|-------------|
| Quelle est la taille d'effet, pas seulement la p-value ? | Significatif statistiquement ≠ important pratiquement |
| L'étude a-t-elle été pré-enregistrée ? | P-hacking et flexibilité analytique |
| A-t-elle été répliquée ? | Un seul résultat = une hypothèse |
| L'axe Y commence-t-il à zéro ? | Graphique trompeur |
| S'agit-il de % relatifs ou de points de % ? | Grossissement optique des effets |
| Voit-on les « perdants » de la sélection ? | Biais de survie |
| Sur combien d'unités/cas ? | Dénominateur manquant |
| Moyenne ou médiane ? | Sensibilité aux valeurs extrêmes |

---

## Flash-cards

**Q1 : Qu'est-ce que le p-hacking et pourquoi est-il problématique ?**
> R : Le p-hacking consiste à exploiter la flexibilité analytique (choix des variables, des exclusions, du moment d'arrêt) pour obtenir p < 0,05. Il augmente le taux de faux positifs bien au-delà de 5 % sans que ce soit forcément intentionnel.

**Q2 : Quelle différence entre significatif statistiquement et important pratiquement ?**
> R : La significativité statistique dépend aussi de la taille de l'échantillon — avec N assez grand, une différence infime devient significative. La taille d'effet (d de Cohen, etc.) mesure l'ampleur réelle de l'effet, indépendamment de N.

**Q3 : Que signifie la crise de réplication ?**
> R : Une majorité d'études en sciences comportementales n'a pas pu être reproduite avec des résultats similaires (OSC 2015 : ~36 % de réplications significatives). Cela signifie qu'un seul résultat publié est une hypothèse à confirmer, pas une vérité établie.

**Q4 : Comment repérer un graphique tronqué ?**
> R : Regarder si l'axe Y commence à zéro. S'il commence à une valeur élevée, une faible variation paraîtra dramatique. Mentalement, remettre l'axe à zéro pour évaluer l'amplitude réelle.

**Q5 : Pourquoi la médiane est-elle souvent plus utile que la moyenne ?**
> R : La moyenne est tirée vers les valeurs extrêmes (outliers). Dans une distribution asymétrique (revenus, temps de traitement, tailles de populations), la médiane représente mieux la valeur typique.

---

## Points clés à retenir

- La p-value dit « est-ce rare si H₀ vraie ? » — pas « est-ce vrai ? » ni « est-ce important ? ».
- La taille d'effet est l'information pratique ; la p-value seule est insuffisante.
- Le p-hacking gonfle les faux positifs ; le pré-enregistrement en est le principal antidote.
- Un seul résultat publié = une hypothèse. La réplication et la convergence de plusieurs études construisent la confiance.
- Dans les médias : vérifier l'axe Y, distinguer % relatifs et points de %, chercher le dénominateur, regarder si les non-sélectionnés existent, choisir la bonne mesure de tendance centrale.

---

## Pour aller plus loin

- **Ioannidis, J.P.A.** (2005). *Why Most Published Research Findings Are False.* PLoS Medicine 2(8):e124. https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124
- **Simmons, J.P., Nelson, L.D. & Simonsohn, U.** (2011). *False-Positive Psychology: Undisclosed Flexibility in Data Collection and Analysis Allows Presenting Anything as Significant.* Psychological Science 22(11):1359-1366. https://journals.sagepub.com/doi/10.1177/0956797611417632
- **Open Science Collaboration** (2015). *Estimating the Reproducibility of Psychological Science.* Science 349(6251):aac4716. https://www.science.org/doi/10.1126/science.aac4716
