# Module 06 — Rationalité Écologique

> **Temps estimé** : 45 min | **Prérequis** : Modules 01-05

> **Objectif** : Comprendre le programme de Gigerenzer sur les heuristiques rapides et frugales, saisir quand une règle simple bat un modèle complexe (effet *less-is-more*), et recadrer le message de J5 : les raccourcis cognitifs ne sont pas que des erreurs — ils peuvent être des solutions adaptées à leur environnement.

---

## 1. Le débat fondateur : erreurs vs. adaptations

Le module 05 a présenté la tradition Tversky-Kahneman : les heuristiques produisent des biais systématiques et prévisibles. Cette lecture est solide pour les contextes où un calcul probabiliste précis est possible et où les environnements sont bien définis.

Mais depuis la fin des années 1990, Gerd Gigerenzer et l'**ABC Research Group** (Adaptive Behavior and Cognition, Max Planck Institut) ont construit un contre-programme expérimental : les heuristiques rapides et frugales sont souvent **optimales**, pas malgré leur simplicité, mais *grâce* à elle.

> **À retenir** : les deux programmes ne s'excluent pas. Tversky-Kahneman montre *quand* les heuristiques échouent (environnements à structure aléatoire, tâches probabilistes abstraites). Gigerenzer montre *quand* elles réussissent (environnements stables avec régularités exploitables). La question n'est pas « les heuristiques sont-elles bonnes ou mauvaises ? » mais **« cette heuristique est-elle adaptée à cet environnement ? »**

Source : Gigerenzer, Todd & ABC Research Group, *Simple Heuristics That Make Us Smart*, Oxford University Press, 1999.

---

## 2. La rationalité écologique : le concept central

Une décision n'est pas rationnelle dans l'absolu — elle est rationnelle *relativement à un environnement donné*. Gigerenzer appelle cela la **rationalité écologique** (*ecological rationality*) : une règle est bonne si elle est bien appariée (*matched*) à la structure du monde dans lequel elle s'applique.

Analogie : un couteau suisse est un outil polyvalent, mais dans un restaurant gastronomique, un seul bon couteau à désosser bat le couteau suisse. La « complexité » de l'outil n'est pas la variable pertinente — c'est l'**adéquation outil-tâche**.

**Les trois propriétés des heuristiques rapides et frugales :**
1. **Rapides** (*fast*) : elles exigent peu de temps de calcul.
2. **Frugales** (*frugal*) : elles n'utilisent qu'un sous-ensemble des informations disponibles.
3. **Robustes** : elles généralisent bien à de nouveaux cas, car elles ne surajustent pas aux données d'entraînement.

---

## 3. L'heuristique de reconnaissance (*Recognition Heuristic*)

### 3.1 La règle

> « Si tu reconnais l'objet A mais pas l'objet B, alors A a une valeur critérielle plus élevée que B. »

Autrement dit : la mémoire de reconnaissance — le simple fait d'avoir entendu parler d'une chose — est utilisée comme signal de rang ou de valeur.

### 3.2 Expérience classique : villes allemandes et américaines

Dans une expérience fondatrice, on demandait à des étudiants allemands et américains quelle ville avait la plus grande population, parmi des paires de villes. Résultat contre-intuitif :

- Les **étudiants américains** (qui connaissaient bien leurs propres villes) reconnaissaient toutes les paires → ils devaient estimer, et commettaient des erreurs.
- Les **étudiants allemands** (qui n'en reconnaissaient qu'une sur deux en moyenne) pouvaient appliquer la règle de reconnaissance → ils obtenaient un taux de bonnes réponses **supérieur** à celui des Américains.

**Pourquoi ?** Parce que la reconnaissance n'est pas aléatoire : on tend à avoir entendu parler d'une ville *précisément* parce qu'elle est grande et médiatisée. La méconnaissance partielle est une ressource, pas un déficit.

> **À retenir** : l'heuristique de reconnaissance fonctionne dans les environnements où la **fréquence d'apparition dans les médias/la mémoire est corrélée avec la valeur critérielle** (taille, classement, popularité). Elle échoue quand cette corrélation est absente ou inversée.

### 3.3 Exemple sport : classement de clubs de football

Des participants ignorant la Premier League anglaise devaient prédire le vainqueur de matchs entre clubs. En appliquant la règle « je reconnais A mais pas B → A gagne », ils atteignaient un taux de bonnes réponses d'environ 65 % — supérieur à des modèles paramétrés sur les statistiques récentes des clubs.

La notoriété d'un club (capital médiatique, palmarès passé) est corrélée à ses performances futures. Une règle simple exploite cette régularité de l'environnement sans même avoir besoin de statistiques détaillées.

---

## 4. Take-The-Best : une heuristique de recherche séquentielle

L'heuristique de reconnaissance s'applique à des choix binaires simples. Gigerenzer et ses collègues ont étendu l'idée à des jugements multi-attributs avec **Take-The-Best** (TTB).

### 4.1 La règle en trois étapes

1. **Trier les indices** (*cues*) par validité décroissante — du plus discriminant au moins discriminant.
2. **Chercher séquentiellement** : regarder le premier indice (le plus valide). S'il discrimine, s'arrêter et décider.
3. **Si pas de discrimination**, passer à l'indice suivant. Répéter jusqu'à décision ou épuisement.

### 4.2 Exemple : prédire le chiffre d'affaires d'une ville

Pour prédire lequel de deux marchés a le plus grand chiffre d'affaires, les indices disponibles sont : taille de la ville, présence d'une université, réseau ferroviaire, ligne d'autobus, code postal. Au lieu d'intégrer tous les indices dans une régression linéaire, TTB regarde *d'abord* la taille (indice le plus valide) : si elle discrimine, la décision est prise.

**Résultat empirique** : sur des jeux de données réels (villes allemandes, Gigerenzer & Goldstein, 1996), TTB prédit aussi bien — voire mieux — qu'une régression de moindres carrés dans 19 environnements sur 20 testés.

### 4.3 Pourquoi TTB peut battre une régression ?

La régression linéaire **optimise sur les données d'entraînement**, ce qui la rend vulnérable au **surajustement** (*overfitting*) quand les données sont limitées ou bruitées. TTB, en n'utilisant qu'un seul indice, n'a pas de paramètre à estimer → elle ne surajuste pas.

> **À retenir** : quand les données sont rares, bruitées ou que les environnements sont non-stationnaires, **moins de paramètres = moins d'erreur de généralisation**. La parcimonie n'est pas une concession à la limitation cognitive — c'est parfois la stratégie statistiquement optimale.

---

## 5. L'effet *less-is-more* (moins c'est plus)

Le paradoxe de la rationalité écologique est capturé par l'effet *less-is-more* : **ignorer délibérément une partie de l'information disponible peut améliorer la précision des prédictions**.

### 5.1 Conditions d'apparition

L'effet *less-is-more* se produit quand :
- L'environnement a une structure exploitable (corrélations stables entre indices et critère).
- Les données sont limitées (pas assez d'exemples pour estimer des poids précis).
- Le bruit est élevé (les données d'entraînement contiennent de l'aléatoire).

Dans ces conditions, un modèle complexe (régression, forêt aléatoire, réseau de neurones) apprend le bruit plutôt que le signal. Une heuristique simple, en ignorant le bruit, généralise mieux.

### 5.2 Illustration : prédire la survie d'un patient

Dans une étude sur la prise en charge de patients en infarctus en urgence, les médecins utilisaient un score complexe combinant de nombreuses variables. Breiman & al. ont montré qu'un arbre de décision à **3 nœuds** (fréquence cardiaque > 100 ? présence de certains signes ECG ? âge > 65 ?) classifiait les cas à risque avec une précision identique — et réduisait les erreurs dans les cas rares où le modèle complexe surajustait. (*Classification and Regression Trees*, Breiman et al., 1984 ; repris dans Gigerenzer 1999.)

### 5.3 Ce que le *less-is-more* n'est pas

L'effet ne s'applique pas universellement :
- Dans des environnements stables avec beaucoup de données propres, les modèles complexes reprennent l'avantage.
- La règle « ignorer l'information » doit être justifiée par la structure de l'environnement, pas par la paresse.

> **À retenir** : *less-is-more* est un phénomène empirique conditionnel, pas une philosophie générale. Il remet en cause le mythe que « plus d'information = meilleure décision » — mais seulement dans des conditions précises (données rares, bruit élevé, structure exploitable).

---

## 6. Recadrage : les raccourcis ne sont pas que des erreurs

J5 a établi que les heuristiques produisent des biais systématiques dans des tâches probabilistes abstraites. Ce module ajoute la perspective complémentaire :

| Contexte | Heuristique simple | Modèle complexe |
|---|---|---|
| Environnement régulier, données rares | Meilleure précision | Surajustement |
| Environnement aléatoire, données abondantes | Sous-optimal | Meilleur |
| Décision rapide sous contrainte de temps | Seule option viable | Impossible à calculer |
| Tâche probabiliste abstraite hors contexte | Produit des biais | Correct si on calcule |

Les heuristiques sont des outils adaptatifs : elles ont évolué et se sont affinées culturellement parce qu'elles fonctionnent dans les environnements auxquels elles s'appliquent. La question n'est pas « cet agent est-il rationnel ? » mais **« cette règle est-elle adaptée à cet environnement ? »**

**Implication pratique** : avant de « débiaser » quelqu'un, demandez si l'heuristique utilisée est réellement mal adaptée à la situation, ou si elle fonctionne correctement dans son contexte habituel.

---

## 7. Flash-cards

**Q1 : Qu'est-ce que la rationalité écologique selon Gigerenzer ?**
> R : Une heuristique est rationnelle si elle est bien appariée (*matched*) à la structure de l'environnement dans lequel elle s'applique. La rationalité n'est pas absolue — elle est toujours relative à un contexte.

**Q2 : Expliquer l'heuristique de reconnaissance en une phrase.**
> R : Si je reconnais A mais pas B, je conclus que A a une valeur critérielle plus élevée (ex. : taille, classement) — car la reconnaissance est corrélée à la valeur dans de nombreux environnements informatifs.

**Q3 : Quelles sont les 3 étapes de Take-The-Best ?**
> R : (1) Trier les indices par validité décroissante. (2) Regarder le premier indice — s'il discrimine, décider. (3) Sinon, passer à l'indice suivant et répéter.

**Q4 : Dans quelles conditions l'effet *less-is-more* apparaît-il ?**
> R : Quand les données sont rares ou bruitées, et que l'environnement a une structure stable. Dans ce cas, un modèle simple généralise mieux qu'un modèle complexe qui surajuste le bruit.

**Q5 : Quelle est la différence essentielle entre la thèse de Tversky-Kahneman et celle de Gigerenzer ?**
> R : Tversky-Kahneman : les heuristiques produisent des erreurs systématiques (perspective de la norme probabiliste). Gigerenzer : les heuristiques sont adaptées à leur environnement et peuvent battre des modèles complexes (perspective de la rationalité écologique). Les deux sont complémentaires.

---

## Points clés à retenir

- La **rationalité écologique** évalue une heuristique par son adéquation à un environnement, pas par rapport à un idéal probabiliste abstrait.
- L'**heuristique de reconnaissance** : reconnaître A et pas B → A a une valeur plus haute. Fonctionne quand la reconnaissance est corrélée au critère.
- **Take-The-Best** : chercher séquentiellement par indice de validité décroissante, s'arrêter dès qu'un indice discrimine.
- L'effet **less-is-more** : ignorer de l'information peut améliorer la précision quand les données sont rares ou bruitées.
- Les raccourcis cognitifs ne sont **ni toujours des erreurs ni toujours des solutions** — leur valeur dépend de l'adéquation avec l'environnement.

---

## Pour aller plus loin

- **Simple Heuristics That Make Us Smart** — Gerd Gigerenzer, Peter M. Todd & ABC Research Group, 1999. Oxford University Press. https://global.oup.com/academic/product/simple-heuristics-that-make-us-smart-9780195143812 — Le texte fondateur du programme de rationalité écologique.

- **Rationality for Mortals: How People Cope with Uncertainty** — Gerd Gigerenzer, 2008. Oxford University Press. https://global.oup.com/academic/product/rationality-for-mortals-9780199747092 — Version plus accessible du même programme, avec des chapitres sur la santé et la finance.

- **Rationality Wars** (synthèse du débat) — *Behavioural Public Policy*, Cambridge Core. https://www.cambridge.org/core/journals/behavioural-public-policy/article/rationality-wars-epistemological-boundaries-and-the-limits-of-reductionism/D468AE06999D3216978B5AB4A1644187 — Analyse des deux programmes comme complémentaires plutôt qu'adversaires.

- **Judgment under Uncertainty: Heuristics and Biases** — Tversky & Kahneman, 1974. *Science* 185(4157):1124-1131. https://www.science.org/doi/10.1126/science.185.4157.1124 — Le programme original, à lire en parallèle de Gigerenzer pour avoir les deux perspectives.
