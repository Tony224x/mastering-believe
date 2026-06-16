# Module 04 — Heuristiques & Biais Cognitifs

> **Temps estimé** : 50 min | **Prérequis** : Modules 01-03

> **Objectif** : Identifier les 5 heuristiques/biais les plus robustes, comprendre quand ils aident et quand ils trompent, et adopter la nuance que les raccourcis mentaux ne sont pas des "erreurs" mais des outils adaptés à certains environnements.

---

## 1. Ce que sont (vraiment) les heuristiques

Une **heuristique** est une règle de décision rapide, frugale en information, qui exploite les régularités de l'environnement. Tversky et Kahneman (1974) les ont décrites comme « économiques mais sources d'erreurs systématiques et prévisibles ». Gigerenzer et ses collègues ont ensuite montré l'autre face : dans de nombreux environnements, une heuristique simple **prédit mieux** qu'un modèle statistique complexe — parce qu'elle ignore le bruit.

> **À retenir** : une heuristique n'est ni bonne ni mauvaise en soi. Son efficacité dépend de l'appariement entre la règle et la structure de l'environnement (*rationalité écologique*).

**Exemple concret — l'heuristique de reconnaissance** : pour prédire quel club de football remportera un match, des participants allemands ignorant la ligue anglaise se contentaient de reconnaître ou non le nom du club. Ce score simple battait des modèles paramétrés sur des statistiques détaillées (Gigerenzer & Todd, 1999). La méconnaissance était une *ressource*, pas un handicap.

---

## 2. Les 5 biais robustes (résultats répliqués)

> **Note réplication** : les effets décrits ci-dessous ont été répliqués dans de nombreuses études. À distinguer des effets de *priming social* (café chaud, amorcage « vieillesse ») qui ont largement échoué à la réplication après 2011. Kahneman lui-même a reconnu cette fragilité. On enseigne ici les résultats solides.

### 2.1 Ancrage numérique

**Mécanisme** : lorsqu'un chiffre quelconque est présenté en premier, il tire les estimations suivantes vers lui, même si ce chiffre est manifestement arbitraire.

**Expérience classique (neutre)** : on demande aux participants de faire tourner une roue de loterie (truquée à 10 ou 65), puis d'estimer le pourcentage de pays africains à l'ONU. Groupe roue=10 : médiane 25 %. Groupe roue=65 : médiane 45 %. L'écart est de 20 points pour un nombre sans rapport.

**Quand ça aide** : l'ancrage est utile pour la négociation (annoncer en premier ancre la fourchette).

**Quand ça trompe** : toute estimation chiffrée (budget, durée d'un projet, valeur d'un bien) risque d'être contaminée par le premier chiffre aperçu, même fortuit.

**Contre-mesure** : générer plusieurs estimations indépendantes *avant* de regarder des références, ou challenger l'ancre en cherchant des arguments dans la direction opposée.

---

### 2.2 Disponibilité

**Mécanisme** : on estime la fréquence ou la probabilité d'un événement par la facilité avec laquelle des exemples viennent à l'esprit.

**Exemple neutre** : après une couverture médiatique intensive d'accidents d'avion, les gens surestiment le risque de mort en avion et sous-estiment celui en voiture — alors que les statistiques de mortalité par km parcouru sont inverses.

**Quand ça aide** : si les événements fréquents sont aussi saillants, la disponibilité fonctionne bien.

**Quand ça trompe** : dès que la saillance médiatique ou émotionnelle diverge de la fréquence réelle.

**Contre-mesure** : chercher des données de base (*base rates*) et des statistiques agrégées.

---

### 2.3 Cadrage (Framing)

**Mécanisme** : la présentation d'une même information change les décisions, même quand les options sont mathématiquement identiques.

**Expérience classique (neutre)** — problème des 600 patients (Tversky & Kahneman, 1981) :
- **Cadrage gain** : programme A sauve 200 personnes ; programme B : 1/3 de chance de sauver 600, 2/3 de chance d'en sauver 0. Majorité choisit A.
- **Cadrage perte** : programme C : 400 personnes mourront ; programme D : 1/3 de chance que personne ne meure, 2/3 de chance que 600 meurent. Majorité choisit D.

A = C et B = D en termes d'espérance. Pourtant les préférences s'inversent. Résultat robuste et répliqué.

**Contre-mesure** : reformuler la même option dans les deux cadrages (gain et perte) avant de décider.

---

### 2.4 Négligence du taux de base (*Base Rate Neglect*)

**Mécanisme** : on ignore les fréquences a priori au profit d'informations descriptives individuelles.

**Exemple neutre chiffré** : un test de dépistage pour une maladie touchant 1 % de la population a une sensibilité de 90 % et un taux de faux positifs de 5 %. Un patient teste positif. Quelle est la probabilité qu'il soit réellement malade ?

Réponse bayésienne :

```
P(malade | positif) = (0,01 × 0,90) / [(0,01 × 0,90) + (0,99 × 0,05)]
                    = 0,009 / (0,009 + 0,0495)
                    ≈ 15,4 %
```

Seulement 15 % — la rareté de la maladie (1 %) pèse lourd.

**Contre-mesure** : toujours demander "quelle est la fréquence de base ?" avant d'interpréter un résultat individuel.

---

### 2.5 Biais de confirmation

**Mécanisme** : on cherche, interprète et mémorise les informations qui confirment nos croyances préexistantes.

**Expérience classique neutre — tâche de Wason (version abstraite)** : on vous montre 4 cartes : E, K, 4, 7. Règle à vérifier : "Si une carte a une voyelle d'un côté, elle a un chiffre pair de l'autre." Quelles cartes retourner ? Correct : E et 7. La majorité choisit E et 4 — cherchant à confirmer plutôt qu'à réfuter.

**Contre-mesure** : chercher activement des arguments *contre* sa position.

---

## 3. Rationalité écologique : quand l'heuristique bat le modèle

Gigerenzer et ses collègues ont montré que des règles simples peuvent surpasser des régressions multiples dans des environnements à haute incertitude.

**Règle "Take the Best"** : pour prédire quelle ville a la plus grande population, utiliser un seul indice discriminant et ignorer tous les autres. Cette règle à un seul indice égalait ou battait des modèles à régression multiple — parce qu'elle évite l'*overfitting*.

**Effet "less-is-more"** : dans certaines tâches de prédiction, *moins* d'information conduit à de *meilleures* prédictions.

> **Message pédagogique** : le cadre "heuristiques = erreurs" (Kahneman) et le cadre "heuristiques = outils adaptatifs" (Gigerenzer) sont **complémentaires**, pas opposés.

---

## 4. Récapitulatif pratique

| Biais | Signal d'alerte | Contre-mesure rapide |
|-------|----------------|----------------------|
| Ancrage | Premier chiffre vu avant d'estimer | Estimer en aveugle, puis chercher des références |
| Disponibilité | Exemples dramatiques en tête | Chercher les statistiques agrégées |
| Cadrage | Décision change si on reformule | Reformuler dans les deux sens (gain/perte) |
| Négligence taux de base | Ignorer la fréquence de base | Calculer P(base) explicitement |
| Confirmation | Chercher seulement ce qui valide | Chercher l'argument réfutant le plus fort |

---

> **À retenir** :
> - Les 5 biais ci-dessus sont robustes et répliqués ; les effets de priming social sont fragiles.
> - Une heuristique n'est pas une erreur : c'est un outil dont l'efficacité dépend de l'environnement.
> - Le remède universel : expliciter les hypothèses implicites et chercher à se réfuter soi-même.

---

## Flash-cards (Module 04)

**Q1** : Qu'est-ce que l'ancrage numérique ? Donnez un exemple neutre.
**R1** : Tendance à tirer ses estimations vers un chiffre présenté en premier. Ex : une roue de loterie affichant 65 fait estimer le nombre de pays africains à l'ONU à ~45 %, contre ~25 % avec une roue à 10.

**Q2** : Pourquoi la disponibilité peut-elle induire en erreur dans l'évaluation des risques ?
**R2** : Parce que la saillance médiatique ou émotionnelle d'un événement ne corrèle pas avec sa fréquence réelle.

**Q3** : Dans le problème des 600 patients, pourquoi les préférences s'inversent-elles entre cadrage gain et cadrage perte ?
**R3** : Parce que la formulation active différemment l'aversion à la perte. En cadrage perte, on devient preneur de risque pour éviter une perte certaine, même si l'espérance est identique.

**Q4** : Donnez le calcul de P(malade | test positif) si prévalence = 1 %, sensibilité = 90 %, faux positifs = 5 %.
**R4** : (0,01 × 0,90) / [(0,01 × 0,90) + (0,99 × 0,05)] ≈ 15,4 %.

**Q5** : Quelle est la thèse centrale de Gigerenzer sur les heuristiques ?
**R5** : Les heuristiques frugales sont adaptées à leur environnement. Dans des environnements bruités avec peu de données, une règle simple peut prédire mieux qu'un modèle complexe (effet "less-is-more").

---

## Points clés à retenir

1. Les heuristiques sont des raccourcis adaptatifs, pas des défauts moraux.
2. Ancrage, disponibilité, cadrage, négligence du taux de base et biais de confirmation sont robustes et répliqués.
3. Les effets de priming social sont fragiles — prudence épistémique requise.
4. La rationalité écologique (Gigerenzer) : une heuristique peut battre un modèle complexe quand elle évite l'overfitting.
5. Les deux programmes (Kahneman/Tversky et Gigerenzer) sont complémentaires.

---

## Pour aller plus loin

- **Article fondateur** : Tversky & Kahneman (1974). *Judgment under Uncertainty: Heuristics and Biases.* Science, 185(4157), 1124-1131. https://www.science.org/doi/10.1126/science.185.4157.1124
- **Synthèse grand public** : Kahneman, D. (2011). *Thinking, Fast and Slow.* Farrar, Straus and Giroux. *(À lire avec le recul sur le priming)*
- **Contrepoint essentiel** : Gigerenzer, G., Todd, P. M. & ABC Research Group (1999). *Simple Heuristics That Make Us Smart.* Oxford University Press. https://global.oup.com/academic/product/simple-heuristics-that-make-us-smart-9780195143812
- **Crise de réplication** : Open Science Collaboration (2015). *Science*, 349(6251). https://osf.io/ezcuj/overview
