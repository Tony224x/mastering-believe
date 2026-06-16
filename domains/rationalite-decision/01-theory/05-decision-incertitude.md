# Module 05 — Décision sous Incertitude

> **Temps estimé** : 50 min | **Prérequis** : Modules 01-04

> **Objectif** : Maîtriser les outils fondamentaux de la décision rationnelle sous incertitude — espérance, utilité, arbres de décision — et comprendre pourquoi nous dévions de ces modèles (paradoxes d'Allais et d'Ellsberg).

---

## 1. L'espérance mathématique : la boussole de base

L'**espérance mathématique** est la somme des gains possibles pondérés par leurs probabilités.

**Formule** : E[X] = Σ pᵢ × xᵢ

**Exemple — pari équitable** : un dé à 6 faces. Gain de 6 € sur le 6, perte de 1 € sinon.

```
E = (1/6 × 6) + (5/6 × (−1)) = 1,00 − 0,833 = +0,167 €
```

Pari légèrement favorable.

**Paradoxe de Saint-Pétersbourg** : pile ou face. Pile au ke lancer = 2^k €. Espérance infinie. Pourtant personne ne paie plus de 20-30 € — preuve que maximiser l'espérance monétaire n'est pas toujours le bon modèle.

---

## 2. Utilité et aversion au risque

**L'utilité** mesure la valeur *subjective* d'un résultat. L'utilité croît de façon concave (chaque euro supplémentaire vaut moins quand on en possède déjà beaucoup).

**Aversion au risque** : la plupart préfèrent un gain certain à un pari d'espérance identique.

- **Option A** : 50 € certains.
- **Option B** : 50 % de chance de 110 €, 50 % de 0 €. E[B] = 55 €.

Majorité choisit A malgré l'espérance inférieure. Courbe d'utilité **concave**.

**Formule d'utilité espérée** (von Neumann-Morgenstern, 1944) : EU = Σ pᵢ × U(xᵢ)

**Profils** :
- Neutre au risque : U(x) = x
- Averse au risque : U(x) = log(x) ou √x
- Chercheur de risque : U(x) = x²

---

## 3. Arbres de décision

Un **arbre de décision** structure visuellement les choix, conséquences et probabilités.

**Éléments** : □ Nœud de décision | ○ Nœud de hasard | → Branche avec probabilité et valeur.

**Exemple — assurance illustrative** :

Vélo de 400 €. Assurance 40 €/an. Risque de vol = 8 %.

```
Sans assurance : E = 0,08 × (−400) + 0,92 × 0 = −32 €
Avec assurance : coût certain = −40 €
```

L'espérance monétaire favorise "sans assurance" (−32 > −40). Mais si la perte de 400 € est très pénible pour votre utilité, payer 40 € pour l'éviter peut être rationnel.

**Calcul** : pour chaque branche, multiplier probabilité × valeur ; remonter les nœuds de hasard ; aux nœuds de décision, choisir la branche à l'utilité espérée maximale.

---

## 4. Le paradoxe d'Allais (1953)

**Situation 1** — choisir entre :
- **1A** : 100 % de 1 000 000 €
- **1B** : 89 % de 1 000 000 €, 10 % de 5 000 000 €, 1 % de 0 €

Majorité préfère 1A.

**Situation 2** — choisir entre :
- **2A** : 11 % de 1 000 000 €, 89 % de 0 €
- **2B** : 10 % de 5 000 000 €, 90 % de 0 €

Majorité préfère 2B.

**La contradiction** : 1A > 1B et 2B > 2A violent l'axiome d'indépendance de von Neumann-Morgenstern. En retirant 89 % de 1 000 000 € des deux options de Situation 1, on retrouve Situation 2 — et les préférences s'inversent. Résultat répliqué robustement.

**Interprétation** : la certitude a une valeur spéciale (effet de certitude) non capturée par l'utilité espérée.

---

## 5. Le paradoxe d'Ellsberg (1961)

**Expérience** : 90 billes. 30 rouges. 60 noires ou jaunes (proportion inconnue).

**Situation 1** : A (100 € si rouge) vs B (100 € si noire). Majorité préfère A.
**Situation 2** : C (100 € si rouge ou jaune) vs D (100 € si noire ou jaune). Majorité préfère D.

**La contradiction** : préférer A implique P(rouge) > P(noire), mais préférer D implique P(noire) > P(rouge). L'incohérence vient de l'**ambiguïté** : on fuit les paris sur des probabilités inconnues.

**Leçon** : les humains distinguent risque (probabilités connues) et ambiguïté (probabilités inconnues), et sont *averse à l'ambiguïté*.

---

## 6. Que faire de tout cela ?

| Outil | Usage adapté | Limite |
|-------|--------------|--------|
| Espérance mathématique | Décisions répétées à enjeux faibles | Ne tient pas compte de l'aversion au risque |
| Utilité espérée | Décisions importantes asymétriques | Difficile à calibrer, viole Allais |
| Arbre de décision | Problème multi-étapes | Requiert d'estimer les probabilités |

**Règle pratique en 3 étapes** :
1. Lister les options avec probabilités.
2. Corriger pour l'aversion au risque (question : "préférerais-je cela en certitude réduite ?").
3. Vérifier la cohérence en reformulant dans le cadrage inverse.

---

> **À retenir** :
> - L'espérance mathématique est le point de départ, pas la fin.
> - L'aversion au risque est rationnelle si la douleur de la perte est supérieure au plaisir du gain.
> - Les paradoxes d'Allais et d'Ellsberg : nous traitons différemment certitude, risque et ambiguïté.

---

## Flash-cards (Module 05)

**Q1** : Calculez l'espérance d'un pari où vous gagnez 50 € avec probabilité 0,3 et perdez 10 € avec probabilité 0,7.
**R1** : E = 0,3 × 50 + 0,7 × (−10) = 15 − 7 = **+8 €** (pari favorable).

**Q2** : Qu'est-ce que l'utilité espérée et pourquoi la distingue-t-on de l'espérance monétaire ?
**R2** : L'utilité espérée pondère les outcomes par leur valeur subjective U(x). Cela modélise l'aversion au risque : 100 € certains peut valoir plus qu'un pari d'espérance 110 € si U est concave.

**Q3** : Dans le paradoxe d'Allais, quel axiome est violé ?
**R3** : L'axiome d'indépendance. La préférence entre deux loteries ne devrait pas changer si on retire le même composant des deux. L'effet de certitude crée une violation systématique.

**Q4** : Qu'est-ce que l'aversion à l'ambiguïté (paradoxe d'Ellsberg) ?
**R4** : La tendance à éviter les paris dont les probabilités sont inconnues, même quand l'espérance est identique à un pari à probabilités connues.

**Q5** : Comment construire un arbre de décision simple ?
**R5** : 1) Nœud de décision (□) pour chaque choix. 2) Nœuds de hasard (○) avec probabilités. 3) Valeurs en bout de branches. 4) Remonter en multipliant p × valeur, choisir le nœud à espérance maximale.

---

## Points clés à retenir

1. L'espérance mathématique est la règle de base : sommer les gains pondérés par leurs probabilités.
2. L'utilité espérée raffine en tenant compte de la valeur subjective.
3. L'aversion au risque est rationnelle si votre courbe d'utilité est concave.
4. Le paradoxe d'Allais : l'effet de certitude viole les axiomes de l'utilité espérée de façon robuste.
5. Le paradoxe d'Ellsberg : nous fuyons l'ambiguïté (probabilités inconnues) au-delà du risque (probabilités connues).

---

## Pour aller plus loin

- **Manuel de référence** : Peterson, M. (2017). *An Introduction to Decision Theory*, 2e éd. Cambridge University Press. https://www.cambridge.org/core/books/an-introduction-to-decision-theory/B9EEB3DCE5D0CAFFB6F3F30B1D0A06A6
- **Théorie normative** : Stanford Encyclopedia of Philosophy. https://plato.stanford.edu/entries/rationality-normative-utility/
- **Alternatives** : Stanford Encyclopedia — *Rivals to Expected Utility*. https://plato.stanford.edu/entries/rationality-normative-nonutility/
- **Théorie des perspectives** : Kahneman, D. & Tversky, A. (1979). Prospect Theory. *Econometrica*, 47(2), 263-291.
