# Module 07 — Décision sous incertitude

> **Temps estimé** : 45 min | **Prérequis** : Modules 01-06
>
> **Objectif** : Maîtriser les outils fondamentaux de la décision rationnelle sous incertitude — espérance mathématique, utilité espérée, aversion au risque, arbres de décision — et comprendre pourquoi nos choix dévient de ces modèles (paradoxes d'Allais et d'Ellsberg).

---

## 1. L'espérance mathématique : point de départ

**Scénario concret d'abord.** On vous propose de lancer un dé équilibré à 6 faces : si le résultat est 6, vous gagnez 12 € ; sinon, vous perdez 2 €. Vaut-il la peine de jouer ?

**Formule** : E[X] = Σ pᵢ × xᵢ

```
E = (1/6 × 12) + (5/6 × (−2))
  = 2,00 − 1,67
  = +0,33 €
```

Espérance positive : sur un grand nombre de lancers, vous gagneriez en moyenne 0,33 € par partie. Ce n'est pas un pari nul — il vous est légèrement favorable.

**Pari équitable** : quand E = 0. Exemple — pile ou face, +5 € sur pile, −5 € sur face : E = 0,5 × 5 + 0,5 × (−5) = 0 €.

> **À retenir** : L'espérance mathématique est la valeur moyenne à long terme. Elle suppose que vous pouvez répéter le pari un grand nombre de fois — or beaucoup de décisions importantes sont uniques.

**Limite immédiate — le paradoxe de Saint-Pétersbourg.** Pile ou face : pile au k-ième lancer = 2^k €. L'espérance est infinie. Pourtant, expérimentalement, personne ne paie plus de 20-30 € pour jouer. Conclusion : maximiser l'espérance *monétaire* n'est pas toujours le bon critère — il faut modéliser la valeur *subjective* des gains.

---

## 2. Utilité et aversion au risque

**L'utilité** mesure la valeur subjective d'un résultat pour un individu donné. La fonction d'utilité U(x) traduit la richesse x en satisfaction.

**Exemple central :**

- **Option A** : 300 € certains.
- **Option B** : 50 % de chance de 700 €, 50 % de 0 €.

Calculons les espérances monétaires :

```
E[A] = 300 €
E[B] = 0,5 × 700 + 0,5 × 0 = 350 €
```

Malgré E[B] > E[A], une majorité de personnes choisit A. Pourquoi ? La courbe d'utilité est **concave** : chaque euro supplémentaire apporte moins de satisfaction que le précédent.

**Utilité espérée (von Neumann-Morgenstern, 1944)** :

```
EU = Σ pᵢ × U(xᵢ)
```

Avec U(x) = √x (exemple de courbe concave) :

```
EU[A] = √300 ≈ 17,32
EU[B] = 0,5 × √700 + 0,5 × √0 ≈ 0,5 × 26,46 = 13,23
```

Ici, EU[A] > EU[B] — l'option certaine est préférable malgré l'espérance monétaire inférieure.

**Trois profils d'attitude face au risque :**

| Profil | Courbe U(x) | Exemple | Comportement |
|--------|------------|---------|--------------|
| Averse au risque | Concave (√x, log x) | La plupart des individus | Préfère le certain à l'espérance équivalente |
| Neutre au risque | Linéaire (U = x) | Modèle de base | Indifférent entre certain et pari d'égale espérance |
| Chercheur de risque | Convexe (x²) | Rare à tous les niveaux | Préfère le pari au gain certain équivalent |

> **À retenir** : L'aversion au risque est **rationnelle**, pas irrationnelle. Si perdre 300 € vous fait beaucoup plus de mal que gagner 300 € ne vous fait de bien, votre courbe d'utilité concave capture correctement cette réalité.

---

## 3. Arbres de décision

Un **arbre de décision** structure visuellement un problème multi-étapes en identifiant explicitement les choix, les aléas et leurs valeurs.

**Éléments de base :**
- □ Nœud de décision — on choisit la branche
- ○ Nœud de hasard — la nature choisit avec probabilités p
- Valeur terminale en bout de branche

**Exemple — choix d'assurance illustratif :**

Vous possédez un vélo de 500 €. Une assurance coûte 45 €/an. Probabilité de vol estimée : 10 %.

```
□ DÉCISION
├── Sans assurance
│   ○ Hasard
│   ├── Vol (p=0,10) → −500 €
│   └── Pas de vol (p=0,90) → 0 €
│   E = 0,10 × (−500) + 0,90 × 0 = −50 €
│
└── Avec assurance
    Coût certain = −45 €
    E = −45 €
```

**Espérance monétaire** : "Sans assurance" est légèrement meilleur (−50 € < −45 € en valeur absolue... attends :  −45 > −50, donc sans assurance est plus défavorable). Ici −45 > −50 : payer 45 € est *moins mauvais* que l'espérance sans assurance de −50 €.

Recalculons clairement :

```
E[sans assurance] = −50 €
E[avec assurance] = −45 €
```

L'espérance monétaire favorise l'assurance (−45 € > −50 €). Et si la perte soudaine de 500 € est très pénalisante pour votre utilité (concave), l'assurance est encore plus justifiée.

**Méthode de résolution d'un arbre :**
1. Calculer la valeur de chaque branche terminale.
2. Aux nœuds de hasard : calculer l'espérance (Σ p × valeur).
3. Aux nœuds de décision : choisir la branche à l'utilité espérée maximale.
4. Remonter l'arbre de droite à gauche (élagage).

> **À retenir** : L'arbre de décision force à *expliciter* les probabilités et valeurs — ce qui révèle souvent des hypothèses implicites qu'on n'avait pas conscientisées.

---

## 4. Le paradoxe d'Allais (1953)

Maurice Allais, prix Nobel d'économie 1988, a mis en évidence une violation systématique de l'utilité espérée.

**Situation 1** — choisissez entre :
- **1A** : 100 % de chance de gagner 1 000 000 €
- **1B** : 89 % de 1 000 000 € | 10 % de 5 000 000 € | 1 % de 0 €

La plupart des gens choisissent **1A** (la certitude).

**Situation 2** — choisissez entre :
- **2A** : 11 % de 1 000 000 € | 89 % de 0 €
- **2B** : 10 % de 5 000 000 € | 90 % de 0 €

La plupart des gens choisissent **2B**.

**La contradiction logique :** Si 1A > 1B, alors en soustrayant le même "89 % de 1 000 000 €" des deux options, on devrait obtenir 2A > 2B. Or les gens préfèrent 2B. C'est une violation de l'**axiome d'indépendance** de von Neumann-Morgenstern.

**Pourquoi ?** La certitude des 1 000 000 € dans 1A a une valeur émotionnelle spéciale — l'**effet de certitude**. Quand ce contexte de certitude disparaît (Situation 2), les préférences s'inversent.

**Résultat empirique :** Ce pattern a été répliqué de façon robuste dans de nombreuses cultures et contextes (Peterson, 2017 ; Kahneman & Tversky, 1979).

---

## 5. Le paradoxe d'Ellsberg (1961)

Daniel Ellsberg a distingué deux formes d'incertitude.

**Expérience :** Une urne contient 90 billes. 30 sont rouges. Les 60 restantes sont noires ou jaunes, dans une proportion inconnue.

**Situation 1** — choisissez :
- **A** : 100 € si rouge (p = 30/90 = 1/3 exactement)
- **B** : 100 € si noire (p inconnue, entre 0 et 2/3)

La plupart préfèrent **A** — la probabilité connue.

**Situation 2** — choisissez :
- **C** : 100 € si rouge ou jaune
- **D** : 100 € si noire ou jaune

La plupart préfèrent **D** — la probabilité connue (noire + jaune = 60/90 = 2/3 exactement).

**La contradiction :** Préférer A implique P(rouge) > P(noire). Mais préférer D implique P(noire ou jaune) > P(rouge ou jaune), soit P(noire) > P(rouge). Les deux ne peuvent pas être vraies simultanément.

**La leçon :** Les humains distinguent :
- **Risque** : probabilités connues (urne avec composition connue)
- **Ambiguïté** : probabilités inconnues (urne avec composition partiellement inconnue)

Et nous sommes systématiquement **averses à l'ambiguïté** — nous fuyons les situations où nous ne savons pas les probabilités, même quand l'espérance est identique.

> **À retenir** : L'aversion à l'ambiguïté explique de nombreux comportements : préférer une option familière dont on connaît les risques à une option potentiellement meilleure dont les risques sont flous.

---

## 6. Synthèse — quel outil pour quelle situation ?

| Outil | Quand l'utiliser | Limite principale |
|-------|-----------------|-------------------|
| Espérance mathématique | Décisions répétées, enjeux faibles par rapport à la richesse | Ignore la concavité de l'utilité |
| Utilité espérée | Décisions importantes à enjeux asymétriques | Difficile à calibrer ; viole Allais |
| Arbre de décision | Problèmes multi-étapes avec probabilités estimables | Nécessite d'expliciter les probabilités |

**Règle pratique en 3 étapes :**
1. Calculer l'espérance monétaire de chaque option.
2. Corriger pour l'aversion au risque : "préférerais-je cela si j'y avais droit à coup sûr, mais pour une somme moindre ?" — c'est l'équivalent certain.
3. Vérifier la cohérence en reformulant le problème différemment (test de robustesse contre le cadrage).

---

## Flash-cards (Module 07)

**Q1 : Calculez l'espérance d'un pari où vous gagnez 80 € avec probabilité 0,25 et perdez 20 € avec probabilité 0,75.**
> R : E = 0,25 × 80 + 0,75 × (−20) = 20 − 15 = **+5 €** (pari légèrement favorable).

**Q2 : Qu'est-ce que l'utilité espérée et pourquoi la distingue-t-on de l'espérance monétaire ?**
> R : L'utilité espérée pondère les résultats par leur valeur *subjective* U(x). Avec une courbe concave (aversion au risque), un gain certain de 300 € peut valoir plus qu'un pari d'espérance 350 €, car l'utilité marginale de l'argent décroît.

**Q3 : Dans le paradoxe d'Allais, quel axiome de la théorie de l'utilité espérée est violé ?**
> R : L'**axiome d'indépendance** : si l'on retire le même composant de deux loteries, les préférences ne devraient pas s'inverser. L'effet de certitude crée cette violation systématique.

**Q4 : Qu'est-ce que l'aversion à l'ambiguïté (paradoxe d'Ellsberg) ?**
> R : La tendance à fuir les paris sur des probabilités *inconnues* — même quand l'espérance est identique à un pari à probabilités connues. On distingue risque (probabilités connues) et ambiguïté (probabilités inconnues).

**Q5 : Comment résoudre un arbre de décision (ordre des étapes) ?**
> R : (1) Valoriser les branches terminales. (2) Aux nœuds de hasard, calculer Σ p × valeur. (3) Aux nœuds de décision, choisir la branche à utilité espérée maximale. (4) Remonter de droite à gauche jusqu'à la racine.

---

## Points clés à retenir

1. L'espérance mathématique (Σ pᵢ × xᵢ) est le point de départ — valide pour décisions répétées à faibles enjeux.
2. L'utilité espérée raffine en tenant compte que 1 € supplémentaire vaut moins quand on en possède déjà beaucoup (concavité).
3. L'aversion au risque est *rationnelle* : si la douleur de la perte dépasse le plaisir du gain équivalent, préférer le certain est cohérent.
4. Les arbres de décision forcent à expliciter probabilités et valeurs — et révèlent les hypothèses implicites.
5. Paradoxe d'Allais : la *certitude* a une valeur spéciale qui viole les axiomes de l'utilité espérée.
6. Paradoxe d'Ellsberg : nous traitons différemment le *risque* (probabilités connues) et l'*ambiguïté* (probabilités inconnues).

---

## Pour aller plus loin

- **Manuel de référence** : Peterson, M. (2017). *An Introduction to Decision Theory*, 2e éd. Cambridge University Press. https://www.cambridge.org/core/books/an-introduction-to-decision-theory/B9EEB3DCE5D0CAFFB6F3F30B1D0A06A6
- **Théorie normative** : Stanford Encyclopedia of Philosophy — *Normative Theories of Rational Choice: Expected Utility* (éd. hiver 2023). https://plato.stanford.edu/entries/rationality-normative-utility/
- **Alternatives à l'utilité espérée** : Stanford Encyclopedia — *Rivals to Expected Utility*. https://plato.stanford.edu/entries/rationality-normative-nonutility/
- **Théorie des perspectives (Prospect Theory)** : Kahneman, D. & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. *Econometrica*, 47(2), 263–291. [Lire l'article original pour comprendre pourquoi les pertes sont pondérées plus fortement que les gains.]
