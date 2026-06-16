# Module 05 — Décision sous Incertitude

> **Temps estimé** : 50 min | **Prérequis** : Modules 01-04

> **Objectif** : Maîtriser les outils fondamentaux de la décision rationnelle sous incertitude — espérance, utilité, arbres de décision — et comprendre pourquoi nous dévions systématiquement de ces modèles (paradoxes d'Allais et d'Ellsberg).

---

## 1. L'espérance mathématique : la boussole de base

L'**espérance mathématique** d'un choix est la somme des gains possibles pondérés par leurs probabilités.

**Formule** :

```
E[X] = Σ pᵢ × xᵢ
```

**Exemple — pari équitable** : un dé à 6 faces. Si vous gagnez 6 € sur le 6 et perdez 1 € sinon.

```
E = (1/6 × 6) + (5/6 × (−1)) = 1,00 − 0,833 = +0,167 €
```

Ce pari est légèrement favorable. Un joueur rationnel maximisant l'espérance l'accepte.

**Paradoxe de Saint-Pétersbourg** : on joue à pile ou face. Si pile au 1er lancer : 2 €. Si pile au 2e : 4 €. Au ke : 2^k €. L'espérance est infinie. Pourtant, presque personne ne paie plus de 20-30 € pour y jouer. Cela prouve que **maximiser l'espérance monétaire n'est pas toujours un modèle adéquat du comportement humain** — d'où le concept d'utilité.

---

## 2. Utilité et aversion au risque

**L'utilité** mesure la valeur *subjective* d'un résultat, pas sa valeur monétaire brute. Bernoulli (1738) proposa que l'utilité croît de façon logarithmique : chaque euro supplémentaire vaut moins quand on en possède déjà beaucoup.

**Aversion au risque** : la plupart des gens préfèrent un gain certain à un pari d'espérance identique (ou même légèrement supérieure). Exemple :

- **Option A** : 50 € certains.
- **Option B** : 50 % de chance de gagner 110 €, 50 % de chance de ne rien gagner. E[B] = 55 €.

La majorité choisit A, bien que B ait une espérance supérieure. La courbe d'utilité est **concave** : la douleur d'une perte est plus grande que le plaisir d'un gain équivalent (*aversion à la perte* de Kahneman et Tversky).

**Formule d'utilité espérée** (von Neumann-Morgenstern, 1944) :

```
EU = Σ pᵢ × U(xᵢ)
```

où U est la fonction d'utilité individuelle.

**Types de profils** :
- *Neutre au risque* : U(x) = x (utilité linéaire) → maximise l'espérance.
- *Averse au risque* : U(x) = log(x) ou √x → préfère la certitude.
- *Chercheur de risque* : U(x) = x² → préfère le pari.

---

## 3. Arbres de décision

Un **arbre de décision** structure visuellement les choix, leurs conséquences et leurs probabilités.

**Éléments** :
- □ Nœud de décision (carré) : choix sous contrôle de l'agent.
- ○ Nœud de hasard (cercle) : événements aléatoires.
- → Branche : outcome avec sa probabilité et sa valeur.

**Exemple — assurance illustrative** :

Vous possédez un vélo de 400 €. Une assurance coûte 40 €/an. Sans assurance, risque de vol à 8 % (perte totale).

```
                     ┌── Vol (0,08) → perte 400 €
Sans assurance ──────┤
                     └── Pas de vol (0,92) → 0 €

Espérance sans assurance : 0,08 × (−400) + 0,92 × 0 = −32 €

Avec assurance : coût certain = −40 €
```

L'espérance *monétaire* favorise "sans assurance" (−32 € > −40 €). Pourtant, si la perte de 400 € est pénible (votre utilité chute fortement), payer 40 € pour l'éviter peut être rationnel selon votre courbe d'utilité.

**Calcul de la valeur attendue** :
1. Pour chaque branche, multiplier la probabilité par la valeur.
2. Remonter les nœuds de hasard (sommer les espérances pondérées).
3. Aux nœuds de décision, choisir la branche à l'espérance maximale (ou à l'utilité espérée maximale).

---

## 4. Le paradoxe d'Allais (1953)

Le paradoxe d'Allais révèle que les préférences humaines violent les axiomes de l'utilité espérée.

**Situation 1** — choisir entre :
- **1A** : 100 % de chance de gagner 1 000 000 €
- **1B** : 89 % de chance de 1 000 000 €, 10 % de chance de 5 000 000 €, 1 % de chance de 0 €

La majorité préfère 1A.

**Situation 2** — choisir entre :
- **2A** : 11 % de chance de 1 000 000 €, 89 % de chance de 0 €
- **2B** : 10 % de chance de 5 000 000 €, 90 % de chance de 0 €

La majorité préfère 2B.

**La contradiction** : la préférence 1A > 1B et 2B > 2A violent simultanément les axiomes de von Neumann-Morgenstern. En soustrayant 89 % de chance de 1 000 000 € des deux options dans la situation 1, on obtient exactement la situation 2 — et les préférences s'inversent. Ce phénomène est répliqué robustement.

**Interprétation** : la certitude a une valeur spéciale (un "effet de certitude") que l'utilité espérée ne capture pas. Les théories de la décision alternatives (théorie des perspectives, Kahneman & Tversky 1979) tentent d'en rendre compte.

---

## 5. Le paradoxe d'Ellsberg (1961)

**Expérience** : une urne contient 90 billes. 30 sont rouges, les 60 restantes sont soit noires soit jaunes (proportion inconnue).

**Situation 1** — choisir entre :
- **A** : 100 € si on tire rouge.
- **B** : 100 € si on tire noire.

La majorité préfère A.

**Situation 2** — choisir entre :
- **C** : 100 € si on tire rouge ou jaune.
- **D** : 100 € si on tire noire ou jaune.

La majorité préfère D.

**La contradiction** : préférer A à B signifie qu'on pense P(rouge) > P(noire). Mais préférer D à C signifie P(noire ou jaune) > P(rouge ou jaune), ce qui implique P(noire) > P(rouge) — l'inverse. L'incohérence vient de l'**ambiguïté** : on fuit les paris sur des probabilités inconnues même quand les espérances sont comparables.

**Leçon** : les humains distinguent le risque (probabilités connues) de l'ambiguïté (probabilités inconnues), et sont *averse à l'ambiguïté*. Le modèle d'utilité espérée standard ignore cette distinction.

---

## 6. Que faire de tout cela ?

| Outil | Usage adapté | Limite |
|-------|--------------|--------|
| Espérance mathématique | Décisions répétées à enjeux faibles (paris, jeux) | Ne tient pas compte de l'aversion au risque |
| Utilité espérée | Décisions importantes à enjeux asymétriques | Difficile à calibrer, viole Allais en pratique |
| Arbre de décision | Structurer un problème complexe multi-étapes | Requiert d'estimer les probabilités |
| Théorie des perspectives | Modéliser les comportements réels | Descriptif, pas normatif |

**Règle pratique en 3 étapes** :
1. Lister les options et leurs conséquences possibles avec les probabilités.
2. Estimer les valeurs *après* avoir corrigé pour l'aversion au risque (poser la question : "préférerais-je la même chose en certitude réduite ?").
3. Vérifier la cohérence en reformulant dans le cadrage inverse.

---

> **À retenir** :
> - L'espérance mathématique est le point de départ, pas la fin : l'utilité subjective raffine le calcul.
> - L'aversion au risque est rationnelle si la douleur de la perte est supérieure au plaisir du gain équivalent.
> - Les paradoxes d'Allais et d'Ellsberg montrent que nous traitons différemment certitude, risque et ambiguïté — des résultats robustes et répliqués.

---

## Flash-cards (Module 05)

**Q1** : Calculez l'espérance d'un pari où vous gagnez 50 € avec probabilité 0,3 et perdez 10 € avec probabilité 0,7.
**R1** : E = 0,3 × 50 + 0,7 × (−10) = 15 − 7 = **+8 €** (pari favorable).

**Q2** : Qu'est-ce que l'utilité espérée et pourquoi la distingue-t-on de l'espérance monétaire ?
**R2** : L'utilité espérée pondère les outcomes par leur valeur *subjective* (U(x)), pas leur valeur nominale. Cela permet de modéliser l'aversion au risque : 100 € certains peut valoir plus qu'un pari d'espérance 110 € si la courbe d'utilité est concave.

**Q3** : Dans le paradoxe d'Allais, quelle axiome de l'utilité espérée est violé ?
**R3** : L'axiome d'indépendance : la préférence entre deux loteries ne devrait pas changer si on ajoute (ou retire) le même composant aux deux. L'effet de certitude crée une violation systématique.

**Q4** : Qu'est-ce que l'aversion à l'ambiguïté (paradoxe d'Ellsberg) ?
**R4** : La tendance à éviter les paris dont les probabilités sont inconnues (ambiguïté) même quand l'espérance est identique à un pari à probabilités connues. Résultat répliqué, non capturé par l'utilité espérée standard.

**Q5** : Comment construire un arbre de décision simple ?
**R5** : 1) Nœud de décision (□) pour chaque choix. 2) Nœuds de hasard (○) pour chaque événement aléatoire, avec probabilités sur les branches. 3) Valeurs en bout de branches. 4) Remonter en multipliant probabilités × valeurs, choisir le nœud de décision à espérance maximale.

---

## Points clés à retenir

1. L'espérance mathématique est la règle de base : sommer les gains pondérés par leurs probabilités.
2. L'utilité espérée raffine en tenant compte de la valeur subjective — la douleur d'une perte vaut plus que le plaisir d'un gain symétrique.
3. L'aversion au risque est rationelle si votre courbe d'utilité est concave (et elle l'est pour la plupart des gens).
4. Le paradoxe d'Allais : l'effet de certitude viole les axiomes de l'utilité espérée de façon robuste.
5. Le paradoxe d'Ellsberg : nous fuyons l'ambiguïté (probabilités inconnues) au-delà du risque (probabilités connues).

---

## Pour aller plus loin

- **Manuel de référence** : Peterson, M. (2017). *An Introduction to Decision Theory*, 2e éd. Cambridge University Press. https://www.cambridge.org/core/books/an-introduction-to-decision-theory/B9EEB3DCE5D0CAFFB6F3F30B1D0A06A6
- **Théorie normative** : Stanford Encyclopedia of Philosophy — *Normative Theories of Rational Choice: Expected Utility*. https://plato.stanford.edu/entries/rationality-normative-utility/
- **Alternatives** : Stanford Encyclopedia — *Rivals to Expected Utility*. https://plato.stanford.edu/entries/rationality-normative-nonutility/
- **Article fondateur des paradoxes** : Allais, M. (1953). Le comportement de l'homme rationnel devant le risque. *Econometrica*, 21(4), 503-546.
- **Théorie des perspectives** : Kahneman, D. & Tversky, A. (1979). Prospect Theory. *Econometrica*, 47(2), 263-291.
