# Exercices — Module 05 : Décision sous Incertitude

---

## Exercice 1 — Calcul d'espérance : quel pari accepter ?

### Objectif
Calculer l'espérance mathématique de plusieurs paris et identifier les partenaires favorables.

### Consigne

Vous êtes face à 4 propositions de paris. Pour chacune, calculez l'espérance et indiquez si le pari est favorable (+), défavorable (−) ou neutre (0).

| Pari | Gain si succès | Probabilité de succès | Mise (perte si échec) |
|------|------------|-------------------|----------------|
| A | 10 € | 0,60 | 5 € |
| B | 100 € | 0,08 | 10 € |
| C | 50 € | 0,50 | 50 € |
| D | 200 € | 0,04 | 8 € |

**Questions** :
1. Calculez E[A], E[B], E[C], E[D].
2. Classez les paris du plus favorable au moins favorable.
3. Un joueur qui maximise l'espérance devrait-il accepter le pari B ? Et le pari D ? Justifiez.

### Critères de réussite
- [ ] Les 4 espérances sont calculées correctement (formule : p × gain + (1−p) × (−mise)).
- [ ] Le classement est correct.
- [ ] La conclusion sur B et D est justifiée par les valeurs numériques.

---

## Exercice 2 — Arbre de décision : réparer ou remplacer ?

### Objectif
Construire un arbre de décision simple et calculer l'espérance de chaque branche.

### Consigne

Votre vélo de ville (valeur estimée si réparé : 300 €) est tombé en panne. Deux options :

**Option A — Réparer** : coût de réparation 80 €. Après réparation :
- 70 % de probabilité que le vélo tienne 2 ans (valeur résiduelle : 150 €).
- 30 % de probabilité qu'il retombe en panne dans le mois (valeur résiduelle : 0 €, perte totale de la mise).

**Option B — Remplacer** : acheter un vélo reconditionné à 220 € (certitude, valeur stable).

**Questions** :
1. Dessinez (ou décrivez textuellement) l'arbre de décision avec les nœuds, branches, probabilités et valeurs finales.
2. Calculez la valeur nette espérée de l'option A (en tenant compte du coût de réparation).
3. Calculez la valeur nette de l'option B.
4. Quelle option maximise l'espérance ? Y a-t-il des raisons non monétaires qui pourraient justifier l'autre choix ?

### Critères de réussite
- [ ] L'arbre est correctement structuré (nœud de décision → nœuds de hasard → valeurs finales).
- [ ] Valeur nette espérée A = calculée correctement.
- [ ] Valeur nette B = −220 € (coût certain).
- [ ] La conclusion identifie l'option dominante et mentionne au moins un facteur non monétaire (aversion au risque, commodité, incertitude sur la durée de vie).

---

## Exercice 3 — Paradoxe d'Allais : tester vos propres préférences

### Objectif
Expérimenter le paradoxe d'Allais sur ses propres choix et comprendre pourquoi il viole l'utilité espérée.

### Consigne

**Situation 1** — Choisissez entre :
- **1A** : 100 % de chance de recevoir 1 000 €.
- **1B** : 89 % de chance de 1 000 €, 10 % de chance de 5 000 €, 1 % de chance de 0 €.

Notez votre choix : ________

**Situation 2** — Choisissez entre :
- **2A** : 11 % de chance de 1 000 €, 89 % de chance de 0 €.
- **2B** : 10 % de chance de 5 000 €, 90 % de chance de 0 €.

Notez votre choix : ________

**Questions** :
1. Calculez l'espérance de 1A, 1B, 2A et 2B.
2. Si vous avez choisi 1A et 2B (le pattern le plus fréquent), expliquez pourquoi cela viole les axiomes de l'utilité espérée.
3. Qu'est-ce que l'"effet de certitude" ? Comment expliquez-vous que la certitude de 1A la rende préférable même si 1B a une espérance supérieure ?

### Critères de réussite
- [ ] Les 4 espérances sont calculées correctement.
- [ ] La contradiction logique entre 1A > 1B et 2B > 2A est expliquée (en soustrayant 89 % de 1 000 €, on retrouve les mêmes options).
- [ ] L'effet de certitude est défini : la certitude a une valeur psychologique qui dépasse son poids mathématique.
