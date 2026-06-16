# Exercices — Module 05 : Décision sous Incertitude

## Exercice 1 — Calcul d'espérance

### Objectif
Calculer l'espérance mathématique de plusieurs paris.

### Consigne
Pour chaque pari, calculez E et indiquez favorable (+), défavorable (−) ou neutre (0).

| Pari | Gain si succès | P(succès) | Mise (perte si échec) |
|------|------------|-----------|----------------|
| A | 10 € | 0,60 | 5 € |
| B | 100 € | 0,08 | 10 € |
| C | 50 € | 0,50 | 50 € |
| D | 200 € | 0,04 | 8 € |

1. Calculez E[A], E[B], E[C], E[D]. Formule : p × gain + (1−p) × (−mise).
2. Classez du plus favorable au moins favorable.
3. Un joueur maximisant l'espérance accepte-t-il B ? Et D ? Justifiez.

### Critères de réussite
- [ ] Les 4 espérances sont calculées correctement.
- [ ] Le classement est correct.
- [ ] La conclusion sur B et D est justifiée par les valeurs numériques.

---

## Exercice 2 — Arbre de décision : réparer ou remplacer ?

### Objectif
Construire un arbre de décision et calculer l'espérance de chaque branche.

### Consigne
Vélo en panne (valeur si réparé : 300 €).

**Option A — Réparer** (coût 80 €) :
- 70 % : tient 2 ans (valeur résiduelle 150 €)
- 30 % : retombe en panne dans le mois (valeur 0 €)

**Option B — Remplacer** : vélo reconditionné à 220 € (certitude).

1. Décrivez l'arbre (nœuds, branches, probabilités, valeurs).
2. Calculez la valeur nette espérée de A.
3. Calculez la valeur nette de B.
4. Quelle option maximise l'espérance ? Y a-t-il des raisons non monétaires ?

### Critères de réussite
- [ ] L'arbre est correctement structuré.
- [ ] Valeur nette A calculée correctement (= 105 − 80 = +25 €).
- [ ] Valeur nette B = −220 € (coût).
- [ ] La conclusion identifie l'option dominante et un facteur non monétaire.

---

## Exercice 3 — Paradoxe d'Allais : tester vos propres préférences

### Objectif
Expérimenter le paradoxe d'Allais et comprendre la violation de l'utilité espérée.

### Consigne

**Situation 1** — Choisissez (notez votre choix) :
- **1A** : 100 % de 1 000 €
- **1B** : 89 % de 1 000 €, 10 % de 5 000 €, 1 % de 0 €

**Situation 2** — Choisissez (notez votre choix) :
- **2A** : 11 % de 1 000 €, 89 % de 0 €
- **2B** : 10 % de 5 000 €, 90 % de 0 €

1. Calculez E[1A], E[1B], E[2A], E[2B].
2. Si vous avez choisi 1A et 2B, expliquez pourquoi cela viole les axiomes de l'utilité espérée.
3. Qu'est-ce que l'"effet de certitude" ?

### Critères de réussite
- [ ] Les 4 espérances sont calculées correctement.
- [ ] La contradiction logique est expliquée.
- [ ] L'effet de certitude est défini.
