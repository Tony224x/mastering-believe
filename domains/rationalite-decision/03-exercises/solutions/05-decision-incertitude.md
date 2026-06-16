# Solutions — Module 05 : Décision sous Incertitude

## Exercice 1 — Calcul d'espérance

Formule : E = p × gain + (1−p) × (−mise)

- **E[A]** = 0,60 × 10 + 0,40 × (−5) = 6 − 2 = **+4 €** (favorable)
- **E[B]** = 0,08 × 100 + 0,92 × (−10) = 8 − 9,2 = **−1,2 €** (défavorable)
- **E[C]** = 0,50 × 50 + 0,50 × (−50) = 25 − 25 = **0 €** (neutre)
- **E[D]** = 0,04 × 200 + 0,96 × (−8) = 8 − 7,68 = **+0,32 €** (favorable)

**Classement** : A (+4 €) > D (+0,32 €) > C (0 €) > B (−1,2 €)

**Décision** : un joueur maximisant l'espérance accepte A et D (positifs), refuse B (négatif). C est indifférent.

---

## Exercice 2 — Arbre de décision

**Option A — Réparer** :
- Valeur espérée brute = 0,70 × 150 + 0,30 × 0 = 105 €
- Valeur nette = 105 − 80 (coût réparation) = **+25 €**

**Option B — Remplacer** :
- Coût certain = −220 €

**Option dominante** : A (réparer), car +25 € > −220 €.

**Facteur non monétaire** : si la fiabilité compte (déplacement quotidien, dépendance au vélo), le risque résiduel de 30 % de repanne dans le mois peut justifier de choisir B malgré l'espérance inférieure.

---

## Exercice 3 — Paradoxe d'Allais

**Espérances** :
- E[1A] = 1,00 × 1 000 = **1 000 €**
- E[1B] = 0,89 × 1 000 + 0,10 × 5 000 + 0,01 × 0 = 890 + 500 = **1 390 €**
- E[2A] = 0,11 × 1 000 + 0,89 × 0 = **110 €**
- E[2B] = 0,10 × 5 000 + 0,90 × 0 = **500 €**

**Violation** : choisir 1A (espérance 1 000 €) sur 1B (1 390 €) puis 2B (500 €) sur 2A (110 €) est incohérent. En retirant la composante commune "89 % de 1 000 €" des deux options de S1, on obtient S2 — et les préférences s'inversent. C'est une violation directe de l'axiome d'indépendance.

**Effet de certitude** : la garantie à 100 % de S1A a une valeur psychologique supplémentaire non capturée par l'espérance. Les individus surpondèrent la certitude absolue, même quand elle est moins avantageuse en espérance.
