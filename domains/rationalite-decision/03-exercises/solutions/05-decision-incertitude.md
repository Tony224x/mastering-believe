# Solutions — Module 05 : Décision sous Incertitude

---

## Solution Exercice 1 — Calcul d'espérance

**Formule** : E = p × gain + (1 − p) × (−mise)

**Pari A** :
```
E[A] = 0,60 × 10 + 0,40 × (−5)
     = 6,00 − 2,00
     = +4,00 €  ✓ favorable
```

**Pari B** :
```
E[B] = 0,08 × 100 + 0,92 × (−10)
     = 8,00 − 9,20
     = −1,20 €  ✗ défavorable
```

**Pari C** :
```
E[C] = 0,50 × 50 + 0,50 × (−50)
     = 25,00 − 25,00
     = 0,00 €  → neutre
```

**Pari D** :
```
E[D] = 0,04 × 200 + 0,96 × (−8)
     = 8,00 − 7,68
     = +0,32 €  ✓ favorable (légèrement)
```

**Classement du plus favorable au moins favorable** : A (+4,00 €) > D (+0,32 €) > C (0,00 €) > B (−1,20 €)

**Conclusion sur B et D** :
- **Pari B** : défavorable (−1,20 €). Un joueur maximisant l'espérance *refuse* ce pari, sauf s'il a une utilité particulière pour la chance de gagner 100 € (aversion au risque inverse pour les gros gains rares — cas du billet de loterie).
- **Pari D** : faiblement favorable (+0,32 €). Un joueur maximisant l'espérance *accepte* D. Sur 100 paris identiques, gain net espéré : +32 €. Mais l'écart-type est élevé (mise régulière de 8 € pour un gain rare de 200 €) — un joueur averse au risque pourrait refuser malgré une espérance positive.

---

## Solution Exercice 2 — Arbre de décision

**Structure de l'arbre** :

```
                          ┌── Tient 2 ans (0,70) → valeur 150 €
Réparation (−80 €) ──── ○ 
                          └── Retombe en panne (0,30) → valeur 0 €

Remplacement (−220 €) ── valeur 220 € stable (certitude)
```

**Option A — Valeur nette espérée** :
```
Valeur brute espérée après réparation :
  = 0,70 × 150 + 0,30 × 0
  = 105 + 0
  = 105 €

Valeur nette = valeur espérée − coût de réparation
             = 105 − 80
             = +25 €
```

L'option A génère une valeur nette espérée de **+25 €** (en conservant un vélo qui vaut en moyenne 105 € après réparation à un coût de 80 €).

**Option B — Valeur nette** :
Le vélo reconditionné coûte 220 € et vaut 220 €, donc valeur nette = 0 € (ni gain ni perte sur la valeur, mais vous avez un vélo fiable).

*Note* : la comparaison correcte est entre :
- **A** : dépenser 80 € et obtenir un bien qui vaut en espérance 105 € → gain net espéré +25 €.
- **B** : dépenser 220 € et obtenir un bien valant 220 € → échange équitable (valeur nette = 0 en termes de patrimoine).

**Option dominante** : A est mathématiquement supérieure (+25 € de gain net espéré vs 0 €).

**Facteurs non monétaires pouvant justifier B** :
- Aversion au risque : le scénario à 30 % (perte totale des 80 € + retomber en panne) est pénible si le vélo est nécessaire au quotidien.
- Valeur de la fiabilité certaine (un vélo garanti est plus pratique qu'un vélo incertain).
- Temps et contraintes : trouver un prestataire de réparation a un coût caché.

---

## Solution Exercice 3 — Paradoxe d'Allais

**Calcul des espérances** :

```
E[1A] = 1,00 × 1 000 = 1 000 €

E[1B] = 0,89 × 1 000 + 0,10 × 5 000 + 0,01 × 0
       = 890 + 500 + 0
       = 1 390 €

E[2A] = 0,11 × 1 000 + 0,89 × 0
       = 110 €

E[2B] = 0,10 × 5 000 + 0,90 × 0
       = 500 €
```

**Explication de la contradiction (pattern 1A > 1B et 2B > 2A)** :

Si on retire 89 % de chance de 1 000 € des deux options de la Situation 1 :
- 1A devient : 11 % de chance de 1 000 €, 89 % de chance de 0 € → c'est exactement 2A.
- 1B devient : 10 % de chance de 5 000 €, 1 % de chance de 0 €, 89 % de chance de 0 € → soit 10 % de 5 000 € et 90 % de 0 € → c'est exactement 2B.

Donc préférer 1A à 1B *et* préférer 2B à 2A revient à dire que la même composante (89 % de 1 000 €) a une valeur différente selon le contexte. C'est une violation de l'axiome d'indépendance : ajouter le même composant aux deux options ne devrait pas changer les préférences.

**L'effet de certitude** :
La certitude de 1A (100 %) a une valeur psychologique qui dépasse son poids mathématique. Passer de 89 % à 100 % n'est pas perçu comme "+11 points de probabilité" mais comme l'élimination totale de tout risque de perte — ce qui a une valeur affective spéciale. Cette "prime à la certitude" disparaît dans la Situation 2 (où aucune option n'est certaine), ce qui explique l'inversion des préférences.

Ce résultat est robuste et répliqué ; il a motivé le développement de la Théorie des Perspectives (Kahneman & Tversky, 1979) comme alternative descriptive à l'utilité espérée.
