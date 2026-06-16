# Solutions Medium — Module 02 : Probabilites utiles

*Les calculs peuvent etre verifies avec le script :*
`python domains/rationalite-decision/02-code/02-probabilites-utiles.py`
*(il calcule VP, FP, VN, FN, VPP et VPN a partir de sensibilite/specificite/prevalence).*

*Arrondis : les effectifs tombent ici sur des entiers exacts ; les VPP/VPN sont arrondies a une decimale en pourcentage.*

---

## Exercice 1 — Tableau des frequences naturelles complet (VPP et VPN)

**Donnees** : 2 000 pieces, prevalence = 5 %, sensibilite = 88 %, specificite = 92 %.

**Etape 1 : effectifs de base**
```
Pieces defectueuses : 2 000 x 0,05 = 100
Pieces conformes    : 2 000 x 0,95 = 1 900
```

**Etape 2 : tableau en frequences naturelles**

|              | Defectueuse | Conforme | Total |
|--------------|-------------|----------|-------|
| Alarme (+)   | VP = 100 x 0,88 = **88** | FP = 1 900 x 0,08 = **152** | 240 |
| Conforme (-) | FN = 100 x 0,12 = **12** | VN = 1 900 x 0,92 = **1 748** | 1 760 |
| Total        | 100 | 1 900 | 2 000 |

(Rappel : FP = sains x (1 - specificite) = 1 900 x 0,08.)

**Etape 3 : VPP**
```
VPP = VP / (VP + FP) = 88 / (88 + 152) = 88 / 240 ≈ 0,3667 ≈ 36,7 %
```

**Etape 4 : VPN**
```
VPN = VN / (VN + FN) = 1 748 / (1 748 + 12) = 1 748 / 1 760 ≈ 0,9932 ≈ 99,3 %
```

**Lecture** : le capteur est excellent pour **rassurer** (une piece declaree conforme l'est a 99,3 %), mais une alarme n'est juste que dans ~37 % des cas. Avec une prevalence de seulement 5 %, la majorite des alarmes (152 sur 240) sont des faux positifs. C'est l'effet du taux de base, pas un defaut de sensibilite.

---

## Exercice 2 — Depistage en deux etapes (mise a jour du taux de base)

**Donnees** : sensibilite = 90 %, specificite = 85 %, prevalence initiale = 4 %, population = 10 000. Les deux tests sont supposes independants.

**Etape 1 : premier test sur 10 000 personnes**
```
Reellement positifs : 10 000 x 0,04 = 400
Reellement negatifs : 10 000 x 0,96 = 9 600

VP = 400 x 0,90 = 360
FP = 9 600 x (1 - 0,85) = 9 600 x 0,15 = 1 440

VPP1 = VP / (VP + FP) = 360 / (360 + 1 440) = 360 / 1 800 = 0,20 = 20 %
```

**Etape 2 : le sous-groupe positif devient la nouvelle population**

Parmi les 1 800 personnes positives au premier test :
- 360 sont reellement positives, 1 440 ne le sont pas.
- La prevalence de ce sous-groupe est donc **20 %** = VPP1. C'est le nouveau taux de base.

On applique le second test (memes caracteristiques) **a ces 1 800 personnes** :
```
VP = 360 x 0,90 = 324
FP = 1 440 x 0,15 = 216

VPP2 = VP / (VP + FP) = 324 / (324 + 216) = 324 / 540 = 0,60 = 60 %
```

**Verification par les rapports de vraisemblance (optionnel)**
```
Cote initiale  = 0,04 / 0,96 = 0,041667
LR+ d'un test  = sensibilite / (1 - specificite) = 0,90 / 0,15 = 6
Apres 1 test : 0,041667 x 6   = 0,25  -> P = 0,25 / 1,25 = 0,20  (= VPP1)
Apres 2 tests: 0,041667 x 36  = 1,50  -> P = 1,50 / 2,50 = 0,60  (= VPP2)
```

**Conclusion** : la VPP passe de **20 % a 60 %** — elle triple. Le test n'a pas change : c'est le **taux de base** qui est passe de 4 % a 20 % avant la seconde etape, parce qu'on a concentre l'effort sur un sous-groupe deja enrichi en vrais positifs. C'est le principe meme du depistage sequentiel.

---

## Exercice 3 — Demeler P(A|B) et P(B|A) sur un tableau de contingence

**Tableau observe** (1 000 pieces) :

|              | Defectueuse | Conforme | Total |
|--------------|-------------|----------|-------|
| Alarme       | 45          | 90       | 135   |
| Pas d'alarme | 5           | 860      | 865   |
| Total        | 50          | 950      | 1 000 |

**Etape 1 : P(alarme | defectueuse)** — on conditionne sur la colonne "Defectueuse" (50 pieces)
```
P(alarme | defectueuse) = 45 / 50 = 0,90 = 90 %
```

**Etape 2 : P(defectueuse | alarme)** — on conditionne sur la ligne "Alarme" (135 pieces)
```
P(defectueuse | alarme) = 45 / 135 ≈ 0,333 = 33,3 %
```

**Etape 3 : P(alarme)** — probabilite marginale
```
P(alarme) = 135 / 1 000 = 0,135 = 13,5 %
```

**Etape 4 : erreur de transposition**

L'affirmation "le capteur s'allume pour 90 % des defectueuses, donc une alarme = 90 % de chances de defaut" confond deux conditionnelles **inverses** :
- P(alarme | defectueuse) = **90 %** (ce que mesure le capteur)
- P(defectueuse | alarme) = **33,3 %** (ce qu'on veut reellement savoir)

C'est la **confusion de la transposition** (dite "erreur du procureur" en contexte judiciaire) : P(A|B) ≠ P(B|A).

Le **taux de base** explique l'ecart : seules 50 pieces sur 1 000 sont defectueuses (5 %). Comme les conformes sont 19 fois plus nombreuses, leurs faux positifs (90) noient les vrais positifs (45) parmi les alarmes. Meme un capteur tres sensible produit donc une majorite d'alarmes injustifiees quand l'evenement reel est rare.
