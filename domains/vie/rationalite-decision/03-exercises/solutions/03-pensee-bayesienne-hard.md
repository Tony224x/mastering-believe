# Solutions Hard — Module 03 : Pensee bayesienne

*Les scripts Python peuvent etre utilises pour verifier les calculs :*
`python domains/vie/rationalite-decision/02-code/03-pensee-bayesienne.py`

*Convention d'arrondi : odds et valeurs intermediaires gardes a 4-5 decimales, resultats finaux arrondis a 1 decimale de pourcentage.*

---

## Exercice 1 : Diagnostic en deux temps — une preuve pour, une preuve contre

**Donnees** :
- Taux de base : P(D) = 0,02 ; P(non-D) = 0,98
- Test : P(test+ | D) = 0,95 ; P(test+ | non-D) = 0,08
- Marqueur de confirmation : P(absent | D) = 0,30 ; P(absent | non-D) = 0,85

On travaille en **forme odds** : Odds posterieur = Odds prieur x LR.

### 1) Etape 1 — test positif (preuve POUR D)

```
LR1 = P(test+|D) / P(test+|non-D) = 0,95 / 0,08 = 11,875

Odds prieur = P(D) / P(non-D) = 0,02 / 0,98 = 0,020408
Odds apres test+ = 0,020408 x 11,875 = 0,24235

P(D | test+) = 0,24235 / (1 + 0,24235) = 0,24235 / 1,24235 = 0,19507 = 19,5 %
```

**Verification par la forme complete** :
```
P(test+) = 0,95 x 0,02 + 0,08 x 0,98 = 0,019 + 0,0784 = 0,0974
P(D | test+) = 0,019 / 0,0974 = 0,19507 = 19,5 %  (identique)
```

Le taux de base de 2 % est si bas que meme un test sensible (95 %) ne fait monter la probabilite qu'a ~20 % : c'est l'effet du **taux de base** (base rate fallacy si on l'oublie).

### 2) Etape 2 — marqueur absent (preuve CONTRE D)

```
LR2 = P(absent|D) / P(absent|non-D) = 0,30 / 0,85 = 0,35294
```

LR2 < 1 : c'est une preuve **contre** D (l'absence du marqueur est plus probable chez les non-porteurs).

### 3) Posterior final

On repart de l'odds de l'etape 1 (0,24235) :

```
Odds final = 0,24235 x 0,35294 = 0,08553
P(D | test+, marqueur absent) = 0,08553 / 1,08553 = 0,07879 = 7,9 %
```

### 4) Verification par le LR combine

Les deux preuves etant independantes, on peut multiplier les LR et les appliquer au prior initial — on doit retrouver le meme resultat :

```
LR combine = LR1 x LR2 = 11,875 x 0,35294 = 4,1912
Odds final = Odds prieur x LR combine = 0,020408 x 4,1912 = 0,08553
P final = 0,08553 / 1,08553 = 7,9 %  (identique a l'etape 3)
```

### 5) Interpretation

La croyance a evolue ainsi : **2 % (base) → 19,5 % (apres test+) → 7,9 % (apres marqueur absent)**.

- La preuve contraire (marqueur absent) **ramene la croyance en arriere** : on ne reste pas accroche au 19,5 % juste parce qu'on l'avait "monte" auparavant. C'est le principe **"change ton avis proportionnellement a la preuve"**, et il fonctionne dans les deux sens.
- Le posterior final (7,9 %) reste **superieur au taux de base** (2 %) car la preuve POUR (LR1 = 11,875) etait plus forte que la preuve CONTRE (LR2 = 0,353) : le LR combine 4,19 reste > 1. La balance penche encore legerement vers D, mais beaucoup moins qu'apres le seul test positif.
- Lecon : ne jamais traiter une preuve isolement. Le posterior integre **toutes** les preuves disponibles ; ignorer la preuve contraire (biais de confirmation) aurait laisse a tort la croyance a 19,5 %.

---

## Exercice 2 : Transfert — de quelle ligne de production vient ce lot ?

### 1) Extraction du prior et des vraisemblances

L'enonce decrit le scenario en mots ; on traduit :

```
Prior (part de volume des lots) :
  P(L1) = 0,70   P(L2) = 0,30

Vraisemblances (taux de defaut par piece) :
  P(D|L1)  = 0,06   P(OK|L1) = 0,94
  P(D|L2)  = 0,02   P(OK|L2) = 0,98
```

Observation : sequence D, OK, D, OK (pieces independantes).

### 2) Mise a jour sequentielle

A chaque etape : P(L1|obs) = L(obs|L1) x P(L1) / [L(obs|L1) x P(L1) + L(obs|L2) x P(L2)].

**Piece 1 — D (prior P(L1) = 0,70)**
```
P(D) = 0,06 x 0,70 + 0,02 x 0,30 = 0,042 + 0,006 = 0,048
P(L1 | D) = 0,042 / 0,048 = 0,87500 = 87,5 %
```

**Piece 2 — OK (prior P(L1) = 0,87500)**
```
P(OK) = 0,94 x 0,87500 + 0,98 x 0,12500 = 0,82250 + 0,12250 = 0,94500
P(L1 | OK) = 0,82250 / 0,94500 = 0,87037 = 87,0 %
```
Une piece conforme est legerement plus probable sous L2 (moins de defauts), donc P(L1) baisse tres legerement.

**Piece 3 — D (prior P(L1) = 0,87037)**
```
P(D) = 0,06 x 0,87037 + 0,02 x 0,12963 = 0,052222 + 0,002593 = 0,054815
P(L1 | D) = 0,052222 / 0,054815 = 0,95270 = 95,3 %
```

**Piece 4 — OK (prior P(L1) = 0,95270)**
```
P(OK) = 0,94 x 0,95270 + 0,98 x 0,04730 = 0,895538 + 0,046354 = 0,941892
P(L1 | OK) = 0,895538 / 0,941892 = 0,95079 = 95,1 %
```

Posterior final : **P(L1) ≈ 95,1 %**.

### 3) Verification par calcul global (2 defauts, 2 conformes)

```
L(seq|L1) = 0,06^2 x 0,94^2 = 0,0036 x 0,8836 = 0,00318096
L(seq|L2) = 0,02^2 x 0,98^2 = 0,0004 x 0,9604 = 0,00038416

P(seq) = 0,00318096 x 0,70 + 0,00038416 x 0,30
       = 0,00222667 + 0,00011525 = 0,00234192
P(L1 | seq) = 0,00222667 / 0,00234192 = 0,95079 = 95,1 %   (identique au sequentiel)
```

### 4) Critique du prior et analyse de sensibilite

On refait **uniquement** le calcul global final en changeant le prior, vraisemblances inchangees :

**(a) Prior 50/50 (aucune information de volume)**
```
P(L1) = (0,00318096 x 0,50) / (0,00318096 x 0,50 + 0,00038416 x 0,50)
      = 0,00159048 / 0,00178256 = 0,89225 = 89,2 %
```

**(b) Prior 90/10**
```
P(L1) = (0,00318096 x 0,90) / (0,00318096 x 0,90 + 0,00038416 x 0,10)
      = 0,00286286 / 0,00290128 = 0,98675 = 98,7 %
```

**Synthese de la sensibilite** :

| Prior P(L1) | Posterior P(L1) |
|-------------|-----------------|
| 50 %        | 89,2 %          |
| 70 % (enonce) | 95,1 %        |
| 90 %        | 98,7 %          |

**Critique** :
- Le prior 70/30 de l'enonce est **defendable** : il est ancre dans une donnee mesuree (la part de volume reelle des deux lignes), pas dans une intuition. C'est ce qui distingue un prior bayesien legitime d'une opinion arbitraire.
- Le prior 50/50 n'est pas "neutre" par defaut — c'est le choix de l'**ignorance** (principe d'indifference). Il est honnete quand on n'a aucune information, mais ici on **a** une information de volume, donc l'ignorer reviendrait a jeter une donnee utile.
- Point cle anti-relativisme : meme en faisant varier le prior de 50 % a 90 %, le posterior reste **toujours superieur a 89 %**. La conclusion "ce lot vient probablement de L1" est donc **robuste** au choix du prior, parce que les donnees (deux defauts sur quatre, alors que L1 a un taux de defaut 3 fois plus eleve) parlent fort. Quand la preuve est forte, les priors raisonnables convergent — un prior n'est pas un permis de croire ce qu'on veut, et il doit toujours pouvoir etre justifie par une source.
