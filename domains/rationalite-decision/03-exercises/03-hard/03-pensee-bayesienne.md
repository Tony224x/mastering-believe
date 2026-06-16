# Exercices Hard — Module 03 : Pensee bayesienne

> **Niveau** : Hard | **Temps estime** : ~50 min

---

## Exercice 1 : Diagnostic en deux temps — une preuve pour, une preuve contre

### Objectif

Conduire une mise a jour multi-etapes a partir d'un **taux de base** (base rate), avec deux preuves **independantes** de forces opposees qui arrivent en sequence : une preuve qui soutient l'hypothese (LR > 1) puis une preuve qui la contredit (LR < 1). Comprendre concretement le principe "change ton avis proportionnellement a la preuve" — y compris en arriere.

### Consigne

Une condition abstraite (notee D) touche **2 % d'une population** : c'est le taux de base, donc le prior P(D) = 0,02. Aucune information individuelle n'est encore disponible.

**Etape 1 — premier examen (preuve POUR).** Un test depiste D. Sa sensibilite est de 95 % et son taux de faux positifs de 8 % :
- P(test+ | D) = 0,95
- P(test+ | non-D) = 0,08

Le test revient **positif**.

**Etape 2 — second examen independant (preuve CONTRE).** Un marqueur de confirmation, independant du premier test, est en general **present** chez les porteurs de D et **absent** chez les non-porteurs. Ici, le marqueur est **absent** :
- P(marqueur absent | D) = 0,30
- P(marqueur absent | non-D) = 0,85

**Travail demande** :
1. Calculez le LR du test positif, puis le posterior P(D | test+) en forme odds.
2. Calculez le LR du marqueur absent (il sera < 1 : c'est une preuve contre D).
3. En repartant du posterior de l'etape 1, calculez le posterior final P(D | test+, marqueur absent).
4. Verifiez que vous obtenez le meme resultat via le **LR combine** (produit des deux LR applique au prior initial).
5. Interpretez : commentez comment la preuve contraire "ramene en arriere" la croyance, et pourquoi le posterior final reste neanmoins superieur au taux de base.

### Criteres de reussite

- [ ] LR1 (test+) = 0,95 / 0,08 = **11,875**
- [ ] Odds prieur = 0,02 / 0,98 = 0,02041 ; Odds apres test+ = 0,02041 x 11,875 = 0,24235 ; P(D | test+) = **19,5 %**
- [ ] LR2 (marqueur absent) = 0,30 / 0,85 = **0,353** (preuve contre D, car < 1)
- [ ] Odds final = 0,24235 x 0,353 = 0,08553 ; P(D | test+, marqueur absent) = 0,08553 / 1,08553 = **7,9 %**
- [ ] LR combine = 11,875 x 0,353 = 4,191 ; Odds = 0,02041 x 4,191 = 0,08553 → meme **7,9 %**
- [ ] Interpretation : la preuve contraire fait chuter la croyance de 19,5 % a 7,9 % ; elle reste > 2 % (taux de base) car la preuve POUR etait plus forte que la preuve CONTRE
- [ ] Le sens de la mise a jour est symetrique : on revise a la hausse OU a la baisse selon le LR, jamais en ignorant une preuve

---

## Exercice 2 : Transfert — de quelle ligne de production vient ce lot ? (et critique du prior)

### Objectif

Mobiliser tout le pipeline bayesien sur un scenario decrit **en mots** : extraire soi-meme le prior et les vraisemblances, conduire la mise a jour sur une sequence d'observations, puis **critiquer le prior** (est-il ancre dans des donnees ? que se passe-t-il si on le change ?). Ce dernier point est l'antidote au relativisme : un bon prior se justifie par des donnees, pas par une preference.

### Consigne

Un entrepot recoit des lots de pieces provenant de **deux lignes de production** d'une meme usine :

- Ligne **L1** : ancienne, **taux de defaut de 6 %** par piece.
- Ligne **L2** : recente, **taux de defaut de 2 %** par piece.

L'historique de volume indique que **70 % des lots** proviennent de L1 et **30 %** de L2. On vous tend un lot **sans etiquette** et on vous demande de deviner sa ligne d'origine.

Vous inspectez **4 pieces** du lot, considerees independantes, et vous observez dans l'ordre : **defectueuse, conforme, defectueuse, conforme** (D, OK, D, OK).

**Travail demande** :
1. Identifiez explicitement le prior P(L1), P(L2) et les vraisemblances P(D|L1), P(OK|L1), P(D|L2), P(OK|L2) a partir de l'enonce.
2. Conduisez la mise a jour **sequentielle** sur les 4 observations (donnez les posteriors intermediaires de P(L1)).
3. Verifiez par le **calcul global** : vraisemblance de la sequence (2 defauts, 2 conformes) sous chaque ligne, ponderee par le prior.
4. **Critique du prior** : l'enonce ancre le prior 70/30 dans des **donnees de volume** — est-ce legitime ? Refaites le calcul final avec deux priors alternatifs : (a) 50/50 (aucune info de volume) et (b) 90/10. Commentez la sensibilite du resultat au prior et ce qui rendrait un prior plus ou moins defendable.

### Criteres de reussite

- [ ] Prior pose : P(L1) = 0,70 ; P(L2) = 0,30. Vraisemblances : P(D|L1) = 0,06, P(OK|L1) = 0,94, P(D|L2) = 0,02, P(OK|L2) = 0,98
- [ ] Apres piece 1 (D) : P(L1) = (0,06 x 0,70) / (0,06 x 0,70 + 0,02 x 0,30) = 0,042 / 0,048 = **87,5 %**
- [ ] Apres piece 2 (OK) : P(L1) = **87,0 %** (une piece conforme favorise legerement L2, taux de defaut plus bas)
- [ ] Apres piece 3 (D) : P(L1) = **95,3 %**
- [ ] Apres piece 4 (OK) : P(L1) = **95,1 %**
- [ ] Verification globale : L(seq|L1) = 0,06^2 x 0,94^2 = 0,003181 ; L(seq|L2) = 0,02^2 x 0,98^2 = 0,000384 ; P(L1) = (0,003181 x 0,70) / (0,003181 x 0,70 + 0,000384 x 0,30) = **95,1 %**
- [ ] Sensibilite au prior : avec 50/50 → P(L1) = **89,2 %** ; avec 90/10 → P(L1) = **98,7 %**
- [ ] Critique : le prior 70/30 est defendable car ancre dans des donnees de volume mesurees ; un prior 50/50 serait le choix "ignorant" par defaut ; le posterior reste > 89 % dans les trois cas, donc la conclusion (lot probablement de L1) est **robuste** au choix du prior. Un prior n'est pas une opinion libre : il doit etre justifie par une source

---

> **Rappel anti-relativisme** : la pensee bayesienne n'autorise pas a "croire ce qu'on veut". Les priors doivent etre ancres dans des donnees (taux de base, frequences mesurees), les vraisemblances proviennent de proprietes verifiables des tests/sources, et deux personnes honnetes partant de priors raisonnables convergent quand les preuves s'accumulent. La methode prime sur les conclusions.
