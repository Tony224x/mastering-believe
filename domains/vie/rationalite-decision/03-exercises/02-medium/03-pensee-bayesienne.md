# Exercices Medium — Module 03 : Pensee bayesienne

> **Niveau** : Medium | **Temps estime** : ~35 min

---

## Exercice 1 : Mise a jour sequentielle sur 4 tirages, puis verification globale

### Objectif

Reviser un prior tirage apres tirage (chaque posterior devient le prior suivant), puis verifier que la mise a jour sequentielle donne exactement le meme resultat que la mise a jour globale.

### Consigne

On dispose d'une urne dont on ignore la composition. Deux hypotheses exclusives :

- Urne **A** : 75 % de billes rouges, 25 % de billes bleues — donc P(rouge|A) = 0,75, P(bleue|A) = 0,25
- Urne **B** : 40 % de billes rouges, 60 % de billes bleues — donc P(rouge|B) = 0,40, P(bleue|B) = 0,60

Au depart, les deux hypotheses sont equiprobables : P(A) = P(B) = 0,50.

On tire **4 billes avec remise** et on observe, dans l'ordre : **rouge, rouge, bleue, rouge**.

1. **Mise a jour sequentielle** : calculez P(A) apres chaque tirage. A chaque etape, le posterior obtenu devient le **prior** du tirage suivant. Donnez les 4 posteriors intermediaires.
2. **Mise a jour globale** : recalculez P(A) en une seule fois en utilisant la vraisemblance de la sequence complete (3 rouges, 1 bleue) sous chaque hypothese, sans passer par les etapes.
3. **Comparez** les deux resultats et tirez la conclusion sur l'ordre des observations.

### Criteres de reussite

- [ ] Apres tirage 1 (rouge) : P(A) = (0,75 x 0,50) / (0,75 x 0,50 + 0,40 x 0,50) = 0,375 / 0,575 = **65,2 %**
- [ ] Apres tirage 2 (rouge) : P(A) = **77,9 %**
- [ ] Apres tirage 3 (bleue) : P(A) = **59,4 %** (la bille bleue tire le posterior vers le bas)
- [ ] Apres tirage 4 (rouge) : P(A) = **73,3 %**
- [ ] Vraisemblances globales : L(3R,1B | A) = 0,75^3 x 0,25 = 0,10547 ; L(3R,1B | B) = 0,40^3 x 0,60 = 0,0384
- [ ] Calcul global : P(A) = (0,10547 x 0,50) / (0,10547 x 0,50 + 0,0384 x 0,50) = **73,3 %**
- [ ] Conclusion : sequentiel = global ; l'ordre d'arrivee des billes ne change pas le posterior final

---

## Exercice 2 : Controle qualite en forme odds (prior non-50 %, LR non-trivial)

### Objectif

Manipuler la **forme odds** de Bayes avec un prior faible et un rapport de vraisemblance eleve : Odds posterieur = Odds prieur x LR, puis reconversion en probabilite.

### Consigne

Dans un atelier, une machine peut etre **mal calibree** (hypothese M) ou bien calibree (non-M). D'apres l'historique de maintenance, une machine donnee est mal calibree dans **5 % des cas** au demarrage d'une serie : P(M) = 0,05.

Un capteur declenche une **alarme**. On sait que :

- quand la machine est mal calibree, l'alarme se declenche dans **90 %** des cas : P(alarme | M) = 0,90
- quand la machine est bien calibree, l'alarme se declenche quand meme (faux positif) dans **12 %** des cas : P(alarme | non-M) = 0,12

1. Calculez le rapport de vraisemblance LR de l'alarme.
2. Exprimez le prior en **odds**, puis calculez les **odds posterieurs** apres l'alarme.
3. Reconvertissez en probabilite P(M | alarme).
4. Une **deuxieme alarme independante** se declenche (meme capteur, memes vraisemblances). En repartant du posterior precedent comme nouveau prior, recalculez P(M | 2 alarmes) en forme odds.

### Criteres de reussite

- [ ] LR = P(alarme|M) / P(alarme|non-M) = 0,90 / 0,12 = **7,5**
- [ ] Odds prieur = 0,05 / 0,95 = **0,0526**
- [ ] Odds posterieur = 0,0526 x 7,5 = **0,3947**
- [ ] P(M | alarme) = 0,3947 / (1 + 0,3947) = **28,3 %** (verifie aussi par la forme complete : 0,90 x 0,05 / (0,90 x 0,05 + 0,12 x 0,95))
- [ ] Apres 2e alarme : Odds = 0,3947 x 7,5 = 2,9605 ; P(M | 2 alarmes) = 2,9605 / 3,9605 = **74,8 %**
- [ ] Interpretation : un prior de 5 % reste prudent — une seule alarme ne suffit pas a conclure ; deux alarmes font basculer la croyance

---

## Exercice 3 : Trois hypotheses concurrentes (normalisation sur 3 pieces)

### Objectif

Passer du cas binaire au cas a **trois hypotheses** : calculer les vraisemblances, ponderer par un prior **non uniforme**, puis **normaliser** le posterior pour que la somme des trois probabilites fasse 1.

### Consigne

On vous presente trois pieces de monnaie d'apparence identique, mais de biais differents quant a la probabilite de tomber sur **face** :

- Piece **C1** : equilibree, P(face | C1) = 0,5
- Piece **C2** : legerement biaisee, P(face | C2) = 0,7
- Piece **C3** : fortement biaisee, P(face | C3) = 0,9

On en choisit une au hasard, mais **pas uniformement** : les pieces equilibrees sont plus courantes dans le lot. Le prior est donc :

- P(C1) = 0,50, P(C2) = 0,30, P(C3) = 0,20

On lance la piece choisie **2 fois** et on obtient **face, face** (FF).

1. Calculez la vraisemblance de "FF" sous chacune des trois hypotheses.
2. Multipliez chaque vraisemblance par son prior pour obtenir les valeurs **non normalisees**.
3. Calculez la constante de normalisation Z (somme des trois) et **normalisez** : donnez P(C1|FF), P(C2|FF), P(C3|FF) et verifiez que leur somme vaut 1.
4. **Bonus** : on lance une 3e fois et on obtient encore **face**. Mettez a jour (sequentiellement) les trois posteriors.

### Criteres de reussite

- [ ] Vraisemblances : L(FF|C1) = 0,5^2 = 0,25 ; L(FF|C2) = 0,7^2 = 0,49 ; L(FF|C3) = 0,9^2 = 0,81
- [ ] Non normalise : C1 = 0,25 x 0,50 = 0,125 ; C2 = 0,49 x 0,30 = 0,147 ; C3 = 0,81 x 0,20 = 0,162
- [ ] Z = 0,125 + 0,147 + 0,162 = **0,434**
- [ ] Posteriors : P(C1|FF) = 0,125/0,434 = **28,8 %** ; P(C2|FF) = 0,147/0,434 = **33,9 %** ; P(C3|FF) = 0,162/0,434 = **37,3 %**
- [ ] Verification : 0,288 + 0,339 + 0,373 = **1,00**
- [ ] Bonus (3e face) : P(C1) ≈ **20,1 %** ; P(C2) ≈ **33,1 %** ; P(C3) ≈ **46,9 %**
- [ ] Interpretation : malgre un prior favorable a C1, deux faces suffisent a faire passer C3 (la plus biaisee) en tete — la preuve deplace la croyance proportionnellement a sa force
