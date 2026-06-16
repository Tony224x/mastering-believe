# Exercices Easy — Module 03 : Pensee bayesienne

---

## Exercice 1 : Mise a jour bayesienne simple

### Objectif

Appliquer la formule de Bayes pour mettre a jour une croyance a partir d'une seule observation.

### Consigne

Un club de sport organise un tournoi de badminton. Deux joueurs s'affrontent : Alice et Bruno. Historiquement, Alice gagne **60 % des matchs** contre Bruno.

Au debut du premier set, Alice enchaine 4 fautes consecutives, ce qui ne se produit que dans **10 %** de ses bonnes parties (quand elle va gagner) mais dans **60 %** de ses mauvaises parties (quand elle va perdre).

Calculez la probabilite qu'Alice gagne le match apres ces 4 fautes consecutives, en utilisant le theoreme de Bayes.

Posez clairement :
- L'hypothese H et son contraire non-H
- Le prior P(H)
- La vraisemblance P(E|H) et P(E|non-H)
- Le calcul du posterior P(H|E)

### Criteres de reussite

- [ ] H = "Alice gagne", P(H) = 0,60 est pose correctement
- [ ] P(E|H) = 0,10 et P(E|non-H) = 0,60 sont identifies correctement
- [ ] P(E) = 0,10 x 0,60 + 0,60 x 0,40 = 0,06 + 0,24 = 0,30 est calcule
- [ ] P(H|E) = 0,06 / 0,30 = **0,20 = 20 %** est le resultat final
- [ ] Interpretation : les 4 fautes ont fait chuter la probabilite de victoire d'Alice de 60 % a 20 %

---

## Exercice 2 : Mise a jour sequentielle

### Objectif

Comprendre comment les preuves s'accumulent de facon coherente dans le cadre bayesien.

### Consigne

Une boite contient des billes. On ne sait pas si c'est :
- Urne "Majorite rouge" (MR) : 80 % rouges, 20 % bleues
- Urne "Majorite bleue" (MB) : 30 % rouges, 70 % bleues

Au depart, les deux hypotheses sont equiprobables : P(MR) = P(MB) = 0,50.

On tire 3 billes avec remise, et on obtient dans l'ordre : rouge, rouge, bleue.

**Etape 1** : Calculez P(MR) apres la premiere bille rouge.
**Etape 2** : Utilisez ce posterior comme nouveau prior. Calculez P(MR) apres la deuxieme bille rouge.
**Etape 3** : Utilisez ce posterior comme nouveau prior. Calculez P(MR) apres la troisieme bille bleue.

Comparez le resultat final avec ce qu'on obtiendrait en calculant tout en une seule fois (en considerant l'evidence globale "2 rouges et 1 bleue").

### Criteres de reussite

- [ ] Etape 1 : P(MR | 1ere rouge) = (0,80 x 0,50) / (0,80 x 0,50 + 0,30 x 0,50) = 0,40 / 0,55 = **72,7 %**
- [ ] Etape 2 : P(MR | 2eme rouge) = **87,7 %**
- [ ] Etape 3 : P(MR | 1 bleue) = **67,1 %**
- [ ] La verification par calcul global donne le meme resultat (~67 %)
- [ ] Conclusion : la mise a jour sequentielle est equivalente a la mise a jour globale

---

## Exercice 3 : Evaluer la force d'une preuve avec le rapport de vraisemblance

### Objectif

Utiliser le rapport de vraisemblance pour comparer des preuves de force differente.

### Consigne

Deux tests pour detecter si un joueur de poker bluffe :

- **Test A** : quand un joueur bluffe, il tapote la table dans 70 % des cas. Quand il ne bluffe pas, il tapote dans 30 % des cas.
- **Test B** : quand un joueur bluffe, il evite le regard dans 60 % des cas. Quand il ne bluffe pas, il evite le regard dans 50 % des cas.

**Question 1** : Calculez le rapport de vraisemblance de chaque test.
**Question 2** : Lequel est le plus informatif ? Exprimez votre reponse quantitativement.
**Question 3** : Si votre prior sur le bluff est P(bluff) = 0,25, calculez le posterior apres avoir observe "tape la table" avec le Test A (utilisez la forme odds).

### Criteres de reussite

- [ ] LR(A) = 0,70 / 0,30 = **2,33** est calcule correctement
- [ ] LR(B) = 0,60 / 0,50 = **1,20** est calcule correctement
- [ ] Question 2 : Test A est plus informatif (LR plus eloigne de 1)
- [ ] Question 3 : Odds prior = 1/3 ; Odds posterior = (1/3) x 2,33 = 0,78 ; P(bluff|tape) = **43,7 %**
- [ ] Interpretation : l'observation augmente la probabilite de bluff de 25 % a ~44 %
