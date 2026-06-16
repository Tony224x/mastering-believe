# Solutions — Module 03 : Pensee bayesienne

*Les scripts Python peuvent etre utilises pour verifier les calculs :*
`python domains/rationalite-decision/02-code/03-pensee-bayesienne.py`

---

## Exercice 1 : Mise a jour bayesienne (Alice et Bruno)

**Posons le probleme** :
- H = "Alice gagne le match"
- E = "4 fautes consecutives en debut de set"

**Prior** :
- P(H) = 0,60 (Alice gagne 60 % des matchs contre Bruno historiquement)
- P(non-H) = 0,40

**Vraisemblances** :
- P(E | H) = 0,10 (4 fautes consecutives dans 10 % des victoires d'Alice)
- P(E | non-H) = 0,60 (4 fautes dans 60 % des defaites d'Alice)

**Calcul de P(E) — probabilite totale** :
```
P(E) = P(E|H) x P(H) + P(E|non-H) x P(non-H)
     = 0,10 x 0,60 + 0,60 x 0,40
     = 0,06 + 0,24
     = 0,30
```

**Posterior** :
```
P(H|E) = P(E|H) x P(H) / P(E)
        = 0,10 x 0,60 / 0,30
        = 0,06 / 0,30
        = 0,20 = 20 %
```

**Interpretation** : les 4 fautes consecutives constituent une preuve forte contre la victoire d'Alice (rapport de vraisemblance = 0,10 / 0,60 = 1/6 — la preuve est 6 fois moins probable dans ses bonnes parties). La probabilite de victoire chute de 60 % a 20 %.

---

## Exercice 2 : Mise a jour sequentielle (urnes)

**Donnees** : P(MR) = P(MB) = 0,50 initialement.
- P(rouge|MR) = 0,80, P(bleue|MR) = 0,20
- P(rouge|MB) = 0,30, P(bleue|MB) = 0,70

### Etape 1 : apres la 1ere bille rouge

Prior : P(MR) = 0,50

```
P(rouge) = 0,80 x 0,50 + 0,30 x 0,50 = 0,40 + 0,15 = 0,55

P(MR | rouge) = (0,80 x 0,50) / 0,55 = 0,40 / 0,55 = 72,7 %
```

### Etape 2 : apres la 2eme bille rouge

Prior : P(MR) = 0,727

```
P(rouge) = 0,80 x 0,727 + 0,30 x 0,273 = 0,5816 + 0,0819 = 0,6635

P(MR | 2eme rouge) = (0,80 x 0,727) / 0,6635 = 0,5816 / 0,6635 = 87,7 %
```

### Etape 3 : apres la bille bleue

Prior : P(MR) = 0,877

```
P(bleue) = 0,20 x 0,877 + 0,70 x 0,123 = 0,1754 + 0,0861 = 0,2615

P(MR | bleue) = (0,20 x 0,877) / 0,2615 = 0,1754 / 0,2615 = 67,1 %
```

### Verification par calcul global (2 rouges, 1 bleue)

```
P(2R1B | MR) = 0,80 x 0,80 x 0,20 = 0,128
P(2R1B | MB) = 0,30 x 0,30 x 0,70 = 0,063

P(2R1B) = 0,128 x 0,50 + 0,063 x 0,50 = 0,0955

P(MR | 2R1B) = (0,128 x 0,50) / 0,0955 = 0,064 / 0,0955 = 67 %
```

**Verification** : les deux methodes donnent ~67 %. La mise a jour sequentielle est exactement equivalente a la mise a jour globale. L'ordre d'arrivee des preuves ne change pas le resultat final.

---

## Exercice 3 : Rapport de vraisemblance (poker)

### Question 1 : Rapports de vraisemblance

```
LR(A) = P(tape|bluff) / P(tape|pas bluff) = 0,70 / 0,30 = 2,33

LR(B) = P(evite regard|bluff) / P(evite regard|pas bluff) = 0,60 / 0,50 = 1,20
```

### Question 2 : Quel test est plus informatif ?

Le **Test A** est plus informatif : LR = 2,33 vs LR = 1,20.

- LR(A) = 2,33 : observer "tape la table" rend le bluff 2,33 fois plus probable
- LR(B) = 1,20 : observer "evite le regard" ne fait que multiplier les odds par 1,20 — effet tres faible

Un LR plus eloigne de 1 indique une preuve plus forte.

### Question 3 : Posterior apres "tape la table" (Test A)

**Forme odds** :

```
Prior : P(bluff) = 0,25, P(pas bluff) = 0,75
Odds prior = 0,25 / 0,75 = 1/3 = 0,333

Odds posterior = Odds prior x LR(A) = 0,333 x 2,33 = 0,777
```

**Reconversion en probabilite** :
```
P(bluff | tape) = 0,777 / (1 + 0,777) = 0,777 / 1,777 = 43,7 %
```

**Interpretation** : l'observation "tape la table" augmente la probabilite de bluff de 25 % (prior) a ~44 % (posterior). C'est une hausse significative (+19 points) mais loin de la certitude. Il faudrait combiner plusieurs observations pour etre tres confident.
