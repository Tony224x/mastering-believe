# Solutions Medium — Module 03 : Pensee bayesienne

*Les scripts Python peuvent etre utilises pour verifier les calculs :*
`python domains/rationalite-decision/02-code/03-pensee-bayesienne.py`

*Convention d'arrondi : valeurs intermediaires gardees a 4-5 decimales, resultats finaux arrondis a 1 decimale de pourcentage.*

---

## Exercice 1 : Mise a jour sequentielle sur 4 tirages, puis verification globale

**Donnees** :
- Urne A : P(rouge|A) = 0,75 ; P(bleue|A) = 0,25
- Urne B : P(rouge|B) = 0,40 ; P(bleue|B) = 0,60
- Prior : P(A) = P(B) = 0,50
- Observations : rouge, rouge, bleue, rouge

A chaque etape : P(A|obs) = L(obs|A) x P(A) / [L(obs|A) x P(A) + L(obs|B) x P(B)], puis ce posterior devient le prior suivant.

### Etape 1 — tirage rouge (prior P(A) = 0,50)

```
P(rouge) = 0,75 x 0,50 + 0,40 x 0,50 = 0,375 + 0,200 = 0,575
P(A | rouge) = 0,375 / 0,575 = 0,65217 = 65,2 %
```

### Etape 2 — tirage rouge (prior P(A) = 0,65217)

```
P(rouge) = 0,75 x 0,65217 + 0,40 x 0,34783 = 0,48913 + 0,13913 = 0,62826
P(A | 2e rouge) = (0,75 x 0,65217) / 0,62826 = 0,48913 / 0,62826 = 0,77855 = 77,9 %
```

### Etape 3 — tirage bleue (prior P(A) = 0,77855)

La bille bleue est une preuve **contre** A (l'urne A produit peu de bleues), donc le posterior baisse.

```
P(bleue) = 0,25 x 0,77855 + 0,60 x 0,22145 = 0,19464 + 0,13287 = 0,32751
P(A | bleue) = (0,25 x 0,77855) / 0,32751 = 0,19464 / 0,32751 = 0,59429 = 59,4 %
```

### Etape 4 — tirage rouge (prior P(A) = 0,59429)

```
P(rouge) = 0,75 x 0,59429 + 0,40 x 0,40571 = 0,44572 + 0,16228 = 0,60800
P(A | 4e rouge) = (0,75 x 0,59429) / 0,60800 = 0,44572 / 0,60800 = 0,73309 = 73,3 %
```

### Verification par calcul global (3 rouges, 1 bleue)

On calcule la vraisemblance de la sequence complete sous chaque hypothese :

```
L(3R,1B | A) = 0,75^3 x 0,25 = 0,421875 x 0,25 = 0,10547
L(3R,1B | B) = 0,40^3 x 0,60 = 0,064 x 0,60 = 0,0384

P(seq) = 0,10547 x 0,50 + 0,0384 x 0,50 = 0,052734 + 0,019200 = 0,071934
P(A | seq) = 0,052734 / 0,071934 = 0,73309 = 73,3 %
```

**Conclusion** : le sequentiel (73,3 %) et le global (73,3 %) coincident exactement. L'ordre d'arrivee des billes ne change pas le posterior final — la mise a jour bayesienne est commutative. La mise a jour sequentielle (reviser a chaque observation) et la mise a jour globale (tout d'un coup) sont deux chemins vers le meme resultat.

---

## Exercice 2 : Controle qualite en forme odds

**Donnees** :
- Prior : P(M) = 0,05 (machine mal calibree), P(non-M) = 0,95
- P(alarme | M) = 0,90 ; P(alarme | non-M) = 0,12

### 1) Rapport de vraisemblance

```
LR = P(alarme|M) / P(alarme|non-M) = 0,90 / 0,12 = 7,5
```

L'alarme est 7,5 fois plus probable quand la machine est mal calibree : c'est une preuve forte, mais le prior tres bas va temperer la conclusion.

### 2) Prior et posterieur en odds

```
Odds prieur = P(M) / P(non-M) = 0,05 / 0,95 = 0,05263
Odds posterieur = Odds prieur x LR = 0,05263 x 7,5 = 0,39474
```

### 3) Reconversion en probabilite

```
P(M | alarme) = Odds / (1 + Odds) = 0,39474 / 1,39474 = 0,28302 = 28,3 %
```

**Verification par la forme complete** :
```
P(alarme) = 0,90 x 0,05 + 0,12 x 0,95 = 0,045 + 0,114 = 0,159
P(M | alarme) = 0,045 / 0,159 = 0,28302 = 28,3 %  (identique)
```

### 4) Deuxieme alarme independante

Le posterior precedent (28,3 %) devient le nouveau prior. On reapplique le meme LR = 7,5.

```
Odds prieur (2) = 0,28302 / 0,71698 = 0,39474   (= l'ancien odds posterieur, coherent)
Odds posterieur (2) = 0,39474 x 7,5 = 2,96053
P(M | 2 alarmes) = 2,96053 / 3,96053 = 0,74751 = 74,8 %
```

**Interpretation** : avec un prior aussi faible (5 %), une **seule** alarme ne fait monter la croyance qu'a 28 % — il serait premature d'arreter la machine sur cette base. Une **deuxieme** alarme independante fait basculer la croyance a 75 %. La forme odds rend cela limpide : chaque preuve multiplie les odds par 7,5, l'effet est multiplicatif.

---

## Exercice 3 : Trois hypotheses concurrentes

**Donnees** :
- P(face | C1) = 0,5 ; P(face | C2) = 0,7 ; P(face | C3) = 0,9
- Prior non uniforme : P(C1) = 0,50 ; P(C2) = 0,30 ; P(C3) = 0,20
- Observation : face, face (FF)

Pour 3 hypotheses, on ne peut plus utiliser "non-H" : on calcule la valeur non normalisee de chaque hypothese, puis on divise par leur somme Z.

### 1) Vraisemblances de FF

```
L(FF | C1) = 0,5^2 = 0,25
L(FF | C2) = 0,7^2 = 0,49
L(FF | C3) = 0,9^2 = 0,81
```

### 2) Valeurs non normalisees (vraisemblance x prior)

```
C1 : 0,25 x 0,50 = 0,125
C2 : 0,49 x 0,30 = 0,147
C3 : 0,81 x 0,20 = 0,162
```

### 3) Normalisation

```
Z = 0,125 + 0,147 + 0,162 = 0,434

P(C1 | FF) = 0,125 / 0,434 = 0,2880 = 28,8 %
P(C2 | FF) = 0,147 / 0,434 = 0,3387 = 33,9 %
P(C3 | FF) = 0,162 / 0,434 = 0,3733 = 37,3 %

Verification : 0,2880 + 0,3387 + 0,3733 = 1,0000  (la somme fait bien 1)
```

### 4) Bonus — une 3e face

On repart des trois posteriors comme nouveaux priors, vraisemblance d'une face = P(face|Ck).

```
Non normalise :
  C1 : 0,5 x 0,2880 = 0,14401
  C2 : 0,7 x 0,3387 = 0,23710
  C3 : 0,9 x 0,3733 = 0,33594
Z' = 0,14401 + 0,23710 + 0,33594 = 0,71705

P(C1 | FFF) = 0,14401 / 0,71705 = 0,2008 = 20,1 %
P(C2 | FFF) = 0,23710 / 0,71705 = 0,3307 = 33,1 %
P(C3 | FFF) = 0,33594 / 0,71705 = 0,4685 = 46,9 %
```

**Interpretation** : au depart le prior favorisait nettement C1 (50 %). Apres seulement deux faces, C3 (la piece la plus biaisee) passe deja en tete (37,3 %), et apres une troisieme face son avance se creuse (46,9 %). La preuve deplace la croyance **proportionnellement a sa force** : chaque face est plus "attendue" sous C3 que sous C1, donc C3 gagne du poids relatif a chaque tirage. C'est exactement le mecanisme bayesien etendu a plusieurs hypotheses — la normalisation garantit que les probabilites restent une distribution coherente (somme = 1).
