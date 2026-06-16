# Exercices Easy — Module 02 : Probabilites utiles

---

## Exercice 1 : Calculer un taux de vrais positifs

### Objectif

Construire le tableau des frequences naturelles et calculer la valeur predictive positive (VPP).

### Consigne

Un laboratoire developpe un test de detection de doping dans le sport. Le test a les caracteristiques suivantes :
- **Sensibilite** : 92 % (il detecte 92 % des athletes qui dopent reellement)
- **Specificite** : 97 % (il donne un negatif correct pour 97 % des athletes sains)
- **Prevalence reelle du dopage** dans la population testee : **3 %**

Sur une competition de **500 athletes**, calculez :

1. Le nombre d'athletes reellement dopes et non dopes.
2. Le tableau complet (vrais positifs, faux positifs, vrais negatifs, faux negatifs).
3. La valeur predictive positive (VPP) : si un athlete est teste positif, quelle est la probabilite qu'il soit reellement dope ?
4. La valeur predictive negative (VPN) : si un athlete est teste negatif, quelle est la probabilite qu'il soit reellement sain ?

### Criteres de reussite

- [ ] Le nombre de dopes (15) et de non-dopes (485) est calcule correctement
- [ ] Les 4 cellules du tableau sont correctes : VP ≈ 14, FP ≈ 15, VN ≈ 470, FN ≈ 1
- [ ] La VPP est calculee : VP / (VP + FP) = 14 / (14 + 15) ≈ **48 %**
- [ ] La VPN est calculee : VN / (VN + FN) = 470 / (470 + 1) ≈ **99,8 %**
- [ ] La conclusion contre-intuitive est nommee : meme avec un bon test, la VPP est proche de 50 % car la prevalence est faible

---

## Exercice 2 : Probabilite conditionnelle et piege de la transposition

### Objectif

Distinguer P(A|B) de P(B|A) et calculer les deux dans un contexte concret.

### Consigne

Dans une ville, les statistiques meteo des 10 dernieres annees montrent que :
- Il a plu **30 %** des jours (P(pluie) = 0,30)
- Quand il pleut, le ciel est couvert dans **90 %** des cas (P(couvert | pluie) = 0,90)
- Quand il ne pleut pas, le ciel est couvert dans **40 %** des cas (P(couvert | pas de pluie) = 0,40)

Vous regardez par la fenetre un matin : le ciel est couvert. Calculez :

1. La probabilite que le ciel soit couvert (P(couvert)) — denomminateur de Bayes.
2. La probabilite qu'il pleuve sachant que le ciel est couvert — P(pluie | couvert).
3. L'erreur de transposition : si quelqu'un affirme "le ciel est couvert dans 90 % des jours de pluie, donc quand c'est couvert il y a 90 % de chance de pluie", expliquez le probleme.

### Criteres de reussite

- [ ] P(couvert) = 0,30 × 0,90 + 0,70 × 0,40 = 0,27 + 0,28 = **0,55** est calcule correctement
- [ ] P(pluie | couvert) = (0,90 × 0,30) / 0,55 = 0,27 / 0,55 ≈ **49 %** est calcule correctement
- [ ] L'erreur de transposition est expliquee : P(couvert|pluie) ≠ P(pluie|couvert)
- [ ] Le role du taux de base P(pluie) = 30 % dans la difference est explique

---

## Exercice 3 : Impact du taux de base sur la VPP

### Objectif

Comprendre experimentalement comment la prevalence modifie l'interpretation d'un test, a caracteristiques fixes.

### Consigne

Utilisez un test fixe avec **sensibilite = 95 %** et **specificite = 95 %**. Calculez la VPP (sur 10 000 personnes) pour les trois prevalences suivantes :

- **Scenario 1** : prevalence = 50 % (maladie tres commune)
- **Scenario 2** : prevalence = 10 % (maladie moderement repandue)
- **Scenario 3** : prevalence = 0,5 % (maladie rare)

Pour chaque scenario :
1. Construisez le tableau (VP, FP, VN, FN).
2. Calculez la VPP.
3. Concluez : dans quel scenario le test est-il le plus "utile" en termes de valeur predictive positive ?

*Conseil : le script `02-code/02-probabilites-utiles.py` peut verifier vos calculs.*

### Criteres de reussite

- [ ] Scenario 1 : VPP ≈ 95 % (calculee correctement)
- [ ] Scenario 2 : VPP ≈ 68 % (calculee correctement)
- [ ] Scenario 3 : VPP ≈ 8,7 % (calculee correctement)
- [ ] La conclusion est claire : la VPP diminue drastiquement quand la prevalence baisse, meme avec un test excellent
- [ ] L'implication pratique est nommee : il ne faut pas tester tout le monde pour une maladie rare sans stratification du risque
