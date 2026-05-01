# Exercices Faciles — Jour 13 : Emergent abilities & reasoning

---

## Exercice 1 : Ecrire un CoT sur un probleme piege

### Objectif

Savoir construire un raisonnement step-by-step qui explicite chaque etape, et comprendre pourquoi ca aide.

### Consigne

Pour chacun des problemes suivants :
1. Donner la reponse "directe" qu'un modele pourrait donner (souvent fausse)
2. Ecrire le CoT complet (4-6 etapes) qui arrive a la bonne reponse

**Probleme 1** : Dans une piece, il y a 5 personnes. Chacune a 2 bras. Combien de bras y a-t-il ?

**Probleme 2** : Marie a 3 fois plus de pommes que Pierre. Ensemble ils ont 16 pommes. Combien en a chacun ?

**Probleme 3** : Un train part de Paris a 8h et roule a 100 km/h. Un autre part de Lyon a 9h et roule a 120 km/h vers Paris. Paris-Lyon = 460 km. A quelle heure se croisent-ils ? (plus dur)

**Probleme 4** (piege linguistique) : "La balle qui coute 1.10 euro est 1 euro plus chere que la batte. Combien coute la batte ?" (classique de Kahneman — beaucoup se trompent en disant 0.10 euro)

Pour chaque probleme :
- Reponse directe plausible mais potentiellement fausse
- CoT avec les etapes numerotees
- Reponse finale

### Criteres de reussite

- [ ] P1 : direct "10 bras" (correct, trivial). CoT : 5 personnes × 2 bras = 10 bras
- [ ] P2 : direct "12 et 4" ou "4 et 12" (correct en fait). CoT explicit : soit p = pommes Pierre, 3p + p = 16, p = 4, Marie = 12
- [ ] P3 : direct impossible a deviner, CoT necessaire. Reponse : ~11h04 (depend de l'interpretation)
- [ ] P4 : direct "0.10 euro" (FAUX). CoT : batte + balle = 1.10, batte = balle + 1.00, donc batte = (1.10 - 1.00) / 2 + 1.00 = 1.05, balle = 0.05
- [ ] Tu comprends pourquoi le CoT aide : il force a poser les equations et eviter les raccourcis intuitifs

---

## Exercice 2 : Self-consistency a la main

### Objectif

Comprendre comment la majority vote ameliore la fiabilite d'un reasoner imparfait.

### Consigne

Soit un LLM imaginaire qui a 60% de chance de donner la bonne reponse sur une question donnee. Les 40% d'erreurs sont reparties uniformement sur plein de reponses differentes.

1. **Probabilite 1 sample** : quelle est la probabilite d'obtenir la bonne reponse avec 1 appel ?

2. **Probabilite 3 samples, majority vote** : approximer la probabilite que la majorite de 3 samples soit correcte. 
   - Majorite correcte si >= 2 corrects sur 3
   - P(>=2 corrects) = C(3,2) * 0.6^2 * 0.4 + C(3,3) * 0.6^3 = ?

3. **Probabilite 5 samples** : meme raisonnement pour 5 samples.
   - P(>=3 corrects) = C(5,3)*0.6^3*0.4^2 + C(5,4)*0.6^4*0.4 + C(5,5)*0.6^5 = ?

4. **Comparer** : combien de samples faut-il pour atteindre 95% de fiabilite ?

5. **Attention** : cette formule suppose que les samples sont independants. En pratique, un LLM a des biais systematiques qui rendent certains samples correles. L'amelioration reelle est souvent un peu moins.

6. **Cout** : si 1 sample coute 1 cent, combien coute self-consistency avec 5 samples ? Avec 20 samples ? Quand ca vaut la peine ?

### Criteres de reussite

- [ ] 1 sample : 60%
- [ ] 3 samples : 3 * 0.36 * 0.4 + 0.216 = 0.432 + 0.216 = 0.648 (65%)
- [ ] 5 samples : 0.3456 + 0.2592 + 0.0778 = 0.6826 (68%)
- [ ] En fait pour depasser 95% il faut ~25+ samples. Le gain est non-lineaire.
- [ ] Comprehension : 1 sample = 1 cent, 20 samples = 20 cents. Vaut la peine quand la bonne reponse est critique (math, code, medical)

---

## Exercice 3 : Identifier quand CoT aide vs quand ca ne sert a rien

### Objectif

Developper l'intuition sur quel type de probleme beneficie du CoT.

### Consigne

Pour chaque tache suivante, indiquer :
- **CoT aide** (+) ou **CoT n'aide pas** (-)
- **Pourquoi** (en 1-2 lignes)

1. "Quelle est la capitale de la France ?"

2. "Calculer la somme de 1 + 2 + 3 + ... + 100"

3. "Reconnaitre un sentiment dans une phrase : 'Ce film etait super ennuyeux'"

4. "Quelle est la 7eme lettre de l'alphabet ?"

5. "Ecrire une fonction Python qui trie une liste par ordre croissant"

6. "Une personne achete 3 pommes a 1 euro, 2 oranges a 1.5 euro, et une banane a 0.8 euro. Elle paie avec un billet de 10 euros. Combien de monnaie recoit-elle ?"

7. "Ecrire un haiku sur l'automne"

8. "Debuguer le code : `def sum(l): for x in l: s = x; return s`"

9. "Lister les 5 plus gros pays d'Europe par population"

10. "Resoudre : si 2x + 5 = 15, trouver x"

### Criteres de reussite

- [ ] CoT n'aide pas : Q1, Q3, Q4, Q7, Q9 (faits directs ou generation creative)
- [ ] CoT aide : Q2, Q6, Q10 (calcul multi-etapes)
- [ ] CoT aide parfois : Q5, Q8 (decomposition du probleme)
- [ ] Tu comprends la regle : CoT aide quand le probleme necessite plusieurs inferences consecutives ou que le modele doit "poser sa reflexion" pour eviter les raccourcis
- [ ] Corollaire : CoT ne sert a rien si la reponse est un fait que le modele connait ou ne connait pas
