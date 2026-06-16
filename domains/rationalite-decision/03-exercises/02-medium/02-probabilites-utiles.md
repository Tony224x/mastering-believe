# Exercices Medium — Module 02 : Probabilites utiles

> **Niveau** : Medium | **Temps estime** : ~35 min

---

## Exercice 1 : Tableau des frequences naturelles complet (VPP et VPN)

### Objectif

Construire integralement le tableau des frequences naturelles a partir de la sensibilite, de la specificite et de la prevalence, puis en deduire la valeur predictive positive (VPP) **et** la valeur predictive negative (VPN).

### Consigne

Un atelier utilise un **capteur de controle qualite** qui signale les pieces defectueuses. Le capteur a les caracteristiques suivantes :

- **Sensibilite** : 88 % (il detecte 88 % des pieces reellement defectueuses)
- **Specificite** : 92 % (il donne un verdict "conforme" correct pour 92 % des pieces saines)
- **Taux de base (prevalence des defauts)** dans le lot : **5 %**

Le lot inspecte comporte **2 000 pieces**. Calculez :

1. Le nombre de pieces reellement defectueuses et reellement conformes.
2. Le tableau complet en frequences naturelles : vrais positifs (VP), faux positifs (FP), vrais negatifs (VN), faux negatifs (FN).
3. La VPP : si le capteur signale une piece, quelle est la probabilite qu'elle soit reellement defectueuse ?
4. La VPN : si le capteur dit "conforme", quelle est la probabilite que la piece le soit reellement ?

*Conseil : le script `domains/rationalite-decision/02-code/02-probabilites-utiles.py` calcule VPP et VPN a partir de sensibilite/specificite/prevalence — utilisez-le pour verifier.*

### Criteres de reussite

- [ ] Le nombre de defectueuses (100) et de conformes (1 900) est calcule correctement
- [ ] Les 4 cellules du tableau sont correctes : VP = 88, FN = 12, VN = 1 748, FP = 152
- [ ] La VPP est calculee : VP / (VP + FP) = 88 / (88 + 152) = 88 / 240 ≈ **36,7 %**
- [ ] La VPN est calculee : VN / (VN + FN) = 1 748 / (1 748 + 12) = 1 748 / 1 760 ≈ **99,3 %**
- [ ] La lecture est faite : le capteur exclut tres bien les defauts (VPN 99,3 %) mais une alerte n'est juste que dans ~37 % des cas (effet du taux de base a 5 %)

---

## Exercice 2 : Depistage en deux etapes (mise a jour du taux de base)

### Objectif

Comprendre qu'un second test applique aux seuls positifs du premier remonte la VPP, parce que le **taux de base** a change apres la premiere etape.

### Consigne

Un service applique un **depistage en deux temps** sur une population a faible prevalence. Le meme test est utilise aux deux etapes (resultats supposes **independants**) :

- **Sensibilite** : 90 %
- **Specificite** : 85 %
- **Prevalence initiale (taux de base)** dans la population : **4 %**

Procedure : on teste tout le monde une premiere fois ; seuls les positifs du premier test passent un **second test** ; on ne retient comme "cas confirme" que ceux positifs **aux deux tests**.

Sur une population de **10 000 personnes**, calculez :

1. Apres le **premier test** : le tableau (VP, FP) et la VPP1 chez les positifs.
2. La nouvelle prevalence parmi les positifs du premier test (= VPP1) — c'est le taux de base de la seconde etape.
3. Apres le **second test** applique a ce sous-groupe : la VPP2 finale (probabilite d'etre reellement positif sachant deux tests positifs).
4. Le facteur d'amelioration de la VPP entre une etape et deux.

*Conseil : vous pouvez verifier la VPP1 et la VPP2 separement avec `02-code/02-probabilites-utiles.py` (la VPP2 = celle obtenue en relancant le calcul avec prevalence = VPP1).*

### Criteres de reussite

- [ ] Etape 1 : VP = 360, FP = 1 440, VPP1 = 360 / 1 800 = **20 %**
- [ ] Le sous-groupe positif compte 360 vrais et 1 440 faux ; sa prevalence est donc 20 % (= nouveau taux de base)
- [ ] Etape 2 sur ce sous-groupe : VP = 324, FP = 216, VPP2 = 324 / 540 = **60 %**
- [ ] La VPP passe de 20 % a 60 % : le second test triple la valeur predictive
- [ ] L'explication est nommee : ce n'est pas le test qui s'ameliore, c'est le **taux de base** qui est passe de 4 % a 20 % avant la seconde etape

---

## Exercice 3 : Demeler P(A|B) et P(B|A) sur un tableau de contingence

### Objectif

A partir d'un tableau de contingence en effectifs, calculer les deux probabilites conditionnelles inverses et nommer l'erreur de transposition.

### Consigne

Un atelier a inspecte **1 000 pieces**. Chaque piece est soit reellement **defectueuse**, soit **conforme** ; pour chacune, un capteur a soit **declenche une alarme**, soit non. Voici le tableau observe :

|              | Defectueuse | Conforme | Total |
|--------------|-------------|----------|-------|
| Alarme       | 45          | 90       | 135   |
| Pas d'alarme | 5           | 860      | 865   |
| Total        | 50          | 950      | 1 000 |

Calculez :

1. P(alarme | defectueuse) — parmi les pieces defectueuses, la proportion qui declenche l'alarme.
2. P(defectueuse | alarme) — parmi les pieces ayant declenche l'alarme, la proportion reellement defectueuse.
3. P(alarme) — la probabilite marginale d'une alarme.
4. L'erreur de transposition : un responsable affirme "le capteur s'allume pour 90 % des pieces defectueuses, donc quand il s'allume il y a 90 % de chances que la piece soit defectueuse". Expliquez pourquoi c'est faux et le role du taux de base.

### Criteres de reussite

- [ ] P(alarme | defectueuse) = 45 / 50 = **90 %**
- [ ] P(defectueuse | alarme) = 45 / 135 = **33,3 %**
- [ ] P(alarme) = 135 / 1 000 = **13,5 %**
- [ ] L'erreur de transposition est nommee : confondre P(alarme|defectueuse) = 90 % avec P(defectueuse|alarme) = 33,3 % (ce sont deux conditionnelles inverses, pas la meme quantite)
- [ ] Le role du taux de base est explicite : seules 5 % des pieces sont defectueuses, donc la masse des alarmes (90 sur 135) provient de pieces conformes (faux positifs)
