# Module 02 — Probabilites utiles en 45 minutes

> **Temps estime** : 45 min | **Prerequis** : Module 01
> **Objectif** : maitriser les 4 outils probabilistes les plus rentables au quotidien — frequences, probabilite conditionnelle, taux de base, faux positifs.

---

## 1. Commencer par un exemple concret

> Un test de depistage d'une maladie a une **sensibilite de 90 %** (il detecte 90 % des malades) et une **specificite de 95 %** (il donne un resultat negatif pour 95 % des personnes saines). La maladie touche **1 % de la population**. Vous testez positif. Quelle est la probabilite que vous soyez reellement malade ?

Reponse intuitive frequente : "90 %" ou "95 %".
Reponse correcte : **environ 15 %**.

Ce resultat surprend. Il illustre pourquoi ignorer le **taux de base** (prevalence de 1 %) conduit a une conclusion radicalement fausse. On va construire ce calcul pas a pas.

---

## 2. Frequences naturelles : penser en effectifs

**Astuce de Gigerenzer** : transformer les pourcentages en effectifs concrets facilite le raisonnement.

Au lieu de penser "probabilite de 1 %", imaginez **1 000 personnes** testees :
- 10 sont malades (1 % de 1 000)
- 990 sont saines

Pour les 10 malades, le test detecte 90 % → **9 vrais positifs** (test + et malade), 1 faux negatif.
Pour les 990 saines, le test est negatif dans 95 % des cas → 940,5 vrais negatifs, **49,5 faux positifs** (disons 50).

Total de personnes testees positif : 9 + 50 = **59**.
Parmi ces 59, combien sont vraiment malades ? **9**.
Probabilite post-test : 9 / 59 ≈ **15 %**.

```
Tableau des frequences (sur 1 000 personnes, prevalence 1 %) :

                    Malade    Sain      Total
Test positif          9        50         59
Test negatif          1       940        941
Total                10       990       1000

Probabilite post-test positive = 9 / 59 ≈ 15 %
```

> **A retenir** : quand la prevalence est faible, meme un bon test produit majoritairement des faux positifs. Connaitre le taux de base change completement l'interpretation du resultat.

---

## 3. Les 4 concepts cles

### 3.1 Probabilite simple et frequence

La **probabilite** d'un evenement A, notee P(A), est comprise entre 0 et 1 :
- P(A) = 0 → impossible
- P(A) = 1 → certain
- P(A) = 0,5 → une chance sur deux (pile ou face equitable)

**Frequence relative** = nombre d'occurrences / nombre total d'essais. Pour de grands echantillons, la frequence relative converge vers la probabilite theorique.

Exemple : un de non truque, 600 lancers → on s'attend a ≈ 100 fois chaque face (probabilite 1/6 ≈ 16,7 %).

### 3.2 Probabilite conditionnelle

**P(A | B)** se lit "probabilite de A sachant que B s'est produit".

Formule : `P(A | B) = P(A et B) / P(B)`

Exemple concret :
- P(test positif | malade) = sensibilite = 90 % = 0,90
- P(test positif | sain) = 1 - specificite = 5 % = 0,05
- Ces deux probabilites sont tres differentes de P(malade | test positif) — c'est le piege classique.

**Erreur frequente** : confondre P(A|B) et P(B|A). On l'appelle la "confusion de la transposition" ou "erreur du procureur" (en contexte judiciaire).

### 3.3 Taux de base (base rate)

Le **taux de base** est la prevalence naturelle d'un evenement dans la population, avant tout test ou information supplementaire.

- Prevalence d'une maladie rare : 1 % → taux de base = 0,01
- Proportion de taxis verts dans la ville : 85 % → taux de base = 0,85
- Proportion de vols annules un lundi pluvieux : 12 % → taux de base = 0,12

**Le taux de base doit toujours etre le point de depart.** L'oublier est l'une des erreurs les plus courantes et les plus consequentes du raisonnement quotidien.

### 3.4 Faux positifs, vrais positifs, valeur predictive

Dans tout systeme de detection (test medical, filtre anti-spam, detecteur de fraude) :

| | Realite : Positif (malade) | Realite : Negatif (sain) |
|---|---|---|
| **Test : Positif** | Vrai positif (VP) | Faux positif (FP) |
| **Test : Negatif** | Faux negatif (FN) | Vrai negatif (VN) |

- **Sensibilite** = VP / (VP + FN) — capacite a detecter les vrais cas (taux de detection)
- **Specificite** = VN / (VN + FP) — capacite a rejeter les vrais negatifs
- **Valeur predictive positive (VPP)** = VP / (VP + FP) — c'est ce qu'on veut savoir apres un test positif

La VPP depend du taux de base : meme avec sensibilite = specificite = 99 %, une maladie a 0,1 % de prevalence donne une VPP d'environ 9 % seulement.

---

## 4. La formule de Bayes (premiere approche)

On peut calculer la VPP directement sans tableau, avec la formule de Bayes :

```
P(malade | test+) = [P(test+ | malade) × P(malade)] / P(test+)

P(test+) = P(test+ | malade) × P(malade) + P(test+ | sain) × P(sain)
```

Avec nos chiffres :
```
P(malade)  = 0,01  (taux de base)
P(sain)    = 0,99

P(test+ | malade) = 0,90  (sensibilite)
P(test+ | sain)   = 0,05  (1 - specificite)

P(test+) = 0,90 × 0,01 + 0,05 × 0,99 = 0,009 + 0,0495 = 0,0585

P(malade | test+) = 0,009 / 0,0585 ≈ 0,154 ≈ 15 %
```

Le module 03 approfondira cette formule et son extension a plusieurs mises a jour successives.

> **A retenir** : la probabilite post-test n'est pas la meme que la sensibilite du test. Elle depend du taux de base. Quand la prevalence est faible, meme un bon test produit principalement des faux positifs.

---

## 5. Exercice rapide : un test de qualite

> Une usine produit des pieces dont **2 %** sont defectueuses. Un capteur detecte les pieces defectueuses avec une **sensibilite de 95 %** et une **specificite de 90 %**.
> Une piece est signalee defectueuse par le capteur. Quelle est la probabilite qu'elle le soit vraiment ?

*Calculez avant de lire la solution.*

Sur 1 000 pieces :
- 20 defectueuses (2 %)
- 980 bonnes

Pieces signalees defectueuses :
- 20 × 0,95 = 19 vrais positifs
- 980 × 0,10 = 98 faux positifs
Total : 117 signalements

VPP = 19 / 117 ≈ **16 %**

Meme ici, avec une prevalence de seulement 2 %, la majorite des alertes sont des faux positifs. Connaitre ce chiffre evite de rejeter 84 % de la production inutilement.

---

## Flash-cards (5)

**Q1** : Quelle est la formule de la valeur predictive positive (VPP) ?
**R1** : VPP = Vrais positifs / (Vrais positifs + Faux positifs). C'est la proportion de cas vraiment positifs parmi tous les tests positifs.

**Q2** : Que signifie "sensibilite = 80 %" pour un test ?
**R2** : Sur 100 personnes reellement malades, le test en detecte 80 correctement (vrais positifs) et en rate 20 (faux negatifs).

**Q3** : Pourquoi le taux de base est-il crucial pour interpreter un test ?
**R3** : Parce qu'une maladie rare produit inevitablement beaucoup de faux positifs meme avec un bon test : la majorite des "positifs" sont des personnes saines. Ignorer le taux de base surestime enormement la probabilite d'etre malade.

**Q4** : Quelle est la difference entre P(test+ | malade) et P(malade | test+) ?
**R4** : P(test+ | malade) est la sensibilite du test. P(malade | test+) est la VPP — ce qu'on veut vraiment savoir apres un resultat positif. Ces deux valeurs peuvent etre tres differentes.

**Q5** : Comment la methode des frequences naturelles (Gigerenzer) facilite-t-elle le calcul ?
**R5** : En remplacant les pourcentages par des effectifs concrets ("sur 1 000 personnes, 10 sont malades"). Le tableau a 4 cases devient intuitif et les erreurs diminuent significativement.

---

## Points cles a retenir

- Penser en frequences naturelles (effectifs) plutot qu'en pourcentages reduit les erreurs de raisonnement.
- Le taux de base est le point de depart obligatoire de tout raisonnement probabiliste.
- Sensibilite et VPP sont deux choses differentes : la confusion entre les deux est une erreur frequente et consequente.
- Quand la prevalence est faible, meme un excellent test produit majoritairement des faux positifs.
- Le script `02-code/02-probabilites-utiles.py` permet de calculer automatiquement VPP et VPN pour n'importe quelle combinaison sensibilite/specificite/prevalence.

---

## Pour aller plus loin

- **Peterson, M.** (2017). *An Introduction to Decision Theory* (2e ed.). Cambridge University Press. https://www.cambridge.org/core/books/an-introduction-to-decision-theory/B9EEB3DCE5D0CAFFB6F3F30B1D0A06A6 — Chapitre 4 : probabilite conditionnelle et cadre bayesien.
- **Gigerenzer, G. & Hoffrage, U.** (1995). How to improve Bayesian reasoning without instruction: Frequency formats. *Psychological Review* 102(4):684-704. — Article originel sur les frequences naturelles.
- **Script interactif** : `02-code/02-probabilites-utiles.py` — calculateur de taux de base / faux positifs avec demonstrations chiffrees.
