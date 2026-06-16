# Solutions — Module 06 : Indépendance financière et retraite anticipée

> Ces corrigés sont des réponses modèles. Pour les exercices de calcul, une tolérance de ± 5 % sur les chiffres est acceptable selon la méthode (annuelle vs mensuelle, arrondis).

---

## Solution Exercice 1 — Calculer son taux d'épargne et son horizon

### Yasmine

- Taux d'épargne : 450 / 2 800 × 100 = **16,1 %**
- Dépenses annuelles : 2 350 × 12 = 28 200 €
- Capital cible (règle des 4 %) : 28 200 × 25 = **705 000 €**
- Horizon approximatif : capital à constituer = 705 000 − 5 000 = 700 000 €

En utilisant la formule : 5 000 × (1,05)^n + (450 × 12) × [(1,05)^n − 1] / 0,05 = 705 000  
Par itération (ou simulateur) : environ **40 ans**

### Antoine

- Taux d'épargne : 2 300 / 4 200 × 100 = **54,8 %**
- Dépenses annuelles : 1 900 × 12 = 22 800 €
- Capital cible (règle des 4 %) : 22 800 × 25 = **570 000 €**
- Horizon approximatif : 30 000 × (1,05)^n + (2 300 × 12) × [(1,05)^n − 1] / 0,05 = 570 000  
Par itération : environ **16 ans**

### Double effet du taux d'épargne élevé d'Antoine

"Antoine épargne 3,4 fois plus que Yasmine chaque mois, ce qui accélère massivement l'accumulation. Mais le double effet est encore plus frappant : parce qu'il dépense moins (1 900 € vs 2 350 €/mois), son capital cible d'indépendance est lui aussi plus petit (570 000 € vs 705 000 €). Un taux d'épargne élevé comprime l'horizon par les deux bouts simultanément : on accumule plus vite ET on a besoin de moins. Résultat : Yasmine atteint l'objectif en ~40 ans, Antoine en ~16 ans — pour un revenu 50 % plus élevé certes, mais surtout grâce à des dépenses maîtrisées."

---

## Solution Exercice 2 — Analyser les limites de la règle des 4 %

### Partie A : Analyse des 3 profils

**Profil A — Camille, 63 ans, retraite à 65 ans, horizon 25 ans**
La règle des 4 % s'applique relativement bien ici. Elle a été calibrée pour un horizon de 30 ans, et 25 ans est légèrement plus court, donc le risque est même un peu plus faible. Elle se rapproche du profil type de Bengen. Avertissement : son portefeuille est diversifié US/Europe (le SPIVA est américain, les données européennes historiques peuvent légèrement varier).

**Profil B — Rémi, 38 ans, horizon 50+ ans, 100 % actions**
La règle des 4 % est inadaptée pour cet horizon. Elle a été calibrée pour 30 ans. Sur 50 ans, le risque d'épuisement du capital est significativement plus élevé, notamment à cause du risque de séquence des rendements. Par ailleurs, un portefeuille 100 % actions est plus volatile — en phase de décumulation (retraits réguliers), un krach dans les premières années est beaucoup plus dommageable.

**Profil C — Emma, 45 ans, indépendance partielle (60 % du portefeuille)**
La règle des 4 % doit être adaptée : Emma ne retire pas 4 % de son portefeuille total, mais seulement de quoi couvrir 40 % de ses dépenses (les 60 % restants viennent de son activité). Son capital cible est donc réduit (elle ne vise que 40 % × dépenses annuelles × 25). Le revenu d'activité partielle réduit aussi le risque de séquence des rendements (moins de retraits nécessaires en cas de krach).

### Partie B : Taux plus prudent pour Rémi

Pour un horizon de 50 ans, des chercheurs et la communauté Bogleheads recommandent généralement **3 % à 3,5 %** de taux de retrait.

Avec un taux de 3 % et des dépenses annuelles = 28 000 € :
- Capital cible = 28 000 / 0,03 = **933 333 €** (≈ 33 × les dépenses annuelles)

Avec un taux de 3,5 % :
- Capital cible = 28 000 / 0,035 = **800 000 €** (≈ 28,6 × les dépenses annuelles)

La règle des 4 % donnerait : 28 000 × 25 = 700 000 €. **Rémi doit viser 800 000 à 933 000 €**, soit 14 à 33 % de plus que la règle des 4 % standard.

### Partie C : 3 limites de la règle des 4 %

*Réponse modèle :*

"Premièrement, la règle des 4 % est fondée sur les données historiques des marchés américains — parmi les plus performants au monde au 20e siècle. Elle peut surestimer la soutenabilité pour des portefeuilles investis sur d'autres marchés ou pour un avenir différent du passé. Deuxièmement, elle a été calibrée pour un horizon de 30 ans : pour une indépendance à 40 ans visant 50 ans de retraite, le risque d'épuisement est nettement plus élevé et les chercheurs recommandent 3 à 3,5 % à la place. Troisièmement, le risque de séquence des rendements montre qu'un krach dans les premières années de la phase de retrait est disproportionnellement dommageable : on vend des actifs dépréciés pour vivre, réduisant le capital disponible pour le rebond. La règle des 4 % ne tient pas compte de cette asymétrie temporelle sans une flexibilité intégrée dans le plan."

---

## Solution Exercice 3 — Stratégie FIRE pour Nadia

### Partie A : Variante FIRE

La variante la plus adaptée est le **Barista FIRE**.

"Nadia ne vise pas l'indépendance financière totale mais la liberté de ne travailler qu'à mi-temps. Son portefeuille n'a pas besoin de couvrir toutes ses dépenses — seulement la partie non couverte par son activité partielle (1 400 €/mois). C'est exactement la définition du Barista FIRE : générer suffisamment de revenus du portefeuille pour compléter un revenu d'activité partielle, réduisant massivement le capital cible par rapport au FIRE classique."

### Partie B : Capital cible

- Dépenses couvertes par le portefeuille : 1 400 €/mois = 16 800 €/an
- Capital cible selon règle des 4 % : 16 800 × 25 = **420 000 €**

(Si horizon > 30 ans, appliquer 3,5 % → 16 800 / 0,035 = 480 000 €. Fourchette cible : 420 000 à 480 000 €.)

### Partie C : Horizon approximatif

Capital de départ : 0 €, versement mensuel 600 € (7 200 €/an), rendement net 5 %, cible 420 000 €.

0 × ... + 7 200 × [(1,05)^n − 1] / 0,05 = 420 000  
[(1,05)^n − 1] / 0,05 = 420 000 / 7 200 = 58,33  
(1,05)^n = 58,33 × 0,05 + 1 = 3,917  
n = ln(3,917) / ln(1,05) = 1,366 / 0,0488 ≈ **28 ans**

> Nadia atteindrait son objectif Barista FIRE à environ 58 ans. Pour atteindre à 45 ans, il faudrait augmenter les versements à ~1 400-1 500 €/mois.

### Partie D : 2 risques et adaptations

**Risque 1 — Le risque de séquence des rendements au début du barista FIRE**
Si un krach survient juste quand Nadia passe à mi-temps et commence à retirer, elle vend des actifs dépréciés pour vivre. Adaptation : constituer une réserve de liquidités de 12 à 24 mois de dépenses résiduelles (1 400 €/mois × 18 = 25 000 €) hors du portefeuille investi, pour ne pas avoir à vendre en période de baisse.

**Risque 2 — La durée imprévue (longévité)**
Si Nadia est en bonne santé à 70 ans, son horizon de décumulation dépasse 30 ans. La règle des 4 % peut insuffisamment couvrir une vie à 80 ou 90 ans. Adaptation : prévoir un taux de retrait de 3,5 % (capital cible ajusté à ~480 000 €) et maintenir la capacité d'ajuster les dépenses ou de reprendre une activité légère si les marchés sous-performent durablement.
