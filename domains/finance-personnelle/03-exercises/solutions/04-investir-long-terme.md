# Solutions — Module 04 : Investir simplement et sur le long terme

> ⚠️ Ces solutions sont des corrigés modèles. Vos chiffres peuvent différer légèrement selon la méthode de calcul (annuelle vs mensuelle). L'important est la démarche et l'ordre de grandeur.

---

## Solution Exercice 1 — L'impact des frais sur 20 ans

### Partie A : Calcul des capitaux finals

**Données** : capital initial 5 000 €, versement mensuel 150 €, rendement brut 7 %, durée 20 ans.

Rendement net Scénario 1 : 7,00 % − 0,20 % = **6,80 %** par an  
Rendement net Scénario 2 : 7,00 % − 1,80 % = **5,20 %** par an

*Méthode de calcul (formule de capitalisation annuelle avec versements) :*
```
Capital final = Capital_initial × (1 + r)^n 
              + Versement_annuel × [(1 + r)^n - 1] / r
```
Où versement annuel = 150 € × 12 = 1 800 €/an

**Scénario 1 (r = 0,068, n = 20) :**
- 5 000 × (1,068)^20 = 5 000 × 3,741 ≈ 18 705 €
- 1 800 × [(1,068)^20 − 1] / 0,068 = 1 800 × 40,31 ≈ 72 558 €
- **Total ≈ 91 263 €**

**Scénario 2 (r = 0,052, n = 20) :**
- 5 000 × (1,052)^20 = 5 000 × 2,765 ≈ 13 825 €
- 1 800 × [(1,052)^20 − 1] / 0,052 = 1 800 × 34,34 ≈ 61 812 €
- **Total ≈ 75 637 €**

*Note : le simulateur Python (`02-code/04-investir-long-terme.py`) utilise la capitalisation mensuelle et donne des résultats légèrement différents (plus précis), dans la même fourchette.*

### Partie B : Différence

Différence : 91 263 − 75 637 = **15 626 €**  
Réduction en % : (91 263 − 75 637) / 91 263 × 100 ≈ **17,1 %**

> Un écart de seulement 1,60 point de frais par an réduit le capital final de 17 % sur 20 ans.

### Partie C : Argumentation SPIVA/Sharpe

*Exemple de réponse modèle :*

"Selon le rapport SPIVA Year-End 2024, environ 65 % des fonds actifs large-cap US sous-performent leur indice sur 1 an, et environ 92 % sur 20 ans. L'arithmétique de Sharpe (1991) explique pourquoi : avant frais, l'ensemble des gérants actifs représente exactement le marché et ne peut donc obtenir en moyenne que le rendement du marché. Après frais, ils sont en moyenne perdants face à l'indice. Payer 1,80 % de frais annuels pour une gestion active qui, dans 9 cas sur 10 sur 20 ans, sous-performera l'indice net de frais, est statistiquement un mauvais pari. Le bas coût des fonds indiciels est un avantage structurel, pas une question de chance."

---

## Solution Exercice 2 — Lire et comparer des fonds

### Question 1 : Impact des frais sur 10 ans pour 10 000 €, rendement brut 8 %

Rendement net Fonds A : 8 % − 0,12 % = 7,88 %  
Rendement net Fonds B : 8 % − 2,10 % = 5,90 %

Fonds A : 10 000 × (1,0788)^10 = 10 000 × 2,109 ≈ **21 090 €**  
Fonds B : 10 000 × (1,0590)^10 = 10 000 × 1,779 ≈ **17 790 €**

Différence : **3 300 €** (soit 15,6 % de capital en moins pour le Fonds B, uniquement à cause des frais).

### Question 2 : Évaluation de la performance du Fonds B

Le Fonds B a fait +8,2 %/an sur 5 ans. C'est une performance brute. Il manque :
- La performance **nette de frais** : 8,2 % − 2,10 % = **6,1 %** net (vs. MSCI World à +9,1 %)
- La **période complète** : 5 ans est court ; les krachs à 1-2 ans font basculer les classements
- La **persistance** : les gérants qui surperforment sur 5 ans sont-ils les mêmes sur 10 ans ? (Réponse statistique : rarement)
- Le **risque ajusté** : a-t-il pris plus de risque pour obtenir ce résultat ?

Conclusion : le Fonds B sous-performe le MSCI World **net de frais** sur 5 ans. C'est le critère pertinent.

### Question 3 : Choix argumenté

*Réponse modèle :*

"Je choisirais le Fonds A (indiciel MSCI World, TER 0,12 %). Il offre une diversification immédiate sur ~1 500 entreprises de 23 pays développés, un coût annuel de 0,12 % seulement, et une performance nette sur 5 ans (+9,1 % − 0,12 % ≈ 9,0 %) supérieure au Fonds B net de frais (+6,1 %). Conformément aux données SPIVA et à l'arithmétique de Sharpe, un fonds passif à bas coût surperforme la grande majorité des fonds actifs sur le long terme."

---

## Solution Exercice 3 — Allocation "3 fonds" pour Camille

### Partie A : Allocation proposée

**Profil** : 35 ans, horizon 25 ans, tolérance modérée.

Allocation proposée :
- **60 % actions domestiques / large-cap développées** (fonds indiciel type MSCI World ou marché local)
- **20 % actions internationales émergentes** (diversification géographique complémentaire)
- **20 % obligations** (amortisseur de volatilité, tolérance modérée)

*Justification :*
"Avec 25 ans d'horizon, Camille peut tolérer une volatilité élevée à court terme. La proportion de 80 % en actions (diversifiées géographiquement) vise à capter la prime de risque sur le long terme. Les 20 % en obligations amortissent les krachs et permettent un rééquilibrage (acheter des actions en solde pendant les baisses en vendant les obligations qui ont moins baissé). À 50 ans (dans 15 ans), elle pourrait réévaluer vers 60/40 actions/obligations."

### Partie B : Projection à 25 ans

Rendement pondéré (net de frais de 0,20 %) :
- 80 % × (7 % − 0,20 %) + 20 % × (3 % − 0,20 %)
- = 80 % × 6,80 % + 20 % × 2,80 %
- = 5,44 % + 0,56 % = **6,00 % net / an**

Capital initial : 8 000 €, versement mensuel : 400 € (→ 4 800 €/an), n = 25 ans, r = 6 %

- 8 000 × (1,06)^25 = 8 000 × 4,292 ≈ 34 336 €
- 4 800 × [(1,06)^25 − 1] / 0,06 = 4 800 × 54,86 ≈ 263 328 €
- **Total ≈ 297 664 €** (≈ 298 000 €)

### Partie C : Impact de 0,20 % → 0,10 % de frais

Nouveau rendement pondéré à 0,10 % de frais :
- 80 % × 6,90 % + 20 % × 2,90 % = 5,52 % + 0,58 % = **6,10 % net**

Nouvelle projection :
- 8 000 × (1,061)^25 = 8 000 × 4,383 ≈ 35 064 €
- 4 800 × [(1,061)^25 − 1] / 0,061 = 4 800 × 55,53 ≈ 266 544 €
- **Total ≈ 301 608 €**

Différence : 301 608 − 297 664 ≈ **3 944 €** supplémentaires  
Soit **+1,3 %** de capital final en plus.

> Réduire les frais de seulement 0,10 point pendant 25 ans rapporte près de 4 000 € supplémentaires sur ce profil. Sur des montants plus importants ou des durées plus longues, l'effet est démultiplié.
