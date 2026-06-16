# Module 12 — Independance financiere et retrait soutenable

> **Temps estime** : 45 min | **Prerequis** : Modules 01-11
>
> **Objectif** : Comprendre le **taux d'epargne** comme moteur de l'independance financiere, connaitre la **regle des 4 %** (issue du Trinity Study) et surtout **ses limites** (sequence des rendements, hypothese 30 ans), et apprehender le mouvement **FIRE** comme un cadre de reflexion — non comme une promesse ou une recette universelle.

> ⚠️ **Disclaimer** : ce module est **purement educatif** et ne constitue pas un conseil financier, fiscal ou en investissement personnalise. Les projections reposent sur des hypotheses historiques qui ne garantissent pas les resultats futurs. Tout investissement comporte un risque de perte en capital. Consultez un conseiller agree pour toute decision reelle.

---

## 1. Exemple d'abord : le taux d'epargne decide tout

Sophie et Theo gagnent le meme salaire net : 3 000 €/mois.

- **Sophie** epargne 10 % (300 €/mois) → il lui faudra environ **43 ans** pour atteindre l'independance financiere (hypothese de rendement reel ~5 %, depenses stables).
- **Theo** epargne 40 % (1 200 €/mois) → environ **22 ans**.
- **Si Theo monte a 60 %** (1 800 €/mois) → environ **12 a 13 ans**.

> La variable decisive n'est pas le revenu (ils gagnent pareil). C'est le **taux d'epargne**, qui agit sur deux leviers a la fois : on accumule plus vite **et** on a besoin d'un capital total plus petit (car on depense moins).

> **A retenir** : a revenu egal, c'est le taux d'epargne — pas le salaire — qui determine en combien de temps on atteint l'independance. (Et le revenu, lui, se travaille : voir Module 11.)

---

## 2. Le taux d'epargne : definition et double effet

**Taux d'epargne = (Revenu disponible − Depenses) / Revenu disponible × 100**

C'est le pourcentage de votre revenu que vous ne depensez pas et que vous investissez.

**Pourquoi un taux d'epargne eleve raccourcit doublement le chemin :**
1. **Accumulation plus rapide** : plus de capital investi chaque mois → plus d'interets composes (Module 01).
2. **Cible plus basse** : si vous vivez avec 2 000 €/mois, votre patrimoine cible est plus petit que si vous vivez avec 4 000 €/mois.

**Tableau indicatif (hypotheses : rendement reel 5 %, depenses stables, depart de zero) :**

| Taux d'epargne | Annees jusqu'a l'independance |
|---|---|
| 10 % | ~43 ans |
| 20 % | ~37 ans |
| 30 % | ~28 ans |
| 50 % | ~17 ans |
| 70 % | ~8,5 ans |

> Ces chiffres sont **illustratifs**. Ils supposent un rendement reel constant et des depenses stables — deux hypotheses qui ne tiennent pas dans la vraie vie. Ils illustrent le **principe de levier**, pas un plan a l'euro pres. Le script `02-code/12-independance-financiere.py` recalcule ces durees avec vos propres parametres.

---

## 3. La regle des 4 % : origine, logique et limites

### 3.1 Origine

William Bengen (*Journal of Financial Planning*, 1994) a analyse les donnees historiques des marches americains depuis 1926, cherchant le taux de retrait annuel qu'un retraite pouvait se permettre **sans jamais epuiser son capital sur 30 ans, meme dans les pires scenarios historiques**.

Conclusion : **4 % du capital initial**, ajustes de l'inflation chaque annee, n'ont jamais epuise un portefeuille diversifie (~50 % actions / 50 % obligations) sur 30 ans dans les donnees historiques americaines.

La **Trinity Study** (Cooley, Hubbard & Walz, 1998) a confirme et enrichi cette analyse, popularisant la regle.

### 3.2 Ce que ca signifie en pratique

> Si vous avez besoin de 30 000 €/an pour vivre, vous visez un patrimoine investi de **30 000 / 0,04 = 750 000 €** (= 25 fois vos depenses annuelles).

Calcul rapide : **capital cible = depenses annuelles × 25** (car 1 / 0,04 = 25).

### 3.3 Les limites importantes (a ne JAMAIS omettre)

La regle des 4 % est un **point de depart de reflexion**, pas une garantie. Ses limites :

**a) Sequence des rendements (*sequence of returns risk*) — la limite la plus sous-estimee**
Deux retraites avec le **meme rendement moyen** sur 30 ans peuvent avoir des issues opposees selon **l'ordre** des rendements. Si un krach severe survient dans les premieres annees de retrait, vous vendez des actifs deprecies pour vivre, ce qui ampute le capital disponible pour le rebond futur. Un krach en annee 3 est bien plus destructeur qu'un krach identique en annee 20 — alors que la moyenne, elle, est inchangee. C'est pourquoi la moyenne historique ne suffit pas a garantir un retrait.

**b) Hypothese d'un horizon de 30 ans**
Bengen calculait pour une retraite classique (~65 → 95 ans). Si vous visez l'independance a 40 ans pour **50 ans** de retrait, la regle de 4 % **sous-estime** le risque d'epuisement : plus l'horizon est long, plus une longue serie defavorable peut survenir. Plusieurs chercheurs suggerent **3 % a 3,5 %** pour les horizons tres longs.

**c) Donnees historiques americaines uniquement**
Les marches US ont connu une performance exceptionnelle au 20e siecle. D'autres marches (Europe, Japon) auraient donne des resultats moins favorables. Une regle issue d'un seul marche est souvent presentee a tort comme universelle.

**d) Retrait fixe vs flexibilite reelle**
La regle suppose un retrait fixe, indexe sur l'inflation, sans revenus complementaires ni ajustement. En pratique, la plupart des personnes en independance financiere gardent de la flexibilite : reduire les depenses en periode de baisse, reprendre une activite partielle. Cette flexibilite ameliore nettement la robustesse.

**e) Fiscalite et frais non inclus**
Les simulations historiques n'integrent ni la fiscalite des retraits ni les frais de gestion. Ces couts **reduisent** le taux de retrait soutenable reel (Modules 06 et 07).

> **A retenir** : "depenses × 25" est un **ordre de grandeur de reflexion**, pas une promesse. Les trois limites a retenir en priorite : **sequence des rendements**, **horizon de 30 ans** (trop court pour un FIRE precoce), et **flexibilite** qui change tout.

> Synthese neutre des etudes : [bogleheads.org/wiki/Safe_withdrawal_rates](https://www.bogleheads.org/wiki/Safe_withdrawal_rates).

---

## 4. Le mouvement FIRE : un cadre, pas une promesse

**FIRE = Financial Independence, Retire Early** (independance financiere, retraite anticipee).

Popularise dans les annees 2010 (notamment via *The Simple Path to Wealth*, J. L. Collins, et des communautes en ligne). Le principe : vivre volontairement en dessous de ses moyens, epargner et investir avec regularite, viser l'independance assez tot pour ne plus **dependre** d'un salaire.

**Variantes (vocabulaire courant) :**
- **Lean FIRE** : independance avec un budget reduit (mode de vie minimaliste).
- **Fat FIRE** : independance avec un niveau de vie confortable (capital cible plus eleve).
- **Coast FIRE** : atteindre assez tot un capital pour que la croissance composee seule mene jusqu'a la retraite classique, sans epargner davantage.
- **Barista FIRE** : revenus partiels d'une activite legere couvrant les frais courants, reduisant le capital necessaire.

**Ce que le FIRE enseigne de valable :**
- La clarte sur ses depenses reelles est le point de depart de toute planification.
- Le taux d'epargne est plus decisif que le revenu brut (section 1).
- L'investissement indiciel a bas cout (Module 06) est coherent avec un horizon long.

**Ce que le FIRE ne dit pas (et ses angles morts) :**
- Ce n'est **pas** universellement applicable : revenus faibles, soins medicaux couteux, charges familiales, fiscalite elevee changent radicalement l'equation.
- "Retraite" a 35-40 ans est ambigu : beaucoup continuent une activite choisie.
- Les projections a 40-50 ans sont **tres** sensibles aux hypotheses de rendement et a la sequence des rendements (section 3.3).

> **A retenir** : presentez le FIRE comme **un cadre de reflexion** sur la liberte financiere, a adapter a sa situation — jamais comme une promesse chiffree ni un dogme.

---

## 5. L'independance financiere est un spectre, pas un interrupteur

L'independance n'est pas binaire (tout ou rien). Chaque palier de patrimoine investi achete de la liberte :

- **Filet de securite progressif** : 1 an de depenses investies, puis 3 ans, puis 5 ans. Chaque palier reduit la dependance au salaire.
- **Revenu passif partiel** : un portefeuille couvrant 30-50 % des besoins allege deja fortement la pression du travail obligatoire.
- **Reconversion facilitee** : un patrimoine donne la liberte de prendre des risques professionnels (changer de secteur, creer une activite) sans panique financiere.
- **Retraite classique amelioree** : combiner retraite legale et patrimoine investi pour maintenir son niveau de vie.

---

## 6. Les vraies questions a se poser

Avant de viser l'independance financiere :

1. **Quel niveau de depenses en "independance" ?** (souvent different du niveau actuel)
2. **Quel horizon reel ?** (retraite a 65, 50 ou 40 ans → 20, 30 ou 50 ans de retrait)
3. **Quelle tolerance au risque en phase de decumulation ?** (distincte de la phase d'accumulation)
4. **Quelles sources de revenus complementaires ?** (retraite legale, loyers, activite partielle)
5. **Quelle flexibilite puis-je integrer ?** (reduire les depenses ou reprendre une activite en cas de krach)

---

## 7. Flash-cards (spaced repetition)

**Q1** — Quel est le double effet d'un taux d'epargne eleve sur l'independance financiere ?
> **R** : On accumule plus de capital plus vite (plus investi = plus de compose) **et** on a besoin d'un capital total plus petit (car on depense moins). Les deux effets se cumulent et raccourcissent fortement l'horizon.

**Q2** — Quelle est la formule rapide du capital cible selon la regle des 4 % ?
> **R** : Capital cible = depenses annuelles × 25 (car 1 / 0,04 = 25). Exemple : 30 000 €/an → 750 000 €.

**Q3** — Qu'est-ce que le risque de sequence des rendements, et pourquoi rend-il la regle des 4 % fragile ?
> **R** : Deux retraites avec le meme rendement moyen peuvent finir tres differemment selon l'ordre des rendements. Un krach en debut de retrait force a vendre des actifs deprecies pour vivre, amputant le capital pour le rebond. La moyenne ne suffit donc pas a garantir le retrait.

**Q4** — Pourquoi la regle des 4 % est-elle moins fiable pour une independance a 40 ans ?
> **R** : Elle a ete calibree pour un horizon de 30 ans (retraite classique). Sur 50 ans, le risque d'epuisement augmente ; certains chercheurs suggerent 3 % a 3,5 %.

**Q5** — Qu'est-ce que le "Coast FIRE" ?
> **R** : Atteindre assez tot un capital pour que la croissance composee le mene seul jusqu'a la retraite classique, sans epargner davantage. On peut alors reduire son taux d'epargne et "laisser composer".

---

## Points cles a retenir

1. **Le taux d'epargne est le levier numero un** : l'augmenter comprime l'horizon ET reduit le capital necessaire (double effet).
2. **Regle des 4 % = point de depart historique, pas une garantie** : la comprendre, c'est connaitre ses limites.
3. **Trois limites prioritaires** : sequence des rendements, hypothese 30 ans (trop courte pour un FIRE precoce), absence de flexibilite et de fiscalite dans le modele d'origine.
4. **Le FIRE est un cadre de reflexion**, pas un dogme ni une promesse — a adapter a son contexte.
5. **L'independance est un spectre progressif** : chaque palier de patrimoine investi augmente la liberte.
6. **La phase de decumulation a sa propre logique** (sequence des rendements, flexibilite), distincte de l'accumulation.

---

## Pour aller plus loin

- **The Simple Path to Wealth** — J. L. Collins (2016, ed. augmentee 2025) : synthese sobre du mouvement FIRE. https://www.simonandschuster.com/books/The-Simple-Path-to-Wealth/J-L-Collins/9798893310474
- **Safe Withdrawal Rates** — Bogleheads Wiki : synthese neutre des etudes Bengen et Trinity. https://www.bogleheads.org/wiki/Safe_withdrawal_rates
- **"Determining Withdrawal Rates Using Historical Data"** — W. Bengen, *Journal of Financial Planning*, 1994 (reference originale de la regle des 4 %)
- **Calculateur du domaine** — `02-code/12-independance-financiere.py` (stdlib Python : taux d'epargne → annees jusqu'a l'independance + montant de retrait a 4 %).
- **La finance pour tous** — IEFP (Banque de France / AMF) : epargne et retraite (specificites locales). https://www.lafinancepourtous.com/

> **Disclaimer** : ce module est educatif et ne constitue pas un conseil financier personnalise. Les projections reposent sur des hypotheses historiques (rendement, inflation, horizon) qui ne garantissent rien ; les performances passees ne prejugent pas des performances futures et tout investissement comporte un risque de perte en capital.
