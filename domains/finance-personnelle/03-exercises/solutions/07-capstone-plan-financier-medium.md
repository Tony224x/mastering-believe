# Solution — Capstone : extensions du plan financier (niveau intermédiaire)

> Corrigés modèles pour le profil fictif "Alex". Les rendements sont **illustratifs** et ne garantissent aucun résultat. Ce document est **purement éducatif** et ne constitue pas un conseil financier personnalisé.
>
> *Note : ces solutions utilisent la capitalisation mensuelle (`capital_final_mensuel` / `02-code/07-capstone-plan-financier.py`). Une capitalisation annuelle simplifiée donne des chiffres légèrement inférieurs, dans la même fourchette.*

---

## Solution Exercice 1 — Sensibilité au rendement (Alex, 280 €/mois, 0 € de départ)

### Question 1 : tableau 3 scénarios × 3 horizons

| Horizon | Défavorable (3 %) | Central (5 %) | Favorable (7 %) |
|---|---|---|---|
| 10 ans | ≈ 39 045 € | ≈ 43 222 € | ≈ 47 894 € |
| 20 ans | ≈ 91 519 € | ≈ 113 625 € | ≈ 142 110 € |
| 30 ans | ≈ **162 040 €** | ≈ **228 305 €** | ≈ **327 447 €** |

### Question 2 : écart à 30 ans

Défavorable -> favorable : 327 447 − 162 040 = **~165 407 €**, soit un facteur **×2,0** entre les bornes. Le même plan (mêmes versements, même durée) produit du simple au double selon une seule hypothèse de rendement. Une projection à **chiffre unique** est donc trompeuse : il faut présenter une fourchette.

### Question 3 : versements vs intérêts

Total versé sur 30 ans : 280 × 12 × 30 = **100 800 €**.
Au scénario central (228 305 €) : intérêts composés = 228 305 − 100 800 = 127 505 €, soit **~55,8 %** du capital final ; les versements ne pèsent que **~44,2 %**. Sur 30 ans, la majorité du capital vient des intérêts.

### Question 4 : disclaimer (exemple)

« Ces projections reposent sur des rendements **illustratifs** (3 % / 5 % / 7 %) ; les performances passées ne préjugent pas des performances futures et aucun rendement n'est garanti. Tout investissement comporte un risque de perte en capital. Ce plan est éducatif, pas un conseil personnalisé. »

---

## Solution Exercice 2 — Versement vs durée (5 % net, cible 228 000 €)

### Question 1 : capital à 30 ans selon le versement

| Versement | Capital à 30 ans (5 % net) |
|---|---|
| 200 €/mois | ≈ 163 075 € |
| 280 €/mois | ≈ 228 305 € |
| 350 €/mois | ≈ 285 382 € |
| 450 €/mois | ≈ 366 919 € |

### Question 2 : horizon pour atteindre 228 000 €

| Versement | Années pour 228 000 € |
|---|---|
| 280 €/mois | ≈ **30 ans** |
| 350 €/mois | ≈ **27 ans** |
| 450 €/mois | ≈ **23 ans** |

### Question 3 : gain d'années (280 -> 450 €)

Passer de 280 € à 450 €/mois (+170 €) fait atteindre le jalon en ~23 ans au lieu de ~30 ans, soit **~7 ans gagnés**. C'est l'illustration directe du Module 06 : le **taux d'épargne** (ici, le montant investi) est le levier principal — augmenter l'effort raccourcit fortement l'horizon, davantage que d'espérer un meilleur rendement (qu'on ne contrôle pas).

### Question 4 : réduire les dépenses vs attendre la fin du crédit

- **Réduire les dépenses non essentielles** : libère des fonds **immédiatement** et a un double effet (Module 06 : on investit plus ET on réduit la cible d'indépendance, car la cible = 25 × dépenses). Levier le plus puissant, mais demande de l'effort/discipline.
- **Attendre l'extinction du crédit auto** (libère 230 €/mois) : sans effort supplémentaire, mais **différé** (~35 mois). Ces 230 € serviront d'abord, selon l'ordre des priorités du capstone, à finaliser le fonds d'urgence si besoin, puis à l'investissement.

Arbitrage raisonnable : combiner les deux — réduire un peu les dépenses dès maintenant pour ne pas perdre de temps (le temps compte, Module 01), et rediriger les 230 € libérés vers l'investissement une fois le crédit soldé. L'ordre reste : fonds d'urgence complet -> dette traitée -> investissement renforcé.

---

## Solution Exercice 3 — Intégrer une augmentation (lifestyle creep)

### Question 1 : capital final avec augmentation à l'année 5

- Capital à l'année 5 (280 €/mois, 5 %) : ≈ **18 988 €**.
- Projection de ce capital + 450 €/mois pendant 25 ans (5 %) : ≈ **327 880 €**.

### Question 2 : gain vs scénario sans augmentation

- Sans augmentation (280 €/mois pendant 30 ans) : ≈ **228 305 €**.
- Avec augmentation réinvestie : ≈ **327 880 €**.
- Gain : 327 880 − 228 305 = **~99 575 €** (**+43,6 %**) à 30 ans, simplement en réinvestissant l'augmentation au lieu de la consommer.

### Question 3 : inflation du train de vie et taux d'épargne

L'"inflation du train de vie" (lifestyle creep) consiste à augmenter ses dépenses chaque fois que le revenu monte — annulant l'effet de la hausse sur l'épargne. Si Alex **dépense** son augmentation, son taux d'épargne stagne ; s'il l'**investit**, il enclenche le double effet du Module 06 : il accumule plus vite ET (s'il ne gonfle pas ses dépenses) sa cible d'indépendance reste basse. C'est l'un des leviers les plus efficaces : "garder le même train de vie quand le revenu augmente" transforme chaque hausse en accélérateur d'indépendance.

### Question 4 : 3 éléments de révision annuelle (Module 07 §5)

1. **Mettre à jour le virement automatique** : porter l'investissement mensuel de 280 € à 450 € (acter la hausse dans le système).
2. **Recalculer les projections et l'horizon d'indépendance** avec le nouveau versement (et vérifier réel vs plan).
3. **Vérifier l'allocation et la rééquilibrer** si une classe d'actifs s'est écartée de plus de ~5 points de la cible ; confirmer que les frais (TER) restent bas.

---

## Résumé des enseignements clés (medium)

1. Une projection honnête se présente en **fourchette** (défavorable / central / favorable), jamais en chiffre unique.
2. Sur 30 ans, la **majorité** du capital final vient des intérêts composés.
3. Le **montant investi** (taux d'épargne) est un levier puissant : +170 €/mois peut gagner ~7 ans sur un jalon d'indépendance.
4. Réinvestir une **augmentation** (éviter le lifestyle creep) peut ajouter +40 %+ de capital à 30 ans.
5. La **révision annuelle** maintient le plan vivant : acter les hausses, recalculer, rééquilibrer.
6. Rendements **illustratifs**, aucun garanti.
