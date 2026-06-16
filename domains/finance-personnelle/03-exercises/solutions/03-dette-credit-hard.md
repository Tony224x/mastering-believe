# Solutions — Maitriser la dette et le credit (Module 03, niveau avance)

> **Disclaimer** : solutions educatives, situations fictives. Les taux sont **illustratifs**, jamais des offres. Ce contenu est educatif et **ne constitue pas un conseil financier**. Aucun rendement n'est garanti ; tout investissement comporte un risque de perte.

---

## Solution Exercice 1 — Rembourser la dette ou investir ? (Yann, 10 000 €)

### Question 1 : le "rendement" du remboursement

Rembourser 10 000 € d'une dette a 8 % evite de payer 8 % d'interets sur ces 10 000 € — c'est equivalent a obtenir un **rendement de 8 %**. Et ce rendement est **certain et sans risque** : il ne depend d'aucun marche, d'aucune hypothese. Chaque euro rembourse "rapporte" mecaniquement le taux de la dette en interets non payes. C'est l'un des rares "placements" a rendement garanti.

### Question 2 : option investissement vs interets evites

- Investir 10 000 € a **6 %** (central) sur 5 ans : `capital_final(10000, 0.06, 5)` ≈ **13 489 €** (gain ~3 489 €).
- Investir 10 000 € a **3 %** (prudent) sur 5 ans : ≈ **11 616 €** (gain ~1 616 €).
- Interets evites en remboursant la dette a 8 % (proxy `10 000 × (1,08)^5`) ≈ **14 693 €** equivalent, soit un "gain garanti" de ~4 693 €.

Le remboursement de la dette a 8 % produit un benefice (~4 700 €) **superieur** au gain espere de l'investissement central a 6 % (~3 500 €) — et tres superieur au scenario prudent.

### Question 3 : regle de decision

Regle du module : **quand le taux de la dette depasse le rendement attendu net de l'investissement, on rembourse la dette en priorite.** Ici, 8 % (dette) > 6 % (rendement central espere) -> **rembourser**. Le remboursement est le meilleur "investissement" disponible : 8 % garantis.

### Question 4 : certain vs hypothetique/risque

On ne peut pas comparer 8 % et 6 % a egalite car ils n'ont pas le meme **statut** :
- Le 8 % de la dette est **certain** : il sera reellement evite.
- Le 6 % de l'investissement est **espere et risque** : il peut etre 6 %, mais aussi 0 % ou negatif sur 5 ans (les marches peuvent baisser durablement).

A esperance brute deja inferieure (6 < 8) **et** avec un risque en plus, l'investissement est doublement perdant ici. L'incertitude **renforce** la conclusion : rembourser la dette a 8 %.

### Question 5 : cas limite (dette a 2 %)

Si la dette etait a **2 %** (pret aide), l'arbitrage **basculerait** : 2 % (dette) < 6 % central espere -> on pourrait privilegier l'investissement. Mais avant de le faire, verifier :
- **Fonds d'urgence complet** (Module 02) — ici, oui.
- **Horizon long** : n'investir que des fonds dont on n'aura pas besoin avant 5-10 ans (Module 04).
- **Tolerance au risque** : accepter que l'investissement puisse sous-performer la dette a 2 % certains annees.
- **Comparer au rendement reel** : a 2 % de dette et 6 % espere, la marge est reelle mais incertaine ; certains preferent quand meme la securite du remboursement. Pas de reponse unique — c'est un arbitrage de profil.

---

## Solution Exercice 2 — Cout reel d'un achat a credit (Nadia, voiture 18 000 €)

### Question 1 : mensualite et cout du credit

`M = P × i / (1 - (1+i)^(-n))`, P = 18 000, i = 0,059/12 ≈ 0,004917, n = 60 :
- Mensualite ≈ **347,15 €/mois**
- Total rembourse ≈ 347,15 × 60 ≈ **20 829 €**
- Cout du credit (interets) ≈ 20 829 - 18 000 ≈ **2 829 €**

### Question 2 : cout d'opportunite cote credit

Si Nadia prend le credit et garde ses 18 000 € au livret a 2,5 % pendant 5 ans :
`18 000 × (1,025)^5` ≈ 20 366 €, soit **~2 366 € d'interets gagnes**.

Bilan net : cout du credit (2 829 €) - interets du livret (2 366 €) = **~463 € de cout net** au desavantage du credit. Comme le TAEG (5,9 %) > taux du livret (2,5 %), le credit reste **defavorable** : on paie plus d'interets qu'on n'en gagne.

### Question 3 : cout d'opportunite cote comptant (mensualite investie)

Si, au lieu de rembourser un credit, Nadia investissait l'equivalent de la mensualite (~347 €/mois) a 5 % illustratif pendant 5 ans : `capital_final(0, 0.05, 5, 347)` ≈ **23 609 €**.

Ce que cela revele : la voiture ne coute pas "18 000 €" — elle **mobilise une capacite d'epargne de ~347 €/mois pendant 5 ans**, qui aurait pu devenir ~23 600 €. Le vrai cout d'un achat est ce a quoi on renonce (cout d'opportunite, Module 03), pas seulement le prix affiche ni meme les interets.

### Question 4 : decision et nuances

Selon la regle du module (TAEG 5,9 % > rendement sans risque 2,5 %), **payer comptant est preferable** ici, sauf si cela compromet la liquidite (mais le fonds d'urgence est separe et intact, donc OK).

Deux situations ou le credit peut neanmoins se defendre (sans en faire une regle) :
1. **Taux promotionnel reellement bas** (ex. TAEG 0-2 % subventionne par le vendeur) : si le TAEG passe sous le rendement sans risque, garder son epargne placee peut etre rationnel.
2. **Preservation d'un coussin de liquidite** : si payer comptant viderait toute l'epargne disponible au-dela du fonds d'urgence, etaler via un petit credit peut eviter de se retrouver sans marge — la securite a une valeur.

### Question 5 : esprit critique sur le montant (depreciation)

La voiture est un actif qui **se deprecie** : c'est, par nature, le terrain de la "mauvaise dette" (Module 03 : objet a depreciation immediate). Cela invite a la prudence non pas seulement sur le **financement**, mais sur le **montant** lui-meme : un vehicule moins cher (ex. 12 000 €) reduit a la fois le capital immobilise, les interets eventuels et le cout d'opportunite. La meilleure decision financiere est souvent en amont — **acheter moins cher** — avant meme de choisir comptant ou credit.

---

## Resume des enseignements cles (hard)

1. Rembourser une dette = obtenir un rendement **certain et sans risque** egal a son taux ; c'est souvent le meilleur "placement".
2. Quand le **taux de la dette > rendement espere net** de l'investissement, rembourser ; l'incertitude de l'investissement **renforce** cette conclusion.
3. On ne compare jamais un rendement certain et un rendement espere a egalite — il faut ajuster pour le risque.
4. Le **vrai cout** d'un achat a credit cumule interets payes + cout d'opportunite (epargne mobilisee) — souvent bien superieur au prix affiche.
5. Pour un actif qui se **deprecie** (mauvaise dette), la prudence porte d'abord sur le **montant** de l'achat, pas seulement sur son financement.
6. Taux et rendements sont **illustratifs** ; aucun rendement n'est garanti.
