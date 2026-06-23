# Solutions Hard — Module 02 : Probabilites utiles

*Les calculs par strate / par scenario peuvent etre verifies avec :*
`python domains/vie/rationalite-decision/02-code/02-probabilites-utiles.py`
*(une execution par couple prevalence donne VP, FP, VN, FN, VPP, VPN).*

*Arrondis : effectifs gardes a une decimale quand ils ne tombent pas juste ; VPP/VPN arrondies a une decimale en pourcentage.*

---

## Exercice 1 — Stratification du risque : qui faut-il tester ?

**Donnees** : test unique, sensibilite = 95 %, specificite = 96 %. Deux groupes de 10 000 personnes : faible risque (taux de base 0,2 %) et haut risque (8 %).

### Groupe a faible risque (prevalence 0,2 %)
```
Reellement positifs : 10 000 x 0,002 = 20
Reellement negatifs : 10 000 x 0,998 = 9 980

VP = 20 x 0,95     = 19
FP = 9 980 x 0,04  = 399,2  (≈ 399)

VPP_faible = 19 / (19 + 399,2) = 19 / 418,2 ≈ 0,0454 ≈ 4,5 %
```

### Groupe a haut risque (prevalence 8 %)
```
Reellement positifs : 10 000 x 0,08 = 800
Reellement negatifs : 10 000 x 0,92 = 9 200

VP = 800 x 0,95   = 760
FP = 9 200 x 0,04 = 368

VPP_haut = 760 / (760 + 368) = 760 / 1 128 ≈ 0,6738 ≈ 67,4 %
```

### Depistage indifferencie (les deux groupes melanges, 20 000 personnes)

On additionne les cellules des deux strates :
```
VP total = 19 + 760   = 779
FP total = 399,2 + 368 = 767,2

VPP_global = 779 / (779 + 767,2) = 779 / 1 546,2 ≈ 0,5038 ≈ 50,4 %
```
Verification par la prevalence melangee : (20 + 800) / 20 000 = 820 / 20 000 = 4,1 %.
VP = 820 x 0,95 = 779 ; sains = 19 180, FP = 19 180 x 0,04 = 767,2 -> meme resultat. OK.

### Decision

| Strate | Prevalence | VPP |
|--------|-----------|-----|
| Faible risque | 0,2 % | ~4,5 % |
| Haut risque | 8 % | ~67,4 % |
| Melange indifferencie | 4,1 % | ~50,4 % |

Le **meme test** donne une VPP de 4,5 % a 67,4 % selon le seul taux de base. Politique recommandee :
- **Tester en priorite le groupe a haut risque** : 2 alertes sur 3 y sont justes.
- **Eviter le depistage de masse du groupe a faible risque** : on y obtient environ **21 faux positifs pour 1 vrai positif** (399 / 19 ≈ 21), soit beaucoup d'inquietude et d'examens de confirmation inutiles pour tres peu de vrais cas.
- **Ne pas se fier a la VPP globale (50,4 %)** : elle ne decrit fidelement *aucun* des deux groupes. Agreger des sous-populations de risques tres differents masque l'information — c'est exactement pourquoi la **stratification du risque avant de tester** est plus rentable que tester tout le monde de la meme facon.

---

## Exercice 2 — Transfert : detecteur de fraude sur des transactions

**Donnees** : detecteur sensibilite = 92 %, specificite = 99 %, taux de base de la fraude = 0,3 %, volume = 1 000 000 de transactions. La structure est identique a un test medical : "frauduleuse" joue le role de "malade", "alerte" celui de "test positif".

### Etape 1 : effectifs de base
```
Frauduleuses : 1 000 000 x 0,003 = 3 000
Legitimes    : 1 000 000 x 0,997 = 997 000
```

### Etape 2 : tableau complet

|             | Frauduleuse | Legitime | Total |
|-------------|-------------|----------|-------|
| Alerte (+)  | VP = 3 000 x 0,92 = **2 760** | FP = 997 000 x 0,01 = **9 970** | 12 730 |
| Rien (-)    | FN = 3 000 x 0,08 = **240**   | VN = 997 000 x 0,99 = **987 030** | 987 270 |
| Total       | 3 000 | 997 000 | 1 000 000 |

(FP = legitimes x (1 - specificite) = 997 000 x 0,01.)

### Etape 3 : VPP
```
VPP = VP / (VP + FP) = 2 760 / (2 760 + 9 970) = 2 760 / 12 730 ≈ 0,2168 ≈ 21,7 %
```
Quand le detecteur leve une alerte, la transaction n'est reellement frauduleuse que dans ~22 % des cas : environ **78 % des alertes sont de fausses alertes**, malgre une specificite de 99 %, a cause de la rarete de la fraude (0,3 %).

### Etape 4 : VPN
```
VPN = VN / (VN + FN) = 987 030 / (987 030 + 240) = 987 030 / 987 270 ≈ 0,99976 ≈ 99,98 %
```
Une transaction non signalee est quasi certainement legitime.

### Decision : arbitrage faux positifs / faux negatifs

Sur 1 000 000 de transactions, le detecteur produit :
- **9 970 faux positifs** : des clients legitimes geles, recontactes, frustres -> friction operationnelle, cout de support, risque sur la confiance et la retention.
- **240 faux negatifs** : des fraudes qui passent -> perte financiere directe et risque de litige.

Le "bon" seuil depend du **cout relatif** des deux erreurs : si une fraude coute beaucoup plus cher qu'un blocage injustifie, on tolere plus de faux positifs (on baisse le seuil, on monte la sensibilite) ; sinon on protege l'experience client (on monte le seuil, on privilegie la specificite).

**Quel levier ameliore le plus la VPP ?** A 0,3 % de prevalence, c'est la **specificite** (et/ou un pre-filtrage qui releve le taux de base avant le detecteur), pas la sensibilite :
```
Passer la specificite de 99 % a 99,9 % :
  FP = 997 000 x 0,001 = 997
  VPP = 2 760 / (2 760 + 997) = 2 760 / 3 757 ≈ 73,5 %   (de 22 % a ~74 %)

Passer la sensibilite de 92 % a 100 % (specificite inchangee) :
  VP = 3 000 ; FP = 9 970
  VPP = 3 000 / (3 000 + 9 970) = 3 000 / 12 970 ≈ 23,1 %  (a peine mieux)
```
Quand l'evenement est rare, ce sont les **faux positifs** (donc la specificite) qui dominent la VPP ; gagner en sensibilite reduit surtout les faux negatifs. Une autre voie tres efficace est de **remonter le taux de base** en n'envoyant au detecteur que des transactions deja pre-filtrees a risque eleve (meme logique de stratification qu'a l'exercice 1).
