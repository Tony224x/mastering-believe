# Solutions — Module 02 : Probabilites utiles

*Les scripts Python peuvent etre utilises pour verifier les calculs :*
`python domains/rationalite-decision/02-code/02-probabilites-utiles.py`

---

## Exercice 1 : Test de detection de doping

**Donnees** : 500 athletes, prevalence = 3 %, sensibilite = 92 %, specificite = 97 %.

**Etape 1 : effectifs de base**
- Athletes dopes : 500 x 0,03 = **15**
- Athletes non dopes : 500 x 0,97 = **485**

**Etape 2 : tableau complet**

| | Dope | Non dope | Total |
|---|---|---|---|
| Test positif | VP = 15 x 0,92 = **13,8 ≈ 14** | FP = 485 x 0,03 = **14,6 ≈ 15** | 29 |
| Test negatif | FN = 15 x 0,08 = **1,2 ≈ 1** | VN = 485 x 0,97 = **470,5 ≈ 470** | 471 |
| Total | 15 | 485 | 500 |

**Etape 3 : VPP**
```
VPP = VP / (VP + FP) = 14 / (14 + 15) = 14 / 29 ≈ 48 %
```

**Etape 4 : VPN**
```
VPN = VN / (VN + FN) = 470 / (470 + 1) = 470 / 471 ≈ 99,8 %
```

**Conclusion contre-intuitive** : avec une prevalence de 3 % seulement, environ **la moitie des positifs sont des faux positifs**. Le test est excellent pour exclure la maladie (VPN 99,8 %) mais peu fiable pour la confirmer sur cette population (VPP 48 %).

---

## Exercice 2 : Probabilite conditionnelle et meteo

**Donnees** : P(pluie) = 0,30, P(couvert|pluie) = 0,90, P(couvert|pas de pluie) = 0,40.

**Etape 1 : P(couvert)**
```
P(couvert) = P(couvert|pluie) x P(pluie) + P(couvert|pas de pluie) x P(pas de pluie)
           = 0,90 x 0,30 + 0,40 x 0,70
           = 0,27 + 0,28
           = 0,55
```

**Etape 2 : P(pluie | couvert)**
```
P(pluie | couvert) = P(couvert | pluie) x P(pluie) / P(couvert)
                   = 0,90 x 0,30 / 0,55
                   = 0,27 / 0,55
                   ≈ 49 %
```

**Etape 3 : Erreur de transposition**

L'affirmation confond :
- P(couvert | pluie) = 90 % (ce que la statistique dit)
- P(pluie | couvert) = 49 % (ce qu'on veut savoir)

La difference est expliquee par le taux de base : il ne pleut que 30 % des jours. Meme si le ciel couvert est predictif de la pluie, la pluie reste minoritaire dans l'ensemble des jours couverts.

---

## Exercice 3 : Impact du taux de base

**Test fixe** : sensibilite = specificite = 95 %. Population = 10 000 personnes.

### Scenario 1 : prevalence = 50 %

| | Malade | Sain |
|---|---|---|
| Test + | VP = 4 750 | FP = 250 |
| Test - | FN = 250 | VN = 4 750 |

```
VPP = 4 750 / 5 000 = 95 %
```

### Scenario 2 : prevalence = 10 %

| | Malade | Sain |
|---|---|---|
| Test + | VP = 950 | FP = 450 |
| Test - | FN = 50 | VN = 8 550 |

```
VPP = 950 / 1 400 ≈ 67,9 %
```

### Scenario 3 : prevalence = 0,5 %

| | Malade | Sain |
|---|---|---|
| Test + | VP ≈ 48 | FP ≈ 498 |
| Test - | FN ≈ 3 | VN ≈ 9 452 |

```
VPP = 48 / 546 ≈ 8,8 %
```

**Conclusion** : la VPP chute de 95 % a 9 % simplement en changeant la prevalence de 50 % a 0,5 %, avec le meme test. Pour les maladies rares, le depistage de masse produit majoritairement des faux positifs et doit etre restreint aux sous-populations a risque eleve.
