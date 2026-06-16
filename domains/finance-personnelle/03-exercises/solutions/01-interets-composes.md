# Solutions — Interets composes et valeur du temps (Module 01)

> **Disclaimer** : solutions educatives a titre de reference. Les taux utilises sont illustratifs.
>
> **Conseil** : tentez de resoudre les exercices par vous-meme avant de lire ces corrections. Voir aussi le calculateur `02-code/01-interets-composes.py` pour verifier vos calculs.

---

## Solution Exercice 1 — Calculer la croissance d'un capital initial

### Questions 1-3 : Calcul avec A = P × (1 + r)^t, P = 5 000 €, r = 0,05

**Apres 5 ans :**
```
A = 5 000 × (1,05)^5
  = 5 000 × 1,2763
  = 6 381 €
```

**Apres 20 ans :**
```
A = 5 000 × (1,05)^20
  = 5 000 × 2,6533
  = 13 266 €
```

**Apres 40 ans :**
```
A = 5 000 × (1,05)^40
  = 5 000 × 7,0400
  = 35 200 €
```

Verification avec le script :
```python
# Dans un shell Python
from domains.finance_personnelle import capital_final
# ou : copier la fonction du script
print(capital_final(5000, 0.05, 5))   # 6 381 €
print(capital_final(5000, 0.05, 20))  # 13 267 €
print(capital_final(5000, 0.05, 40))  # 35 200 €
```

### Question 4 : Verification script
Modifier `demo_calcul_de_base()` ou appeler `capital_final(5000, 0.05, X)` directement.

### Question 5 : Pourquoi le gain 20-40 ans > le gain 0-20 ans ?

Le gain entre 0 et 20 ans est : 13 266 - 5 000 = **8 266 €**.
Le gain entre 20 et 40 ans est : 35 200 - 13 266 = **21 934 €**.

Pourtant les deux periodes durent 20 ans. La difference est due a la **croissance exponentielle** : a 20 ans, le capital est de 13 266 € (au lieu de 5 000 €). Ce capital plus grand genere lui-meme plus d'interets, qui generent a leur tour des interets. C'est l'"interet sur l'interet" — chaque euro de gain s'ajoute a la base de calcul des annees suivantes.

---

## Solution Exercice 2 — Comparer deux strategies d'epargne

### Total des sommes versees

- **Strategie A** : 150 €/mois × 120 mois (10 ans) = **18 000 €**
- **Strategie B** : 300 €/mois × 276 mois (23 ans) = **82 800 €**

### Capital a 65 ans (taux 6 %/an)

**Strategie A :**
- Capital accumule de 22 a 32 ans (versements) :
  `C_32 = 150 × [((1.005)^120 - 1) / 0.005] = 150 × 163,9 = 24 585 €`
- Croissance de 32 a 65 ans (33 ans sans versement) :
  `A = 24 585 × (1,06)^33 = 24 585 × 6,841 = 168 200 €`

**Strategie B :**
- Capital accumule de 42 a 65 ans (23 ans de versements) :
  `C = 300 × [((1.005)^276 - 1) / 0.005] = 300 × 495,2 = 148 560 €`

> Utiliser le script `capital_final(0, 0.06, 23, 300)` → environ 148 000-150 000 €.

### Comparaison

| | Strategie A | Strategie B |
|---|---|---|
| Total verse | 18 000 € | 82 800 € |
| Capital a 65 ans | ~168 000 € | ~148 000 € |
| Rapport capital/verse | ~9,3x | ~1,8x |

**Reponses :**
- La **Strategie A** produit un capital final plus important malgre 4,6x moins de sommes versees.
- La Strategie A a un rapport capital/verse de ~9x vs ~2x pour B — elle est bien plus efficace "par euro investi".
- Le "cout de l'attente" de 20 ans (commencer a 42 ans plutot que 22 ans) est ici d'environ **20 000 € de capital final en moins**, pour le double de sommes versees.

> **Lesson key** : le temps investi (en annees) bat le montant investi (en euros), grace a l'exponentielle.

---

## Solution Exercice 3 — Regle des 72 et cout de l'inaction

### Partie A — Tableau regle des 72

| Taux | Estimation (72/taux) | Calcul exact (ln2/ln(1+r)) | Ecart |
|------|---------------------|--------------------------|-------|
| 3 % | 72/3 = **24,0 ans** | ln(2)/ln(1,03) = **23,4 ans** | +2,6 % |
| 7 % | 72/7 = **10,3 ans** | ln(2)/ln(1,07) = **10,2 ans** | +1,0 % |
| 12 % | 72/12 = **6,0 ans** | ln(2)/ln(1,12) = **6,1 ans** | -1,6 % |

La regle des 72 est une excellente approximation (erreur < 3 % sur les taux courants).

### Partie B — Le cout de l'inaction de Marie (250 €/mois, 7 %)

**Si elle commence a 30 ans (35 ans de versements) :**
```
capital_final(0, 0.07, 35, 250) ≈ 370 000 €
```

**Si elle commence a 35 ans (30 ans de versements) :**
```
capital_final(0, 0.07, 30, 250) ≈ 257 000 €
```

**Manque a gagner lié aux 5 ans d'attente :**
```
370 000 - 257 000 = ~113 000 €
```

**Question 4 — Compensation par des versements supplementaires :**
Pour recuperer 113 000 € en commencant a 35 ans sur 30 ans :
Il faudrait verser ~250 × (370/257) ≈ **360 €/mois** au lieu de 250 €, soit 44 % de plus. En pratique, 5 ans d'attente necessite d'augmenter ses versements d'environ 40-50 % pour compenser — ce qui illustre l'irreversibilite du temps perdu.

---

## Resume des enseignements cles (Exercices 1-3)

1. La croissance exponentielle s'accelere avec le temps — le gain des dernieres annees est toujours superieur aux premieres.
2. Verser peu tot bat verser beaucoup tard, meme si l'effort total est bien plus faible.
3. La regle des 72 est un outil de calcul rapide fiable (erreur < 3 %).
4. 5 ans d'attente peuvent couter 100 000+ € de capital final — c'est concret et chiffrable.
5. Tous ces calculs sont illustratifs : les rendements reels varient, ne sont pas garantis, et dependent de choix d'investissement risques.
