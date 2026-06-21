# Solutions — Independance financiere et retrait soutenable (Module 12)

> **Disclaimer** : solutions educatives a titre de reference. Les hypotheses (rendement, taux de retrait) sont illustratives et ne garantissent rien. Tout investissement comporte un risque de perte en capital.
>
> **Conseil** : tentez de resoudre par vous-meme et de relancer le script `02-code/12-independance-financiere.py` avant de lire ces corrections.

---

## Solution Exercice 1 — Capital cible et retrait (regle des 4 %)

### 1. Capital cible de Julie
Cible = depenses annuelles / 0,04 = depenses × 25.
28 000 × 25 = **700 000 €**.

### 2. Retrait de Marc (600 000 €)
- Annuel : 600 000 × 0,04 = **24 000 €/an**.
- Mensuel : 24 000 / 12 = **2 000 €/mois**.

### 3. Verification script
```python
capital_cible(28000)          # -> 700000.0
montant_retrait(600000, 0.04) # -> {'annuel': 24000.0, 'mensuel': 2000.0}
```
Resultats confirmes.

### 4. Ajuste de quoi ?
Le retrait est ajuste **de l'inflation** chaque annee, afin de **maintenir le pouvoir d'achat** constant du retraite. Raisonner en euros nominaux fixes surestimerait ce que le capital peut reellement financer sur la duree (cf. rendement reel, Module 01).

---

## Solution Exercice 2 — Effet non lineaire du taux d'epargne

### Tableau (rendement reel 5 %, taux de retrait 4 %, depart de zero)

| Taux d'epargne | Annees jusqu'a l'independance |
|---|---|
| 20 % | **~36,7 ans** |
| 40 % | **~21,6 ans** |
| 60 % | **~12,4 ans** |

### Reponses

1. Passer de 20 % a 40 % (×2 sur le taux d'epargne) fait passer l'horizon de ~36,7 a ~21,6 ans : l'horizon est divise par **~1,7**... mais surtout on **gagne ~15 ans**, ce qui est bien plus que ce qu'une intuition lineaire suggererait. Et de 40 % a 60 %, on gagne encore ~9 ans. L'effet est **non lineaire** et s'accelere.

   *(Note : la division "exacte par plus de 2" depend des parametres ; l'essentiel pedagogique est que chaque palier de taux d'epargne fait gagner beaucoup d'annees, bien au-dela d'une regle de trois.)*

2. **Double effet** : (a) un taux d'epargne plus eleve signifie qu'on **investit plus** chaque annee, donc on accumule plus vite (plus de capital compose) ; (b) il signifie aussi qu'on **depense moins**, donc la **cible** a atteindre (depenses × 25) est **plus basse**. Les deux poussent dans le meme sens : moins de chemin a parcourir, parcouru plus vite.

3. Le **niveau du salaire n'influence pas la duree** (a taux d'epargne donne, depart de zero) parce que le revenu apparait a la fois au **numerateur** (la cible = revenu × (1 − taux) / taux_retrait) et au **denominateur** (le versement annuel = revenu × taux d'epargne) : il se **simplifie**. C'est pourquoi, dans l'exemple Sophie/Theo du cours, c'est le taux d'epargne — pas le salaire egal — qui fait toute la difference.

---

## Solution Exercice 3 — Critiquer la regle des 4 %

1. **Sofia (independance a 42 ans, 50 ans de retrait)** → limite **horizon trop long**. La regle des 4 % a ete calibree pour un horizon d'environ 30 ans ; sur 50 ans, le risque d'epuisement augmente. Plusieurs chercheurs suggerent **3 % a 3,5 %** pour ces horizons.

2. **Hugo (krach en debut de retrait) vs voisin (krach en milieu)** → limite **sequence des rendements**. A rendement moyen egal, un krach precoce force a vendre des actifs deprecies pour vivre, amputant le capital pour le rebond futur : c'est bien plus destructeur qu'un krach tardif. La moyenne ne suffit donc pas a garantir le retrait.

3. **Lina (retraits imposes + frais de gestion)** → limite **fiscalite et frais non inclus**. Les simulations historiques ignorent l'impot sur les retraits et les frais ; ces couts **reduisent le taux de retrait soutenable reel**. La cible "× 25" brute est donc optimiste.

### Conclusion — un mecanisme qui ameliore la robustesse
Plusieurs reponses valables, par exemple :
- **Flexibilite des depenses** : reduire temporairement les retraits en periode de baisse evite de vendre au plus bas (attenue le risque de sequence).
- **Taux de retrait plus prudent** (3-3,5 %) pour les horizons longs.
- **Reserve de liquidites** (quelques annees de depenses) pour ne pas vendre des actifs en bas de cycle.
- **Revenu complementaire partiel** (type Barista FIRE).

---

## Resume des enseignements cles (Exercices 1-3)

1. Capital cible = depenses annuelles × 25 (a 4 %) ; retrait = capital × 4 %, ajuste de l'inflation.
2. Le taux d'epargne a un **double effet** (accumuler plus + cible plus basse) → gain d'annees non lineaire.
3. A taux d'epargne donne et depart de zero, le **salaire ne change pas la duree** : il se simplifie.
4. La regle des 4 % s'utilise **toujours avec ses limites** : sequence des rendements, horizon 30 ans, donnees US, fiscalite/frais, manque de flexibilite.
5. La robustesse vient de la **flexibilite**, d'un taux plus prudent sur horizon long, et d'une reserve de liquidites — pas d'un chiffre magique.
