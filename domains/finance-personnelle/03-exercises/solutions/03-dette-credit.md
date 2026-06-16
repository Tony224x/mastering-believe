# Solutions — Maitriser la dette et le credit (Module 03)

> **Disclaimer** : solutions educatives, situations fictives. Les taux sont illustratifs et ne constituent pas une offre commerciale.

---

## Solution Exercice 1 — Calculer le cout reel d'un credit et comparer des offres

### Question 1 : Indicateur de comparaison valable

L'indicateur de comparaison valable est le **TAEG (Taux Annuel Effectif Global)**, car il inclut tous les couts du credit : interets + frais de dossier + assurance obligatoire + autres frais.

Le taux nominal seul est insuffisant car il n'inclut que les interets purs et ignore les frais, qui peuvent representer un surcoût significatif (150 € de frais de dossier sur 3 500 € = 4,3 % du capital).

### Question 2 : Classement des offres

Du moins cher au plus cher selon le TAEG :
1. **Banque A** : TAEG 5,9 %
2. **Organisme B** : TAEG 6,8 %
3. **Banque en ligne C** : TAEG 7,1 %

L'Organisme B a le taux nominal le plus bas (4,5 %) mais son TAEG de 6,8 % la classe deuxieme a cause des 150 € de frais de dossier.

### Question 3 : Mensualite de l'offre A (TAEG 5,9 %, 24 mois)

Formule : `M = P × (r/12) / (1 - (1 + r/12)^(-n))`
Avec P = 3 500, r = 0.059, n = 24.

```
r/12 = 0.059 / 12 = 0.004917
(1 + 0.004917)^(-24) = (1.004917)^(-24)

(1.004917)^24 ≈ 1.1254
(1.004917)^(-24) ≈ 0.8886

M = 3 500 × 0.004917 / (1 - 0.8886)
  = 3 500 × 0.004917 / 0.1114
  = 17.21 / 0.1114
  ≈ 154.49 €/mois
```

Mensualite : **~154,49 €/mois** (arrondie a 154-155 €).

### Question 4 : Cout total du credit A

- Total rembourse : 154,49 × 24 = **3 707,76 €**
- Cout du credit : 3 707,76 - 3 500 = **207,76 €**

Le credit Paul lui coute environ **208 € d'interets** sur 2 ans.

### Question 5 : Comparaison avec le livret a 3 %

Interets gagnes sur 3 500 € au livret a 3 % pendant 24 mois :
```
A = 3 500 × (1 + 0.03/12)^24 = 3 500 × (1.0025)^24
  = 3 500 × 1.0618
  = 3 716,30 €
Interets gagnes : 216,30 €
```

**Comparaison :**
- Cout du credit A : **208 €** (certainement paye)
- Interets perdus sur le livret : **216 €** (certainement perdus si on paie comptant)

Bilan : payer comptant lui fait perdre 216 € d'interets sur le livret et lui economise 208 € de cout de credit → il "perd" 8 € en payant comptant dans ce cas precis.

**Mais** : ici le TAEG (5,9 %) est superieur au taux du livret (3 %), donc le credit coute plus qu'il ne rapporte. L'ecart est faible (8 €) mais la logique s'applique : quand le taux du credit > rendement de l'epargne sans risque, payer comptant est preferable.

> **Conclusion** : dans ce cas, payer comptant est financierement legerement preferable. La regle generale : si TAEG > rendement disponible sans risque, utiliser l'epargne. Si TAEG < rendement attendu sans risque (rare), le credit peut valoir le coup.

---

## Solution Exercice 2 — Strategie de remboursement d'Amelie

### Rappel de la situation

| Dette | Solde | Taux | Minimum |
|-------|-------|------|---------|
| Carte renouvelable | 1 800 € | 19,5 % | 55 € |
| Credit auto | 8 500 € | 4,2 % | 180 € |
| Pret personnel | 2 200 € | 7,8 % | 65 € |
| **Total** | **12 500 €** | — | **300 €** |

Surplus disponible : 400 - 300 = **100 €/mois**.

### Partie A — Methode Avalanche

**Cible en premier : carte de credit renouvelable (19,5 %)** — taux le plus eleve.

Repartition mensuelle :
- Carte renouvelable : 55 € (minimum) + 100 € (surplus) = **155 €/mois**
- Credit auto : 180 €/mois (minimum)
- Pret personnel : 65 €/mois (minimum)

Duree estimee pour eliminer la carte (sans interets, approximation) :
```
1 800 / 155 ≈ 11,6 mois → environ 12 mois
```

Avec les interets (19,5 %/an = 1,625 %/mois) : la dette augmente de 29 € au premier mois avant remboursement → duree reelle ≈ 14-15 mois.

### Partie B — Methode Boule de Neige

**Cible en premier : carte de credit renouvelable (1 800 €)** — coincidence ici avec l'Avalanche car c'est aussi le solde le plus faible des trois dettes ! (1 800 € < 2 200 € < 8 500 €).

Repartition identique a l'Avalanche dans ce cas precis.
Duree estimee : identique (~12-15 mois).

**Motivation psychologique** : eliminer entierement une dette (meme si ce n'est pas la plus grande) cree un sentiment de victoire, libere de la tete, et renforce l'engagement. C'est un "succes visible" qui entretient l'effort.

> Note importante : dans cet exercice, les deux methodes ciblent la meme dette par coincidence (la carte renouvelable est a la fois la plus chere ET la plus petite). Dans d'autres situations (ex. si le pret personnel avait un taux de 24 % et un solde de 5 000 €), les cibles des deux methodes divergeraient.

### Partie C — Decision et cout annuel de la carte

**Cout annuel de la carte renouvelable (1 800 € a 19,5 %) :**
```
1 800 × 19,5 % = 351 €/an en interets
```
Soit **29,25 €/mois** "perdus" en interets sur ce seul poste. C'est presque autant que les 35 € de surplus mensuel — l'urgence de l'eliminer est reelle.

**Recommandation pour Amelie :**
Etant donne ses echecs precedents par manque de motivation, la **methode Boule de Neige** est recommandee — meme si ici les deux ciblent la meme dette. Apres avoir elimine la carte, elle aura une victoire concrete qui la motivera a continuer.

Si a l'avenir elle a des dettes avec ecarts importants entre taux et soldes, Boule de Neige reste preferable tant que la discipline est le facteur limitant. Un calcul montre que la difference de cout total entre les deux methodes est souvent < 5 % — la coherence sur la duree l'emporte.

---

## Solution Exercice 3 — Evaluer si s'endetter vaut le coup (Julien, 2 000 € a 9,9 % / 18 mois)

### Question 1 : Mensualite

Formule : `M = P × (r/12) / (1 - (1 + r/12)^(-n))`
Avec P = 2 000, r = 0.099, n = 18.

```
r/12 = 0.099 / 12 = 0.00825
(1.00825)^18 ≈ 1.1601
(1.00825)^(-18) ≈ 0.8620

M = 2 000 × 0.00825 / (1 - 0.8620)
  = 2 000 × 0.00825 / 0.1380
  = 16.50 / 0.1380
  ≈ 119.57 €/mois
```

Mensualite : **~119,57 €/mois** (environ 120 €).

### Question 2 : Cout total du credit

- Total rembourse : 119,57 × 18 = **2 152,26 €**
- Cout du credit : 2 152,26 - 2 000 = **152,26 €**

Le credit velo coute environ **152 €** sur 18 mois.

### Question 3 : Interets gagnes sur le livret a 2,5 % pendant 18 mois

```
A = 2 000 × (1 + 0.025/12)^18 = 2 000 × (1.002083)^18
  = 2 000 × 1.0382
  = 2 076,40 €
Interets gagnes : 76,40 €
```

### Question 4 : Cout net reel du credit

```
Cout net = Cout du credit - Interets gagnes sur livret
         = 152 - 76
         = ~76 €
```

En gardant ses 2 000 € sur le livret pendant qu'il rembourse le credit, Julien "gagne" 76 € d'interets mais "perd" 152 € de cout de credit. Cout net reel : **76 €** au desavantage du credit.

**Payer comptant est preferable** dans ce cas, car le TAEG (9,9 %) > taux du livret (2,5 %).

### Question 5 : Et si l'epargne etait en bourse a 7 % ?

Gain potentiel en bourse sur 18 mois (7 %/an pro-rata) :
```
A = 2 000 × (1 + 0.07/12)^18 = 2 000 × (1.00583)^18
  = 2 000 × 1.1097
  = 2 219,40 €
Gain potentiel : 219,40 €
```

Comparaison : gain bourse (219 €) > cout credit (152 €) → difference de **+67 €** en faveur du credit.

**Mais nuance essentielle** : ce gain est **hypothetique et risque**. Le rendement boursier de 7 % est une moyenne historique a long terme — sur 18 mois, les marchés peuvent perdre 20-30 %. Le cout du credit (152 €) est **certain et garanti**. On ne peut pas comparer un gain certain (livret) ou certain negatif (credit) a un gain hypothetique risque (bourse) sans ajuster pour le risque.

**Regle de decision** :
- TAEG credit > rendement sans risque disponible → payer comptant
- TAEG credit < rendement sans risque garantie → le credit peut valoir le coup (rare)
- Comparaison avec la bourse : ne le faites que si vous auriez investi de toute facon, et uniquement avec un horizon long terme (pas 18 mois)

### Question 6 : Regle de decision generaliste

**Payer comptant est preferable quand :**
- TAEG > rendement de votre epargne "sans risque" (livret, compte epargne)
- Vous n'avez pas d'autre usage productif de l'argent
- Le montant est accessible sans vider le fonds d'urgence

**Le credit peut se justifier quand :**
- TAEG < rendement garanti disponible ailleurs (rare en pratique)
- Le bien finance genere des revenus superieurs au cout du credit (levier productif)
- Vous maintenez votre fonds d'urgence intact et ne prenez pas de risque supplementaire
- La depense est necessaire et vous n'avez genuinement pas le capital disponible

> **Conclusion globale** : dans la majorite des achats de consommation (voiture, electronique, meubles), payer comptant est preferable des lors que le TAEG depasse le taux sans risque — ce qui est le cas avec 9,9 % vs 2,5 %. La "facilite" du credit cache toujours un cout reel.
