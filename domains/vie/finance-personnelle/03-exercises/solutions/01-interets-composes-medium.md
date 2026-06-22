# Solutions — Interets composes et valeur du temps (Module 01, niveau intermediaire)

> **Disclaimer** : solutions educatives a titre de reference. Les taux sont **illustratifs** ; aucun rendement n'est promis. Ce contenu est educatif et **ne remplace pas un conseil financier personnalise**.
>
> **Conseil** : tentez les exercices avant de lire. Verifiez vos calculs avec `02-code/01-interets-composes.py` (`capital_final(P, r, t, M)`).

---

## Solution Exercice 1 — Rendement nominal, reel et inflation

### Question 1 : rendement reel exact (formule de Fisher)

```
r_reel = (1 + r_nominal) / (1 + inflation) - 1
       = (1 + 0,06) / (1 + 0,03) - 1
       = 1,06 / 1,03 - 1
       = 0,02913 = 2,91 %
```

L'approximation par soustraction (6 % - 3 % = 3 %) donne **3,00 %**. L'ecart est de **0,09 point** (3,00 - 2,91). L'approximation par soustraction **surestime legerement** le rendement reel. Elle reste acceptable pour une estimation rapide quand les taux sont bas, mais l'ecart grandit quand les taux montent.

### Question 2 : capital nominal vs pouvoir d'achat sur 30 ans (10 000 €)

- **Capital final nominal** (6 %, 30 ans) : `capital_final(10000, 0.06, 30)` ≈ **60 226 €**
- **Capital en pouvoir d'achat constant** (taux reel 2,913 %, 30 ans) ≈ **23 934 €**

Commentaire : le chiffre nominal (60 226 €) impressionne, mais une fois corrige de l'erosion monetaire, le pouvoir d'achat reel ne represente que **~24 000 € d'aujourd'hui**. L'inflation "mange" silencieusement une grande partie de la croissance affichee. Toujours se demander : ce chiffre est-il nominal ou reel ?

### Question 3 : livret a rendement reel negatif

```
r_reel = 1,02 / 1,03 - 1 = -0,97 %  (negatif)
```

Le rendement reel est **negatif** : meme si le solde nominal augmente, le **pouvoir d'achat diminue**. Pour 10 000 € laisses 10 ans a ce taux reel : `capital_final(10000, -0.0097, 10)` ≈ **9 074 €** de pouvoir d'achat (en euros d'aujourd'hui). On a "perdu" environ 9 % de pouvoir d'achat malgre un solde nominal en hausse.

> C'est exactement le role d'un fonds d'urgence (Module 02) : sa fonction est la securite et la liquidite, pas le rendement. Sur le long terme, garder tout son patrimoine en cash erode le pouvoir d'achat.

### Question 4 : pourquoi un rendement reel dans les projections

Parce qu'un rendement reel (ex. 5 %) integre deja l'inflation : il exprime la projection en pouvoir d'achat constant, ce qui est plus honnete et evite de surpromettre avec des chiffres nominaux gonfles qui ne refletent pas le pouvoir d'achat futur.

---

## Solution Exercice 2 — Isoler chaque levier (taux 7 %, mensuel)

### Question 1 : capitaux finaux et totaux verses

| Epargnant | Detail | Total verse | Capital final |
|---|---|---|---|
| Karim | 20 000 € initial, 0 €/mois, 25 ans | 20 000 € | ≈ **114 508 €** |
| Lina | 0 € initial, 150 €/mois, 25 ans | 45 000 € | ≈ **121 511 €** |
| Sofia | 0 € initial, 150 €/mois, 30 ans | 54 000 € | ≈ **182 996 €** |

(Valeurs via `capital_final()`, capitalisation mensuelle.)

### Question 2 : Karim vs Lina

Non, ils ne versent pas la meme chose : Karim verse **20 000 €** (en une fois, tot), Lina verse **45 000 €** (etale sur 25 ans). Pourtant Lina finit avec un capital legerement superieur (~121 500 € vs ~114 500 €).

Mais le point cle est ailleurs : **Karim multiplie son apport par ~5,7** (114 508 / 20 000) alors que Lina ne multiplie le sien que par ~2,7 (121 511 / 45 000). Le capital initial de Karim a beneficie de **25 ans pleins de croissance** sur chaque euro, tandis que les versements de Lina arrivent progressivement et ont en moyenne moins de temps pour croitre. Lecon : un euro place tot (capital initial) "travaille" plus longtemps qu'un euro verse plus tard.

### Question 3 : Lina vs Sofia (effet des 5 ans)

A versement mensuel identique (150 €), les **5 annees supplementaires** de Sofia apportent :
- Gain en euros : 182 996 - 121 511 = **+61 485 €**
- Gain en % : +61 485 / 121 511 ≈ **+50,6 %**
- Surplus verse par Sofia : 54 000 - 45 000 = **+9 000 €** seulement

Sofia a verse **9 000 € de plus** et obtient **61 485 € de plus**. Le supplement de capital (~61 500 €) est presque 7 fois le supplement verse (~9 000 €). C'est la signature de l'exponentielle : les annees du debut sont celles qui ont le plus de temps pour composer.

### Question 4 : tableau et hierarchie

| Epargnant | Total verse | Capital final | Multiplicateur (capital÷verse) |
|---|---|---|---|
| Karim | 20 000 € | 114 508 € | **×5,7** |
| Lina | 45 000 € | 121 511 € | ×2,7 |
| Sofia | 54 000 € | 182 996 € | ×3,4 |

**Hierarchie des leviers** : (1) le **temps** domine — Karim transforme un petit apport en gros capital grace a 25 ans de croissance ; Sofia surpasse Lina en ajoutant 5 ans. (2) Le **capital initial** est puissant car chaque euro y profite de toute la duree. (3) La **regularite** (versement mensuel) construit le capital de facon fiable, mais les euros verses tard ont moins de temps. Conclusion coherente avec le module : *commencer tot bat verser plus tard*.

---

## Solution Exercice 3 — Du capital cible au versement requis

### Question 1 : inversion de la formule

Valeur future d'une annuite : `A = M × [((1+i)^N - 1) / i]`, donc :

```
M = A × i / ((1+i)^N - 1)
```

Avec A = 200 000 €, i = 0,06/12 = 0,005, N = 360 :

```
(1,005)^360 ≈ 6,0226
M = 200 000 × 0,005 / (6,0226 - 1)
  = 1 000 / 5,0226
  ≈ 199,10 €/mois
```

Versement requis : **~199 €/mois** (environ 200 €).

### Question 2 : verification

`capital_final(0, 0.06, 30, 199.1)` ≈ **200 000 €**. Coherent.

### Question 3 : horizon reduit a 20 ans

```
N = 240 ; (1,005)^240 ≈ 3,3102
M = 200 000 × 0,005 / (3,3102 - 1) = 1 000 / 2,3102 ≈ 432,86 €/mois
```

Versement requis sur 20 ans : **~433 €/mois**, contre ~199 €/mois sur 30 ans. C'est une augmentation de **+117 %** (plus du double) pour seulement 10 ans de moins. Perdre 10 ans ne double pas l'effort : il le **plus que double**, a cause de la croissance exponentielle perdue.

### Question 4 : taux plus prudent (4 % au lieu de 6 %), 30 ans

```
i = 0,04/12 = 0,003333 ; N = 360 ; (1,003333)^360 ≈ 3,3135
M = 200 000 × 0,003333 / (3,3135 - 1) = 666,7 / 2,3135 ≈ 288,16 €/mois
```

Versement requis a 4 % : **~288 €/mois**, contre ~199 €/mois a 6 %, soit **+45 %**.

**Conclusion** : un plan d'epargne est tres sensible a l'hypothese de rendement. Baisser l'hypothese de 6 % a 4 % exige 45 % d'effort en plus pour le meme objectif. D'ou l'importance d'utiliser des hypotheses **prudentes** et de ne jamais traiter un rendement illustratif comme garanti.

---

## Resume des enseignements cles (medium)

1. Le rendement **reel** (net d'inflation, formule de Fisher) mesure le vrai pouvoir d'achat — un solde nominal en hausse peut cacher une perte reelle.
2. Le **temps** est le levier dominant : un euro place tot bat un euro verse tard ; gagner 5 ans peut ajouter +50 % de capital pour un effort marginal.
3. On peut **inverser** la formule pour passer d'un objectif au versement requis — competence cle pour un plan realiste.
4. Un plan est **tres sensible** a l'hypothese de rendement et a l'horizon : d'ou la prudence des hypotheses.
5. Tous ces chiffres sont **illustratifs** : les rendements reels varient, ne sont pas garantis, et dependent de choix risques.
