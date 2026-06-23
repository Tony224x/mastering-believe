# Solutions — Interets composes et valeur du temps (Module 01, niveau avance)

> **Disclaimer** : solutions educatives a titre de reference. Les taux et scenarios sont **illustratifs**, jamais des previsions. Ce contenu est educatif et **ne constitue pas un conseil financier**. Aucun rendement n'est garanti ; tout investissement comporte un risque de perte en capital.

---

## Solution Exercice 1 — Arbitrage : prime initiale ou epargne mensuelle ?

Donnees : horizon 25 ans, taux illustratif central 6 %/an, capitalisation mensuelle.

### Question 1 : options A et B

- **Option A** — 15 000 € investis en une fois + 250 €/mois pendant 25 ans :
  `capital_final(15000, 0.06, 25, 250)` ≈ **240 223 €**
- **Option B** — 15 000 € au livret a 2,5 % + 250 €/mois investis a 6 % :
  `capital_final(15000, 0.025, 25)` + `capital_final(0, 0.06, 25, 250)`
  ≈ 27 906 € + 173 348 € ≈ **201 254 €**

Ecart A - B ≈ **38 969 €**. La "prudence" de l'option B (laisser 15 000 € dormir a 2,5 % au lieu de les investir) coute environ **39 000 €** sur 25 ans — soit plus du double de la prime initiale. Garder une grosse somme en cash sur un horizon long a un cout d'opportunite eleve.

### Question 2 : option C (etalement de la prime)

Decomposition : 30 versements de 500 € (en plus des 250 €/mois) pendant les 30 premiers mois, puis 250 €/mois jusqu'a 65 ans.
- Les 500 €/mois pendant 30 mois (2,5 ans) atteignent ≈ `capital_final(0, 0.06, 2.5, 500)` ≈ 38 800 € au mois 30, puis croissent pendant les 22,5 ans restants : ≈ 38 800 × (1,005)^270 ≈ **~149 000 €**.
- Le flux de 250 €/mois sur 25 ans ≈ **173 348 €** (deja vu).
- Total option C ≈ **235 298 €**.

Comparaison : A ≈ 240 223 € vs C ≈ 235 298 €, soit un ecart d'environ **4 925 €** en faveur de A. **Etaler la prime sur 30 mois coute ~5 000 €** par rapport au lump sum : modeste mais reel, car une partie de la prime perd 2,5 ans de croissance. L'etalement reduit le risque ressenti de "mal timer", mais a un cout.

### Question 3 : "et si le marche chute juste apres ?"

Ce que la theorie du module **permet d'affirmer** :
- Sur un horizon long (25 ans), le temps de croissance est l'atout principal ; chaque mois hors du marche est un mois de composition perdu (cout de l'attente, §5 du module).
- Historiquement, statistiquement, investir tot a souvent surpasse l'attente — mais "souvent" n'est pas "toujours".

Ce que la theorie **ne permet PAS d'affirmer** :
- Qu'il n'y aura pas de baisse apres l'investissement (personne ne sait timer le marche).
- Qu'un rendement de 6 % est garanti — c'est une hypothese illustrative ; le marche peut etre negatif plusieurs annees.

Role de l'horizon de 25 ans : plus l'horizon est long, plus une baisse precoce a de temps pour etre absorbee/recuperee. Le risque de timing est donc **attenue par la duree**, sans etre annule. (Le risque de sequence des rendements, plus aigu en phase de retrait, est traite au Module 06.)

### Question 4 : scenario defavorable a 3 %

`capital_final(15000, 0.03, 25, 250)` ≈ **143 227 €**, contre ≈ 240 223 € a 6 %. Le capital final **chute de ~97 000 €** (-40 %) quand le rendement passe de 6 % a 3 %.

Ce que cela revele : une projection a **scenario unique** est fragile. Le meme plan (memes versements, meme duree) produit de 143 000 € a 240 000 € selon une seule hypothese. D'ou la necessite de presenter une fourchette (voir Exercice 2) et de ne jamais vendre un chiffre unique comme une certitude.

---

## Solution Exercice 2 — Projection multi-scenarios

Donnees : 0 € initial, 300 €/mois, 30 ans, capitalisation mensuelle.

### Question 1 : trois scenarios de rendement

| Scenario | Taux | Capital final a 30 ans | Multiplicateur (capital÷verse) |
|---|---|---|---|
| Defavorable | 3 % | ≈ **174 821 €** | ×1,62 |
| Central | 6 % | ≈ **301 355 €** | ×2,79 |
| Favorable | 8 % | ≈ **447 108 €** | ×4,14 |

Le meme effort produit un capital final allant de ~175 000 € a ~447 000 € selon l'hypothese — un facteur 2,5 entre les bornes.

### Question 2 : versements vs interets

Total verse : 300 × 12 × 30 = **108 000 €** (identique dans les trois cas).

- **Scenario central (6 %)** : interets composes = 301 355 - 108 000 = 193 355 €, soit **64,2 %** du capital final ; les versements ne representent que **35,8 %**.
- **Scenario favorable (8 %)** : interets = 447 108 - 108 000 = 339 108 €, soit **75,8 %** du capital final.

Lecon : sur 30 ans, la majorite du capital final provient des **interets composes**, pas des versements — et cette part grandit avec le taux.

### Question 3 : pourquoi une fourchette, pas un chiffre unique

Presenter uniquement le scenario a 8 % (447 000 €) serait trompeur : ce taux est une hypothese favorable, non une promesse. Les marches ne garantissent aucun rendement, et **les performances passees ne prejugent pas des performances futures**. Une fourchette (175 000 € a 447 000 €) communique honnetement l'incertitude, permet a l'epargnant de planifier sur le scenario prudent, et evite la deception (ou les decisions hatives) si le rendement reel est inferieur. La posture honnete consiste a montrer la dispersion, pas a cueillir le meilleur cas.

### Question 4 : sensibilite a la duree (scenario central 6 %)

| Duree | Capital final (6 %) |
|---|---|
| 25 ans | ≈ 207 898 € |
| 30 ans | ≈ 301 355 € |
| 35 ans | ≈ 427 413 € |

- Passer de 25 a 30 ans : **+93 456 €**
- Passer de 30 a 35 ans : **+126 059 €**

L'effet n'est **pas symetrique** : les 5 dernieres annees (30→35) ajoutent plus que les precedentes (25→30), car a 30 ans le capital est deja important et chaque annee supplementaire compose sur une base plus grande. C'est exactement l'accélération exponentielle vue au Module 01 (le gain 20-40 ans depasse le gain 0-20 ans). Gagner 5 ans en fin de course n'a donc pas la meme valeur que les perdre au debut — mais les annees du **debut** restent les plus precieuses car elles beneficient du plus long temps de composition.

---

## Resume des enseignements cles (hard)

1. Sur un horizon long, **investir tot** (lump sum) bat generalement garder du cash "par prudence" — le cout d'opportunite se chiffre en dizaines de milliers d'euros.
2. La theorie permet d'affirmer l'effet du temps, **pas** le timing ni un rendement garanti : posture honnete obligatoire.
3. Une projection a **scenario unique** est fragile ; toujours montrer une **fourchette** (defavorable / central / favorable).
4. Sur 30 ans, la majorite du capital final vient des **interets composes**, pas des versements.
5. L'effet de la duree est **asymetrique et exponentiel** ; les annees du debut comptent le plus.
6. Tous les taux et scenarios sont **illustratifs** — aucun rendement n'est promis ; risque de perte en capital reel.
