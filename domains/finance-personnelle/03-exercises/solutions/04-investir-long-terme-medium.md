# Solutions — Module 04 : Investir simplement et sur le long terme (niveau intermédiaire)

> ⚠️ Ces solutions sont des corrigés modèles. Vos chiffres peuvent différer légèrement selon la méthode (annuelle vs mensuelle, arrondis). L'important est la démarche et l'ordre de grandeur.
>
> **Disclaimer** : contenu éducatif, rendements **illustratifs**, aucun rendement garanti. **Pas un conseil financier personnalisé.**

---

## Solution Exercice 1 — Rééquilibrer un portefeuille qui a dérivé (Hakim)

### Question 1 : valeur des poches après un an

- Actions : 70 000 × 1,25 = **87 500 €**
- Obligations : 30 000 × 1,02 = **30 600 €**
- Total : **118 100 €**

### Question 2 : nouvelle allocation et dérive

- Actions : 87 500 / 118 100 = **74,1 %**
- Obligations : 30 600 / 118 100 = **25,9 %**

La poche actions a dérivé de 70 % à 74,1 % (**+4,1 points**) ; les obligations de 30 % à 25,9 %. La hausse des actions a déformé l'allocation : le portefeuille est désormais **plus risqué** que la cible choisie.

### Question 3 : montant à vendre pour revenir à 70/30

Cible actions = 118 100 × 0,70 = 82 670 €. Il faut vendre : 87 500 − 82 670 = **~4 830 € d'actions**, et les replacer en obligations (qui passent à ~35 430 €). En pratique, on peut aussi rééquilibrer en **dirigeant les nouveaux versements** vers les obligations, pour éviter de vendre (et limiter la fiscalité).

### Question 4 : mécanisme et discipline anti-biais

Ici le marché a **monté** : le rééquilibrage fait **vendre une partie de la classe qui a le plus monté** (actions) pour racheter celle qui a le moins monté (obligations). C'est mécaniquement "vendre haut, acheter bas". Lors d'une baisse, le mécanisme s'inverse : on rachète les actions "en solde". Cette règle automatique **neutralise les biais** (Module 05) : on ne se laisse pas griser par la hausse (qui pousserait à charger en actions au plus haut), ni paniquer par la baisse. La décision est prise par la règle, pas par l'émotion.

### Question 5 : fréquence et seuil

Le module recommande de rééquilibrer **environ une fois par an**, et seulement si une classe d'actifs s'est écartée de la cible de plus de **~5 points** (sinon, ne rien faire — éviter la suractivité et les frais).

---

## Solution Exercice 2 — Tester l'intuition de Markowitz

### Question 1 : rendement moyen de chaque actif seul

- Actions : (20 + 8 − 25) / 3 = **+1,0 %**
- Obligations : (1 + 3 + 4) / 3 = **+2,67 %**

### Question 2 : portefeuille 70/30 par scénario

Rendement = 0,70 × actions + 0,30 × obligations :
- Bon : 0,70×20 + 0,30×1 = 14 + 0,3 = **+14,3 %**
- Moyen : 0,70×8 + 0,30×3 = 5,6 + 0,9 = **+6,5 %**
- Mauvais : 0,70×(−25) + 0,30×4 = −17,5 + 1,2 = **−16,3 %**

Rendement moyen du portefeuille : (14,3 + 6,5 − 16,3) / 3 = **+1,5 %**.

### Question 3 : amplitude (proxy du risque)

- Actions seules : de +20 % à −25 % -> amplitude = **45 points**.
- Portefeuille 70/30 : de +14,3 % à −16,3 % -> amplitude = **30,6 points**.

Le portefeuille mixte **réduit l'amplitude** (30,6 < 45), donc le risque (mesuré ici par la dispersion), tout en gardant un rendement moyen comparable à celui des actions seules (1,5 % vs 1,0 %). C'est exactement l'effet de diversification.

### Question 4 : rôle d'amortisseur des obligations (scénario Mauvais)

En cas de krach :
- 100 % actions : **−25 %**.
- Portefeuille 70/30 : **−16,3 %**.

Les obligations (+4 % ce scénario-là) **amortissent** la perte de 25 % à 16,3 %, soit **8,7 points de perte en moins**. C'est ce coussin qui, en phase de baisse, permet aussi de rééquilibrer (vendre des obligations stables pour racheter des actions en solde).

### Question 5 : reformulation de Markowitz

En combinant des actifs dont les rendements ne bougent pas de la même façon (actions volatiles + obligations stables), on **réduit le risque global du portefeuille sans sacrifier proportionnellement le rendement espéré**.

> Note : cet exemple est volontairement simplifié et illustratif (3 scénarios). Le principe est robuste, mais les rendements réels ne sont ni connus à l'avance ni garantis.

---

## Solution Exercice 3 — Allocation "3 fonds" et trois profils (Inès, 35 ans)

### Partie A : allocation équilibrée proposée

Allocation équilibrée : **60 % actions monde développé / 20 % actions émergentes / 20 % obligations**.

Justification : à 35 ans avec un horizon de 30 ans, Inès peut supporter une forte exposition actions (80 % au total, diversifiées géographiquement) pour capter la prime de risque long terme ; les 20 % d'obligations amortissent la volatilité et servent de réserve de rééquilibrage. C'est un point de départ, à ajuster selon sa tolérance réelle au risque.

### Partie B : rendement net pondéré et projection à 30 ans

Rendement net = (part_dev × 7 % + part_em × 8 % + part_oblig × 3 %) − TER 0,20 %.

| Allocation | Rendement net pondéré | Capital à 30 ans (10 000 € + 350 €/mois) |
|---|---|---|
| Dynamique (100 % dev) | 7,00 − 0,20 = **6,80 %** | ≈ **466 500 €** |
| Équilibrée (60/20/20) | 6,40 − 0,20 = **6,20 %** | ≈ **414 400 €** |
| Prudente (30/10/60) | 4,70 − 0,20 = **4,50 %** | ≈ **298 900 €** |

(Capitalisation mensuelle, via `capital_final_mensuel`.)

### Partie C : comparaison critique

L'écart entre dynamique (~466 500 €) et prudente (~298 900 €) est important (~167 000 €). Mais cet écart **ne suffit pas** à décider de tout mettre en actions, car le calcul ne montre **que le rendement espéré**, pas le risque. Dimensions absentes :
- **Volatilité** : le profil 100 % actions peut chuter de 30-50 % sur une mauvaise année ; il faut pouvoir le **supporter sans vendre** (Module 05 : aversion à la perte).
- **Comportement** : une allocation "trop chaude" pour son tempérament conduit souvent à paniquer et vendre au creux — détruisant l'avantage théorique.
- **Horizon réel et besoins** : si une partie du capital pourrait être nécessaire avant 5-10 ans, l'exposition actions doit baisser.

La bonne allocation est celle dont on **tiendra le cap** pendant 30 ans, pas celle qui maximise le rendement sur tableur.

---

## Résumé des enseignements clés (medium)

1. Le **rééquilibrage** ramène l'allocation à sa cible et impose mécaniquement "vendre haut / acheter bas" — une discipline anti-biais (≈ 1 fois/an, seuil ~5 points).
2. La **diversification** (Markowitz) réduit l'amplitude (risque) sans sacrifier proportionnellement le rendement espéré ; les obligations amortissent les krachs.
3. Une allocation "3 fonds" se calibre selon l'**horizon et la tolérance au risque**.
4. Un écart de rendement espéré **ne suffit pas** à choisir une allocation : il faut intégrer volatilité, comportement et horizon.
5. Tous les rendements sont **illustratifs** ; aucun n'est garanti.
