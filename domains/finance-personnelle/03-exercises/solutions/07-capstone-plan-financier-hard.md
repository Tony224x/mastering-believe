# Solution — Capstone : extensions du plan financier (niveau avancé)

> Corrigés modèles pour le profil fictif "Alex". Rendements, taux et fiscalité **illustratifs et génériques**. Ce document est **purement éducatif** et ne constitue pas un conseil financier, fiscal ou en investissement personnalisé. Les performances passées ne préjugent pas du futur ; risque de perte en capital.
>
> *Note : capitalisation mensuelle (`02-code/07-capstone-plan-financier.py`).*

---

## Solution Exercice 1 — Optimiser la séquence du plan d'Alex

### Question 1 : séquence des 3 premières années

| Phase | Action | Justification |
|---|---|---|
| **Mois 1-8** | Compléter le fonds d'urgence (1 200 € -> 4 100 €, ~350 €/mois) | Module 02 : le filet précède l'investissement. Sans lui, un imprévu force à vendre au mauvais moment. |
| **Mois 9-35** | Investir ~280 €/mois **+** continuer le crédit auto (230 €/mois jusqu'à extinction) | Module 03 : le crédit auto est à **4,5 %**, proche du rendement espéré (~5 %). L'écart étant faible, on n'accélère pas forcément le remboursement ; on investit en parallèle. |
| **Mois 36+** | Le crédit soldé libère 230 €/mois -> investissement porté à ~400 €/mois | Montée en puissance une fois la dette éteinte. |

Position raisonnable du capstone : quand dette (4,5 %) et investissement (5 %) sont si proches, un **partage 50/50** entre remboursement anticipé et investissement est défendable — il n'y a pas de réponse unique.

### Question 2 : projection comparée à 30 ans (5 % net)

| Trajectoire | Description | Capital à 30 ans |
|---|---|---|
| **T1** | 280 €/mois pendant 30 ans | ≈ **228 305 €** |
| **T2** | 280 €/mois pendant 3 ans, puis 400 €/mois pendant 27 ans | ≈ **308 817 €** |
| **T3** | 400 €/mois pendant 30 ans (hypothétique) | ≈ **326 150 €** |

Commentaire : T2 (la trajectoire réaliste, où le crédit soldé libère de la capacité) atteint ~309 000 €, soit ~80 000 € de plus que T1 et seulement ~17 000 € de moins que l'idéal T3. La libération de la dette au mois 36 a un fort effet cumulé sur 27 ans. Conclusion : ce n'est pas dramatique de commencer à 280 € si on monte ensuite à 400 € — la régularité et la montée en puissance comptent plus que le point de départ exact.

### Question 3 : arbitrage 4,5 % (dette) vs 5 % (investissement), non dogmatique

L'écart (0,5 point) est dans la marge d'incertitude : le 4,5 % de la dette est **certain**, le 5 % de l'investissement est **espéré et risqué** (Module 03 : on ne compare pas un rendement certain et un rendement risqué à égalité). Facteurs non chiffrables (Modules 03 et 05) :
- **Psychologie de la dette** : certaines personnes dorment mieux sans dette ; éteindre le crédit a une valeur émotionnelle réelle qui soutient la discipline globale.
- **Certitude vs risque** : rembourser garantit 4,5 % ; investir peut faire mieux ou moins bien.
- **Tolérance au risque et horizon** : un horizon long favorise l'investissement, mais pas au point de l'imposer ici.

D'où le partage 50/50 du capstone : ni tout l'un, ni tout l'autre — un compromis adapté au profil.

### Question 4 : "erreurs classiques" du Module 07 évitées

- **"Investir avant le fonds d'urgence"** : évité — la phase 1 complète le fonds d'urgence d'abord.
- **"Rembourser toute dette avant d'investir"** : évité — on n'attend pas l'extinction du crédit (taux bas) pour commencer à investir ; on arbitre par le taux.
- **"Plan trop complexe"** : respecté — allocation 3 fonds, virement automatique, révision annuelle. Simplicité = durabilité.

---

## Solution Exercice 2 — Du plan nominal au plan réel

### Question 1 : horizon d'indépendance totale (cible 570 000 €, 5 % réel)

- À **400 €/mois** : atteinte de 570 000 € en ≈ **40 ans**.
- À **600 €/mois** : atteinte en ≈ **33 ans**.

Pousser l'effort de 400 à 600 €/mois (+200 €) fait **gagner ~7 ans** sur l'indépendance totale — encore une illustration du levier "taux d'épargne" (Module 06).

### Question 2 : nominal vs réel (400 €/mois, 30 ans)

- **(a) 7 % nominal** : ≈ **467 781 €**.
- **(b) Pouvoir d'achat constant** (rendement réel ≈ 4,9 % si inflation 2 %) : ≈ **320 574 €** en euros d'aujourd'hui.

Le chiffre nominal (467 781 €) est **trompeur** pour planifier l'indépendance : la cible d'indépendance (25 × dépenses) se mesure en **dépenses réelles**, qui augmentent avec l'inflation. Un capital nominal de 467 781 € dans 30 ans n'achètera "que" ~320 574 € de pouvoir d'achat d'aujourd'hui. Planifier en **rendement réel** évite de croire qu'on est plus proche de l'indépendance qu'on ne l'est réellement (lien avec l'exercice hard du Module 01).

### Question 3 : enveloppe fiscale et taux de retrait

Vérifications génériques avant de loger l'investissement dans une enveloppe fiscale (Module 05/07, sans règles d'un pays) :
1. **Coût total net** : l'avantage fiscal ne doit pas être annulé par des frais de gestion élevés — comparer le net réel.
2. **Conditions et pénalités de retrait** : compatibles avec la liquidité dont Alex pourrait avoir besoin ?
3. **Horizon de blocage** : l'avantage est-il conditionné à une durée minimale cohérente avec son horizon ?
4. **Règles à jour de sa juridiction** : la fiscalité change ; consulter les sources officielles (régulateur, administration fiscale).

Effet sur le taux de retrait (Module 06) : la fiscalité des retraits **réduit le revenu net disponible** par euro retiré, donc le taux de retrait soutenable réel est inférieur au taux brut (4 % brut peut correspondre à moins de 4 % net après impôt). Les simulations de la règle des 4 % n'incluent ni fiscalité ni frais — il faut donc une marge.

### Question 4 : plan final révisé (1 page, exemple)

---

**PLAN FINANCIER RÉALISTE — Alex, 35 ans** — *Document éducatif, pas un conseil personnalisé.*

- **Photographie** : revenu 3 200 €/mois, dépenses ann. ~22 800 €, fonds d'urgence 1 200 € (cible 4 100 €), crédit auto 8 000 € @ 4,5 %.
- **Séquence** : (1) compléter le fonds d'urgence (~8 mois) ; (2) investir ~280 €/mois + servir le crédit ; (3) crédit soldé (~mois 36) -> investir ~400 €/mois.
- **Allocation** : 3 fonds indiciels (ex. 60 % monde dév. / 20 % émergents / 20 % obligations), **TER cible < 0,20 %**, rééquilibrage 1×/an, enveloppe fiscale à vérifier (coût net, blocage, règles à jour).
- **Projection en rendement RÉEL (≈ 4,9 %, pouvoir d'achat constant), 400 €/mois** : ~30 ans -> ~320 000 € réels ; indépendance partielle (10× dépenses ≈ 228 000 € réels) atteinte plus tôt.
- **Cible d'indépendance totale** : 25 × 22 800 = 570 000 € (réels) ; horizon ~40 ans à 400 €/mois, ~33 ans à 600 €/mois.
- **Limites & disclaimer** : projections en rendement réel **illustratif**, non garanti ; règle des 4 % calibrée US/30 ans ; fiscalité et frais réduisent le taux de retrait réel ; risque de séquence des rendements en phase de retrait ; performances passées ≠ futures ; risque de perte en capital. **Réviser chaque année.**

---

## Résumé des enseignements clés (hard)

1. La **libération d'une dette** (timing) a un fort effet cumulé : monter de 280 à 400 €/mois après le crédit (T2) rattrape presque l'idéal (T3).
2. L'arbitrage dette/investissement à taux proches se traite **sans dogmatisme** : certitude vs risque + psychologie de la dette -> partage raisonnable.
3. Planifier en **rendement réel** (et non nominal) évite de surestimer sa proximité de l'indépendance — la cible se mesure en dépenses réelles.
4. Une **enveloppe fiscale** se choisit sur le **net réel** (frais inclus) ; fiscalité et frais réduisent le taux de retrait soutenable.
5. Le plan final reste **simple, lisible, révisé chaque année**, et accompagné de **disclaimers honnêtes**.
