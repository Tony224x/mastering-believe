# Exercices — Capstone : extensions du plan financier (niveau avancé)

> **Niveau** : Avancé | **Temps estimé** : 90-120 min
>
> **Matière première** : Modules 01 à 06 intégrés + simulateur `02-code/07-capstone-plan-financier.py`. **Prérequis** : capstone easy réalisé. Ces exercices poussent le plan vers les **arbitrages et les limites** — pas de redite.
>
> **Disclaimer** : exercices éducatifs, profils fictifs. Rendements, taux et fiscalité sont **illustratifs et génériques**. Ce contenu est éducatif et **ne constitue pas un conseil financier, fiscal ou en investissement personnalisé**. Les performances passées ne préjugent pas du futur ; risque de perte en capital.

---

## Exercice 1 — Optimiser la séquence du plan d'Alex (arbitrage dette / fonds / investissement)

### Objectif
Intégrer tous les modules dans une séquence optimisée et chiffrée, en arbitrant le timing entre fonds d'urgence, remboursement de dette et montée en puissance de l'investissement.

### Consigne

**Profil Alex** (rappel) : revenu 3 200 €/mois, dépenses annuelles ~22 800 €, fonds d'urgence 1 200 € (objectif 4 100 €), **crédit auto 8 000 € à 4,5 %** (mensualité 230 €), capacité d'épargne disponible ~400 €/mois après optimisation budget. Rendement net investissement illustratif 5 %.

1. **Séquence proposée.** Construisez la séquence des 3 premières années : (a) compléter le fonds d'urgence, (b) traiter le crédit auto, (c) monter l'investissement à 400 €/mois. Justifiez l'ordre avec : Module 02 (fonds d'urgence d'abord), Module 03 (arbitrage par le taux — 4,5 % de la dette vs ~5 % espéré de l'investissement). L'écart étant faible, discutez la position raisonnable du capstone (partage 50/50 possible).
2. **Projection comparée à 30 ans.** Calculez le capital à 30 ans pour trois trajectoires (5 % net) :
   - **T1** : 280 €/mois pendant 30 ans (plan de base).
   - **T2** : 280 €/mois pendant 3 ans, puis 400 €/mois pendant 27 ans (après libération du crédit auto).
   - **T3** : 400 €/mois pendant 30 ans (hypothétique : si la capacité était dispo dès le départ).
   Présentez le tableau et commentez l'écart entre T1, T2 et T3.
3. **Arbitrage dette à 4,5 % vs investissement à 5 %.** Le capstone propose un partage 50/50 quand l'écart est faible. Pourquoi ne pas trancher de façon dogmatique ici ? Quels facteurs **non chiffrables** (psychologie de la dette, certitude vs risque) entrent en jeu (Modules 03 et 05) ?
4. **Cohérence du plan.** Vérifiez que la séquence respecte les "erreurs classiques à éviter" du Module 07 (fonds d'urgence avant investissement ; ne pas rembourser toute dette avant d'investir si le taux est bas ; simplicité). Citez-en au moins deux et montrez que le plan les évite.

### Critères de réussite
- [ ] Séquence des 3 ans construite et justifiée (Modules 02 et 03)
- [ ] Projection T1/T2/T3 à 30 ans calculée et commentée
- [ ] L'arbitrage 4,5 % vs 5 % est traité de façon non dogmatique, avec facteurs non chiffrables
- [ ] Au moins deux "erreurs classiques" du Module 07 sont citées et le plan montre qu'il les évite

---

## Exercice 2 — Du plan nominal au plan réel : inflation, fiscalité et horizon d'indépendance

### Objectif
Affiner le plan d'Alex en intégrant l'inflation (rendement réel vs nominal) et le principe de fiscalité, puis calculer un horizon d'indépendance financière réaliste — la limite que beaucoup de plans ignorent.

### Consigne

Alex investit **400 €/mois** (capacité après extinction du crédit), capital de départ 0 €. Dépenses annuelles **22 800 €**. Cible d'indépendance totale = **25 × dépenses = 570 000 €** (règle des 4 %).

1. **Horizon d'indépendance totale.** Estimez en combien d'années Alex atteint 570 000 € avec 400 €/mois à 5 % réel. Puis avec **600 €/mois** (s'il pousse l'effort). Combien d'années l'effort supplémentaire fait-il gagner ?
2. **Nominal vs réel.** Le capstone recommande d'"utiliser un rendement réel (nominal − inflation)". Comparez la projection à 30 ans de 400 €/mois (a) à **7 % nominal**, et (b) en **pouvoir d'achat constant** (rendement réel ≈ 4,9 % si l'inflation est 2 %). Pourquoi le chiffre nominal (a) est-il **trompeur** pour planifier l'indépendance, qui se mesure en dépenses réelles ?
3. **Fiscalité (principe général).** Le capstone et le Module 05 disent qu'un avantage fiscal ne justifie pas des frais élevés. Énoncez, sans entrer dans les règles d'un pays, les 3-4 vérifications à faire avant de loger l'investissement d'Alex dans une enveloppe fiscale. En quoi la fiscalité des retraits réduit-elle le taux de retrait soutenable réel (Module 06) ?
4. **Plan final révisé (1 page).** Rédigez la version "réaliste" du plan d'Alex en intégrant : projection en **rendement réel**, cible d'indépendance, horizon estimé, rappel des limites (inflation, fiscalité, séquence des rendements) et **disclaimer**. Le plan doit rester lisible en moins de 2 minutes.

### Critères de réussite
- [ ] Horizon vers 570 000 € estimé à 400 €/mois et à 600 €/mois, gain d'années chiffré
- [ ] Comparaison nominal (7 %) vs réel (≈ 4,9 %) à 30 ans, et explication du caractère trompeur du nominal
- [ ] 3-4 vérifications d'enveloppe fiscale énoncées (génériques) + effet de la fiscalité sur le taux de retrait
- [ ] Plan final révisé en 1 page, en rendement réel, avec limites et disclaimer présents
