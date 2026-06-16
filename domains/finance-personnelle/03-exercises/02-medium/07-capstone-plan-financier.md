# Exercices — Capstone : extensions du plan financier (niveau intermédiaire)

> **Niveau** : Intermédiaire | **Temps estimé** : 60-75 min
>
> **Matière première** : Module 07 + simulateur `02-code/07-capstone-plan-financier.py`. **Prérequis** : avoir réalisé le capstone (exercices easy). Ces exercices **étendent** le plan d'Alex — ils ne le refont pas.
>
> **Disclaimer** : exercices éducatifs, profil fictif. Les rendements sont **illustratifs** et ne garantissent aucun résultat. Ce contenu est éducatif et **ne constitue pas un conseil financier personnalisé**.

---

> **Rappel du profil "Alex"** (du capstone easy) : 35 ans, revenu net 3 200 €/mois, dépenses annuelles ~22 800 €, investit 280 €/mois (puis ~400 €/mois après extinction du crédit auto), rendement net illustratif 5 %/an, capital de départ 0 €.

## Exercice 1 — Analyse de sensibilité au rendement (3 scénarios)

### Objectif
Transformer la projection à point unique du capstone en une **fourchette de scénarios**, pour présenter un plan honnête face à l'incertitude de marché.

### Consigne

Alex investit 280 €/mois, capital de départ 0 €.

1. Calculez la projection à 10, 20 et 30 ans pour trois rendements illustratifs : **défavorable 3 %**, **central 5 %**, **favorable 7 %**. Présentez un tableau (horizon × scénario). *(Vous pouvez utiliser `02-code/07-capstone-plan-financier.py` ou `capital_final_mensuel`.)*
2. À 30 ans, quel est l'écart entre le scénario défavorable et le favorable (en € et en facteur) ? Que révèle cet écart sur la fiabilité d'une projection à chiffre unique ?
3. Le total versé sur 30 ans est identique dans les trois cas. Calculez-le. Quelle part du capital final (scénario central) vient des **versements** vs des **intérêts composés** ?
4. Rédigez la phrase de **disclaimer** qui devrait accompagner ce tableau dans le plan d'Alex.

### Critères de réussite
- [ ] Tableau 3 scénarios × 3 horizons calculé
- [ ] Écart défavorable/favorable à 30 ans chiffré (€ et facteur)
- [ ] Total versé calculé + part versements/intérêts au scénario central
- [ ] Disclaimer explicite rédigé (performances passées, pas de garantie)

---

## Exercice 2 — Quel levier accélère le plus : le versement ou la durée ?

### Objectif
Comparer l'effet du montant investi et de la durée sur l'atteinte d'un jalon d'indépendance, pour prioriser les leviers d'action d'Alex.

### Consigne

Rendement net illustratif **5 %/an**, capital de départ 0 €. Jalon d'**indépendance partielle** = 10 × dépenses annuelles = **228 000 €**.

1. Calculez le capital à 30 ans pour des versements de **200 €**, **280 €**, **350 €** et **450 €**/mois. Présentez un tableau.
2. Pour chacun de ces versements, estimez en combien d'années Alex atteint le jalon de **228 000 €**. (Itérez, ou utilisez le simulateur.)
3. Passer de 280 € à 450 €/mois (+170 €) gagne combien d'années sur l'atteinte du jalon ? Reliez ce résultat au Module 06 (taux d'épargne comme levier principal).
4. Alex a deux options pour libérer ces 170 € : réduire ses dépenses non essentielles, ou attendre l'extinction de son crédit auto (qui libère 230 €/mois). Discutez l'arbitrage en mobilisant l'ordre des priorités du capstone (fonds d'urgence -> dette -> investissement).

### Critères de réussite
- [ ] Tableau capital à 30 ans pour les 4 niveaux de versement
- [ ] Horizon pour atteindre 228 000 € estimé pour chaque versement
- [ ] Gain d'années (280 -> 450) chiffré et relié au levier "taux d'épargne"
- [ ] Arbitrage "réduire dépenses vs attendre fin de crédit" discuté avec l'ordre des priorités

---

## Exercice 3 — Intégrer un événement de vie dans le plan

### Objectif
Tester la robustesse du plan face à un changement de situation (augmentation de revenu) et pratiquer la **révision** du plan — un réflexe central du Module 07.

### Consigne

Alex obtient une augmentation à la fin de l'**année 5** qui lui permet de porter son investissement de **280 €** à **450 €/mois** (le reste du plan inchangé, 5 % net).

1. Calculez le capital accumulé au bout de l'année 5 (280 €/mois pendant 5 ans). Puis projetez ce capital + 450 €/mois jusqu'à l'année 30. Donnez le capital final.
2. Comparez au scénario "sans augmentation" (280 €/mois pendant 30 ans). Quel gain (en € et en %) l'augmentation, réinvestie et non consommée, apporte-t-elle à 30 ans ?
3. Le Module 07 met en garde contre l'**inflation du train de vie** ("lifestyle creep") : pourquoi est-il décisif qu'Alex **investisse** l'augmentation plutôt que de la dépenser ? Reliez au double effet du taux d'épargne (Module 06).
4. Lors de sa **révision annuelle** (Module 07 §5), quels 3 éléments Alex devrait-il mettre à jour après cette augmentation ?

### Critères de réussite
- [ ] Capital à l'année 5 puis projection avec versement relevé calculés
- [ ] Gain de l'augmentation réinvestie chiffré (€ et %) vs scénario sans
- [ ] Le risque d'inflation du train de vie est expliqué et relié au taux d'épargne
- [ ] 3 éléments de révision annuelle pertinents identifiés
