# Module 07 — Capstone : construire son plan financier personnel

> **Temps estimé** : 90-120 min (capstone, travail actif) | **Prérequis** : Modules 01-06
> **Objectif** : Intégrer l'ensemble des concepts des modules 01 à 06 dans un plan financier personnel cohérent, chiffré, adapté à votre situation — budget, fonds d'urgence, allocation simulée et projection sur 20-30 ans.

> ⚠️ **Disclaimer** : Ce capstone est **purement éducatif** et ne constitue pas un conseil financier, fiscal ou en investissement personnalisé. Les projections sont des estimations basées sur des hypothèses historiques qui ne garantissent pas les résultats futurs. Consultez un conseiller agréé pour toute décision financière réelle. Tout investissement comporte un risque de perte en capital.

---

## 1. Pourquoi un plan écrit fait une différence

Des études en psychologie de l'objectif (notamment les travaux de Peter Gollwitzer sur les "implementation intentions") montrent que mettre par écrit **quoi faire, quand et comment** multiplie significativement la probabilité de passer à l'action et de tenir dans la durée.

Un plan financier écrit n'est pas un document bureaucratique. C'est :
- Un miroir de vos valeurs et priorités financières
- Un outil de prise de décision lors des tentations ou des crises de marché
- Un repère pour mesurer vos progrès annuellement

---

## 2. Structure du livrable capstone

Votre plan financier personnel se compose de **5 blocs** :

### Bloc 1 — Photographie financière actuelle

| Élément | Votre situation |
|---|---|
| Revenus nets mensuels | |
| Dépenses mensuelles (essentielles) | |
| Dépenses mensuelles (non essentielles) | |
| Épargne mensuelle actuelle | |
| Taux d'épargne actuel | |
| Dettes en cours (montant, taux, mensualité) | |
| Épargne et patrimoine investis actuels | |

### Bloc 2 — Budget optimisé (règle 50/30/20 ou personnalisée)

Répartition cible de vos revenus :
- % alloué aux besoins essentiels (logement, alimentation, transport, santé)
- % alloué aux envies / loisirs
- % alloué à l'épargne et remboursement de dettes

Ajustements par rapport à la situation actuelle : où réduire, comment automatiser l'épargne.

### Bloc 3 — Fonds d'urgence

- Objectif cible (3, 4 ou 6 mois de dépenses selon votre situation)
- Capital actuel dans le fonds d'urgence
- Délai estimé pour atteindre l'objectif (si pas encore atteint)
- Où il est placé (compte liquide, sans risque)

### Bloc 4 — Allocation d'investissement simulée

- Taux d'épargne dédié à l'investissement (€/mois, après fonds d'urgence)
- Répartition entre classes d'actifs (exemple "3 fonds" du Module 04)
- Type d'enveloppe fiscale utilisée (selon votre pays — à compléter selon votre situation)
- TER cible (frais annuels des fonds choisis)

### Bloc 5 — Projection sur 20-30 ans

- Capital de départ
- Versements mensuels
- Hypothèse de rendement annuel (ex. 5 % net de frais — hypothèse prudente)
- Projection à 10, 20, 30 ans
- Calcul du "nombre de fois les dépenses annuelles couvertes" à chaque jalon
- Estimation de l'horizon d'indépendance financière (optionnel)

---

## 3. Le simulateur Python du capstone

Le fichier `02-code/07-capstone-plan-financier.py` fournit un simulateur en ligne de commande qui calcule automatiquement :
- L'impact des frais sur votre projection
- La croissance du capital mois par mois
- Les jalons (capital × 5, × 10, × 25 vos dépenses annuelles)
- Un résumé lisible de votre plan

Lancez-le avec :
```
python domains/finance-personnelle/02-code/07-capstone-plan-financier.py
```

---

## 4. Critères de réussite du capstone

Votre plan est complet et opérationnel si :
- [ ] Vous avez une photographie claire de vos revenus, dépenses et épargne actuelle
- [ ] Votre budget cible est défini avec les 3 grandes catégories (besoins/envies/épargne)
- [ ] Votre objectif de fonds d'urgence est chiffré et sa localisation est définie
- [ ] Vous avez une allocation d'investissement en fonds indiciels (au moins 2 fonds, TER < 0,5 %)
- [ ] Vous avez une projection chiffrée à 20 ans et 30 ans avec une hypothèse de rendement explicite
- [ ] Vous avez mis en place (ou planifié) un virement automatique mensuel vers votre investissement
- [ ] Le plan tient sur 1 à 2 pages — il doit être relisible rapidement lors des révisions annuelles

---

## 5. Révision annuelle : le rituel de maintenance

Un plan financier se révise une fois par an (ou après un événement majeur : changement de revenu, naissance, achat immobilier, etc.) :

1. Comparer situation réelle vs plan (taux d'épargne réel, capital investi réel)
2. Rééquilibrer l'allocation si une classe d'actifs s'est trop éloignée de la cible
3. Mettre à jour les projections avec les données réelles
4. Ajuster le plan si les objectifs ou la situation ont changé

> Le plan n'est pas figé. Il évolue avec votre vie. Ce qui compte, c'est de le maintenir vivant et de s'y référer.

---

## 6. Les erreurs classiques à éviter

| Erreur | Réponse |
|---|---|
| Pas de fonds d'urgence avant d'investir | Construire le fonds d'urgence en premier (Module 02) |
| Rembourser toutes les dettes avant d'investir | Arbitrage selon le taux (Module 03) — dettes à taux élevé d'abord |
| Choisir un fonds à frais élevés pour un avantage fiscal marginal | Calculer le coût total net (frais × enveloppe fiscale) |
| Vérifier le portefeuille quotidiennement | Revue trimestrielle maximum, virement automatique |
| Plan trop complexe (15 fonds différents) | Simplicité = durabilité. 2 à 3 fonds suffisent. |
| Ignorer l'inflation dans les projections | Utiliser un rendement réel (rendement nominal − inflation) |

---

> **À retenir** :
> - Un plan financier écrit change le comportement — c'est prouvé par la recherche en psychologie.
> - Il tient en 5 blocs : photographie actuelle, budget, fonds d'urgence, allocation, projection.
> - La révision annuelle est le rituel qui maintient le plan vivant et pertinent.
> - La simplicité est une vertu : un plan qu'on applique prime un plan parfait qu'on abandonne.

---

## Flash-cards (spaced repetition)

**Q1** — Pourquoi écrire son plan financier plutôt que le garder "dans la tête" ?
> **R** : Les "implementation intentions" (Gollwitzer) montrent qu'écrire quoi/quand/comment multiplie la probabilité d'action. Un plan écrit sert aussi de repère lors des crises émotionnelles (krachs, tentations de dépense).

**Q2** — Quels sont les 5 blocs d'un plan financier personnel complet ?
> **R** : Photographie actuelle → Budget optimisé → Fonds d'urgence → Allocation d'investissement → Projection 20-30 ans.

**Q3** — À quelle fréquence réviser son plan financier ?
> **R** : Une fois par an (révision annuelle), ou après un événement majeur. Inclut : comparaison réel vs plan, rééquilibrage de l'allocation, mise à jour des projections.

**Q4** — Quelle est l'erreur la plus courante dans l'ordre des priorités financières ?
> **R** : Investir avant d'avoir constitué le fonds d'urgence. Le fonds d'urgence (3-6 mois de dépenses) protège contre le besoin de vendre des investissements au mauvais moment.

**Q5** — Pourquoi la simplicité du plan est-elle une vertu financière ?
> **R** : Un plan simple (2-3 fonds, virement automatique, révision annuelle) est plus facile à maintenir dans la durée. La complexité génère de la friction, des coûts cachés et des erreurs comportementales.

---

## Points clés à retenir

1. Un plan écrit et chiffré change le comportement — c'est la base de toute réalisation financière durable.
2. Les 5 blocs : photographie, budget, fonds d'urgence, allocation, projection.
3. La simplicité (2-3 fonds indiciels, virement automatique) prime la sophistication.
4. La révision annuelle maintient le plan pertinent et ancré dans la réalité.
5. Ce plan est un point de départ éducatif — adaptez-le à votre situation et consultez un professionnel pour les décisions importantes.

---

## Pour aller plus loin

- **The Little Book of Common Sense Investing** — J. C. Bogle (Wiley, 2017) : pour l'allocation indicielle. https://www.wiley.com/en-us/9781119404507
- **The Psychology of Money** — M. Housel (Harriman House, 2020) : pour ancrer le comportement dans la durée. https://harriman-house.com/authors/morgan-housel/the-psychology-of-money/9780857197689/
- **The Simple Path to Wealth** — J. L. Collins (2016/2025) : un plan de A à Z, sobre et applicable. https://www.simonandschuster.com/books/The-Simple-Path-to-Wealth/J-L-Collins/9798893310474
- **Compound Interest Calculator** — Investor.gov (SEC) : outil officiel de projection. https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator
- **AMF — Espace épargnants** (en français) : https://www.amf-france.org/fr/espace-epargnants
- **La finance pour tous** — IEFP : https://www.lafinancepourtous.com/
