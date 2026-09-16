# Finance personnelle — Maitriser l'argent, une vie durant

> **Pilier : Argent — Ecole de la vie**
>
> **Disclaimer.** Ce domaine est **purement educatif** et ne constitue **pas un conseil financier, fiscal ou en investissement personnalise**. Les situations individuelles varient ; pour des decisions reelles, consultez un conseiller agree et les sources officielles a jour. Les performances passees ne prejugent pas des performances futures. Tout investissement comporte un risque de perte en capital.

---

## Scope

Ce domaine couvre les **fondamentaux universels** de la finance personnelle selon une logique **Pareto-first** : maitriser le 20 % de concepts qui produit 80 % des resultats financiers d'une vie.

**Public vise** : adultes francophones debutants ou intermediaires. **Aucun prerequis math avance** n'est requis — arithmetique de base suffit. Pas de jargon inutile, concret avant abstrait.

**Ce que vous apprendrez** :
- Comprendre et exploiter les interets composes — y compris le rendement **reel** (ajuste de l'inflation)
- Construire un budget solide, automatiser son epargne et constituer un fonds d'urgence
- Definir des objectifs financiers par horizon et etapes de vie
- Gerer la dette et le credit (et comprendre le score de credit)
- Poser les bases de l'investissement (risque/rendement, diversification) puis investir via des fonds indiciels a bas cout
- Comprendre les principes fiscaux generaux et le role des enveloppes
- Reconnaitre vos biais psychologiques et reperer les arnaques financieres
- Assurer et transferer le risque ; decider acheter vs louer par la donnee
- Augmenter et negocier ses revenus (le levier #1 d'une vie)
- Viser l'independance financiere avec methode (regle des 4 % et ses limites)
- Situer crypto et robo-advisors de facon neutre
- Construire votre propre simulateur de plan financier (capstone)

**Ce que ce domaine n'est PAS** :
- Un guide de stock-picking ou de day-trading
- Une promotion de la crypto ou de placements speculatifs (presentes de facon neutre, par la donnee)
- Un conseil fiscal adapte a votre pays ou situation personnelle (principes generaux uniquement)
- Une promesse de "s'enrichir vite"

---

## Prerequis

- Arithmetique de base (pourcentages, multiplication)
- Aucun prerequis financier ou mathematique avance
- Curiosite et volonte de reflechir concretement a sa situation

---

## Planning — 14 modules (~45 min chacun)

| Jour | Slug | Titre | Temps |
|------|------|-------|-------|
| J1 | `01-interets-composes` | Interets composes, valeur du temps et inflation | ~45 min |
| J2 | `02-budget-se-payer-dabord` | Budget et se payer en premier | ~45 min |
| J3 | `03-fonds-urgence-objectifs` | Fonds d'urgence, objectifs et etapes de vie | ~45 min |
| J4 | `04-dette-credit` | Dette, credit et score de credit | ~45 min |
| J5 | `05-bases-investissement` | Bases de l'investissement (risque/rendement, diversification) | ~45 min |
| J6 | `06-fonds-indiciels-allocation` | Fonds indiciels et allocation | ~45 min |
| J7 | `07-fiscalite-enveloppes` | Fiscalite et enveloppes (principes universels) | ~45 min |
| J8 | `08-psychologie-arnaques` | Psychologie de l'argent et arnaques | ~45 min |
| J9 | `09-assurance-risque` | Assurance, gestion du risque et transmission | ~45 min |
| J10 | `10-immobilier-location` | Immobilier vs location et gros achats | ~45 min |
| J11 | `11-revenus-negociation` | Revenus et negociation salariale | ~45 min |
| J12 | `12-independance-financiere` | Independance financiere et retrait soutenable | ~45 min |
| J13 | `13-actifs-numeriques-robo` | Actifs numeriques et robo-advisors (neutre) | ~45 min |
| J14 | `14-capstone-plan-financier` | Capstone : simulateur de plan financier | ~60 min |

**Duree totale estimee** : ~11 h de formation + exercices

---

## Criteres de reussite du domaine

A l'issue des 14 modules, vous devez etre capables de :

- [ ] Calculer la valeur finale d'une epargne reguliere sur 20-30 ans, en nominal **et en reel** (inflation)
- [ ] Rediger votre budget mensuel (50/30/20), automatiser l'epargne et fixer des objectifs par horizon
- [ ] Calculer le cout reel d'un credit (TAEG, total des interets) et expliquer le score de credit
- [ ] Distinguer risque/rendement et expliquer pourquoi la diversification reduit le risque
- [ ] Expliquer (preuves SPIVA, avec leur nuance) pourquoi les fonds indiciels a bas cout sont difficiles a battre nettement de frais
- [ ] Decrire les principes fiscaux generaux et le role des enveloppes
- [ ] Identifier des biais psychologiques et reconnaitre les schemas d'arnaque courants
- [ ] Raisonner acheter vs louer par la donnee, et preparer une negociation salariale
- [ ] Simuler un scenario d'independance financiere (regle des 4 % et ses limites)
- [ ] Produire un simulateur de plan financier parametrable : budget + fonds d'urgence + allocation + projection reelle a 20-30 ans (capstone J14)

---

## Capstone (J14)

Le capstone est un **simulateur de plan financier parametrable** (`02-code/14-capstone-plan-financier.py`) ou vous entrez **vos** chiffres :
1. Budget mensuel detaille et taux d'epargne
2. Objectif et calendrier de constitution du fonds d'urgence
3. Inventaire et plan de remboursement de la dette
4. Allocation d'investissement simulee (vos % en entree, jamais imposes)
5. Projection d'interets composes sur 20-30 ans, **nominale et reelle**
6. Cible optionnelle d'independance financiere (taux de retrait soutenable et ses limites)

C'est un **gabarit educatif** : il affiche des projections a partir d'hypotheses ajustables, **pas une recommandation**.

---

## Structure des fichiers

```
domains/vie/finance-personnelle/
├── README.md          # Ce fichier
├── PLAN.md            # Curriculum fige et notes editoriales
├── REFERENCES.md      # Sources tier-1 par module
├── 01-theory/         # Cours theoriques (Modules 01-14)
├── 02-code/           # Calculateurs et exemples Python (stdlib)
├── 03-exercises/      # Exercices progressifs + solutions
│   ├── 01-easy/
│   ├── 02-medium/
│   ├── 03-hard/
│   ├── solutions/
│   └── workspace/     # Votre espace de travail (gitignore)
├── 04-projects/       # Mini-projets libres
└── 05-projets-guides/ # Projets guidés appliqués (contexte LogiSim)
```

## Projets guidés (Module 05-projets-guides)

Trois projets appliqués qui mettent vos acquis en situation, sur le décor métier
partagé LogiSim (cf. [`shared/logistics-context.md`](../../../shared/logistics-context.md)) :
les protagonistes sont des personnes travaillant chez/autour de LogiSim
confrontées à des décisions d'argent ordinaires (finance **personnelle**, pas
finance d'entreprise). Chaque projet fournit un script exécutable (stdlib) + une
analyse interprétée :

| # | Projet | Modules mobilisés |
|---|--------|-------------------|
| 01 | Le plan d'épargne d'une équipe OCC | 01, 02, 05 |
| 02 | Faut-il s'endetter pour la voiture ? | 03, 01 |
| 03 | Allocation & indépendance financière | 04, 05, 06, 01 |

Voir [`05-projets-guides/README.md`](./05-projets-guides/README.md).
