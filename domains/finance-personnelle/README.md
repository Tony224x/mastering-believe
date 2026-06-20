# Finance personnelle — Maitriser l'argent, une vie durant

> **Pilier : Argent — Ecole de la vie**
>
> **Disclaimer.** Ce domaine est **purement educatif** et ne constitue **pas un conseil financier, fiscal ou en investissement personnalise**. Les situations individuelles varient ; pour des decisions reelles, consultez un conseiller agree et les sources officielles a jour. Les performances passees ne prejugent pas des performances futures. Tout investissement comporte un risque de perte en capital.

---

## Scope

Ce domaine couvre les **fondamentaux universels** de la finance personnelle selon une logique **Pareto-first** : maitriser le 20 % de concepts qui produit 80 % des resultats financiers d'une vie.

**Public vise** : adultes francophones debutants ou intermediaires. **Aucun prerequis math avance** n'est requis — arithmetique de base suffit. Pas de jargon inutile, concret avant abstrait.

**Ce que vous apprendrez** :
- Comprendre et exploiter les interets composes (le moteur de toute richesse a long terme)
- Construire un budget solide et un fonds d'urgence qui vous protege
- Gerer la dette intelligemment et comprendre le cout reel du credit
- Investir simplement, a bas cout et sur le long terme
- Reconnaitre vos biais psychologiques et y resister
- Viser l'independance financiere avec methode et realisme
- Construire votre propre plan financier chiffre (capstone)

**Ce que ce domaine n'est PAS** :
- Un guide de stock-picking ou de day-trading
- Un cours sur la crypto ou les placements speculatifs
- Un conseil fiscal adapte a votre pays ou situation personnelle
- Une promesse de "s'enrichir vite"

---

## Prerequis

- Arithmetique de base (pourcentages, multiplication)
- Aucun prerequis financier ou mathematique avance
- Curiosite et volonte de reflechir concretement a sa situation

---

## Planning — 7 modules (~45 min chacun)

| # | Slug | Titre | Temps |
|---|------|-------|-------|
| 01 | `01-interets-composes` | Le moteur : interets composes et valeur du temps | ~45 min |
| 02 | `02-budget-epargne-fonds-urgence` | Se payer en premier : budget, epargne automatique et fonds d'urgence | ~45 min |
| 03 | `03-dette-credit` | Maitriser la dette et le credit | ~45 min |
| 04 | `04-investir-long-terme` | Investir simplement et sur le long terme | ~45 min |
| 05 | `05-psychologie-argent-frais` | Le facteur humain : psychologie de l'argent et impact des frais | ~45 min |
| 06 | `06-independance-financiere` | Cap long terme : independance financiere et retrait soutenable | ~45 min |
| 07 | `07-capstone-plan-financier` | Capstone : mon plan financier personnel chiffre | ~60 min |

**Duree totale estimee** : ~5,5 h de formation + exercices

---

## Criteres de reussite du domaine

A l'issue des 7 modules, vous devez etre capables de :

- [ ] Calculer la valeur finale d'une epargne reguliere sur 20-30 ans avec les interets composes
- [ ] Rediger votre propre budget mensuel avec la regle 50/30/20 et identifier vos leviers
- [ ] Calculer le cout reel d'un credit (TAEG, total des interets) et decider si s'endetter vaut le coup
- [ ] Expliquer pourquoi les fonds indiciels a bas cout surperforment en moyenne la gestion active nette de frais
- [ ] Identifier trois biais psychologiques qui sabotent les decisions financieres et les contourner
- [ ] Calculer votre taux d'epargne actuel et simuler un scenario d'independance financiere
- [ ] Produire un plan financier personnel chiffre : budget + objectif fonds d'urgence + allocation d'investissement + projection a 20 ans (capstone module 07)

---

## Capstone (Module 07)

Le capstone est un **plan financier personnel chiffre** que vous construirez pas a pas :
1. Budget mensuel detaille et taux d'epargne
2. Objectif et calendrier de constitution du fonds d'urgence
3. Inventaire et plan de remboursement de la dette
4. Allocation d'investissement simulee (fonds indiciels, horizon)
5. Projection d'interets composes sur 20-30 ans avec differents scenarios
6. Cible optionnelle d'independance financiere (taux de retrait soutenable)

Ce plan est le votre — educatif, pas un conseil personnalise.

---

## Structure des fichiers

```
domains/finance-personnelle/
├── README.md          # Ce fichier
├── PLAN.md            # Curriculum fige et notes editoriales
├── REFERENCES.md      # Sources tier-1 par module
├── 01-theory/         # Cours theoriques (Modules 01-07)
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
partagé LogiSim (cf. [`shared/logistics-context.md`](../../shared/logistics-context.md)) :
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
