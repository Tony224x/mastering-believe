# Analyse — Projet 03 (allocation & indépendance financière)

> Lecture commentée. Chiffres issus de l'exécution de `tableau_bord.py`
> (rendements **illustratifs**, non garantis — projet éducatif, pas un conseil).

## 1. Allocation par horizon

| Horizon | Actions | Obligations | Rendement attendu* |
|---:|---:|---:|---:|
| 3 ans | 30 % | 70 % | 4,2 % |
| 8 ans | 42 % | 58 % | 4,7 % |
| 15 ans | 70 % | 30 % | 5,8 % |
| 25 ans | 90 % | 10 % | 6,6 % |
| 35 ans | 90 % | 10 % | 6,6 % |

\* Pondéré, **brut de frais et d'inflation, non garanti**. Plus l'horizon est
long, plus on peut supporter de volatilité (donc d'actions), parce qu'on a le
temps de digérer les baisses. La règle codée est **une heuristique assumée** —
le vrai déterminant n'est pas l'âge mais l'horizon **et** la capacité
psychologique à ne pas vendre en pleine baisse (module 05).

## 2. Actif vs passif (net de frais) — le cadrage SPIVA

6 000 €/an pendant 30 ans, rendement brut 7 % :

| Fonds | Frais | Capital final |
|---|---:|---:|
| Indiciel passif | 0,2 % | **546 774 €** |
| Géré activement | 1,8 % | 412 599 € |

**Manque à gagner dû à 1,6 point de frais : 134 175 € (≈ 25 %).** L'hypothèse
« même rendement brut » n'est pas un cadeau fait au passif : les rapports SPIVA
montrent que, sur 15-20 ans, **~85-95 % des fonds actifs sous-performent leur
indice net de frais**. Payer plus cher pour « battre le marché » est, en moyenne
et statistiquement, un pari perdant — et même *à* performance brute égale, les
frais suffisent à creuser un quart d'écart.

## 3. Indépendance financière — cible et délai

Dépenses visées 30 000 €/an, capital actuel 40 000 €, épargne 18 000 €/an :

| Taux de retrait | Capital cible | Multiple | Délai |
|---:|---:|---:|---:|
| 5,0 % | 600 000 € | ×20 | 18 ans |
| 4,0 % | 750 000 € | ×25 | 21 ans |
| 3,5 % | 857 143 € | ×28,6 | 23 ans |

Le **taux de retrait** choisi change radicalement la cible : plus prudent = plus
gros capital = plus d'années. Et le moteur du délai n'est pas le revenu mais le
**taux d'épargne** (module 06) : c'est lui qui détermine à la fois combien on
accumule **et** combien il faut accumuler (des dépenses basses abaissent la
cible *et* augmentent l'épargne).

⚠️ La règle des 4 % vient de l'étude Trinity (marché US, horizon 30 ans). C'est
une **heuristique historique, pas une loi** : horizon plus long, marché
différent, ou séquence défavorable (cf. §4) peuvent la mettre en défaut. D'où
l'intérêt de regarder aussi 3,5 %.

## 4. Risque de séquence — le piège du début de retraite

Capital 750 000 €, retrait 30 000 €/an, **mêmes 10 rendements** (moyenne 4,6 %),
dans deux ordres inverses :

| Ordre des rendements | Capital final |
|---|---:|
| Krach **en fin** de période | **808 131 €** |
| Krach **au début** de période | 634 121 € |

**Écart : 174 010 € pour une moyenne de rendement identique.** C'est le cœur du
projet. En phase d'**accumulation**, l'ordre des rendements est neutre (seul le
produit compte). En phase de **retrait**, il devient critique : encaisser une
forte baisse *juste après* avoir commencé à puiser dans le capital force à
vendre des parts dépréciées, ce qui ampute la base sur laquelle le rebond
pourra agir. Parades classiques :

- **matelas de cash** de 1-2 ans de dépenses (pour ne pas vendre en baisse) ;
- **taux de retrait flexible** (réduire les retraits les mauvaises années) ;
- ne **pas** être 100 % actions à la veille de vivre de son capital (module 04 + 06).

## Choix de modélisation

- **Tout en boucles annuelles explicites** : on privilégie la lisibilité et
  l'auditabilité sur l'élégance (les volumes sont minuscules).
- **`reversed()` pour le risque de séquence** : garantit que les deux séquences
  ont *exactement* le même multiensemble de rendements, donc la même moyenne —
  sinon la démonstration serait biaisée.
- **Retrait en début d'année** : hypothèse conservatrice (on « vit » avant que
  le marché ne joue), cohérente avec les études de retrait soutenable.
- **Garde-fous** : `annees_jusqu_a_fi` → `None` au-delà de 100 ans ;
  `simuler_retraite` → 0.0 dès épuisement (jamais de capital négatif).

## Limites honnêtes

- **Rendements lissés ou en séquences choisies à la main** : pédagogiques, pas
  prédictifs. La vraie distribution est plus large (d'où l'extension Monte-Carlo).
- **Pas d'inflation explicite** dans les volets 1-3 (les montants sont nominaux).
  Le volet 4 ignore aussi l'indexation du retrait, ce qui **sous-estime** le
  risque réel de séquence.
- **Fiscalité et frais réels** absents des projections FI — ils repoussent la
  cible et allongent le délai.
- La règle des 4 % est **non garantie** (cf. §3).

## Vérification

```
$ python tableau_bord.py           # tourne, sorties cohérentes
$ ruff check .                      # All checks passed!
```
Probes adversariales : `annees_jusqu_a_fi(0, 0, 1e6, 0.0)` → `None` (cible
inatteignable) ; `simuler_retraite(50000, 30000, [-0.5]*10)` → `0.0` (épuisement
propre, sans valeur négative).
