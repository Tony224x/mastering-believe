# Analyse — Projet 02 (crédit véhicule d'une technicienne terrain)

> Lecture commentée. Chiffres issus de l'exécution de `credit_analyzer.py`
> (taux **illustratifs** — projet éducatif, pas un conseil en crédit).

## 1. Les deux offres (8 000 €)

| Offre | Mensualité | Coût du crédit | Total remboursé |
|---|---:|---:|---:|
| A — 4,9 % / 36 mois | 239,41 € | **619 €** | 8 619 € |
| B — 4,5 % / 72 mois | 126,99 € | **1 143 €** | 9 143 € |

**La leçon contre-intuitive** : l'offre B a un taux *affiché plus bas* (4,5 %
contre 4,9 %) **et** une mensualité presque deux fois plus douce… mais elle
coûte **~2× plus cher au total**. Pourquoi ? Parce qu'on paie des intérêts
pendant **72 mois au lieu de 36**. La durée écrase l'avantage du taux. Le
vendeur qui « adapte la mensualité à votre budget » vend surtout… plus
d'intérêts.

> Toujours comparer le **coût du crédit** (total des intérêts), jamais la seule
> mensualité.

## 2. Amortissement — pourquoi rembourser tôt paie

Sur l'offre A, la 1re mensualité de 239,41 € se décompose en **32,67 €
d'intérêts** et seulement **206,74 € de capital**. Au début d'un crédit, on
paie surtout le loyer de l'argent. Conséquence pratique : un remboursement
anticipé **précoce** attaque directement le capital et supprime tous les
intérêts futurs sur cette part — d'où un effet de levier maximal au début.

## 3. Comptant ou garder l'épargne placée ?

À *effort égal* (la mensualité sort du salaire dans les deux scénarios), le
verdict dépend **uniquement** du rapport entre le rendement du placement et le
taux du crédit (4,5 %) :

| Rendement du placement | Patrimoine net si comptant | si crédit | Gagnant |
|---|---:|---:|---|
| 2 % / an | 17 598 € | 16 911 € | **Comptant** (+687 €) |
| 7 % / an | 21 963 € | 22 802 € | **Crédit** (+838 €) |

- Si le placement rapporte **moins** que le crédit ne coûte (2 % < 4,5 %),
  payer comptant gagne.
- S'il rapporte **plus** (7 % > 4,5 %), garder l'argent investi et financer
  l'achat gagne — c'est l'arbitrage de « dette intelligente ».

⚠️ Nuance honnête que le tableau ne capture pas : payer comptant est un
rendement **certain et sans risque** (le taux du crédit évité), alors que le
7 % de placement est **espéré et risqué**. À avantage chiffré comparable, le
sans-risque vaut une prime. Le crédit n'« est rentable » qu'avec un placement
qui bat son taux **net d'impôt et de risque**.

## 4. Avalanche vs boule de neige (3 dettes, 400 €/mois)

Dettes : carte 1 500 € à 18 %, téléphone 600 € à 5 %, prêt étudiant 4 000 € à 3 %.

| Stratégie | Durée | Total intérêts |
|---|---:|---:|
| Avalanche (taux max d'abord) | 17 mois | **213,74 €** |
| Boule de neige (petit solde d'abord) | 17 mois | 252,99 € |

- L'**avalanche** attaque d'abord la carte à 18 % : mathématiquement optimale,
  elle économise ici **39,25 €** d'intérêts.
- La **boule de neige** solde d'abord le petit crédit téléphone : un peu plus
  chère, mais elle procure une **victoire rapide** qui aide à tenir.

> Choix honnête (module 05, le facteur humain) : l'avalanche si tu tiens sur la
> durée ; la boule de neige si tu as besoin de *momentum* pour ne pas
> abandonner. La meilleure stratégie est celle qu'on **applique réellement**.

## Choix de modélisation

- **Formule fermée pour la mensualité**, boucle explicite pour
  l'amortissement et le remboursement multi-dettes (lisibilité > élégance).
- **Garde-fou à 1 000 mois** dans `rembourser` : si le budget ne couvre pas les
  intérêts, la dette diverge ; on stoppe au lieu de boucler à l'infini (probe
  adversariale vérifiée).
- **Copie de travail des dettes** : `rembourser` ne mute pas la liste d'entrée,
  pour pouvoir relancer les deux stratégies sur les mêmes données.

## Limites honnêtes

- Pas d'assurance emprunteur ni de frais de dossier (ils changent souvent le
  classement réel des offres — cf. « Pour aller plus loin »).
- TAEG approximé par le taux nominal annualisé ; le vrai TAEG inclut les frais.
- Fiscalité du placement ignorée (elle réduit le rendement net, donc favorise
  encore le comptant).

## Vérification

```
$ python credit_analyzer.py        # tourne, sorties cohérentes
$ ruff check .                      # All checks passed!
```
Probes adversariales : `mensualite(1200, 0.0, 12) == 100.0` (taux zéro) ;
budget insuffisant → garde-fou stoppe à 1 000 mois (pas de boucle infinie).
