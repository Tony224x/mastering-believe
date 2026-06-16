# Solutions 08 — Calibration & Forecasting

> Corrigé chiffré modèle pour les exercices du Module 08.

---

## Solution Exercice 1 — Calcul de score de Brier à la main

**a) Calcul de `(pᵢ − oᵢ)²` pour chaque prédiction :**

| Jour | Prédiction | p | o | (p − o) | (p − o)² |
|------|-----------|---|---|---------|----------|
| 1 | Pluie avant midi | 0,75 | 1 | −0,25 | **0,0625** |
| 2 | Température > 30°C | 0,90 | 0 | +0,90 | **0,8100** |
| 3 | Vent fort > 60 km/h | 0,20 | 0 | +0,20 | **0,0400** |
| 4 | Orage en soirée | 0,55 | 1 | −0,45 | **0,2025** |
| 5 | Journée ensoleillée | 0,40 | 1 | −0,60 | **0,3600** |

**b) Score de Brier global :**

```
Brier = (0,0625 + 0,8100 + 0,0400 + 0,2025 + 0,3600) / 5
      = 1,4750 / 5
      = 0,295
```

**c) Score de la baseline "toujours 50 %" :**

Pour les 5 prédictions avec outcomes [1, 0, 0, 1, 1] :

```
Baseline = [(0,5−1)² + (0,5−0)² + (0,5−0)² + (0,5−1)² + (0,5−1)²] / 5
         = [0,25 + 0,25 + 0,25 + 0,25 + 0,25] / 5
         = 1,25 / 5
         = 0,25
```

**d) Verdict et analyse :**

- Score obtenu (0,295) > Baseline (0,25) → le prévisionniste **ne bat pas** la baseline. Ses prédictions n'apportent pas d'information utile par rapport à "je ne sais pas (50/50)".
- La prédiction la plus coûteuse est le **Jour 2** (0,81 sur 5 termes). Raison : le prévisionniste était très confiant (90 %) mais faux. Le carré amplifie les erreurs faites avec forte confiance — `(0,90)² = 0,81` vs `(0,45)² = 0,20` pour une erreur à 55 %. C'est exactement pourquoi le score de Brier encourage la prudence et l'honnêteté sur l'incertitude.

---

## Solution Exercice 2 — Lire une courbe de calibration

**a) Classement par tranche (seuil ±10 points de %) :**

| Tranche | Écart | Classement |
|---------|-------|-----------|
| [0 %–20 %) | −4 % | **Calibré** (|écart| < 10 %) |
| [20 %–40 %) | −10 % | **Limite** (exactement à 10 %, classer sur-confiant par prudence) |
| [40 %–60 %) | −1 % | **Calibré** |
| [60 %–80 %) | −20 % | **Sur-confiant** |
| [80 %–100 %) | −25 % | **Sur-confiant** |

**b) Biais systématique global :**

Le prévisionniste souffre de **sur-confiance sur les hautes probabilités**. Lorsqu'il est très confiant (tranches 60-80 % et 80-100 %), la réalité se produit bien moins souvent qu'annoncé (50 % et 60 % vs 70 % et 85 %). Explication intuitive : quand on "sent" qu'une équipe va gagner, on a tendance à surinflater sa probabilité. Le taux de base réel (probabilité de victoire toutes choses égales) est souvent plus modeste que notre ressenti.

**c) Projection pour "75 % de chances de victoire" :**

La tranche [60-80 %) montre un écart de −20 points (annoncé 70 %, réel 50 %). Sur une annonce à 75 %, le biais passé suggère une fréquence réelle d'environ **55 %** (75 % − 20 %). Ce n'est pas une prédiction exacte, c'est une correction heuristique basée sur le biais documenté.

**d) Recommandation :**

Tenir un **journal de prédictions** avec scoring systématique (calcul du score de Brier mensuel, courbe de calibration trimestrielle). Concrètement : dès qu'il est tenté d'annoncer ≥ 70 %, il devrait consulter ses données passées sur cette tranche avant de publier sa prédiction. À terme, cela "rebobine" l'instinct de sur-confiance par un retour empirique régulier.

---

## Solution Exercice 3 — Journal de prévisions (corrigé type)

> Cet exercice est individuel : la solution est un **exemple modèle commenté**, pas un résultat unique.

**Exemple de journal complet (10 entrées fictives, exemples neutres) :**

| # | Date | Prédiction | p | o | (p−o)² |
|---|------|-----------|---|---|--------|
| 1 | 2026-06-16 | Pluie avant 14h | 0,70 | 1 | 0,09 |
| 2 | 2026-06-17 | Match : victoire de l'équipe A | 0,60 | 1 | 0,16 |
| 3 | 2026-06-17 | Bus < 5 min de retard | 0,50 | 0 | 0,25 |
| 4 | 2026-06-18 | Température > 25°C | 0,80 | 1 | 0,04 |
| 5 | 2026-06-18 | Réunion terminée à l'heure | 0,35 | 0 | 0,12 |
| 6 | 2026-06-19 | Pluie dans la journée | 0,65 | 0 | 0,42 |
| 7 | 2026-06-19 | Sprint terminé en < 2h | 0,75 | 1 | 0,06 |
| 8 | 2026-06-20 | Journée ensoleillée | 0,45 | 0 | 0,20 |
| 9 | 2026-06-21 | Résultat sportif attendu | 0,55 | 1 | 0,20 |
| 10 | 2026-06-21 | Record local battu | 0,25 | 0 | 0,06 |

**b) Score de Brier :**

```
Brier = (0,09 + 0,16 + 0,25 + 0,04 + 0,12 + 0,42 + 0,06 + 0,20 + 0,20 + 0,06) / 10
      = 1,60 / 10
      = 0,160
```

**c) Baseline "toujours 50 %" :**

Outcomes : [1, 1, 0, 1, 0, 0, 1, 0, 1, 0] → 5 uns, 5 zéros.
```
Baseline = 5×(0,5−1)² + 5×(0,5−0)² / 10 = 5×0,25 + 5×0,25 / 10 = 0,25
```
Score obtenu (0,160) < Baseline (0,25) → on bat la baseline.

**d) Table de calibration simplifiée (3 tranches) :**

| Tranche | Entrées | p_moy | Réel | Écart |
|---------|---------|-------|------|-------|
| [0–33 %) | #5 (0,35→arrondi), #8 (0,45→arrondi), #10 (0,25) → entries ≤ 0,33 : #10 | 0,25 | 0 % | −25 % |
| [33–67 %) | #1 (0,70→ 67 %), #3 (0,50), #5 (0,35), #6 (0,65), #8 (0,45), #9 (0,55) | 0,53 | 50 % | −3 % |
| [67–100 %) | #2 (0,60→67+), #4 (0,80), #7 (0,75) | 0,72 | 100 % | +28 % |

> Note : avec seulement 10 prédictions, les tranches sont peu peuplées. Il faut 50+ prédictions pour tirer des conclusions robustes.

**e) Biais identifié :**

Dans cet exemple, la tranche haute [67-100 %) réalise 100 % alors qu'on annonçait 72 % en moyenne → **sous-confiance légère sur les hautes probas**. La tranche basse est sur-confiante (annoncé 25 %, réalisé 0 %). Globalement, le prévisionniste est raisonnablement calibré avec un léger biais de sous-confiance.

**f) Meilleure catégorie :**

Dans cet exemple fictif, la météo et le sport ont des contributions faibles (peu d'erreurs confiantes), tandis que les prédictions de logistique personnelle (bus, réunion) sont moins bonnes. Explication probable : les prévisions météo bénéficient d'une classe de référence naturelle (on consulte l'app météo et on calibre intuitivement) ; les prédictions d'organisation personnelle sont soumises au biais d'optimisme (on sous-estime les délais et imprévus).

**Leçon transversale :** 10 prédictions permettent de s'initier au scoring mais pas de tirer de conclusions statistiquement solides. L'objectif à 30 jours : 50+ prédictions, puis regard mensuel sur la courbe de calibration pour identifier et corriger le biais dominant.
