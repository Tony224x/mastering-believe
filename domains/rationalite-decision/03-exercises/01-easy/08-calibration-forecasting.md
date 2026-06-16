# Exercices 08 — Calibration & Forecasting

> **Module** : 08 — Calibration & Forecasting
> **Niveau** : progressif (easy → medium → hard)
> **Prérequis** : avoir lu `01-theory/08-calibration-forecasting.md` et lancé `02-code/08-calibration-forecasting.py`

---

## Exercice 1 — Calcul de score de Brier à la main

### Objectif

Maîtriser le calcul du score de Brier sur un petit jeu de données, identifier le verdict, et comprendre ce que le carré pénalise.

### Consigne

Un prévisionniste météo a noté les prédictions suivantes sur 5 jours consécutifs :

| Jour | Prédiction | Probabilité annoncée | Résultat observé |
|------|-----------|----------------------|-----------------|
| 1 | Pluie avant midi | 0,75 | 1 (il a plu) |
| 2 | Température > 30°C | 0,90 | 0 (non) |
| 3 | Vent fort (> 60 km/h) | 0,20 | 0 (non) |
| 4 | Orage en soirée | 0,55 | 1 (oui) |
| 5 | Journée ensoleillée | 0,40 | 1 (oui) |

**Questions :**

a) Calculez `(pᵢ − oᵢ)²` pour chacune des 5 prédictions.

b) Calculez le score de Brier global.

c) Calculez le score de Brier du prévisionniste "toujours 50 %" sur ce même jeu.

d) Le prévisionniste bat-il la baseline ? Quelle prédiction a le plus coûté en points de Brier ? Pourquoi ?

### Critères de réussite

- [ ] Les 5 termes `(pᵢ − oᵢ)²` sont corrects (tolérance : ±0,001)
- [ ] Le score de Brier final est correct (tolérance : ±0,001)
- [ ] Le score baseline "toujours 50 %" est calculé et vaut 0,25 (ou proche selon la répartition)
- [ ] La question d sur la prédiction la plus coûteuse est justifiée avec le mécanisme du carré

---

## Exercice 2 — Lire une courbe de calibration

### Objectif

Interpréter une table de calibration, identifier les biais systématiques (sur-confiance, sous-confiance), et formuler une recommandation d'ajustement concrète.

### Consigne

Un prévisionniste sportif a scoré 50 prédictions sur des résultats de matchs. Voici sa table de calibration (calculée sur `n_bins = 5`) :

| Tranche | n | Annoncé | Réel | Écart |
|---------|---|---------|------|-------|
| [0 %–20 %) | 8 | 12 % | 8 % | −4 % |
| [20 %–40 %) | 10 | 30 % | 20 % | −10 % |
| [40 %–60 %) | 14 | 51 % | 50 % | −1 % |
| [60 %–80 %) | 12 | 70 % | 50 % | −20 % |
| [80 %–100 %) | 6 | 85 % | 60 % | −25 % |

**Questions :**

a) Pour chaque tranche, indiquez si le prévisionniste est calibré, sur-confiant ou sous-confiant (utilisez le seuil de ±10 points de %).

b) Quel biais systématique global observez-vous ? Proposez une explication intuitive.

c) Ce prévisionniste annonce "75 % de chances de victoire" pour un match. Si son biais persiste, quel taux de victoire réel devrait-il anticiper dans ce scénario ?

d) Quelle habitude pourrait-il adopter pour corriger ce biais à terme ?

### Critères de réussite

- [ ] Chaque tranche est correctement classée (calibré / sur-confiant / sous-confiant)
- [ ] Le biais global est nommé et expliqué intuitivement (sur-confiance sur les hautes probas)
- [ ] La question c est traitée avec un raisonnement explicite (pas juste un chiffre sorti de nulle part)
- [ ] La recommandation de la question d est concrète et pratique (journal + scoring, pas "être plus humble")

---

## Exercice 3 — Construire son premier journal de prévisions

### Objectif

Mettre en pratique la tenue d'un journal de prévisions sur 10 entrées réelles, calculer le score de Brier, et identifier son propre biais de calibration.

### Consigne

Cet exercice se déroule sur **7 jours**. Il nécessite de noter de vraies prédictions *avant* de connaître le résultat.

**Étape 1 — Collecte (jours 1 à 7) :**

Notez au moins 10 prédictions sur des événements vérifiables à court terme. Exemples de catégories (choisissez ceux qui correspondent à votre quotidien) :
- Météo (pluie, température, ensoleillement)
- Sport (résultat d'un match ou tournoi en cours)
- Logistique (arrivée d'un bus/train, durée d'une tâche)

Pour chaque entrée, utilisez ce format :

```
Date        : YYYY-MM-DD
Prédiction  : [énoncé précis et vérifiable]
Probabilité : XX %
Résultat    : [à remplir après vérification]
```

**Étape 2 — Vérification et scoring :**

a) Relevez le résultat de chaque prédiction (0 ou 1) et calculez `(p − o)²`.

b) Calculez votre score de Brier sur les 10 prédictions.

c) Calculez le score de la baseline "toujours 50 %".

d) Construisez une table de calibration simplifiée (3 tranches : [0–33 %), [33–67 %), [67–100 %]). Trouvez la fréquence réelle dans chaque tranche.

**Étape 3 — Analyse :**

e) Êtes-vous globalement sur-confiant, sous-confiant, ou calibré ? Justifiez avec la table.

f) Quelle catégorie de prédictions (météo, sport, logistique, autre) vous a le mieux réussi ? Pourquoi, selon vous ?

### Critères de réussite

- [ ] Le journal contient au moins 10 entrées avec date, énoncé précis, probabilité numérique, et résultat
- [ ] Le score de Brier est calculé correctement sur l'ensemble des prédictions
- [ ] La table de calibration (3 tranches minimum) est construite et remplie
- [ ] L'analyse (questions e et f) est étayée par les données du journal, pas par l'intuition seule
- [ ] Le score de Brier est comparé à la baseline 0,25 et le verdict est tiré
