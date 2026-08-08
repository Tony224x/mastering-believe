# Solution — Formules & tableaux

> Module `07-formules-tableaux` · Badge : **Modèle lisible** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

### Mini-modèle
Onglet Transactions + formules solde. Vérifie : `SOMME(Sens) == total entrées − total sorties`.


## Medium (bonus)

### Onglet Résumé séparé
Total entrées, sorties, solde, % sorties/entrées (si entrées > 0). Références stables (pas de plages fragiles).


## Hard (bonus)

### Scénarios base / optimiste / pessimiste
3 colonnes d'hypothèses + résumé qui bascule (ou 3 blocs clairs).


## Clés structurées (rappel)

### easy

- **sens_formula_fr** : =SI(E2="entree";D2;-D2)
- **check** : SOMME(Sens) == total entrees - total sorties

### medium

- **resume_cells** :
  - B2 entrees
  - B3 sorties
  - B4 solde
  - B5 ratio
- **div_guard_fr** : =SI(B2=0;"n/a";B3/B2)
- **copy_errors** :
  - mauvaise plage
  - formule EN
  - entetes inclus dans SOMME

### hard

- **min_categories** : 4
- **example_categories** :
  - loyer
  - salaires
  - marketing
  - fournitures
- **validation** : recalcul manuel 1 categorie complete
