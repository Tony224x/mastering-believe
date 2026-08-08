# Solution — Projet trésorerie

> Module `09-projet-tresorerie` · Badge : **Trésorier demo** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

### Livrable minimum `Budget-PME-Demo.xlsx` (fictif)
Onglets suggérés : Transactions · Résumé · (option) Scénarios.
Résumé 5 lignes en français pour un non-financier.
**Jamais** de données employeur réelles.


## Medium (bonus)

### 3 scénarios + notes d'hypothèses
Base / optimiste / pessimiste avec écarts explicités.


## Hard (bonus)

### Classeur réutilisable + README
Checklist J9 cochée · noms d'onglets clairs · formules documentées.


## Clés structurées (rappel)

### easy

- **csv_header** : date,libelle,categorie,montant,type
- **n_min** : 15
- **audit** :
  - pas de vrais noms
  - mix entrees/sorties
  - categories stables

### medium

- **sheets** :
  - Transactions
  - Resume
  - Readme
- **n_min** : 20
- **resume_formulas** :
  - SOMME.SI entrees
  - SOMME.SI sorties
  - solde

### hard

- **sheets** :
  - Transactions
  - Resume
  - Scenarios
  - Readme
- **scenarios** :
  - **base** :
    - 1.0
    - 1.0
  - **optimiste** :
    - 1.1
    - 1.0
  - **pessimiste** :
    - 0.9
    - 1.05
- **python_helper** : domains/vie/ia-quotidien/02-code/09-projet-tresorerie.py
- **example_math** : base 10000/8000 -> solde 2000; pessimiste 9000/8400 -> solde 600
