# Solution — Projet trésorerie

> Module `09-projet-tresorerie` · Badge : **Trésorier demo** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

Voici un exemple qui marche pour le livrable `Budget-PME-Demo.xlsx` (100 % fictif).

### Schéma CSV / Excel

En-tête :

```text
date,libelle,categorie,montant,type
```

Minimum **15** transactions. Mix entrées / sorties · catégories stables · **aucun** vrai nom de personne ni donnée employeur.

### Onglets suggérés

| Onglet | Rôle |
|--------|------|
| Transactions | lignes brutes |
| Résumé | totaux + solde en français clair |
| (option) Scénarios | plus tard |

### Prompt type (café mobile fictif, janv. 2026)

```
Rôle : assistant tableur pour non-financier.
Contexte : café mobile en ville, janvier 2026, données 100 % fictives.
Tâche : génère 15 lignes CSV colonnes date,libelle,categorie,montant,type.
Contraintes : pas de vrais noms ; montants réalistes ; mix entrées/sorties.
Format : CSV prêt à coller dans Excel.
```

Résumé en **5 lignes** pour un non-financier (quoi est entré, quoi est sorti, solde, 1 alerte, 1 prochaine action).


## Medium (bonus)

### 20+ lignes · 3 onglets

Onglets : **Transactions** · **Resume** · **Readme**

Formules Résumé :

- `SOMME.SI` entrées  
- `SOMME.SI` sorties  
- solde  

Notes d’hypothèses sous le résumé (1–3 phrases).


## Hard (bonus)

### Classeur réutilisable + scénarios

| Scénario | Coeff. entrées | Coeff. sorties | Exemple (base 10 000 / 8 000) |
|----------|----------------|----------------|--------------------------------|
| base | 1.0 | 1.0 | solde **2 000** |
| optimiste | 1.1 | 1.0 | entrées 11 000 · solde 3 000 |
| pessimiste | 0.9 | 1.05 | 9 000 / 8 400 → solde **600** |

Checklist : noms d’onglets clairs · formules documentées dans Readme · jamais de données réelles.

Helper optionnel (code) : `domains/vie/ia-quotidien/02-code/09-projet-tresorerie.py`


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
