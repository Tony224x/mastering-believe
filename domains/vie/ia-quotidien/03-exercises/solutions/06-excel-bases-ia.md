# Solution — Excel + IA : les bases

> Module `06-excel-bases-ia` · Badge : **Formule qui matche** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

### Cas de test (nombres fictifs)
| Date | Libellé | Catégorie | Montant | Type |
|------|---------|-----------|---------|------|
| 2025-01-05 | Vente atelier | Recettes | 600 | entrée |
| 2025-01-08 | Fournitures | Achats | 150 | sortie |

Formules attendues (FR) :
- Total entrées : `=SOMME.SI(E:E;"entrée";D:D)` → **600**
- Total sorties : `=SOMME.SI(E:E;"sortie";D:D)` → **150**
- Solde : `=total_entrées - total_sorties` → **450**

Écran de référence : `assets/screens/screen-excel-somme-si.png`


## Medium (bonus)

### Tableau de 10 lignes + 3 formules
Demande à ChatGPT un tableau | Objectif | Formule FR | Explication | Test | puis **vérifie** chaque total à la main.


## Hard (bonus)

### Gestion d'erreurs
Documente #DIV/0! #REF! #VALEUR! avec cause + correctif (sans VBA).


## Clés structurées (rappel)

### easy

- **sample_data** :
  -
    - 2026-01-02
    - Vente A
    - 200
    - entree
  -
    - 2026-01-03
    - Vente B
    - 250
    - entree
  -
    - 2026-01-04
    - Vente C
    - 150
    - entree
  -
    - 2026-01-05
    - Loyer
    - 80
    - sortie
  -
    - 2026-01-06
    - Fournitures
    - 40
    - sortie
  -
    - 2026-01-07
    - Pub
    - 30
    - sortie
- **formula_fr** : =SOMME.SI(D2:D7;"entree";C2:C7)
- **formula_fr_alt** : selon ordre colonnes Montant/Type — adapter plages
- **expected_entrees** : 600
- **expected_sorties** : 150
- **expected_solde** : 450

### medium

- **formulas** :
  - **entrees** : SOMME.SI sur Type=entree
  - **sorties** : SOMME.SI sur Type=sortie
  - **solde** : entrees - sorties
- **error_drill** : reduire la plage d'une ligne puis lire # éventuel / total faux

### hard

- **sheets** :
  - Transactions
  - Resume
- **indicators** :
  - total entrees
  - total sorties
  - solde
  - NB transactions
- **locale_trap** : #NOM? si SUM au lieu de SOMME
