# Solution — Excel + IA : les bases

> Module `06-excel-bases-ia` · Badge : **Formule qui matche** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

Voici un exemple qui marche — mêmes nombres que la mission.

### Données à coller (fictives)

| Date | Libellé | Montant | Type |
|------|---------|---------|------|
| 2026-01-02 | Vente A | 200 | entrée |
| 2026-01-03 | Vente B | 250 | entrée |
| 2026-01-04 | Vente C | 150 | entrée |
| 2026-01-05 | Loyer | 80 | sortie |
| 2026-01-06 | Fournitures | 40 | sortie |
| 2026-01-07 | Pub | 30 | sortie |

**Calcul manuel :** entrées **600** · sorties **150** · solde **450**

### Formules FR (si colonnes : A Date · B Libellé · C Montant · D Type)

```excel
=SOMME.SI(D2:D7;"entrée";C2:C7)   → 600
=SOMME.SI(D2:D7;"sortie";C2:C7)   → 150
=B_total_entrees - B_total_sorties → 450
```

> **Piège fréquent :** si ta colonne Type contient `entree` **sans accent**, la formule doit utiliser `"entree"` (même orthographe exacte). Sinon le total reste à 0.

Écran de référence : `assets/screens/screen-excel-somme-si.png`

### Prompt ChatGPT utile

```
Rôle : formateur Excel FR pour non-tech.
Contexte : tableau A1:D7, en-têtes Date|Libellé|Montant|Type, lignes 2–7 comme ci-dessus.
Tâche : formule SOMME.SI pour totaliser les entrées ; explique en 3 phrases ; donne un cas de test.
Contraintes : Excel français ; pas de VBA ; pas de données réelles.
```


## Medium (bonus)

### Tableau de 10 lignes + 3 formules

Demande à ChatGPT un mini-tableau de suivi, puis un livrable du type :

| Objectif | Formule FR | Explication | Test |
|----------|------------|-------------|------|
| Total entrées | `=SOMME.SI(...)` | … | = 600 sur le jeu fixe |
| Total sorties | `=SOMME.SI(...)` | … | = 150 |
| Solde | `=entrées-sorties` | … | = 450 |

**Vérifie chaque total à la main.** Drill d’erreur : réduis la plage d’une ligne et observe le total faux / message d’erreur.


## Hard (bonus)

### Gestion d’erreurs (sans VBA)

| Erreur | Cause typique | Correctif |
|--------|---------------|-----------|
| `#DIV/0!` | Division par zéro (ratio si entrées = 0) | `=SI(denominateur=0;"n/a";…)` |
| `#REF!` | Plage supprimée / copiée mal | reconstruire la référence |
| `#VALEUR!` | Texte là où Excel attend un nombre | nettoyer montants |
| `#NOM?` | Formule EN (`SUM`) dans Excel FR | utiliser `SOMME` / `SOMME.SI` |

Classeur type : onglets **Transactions** + **Résumé** (totaux, solde, NB de lignes).


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
- **error_drill** : reduire la plage d’une ligne puis lire # éventuel / total faux

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
