# Solution — Formules & tableaux

> Module `07-formules-tableaux` · Badge : **Modèle lisible** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

Voici un exemple qui marche pour un mini-modèle trésorerie.

### Onglet Transactions

Colonnes type : Date · Libellé · Catégorie · Montant · Type · **Sens**

Formule **Sens** (signe +/− selon entrée/sortie), ligne 2 :

```excel
=SI(E2="entree";D2;-D2)
```

(Adapte la lettre de colonne si ton Type n’est pas en E.)

### Contrôle d’intégrité

```
SOMME(Sens)  ==  total entrées − total sorties
```

Si les deux côtés diffèrent : une ligne a un Type mal orthographié ou un montant texte.


## Medium (bonus)

### Onglet Résumé séparé

| Cellule | Contenu |
|---------|---------|
| B2 | total entrées (`SOMME.SI`) |
| B3 | total sorties |
| B4 | solde (= B2 − B3) |
| B5 | ratio sorties / entrées |

Garde-fou division par zéro :

```excel
=SI(B2=0;"n/a";B3/B2)
```

Références **stables** (plages nommées ou colonnes de tableau Excel), pas de plages fragiles copiées à la main.

Erreurs de copie fréquentes : mauvaise plage · formule EN · en-têtes inclus dans `SOMME`.


## Hard (bonus)

### Scénarios base / optimiste / pessimiste

3 colonnes d’hypothèses (ou 3 blocs clairs) + résumé qui bascule.

Catégories mini (4) : loyer · salaires · marketing · fournitures.

Validation : recalcul manuel d’**une** catégorie complète (ligne par ligne) pour coller au total affiché.


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
