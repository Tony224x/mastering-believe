# Solution — Nettoyer, analyser, visualiser

> Module `08-nettoyer-analyser` · Badge : **Données propres** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

### Checklist de nettoyage (ordre)
1. Doublons
2. Espaces superflus
3. Dates au même format
4. Catégories standardisées
5. Types (entrée/sortie) homogènes
6. Montants en nombres
Puis **1 graphique sobre** (barres ou lignes, pas de camembert inutile).


## Medium (bonus)

### 12 lignes propres + totaux
Dates ISO · Title Case catégories · Type minuscule · montants numériques.


## Hard (bonus)

### Avant/après documenté
Capture ou tableau « sale » vs « propre » + justification du type de graphique.


## Clés structurées (rappel)

### easy

- **problems** :
  - formats de dates heterogenes
  - espaces autour des libelles
  - casse Type inconsistante (sortie/Sortie)
  - separateur decimal virgule vs point possible
  - doublon loyer potentiel
  - separateur champs ; vs colonnes Excel

### medium

- **standards** :
  - **dates** : YYYY-MM-DD
  - **type** : entree|sortie minuscules
  - **categories** : Title Case
  - **montants** : nombre pur

### hard

- **chart** : barres categories vs total sorties
- **title_example** : Le loyer concentre plus de la moitie des sorties
- **insights_min** : 3
