# Solution — Nettoyer, analyser, visualiser

> Module `08-nettoyer-analyser` · Badge : **Données propres** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

Voici un exemple qui marche — dans cet ordre, sans sauter d’étape.

### Checklist de nettoyage

1. Doublons (ex. loyer collé deux fois)
2. Espaces superflus autour des libellés
3. Dates au **même** format
4. Catégories standardisées
5. Types (`entrée` / `sortie`) homogènes — même casse
6. Montants en **nombres** (pas de texte « 80 € »)

Puis **1 graphique sobre** (barres ou lignes). Évite le camembert sauf si les parts sont vraiment le message.

### Problèmes typiques dans un export « sale »

| Problème | Symptôme |
|----------|----------|
| Dates hétérogènes | `01/02/2026` vs `2026-02-01` |
| Espaces | `" Loyer "` ≠ `"Loyer"` dans un `SOMME.SI` |
| Casse Type | `Sortie` vs `sortie` |
| Décimales | virgule vs point selon locale |
| Doublon | même loyer deux fois |
| Séparateur | `;` vs colonnes Excel |


## Medium (bonus)

### 12 lignes propres + totaux

Standards à appliquer :

| Champ | Standard |
|-------|----------|
| dates | `YYYY-MM-DD` |
| type | `entree` \| `sortie` (minuscules, orthographe **unique**) |
| catégories | Title Case (ex. `Fournitures`) |
| montants | nombre pur |

Prompt utile : « Voici 12 lignes sales (fictives). Renvoie un tableau propre selon ces standards + 3 totaux. »


## Hard (bonus)

### Avant / après documenté

- Tableau « sale » vs « propre » (capture ou 2 tableaux côte à côte)
- Graphique : **barres** catégories vs total sorties
- Titre d’insight type : « Le loyer concentre plus de la moitié des sorties »
- ≥ 3 insights en une phrase chacun (pas de chiffre inventé hors tableau)


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
