# J23 — Exercice MEDIUM : pipeline filter + stats sur un dataset jouet

## Objectif

Reproduire l'**étape de filtering** du pipeline GR00T (REFERENCES.md #15) qui rejette typiquement ~30% des trajectoires synthétiques générées. Tu vas écrire une fonction de filtre, l'appliquer à un mini-dataset, et produire des **statistiques de rejet** par catégorie de raison.

## Consigne

1. Génère un mini-dataset de **100 trajectoires jouets** où chaque trajectoire contient :
   - `state : np.ndarray (T=20, 4)`,
   - `action : np.ndarray (T=20, 2)`,
   - `image : np.ndarray (T=20, 8, 8, 3)`.

2. Injecte volontairement des **défauts** dans ~40% des trajectoires :
   - 10% : un saut d'action > 5.0 (simule une IK qui diverge),
   - 10% : un état avec norme > 100 (simule trajectoire explosée),
   - 10% : une valeur NaN dans `image` (artefact rendering),
   - 10% : une valeur Inf dans `state` (sensor sanity fail).
   Les 60% restants sont propres.

3. Écris une fonction `filter_episode(ep, max_action_jump=3.0, max_state_norm=50.0) -> (bool, str)` qui retourne `(keep, reason)` :
   - `keep=False, reason="non-finite values in <field>"` si NaN/Inf dans state/action/image,
   - `keep=False, reason="action jump <X> > <max>"` si un saut d'action dépasse `max_action_jump`,
   - `keep=False, reason="state norm <X> > <max>"` si norme état max dépasse `max_state_norm`,
   - `keep=True, reason="ok"` sinon.

4. Boucle sur le dataset, classe les rejetés par raison, et **affiche un tableau** :

```
total       : 100
kept        : XX (XX.X%)
rejected    : XX (XX.X%)
  - non-finite values in image    : XX
  - non-finite values in state    : XX
  - action jump XX.XX > 3.0       : XX
  - state norm XX.XX > 50.0       : XX
```

5. **Bonus** : trace un histogramme matplotlib des `max(jump)` par épisode pour visualiser la distribution des sauts d'action — c'est le diagnostic typique pour ajuster le seuil `max_action_jump`.

## Étapes suggérées

1. Fonction de génération de trajectoire propre (parametric curve simple).
2. Fonction d'injection de défauts (4 types, sample uniformément).
3. La fonction `filter_episode` (faire les checks dans l'ordre coût-décroissant : NaN d'abord, puis seuils numériques).
4. Boucle de filtrage avec accumulation des stats dans un `dict[str, int]`.
5. Print formaté.

## Critères de réussite

- [ ] Le script tourne sans crash et termine en < 5 secondes.
- [ ] Le taux de rejet observé est **proche de 40%** (à ±5% près, dépend du seed).
- [ ] Les 4 catégories de défauts sont **chacune** détectées par la fonction de filtre (au moins 1 occurrence chacune).
- [ ] Le tableau de stats est lisible et inclut les **noms exacts** des raisons de rejet.
- [ ] Tu peux expliquer en 2 phrases pourquoi **GR00T rejette ~30%** de ses trajectoires synthétiques en pratique (REFERENCES.md #15).

## Pièges classiques

- **Oublier le check NaN/Inf en premier** : si tu fais `state.max()` sur un array contenant des NaN, le résultat est NaN et tu peux passer à côté du filtre.
- **Mettre `max_state_norm` trop bas** : tu vas rejeter aussi les trajectoires propres dont la norme est légitimement élevée (ex. mobile manipulator). Calibre sur la distribution des trajectoires propres d'abord.
- **Ne pas tracer la distribution avant de fixer le seuil** : `max_action_jump=3.0` est un nombre arbitraire ; en pratique tu prends le percentile 99 de la distribution observée sur un set "propre" connu.
- **Compter les rejets en double** : si une trajectoire a 2 défauts, ta logique doit la rejeter pour le **premier** trouvé (et compter UNE fois), pas itérer tous les checks.
