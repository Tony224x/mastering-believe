# J24 — Exercice EASY : explorer le dataset PushT généré

## Objectif

Vérifier que tu sais :
- **charger** le dataset produit par `02-code/24-capstone-setup.py`,
- **compter** correctement le nombre d'épisodes et de transitions,
- **plotter** quelques trajectoires pour valider visuellement la génération.

Aucune théorie de Diffusion Policy n'est requise. C'est un exercice de "data hygiene" — savoir parler à un dataset avant de l'entraîner.

## Pré-requis

Avoir exécuté au préalable :

```bash
python domains/robotics-ai/02-code/24-capstone-setup.py
```

Ce qui crée `artifacts/pusht_demos/data.npz`, `artifacts/pusht_demos/meta.json`, et `artifacts/pusht_demos/episodes.jsonl`.

## Consigne

Écris un script `solution_easy.py` (ou un notebook) qui :

1. **Charge** `data.npz` avec `numpy.load(...)` et lit le `meta.json`.
2. **Affiche** un mini-rapport :
   - nombre d'épisodes,
   - nombre total de transitions,
   - longueur moyenne / min / max des épisodes,
   - shape exacte de `obs` et de `action`,
   - dtypes.
3. **Vérifie** la cohérence :
   - `ep_start` est strictement croissant et `ep_start[0] == 0`,
   - `sum(ep_length) == obs.shape[0] == action.shape[0]`,
   - `max(action[:, 0]**2 + action[:, 1]**2) <= a_max**2 + epsilon` (avec `a_max` lu depuis `meta.json`).
4. **Plotte** sur une figure les trajectoires de l'agent (`obs[:, 0:2]`) **et** du block (`obs[:, 2:4]`) pour les 8 premiers épisodes. Couleur différente par épisode, le block en pointillés. Sauvegarde la figure dans `artifacts/pusht_demos/exo_easy_trajectories.png`.

## Étapes suggérées

1. `import numpy as np ; import json` (pas besoin de pandas).
2. `meta = json.load(open("artifacts/pusht_demos/meta.json"))`.
3. `data = np.load("artifacts/pusht_demos/data.npz")` puis `obs, action, ep_start, ep_length = data["obs"], data["action"], data["ep_start"], data["ep_length"]`.
4. Pour chaque épisode `i`, `s, l = int(ep_start[i]), int(ep_length[i])` puis `obs[s:s+l]`.
5. Pour le plot, utilise `plt.plot(traj[:, 0], traj[:, 1])` pour l'agent.

## Critères de réussite

- [ ] Le script s'exécute sans erreur sur le dataset par défaut (200 épisodes).
- [ ] Les 3 vérifications de cohérence passent (assertions vraies).
- [ ] La figure montre clairement 8 trajectoires distinctes, agent en plein, block en pointillés.
- [ ] Tu peux dire en 1 phrase ce que tu observes : par exemple "les trajectoires d'agent contournent visiblement le block, pas en ligne droite".

## Indices

- `np.diff(ep_start) > 0` te dit en une ligne si `ep_start` est strictement croissant.
- Pour la borne d'action : `np.linalg.norm(action, axis=1).max()` doit être `<= a_max + 0.5` environ (le bruit gaussien d'expert ajoute un peu, mais l'env clippe — le clip se fait *après* le bruit, donc c'est borné).
- Si tu veux compter les sides (gauche/droite), lis `episodes.jsonl` (un dict par ligne, parser avec `json.loads`).
