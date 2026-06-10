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

## Criteres de reussite

- [ ] Le script s'exécute sans erreur sur le dataset par défaut et le rapport imprime `n_episodes == 200` et `sum(ep_length) == obs.shape[0] == action.shape[0]`.
- [ ] Les 3 vérifications de cohérence sont des `assert` dans le script (pas des prints), et elles passent : `ep_start` strictement croissant avec `ep_start[0] == 0`, comptage des transitions, borne sur la norme des actions (`<= a_max + epsilon`, `a_max` lu depuis `meta.json`).
- [ ] La figure `exo_easy_trajectories.png` est créée et montre 8 trajectoires distinctes, agent en trait plein, block en pointillés, une couleur par épisode.
- [ ] Tu as écrit en 1 phrase (commentaire de fin de script) une observation qualitative vérifiable sur la figure, par exemple "les trajectoires d'agent contournent le block au lieu d'aller en ligne droite" — et la figure la confirme.

## Indices

- `np.diff(ep_start) > 0` te dit en une ligne si `ep_start` est strictement croissant.
- Pour la borne d'action : `np.linalg.norm(action, axis=1).max()` doit être `<= a_max + 0.5` environ (le bruit gaussien d'expert ajoute un peu, mais l'env clippe — le clip se fait *après* le bruit, donc c'est borné).
- Si tu veux compter les sides (gauche/droite), lis `episodes.jsonl` (un dict par ligne, parser avec `json.loads`).
