# J23 — Exercice HARD : pipeline complet bout-en-bout + benchmark "couverture vs robustesse"

## Objectif

Construire **ton propre mini-pipeline GR00T-style** end-to-end et **mesurer empiriquement** l'effet de chaque axe d'augmentation sur la performance d'une mini-policy. C'est l'exercice qui matérialise le pari "augmentation = robustesse" du module.

Tu vas :
1. Générer un dataset synthétique multi-embodiment / multi-task,
2. Appliquer les 4 axes d'augmentation (background / lighting / distractors / dynamics),
3. Filtrer avec rejet ~30%,
4. Sauvegarder en format LeRobotDataset-like (Parquet ou pickle),
5. Entraîner une **petite policy MLP** (state, image_features → action) sur 4 versions du dataset :
   - sans augmentation,
   - avec **uniquement** lighting,
   - avec **uniquement** dynamics,
   - avec les **4 axes**,
6. **Évaluer** chaque policy sur un set de test **out-of-distribution** (lighting très différent du train) et reporter le delta de performance.

## Consigne détaillée

### A. Génération + augmentation + filtering (réutilisation du code J23)

1. Pars de `02-code/23-synthetic-data-scale.py`. Tu peux **importer** ses fonctions ou recoder les tiennes — au choix.
2. Génère **150 épisodes raw** sur 3 embodiments × 2 tasks, horizon 24, image 8x8x3.
3. Pour chaque expérience, applique l'augmentation appropriée avec `n_augments_per_episode=4`.
4. Filtre avec `max_action_jump=3.0`. Vise ~25-35% de rejet.

### B. Mini-policy MLP

Architecture suggérée (adapte si tu veux, mais reste léger pour CPU) :

```
state (state_dim) ─┐
                   ├─► concat ─► MLP(128, 64) ─► action (action_dim)
image_flat ────────┘
```

- `image_flat = image.flatten() / 255.` (mais dans le code J23 c'est déjà `[0,1]`).
- Loss : MSE sur l'action prédite vs ground-truth (BC simple).
- Optimizer : Adam lr=3e-4.
- Epochs : 5-10 sur tout le dataset.

### C. Set d'évaluation out-of-distribution (OOD)

C'est la partie cruciale. Génère **30 épisodes propres** avec :
- les **mêmes** embodiments / tasks que le train,
- mais **shift agressif** sur lighting (`gain=0.3` ou `gain=1.7`, hors range training) ET sur background (couleurs jamais vues).

C'est l'**OOD lighting**. Tu mesures la MSE moyenne par-step de chaque policy sur ce set.

### D. Tableau de résultats attendu

Après tes 4 entraînements, produis un tableau type :

```
Train augmentation          | MSE in-distribution | MSE OOD-lighting | Delta (%)
----------------------------+--------------------+------------------+----------
none (no aug)               |  X.XXX             |  X.XXX           |  +XXX%
lighting only               |  X.XXX             |  X.XXX           |  +XX%
dynamics only               |  X.XXX             |  X.XXX           |  +XXX%
all 4 axes                  |  X.XXX             |  X.XXX           |  +XX%
```

Et **commente** dans la console au moins 3 phrases :
1. Quel axe seul donne déjà la meilleure robustesse OOD ?
2. Y a-t-il un **gain super-additif** quand on combine les 4 (vs lighting seul) ?
3. Quel est le **coût** en perte de performance in-distribution (s'il y en a un) quand on augmente trop ?

### E. Bonus optionnel

- Sauvegarde au format LeRobotDataset-like (Parquet si pyarrow dispo) et **re-charge** via `ToyLeRobotIterableDataset` pour le training. Vérifie que ça change rien (le format est neutre côté ML).
- Ajoute un **5ᵉ axe** custom (ex. random crop, flip horizontal contraint à laisser l'objet visible) et compare.
- Visualise 1 image avant/après chaque augmentation côte-à-côte avec `matplotlib.subplots`.

## Critères de réussite

- [ ] Le script tourne end-to-end en < 3 minutes sur CPU.
- [ ] Les 4 policies sont entraînées avec exactement le même budget compute (mêmes epochs, lr, batch).
- [ ] Tu observes que **"all 4 axes"** a la **meilleure MSE OOD** (ou très proche). Si ce n'est pas le cas, débugger ton OOD set : il est peut-être trop facile.
- [ ] Tu observes que **"none"** a la **pire MSE OOD** (overfitting au lighting fixe du train).
- [ ] Le tableau est lisible et le commentaire de 3 phrases est concret (pas du blabla).
- [ ] Tu produis (au moins) 1 image qui visualise le shift train→OOD pour illustrer l'écart.

## Pièges classiques

- **OOD trop agressive** : si ton OOD set rend les images quasi-noires (`gain=0.05`), même la policy "all 4 axes" plante. Calibre pour que le shift soit en bordure de la distribution train, pas hors-monde.
- **Set d'évaluation contenant les mêmes épisodes que le train** : oublier de re-générer un set indépendant pour l'OOD = leak. Utilise un seed différent.
- **Comparer des MSE absolues** : selon les graines, les valeurs absolues bougent. Le **ratio MSE-OOD / MSE-in-distribution** est plus stable pour comparer entre runs.
- **Random seed unique** : faire 1 seul run par config peut donner des résultats trompeurs. Si possible, fais 3 seeds et reporte la **médiane** (ou au moins mentionne ta seed exacte).
- **Oublier le filtering** : si tu sautes le filter, tu pollues ton train avec des trajectoires non-physiques (action jumps énormes), la MSE est artificiellement haute. Toujours filtrer.

## Pour aller plus loin

- Implémenter un **vrai relighting diffusion** (lite) : prends un modèle pré-entraîné MobileViT ou un mini-UNet avec conditional channel + transfer-learned tint. C'est l'équivalent jouet d'un `Cosmos-relight` (REFERENCES.md #22).
- Comparer **MSE vs success rate** sur un environnement Gymnasium pédagogique (CartPole conditionné par image) — c'est plus représentatif que la MSE pure.
- Lire le papier GR00T N1 §3.2 (REFERENCES.md #15) et identifier les **5 catégories** de filtres qu'ils utilisent en production. Reporter dans une note quelles tu as reproduites et lesquelles tu as omises.
