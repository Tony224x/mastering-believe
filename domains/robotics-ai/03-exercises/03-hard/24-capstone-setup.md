# J24 — Exercice HARD : pipeline d'augmentation + split train/val stratifié

## Objectif

Construire la **DataModule complète** que J26 utilisera pour entraîner Diffusion Policy. Tu dois :
- charger le dataset brut produit par J24,
- implémenter un **action chunking** (le formatage de Diffusion Policy : observation à `t`, séquence d'actions de `t` à `t+H-1`),
- ajouter un **pipeline d'augmentation** (flip horizontal + bruit gaussien sur obs),
- implémenter un **split train/val 80/20 par épisode**, stratifié sur le `side` (gauche/droite).

C'est l'exercice qui matérialise concrètement le **contrat de format** entre le dataset brut et le DataLoader d'entraînement.

## Pré-requis

- Le dataset doit avoir été généré (`python domains/robotics-ai/02-code/24-capstone-setup.py`).
- Tu peux utiliser numpy seul ou ajouter PyTorch si tu veux un vrai `Dataset` torch.

## Consigne

### 1. Action chunking

Implémente une fonction `make_chunks(obs, action, ep_start, ep_length, horizon=8)` qui retourne :
- `obs_t`     : `(N_samples, obs_dim)` — l'observation à l'instant `t`
- `action_chunk` : `(N_samples, horizon, action_dim)` — les `horizon` actions à partir de `t`
- `ep_id_t`  : `(N_samples,)` — l'épisode source de chaque échantillon

**Règle** : on ne génère un échantillon que si `t + horizon <= ep_start + ep_length`, i.e. on ne traverse jamais une frontière d'épisode (pad ou skip — choisis skip).

### 2. Augmentation pipeline

Crée une classe `Augmenter` avec un paramètre `p_flip ∈ [0, 1]` et un `sigma_obs ∈ [0, ∞)` qui, étant donné `(obs_t, action_chunk)`, retourne une version augmentée :
- avec proba `p_flip`, **flip horizontal** : `obs_t[0] ← W - obs_t[0]`, `obs_t[2] ← W - obs_t[2]`, `obs_t[4] ← -obs_t[4]` (theta du block) ; et `action_chunk[:, 0] ← -action_chunk[:, 0]` (delta_x s'inverse). `W` = `meta["table_size"]`.
- bruit gaussien sur l'obs : `obs_t ← obs_t + N(0, sigma_obs)` *uniquement* sur les coordonnées spatiales (indices 0..3, pas sur theta — ou avec un sigma plus petit pour theta).
- **important** : l'augmentation ne touche jamais les actions sauf le flip de signe ci-dessus.

### 3. Split train/val stratifié

Implémente `stratified_episode_split(episodes_jsonl, ratio=0.8, seed=0)` qui :
- lit `episodes.jsonl` (un dict par ligne avec `episode_index`, `length`, `success`, `side`),
- partitionne les épisodes en deux groupes selon `side ∈ {"left", "right"}`,
- pour **chaque** groupe, prend 80% en train et 20% en val (random sous seed),
- retourne `(train_ids, val_ids)` deux listes d'indices d'épisodes.

Le but : garantir que train ET val contiennent une proportion équilibrée des deux modes (sinon la val score sur la multimodalité est biaisée).

### 4. Mini DataModule

Assemble le tout :
- une classe `PushTDataModule` qui en `__init__` charge le dataset, splitte, fabrique les chunks pour train et pour val séparément.
- une méthode `iterate(split="train", batch_size=64, augment=True, shuffle=True)` qui yield des `(obs_b, action_b)` numpy ou torch tensors. L'augmentation s'applique uniquement au train.

### 5. Sanity checks à imprimer

- Nombre d'échantillons train et val.
- Ratio left/right dans train et val (doit être ≈ 0.5/0.5 dans les deux).
- Pour 1 batch tiré au hasard : moyenne et std des observations avec et sans bruit (pour vérifier que le bruit a bien l'amplitude attendue).
- Visualiser **1 chunk avant flip** et **1 chunk après flip** pour confirmer visuellement que le flip est cohérent géométriquement.

## Critères de réussite

- [ ] `make_chunks` retourne des shapes cohérentes : `obs_t.shape[0] == action_chunk.shape[0]`.
- [ ] Aucun chunk ne traverse une frontière d'épisode (vérifié via une assertion sur les `ep_id_t`).
- [ ] Le split est stratifié : abs(ratio_left_train - ratio_left_val) < 0.05.
- [ ] L'augmentation flip est **involutive** : appliquer flip deux fois rend l'identité (à epsilon près).
- [ ] Le bruit gaussien produit un std empirique ≈ `sigma_obs` (à 10% près sur 1000+ samples).
- [ ] Le DataModule itère un epoch complet en < 5 s sur CPU.

## Étapes suggérées

1. Charger `obs`, `action`, `ep_start`, `ep_length` (déjà fait à l'EASY).
2. Pour l'action chunking, boucler par épisode : pour `i in range(N)`, pour `t in range(ep_length[i] - horizon + 1)`, ajouter `(obs[ep_start[i] + t], action[ep_start[i] + t : ep_start[i] + t + horizon])`.
3. Lecture de `episodes.jsonl` ligne par ligne avec `json.loads`.
4. Split : `np.random.default_rng(seed).permutation(...)` puis slice 80/20.
5. Augmenter : implémenter une seule passe `_apply(obs, chunk)` puis l'appeler à chaque itération si `augment=True`.

## Pièges classiques

- **Flip incohérent** : oublier de flipper `theta` (ou flipper avec le mauvais signe). Vérifie en plottant : un T flippé doit toujours ressembler à un T (la symétrie miroir d'un T autour de la verticale centrale est un T).
- **Bruit sur theta** : si tu mets le même sigma sur theta que sur les pixels, tu perturbes massivement (theta est en radians, ~0.5 rad de bruit fait tourner le block visuellement de 30°). Soit applique sigma_obs uniquement sur les 4 premiers indices, soit utilise un sigma_theta plus petit (0.01 rad).
- **Stratification mal codée** : si tu fais un split global puis que tu vérifies après, tu peux tomber par malchance sur une distribution déséquilibrée. Splitte à l'intérieur de chaque groupe.
- **Action chunk borderline** : pour `t = ep_length - horizon`, l'inclusion est valide (dernier chunk strict) mais une faute d'off-by-one est facile. Teste avec un mini-dataset (3 épisodes de longueurs 5, 6, 7) et un horizon=3 : tu dois obtenir 3 + 4 + 5 = 12 chunks au total.

## Pour aller plus loin (bonus)

- Implémente une variante **observation history** : au lieu d'`obs_t` seul, retourner `obs_{t-2:t+1}` (3 frames) — c'est ce que Diffusion Policy original utilise. Comment l'augmentation flip se comporte-t-elle alors ? (Réponse : flip s'applique frame par frame, c'est cohérent.)
- Implémente une **normalisation** par dimension : pour J26, on s'attend à ce que les obs soient zero-mean unit-std. Calcule sur le **train uniquement** un (mean, std), applique aux deux splits, sauvegarde dans `meta.json` un champ `obs_normalization` pour que J26 puisse l'utiliser à l'inverse au moment de l'eval.
- Sortie au format **Parquet** (avec `pyarrow`) : convertir le mini DataModule en un format directement consommable par `lerobot` officiel. Tu prépares ainsi le terrain pour J28 si l'on veut migrer du `.npz` au format LeRobotDataset v3.0 standard.
