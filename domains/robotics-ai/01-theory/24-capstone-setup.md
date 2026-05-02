# J24 — Capstone setup : PushT environment + génération de démos + dataset

> Durée d'étude : 45-60 min
> Prérequis : J13 (imitation learning, BC vs DAgger), J15 (diffusion + flow matching), J16 (Diffusion Policy deep dive), J23 (synthetic data + sim-to-real à scale).
> Sources principales : REFERENCES.md #19 (Diffusion Policy repo — environment PushT), #27 (LeRobot v0.4 + LeRobotDataset v3.0), #24 (MuJoCo Documentation 3.x).

Tu es **jour 1/5** du capstone. Sur 5 jours (J24→J28) on construit Diffusion Policy from scratch sur la tâche canonique **PushT**. Aujourd'hui : produire le **dataset de démos expertes** que les jours suivants vont consommer pour entraîner et évaluer la policy. Pas de réseau de neurones aujourd'hui — du code "monde + expert + logging" propre.

---

## 1. Hook — pourquoi PushT est devenue *le* benchmark Diffusion Policy

En 2023, Cheng Chi et al. (Columbia/TRI, RSS best paper) cherchent une tâche assez **simple** pour itérer vite, mais assez **subtile** pour exposer les limites de Behavior Cloning naïf. Ils retiennent **PushT** : un agent circulaire pousse un **bloc en forme de T** sur une table 2D pour le faire coïncider avec une **target T** dessinée au sol. Variantes en pixel et en state.

Pourquoi ce choix précis :

1. **Action multimodale évidente.** Pour amener le T au target, l'expert peut commencer par pousser à droite OU à gauche du bloc — deux trajectoires valides. Un BC MSE moyenne les deux et produit une trajectoire moyenne *qui ne pousse rien*. C'est exactement le mode-collapse que la diffusion résout.
2. **Action 2D continue + observation 2D.** On peut tout coder sans MuJoCo, sans GPU, sans rendu : un T en polygones, un agent en cercle, des collisions analytiques. Le pipeline pédagogique reste lisible.
3. **Shaping de reward facile.** IoU entre le T courant et le T target → score ∈ [0, 1] continu, parfait pour mesurer un success rate.
4. **Démos scriptables.** Une heuristique géométrique simple (s'aligner derrière le bloc puis pousser vers la target) génère des trajectoires "expertes" en quelques ms.

> Concrètement, dans le repo `real-stanford/diffusion_policy`, l'env vit dans `diffusion_policy/env/pusht/pusht_env.py`. Il est en pure pymunk + pygame. Nous, on va faire encore plus simple : numpy seul, simulation 2D sans physics engine externe — on garde la sémantique (T-block + agent + target), pas le rigid body solver.

> **Key takeaway #1**
> PushT = T-block 2D à pousser sur une target T avec un agent circulaire. Les choix de design (multimodalité, action 2D continue, IoU comme reward) en font le benchmark où Diffusion Policy bat tous ses concurrents. C'est *la* tâche de validation du capstone.

---

## 2. Anatomie de l'environnement PushT — état, action, reward

### 2.1 État du monde

Trois entités :

| Objet | Représentation | DoF |
|---|---|---|
| **Agent (pusher)** | Cercle de rayon `r_agent` (typique 15 px) | position 2D `(x, y)` |
| **T-block** | Polygone en T (assemblage de 2 rectangles) | position 2D + orientation `(x, y, θ)` |
| **Target T** | Polygone en T fixe sur le plan | (statique, paramètre du monde) |

**Observation** dans le mode "state" : `obs = [agent_x, agent_y, block_x, block_y, block_theta]` — 5 floats. Suffisant pour Diffusion Policy state-based, et c'est ce qu'on vise pour le capstone (plus rapide à entraîner que la version pixel).

### 2.2 Action

`action = [delta_x, delta_y]` — déplacement 2D demandé pour l'agent au prochain step. Bornée par `|delta| ≤ a_max`. Le step de simulation déplace l'agent puis applique la collision agent↔block (push) et bord↔block (clipping).

### 2.3 Reward / success

`success = IoU(T_block, T_target) ≥ 0.95`. Ce critère est *binaire* à la fin de l'épisode mais peut être tracé continu en cours d'épisode pour debugging.

```
table 512 × 512 px
                ┌────────────────────────────┐
                │        T target (gris)     │
                │     ┌────┬────┬────┐       │
                │     │    │    │    │       │
                │     │    └────┘    │       │
                │     │              │       │
                │     └──────────────┘       │
                │                            │
                │   ┌─T-block (bleu)──┐      │
                │   │                 │      │
                │   │       O ← agent │      │
                │   └─────────────────┘      │
                └────────────────────────────┘
```

### 2.4 Multimodalité — la question centrale

Pour pousser le block vers le haut, l'agent peut :
- contourner par la **gauche** puis pousser depuis le bas-gauche,
- contourner par la **droite** puis pousser depuis le bas-droite.

Les deux séquences `(x_t, y_t)` sont valides étant donné le même état initial. **Une policy gaussienne unimodale échoue parce qu'elle moyenne** — elle pousse "tout droit" sans contourner et ne fait rien bouger. Diffusion Policy modélise une distribution multimodale dans l'espace des trajectoires d'actions, donc elle peut échantillonner l'une OU l'autre.

> **Key takeaway #2**
> Le multimodal action distribution est *le* phénomène que PushT exhibe et que Diffusion Policy adresse. Sans ça, BC suffirait. C'est pour ça que le dataset doit contenir **les deux types** de trajectoires expertes (gauche-puis-pousse, droite-puis-pousse), sinon on perd la raison d'être du capstone.

---

## 3. Génération de démos — téléopération vs script

### 3.1 Deux paradigmes industriels

Dans LeRobot (REFERENCES.md #27), deux pipelines coexistent pour produire des démos :

| Pipeline | Outil | Avantage | Inconvénient |
|---|---|---|---|
| **Téléopération keyboard / leader-follower** | `lerobot record` + clavier ou bras leader | Démos *humaines* avec stratégies riches | Lent : ≈ 1 démo / 30 s humain |
| **Script expert (heuristique ou planneur)** | Code Python, MPC, RRT | Génère 1000+ démos en minutes | Diversité bornée par l'heuristique |

En pratique pour PushT :
- Le repo `real-stanford/diffusion_policy` fournit **200 démos téléopérées humaines** dans son dataset officiel. C'est ce qui "entraîne sérieusement" la policy.
- Pour un capstone pédagogique, on **scripte un expert géométrique** : 100-200 démos générées en < 30 s sur CPU. La diversité est plus faible mais on injecte de l'aléa pour conserver la multimodalité (cf. §3.3).

### 3.2 Anatomie de l'expert scripté

L'heuristique pour amener le T-block vers le target :

1. Calculer le vecteur `e = pos_target - pos_block`.
2. Choisir un **côté de poussée** : si `e.y > 0` (target au-dessus), il faut pousser depuis le bas ; calculer le point d'application `contact = pos_block + offset_below`.
3. **Phase 1 — alignement** : déplacer l'agent vers `contact` (en contournant le block, jamais à travers).
4. **Phase 2 — poussée** : déplacer l'agent dans la direction `e` normalisée tant que `IoU < 0.95`.

Pour générer la **multimodalité**, on randomise :
- le **side de contournement** (gauche ou droite) avec une bernoulli 0.5,
- la position initiale du block et de l'agent,
- un petit bruit gaussien sur chaque action `a ← a + ε, ε ~ N(0, 0.5 px)`.

Sans cette randomisation, les 200 démos seraient quasi identiques et la policy n'aurait rien à apprendre de "multimodal".

### 3.3 Rejection sampling — ne garder que les démos qui réussissent

Toutes les démos générées ne convergent pas (heuristique imparfaite + bruit). On garde uniquement celles avec `success_final = True` ET `episode_length < max_steps`. En pratique on génère 250 candidats pour obtenir 100-200 valides — exactement le pattern qu'utilise NVIDIA Cosmos pour curer ses 780k trajectoires synthétiques (REFERENCES.md #15, vu en J23).

> **Key takeaway #3**
> Un expert scripté + randomisation de stratégie + rejection sampling sur le success final = recette minimale pour produire un dataset multimodal de qualité. C'est exactement la philosophie data-curation des frontier labs (NVIDIA, TRI), juste à petite échelle pédagogique.

---

## 4. Format de dataset — LeRobotDataset, structure et alternatives

### 4.1 LeRobotDataset v3.0 (REFERENCES.md #27)

Le format standard 2026 de Hugging Face pour les démos robotiques. Couches :

| Couche | Contenu | Format |
|---|---|---|
| **`meta/info.json`** | Schéma : nom des features, shapes, dtypes, fps | JSON |
| **`meta/episodes.jsonl`** | Une ligne par épisode : `episode_index`, `length`, `task` | JSONL |
| **`data/chunk-XXX/`** | Tableaux temporels des observations + actions | Parquet (1 fichier / chunk d'épisodes) |
| **`videos/chunk-XXX/`** | Frames d'observation (si caméra) | MP4 |

Avantages vs un gros pickle ou H5 :
- **Streamable** depuis HuggingFace Hub (pas besoin de tout télécharger pour itérer dataset).
- **Schema-first** : les shapes sont déclarées une fois, le code de chargement est uniforme.
- **MP4 pour les vidéos** : 100× plus compact que des frames numpy brutes.
- **Compatible `lerobot.policy.diffusion`** out-of-the-box : J26 pourra entraîner sans glue code.

### 4.2 Compromis pour le capstone

LeRobotDataset officiel demande l'install complète de `lerobot` (qui compile `pyav` etc.). Pour un capstone pédagogique on choisit un **mini-format compatible mais simplifié** :

```
artifacts/pusht_demos/
├── meta.json                # {n_episodes, fps, obs_dim, action_dim, max_episode_len}
├── episodes.jsonl           # 1 ligne par épisode : {idx, length, success, side}
└── data.npz                 # tous les tenseurs, indexés par offsets
                              #   obs       (T_total, 5)         float32
                              #   action    (T_total, 2)         float32
                              #   ep_id     (T_total,)           int32
                              #   ep_start  (n_episodes,)        int32
                              #   ep_length (n_episodes,)        int32
```

Pourquoi `.npz` plutôt que Parquet : zéro dépendance externe (numpy seul), suffit pour < 10 000 transitions, et la migration vers Parquet/LeRobot en J28 est triviale (3 lignes `pd.DataFrame(...).to_parquet(...)`).

### 4.3 Contrat exact pour les jours suivants (J25-J28)

Toi tu dois écrire ce dataset, **les autres jours doivent pouvoir le charger sans te demander quoi que ce soit**. Le contrat figé :

```python
# Charger tout le dataset
data = np.load("artifacts/pusht_demos/data.npz")
obs       = data["obs"]        # (T_total, 5)  : [agent_x, agent_y, block_x, block_y, block_theta]
action    = data["action"]     # (T_total, 2)  : [dx, dy] in pixels per step, bornée à a_max
ep_start  = data["ep_start"]   # (N,) int32   : indice de début de chaque épisode dans obs/action
ep_length = data["ep_length"]  # (N,) int32   : longueur de chaque épisode

# Itérer un épisode i
s, l = ep_start[i], ep_length[i]
ep_obs    = obs[s : s + l]
ep_action = action[s : s + l]
```

Le code du jour produit ce fichier dans `artifacts/pusht_demos/data.npz`. Les modules J25-J28 le consomment.

> **Key takeaway #4**
> Un dataset propre = schéma documenté + contrat figé. Tu produis aujourd'hui un .npz avec exactement les clés `obs`, `action`, `ep_start`, `ep_length`, plus un `meta.json`. C'est le pacte entre J24 et les jours suivants.

---

## 5. Augmentation et split train/val

### 5.1 Augmentation — pourquoi et comment

Sur PushT state-based (5 floats), les augmentations possibles sont **modestes** comparé au mode pixel :
- **Symétrie horizontale** : si on flip `x`, le problème reste valide en flippant aussi `target_x` et `block_theta → -block_theta`. Double effectivement le dataset.
- **Bruit gaussien** sur les observations : `obs ← obs + N(0, σ_obs)` simule une caméra/encoder bruité. σ ≈ 1-2 px.
- **Bruit sur les actions** : déjà fait à la génération (§3.2), inutile de re-bruiter.

Sur PushT pixel-based (cas non utilisé ici mais à savoir) :
- **Random crop** (Diffusion Policy original) : crop aléatoire des 96×96 frames.
- **Color jitter** : variations RGB pour robustesse à l'éclairage.

Ces augmentations sont **appliquées dans le DataLoader** au moment du training (J26), pas à la génération. Le dataset brut reste fidèle.

### 5.2 Split train/val

Convention Diffusion Policy : **80/20 par épisode** (pas par transition). Pourquoi par épisode :
- Sinon des transitions consécutives du même épisode se retrouvent dans train ET val → leak temporel, val score gonflé artificiellement.
- 200 épisodes → 160 train + 40 val — assez pour tracker la perte de validation.

On split **après** le rejection sampling, en shuffle stratifié sur le `side` de contournement (50% gauche, 50% droite dans chaque split) pour que les deux modes soient représentés en val.

> **Key takeaway #5**
> Split par épisode, jamais par transition. Stratifier sur le mode (gauche/droite) garantit que la val mesure bien la généralisation multimodale.

---

## 6. Synthèse / Mind map du jour

```
J24 — Capstone Day 1 — Setup PushT + dataset
├─ Environment
│  ├─ Agent (cercle r=15px) + T-block (polygone) + Target T (fixe)
│  ├─ Obs state-based (5 floats) — pas de pixel pour rester rapide
│  ├─ Action 2D continue (delta_x, delta_y), bornée a_max
│  └─ Success = IoU(block, target) >= 0.95
├─ Expert scripté
│  ├─ Phase 1 : aligner agent derrière block (côté randomisé)
│  ├─ Phase 2 : pousser dans direction (target - block)
│  └─ Bruit gaussien + multi-côté => multimodalité préservée
├─ Génération
│  ├─ 200 démos cibles (génère ~250 candidats)
│  ├─ Rejection sampling sur success final
│  └─ Mélange ~50% gauche / ~50% droite
├─ Dataset (artifacts/pusht_demos/)
│  ├─ data.npz : obs, action, ep_start, ep_length
│  ├─ meta.json : n_episodes, fps, dims
│  └─ episodes.jsonl : un dict par épisode (success, side, length)
├─ Augmentation (à appliquer en J26 dataloader, pas ici)
│  ├─ Flip horizontal (double dataset)
│  └─ Bruit gaussien obs (σ ≈ 1-2 px)
└─ Split train/val
   ├─ 80/20 par épisode
   └─ Stratifié sur side (gauche/droite)

→ J25 chargera obs+action et entraînera ResNet18 + UNet1D + DDPM
→ J27 utilisera le même env pour rollouts d'évaluation
→ J28 packagera tout pour démo finale
```

---

## 7. Q&A spaced repetition

**Q1.** Pourquoi PushT est-il *le* benchmark canonique de Diffusion Policy plutôt qu'une tâche manipulation MuJoCo standard ?
**R.** Parce que sa structure expose explicitement la **multimodalité** des actions valides (pousser par la gauche OU par la droite), ce qui est *exactement* le phénomène que la diffusion modélise mieux que les baselines unimodales (BC MSE, gaussienne).

**Q2.** Pourquoi générer les démos par script plutôt que par téléopération humaine pour le capstone ?
**R.** Vitesse : 200 démos en < 30 s contre ≈ 1 h en téléopération. On accepte une diversité moindre — compensée par randomisation du côté de contournement et du bruit, suffisante pour exhiber la multimodalité.

**Q3.** Quelles sont les 4 clés exactes du fichier `data.npz` produit aujourd'hui, et que représentent-elles ?
**R.** `obs` (T_total, 5) états successifs ; `action` (T_total, 2) actions correspondantes ; `ep_start` (N,) indice de début de chaque épisode dans les tableaux plats ; `ep_length` (N,) longueur de chaque épisode.

**Q4.** Pourquoi splitter le dataset par épisode plutôt que par transition pour le train/val ?
**R.** Pour éviter le **leak temporel** : des transitions consécutives du même épisode contiennent presque la même info, les retrouver à la fois en train et en val gonfle artificiellement la val score et masque l'overfitting.

**Q5.** Pourquoi appliquer le bruit gaussien sur les observations dans le DataLoader (J26) et non à la génération (J24) ?
**R.** Pour pouvoir varier σ pendant le training/eval sans regénérer le dataset, et pour échantillonner du bruit *différent* à chaque epoch (data augmentation). Si on bruitait à la génération, on figerait un seul réalisation du bruit.

**Q6 (bonus).** Pourquoi un mini-format `.npz` au lieu du LeRobotDataset officiel ?
**R.** Zéro dépendance externe (numpy seul), pas d'install lerobot ni pyav, suffit pour 200 épisodes × 100 steps = 20 000 transitions. Migration vers Parquet/LeRobot triviale au moment du packaging J28 si besoin.

---

## 8. Sources citées (REFERENCES.md)

- **#19** — Chi et al. (Columbia/TRI), *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023, https://diffusion-policy.cs.columbia.edu/. Repo : https://github.com/real-stanford/diffusion_policy. **Source de l'environnement PushT** (pusht_env.py) et du format de démos. C'est notre référence directe pour la sémantique de la tâche, la définition du success (IoU 0.95), et la structure des trajectoires expertes.
- **#27** — Hugging Face, *LeRobot v0.4 + LeRobotDataset v3.0*, 2025-2026, https://huggingface.co/docs/lerobot/index. **Source du format de dataset cible** (Parquet + MP4 streamable). On en adopte la philosophie schema-first et la convention episode-level dans une version simplifiée `.npz` adaptée au capstone.
- **#24** — Google DeepMind, *MuJoCo Documentation 3.x*, 2026, https://mujoco.readthedocs.io/. Contexte stack physique pour la suite ; pas utilisé directement aujourd'hui (on simule en 2D pur numpy) mais cité pour expliquer pourquoi on simplifie : MuJoCo serait surdimensionné pour une tâche 2D collision-only.

---

## 9. Pour aller plus loin (optionnel)

- Cloner `real-stanford/diffusion_policy` et lire `diffusion_policy/env/pusht/pusht_env.py` (~150 lignes pymunk). Le comparer à notre version 2D numpy pour mesurer le gain de simplicité.
- Explorer le dataset officiel PushT : 200 démos téléopérées disponibles sur le site Diffusion Policy. Comparer la diversité humaine vs scriptée en plottant les histogrammes d'angle initial de poussée.
- Lire la doc LeRobotDataset v3.0 (REFERENCES.md #27) section "Schema reference" pour voir comment migrer notre `.npz` vers Parquet en J28.
