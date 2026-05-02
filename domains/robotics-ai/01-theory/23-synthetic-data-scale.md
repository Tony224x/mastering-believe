# J23 — Synthetic data + sim-to-real à scale

> Durée d'étude : 45-60 min
> Prérequis : J14 (sim-to-real, domain randomization), J18 (NVIDIA Cosmos), J19 (Open X-Embodiment), J20-J21 (VLA OpenVLA / π0).
> Sources principales : REFERENCES.md #22 (NVIDIA Cosmos), #15 (GR00T N1 + data pipeline), #27 (LeRobotDataset format).

---

## 1. Hook — pourquoi NVIDIA a fabriqué 780 000 trajectoires synthétiques pour GR00T N1

Quand NVIDIA publie GR00T N1 en mars 2025 (REFERENCES.md #15), une ligne du papier saute aux yeux : le modèle est entraîné sur **88h de téléopération réelle** + **780 000 trajectoires synthétiques générées en simulation**. Le ratio est ~9 000:1 en faveur du synthétique. C'est l'inverse complet de l'idée naïve "le réel domine, la sim est un fallback".

Pourquoi ce pari ? Parce que **collecter 1 démonstration réelle coûte ~5 minutes de téléopérateur humain** sur un robot Apptronik / 1X / Figure (mouvement bras + gripper, scène à filmer, étiquetage). À 100 demos/heure et un humanoïde par opérateur, atteindre 1 million de trajectoires demanderait 10 000 heures-humain — soit ~5 ans-personne à temps plein, sans compter le coût hardware. **En sim parallélisée sur 1 GPU node** (Isaac Lab + Cosmos), on génère 780 000 trajectoires variées (objets, lighting, distractors) en quelques jours.

Le problème : **le reality gap reste**. Une policy entraînée 100% en sim s'effondre sur un robot réel. La recette industrielle 2025 mélange donc :
1. **Beaucoup de synthétique** pour la couverture (skills, objets, scènes),
2. **Un peu de réel** pour ancrer la dynamique fine (frottements, contacts, sensor noise),
3. **De la randomization agressive** pour rendre la policy robuste à l'écart résiduel.

> **Key takeaway #1**
> En 2025-2026 le dataset d'entraînement d'un VLA généraliste n'est plus "des démos humaines" mais un **stack hiérarchique** : Open X (réel multi-robots) + sim massive (Isaac Lab) + post-traitement vidéo (Cosmos). Le réel est la *pièce ancre*, le synthétique est le *volume*.

---

## 2. Pourquoi le synthétique est devenu viable en 2024-2025

### 2.1 Trois progrès simultanés

| Domaine | Avant 2023 | 2024-2025 |
|---|---|---|
| **Rendu visuel sim** | OpenGL classique, "look 3D des années 2010", reality gap énorme | Path tracing GPU temps-réel (Isaac Sim/Lab, Omniverse), photoreal proche du réel |
| **Génération de scènes** | Manuelle, par scientifique data | Procédurale + LLM-driven layout + asset libraries 100k+ objets |
| **Augmentation post-rendu** | Crops, color jitter | **Cosmos video tokenizer** : reskin complet (background swap, relighting, distractors) |

### 2.2 Le coup d'accélérateur Cosmos (REFERENCES.md #22)

NVIDIA Cosmos (janvier 2025) a entraîné un **foundation model vidéo** sur 20 millions d'heures de vidéo physique réelle. Il sert deux usages distincts :

1. **Curation** : tokenizer vidéo (CV8x8x8) qui compresse 8x8x8 pixels-temps en 1 token → on peut filtrer / rechercher dans des datasets vidéo massifs (Open X-Embodiment, ego4D, robotique YouTube).
2. **Augmentation** : modèles diffusion + autoregressifs qui prennent une trajectoire sim "moche" et la **reskinent** en photoreal (repaint background, relighting, distractor objects), tout en **préservant la trajectoire d'action** (les contacts, le geste). C'est un **video-to-video conditionné par les actions du robot**.

Concrètement : avec Cosmos, ta démo simulée d'un bras Franka qui pose une tasse devient *visuellement* une vidéo réaliste de cuisine d'Airbnb, **sans avoir à modéliser cette cuisine en 3D**. Tu multiplies arbitrairement la diversité visuelle.

> **Key takeaway #2**
> Cosmos transforme une simu "moche mais physiquement correcte" en une vidéo "photoreal mais physiquement consistante". On découple la **vérité physique** (Isaac Lab/MuJoCo) de la **vérité visuelle** (Cosmos diffusion). Le VLA voit des images quasi-réelles avec des actions ground-truth.

---

## 3. Pipeline NVIDIA GR00T pour 780k trajectoires (REFERENCES.md #15)

C'est le pipeline le plus documenté — autant prendre celui-là comme cas concret.

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Source : ~88h démonstrations réelles téléopération         │
│    - 7 humanoides (Fourier GR1, Apptronik, Unitree, etc.)    │
│    - Tâches : pick & place, pour, plier, ouvrir tiroir, ...  │
└────────────────────────┬─────────────────────────────────────┘
                         │
              ┌──────────▼───────────┐
              │ 2. Isaac Lab replay  │  ← injection en simulation
              │    + retargeting     │
              └──────────┬───────────┘
                         │
       ┌─────────────────┴────────────────┐
       │                                  │
       ▼                                  ▼
┌─────────────┐                  ┌────────────────┐
│ 2a. Domain  │                  │ 2b. Trajectory │
│ randomization│                 │ perturbation   │
│ (per replay) │                 │ (init noise,   │
│              │                 │  goal jitter)  │
└──────┬───────┘                 └────────┬───────┘
       │                                  │
       └──────────┬───────────────────────┘
                  ▼
        ┌────────────────────┐
        │ 3. Multi-camera    │  ← captures depuis 3-5 viewpoints
        │    rendering       │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ 4. Cosmos relight  │  ← reskin photoreal (REFERENCES.md #22)
        │    + bg swap       │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ 5. Filtering :     │  ← rejet trajectoires non-physiques
        │    contact valid,  │     (tasse qui passe à travers main),
        │    success check   │     non-success, NaN sensors
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ 6. LeRobotDataset  │  ← Parquet + MP4 streamables
        │    (Hugging Face)  │     (REFERENCES.md #27)
        └────────────────────┘
                  ▼
              780k trajectoires augmentées synthétiques
              + 88h réelles (gardées telles quelles)
```

### 3.1 Étape critique : le retargeting

Une démo réelle est captée sur le robot A (ex. Fourier GR1, 32 DoF). Pour démultiplier, il faut la **rejouer sur les robots B, C, D** en simu (Apptronik, Unitree H1, etc.) — chacun a sa propre cinématique. **Retargeting** = solveur IK qui mappe les trajectoires end-effector du robot source vers les configurations articulaires du robot cible. Quand ça marche, **1 démo réelle → 7 démos synthétiques différentes**, multi-embodiment "for free".

### 3.2 Étape critique : le filtering

Sans filtrage, ton dataset sera pollué par des trajectoires impossibles : la cinématique inverse a divergé, le bras passe à travers la table, le gripper ferme à vide. **GR00T N1 rejette ~30% de ses trajectoires** générées (chiffre du papier). Les filtres types :

- **Contact validity** : la pénétration objet-objet < 1cm pendant la trajectoire.
- **Goal completion** : à la fin, l'état du monde matche bien le but (objet sur cible).
- **Smoothness** : pas de saut articulaire > seuil (signe de divergence solver).
- **Sensor sanity** : pas de NaN dans les images (artefact rendering).

> **Key takeaway #3**
> Le pipeline n'est pas "rendre puis donner au modèle" — c'est **filtrer agressivement**. Mieux vaut 700k trajectoires propres que 1M sales. Le filtrage est une étape *non-négligeable* en compute (souvent ~20% du budget total du pipeline).

---

## 4. Augmentation : les 4 axes qui comptent

Les VLA généralistes 2025 (GR00T, Helix, π0.5) random-isent au minimum sur ces 4 axes pendant la génération de données :

### 4.1 Background

Swap de la **scène** entière. En sim : on charge des asset libraries d'environnements (Habitat-Sim, ProcTHOR, scenes 3D Cosmos). En post-prod : reskin Cosmos qui change le fond mais pas le robot ni les objets manipulés.

**But** : la policy ne doit pas s'accrocher à des features de fond ("cette nuance de mur signifie qu'il y a une tasse à gauche"). C'est **anti-overfitting visuel**.

### 4.2 Lighting

Random sur la position et l'intensité des sources, sur la HDRI background, sur la couleur (température) de l'éclairage.

**But** : robustesse au passage de jour/nuit, lampe/spot, intérieur/extérieur. C'est l'augmentation la moins chère (juste un changement de matrices de rendering) et celle qui donne le meilleur rapport robustesse/coût.

### 4.3 Distractors

Ajout d'**objets non-pertinents** dans la scène (autre tasses, jouets, papiers). En sim : sample N objets random depuis une asset library, place-les avec collision check.

**But** : la policy doit apprendre à **focuser sur l'objet d'intérêt** (donné par l'instruction langage), pas à mémoriser un layout. C'est l'augmentation qui force le langage à *vraiment* contribuer.

### 4.4 Dynamics

Random sur **mass**, **friction**, **damping**, **latence sensor/action**, **noise sur les caméras**. Inspiré de Tobin 2017 (domain randomization).

**But** : fermer le **dynamics gap** sim-to-real. Un robot réel a des frottements imprévus, des actuateurs avec backlash, des caméras avec rolling shutter — la policy s'entraîne sur une **distribution de dynamiques** plutôt qu'une dynamique fixée.

### 4.5 Tableau résumé

| Axe | Coût compute | Couverture sim-to-real | Risque overfitting si absent |
|---|---|---|---|
| Background | Moyen (re-rendering) | Visuel | Énorme (la policy apprend la scène) |
| Lighting | Faible | Visuel | Moyen |
| Distractors | Moyen (collision check) | Sémantique | Énorme (langage non-utilisé) |
| Dynamics | Faible (rééchantillonner masses) | Physique | Énorme (transferts ratés) |

> **Key takeaway #4**
> Les 4 axes sont **complémentaires, pas redondants**. Une policy entraînée avec 3 sur 4 a une faiblesse identifiable. GR00T, Helix et π0.5 randomisent les 4. C'est l'état de l'art 2025.

---

## 5. Open X-Embodiment — le standard datasets multi-robots

Open X-Embodiment (Padalkar et al., 2024) n'est pas un *modèle* mais un **standard de dataset** : ~1M trajectoires aggrégées de 22 institutions, ~22 embodiments différents (Franka, UR5e, Kuka, ALOHA, Sawyer, ...), ~500 skills.

### 5.1 Pourquoi ça compte

Avant Open X, chaque labo avait son format propriétaire. Octo (J19), OpenVLA (J20) et π0 (J21) **n'auraient pas pu exister** sans ce standard : impossible d'entraîner un modèle généraliste si les datasets sources n'ont pas une API commune.

### 5.2 Le format (préfigure LeRobotDataset)

Chaque trajectoire = séquence de transitions :
```
{
  "observation/image_primary":  uint8 (T, H, W, 3)
  "observation/image_wrist":    uint8 (T, H, W, 3)   # optionnel
  "observation/state":          float32 (T, dim_state)
  "action":                     float32 (T, dim_action)
  "language_instruction":       str
  "embodiment":                 str ("franka_panda", ...)
  "task":                       str
}
```

### 5.3 Limitations connues

- **Distribution biaisée** : 80% des trajectoires sont du tabletop pick-and-place. Pas de manipulation longue, peu de mobile.
- **Qualité hétérogène** : labos avec démos imparfaites (jitter, instructions vagues).
- **Pas de synthétique** : c'est uniquement du réel — d'où l'intérêt de combiner avec sim massive.

GR00T N1 et Helix l'utilisent comme **base réelle**, puis ajoutent leur propre génération synthétique par-dessus. C'est le pattern dominant 2025.

---

## 6. LeRobotDataset v3.0 — le format standard 2026 (REFERENCES.md #27)

Hugging Face `lerobot` (v0.4, 2025-2026) a standardisé le format pour les datasets robotique modernes. C'est le format **de fait** pour publier un dataset VLA en 2026.

### 6.1 Layout disque

```
my_dataset/
├── meta/
│   ├── info.json          # version, robot_type, fps, total_episodes, ...
│   ├── stats.json         # mean/std par feature (pour normalisation)
│   ├── tasks.jsonl        # {"task_index": 0, "task": "pick the red cup"}
│   └── episodes/
│       └── chunk-000.parquet  # episode_index → length, task_index, ...
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # actions/state au fps natif
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.cam_high/episode_000000.mp4
        ├── observation.images.cam_low/episode_000000.mp4
        └── ...
```

### 6.2 Pourquoi ce format

1. **Parquet** = colonnar, compressé, streamable depuis Hugging Face Hub sans tout télécharger.
2. **MP4 séparé** = vidéo encodée en H.264, ~10-50× plus compact que des frames PNG, décodage rapide via PyAV/torchvision.
3. **Métadonnées JSON** = lisibles humain, intégrables `datasets.load_dataset(...)` direct.
4. **Chunking** = un chunk de 1000 épisodes = 1 fichier Parquet = adapté au Hub (limite 50GB / fichier).

### 6.3 Pourquoi pas TFDS ?

Open X était en TFDS (TensorFlow Datasets). LeRobotDataset gagne sur :
- **Pas de dépendance TF** (PyTorch first → écosystème robotique 2025).
- **Streaming natif** sans téléchargement complet.
- **Vidéos H.264** vs frames raw → 10-50× moins de stockage.

> **Key takeaway #5**
> Quand tu publies un dataset robotique en 2026, pousse-le au format LeRobotDataset sur le Hugging Face Hub. C'est lisible par PI0, GR00T, OpenVLA, Octo et tous les frameworks modernes. C'est le **`tokenizers/transformers` du dataset robotique**.

---

## 7. Limitations sim-to-real à scale — le reality gap résiduel

Tout n'est pas rose. Trois limitations structurelles persistent en 2026 :

### 7.1 Le contact gap

Les **contacts** sont la partie la moins fidèle de la sim. Frottements, déformations, glissement résiduel — MuJoCo et Isaac Lab simplifient. Une policy qui **agrippe parfaitement en sim** peut **glisser en réel**. Random sur friction aide mais ne couvre pas tout.

**Mitigation industrielle** : ajouter un bloc de **fine-tuning sur ~50-200 démos réelles** après pré-entraînement synthétique. C'est ce que fait Helix Logistics (REFERENCES.md #16).

### 7.2 Le sensor gap

Les caméras réelles ont du **rolling shutter**, du **motion blur**, de la **compression JPEG inline**, des artefacts d'autofocus. Cosmos relight produit des images plus belles que la réalité — paradoxe : trop photoréaliste devient *out-of-distribution* par rapport au capteur final.

**Mitigation** : ajouter explicitement les artefacts capteurs dans le pipeline de rendering (motion blur, JPEG noise, color shift).

### 7.3 Le distribution shift "long tail"

Le synthétique couvre bien les **cas centraux** (objet propre, lumière neutre, distractor visible). Il rate les **cas rares** (reflet sur objet métallique, poussière, occlusion partielle bizarre). En déploiement, ces cas rares dominent les échecs.

**Mitigation** : **online learning / RLHF post-deploy** (cf. J22 Helix, où le déploiement même sert à collecter les failure modes). C'est la frontière 2026-2027.

> **Key takeaway #6**
> Le synthétique ne **remplace pas** le réel. Il **multiplie** sa couverture. La recette gagnante 2025 est : **synthétique massif + petite ancre réelle + fine-tune réel post-déploiement**. Un labo qui claim "100% synthétique en prod" sur des tâches non-triviales ment ou est dans un domaine très restreint.

---

## 8. Synthèse — comment on assemble un dataset pour entraîner un VLA généraliste

Décision pratique en 2026, ordre suggéré :

```
1. Définir le périmètre de tâches/embodiments cibles.
   → Si single-arm tabletop : Open X-Embodiment + 50h propres = peut suffire pour finetune.
   → Si humanoide multi-task : il FAUT générer du synthétique massif (>100k trajectoires).

2. Collecter le réel ancre.
   → 50-200h de demos téléopération propres, 1-3 robots, ~10-30 tâches.
   → Format LeRobotDataset directement.

3. Set up le pipeline sim.
   → Isaac Lab (NVIDIA) ou MuJoCo MJX (Google) pour la sim massive parallélisée.
   → Asset library : Objaverse + ProcTHOR pour scènes diverses.
   → Retargeting cinématique vers tous les embodiments cibles.

4. Génération massive avec randomization 4-axes.
   → Background / lighting / distractors / dynamics.
   → Multi-camera viewpoints (3-5 caméras par scène).
   → Cosmos relighting si tu vises le photoreal sans modeler 3D.

5. Filtering aggressif.
   → Contact validity + success + smoothness + sensor sanity.
   → Rejeter ~20-40% des trajectoires.

6. Mix réel + synthétique avec ratio agnostic.
   → Pas 50/50 — souvent 1:10 ou 1:100 réel:synthétique pour generalist VLA.
   → Sample weighting : sur-pondérer le réel pendant le training (~10× le poids du synthétique).

7. Évaluer en réel + détecter failure modes.
   → Online collection des échecs → re-injection au prochain cycle de training.
```

---

## 9. Mind map du jour

```
Synthetic data + sim-to-real à scale
├─ Pourquoi
│  ├─ Coût téléopération réelle prohibitif (5min/démo)
│  └─ Sim 2024+ devenue photoréaliste (Cosmos, Isaac Lab path tracing)
├─ Pipeline GR00T-style (REFERENCES.md #15)
│  ├─ 88h réel téléop ancrage
│  ├─ Replay multi-embodiment + DR
│  ├─ Multi-camera rendering
│  ├─ Cosmos relight (REFERENCES.md #22)
│  ├─ Filtering 30% rejet
│  └─ Export LeRobotDataset (REFERENCES.md #27)
├─ Augmentation 4 axes
│  ├─ Background (anti-scene-overfit)
│  ├─ Lighting (cheap & robuste)
│  ├─ Distractors (force langage)
│  └─ Dynamics (sim-to-real physique)
├─ Standards
│  ├─ Open X-Embodiment (réel, multi-labos)
│  └─ LeRobotDataset v3.0 (Parquet + MP4, HF Hub)
└─ Limites résiduelles
   ├─ Contact gap (frottements)
   ├─ Sensor gap (rolling shutter, JPEG)
   └─ Long-tail (cas rares non-couverts par sim)
```

---

## 10. Q&A spaced repetition

**Q1.** Pourquoi 780k trajectoires synthétiques + seulement 88h réelles dans GR00T N1 ?
**R.** Parce que collecter du réel coûte ~5 min de téléopération humaine par démo, donc 1M démos = 10 000 heures-humain inatteignable. Le synthétique parallélisé (Isaac Lab + Cosmos) génère le même volume en quelques jours-GPU. Le réel sert d'**ancre dynamique** ; le synthétique sert de **volume de couverture**.

**Q2.** Qu'apporte Cosmos (REFERENCES.md #22) à un pipeline de génération synthétique par rapport à de la sim "classique" Isaac Lab ?
**R.** Cosmos prend une démo simulée (visuellement "moche" mais physiquement correcte) et la **reskin** en photoreal (background swap, relighting) **tout en préservant les actions ground-truth**. Cela découple la **vérité physique** (sim) de la **vérité visuelle** (Cosmos), et multiplie arbitrairement la diversité visuelle sans modéliser de nouvelles scènes 3D.

**Q3.** Quels sont les 4 axes d'augmentation utilisés par tous les VLA généralistes 2025 ?
**R.** **Background** (scène/fond), **lighting** (sources, HDRI, température), **distractors** (objets non-pertinents pour forcer l'attention guidée par langage), **dynamics** (mass, friction, latence, sensor noise). Les 4 sont **complémentaires** — en oublier un crée une faiblesse identifiable.

**Q4.** Pourquoi LeRobotDataset (REFERENCES.md #27) a remplacé TFDS comme format standard en 2026 ?
**R.** Parquet + MP4 séparés ⇒ streamable depuis HF Hub, vidéos compressées 10-50× (vs frames PNG), pas de dépendance TensorFlow (PyTorch first), métadonnées JSON lisibles, chunking adapté aux limites Hub. Compatible nativement PI0, GR00T, OpenVLA, Octo.

**Q5.** Quel est le reality gap résiduel le plus difficile à fermer même avec une sim photoréaliste ?
**R.** Le **contact gap** (frottements, déformations, glissement) — MuJoCo et Isaac Lab simplifient la mécanique des contacts. La mitigation industrielle (Helix, π0.5) consiste à fine-tuner sur ~50-200 démos réelles après pré-entraînement synthétique. Le **sensor gap** (rolling shutter, JPEG, motion blur) est le second sur la liste — la sim peut être *trop* belle et devenir OOD par rapport au capteur final.

**Q6 (bonus).** Quel ratio réel:synthétique recommanderais-tu pour un VLA généraliste humanoide en 2026 ?
**R.** ~1:10 à 1:100 en volume brut, mais avec un **sample weighting** qui multiplie le poids du réel par ~10× pendant le training (le réel apparaît plus souvent dans les batchs qu'en proportion brute). Et toujours un fine-tune final sur ~50-200 démos réelles spécifiques au déploiement cible.

---

## 11. Sources citées (REFERENCES.md)

- **#22** — NVIDIA (Balaji et al.), *"Cosmos World Foundation Model Platform for Physical AI"*, janvier 2025, https://arxiv.org/abs/2501.03575 ; code https://github.com/nvidia-cosmos. **Source principale pour la curation/augmentation vidéo.**
- **#15** — NVIDIA Research, *"GR00T N1: An Open Foundation Model for Generalist Humanoid Robots"*, mars 2025, https://arxiv.org/abs/2503.14734 ; repo https://github.com/NVIDIA/Isaac-GR00T. **Source principale pour le pipeline 780k trajectoires.**
- **#27** — Hugging Face, *"LeRobot v0.4 + LeRobotDataset v3.0"*, 2025-2026, https://huggingface.co/docs/lerobot/index. **Source principale pour le format dataset standard.**
- **#16** — Figure AI, *"Helix Logistics"*, 2025, https://www.figure.ai/news/helix-logistics (mention fine-tune réel post-deploy).
- **#11** — Berkeley CS285 Lecture 13 — Sergey Levine (sim-to-real, domain randomization).

---

## 12. Pour aller plus loin (optionnel)

- Lire le papier Cosmos (REFERENCES.md #22) en focus sur **§4 (Tokenizer)** et **§5 (Post-training for Physical AI)** — c'est ce qui justifie l'usage en data augmentation.
- Survoler GR00T N1 §3.2 sur le pipeline de données — chiffres exacts par catégorie de tâche.
- Code à manipuler : le module `lerobot.common.datasets` (REFERENCES.md #27) — particulièrement `LeRobotDataset` et `LeRobotDatasetMetadata` qu'on touche dans le code J23.
- Pour le capstone J24+ on utilisera ce format directement pour stocker les démos PushT.
