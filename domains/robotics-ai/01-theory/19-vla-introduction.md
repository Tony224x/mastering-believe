# J19 — VLA introduction : RT-1, RT-2, Open X-Embodiment, Octo

> Durée d'étude : 45-60 min. Objectif : comprendre ce qu'est un Vision-Language-Action (VLA), suivre la généalogie RT-1 → RT-2 → Open X-Embodiment → Octo, et savoir distinguer **action tokenization** (discrétisation + classification) de **action regression** (sortie continue).

---

## 1. Exemple concret avant tout

Imagine une scène simple : un bras robotique Franka pose ses caméras sur une table avec une pomme rouge, une banane et une éponge. Tu lui dis en langage naturel :

> "Pick up the red apple."

Trois entrées arrivent simultanément à la policy :

1. **Vision** : un (ou plusieurs) flux RGB de caméras tiers + caméra-poignet, par ex. 224×224×3 à 5 Hz.
2. **Langage** : la phrase tokenisée (`Pick`, `up`, `the`, `red`, `apple`).
3. **Proprioception** : configuration articulaire actuelle `q ∈ R^7`, vélocité `q̇`, état du gripper.

Sortie attendue : un **delta de pose end-effector** (Δx, Δy, Δz, Δroll, Δpitch, Δyaw) et un état de gripper (open/close), à 5-10 Hz, jusqu'à ce que la pomme soit saisie.

C'est exactement ce qu'un **VLA** fait : il consomme des images + du texte et émet des actions robot. Ce qui change selon le modèle, c'est **comment** la sortie est représentée : **un vecteur continu** (regression) ou **une suite de tokens discrets** (tokenization, comme un LLM qui prédit le prochain mot).

> **Key takeaway** — Un VLA = `(images, instruction texte, état robot) → action(s)`. Le reste (tokenizer, backbone, action head) sont des choix d'architecture qui changent la sample efficiency, la généralisation et le coût d'inférence.

---

## 2. Définition VLA et pourquoi ça émerge en 2022-2024

**VLA = Vision-Language-Action model**. C'est une policy robotique qui hérite de deux univers :

- Les **VLMs** (Vision-Language Models) — type LLaVA, PaLI, SigLIP — qui savent associer image + texte.
- Les **policies robotiques** — qui produisent des actions exécutables sur un robot réel.

Pourquoi cette fusion explose entre 2022 et 2024 ? Trois raisons convergentes :

1. **VLMs pré-entraînés à grande échelle disponibles** (CLIP 2021, PaLI/PaLM-E 2023). On peut piggyback sur leur représentation visuelle/sémantique au lieu de réapprendre depuis zéro.
2. **Datasets robot massifs et standardisés** (Open X-Embodiment, 2024 : 1M+ trajectoires, 22 embodiments). Avant, chaque labo avait ses propres données, non interopérables.
3. **L'échec du multitask par-tâche** : entraîner une policy par tâche ne scale pas. On veut un modèle généraliste qui suit n'importe quelle instruction texte.

Un VLA hérite donc du paradigme "**foundation model**" : une seule policy fine-tunable, qui généralise à des instructions et objets jamais vus en entraînement.

---

## 3. Généalogie : RT-1 (2022) → RT-2 (2023) → Open X-Embodiment (2024) → Octo (RSS 2024)

### 3.1 RT-1 (Robotics Transformer 1) — Google, fin 2022

Première démonstration d'une policy transformer "à la BERT" pour la robotique. Architecture :

- **Vision** : EfficientNet pré-entraîné, conditionné par un embedding texte (FiLM) → 81 tokens visuels par image.
- **Texte** : USE (Universal Sentence Encoder) → un embedding par instruction.
- **Policy** : transformer décodeur qui produit **11 actions discrètes** (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper, mode, terminate, etc.) chacune **discrétisée en 256 bins** → un classifieur softmax sur 256 classes par dimension.
- **Données** : 130k démos sur 700+ tâches (Everyday Robots, 17 mois de collecte).

Acquis pédagogique : RT-1 prouve qu'**un seul transformer policy peut absorber des dizaines de tâches** distinctes si on lui donne assez de démos. Mais il ne profite **pas** d'un pré-entraînement web : il part de zéro côté connaissance du monde.

### 3.2 RT-2 (Robotics Transformer 2) — Google DeepMind, juillet 2023

Saut conceptuel majeur : **co-fine-tuner un VLM web-scale (PaLI-X 55B / PaLM-E 12B) directement sur des données robot**, en représentant les actions comme des **tokens texte** que le VLM apprend à émettre.

Concrètement :

- Une action `(Δx=0.02, Δy=-0.01, ..., gripper=0.8)` est sérialisée en texte → `"1 128 91 241 ..."` (bins discrétisés rendus comme des chaînes de caractères dans le vocabulaire du VLM).
- L'entraînement mélange `vision-language Q&A web` ET `(image robot, instruction) → tokens action`.
- Inférence : le VLM autocomplète les tokens action comme s'il continuait une phrase.

Résultat : **généralisation émergente** à des objets/scènes jamais vus, parce que le backbone connaît "rouge", "fragile", "petit cube vert" depuis le web. RT-2 est le moment où la robotique **hérite directement** des LLMs/VLMs.

> Référence canonique pour le contexte RT-2 : voir REFERENCES.md #13 (OpenVLA paper), §2 "Background and Related Work" — OpenVLA est la version open-source de cette idée.

### 3.3 Open X-Embodiment (OXE) — RSS 2024 + extensions

Avant 2024, chaque labo entraînait sur son propre dataset, ses propres caméras, son propre format d'action. Open X-Embodiment résout ce chaos :

- **22 embodiments** (Franka, UR5, xArm, ALOHA, mobile manipulators, etc.).
- **160 000+ tâches** réparties sur **1M+ trajectoires**.
- Format unifié RLDS (Reinforcement Learning Datasets, basé sur TFDS).

L'idée : si une policy s'entraîne sur **plusieurs robots à la fois**, elle développe des représentations qui transfèrent — comme un LLM entraîné sur 100 langues parle mieux chacune. C'est la **standardisation des datasets** qui rend possible l'ère VLA généraliste.

### 3.4 Octo — RSS 2024

Octo est la première policy **open-source** qui prend OXE au sérieux à l'échelle 800k trajectoires.

Architecture (REFERENCES.md #17) :

- **Encoder vision** : ViT-S ou ViT-B, applique un patch embedding par image, plusieurs caméras supportées.
- **Encoder langage** : T5-base (pour conditioning texte), ou alternative goal-image (pour conditioning par image-but).
- **Backbone** : transformer block-causal qui consomme `[task_tokens, obs_tokens]` et produit des `readout_tokens` à chaque step.
- **Action head** : **diffusion head** (DDIM) qui décode les readout tokens en **séquence d'actions continues** (chunking de 4 actions).

Octo se distingue de RT-1/RT-2 par :

- **Open-source** (vs RT-1/RT-2 fermés).
- **Action regression via diffusion** (vs tokenization discrète chez RT-1/RT-2).
- **Multi-embodiment de naissance** : entraîné sur 9 plateformes simultanées.
- Fine-tuning rapide (10-100 démos) pour une nouvelle config robot.

> **Key takeaway** — La généalogie 2022-2024 va du "transformer policy from scratch" (RT-1) au "VLM web-scale qui parle robot" (RT-2) au "dataset standardisé multi-robot" (OXE) au "policy open-source diffusion-based pour la communauté" (Octo).

---

## 4. Action tokenization vs action regression

C'est le clivage architectural central des VLAs. Connaître ses tradeoffs = comprendre 80% des choix de design.

### 4.1 Action tokenization (RT-1, RT-2, OpenVLA)

Principe : on **discrétise** chaque dimension d'action en N bins (typiquement N = 256). Une action `Δx ∈ [-0.05, 0.05]` devient un entier `bin_id ∈ [0, 255]`. Le modèle prédit ce bin via un **classifieur softmax** (cross-entropy loss).

**Exemple concret** :

```
Δx = 0.012  →  bin = round((0.012 + 0.05) / 0.1 × 255) = 158
```

Avantages :
- Compatible directement avec un LLM autoregressif : on étend le vocabulaire avec des tokens action.
- Action sequence modeling = sequence modeling classique (cross-entropy).
- Capture des distributions multimodales (la pomme peut être saisie par 3 angles différents — les 3 modes ont des bins distincts qui co-existent dans le softmax).

Inconvénients :
- **Quantization error** : la précision est limitée par N. À 256 bins sur une plage de 10 cm, granularité = 0.4 mm — souvent suffisant, mais pas pour de la chirurgie.
- Inférence séquentielle si les dimensions sont émises une par une (lent à 200 Hz).
- Les bins voisins ne sont pas "proches" pour la loss (bin 100 et bin 101 sont aussi loin que bin 100 et bin 250 sous une cross-entropy naïve).

### 4.2 Action regression (Diffusion Policy, Octo, π0)

Principe : sortie **continue**, on prédit directement `(Δx, Δy, ...) ∈ R^d`. Loss MSE classique, ou plus sophistiquée via diffusion / flow matching.

Avantages :
- Précision continue (pas de quantization).
- Plus naturel pour des contrôleurs hauts-débit (200 Hz Helix).
- La diffusion / flow matching capture la **multimodalité** (modes multiples) sans bins discrets.

Inconvénients :
- Une simple MSE produit des actions "moyennes" (mode collapse) sur des distributions multimodales — d'où la nécessité de diffusion / flow matching.
- Pas de fit naturel avec un LLM autoregressif → il faut un "action head" séparé en sortie.

### 4.3 Tableau récap

| Aspect | Tokenization (RT-1/2, OpenVLA) | Regression (Octo, π0, DP) |
|--------|--------------------------------|---------------------------|
| Sortie | bins discrets + softmax        | vecteur continu (diffusion/flow) |
| Loss   | Cross-entropy                  | MSE / score matching      |
| Multimodalité | Native via softmax       | Native via diffusion      |
| Précision | Quantization (N=256 typique) | Continue                  |
| Compatibilité LLM | Très élevée (extension du vocab) | Nécessite head séparé |
| Vitesse inférence | Faible si autoregressif | Élevée si chunking parallèle |

> **Key takeaway** — Tokenization = "le robot parle une langue avec mots = bins d'action". Regression = "le robot prédit un vecteur réel". Les deux sont valides, le choix dépend de la précision requise, de la vitesse de contrôle, et de la pile (LLM vs head custom).

---

## 5. Embodiment et standardisation des datasets

Le mot **embodiment** désigne **la combinaison physique** d'un robot : nombre de DoF, type de gripper, layout des caméras, plage articulaire, mode de contrôle (position, vélocité, torque), fréquence. Deux Franka identiques avec deux gripper différents = deux embodiments.

Pourquoi c'est important pour les VLAs ?

- Une policy entraînée sur un seul embodiment **n'apprend pas** à séparer "ce qui est universel" (saisir un objet rouge) de "ce qui est spécifique au robot" (mes 7 articulations + ce gripper Robotiq).
- Une policy multi-embodiment, à condition d'avoir un format unifié, **généralise mieux** car le réseau est forcé d'extraire des features invariantes au robot.

OXE a posé les bases avec :

- Un **format RLDS unifié** (TFDS-compatible).
- Une convention **action_space** : delta-EE 7-DoF (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper) comme représentation commune, même si chaque robot a son contrôleur natif différent en backend.
- **Métadonnées d'embodiment** explicites pour que le modèle puisse conditionner sur le robot s'il en a besoin.

LeRobot (Hugging Face, 2025-2026) a pris le relais avec un format Parquet+MP4 streamable, conventions normalisées, et 100+ datasets indexés sur le Hub.

> **Key takeaway** — Un VLA généraliste **a besoin** d'embodiment-awareness et de datasets standardisés. Sans ça, on retombe dans l'ère "une policy par robot par tâche".

---

## 6. Limites actuelles des VLAs (état mai 2026)

À retenir pour ne pas survendre la techno :

- **Coût d'inférence** : RT-2 55B ne tourne qu'en cloud. OpenVLA 7B sur GPU consumer marche mais à 5-10 Hz max — insuffisant pour du contrôle réactif (200 Hz Helix). D'où les architectures **dual-system** System1/System2 (J22).
- **Dynamique** : la plupart des VLAs sont entraînés sur démos quasi-statiques (pick-and-place lent). Ils ne savent pas verser un café à toute vitesse ni rattraper un objet qui tombe.
- **Long-horizon** : suivre "fais-moi un café" reste hors de portée — il faut un planner symbolique au-dessus, ou un LLM qui décompose en sous-instructions.
- **Données** : 1M trajectoires OXE c'est ~7 ordres de grandeur de moins que les corpus texte (10T tokens). On est encore data-starved.

---

## 7. Récap visuel

```
              VLM web-scale (PaLI, SigLIP, Llama2)
                          │
                          ▼
   ┌──────────────────────┴──────────────────────┐
   │                                             │
RT-1 (2022)                                  RT-2 (2023)
transformer + action bins                    VLM co-fine-tuned + action tokens
130k démos, 1 robot                          PaLI-X 55B, généralisation web
   │                                             │
   └──────────────┬──────────────────────────────┘
                  ▼
       Open X-Embodiment (RSS 2024)
       1M trajectoires, 22 embodiments
       format RLDS unifié
                  │
                  ▼
              Octo (RSS 2024)
              ViT + T5 + diffusion head
              open-source, 800k traj OXE
                  │
                  ▼
          OpenVLA (2024) → π0 (2024) → GR00T N1 (2025) → Helix (2025)
```

---

## 8. Q&A spaced-repetition (5 cartes)

**Q1** — Quelle est la différence essentielle entre RT-1 et RT-2 ?
**R1** — RT-1 entraîne un transformer policy **from scratch** sur des démos robot (pas de connaissance du monde héritée). RT-2 **co-fine-tune un VLM web-scale** (PaLI-X / PaLM-E) directement sur des actions sérialisées en tokens texte → généralisation émergente à des objets/scènes inédits.

**Q2** — Donne deux avantages de l'action tokenization par rapport à l'action regression naïve (MSE).
**R2** — (1) Capture native des distributions multimodales (les bins coexistent dans le softmax, pas de mode collapse). (2) Compatibilité directe avec un LLM autoregressif : on étend le vocabulaire avec les tokens action et on entraîne avec une cross-entropy classique.

**Q3** — Pourquoi Open X-Embodiment a été un tournant ?
**R3** — Avant OXE, chaque labo avait son format de données → impossible de combiner. OXE a unifié 22 embodiments / 1M+ trajectoires sous un format RLDS commun (action_space delta-EE 7-DoF), permettant l'entraînement de policies multi-embodiment et l'apparition de modèles généralistes comme Octo, OpenVLA, π0.

**Q4** — Quelle est l'architecture d'action head d'Octo, et pourquoi ce choix ?
**R4** — Octo utilise une **diffusion head (DDIM)** qui décode les readout tokens en séquence continue d'actions (chunking de 4). Choix motivé par : (1) précision continue (pas de quantization), (2) capture native de la multimodalité via diffusion, (3) chunking parallèle = inférence rapide.

**Q5** — Qu'est-ce qu'un **embodiment** et pourquoi c'est central pour les VLAs ?
**R5** — Un embodiment = la combinaison physique d'un robot (DoF, gripper, caméras, plage articulaire, mode de contrôle, fréquence). Une policy mono-embodiment ne sépare pas universel (saisir un objet) du spécifique (ce gripper). Une policy multi-embodiment, sur un format de données standardisé, est forcée d'extraire des features invariantes au robot → meilleure généralisation.

---

## 9. Sources principales utilisées

- **REFERENCES.md #17 — Octo: An Open-Source Generalist Robot Policy**, Octo Model Team, RSS 2024 (https://arxiv.org/abs/2405.12213). Source primaire pour l'architecture transformer + diffusion head, OXE, multi-embodiment fine-tuning.
- **REFERENCES.md #13 — OpenVLA: An Open-Source Vision-Language-Action Model**, Kim et al., 2024 (https://arxiv.org/abs/2406.09246). Source primaire pour le contexte RT-1 / RT-2, action tokenization à 256 bins, héritage VLM.
- Pour aller plus loin : Open X-Embodiment paper original (Padalkar et al., RSS 2024) référencé dans REFERENCES.md #13 §2 et REFERENCES.md #17 §3.
