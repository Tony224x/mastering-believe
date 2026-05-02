# J16 — Diffusion Policy (Chi 2023) deep dive

> Lecture : 45-60 min. Pré-requis : J13 (Imitation Learning, Behavior Cloning, distribution shift) et J15 (DDPM, score matching, sampling). Source principale : **REFERENCES.md #19** (Chi et al., RSS 2023, best paper) — site et repo `real-stanford/diffusion_policy`. Background mathématique : **REFERENCES.md #23** (MIT 6.S184).

---

## 0. Exemple concret avant la théorie

Imagine une démonstration de "push-T" : un opérateur humain pousse une pièce en T sur une table jusqu'à une cible. Tu enregistres 200 démos. À un instant `t` donné, l'état est : *pièce à 12 cm de la cible, légèrement tournée à droite*.

Que va faire un humain ? Deux stratégies sont parfaitement valides :
- **Stratégie A** : contourner par la gauche, pousser le coin droit pour redresser, puis avancer.
- **Stratégie B** : contourner par la droite, pousser l'autre flanc, faire pivoter dans l'autre sens.

Sur 200 démos, environ 50% suivent A, 50% suivent B. La distribution conditionnelle `p(action | observation)` a **deux modes** bien séparés.

Maintenant entraîne un Behavior Cloning classique (MLP qui régresse `a = f_θ(o)` avec une MSE) :
- À l'optimum MSE, le réseau prédit la **moyenne** des deux modes : `(A+B)/2`.
- Cette moyenne est… **droit sur la pièce**, ce qui ne fait rien d'utile (souvent même une collision frontale).
- Résultat : success rate ≈ 0%.

Diffusion Policy (Chi 2023) résout ce problème : au lieu de régresser une action, elle modélise **toute la distribution multimodale** `p(a₁:k | o)` via un processus de débruitage itératif. À l'inférence, on échantillonne — on récupère soit A, soit B, jamais leur moyenne.

> **Key takeaway 0** : la raison d'être de Diffusion Policy n'est PAS "c'est joli avec des U-Nets", c'est **régler le mode-collapse de BC sur des distributions d'actions multimodales**, qui est la norme et non l'exception en manipulation humaine.

---

## 1. Le problème central : multimodalité des actions humaines

### 1.1 Pourquoi BC échoue

Behavior Cloning suppose implicitement une distribution **unimodale gaussienne** autour de la prédiction (perte MSE = log-vraisemblance d'une N(μ, σ²I)). Sur une démonstration humaine :

- Trajectoires multi-stratégies (gaucher/droitier, court/long).
- Pauses, corrections, hésitations (l'humain n'est pas un MDP propre).
- Goûts personnels (un opérateur attaque un objet par un côté préféré).

Une perte MSE moyenne ces stratégies → action floue, souvent **non admissible physiquement** (au milieu de l'objet).

### 1.2 Solutions historiques (et leurs limites)

| Approche | Idée | Limite |
|----------|------|--------|
| Mixture Density Network (MDN) | Sortie = K gaussiennes | Effondrement sur 1 mode, instable à entraîner |
| Energy-Based Model (IBC, Florence 2021) | Apprendre `E(o,a)` | Inférence chère (échantillonnage MCMC), instable |
| VAE conditionnel | Latent code z, decode action | Mode collapse fréquent |
| **Diffusion Policy** | Apprendre à débruiter une séquence d'actions | Stable, multimodal, exact |

Chi et al. (2023) montrent empiriquement que **diffusion policy bat IBC, BET, LSTM-GMM** sur 11/12 tâches de manipulation, avec des gains de +20 à +40 points de success rate.

> **Key takeaway 1** : DDPM transforme l'apprentissage d'une distribution complexe en une cascade de petits pas de débruitage gaussien — chacun individuellement gérable par un MLP/UNet, mais qui composés représentent n'importe quelle distribution multimodale.

---

## 2. Anatomie de Diffusion Policy

### 2.1 Vue d'ensemble (3 briques)

```
        observation o_t (image + état)
                  │
                  ▼
   ┌───────────────────────────────┐
   │  Visual encoder (ResNet18)    │  ← brique 1 : conditioning
   │  + state MLP                  │
   └───────────────┬───────────────┘
                   │  feature c (vecteur ~512)
                   ▼
   ┌───────────────────────────────┐
   │  Denoising network ε_θ        │  ← brique 2 : prédiction de bruit
   │  (UNet 1D ou Transformer)     │
   └───────────────┬───────────────┘
                   │
                   ▼
   ┌───────────────────────────────┐
   │  DDPM scheduler (forward +    │  ← brique 3 : process diffusif
   │  reverse, K=100 steps train,  │
   │  10-16 DDIM en eval)          │
   └───────────────┬───────────────┘
                   ▼
        action chunk a_{t:t+T_a}
```

### 2.2 Brique 1 — Conditioning visuel

Le repo officiel `real-stanford/diffusion_policy` propose deux variantes :

- **`DiffusionUnetImagePolicy`** : encoder = ResNet18 (depuis `torchvision`), pretrained ImageNet, **GroupNorm** au lieu de BatchNorm (BN crashe avec petits batch + EMA). On retire le `fc` final → vecteur 512.
- **`DiffusionUnetHybridImagePolicy`** : ResNet18 + état low-dim concaténé.
- **Variante Transformer** : remplacer UNet par decoder transformer. Plus expressif, plus dur à entraîner sur petits datasets.

Astuce du repo : on **n'envoie pas o_t seul**, mais la fenêtre des `T_o` dernières observations (`T_o = 2` typiquement). Ça donne un signal de vitesse implicite.

### 2.3 Brique 2 — UNet 1D pour débruiter une séquence

On ne débruite pas une image (2D), mais une **séquence d'actions** (1D : axe temporel × dim_action).

- Entrée : `a^k ∈ ℝ^{T_p × d_a}` (tenseur "trajectoire bruitée").
- Conditioning : `c` (feature visuelle) injecté via **FiLM** (`γ(c) * h + β(c)`) à chaque bloc résiduel.
- Timestep `k` : embedding sinusoïdal + MLP, ajouté pareil.
- Architecture : `Conv1D + GroupNorm + Mish` × 4-6 niveaux down/up + skip connections (UNet classique).

Pourquoi UNet 1D et pas un MLP simple ? La structure temporelle des actions a une régularité forte (smoothness, périodicité) que les convolutions 1D capturent gratuitement, et les skip connections permettent au modèle de retrouver les détails fins même quand `k` est grand (très bruité).

### 2.4 Brique 3 — DDPM scheduler

Forward (training) : on tire `k ~ Uniform(1, K)`, `ε ~ N(0, I)`, et :

```
a^k = √(ᾱ_k) · a^0 + √(1 - ᾱ_k) · ε
```

avec `ᾱ_k = ∏_{i=1}^{k} (1 - β_i)` et `β_k` la schedule (linear ou squared cosine — le repo utilise **squared cosine**, qui dégrade plus doucement aux extrêmes).

**Loss simplifiée Ho 2020** :

```
L = E_{t, k, ε} [ ‖ ε - ε_θ(a^k, k, c) ‖² ]
```

Soit : "le réseau apprend à prédire le bruit qu'on a ajouté".

Reverse (inference) : on part de `a^K ~ N(0,I)` et on itère :

```
a^{k-1} = (a^k - σ_k · ε_θ(a^k, k, c)) / √(1 - β_k) + bruit
```

En pratique on remplace DDPM par **DDIM** (Song 2020) en eval pour passer de 100 → 10-16 steps sans perte de qualité — critique pour la latence (10-20Hz contrôle).

> **Key takeaway 2** : DDPM training = MSE sur du bruit gaussien. C'est exactement aussi simple à coder qu'un BC. La complexité est dans le **schedule** et le **sampling** — pas dans la loss.

---

## 3. Action chunking + receding horizon

### 3.1 Pourquoi prédire une **séquence** d'actions

BC classique prédit `a_t` à partir de `o_t`. Diffusion Policy prédit `a_{t:t+T_p}` (séquence de longueur `T_p = 16` typiquement). Trois bénéfices majeurs :

1. **Cohérence temporelle** : un échantillon = une intention complète. On ne risque plus de "switcher de mode" entre deux pas de temps (ex : à t=0 on échantillonne stratégie A, à t=1 stratégie B → comportement chaotique).
2. **Idle handling** : pendant les pauses humaines (latence de 100-200 ms), BC sample-by-sample voit un signal `dx=0` mais la queue d'actions ramène la trajectoire. Diffusion Policy modélise nativement ces pauses comme partie de la séquence.
3. **Compression observationnelle** : on peut sortir 16 actions à partir d'une seule observation → 16× moins d'inférences = latence amortie.

### 3.2 Receding horizon (MPC-style)

À l'exécution, on ne joue PAS toute la séquence prédite. On joue les `T_a` premières (`T_a = 8` typiquement), puis on **replanifie** depuis la nouvelle observation. C'est exactement l'idée de Model Predictive Control :

```
T_o = 2  (observation horizon)  → on regarde 2 frames passées
T_p = 16 (prediction horizon)   → on prédit 16 actions
T_a = 8  (action horizon)        → on en exécute 8 puis on replannifie
```

Compromis :
- `T_a` grand → moins de latence cumulée, mais plus de drift si l'état dérive.
- `T_a` petit → réactif, mais on appelle souvent l'inférence (coûteux).

Le paper Chi 2023 montre que `T_a ∈ [4, 8]` est l'optimum sur PushT, kitchen, etc.

> **Key takeaway 3** : Diffusion Policy est un **planner court-horizon** déguisé en policy. La replanification fréquente (receding horizon) compense l'absence de feedback strict pendant `T_a` steps.

---

## 4. Pourquoi diffusion > BC sur multimodal — la démonstration de Chi 2023

Le paper compare sur 12 tâches (Push-T, Block-Push, Kitchen, ToolHang, Square, Lift, Can, Transport, Robomimic + benchmark IBC). Résultats clés :

| Tâche | LSTM-GMM (BC) | IBC (EBM) | BET | **Diffusion Policy** |
|-------|---------------|-----------|-----|----------------------|
| Push-T (state) | 0.65 | 0.52 | 0.79 | **0.95** |
| Push-T (image) | 0.69 | 0.06 | 0.78 | **0.91** |
| Tool Hang (image) | 0.39 | — | 0.58 | **0.85** |
| Robomimic Square | 0.73 | 0.04 | 0.62 | **0.92** |

**Le delta vient explicitement des tâches où la distribution d'actions est multimodale** (Push-T : deux stratégies de contournement, Block-Push : ordre des cubes libre, ToolHang : pince ouverte/fermée). Sur tâches unimodales, le gain est nul ou marginal — ce qui confirme la thèse : la valeur ajoutée n'est pas "j'ai un modèle plus gros", c'est "je gère la multimodalité".

### Ablations clés du paper (§6)

1. **Sans action chunking** (`T_p=1`) : success rate chute de 20-30 points → la cohérence temporelle compte.
2. **Sans EMA** : training oscillant, mauvaise convergence.
3. **DDIM 16 steps vs DDPM 100 steps** : <1% delta en success rate, 6× plus rapide en inférence.
4. **Transformer vs UNet** : Transformer meilleur sur gros datasets, UNet plus stable sur petits (<200 démos) → recommandation par défaut UNet.

---

## 5. Lecture du repo `real-stanford/diffusion_policy`

Repère ces fichiers à la prochaine lecture du repo (REFERENCES.md #19) :

```
diffusion_policy/
├── policy/
│   ├── diffusion_unet_image_policy.py     ← brique 1+2+3 assemblées
│   ├── diffusion_transformer_image_policy.py
│   └── base_image_policy.py               ← interface predict_action()
├── model/
│   ├── vision/
│   │   ├── model_getter.py                ← ResNet18 + GroupNorm replace
│   │   └── multi_image_obs_encoder.py
│   ├── diffusion/
│   │   ├── conditional_unet1d.py          ← UNet 1D + FiLM
│   │   ├── transformer_for_diffusion.py
│   │   ├── ema_model.py                   ← EMA wrapper
│   │   └── positional_embedding.py        ← sinusoidal timestep emb
│   └── common/
│       └── normalizer.py                  ← min/max normalisation actions [-1,1]
├── workspace/
│   └── train_diffusion_unet_image_workspace.py  ← training loop
└── config/
    └── task/                              ← YAMLs Hydra par tâche
```

Le repo utilise `diffusers` de HuggingFace pour le scheduler (`DDPMScheduler`, `DDIMScheduler`) — ne pas réinventer.

---

## 6. Pièges à connaître

1. **Normalisation des actions** : DDPM marche en `[-1, 1]`. Toujours min-max normaliser depuis le dataset, et dénormaliser en sortie.
2. **GroupNorm obligatoire** : BatchNorm pète avec EMA et batch size variable.
3. **EMA décroissance** : 0.75 typique en début (warmup), puis 0.9999 — sans ça l'entraînement diverge.
4. **Squared cosine schedule** : surtout pas de schedule linéaire pour les actions (les linéaires sont pour les images, pas les trajectoires courtes).
5. **Pretrain ResNet** : laisser pretrained ImageNet, GroupNorm, **fine-tune** (ne pas freeze entièrement — l'embedding visuel doit s'adapter au domaine robot).
6. **Inférence en temps réel** : DDIM 10-16 steps, fp16, batch 1 → ~30-50 ms sur RTX 3090. Compatible 10-20 Hz, pas 100 Hz.

---

## 7. Spaced repetition — Q&A

**Q1.** Pourquoi un Behavior Cloning naïf échoue sur des démonstrations humaines de "push-T" alors que sur un robot téléopéré scripté, il marche ?
> **R.** Les démos humaines ont une distribution d'actions **multimodale** (plusieurs stratégies pour le même état). MSE force la prédiction vers la moyenne, qui n'est aucune des stratégies valides. Un robot scripté est unimodal par construction → BC y suffit.

**Q2.** Que prédit le réseau `ε_θ` pendant le training ?
> **R.** Le **bruit gaussien** ε qu'on a ajouté à l'action propre, conditionné sur (action bruitée, timestep k, observation). Pas l'action propre directement.

**Q3.** Quelle est la différence entre `T_p` (prediction horizon) et `T_a` (action horizon) ?
> **R.** `T_p` = nombre d'actions prédites par appel au modèle (ex: 16). `T_a` = nombre d'actions effectivement exécutées avant de replanifier (ex: 8). On a toujours `T_a ≤ T_p`. C'est la replanification receding-horizon façon MPC.

**Q4.** Pourquoi DDIM en inférence et pas DDPM ?
> **R.** DDPM nécessite ~100 steps de débruitage → ~300 ms. DDIM (Song 2020) est un sampler déterministe qui obtient ~la même qualité en 10-16 steps → ~30 ms, compatible avec un contrôle 10-20 Hz.

**Q5.** Sur quels types de tâches Diffusion Policy n'apporte rien vs un BC bien tuné ?
> **R.** Tâches **unimodales** : trajectoires scriptées, démos d'un seul opérateur cohérent, environnements parfaitement déterministes. Le gain principal vient explicitement de la multimodalité (paper Chi 2023 §6).

---

## 8. Sources citées

- **REFERENCES.md #19** — Chi, Feng, Du, Burchfiel, Tedrake, Song. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 (best paper) / IJRR 2024. Site : <https://diffusion-policy.cs.columbia.edu/> · Repo : <https://github.com/real-stanford/diffusion_policy>
- **REFERENCES.md #23** — MIT 6.S184 (Holderrieth & Erives, IAP 2025), notes <https://diffusion.csail.mit.edu/docs/lecture-notes.pdf> — pour les fondations DDPM/score matching.

Pour aller plus loin (hors scope J16, mais lecture naturelle) : Ho 2020 (DDPM original), Song 2020 (DDIM), Florence 2021 (IBC, le concurrent direct battu par Diffusion Policy).
