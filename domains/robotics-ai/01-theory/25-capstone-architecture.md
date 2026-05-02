# J25 — Capstone Architecture : Vision encoder + UNet 1D + DDPM

> **Pareto** : Diffusion Policy = (image -> embedding visuel) + (state -> embedding) -> conditionnement d'un **denoiser** qui transforme du bruit gaussien en **sequence d'actions** via un schedule DDPM. Cette page construit le squelette bloc par bloc, sans encore l'entrainer (le training arrive J26).

---

## 1. Concret d'abord : que fait reellement la policy a l'inference ?

A chaque step du robot (PushT, picking, etc.) :

1. On prend les `T_obs` dernieres images (typiquement 2) et les `T_obs` derniers states (pose, velocite).
2. L'**encoder visuel** (ResNet18) sort un vecteur `f_img` de dimension 512 par image.
3. Le **state encoder** (petit MLP) sort un vecteur `f_state`.
4. On concatene -> `cond` (vecteur de conditionnement, ~1024 dims selon T_obs).
5. On part d'un tenseur `a_T` de bruit gaussien pur de shape `(B, T_act, action_dim)` avec par ex. `T_act=16`, `action_dim=2` pour PushT (x, y).
6. On itere `T = num_diffusion_steps` fois (100 pour DDPM, 10-50 pour DDIM) :
   - Le **denoiser** (UNet 1D) prend `(a_t, t, cond)` et predit le bruit `eps_theta`.
   - Le **scheduler** applique la formule reverse : `a_{t-1} = scheduler.step(eps_theta, a_t, t)`.
7. On sort `a_0`, sequence de `T_act` actions. On en execute `T_a` (par ex. 8) puis on replanifie : c'est le **receding horizon**.

> **Cle** : on ne predit **jamais** une action, on predit toujours une **sequence** debruitee. C'est ce **chunking** qui rend la policy capable de gerer la **multimodalite** (plusieurs trajectoires valides : passer a gauche ou a droite d'un obstacle, par exemple).

---

## 2. Vue bloc par bloc

```
        +-------------+         +-------------+
images  | ResNet18    |  f_img  |             |
------> | (frozen ou  |-------->|             |
        |  fine-tune) |         |             |
        +-------------+         |   cond      |
                                | (concat +   |
        +-------------+         |  optionnel  |
state   | MLP state   |  f_st   |  MLP)       |
------> | encoder     |-------->|             |
        +-------------+         +------+------+
                                       |
                                       v
   bruit a_t  (B, T_act, A)     +------+------+
   timestep t                   |  UNet 1D    |   eps_theta
   ------------------>          |  denoiser   |  ----------+
                                |  (FiLM cond)|            |
                                +-------------+            |
                                                           v
                                          +----------------+--+
                                          |  DDPM scheduler   |
                                          |  step (a_t -> a_{t-1})|
                                          +-------------------+
```

Quatre composants, pas un de plus :

| Bloc | Role | Choix dans ce capstone |
|------|------|------------------------|
| Vision encoder | image -> vecteur | ResNet18 torchvision (fallback CNN simple si torchvision absent), spatial softmax optionnel |
| State encoder | (pose, vel) -> vecteur | MLP 2 couches |
| Denoiser | predit le bruit a partir de (a_t, t, cond) | UNet 1D conditionne par FiLM |
| Scheduler | gere forward (entrainement) et reverse (echantillonnage) | DDPM cosine schedule |

---

## 3. Vision encoder : pourquoi ResNet18

Diffusion Policy (Chi et al. RSS 2023) utilise **ResNet18** pour 3 raisons :
- **Compromis** : suffisamment expressif (entraine sur ImageNet) sans exploser le compute (~11M params).
- **Spatial softmax** (option) : on remplace le global average pool par une carte de softmax 2D qui produit des coordonnees `(x, y)` apprises -> tres efficace pour la robotique (Levine et al.).
- **Fine-tuning leger** : on degele en general la derniere block, le reste reste frozen pour eviter l'overfit sur 200 demos.

Dans ce module, on supporte les deux : torchvision si dispo, sinon un mini-CNN qui produit le meme contrat (vecteur 512). Le code aval est agnostique.

---

## 4. State encoder et conditionnement

L'**etat** ce sont les `T_obs` dernieres poses + velocites du robot/objet. On le passe dans un MLP 2 couches qui sort un vecteur.

Le vecteur de **conditionnement final** = `concat([f_img_t-1, f_img_t, f_state_t-1, f_state_t])` puis on traverse une projection lineaire pour fixer la dimension globale `cond_dim` (par ex. 256).

Ce `cond` est diffuse dans toutes les couches du UNet via **FiLM** (Feature-wise Linear Modulation) : pour chaque couche, on apprend `(gamma, beta) = MLP(cond, t_embed)` puis `h = gamma * h + beta`. C'est ce qui evite d'inserer la condition uniquement a l'entree du UNet.

---

## 5. UNet 1D pour denoising sur sequences d'actions

Pourquoi 1D et pas 2D ?
- Les actions forment une serie temporelle `(T_act, action_dim)` avec `T_act` petit (8-16) et `action_dim` petit (2 pour PushT, 7 pour Franka). Pas d'image -> on convolue le long de l'axe **temps**.
- Convolutions 1D, downsample et upsample pour capter des dependances temporelles a plusieurs echelles.

Architecture type (Chi 2023) :

```
input  (B, A, T_act)               # transpose pour la conv 1D : channels = action_dim
  |  Conv1D
  v
[ResBlock1D + FiLM(cond, t)]  -> down1   (T_act/2, ch_d1)
[ResBlock1D + FiLM(cond, t)]  -> down2   (T_act/4, ch_d2)
[ResBlock1D + FiLM(cond, t)]   bottleneck
[ResBlock1D + FiLM(cond, t)]  -> up1   (concat avec down2)
[ResBlock1D + FiLM(cond, t)]  -> up2   (concat avec down1)
Conv1D final
output (B, A, T_act)               # le bruit predit eps_theta
```

Chaque **ResBlock1D** = `GroupNorm -> Mish -> Conv1D -> FiLM(cond, t_embed) -> GroupNorm -> Mish -> Conv1D + skip`.

**Embedding du timestep** `t` : sinusoidal positional embedding (comme dans les Transformers / DDPM) puis MLP, partage avec le `cond` dans la projection FiLM.

---

## 6. DDPM scheduler : forward et reverse

### 6.1 Forward (training)

On corrompt progressivement `a_0` (action propre du dataset) :

```
a_t = sqrt(alpha_bar_t) * a_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)
```

ou `alpha_bar_t = product_{s=1}^{t} (1 - beta_s)`. C'est `q_sample`. La cible du reseau est `eps`, et la loss est `MSE(eps_theta(a_t, t, cond), eps)`.

### 6.2 Reverse (sampling)

On part de `a_T ~ N(0, I)` puis pour `t = T..1` :

```
mu_theta = (1 / sqrt(alpha_t)) * (a_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta)
a_{t-1} = mu_theta + sigma_t * z,   z ~ N(0, I)  (z=0 si t=1)
```

C'est `p_sample`. Apres `T` etapes, on a une sequence d'actions debruitee.

### 6.3 Beta schedule cosinus (Nichol & Dhariwal 2021)

Plus stable que le lineaire, surtout pour les sequences d'actions courtes :

```
alpha_bar_t = cos((t/T + s) / (1 + s) * pi/2)^2,    s = 0.008
beta_t = clip(1 - alpha_bar_t / alpha_bar_{t-1}, 0, 0.999)
```

C'est l'option par defaut de Diffusion Policy.

---

## 7. Action chunking et receding horizon

| Hyperparam | Symbole | Valeur PushT |
|------------|---------|--------------|
| Action dim | `action_dim` | 2 (x, y de la cible TCP) |
| Action horizon | `T_act` | 16 |
| Action exec horizon | `T_a` | 8 |
| Obs horizon | `T_obs` | 2 |
| Diffusion steps train | `T_train` | 100 |
| Diffusion steps eval | `T_eval` | 100 (DDPM) ou 16 (DDIM) |
| Beta schedule | | cosinus |

**Receding horizon** : on predit `T_act = 16` actions, on en execute `T_a = 8`, puis on replanifie. Trade-off : grand `T_a` = inference moins frequente mais policy moins reactive ; petit `T_a` = reactive mais coute en compute.

---

## 8. Hyperparametres globaux (a documenter)

```python
ACTION_DIM = 2          # PushT (x, y)
ACTION_HORIZON = 16     # T_act
OBS_HORIZON = 2         # T_obs
NUM_DIFFUSION_STEPS = 100
BETA_SCHEDULE = "cosine"   # ou "linear" / "scaled_linear"
COND_DIM = 256
UNET_DOWN_CHANNELS = (256, 512, 1024)
KERNEL_SIZE = 5
N_GROUPS = 8            # GroupNorm
```

Ces valeurs sont alignees avec le repo `real-stanford/diffusion_policy` (config `diffusion_policy/config/policy/diffusion_policy_cnn.yaml`).

---

## 9. Pourquoi ce design plutot qu'un Transformer ?

Le repo Diffusion Policy propose les **deux** : `DiffusionUnetPolicy` et `DiffusionTransformerPolicy`. Empiriquement (papier Chi 2023, table 4) :

- **UNet 1D** : meilleur sur tache visuelle (PushT image-based), inference plus rapide, plus simple a debugger.
- **Transformer denoiser** : leger avantage sur taches state-based avec long horizon. Sera explore en exercice hard.

---

## 10. Ce qu'on N'a PAS encore

- Pas de boucle d'entrainement (J26).
- Pas d'EMA sur les poids (J26).
- Pas d'evaluation rollouts (J27).
- Pas de packaging / demo (J28).

L'objectif J25 est strict : **monter l'architecture, faire un forward dummy, verifier les shapes, documenter les hyperparametres**.

---

## Key takeaway

> [!tip] Diffusion Policy en une formule
> **`a_0 = denoise_T_steps(noise, cond=encode(image, state))`**
> ou `denoise` = UNet 1D conditionne par FiLM(cond, t), entraine a predire le bruit avec `MSE(eps_theta, eps_true)` et un schedule DDPM cosine.
> Le reste (action chunking, receding horizon, ResNet18) est ce qui transforme cette equation en policy robotique deployable.

---

## Mnemonique

**V-S-U-D** : **V**ision encoder, **S**tate encoder, **U**Net 1D denoiser, **D**DPM scheduler. Quatre blocs, c'est tout.

---

## Spaced repetition (a relire J+1, J+3, J+7)

1. **Q** : Pourquoi predire une **sequence** d'actions et non une seule ?
   **R** : Action chunking permet de modeliser des distributions multimodales d'actions et de reduire la frequence d'inference. Une policy qui predit une seule action a chaque step a tendance a moyenner les modes (jittering).

2. **Q** : Que predit exactement le UNet 1D ?
   **R** : Il predit le **bruit** `eps_theta(a_t, t, cond)`, pas l'action propre. C'est la parametrisation `epsilon-prediction` (Ho 2020).

3. **Q** : Quelle est la loss d'entrainement ?
   **R** : `MSE(eps_theta(a_t, t, cond), eps)` ou `a_t = sqrt(alpha_bar_t) * a_0 + sqrt(1 - alpha_bar_t) * eps`.

4. **Q** : Pourquoi un schedule **cosinus** plutot que lineaire ?
   **R** : Le lineaire detruit trop vite le signal en debut de schedule pour des sequences courtes. Le cosinus (Nichol & Dhariwal 2021) preserve plus d'information aux premiers timesteps -> meilleur denoising.

5. **Q** : C'est quoi FiLM et pourquoi on l'utilise ici ?
   **R** : Feature-wise Linear Modulation : `h = gamma * h + beta` avec `(gamma, beta) = MLP(cond, t_embed)`. Permet d'injecter le conditionnement (image+state+timestep) dans **chaque** couche du UNet, pas seulement a l'entree.

---

## Sources citees

- **REFERENCES.md #19** — Diffusion Policy (Chi et al., RSS 2023). Repo `real-stanford/diffusion_policy`, fichiers `diffusion_policy/model/diffusion/conditional_unet1d.py` et `diffusion_policy/policy/diffusion_unet_image_policy.py`.
- **REFERENCES.md #23** — MIT 6.S184 (Holderrieth & Erives 2025). Notes lecture 4-6 pour les fondations DDPM (forward/reverse SDE, score matching, beta schedule).
- Nichol & Dhariwal 2021, "Improved Denoising Diffusion Probabilistic Models" (arxiv 2102.09672) pour le cosinus schedule.
- Ho et al. 2020, "Denoising Diffusion Probabilistic Models" (arxiv 2006.11239) pour la parametrisation `epsilon`.
