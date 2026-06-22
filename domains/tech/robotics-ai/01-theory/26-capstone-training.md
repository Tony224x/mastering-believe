# J26 — Capstone : Training loop Diffusion Policy

> Jour 3/5 du capstone. On a un dataset de démos (J24) et une architecture (J25). Aujourd'hui on entraîne — proprement, avec les recettes qui font la différence entre "ça converge" et "ça plafonne à 30% de succès".

---

## 1. L'exemple concret avant la théorie

Imaginons : on a 200 démonstrations de la tâche PushT. Chaque démo dure ~80 steps. Une "trajectoire d'entraînement" c'est une fenêtre de 16 actions consécutives (action chunk). On a donc ~12 800 chunks dans le dataset.

Une étape d'entraînement, c'est :

```
1. Sample un batch de 64 (state, action_chunk) du dataset
2. Tire un t aléatoire dans [1, T=100]  (T = nombre de pas de diffusion)
3. Calcule alpha_t (schedule cosine ou linear)
4. Bruite les actions : a_noisy = sqrt(alpha_t) * a + sqrt(1-alpha_t) * eps   avec eps ~ N(0, I)
5. Le modèle predit : eps_hat = model(a_noisy, t, state_embedding)
6. Loss = MSE(eps_hat, eps)        # <-- la subtilité du papier Chi 2023
7. Backward, AdamW step, EMA update, log
```

À retenir : **on n'apprend pas à prédire l'action, on apprend à prédire le bruit qui a corrompu l'action**. C'est l'astuce DDPM (Ho 2020) reprise par Diffusion Policy (Chi et al., 2023, REFERENCES.md #19).

---

## 2. Pourquoi la loss est MSE sur le bruit, pas sur l'action

Naïvement on serait tenté de faire `loss = MSE(action_predite, action_vraie)`. Erreur. Voici pourquoi :

- **Multimodalité** : sur PushT, pousser le T par la gauche ou par la droite peut être également bon. Une regression directe moyennerait les deux modes → action moyenne nulle, échec total. C'est exactement le problème que Diffusion Policy résout vs BC.
- **Gradient signal** : prédire le bruit donne un gradient bien conditionné à tous les niveaux de t (le bruit a toujours variance unitaire). Prédire l'action directement donne des gradients qui s'effondrent quand t est petit (action ~= action vraie, signal trivial) ou explosent quand t est grand (action ~= bruit pur).
- **Équivalence formelle** : prédire `eps` ou prédire `x_0` ou prédire la `score function` sont équivalents (reparamétrisation), mais `eps`-prediction donne en pratique la meilleure stabilité numérique (Ho 2020, eq. 12).

> **Key takeaway** > **`loss = MSE(model(x_t, t, cond), eps)`** où `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`. Une seule ligne, mais tout le succès de DDPM/Diffusion Policy tient dedans.

---

## 3. EMA — l'arme secrète qui double presque les performances

EMA = **Exponential Moving Average** des poids. À chaque step :

```
ema_param = decay * ema_param + (1 - decay) * model_param
```

avec `decay = 0.9999` typiquement. Au moment de l'inférence/eval, on utilise les poids EMA, pas les poids "live".

Pourquoi ça marche :
- **Lissage temporel** des poids → réduit la variance entre checkpoints proches
- **Effet ensemble implicite** : EMA = moyenne pondérée des derniers ~10k états du modèle
- **Empirique** : Diffusion Policy paper §A.4 observe **+10 à +20 points de success rate** avec EMA. Coût : 0 (juste un buffer de poids dupliqué).

Subtilité d'implémentation : ne PAS appliquer EMA pendant les premiers ~500 steps (warmup), sinon EMA traîne sur des poids initiaux random. Implémentations sérieuses utilisent un `decay` qui monte progressivement de 0 à 0.9999.

---

## 4. Optimizer + LR schedule

### AdamW, pas Adam
On utilise **AdamW** (decoupled weight decay) avec `weight_decay=1e-6` typiquement. La décorrelation du weight decay et du gradient évite la pénalisation excessive observée avec Adam vanilla. Diffusion Policy paper utilise `lr=1e-4`, `betas=(0.95, 0.999)`.

### Schedule cosine avec warmup linéaire
```
warmup_steps = 500
total_steps = 100_000

lr(t) = lr_max * t / warmup_steps                                        si t < warmup
lr(t) = lr_max * 0.5 * (1 + cos(pi * (t - warmup) / (total - warmup)))    sinon
```

Pourquoi :
- **Warmup linéaire** : évite l'explosion de gradients en early training, surtout avec EMA (les poids ne sont pas encore stabilisés).
- **Cosine decay** : suit la "loss landscape geometry" intuition de Loshchilov, plus doux qu'un step-wise. Empiriquement supérieur à exponential decay sur diffusion.
- **Pas de restart** : on entraîne typiquement 100k-200k steps, un seul cycle suffit.

---

## 5. Batch size et gradient accumulation

Diffusion Policy paper utilise **batch=256** sur GPU A100 (40GB). Sur GPU consumer (RTX 3060 12GB) on est limité à ~32 ou 64 selon la taille de l'image.

Solution : **gradient accumulation**.

```python
optimizer.zero_grad()
for micro_batch in range(accum_steps):
    loss = compute_loss(batch[micro_batch]) / accum_steps   # important !
    loss.backward()
optimizer.step()
ema.update()
```

Ça permet de simuler un batch de 256 avec un micro-batch de 32 et `accum_steps=8`. Coût : 8× le forward/backward par step "logique", mais mémoire constante.

Note : **diviser la loss par accum_steps** sinon le gradient est mécaniquement 8× trop grand → AdamW compense partiellement mais c'est une mauvaise hygiène.

---

## 6. Mixed precision (fp16 / bf16) sur GPU consumer

Sur GPU consumer (sans tensor cores serious), le gain en vitesse est de ~1.5-2× et la mémoire descend de ~40%. Avec `torch.amp` :

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for batch in loader:
    optimizer.zero_grad()
    with autocast(device_type="cuda", dtype=torch.float16):
        loss = compute_loss(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Pièges :
- **bf16 > fp16** si dispo (Ampere+) : range numérique de fp32, pas besoin de GradScaler.
- **EMA toujours en fp32** : les poids EMA sont stockés en fp32 séparément, sinon ils dérivent.
- **Loss en fp32** : `loss.float()` avant `.backward()` si tu vois des NaN.

---

## 7. Logging : wandb + reconstructions visuelles

À chaque step on logge `loss`, `lr`, `grad_norm`. Toutes les ~1000 steps on logge :
- **Sample reconstruction** : on prend une vraie observation du val set, on échantillonne une action via le modèle EMA, on plotte côte-à-côte (ground truth vs predicted) → permet de détecter early "mode collapse" ou divergence.
- **Histogramme du bruit prédit** vs bruit vrai → si distribution de eps_hat dérive, on a un bug.

Wandb (ou TensorBoard) n'est pas optionnel sur ce genre de run : 100k steps × 5h, sans logging continu tu rates les early warnings.

---

## 8. Diagnostic de courbes de loss

Loss MSE typique sur Diffusion Policy PushT :
- Step 0 : ~1.0 (eps ~ N(0,I), prediction random)
- Step 5k : ~0.2 (warmup terminé, modèle apprend la structure)
- Step 50k : ~0.05 (plateau de qualité)
- Step 100k : ~0.04 (convergence asymptotique)

Signaux d'alarme :
- **Loss qui remonte** : LR trop haut, ou bug de schedule (warmup mal calibré).
- **Loss plateau à 0.5+** : le modèle n'apprend rien — vérifier le bruitage (forward process), la cond visuelle (encoder gelé ?), la forme des tenseurs.
- **Loss qui descend mais success rate plafonne** : c'est le cas le plus pernicieux — la loss MSE ne corrèle pas parfaitement avec le success rate (proxy, pas objectif). C'est pour ça qu'on évalue rollouts régulièrement (sujet J27).

---

## 9. Recette finale (cheatsheet)

```python
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6, betas=(0.95, 0.999))
scheduler = CosineLRScheduler(warmup_steps=500, total_steps=100_000)
ema = EMA(model, decay=0.9999, warmup=500)
scaler = GradScaler()  # optionnel, fp16

for step, batch in enumerate(loader):
    optimizer.zero_grad()
    with autocast(...):  # optionnel
        x_0, cond = batch
        t = randint(1, T, (B,))
        eps = randn_like(x_0)
        x_t = q_sample(x_0, t, eps)             # forward noising
        eps_hat = model(x_t, t, cond)
        loss = F.mse_loss(eps_hat, eps)         # <-- LE coeur

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    ema.update(model)

    if step % 1000 == 0:
        log({"loss": loss, "lr": scheduler.get_lr(), "grad_norm": grad_norm})
        save_checkpoint(model, ema, optimizer, step)
```

---

## 10. Q&A spaced-repetition

**Q1** : Pourquoi prédit-on `eps` et pas `x_0` directement ?
**R** : Parce que `eps ~ N(0, I)` à variance constante donne un signal de gradient bien conditionné quel que soit le pas de bruit `t`. Prédire `x_0` donne un signal qui s'effondre à petit `t` (cible ~= input). Les deux sont mathématiquement équivalents par reparamétrisation, mais empiriquement `eps`-prediction est plus stable (Ho 2020).

**Q2** : Que fait l'EMA et pourquoi le decay typique est 0.9999 ?
**R** : EMA maintient une moyenne mobile exponentielle des poids. À l'inférence on utilise les poids EMA, qui sont plus stables (variance temporelle réduite, effet ensemble implicite). Decay 0.9999 → fenêtre effective ~10 000 steps, ce qui matche la durée typique d'un pseudo-cycle de convergence.

**Q3** : Pourquoi cosine + warmup et pas LR constant ?
**R** : Warmup linéaire évite gradient blow-up early (EMA pas encore stable, init naïve). Cosine decay donne une décroissance douce qui lisse la fin de training (loss landscape de plus en plus plate near optimum). Combo empiriquement optimal sur diffusion (cf. ADM, Imagen, Diffusion Policy paper).

**Q4** : Comment simuler un batch de 256 sur un GPU 12GB qui ne tient que 32 ?
**R** : Gradient accumulation : faire 8 forwards/backwards consécutifs sans `optimizer.step()`, en divisant la loss par 8 pour conserver l'échelle de gradient. Mémoire constante, throughput légèrement réduit, mais batch effectif identique.

**Q5** : Pourquoi la loss MSE peut descendre alors que le success rate plafonne ?
**R** : Parce que MSE est un proxy de qualité, pas l'objectif final. Une légère erreur sur `eps_hat` peut s'amplifier au cours des T pas de denoising (T=100 typique) et donner une action qui rate le but. C'est pour ça qu'on évalue par rollouts (J27), pas seulement via la loss.

---

## Sources

- **#19 Diffusion Policy** (Chi et al., 2023) — config training officielle, EMA, AdamW, hyperparams. Source principale du jour. https://diffusion-policy.cs.columbia.edu/
- **#23 MIT 6.S184** (Holderrieth & Erives, 2025) — fondations math de la loss DDPM, équivalence eps/x_0/score. https://diffusion.csail.mit.edu/2025/index.html
