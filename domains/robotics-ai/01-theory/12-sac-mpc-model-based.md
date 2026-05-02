# J12 — SAC, TD3, MPC, Model-Based RL

> Suite directe de J11 (PPO). On répond à : « PPO marche, mais comment faire mieux quand chaque step de simulation est précieux ? »

## 1. Pendule, deux philosophies

Imagine un pendule inversé. Tu dois le maintenir vertical en appliquant un couple `u` sur l'axe.

**Approche 1 — model-free (SAC)** :
- Tu lances 1000 épisodes random.
- Le réseau de politique apprend par essai-erreur : « si l'angle est X et la vitesse Y, le couple optimal est Z ».
- Aucune connaissance physique injectée. Le pendule est une boîte noire.
- Ça marche mais ça coûte ~100k-1M steps.

**Approche 2 — MPC (model-based, planning à chaque step)** :
- Tu connais (ou apprends) la dynamique : `θ̈ = (g/l)·sin(θ) + u/(m·l²)`.
- À chaque step `t`, tu simules dans ta tête 50 séquences d'actions sur 20 steps (`H=20`).
- Tu choisis la première action de la meilleure séquence. Tu réexécutes au step suivant.
- Si ton modèle est correct : ça marche en **0 step d'entraînement**.

**La tension du jour** :
- Model-free (SAC, TD3) = pas besoin de modèle, mais data-hungry.
- Model-based (MPC, MBPO, Dreamer) = sample-efficient, mais demande un modèle (analytique ou appris).
- En 2026, les frontière sont floues : SAC + world model appris = MBPO ; PPO + sim randomization = quasi-MPC implicite.

---

## 2. SAC — Soft Actor-Critic (Haarnoja et al., 2018)

PPO est on-policy : à chaque mise à jour, il jette les anciennes données. C'est gaspilleur.

SAC est **off-policy** + **maximum-entropy RL** : il garde un replay buffer (comme DQN) et il **maximise reward + entropie de la policy**.

### Objectif max-entropy

```
J(π) = Σ_t E[r(s_t, a_t) + α · H(π(·|s_t))]
```

- `H(π(·|s)) = -E_a[log π(a|s)]` : entropie de la distribution d'actions au state `s`.
- `α` : température. Plus `α` est grand, plus la policy est encouragée à explorer.

**Pourquoi c'est malin** :
1. **Exploration native** — pas besoin de bricoler ε-greedy ou OU noise (DDPG).
2. **Robustesse** — la policy ne se "lock" pas sur une mode. Si plusieurs actions sont quasi-équivalentes, elle garde la diversité.
3. **Stabilité** — l'entropie agit comme régularisation contre l'overfit du critic.

### Architecture

3 réseaux (en pratique 5 avec les targets) :
- **Actor** `π_φ(a|s)` : policy stochastique, en général Gaussienne `N(μ_φ(s), σ_φ(s))` squashée par `tanh` pour les bornes.
- **Critic 1** `Q_θ1(s,a)` et **Critic 2** `Q_θ2(s,a)` : double Q (héritage TD3) pour réduire le biais d'overestimation.
- **Targets** `Q_θ1'`, `Q_θ2'` : copies polyak-averaged (`θ' ← τ·θ + (1-τ)·θ'` avec τ≈0.005).

### Mises à jour (sketch)

À chaque update step (typique : 1 update par step env) :

1. Échantillonner batch `(s, a, r, s', done)` du replay buffer.
2. Cible Q :
   ```
   a' ~ π_φ(·|s')
   y = r + γ · (1-done) · [min(Q_θ1'(s',a'), Q_θ2'(s',a')) - α · log π_φ(a'|s')]
   ```
3. Loss critics : `L = MSE(Q_θi(s,a), y)` pour `i=1,2`.
4. Loss actor (reparam trick) :
   ```
   a_φ = μ_φ(s) + σ_φ(s) · ε,  ε ~ N(0,I)
   L_π = E[α · log π_φ(a_φ|s) - min(Q_θ1(s,a_φ), Q_θ2(s,a_φ))]
   ```
5. (Auto-tune `α`) loss `L_α = -α · (log π_φ(a|s) + H_target)` où `H_target = -dim(A)`.
6. Soft update des targets.

### SAC vs PPO — tableau qui résume tout

| Critère | PPO | SAC |
|---|---|---|
| On/off-policy | on | off (replay buffer) |
| Sample efficiency | faible (jette les rollouts) | élevée (3-10× moins de steps) |
| Stabilité | très haute | haute (mais sensible à `α`) |
| Hyperparam-friendly | oui (peu de tuning) | moyen (lr, τ, α) |
| Discret/continu | les deux | continu (extension SAC-discrete existe) |
| Coût compute par step | faible | élevé (plus de réseaux, plus d'updates/step) |
| MuJoCo benchmark | bon baseline | souvent meilleur en final reward |

**Règle pratique 2026** : sur MuJoCo continu, SAC bat PPO en sample efficiency. Sur tâches très bruyantes ou multi-env (vectorized), PPO reste plus simple à scaler. Cf. `[CS285 L13]`.

---

## 3. TD3 — Twin Delayed DDPG (Fujimoto et al., 2018)

TD3 est l'**ancêtre déterministe de SAC**. Trois ingrédients :

1. **Twin Q-networks** : deux critics `Q_θ1`, `Q_θ2`, on prend `min` pour la cible. Combat l'overestimation TD bias héritée de DDPG. SAC reprend ça.
2. **Delayed policy updates** : on met à jour l'actor toutes les `d` updates critics (typique `d=2`). Le critic doit converger avant que l'actor ne le suive.
3. **Target policy smoothing** : on ajoute un bruit `ε ~ clip(N(0,σ), -c, c)` à `a' = π_φ'(s') + ε` avant le calcul de la cible Q. Empêche l'actor de surexploiter des pics du critic.

**Différences clés vs SAC** :
- TD3 est **déterministe** (DDPG-style), donc bruit d'exploration externe (gaussien).
- TD3 n'a **pas** de bonus d'entropie. Plus de tuning manuel pour l'exploration.
- TD3 est souvent **plus rapide** à converger sur tâches déterministes simples (Pendulum, HalfCheetah).
- SAC gagne sur tâches multi-mode où l'entropie aide.

En 2026 : SAC est devenu le défaut. TD3 reste utile en baseline et quand la stochasticité de SAC nuit (par ex. tâches de précision). Cf. `[Spinning Up — SAC vs TD3]`.

---

## 4. MPC — Model Predictive Control

**Idée centrale** : à chaque step, je résous un problème d'optimisation court-terme sur un horizon `H` en utilisant un modèle de la dynamique, j'applique la **première** action, je re-planifie au step suivant (receding horizon).

```
À chaque t :
  Trouver U* = (u_t, u_{t+1}, ..., u_{t+H-1}) qui maximise
    Σ_{k=0}^{H-1} r(s_{t+k}, u_{t+k})  sous  s_{t+k+1} = f(s_{t+k}, u_{t+k})
  Appliquer u_t = U*[0]
  Observer s_{t+1}, recommencer.
```

**Pourquoi receding horizon ?**
- Ça absorbe les erreurs de modèle (au step `t+1` on re-planifie avec la vraie observation).
- Ça gère l'horizon infini sans calculer une politique globale.

### Méthodes pour résoudre l'optim

#### CEM — Cross-Entropy Method
- Échantillonner `N` séquences `U_i ~ N(μ, Σ)` (par défaut `μ=0`).
- Évaluer chaque rollout via le modèle, obtenir `R_i`.
- Garder les `K` meilleures (élites, `K << N`).
- Ré-estimer `μ, Σ` sur les élites. Itérer.
- Renvoyer `μ[0]` (la première action de la meilleure trajectoire moyenne).

Simple, parallélisable, dérivée-free. Marche très bien sur pendule.

#### MPPI — Model Predictive Path Integral (Williams 2017)
- Échantillonner `N` séquences perturbées `U_i = U_nom + δU_i`, `δU_i ~ N(0,Σ)`.
- Calculer rewards `R_i`.
- **Pondération par exponentielle** : `w_i = exp(R_i/λ) / Z`.
- Mise à jour : `U_nom ← Σ_i w_i · U_i`.
- Renvoyer `U_nom[0]`.

Plus lisse que CEM (toute la masse de probabilité contribue, pas juste les top-K). Standard en robotique pour quadrupèdes (Cheetah, Spot).

#### iLQR / DDP
Si le modèle est différentiable (rare en réel, courant en sim), on peut résoudre par programmation dynamique linéaire-quadratique itérative. Cf. `[Tedrake ch. 10]`.

### Forces / faiblesses MPC

| + | - |
|---|---|
| 0 entraînement si modèle connu | Coûteux à chaque step (planning online) |
| Contraintes faciles (couples max, etc.) | Sensible à l'erreur de modèle |
| Interprétable | Horizon court → myopie |
| Gère les changements (goal, env) | Pas d'apprentissage par feedback |

**Combinaison classique** : MPC pour le bas-niveau (suivi de trajectoire) + RL pour le haut-niveau (choix de goal). Voir aussi : Helix (Figure 2025) — System2 LLM + System1 80M policy temps-réel.

---

## 5. Model-Based RL — apprendre f(s,a) puis planifier

Si on n'a pas de modèle analytique, on l'apprend.

### Dyna (Sutton, 1990) — la matrice originelle

```
1. Boucle :
   - Step env réel, stocker (s, a, r, s')
   - Update Q via TD-learning (model-free part)
   - Apprendre modèle f̂(s,a) → s', r̂(s,a) → r
   - Pour k=1..n :
       - Sampler (s, a) du buffer
       - s', r = f̂(s,a), r̂(s,a)  (rollout imaginé)
       - Update Q via TD sur (s, a, r, s')
```

L'idée : 1 step réel produit `n+1` updates Q (1 réel + `n` imaginés). Sample efficiency × `n+1`.

### MBPO — Model-Based Policy Optimization (Janner 2019)

Combine **SAC** + **ensemble de modèles dynamiques** :
- Apprend `K` modèles (typique K=5-7) par bootstrapping. Réduit l'overconfidence.
- À chaque step, génère des rollouts courts (H=1-15) à partir de states du replay réel.
- SAC update sur le mix réel + imaginé.
- **Rollouts courts** parce que les erreurs de modèle s'accumulent. Pivot conceptuel : préfère `H=5` rollouts depuis 1000 vrais states que `H=200` depuis 1.

Résultat : on atteint le perf de SAC pur en **10-100× moins** de steps réels. Cf. `[CS285 L11-12]`.

### Dreamer V1→V3 (Hafner)

Voir J17 (deep dive). En une phrase : **world model latent (RSSM) + actor-critic dans l'imagination**. DreamerV3 (2023) marche avec une seule config sur 150+ tâches, du contrôle continu à Minecraft.

### Quand model-based, quand model-free ?

| Critère | Model-free (SAC, PPO) | Model-based (MBPO, Dreamer) |
|---|---|---|
| Sample efficiency | faible | haute |
| Asymptotic perf | souvent meilleure | parfois plafonne (model bias) |
| Compute par step | faible | élevé (apprend modèle + rollouts) |
| Bruit env | robuste | sensible (modèle apprend du bruit) |
| Hardware réel coûteux | déconseillé | indiqué |
| Sim cheap (MuJoCo) | OK | overkill souvent |

**Heuristique 2026** :
- Tu as MuJoCo et un GPU → SAC ou PPO.
- Tu as un robot réel à 100k€ et chaque rollout coûte 30s humain → Dreamer V3 ou MBPO.
- Tu as un modèle analytique fiable → MPC pur.

---

## 6. Récap mental

```
Continu, sample-efficient, sim → SAC (ou TD3 si déterministe)
Continu, on-policy, robuste → PPO (J11)
Modèle connu (analytique) → MPC (CEM/MPPI)
Modèle inconnu, peu de data réelle → MBPO ou Dreamer
LLM-style généralisation → World models (J17), VLA (J19+)
```

---

## Flashcards (spaced repetition)

1. **Q** : Pourquoi SAC ajoute un terme `α · H(π)` à l'objectif ?
   **R** : Pour encourager l'exploration native (pas de ε-greedy ni OU noise) et stabiliser l'apprentissage en évitant que la policy ne s'effondre sur une seule action.

2. **Q** : Quels sont les 3 ingrédients de TD3 vs DDPG ?
   **R** : (1) Twin Q-networks (`min` pour combat overestimation), (2) Delayed policy updates (actor mis à jour 1× toutes les `d=2` updates critic), (3) Target policy smoothing (bruit clipped sur `a'`).

3. **Q** : MPC = receding horizon. Que veut dire "receding" ?
   **R** : À chaque step `t`, on planifie sur `H` futurs steps, on applique la **première action** seulement, puis on re-planifie au step `t+1` avec les nouvelles observations. L'horizon "recule" avec le temps.

4. **Q** : Différence entre CEM et MPPI ?
   **R** : CEM garde les top-K élites pour ré-estimer μ,Σ (élitisme dur). MPPI pondère **toutes** les samples par `exp(R/λ)` (élitisme doux), donnant un update plus lisse.

5. **Q** : Pourquoi MBPO utilise des rollouts courts (H≈5) plutôt que longs (H≈200) ?
   **R** : Les erreurs de modèle s'accumulent exponentiellement avec l'horizon. Mieux vaut 200 rollouts courts depuis 200 states réels qu'un seul rollout long depuis 1 state.

6. **Q** : Quand préférer model-based à model-free ?
   **R** : Quand chaque step env coûte cher (robot réel, simu lente, hardware) ou quand on a un bon modèle analytique. Sinon, en sim cheap (MuJoCo + GPU), model-free reste plus simple et atteint souvent meilleur asymptotic.

---

**Sources** : `[Haarnoja et al., 2018, SAC]` (arxiv 1801.01290), `[Fujimoto et al., 2018, TD3]` (arxiv 1802.09477), `[Janner et al., 2019, MBPO]` (arxiv 1906.08253), `[CS285 L11-12]` (Levine — model-based RL), `[Spinning Up — SAC]`, `[Tedrake ch. 10 — MPC + trajopt]`.
