# J11 — Policy gradients & PPO sur MuJoCo

> **Acquis fin de jour** : entrainer PPO sur CartPole en moins d'une minute, lancer PPO sur HalfCheetah, comprendre pourquoi le clip ratio fait toute la difference entre TRPO et PPO.

---

## 1. Le mini-exemple qui motive tout — CartPole, "pousse a droite si penche a droite"

Avant n'importe quelle equation, fixons la scene. Dans `CartPole-v1` :

- L'observation est `s = (x, x_dot, theta, theta_dot)`. Le pole est en equilibre instable.
- L'action est binaire : `a = 0` pousser a gauche, `a = 1` pousser a droite.
- La recompense est `+1` par step tant que le pole tient (max 500 steps).

L'intuition humaine : **si le pole penche a droite (`theta > 0`), il faut pousser a droite pour glisser dessous**. Une policy logique serait `pi(a=1 | s) = sigmoid(w * theta)` avec `w > 0`.

**Question fondamentale** : comment apprendre ce `w` par gradient, sans connaitre le modele de physique ? On ne peut pas "regresser" parce qu'on n'a pas les bonnes actions — on a juste un signal scalaire `R` (somme des `+1`) a la fin de chaque trajectoire.

La reponse, en une phrase :
> *"Augmente la log-probabilite des actions qui ont ete suivies de bonnes recompenses, diminue celle des actions suivies de mauvaises."*

C'est la **policy gradient theorem** [Sutton & Barto, ch. 13]. Tout le reste — REINFORCE, A2C, PPO — n'est qu'une chasse a la variance et a la stabilite autour de cette idee.

---

## 2. Policy gradient theorem (sans douleur)

### 2.1 Le setup

On parametrise une policy stochastique `pi_theta(a | s)` par un reseau de neurones de parametres `theta`. On veut maximiser le retour espere :

```
J(theta) = E_{tau ~ pi_theta} [ R(tau) ]
```

ou `tau = (s_0, a_0, r_0, s_1, a_1, r_1, ...)` est une trajectoire et `R(tau) = sum_t gamma^t r_t`.

**Probleme** : `J` depend de la distribution des trajectoires, qui depend de `theta` qui depend de... bref, on ne peut pas deriver naivement.

### 2.2 L'astuce log-derivative

Le truc magique (identite triviale en proba) :

```
nabla_theta log p_theta(x) = nabla_theta p_theta(x) / p_theta(x)
```

Applique a la distribution des trajectoires `p_theta(tau) = p(s_0) * prod_t pi_theta(a_t|s_t) * P(s_{t+1}|s_t, a_t)`, on obtient apres un peu d'algebre (les termes de transition `P` ne dependent pas de `theta` et disparaissent) :

```
nabla_theta J(theta) = E_{tau} [ sum_t nabla_theta log pi_theta(a_t | s_t) * R(tau) ]
```

**Lecture** : on evalue le gradient de la log-policy a chaque `(s_t, a_t)` pris dans une trajectoire, on le multiplie par le retour total `R(tau)`, on moyenne. C'est tout.

### 2.3 REINFORCE — vanilla policy gradient

L'algo en 5 lignes ([Williams, 1992]; [Sutton & Barto, ch. 13.3]) :

```
1. Roll out N trajectoires avec pi_theta
2. Pour chaque (s_t, a_t) dans chaque tau, calcul de R_t = sum_{k>=t} gamma^{k-t} r_k  (return-to-go)
3. loss = - mean( log pi_theta(a_t | s_t) * R_t )
4. theta <- theta - lr * nabla loss
5. Repeter
```

> **Pourquoi `R_t` (return-to-go) plutot que `R(tau)` complet ?**
> Le passe ne depend pas des actions futures — utiliser `R(tau)` ajoute du bruit gratuit. Le return-to-go est mathematiquement equivalent en esperance et reduit la variance. C'est le **premier vinaigre anti-variance**.

REINFORCE *fonctionne* sur CartPole en environ 200-500 episodes. Mais sur MuJoCo (HalfCheetah, espace d'action continu), la variance explose et la convergence est tres lente. D'ou les ameliorations qui suivent.

---

## 3. Variance reduction — baseline, A2C, GAE

### 3.1 Baseline : soustraire ce qu'on attend

**Constat** : `R_t` est tres bruite. Si on soustrait *n'importe quelle fonction `b(s_t)` qui ne depend pas de `a_t`*, l'esperance du gradient ne change pas (preuve : le terme additionnel s'annule en moyenne sur les actions). Mais la *variance* peut chuter drastiquement.

Le choix canonique : `b(s) = V(s)`, la value function de la policy courante. On apprend `V_phi(s)` en parallele par regression MSE sur les `R_t` observes.

### 3.2 Advantage et A2C

L'**advantage** est `A_t = R_t - V(s_t)` : "combien l'action choisie a-t-elle fait mieux que la moyenne attendue dans cet etat". Plug-in dans REINFORCE :

```
loss_pi = - mean( log pi_theta(a_t | s_t) * A_t )
loss_v  = mean( (V_phi(s_t) - R_t)^2 )
```

C'est **A2C** (Advantage Actor-Critic) [Mnih et al., 2016, A3C async version]. Le "actor" est `pi_theta`, le "critic" est `V_phi`.

### 3.3 GAE — l'estimateur de variance optimal

Le return-to-go `R_t` est *unbiased mais haute variance* (somme de tous les rewards futurs). Le **TD error** `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)` est *low variance mais biaise* (depend de la qualite de V).

**Generalized Advantage Estimation (GAE)** [Schulman et al., 2016, GAE] interpole les deux par une moyenne exponentielle pondere par `lambda in [0,1]` :

```
A_t^GAE = sum_{l=0}^{infty} (gamma * lambda)^l * delta_{t+l}
```

- `lambda = 0` : `A_t = delta_t` (1-step TD, faible variance, biais eleve)
- `lambda = 1` : `A_t = R_t - V(s_t)` (Monte Carlo, faible biais, haute variance)
- En pratique : `lambda = 0.95`, `gamma = 0.99` — sweet spot etabli empiriquement.

GAE est le **deuxieme vinaigre** : variance ↓ sans tuer le signal.

---

## 4. Le probleme qui reste — gros pas = boom

Meme avec GAE et baseline, vanilla policy gradient (VPG) souffre d'un probleme structurel : **un seul gros gradient peut detruire la policy**. Si l'update fait basculer `pi_theta` loin de la policy qui a genere les donnees, les advantages estimes deviennent denues de sens, et la prochaine iteration s'effondre.

**Diagnostic** : on fait du *on-policy learning*. Les donnees `(s_t, a_t, A_t)` ne sont valides que pour la policy *qui les a produites*. Des qu'on bouge trop, on est hors-distribution.

**Solution naive** : faire un petit pas (`lr` faible). Inefficace : on jette des donnees apres une seule update.

**Solution TRPO** [Schulman et al., 2015] : contrainte explicite de divergence KL entre nouvelle et ancienne policy, resolue par natural gradient + line search. Mathematiquement propre, mais **ingenierie penible** (Fisher matrix, conjugate gradient, line search).

---

## 5. PPO — la solution pragmatique

[Schulman et al., 2017, PPO] propose une approximation **redoutablement simple** de l'idee TRPO. Le coeur : le **clip objective**.

### 5.1 Le ratio de probabilite

On note `r_t(theta) = pi_theta(a_t|s_t) / pi_{theta_old}(a_t|s_t)`. Si `theta = theta_old`, alors `r_t = 1`. Plus on s'eloigne, plus `r_t` s'ecarte de 1.

L'objectif "surrogate" non-clippe (qui generalise REINFORCE en off-policy via importance sampling) est :

```
L^CPI(theta) = E_t [ r_t(theta) * A_t ]
```

Probleme : si `A_t > 0` et `r_t` explose vers le haut, l'optimiseur va pousser `r_t` aussi loin que possible — meme si on quitte le voisinage de la policy d'origine.

### 5.2 Le clip

PPO clipe `r_t` dans `[1 - epsilon, 1 + epsilon]` (typiquement `epsilon = 0.2`) :

```
L^CLIP(theta) = E_t [ min( r_t(theta) * A_t,  clip(r_t(theta), 1 - eps, 1 + eps) * A_t ) ]
```

Lecture intuitive — quatre cas :

| `A_t` | `r_t` | objectif | gradient |
|---|---|---|---|
| `> 0` | `< 1+eps` | `r_t * A_t` | pousse `r_t` vers le haut |
| `> 0` | `>= 1+eps` | `(1+eps) * A_t` (constante en `theta`) | **gradient = 0**, on s'arrete |
| `< 0` | `> 1-eps` | `r_t * A_t` | pousse `r_t` vers le bas |
| `< 0` | `<= 1-eps` | `(1-eps) * A_t` (constante) | **gradient = 0** |

> **Idee centrale** : *des qu'on s'est trop eloigne de l'ancienne policy dans la bonne direction, on coupe le gradient*. Dans la mauvaise direction, le min impose qu'on prenne le pire des deux — ca empeche d'amplifier les erreurs.

C'est pas une borne KL stricte (TRPO l'est), c'est un mecanisme **pessimiste** qui rend l'optimiseur incapable de trop bouger en une update. En pratique, ca permet de faire **plusieurs epochs (4-10) sur le meme batch** sans tout casser.

### 5.3 La loss totale

```
L^PPO = L^CLIP - c1 * L^VF + c2 * S[pi_theta]
```

ou :
- `L^VF = (V_phi(s_t) - R_t)^2` : value loss, MSE sur les returns.
- `S[pi]` : entropie de la policy, pour encourager l'exploration. Coefficient typique `c2 = 0.01`.
- `c1 ~ 0.5`.

### 5.4 Pseudocode complet

```
Init pi_theta, V_phi
for iteration in 1..N:
    # Rollout
    Collect T steps avec pi_theta_old (theta_old <- theta)
    Compute advantages A_t via GAE
    Compute returns R_t = A_t + V(s_t)

    # K epochs de SGD sur le meme batch
    for epoch in 1..K:
        for minibatch:
            r_t = pi_theta(a|s) / pi_theta_old(a|s)
            L_clip = mean( min(r_t * A, clip(r_t, 1-eps, 1+eps) * A) )
            L_vf   = mean( (V_phi(s) - R)^2 )
            S_ent  = mean( entropy(pi_theta(.|s)) )
            loss = -L_clip + 0.5 * L_vf - 0.01 * S_ent
            optimizer.step()
```

C'est l'algo central de [Spinning Up, PPO]. Le repo de reference single-file pour MuJoCo : [CleanRL `ppo_continuous_action.py`].

---

## 6. PPO sur MuJoCo — config standard

Pour HalfCheetah-v4 / Ant-v4 / Walker2d-v4, les hyperparametres "qui marchent toujours" (issus du paper original + tuning communautaire) :

| Hyperparametre | Valeur typique |
|---|---|
| `lr` (Adam) | `3e-4` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_eps` | `0.2` |
| `n_envs` parallele | 1 a 8 |
| `n_steps` rollout | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `total_timesteps` | 1M (CartPole) a 10M (HalfCheetah) |
| Network | MLP `[64, 64]` ou `[256, 256]` tanh |
| Action distrib | `Normal(mu, sigma)` avec `sigma` apprise (state-independent) |

> **Astuce continuous action** : la policy sort `mu` ; `log_sigma` est un parametre libre du reseau (pas conditionne sur `s`). Tres simple, tres efficace ([CleanRL] confirme).

Sur HalfCheetah-v4, PPO atteint typiquement 3000-6000 de retour en 1M-3M steps avec ces hyperparametres (CPU 8 cores : ~30-60 min).

---

## 7. Stable-Baselines3 vs CleanRL — quand utiliser quoi

| Critere | Stable-Baselines3 | CleanRL |
|---|---|---|
| **API** | `model = PPO("MlpPolicy", env); model.learn(1e6)` | un seul fichier `ppo_continuous_action.py`, ~400 lignes |
| **Force** | rapide a brancher, callbacks, save/load, vec envs gerees | tout est lisible ligne par ligne, instrumente Wandb |
| **Faiblesse** | "boite noire" pedagogiquement, debug penible | pas d'abstraction, copie-colle pour adapter |
| **Quand l'utiliser** | baseline production, comparer 5 algos vite | **comprendre/modifier** PPO (recherche, ablation) |

**Heuristique** : SB3 quand on veut un resultat ; CleanRL quand on veut savoir *pourquoi* ca marche. Pour un capstone d'apprentissage, **CleanRL d'abord**.

---

## 8. Pourquoi PPO a-t-il "gagne" l'industrie ?

- **Robuste** au choix d'hyperparametres (le clip impose une regularisation implicite).
- **Simple a implementer** : pas de Fisher matrix, pas de line search.
- **On-policy compatible avec les workflows actuels** : facile a paralleliser sur CPU (vec envs).
- **Universel** : action discrete (Categorical), continue (Normal), dictionnaire (mixte) — meme code, juste la distribution change.
- **Adopte par OpenAI, DeepMind (en partie), tous les labs robotique** comme baseline RL.

C'est aussi pour ca que **PPO est le backbone de RLHF** sur les LLMs : meme algo, autre application.

---

## 9. Limites et suite logique

PPO reste **on-policy** : il jette les rollouts apres K epochs. **Sample efficiency** mediocre comparee a SAC/TD3 (off-policy, replay buffer). Sur un bras reel ou un humanoid, on prefere SAC. Ca, c'est J12.

Et PPO + LLM = RLHF, qui a son propre folklore (KL penalty, reward model). Ca, c'est ailleurs.

---

## 10. Flash-cards (5 Q/R)

1. **Q** : Qu'est-ce que le policy gradient theorem dit en une phrase ?
   **R** : Le gradient du retour espere est `E[ nabla log pi(a|s) * R ]` — augmente la log-prob des actions qui rapportent.

2. **Q** : Pourquoi soustraire une baseline `V(s)` ne change pas le gradient en esperance ?
   **R** : Parce que `E_a[ nabla log pi(a|s) * b(s) ] = b(s) * nabla 1 = 0`. Mais ca reduit la variance du sample.

3. **Q** : Quelles sont les deux extremes de GAE ?
   **R** : `lambda = 0` -> 1-step TD (low var, high bias) ; `lambda = 1` -> Monte Carlo (high var, low bias).

4. **Q** : Que clipe PPO et pourquoi ?
   **R** : Le ratio `r_t = pi_new / pi_old` dans `[1-eps, 1+eps]`. Empeche un seul gros pas de detruire la policy + permet plusieurs epochs sur le meme batch.

5. **Q** : SB3 ou CleanRL pour comprendre PPO ?
   **R** : CleanRL — un seul fichier, lisible ligne par ligne. SB3 est un workhorse mais opaque pedagogiquement.

---

## Sources

- [Schulman et al., 2017, PPO] — *Proximal Policy Optimization Algorithms*. arxiv 1707.06347.
- [OpenAI Spinning Up] — Pages VPG et PPO. https://spinningup.openai.com/
- [CleanRL] — `ppo_continuous_action.py`. https://github.com/vwxyzjn/cleanrl
- [Sutton & Barto, ch. 13] — REINFORCE, baseline, actor-critic.
- [Schulman et al., 2016, GAE] — *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arxiv 1506.02438.
