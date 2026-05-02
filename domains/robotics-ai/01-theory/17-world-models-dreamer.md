# J17 — World models : Dreamer V1 → V3

> **Pré-requis** : J9 (MDP, Bellman), J11 (PPO, advantage), J12 (SAC, MPC, model-based RL).
> **Objectif lecture (40-55 min)** : comprendre ce qu'est un *world model*, disséquer le RSSM (Recurrent State-Space Model), saisir pourquoi DreamerV3 (Hafner 2023) tourne avec **une seule config** sur 150+ tâches, et savoir quand ça bat le model-free.

---

## 1. L'image qui amorce : un agent qui rêve avant d'agir

Imagine que tu joues à Mario sans manette. Tu observes la scène, puis pendant 3 secondes tu **fermes les yeux** et tu *imagines* mentalement ce qui se passerait si tu sautais maintenant : la silhouette du Goomba, la trajectoire de Mario, le tuyau au bout. Tu *roule* mentalement plusieurs scénarios, tu choisis le meilleur, tu rouvres les yeux, tu agis.

Cette scène mentale — la simulation interne — c'est *exactement* ce que fait un agent **world-model** :

1. Il encode l'observation (pixels) dans un **état latent compact** `s_t`.
2. Il apprend une fonction de transition `p(s_{t+1} | s_t, a_t)` qui *prédit* l'état suivant.
3. Il déroule cette transition dans sa tête sur 15-20 pas (« imagination »).
4. Il entraîne sa policy `π(a|s)` à maximiser la reward **prédite par le modèle**, pas dans le vrai env.
5. Il exécute la policy dans l'env réel, collecte de nouvelles données, raffine le modèle.

Conséquence pratique : un agent world-model peut s'entraîner avec **10-100×** moins d'interactions réelles qu'un PPO ou un SAC. Crucial quand chaque interaction coûte cher (robot physique, simu lente, atari avec budget de frames).

> **Key takeaway #1** : un *world model* = modèle génératif appris de la dynamique de l'env (état latent + transition + reward). L'agent rêve dans ce modèle pour gagner en sample-efficiency.

---

## 2. Le contrat formel d'un world model

Plutôt que `Q(s,a)` ou `π(a|s)` directement, on apprend trois fonctions :

| Composant | Notation | Rôle |
|-----------|----------|------|
| **Encoder** | `e_t = enc(o_t)` | Compresse l'observation brute (image 64×64) en embedding |
| **Dynamique** | `p(s_{t+1} \| s_t, a_t)` | Prédit l'état latent suivant (récurrent + stochastique) |
| **Decoder** | `\hat{o}_t = dec(s_t)` | Reconstruit l'obs (signal d'apprentissage) |
| **Reward head** | `\hat{r}_t = R(s_t)` | Prédit la reward |
| **Continue head** | `\hat{c}_t = C(s_t)` | Prédit si l'épisode continue (γ effectif) |

Le tout est entraîné par **maximum likelihood** sur les trajectoires collectées (loss = reconstruction + reward + KL prior/posterior). C'est de l'apprentissage **non-supervisé** au sens où la reward n'est qu'un signal parmi d'autres : le gros du gradient vient de la reconstruction.

Pourquoi un *latent* ? Parce que travailler en pixels est :
- **lent** (un rollout d'imagination 15 pas × 64×64×3 = beaucoup de mémoire),
- **bruité** (texture, éclairage, parallaxe sont des distractions pour la décision),
- **redondant** (la position d'un agent tient en 4 floats, pas 12 288).

L'encoder agit comme un *bottleneck* qui force le modèle à garder l'information pertinente pour prédire le futur.

---

## 3. RSSM : le cœur de Dreamer

Le **Recurrent State-Space Model** (Hafner 2019, repris à l'identique en V2/V3) est l'architecture qui a *établi* la lignée Dreamer. Sa particularité : l'état latent `s_t` est **dual** :

```
s_t = (h_t, z_t)
       ^    ^
       |    └── partie stochastique (catégorielle 32x32 dans V3)
       └── partie déterministe (état GRU/LSTM)
```

Pourquoi ce dédoublement ?

- **`h_t` déterministe** : porte la *mémoire à long terme* (où je suis dans l'épisode, mon objectif, mon historique). Un GRU classique : `h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])`.
- **`z_t` stochastique** : capture l'incertitude / la stochasticité de l'env. Échantillonné depuis une distribution apprise.

Dans Dreamer, on entraîne **deux distributions** sur `z_t` :

1. **Prior** `p(z_t | h_t)` — distribution **prédite** par le modèle, conditionnée juste sur `h_t` (sans regarder l'obs réelle). C'est ce qu'on utilise en imagination.
2. **Posterior** `q(z_t | h_t, e_t)` — distribution **inférée** quand on a vu l'obs. C'est ce qu'on utilise en training (encoder le réel).

Le KL `KL(q || p)` est minimisé : on veut que la prédiction sans obs (prior) soit la plus proche possible de l'inférence avec obs (posterior). C'est littéralement *« apprends à imaginer ce que tu vois »*.

DreamerV3 introduit deux astuces clés sur RSSM :

- **`z_t` discret catégoriel 32 × 32** (32 variables catégorielles à 32 classes) au lieu de gaussien — empiriquement plus stable, **straight-through estimator** pour le gradient.
- **`symlog` transform** sur les rewards et reconstructions : `symlog(x) = sign(x) * log(|x| + 1)`. Stabilise sur tâches à reward range énorme (Atari) sans tuner.

> **Key takeaway #2** : RSSM = GRU déterministe (mémoire) + variable stochastique catégorielle (incertitude). KL prior↔posterior = colle qui force l'imagination à coïncider avec la perception.

---

## 4. Latent imagination : entraîner l'actor-critic dans le rêve

Une fois RSSM entraîné, l'astuce est radicale :

1. Tire un état réel `s_0` depuis le replay buffer.
2. **Déroule H=15 pas dans l'imagination** : à chaque pas, sample `a_t ~ π(s_t)`, prédit `s_{t+1} ~ prior(z|h)`, prédit `\hat{r}_t`, `\hat{c}_t`.
3. Sur cette trajectoire imaginée, entraîne :
   - **Critic** `V(s_t)` par TD(λ) sur les `\hat{r}` prédites.
   - **Actor** `π(a|s)` par maximisation de `V` (reparametrization trick + entropy bonus).

Tout le gradient passe **à travers** la dynamique du modèle (pas à travers l'env réel). On obtient typiquement **500 trajectoires imaginées par batch**, là où PPO en exigerait des milliers de réelles.

Schéma mental :

```
Réel:        o_0 ──► o_1 ──► o_2  (collecté, lent, cher)
              │       │       │
              ▼       ▼       ▼
RSSM train: s_0 ←─► s_1 ←─► s_2  (encode + reconstruct)
              │
              ▼ (H=15 imagined steps, fast, on GPU)
Imagine:    s_0 ──► s_1' ──► ... ──► s_15'
            actor + critic update on this dream
```

C'est ce qui permet à DreamerV3 de battre PPO/SAC en **wall-clock time** *et* en **sample efficiency** sur la plupart des benchmarks.

---

## 5. Pourquoi DreamerV3 = single-config qui marche partout

Référence : **Hafner et al., "Mastering Diverse Domains through World Models", 2023** (arxiv 2301.04104, REFERENCES.md #20).

Avant V3, chaque suite de tâches (DMC, Atari, Crafter) demandait son propre tuning. V3 expose **une config unique** qui obtient le top score humain sur 150+ tâches incluant Atari, ProcGen, DMC, Crafter, et — première mondiale — collecte de diamants en Minecraft *from scratch* sans curriculum.

Les 4 ingrédients qui rendent V3 robuste à l'échelle :

1. **Symlog predictions** sur reward et obs (cf. §3) — gère les ordres de grandeur arbitraires.
2. **Discrete latents** (32×32 catégoriel) — variance de gradient maîtrisée.
3. **Free bits + KL balancing** — empêche le posterior de collapser ou d'exploser.
4. **Critic regularization vers EMA target** — stabilise la valeur sans tuner γ ou la learning rate.

Et un *scaling law* propre : doubler la taille du modèle (12M → 200M params) double le sample efficiency sur Atari, sans changer les hyperparams. C'est le premier algo RL qui se comporte comme un LLM côté scaling.

> **Key takeaway #3** : DreamerV3 = RSSM + symlog + latents discrets + KL balancing. Une config, 150+ tâches, scaling propre. Premier RL « plug-and-play à l'échelle ».

---

## 6. Quand utiliser un world model — et quand pas

### Pour
- **Sample efficiency critique** : robot réel, simu lente, budget de frames serré.
- **Multi-tâches** : le RSSM apprend une représentation réutilisable (transfer entre tâches qui partagent la dynamique).
- **Exploration dirigée** : on peut planifier dans l'imagination (MPC sur le modèle, ex. PlaNet).
- **Long-horizon** : H=15 imagination steps découplés du wall-clock du simulateur.

### Contre
- **Coût compute** : entraîner RSSM + actor + critic = 3 réseaux à backprop. Sur tâches simples (CartPole), PPO single-file est plus rapide en wall-clock.
- **Model bias** : si la dynamique apprise diverge du réel, l'actor optimise un fantasme. DreamerV3 mitige par KL, mais sur env très stochastiques (cartes ouvertes, partial observability extrême), le modèle peut hallucine.
- **Implémentation lourde** : code RSSM ≈ 800-1500 LoC vs ~300 pour PPO CleanRL. Pour ce qui est de la lisibilité pédagogique, on fera une *toy version* (code de J17) — pour de la prod, repo officiel.

### Comparaison rapide (Source : Hafner 2023, Levine CS285 L11)

| Approche | Sample efficiency | Wall-clock | Robustesse env stochastique | Tuning |
|----------|-------------------|------------|------------------------------|--------|
| PPO (model-free on-policy) | basse | rapide | bonne | moyen |
| SAC (model-free off-policy) | moyenne | moyen | bonne | élevé |
| DreamerV3 (model-based latent) | **haute** | moyen | moyenne | **bas** (single config) |
| MPC + modèle appris | moyenne | lent (planning) | excellente | moyen |

---

## 7. Perspective : Dreamer vs JEPA vs Cosmos (préview J18)

Trois écoles de world models en 2025-2026, qu'on disséquera demain :

- **Dreamer (générative pixel-level)** — reconstruit l'image, KL prior↔posterior. Critique de LeCun : « gaspille du compute à reconstruire l'inutile ».
- **JEPA (V-JEPA 2, REFERENCES.md #21)** — prédiction dans l'**espace latent** seulement, pas de pixel decoder. Plus efficace, pas de génération.
- **NVIDIA Cosmos (REFERENCES.md #22)** — foundation models pré-entraînés sur 20M h vidéo, tokenizer vidéo réutilisable. World model à l'échelle d'un LLM.

Dreamer reste **le standard pédagogique** : c'est lui qu'on lit pour comprendre le pattern. Les autres innovent sur le « comment » pas le « quoi ».

---

## 8. Q&A spaced-repetition

**Q1** — Quelle est la différence entre prior et posterior dans RSSM, et pourquoi minimiser leur KL ?

> **R** — Prior `p(z|h)` prédit `z` sans regarder l'obs réelle (utilisé en imagination). Posterior `q(z|h,e)` infère `z` en voyant l'obs (utilisé en training). Minimiser `KL(q||p)` force la prédiction à converger vers l'inférence : « apprends à imaginer ce que tu observes ».

**Q2** — Pourquoi `s_t = (h_t, z_t)` plutôt qu'un seul vecteur ?

> **R** — `h_t` (déterministe, GRU) porte la mémoire long-terme. `z_t` (stochastique, échantillonné) capture l'incertitude/stochasticité de l'env. Séparer permet de propager une mémoire stable même si l'aléa de `z` change à chaque pas.

**Q3** — Cite trois choix techniques qui rendent DreamerV3 « single-config ».

> **R** — (1) Symlog sur rewards/obs (gère les échelles), (2) latents discrets catégoriels 32×32 + straight-through (variance contrôlée), (3) KL balancing + free bits (évite collapse posterior). Source : Hafner 2023.

**Q4** — Pourquoi entraîne-t-on l'actor-critic dans l'imagination plutôt que sur les vraies trajectoires ?

> **R** — Sample efficiency : 500 trajectoires imaginées (rapides, sur GPU) >> 500 trajectoires réelles (lentes, simu/robot). Le gradient passe par la dynamique du modèle, ce qui découple le compute du temps env.

**Q5** — Quand un model-free PPO est-il *préférable* à DreamerV3 ?

> **R** — Quand l'env est rapide et bon marché (CartPole, gridworld), quand l'env est très stochastique (Dreamer hallucine), quand on veut un code lisible/maintenable rapide à debugger. Sample efficiency n'est pas critique → la complexité de RSSM ne paie pas.

---

## Sources citées

- **REFERENCES.md #20** — Hafner et al. 2023, *Mastering Diverse Domains through World Models* (DreamerV3). Source principale §3-§6. https://arxiv.org/abs/2301.04104
- **REFERENCES.md #11** — Berkeley CS285 L11 (Levine, model-based RL). Cadre §2 (contrat formel) et §6 (model bias). https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps
- **REFERENCES.md #21** — V-JEPA 2 (préview §7). https://arxiv.org/abs/2506.09985
- **REFERENCES.md #22** — NVIDIA Cosmos (préview §7). https://arxiv.org/abs/2501.03575
