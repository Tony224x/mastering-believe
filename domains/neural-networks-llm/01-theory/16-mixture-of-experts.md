# Jour 16 — Mixture of Experts : decoupler parametres et compute

> **Temps estime** : 5h | **Prerequis** : J1-J9 (transformers, FFN, MLP)

---

## 1. Pourquoi MoE existe — le probleme du FFN dense

Dans un transformer dense classique (GPT, Llama dense), **environ 70% des parametres vivent dans les FFN** (les couches feed-forward apres chaque attention). Et a chaque token, **tous ces parametres sont actives**. C'est l'origine du probleme :

```
Modele Llama 3 70B dense :
  - 70 milliards de params au total
  - ~50 milliards dans les FFN
  - Pour produire 1 token : 70 milliards de params actifs
  - Cout/token = ~140 GFLOPs (2 * params)
```

Or, **un token donne (le mot "stack" dans un contexte de programmation) n'a pas besoin de toute la connaissance du modele**. Il a besoin de la connaissance "code/programmation". Un autre token ("Verdun") a besoin de la connaissance "histoire/geographie". On gaspille du compute.

**L'idee MoE** (Shazeer 2017, popularise par GShard, Switch Transformer, Mixtral) : remplacer le FFN dense par **N experts FFN specialises**, et un **routeur** qui choisit, pour chaque token, **k experts parmi N** (typiquement k=1 ou 2).

Resultat :
- Total params **eleve** (somme des experts) → grande capacite de stockage
- Params actifs par token **bas** (k experts seulement) → compute faible
- **On decouple capacite memoire et compute par token**

C'est *la* raison d'etre de MoE.

---

## 2. L'architecture de base

```
                Input token (vecteur d_model)
                          │
                          ▼
                    [LayerNorm]
                          │
         ┌────────────────┼────────────────┐
         │                                  │
         ▼                                  ▼
    [Attention]                      [Routeur (gating)]
         │                                  │
         │                       softmax sur N experts
         │                                  │
         │                       top-k selection (k=2)
         │                                  │
         │                  ┌──────┬───────┴───────┬──────┐
         │                  ▼      ▼               ▼      ▼
         │            [Expert1][Expert2]    ...[ExpertN]  (FFN paralleles)
         │                  │      │               │
         │                  └──────┴───────┬───────┘
         │                       weighted sum (par gate)
         ▼                                  │
       (residual) ◄────────────────────────┘
```

### Le routeur (gating network)

Une simple matrice `W_g` de taille `(d_model, N)` :

```
logits = x @ W_g            # (batch, N)
weights = softmax(logits)   # probabilites par expert
top_k_indices = topk(weights, k=2)
top_k_weights = renormalize(weights[top_k_indices])
```

Pour chaque token, on garde les **k plus grands** poids et on les renormalise pour qu'ils somment a 1. Le token est ensuite envoye **uniquement** vers ces k experts.

### Les experts

Chaque expert est un FFN classique :

```
expert_i(x) = down_proj_i(SwiGLU(up_proj_i(x)))
```

avec ses propres matrices `up_proj_i` et `down_proj_i`. Memes formes que le FFN dense, mais **multiplie par N**.

### Le forward final

```
output = sum_{i in top_k}  weights[i] * expert_i(x)
```

C'est une combinaison lineaire ponderee des sorties des k experts choisis.

---

## 3. Mixtral 8x7B en detail

Mixtral 8x7B (Mistral, decembre 2023) a popularise MoE en open-source. **Comprendre ses chiffres = comprendre MoE.**

```
N = 8 experts par layer
k = 2 (top-2 routing)
32 layers
d_model = 4096
FFN hidden dim = 14336

Params par expert FFN ≈ 3 * 4096 * 14336 ≈ 176 M
Params totaux des 8 experts par layer ≈ 1.4 G
x 32 layers ≈ 45 G de params FFN

+ attention, embeddings, etc. ≈ 47 G total params
```

### Le piege du nom "8x7B"

Le nom "8x7B" suggere 56B. **Faux.** Les experts ne sont pas 8 modeles de 7B independants, ils ne sont que les **FFN** d'un modele. L'attention, les embeddings et le LayerNorm sont **partages**. Total reel : **~47B**.

### Params actifs par token

```
A chaque token, k=2 experts sur 8 sont actives :
  Params FFN actifs ≈ 2 * 176 M * 32 layers ≈ 11 G
  + attention partagee ≈ 2 G
  ≈ 13 G params actifs par token
```

Donc Mixtral est **un modele 47B avec le compute d'un 13B**. C'est l'argument commercial.

### Performance reelle

Mixtral 8x7B bat Llama 2 70B dense sur la plupart des benchmarks, avec **5x moins de FLOPs/token**. Mais **prend 47B de VRAM**, comme un modele 47B dense — il faut tous les experts en memoire au cas ou un token aurait besoin d'eux.

---

## 4. DeepSeekMoE : shared experts + fine-grained

DeepSeek-V3 (decembre 2024, 671B params, 37B actifs) a pousse MoE plus loin avec **deux innovations**.

### Innovation 1 — Fine-grained experts

Au lieu de **8 experts gros**, **256 experts petits** (chacun ~1/8 de la taille d'un expert Mixtral). Avec top-k=8, on active toujours ~ la meme proportion totale, mais la combinaison de specialisations est **beaucoup plus flexible**.

```
Mixtral 8x7B   : 8 experts, top-2 → C(8,2) = 28 combinaisons
DeepSeek-V3    : 256 experts, top-8 → C(256,8) ≈ 4*10^14 combinaisons
```

Plus de granularite = meilleure specialisation, moins de redondance entre experts. **Nuance** : cette intuition combinatoire est une borne superieure ; en pratique le gain vient de la granularite de specialisation, pas du nombre de combinaisons.

### Innovation 2 — Shared experts

DeepSeek ajoute **1 expert "partage"** (shared) qui est **toujours active** pour tous les tokens, en plus des k experts routes. Idee : factoriser la connaissance commune (grammaire de base, raisonnement general) dans le shared expert, et laisser les routed experts se specialiser sur des niches.

```
Output = shared_expert(x) + sum_{i in top-k routed}  weight_i * expert_i(x)
```

Ce design reduit le besoin de **dupliquer** la connaissance commune dans plusieurs experts routes — un probleme reel observe sur Mixtral.

### Les chiffres DeepSeek-V3

```
671 B params totaux
37 B params actifs par token (5.5% du total)
256 routed experts + 1 shared expert
top-8 routing
```

C'est **~14x plus de params totaux que Mixtral pour ~3x plus de compute** (671/47 ≈ 14.3). Le ratio sparsite est radicalement plus agressif.

---

## 5. Le probleme du load balancing

Si on laisse le routeur libre, **un piege** : il converge vers **2-3 experts favoris** qui gagnent toutes les requetes. Les autres experts ne recoivent jamais de tokens, ne s'entrainent pas, et meurent.

C'est un **collapse** classique : meme cause que le mode collapse en GAN.

### Solution 1 — Load balancing loss (Shazeer 2017)

Loss auxiliaire qui penalise un routage desequilibre :

```
f_i = fraction de tokens routes vers expert i  (dans le batch)
P_i = probabilite moyenne attribuee a expert i par le softmax
L_aux = N * sum_i (f_i * P_i)
```

Si tous les experts recoivent une part egale (`f_i = P_i = 1/N`), alors `L_aux = 1`. Si un expert prend tout (`f_1 = P_1 = 1`), alors `L_aux = N`. La loss totale devient :

```
L = L_task + alpha * L_aux       avec alpha ~ 0.01
```

Le gradient de `L_aux` pousse le routeur a equilibrer.

### Solution 2 — Z-loss (ST-MoE, Zoph 2022)

Penalise les logits trop grands en valeur absolue (eviter softmax instable) :

```
L_z = mean( logsumexp(logits)^2 )
```

Stabilise le training.

### Solution 3 — Expert-choice routing (Zhou 2022)

**Inversion du probleme** : au lieu que chaque token choisisse k experts, **chaque expert choisit ses M tokens preferes** dans le batch. Garantit par construction un parfait equilibre (chaque expert a exactement M tokens). Mais : **certains tokens ne sont selectionnes par aucun expert** → drops.

```
Token-choice : token → top-k experts        (peut deborder un expert)
Expert-choice : expert → top-M tokens       (peut dropper un token)
```

DeepSeek-V3 a utilise une variante : **auxiliary-loss-free balancing** via un biais ajoute aux logits du routeur, ajuste dynamiquement par expert (si un expert est sur-utilise on baisse son biais). Plus simple, pas de loss auxiliaire a tuner.

---

## 6. Capacity factor & token dropping

Probleme pratique : sur GPU, on **alloue un buffer fixe par expert**. Si trop de tokens veulent le meme expert, **le buffer deborde** → on **drop** les tokens en exces (ils passent par le residual seul).

### Capacity factor (CF)

```
capacity = CF * (tokens_par_batch / N)
```

- `CF = 1.0` : equilibre theorique parfait, mais en pratique 1-3% des tokens sont droppes (selon le paper Switch Transformer)
- `CF = 1.25` : marge de 25%, ~1% droppes (defaut Switch Transformer)
- `CF = 2.0` : presque 0% droppes, mais 2x plus de memoire allouee

C'est un **tradeoff direct memoire vs qualite**. Plus la load balancing loss est efficace, plus on peut baisser CF.

### Token dropping en inference

A l'inference, on ne **drop** generalement **pas** : on attend que l'expert traite tous ses tokens (pas de buffer fixe). C'est le training qui souffre du capacity factor. Bonne nouvelle pour la production.

---

## 7. Communication cost & expert parallelism

Sur multi-GPU, MoE introduit un **goulot d'etranglement reseau** specifique : **tous-vers-tous (all-to-all)**.

```
GPU 0 : experts 0-1
GPU 1 : experts 2-3
GPU 2 : experts 4-5
GPU 3 : experts 6-7

Pour chaque token :
  1. Calcule routing sur son GPU local
  2. ENVOIE le token vers les GPU(s) qui hebergent les experts choisis
  3. L'expert calcule
  4. RENVOIE le resultat
  5. Combine

= 2 all-to-all communications par layer MoE
```

L'all-to-all sature vite la bande passante (NVLink ~600 GB/s, Ethernet beaucoup moins). Sur les gros modeles MoE distribues, **20-50% du temps total est en communication**, pas en compute.

C'est pourquoi MoE est un **investissement infra** : sans NVLink/IB, le MoE perd ses gains. C'est aussi pourquoi DeepSeek a eu besoin d'innovations specifiques (DualPipe, FP8 communication) pour scaler V3.

---

## 8. Tradeoffs en pratique : MoE ou dense ?

| Critere | Dense | MoE |
|---|---|---|
| FLOPs / token | x1 | **0.2-0.3x** |
| VRAM totale | x1 | **3-15x** (faut tout charger) |
| Throughput inference (avec VRAM ok) | x1 | **3-5x** |
| Latence single-batch | x1 | similaire |
| Qualite a budget compute training egal | x1 | **+10-30%** |
| Stabilite training | facile | hard (load balance, drops) |
| Fine-tuning | facile | tricky (les experts collapse) |
| Tasks low-resource (langues rares) | meilleur | moins bon (data sparse → experts non couverts) |

### Quand utiliser MoE en 2026

1. **Tu sers du volume** : throughput-bound, pas latency-bound. MoE excelle.
2. **Tu as la VRAM** : H100/B200 cluster. Si tu es sur RTX 4090, oublie.
3. **Tu fais du pretraining a grande echelle** : MoE donne plus de qualite par GPU-hour. C'est pour ca que tous les frontier models 2025-2026 (GPT-5, Gemini 2.5, Claude 4.5/4.6, DeepSeek V3) sont MoE.

### Quand **ne pas** utiliser MoE

1. **Edge/mobile/single-GPU** : VRAM dominee par les experts dormants. Un dense 7B est plus efficient.
2. **Fine-tuning sur petit dataset** : risque de degrader le routing equilibre.
3. **Tasks tres specialisees** : un dense fine-tune > un MoE generaliste.

---

## 9. Idees fausses repandues

**Idee fausse 1 — "Mixtral 8x7B = 8 modeles 7B en ensemble"**
Faux. Les experts sont uniquement les FFN. L'attention, les embeddings, les LayerNorm sont partages. Et l'output est une combinaison ponderee des **2** experts choisis, pas un vote sur 8.

**Idee fausse 2 — "MoE consomme moins de VRAM"**
Faux. MoE consomme **plus** de VRAM totale (tous les experts doivent etre charges au cas ou). MoE consomme moins de **compute** par token. C'est different.

**Idee fausse 3 — "Chaque expert se specialise sur un domaine semantique humain (math, code, francais, ...)"**
Vrai pour Mixtral (paper Mistral §5) : les experts se specialisent sur des **patterns syntaxiques** (ponctuation, tokens numeriques, debuts de mots) plutot que sur des domaines semantiques. La semantique reste distribuee. **Pour DeepSeek-V3 cependant, le fine-grained routing produit une specialisation par domaine plus marquee** — la granularite plus fine permet aux experts de capturer des niches semantiques que les 8 gros experts de Mixtral ne pouvaient pas isoler.

**Idee fausse 4 — "Plus d'experts = toujours mieux"**
Faux. Au-dela d'une certaine sparsite, le routeur devient instable, les experts collapse, et la communication GPU explose. Le sweet spot est typiquement 16-256 experts avec sparsite ~5-15% (DeepSeek-V3 = 5.5%).

**Idee fausse 5 — "Le MoE remplace le dense partout"**
Faux. Les modeles edge/on-device (Phi-4, Gemma 3, les variantes 3B-8B) restent **dense**. MoE n'a de sens qu'a l'echelle du datacenter.

---

## Key takeaways (flashcards)

**Q1** — Quelle est la promesse fondamentale de MoE ?
> Decoupler **params totaux** (capacite de stockage) et **compute par token** (FLOPs). Un MoE 47B peut tourner avec le compute d'un 13B en routant chaque token vers seulement 2 experts sur 8.

**Q2** — Pourquoi Mixtral 8x7B fait 47B et pas 56B ?
> Les "8 experts" sont uniquement les FFN. L'attention, les embeddings et les LayerNorm sont partages entre les experts. Ils ne sont pas 8 modeles independants.

**Q3** — A quoi sert la load balancing loss de Shazeer ?
> Empecher le routeur de favoriser 2-3 experts en collapse. Penalise un routage desequilibre via `N * sum(f_i * P_i)`, pousse vers une utilisation uniforme.

**Q4** — Quelle est la difference entre token-choice et expert-choice routing ?
> Token-choice : chaque token choisit ses k experts (peut faire deborder un expert → token drops). Expert-choice : chaque expert choisit ses M tokens (equilibre garanti, mais certains tokens ne recoivent aucun expert).

**Q5** — Quand ne PAS utiliser MoE ?
> Sur un seul GPU avec VRAM limitee (RTX 4090, edge, mobile) : MoE consomme la VRAM des experts dormants pour rien. En fine-tuning sur petit dataset : le routing equilibre se degrade. Pour des taches tres specialisees ou un dense fine-tune fait mieux.

---

## Sources

- Shazeer et al. (2017) — *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. https://arxiv.org/abs/1701.06538
- Lepikhin et al. (2020) — *GShard: Scaling Giant Models with Conditional Computation*. https://arxiv.org/abs/2006.16668
- Fedus, Zoph, Shazeer (2021) — *Switch Transformer: Scaling to Trillion Parameter Models*. https://arxiv.org/abs/2101.03961
- Zoph et al. (2022) — *ST-MoE: Designing Stable and Transferable Sparse Expert Models*. https://arxiv.org/abs/2202.08906
- Jiang et al. / Mistral AI (2024) — *Mixtral of Experts*. https://arxiv.org/abs/2401.04088
- DeepSeek-AI (2024) — *DeepSeekMoE: Towards Ultimate Expert Specialization*. https://arxiv.org/abs/2401.06066
- DeepSeek-AI (2024) — *DeepSeek-V3 Technical Report*. https://arxiv.org/abs/2412.19437


---

## Pour aller plus loin

Lectures couvrant ce sujet (playlists dans [`shared/external-courses.md`](../../../shared/external-courses.md)) :

- **Stanford CS336 — Lec. 4 (Mixture of Experts)** — derivation complete du routing, load balancing, training MoE.
- **CMU 11-711 (Neubig) — Lec. 14 (Ensembling & Mixture of Experts)** — perspective NLP sur les architectures sparses.
