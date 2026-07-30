# Jour 9 — Architecture des LLMs modernes : de GPT-2 a Kimi K3

> **Temps estime** : 6h | **Prerequis** : Jours 1-8 (transformers, attention, tokenization)

---

## 1. Pourquoi l'architecture du Transformer a evolue

> **Pont / ordre de lecture** — ce module empile 7-8 innovations. Ne cherche pas a tout absorber d'un coup. Lis-le en deux passes :
> - **Core (a maitriser absolument)** : RoPE (§2), RMSNorm (§3), SwiGLU (§4), GQA (§5). Ces 4 briques sont le standard 2024 (Llama 3, Mistral, Qwen 2.5) — c'est le minimum vital pour lire un paper SOTA.
> - **Approfondissement (frontiere, 2e passe)** : MLA (§6), MoE fine-grained (§7), attention hybride lineaire + globale (§8), attention residuals (§8bis). Ces 4 definissent la frontiere 2025-2026 ; les deux premieres sont creusees en profondeur dans les modules dedies J16 (MoE) et J17 (SSM / attention lineaire). Survole-les ici, reviens-y avec ces modules.
>
> Strategie : fais d'abord les 4 sections core en comprenant le *pourquoi* de chaque remplacement, puis attaque la frontiere.

Le Transformer original (Vaswani et al., 2017) a tenu 5 ans avec peu de modifications. Mais entre 2022 et 2026, **8 innovations** sont devenues standard dans les LLMs SOTA :

1. **RoPE** (Rotary Position Embedding) remplace les positional embeddings sinusoidaux
2. **RMSNorm** remplace LayerNorm
3. **SwiGLU** remplace le ReLU/GeLU dans le FFN
4. **GQA** (Grouped Query Attention) remplace la Multi-Head Attention classique
5. **MLA** (Multi-Latent Attention, DeepSeek V3) pousse la compression du KV cache plus loin
6. **MoE fine-grained** (Mixture of Experts) decouple params totaux et flops actifs
7. **SSM / attention lineaire hybrides** (Mamba, KDA + attention) pour le tres long contexte
8. **Attention residuals** (AttnRes, Kimi K3) : chaque couche va rechercher selectivement les representations des couches precedentes au lieu de les accumuler uniformement

Les 4 premieres sont le standard 2024 (Llama 3, Mistral, Qwen 2.5). Les 4 dernieres definissent la frontiere 2025-2026 (Llama 4, DeepSeek V3/R1, Qwen 3, Jamba, Kimi K3 ; les familles fermees comme Gemini et GPT-5 sont presumees utiliser des recettes MoE similaires, sans details publics).

### Vue d'ensemble des differences

| | GPT-2 (2019) | LLaMA 2 (2023) | Llama 3 / Qwen 2.5 (2024) | DeepSeek V3 / Llama 4 (2025) | Kimi K3 (2026) |
|---|---|---|---|---|---|
| Position encoding | Learned absolute | RoPE | RoPE (+ YaRN) | RoPE (YaRN, 1M+ tokens) | **NoPE** — position portee par la recurrence KDA |
| Normalization | LayerNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm (+ un RMSNorm insere avant l'up-projection MoE) |
| Norm position | Pre-norm | Pre-norm | Pre-norm | Pre-norm | Pre-norm + **AttnRes** (residuel par attention sur la profondeur) |
| FFN activation | GeLU | SwiGLU | SwiGLU | SwiGLU | **SiTU-GLU** (SwiGLU borne) |
| Attention | MHA | MHA/GQA | GQA | **MLA** (DeepSeek) / GQA (Llama 4) | **Hybride KDA + Gated MLA** (3 couches lineaires : 1 globale) |
| Dense vs MoE | Dense | Dense | Dense | **MoE fine-grained** (256 experts, top-8) | **Stable LatentMoE** (896 routed top-16 + 2 shared) |
| Vocab size | 50 257 | 32 000 | 128 256 | 129 280 (DeepSeek V3) | 160 000 |

**Modeles SOTA 2025-2026** : Llama 4 (Meta, 2025, MoE), DeepSeek V3 (dec 2024, 671B totaux/37B actifs) et R1 (jan 2025, reasoning), Qwen 3 (Alibaba, 2025), Gemini 2 (Google, 2025), GPT-5 (OpenAI, 2025), Claude 4 (Anthropic, 2025), **Kimi K3** (Moonshot AI, juillet 2026, 2.78T totaux / 104B actifs, open-weight). GPT-3 et LLaMA 2 servent encore de repere historique mais ne sont plus SOTA.

> **Le repere open-weight de 2026 — Kimi K3.** Publie le 16 juillet 2026 avec son rapport technique et ses poids, c'est le plus gros modele open-weight a ce jour : **2.78 T de parametres totaux, 104.2 B actifs par token, 93 couches, contexte d'entrainement de 1 M de tokens, multimodal natif** (encodeur vision MoonViT-V2, 401 M). Son interet pedagogique n'est pas la taille mais le fait qu'il deplace **trois** briques du tableau ci-dessus d'un coup : l'attention (hybride lineaire/globale), le FFN (SiTU-GLU + experts en espace latent) et le chemin residuel (AttnRes). Moonshot annonce ~**2.5x d'efficacite de scaling** par rapport a Kimi K2 (meme loss de validation pour 2.5x moins de FLOPs, Fig. 7 du rapport). C'est la reference a lire quand on veut voir comment les briques de ce module se combinent reellement a l'echelle frontier.

> **Note sur les tailles** : seules les tailles des modeles open-weight (DeepSeek, Qwen, Llama) sont confirmees. Pour les modeles fermes (GPT-4/GPT-5, Claude, Gemini), aucune taille n'est officielle — les chiffres qui circulent sont **non confirmes, estimations**.

---

## 2. RoPE — Rotary Positional Embedding

### Le probleme avec les positional embeddings sinusoidaux

Le Transformer original (2017) utilise des sinusoides pre-calculees ajoutees aux embeddings de tokens :

```
input_embedding[i] = token_embedding[i] + pos_embedding[i]

avec pos_embedding[i, 2j] = sin(i / 10000^(2j/d))
     pos_embedding[i, 2j+1] = cos(i / 10000^(2j/d))
```

**Probleme 1 — Absolu, pas relatif** : la position 5 est encode pareil, que la sequence fasse 10 ou 10 000 tokens. Mais ce qui compte pour le language, c'est souvent la position **relative** (distance entre deux tokens), pas absolue.

**Probleme 2 — Extrapolation** : si on entraine avec des sequences de 2048 tokens max, le modele ne sait pas quelle position assigner au token 5000 en inference.

**Probleme 3 — Additif** : la somme `token + pos` melange les signaux. Le modele doit apprendre a les separer.

### L'idee de RoPE (Su et al., 2021)

Au lieu d'**ajouter** une position, on fait **tourner** les queries et keys dans l'espace 2D.

**L'insight fondamental** : le produit scalaire entre deux vecteurs tournes depend SEULEMENT de leur difference d'angle.

```
q_m = R(m * θ) @ q    # query a la position m, tourne de m*θ
k_n = R(n * θ) @ k    # key a la position n, tourne de n*θ

q_m · k_n = q · R(m*θ)^T · R(n*θ) · k
          = q · R((n-m) * θ) · k     ← depend seulement de (n - m)
```

Donc le produit scalaire d'attention capture automatiquement la **position relative** entre les tokens m et n.

### Comment ca marche concretement

Pour un vecteur `q` de dimension `d`, on le divise en paires `(q_0, q_1), (q_2, q_3), ...`. Chaque paire est un vecteur 2D qu'on fait tourner d'un angle qui depend de la position.

```
Paire j de q, a la position m:
  theta_j = 1 / 10000^(2j/d)    # frequence pour la paire j
  angle = m * theta_j            # angle de rotation

  [q_2j]      [cos(angle) -sin(angle)]   [q_2j]
  [q_2j+1] -> [sin(angle)  cos(angle)] * [q_2j+1]
```

Chaque paire tourne a une vitesse differente : avec theta_j = 10000^(-2j/d), les premieres paires (j petit) tournent vite (capturent les relations locales, courtes distances), les dernieres tournent lentement (relations longues).

### Avantages de RoPE

1. **Position relative native** : le produit scalaire d'attention encode directement `(n - m)`
2. **Extrapolation** : on peut tourner les vecteurs a n'importe quel angle, y compris pour des positions jamais vues
3. **Pas de parametres supplementaires** : les angles sont deterministes, calcules a partir de la position
4. **Compatible avec l'attention causale** : s'applique a q et k avant le produit scalaire

### Extrapolation en pratique

Meme avec RoPE, un modele entraine sur 2048 tokens a du mal a generaliser a 100k tokens. Solution : **NTK-aware scaling**, **YaRN**, **Position Interpolation** — des techniques qui modifient legerement les frequences RoPE pour supporter des contextes plus longs. C'est comme ca qu'on arrive a GPT-4-128k et Claude 200k.

---

## 3. RMSNorm — la simplification de LayerNorm

### LayerNorm — rappel

```
LayerNorm(x) = ((x - mean(x)) / std(x)) * γ + β
```

- Calcule mean et std sur la dimension des features
- Centre (soustrait la mean) et normalise (divise par std)
- Applique une scale `γ` et un bias `β` learnable

### L'observation de RMSNorm (Zhang & Sennrich, 2019)

Experiences empiriques : le **centrage** (soustraire la mean) n'apporte presque rien aux performances. Ce qui compte, c'est la **re-scaling** (diviser par la magnitude).

### La formule

```
RMSNorm(x) = (x / RMS(x)) * γ

avec RMS(x) = sqrt(mean(x²)) = sqrt((1/d) Σ x_i²)
```

Differences avec LayerNorm :
- Pas de soustraction de la mean (RMS au lieu de std)
- Pas de bias β
- Meme scale γ learnable

### Pourquoi c'est mieux

1. **Moins de calcul** : 1 moyenne au lieu de 2 (pas de centrage)
2. **Plus stable numeriquement** : pas de soustraction (source de perte de precision en float16)
3. **Memes performances empiriques** : les benchmarks montrent que RMSNorm ≈ LayerNorm
4. **Moins de parametres** : pas de bias β (economie marginale mais symbolique)

Utilise par : LLaMA, Mistral, Gemma, Qwen, DeepSeek — c'est devenu le standard.

### Pre-norm vs Post-norm

Le transformer original utilise **post-norm** : LayerNorm apres l'attention/FFN et le residual.

```
Post-norm:  x -> x + Attention(x)  -> LayerNorm(x + Attention(x))
```

**Probleme** : pour les reseaux tres profonds (> 100 couches), post-norm converge mal. Les gradients explosent ou disparaissent.

**Pre-norm** : LayerNorm AVANT l'attention/FFN, puis residual.

```
Pre-norm:  x -> LayerNorm(x) -> Attention -> x + result
```

**Avantage** : le residual passe inchange, donc les gradients coulent mieux. Tous les LLMs modernes utilisent pre-norm.

---

## 4. SwiGLU — l'activation qui a supplante ReLU

### FFN classique dans le Transformer

```
FFN(x) = Linear_2(activation(Linear_1(x)))

Avec Linear_1 : d_model -> d_ff (d_ff ≈ 4 * d_model)
     activation : ReLU ou GeLU
     Linear_2 : d_ff -> d_model
```

### GLU (Gated Linear Unit) — l'idee generale

Au lieu d'une seule transformation, on fait **deux** transformations lineaires de l'input et on multiplie element-wise l'une par l'autre :

```
GLU(x) = (W_a @ x) * sigmoid(W_b @ x)
                     └─────────────┘
                        "gate" qui laisse passer ou non chaque valeur
```

L'une des deux branches joue le role d'une **porte** (gate) qui decide, pour chaque feature, combien laisser passer.

### SwiGLU — GLU avec Swish (Shazeer, 2020)

```
SwiGLU(x) = Swish(W_a @ x) * (W_b @ x)

avec Swish(x) = x * sigmoid(x)    (aussi appele SiLU)
```

Donc dans un FFN complet :

```
FFN_SwiGLU(x) = W_down @ (SwiGLU(x))
             = W_down @ (Swish(W_gate @ x) * (W_up @ x))
```

Il y a maintenant **3 matrices** au lieu de 2 : W_gate, W_up, W_down. Pour garder le meme nombre de parametres qu'un FFN classique, on reduit `d_ff` de 4 * d_model a 8/3 * d_model.

### Pourquoi SwiGLU marche mieux

Shazeer (2020) a teste toutes les variantes (GeLU, ReLU, Swish, GLU avec chaque activation) et SwiGLU gagne systematiquement de ~0.5% sur les benchmarks. Pas d'explication theorique satisfaisante, juste : "ca marche mieux empiriquement" (citation du paper : "we offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence").

Hypothese intuitive : le gating permet au modele de desactiver dynamiquement des neurones en fonction de l'input, un peu comme un MoE (Mixture of Experts) a echelle fine.

### 2026 — SiTU-GLU : borner SwiGLU pour survivre au FP4

Les **deux** facteurs de SwiGLU (`Swish(W_gate @ x)` et `W_up @ x`) sont non bornes. Si deux grandes coordonnees tombent au meme endroit, le produit explose : ce sont les **activation outliers**, et ils deviennent un vrai risque d'overflow des qu'on calcule en basse precision (FP8, FP4 — cf. J19).

Kimi K3 remplace SwiGLU par **SiTU-GLU** (Sigmoid Tanh Unit GLU) : on applique un *soft cap* `softcap(x, beta) = beta * tanh(x / beta)` **independamment** sur les deux branches.

```
SiTU-GLU(x) = [ beta_1 * tanh(W_gate @ x / beta_1) * sigmoid(W_gate @ x) ]
              * [ beta_2 * tanh(W_up   @ x / beta_2) ]

Kimi K3 : beta_1 = 4 (branche gate), beta_2 = 25 (branche up)
     ->  |SiTU-GLU(x)| <= beta_1 * beta_2 = 100   (borne dure)
```

Deux proprietes recherchees :
- **Pres de 0**, `tanh(u) ≈ u`, donc SiTU-GLU se comporte quasi exactement comme SwiGLU — on ne perd pas ce qui fait marcher SwiGLU.
- **Loin de 0**, la sortie sature a 100 au lieu de diverger.

Pourquoi **100** et pas une autre valeur ? Parce que les activations de Kimi K3 sont servies en **MXFP8 E4M3**, dont le maximum representable est **448** (cf. J19). La borne est donc choisie avec une marge confortable sous le plafond du format : au-dela, les valeurs ne sont pas "imprecises", elles deviennent des `inf` — et un `inf` dans une activation contamine tout le reste du forward. Le code du module mesure les deux regimes : en activations normales SiTU-GLU s'ecarte de SwiGLU d'environ 1 % (reparametrisation quasi transparente), mais des qu'on injecte des *massive activations*, SwiGLU depasse 448 de plusieurs ordres de grandeur la ou SiTU-GLU reste sous 100.

Difference avec un simple `clamp` : le soft cap garde un **gradient non nul** au-dela du seuil, la ou un clamp dur tue le gradient. Retenir la logique generale : a mesure que le training descend en precision numerique, les activations non bornees deviennent le maillon faible, et les architectures se mettent a **borner par construction**.

---

## 5. GQA — Grouped Query Attention

### Le probleme : le KV cache en inference

En generation autoregressive, a chaque nouveau token, on doit :
1. Calculer la query du nouveau token
2. La comparer avec TOUTES les keys des tokens precedents
3. Calculer l'attention sur toutes les values

Pour eviter de recalculer les K et V a chaque fois, on les **cache** :

```
KV cache size = 2 * n_layers * n_heads * head_dim * seq_len * batch_size * bytes_per_elem

Exemple LLaMA-7B en full MHA:
  n_layers = 32, n_heads = 32, head_dim = 128
  seq_len = 4096, batch = 1, bf16 (2 bytes)
  
  KV cache = 2 * 32 * 32 * 128 * 4096 * 1 * 2 = 2.15 GB

Pour un batch de 8 et seq 8k: 34 GB juste pour le cache.
```

**Probleme** : le KV cache devient plus gros que les poids du modele pour des longues sequences. C'est devenu le bottleneck principal en inference.

### MHA vs MQA vs GQA

**MHA (Multi-Head Attention)** — classique : chaque head a sa propre query, key, value.
```
Q: (n_heads, head_dim)
K: (n_heads, head_dim)
V: (n_heads, head_dim)
```

**MQA (Multi-Query Attention)** — Shazeer (2019) : toutes les heads partagent UNE seule key et value.
```
Q: (n_heads, head_dim)      ← n_heads queries
K: (1, head_dim)             ← une seule key partagee
V: (1, head_dim)
```

Reduction du KV cache par `n_heads` (ex: 32x plus petit). Mais la perte de performance est visible sur les benchmarks.

**GQA (Grouped Query Attention)** — LLaMA 2 (2023) : compromis. On groupe les queries en `n_kv_heads` groupes, chacun partageant une K/V.
```
Q: (n_heads=32, head_dim)
K: (n_kv_heads=8, head_dim)   ← 8 groupes de 4 queries
V: (n_kv_heads=8, head_dim)
```

Chaque groupe de 4 queries partage les memes K et V. Reduction du KV cache par `n_heads / n_kv_heads` = 4.

### Le bon compromis

```
MHA:  n_heads=32, n_kv_heads=32  → cache 100%, qualite max
MQA:  n_heads=32, n_kv_heads=1   → cache 3%, qualite -2%
GQA:  n_heads=32, n_kv_heads=8   → cache 25%, qualite -0.3%
```

GQA est le sweet spot : 4x de reduction de cache avec une perte de qualite negligeable. LLaMA 2 70B, Llama 3, Llama 4, Mistral, Gemma, Qwen 2.5 — tous utilisent GQA.

---

## 6. MLA — Multi-Latent Attention (DeepSeek V3)

### La motivation : GQA ne suffit plus

GQA divise le KV cache par le ratio `n_heads / n_kv_heads` (typiquement 4 a 8x). Mais sur les modeles frontier de 2025 (200B+ params, contextes de 128k a 1M), le KV cache redevient le bottleneck. DeepSeek V2 (mai 2024) puis V3 (dec 2024) ont introduit une idee plus radicale : **ne plus stocker K et V du tout**, seulement une projection latente de dimension reduite.

### L'idee : projection dans un espace latent

Au lieu de calculer et de stocker `K` et `V` en pleine dimension, on apprend une matrice de compression `W_DKV` qui projette l'input `x` dans un vecteur latent `c_kv` de petite dimension (ex: 512 au lieu de n_heads * head_dim = 4096). C'est **ce vecteur latent** qu'on cache.

```
Flow classique (MHA) :
  x -> W_K -> K (grand)  -> cache K
  x -> W_V -> V (grand)  -> cache V

Flow MLA :
  x -> W_DKV -> c_kv (petit, ex: 512 dims) -> cache c_kv
  Au decoding :
    K = W_UK @ c_kv    # decompression up-projection
    V = W_UV @ c_kv
```

Le vecteur latent `c_kv` est compresse ~10x par rapport au KV cache GQA equivalent. Pendant l'attention, on reconstruit `K` et `V` a la volee avec des matrices de decompression (`W_UK`, `W_UV`).

### Trade-off

- **Gain** : KV cache divise par ~10 (DeepSeek V3 avec 61 couches x head_dim=128 stocke seulement ~70 KB par token au lieu de ~700 KB en GQA standard)
- **Cout** : calcul supplementaire a chaque step de decoding (la decompression `W_UK @ c_kv`). Mais comme le decode est memory-bound, ce calcul ajoute est gratuit en wall-clock time
- **Astuce RoPE** : les matrices de decompression peuvent etre absorbees dans `W_Q` et `W_O` par fusion de matrices, donc l'overhead est minimal. Sauf pour les dimensions portant RoPE, qui doivent etre traitees separement (DeepSeek les garde en clair, c'est la partie "decoupled RoPE")

### Impact concret

DeepSeek V3 : 671B params totaux, 37B actifs (grace au MoE, voir section 7), MLA pour le KV cache. Resultat : le modele tient en inference sur 8 x H100 alors qu'un equivalent dense GQA demanderait le double. Ouvre la voie au long contexte 128k+ sur des modeles open-source.

### Le flow MHA -> MQA -> GQA -> MLA

```
MHA  : chaque head a sa propre K, V         [cache = 100%]
MQA  : toutes les heads partagent 1 K, V    [cache = 1/n_heads, qualite -2%]
GQA  : groupes de heads partagent K, V      [cache = 1/group_size, qualite -0.3%]
MLA  : cache = projection latente c_kv      [cache = 1/10, qualite equivalente MHA]
        -> K, V reconstruits a la volee
```

MLA est plus complexe a implementer mais c'est la nouvelle frontiere pour les modeles frontier.

### 2026 — Gated MLA : MLA gardee, mais devenue minoritaire

Kimi K3 conserve MLA, avec deux modifications, et surtout **ne la met plus que dans 1 couche sur 4** (les 3 autres sont en attention lineaire KDA, cf. §8) :

1. **Gate de sortie full-rank** : `y = W_o [ sigmoid(W_g @ x) * o_MLA ]`. Une porte dependante de l'input, canal par canal, laisse chaque token moduler ce qu'il lit de l'attention globale.
2. **NoPE** — *No Positional Encoding*. Aucun RoPE n'est applique aux queries/keys des couches MLA. La position est portee implicitement par la **recurrence et les gates de decroissance** des couches KDA intercalees ; les couches MLA ne font plus que de l'interaction globale de contenu.

Consequence pratique majeure : **plus de "decoupled RoPE" a gerer, et plus de RoPE scaling du tout** pour etendre le contexte — pas de retuning de base RoPE, pas de YaRN (cf. J18). C'est ce qui permet a K3 d'extrapoler jusqu'a 1 M de tokens en jouant seulement sur le curriculum de longueur (8K -> 64K en pre-training, 256K -> 1M en cooldown).

---

## 7. MoE modernes — decoupler params et flops

### Motivation : scaling dense bute sur le cost

Un modele dense de 400B params demande 400B de flops PAR token genere. Couteux a entrainer (millions de GPU-hours) et a servir (batch size ridicule sur H100). Mais la loi d'echelle (Chinchilla, scaling laws) dit : plus de params = meilleur modele. Comment avoir 400B de "savoir" sans payer 400B de flops ?

**Reponse MoE** : activer seulement **k experts parmi N** a chaque token. Le FFN est duplique N fois (les "experts"), un routeur choisit quels k experts appellent pour chaque token.

### Architecture d'un bloc MoE

Dans un bloc Transformer MoE, on remplace le FFN par :

```
def moe_block(x):
    # x : (batch, seq, d_model)
    scores = Linear(x)              # (batch, seq, N_experts)
    top_k_idx = topk(scores, k)     # indices des k experts elus
    top_k_w = softmax(scores[top_k_idx])  # poids de combinaison

    output = 0
    for i in top_k_idx:
        output += top_k_w[i] * Expert_i(x)   # Expert_i est un FFN_SwiGLU
    return output
```

Chaque token n'active que k experts sur N (ex: k=2, N=8 pour Mixtral ; k=8, N=256 pour DeepSeek V3). Le routeur est une simple `Linear + softmax`, ses flops sont negligeables. Les params totaux sont grands (N fois plus d'experts), mais les flops par token restent `k * flops_expert`.

### Exemples SOTA 2024-2025

- **Mixtral 8x22B** (Mistral, avril 2024) : 8 experts, top-2 actifs -> 176B totaux, ~39B actifs
- **DeepSeek V3** (dec 2024) : 256 experts **fine-grained** + 1 expert partage, top-8 actifs -> 671B totaux, **37B actifs**. "Fine-grained" = experts plus petits mais plus nombreux, meilleur balance
- **Qwen 3 MoE** (Alibaba, 2025) : variantes 30B et 235B avec MoE fine-grained
- **Llama 4** (Meta, 2025) : architecture MoE confirmee, plusieurs tailles
- **Kimi K3** (Moonshot, juillet 2026) : **896 routed experts + 2 shared, top-16** -> 2.78T totaux, **104B actifs** (3.7%). Les experts routes ne travaillent plus en pleine largeur `d = 7168` mais dans un **espace latent** de 3584 (0.5x) — voir J16 §4bis
- **GPT-5** et **Claude 4** : rumored MoE (architecture non publique mais indices par la tarification et les patterns de latence)

### Challenges

1. **Load balancing** : si le routeur envoie toujours les memes experts, les autres sont sous-utilises (expert collapse). Solution : auxiliary loss qui penalise le desequilibre (DeepSeek V3 va plus loin avec un "auxiliary-loss-free" balancing par bias adaptatif ; Kimi K3, avec ~10^3 experts par couche, remplace l'ajustement pas-a-pas du bias par un calcul de **quantile** — voir J16 §5).
2. **Expert parallelism** : distribuer les experts sur plusieurs GPUs. All-to-all communication devient le bottleneck entrainement.
3. **Inference** : batching MoE est delicat car les tokens d'un batch n'activent pas les memes experts. vLLM et SGLang ont des kernels dedies.

### Quand utiliser MoE

- **Oui** : on veut plus de savoir (params totaux) sans payer plus de flops par token. Ideal pour le training a grande echelle et l'inference grand batch
- **Non** : on optimise pour la VRAM (on doit quand meme charger tous les experts en memoire, meme si on en active peu)
- **Hybride** : les modeles Llama 4 combinent dense et MoE sur differentes tailles

---

## 8. Hybrides lineaire + global — au-dela de l'attention quadratique

### Motivation : l'attention est O(n^2)

L'attention standard est O(n^2) en compute et memoire sur la longueur de contexte. FlashAttention et GQA/MLA atenuent, mais ne changent pas la complexite. Pour un contexte de 1M tokens, O(n^2) = 1e12 operations par token, infaisable.

Les **State Space Models** (SSM) offrent une alternative O(n) : ils maintiennent un etat fixe qui compresse tout le passe, comme un RNN moderne avec un formalisme mathematique d'equation d'etat continue discretisee.

### Mamba (Gu & Dao, 2023)

Mamba introduit la **S6 layer** (Selective SSM). La nouveaute par rapport aux SSM precedents (S4, H3) : les parametres de l'equation d'etat dependent de l'input, ce qui permet au modele de **choisir selectivement** ce qu'il memorise ou oublie. En termes pratiques : Mamba peut "reseter" son etat sur un nouveau paragraphe, ce que S4 ne pouvait pas.

- Memoire : O(1) par token (etat fixe, pas de cache croissant)
- Compute : O(n) en forward pass
- Apprentissage : parallelisable via un scan associatif sur GPU

### Realite 2026 : les hybrides gagnent

Mamba **pur** n'a pas battu les Transformers sur les benchmarks LLM (GSM8K, MMLU, HumanEval). L'attention reste superieure sur le in-context learning et le reasoning a courte portee. Mais les **hybrides** — mettre un bloc d'attention toutes les N couches Mamba — sont prometteurs pour le long contexte et l'edge inference.

Exemples 2024-2026 :
- **Jamba** (AI21, 2024) : 52B params, alternance Mamba + attention + MoE. 256k context
- **Zamba** (Zyphra, 2024) : hybride SSM + attention partagee
- **Gemini 2** (Google, 2025) : rumored d'utiliser une architecture hybride pour son 1M+ token context
- **Llama 4** : rumeurs d'elements hybrides pour le long contexte
- **Kimi K3** (Moonshot, juillet 2026) : **la premiere demonstration a l'echelle frontier**. 93 couches = **69 couches KDA (attention lineaire) + 24 couches Gated MLA (attention globale)**, en blocs de `3 KDA -> 1 MLA`, avec une derniere couche MLA en fin de backbone pour que la sortie passe toujours par de l'attention globale. 2.78T params, open-weight, contexte 1M

**KDA** (*Kimi Delta Attention*) n'est pas un SSM au sens Mamba : c'est une **attention lineaire a delta rule** (lignee DeltaNet / Gated Linear Attention) avec une **porte d'oubli par canal**. Meme promesse pratique que Mamba — etat recurrent de taille **fixe**, cout lineaire en longueur — mais une regle de mise a jour differente (`S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T` : on *efface* la valeur deja associee a cette cle avant d'ecrire la nouvelle). Detail dans J17.

### Quand utiliser un backbone lineaire (SSM / attention lineaire)

- **Oui** : tres long contexte (> 200k tokens) ou l'O(n^2) de l'attention devient prohibitif
- **Oui** : edge inference (memoire fixe, pas de cache croissant)
- **Non** : en **pur**, pour du raisonnement ou du recall associatif dense — l'attention globale reste necessaire
- **Compromis** : les hybrides sont le sweet spot. Le ratio empirique s'est deplace de `1 attention / 7 Mamba` (Jamba, 2024) a `1 attention globale / 3 lineaires` (Kimi K3, 2026) : plus d'attention globale que Jamba, mais toujours 75 % de couches en cout lineaire

---

## 8bis. Attention Residuals — quand le residuel devient de l'attention (Kimi K3, 2026)

Les 8 sections precedentes optimisent toutes le **melange des tokens** (attention) ou le **melange des canaux** (FFN/MoE). AttnRes s'attaque a une troisieme dimension, restee inchangee depuis ResNet (2015) : le melange **des couches**.

### Le constat

La connexion residuelle standard fait ceci :

```
h_l = h_{l-1} + f_l(h_{l-1})
```

Toute l'information des couches precedentes est **compressee dans un unique etat** `h_{l-1}`. C'est exactement le goulot d'etranglement d'un RNN — mais sur la profondeur au lieu du temps. Une couche 60 qui aurait besoin de la sortie brute de la couche 12 doit esperer qu'elle a survecu a 48 additions successives.

Or on connait le remede a ce goulot, puisque c'est le probleme que le Transformer a resolu en 2017 sur la dimension temporelle : **remplacer l'accumulation par de l'attention**. AttnRes applique la meme methode a la profondeur.

### Le mecanisme

Chaque couche `l` porte une **pseudo-query apprise** `q_l = w_l` (un simple vecteur de parametres, pas une projection de l'input). Les cles/valeurs sont les **sorties des couches precedentes** (plus l'embedding) :

```
k_i = v_i = f_i(h_i)        pour 1 <= i <= l-1
k_0 = v_0 = h_1             (l'embedding de token, toujours accessible)

alpha_{i->l} = softmax_i( exp( q_l^T RMSNorm(k_i) ) )
h_l = sum_i  alpha_{i->l} * v_i
```

Chaque couche **choisit** dans quelles couches anterieures elle va lire, au lieu de recevoir une somme imposee. Le `RMSNorm` sur les cles est indispensable : sans lui, une couche a forte magnitude ecraserait les poids d'attention de toutes les autres.

### Pourquoi c'est payable

Naivement c'est `O(L^2 d)` en calcul — mais la profondeur d'un LLM reste modeste (`L < 100`), donc le calcul n'est pas le probleme. Le vrai cout est la **memoire** `O(L * d)` : il faut garder vivantes les sorties de toutes les couches (et les transmettre entre etages sous pipeline parallelism).

D'ou la variante reellement deployee, **Block AttnRes** : on decoupe les `L` couches en `N` blocs ; a l'interieur d'un bloc on somme (residuel classique), et l'attention ne porte que sur les `N` representations **de bloc**. La memoire retombe a `O(N * d)`.

```
Kimi K3 : 93 couches decoupees en blocs de 12 couches
          -> l'essentiel du benefice est deja atteint vers N ~ 8 blocs
```

> **A retenir** : le motif "remplacer une accumulation aveugle par une selection apprise" est le meme geste intellectuel que l'attention de 2017 — applique cette fois a la profondeur du reseau. Quand tu lis un rapport d'architecture, demande-toi toujours **sur quelle dimension** l'information circule mal : les tokens, les canaux, ou les couches.

---

## 9. Putting it all together — une couche LLaMA

Voici a quoi ressemble une couche LLaMA moderne, compare a GPT-2 :

### GPT-2 (2019)
```
def gpt2_block(x):
    # Post-norm: norm applied after the residual
    x = x + MHA(LayerNorm(x))
    x = x + FFN_GeLU(LayerNorm(x))
    return x
```

### LLaMA 3 / Mistral (2024)
```
def llama_block(x, pos):
    # Pre-norm: norm before the layer, residual outside
    h = RMSNorm(x)
    h = GQA(h, pos)  # RoPE is applied INSIDE GQA on q and k
    x = x + h

    h = RMSNorm(x)
    h = FFN_SwiGLU(h)
    x = x + h
    return x
```

### Combien d'innovations par couche ?

Une couche LLaMA moderne differe de GPT-2 sur :
1. **RoPE** dans GQA (au lieu d'embeddings de position en input)
2. **RMSNorm** (au lieu de LayerNorm)
3. **SwiGLU** (au lieu de GeLU)
4. **GQA** (au lieu de MHA)
5. **Pre-norm** (deja dans GPT-2 en fait, c'etait le Transformer original qui avait post-norm)

Chaque innovation apporte 0.5% a 2% de gains, mais l'effet cumulatif est enorme. Un LLaMA de 7B bat un GPT-2 de 1.5B par un enorme ecart, meme a compute equivalent.

---

## 10. Mistral specifiques

Mistral (2023) ajoute deux choses au-dela de LLaMA :

### 10.1 Sliding Window Attention (SWA)

Au lieu que chaque token voit tous les tokens precedents, il ne voit que les `W` derniers (typiquement W=4096).

```
Token a la position 10k : voit seulement les tokens 6k a 10k
```

Avantage : complexite d'attention O(N × W) au lieu de O(N²). Le KV cache est aussi limite a W.

Mais comment faire pour les dependances longues ? Recursivement ! A la couche 2, chaque token voit les W precedents. Parmi eux, certains viennent deja d'avoir integre de l'information plus lointaine dans la couche 1. Apres L couches, l'information peut parcourir `L × W` tokens. C'est le "theoretical receptive field".

### 10.2 Rolling buffer KV cache

Le cache n'est pas un buffer qui grandit, mais un **anneau** de taille W. Les anciens tokens sont overwrites. Memory = O(W) au lieu de O(N).

Mistral 7B batt LLaMA 2 13B grace a ces deux optimisations + un entrainement plus soigne.

---

## 11. Flash Cards — Active Recall

### Q1 : Pourquoi RoPE est-il meilleur que les positional embeddings sinusoidaux ?

<details>
<summary>Reponse</summary>

RoPE applique une **rotation** aux queries et keys avant le produit scalaire d'attention. La propriete cle :

```
q_m · k_n depend SEULEMENT de (n - m), pas de m et n separement.
```

Donc le produit scalaire d'attention encode directement la **position relative** entre les tokens, ce qui est ce qu'on veut pour le language (la distance entre deux mots compte plus que leur position absolue dans le texte).

Autres avantages :
- Pas de parametres supplementaires (les angles sont deterministes)
- Meilleure extrapolation aux sequences longues (avec des astuces comme NTK-aware)
- S'integre naturellement avec l'attention causale

</details>

### Q2 : Quelle est la difference entre LayerNorm et RMSNorm ?

<details>
<summary>Reponse</summary>

**LayerNorm** : `((x - mean(x)) / std(x)) * γ + β`
- Centre (soustrait la mean)
- Normalise par l'ecart-type
- Applique scale γ et bias β learnable

**RMSNorm** : `(x / RMS(x)) * γ` avec `RMS(x) = sqrt(mean(x²))`
- Pas de centrage (on divise juste par la magnitude)
- Pas de bias β
- Meme scale γ

**Pourquoi** : empiriquement, le centrage n'apporte rien pour le language modeling. RMSNorm est plus simple, plus rapide, plus stable en float16, pour des performances equivalentes.

Utilise par : LLaMA, Mistral, Gemma, tous les LLMs modernes.

</details>

### Q3 : Quel est le probleme du KV cache et comment GQA le resout-il ?

<details>
<summary>Reponse</summary>

**Le probleme** : en generation autoregressive, on cache les K et V de tous les tokens precedents. La taille du cache croit lineairement avec la longueur de la sequence et peut devenir plus grosse que les poids du modele.

```
KV cache = 2 * n_layers * n_heads * head_dim * seq_len * batch * bytes_per_elem
```

**MQA** (Multi-Query Attention) : toutes les heads partagent une seule K/V. Reduction maximale (n_heads fois plus petit) mais perte de qualite visible (~2%).

**GQA** (Grouped Query Attention) : compromis. On groupe les heads en `n_kv_heads` groupes (ex: 32 heads → 8 groupes de 4). Chaque groupe partage K/V. Reduction de 4x avec perte de qualite negligeable (~0.3%).

GQA est devenu le standard pour tous les LLMs 70B+ car le bottleneck d'inference est memoire-bound.

</details>

### Q4 : Pourquoi SwiGLU remplace-t-il ReLU/GeLU dans le FFN ?

<details>
<summary>Reponse</summary>

SwiGLU est un **Gated Linear Unit** avec l'activation Swish :

```
SwiGLU(x) = Swish(W_gate @ x) * (W_up @ x)
         = (x_gate * sigmoid(x_gate)) * x_up
```

Le principe : deux transformations lineaires, l'une joue le role de **porte** (sigmoid qui decide combien laisser passer), l'autre porte l'information.

**Avantage** : le modele peut desactiver dynamiquement des neurones en fonction de l'input. C'est une forme de MoE a echelle fine.

**Preuve empirique** : Shazeer (2020) a teste toutes les variantes et SwiGLU gagne systematiquement de ~0.5-1% sur les benchmarks, sans explication theorique (le paper dit "we attribute their success to divine benevolence").

Utilise par : LLaMA, Mistral, PaLM, Gemma.

</details>

### Q5 : Quelles sont les 4 innovations architecturales qui separent GPT-2 de LLaMA ?

<details>
<summary>Reponse</summary>

1. **RoPE** (Rotary Positional Embedding) au lieu de positional embeddings apprises
   → encode la position relative dans le produit scalaire d'attention

2. **RMSNorm** au lieu de LayerNorm
   → plus simple, plus stable, meme performance

3. **SwiGLU** au lieu de GeLU/ReLU dans le FFN
   → gating dynamique, +0.5% de qualite

4. **GQA** (Grouped Query Attention) au lieu de MHA
   → reduit le KV cache de 4x avec perte de qualite negligeable

Toutes sont des ameliorations relativement petites individuellement (0.3-2%) mais leur somme explique pourquoi LLaMA 7B bat GPT-2 1.5B malgre une architecture superficiellement similaire.

Bonus : LLaMA utilise aussi **pre-norm** (RMSNorm avant la layer) au lieu de post-norm. Mais c'etait deja dans GPT-2, le vrai Transformer de 2017 etait le seul a etre post-norm.

</details>

### Q6 : Que fait MLA (Multi-Latent Attention) et pourquoi DeepSeek V3 l'utilise ?

<details>
<summary>Reponse</summary>

**Idee** : au lieu de cacher `K` et `V` pleine dimension comme MHA/GQA, MLA projette l'input dans un vecteur latent `c_kv` de petite dimension (ex: 512) et ne cache QUE ce vecteur. Au decoding, `K` et `V` sont reconstruits a la volee par decompression (`K = W_UK @ c_kv`, `V = W_UV @ c_kv`).

**Gain** : KV cache divise par **~10x** par rapport a GQA. DeepSeek V3 (671B params, 37B actifs MoE) tient en inference sur 8 x H100.

**Cout** : calcul supplementaire par token (decompression). Mais le decode est memory-bound, donc ce cout compute est gratuit en wall-clock.

**Le flow complet** :
```
MHA  -> MQA  -> GQA  -> MLA
100%     3%    25%    10% (cache size vs MHA)
```

**Detail technique** : les dimensions portant RoPE doivent etre traitees separement (decoupled RoPE) car on ne peut pas absorber la rotation dans la decompression.

Utilise par : DeepSeek V2, V3, R1. C'est une des innovations architecturales majeures de 2024-2025 pour les modeles frontier open-source.

</details>

### Q7 : Qu'est-ce qu'un MoE fine-grained et pourquoi DeepSeek V3 utilise 256 experts top-8 ?

<details>
<summary>Reponse</summary>

**MoE (Mixture of Experts)** : le FFN est remplace par N experts (FFN dupliques). Un routeur (`Linear + softmax`) choisit les top-k experts a activer pour chaque token. Seuls `k * flops_expert` flops sont depenses, mais le modele a N fois plus de params au total.

**Fine-grained** : au lieu d'avoir peu de gros experts (ex: Mixtral 8x22B, 8 experts top-2), on a **beaucoup d'experts plus petits** (DeepSeek V3 : 256 experts top-8). Avantages :
- Plus de combinaisons possibles (C(256, 8) >> C(8, 2)), donc specialisation plus fine
- Meilleur load balancing (la variance de charge est plus faible)
- Meilleure qualite par param actif

**DeepSeek V3** : 671B params totaux, **37B actifs** par token (grace au MoE + 1 shared expert). On a le savoir d'un 671B mais le cout d'inference d'un 37B.

**Challenges** : load balancing (eviter qu'un expert prenne toute la charge), expert parallelism (distribuer les experts sur plusieurs GPUs), inference batching. DeepSeek V3 utilise un auxiliary-loss-free balancing par bias adaptatif.

**Autres exemples MoE 2024-2026** : Mixtral 8x22B, Qwen 3 MoE, Llama 4 (MoE confirme), Kimi K3 (896 routed top-16 + 2 shared, 2.78T/104B), GPT-5 et Claude 4 (rumored).

</details>

### Q8 : Quand utiliser un modele Mamba/SSM plutot qu'un Transformer ?

<details>
<summary>Reponse</summary>

**Mamba (S6 layer, Gu & Dao 2023)** : State Space Model selectif, O(n) au lieu de O(n^2), memoire **fixe** (etat compresse au lieu de cache croissant).

**Quand utiliser** :
- **Tres long contexte** (> 200k tokens) : l'O(n^2) de l'attention devient prohibitif
- **Edge inference** : memoire fixe, pas de cache qui explose
- **Streaming infini** : on peut decoder indefiniment sans accumuler de cache

**Quand NE PAS utiliser** :
- **Reasoning complexe a courte portee** : l'attention reste superieure (in-context learning, retrieval precis)
- **Benchmarks LLM standards** : Mamba pur n'a pas battu les Transformers

**Realite 2026** : le backbone lineaire PUR n'a pas gagne, mais les **hybrides** (couches lineaires + attention globale toutes les N couches) ont gagne. Exemples : Jamba (AI21, 52B, 256k context, 1 attn / 7 Mamba), Zamba (Zyphra), Gemini 2 (rumored hybrid pour son 1M+ context), et surtout **Kimi K3** (Moonshot, juillet 2026) qui porte le pattern a l'echelle frontier : 69 couches KDA (attention lineaire a delta rule) + 24 couches Gated MLA, ratio 3:1, 2.78T params open-weight, contexte 1M.

**Intuition** : la couche lineaire excelle pour compresser le passe ancien (resume, etat de taille fixe), l'attention excelle pour rappeler exactement un token precis (retrieval). Les hybrides combinent les deux forces.

</details>


---

## Pour aller plus loin

Lectures couvrant ce sujet (playlists dans [`shared/external-courses.md`](../../../shared/external-courses.md)) :

- **Stanford CS336 — Lec. 3 (Architectures), Lec. 4 (MoE)** — variantes modernes (LLaMA, GQA, RoPE, MoE) en detail.
- **CMU 11-711 (Neubig) — Lec. 14 (Ensembling & MoE)** — tour d'horizon des architectures sparses.
- **CMU 11-711 (Welleck) — Lec. 6 (Pretraining)** — choix d'architecture vu sous l'angle pretraining 2025.
