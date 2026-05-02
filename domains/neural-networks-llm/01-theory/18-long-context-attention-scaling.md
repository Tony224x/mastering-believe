# Jour 18 — Long context : Flash Attention, RoPE scaling, ring attention

> **Temps estime** : 5h | **Prerequis** : J5-J6 (attention, transformer), J9 (RoPE)

---

## 1. Le mur memoire : pourquoi 1M tokens c'est dur

L'attention vanilla (Vaswani 2017) calcule une matrice `S = Q @ K^T` de taille `N x N` ou `N` est la longueur de la sequence. Cette matrice doit etre materialisee en VRAM pour appliquer le softmax.

### Calcul du pic memoire (FP16, 1 head)

```
N = 1 024     -> S = 1024 * 1024 * 2 octets = 2 MB
N = 4 096     -> 32 MB
N = 16 384    -> 512 MB
N = 100 000   -> 19 GB (!)
N = 1 000 000 -> 1.9 TB par head (!!)
```

Multiplier par le nombre de heads (32-128) et de couches (32-80) : la matrice d'attention seule explose la VRAM bien avant les poids du modele. Pour un Llama 70B avec contexte 1M, le naif demande **plus de petaoctets** de VRAM. C'est physiquement impossible.

### Les deux complexites a separer

| Complexite | Vanilla attention | Importance pratique |
|---|---|---|
| FLOPs | O(N^2 * d) | Dominante apres N >> 10K |
| Memoire | O(N^2) | Le BOTTLENECK reel jusque la |
| IO HBM <-> SRAM | O(N^2) | Le VRAI bottleneck (nouveau insight 2022) |

L'insight de Tri Dao (Flash Attention 2022) : **le bottleneck n'est pas le compute, c'est les transferts memoire** entre HBM (VRAM lente, abondante) et SRAM (cache GPU rapide, minuscule). Une A100 SXM4 a ~2.0 TB/s de bandwidth HBM (A100 PCIe ~1.55 TB/s) mais ~19 TB/s en SRAM. Reduire les allers-retours = vitesse x10.

---

## 2. Flash Attention : tiling + recomputation

### Le bon vieux trick : trade compute for memory

L'idee est familiere de l'ingenierie systeme : si la memoire est saturee, **recalcule** plutot que de stocker. Flash Attention applique ce principe a l'attention.

### Flash Attention v1 (Dao, Mai 2022)

Algorithme :

```
1. Decouper Q, K, V en blocs (tiles) qui tiennent en SRAM (taille B = 128-256)
2. Pour chaque bloc de Q :
     Pour chaque bloc de K, V :
        Charger les tiles en SRAM
        Calculer S_ij = Q_i @ K_j^T en SRAM
        Mettre a jour le softmax en streaming (online softmax)
        Calculer la sortie partielle O_i += P_ij @ V_j
3. Au backward : on ne stocke PAS la matrice attention (N^2)
   On recalcule le bloc a la volee pendant le backward
```

Resultat : memoire **lineaire** O(N) au lieu de O(N^2). Vitesse 2-4x plus rapide sur les longues sequences. Adopte par PyTorch (`scaled_dot_product_attention`), HuggingFace, vLLM en 2023.

### Flash Attention v2 (Juillet 2023)

Trois optimisations :
1. **Reorder loops** : la boucle externe est maintenant sur Q (au lieu de K). Reduit les ecritures cross-warp.
2. **Less non-matmul FLOPs** : delaye la division par `sum(exp)` au tout dernier moment.
3. **Better parallelism** : parallelise sur la dimension de sequence en plus de batch/heads.

Gain : 2x sur A100, et premiere version vraiment exploitable sur des sequences > 32K en training.

### Flash Attention v3 (Juillet 2024, Hopper)

Les H100 ont des **Tensor Memory Accelerators** (TMA) et du support FP8. v3 exploite :
- **Asynchrony** : overlap copy HBM->SRAM avec compute matmul (warp specialization).
- **FP8** : 2x throughput par rapport a FP16, avec recalibration en ligne.
- **Block size adaptatif** selon la taille de la SRAM disponible.

Speedup ~1.5-2x vs FA2 sur H100. C'est ce qui rend possible l'entrainement Llama 4 / Claude 4.x avec 1M tokens sans cout prohibitif.

### Mental model

```
Vanilla:    Q, K, V dans HBM -> S (N^2) dans HBM -> softmax(S) dans HBM -> O dans HBM
                                ^^^^^^^^^^^^^
                              Le mur memoire

Flash:      Q, K, V dans HBM -> tiles dans SRAM -> O dans HBM
                                  ^^^^^^^^^^^^^
                              Jamais materialise
```

---

## 3. RoPE scaling : comment etendre 4K -> 128K -> 1M

### Rappel RoPE (J9)

RoPE encode la position en **rotant** chaque paire de dimensions du Q et K par un angle proportionnel a la position. Les frequences de rotation sont :

```
theta_i = base^(-2i/d)   avec base = 10 000 (typiquement)
```

Pour la position `m`, la dimension `i` est rotee de `m * theta_i`. Resultat : le produit `q_m . k_n` ne depend que de `(m - n)`, donnant l'invariance par translation.

**Le probleme** : si on a entraine avec contexte max `L = 4 096` et qu'on teste a `m = 32 000`, les angles `m * theta_i` pour les basses frequences sont **hors distribution** — le modele n'a jamais vu ces angles. Performance s'effondre.

### Solution 1 : Position Interpolation (PI, Meta, Juin 2023)

```
pos_effective = m * (L_train / L_target)
```

Pour passer de 4K a 32K : on **divise** la position par 8. Le modele voit toujours des angles dans la plage entrainee. Necessite un fine-tuning court (~1B tokens). Fonctionne mais comprime trop les positions courtes -> perte de precision locale.

### Solution 2 : NTK-aware scaling (bloc97, Juillet 2023)

Idee : ne pas interpoler uniformement. Les **hautes frequences** (resolution locale) doivent rester intactes, les **basses frequences** (long-range) doivent etre etalees.

```
new_base = base * (L_target / L_train)^(d / (d - 2))
```

On change la **base** (10 000 -> ~86 800 pour d=64, scale=8 : 10 000 x 8^(32/31) ~ 86 800) au lieu de comprimer les positions. Resultat : preservation des hautes frequences, fine-tuning souvent inutile. Adopte par Code Llama, Qwen.

### Solution 3 : YaRN (Peng et al., Septembre 2023)

YaRN = "Yet another RoPE extensioN". Combine NTK avec deux raffinements :
1. **Scaling par bande** : interpolation differenciee selon que la frequence est haute (intacte), moyenne (NTK) ou basse (PI).
2. **Attention temperature** : reduit la temperature softmax pour compenser la dispersion accrue des positions etendues.

```
Pour chaque dimension i :
  Si lambda_i (longueur d'onde) << L_train -> garde theta_i (haute freq)
  Si lambda_i >> L_train                   -> applique PI complete
  Sinon                                     -> NTK partiel (transition douce)
```

Resultat : meilleur fine-tune efficiency (~250M tokens pour 32K -> 128K), meilleure perplexity sur les benchmarks long-context (PG19, GovReport, Needle-in-Haystack). C'est ce qu'utilisent Llama 3.1 (128K), Mistral, Qwen 2.5 long-context.

### Comparaison rapide

| Methode | Tuning | Quality 32K | Quality 128K |
|---|---|---|---|
| PI | 1B tokens | Bon | Mediocre |
| NTK-aware | 0-200M tokens | Tres bon | Bon |
| YaRN | 250M tokens | Excellent | Excellent |

**Regle** : YaRN est le standard 2024-2026 pour etendre un modele court-contexte. PI seulement si tu n'as pas de budget de fine-tuning.

---

## 4. Sliding window vs full attention

### Sliding window attention (Mistral 7B, 2023)

Au lieu d'attention sur toute la sequence, chaque token n'attend que sur les `W` tokens precedents (W = 4096 par exemple).

```
Token m attend a [m-W, m-1]   (au lieu de [0, m-1])
```

Avantages :
- Memoire et compute **lineaires** O(N * W) au lieu de O(N^2).
- KV cache **constant** : on ne garde que les W derniers KV.

Receptive field implicite : apres `L` couches, l'information se propage sur `L * W` tokens. Mistral 7B (32 couches, W=4096) a un receptive field theorique de ~131K tokens, alors que chaque attention voit seulement 4K.

Limite : ce receptive field est **theorique**. En pratique, l'information se dilue a chaque hop. Mixtral 8x7B utilise aussi du sliding window (ce n'est pas un hybride couches alternees), et les versions ulterieures de Mistral ont adapte leur strategie au cas par cas.

### Full vs sliding : quand utiliser quoi

| Critere | Full attention | Sliding window |
|---|---|---|
| Comprehension long-range exacte | OUI | Limite |
| Memoire / cout | Cher | Pas cher |
| Bon pour tasks chat / Q&A long | OUI | Marginal |
| Bon pour streaming / long monologue | Non (trop cher) | OUI |

En 2026, la plupart des modeles frontier utilisent **attention hybride** : couches paires en full, couches impaires en sliding. C'est le pattern de Llama 4 et Claude 4.x (selon les fuites). Compromis VRAM/qualite optimal.

---

## 5. Attention sinks : pourquoi les premiers tokens sont speciaux

### Le probleme du streaming

Si on fait du streaming infini (chatbot qui parle pendant des heures), on ne peut pas garder tout le KV cache. Solution naive : sliding window — on jette les KV des tokens > W.

**Surprise** (StreamingLLM, Xiao et al., Septembre 2023) : si on jette les **premiers** tokens, la perplexity explose et le modele genere du gibberish. Si on jette les tokens du milieu, ca tient.

### L'explication

Le softmax de l'attention force la somme des poids a 1.0. Pour les tokens recents qui n'ont rien d'interessant a regarder en arriere (ex: un token de ponctuation), le modele "depose" l'attention quelque part — typiquement sur les **premiers tokens** de la sequence (les "sinks"). C'est comme un drain pour evacuer l'attention non-utilisee.

Si on retire les premiers tokens, le softmax doit redistribuer cette attention sur des tokens significatifs, ce qui **corrompt** leurs representations. Collapse.

### La fix : attention sinks

StreamingLLM propose de **toujours garder les 4 premiers tokens** (les sinks) + une fenetre glissante des tokens recents. 4 tokens suffisent pour drain l'attention. Resultat : streaming infini stable, sans degradation de perplexity.

Plus elegant : certaines equipes (Cohere, etc.) entrainent desormais des **tokens sinks dedies** (un token special au debut, jamais retire). Mistral et Llama 3+ exploitent plutot le BOS naturel comme sink emergent (pas un sink dedie entraine specifiquement, mais le meme effet en pratique).

### Mental model

```
Sequence:  [SINK] [SINK] [SINK] [SINK] ... ... ... [recent_W tokens]
                  ^^^^^^^^^^^^^^^^                  ^^^^^^^^^^^^^^^^
                  toujours gardes                  fenetre glissante
                  (drain l'attention)              (info contextuelle)
```

---

## 6. Ring attention : sequence parallelism pour 1M+

### Le probleme

Meme avec Flash Attention, la sequence elle-meme (KV cache) ne tient plus sur un seul GPU au-dela de ~200K tokens en FP16 (un Llama 70B a 80 layers x 1024 dims x 2 octets x 200K tokens = ~26 GB juste pour les KV). A 1M tokens : 130 GB. Une H100 a 80 GB.

### Solution : Ring Attention (Liu et al., Octobre 2023)

Au lieu de **data parallelism** (chaque GPU voit toute la sequence) ou **tensor parallelism** (chaque GPU voit une partie des heads), on fait **sequence parallelism** : chaque GPU detient un **chunk** de la sequence.

```
GPU 1: tokens [0     ... N/4]   -> Q1, K1, V1
GPU 2: tokens [N/4   ... N/2]   -> Q2, K2, V2
GPU 3: tokens [N/2   ... 3N/4]  -> Q3, K3, V3
GPU 4: tokens [3N/4  ... N]     -> Q4, K4, V4

Ring : a chaque step, chaque GPU passe son K, V au voisin (anneau).
       Apres K steps, chaque GPU a vu tous les K, V.
       Compute attention en streaming par chunk (style Flash).
```

L'overlap **communication / compute** est cle : pendant qu'un GPU calcule son attention sur le chunk recu, il envoie deja le suivant au voisin. Ratio idealement zero overhead.

Souvent evoque dans la communaute pour Gemini 1.5 Pro (1M-10M tokens) et Llama 4 long context, mais non confirme officiellement par Google. Demande un cluster bien interconnect (NVLink ou InfiniBand >200 Gb/s). Sur du commodity, le bandwidth limite la scalabilite.

---

## 7. Lost in the middle et needle-in-haystack

Avoir un contexte de 1M tokens **ne signifie pas** que le modele utilise 1M tokens. Trois realites mesurees :

### Lost in the middle (Liu et al., 2023)

Si on place une information critique au **milieu** d'un long contexte, les modeles la trouvent moins bien que si elle est au debut ou a la fin. Performance suit une courbe en U.

```
                  ▲
   Recall         │  *                                       *
                  │     *                              *
                  │        *                       *
                  │           *                 *
                  │              *      *  *  *
                  │
                  └────────────────────────────────────► position
                  debut          milieu              fin
```

C'est lie a la distribution d'attention pendant le training : les modeles s'habituent a regarder les tokens proches (recence) et l'instruction (debut). Le milieu est sous-represente.

### Needle in a Haystack (NIAH)

Test devenu standard : on cache une phrase ("the secret password is 42") dans un long contexte non-pertinent, et on demande de la recuperer. Les modeles 2024-2026 atteignent 95%+ sur ce test jusqu'a leur contexte annonce.

**Mais** : NIAH est un benchmark facile. Il teste **un fait isole**, pas la **comprehension multi-faits**. Les benchmarks plus durs (RULER, LongBench, Babilong) montrent que la performance reelle decroche bien avant le contexte annonce.

### Reality check 2026

Les chiffres "contexte effectif" ci-dessous sont des **estimations communautaires / non publiees officiellement** ; les valeurs RULER officielles ne sont generalement pas releasees par les labs frontier. A prendre comme ordre de grandeur, pas comme chiffre exact.

| Modele | Contexte annonce | Contexte effectif (RULER 80%+, estimation) |
|---|---|---|
| Claude 4.5 Opus | 1M | non publie (~moitie annoncee) |
| Gemini 2.5 Pro | 2M | non publie (~moitie annoncee) |
| GPT-5.4 | 1M | non publie (~moitie annoncee) |
| Llama 4 405B | 256K | non publie (~moitie annoncee) |

**Regle pragmatique** : pour de la production, considere le contexte effectif comme la moitie du contexte annonce. Au-dela, RAG ou chunking reste plus fiable.

---

## 8. Idees fausses repandues

**"Flash Attention reduit le compute"** — Non, il reduit les transferts memoire. Le compute (FLOPs) est le meme. Le speedup vient de la VRAM bandwidth saturee remplacee par de la SRAM bandwidth.

**"YaRN remplace le re-training"** — Non, YaRN demande quand meme un fine-tuning (~250M tokens). C'est juste plus efficient que PI pur (1B+ tokens).

**"Un contexte plus long est toujours mieux"** — Faux. Au-dela du contexte effectif, le bruit augmente, les hallucinations augmentent, les couts explosent. Souvent un RAG bien fait sur 8K tokens bat un dump 1M.

**"Sliding window = pas d'attention long-range"** — Faux. Le receptive field cumule via les couches permet de l'info long-range, mais diluee. Pour de l'extraction precise, full attention est requise.

**"Les attention sinks sont une astuce de streaming, ca ne concerne pas le training"** — Faux. Les modeles entraines en ignorant ce phenomene developpent quand meme des sinks emergents (souvent BOS ou les 2-3 premiers tokens). Mieux vaut les anticiper.

---

## Key takeaways (flashcards)

**Q1** — Pourquoi Flash Attention est-il rapide alors que les FLOPs sont identiques au vanilla ?
> Le bottleneck reel n'est pas le compute mais les transferts memoire HBM <-> SRAM. Flash tile la matrice d'attention pour qu'elle tienne en SRAM, evitant la materialisation N^2 en HBM. Bandwidth SRAM est 10-15x celle de HBM.

**Q2** — Quelle est la difference fondamentale entre Position Interpolation et NTK-aware scaling pour etendre un RoPE ?
> PI comprime les positions (divise m par L_target/L_train). NTK-aware modifie la base de RoPE (10 000 -> ~86 800 pour d=64, scale=8 via `new_base = base * scale^(d/(d-2))`) pour preserver les hautes frequences (resolution locale) et n'etaler que les basses frequences. NTK demande moins de fine-tuning et preserve la qualite locale.

**Q3** — Pourquoi enlever les premiers tokens d'un KV cache fait collapse le modele ?
> Les premiers tokens servent de "sinks" pour l'attention residuelle (le softmax doit sommer a 1, l'attention non-utilisee se depose la). Les retirer force le softmax a redistribuer cette masse sur des tokens significatifs, corrompant leurs representations. Solution : toujours garder ~4 sinks + fenetre glissante.

**Q4** — Quand prefererais-tu attention sliding window vs full attention ?
> Sliding pour streaming long, latence stricte, ou modeles avec contexte deep mais cout VRAM contraint. Full pour tasks de comprehension longue precise (Q&A doc, code review long, math multi-step). Le pattern hybride (couches alternees) est le standard 2026.

**Q5** — Quelle est la difference entre contexte annonce et contexte effectif ?
> Le contexte annonce est la longueur max techniquement supportee. Le contexte effectif (mesure par RULER, LongBench) est la longueur ou le modele recupere fiablement >80% des informations. Typiquement le contexte effectif est ~30-50% du annonce. Au-dela, RAG est preferable.

---

## Sources

- Dao et al. (2022) — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. https://arxiv.org/abs/2205.14135
- Dao (2023) — *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. https://arxiv.org/abs/2307.08691
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao (2024) — *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*. https://arxiv.org/abs/2407.08608
- Su et al. (2021) — *RoFormer: Enhanced Transformer with Rotary Position Embedding*. https://arxiv.org/abs/2104.09864
- Chen, Wong, Chen, Tian (2023) — *Extending Context Window of Large Language Models via Positional Interpolation*. https://arxiv.org/abs/2306.15595
- Peng, Quesnelle, Fan, Shippole (2023) — *YaRN: Efficient Context Window Extension of Large Language Models*. https://arxiv.org/abs/2309.00071
- Xiao, Tian, Chen, Han, Lewis (2023) — *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM). https://arxiv.org/abs/2309.17453
- Liu, Zaharia, Abbeel (2023) — *Ring Attention with Blockwise Transformers*. https://arxiv.org/abs/2310.01889
- Beltagy, Peters, Cohan (2020) — *Longformer: The Long-Document Transformer*. https://arxiv.org/abs/2004.05150
- Liu et al. (2023) — *Lost in the Middle: How Language Models Use Long Contexts*. https://arxiv.org/abs/2307.03172


---

## Pour aller plus loin

Lectures couvrant ce sujet (playlists dans [`shared/external-courses.md`](../../../shared/external-courses.md)) :

- **CMU 11-711 (Neubig) — Lec. 13 (Long Sequence Models), Lec. 23 (MagicPIG & Factor — Long Context)** — etat de l'art pre/post 2024 sur le long contexte.
- **CMU 11-711 (Welleck) — Lec. 16 (Long-Context Models)** — techniques 2025 (RoPE scaling, sliding window, attention sinks).
