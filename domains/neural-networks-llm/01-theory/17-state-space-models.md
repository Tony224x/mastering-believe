# Jour 17 — State Space Models : l'alternative a l'attention

> **Temps estime** : 5h | **Prerequis** : J5-J6 (attention, transformer)

---

## 1. Le probleme : attention en O(N^2), pourquoi c'est un mur

L'attention auto-regressive d'un transformer calcule, pour chaque token, un produit scalaire avec **tous** les autres tokens du contexte. Cout = O(N^2) en compute et O(N^2) en memoire (la matrice de scores).

### Chiffres concrets

Pour un modele 7B (d_model=4096, 32 heads), une sequence de N tokens en FP16 :

| N (contexte) | Memoire matrice attention par layer | Memoire totale 32 layers |
|---|---|---|
| 2 048 | 16 MB | 512 MB |
| 8 192 | 256 MB | 8 GB |
| 32 768 | 4 GB | **128 GB** |
| 131 072 | 64 GB | **2 TB** |

A 128k tokens, on n'entraine plus rien sur un GPU H100 (80 GB). Meme l'inference KV-cache devient le goulot. **C'est le mur quadratique**.

### Les patches actuels du transformer

- **FlashAttention** (Tri Dao, 2022) : meme complexite asymptotique mais constante divisee par 5-10 grace a la fusion des kernels GPU.
- **Sliding window attention** (Mistral, Longformer) : O(N*W) ou W est la taille de fenetre. On perd la portee globale.
- **Sparse / linear attention** (Performer, Linformer) : O(N*log N) ou O(N) mais qualite degradee sur le recall.
- **MQA/GQA** : reduit la taille du KV-cache, pas la complexite.

Aucun n'est une victoire propre. D'ou la question : **et si on remplacait l'attention par une primitive O(N) native ?**

---

## 2. L'idee SSM en une equation

Un **State Space Model** vient de la theorie du controle (annees 60). Il modelise une sequence comme un systeme dynamique a etat cache **continu** :

```
h'(t) = A h(t) + B x(t)        (equation d'etat, continue)
y(t)  = C h(t) + D x(t)        (equation de sortie)
```

- `x(t)` : input scalaire ou vectoriel au temps t
- `h(t)` : etat cache (vecteur de taille N, l'etat interne)
- `y(t)` : output au temps t
- `A, B, C, D` : matrices apprises (D est souvent nul ou skip-connection)

Pour un LLM, on **discretise** : le pas de temps `Delta` devient un parametre. La recurrence discrete (Zero-Order Hold) :

```
h_t = A_bar h_{t-1} + B_bar x_t
y_t = C h_t
```

avec `A_bar = exp(A * Delta)` et `B_bar = (A_bar - I) A^{-1} B`.

> **Note** : forme exacte ZOH ; Mamba utilise la variante equivalente B̄ = (ΔA)^{-1}(exp(ΔA)−I)·ΔB.

### Les deux modes equivalents

C'est **la** propriete cle des SSM :

1. **Mode recurrent** (inference) : applique la recurrence ci-dessus, O(N) en temps, O(1) en memoire par step. Comme un RNN.
2. **Mode convolutionnel** (training) : on deroule la recurrence et on observe que `y = K * x` ou `K = (CB, CAB, CA^2 B, ..., CA^{N-1} B)` est un kernel fixe. S4 calcule ce kernel en O(N log N) via une factorisation de Cauchy (sur la representation diagonale + low-rank de A), puis la convolution `y = K * x` se fait egalement en O(N log N) par FFT — parallelement sur toute la sequence.

**On a le meilleur des deux mondes** : l'efficacite memoire d'un RNN a l'inference, le parallelisme d'un transformer au training.

---

## 3. De S4 a Mamba : la selectivite

### S4 (Gu, Goel, Re — 2021) : le premier SSM qui marche

Le defi : initialiser A pour que les long-range dependencies ne s'effondrent pas (vanishing memory). Solution : **HiPPO** (High-order Polynomial Projection Operators) — A est initialise pour que h_t soit la meilleure approximation polynomiale de l'historique de x.

Resultat : sur Long Range Arena (sequences de 16k tokens), S4 explose les transformers. Mais sur le langage, il reste en-dessous : **probleme de recall**.

### Pourquoi S4 echoue en LM : le probleme du recall

Considere la tache "selective copying" : on donne des tokens, certains marques avec un flag "remember", d'autres avec "ignore". Le modele doit reproduire seulement les tokens flagges.

- **Transformer** : trivial, l'attention regarde le flag a chaque step.
- **S4** : echoue. Pourquoi ? Parce que A, B, C sont **fixes** — ils ne peuvent pas "regarder" le token courant et decider d'absorber ou ignorer. La recurrence est lineaire et **content-independent**.

### Mamba / S6 (Gu & Dao, decembre 2023) : la selectivite

Mamba modifie une chose, mais c'est tout :

```
B_t = Linear_B(x_t)            <-- B depend de x maintenant
C_t = Linear_C(x_t)            <-- C depend de x maintenant
Delta_t = softplus(Linear_d(x_t))  <-- pas de discretisation depend de x
A reste fixe (mais expressif via HiPPO)
```

**A_bar et B_bar sont maintenant fonction du token courant**. Le modele peut "decider" d'ouvrir ou fermer la memoire selon l'input. C'est un **gating** dynamique — l'equivalent fonctionnel de l'attention, mais en O(N).

> **Note importante** : A reste fixe en continu, mais Ā = exp(Δ_t · A) devient input-dependent via Δ_t. C'est par Δ que la selectivite agit sur la dynamique d'etat.

Resultat : Mamba egale ou bat les transformers de meme taille sur LM jusqu'a 7B parametres, avec contexte illimite en theorie et ~5x moins de memoire a l'inference longue.

---

## 4. Mamba en detail (selective scan, gating)

### Le bloc Mamba

```
                  x_t
                   |
          ┌────────┼────────┐
          |                 |
       Linear            Linear (gate)
          |                 |
       SSM(A,B,C,D)         |
       avec selectivite     |
          |                 |
          └────── * ────────┘   <-- multiplication element-wise (gate type GLU)
                   |
                Linear
                   |
                  y_t
```

Trois ingredients :
1. **SSM selectif** (cur du bloc) : applique la recurrence avec B_t, C_t, Delta_t calcules a partir de x_t.
2. **Gate multiplicatif** : un second chemin lineaire qui module la sortie SSM (style GLU/SwiGLU).
3. **Convolution causale 1D** locale avant le SSM : capture les dependances a tres courte portee (~3-4 tokens) que le SSM gere mal.

Empilement : un modele Mamba = N blocs Mamba + RMSNorm + residual, comme un transformer mais sans attention.

### Selective scan

L'algorithme cur de Mamba. Probleme : avec B_t et A_t qui dependent du token, on ne peut plus utiliser la FFT (le kernel n'est plus stationnaire). Il faut faire un **scan** : calcul sequentiel.

Naivement, c'est O(N) sequentiel — donc lent sur GPU malgre la complexite asymptotique correcte. Tri Dao a ecrit un kernel CUDA custom qui :
- Garde les etats `h_t` en SRAM (pas en HBM, evite les allers-retours memoire couteux)
- Implemente un **parallel scan** (algorithme de Blelloch) : O(N) work mais O(log N) depth en parallele
- Recompute selectivement plutot que stocker tous les `h_t` (gradient checkpointing implicite)

Resultat : Mamba est ~5x plus rapide qu'un transformer optimise pour les longues sequences a parametres egaux.

---

## 5. Le hardware : pourquoi le scan parallele de Mamba est hard

### Le defi GPU

Une attention transformer = matmul = tres bien parallelisable, le GPU adore. Un scan recurrent = chaine de dependances = le GPU deteste.

Le truc de Tri Dao :

```
GPU memoire :
  HBM (80 GB)  : lent, mais grand. C'est ici qu'on stocke poids et activations.
  SRAM (~20 MB par SM) : 100x plus rapide, mais minuscule.

Naivement, un SSM ferait :
  for t in range(N):
      lire h_{t-1} depuis HBM         <-- lent
      lire A_t, B_t, x_t depuis HBM   <-- lent
      calculer h_t                    <-- rapide
      ecrire h_t en HBM               <-- lent
  Bande passante HBM = goulot.

Le selective scan kernel :
  Charge un chunk en SRAM
  Fait tout le scan local en SRAM
  Ne touche HBM qu'au debut/fin du chunk
  Bande passante divisee par ~10x
```

C'est le meme principe que FlashAttention (memoire-aware kernels). Tri Dao est l'auteur des deux. **Sans ce kernel, Mamba serait inutilisable.**

### Mamba-2 (mai 2024) : SSD = State Space Duality

Le papier suivant montre que les SSM selectifs et l'**attention lineaire** sont mathematiquement la meme chose — sous structure semi-separable scalaire (Mamba-2 introduit la contrainte A_t = a_t · I, c.-a-d. A_t scalaire fois identite par head). Plus precisement, Mamba-2 reformule le SSM comme une matrice triangulaire `M` ou les blocs diagonaux sont des SSM et les blocs hors-diagonale sont une forme d'attention low-rank. On peut alors :
- Utiliser des matmuls (rapides sur GPU) au lieu de scans
- 2-8x plus rapide que Mamba-1
- Compatible avec les techniques transformer existantes (tensor parallelism, etc.)

**SSD** est devenu la base de tous les SSM 2024-2026.

---

## 6. Hybrides Transformer + Mamba : Jamba, Zamba, Samba

### Pourquoi hybrider

Mamba est excellent en throughput et long context. Mais il a une faiblesse documentee : **recall associatif**. Un transformer peut faire "what was the password that user X gave 50k tokens ago ?" car la KV-cache stocke litteralement le mapping. Mamba doit compresser tout l'historique dans h_t (taille fixe). Si le contexte est dense, l'info se perd.

Solution pragmatique : **garder quelques layers d'attention** dans une stack majoritairement Mamba.

### Jamba (AI21, mars 2024)

Architecture : 1 layer attention pour 7 layers Mamba. + MoE.

Resultat :
- 256k tokens de contexte sur GPU H100 80GB (vs ~32k pour un Mistral comparable)
- 3x plus de throughput
- Qualite egale ou meilleure que Mixtral 8x7B sur la plupart des benchmarks

**Pourquoi 1/8 marche** : les rares layers d'attention permettent le recall ponctuel quand necessaire. Les layers Mamba font le gros du travail.

### Zamba 2 (Zyphra, 2024)

Architecture similaire, avec un seul block d'attention partage entre tous les layers. Reduit encore le KV-cache. Petits modeles (1.2B, 2.7B) qui battent Llama 3.2 a taille egale.

### Samba (Microsoft, 2024)

Mamba + Sliding Window Attention au lieu de full attention. Le SWA gere le local fin, le Mamba gere le long. Tres bon equilibre simplicite/perf.

### RWKV (Bo Peng, depuis 2021, evolutions continues)

Pas un SSM stricto sensu mais meme philosophie : recurrence lineaire formulee comme une attention "time-mixing" avec parametres exponentiels. Mode entrainable parallele, mode inference recurrent. Versions 5 et 6 (Eagle, Finch) competitives en 2025-2026, surtout en inference offline.

---

## 7. Forces & faiblesses : ou SSM gagne, ou il perd

### Ou SSM domine

| Domaine | Pourquoi |
|---|---|
| **Audio** (whisper-like, codecs neuraux) | Sequences ~50k-500k tokens, structure tres locale |
| **ADN / proteomique** (Evo, Caduceus) | Genome humain = 3 milliards de bases, transformer impossible |
| **Time-series** longues (signaux, capteurs) | Recurrence naturelle |
| **Vision haute-res** (ViM, Vision Mamba) | Patchs nombreux, pas besoin d'attention all-to-all |
| **Streaming / on-device** | Memoire constante a l'inference |

### Ou SSM perd encore (en 2026)

| Tache | Probleme |
|---|---|
| **Recall associatif dense** | Compression d'etat fixe. Benchmark **MQAR** (Multi-Query Associative Recall) le mesure precisement. Mamba pur = 30-50% vs transformer = 95%+. |
| **In-context learning few-shot** | Recall associatif sous-jacent. SSM moins bon. |
| **Code** (long-range refs sur classes/symbols) | Idem. Hybrides necessaires. |
| **Reasoning multi-step structure** | Les SSM peinent a maintenir des "registres" symboliques. |

### Le verdict pragmatique 2026

- **Pure SSM** : excellent pour throughput sur longs contextes, recherche, niches non-textuelles.
- **Pure Transformer** : encore le defaut pour LMs frontier qualite max < 128k.
- **Hybride (Jamba-like)** : **le sweet spot** pour les LMs production qui ont besoin de contexte > 128k. Tous les labs frontier explorent cette voie.

---

## 8. Idees fausses repandues

1. **"Mamba va remplacer le transformer"** : faux. Mamba egale le transformer a echelle moyenne mais perd sur le recall associatif. Les hybrides gagnent. Mamba est une **brique**, pas un remplacement.

2. **"SSM = RNN"** : non. Un RNN classique a une non-linearite (tanh, gate LSTM) dans la recurrence, ce qui interdit le mode convolutionnel parallele. Un SSM est **lineaire en h** dans la recurrence (la non-linearite est ailleurs, dans le bloc gating). C'est ce qui le rend parallelisable.

3. **"Mamba est lineaire donc moins expressif"** : la lineraite est dans la recurrence d'etat, pas dans le modele global. Les MLPs et le gate apportent la non-linearite. Mamba est universel sous certaines conditions (Wang et al., 2024).

4. **"Le contexte infini de Mamba implique un recall infini"** : le contexte est infini mais l'etat h_t est de taille **fixe** (typiquement 16-64 dims par canal). C'est un **compresseur lossy**. Plus le contexte est long, plus l'info ancienne est diluee.

5. **"FlashAttention rend l'attention O(N)"** : non. FlashAttention reduit la constante (memoire), pas l'asymptotique. C'est toujours O(N^2) en compute.

---

## 9. Cadre mental : quel choix en 2026

```
Question : quel backbone pour mon LM ?

Contexte typique > 128k tokens ET throughput critique ?
  ├── Oui  ──> Hybride Mamba+Attn (Jamba-style) ou MoE+Mamba
  └── Non
      └── Recall associatif critique (RAG dense, code, agents) ?
            ├── Oui ──> Transformer classique (FlashAttn-2, GQA)
            └── Non
                └── Audio / DNA / time-series ?
                      ├── Oui ──> Pure Mamba / Mamba-2
                      └── Non ──> Transformer (defaut sur)
```

**Regle d'or** : ne pas adopter un SSM par hype. Mesurer sur **ton** workload : MQAR-like benchmarks, throughput, qualite. Beaucoup d'equipes ont migre Mamba puis backtracke car leur usage etait recall-heavy.

---

## Key takeaways (flashcards)

**Q1** — Pourquoi l'attention transformer est dite quadratique et ou ca pose probleme ?
> Pour une sequence de N tokens, l'attention calcule une matrice N x N de scores. Cout = O(N^2) compute et memoire. Pour N > 32k, la matrice ne tient plus en VRAM. C'est le mur asymptotique qui motive les SSM.

**Q2** — Quelle est la propriete miracle des SSM lineaires ?
> Equivalence entre mode recurrent (O(N) inference, O(1) memoire par step) et mode convolutionnel (O(N log N) parallele au training via FFT). On a la rapidite RNN a l'inference + le parallelisme transformer au training.

**Q3** — Quelle est l'innovation cle de Mamba par rapport a S4 ?
> La selectivite : B, C et le pas de discretisation Delta dependent du token courant (Linear(x_t)). Le SSM peut alors "decider" d'absorber ou ignorer un token, ce que S4 ne pouvait pas faire. C'est l'equivalent fonctionnel d'un gate dynamique.

**Q4** — Pourquoi Mamba est connu pour echouer sur le recall associatif ?
> L'etat cache h_t a une taille fixe (16-64 dims). Tout l'historique y est compresse de facon lossy. Pour repondre "what was the password X gave 50k tokens ago", Mamba doit avoir conserve cette info dans h_t — ce qui est statistiquement difficile en presence de bruit. Un transformer y accede directement via la KV-cache. Benchmark de reference : MQAR.

**Q5** — Pourquoi Jamba garde 1 layer d'attention sur 8 et pas zero ?
> Les rares layers d'attention sauvent le recall ponctuel (associatif dense) que Mamba seul rate. Les 7 autres layers Mamba font le gros du travail en O(N). On obtient ~95% des benefices Mamba (memoire, throughput, long context) tout en gardant la qualite recall d'un transformer. Le ratio empirique "1 attention pour 7 Mamba dans un bloc de 8 layers" vient de l'experimentation interne d'AI21.

---

## Sources

- Gu, Dao, Ermon, Rudra, Ré (2020) — *HiPPO: Recurrent Memory with Optimal Polynomial Projections*. https://arxiv.org/abs/2008.07669
- Gu, Goel, Ré (2021) — *Efficiently Modeling Long Sequences with Structured State Spaces* (S4). https://arxiv.org/abs/2111.00396
- Gu, Dao (2023) — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. https://arxiv.org/abs/2312.00752
- Dao, Gu (2024) — *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2). https://arxiv.org/abs/2405.21060
- Lieber et al., AI21 (2024) — *Jamba: A Hybrid Transformer-Mamba Language Model*. https://arxiv.org/abs/2403.19887
- Peng et al. (2023) — *RWKV: Reinventing RNNs for the Transformer Era*. https://arxiv.org/abs/2305.13048


---

## Pour aller plus loin

Lectures couvrant ce sujet (playlists dans [`shared/external-courses.md`](../../../shared/external-courses.md)) :

- **Stanford CS25 V6 — Lec. 1 (Tradeoffs of State Space Models and Transformers)** — comparatif rigoureux SSM vs attention, par les auteurs Mamba.
