# Jour 19 — Production inference serving : vLLM, SGLang, batching, speculative

> **Temps estime** : 5h | **Prerequis** : J11 (inference optimisee, KV cache)

---

## 1. Pourquoi ce chapitre existe

J11 couvrait les techniques **d'algorithme** (KV cache, Flash Attention, quantization). J19 couvre les techniques **de systeme** : comment servir mille utilisateurs simultanes avec un GPU, ou comment reduire la latence user-facing de 3x sans toucher au modele.

En 2026, deployer un LLM open source (Llama 4, Qwen 3, DeepSeek V3) localement est redevenu strategique :
- Souverainete donnees (France, UE, defense — Masa Group inclus)
- Couts a grande echelle (> 10k req/jour, l'open source bat le pay-per-token)
- Latence (moins de hops reseau)
- Customisation (fine-tuning, prompt caching agressif)

Maitriser **vLLM**, **SGLang**, **TensorRT-LLM** et leurs equivalents est devenu aussi important que maitriser Transformers.

---

## 2. Le bottleneck fondamental : memory-bound decoding

### La realite physique

Un H100 a :
- ~3 TB/s de bande passante memoire (HBM3)
- ~2 TFlops FP16
- Ratio flops / bytes ≈ 600 (operations par byte lu)

Pour un Llama 3 70B en FP16 : ~140GB de poids. Pour decoder **UN seul token**, le GPU doit lire ces 140GB. Temps minimum = 140 GB / 3 TB/s ≈ 47ms → max 21 tokens/seconde par utilisateur sur un seul GPU si aucune astuce n'est employee.

Le compute utilise pendant cette lecture est **negligeable** (~1% du GPU). D'ou **memory-bound** : on est limite par la bande passante memoire, pas par les flops.

### L'insight fondamental : batching

Si on decode **64 utilisateurs simultanement**, on lit les memes 140GB une seule fois, mais on fait le compute pour 64 tokens. On reste memory-bound, mais on serve 64x plus d'utilisateurs par seconde. **Throughput x64 pour une latence a peine superieure.**

```
Batch 1  : 20 tokens/s/user, 20 tokens/s throughput
Batch 8  : 18 tokens/s/user, 144 tokens/s throughput
Batch 32 : 15 tokens/s/user, 480 tokens/s throughput
Batch 128: 8 tokens/s/user, 1024 tokens/s throughput  (saturation compute)
```

**Implication** : la bonne taille de batch depend du SLA latence user-facing. Chat interactif = batch 8-16. Offline processing = batch 64-256.

---

## 3. Continuous batching (vLLM, SGLang, TGI)

### Le probleme du static batching

Dans le serving naif (HuggingFace `generate()`), on batch N requests ensemble. Mais :
- Les sequences ont des longueurs differentes (100 vs 2000 tokens)
- Les **courtes** finissent vite et attendent les **longues** → GPU idle
- Une nouvelle requete qui arrive attend le batch complet finisse

### L'idee : continuous batching (paper "Orca", 2022, implemente par vLLM en 2023)

A chaque iteration de decodage, on peut :
- Injecter de nouvelles requetes dans le batch (elles demarrent leur prefill)
- Retirer celles qui ont fini
- Continuer les autres

```
Step 1: batch = [A decode, B decode, C decode]
Step 2: batch = [A decode, B decode, C decode, D prefill]  ← D inject
Step 3: batch = [A decode, B decode, D decode]  ← C fini, retire
Step 4: batch = [A decode, B decode, D decode, E prefill]  ← E inject
```

Gain mesure : **throughput x2-5** vs static batching, a qualite egale, meme modele, meme hardware.

### Tradeoff prefill/decode dans un meme batch

Le prefill (ex 500 tokens d'un coup) est **compute-bound**. Le decode (1 token nouveau) est **memory-bound**. Les mixer dans un batch fait que :
- Pendant le prefill, les decodes sont ralentis
- L'utilisation GPU est meilleure, mais le latence inter-token varie

**Chunked prefill** (SGLang, vLLM 0.5+) : couper un gros prefill en chunks de 512 tokens, intercaler avec les decodes. Latence decode stable, throughput prefill presque egal.

---

## 4. PagedAttention — gerer le KV cache comme de la RAM

Le KV cache de chaque sequence peut faire 1-100MB. Pour 64 sequences concurrentes : 1-6GB. Mais les longueurs sont **dynamiques** — on ne sait pas a l'avance.

### Le probleme de la fragmentation

Allouer un KV cache continu par sequence :
- Surallocation si on reserve la taille max (gaspillage)
- Realloc + copie si on agrandit en cours (latence spike)
- Fragmentation a la Tetris : impossible de placer une nouvelle sequence meme avec assez de memoire totale

### PagedAttention (Kwon et al., vLLM, 2023)

Inspiration : la memoire virtuelle de l'OS. Decouper le KV cache en **pages fixes** (ex: 16 tokens par bloc). Chaque sequence a une **table de pages** qui pointe vers des blocs physiques non contigus.

```
Sequence A : [page_5] [page_12] [page_7] [page_20]  (tokens 0-63)
Sequence B : [page_3] [page_9]                       (tokens 0-31)
Sequence C : [page_15] [page_11] [page_18]           (tokens 0-47)
```

Gain : **fragmentation < 5%** (vs 30-60% sans paging). On peut soutenir 2-4x plus de sequences simultanees.

Bonus : partage de KV cache entre sequences (prefix caching) devient trivial — pointer aux memes pages.

---

## 5. Prefix caching (a la vLLM)

Si deux requetes partagent le meme system prompt, au lieu de re-calculer les K/V du prefixe, on reutilise les pages deja en GPU memory.

```
System prompt : 8000 tokens → 500 pages (block 16)
User A : [system_pages] + [user_A_pages]
User B : [system_pages] + [user_B_pages]   ← memes system_pages
```

Difference avec le **prompt caching API** (J18) :
- vLLM prefix caching = **cote serveur**, au niveau KV cache physique, automatique
- API prompt caching = **cote provider**, avec TTL, facture differemment

Les deux **s'additionnent** : un client peut beneficier des deux niveaux.

---

## 6. Speculative decoding

### L'idee en une phrase

Faire tourner un **petit modele** (draft) qui propose K tokens d'un coup. Le **gros modele** (target) les verifie tous en parallele (un seul forward pass). Si les K tokens du draft sont corrects → K tokens generes pour le prix d'un prefill. Sinon → fallback sur le gros modele.

### Pourquoi ca marche

Le gros modele est memory-bound en decode. Verifier K tokens = lire les poids une fois. Donc verifier 5 tokens = meme cout que generer 1 token. Si le draft a 70% raison en moyenne, on genere 1 + 0.7 × 4 = ~3.8 tokens par forward. **Speedup x3-4**.

### Variantes 2025-2026

- **Classic spec decoding** : draft model = 7B, target = 70B
- **Medusa** (2024) : plusieurs heads dans le meme modele predisent en parallele, pas besoin de draft separe
- **EAGLE** (2024-2025) : draft via une petite tete transformer sur les hidden states du target. SOTA vitesse/qualite
- **N-gram speculation** : pour des cas deterministes (code, JSON), matche des n-grams du contexte comme draft, gratuit

En prod : vLLM supporte EAGLE et ngram. Medusa chez TGI. Gain typique : x2-3 sur decode, aucun impact qualite (verifie).

---

## 7. Quantization en production

J11 a presente les principes (INT8, INT4). J19 couvre les choix **de deploiement** :

### Les formats gagnants 2026

| Format | Qualite | Speed up | Support |
|---|---|---|---|
| FP16 / BF16 | 100% reference | 1x | partout |
| FP8 (H100+, Blackwell) | 99% | 1.5-2x | H100, B200, SXM5 |
| INT8 (W8A8 SmoothQuant) | 98% | 1.5-2x | partout |
| INT4 (AWQ, GPTQ) | 95-98% | 2-3x | partout |
| INT4 (W4A16) | 96% | 1.5-2x | defaut Qwen3/Llama4 releases |
| 1.58-bit (BitNet) | 90% | 4-8x | experimental 2025-2026 |

**Regle pratique** :
- Serving a large echelle + H100+ → FP8 (best quality/speed, natif)
- Serving sur GPU commodity (A100, L40, 4090) → INT4 AWQ
- Edge (Mac M3, 3090, serveur legacy) → INT4 GGUF (llama.cpp)

### Les pieges

- Benchmark "perplexity" egal en quantized vs fp16 ≠ qualite egale en downstream task. Toujours evaluer end-to-end sur tes vraies taches.
- Reasoning models sont plus fragiles a la quantization (chaines longues amplifient les erreurs). INT4 peut chuter de 5-15% sur AIME/MATH. FP8 est safer.
- La calibration data pour AWQ/GPTQ importe. 512 samples representatifs du domaine > 10k samples generiques.

---

## 8. Disaggregated serving (2024-2026 frontier)

Idee : separer les GPUs qui font le **prefill** de ceux qui font le **decode**. Raisons :
- Le prefill est compute-bound → GPUs avec max flops (ex: B200 fp8)
- Le decode est memory-bound → GPUs avec max bande passante / cost
- Ils ont des patterns d'utilisation differents

Architecture :
```
Prefill pool (2x B200) → KV cache transfer (RDMA) → Decode pool (8x H100)
```

Open-source : **SGLang** (2024, Berkeley) a popularise cette architecture. **DeepSeek V3** est deploye ainsi dans leurs datacenters (publie dans le paper). Gain : **+50% throughput** pour un cout hardware equivalent.

Pour un AI engineer en 2026, ce n'est pertinent qu'au-dela de ~100k requests/h. En dessous, un pool unique suffit.

---

## 9. Choix de stack pour 2026

| Besoin | Stack recommande |
|---|---|
| Prototype / dev | transformers + accelerate |
| Petite prod self-hosted | **vLLM** (le plus utilise, stable, large support) |
| Latence minimum | **SGLang** (prefix caching agressif, structured outputs) |
| NVIDIA max perf | **TensorRT-LLM** (compile, moins flexible) |
| Batch offline massif | **vLLM** offline mode ou **llama.cpp** multi-GPU |
| Edge / CPU / Mac | **llama.cpp** / **ollama** / **mlx** |
| Multi-modele dynamique | **LMDeploy** ou SGLang router |

vLLM est de facto le defaut 2026 pour "serving un LLM open-source en prod Kubernetes". Pour optimisations plus agressives (structured outputs, prompt caching visible, multi-model routing), SGLang.

---

## 10. Les metriques que tu dois monitorer

### Les 5 metriques SLO d'un endpoint LLM

1. **TTFT** (time to first token) — principalement lie au prefill. Target < 500ms pour chat, < 3s pour agents.
2. **Inter-token latency** — temps entre tokens. Target 30-50ms pour chat reactif.
3. **Tokens/s/user** (decode throughput perceptible) — > 20 tok/s pour un chat usable, > 50 pour excellent.
4. **Tokens/s/GPU** (throughput server, tous users) — mesure l'efficacite du batching.
5. **KV cache utilization** — taux d'occupation memoire. > 80% = risque eviction, < 40% = oversized hardware.

### Bonus

- **Preemption rate** : fraction de sequences preemptees par eviction KV cache. Si > 5%, augmenter memoire ou reduire max context.
- **Speculative acceptance rate** : pourcentage de tokens du draft acceptes. Cible > 0.6 ; si < 0.4, desactiver (overhead net).

---

## Key takeaways (flashcards)

**Q1** — Pourquoi le decode d'un LLM est-il memory-bound ?
> Pour generer 1 token, il faut lire tous les poids du modele depuis la HBM. Le compute est minuscule vs la lecture. Batcher N sequences amortit la lecture sur N sorties.

**Q2** — Qu'est-ce que continuous batching ?
> Ajouter/retirer des sequences du batch a chaque iteration de decodage au lieu d'attendre le batch entier. Throughput x2-5 vs static batching.

**Q3** — Que resout PagedAttention ?
> La fragmentation memoire du KV cache. Decoupe en pages fixes (16 tokens), chaque sequence a une table de pages vers des blocs non contigus. Supporte 2-4x plus de sequences concurrentes.

**Q4** — Quel speedup attendre de speculative decoding ?
> x2-4 sur le decode, sans perte de qualite (verification exacte). Le draft doit avoir > 60% d'acceptance rate, sinon c'est un net negatif.

**Q5** — FP8 vs INT4 : quel choix en 2026 ?
> FP8 sur H100+ (natif, +99% qualite, x1.5-2). INT4 sur GPU commodity (x2-3 memoire, qualite 95-98%). Reasoning models preferent FP8 (chaines longues sensibles a l'erreur).

**Q6** — Qu'est-ce que disaggregated serving ?
> Separer physiquement les GPUs qui font prefill (compute-bound) des GPUs qui font decode (memory-bound). +50% throughput vs pool unique. Pertinent > 100k req/h.
