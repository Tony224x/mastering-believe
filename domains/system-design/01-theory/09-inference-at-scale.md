# Jour 9 — Inference at Scale

## Pourquoi l'inference est un probleme de systems, pas de ML

**Exemple d'abord** : Tu deploies Llama-3-70B sur une A100 80Go. Requete par requete, tu sers 1 user a la fois, 800 ms par reponse. Throughput : ~1.25 req/s. Tu utilises 15% du GPU. Avec le **continuous batching** de vLLM, la meme machine sert ~50 users en parallele, p99 latency 1500 ms, throughput ~40 req/s, 90% d'utilisation GPU. Gain : **32x**.

C'est toute la problematique de l'inference at scale. Le modele est deja entraine, le code est simple. Mais faire passer un GPU de 15% a 95% d'utilisation demande une ingenierie system non triviale : scheduling, batching, gestion memoire, quantization, autoscaling, routing.

**Key takeaway** : Un GPU a 15% d'utilisation coute 100% du prix. L'optimisation de l'inference est directement proportionnelle a la facture cloud. Un facteur 5x sur le throughput = un facteur 5x sur le cout par requete.

---

## Les metriques qui comptent

| Metrique | Definition | Cible typique |
|---|---|---|
| **Latence p50** | 50% des requetes plus rapides | Modeles classiques : < 50 ms. LLM : < 1 s |
| **Latence p99** | 99% des requetes plus rapides | 3-5x le p50 max |
| **Throughput** | Requetes servies par seconde | Depend du modele et du GPU |
| **TTFT** (time to first token) | Latence avant le 1er token (LLM) | < 500 ms pour UX fluide |
| **TPOT** (time per output token) | Temps entre 2 tokens streames | < 50 ms = experience humaine |
| **GPU utilization** | % du temps ou le GPU calcule | Cible : > 80% |
| **Cost per 1K tokens** | $ / 1K tokens servis | Directement dicte par le throughput |

**Piege frequent** : optimiser seulement la latence p50 alors que p99 est 10x plus haut. Un p99 degrade = 1% des users qui attendent 10 s = reviews 1 etoile.

---

## Frameworks de serving

### Modeles classiques (torch / tf / sklearn)

| Framework | Force | Faiblesse |
|---|---|---|
| **TorchServe** | Natif PyTorch, simple | Pas le plus performant |
| **Triton Inference Server** (NVIDIA) | Multi-framework, multi-GPU, dynamic batching, modele d'ensemble | Courbe d'apprentissage |
| **TensorFlow Serving** | Eprouve sur TF | Specifique TF |
| **KServe** (Kubernetes) | Orchestration native K8s, autoscaling | Complexite K8s |
| **BentoML** | Bon DX Python, packaging facile | Moins optimise que Triton a grande echelle |

### LLMs

| Framework | Force | Quand l'utiliser |
|---|---|---|
| **vLLM** | PagedAttention, continuous batching, le plus rapide en 2025 | Standard pour Llama/Mistral/Qwen |
| **TGI** (Text Generation Inference, HuggingFace) | Integration ecosystem HF, streaming | Bon defaut si tu es deja sur HF |
| **TensorRT-LLM** (NVIDIA) | Le plus optimise par GPU, graph compilation | Prod lourde, equipe experte |
| **SGLang** | Radix attention, excellent pour des prompts avec beaucoup de prefixes communs | Cas d'usage specifique (few-shot, agents) |
| **llama.cpp / Ollama** | CPU / Apple Silicon, petit | Edge, local, demos |

---

## Le probleme central : batching

Un GPU est optimise pour le calcul parallele de grosses matrices. Faire une requete a la fois = gacher 99% du silicium. Il faut grouper les requetes pour que le GPU travaille sur plusieurs en meme temps.

### Static batching (la mauvaise idee)

Tu fixes `batch_size=32`, tu attends que 32 requetes arrivent, tu les traites, tu renvoies.

**Problemes** :
- Si seulement 5 requetes arrivent, elles attendent indefiniment OU tu les envoies a moitie batch -> gaspillage.
- Pour les LLMs, les sequences ont des longueurs differentes. Si une seq prend 1000 tokens et 31 autres 50 tokens, tu attends la plus longue. Le GPU tourne au ralenti sur 31 sur 32 slots.

### Dynamic batching

Tu attends les requetes dans une queue. Tu flushes quand :
- **batch_size atteint** (ex: 32)
- **OU max_wait depasse** (ex: 50 ms)

```
  t=0   t=10ms   t=20ms   t=30ms   t=40ms   t=50ms
   │      │        │        │        │        │
 req1    req2                                  FLUSH (3 req, timeout 50ms)
          │        req3
                                                processing...
```

Bon compromis : limite la latence max, remplit mieux le batch. Standard dans Triton.

### Continuous batching (pour les LLMs)

Innovation de vLLM / TGI : au lieu d'attendre que toutes les seq du batch soient finies, on **remplace** une seq finie par une nouvelle dans le meme batch.

```
  Batch slot :  1   2   3   4
  t=0 :        A   B   C   D      (4 sequences in)
  t=1 :        A   B   C   D
  ...
  t=n :        A   -   C   D      (B finished)
  t=n+1 :      A   E   C   D      (E started in the free slot)
```

Resultat : le GPU est en permanence a 100% d'occupation. C'est ce qui fait passer Llama-70B de 1 req/s a 40 req/s.

**Pre-requis** : paged attention (gestion memoire KV cache non-contigue) -> c'est la contribution de vLLM.

---

## Le KV cache : ami et tyran

Pour generer un token, un LLM doit calculer l'attention sur tous les tokens precedents. Pour eviter de recalculer a chaque step, on cache les `key` et `value` de chaque couche. C'est le **KV cache**.

### Couts memoire

Pour Llama-3-70B, 1 token de KV cache consomme environ 320 Ko. Pour une sequence de 4000 tokens : **1.25 Go par sequence**. Avec un contexte de 32K tokens et un batch de 16 : **160 Go** rien que pour le cache -> depasse la memoire GPU.

### Pourquoi PagedAttention ?

Les implementations naives allouent le KV cache de maniere contigue (une zone par sequence). Mal : quand une seq est courte, la memoire reservee est gachee. Quand une seq est longue, il faut realloc -> latence.

**PagedAttention** (vLLM) alloue le KV cache par **pages** (blocs de 16 tokens) qui n'ont pas besoin d'etre contigues en memoire. Une table de pages fait le mapping. Gain : 2-4x plus de sequences simultanees par GPU.

### Cache prefix sharing

Si plusieurs requetes partagent le meme prompt system, on peut partager les KV entries. Exemple : toutes les requetes d'un agent ont le meme systeme de 2000 tokens. Au lieu de recomputer pour chaque requete, on calcule une fois et on partage.

---

## Quantization : diviser la memoire et accelerer

Un modele Llama-70B en fp16 pese **140 Go**. Meme sur une H100 80Go, il ne rentre pas. Solution : la **quantization**, qui reduit la precision des poids.

| Format | Taille Llama-70B | Perte qualite | Vitesse |
|---|---|---|---|
| **fp32** (reference) | 280 Go | 0 | 1x |
| **fp16 / bf16** (defaut moderne) | 140 Go | 0 (pratique) | 2x |
| **int8** (GPTQ, AWQ) | 70 Go | -1 a -2% benchmarks | 2-3x |
| **int4** (GPTQ, AWQ) | 35 Go | -3 a -5% benchmarks | 3-4x |
| **int4 + kv cache int8** | 30 Go | -5% | 4-5x |

**Tradeoffs** :
- int8 : perte negligeable, gain memoire 2x, quasi standard
- int4 : bon pour les modeles open source, support variable selon le hardware
- fp8 : supporte sur H100+, tres bon compromis

**Regle** : teste sur tes propres benchmarks metier, pas MMLU. Un modele int4 peut performer identiquement sur ton use case mais exploser sur un benchmark general.

---

## Autoscaling pour l'inference

### Scale-up : plus de GPUs

Le GPU est une ressource chere et lente a demarrer (pulls d'images plusieurs Go, warmup du modele 30-60s). Un autoscaling naif sur CPU metric ne marche pas.

**Metriques utiles pour trigger le scale-up** :
- **Queue depth** : nombre de requetes en attente dans la queue de batching
- **p99 latency** : si la latence degrade -> saturation
- **GPU utilization** : > 80% soutenu

### Scale-to-zero

Supprimer les GPUs quand pas de trafic. Probleme : cold start 30-60 s. Inacceptable pour un user en attente.

**Mitigations** :
- Garder 1 replica "hot" en permanence, scale au-dessus a la demande
- **Model loading depuis shared storage** (S3 + tmpfs) pour reduire le cold start
- **Snapshotting** (CUDA graph capture + serialisation) pour demarrer en 5-10 s

### Scale horizontal vs vertical

Pour un LLM 70B+ : un seul GPU ne suffit pas -> **tensor parallelism** (decouper les poids entre 2-8 GPUs) ou **pipeline parallelism** (decouper les couches). vLLM et TGI supportent les deux.

---

## Optimisations 2025-2026 : au-dela du continuous batching

Le continuous batching + PagedAttention reste la fondation, mais l'etat de l'art 2025-2026 empile par-dessus 5 techniques qui font encore gagner un facteur 2-5x sur le cout et la latence. Les implementer correctement est desormais la difference entre un serving "bon" et un serving competitif.

### Speculative decoding

Un petit modele **draft** (ex: Llama-3.2-1B) propose N tokens en quelques ms. Le gros modele cible (ex: Llama-3-70B) les **verifie en une seule passe forward**. Si les tokens matchent, on a gagne N-1 forwards. Si un token diverge, on repart de la.

- **Gain typique** : 1.5-3x sur le TPOT, jusqu'a 4x sur des prompts tres previsibles (code, generation structuree).
- **Techniques avancees** : **EAGLE** (draft integre au modele cible, reutilise ses features) et **Medusa** (plusieurs tetes de prediction paralleles sur le meme modele). Plus complexe a deployer mais plus robuste.
- **Support** : vLLM, TensorRT-LLM et SGLang proposent tous du spec decoding out-of-the-box depuis 2024.

### Prefix caching (different du KV cache classique)

Le KV cache classique vit pendant **une seule requete**. Le **prefix caching** reutilise le KV cache d'un prefixe identique **entre requetes**. Exemple : un system prompt de 3000 tokens partage par 1M requetes/jour ne se calcule plus qu'une fois, puis est reference par pointeur.

- **Anthropic** le fait via `cache_control` explicite dans l'API Messages (ephemeral, 5 min).
- **OpenAI** le fait automatiquement depuis fin 2024 sur les prompts > 1024 tokens.
- **vLLM** l'implemente via l'option `enable_prefix_caching=True` (radix tree sur les KV blocks).
- **Impact cout** : jusqu'a 90% de reduction sur les tokens de system prompt longs (les tokens caches sont factures 10% du prix normal chez Anthropic).

### Chunked prefill

Le **prefill** (traiter le prompt) est compute-bound et bursty. Le **decode** (generer les tokens) est memory-bound et continu. Les melanger naivement cree des **stalls** : quand un gros prefill arrive, le decode s'arrete.

**Solution** : decouper le prefill en chunks (ex: 512 tokens) et les interleaver avec les steps de decode dans le meme batch. Resultat : pas de stall, GPU utilise en permanence, p99 TTFT divise par 2-3. Standard dans vLLM 0.5+ via `--enable-chunked-prefill`.

### Disaggregated serving (prefill/decode split)

Pattern devenu dominant chez OpenAI, Anthropic et Meta en 2025 : **separer physiquement** les GPUs qui font prefill de ceux qui font decode.

- **Prefill GPUs** : compute-heavy, batch-friendly, on maximise le throughput (TensorRT-LLM optimise, batch large).
- **Decode GPUs** : memory-heavy, latency-sensitive, on maximise le KV cache et le nombre de sequences simultanees.
- **Transfer** : le KV cache de la phase prefill est transmis aux decode GPUs via NVLink ou RDMA.

**Pourquoi ca marche** : les deux phases ont des profils de ressources opposes. Les faire cohabiter sur le meme GPU sacrifie l'une pour l'autre. Les separer permet d'optimiser independamment et de scale asymetriquement (ex: 2 prefill GPUs pour 8 decode GPUs). Papers de reference : **DistServe** (2024) et **Splitwise** (Microsoft 2024).

### Semantic routing

Ne pas envoyer toutes les queries au gros modele. Un **routeur leger** (classifier ou matching d'embeddings) dispatche vers le bon tier :

- Requete triviale ("bonjour", "merci") -> Haiku / gpt-5.4-nano
- Q&A factuelle courte -> Sonnet / gpt-5.4-mini
- Reasoning complexe -> Opus / gpt-5.4

**Outils** : **LiteLLM router**, **Portkey**, **OpenRouter**. Gain typique : 40-70% de cout en moins pour une qualite percue equivalente, a condition que le routeur soit bien calibre sur les vraies distributions de trafic. Voir J11 pour le detail.

**Key takeaway** : en 2026, un serving LLM "standard" empile tout : continuous batching + PagedAttention + prefix caching + chunked prefill + speculative decoding + disaggregated serving + semantic routing. Chaque technique est un levier multiplicatif, et l'absence d'une seule peut couter 2-5x sur la facture cloud.

---

## Pattern architectural : queue + workers

```
  clients ──> API gateway ──> ingress queue (Redis / SQS) ──> batcher ──> model workers (GPUs)
                                                                  │                │
                                                                  │                v
                                                                  └──── response queue ──> client
```

Avantages :
- Decouple le rate client du rate GPU
- Permet le batching dynamique
- Resilience : si un worker crash, les requetes sont rejouees
- Observability : on mesure la taille de la queue -> trigger autoscaling

---

## Exemples reels

- **OpenAI** : racks de H100 + TensorRT-LLM custom + routing hierarchique
- **Anthropic** : infra proprietaire, heavy continuous batching, caching agressif
- **Together AI, Replicate, Fireworks** : vLLM / SGLang sur H100, multi-tenancy, facturation au token
- **HuggingFace Inference Endpoints** : TGI par defaut, deploiement 1-click
- **Modal, Baseten** : platforms d'inference avec scale-to-zero et model caching

---

## Flash cards

**Q: Pourquoi le static batching est une mauvaise idee pour les LLMs ?**
R: Les sequences ont des longueurs differentes. Tu attends la plus longue, le GPU est sous-utilise 90% du temps.

**Q: Qu'est-ce que le continuous batching ?**
R: Une technique ou, des qu'une sequence finit dans le batch, une nouvelle la remplace immediatement. Le GPU reste a 100% d'occupation.

**Q: Quelle est la contribution cle de vLLM ?**
R: PagedAttention : gestion du KV cache par pages non-contigues. Permet 2-4x plus de sequences simultanees.

**Q: Quantization int4 : combien de perte qualite ?**
R: Environ 3-5% sur les benchmarks generaux. A tester sur tes propres donnees, parfois nulle pour le use case specifique.

**Q: Quelle metrique trigger le scale-up d'un pool GPU ?**
R: Queue depth + p99 latency. Jamais le CPU seul. GPU utilization > 80% soutenu est aussi un bon signal.

---

## Key takeaways

- L'inference n'est pas du ML : c'est du systems engineering. Le scheduling et le batching dominent la performance.
- Continuous batching + PagedAttention (vLLM) = 10-30x plus de throughput qu'une implementation naive.
- KV cache est la ressource critique des LLMs. Toute optimisation memoire (int8 KV, sharing) paye double.
- Quantization n'est plus optionnelle : int8 est quasi gratuit, int4 est le standard pour les open-source en prod.
- Autoscaling : metrics = queue depth + p99 latency. Cold start est l'anomalie principal.
