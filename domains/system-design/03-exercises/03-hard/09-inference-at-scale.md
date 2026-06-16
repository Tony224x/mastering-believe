# Exercices Hard — Inference at Scale

---

## Exercice 1 : Dimensionner et architecturer le serving LLM d'un produit a 100K req/s

### Objectif
Concevoir l'infra d'inference LLM complete d'un produit a grande echelle : sizing GPU, batching, KV cache, autoscaling, multi-tier routing, et le stack d'optimisations 2025-2026. Design d'entretien "LLM infra".

### Consigne
Tu designs le serving d'un produit LLM grand public :
- 100K req/s en peak, mix de tailles de requetes
- Modele principal : Llama-3-70B (servi en int4/fp8 sur H100 80 Go)
- SLA : TTFT < 500 ms p95, TPOT < 50 ms, p99 latency raisonnable
- Cout : la facture GPU est le poste #1, a minimiser
- 60% du trafic est trivial, 30% mid, 10% complexe

**Livre :**

1. **Sizing GPU** :
   - Une H100 en int4 + continuous batching sert ~X sequences simultanees (KV cache limitant). Estime X pour des sequences de 4000 tokens (KV ~320 Ko/token, ~35 Go de room apres poids+overhead).
   - Pour 100K req/s avec des requetes de ~10 s en moyenne, combien de requetes "in-flight" simultanees ? Combien de H100 au minimum ?

2. **Batching** :
   - Pourquoi static batching est exclu ? Pourquoi continuous batching + PagedAttention est la fondation ?
   - Quel est le pre-requis memoire de PagedAttention et qu'est-ce qu'il economise ?

3. **KV cache & memoire** :
   - Le KV cache est la ressource critique : comment le reduis-tu (int8 KV, prefix sharing) ?
   - Prefix caching inter-requetes : sur un system prompt de 3000 tokens partage par 1M req/jour, quel gain ?

4. **Stack d'optimisations 2025-2026** :
   - Empile et explique : continuous batching, PagedAttention, prefix caching, chunked prefill, speculative decoding, disaggregated serving (prefill/decode split), semantic routing.
   - Pour chacun, le levier (latence ? throughput ? cout ?).

5. **Routing multi-tier** :
   - Avec 60/30/10, route vers nano / mid / frontier. Estime l'economie de cout vs tout-frontier.
   - Quel est le risque d'un routeur mal calibre et quelle metrique le detecte ?

6. **Autoscaling & resilience** :
   - Quelle metrique trigger le scale-up (pas le CPU) ?
   - Cold start GPU 30-60 s : comment tu le geres (hot replica, snapshotting, scale-to-zero) ?
   - Pattern queue + workers : quels avantages (decouplage, batching, observability) ?

### Criteres de reussite
- [ ] Sizing GPU : ~27 sequences/H100 en int4 (KV limitant) ; in-flight ≈ req/s * duree moyenne (100K * 10 s = 1M in-flight) -> ordre de grandeur de milliers de H100, sizing assume + headroom
- [ ] Static batching exclu (seq de longueurs inegales -> GPU sous-utilise) ; continuous batching + PagedAttention = fondation
- [ ] PagedAttention : KV pagine non-contigu (16-token blocks) -> moins de gaspillage, plus de seq simultanees
- [ ] Reduction KV : int8 KV (~2x), prefix sharing ; prefix caching inter-requetes -> jusqu'a 90% sur les tokens de system prompt longs
- [ ] Les 7 optimisations sont empilees et chacune a son levier (TTFT, TPOT, throughput, cout) correctement attribue
- [ ] Routing multi-tier : economie 40-70% vs tout-frontier ; routeur mal calibre detecte par taux d'escalade/fallback
- [ ] Autoscaling sur queue depth + p99 (jamais CPU seul) ; cold start gere (hot replica + snapshotting/CUDA graph) ; pattern queue+workers justifie

---

## Exercice 2 : Post-mortem — le serving LLM ou un GPU a 15% facturait 100%

### Objectif
Analyser un incident d'inference mal optimisee (cout + latence) et concevoir le serving competitif. La difference entre "ca marche" et "ca tient l'echelle".

### Consigne
Voici le rapport d'incident (resume) du serving LLM d'une startup.

**Contexte** : Llama-3-70B servi en fp16 (140 Go) sur 2 H100 en tensor parallelism, **static batching** (batch fixe = 8), pas de continuous batching, pas de prefix caching, pas de routing (tout va au 70B), autoscaling sur **CPU utilization**. La facture cloud explose, la latence p99 est mauvaise, le GPU est a 15-20% d'utilisation.

**Timeline / symptomes :**

| Symptome | Detail |
|---|---|
| GPU utilization | 15-20% en moyenne (le GPU attend des batchs complets / des seq longues). |
| Throughput | ~1.25 req/s par paire de GPU. |
| p99 latency | 8 s (les requetes courtes attendent la plus longue du batch). |
| Cout | $/req tres eleve, marge negative. |
| Autoscaling | Ne scale jamais (CPU reste bas car le bottleneck est le GPU) -> queue qui deborde en peak. |
| System prompt | 3000 tokens identiques recomputes a chaque requete. |
| Trafic | 70% des requetes sont triviales mais vont quand meme au 70B. |

**Questions :**

1. **Root cause analysis** :
   - Pour chaque symptome, la cause technique et l'optimisation manquante. Classe : batching, memoire/KV, routing, autoscaling.
   - Pourquoi un GPU a 15% coute 100% du prix ?

2. **Le chiffre du gain** :
   - Static batching (batch 8, seq inegales) vs continuous batching : explique le facteur ~32x (de 1.25 req/s a ~40 req/s). Quel mecanisme ?
   - Quantization fp16 (140 Go) vs int4 (35 Go) : qu'est-ce que ca debloque cote KV cache / batch ?

3. **Autoscaling casse** :
   - Pourquoi scaler sur le CPU ne marche jamais pour de l'inference GPU ?
   - Quelles 2 metriques utiliser a la place ?

4. **Prefix caching** :
   - Le system prompt de 3000 tokens est recompute a chaque requete. Quel gain avec le prefix caching, et comment ca marche (KV reutilise entre requetes) ?

5. **Routing** :
   - 70% du trafic est trivial mais va au 70B. Estime l'economie d'un routing vers un petit modele pour ces 70%.

6. **Serving corrige** :
   - Liste le stack complet a deployer (continuous batching, PagedAttention, int4, prefix caching, chunked prefill, spec decoding, routing, autoscaling sur queue depth).
   - Donne l'ordre de priorite (quelle optim deployer en PREMIER pour le plus gros gain immediat).

### Criteres de reussite
- [ ] Chaque symptome relie a sa cause + optim manquante, classe (batching : static -> continuous ; memoire : fp16 -> int4 + KV ; routing : tout-70B -> tiers ; autoscaling : CPU -> queue depth)
- [ ] "GPU a 15% = 100% du prix" explique : on paie le GPU a l'heure quel que soit son taux d'utilisation
- [ ] Le facteur ~32x est explique par le continuous batching (slots remplis en continu, plus d'attente de la seq la plus longue) ; int4 debloque du KV cache -> plus grand batch
- [ ] Autoscaling : le CPU reste bas car le bottleneck est le GPU -> scaler sur queue depth + p99 latency
- [ ] Prefix caching : le KV du system prompt est calcule une fois et reutilise entre requetes -> jusqu'a 90% d'economie sur ces tokens
- [ ] Routing : 70% vers un petit modele (~5x moins cher) -> economie majeure sur le cout LLM brut
- [ ] Stack corrige complet + ordre de priorite credible (continuous batching/PagedAttention en premier = plus gros gain throughput, puis quantization, puis prefix caching + routing)
