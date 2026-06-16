# REFERENCES — `system-design` (14 jours)

> Sources de tier-1 par module pour le domaine **System Design — Architecture Backend & IA**.
> Source de verite pour les modules J1..J14 (Semaine 1 : fondations system design ; Semaine 2 : architecture IA & production ML).
> Titres, auteurs, annees et identifiants arXiv verifies via WebSearch/WebFetch (juin 2026). Les rares items non confirmes sont marques **(a verifier)**.
> Format des puces : `Auteurs (annee). *Titre*. Venue/arXiv:id — note 1 ligne.`

---

## Module 01 — Principes fondamentaux

- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly — chap. 1-2 (reliability, scalability, maintainability) et chap. 9 (consistency/consensus, CAP) : socle theorique de tout le module.
- Gilbert, S. & Lynch, N. (2002). *Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services*. ACM SIGACT News — la preuve formelle du theoreme CAP (souvent mal cite).
- Brewer, E. (2012). *CAP Twelve Years Later: How the "Rules" Have Changed*. IEEE Computer — l'auteur de CAP nuance lui-meme : C/A n'est pas binaire, le vrai tradeoff est latence vs consistance sous partition.
- Abadi, D. (2012). *Consistency Tradeoffs in Modern Distributed Database System Design (PACELC)*. IEEE Computer — etend CAP : meme sans partition, on arbitre Latency vs Consistency.
- Dean, J. & Barroso, L.A. (2013). *The Tail at Scale*. Communications of the ACM — la reference sur la latence p99/tail et pourquoi elle domine l'experience a l'echelle.

## Module 02 — Stockage & Databases

- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly — chap. 3 (storage engines, B-trees vs LSM), 5 (replication), 6 (partitioning/sharding) : reference centrale du module.
- DeCandia, G. et al. (2007). *Dynamo: Amazon's Highly Available Key-value Store*. SOSP 2007 — consistent hashing, quorums, object versioning ; blueprint de Cassandra/DynamoDB (BASE, leaderless).
- Chang, F. et al. (2006). *Bigtable: A Distributed Storage System for Structured Data*. OSDI 2006 (Best Paper) — modele wide-column, SSTables/LSM ; ancetre de HBase/Cassandra.
- Corbett, J.C. et al. (2012). *Spanner: Google's Globally-Distributed Database*. OSDI 2012 — DB SQL globalement distribuee, transactions externally-consistent via TrueTime ; le contre-exemple "NewSQL" a CAP.
- Lakshman, A. & Malik, P. (2010). *Cassandra: A Decentralized Structured Storage System*. ACM SIGOPS OSR — synthese Dynamo + Bigtable, modele de coherence tunable.

## Module 03 — Caching & CDN

- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly — materialized views, derived data et invalidation : cadre conceptuel du caching.
- Nishtala, R. et al. (2013). *Scaling Memcache at Facebook*. NSDI 2013 — etude de cas canonique : cache-aside a tres grande echelle, thundering herd, leases, invalidation cross-region.
- Redis. *Redis Documentation* (consulte juin 2026). https://redis.io/docs/ — reference officielle (structures de donnees, eviction LRU/LFU, persistence, Redis Cluster).
- Cloudflare. *Learning Center: What is a CDN? / How caching works* (consulte juin 2026). https://www.cloudflare.com/learning/cdn/what-is-a-cdn/ — vulgarisation tier-1 sur edge caching, TTL, cache headers.
- Fielding, R. & Reschke, J. (2014). *HTTP/1.1 Caching*. RFC 7234 — semantique normative de `Cache-Control`, `ETag`, validation conditionnelle (base du browser/CDN cache).

## Module 04 — Message Queues & Event-Driven

- Kreps, J., Narkhede, N. & Rao, J. (2011). *Kafka: a Distributed Messaging System for Log Processing*. NetDB 2011 — le papier fondateur de Kafka (log append-only partitionne, consumer groups).
- Apache Kafka. *Kafka Documentation: Design & Implementation* (consulte juin 2026). https://kafka.apache.org/documentation/#design — reference officielle (partitions, offsets, delivery semantics, exactly-once).
- Fowler, M. (2005, maj 2011). *Event Sourcing*. martinfowler.com. https://martinfowler.com/eaaDev/EventSourcing.html — definition canonique de l'event sourcing.
- Fowler, M. (2011). *CQRS*. martinfowler.com. https://martinfowler.com/bliki/CQRS.html — separation command/query, complement direct de l'event sourcing.
- Richardson, C. (2018). *Microservices Patterns* (chap. Saga). Manning — reference du Saga pattern (orchestration vs choregraphie) ; voir aussi microservices.io/patterns/data/saga.html.

## Module 05 — Load Balancing & Networking

- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly — chap. 8 (the trouble with distributed systems : reseau, timeouts, partitions).
- NGINX. *NGINX Documentation / Admin Guide: HTTP & TCP Load Balancing* (consulte juin 2026). https://docs.nginx.com/nginx/admin-guide/load-balancer/ — reference reverse proxy / L7 vs L4, healthchecks, sticky sessions.
- AWS. *Elastic Load Balancing — ALB vs NLB User Guides* (consulte juin 2026). https://docs.aws.amazon.com/elasticloadbalancing/ — distinction L7 (ALB) vs L4 (NLB) cote cloud, autoscaling.
- Nygard, M. (2018, 2e ed.). *Release It! Design and Deploy Production-Ready Software*. Pragmatic Bookshelf — source canonique des patterns Circuit Breaker, Bulkhead, Timeout, Backpressure.
- Cloudflare. *Rate limiting / DDoS — Learning Center* (consulte juin 2026). https://www.cloudflare.com/learning/ — algorithmes de rate limiting (token bucket, sliding window) expliques tier-1.

## Module 06 — API Design & Patterns

- Fielding, R. (2000). *Architectural Styles and the Design of Network-based Software Architectures* (these de doctorat, UC Irvine) — la definition originale de REST par son auteur.
- gRPC Authors. *gRPC Documentation* (consulte juin 2026). https://grpc.io/docs/ — reference officielle (Protocol Buffers, HTTP/2, streaming, codegen).
- GraphQL Foundation. *GraphQL Specification & Learn* (consulte juin 2026). https://graphql.org/learn/ — reference officielle (schema, resolvers, over/under-fetching).
- Microsoft. *REST API Guidelines* (GitHub, maj continue). https://github.com/microsoft/api-guidelines — guide d'entreprise tier-1 : versioning, pagination, erreurs, idempotency.
- IETF. *Idempotency-Key HTTP Header Field* (draft-ietf-httpapi-idempotency-key-header, en cours). https://datatracker.ietf.org/doc/draft-ietf-httpapi-idempotency-key-header/ — standardisation des cles d'idempotence **(a verifier : draft non finalise mi-2026)**.

## Module 07 — Designs classiques (entretiens)

- Xu, A. (2020). *System Design Interview — An Insider's Guide, Vol. 1*. ByteByteGo — framework d'entretien + designs URL shortener, rate limiter, chat, news feed.
- Xu, A. (2022). *System Design Interview, Vol. 2*. ByteByteGo — designs avances (paiements, proximity service, hotel reservation, metrics monitoring).
- Martin, D. (2018+). *The System Design Primer* (GitHub). https://github.com/donnemartin/system-design-primer — recueil open-source de reference (estimations, back-of-envelope, designs types).
- ByteByteGo. *System Design Newsletter / blog* (consulte juin 2026). https://blog.bytebytego.com/ — etudes de cas a jour (Twitter timeline fan-out, Dropbox, etc.).
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly — chap. 11-12 (stream processing, future of data systems) : assemblage des briques en systeme complet.

## Module 08 — ML System Design : Introduction

- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS 2015 — le papier canonique : le code ML est < 5% du systeme ; entanglement, feedback loops, training-serving skew.
- Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly (ISBN 978-1098107963) — manuel de reference (data, feature engineering, training-serving skew, deployment, monitoring) ; base du cours Stanford CS329S.
- Stanford CS329S. *Machine Learning Systems Design — lecture notes* (Chip Huyen). https://stanford-cs329s.github.io/ — notes de cours tier-1 sur le cycle de vie ML en prod.
- Feast Authors. *Feast — Feature Store Documentation* (consulte juin 2026). https://docs.feast.dev/ — reference open-source du feature store (point-in-time correctness, online/offline store).
- Breck, E. et al. (2017). *The ML Test Score: A Rubric for ML Production Readiness*. IEEE Big Data — checklist Google des tests data/model/infra/monitoring pour la prod.

## Module 09 — Inference at Scale

- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023 / arXiv:2309.06180 — vLLM + PagedAttention + continuous batching ; reference du serving LLM (gain throughput 2-4x).
- Leviathan, Y., Kalman, M. & Matias, Y. (2022). *Fast Inference from Transformers via Speculative Decoding*. arXiv:2211.17192 (ICML 2023) — le papier fondateur du speculative decoding (draft + verify, sortie identique).
- Cai, T. et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*. arXiv:2401.10774 (ICML 2024) — tetes de decodage multiples + tree attention, speedup 2.2-3.6x.
- Li, Y. et al. (2024). *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. arXiv:2401.15077 (ICML 2024) — speculation au niveau feature ; voir aussi EAGLE-2 (arXiv:2406.16858) et EAGLE-3 (arXiv:2503.01840).
- Zhong, Y. et al. (2024). *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving*. OSDI 2024 / arXiv:2401.09670 — separation prefill/decode sur GPUs distincts pour respecter TTFT et TPOT.
- Patel, P. et al. (2023). *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*. arXiv:2311.18677 (ISCA 2024) — phase-splitting prompt/token sur hardware adapte (1.4x throughput, -20% cout).
- Zheng, L. et al. (2023). *SGLang: Efficient Execution of Structured Language Model Programs*. arXiv:2312.07104 (NeurIPS 2024) — RadixAttention (reuse KV cache par prefixes communs), utile pour few-shot/agents.
- Lin, J. et al. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. arXiv:2306.00978 (MLSys 2024, Best Paper) — quantization weight-only int4 preservant les poids saillants ; reference pour la compression a l'inference.
- NVIDIA. *Triton Inference Server Documentation* (consulte juin 2026). https://docs.nvidia.com/deeplearning/triton-inference-server/ — reference du serving multi-framework (dynamic batching, ensembles, multi-GPU).

## Module 10 — RAG Architecture

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020 / arXiv:2005.11401 — le papier qui invente le terme RAG (retriever DPR + generateur seq2seq).
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020 / arXiv:2004.04906 — DPR : dual-encoder dense battant BM25 ; brique de retrieval du RAG dense.
- Khattab, O. & Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. SIGIR 2020 / arXiv:2004.12832 — late interaction (multi-vecteurs) ; voir aussi ColBERTv2 (arXiv:2112.01488, NAACL 2022).
- Malkov, Y.A. & Yashunin, D.A. (2016). *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs (HNSW)*. arXiv:1603.09320 (IEEE TPAMI 2020) — l'index ANN dominant des vector DBs (Qdrant, pgvector, FAISS, Weaviate).
- Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997 — taxonomie Naive/Advanced/Modular RAG, chunking, hybrid search, reranking ; survey de reference.
- Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv:2404.16130 (Microsoft) — GraphRAG : indexation par graphe d'entites pour les questions "globales".

## Module 11 — LLM Infrastructure

- Huyen, C. (2025). *AI Engineering: Building Applications with Foundation Models*. O'Reilly (ISBN 978-1098166304) — manuel de reference 2025 : prompt routing, evaluation, caching, guardrails, cost/latency tradeoffs.
- Anthropic. *Building Effective Agents* (dec. 2024). https://www.anthropic.com/research/building-effective-agents — distinction workflows vs agents et patterns composables ; cadre des couches LLM infra (routing, gateway).
- Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. arXiv:2212.08073 (Anthropic) — fondement des guardrails de sortie bases sur des principes.
- LiteLLM. *LiteLLM Documentation* (consulte juin 2026). https://docs.litellm.ai/ — reference d'un LLM gateway open-source (routing multi-provider, fallback, budgets, cles virtuelles).
- Bang, F. (2023). *GPTCache: An Open-Source Semantic Cache for LLM Applications*. ACL 2023 (NLP-OSS workshop). https://aclanthology.org/2023.nlposs-1.24/ — reference du semantic caching (cache par similarite d'embedding).

## Module 12 — Agent Systems Architecture

- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629 (ICLR 2023) — la boucle reason-act-observe ; base de l'agent single-loop.
- Anthropic. *Building Effective Agents* (dec. 2024). https://www.anthropic.com/research/building-effective-agents — patterns d'orchestration (prompt chaining, routing, parallelization, orchestrator-worker, evaluator-optimizer).
- Anthropic. *Introducing the Model Context Protocol (MCP)* (nov. 2024). https://www.anthropic.com/news/model-context-protocol — standard ouvert de connexion outils/donnees ; specs sur https://modelcontextprotocol.io.
- LangGraph. *LangGraph Documentation* (consulte juin 2026). https://langchain-ai.github.io/langgraph/ — reference d'orchestration stateful (graphes, supervisor, swarm, memoire, checkpointing).
- Wang, L. et al. (2023). *A Survey on Large Language Model based Autonomous Agents*. arXiv:2308.11432 — taxonomie agents (profil, memoire, planning, action) ; vue d'ensemble multi-agent.

## Module 13 — Observabilite & MLOps

- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS 2015 — motive le monitoring (feedback loops, drift, undeclared consumers).
- Beyer, B. et al. (2016). *Site Reliability Engineering* (Google SRE Book). O'Reilly. https://sre.google/sre-book/ — SLI/SLO/error budgets, les 4 golden signals : socle de l'observabilite.
- OpenTelemetry. *OpenTelemetry Documentation* (consulte juin 2026). https://opentelemetry.io/docs/ — standard tracing/metrics/logs (spans), substrat de Langfuse/Phoenix/Helicone.
- Langfuse. *Langfuse Documentation* (consulte juin 2026). https://langfuse.com/docs — reference open-source du tracing LLM/agents (spans, cost/tokens, evals, prompt management).
- Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217 — metriques d'eval RAG sans reference (faithfulness, answer/context relevancy) pour le monitoring qualite.

## Module 14 — Capstone

> Le capstone reutilise les sources des modules precedents. Les references ci-dessous couvrent les designs types travailles (Dropbox/file storage + un design IA) et la methode d'entretien.

- Xu, A. (2020 & 2022). *System Design Interview, Vol. 1 & 2*. ByteByteGo — designs de reference (file storage, news feed, chat) et structure d'entretien 45 min.
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly — reference transverse pour justifier chaque choix (consistance, partitioning, replication).
- Ghemawat, S., Gobioff, H. & Leung, S.-T. (2003). *The Google File System*. SOSP 2003 — fondations du stockage de fichiers distribue (chunking, replication, metadata master) ; utile pour le design Dropbox.
- Martin, D. (2018+). *The System Design Primer* (GitHub). https://github.com/donnemartin/system-design-primer — back-of-envelope estimations et checklists pour les deux designs du capstone.

---

## Ressources transversales

- **Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly** — la bible du domaine (S1). Une 2e edition est annoncee/en cours chez O'Reilly **(a verifier : statut de parution mi-2026)** ; citer l'edition 2017 par defaut.
- **Xu, A. *System Design Interview*, Vol. 1 (2020) & Vol. 2 (2022). ByteByteGo** — la reference orientee entretien ; blog/newsletter ByteByteGo (https://bytebytego.com) pour les cas a jour.
- **Huyen, C. *Designing Machine Learning Systems* (2022) & *AI Engineering* (2025). O'Reilly** — le couple ML systems + LLM apps pour la S2.
- **Beyer, B. et al. *Site Reliability Engineering* (2016) + *The Site Reliability Workbook* (2018). Google, https://sre.google/books/** — SLI/SLO/error budgets, fiabilite en prod (gratuit en ligne).
- **MIT 6.5840 (ex-6.824) — Distributed Systems** (R. Morris), https://pdos.csail.mit.edu/6.824/ — lectures + labs Go (MapReduce, Raft, KV store, sharding) ; le meilleur cours ouvert sur les systemes distribues.
- **Blogs d'ingenierie tier-1** : Netflix Tech Blog (https://netflixtechblog.com), Uber Engineering (https://www.uber.com/blog/engineering/), AWS Architecture Blog (https://aws.amazon.com/blogs/architecture/), Cloudflare Blog (https://blog.cloudflare.com), Discord/Stripe/Figma engineering — etudes de cas reelles d'architectures a l'echelle.
- **The System Design Primer** (D. Martin), https://github.com/donnemartin/system-design-primer — aide-memoire open-source (latency numbers, estimations, designs).

## Notes

- **Decoupage S1/S2** : modules 01-07 = fondations system design (sources livres + papers distribues classiques) ; modules 08-14 = ML/IA en prod (sources : papers recents + docs officielles + 2 livres O'Reilly).
- **Papers IA verifies (arXiv)** : vLLM 2309.06180, speculative decoding 2211.17192, Medusa 2401.10774, EAGLE 2401.15077 / 2406.16858 / 2503.01840, DistServe 2401.09670, Splitwise 2311.18677, SGLang 2312.07104, AWQ 2306.00978, RAG 2005.11401, DPR 2004.04906, ColBERT 2004.12832 / 2112.01488, HNSW 1603.09320, RAG survey 2312.10997, GraphRAG 2404.16130, ReAct 2210.03629, RAGAS 2309.15217.
- **Papers systemes verifies (venue)** : Dynamo (SOSP 2007), Bigtable (OSDI 2006), Spanner (OSDI 2012), Raft (USENIX ATC 2014, via les Ressources transversales), Kafka (NetDB 2011), GFS (SOSP 2003), Hidden Technical Debt (NeurIPS 2015), Memcache@Facebook (NSDI 2013).
- **Items marques (a verifier)** : draft IETF Idempotency-Key (non finalise) ; statut 2e edition DDIA ; date exacte de RAGAS et Cassandra OSR a recouper si citation precise necessaire.
- **Docs officielles** datees "consulte juin 2026" : pages vivantes (Redis, Kafka, NGINX, AWS, gRPC, GraphQL, Triton, Feast, LiteLLM, Langfuse, OpenTelemetry, LangGraph, MCP) — verifier la version a l'usage.
