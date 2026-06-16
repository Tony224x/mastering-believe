# Plan figé domaine `system-design` (14 modules / 2 semaines)

> Curriculum figé, dérivé du [`README.md`](README.md). Source de vérité pour la structure du domaine.
> **Double palier** : Semaine 1 (J1→J7, fondations system design, autonome) + Semaine 2 (J8→J14, architecture IA, s'appuie sur la S1). Le **module J8 est le pont** entre les deux.
> **Convention de slug** : pour un module N, `01-theory/NN-x.md`, `01-theory-qd/NN-x.qd` et les exercices/solutions partagent le même slug numéroté. Les modules avec code applicable ont un `02-code/NN-x.py`.
> **REFERENCES.md** : non encore produit (phase 2). Les ressources canoniques sont listées dans le `README.md` (section « Ressources externes »).

---

## Semaine 1 — Fondations System Design (parcours généraliste)

## J1 — Principes fondamentaux

- **Concepts clés** :
  - CAP theorem (consistency / availability / partition tolerance)
  - Scalabilité horizontale vs verticale
  - Latence vs throughput
  - SLA / SLO / SLI, disponibilité (les « neuf »), p50/p95/p99
  - Estimations d'ordre de grandeur (QPS, stockage)
- **Acquis fin de module** : raisonner CAP sur un cas, estimer QPS et stockage d'un service, parler p99/throughput naturellement.
- **Stack** : diagrammes + stdlib Python (estimations)
- **Slug** : `01-principes-fondamentaux`
- **Temps** : ~3h

## J2 — Stockage & Databases

- **Concepts clés** :
  - SQL vs NoSQL (familles : KV, document, colonne, graphe)
  - Sharding (par hash, par range), partitioning
  - Réplication (leader-follower, multi-leader, leaderless), quorums
  - Indexing, choix de DB selon le pattern d'accès
- **Acquis fin de module** : choisir une DB justifiée par le pattern d'accès, expliquer sharding + réplication + tradeoffs de cohérence.
- **Stack** : diagrammes + stdlib Python
- **Slug** : `02-stockage-databases`
- **Temps** : ~3h

## J3 — Caching & CDN

- **Concepts clés** :
  - Stratégies : cache-aside, write-through, write-back
  - Invalidation, TTL, éviction (LRU/LFU)
  - Redis (structures, usages), CDN (edge caching)
  - Hit rate, hot keys, thundering herd
- **Acquis fin de module** : choisir une stratégie de cache + invalidation pour un cas, estimer un hit rate, éviter les pièges (stampede, hot key).
- **Stack** : diagrammes + stdlib Python
- **Slug** : `03-caching-cdn`
- **Temps** : ~3h

## J4 — Message Queues & Event-Driven

- **Concepts clés** :
  - Kafka vs RabbitMQ (log vs broker), pub/sub
  - Event sourcing, CQRS, saga pattern
  - Garanties de livraison (at-most/at-least/exactly-once), idempotence, ordering
  - Backpressure, dead letter queue
- **Acquis fin de module** : designer un flux event-driven, choisir le bon broker, raisonner les garanties de livraison.
- **Stack** : diagrammes + stdlib Python
- **Slug** : `04-message-queues-event-driven`
- **Temps** : ~3h

## J5 — Load Balancing & Networking

- **Concepts clés** :
  - Load balancing L4 vs L7, algorithmes (round-robin, least-conn, hash)
  - Reverse proxy, DNS, anycast
  - Rate limiting (token bucket, leaky bucket), circuit breaker
  - Health checks, sticky sessions
- **Acquis fin de module** : positionner les LB dans une archi, choisir L4/L7, dimensionner un rate limiter, expliquer un circuit breaker.
- **Stack** : diagrammes + stdlib Python
- **Slug** : `05-load-balancing-networking`
- **Temps** : ~3h

## J6 — API Design & Patterns

- **Concepts clés** :
  - REST vs gRPC vs GraphQL (quand quoi)
  - API gateway, versioning, pagination (offset vs cursor)
  - Idempotency keys, retries, error handling
- **Acquis fin de module** : designer une API propre (versioning, pagination, idempotence), choisir le protocole adapté au use case.
- **Stack** : diagrammes + stdlib Python
- **Slug** : `06-api-design-patterns`
- **Temps** : ~3h

## J7 — Design classiques (capstone Semaine 1)

- **Concepts clés** :
  - URL shortener (encoding base62, compteur + range allocation vs hash, birthday paradox, KV shardé, cache)
  - Twitter feed (fanout-on-write vs fanout-on-read, hybride célébrités)
  - Chat system (WebSocket, connection registry, persist + delivery)
  - Exercices chronométrés en conditions d'entretien
- **Acquis fin de module** : designer un système complet en 45 min (diagramme + estimations + tradeoffs).
- **Stack** : diagrammes + Python (mocks runnable : `02-code/07-design-classiques.py`)
- **Slug** : `07-design-classiques`
- **Temps** : ~4h

---

## Semaine 2 — Architecture IA & Production ML (parcours IA avancée)

## J8 — ML System Design : introduction (le PONT S1 → S2)

- **Concepts clés** :
  - Traduction des patterns S1 vers le ML (feature store = DB + cache, retraining = message queue, serving = LB + API)
  - 5 termes-clés S2 : embedding, inference, feature, drift, point-in-time correctness
  - Training-serving skew, cycle de vie d'un système ML (boucle, pas pipeline linéaire)
  - Feature store : réutilisation, consistance offline/online, point-in-time correctness
  - Model registry, training vs serving, batch vs real-time
- **Acquis fin de module** : maîtriser le vocabulaire S2, expliquer pourquoi un système ML est un système distribué dont le modèle est un composant, designer un feature store.
- **Rôle** : maillon faible d'accessibilité — module-charnière à ne pas sauter pour qui enchaîne sur l'IA.
- **Stack** : diagrammes + stdlib Python
- **Slug** : `08-ml-system-design-intro`
- **Temps** : ~3h

## J9 — Inference at scale

- **Concepts clés** :
  - Model serving (TorchServe, Triton, vLLM)
  - GPU scheduling, batching (dynamic/continuous), quantization
  - Latence vs throughput côté inférence, KV cache (LLM)
- **Acquis fin de module** : dimensionner une couche de serving, choisir batching + quantization, raisonner le coût GPU.
- **Stack** : diagrammes + Python
- **Slug** : `09-inference-at-scale`
- **Temps** : ~3h

## J10 — RAG Architecture

- **Concepts clés** :
  - Vector DBs, embedding pipelines
  - Chunking strategies, hybrid search (dense + sparse), reranking
  - Évaluation RAG, fraîcheur de l'index
- **Acquis fin de module** : designer un pipeline RAG de bout en bout (ingestion → embedding → retrieval → rerank).
- **Stack** : diagrammes + Python
- **Slug** : `10-rag-architecture`
- **Temps** : ~3h

## J11 — LLM Infrastructure

- **Concepts clés** :
  - Prompt routing, caching sémantique
  - Guardrails, cost optimization, fallbacks (modèle dégradé)
  - Observabilité des appels LLM, gestion des coûts
- **Acquis fin de module** : designer une infra LLM résiliente (routing, cache sémantique, guardrails, fallbacks).
- **Stack** : diagrammes + Python
- **Slug** : `11-llm-infrastructure`
- **Temps** : ~3h

## J12 — Agent Systems Architecture

- **Concepts clés** :
  - Orchestration patterns (supervisor, swarm), tool routing
  - Mémoire (court/long terme), coordination multi-agent
  - Boucles, garde-fous, terminaison
- **Acquis fin de module** : designer une archi multi-agent avec orchestration, mémoire et routing d'outils.
- **Stack** : diagrammes + Python
- **Slug** : `12-agent-systems-architecture`
- **Temps** : ~3h

## J13 — Observabilité & MLOps

- **Concepts clés** :
  - Monitoring, drift detection
  - A/B testing, CI/CD ML, feature flags
  - Langfuse, tracing, retraining déclenché par drift
- **Acquis fin de module** : instrumenter un système ML (monitoring + drift + A/B + CI/CD ML).
- **Stack** : diagrammes + Python
- **Slug** : `13-observabilite-mlops`
- **Temps** : ~3h

## J14 — Capstone

- **Concepts clés** :
  - Designer 2 systèmes complets (1 classique + 1 IA) en conditions d'entretien
  - Application du framework de résolution (clarifier → estimer → high-level → deep dive → bottlenecks → extensions)
- **Acquis fin de module** : produire 2 designs complets en temps limité, prêts pour un entretien senior/staff.
- **Stack** : diagrammes + Python
- **Slug** : `14-capstone`
- **Temps** : ~4h

---

## Framework de résolution (transversal, appliqué dès J7)

1. **Clarifier** — requirements fonctionnels et non-fonctionnels (2 min)
2. **Estimer** — utilisateurs, QPS, stockage, bande passante (3 min)
3. **High-level design** — composants principaux et flux de données (10 min)
4. **Deep dive** — 2-3 composants critiques en détail (20 min)
5. **Bottlenecks & tradeoffs** — identifier, proposer des alternatives (5 min)
6. **Extensions** — comment évoluer le système (5 min)
