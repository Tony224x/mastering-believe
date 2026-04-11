# System Design — Architecture Backend & IA

## Scope

Concevoir des architectures scalables, maintenables et production-ready pour des systemes backend et IA. Couvre le system design classique (entretiens senior/staff) ET les patterns specifiques aux systemes ML/IA en production.

## Prerequisites

- Experience backend (API REST, bases de donnees relationnelles)
- Notions de cloud (containers, deploiement)
- Bases ML/IA (savoir ce qu'est un modele, une inference)

## Planning (2 semaines)

### Semaine 1 — Fondations System Design

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J1 | Principes fondamentaux | CAP theorem, scalabilite horizontale/verticale, latence vs throughput, SLA/SLO | 3h |
| J2 | Stockage & Databases | SQL vs NoSQL, sharding, replication, indexing, partitioning, choix de DB | 3h |
| J3 | Caching & CDN | Strategies de cache (write-through, write-back, aside), invalidation, Redis, CDN | 3h |
| J4 | Message Queues & Event-Driven | Kafka, RabbitMQ, pub/sub, event sourcing, CQRS, saga pattern | 3h |
| J5 | Load Balancing & Networking | L4/L7 LB, reverse proxy, DNS, rate limiting, circuit breaker | 3h |
| J6 | API Design & Patterns | REST, gRPC, GraphQL, API gateway, versioning, idempotency, pagination | 3h |
| J7 | **Design classiques** | URL shortener, Twitter feed, Chat system — exercices chronomentres | 4h |

### Semaine 2 — Architecture IA & Production ML

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J8 | ML System Design intro | Feature stores, model registry, training vs serving, batch vs real-time | 3h |
| J9 | Inference at scale | Model serving (TorchServe, Triton, vLLM), GPU scheduling, batching, quantization | 3h |
| J10 | RAG Architecture | Vector DBs, embedding pipelines, chunking strategies, hybrid search, reranking | 3h |
| J11 | LLM Infrastructure | Prompt routing, caching (semantic), guardrails, cost optimization, fallbacks | 3h |
| J12 | Agent Systems Architecture | Orchestration patterns, tool routing, memory, multi-agent coordination | 3h |
| J13 | Observabilite & MLOps | Monitoring, drift detection, A/B testing, CI/CD ML, feature flags, Langfuse | 3h |
| J14 | **Capstone** | Designer 2 systemes complets (1 classique + 1 IA) en conditions d'entretien | 4h |

## Criteres de reussite

- [ ] Designer un systeme complet en 45 min avec diagramme, estimations de charge, et tradeoffs
- [ ] Identifier spontanement les 3 bottlenecks principaux de toute architecture proposee
- [ ] Justifier chaque choix technique (pourquoi X et pas Y) avec des chiffres
- [ ] Maitriser le vocabulaire : parler de p99 latency, throughput, consistency models naturellement
- [ ] Designer une architecture RAG + agents production-ready de A a Z

## Framework de resolution (methode pour les entretiens)

1. **Clarifier** — requirements fonctionnels et non-fonctionnels (2 min)
2. **Estimer** — utilisateurs, QPS, stockage, bande passante (3 min)
3. **High-level design** — composants principaux et flux de donnees (10 min)
4. **Deep dive** — 2-3 composants critiques en detail (20 min)
5. **Bottlenecks & tradeoffs** — identifier, proposer des alternatives (5 min)
6. **Extensions** — comment evoluer le systeme (5 min)

## Ressources externes

- **Designing Data-Intensive Applications** (Kleppmann) — la bible
- **System Design Interview** (Alex Xu, vol. 1 & 2)
- **ML System Design** (Stanford CS 329S notes)
- **Architecture of Open Source Applications** — etudes de cas reels
