# System Design — Architecture Backend & IA

## Scope

Concevoir des architectures scalables, maintenables et production-ready pour des systemes backend et IA. Couvre le system design classique (entretiens senior/staff) ET les patterns specifiques aux systemes ML/IA en production.

## Carte d'entree

> **Peux-tu commencer ?** Reponds a ces 3 questions :
> 1. **Sais-tu ce qu'est une API REST et une base de donnees relationnelle ?** (tu as deja ecrit un endpoint qui lit/ecrit en DB)
> 2. **As-tu deja deploye quelque chose ?** (un container, une VM, un service cloud — meme un simple `docker run`)
> 3. **Sais-tu, en une phrase, ce qu'est un modele ML et une inference ?** (un modele entraine qui produit une prediction)
>
> - **3 oui** → tu es pret pour le parcours complet (S1 + S2).
> - **Oui aux questions 1-2, non a la 3** → fais la Semaine 1 (parcours generaliste), la Semaine 2 viendra apres une remise a niveau IA.
> - **Non a 1 ou 2** → commence par consolider les bases backend/cloud avant ce domaine.
>
> **Prerequis** : backend (REST + SQL), notions cloud (containers/deploiement), et pour la S2 seulement, des bases ML.
> **Temps** : ~3h par module. Parcours express ≈ 12-15h, parcours complet ≈ 40-45h (voir ci-dessous).

## Prerequisites

- Experience backend (API REST, bases de donnees relationnelles)
- Notions de cloud (containers, deploiement)
- Bases ML/IA (savoir ce qu'est un modele, une inference) — **uniquement pour la Semaine 2**

## Deux parcours : express vs complet

Ce domaine a un **double palier** : la Semaine 1 (fondations system design, autonome) et la Semaine 2 (architecture IA, qui s'appuie sur la S1). Selon ton objectif, suis l'un des deux parcours.

| | **Parcours generaliste** (Semaine 1) | **Parcours IA avancee** (Semaine 1 + 2) |
|---|---|---|
| **Pour qui** | Entretien backend/infra, fondations solides | Entretien ML/IA senior-staff, plateformes ML en prod |
| **Modules** | J1 → J7 | J1 → J14 |
| **Charniere** | termine au capstone classique (J7) | le **module J8 est le pont** vers l'IA |
| **Temps** | ~21h (7 modules x 3h) | ~43h (14 modules) |
| **Sortie** | designer URL shortener / Twitter / chat en 45 min | + designer RAG + agents production-ready |

> **Parcours express (12-15h, revision avant entretien)** : J1 (principes) → J2 (storage) → J3 (caching) → J5 (load balancing) → J7 (designs classiques). On saute J4/J6 en premiere passe et on les recupere via les exercices. Objectif : couvrir le 20% le plus interroge en entretien.

> **Le module J8 est le maillon S1↔S2.** Si tu enchaines vers l'IA, ne saute pas J8 : il traduit le vocabulaire et les patterns de la Semaine 1 vers le monde ML (feature store = DB + cache, retraining = message queue, serving = LB + API). Il pose aussi les 5 termes-cles de la S2 (embedding, inference, feature, drift, point-in-time correctness).

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
| J7 | **Design classiques** | URL shortener, Twitter feed, Chat system — exercices chronometres | 4h |

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

## Exercices & projets guides

Les exercices de `03-exercises/` couvrent tous les modules, avec les niveaux medium et hard concentres sur les fondamentaux (modules 01-03) — la ou la pratique deliberee rapporte le plus.

Le dossier [`05-projets-guides/`](05-projets-guides/) propose 3 projets appliques au contexte LogiSim (simulation logistique automatisee, voir [`shared/logistics-context.md`](../../shared/logistics-context.md)) : simulation distribuee, pipeline EOD, plateforme multi-tenant air-gapped.

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
