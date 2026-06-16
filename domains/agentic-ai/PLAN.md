# PLAN — Systemes IA Agentiques

> Plan fige du curriculum de ce domaine : 14 modules ("jours"), repartis en 2 vagues
> (Fondations Agent, puis Multi-Agent & Production). Pour chaque module : titre,
> objectif, et acquis vises (ce que l'apprenant doit savoir faire en sortie).
>
> Ce fichier est la reference figee du curriculum. La progression detaillee, les
> prerequis, le parcours express et l'etat de couverture des exercices sont dans
> [`README.md`](./README.md). La source-of-truth pedagogique reste `01-theory/`.

## Vue d'ensemble

| # | Module | Slug | Temps | Niveaux d'exercices |
|---|--------|------|-------|---------------------|
| J1 | Anatomie d'un agent | `01-anatomie-agent` | 3h | easy, medium, hard |
| J2 | Tool Use & Function Calling | `02-tool-use-function-calling` | 3h | easy, medium, hard |
| J3 | Memory & State | `03-memory-state` | 3h | easy, medium, hard |
| J4 | Planning & Reasoning | `04-planning-reasoning` | 3h | easy |
| J5 | LangGraph fondamentaux | `05-langgraph-fondamentaux` | 4h | easy |
| J6 | LangGraph avance | `06-langgraph-avance` | 4h | easy |
| J7 | Build : agent complet | `07-agent-complet` | 4h | easy |
| J8 | RAG agentique | `08-rag-agentique` | 3h | easy |
| J9 | Multi-agent patterns | `09-multi-agent-patterns` | 3h | easy |
| J10 | MCP (Model Context Protocol) | `10-mcp` | 3h | easy |
| J11 | Evaluation & Testing | `11-evaluation-testing` | 3h | easy |
| J12 | Production & Observabilite | `12-production-observabilite` | 3h | easy |
| J13 | Securite & Robustesse | `13-securite-robustesse` | 3h | easy |
| J14 | Capstone | `14-capstone` | 5h | easy |

Total indicatif : ~47h. Voir le **Parcours express** du README pour un sous-ensemble ~20h.

---

## Vague 1 — Fondations Agent (J1-J7)

### J1 — Anatomie d'un agent
- **Objectif** : comprendre ce qu'est reellement un agent IA, quand l'utiliser, et maitriser la boucle ReAct from scratch.
- **Acquis** :
  - Distinguer agent vs chatbot vs pipeline, et reconnaitre quand un agent est justifie (et quand il ne l'est pas).
  - Decrire la boucle ReAct (observation → reflexion → action → observation).
  - Implementer un agent ReAct minimal sans framework (stdlib + httpx), en mode simule comme en mode live.

### J2 — Tool Use & Function Calling
- **Objectif** : maitriser le tool use de A a Z — du design de tools a l'execution parallele, en passant par la securite et le structured output.
- **Acquis** :
  - Definir des tools (name, description, JSON schema) et comprendre le function calling natif sous le capot.
  - Designer des tools efficaces (granularite, naming, nombre, descriptions) et router fiablement.
  - Forcer un output structure (JSON mode, JSON schema, structured outputs natifs, `tool_choice`).
  - Gerer les erreurs de tools (feedback plutot que crash) et paralleliser les appels independants.

### J3 — Memory & State
- **Objectif** : comprendre et implementer les differents types de memoire d'un agent IA, maitriser le state management, et savoir quand/comment persister l'etat.
- **Acquis** :
  - Distinguer memoire short-term (fenetre de contexte), long-term (vector store) et working memory (scratchpad).
  - Implementer buffer / sliding window / summary / hybrid memory et arbitrer selon le cout et la perte d'info.
  - Faire de la recherche par similarite (embeddings + cosine) et du checkpointing (sauvegarde/restauration d'etat).

### J4 — Planning & Reasoning
- **Objectif** : comprendre comment forcer un LLM a raisonner explicitement, maitriser les patterns de planification (CoT, ToT, ReAct, plan-and-execute, Reflexion), et savoir quand chacun aide vs quand chacun fait perdre du temps.
- **Acquis** :
  - Appliquer chain-of-thought, tree-of-thought, plan-and-execute, self-critique / Reflexion.
  - Connaitre l'extended thinking / test-time compute et son rapport cout/benefice.
  - Choisir le bon pattern de raisonnement selon la tache (raisonnement dur vs conversationnel/extraction).

### J5 — LangGraph fondamentaux
- **Objectif** : comprendre le modele mental de LangGraph (StateGraph + nodes + edges + state), savoir construire un graph minimal from scratch, et maitriser invoke/stream + human-in-the-loop.
- **Acquis** :
  - Modeliser un agent comme un graphe : state typé (TypedDict + reducers), nodes, edges (inconditionnels et conditionnels).
  - Compiler et executer un graphe (`invoke`, `stream`), avec routing dynamique.
  - Mettre en place un human-in-the-loop via interrupts.

### J6 — LangGraph avance
- **Objectif** : maitriser les patterns avances de LangGraph (subgraphs, parallel execution, streaming, persistence, time-travel debugging) pour construire des agents production-grade.
- **Acquis** :
  - Composer des subgraphs et executer des branches en parallele (fan-out / fan-in).
  - Brancher un checkpointer (Memory / Sqlite / Postgres) et gerer des threads multi-conversation.
  - Faire du time-travel debugging (reprendre/rejouer depuis un checkpoint).

### J7 — Build : agent complet
- **Objectif** : assembler tout ce qu'on a appris en un agent production-credible — un agent de recherche qui planifie, execute des outils, maintient une memoire, gere les erreurs, et synthetise une reponse.
- **Acquis** :
  - Integrer tool use + memory + planning + error handling dans un seul agent coherent.
  - Orchestrer un workflow multi-etapes de bout en bout.
  - Produire une synthese finale a partir d'observations collectees.

---

## Vague 2 — Multi-Agent & Production (J8-J14)

### J8 — RAG agentique
- **Objectif** : comprendre ce qui separe un RAG "bete" d'un RAG agentique, implementer query decomposition, routing, retrieval grading et multi-hop reasoning.
- **Acquis** :
  - Decomposer une requete, router vers la bonne source, noter la pertinence du retrieval (grading).
  - Faire du raisonnement multi-hop (retrieval iteratif guide par l'agent).
  - Reconnaitre les faiblesses RAG cote securite (empoisonnement d'index, fuite cross-tenant).

### J9 — Multi-agent patterns
- **Objectif** : savoir quand et comment faire collaborer plusieurs agents specialises, connaitre les 4 patterns de reference, et arbitrer entre complexite et qualite.
- **Acquis** :
  - Comparer supervisor, hierarchical, collaborative/debate et swarm (handoff).
  - Choisir le pattern adapte et justifier le surcout de complexite par un gain de qualite.
  - Implementer la coordination (distribution de taches, handoff, agregation).

### J10 — MCP (Model Context Protocol)
- **Objectif** : comprendre ce qu'est MCP, pourquoi c'est important, comment construire un serveur MCP, et comment s'en servir dans un agent.
- **Acquis** :
  - Decrire le modele client/serveur MCP (tools, resources, prompts, sampling, elicitation).
  - Construire et brancher un serveur MCP sur un agent.
  - Situer l'evolution de la spec (2024-11-05 → 2025-11-25) et sa gouvernance ouverte.

### J11 — Evaluation & Testing
- **Objectif** : comprendre les differents niveaux d'evaluation d'un agent, maitriser le LLM-as-judge, ecrire des regression tests, et savoir quels benchmarks utiliser.
- **Acquis** :
  - Distinguer evaluation de sortie, de trajectoire et de bout en bout.
  - Mettre en place un LLM-as-judge et des tests de regression.
  - Connaitre les benchmarks de reference (GAIA, SWE-bench, TAU-bench) et leurs limites.

### J12 — Production & Observabilite
- **Objectif** : savoir comment tracer, monitorer, recuperer et budgeter un agent en production.
- **Acquis** :
  - Tracer et monitorer un agent (latence, cout, erreurs) via tracing structure.
  - Mettre en place cost tracking, budgets et fallback chains.
  - Concevoir une recuperation d'erreur gracieuse (retry, degradation).

### J13 — Securite & Robustesse
- **Objectif** : comprendre les surfaces d'attaque d'un agent, maitriser les techniques d'injection et leurs defenses, savoir ou placer les humains dans la boucle.
- **Acquis** :
  - Reconnaitre prompt injection (directe/indirecte), tool abuse, confused deputy, DoS/cost exhaustion.
  - Appliquer une defense en profondeur (input/tool/output guardrails, trust boundaries, sandboxing).
  - Auditer un agent contre l'OWASP LLM Top 10 (2025) et placer les points HITL.

### J14 — Capstone
- **Objectif** : reunir tout ce qu'on a appris pour designer et implementer un systeme multi-agent production-ready — quelque chose qui pourrait vraiment tourner en prod.
- **Acquis** :
  - Concevoir l'architecture d'un systeme multi-agent complet (supervisor + workers + memoire + tools + HITL + observabilite).
  - Implementer le systeme de bout en bout, executable offline.
  - Integrer un harness d'evaluation simple pour valider le comportement.
