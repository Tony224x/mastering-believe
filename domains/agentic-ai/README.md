# Systemes IA Agentiques — Concevoir des Agents Autonomes

## Scope

Maitriser la conception, l'implementation et le deploiement de systemes IA agentiques : du single-agent au multi-agent, avec tool use, memory, planning, et orchestration. Stack : Python, LangGraph, Claude/OpenAI APIs, MCP.

## Prerequisites

- Python courant (async, classes)
- Experience avec les APIs LLM (au moins avoir fait des appels API)
- Notions de prompting (system prompt, few-shot)

## Planning (4 semaines)

Le parcours se fait en deux temps : **S1-S2 = fondations** (J1-J14, du single-agent au multi-agent en production), puis **S3-S4 = niveau avance/frontier 2025-2026** (J15-J28 : context engineering, memoire long-horizon, verifiers, protocoles inter-agents, durabilite, coding & computer-use agents, serving a l'echelle, capstone avance). Les S3-S4 supposent les S1-S2 acquises.

> 🎮 **Suis ta progression en mode jeu** : [`PROGRESS.md`](./PROGRESS.md) — carnet XP / niveaux / badges + carte de quete (skill-tree). Chaque point est branche sur un principe d'apprentissage prouve (active recall, pratique deliberee, repetition espacee).

### Semaine 1 — Fondations Agent

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J1 | Anatomie d'un agent | Boucle ReAct, observation/action/reflexion, agent vs chatbot vs pipeline | 3h |
| J2 | Tool Use & Function Calling | Definir des tools, routing, structured output, schema JSON, error handling | 3h |
| J3 | Memory & State | Short-term (context window), long-term (vector store), working memory, checkpointing | 3h |
| J4 | Planning & Reasoning | Chain-of-thought, tree-of-thought, reflexion, plan-and-execute, self-critique | 3h |
| J5 | LangGraph fondamentaux | StateGraph, nodes, edges, conditional routing, human-in-the-loop | 4h |
| J6 | LangGraph avance | Subgraphs, parallel execution, streaming, persistence, time-travel | 4h |
| J7 | **Build : agent complet** | Agent recherche web + analyse docs avec memory et tools | 4h |

### Semaine 2 — Multi-Agent & Production

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J8 | RAG agentique | Retrieval augmente par agents : query decomposition, routing, adaptive RAG | 3h |
| J9 | Multi-agent patterns | Supervisor, hierarchical, collaborative, debate, swarm | 3h |
| J10 | MCP (Model Context Protocol) | Servers, resources, tools, prompts — connecter agents a tout | 3h |
| J11 | Evaluation & Testing | Agent evals, trajectory testing, LLM-as-judge, benchmarks, regression | 3h |
| J12 | Production & Observabilite | Langfuse/LangSmith, cost tracking, latency, error recovery, guardrails | 3h |
| J13 | Securite & Robustesse | Prompt injection, tool abuse, sandboxing, rate limiting, human oversight | 3h |
| J14 | **Capstone** | Systeme multi-agent production-ready (ex: assistant de recherche autonome) | 5h |

### Semaine 3 — Frontier patterns & orchestration avancee

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J15 | Context engineering & compaction | Curation du context window, compaction/offloading, deep-agent scratchpad/virtual FS, token & cost budgeting | 3h |
| J16 | Memoire long-horizon | Episodique/semantique/procedurale, MemGPT/Letta, consolidation, decay & scoring de pertinence | 3h |
| J17 | Verifiers & self-improvement | Verifiers / process reward models, best-of-N, self-improvement persiste, expo RL/fine-tuning | 3h |
| J18 | Orchestration comparee & failure modes | LangGraph vs CrewAI vs AutoGen vs OpenAI SDK vs Swarm vs Google ADK, quand le multi-agent casse | 3h |
| J19 | Protocoles inter-agents | A2A, agent cards, ACP, complementarite MCP, interop multi-vendor | 3h |
| J20 | Durable & event-driven agents | Durable execution (Temporal), reprise sur crash, event-driven, HITL avance (interrupt/resume) | 3h |
| J21 | Architecture des coding agents | SWE-agent & ACI, boucle edit/search/run, SWE-bench, Aider | 3h |

### Semaine 4 — Computer-use, production a l'echelle & capstone

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J22 | Computer use & GUI/browser agents | Claude computer use, CUA/Operator, browser-use, set-of-marks, action grounding | 3h |
| J23 | Sandboxing & execution sure (infra) | gVisor/microVM, sandbox-runtime, egress filtering, capability-based access | 3h |
| J24 | Inference engineering | Structured outputs/constrained decoding, model routing (RouteLLM), prompt caching | 3h |
| J25 | Serving stateful & sessions a l'echelle | Backends checkpointer (SQLite/Postgres/Redis), scaling horizontal, online eval/drift | 3h |
| J26 | Benchmarking pratique | Harness d'eval sur SON agent, pass^k, rapport de regression | 3h |
| J27 | Capstone avance — architecture | Deep ops agent durable : conception, contrats, setup | 4h |
| J28 | **Capstone avance — build & eval** | Implementation runnable de bout en bout + harness d'evaluation | 5h |

## Criteres de reussite

- [ ] Implementer un agent ReAct from scratch (sans framework) en < 100 lignes
- [ ] Concevoir un graph LangGraph multi-agent avec persistence et human-in-the-loop
- [ ] Expliquer les tradeoffs entre patterns multi-agent (supervisor vs swarm vs hierarchical)
- [ ] Deployer un serveur MCP fonctionnel avec resources + tools
- [ ] Mettre en place un pipeline d'evaluation qui detecte les regressions
- [ ] Designer l'architecture d'un systeme agentique complet sur whiteboard

### Avance (S3-S4, J15-J28)

- [ ] Maitriser le context engineering : compaction, offloading, budgeting tokens par sous-agent
- [ ] Implementer une memoire long-horizon (episodique/semantique/procedurale) avec consolidation
- [ ] Construire une boucle verifier/PRM (best-of-N) et un agent qui s'ameliore entre runs
- [ ] Comparer 5 frameworks d'orchestration et expliquer les multi-agent failure modes
- [ ] Decrire A2A (agent cards, JSON-RPC) et sa complementarite avec MCP
- [ ] Implementer un agent durable qui reprend apres un crash du process
- [ ] Dissequer la boucle edit/search/run d'un coding agent (ACI) sur un repo
- [ ] Expliquer set-of-marks et la fragilite des GUI/computer-use agents
- [ ] Choisir un niveau de sandboxing (subprocess/gVisor/microVM) selon la menace
- [ ] Mettre en place routing multi-modele + prompt caching et chiffrer le gain
- [ ] Choisir un backend de checkpointer et scaler horizontalement des sessions
- [ ] Construire un harness d'eval sur son agent avec metrique pass^k et regression

## Au-dela des 14 jours

- **`05-projets-guides/`** — 3 projets appliques au contexte d'editeur de simulation logistique LogiSim/FleetSim (voir [`shared/logistics-context.md`](../../shared/logistics-context.md)). Projet phare : `02-supervisor-swarm-multi-tier/`, qui combine les patterns supervisor et swarm de LangGraph sur une operation multi-flotte.
- **`04-projects/`** — espace libre pour mini-projets et capstones supplementaires lies au domaine.

**Note sur les exercices hard (modules 01-03)** : ce sont des mini-projets a part entiere. Des corriges complets et executables existent desormais dans `03-exercises/solutions/` (fichiers `NN-<slug>-hard.py`), en plus des solutions easy/medium.

## Patterns d'architecture agentique

### Single Agent
- **ReAct** — reason + act en boucle (le classique)
- **Plan-and-Execute** — planifier d'abord, executer ensuite
- **Reflexion** — self-critique et amelioration iterative

### Multi-Agent
- **Supervisor** — un agent chef distribue les taches
- **Hierarchical** — superviseurs en cascade
- **Collaborative** — agents pairs qui debattent/iteres
- **Swarm** — handoff dynamique entre agents specialises

### Production Patterns
- **Human-in-the-loop** — validation humaine aux points critiques
- **Guardrails** — filtrage input/output, limites de cout/temps
- **Fallback chains** — degradation gracieuse si un agent echoue

## Ressources externes

- **LangGraph Documentation** — reference officielle, tutorials
- **Claude Agent SDK** — patterns Claude-natifs
- **"Building Effective Agents"** (Anthropic blog) — principles de design
- **MCP Specification** — protocol officiel
- **DeepLearning.AI — AI Agents in LangGraph** (Andrew Ng)
- **Google × Kaggle — "5-Day AI Agents Intensive"** — cours gratuit (Gemini + ADK + Vertex AI Agent Engine, complete par MCP & A2A) ; whitepapers + codelabs Kaggle
- **Google ADK (Agent Development Kit)** — framework multi-agent open-source (workflow agents types Sequential/Parallel/Loop)
