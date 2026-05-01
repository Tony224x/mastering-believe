# Systemes IA Agentiques — Concevoir des Agents Autonomes

## Scope

Maitriser la conception, l'implementation et le deploiement de systemes IA agentiques : du single-agent au multi-agent, avec tool use, memory, planning, et orchestration. Stack : Python, LangGraph, Claude/OpenAI APIs, MCP.

## Prerequisites

- Python courant (async, classes)
- Experience avec les APIs LLM (au moins avoir fait des appels API)
- Notions de prompting (system prompt, few-shot)

## Planning (2 semaines)

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

## Criteres de reussite

- [ ] Implementer un agent ReAct from scratch (sans framework) en < 100 lignes
- [ ] Concevoir un graph LangGraph multi-agent avec persistence et human-in-the-loop
- [ ] Expliquer les tradeoffs entre patterns multi-agent (supervisor vs swarm vs hierarchical)
- [ ] Deployer un serveur MCP fonctionnel avec resources + tools
- [ ] Mettre en place un pipeline d'evaluation qui detecte les regressions
- [ ] Designer l'architecture d'un systeme agentique complet sur whiteboard

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
- **Anthropic Agent SDK** — patterns Claude-natifs
- **"Building Effective Agents"** (Anthropic blog) — principles de design
- **MCP Specification** — protocol officiel
- **DeepLearning.AI — AI Agents in LangGraph** (Andrew Ng)
