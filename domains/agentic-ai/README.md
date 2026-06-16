# Systemes IA Agentiques — Concevoir des Agents Autonomes

> ### Carte d'entree — Peux-tu commencer ?
>
> Reponds a ces 3 questions avant de te lancer :
>
> 1. **Sais-tu lire et ecrire du Python "normal"** (fonctions, classes, dict/list, un peu d'async) sans bloquer ? → si non, fais d'abord un domaine Python avant celui-ci.
> 2. **As-tu deja appele une API LLM** (ne serait-ce qu'un `chat.completions.create` ou `messages.create`) ? → si non, fais-le une fois aujourd'hui, ca suffit pour demarrer J1.
> 3. **Sais-ce qu'est un system prompt / du few-shot** ? → si non, lis 15 min de doc prompting, c'est le seul vrai prerequis conceptuel.
>
> **Prerequis durs** (sans eux tu vas souffrir) : Python courant, avoir deja fait un appel API LLM.
> **Prerequis souples** (utiles mais rattrapables en route) : async/await, notions de vecteurs/embeddings (pour J8), avoir touche un graphe d'etats (pour J5-J6).
>
> **Temps reel** : le planning affiche ~3-5h/jour soit **~47h au total** sur 14 modules. En rythme soir/weekend, compte **3 a 5 semaines** plutot que 2 — c'est normal, le but est la maitrise, pas la vitesse. Tu peux aussi suivre le **Parcours express** ci-dessous (~20h) si tu veux le 20% qui donne 80%.

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

## Parcours express vs complet

Tu n'es pas oblige de tout faire dans l'ordre, ni tout court. Deux trajectoires :

- **Parcours express (~20h) — le 20% qui donne 80%.** Pour avoir un agent fonctionnel et comprendre les mecanismes essentiels : **J1 → J2 → J3 → J5 → J7 → J9 → J12**. Tu sais alors construire une boucle ReAct, brancher des tools, gerer la memoire, modeliser un agent en graphe LangGraph, livrer un agent complet, coordonner plusieurs agents, et l'observer en production. C'est suffisant pour un premier projet serieux.
- **Parcours complet (~47h) — la maitrise.** Les 14 modules dans l'ordre. Les modules "complementaires" (J4 planning avance, J6 LangGraph avance, J8 RAG agentique, J10 MCP, J11 eval, J13 securite) approfondissent des dimensions que tu rencontreras en vrai des que tu passes en production ou en multi-agent serieux.

> **Conseil** : commence par l'express. Si un sujet te bloque ou te passionne, va lire le module complementaire correspondant. L'interleaving (alterner les sujets) retient mieux qu'un tunnel lineaire.

## Criteres de reussite

- [ ] Implementer un agent ReAct from scratch (sans framework) en < 100 lignes
- [ ] Concevoir un graph LangGraph multi-agent avec persistence et human-in-the-loop
- [ ] Expliquer les tradeoffs entre patterns multi-agent (supervisor vs swarm vs hierarchical)
- [ ] Deployer un serveur MCP fonctionnel avec resources + tools
- [ ] Mettre en place un pipeline d'evaluation qui detecte les regressions
- [ ] Designer l'architecture d'un systeme agentique complet sur whiteboard

## Au-dela des 14 jours

- **`05-projets-guides/`** — 3 projets appliques au contexte d'editeur de simulation logistique LogiSim/FleetSim (voir [`shared/logistics-context.md`](../../shared/logistics-context.md)). Projet phare : `02-supervisor-swarm-multi-tier/`, qui combine les patterns supervisor et swarm de LangGraph sur une operation multi-flotte.
- **`04-projects/`** — espace libre pour mini-projets et capstones supplementaires lies au domaine.
- **`01-theory-qd/`** — version Quarkdown enrichie de la theorie (math LaTeX, mermaid, callouts). Build : `pwsh quarkdown/scripts/build-all.ps1 -Domain agentic-ai`. Les `.md` de `01-theory/` restent la source-of-truth.

### Etat de la couverture des exercices (honnete)

Tous les modules n'ont pas encore les 3 niveaux d'exercices :

| Niveau | Modules couverts |
|--------|------------------|
| **Easy** (`01-easy/`) | **J1 → J14** (les 14 modules) |
| **Medium** (`02-medium/`) | **J1, J2, J3 uniquement** |
| **Hard** (`03-hard/`) | **J1, J2, J3 uniquement** |

Autrement dit : **J4 a J14 n'ont pour l'instant que le niveau easy.** Les niveaux medium/hard pour J4-J14 sont une **extension prevue** (contributions bienvenues — c'est un repo public). Les exercices hard existants (J1-J3) sont de vrais mini-projets, avec corriges complets et executables dans `03-exercises/solutions/` (fichiers `NN-<slug>-hard.py`), en plus des solutions easy/medium.

En attendant, pour J4-J14 : le module `04-projects/` (mini-projets libres) et les `05-projets-guides/` offrent de la pratique avancee appliquee.

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
