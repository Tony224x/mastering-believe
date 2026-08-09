# Assets visuels — agentic-ai

**Standard qualité** (SSOT) :  
[`.claude/skills/mastering-domain-creator/references/svg-pedagogique.md`](../../../../.claude/skills/mastering-domain-creator/references/svg-pedagogique.md)

## Design system (Teal Trust)

| Rôle | Hex |
|------|-----|
| Primary | `#0F766E` |
| Accent | `#F59E0B` |
| Danger | `#DC2626` |
| Surface | `#F8FAFC` |
| Ink | `#0F172A` |

Chaque SVG : **1200×680**, ombre système, barre latérale teal, `title`+`desc`, 1 idée.

## Style KaView (légendes positives)

Sous chaque image — toujours utile, jamais un message d'échec :

```markdown
![alt descriptif](../assets/NN-slug.svg)

> **En une phrase :** takeaway pédagogique.
>
> **Visuel :** ce que le schéma montre (caption lisible hors image).
```

L'`alt` + le bloc **Visuel** couvrent accessibilité et lecture KaView hors-ligne.

## Inventaire

| Fichier | Contenu |
|---------|---------|
| `01-anatomie-agent.svg` | Boucle Perceive → Think → Act d'un agent IA |
| `02-tool-use-function-calling.svg` | Flux tool use en quatre étapes |
| `03-memory-state.svg` | Trois couches de mémoire + checkpoint |
| `04-planning-reasoning.svg` | Quatre patterns de planning du simple au cher |
| `05-langgraph-fondamentaux.svg` | StateGraph LangGraph : START agent tools END |
| `06-langgraph-avance.svg` | Quatre leviers LangGraph avancés |
| `07-agent-complet.svg` | Pipeline agent complet cinq étapes |
| `08-rag-agentique.svg` | RAG vanilla versus RAG agentique |
| `09-multi-agent-patterns.svg` | Quatre patterns multi-agent |
| `10-mcp.svg` | Architecture MCP host client server |
| `11-evaluation-testing.svg` | Pyramide d'évaluation d'un agent |
| `12-production-observabilite.svg` | Quatre piliers d'observabilité prod |
| `13-securite-robustesse.svg` | Surfaces d'attaque et défense en profondeur |
| `14-capstone.svg` | Capstone multi-agent supervisor et workers |
| `15-context-engineering-compaction.svg` | Context rot, compaction et offloading |
| `16-memoire-long-horizon.svg` | Mémoire épisodique sémantique procédurale |
| `17-verifiers-self-improvement.svg` | Boucle generate verifier select persist |
| `18-orchestration-comparee-failure-modes.svg` | Comparaison frameworks d'orchestration agents |
| `19-protocoles-inter-agents.svg` | A2A entre agents et complémentarité MCP |
| `20-durable-event-driven-agents.svg` | Checkpoint versus durable execution |
| `21-coding-agents-architecture.svg` | Boucle SEARCH EDIT RUN OBSERVE d'un coding agent |
| `22-computer-use-gui-agents.svg` | Boucle GUI screenshot grounding action |
| `23-sandboxing-execution-sure.svg` | Quatre couches de sandboxing infra |
| `24-inference-engineering.svg` | Structured outputs, routing et caching |
| `25-serving-stateful-sessions.svg` | Serving : clients, workers stateless, checkpointer |
| `26-benchmarking-pratique.svg` | Pipeline harness pass^k |
| `27-capstone-architecture.svg` | Architecture deep ops agent en huit briques |
| `28-capstone-build-eval.svg` | Quatre étapes du capstone build et eval |
| `parcours-28j.svg` | Carte du parcours agentic-ai 28 jours en 4 semaines |
| `08-rag-pipeline-complet.svg` | Pipeline RAG agentique complet |
| `14-capstone-pipeline.svg` | Pipeline capstone de production |
| `03-memory-production.svg` | Architecture mémoire d'un agent en production |
| `13-defense-profondeur.svg` | Défense en profondeur — cinq couches |
| `07-research-agent-archi.svg` | Architecture du research agent |
| `11-eval-pipeline.svg` | Pipeline d'évaluation Dev → Prod |
| `09-hierarchical.svg` | Pattern multi-agent hiérarchique |
| `09-debate.svg` | Pattern débat multi-agent |
| `09-swarm-handoff.svg` | Pattern Swarm handoff |
| `09-decision-tree.svg` | Arbre décision single vs multi-agent |
| `06-agent-prod-complet.svg` | Patterns LangGraph combinés production |
| `06-parallel-vs-seq.svg` | Exécution séquentielle vs parallèle |
| `16-main-vs-external.svg` | Main context versus external context |
| `16-memory-flow.svg` | Flux mémoire agent long-horizon |
| `10-mcp-lifecycle.svg` | Lifecycle connexion MCP |
| `04-plan-execute.svg` | Pattern Plan-and-Execute |
| `04-tree-of-thought.svg` | Tree-of-Thought |
| `22-perceive-mark-act.svg` | Boucle perceive mark act GUI |
| `22-set-of-marks.svg` | Set-of-Marks prompting |
| `25-stateful-problem.svg` | Stateful vs workers stateless |
| `03-hybrid-memory.svg` | Hybrid memory summary plus fenêtre |
| `08-hybrid-rrf.svg` | Hybrid retrieval RRF rerank |
| `14-deployment.svg` | Stack déploiement capstone |
| `19-a2a-task-lifecycle.svg` | Lifecycle tâche A2A |
| `20-durable-combined.svg` | Architecture durable event-driven HITL |
| `24-three-levers.svg` | Trois leviers inference engineering |
| `23-isolation-layers.svg` | Couches d'isolation sandbox |
| `15-context-isolation.svg` | Isolation contexte sous-agent |
| `17-self-refine-loop.svg` | Boucle Self-Refine verifier |

## Régénérer

```bash
python domains/tech/agentic-ai/scripts/generate_assets.py
```

Structure du générateur :

```
scripts/
  generate_assets.py          # CLI
  assets_gen/
    kit.py                    # palette + primitives SVG
    captions.py               # alt / légende / takeaway
    registry.py               # SPECS
    inject.py                 # injection MD style KaView
    builds/
      s1_fondations.py        # J1–J7
      s2_multi_prod.py        # J8–J14
      s3_frontier.py          # J15–J21
      s4_scale.py             # J22–J28 + parcours
      s5_body_diagrams.py     # schémas corps lot A
      s6_enrichment.py        # enrichissement lots B/C
```
