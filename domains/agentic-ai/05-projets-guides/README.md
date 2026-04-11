# 05 — Projets guides (Agentic AI)

> Voir `shared/masa-context.md` pour le contexte metier de MASA Group.

Trois projets qui illustrent comment transformer le paradigme de SWORD (agents autonomes scripted via Direct AI) en **systemes multi-agents modernes pilotes par LLM**. Le projet phare est le **02 — Supervisor + Swarm** qui combine les deux patterns principaux de LangGraph dans un scenario interarmes.

## Projets

| # | Projet | Pattern | Difficulte |
|---|---|---|---|
| 01 | **Agent commandant de peloton** | Single-agent avec tools | medium |
| 02 | **Supervisor + Swarm interarmes** ⭐ | Multi-agent hierarchique + horizontal | hard |
| 03 | **Agent AAR conversationnel** | Agent + RAG sur events | medium |

## Methodologie

Pour chaque projet :
1. Lire le contexte metier
2. Lire les specs de l'architecture agent (nodes, state, tools)
3. Coder la v0 a partir du squelette fourni
4. Confronter a la correction commentee
5. Faire tourner la demo et observer la trace d'execution

## Pourquoi pas juste Direct AI ?

Direct AI (le moteur historique de MASA) est excellent pour des comportements **reactifs** : etat du monde -> action optimale. Mais il est limite pour :
- Raisonnement **multi-etapes** avec negociation ("je vais attendre le support artillerie avant d'engager")
- Traitement du **langage naturel** (interpreter un OPORD en francais)
- **Creativite** tactique (trouver des ruses non scriptees)
- Adaptation a un **scenario inedit** sans reprogrammation

Les agents LLM complementent Direct AI sans le remplacer. Dans la realite, on aurait un systeme hybride : Direct AI pour le tempo rapide (sub-seconde), LLM agents pour les decisions lentes (minute, echelon etat-major).

## Stack technique

- **LangGraph** pour la structure agentique (Graph API + prebuilt helpers)
- **LangChain** pour les tools et les messages
- **LLM** : claude-haiku-4-5 pour la v0 (rapide et bon marche), upgrade vers claude-opus-4-6 si les decisions demandent du reasoning lourd
- **Observabilite** : les projets incluent des prints structures pour tracer les decisions — en prod on brancherait LangSmith ou un logger equivalent

## Requirements

```bash
pip install langgraph langchain-anthropic langchain-core
export ANTHROPIC_API_KEY=sk-ant-...
```

Sans cle API, les scripts tombent en mode "stub" qui simule des reponses — permet de tester la structure du graphe sans consommer de tokens.
