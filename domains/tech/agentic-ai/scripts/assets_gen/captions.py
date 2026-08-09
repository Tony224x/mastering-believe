"""Métadonnées pédagogiques des schémas (alt, légende, takeaway)."""

from __future__ import annotations

CAPTIONS: dict[str, dict[str, str]] = {
    "01-anatomie-agent": {
        "alt": "Boucle Perceive → Think → Act d'un agent IA",
        "legend": "Trois boîtes en boucle : PERCEIVE (observations), THINK (raisonnement), ACT (outil ou réponse). Le LLM est dans la boucle de contrôle — pas un pipeline fixe.",
        "phrase": "Un agent = LLM qui choisit ses actions jusqu'à l'objectif.",
    },
    "02-tool-use-function-calling": {
        "alt": "Flux tool use en quatre étapes",
        "legend": "1) Schéma JSON du tool → 2) le LLM choisit nom + args → 3) exécution → 4) observation réinjectée. Description claire + validation + structured output.",
        "phrase": "Le tool use transforme un LLM en acteur dans le monde réel.",
    },
    "03-memory-state": {
        "alt": "Trois couches de mémoire + checkpoint",
        "legend": "Short-term (fenêtre de contexte), working memory (scratchpad/state), long-term (vector store). À côté : checkpoint pour sauvegarder et reprendre l'état.",
        "phrase": "Sans mémoire structurée, l'agent redécouvre tout à chaque tour.",
    },
    "04-planning-reasoning": {
        "alt": "Quatre patterns de planning du simple au cher",
        "legend": "CoT (1 passe) → ReAct (boucle outils) → Plan-and-Execute (2 phases) → ToT/Reflexion (plus cher). Commencer simple, monter seulement si besoin.",
        "phrase": "Plus de raisonnement n'est pas toujours mieux — mesure le ROI tokens/qualité.",
    },
    "05-langgraph-fondamentaux": {
        "alt": "StateGraph LangGraph : START agent tools END",
        "legend": "START → nœud agent → edge conditionnel needs_tool : oui vers tools puis retour agent, non vers END. Le state (dict) circule ; HITL = interrupt.",
        "phrase": "LangGraph = Redux pour agents : nodes purs + state + edges.",
    },
    "06-langgraph-avance": {
        "alt": "Quatre leviers LangGraph avancés",
        "legend": "Subgraphs (sous-agents), parallèle, persistence (checkpointer), time-travel (rejouer un checkpoint). Sans persistence, crash = tout perdre.",
        "phrase": "La persistence transforme un démo en système opérable.",
    },
    "07-agent-complet": {
        "alt": "Pipeline agent complet cinq étapes",
        "legend": "Question → Plan → Tools → Mémoire → Synthèse. Capstone semaine 1 : recherche + analyse avec erreurs gérées et state checkpointé.",
        "phrase": "Assembler les briques vaut mieux que perfectionner une seule.",
    },
    "08-rag-agentique": {
        "alt": "RAG vanilla versus RAG agentique",
        "legend": "Vanilla : query → embed → top-k → LLM (une passe). Agentique : décompose, route, grade, multi-hop, re-query si insuffisant.",
        "phrase": "RAG agentique = la recherche est un raisonnement, pas un lookup.",
    },
    "09-multi-agent-patterns": {
        "alt": "Quatre patterns multi-agent",
        "legend": "Supervisor (chef + workers), Swarm (handoffs peer-to-peer), Hiérarchie, Débat (critique croisée). Commencer single-agent.",
        "phrase": "Multi-agent seulement si les rôles sont vraiment distincts.",
    },
    "10-mcp": {
        "alt": "Architecture MCP host client server",
        "legend": "Host/Agent → MCP Client → MCP Server exposant tools, resources, prompts (FS, DB, APIs). Un protocole, plusieurs hosts.",
        "phrase": "MCP = le USB des LLMs pour brancher des capacités.",
    },
    "11-evaluation-testing": {
        "alt": "Pyramide d'évaluation d'un agent",
        "legend": "Base large : unit tools/prompts. Puis trajectory tests. Puis E2E + LLM-as-judge. Sommet : prod/online (drift, feedback).",
        "phrase": "Sans tests de trajectoire, tu ne sais pas comment l'agent a « réussi ».",
    },
    "12-production-observabilite": {
        "alt": "Quatre piliers d'observabilité prod",
        "legend": "Traces (nœuds & tools), coûts (tokens), latence (p50/p95), recovery (retry, fallback, HITL). Voir la trajectoire, pas seulement le log final.",
        "phrase": "Ce qui n'est pas tracé n'est pas débuggable en prod.",
    },
    "13-securite-robustesse": {
        "alt": "Surfaces d'attaque et défense en profondeur",
        "legend": "Menaces : prompt injection, tool abuse, exfiltration, boucles infinies. Défense : whitelist tools, sandbox, validation args, rate limits, HITL.",
        "phrase": "Ne jamais traiter le contenu récupéré comme une instruction de confiance.",
    },
    "14-capstone": {
        "alt": "Capstone multi-agent supervisor et workers",
        "legend": "Supervisor qui délègue à Researcher, Analyst, Critic, Writer. Livrable : runnable + traces + tests de trajectoire + garde-fous.",
        "phrase": "Production-ready = happy path + erreurs + timeouts + cas ambigus.",
    },
    "15-context-engineering-compaction": {
        "alt": "Context rot, compaction et offloading",
        "legend": "Sans curation : historique monstrueux, coût ↗ qualité ↘. Compaction = résumer. Offloading = scratchpad hors fenêtre + budget par sous-agent.",
        "phrase": "La fenêtre de contexte est une ressource rare à budgéter.",
    },
    "16-memoire-long-horizon": {
        "alt": "Mémoire épisodique sémantique procédurale",
        "legend": "Épisodique (événements), sémantique (faits), procédurale (skills). Scoring récence + importance + similarité ; consolidation par réflexion.",
        "phrase": "Long-horizon = se souvenir entre les sessions, pas seulement dans le chat.",
    },
    "17-verifiers-self-improvement": {
        "alt": "Boucle generate verifier select persist",
        "legend": "Generate N → Verifier (PRM/règles) → Select best-of-N → Persist les leçons. Outcome vs process reward.",
        "phrase": "S'améliorer entre les runs exige de persister les leçons.",
    },
    "18-orchestration-comparee-failure-modes": {
        "alt": "Comparaison frameworks d'orchestration agents",
        "legend": "LangGraph, CrewAI, AutoGen, OpenAI SDK, Swarm/ADK — trade-offs contrôle vs vitesse. Failure modes : loops, ping-pong, contexte pollué, coût.",
        "phrase": "Choisir selon le contrôle requis, pas la hype du framework.",
    },
    "19-protocoles-inter-agents": {
        "alt": "A2A entre agents et complémentarité MCP",
        "legend": "Agent A et Agent B (vendeurs différents) via A2A/ACP (découverte, cards, confiance). MCP reste pour brancher des tools.",
        "phrase": "MCP = capacités · A2A = collaboration entre agents hétérogènes.",
    },
    "20-durable-event-driven-agents": {
        "alt": "Checkpoint versus durable execution",
        "legend": "Checkpoint : snapshot d'état pour debug/HITL. Durable (Temporal…) : workflow rejouable, crash machine ≠ perte, timers et events.",
        "phrase": "Longs runs et reprises fiables → durable execution, pas seulement un checkpointer.",
    },
    "21-coding-agents-architecture": {
        "alt": "Boucle SEARCH EDIT RUN OBSERVE d'un coding agent",
        "legend": "SEARCH le code → EDIT un patch → RUN tests/CLI → OBSERVE logs, en boucle jusqu'aux tests verts. SWE-bench évalue cette boucle.",
        "phrase": "Un coding agent est un utilisateur automatisé du dépôt (ACI).",
    },
    "22-computer-use-gui-agents": {
        "alt": "Boucle GUI screenshot grounding action",
        "legend": "Screenshot → grounding (set-of-marks, coords) → action (click/type/scroll). Grounding fragile : sandbox obligatoire.",
        "phrase": "Piloter un écran = perception + grounding fiable + isolation.",
    },
    "23-sandboxing-execution-sure": {
        "alt": "Quatre couches de sandboxing infra",
        "legend": "Process limité → container/cgroups → gVisor/microVM → egress réseau filtré. Plus bas dans la pile = plus sûr et plus cher.",
        "phrase": "J13 = principes · J23 = comment l'infra isole vraiment.",
    },
    "24-inference-engineering": {
        "alt": "Structured outputs, routing et caching",
        "legend": "Structured outputs (tool calls fiables), model routing (petit modèle si simple → −coût), prompt caching (préfixe réutilisé → −latence).",
        "phrase": "Fiabilité des appels + coût + latence se pilotent ensemble.",
    },
    "25-serving-stateful-sessions": {
        "alt": "Serving : clients, workers stateless, checkpointer",
        "legend": "Clients → workers stateless × K → checkpointer partagé (Postgres/Redis). Online eval, drift, limites de session.",
        "phrase": "Scaler = externaliser l'état, garder les workers interchangeables.",
    },
    "26-benchmarking-pratique": {
        "alt": "Pipeline harness pass^k",
        "legend": "Cas de test → run × k → score → pass^k → rapport vs baseline. pass^k = proba d'au moins un succès sur k essais.",
        "phrase": "Évalue TON agent avec un harness, pas seulement les leaderboards publics.",
    },
    "27-capstone-architecture": {
        "alt": "Architecture deep ops agent en huit briques",
        "legend": "Ingest, Planner, Workers sandboxed, Verifier, Memory long-horizon, Durable, Observability, Eval harness.",
        "phrase": "L'architecture avant le code : contrats clairs entre briques.",
    },
    "28-capstone-build-eval": {
        "alt": "Quatre étapes du capstone build et eval",
        "legend": "1 Build runnable → 2 scénario bug→fix → 3 crash test reprise → 4 eval pass^k + rapport. Done = démo live + métriques.",
        "phrase": "Si tu ne peux pas rejouer et expliquer un échec, ce n'est pas prêt.",
    },
    "parcours-28j": {
        "alt": "Carte du parcours agentic-ai 28 jours en 4 semaines",
        "legend": "S1 J1–J7 fondations agent + LangGraph · S2 J8–J14 multi-agent + prod · S3 J15–J21 frontier · S4 J22–J28 scale + capstone.",
        "phrase": "28 jours : du single-agent au deep ops évalué.",
    },
    # Schémas corps (pas injectés en tête de module — remplacent ASCII denses)
    "08-rag-pipeline-complet": {
        "alt": "Pipeline RAG agentique complet",
        "legend": "Query → decomposer → router → sources → grader (reformulate si besoin) → next hop ? → synthesizer → answer.",
        "phrase": "Le LLM pilote décomposition, routing, grading et multi-hop avant la synthèse.",
    },
    "14-capstone-pipeline": {
        "alt": "Pipeline capstone de production",
        "legend": "Input guardrail → rate limit → budget → supervisor fan-out (researcher, analyzer, writer) → output guardrail → HITL → rapport.",
        "phrase": "Guardrails et budget avant le supervisor ; workers puis HITL optionnel.",
    },
    "03-memory-production": {
        "alt": "Architecture mémoire d'un agent en production",
        "legend": "Context window + working memory alimentent le LLM ; tools et checkpoint à côté ; vector store et key-value en bas.",
        "phrase": "Trois mémoires dans la boucle, plus stores externes pour le long terme.",
    },
    "13-defense-profondeur": {
        "alt": "Défense en profondeur — cinq couches",
        "legend": "L1 input guardrails → L2 trust boundaries → L3 tool guardrails → L4 output → L5 monitoring / kill switch.",
        "phrase": "Empile input, trust, tools, output et monitoring : une seule couche ne suffit pas.",
    },
    "07-research-agent-archi": {
        "alt": "Architecture du research agent",
        "legend": "START → planner → executor (boucle replan) → analyzer → END ; state question/plan/mémoires/findings.",
        "phrase": "Planifier, exécuter avec replan, analyser, synthétiser — state et tools partagés.",
    },
    "11-eval-pipeline": {
        "alt": "Pipeline d'évaluation Dev → Prod",
        "legend": "Quatre étages en cascade : DEV → CI (bloque PR) → NIGHTLY → PROD online.",
        "phrase": "Unit tests, CI, nightly large, puis monitoring live sur le traffic réel.",
    },
    # Lot B/C
    "09-hierarchical": {
        "alt": "Pattern multi-agent hiérarchique",
        "legend": "CEO agent délègue à des sub-supervisors qui pilotent des workers spécialisés.",
        "phrase": "Hiérarchie = org métier en couches ; borner la profondeur.",
    },
    "09-debate": {
        "alt": "Pattern débat multi-agent",
        "legend": "Agents A B C s'échangent critiques sur plusieurs rounds, puis un merge/judge produit la réponse.",
        "phrase": "Le débat améliore la qualité si les rounds sont bornés.",
    },
    "09-swarm-handoff": {
        "alt": "Pattern Swarm handoff",
        "legend": "User vers agent triage, handoff vers agent spécialisé, puis éventuellement review — contrôle peer-to-peer.",
        "phrase": "Swarm = handoffs ; attention aux boucles A↔B.",
    },
    "09-decision-tree": {
        "alt": "Arbre décision single vs multi-agent",
        "legend": "Si tâche simple : single agent. Sinon, multi seulement si plusieurs rôles ou besoin de critique parallèle.",
        "phrase": "Commence single. Multi seulement si rôles vraiment distincts.",
    },
    "06-agent-prod-complet": {
        "alt": "Patterns LangGraph combinés production",
        "legend": "Main graph avec router, subgraph workers en parallèle, merge, HITL, le tout sur checkpointer.",
        "phrase": "Subgraphs + parallèle + persistence + HITL = agent opérable.",
    },
    "06-parallel-vs-seq": {
        "alt": "Exécution séquentielle vs parallèle",
        "legend": "À gauche chaîne séquentielle coûteuse ; à droite fan-out Send API puis merge.",
        "phrase": "Paralléliser seulement des sous-tâches indépendantes.",
    },
    "16-main-vs-external": {
        "alt": "Main context versus external context",
        "legend": "À gauche la fenêtre LLM (system, working, tools). À droite archival, recall et fichiers externes.",
        "phrase": "MemGPT : pager le contexte comme un OS.",
    },
    "16-memory-flow": {
        "alt": "Flux mémoire agent long-horizon",
        "legend": "Pipeline événement, scoring, stockage, retrieval et injection dans le contexte.",
        "phrase": "Score → store → retrieve → inject, en boucle.",
    },
    "10-mcp-lifecycle": {
        "alt": "Lifecycle connexion MCP",
        "legend": "Quatre étapes : launch, initialize, list des capacités, call d'outil avec résultat.",
        "phrase": "Découvrir les capacités avant d'invoquer.",
    },
    "04-plan-execute": {
        "alt": "Pattern Plan-and-Execute",
        "legend": "Question, planner qui produit des étapes, executor outillé, réponse ; boucle de replan si besoin.",
        "phrase": "Planifier puis exécuter : le pattern production.",
    },
    "04-tree-of-thought": {
        "alt": "Tree-of-Thought",
        "legend": "Arbre de pensées avec scores : branches gardées ou élaguées jusqu'à la meilleure feuille.",
        "phrase": "ToT = explorer et élaguer — cher, réservé au dur.",
    },
    "22-perceive-mark-act": {
        "alt": "Boucle perceive mark act GUI",
        "legend": "Screenshot ou DOM, grounding Set-of-Marks, action souris clavier, en boucle.",
        "phrase": "GUI agent = percevoir, ancrer, agir.",
    },
    "22-set-of-marks": {
        "alt": "Set-of-Marks prompting",
        "legend": "Comparaison screenshot brut versus éléments numérotés pour guider les actions du LLM.",
        "phrase": "Numéroter les cibles rend le grounding actionnable.",
    },
    "25-stateful-problem": {
        "alt": "Stateful vs workers stateless",
        "legend": "Anti-pattern workers avec état local ; cible workers interchangeables et checkpointer partagé.",
        "phrase": "Scaler = externaliser l'état hors du worker.",
    },
    "03-hybrid-memory": {
        "alt": "Hybrid memory summary plus fenêtre",
        "legend": "Résumé des anciens messages à gauche, fenêtre récente à droite, assemblés pour le LLM.",
        "phrase": "Summary + fenêtre récente = compromis coût/fidélité.",
    },
    "08-hybrid-rrf": {
        "alt": "Hybrid retrieval RRF rerank",
        "legend": "Query vers dense et sparse, fusion RRF, puis cross-encoder rerank avant le LLM.",
        "phrase": "Dense + sparse + rerank : le retrieval prod.",
    },
    "14-deployment": {
        "alt": "Stack déploiement capstone",
        "legend": "Couches clients, FastAPI SSE, runtime agent, stores et observabilité.",
        "phrase": "API + runtime + stores + obs : le squelette deploy.",
    },
    "19-a2a-task-lifecycle": {
        "alt": "Lifecycle tâche A2A",
        "legend": "États submitted, working, completed, avec branches input-required, failed et canceled.",
        "phrase": "A2A orchestre des tâches entre agents avec états explicites.",
    },
    "20-durable-combined": {
        "alt": "Architecture durable event-driven HITL",
        "legend": "Workflow durable avec activities LLM et tools, attente signal HITL, alimenté par un bus d'événements.",
        "phrase": "Long run fiable = durable workflow + events + HITL.",
    },
    "24-three-levers": {
        "alt": "Trois leviers inference engineering",
        "legend": "Requête, model router, puis structured outputs et prompt caching vers la réponse.",
        "phrase": "Router, contraindre, cacher : fiabilité et coût.",
    },
    "23-isolation-layers": {
        "alt": "Couches d'isolation sandbox",
        "legend": "Sept couches empilées de l'audit log jusqu'aux limites process.",
        "phrase": "Plus bas dans la pile = plus isolé (et plus cher).",
    },
    "15-context-isolation": {
        "alt": "Isolation contexte sous-agent",
        "legend": "Superviseur à gros historique qui délègue un prompt minimal à un sous-agent.",
        "phrase": "Déléguer le minimum utile, pas tout le contexte.",
    },
    "17-self-refine-loop": {
        "alt": "Boucle Self-Refine verifier",
        "legend": "Generator, verifier, refiner, select, avec boucle de retry bornée et leçons optionnelles.",
        "phrase": "Générer puis vérifier bat le simple retry aveugle.",
    },
}

# Slugs injectés en tête de module (1 hero par fichier theory).
# Les schémas corps restent manuels dans le MD.
HERO_SLUGS: frozenset[str] = frozenset(
    {
        "01-anatomie-agent",
        "02-tool-use-function-calling",
        "03-memory-state",
        "04-planning-reasoning",
        "05-langgraph-fondamentaux",
        "06-langgraph-avance",
        "07-agent-complet",
        "08-rag-agentique",
        "09-multi-agent-patterns",
        "10-mcp",
        "11-evaluation-testing",
        "12-production-observabilite",
        "13-securite-robustesse",
        "14-capstone",
        "15-context-engineering-compaction",
        "16-memoire-long-horizon",
        "17-verifiers-self-improvement",
        "18-orchestration-comparee-failure-modes",
        "19-protocoles-inter-agents",
        "20-durable-event-driven-agents",
        "21-coding-agents-architecture",
        "22-computer-use-gui-agents",
        "23-sandboxing-execution-sure",
        "24-inference-engineering",
        "25-serving-stateful-sessions",
        "26-benchmarking-pratique",
        "27-capstone-architecture",
        "28-capstone-build-eval",
    }
)
