# J18 — Orchestration comparee : LangGraph vs CrewAI vs AutoGen vs OpenAI Agents SDK & multi-agent failure modes

> **Temps estime** : 3h | **Prerequis** : J1-J17 (surtout J5/J6 pour LangGraph, J9 pour les patterns multi-agents)
> **Objectif** : comparer les 5 grands frameworks d'orchestration sur leurs modeles d'execution, savoir choisir lequel deployer selon le contexte, et identifier les failure modes classiques du multi-agent pour ne pas reproduire les erreurs couteuses.

---

## 1. Pourquoi comparer les frameworks ?

En 2024-2025, l'ecosysteme multi-agent a explose. Chaque framework propose une abstraction differente du meme probleme fondamental : **comment coordonner plusieurs LLM-calls et tools de facon fiable ?**

Le choix du framework n'est pas anodin — il dicte :
- La **complexite operationnelle** (courbe d'apprentissage, debug)
- La **flexibilite architecturale** (graphe vs roles vs acteurs)
- Le **cout en tokens** (etat partage vs copies, verbosity des prompts)
- La **capacite de reprise sur erreur** (checkpointing, replay)

> **Rappel J5/J6** : LangGraph fondamentaux et avance. **Rappel J9** : patterns supervisor / swarm / hierarchical / debate. Ce module se concentre sur le *comparatif inter-frameworks* et sur *pourquoi le multi-agent casse*.

---

## 2. Les 5 frameworks en bref

### 2.1 LangGraph (LangChain, 2024)

**Modele d'execution** : graphe oriente d'etats. Chaque noeud est une fonction Python, chaque arete est une transition (conditionnelle ou fixe). L'etat est un `TypedDict` partage entre tous les noeuds.

**Points forts** :
- Controle total du flux (boucles, branchements, sous-graphes)
- **Checkpointing** natif (SQLite, Redis, Postgres) — reprise apres crash
- Streaming token-by-token integre
- Subgraphs pour hierarchie d'agents

**Limites** :
- Verbeux a initialiser pour des cas simples
- Concept "graphe" moins intuitif pour les profils non-engineers

### 2.2 CrewAI (CrewAI Inc., 2024)

**Modele d'execution** : role-based crews. On definit des `Agent` (role, backstory, tools) et des `Task` (description, expected output, agent assignee). Le `Crew` execute les tasks en mode `sequential` (une apres l'autre) ou `hierarchical` (un manager LLM delegue).

**Points forts** :
- Abstraction haut niveau tres lisible (proche du langage metier)
- Onboarding rapide — quelques lignes suffisent
- Mode hierarchical avec manager LLM automatique

**Limites** :
- Moins de controle fin sur le flux (boite noire sous le capot)
- Etat inter-tasks passe via le contexte des task outputs (stringifie)
- Debug difficile quand le manager LLM prend de mauvaises decisions

### 2.3 AutoGen 0.4 (Microsoft Research, jan. 2025)

**Modele d'execution** : event-driven / actor model. Chaque agent est un acteur asynchrone qui envoie et recoit des **messages types** via un runtime. Deux layers :
- **Core** : acteurs bas niveau, full async, aucune opinion sur les LLMs
- **AgentChat** : surcouche opinee (AssistantAgent, UserProxyAgent, GroupChat)

**Points forts** :
- Scalabilite : les acteurs tournent en parallele sans partage memoire
- Tres decoupled — echange des LLMs sans refactoring du flux
- Suites de test et observabilite natives

**Limites** :
- Rupture de compatibilite v0.2 → v0.4 significative
- Courbe d'apprentissage elevee pour le Core async
- AgentChat plus simple mais moins flexible que LangGraph

### 2.4 OpenAI Agents SDK (OpenAI, 2025)

**Modele d'execution** : handoffs tool-centric. Un `Agent` est une entite LLM + tools. La coordination se fait via des **handoffs** : un agent delegue le controle a un autre agent via un appel de tool special. Pas d'etat partage central — l'etat passe dans les messages de la conversation.

**Points forts** :
- Integre native avec l'API OpenAI (parallel tool calls, structured outputs)
- Tres leger — faible overhead d'abstraction
- `Runner.run()` gere automatiquement le loop LLM → tool → LLM
- Guardrails integres (input/output validation)

**Limites** :
- Vendor lock-in OpenAI (bien qu'un adaptateur Anthropic existe)
- Handoffs stateless : l'etat doit etre reconstruit depuis la conversation history
- Moins adapte aux pipelines complexes avec branches multiples

### 2.5 OpenAI Swarm (OpenAI, archive 2024)

**Modele d'execution** : stateless handoffs, predecessor de l'Agents SDK. Chaque `Agent` passe le contexte via des variables (`context_variables`) et peut transferer le controle a un autre agent en retournant un objet `Agent`.

**Statut** : **archive, remplace par l'Agents SDK**. Ne pas utiliser pour de nouveaux projets.

> **Pourquoi le mentionner ?** Swarm a popularise le concept de handoffs legers. Beaucoup de ressources en ligne l'utilisent encore. Comprendre Swarm aide a comprendre l'Agents SDK.

### 2.6 Google ADK — Agent Development Kit (Google, 2025)

**Modele d'execution** : hybride **workflow agents deterministes** + **LLM agents dynamiques**. ADK est un framework open-source (Python/Java) code-first, deployable sur Vertex AI Agent Engine (J25). Sa signature : exposer les topologies multi-agents comme des **classes pretes a l'emploi** plutot que comme du graphe a cabler soi-meme.

- **`SequentialAgent`** : pipeline en chaine. La sortie de chaque sous-agent est ecrite dans le `state` partage via une cle `output_key`, lue par le suivant.
- **`ParallelAgent`** : execute des sous-agents independants en concurrence (fan-out), puis fan-in.
- **`LoopAgent`** : repete un sous-agent jusqu'a une condition d'arret (raffinement iteratif).
- **LLM-driven routing** : pour le comportement adaptatif, un agent LLM decide dynamiquement du sous-agent a appeler (equivalent d'un supervisor, J9).

**Points forts** :
- **Workflow agents types** : Sequential / Parallel / Loop comme primitives nommees — la topologie est explicite dans le type, pas implicite dans le cablage du graphe.
- DevX soignee : CLI + Web UI locale pour inspecter pas-a-pas le state, les events et l'execution.
- Built-in tools manages (Google Search, **Code Execution**, RAG Engine), agent-as-a-tool, long-running tools async.
- Securite en couches via **plugins au niveau runner** (policies globales : checks pre-LLM, tool execution policy-enforced, validation post-LLM).

**Limites** :
- Pleinement integre quand on deploie sur **Google Cloud / Vertex AI Agent Engine** ; hors de cet ecosysteme, on perd le runtime manage (J25).
- Plus jeune que LangGraph ; ecosysteme d'integrations tierces encore en construction.
- Le modele cible est **Gemini** (multi-modele possible, mais le chemin doux reste Google).

> **Le clivage a retenir (transferable hors ADK)** : *workflow agents deterministes* (Sequential/Parallel/Loop — tu sais a l'avance qui s'execute et dans quel ordre) **vs** *LLM-driven dynamic routing* (le LLM choisit a la volee). C'est une grille de design utile dans **tous** les frameworks : LangGraph te laisse coder les deux, ADK les nomme. Choisis le deterministe quand l'ordre est connu (moins cher, plus debuggable) ; le dynamique quand l'ordre depend de l'input.

---

## 3. Table de decision — quel framework choisir ?

| Critere | **LangGraph** | **CrewAI** | **AutoGen 0.4** | **OpenAI Agents SDK** | **Swarm** | **Google ADK** |
|---|---|---|---|---|---|---|
| **Modele d'execution** | Graphe d'etats | Role-based crew | Event-driven / acteurs | Handoffs tool-centric | Handoffs stateless | Workflow agents types (Seq/Par/Loop) + LLM-driven |
| **Etat / persistence** | TypedDict partage + checkpointing natif | Task outputs stringifies | Messages types par acteur | Conversation history | context_variables dict | `state` partage via `output_key` + sessions (Agent Engine) |
| **Handoff / delegation** | Edges conditionnels + subgraphs | Manager LLM ou sequence | Messages inter-acteurs | Tool `transfer_to_X` | Return `Agent` object | Sous-agents types + agent-as-tool + routing LLM |
| **Parallelisme** | Send() pour branches paralleles | Limite (sequential/hierarchical) | Natif (async acteurs) | Parallel tool calls | Non | Natif (`ParallelAgent`) |
| **Courbe d'apprentissage** | Moyenne (graphe a apprendre) | Faible (role/task intuitif) | Elevee (actor model async) | Faible-Moyenne | Faible (archive) | Moyenne (classes claires + Web UI) |
| **Debug / observabilite** | LangSmith, streaming natif | Limitee | OpenTelemetry natif | Traces OpenAI | Aucune | Web UI locale + traces ; Agent Engine en prod |
| **Vendor lock-in** | LangChain ecosystem | Aucun | Aucun | OpenAI fort | OpenAI fort | Google Cloud fort (runtime manage Vertex) |
| **Checkpointing / reprise** | Natif | Non | Non (a implanter) | Non | Non | Via Agent Engine sessions (manage) |
| **Quand choisir** | Pipelines complexes, long-running, besoin de reprise | Automatisation de workflows metier en quelques heures | Multi-agent distribue, scalabilite, recherche | Apps OpenAI, handoffs simples, prototypage rapide | Ne pas choisir (archive) | Stack Google Cloud / Gemini, topologies typees + runtime manage |

> **Analogie** : LangGraph = un chef de projet avec un organigramme precis. CrewAI = une equipe de freelances avec des fiches de poste. AutoGen = un reseau d'agents autonomes qui se passent des notes. OpenAI Agents SDK = un standard telephonique qui transfere l'appel au bon service. ADK = une chaine de montage avec des postes types (sequentiel / parallele / boucle) qu'on assemble comme des Lego.

---

## 4. Multi-agent failure modes — pourquoi ca casse

### 4.1 Propagation d'erreurs en cascade

Dans un pipeline A → B → C, si B produit une output incorrecte, C l'amplifie. Le LLM final peut avoir confiance dans une donnee corrompue parce qu'elle vient "d'un autre agent".

**Symptome** : l'erreur finale est loin de la source, le debug est difficile.

**Defense** : validation de schema entre agents, checkpointing, retry avec backoff.

### 4.2 Desaccords non resolus (agent debate boucle infinie)

Deux agents avec des instructions contradictoires peuvent entrer dans une boucle : A contredit B, B contredit A, etc. Sans mecanisme de resolution (arbitre, vote, max_turns), le cout en tokens explose.

**Symptome** : la conversation agent-to-agent ne converge pas, le budget tokens est consomme sans resultat.

**Defense** : `max_turns` strict, agent arbitre, ou escalade vers un humain apres N echanges.

### 4.3 Explosion de cout / tokens

Chaque agent dans une chaine reinjeste le contexte complet des agents precedents. Pour N agents et un contexte de T tokens par etape, le cout total peut atteindre O(N * T). Les frameworks verbose (AgentChat GroupChat) aggravent ce phenomene en prefixant chaque message du nom de l'agent et des metadonnees.

**Exemple reel** : un pipeline de 5 agents avec 10k tokens de contexte chacun = 50k tokens par execution. A $15/M tokens (GPT-4o), 1000 executions/jour = $750/jour.

**Defense** : context truncation, summarization intermediaire, shared memory (J10), et surtout... remettre en question si le multi-agent est justifie.

### 4.4 Hallucination cross-agent amplifiee

Un agent hallucine un fait. L'agent suivant le cite comme source. L'agent final le presente comme verifie. L'erreur s'est **legitimisee** en traversant des couches d'agents.

**Defense** : grounding sur des sources externes (RAG, tools de recherche), validation par un agent critique dédié.

### 4.5 Latence additive

Si les agents s'executent sequentiellement, les latences s'additionnent. Un pipeline de 5 agents a 3s chacun = 15s de latence perçue. Les utilisateurs attendent.

**Defense** : paralleliser quand les taches sont independantes (Send() dans LangGraph, acteurs AutoGen), reunir les resultats avec un fan-in.

---

## 5. Le debat : "un agent bien outille > N agents"

### 5.1 La position Cognition / "Don't Build Multi-Agents"

Walden Yan (Cognition, 2025) argumente que la plupart des use cases qu'on confie a N agents seraient mieux servis par **un seul agent avec plus de tools et un meilleur systeme de planification**.

Les arguments :
- Chaque handoff est un point de failure potentiel
- La coordination consomme des tokens sans produire de valeur
- Un agent unique a acces a tout le contexte, sans perte a la translation
- Les bugs inter-agents sont les plus difficiles a reproduire et corriger

**Corollaire pratique** : avant de creer un deuxieme agent, demande-toi si un tool supplementaire suffira.

### 5.2 Les cas ou le multi-agent est justifie (Anthropic, 2024)

Anthropic identifie 3 patterns ou le multi-agent apporte une valeur reelle :

1. **Taches parallelisables** : des sous-taches independantes qui beneficient d'une execution en parallele (ex : analyser 10 documents simultanement)
2. **Specialisation** : des domaines si differents qu'un seul prompt system ne peut pas bien les couvrir (ex : agent code + agent securite + agent documentation)
3. **Verification croisee** : un agent produit, un autre critique — le desaccord structure produit de meilleurs resultats que l'auto-critique

**La bonne question** : *est-ce que la complexite de coordination est compensee par un gain reel en qualite, vitesse ou cout ?*

### 5.3 Heuristique pratique

```
si tache_parallelisable:
    → multi-agent justifie (fan-out / fan-in)
si domaines_tres_differents ET context_window_sature:
    → multi-agent justifie (specialisation)
si besoin_verification_independante:
    → multi-agent justifie (critique/verifier pattern)
sinon:
    → single agent + tools (plus simple, moins cher, plus debuggable)
```

---

## 6. Patterns de resilience inter-frameworks

Quelques patterns qui s'appliquent independamment du framework choisi :

| Pattern | Description | Implementation |
|---|---|---|
| **Retry with backoff** | Relance l'appel LLM en cas d'erreur transitoire | Decorator sur l'appel LLM, max 3 retries |
| **Schema validation** | Valide la sortie d'un agent avant de la passer au suivant | Pydantic / TypedDict / JSON schema |
| **Budget guard** | Coupe le pipeline si le compteur de tokens depasse un seuil | Compteur global, exception si depassement |
| **Max turns** | Limite les echanges dans un debate/loop | Parametre dans le crew / graphe / runtime |
| **Summarization intermediaire** | Resumer le contexte avant de le passer a l'agent suivant | Noeud "summarize" dans le graphe |
| **Fallback to single agent** | Si le pipeline multi-agent echoue N fois, retry en single-agent | Circuit breaker pattern |

---

## Flash-cards

**Q1 :** Quelle est la difference fondamentale entre le modele d'etat de LangGraph et celui de CrewAI ?
> **R :** LangGraph maintient un `TypedDict` partage et persiste entre les noeuds avec checkpointing. CrewAI passe l'etat entre tasks via les outputs stringifies (texte brut), sans persistence native.

**Q2 :** Pourquoi AutoGen 0.4 est-il dit "event-driven" et en quoi ca change la scalabilite ?
> **R :** Chaque agent est un acteur asynchrone qui communique par messages types. Les acteurs s'executent en parallele sans partage memoire, donc on peut scaler horizontalement sans locks — contrairement aux graphes sequentiels ou role-based.

**Q3 :** Qu'est-ce qu'un handoff dans l'OpenAI Agents SDK et en quoi differe-t-il d'un edge LangGraph ?
> **R :** Un handoff est un tool special qui transfère le controle de l'agent courant vers un autre agent, en passant l'historique de conversation. Un edge LangGraph est une transition deterministe (ou conditionnelle) entre noeuds d'un graphe qui partagent un etat commun structure — plus de controle, plus de verbosity.

**Q4 :** Cite 2 failure modes multi-agent et leur defense associee.
> **R :** (1) Propagation d'erreurs en cascade → validation de schema entre agents + checkpointing. (2) Desaccords non resolus → `max_turns` strict + agent arbitre + escalade humaine.

**Q5 :** Selon l'heuristique pratique J18, quand le multi-agent est-il justifie vs un agent unique bien outille ?
> **R :** Multi-agent justifie pour : taches parallelisables (fan-out), domaines tres differents saturant le context window, verification croisee independante. Sinon : single agent + tools (plus simple, moins cher, plus debuggable).

**Q6 :** Qu'est-ce qui distingue les *workflow agents* d'ADK et quel clivage de design en retenir ?
> **R :** ADK expose les topologies comme des classes pretes a l'emploi : `SequentialAgent` (chaine via `output_key`), `ParallelAgent` (fan-out concurrent), `LoopAgent` (iteration jusqu'a condition d'arret). Le clivage transferable : *workflow deterministe* (ordre connu a l'avance — moins cher, plus debuggable) vs *LLM-driven routing* (le LLM choisit a la volee — adaptatif). LangGraph laisse coder les deux ; ADK les nomme.

---

## Points cles a retenir

- **LangGraph** = controle maximal + checkpointing → pipelines complexes long-running
- **CrewAI** = onboarding rapide role-based → automatisation metier simple
- **AutoGen 0.4** = actor model async → multi-agent distribue / recherche
- **OpenAI Agents SDK** = handoffs legers → apps OpenAI, prototypage rapide
- **Swarm** = archive, ne pas utiliser pour de nouveaux projets
- **Google ADK** = workflow agents types (Sequential/Parallel/Loop) + runtime manage Vertex → stack Google Cloud / Gemini ; retenir le clivage *deterministe vs LLM-driven*
- Les failure modes principaux : cascade d'erreurs, boucles de desaccord, explosion de cout tokens, hallucination cross-agent amplifiee, latence additive
- Avant de creer un 2e agent, demande-toi si un tool supplementaire suffira (Cognition)
- Multi-agent justifie dans 3 cas : parallelisme, specialisation, verification croisee (Anthropic)

---

## Pour aller plus loin

- LangChain, "LangGraph — Multi-agent" (2025) — https://docs.langchain.com/oss/python/langchain/multi-agent
- Microsoft Research, "AutoGen v0.4: Reimagining the Foundation of Agentic AI for Scale, Extensibility, and Robustness" (jan. 2025) — https://www.microsoft.com/en-us/research/blog/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/
- OpenAI, "Agents SDK — Handoffs" — https://openai.github.io/openai-agents-python/handoffs/
- OpenAI, "swarm" (archive, remplace par l'Agents SDK) — https://github.com/openai/swarm
- CrewAI docs — https://docs.crewai.com/
- Walden Yan (Cognition), "Don't Build Multi-Agents" (2025) — https://cognition.ai/blog/dont-build-multi-agents
- Google, "Agent Development Kit (ADK) — Easy to build multi-agent applications" (2025) — https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/
- Google, "ADK docs — Workflow agents (Sequential / Parallel / Loop)" — https://google.github.io/adk-docs/agents/workflow-agents/
