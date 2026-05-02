# J9 — Multi-Agent Patterns : supervisor, swarm, debat, hierarchie

> **Temps estime** : 3h | **Prerequis** : J1-J8
> **Objectif** : savoir quand et comment faire collaborer plusieurs agents specialises, connaitre les 4 patterns de reference, et arbitrer entre complexite et qualite.

---

## 1. Un seul agent ne suffit pas toujours

Tu as construit un agent qui peut utiliser des outils, raisonner, planifier. Il marche bien sur des taches simples. Puis arrive une tache complexe :

> "Analyse la documentation de notre API, trouve les 3 endpoints les moins performants, propose des optimisations et ecris un rapport technique."

Un seul agent va :
- Se perdre dans les responsabilites (analyse ? code ? redaction ?)
- Melanger les prompts systemes (analyste vs developpeur vs redacteur)
- Degrader en qualite sur au moins une des competences
- Exploser sa context window avec tous les outils charges en meme temps

**La solution** : **specialiser**. Plusieurs agents, chacun expert dans son domaine, qui collaborent.

### Quand le multi-agent bat le single agent

| Situation | Single agent | Multi-agent |
|-----------|--------------|-------------|
| Tache simple (< 10 etapes) | Parfait | Sur-ingenierie |
| Tache composee (2-3 competences) | Decent | Meilleur |
| Tache complexe (4+ competences) | Degrade | Net avantage |
| Besoin d'expertise tres pointue | Un prompt general | Un prompt specialise par role |
| Ouputs verifiables par les pairs | Pas de critique | Critique croisee |
| Exploration parallele | Sequentiel | Parallele natif |

**Regle pragmatique** : commence avec un seul agent. **Passe au multi-agent quand** tu vois que le prompt commence a melanger des roles (ex: "tu es un analyste qui sait aussi coder et rediger"), ou quand tu as besoin d'exploration parallele.

---

## 2. Les 4 patterns de reference

Il y a une dizaine de variantes dans la literature, mais elles se ramenent toutes a 4 archetypes.

### 2.1 Supervisor pattern — chef + workers

**Principe** : un agent "chef" (supervisor) recoit la tache, la decoupe, delegue a des workers specialises, collecte les resultats, synthetise la reponse finale.

```
                    ┌─────────────┐
                    │  SUPERVISOR │   ← recoit la tache, decide qui fait quoi
                    └──────┬──────┘
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │ Researcher│    │   Coder   │    │  Writer  │
   └──────────┘    └──────────┘    └──────────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    ┌─────────────┐
                    │  SUPERVISOR │   ← synthese finale
                    └──────┬──────┘
                           ▼
                        ANSWER
```

**Avantages** :
- Controle centralise — facile a debugger
- Synthese coherente — le supervisor homogeneise le style
- Budget previsible — le supervisor decide quand arreter

**Inconvenients** :
- Goulot d'etranglement — toutes les decisions passent par le supervisor
- Latence additionnelle — aller-retours multiples
- Le supervisor doit etre bon a **router** (choisir le bon worker) ET a **synthetiser**

**Quand l'utiliser** : c'est le pattern par defaut. Commence par la. Il couvre 80% des cas.

**Implementation typique** :

```python
def supervisor_loop(task):
    plan = supervisor_llm(f"Decompose this task: {task}")
    results = {}
    for step in plan.steps:
        worker = select_worker(step.type)  # researcher, coder, writer...
        results[step.id] = worker(step.input)
    return supervisor_llm(f"Synthesize: {task} / results: {results}")
```

### 2.2 Hierarchical pattern — supervisors of supervisors

**Principe** : comme le supervisor mais sur plusieurs niveaux. Un supervisor de haut niveau delegue a des sub-supervisors, qui eux-memes delegent a des workers.

```
                     ┌──────────────┐
                     │  CEO agent   │
                     └──────┬───────┘
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │ Research│   │   Dev    │   │   Ops   │
        │  Manager│   │  Manager │   │  Manager│
        └────┬────┘   └────┬────┘   └────┬────┘
             │             │             │
         workers       workers       workers
```

**Avantages** :
- Scale a des taches tres complexes (dizaines d'agents)
- Division des responsabilites en "departements"
- Chaque sub-supervisor gere son domaine sans polluer les autres

**Inconvenients** :
- Latence encore plus grande (cascades d'aller-retours)
- Cout eleve (chaque niveau ajoute des appels LLM)
- Debugging difficile (ou ca a casse, a quel niveau ?)
- Sur-ingenierie dans 90% des cas

**Quand l'utiliser** : projets tres grands, long-running, avec > 15 agents. En pratique : assistants pour code reviews de monorepos, operations autonomes sur infrastructure, etc.

### 2.3 Collaborative debate — faire debattre des agents

**Principe** : plusieurs agents donnent leur avis sur le meme probleme, debattent, et convergent vers une reponse consensus (ou un arbitre tranche).

```
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Agent A  │   │ Agent B  │   │ Agent C  │   ← meme tache, 3 avis
  └────┬─────┘   └────┬─────┘   └────┬─────┘
       │              │              │
       └──────────────┼──────────────┘
                      ▼
              [round de debat]
                      │
                      ▼
              [nouveau round]
                      │
                      ▼
              ┌──────────────┐
              │   MODERATOR  │   ← tranche ou certifie le consensus
              └──────┬───────┘
                     ▼
                   ANSWER
```

**Comment ca marche concretement** :
1. Chaque agent produit une reponse initiale
2. On montre aux agents les reponses des autres
3. Chaque agent revise sa reponse (peut changer d'avis)
4. On repete N rounds
5. Un moderateur valide le consensus ou tranche en cas de desaccord

**Avantages** :
- Reduit les hallucinations — si 3 agents sont d'accord, c'est plus fiable qu'un seul
- Capture plusieurs perspectives
- Amplifie la qualite sur des taches subjectives (ethique, design, strategie)

**Inconvenients** :
- **Tres cher** — chaque round multiplie le cout par N agents
- Lent — N agents * M rounds d'appels LLM sequentiels
- Peut converger vers un mauvais consensus (biais groupthink)
- Debugging complexe

**Quand l'utiliser** : decisions importantes ou la verification croisee vaut le cout (analyses juridiques, revues de code sensibles, decisions strategiques). Rarement pour du high-throughput.

### 2.4 Swarm / Handoff pattern — passe le bebe

**Principe** : pas de chef. Chaque agent peut **passer la main** (handoff) a un autre agent quand il estime qu'il n'est plus le bon a traiter la tache. C'est une circulation laterale, pas une delegation hierarchique.

```
    User → Agent A (triage)
                │
                │ handoff: "this is a coding task"
                ▼
          Agent B (coder)
                │
                │ handoff: "tests are failing, need QA"
                ▼
          Agent C (tester)
                │
                │ handoff: "fixed, back to coder"
                ▼
          Agent B (coder)
                │
                ▼
             ANSWER
```

**Difference cle avec le supervisor** :
- Supervisor : un chef decide a chaque etape qui doit parler
- Swarm : chaque agent decide lui-meme si la tache est pour lui ou s'il doit passer la main

**Avantages** :
- **Local decision-making** — pas de goulot central
- Flexible — les agents peuvent collaborer sans plan central
- Naturellement reseau (chaque agent connait les autres)

**Inconvenients** :
- Risque de boucles infinies (A passe a B, B passe a A, etc.)
- Difficile a tracer — qui a fait quoi quand ?
- Moins de controle global

**Quand l'utiliser** : systemes d'assistance multi-domaine ou le premier contact ne sait pas d'avance qui doit traiter (ex: bot de support client — tu commences par un triage, puis ca handoff vers billing, tech support, sales...).

**Reference** : c'est le pattern popularise par [OpenAI Swarm](https://github.com/openai/swarm), maintenant remplace par Agents SDK.

---

## 3. Tradeoffs comparatifs

| Pattern | Cout | Latence | Qualite | Complexite |
|---------|------|---------|---------|------------|
| Supervisor | Moyen | Moyenne | Bonne | Faible |
| Hierarchical | Tres eleve | Haute | Tres bonne | Haute |
| Debate | Eleve | Haute | Excellente (subjectif) | Moyenne |
| Swarm | Moyen | Variable | Bonne si bien tune | Moyenne |

### Arbre de decision pragmatique

```
Ta tache est simple ? (< 10 etapes, 1 competence)
├─ Oui → single agent
└─ Non → besoin de plusieurs roles ?
         ├─ Oui → y a-t-il un flux clair (input → analyse → output) ?
         │        ├─ Oui → supervisor
         │        └─ Non → swarm (handoff lateral)
         └─ Non, besoin de verification croisee
                  ├─ Oui → debate
                  └─ Non, mais c'est un enorme projet
                           └─ hierarchical
```

---

## 4. Communication entre agents

Quel que soit le pattern, les agents doivent **se parler**. Plusieurs formats.

### 4.1 Message passing (le standard)

```python
@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    content: str
    metadata: dict  # timestamp, task_id, trace_id, cost_so_far...
```

Chaque agent lit les messages qui lui sont adresses, produit des messages en sortie. Le systeme (supervisor ou broker) route les messages.

### 4.2 Shared state

Les agents partagent un etat global (un dict, un store, un blackboard). Chaque agent lit et ecrit dans cet etat.

```python
state = {
    "task": "...",
    "research_done": False,
    "code_written": False,
    "findings": [...],
    "final_report": None,
}
```

**Avantage** : simple, visible, facile a debugger.
**Inconvenient** : couplage fort. Modifier le schema du state casse tous les agents.

### 4.3 Tool-like handoff

Dans le swarm pattern, chaque agent a un "outil" `transfer_to(agent_name)`. L'appeler termine le tour courant et transfere le contexte a un autre agent.

```python
def transfer_to_coder(state):
    return {"next_agent": "coder", "state": state}
```

C'est le pattern utilise par OpenAI Swarm/Agents SDK.

### 4.4 Protocoles

Pour les systemes plus grands ou distribues :
- **Message Queue** (Redis, RabbitMQ) — decouple les agents
- **Pub/sub** — diffusion sans connaitre les abonnes
- **gRPC / HTTP** — si chaque agent tourne dans un process different

**En pratique** : pour 95% des cas, un simple passing de messages ou un shared state dans le meme process suffit.

---

## 5. Prompts specialises — la cle du multi-agent

Le multi-agent ne marche que si chaque agent a un prompt specialise **focus et court**. Melanger les roles dans un meme prompt degrade tous les roles.

### Anti-pattern : le prompt frankenstein

```
You are an agent that can research topics, write code, test it, and produce
technical reports. You know Python, SQL, statistics, prose writing, and project
management. Do your best to help the user.
```

**Probleme** : le LLM essaie d'etre moyen partout plutot qu'excellent quelque part.

### Bon pattern : un prompt par role

```
# Researcher prompt
You are a research agent. Your ONLY job is to find factual information
from the knowledge base. You do NOT write code, you do NOT summarize beyond
the essentials. For each question, produce a JSON list of facts with citations.

# Coder prompt
You are a coding agent. You receive a specification and produce runnable
Python code. You do NOT explain prose, you do NOT research new information.
If the spec is unclear, ask ONE clarifying question.

# Writer prompt
You are a writer agent. You receive a collection of facts and code, and you
produce a clean technical report. You do NOT add new facts. You only
reorganize and present the provided content clearly.
```

**Pourquoi ca marche** :
- Chaque role est clair, pas de chevauchement
- Le LLM a moins de decisions a prendre — il peut se concentrer sur la qualite
- Debugging simple : si le rapport est mauvais, tu sais que c'est le writer (pas le researcher)

---

## 6. Pieges courants du multi-agent

### 6.1 Boucles infinies (surtout en swarm)

**Symptome** : Agent A → Agent B → Agent A → Agent B → ...

**Cause** : chaque agent pense que l'autre est mieux place.

**Defense** :
- Compteur global d'iterations avec limite
- Tracer les handoffs et detecter les cycles
- Le swarm doit avoir une **sortie finale** claire (ex: un agent "responder" qui produit la reponse terminale)

### 6.2 Explosion du cout

**Symptome** : une tache qui prenait 5 appels LLM en single-agent en prend 50 en multi-agent.

**Cause** : chaque agent a son propre context, chaque handoff ajoute des messages.

**Defense** :
- Budget global d'appels LLM et de tokens
- Messages compacts entre agents (extraire l'essentiel, pas tout le dialogue)
- Skipping : si un worker n'est pas necessaire, ne pas l'invoquer

### 6.3 Loss of context entre agents

**Symptome** : l'agent B ne sait pas ce que l'agent A a fait, refait le meme travail ou rate une information.

**Cause** : les messages transmis sont trop compacts ou mal structures.

**Defense** :
- Schema clair pour les messages inter-agents
- Inclure un resume du travail deja fait
- Tester : "si je lance l'agent B avec juste les messages qu'il recevra, est-ce qu'il a assez de contexte ?"

### 6.4 Style incoherent en sortie

**Symptome** : le rapport final melange plusieurs voix, plusieurs formats.

**Cause** : le supervisor ne reformate pas, il concatene.

**Defense** : toujours avoir une etape finale de synthese qui **reecrit** dans un style unique, plutot que de coller les bouts.

### 6.5 Competences qui se chevauchent

**Symptome** : le researcher code un peu, le coder recherche un peu. Les prompts derivent.

**Defense** : prompts tres stricts ("You do NOT do X"). Re-verifier regulierement que chaque agent reste dans son perimetre.

---

## 7. Workflows Anthropic (Building Effective Agents, oct 2024)

En octobre 2024, Anthropic a publie [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — un article de reference qui distingue **workflows** (LLM calls orchestres par du code) et **agents** (LLM dirige son propre flow). Cette distinction est cruciale.

### Workflow vs agent

| Aspect | Workflow | Agent |
|--------|----------|-------|
| Qui decide le flux | Le code (determinisme) | Le LLM (autonomie) |
| Previsibilite | Haute | Faible |
| Cout | Previsible | Variable (explose vite) |
| Complexite | Faible a moyenne | Moyenne a elevee |
| Quand utiliser | Task decomposable en steps clairs | Task exploratoire, flux inconnu |

**Regle cle** : **commence toujours par un workflow**. Passe a un agent **seulement** si la task l'exige vraiment (decouverte, incertitude sur le flux, N etapes inconnues a l'avance).

### Les 5 patterns workflow

#### 1. Prompt chaining

Une sequence lineaire de prompts ou chaque step enrichit le contexte du suivant.

```
Task : Rediger un article de blog sur l'agentic AI

Step 1 : draft   — "Genere un brouillon de 500 mots sur {topic}"
Step 2 : edit    — "Ameliore ce brouillon : {draft}. Corrige le ton, elimine les repetitions"
Step 3 : summarize — "Genere le meta description SEO de cet article : {edited}"
```

**Quand utiliser** : taches decomposables en etapes sequentielles claires (draft -> edit -> polish, extract -> classify -> summarize).

#### 2. Routing

Un LLM classe l'input et route vers **differents prompts/modeles** specialises.

```
User query
  │
  ▼
[LLM classifier : "billing" | "tech_support" | "sales" | "other"]
  │
  ├─> "billing"      → prompt specialise facturation (+ modele cheap)
  ├─> "tech_support" → prompt specialise technique (+ modele reasoning)
  ├─> "sales"        → prompt commercial
  └─> "other"        → fallback generique
```

**Quand utiliser** : inputs varies qui beneficient chacun d'un prompt tres cible. Bonus : permet d'utiliser un modele cheap sur les cas simples, un modele premium uniquement sur les cas complexes.

#### 3. Parallelization

N appels LLM independants executes **en parallele**, puis merge des resultats.

**Deux sous-patterns** :
- **Voting / self-consistency** : meme prompt, N runs avec temperature > 0, vote majoritaire
- **Sectioning** : N prompts differents executes en parallele, chacun traite un angle (review produit : perspective prix, perspective design, perspective experience)

```
Input : "Revois ce code"
  │
  ├─> reviewer_security    → issues securite
  ├─> reviewer_performance → issues perf
  └─> reviewer_style       → issues style
       │
       ▼
  [Merge : liste consolidee]
```

**Quand utiliser** : review multi-angles, self-consistency sur des decisions critiques, collecte d'informations en parallele.

#### 4. Orchestrator-worker

Un **orchestrator LLM** decompose dynamiquement la task, delegue a des workers, agrege les resultats. C'est comme un supervisor mais **plus leger** — l'orchestrator prend typiquement 2 decisions (decompose, synthesize) et pas plus.

**Difference avec le supervisor classique** : le supervisor (section 2.1) peut prendre des decisions a chaque tour, re-router, re-planer. L'orchestrator-worker fait **un seul plan**, un seul dispatch, une seule synthese.

**Quand utiliser** : taches ou on peut planifier en une passe. Plus simple et moins cher qu'un supervisor iteratif.

#### 5. Evaluator-optimizer

Un LLM genere une reponse, un **autre LLM evalue** selon des criteres, loop jusqu'a ce que l'evaluation soit satisfaisante.

```
Loop :
  1. generator : produit une reponse
  2. evaluator : score la reponse (1-5 + critique)
  3. si score >= 4 : return
  4. sinon : regenerate en utilisant la critique
  5. max 3 iterations
```

**Quand utiliser** : translation (qualite difficile a juger a la premiere passe), code review (critique -> fix -> recritique), creative writing.

### Trade-off avec les patterns multi-agent classiques

| Aspect | Workflows Anthropic | Agents classiques (supervisor/swarm) |
|--------|-------------------|--------------------------------------|
| Complexite | Faible | Moyenne a elevee |
| Nombre d'appels LLM | Previsible (3-10) | Variable (10-100+) |
| Cout | Maitrise | Explose sur les mauvaises tasks |
| Debugging | Simple (code deterministe) | Difficile (decisions LLM a chaque tour) |
| Flexibilite | Limitee | Haute |
| Quand | 80% des cas prod | Tasks vraiment agentiques |

**Regle 2026** : commence par un workflow, passe a un agent seulement si ton workflow ne couvre pas la task. La plupart des "agents" en prod sont en fait des workflows deguises — et c'est tres bien comme ca.

---

## 8. Frameworks pour faire du multi-agent

| Framework | Pattern par defaut | Forces | Limites |
|-----------|-------------------|--------|---------|
| **LangGraph** | Graphe (tout pattern possible) | Flexible, production-ready, checkpointing | Courbe d'apprentissage |
| **CrewAI** | Crew = supervisor implicite | Simple, roles/tasks/agents clairs | Moins flexible |
| **AutoGen** (Microsoft) | Group chat | Bon pour debate et collaboration | Moins de controle |
| **OpenAI Agents SDK** (ex-Swarm) | Swarm / handoff | Simple, natif OpenAI | Peu d'outils avances |
| **MetaGPT** | Hierarchical (company analogy) | Marche pour la generation de code complet | Tres opinionated |

**Recommandation** : commence par **LangGraph** pour comprendre les mecanismes. C'est bas niveau mais tu controles tout. Si tu veux aller plus vite en prod, passe ensuite a CrewAI ou Agents SDK selon tes preferences.

### 8.1 Paysage frameworks 2025-2026 — vue actualisee

L'ecosysteme a explose en 2024-2025. Nouveaux entrants, repositionnement des anciens, standardisation autour de MCP et des SDKs officiels. Table comparative a jour :

| Framework | Force | Limite | Quand l'utiliser |
|-----------|-------|--------|------------------|
| **Pydantic AI** (Q4 2024) | Type-safe, Pydantic-native, multi-LLM provider | Jeune, ecosysteme en croissance, docs encore en dev | Prod Python, teams deja Pydantic-heavy, besoin de structured output natif |
| **Claude Agent SDK** (sept 2025) | Official Anthropic, prompts optimises pour Claude, `tool_choice` natif, extended thinking integre | Claude-only by design | Agents Anthropic-first, qualite max |
| **OpenAI Agents SDK** (mars 2025) | Simple, pattern handoff (swarm-inspired), tracing natif | GPT-only, customization limitee | Prototypage OpenAI-only, POCs rapides |
| **LangGraph** (2023+) | Maximum de flexibilite, checkpointing, state machine explicite, ecosysteme LangChain | Complexite elevee, boilerplate | Workflows complexes, multi-step, production avec contraintes fortes |
| **CrewAI** | Prototyping rapide, roles/tasks clairs, DX agreable | Abstraction trop epaisse (dur a debugger quand ca derape) | Demos, early prototypes, onboarding |
| **smolagents (HuggingFace)** | Minimaliste, "code as action" (agents qui ecrivent du code Python) | Experimental, pas de production stability | Recherche, experimentation |
| **DSPy** (Stanford) | Optimisation automatique des prompts (compilation), eval-driven | Courbe d'apprentissage, moins intuitif | Taches repetables avec eval dataset solide |

### 8.2 Regle de choix 2026

```
Tu construis un nouvel agent :
│
├─ Anthropic-first + qualite max      → Claude Agent SDK
├─ OpenAI-first + simple              → OpenAI Agents SDK
├─ Multi-provider + type safety       → Pydantic AI
├─ Workflow complexe, state machine   → LangGraph
├─ Prototype rapide, demo             → CrewAI
└─ Taches repetables avec eval        → DSPy
```

**Le grand gagnant 2026** : pour une nouvelle app Python en production, **Pydantic AI** (si tu veux type safety multi-LLM) ou **Claude Agent SDK** (si tu es Anthropic-first) sont les choix par defaut. **LangGraph** reste incontournable pour les workflows complexes avec state machine explicite et checkpointing. Les autres sont des niches.

> **Opinion terrain** : en pratique, on garde LangGraph comme default sur les workers production (flexibilite, checkpointing) et Pydantic AI sur les nouvelles features ou la type safety est critique. Claude Agent SDK est un bon pick pour les POCs premium.

---

## 9. Flash Cards — Test de comprehension

**Q1 : Dans quels cas le multi-agent bat le single-agent ?**
> R : Quand la tache necessite **plusieurs competences distinctes** (ex: analyse + code + redaction), quand le prompt d'un agent unique commence a melanger des roles incompatibles, quand on a besoin d'exploration parallele, ou quand la verification croisee ajoute de la qualite. Pour les taches simples (< 10 etapes, 1 competence), le multi-agent est de la sur-ingenierie.

**Q2 : Quelle est la difference fondamentale entre le supervisor pattern et le swarm pattern ?**
> R : **Supervisor** = decision centrale. Un chef recoit la tache, decide qui fait quoi, collecte et synthetise. **Swarm** = decision locale. Chaque agent decide lui-meme si la tache est pour lui ou s'il doit passer la main (handoff) a un autre. Pas de chef, circulation laterale.

**Q3 : Pourquoi le collaborative debate est-il puissant mais peu utilise en production ?**
> R : Puissant parce qu'il reduit les hallucinations (consensus entre plusieurs agents) et capture plusieurs perspectives — tres utile pour des taches subjectives. Peu utilise en prod parce que c'est **tres cher** (N agents * M rounds) et lent. Reserve aux decisions critiques ou la verification croisee vaut le cout.

**Q4 : Quel est l'anti-pattern du "prompt frankenstein" et comment l'eviter ?**
> R : Un seul prompt qui essaie de couvrir plusieurs roles a la fois ("tu es un agent qui recherche, code, teste et redige"). Le LLM devient moyen partout plutot qu'excellent quelque part. Defense : prompts specialises, courts, avec des "You do NOT do X" explicites pour delimiter le perimetre de chaque role.

**Q5 : Quel pattern choisir en premier quand tu commences un projet multi-agent ?**
> R : **Le supervisor pattern**. C'est le plus simple a implementer et a debugger, il couvre 80% des cas, le controle est centralise. On passe aux autres patterns seulement si les contraintes le demandent (swarm si pas de chef naturel, hierarchical si > 15 agents, debate si verification critique).

---

## Points cles a retenir

- Un agent unique suffit pour 80% des taches — ne sur-ingenierise pas
- 4 patterns de reference : **supervisor**, **hierarchical**, **debate**, **swarm**
- Supervisor = decision centrale. Swarm = decision locale (handoff). Hierarchical = supervisors de supervisors. Debate = plusieurs avis + moderation
- Toujours commencer par le supervisor pattern
- Un prompt specialise par role — jamais de "prompt frankenstein"
- Message passing ou shared state pour la communication ; protocoles (MQ, gRPC) pour les systemes distribues
- Pieges principaux : boucles infinies (swarm), explosion du cout, perte de contexte entre agents, style incoherent en sortie
- Budget strict sur les appels LLM et les iterations — le multi-agent amplifie les couts
- LangGraph pour apprendre en profondeur, CrewAI/Agents SDK pour aller vite en prod


---

## Pour aller plus loin

Lectures couvrant ce sujet (playlists dans [`shared/external-courses.md`](../../../shared/external-courses.md)) :

- **Berkeley CS294-196 (Fa25) — Lec. 3 (Multi-Agent Systems in Era of LLMs, Vinyals)** — vue DeepMind sur les systemes multi-agents.
- **Berkeley CS294-196 (Fa25) — Lec. 7 (Multi-Agent AI, Noam Brown)** — perspective game theory et coordination par l'auteur de Cicero/Pluribus.
