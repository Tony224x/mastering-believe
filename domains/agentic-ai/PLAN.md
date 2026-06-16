# PLAN — agentic-ai, extension avancee (S3-S4, J15-J28)

Plan fige du curriculum avance (frontier 2025-2026), greffe sur les fondations
J1-J14. Mode **full** : chaque jour = theorie + code commente runnable +
exercices gradues + solutions + flash-cards. Niveau : ingenieur Python ayant
fait J1-J14.

> Source-of-truth pedagogique. Toute deviation lors du build doit etre
> justifiee. Les sources tier-1 par module sont dans `REFERENCES.md`.

## Principe directeur : zero redite avec J1-J14

Le plan a ete challenge par un relecteur adverse. Chaque module avance a une
**delimitation ecrite** vis-a-vis des fondations pour eviter de re-payer du
contenu deja vu. Rappels d'une ligne uniquement, puis renvoi au jour concerne.

| Jour | Module | Neuf vs J1-J14 (delimitation) |
|------|--------|-------------------------------|
| J15 | Context engineering & compaction | vs J3 (memoire) / J9 (topologie) / J12 (cost) : ici **tokens & fenetre de contexte**, compaction/offloading, deep-agent virtual FS, budgeting fin |
| J16 | Memoire long-horizon | vs J3 (bases) : **architectures** episodique/semantique/procedurale, MemGPT, consolidation |
| J17 | Verifiers & self-improvement | vs J4 (Reflexion/CoT/test-time deja vus) : **verifiers/PRM**, self-improvement **persiste entre runs**, expo RL/fine-tuning |
| J18 | Orchestration comparee & failure modes | vs J5/J6/J9 (patterns) : **comparatif inter-frameworks** + **quand le multi-agent casse** |
| J19 | Protocoles inter-agents | vs J10 (MCP) : **A2A/ACP/agent cards**, MCP < 10% (rappel) |
| J20 | Durable & event-driven agents | vs J6 (checkpointing) / J12 (retry) : **durable execution** (survit au crash process) + HITL avance |
| J21 | Architecture des coding agents | nouveau : **ACI**, boucle edit/search/run, SWE-bench |
| J22 | Computer use & GUI agents | nouveau : perception d'ecran, set-of-marks, grounding (place apres J21, plus fragile) |
| J23 | Sandboxing & execution sure (infra) | vs J13 (principes) : **mise en oeuvre infra** (gVisor/microVM/sandbox-runtime/egress) |
| J24 | Inference engineering | vs J12 (cost basique) : structured outputs/constrained decoding + routing (RouteLLM) + caching |
| J25 | Serving stateful & sessions a l'echelle | vs J6/J12 : **choix de backend, scaling horizontal, online eval** |
| J26 | Benchmarking pratique | vs J11 (catalogue) : **faire tourner un harness sur SON agent**, pass^k, regression |
| J27 | Capstone avance — architecture | assemblage |
| J28 | Capstone avance — build & eval | assemblage runnable |

## Ordre & progression

- S3 (J15-J21) : ce qui se passe **a l'interieur** d'un agent avance (contexte,
  memoire, auto-amelioration) puis **entre agents** (orchestration, protocoles,
  durabilite) puis premier agent specialise (coding).
- S4 (J22-J26) : computer-use, puis **production a l'echelle** (infra, inference,
  serving, evaluation), avant le capstone.
- Decision d'ordre cle (issue du challenge) : **coding agents (J21) AVANT
  computer-use (J22)** car l'ACI texte est plus fondatrice que le grounding
  visuel.

## Capstone J27-J28 — scope fige (runnable, sans GPU ni infra lourde)

**"Deep ops agent durable, observable et auto-evalue" — 100% local-mockable.**

- **Coeur** : un *deep agent* (planner + todo/scratchpad + virtual filesystem
  sur disque local) orchestrant 2-3 sous-agents avec **isolation de contexte**
  (reinvestit J15/J16).
- **Durabilite app-level** : checkpointer **SQLite** + reprise apres `kill -9`
  (reinvestit J20 ; **pas de serveur Temporal** — montre en encart theorique).
- **Routing** : routeur de modele **mocke** (regle cout/complexite, J24).
- **Coding tool** : un outil edit/search/run scope a un **mini-repo jouet
  fourni**, execution en sous-process restreint (J21/J23, principe).
- **Harness d'eval** : 8-12 cas, eval trajectory + final-answer, LLM-as-judge
  **deterministe/mocke**, rapport de regression vs baseline (J11/J26).
- **Critere de reussite** : `python ...28-....py` tourne **sans cle API** (LLM
  mocke) et avec une vraie cle ; **survit a un kill** en cours de run ; produit
  un rapport d'eval.
- **Hors livrable** (enseignes mais non exiges dans le binaire) : Temporal
  deploye, computer-use/GUI (trop fragile pour etre reproductible).

## Gates qualite (Phases 5-6)

1. **Facts-checker** : verifier les affirmations techniques contre `REFERENCES.md`.
2. **Code-runner** : chaque `02-code/*.py` et `03-exercises/solutions/*.py`
   tourne en stdlib pur, sans cle API, exit 0.
3. **Pedagogy-reviewer** : delimitations respectees, zero redite, flash-cards
   presentes, progression tenue.
