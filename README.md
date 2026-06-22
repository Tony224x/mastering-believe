# Mastering Believe

> Parcours d'apprentissage communautaires sur des sujets a creuser en profondeur. Repo public et ouvert — utilisation, fork et contributions bienvenus.

Theorie + (quand pertinent) code applique runnable + exercices progressifs avec solutions. Le format s'adapte au domaine : tech, sciences, langues, sciences humaines, autre. La duree d'un domaine n'est pas figee — quelques jours a plusieurs semaines selon le sujet.

## Philosophie

Chaque domaine est concu autour de techniques d'apprentissage prouvees :

- **Pareto first** — les 20% qui donnent 80% des resultats, en premier
- **Concret avant abstrait** — toujours un exemple, puis le principe
- **Deliberate practice** — exercices cibles, gradues, chronometrables
- **Active recall** — flash cards Q&A a la fin de chaque module de theorie
- **Capstone reel** — chaque domaine se conclut par un projet qui pourrait shipper

## Domaines disponibles

Les domaines sont ranges par **track** dans `domains/<track>/<domaine>/` (`tech`, `vie`, `exploratoire`). Le tableau ci-dessous est **genere** depuis les `meta.toml` ; inventaire complet (modules, prerequis, garde-fous, statuts) : [`domains/CATALOG.md`](./domains/CATALOG.md).

<!-- CATALOG:START -->
**Track Tech — maitrise d'ingenierie** :

| Domaine | Stack | Focus | Duree |
|---|---|---|---|
| [Systemes IA Agentiques](./domains/tech/agentic-ai/) | Python, LangGraph, MCP, Claude/OpenAI APIs | Concevoir des agents autonomes du single-agent au multi-agent en production, jusqu'aux patterns frontier 2025-2026. | 4 semaines (J1-J28) |
| [Algorithmie & Data Structures — Live Coding Python](./domains/tech/algorithmie-python/) | Python, stdlib | Structures de donnees et algorithmes pour le live coding : patterns LeetCode, complexite, entretiens tech FAANG. | ~45h sur 2 semaines (14 modules) |
| [Gouvernance de l'IA](./domains/tech/gouvernance-ia/) | Python, stdlib | Gouverner une flotte d'agents : EU AI Act, NIST RMF, ISO 42001, RGPD + 4 piliers (identite, owner, permissions, audit). | 15 modules (~45-60 min chacun) |
| [Réseaux de Neurones & LLMs](./domains/tech/neural-networks-llm/) | Python, numpy, PyTorch | Mécanismes internes des LLMs from scratch, du neurone au Transformer puis frontière NN 2026. | 3 semaines (core J1-J14 ≈ 75-80h + bloc frontière J15-J22 ≈ 38h optionnel) |
| [Robotics & AI](./domains/tech/robotics-ai/) | Python, PyTorch, MuJoCo, Gymnasium, NumPy, LeRobot | Robotique moderne : fondations classiques (SE(3), FK/IK, contrôle), RL/IL, diffusion policies, VLA frontier, capstone Diffusion Policy from scratch. | 28 jours |
| [System Design — Architecture Backend & IA](./domains/tech/system-design/) | Python, Kafka, RabbitMQ, Redis, gRPC, GraphQL, TorchServe, Triton, vLLM, Langfuse | Architectures scalables backend et IA (RAG, agents) pour entretiens senior/staff et ML en production. | 2 semaines (14 modules / ~43h complet, 12-15h express) |

**Track Vie — l'ecole de la vie** :

| Domaine | Pilier | Focus | Duree |
|---|---|---|---|
| [Apprendre a apprendre](./domains/vie/apprendre-a-apprendre/) | Esprit | Meta-competence fondatrice : retrieval practice, spaced repetition, deep work, metacognition, apprendre avec l'IA. | 14 modules x ~45 min (~10 h 30) |
| [Communication, persuasion & influence](./domains/vie/communication-persuasion/) | Relations | Communiquer clairement et persuader honnetement : ecoute, clarte, recit, negociation, feedback, prise de parole, influence ethique. | 14 modules (~45 min chacun), sur 2 semaines |
| [Finance personnelle & investissement](./domains/vie/finance-personnelle/) | Argent | Fondamentaux Pareto-first de la finance perso : interets composes, budget, dette, fonds indiciels, psychologie, independance financiere. | 14 modules (~45 min chacun), ~11 h |
| [Pensee critique, rationalite & decision](./domains/vie/rationalite-decision/) | Jugement | Methode de raisonnement neutre : probas/Bayes, biais, decision sous incertitude, calibration (Brier), verification (SIFT). | 14 modules (~45 min chacun) |
| [Sante, nutrition & longevite](./domains/vie/sante-longevite/) | Corps | Fondations evidence-based: sommeil, activite physique, nutrition, sante metabolique, stress & lien social, plan sante personnel. | 7 modules (~45 min chacun), ~8-10h sur 2-3 semaines |

**Exploratoire — ajouts sous cadrage adverse** :

| Domaine | Stack | Focus | Duree |
|---|---|---|---|
| [Psychedeliques, creativite & IA — Comprendre sans consommer](./domains/exploratoire/psychedeliques-creativite-ia/) | — | Etude educative, evidence-based et reduction des risques : securite/legalite, neuroscience, esprit critique creativite, integration legale, art & IA. | 6 modules + capstone (~45 min chacun) |

> Inventaire complet (modules, prerequis, garde-fous, statuts) : [`domains/CATALOG.md`](./domains/CATALOG.md).
<!-- CATALOG:END -->

Le format de duree reste indicatif — un domaine peut etre plus court ou plus long selon le sujet.

## Structure d'un domaine

```
domains/<track>/<domain>/    # track = tech | vie | exploratoire
├── README.md                # Scope, planning (duree libre), criteres de reussite
├── meta.toml                # Metadonnees (track, statut, niveau, stack, focus, prerequis...) -> CATALOG.md
├── 01-theory/               # Theorie progressive (Markdown)
├── 02-code/                 # Exemples runnable, commentes ligne a ligne
├── 03-exercises/
│   ├── 01-easy/             # Enonces faciles
│   ├── 02-medium/           # Enonces intermediaires
│   ├── 03-hard/             # Enonces difficiles
│   ├── solutions/           # Corriges commentes
│   └── workspace/           # Espace personnel pour ecrire ses solutions (gitignore)
├── 04-projects/             # Mini-projets / capstones libres lies au domaine
└── 05-projets-guides/       # Projets reels appliques (contexte logistique automatisee)
```

> Les **05-projets-guides** sont des projets longs appliques a un editeur fictif de simulation logistique (LogiSim — entrepots automatises, flottes de robots, AGV, drones d'inventaire). Ils illustrent comment combiner les patterns appris sur un cas concret. Voir [`shared/logistics-context.md`](./shared/logistics-context.md).

## Comment utiliser ce repo

### Pour suivre un domaine

1. Lire le `README.md` du domaine — planning, prerequis, criteres de reussite
2. Pour chaque jour : lire la theorie (`01-theory/0X-...md`), puis faire tourner le code applique (`02-code/0X-...py`)
3. Faire les exercices (`03-exercises/01-easy/0X-...md`) dans le **workspace** : `03-exercises/workspace/01-easy/0X-...py`
4. Comparer la solution avec celle de `03-exercises/solutions/`
5. A la fin du domaine, faire le capstone

### Pour les exercices

Le dossier `domains/<domain>/03-exercises/workspace/` est **ignore par git** : solutions, notes, notebooks peuvent y etre stockes sans risque de polluer le repo en cas de fork.

```bash
# Exemple : faire le premier exercice d'algorithmie
cat domains/tech/algorithmie-python/03-exercises/01-easy/01-complexite-big-o.md
# Coder la solution
$EDITOR domains/tech/algorithmie-python/03-exercises/workspace/01-easy/01-complexite-big-o.py
# Executer
python domains/tech/algorithmie-python/03-exercises/workspace/01-easy/01-complexite-big-o.py
# Comparer avec la correction
diff domains/tech/algorithmie-python/03-exercises/workspace/01-easy/01-complexite-big-o.py \
     domains/tech/algorithmie-python/03-exercises/solutions/01-complexite-big-o.py
```

## Installation

Python 3.11+ recommande.

```bash
git clone <ce-repo>
cd mastering-believe

# Selon le domaine aborde :
pip install numpy                       # Neural Networks & LLMs (02-code en numpy ; torch seulement pour certains 05-projets-guides)
pip install langgraph langchain         # Agentic AI
# Algorithmie & System Design : stdlib seulement
```

Les exemples sont concus pour tourner **standalone**, sans frais d'API obligatoires (mocks LLM fournis dans Agentic AI).

## Conventions

- **Langue** : francais pour la theorie/explications, anglais pour le code/commentaires
- **Numerotation** : tous les dossiers et fichiers sont numerotes (`01-`, `02-`, ...) pour imposer l'ordre d'apprentissage
- **Code** : runnable standalone, chaque ligne non-evidente expliquee (le **why**, pas le what)
- **Exercices** : chaque enonce a `## Objectif`, `## Consigne`, `## Criteres de reussite`

## Skill Claude Code : `mastering-domain-creator`

Le repo embarque un skill Claude Code (dans `.claude/skills/mastering-domain-creator/`) qui automatise la creation d'un nouveau domaine en respectant la convention ci-dessus. Pipeline 7 phases avec gates :

1. **Discovery** — interview utilisateur en 2 vagues (scope, niveau, stack, capstone, contraintes, mode full/lite, N jours)
2. **Sourced research** — subagents en parallele cherchent 3-5 sources tier-1 par axe → `REFERENCES.md`
3. **Plan + challenge** — plan N jours soumis a un subagent adverse avant validation utilisateur
4. **Bootstrap** — squelette + `PLAN.md` fige (contrat pour Phase 4)
5. **Creation parallele** — 1 subagent par jour, ecrit theorie + code + 3 niveaux d'exercices + solutions, anti-collision sur les fichiers transverses
6. **Pass 1 verification** — relecture sourcee toi-meme (citations, Q&A, concrete-before-abstract)
7. **Pass 2 verification** — 3 subagents en parallele : facts-checker, code-runner, pedagogy-reviewer
8. **Capstone & cloture** — verifie capstone, met a jour `CLAUDE.md` + `tasks/todo.md`, commit final

**Comment l'invoquer** : dans Claude Code, taper `/mastering-domain-creator` ou demander "ajoute un domaine sur X". Le skill ne se declenche que dans ce repo. Cross-platform (Windows / Linux / macOS).

## License

MIT — voir [LICENSE](./LICENSE).

## Contributions

Repo public et communautaire. Issues et PRs bienvenues pour corriger des erreurs, ajouter des exercices, proposer un nouveau domaine ou ameliorer un existant.
