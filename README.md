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

| Domaine | Stack | Focus |
|---------|-------|-------|
| [Algorithmie Python](./domains/algorithmie-python/) | Python | Live coding, patterns LeetCode, entretiens tech |
| [System Design](./domains/system-design/) | Diagrammes + Python | Architecture backend & IA, entretiens senior |
| [Neural Networks & LLMs](./domains/neural-networks-llm/) | Python, PyTorch | Mecanismes internes des LLMs, from scratch |
| [Agentic AI](./domains/agentic-ai/) | Python, LangGraph, MCP | Agents autonomes, multi-agent, production |

Les domaines actuels sont organises sur ~14 jours (3-5h/jour) avec un capstone, mais ce format reste indicatif — un nouveau domaine peut etre plus court ou plus long selon le sujet.

## Structure d'un domaine

```
domains/<domain>/
├── README.md                # Scope, planning (duree libre), criteres de reussite
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
cat domains/algorithmie-python/03-exercises/01-easy/01-two-sum.md
# Coder la solution
$EDITOR domains/algorithmie-python/03-exercises/workspace/01-easy/01-two-sum.py
# Executer
python domains/algorithmie-python/03-exercises/workspace/01-easy/01-two-sum.py
# Comparer avec la correction
diff domains/algorithmie-python/03-exercises/workspace/01-easy/01-two-sum.py \
     domains/algorithmie-python/03-exercises/solutions/01-two-sum.py
```

## Installation

Python 3.11+ recommande.

```bash
git clone <ce-repo>
cd mastering-believe

# Selon le domaine aborde :
pip install torch                       # Neural Networks & LLMs
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
