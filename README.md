# Mastering Believe

> Systeme d'apprentissage accelere — atteindre un niveau world-class dans un domaine technique en moins de 2 semaines.

Theorie + code applique runnable + exercices progressifs avec solutions, sur des domaines cles de l'ingenierie IA et backend moderne.

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

Chaque domaine contient ~14 jours de contenu (3-5h/jour) et un capstone.

## Structure d'un domaine

```
domains/<domain>/
├── README.md                # Scope, planning 2 semaines, criteres de reussite
├── 01-theory/               # Theorie progressive (Markdown)
├── 02-code/                 # Exemples runnable, commentes ligne a ligne
├── 03-exercises/
│   ├── 01-easy/             # Enonces faciles
│   ├── 02-medium/           # Enonces intermediaires
│   ├── 03-hard/             # Enonces difficiles
│   ├── solutions/           # Corriges commentes
│   └── workspace/           # TON espace pour ecrire tes solutions (gitignore)
└── 05-projets-guides/       # Projets reels appliques (contexte logistique automatisee)
```

> Les **05-projets-guides** sont des projets longs appliques a un editeur fictif de simulation logistique (LogiSim — entrepots automatises, flottes de robots, AGV, drones d'inventaire). Ils illustrent comment combiner les patterns appris sur un cas concret. Voir [`shared/logistics-context.md`](./shared/logistics-context.md).

## Comment utiliser ce repo

### Pour suivre un domaine

1. Lis le `README.md` du domaine — tu vois le planning, les prerequis, les criteres de reussite
2. Pour chaque jour : lis la theorie (`01-theory/0X-...md`), puis fais tourner le code applique (`02-code/0X-...py`)
3. Fais les exercices (`03-exercises/01-easy/0X-...md`) dans **ton workspace** : `03-exercises/workspace/01-easy/0X-...py`
4. Compare ta solution avec celle de `03-exercises/solutions/`
5. A la fin du domaine, fais le capstone

### Pour les exercices

Le dossier `domains/<domain>/03-exercises/workspace/` est **ignore par git**. Tu peux y mettre tes solutions, tes notes, tes notebooks — rien ne sera commit. Aucun risque de polluer le repo si tu fork.

```bash
# Exemple : faire le premier exercice d'algorithmie
cat domains/algorithmie-python/03-exercises/01-easy/01-two-sum.md
# Code ta solution
$EDITOR domains/algorithmie-python/03-exercises/workspace/01-easy/01-two-sum.py
# Execute
python domains/algorithmie-python/03-exercises/workspace/01-easy/01-two-sum.py
# Compare avec la correction
diff domains/algorithmie-python/03-exercises/workspace/01-easy/01-two-sum.py \
     domains/algorithmie-python/03-exercises/solutions/01-two-sum.py
```

## Installation

Python 3.11+ recommande.

```bash
git clone <ce-repo>
cd mastering-believe

# Selon le domaine que tu attaques :
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

## License

MIT — voir [LICENSE](./LICENSE).

## Contributions

Repo personnel publie en lecture publique. Issues et PRs bienvenues pour corriger des erreurs ou ajouter des exercices, mais le scope reste celui d'un parcours personnel.
