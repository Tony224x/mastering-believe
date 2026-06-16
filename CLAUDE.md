# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

**Mastering Believe** is a community learning repository, **public and open to anyone** who wants to use, fork, or contribute. Goal: structure deep dives on chosen topics using evidence-based learning techniques (deliberate practice, spaced repetition, interleaving, active recall, Feynman technique).

Scope is **not limited to AI/backend or to a 2-week format**: a domain can cover anything (tech, sciences, languages, humanities, etc.) and span any duration that fits the topic — from a few days to several weeks.

Each domain is a self-contained module composed of:
1. **Cours theorique** — structured theory (Markdown), progressive, concise, no fluff
2. **Code applique** *(when relevant)* — real, runnable, heavily commented examples
3. **Exercices** — graded exercises (easy → hard → real-world challenge) with solutions

## Architecture

```
mastering-believe/
├── CLAUDE.md
├── .claude/skills/
│   └── mastering-domain-creator/   # Skill Claude Code : pipeline 7 phases de creation de domaine
├── domains/
│   └── <domain-name>/       # One folder per mastery domain
│       ├── README.md        # Domain overview, learning path, time budget
│       ├── PLAN.md          # (si cree via le skill) Plan fige du curriculum
│       ├── REFERENCES.md    # (si cree via le skill) Sources tier-1 par module
│       ├── 01-theory/       # Numbered theory modules (Markdown, source-of-truth)
│       │   ├── 01-fundamentals.md
│       │   └── ...
│       ├── 01-theory-qd/    # Site Quarkdown enrichi (optionnel, 1 par domaine)
│       │   ├── main.qd      # Index avec sidebar nav
│       │   └── ...
│       ├── 02-code/         # Runnable commented examples matching theory
│       ├── 03-exercises/    # Progressive exercises + solutions
│       │   ├── 01-easy/
│       │   ├── 02-medium/
│       │   ├── 03-hard/
│       │   ├── solutions/
│       │   └── workspace/   # Espace perso de l'apprenant (gitignore, sauf README/.gitkeep)
│       ├── 04-projects/     # Mini-projets / capstones libres lies au domaine
│       └── 05-projets-guides/   # Guided real-world projects in LogiSim context
│           ├── 01-<project>/
│           │   ├── README.md    # Contexte, consigne, etapes, criteres
│           │   └── solution/    # Corrige commente
│           └── ...
├── .github/workflows/
│   └── quarkdown-release.yml # CI : build des sites + bundles en Release sur tag v*
├── quarkdown/               # Pipeline de rendu .qd -> site HTML statique
│   ├── README.md            # Prerequis (Java 17+, Quarkdown CLI), install, build
│   ├── post-build-fix-links.py
│   └── scripts/
│       ├── build-all.ps1            # Build tous les sites Windows (ou -Domain X, -Watch)
│       ├── build-all.sh             # Idem Linux/macOS/CI (--domain X, --out)
│       └── scaffold-domain.py       # Genere 01-theory-qd/ depuis 01-theory/
└── shared/
    ├── templates/           # Templates for new domains
    ├── external-courses.md  # Index de cours universitaires (Stanford, MIT, CMU, ...)
    └── logistics-context.md # Shared LogiSim context referenced by all 05-projets-guides
```

Note : `tasks/` (todo.md, lessons/) est un espace de suivi **local et gitignore** — il peut exister sur une machine de dev mais n'est jamais commite. Idem pour `references/`, `docs/plans/` et les outputs Quarkdown (`quarkdown/output*/`).

## Conventions

- **Language**: French for theory/explanations, English for code/comments when the domain is tech
- **Numbering**: All folders and files are numbered (`01-`, `02-`, ...) to enforce learning order
- **Theory files**: Markdown with clear headings, key takeaways boxed, mnemonics highlighted
- **Code files** (when the domain has code): runnable standalone. Every non-obvious line has a comment explaining WHY, not WHAT
- **Exercises**: un fichier d'exercices regroupe 2-3 exercices ; chaque exercice a ses sections `### Objectif`, `### Consigne`, `### Criteres de reussite` (sous un `## Exercice N`)
- **Solutions**: Separate folder, never mixed with exercise files
- **Workspace**: `03-exercises/workspace/` est gitignore — l'apprenant y ecrit ses solutions sans polluer le repo
- **Naming coherence**: pour un module N, `01-theory/NN-x.md`, `02-code/NN-x.py`, `01-theory-qd/NN-x.qd` et les exercices/solutions partagent le meme slug numerote
- **Public repo**: contenu lisible par tous, pas d'info perso/sensible dans les fichiers commites

## Learning Methodology Rules

When creating content for a domain:
1. **Start with the 20% that gives 80% of results** — Pareto-first structure
2. **Concrete before abstract** — always lead with an example, then extract the principle
3. **Spaced repetition hooks** — end each theory module with 3-5 flash-card-style Q&A
4. **Deliberate practice** — exercises must target specific weaknesses, not repeat strengths
5. **Progressive overload** — each level must feel slightly beyond current comfort zone
6. **Real-world capstone** — every domain ends with a project that could ship or be shown in portfolio. Le capstone "fil-rouge" du planning vit dans `02-code/<dernier-jour>.py` ; `04-projects/` accueille des mini-projets libres ou capstones supplementaires

## Creating a New Domain

**Voie recommandee** : le skill `mastering-domain-creator` (`.claude/skills/mastering-domain-creator/`) automatise tout le pipeline en 7 phases avec gates : interview en 2 vagues, recherche sourcee (→ `REFERENCES.md`), plan challenge par un subagent adverse (→ `PLAN.md`), creation parallele (1 subagent par module), 2 passes de verification (facts-checker, code-runner, pedagogy-reviewer), capstone. Modes `full` / `lite`, duree N jours parametrable. Se declenche sur "ajouter un domaine", "create a domain", etc. — uniquement dans ce repo.

Voie manuelle :
1. Copy `shared/templates/` structure into `domains/<domain-name>/`
2. Write `README.md` with: scope, prerequisites, schedule (durée libre — quelques jours à plusieurs semaines selon le sujet), success criteria
3. Build theory → (code if applicable) → exercises in order, numbering consistently
4. Each theory module should take 30-60 min to study
5. Each code example (if any) should take 15-30 min to read and run
6. Exercises: minimum 3 easy, 3 medium, 2 hard, 1 capstone project

## Domains actifs

| Domain | Folder | Stack | Focus |
|--------|--------|-------|-------|
| Algorithmie Python | `domains/algorithmie-python/` | Python | Live coding, LeetCode patterns, entretiens tech |
| System Design | `domains/system-design/` | Diagrammes + Python/infra | Architecture backend & IA, entretiens senior |
| Neural Networks & LLMs | `domains/neural-networks-llm/` | Python, PyTorch | Mecanismes internes des LLMs, from scratch |
| Agentic AI | `domains/agentic-ai/` | Python, LangGraph, MCP | Agents autonomes, multi-agent, production |
| Robotics & AI | `domains/robotics-ai/` | Python, MuJoCo, PyTorch, LeRobot | Robotique moderne (VLA, world models, diffusion policies), capstone Diffusion Policy 28j — **en cours** : pas encore de `04-projects/` ni `05-projets-guides/` |

**Projets guides (contexte logistique automatisee)** : les domaines finalises ont un dossier `05-projets-guides/` avec 3 projets appliques a un contexte d'editeur de simulation logistique (inspire de LogiSim / produit FleetSim, fictif). Voir `shared/logistics-context.md` pour le contexte metier complet. Le projet phare est `domains/agentic-ai/05-projets-guides/02-supervisor-swarm-multi-tier/` qui illustre la combinaison des patterns supervisor et swarm de LangGraph sur un scenario d'operation multi-flotte.

**Quarkdown** : les **5 domaines actifs** ont un `01-theory-qd/` buildable (agentic-ai partiellement enrichi, les 4 autres en placeholders scaffoldes). Les `.md` de `01-theory/` restent la source-of-truth lisible sur GitHub ; les `.qd` sont des versions enrichies (math LaTeX, mermaid, callouts). **Toute correction d'un `.md` de theorie doit etre repercutee dans le `.qd` miroir s'il existe.** La CI (`.github/workflows/quarkdown-release.yml`) build tous les sites et publie un bundle `tar.gz` par domaine en asset de GitHub Release sur tag `v*`.

## Commands

Code examples are standalone scripts — run them directly (Python 3.11+ recommande):
- **Python**: `python domains/<domain>/02-code/<file>.py`
- **Neural Networks & LLMs**: `python domains/neural-networks-llm/02-code/<file>.py` (les `02-code/` sont en **numpy** + stdlib, pas de torch ; `torch` n'est requis que pour certains `05-projets-guides/`)
- **LangGraph**: `python domains/agentic-ai/02-code/<file>.py` (requires `langgraph`, `langchain` — mocks LLM fournis, pas de cle API obligatoire)
- **Robotics**: `python domains/robotics-ai/02-code/<file>.py` (deps **selon le module** : `numpy` partout, `torch` + `gymnasium` pour les modules RL/learning — classic-control `CartPole`/`Pendulum` + `PushT`, `matplotlib`/`scipy` ponctuels ; `mujoco` est **optionnel**, utilise seulement par quelques modules — voir l'entete `# requires:` de chaque fichier ; headless mujoco : `MUJOCO_GL=osmesa`)
- Algorithmie & System Design : stdlib seulement
- Pas de suite de tests ni de linter au niveau repo — verifier un exemple = le lancer

Quarkdown (build des sites de cours — requiert Java 17+ et le CLI `quarkdown`, voir `quarkdown/README.md`) :
- **Build tous les domaines** : `pwsh quarkdown/scripts/build-all.ps1` (Windows) / `bash quarkdown/scripts/build-all.sh` (Linux/macOS/CI)
- **Build un domaine** : `pwsh quarkdown/scripts/build-all.ps1 -Domain agentic-ai` / `bash quarkdown/scripts/build-all.sh --domain agentic-ai`
- **Live preview** : `pwsh quarkdown/scripts/build-all.ps1 -Domain agentic-ai -Watch` (PowerShell uniquement)
- **Scaffold un 01-theory-qd/** : `python quarkdown/scripts/scaffold-domain.py <domain>` (idempotent, `--force` pour reecrire ; `main.qd` toujours regenere)
- Output : `quarkdown/output-site/<domain>/` (gitignore)
