# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

**Mastering Believe** is a personal accelerated mastery system for Anthony. Goal: reach world-class level in targeted domains in < 2 weeks per topic, using evidence-based learning techniques (deliberate practice, spaced repetition, interleaving, active recall, Feynman technique).

Each domain is a self-contained module composed of:
1. **Cours theorique** — structured theory (Markdown), progressive, concise, no fluff
2. **Code applique** — real, runnable, heavily commented code examples illustrating each concept
3. **Exercices** — graded exercises (easy → hard → real-world challenge) with solutions

## Architecture

```
mastering-believe/
├── CLAUDE.md
├── tasks/
│   ├── todo.md              # Current task tracking
│   └── lessons/             # Self-improvement loop
│       └── *.md
├── domains/
│   └── <domain-name>/       # One folder per mastery domain
│       ├── README.md        # Domain overview, learning path, time budget
│       ├── 01-theory/       # Numbered theory modules (Markdown)
│       │   ├── 01-fundamentals.md
│       │   ├── 02-intermediate.md
│       │   └── ...
│       ├── 02-code/         # Runnable commented examples matching theory
│       │   ├── 01-fundamentals/
│       │   └── ...
│       ├── 03-exercises/    # Progressive exercises + solutions
│       │   ├── 01-easy/
│       │   ├── 02-medium/
│       │   ├── 03-hard/
│       │   └── solutions/
│       └── 04-projects/     # Capstone mini-projects (real-world application)
└── shared/
    └── templates/           # Templates for new domains
```

## Conventions

- **Language**: French for theory/explanations, English for code/comments when the domain is tech
- **Numbering**: All folders and files are numbered (`01-`, `02-`, ...) to enforce learning order
- **Theory files**: Markdown with clear headings, key takeaways boxed, mnemonics highlighted
- **Code files**: Must be runnable standalone. Every non-obvious line has a comment explaining WHY, not WHAT
- **Exercises**: Each exercise file starts with `## Objectif`, `## Consigne`, `## Criteres de reussite`
- **Solutions**: Separate folder, never mixed with exercise files

## Learning Methodology Rules

When creating content for a domain:
1. **Start with the 20% that gives 80% of results** — Pareto-first structure
2. **Concrete before abstract** — always lead with an example, then extract the principle
3. **Spaced repetition hooks** — end each theory module with 3-5 flash-card-style Q&A
4. **Deliberate practice** — exercises must target specific weaknesses, not repeat strengths
5. **Progressive overload** — each level must feel slightly beyond current comfort zone
6. **Real-world capstone** — every domain ends with a project that could ship or be shown in portfolio

## Creating a New Domain

1. Copy `shared/templates/` structure into `domains/<domain-name>/`
2. Write `README.md` with: scope, prerequisites, 2-week schedule, success criteria
3. Build theory → code → exercises in order, numbering consistently
4. Each theory module should take 30-60 min to study
5. Each code example should take 15-30 min to read and run
6. Exercises: minimum 3 easy, 3 medium, 2 hard, 1 capstone project

## Domains actifs

| Domain | Folder | Stack | Focus |
|--------|--------|-------|-------|
| Algorithmie Python | `domains/algorithmie-python/` | Python | Live coding, LeetCode patterns, entretiens tech |
| System Design | `domains/system-design/` | Diagrammes + Python/infra | Architecture backend & IA, entretiens senior |
| Neural Networks & LLMs | `domains/neural-networks-llm/` | Python, PyTorch | Mecanismes internes des LLMs, from scratch |
| Agentic AI | `domains/agentic-ai/` | Python, LangGraph, MCP | Agents autonomes, multi-agent, production |

## Commands

Code examples are standalone scripts — run them directly:
- **Python**: `python domains/<domain>/02-code/<file>.py`
- **PyTorch**: `python domains/neural-networks-llm/02-code/<file>.py` (requires `torch`)
- **LangGraph**: `python domains/agentic-ai/02-code/<file>.py` (requires `langgraph`, `langchain`)
