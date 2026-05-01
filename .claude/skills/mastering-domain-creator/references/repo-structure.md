# Convention exacte du repo `mastering-believe`

**Source de verite** : structure verifiee contre `algorithmie-python`, `system-design`, `neural-networks-llm`, `agentic-ai` au 2026-05-01. Si tu lis ce fichier dans le futur et que la realite a derive, RE-VERIFIE avant d'appliquer.

## Arborescence reelle

```
domains/<nom-du-domaine>/
├── README.md                          # scope, prereqs, planning, criteres
├── REFERENCES.md                      # Phase 1 du skill (nouveau, pas dans domaines existants)
├── PLAN.md                            # Phase 3 du skill — fige le brief de chaque jour
├── REVIEW-pass1.md                    # Phase 5 du skill
├── REVIEW-pass2.md                    # Phase 6 du skill
├── 01-theory/
│   ├── 01-<slug>.md                   # 1 fichier markdown par jour
│   ├── 02-<slug>.md
│   └── ... jusqu'a NN-<slug>.md
├── 02-code/
│   ├── 01-<slug>.py                   # 1 FICHIER PLAT par jour, PAS un dossier
│   ├── 02-<slug>.py
│   └── ...
├── 03-exercises/
│   ├── 01-easy/
│   │   ├── 01-<slug>.md               # MEME slug que la theorie du jour
│   │   ├── 02-<slug>.md
│   │   └── ...
│   ├── 02-medium/
│   │   ├── 01-<slug>.md               # MEME slug, contenu different (medium)
│   │   └── ...
│   ├── 03-hard/
│   │   ├── 01-<slug>.md               # MEME slug, contenu different (hard)
│   │   └── ...
│   └── solutions/
│       ├── 01-<slug>.py               # 1 fichier qui solutionne les 3 niveaux
│       └── ...
├── 04-projects/                       # SOUVENT VIDE dans les domaines existants
│   └── (le capstone vit dans 02-code/<dernier-jour>.py par convention)
└── 05-projets-guides/                 # OPTIONNEL — uniquement si fil-rouge metier
    ├── 01-<projet>/
    │   ├── README.md
    │   └── solution/
    ├── 02-<projet>/
    └── 03-<projet>/
```

## Faits importants (verifies)

1. **`02-code/` contient des fichiers `.py` plats**, PAS des dossiers `<NN>-<slug>/main.py`. Exemples reels :
   ```
   domains/algorithmie-python/02-code/01-complexite-big-o.py
   domains/algorithmie-python/02-code/02-arrays-strings.py
   domains/system-design/02-code/01-principes-fondamentaux.py
   domains/agentic-ai/02-code/01-anatomie-agent.py
   ```

2. **`03-exercises/` reproduit le MEME slug dans les 3 niveaux** :
   ```
   03-exercises/01-easy/01-complexite-big-o.md       # version easy du theme du J1
   03-exercises/02-medium/01-complexite-big-o.md     # version medium du meme theme
   03-exercises/03-hard/01-complexite-big-o.md       # version hard du meme theme
   03-exercises/solutions/01-complexite-big-o.py     # solutions des 3 niveaux
   ```

3. **`04-projects/` est vide** dans tous les domaines verifies. Le capstone vit dans le fichier `02-code/14-<slug>.py` (dernier jour). Exemple : `02-code/14-mock-interviews.py`, `02-code/14-capstone.py`.

4. **N peut depasser 14** : `neural-networks-llm/02-code/` contient jusqu'a `22-...py` (frontier 2026 lessons). Le N est un parametre, pas un hard-cap.

5. **`05-projets-guides/` n'existe que dans les 4 domaines actifs** depuis l'ajout du fil-rouge LogiSim/FleetSim. Optionnel pour un nouveau domaine.

## Regles de nommage

- Prefixes `NN-` zero-padded sur 2 chiffres : `01`, `02`, ..., `14`, ..., `22`.
- Slug en kebab-case, ASCII pur, sans accents (`big-o`, pas `complexité-big-o`).
- **Le slug est le MEME entre `01-theory/`, `02-code/`, et les 3 niveaux d'exercices** + solutions, pour qu'on retrouve facilement le materiel d'un jour.
- Code : English pour identifiers et comments. Markdown : francais pour la prose.

## Contenu attendu par fichier

### `README.md` du domaine
Suit `shared/templates/README.md`. Sections obligatoires :
- `## Scope`
- `## Prerequisites`
- `## Planning (2 semaines)` avec tableau Jour/Module/Temps estime
- `## Criteres de reussite` — concrets, mesurables
- `## Ressources externes` — max 5, pointe vers `REFERENCES.md` pour la liste exhaustive

### `PLAN.md` (nouveau, contrat Phase 3 → Phase 4)
Format :
```markdown
# Plan fige domaine <nom>

## J1 — <titre>
- **Concepts cles** : 4-6 bullets
- **Acquis a la fin du jour** : 2-3 phrases
- **Sources autorisees** (max 3 extraites de REFERENCES.md) :
  - <ref 1>
- **Stack du jour** : <python / pytorch / langgraph / ...>
- **Slug** : 01-<slug>

## J2 — ...
```

### Theorie (`01-theory/<NN>-<slug>.md`)
- Heading H1 = titre du module (ASCII)
- Section "Pourquoi ce module" en 3 lignes max
- Exemple concret AVANT principe abstrait
- Sections numerotees H2 (`## 1. ...`)
- Encadre "Key takeaway" a la fin de chaque section
- Bloc final `## Spaced repetition` : 3 a 5 Q&A flash-card
- Au moins 1 citation explicite vers `REFERENCES.md` (format `[Auteur, Annee, ch. X]`)

### Code (`02-code/<NN>-<slug>.py`)
- Fichier `.py` plat, **pas** un dossier
- Docstring d'en-tete decrivant ce que le script demontre
- `if __name__ == "__main__":` toujours
- Commentaires expliquent le **WHY**, pas le WHAT
- Dependances listees en commentaire en tete (ex : `# requires: torch>=2.0`)
- Doit tourner standalone : `python <fichier>` produit une sortie visible si les deps sont la
- Doit AU MINIMUM compiler : `python -m py_compile <fichier>` reussit toujours

### Exercices (`03-exercises/0X-<niveau>/<NN>-<slug>.md`)
Le meme slug existe dans `01-easy/`, `02-medium/`, `03-hard/`. Format obligatoire :
```markdown
# <Titre> — niveau <easy/medium/hard>

## Objectif
<une phrase>

## Consigne
<numerotee, etapes claires>

## Criteres de reussite
- [ ] critere 1
- [ ] critere 2
```

### Solutions (`03-exercises/solutions/<NN>-<slug>.py`)
Un seul fichier qui couvre les 3 niveaux du jour. Sections separees par commentaires :
```python
# === EASY ===
def easy_solution(...): ...

# === MEDIUM ===
def medium_solution(...): ...

# === HARD ===
def hard_solution(...): ...

if __name__ == "__main__":
    # smoke test des 3 niveaux
    ...
```

### Capstone
**Convention** : pas de `04-projects/<capstone>/`. Le capstone est le code du dernier jour (`02-code/<NN>-<slug>.py` ou `02-code/<NN>-capstone.py`). Plus consequent que les jours precedents (~2-3x la taille), runnable end-to-end.

### Projets guides (`05-projets-guides/`) — optionnel
Si Phase 0 demande un fil-rouge metier :
- 2 ou 3 projets numerotes (`01-`, `02-`, `03-`)
- Chacun a son `README.md` (contexte metier, consigne, etapes guidees, criteres) + dossier `solution/` avec le corrige commente
- Chaque `README.md` reference `shared/logistics-context.md` (ou autre contexte cree)

## Mise a jour transverse obligatoire (faite par Claude principal, PAS les subagents)

A chaque nouveau domaine :
1. `CLAUDE.md` racine — section "## Domains actifs" : ajouter une ligne dans le tableau (Domain / Folder / Stack / Focus)
2. `tasks/todo.md` — checklist J1..JN cochee a la fin

## Conventions de qualite (rappel CLAUDE.md repo)

- Pareto-first : J1 a J(N/4) = 20% qui donne 80%
- Concrete-before-abstract : exemple → principe
- Spaced repetition hooks : 3-5 Q&A par module theorique
- Deliberate practice : exos ciblent faiblesses, ne repetent pas forces
- Progressive overload : chaque jour > J-1
- Capstone reel : J(N-1) a JN shippable

## Rappel cross-platform

Toutes les commandes shell utilisees dans le pipeline doivent etre portables ou utiliser le tool natif de l'OS courant :
- `python -m py_compile <fichier>` : portable
- `python <fichier>` : portable (le binaire `python` ou `py` selon Windows)
- `Glob "domains/<nom>/**/*"` (tool Glob) : portable, remplace `tree`, `find`, `dir /s`
- `git` : portable
- Eviter : `tree`, `find ... -exec`, `/dev/null`, `python3` hardcode
