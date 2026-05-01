---
name: mastering-domain-creator
description: Create a new accelerated-mastery domain inside the `mastering-believe` repo. Triggers a guided 7-phase pipeline (interview, sourced research, plan + challenge, parallel course creation, sourced verification pass 1, subagent verification pass 2, capstone). Use this skill ONLY when working in the `mastering-believe` repo (CWD ends with `mastering-believe` or user explicitly references it). Triggers on phrases like "ajouter un domaine", "create a domain", "monter une formation", "build a learning track", "ajouter un module de maitrise", or "I want a 14-day mastery track on X". Do NOT trigger on generic "I want to learn X" outside this repo — for that, just give a study plan inline.
---

# Mastering Domain Creator

Pipeline pour creer un nouveau domaine de maitrise accelere (theorie + code + exercices + projets guides) dans le repo `mastering-believe`, en gardant la barre de qualite des domaines existants (`algorithmie-python`, `system-design`, `neural-networks-llm`, `agentic-ai`).

**7 phases avec gates explicites. Ne pas sauter de phase.** Si une gate echoue, on boucle au lieu de pousser.

## Detection d'environnement

Toutes les commandes shell de ce skill doivent fonctionner sur **Windows (PowerShell)** ET **Linux/macOS (bash)**. Detecte au demarrage :

```python
# Detection
import platform, sys
IS_WINDOWS = platform.system() == "Windows"
PYTHON = sys.executable  # absolu, evite python vs python3 vs py
```

Ou en shell :
- Bash dispo via le tool `Bash` sur les 3 OS → privilegier `Bash` pour les commandes simples (`ls`, `find`, `python`).
- Pour les commandes specifiques OS, utiliser le tool natif : `PowerShell` sur Windows, `Bash` sinon.
- `tree` n'est pas portable → utiliser le tool `Glob` pour verifier l'arborescence.
- `python -m py_compile <fichier>` est portable et fiable → c'est le test de compilation par defaut.
- `python <fichier>` lance un script ; si le script importe `torch`/`langgraph`, il peut echouer faute de deps — voir Phase 6 pour la gestion.

**Pas de hardcode `/dev/null`, `python3`, `tree -L 3`, ou path absolus Linux.** A la place : noms relatifs, `Glob`, `Bash` portable.

## Parametre N (nombre de jours)

Par defaut **N=14** mais peut etre fixe a 7, 10, 14, 21 selon Phase 0. Toutes les references "J1..J14", "14 jours", "14 modules" doivent etre lues comme "J1..JN". Le SKILL substitue N une fois fixe en Phase 0.

## Mode "lite" vs "full"

Cette pipeline lance ~20 subagents (research + challenge + N modules + 3 reviewers). Sur Opus, ca chiffre. Phase 0 doit demander :
- **full** : N modules + 3 reviewers Phase 6 (defaut)
- **lite** : N/2 modules consolides + 1 reviewer (code-runner) en Phase 6 — pour un POC ou un domaine bien maitrise

---

## Phase 0 — Discovery interview

Avant tout, demande des questions de clarification. Cout d'asker = bas ; cout de batir un curriculum sur la mauvaise hypothese = enorme.

**Pose les questions en 2 vagues** (utiliser `AskUserQuestion` si dispo, max 4 questions par vague) :

**Vague 1 — Cadrage**
1. **Domaine & scope** : Quel domaine exactement ? Frontieres ? Qu'est-ce qu'on EXCLUT ?
2. **Niveau de depart** : Prerequis acquis ? Background pertinent ?
3. **Niveau cible** : "World-class" veut dire quoi ici ? (entretien senior / shipper un projet / ecrire un papier / expliquer a un junior)
4. **Stack / langage** : Python ? PyTorch ? LangGraph ? Rust ? K8s ? Autre ?

**Vague 2 — Contraintes & ambition**
5. **Capstone** : Projet final reve ? Sinon on en propose un en Phase 2.
6. **Contexte metier optionnel** : Veut-il un `05-projets-guides/` rattache a `shared/logistics-context.md` (LogiSim/FleetSim), un autre fil-rouge a creer, ou rien ?
7. **Contraintes** : Temps reel/jour ? GPU ? Budget API ? Hors-ligne requis ? **Mode full ou lite ?** **N jours (defaut 14) ?**

**Gate** : reformule la cible en 3 lignes (domaine, public cible, capstone, mode, N) et fais valider explicitement avant Phase 1.

## Phase 1 — Sourced research

Lance des subagents `general-purpose` (PAS `Explore` — il n'a pas WebFetch/WebSearch necessaires pour valider) **en parallele**, un par axe. Vise 3-5 sources tier-1 par axe.

**Axes par defaut** (a adapter au domaine) :
- Fondations theoriques (livre canonique, paper seminal)
- Tooling pratique (doc officielle du framework principal)
- Patterns avances / production
- Evaluation / benchmarks de reference
- Repo open-source de reference

Pour chaque source : titre, auteur, annee, URL, **pourquoi elle est dans la liste** (1 phrase), **modules qu'elle alimente**.

Stocke dans `domains/<nom>/REFERENCES.md`. C'est la source de verite pour Phase 5/6.

**Gate** : presente les references a l'utilisateur. Demande "manque-t-il une source que tu utilises personnellement ?"

Voir `references/subagent-prompts.md` section Phase 1 pour le template.

## Phase 2 — Plan + challenge

Construis un plan **N jours** suivant la convention reelle du repo (voir `references/repo-structure.md`). Format minimum :

| Jour | Module | Concepts cles | Sources principales |
|------|--------|---------------|---------------------|

Principes obligatoires (rappel CLAUDE.md repo) :
- **Pareto-first** : J1 a J(N/4) couvrent le 20% qui donne 80%.
- **Concrete before abstract** : exemple d'abord, principe ensuite.
- **Progressive overload** : chaque jour legerement au-dessus du precedent.
- **Capstone reel** : J(N-1) a JN = projet shippable.
- **Spaced repetition hooks** : 3-5 Q&A flash a la fin de chaque module.

**Challenge obligatoire** : avant presentation utilisateur, lance un subagent `general-purpose` avec le prompt de challenge (voir `references/subagent-prompts.md`). Integre les corrections high/med severity.

**Gate** : validation utilisateur explicite avant Phase 3.

## Phase 3 — Bootstrap + PLAN.md fige

Cree la structure squelette EXACTE selon `references/repo-structure.md` (fichiers plats, pas de sous-dossiers `<NN>-<slug>/main.py`).

**Etapes** :
1. Cree les dossiers vides : `01-theory/`, `02-code/`, `03-exercises/{01-easy,02-medium,03-hard,solutions}/`, `04-projects/`, et `05-projets-guides/` si Phase 0 le demande.
2. Copie `shared/templates/README.md` dans `domains/<nom>/README.md`, remplis scope/prereqs/planning/criteres.
3. Copie `REFERENCES.md` issu de Phase 1.
4. **Cree `domains/<nom>/PLAN.md`** : fige le brief de chaque jour. C'est le contrat que liront les subagents Phase 4 — ils ne se lisent PAS entre eux.

Format de `PLAN.md` :
```markdown
# Plan fige domaine <nom>

## J1 — <titre>
- Concepts cles (4-6 bullets)
- Acquis a la fin du jour
- Sources autorisees (max 3, extraites de REFERENCES.md)
- Stack du jour

## J2 — <titre>
...
```

5. Met a jour `tasks/todo.md` du repo avec checklist J1..JN.
6. Met a jour la section "Domains actifs" du `CLAUDE.md` racine.

**Gate** : verifier l'arborescence avec `Glob "domains/<nom>/**/*"`. PLAN.md couvre N jours.

## Phase 4 — Creation parallele des cours

Pour chaque jour J1..JN, **lance un subagent `general-purpose` dedie** qui produit les 3 livrables (theorie + code + exercices+solutions). Spawne **par lots de 3 a 5** dans un seul message pour la concurrence, mais lance les lots sequentiellement.

**Regle anti-collision** : chaque subagent peut UNIQUEMENT ecrire dans :
- `domains/<nom>/01-theory/<NN>-<slug>.md`
- `domains/<nom>/02-code/<NN>-<slug>.py`
- `domains/<nom>/03-exercises/01-easy/<NN>-<slug>.md`
- `domains/<nom>/03-exercises/02-medium/<NN>-<slug>.md`
- `domains/<nom>/03-exercises/03-hard/<NN>-<slug>.md`
- `domains/<nom>/03-exercises/solutions/<NN>-<slug>.py`

**Interdit** : toucher `tasks/todo.md`, `CLAUDE.md` racine, `PLAN.md`, `REFERENCES.md`, autres jours. Ces fichiers sont reserves a Claude principal en Phase 7.

Le subagent doit **lire `PLAN.md`** pour comprendre le contexte global, pas explorer les autres jours (qui n'existent pas encore).

Briefing complet dans `references/subagent-prompts.md` section Phase 4.

**Convention de fichier** (verifiee contre repo reel) :
- `02-code/<NN>-<slug>.py` est un **fichier plat unique**, pas un dossier.
- Les exercices ont **le meme slug** dans `01-easy/`, `02-medium/`, `03-hard/`, mais contenu different (3 niveaux de difficulte sur le meme theme du jour).
- Les solutions sont dans `03-exercises/solutions/<NN>-<slug>.py` — un fichier qui solutionne les 3 niveaux.

**Gate** : tous les jours presents (verifier avec `Glob`), tous les `python -m py_compile <fichier>` passent. Si un module manque, relance son subagent.

**Checkpoint commit** apres Phase 4 reussie :
```
git add domains/<nom>/ && git commit -m "chore(<nom>): scaffold + day modules [WIP]"
```
Ca evite de perdre 56 fichiers si Phase 6 crash.

## Phase 5 — Pass 1 : verification sourcee (toi-meme)

Lis CHAQUE `01-theory/<NN>-<slug>.md`. Pour chacun :
1. Au moins **1 reference du REFERENCES.md** est citee.
2. La citation est **plausible** (pas un n° de page hallucine).
3. Q&A spaced-repetition existent (3-5) et sont non-triviaux.
4. Le module **commence par un exemple concret** avant la theorie abstraite.

Logue dans `domains/<nom>/REVIEW-pass1.md` (probleme, fichier, severite).

**Definition des severites** :
- **HIGH** : citation hallucinee (auteur/annee/page faux verifiable), code qui ne compile pas, fait demonstrablement faux dans la theorie, structure cassee (manque Q&A, manque exemple concret).
- **MED** : imprecision verifiable (date approximative, statistique sans source), exemple peu pedagogique, deviation de "concrete-before-abstract".
- **LOW** : formulation, typo, slug suboptimal.

Tu fais cette passe **toi-meme**. Cette passe attrape les hallucinations structurelles avant qu'elles soient figees.

**Gate** : 0 high-severity restant. Med toleres si trackes dans REVIEW-pass1.md.

## Phase 6 — Pass 2 : verification subagents

**Mode full** : 3 subagents `general-purpose` en parallele.
**Mode lite** : seulement le code-runner.

1. **Facts checker** (subagent_type: `general-purpose`, doit avoir WebFetch/WebSearch) — relit la theorie, verifie claims numeriques/historiques contre sources web.
2. **Code runner** (subagent_type: `general-purpose`, doit avoir Bash) — execute chaque `02-code/<NN>-<slug>.py` et chaque `solutions/<NN>-<slug>.py`. Capture stdout/stderr. **Si dependance manque** (`ImportError: torch`), reporte FAIL avec note `missing dep: <pkg>` au lieu d'installer (pas de `pip install` autonome).
3. **Pedagogy reviewer** (`general-purpose`) — relit la sequence J1..JN, cherche frictions pedagogiques.

Briefings complets dans `references/subagent-prompts.md`.

Consolide dans `REVIEW-pass2.md`. Applique les fixes.

**Gate** : code-runner = 100% PASS sur les jours sans dep externe ; jours avec dep externe doivent au moins faire `python -m py_compile`. Facts-checker = 0 "HIGH confiance faux".

## Phase 7 — Capstone & cloture

1. Verifie que J(N-1) ou JN dans `02-code/` constitue le capstone runnable de bout en bout (convention du repo : pas de dossier `04-projects/<capstone>/`, le capstone vit dans le code du dernier jour).
2. Si Phase 0 a demande un fil-rouge metier : cree `05-projets-guides/01-<projet>/`, `02-...`, `03-...` chacun avec `README.md` + `solution/`. Chaque projet reference `shared/logistics-context.md` (ou nouveau contexte cree par Phase 0).
3. Met a jour `CLAUDE.md` racine — section "## Domains actifs" (1 ligne dans le tableau).
4. Met a jour `tasks/todo.md` (cocher J1..JN).
5. Commit final :
   ```
   git add -A && git commit -m "feat(<nom>): full <N>-day mastery track for <domaine>"
   ```
   Ou si le checkpoint Phase 4 existe, amend ou nouveau commit selon preference utilisateur.

**Gate final** : resume 8-10 lignes a l'utilisateur — quoi cree, combien de fichiers, capstone, points faibles connus restants (issus de REVIEW-pass1/2 medium-severity).

---

## Anti-patterns a fuir

- **Sauter Phase 0** parce que "le domaine est evident" → 1 fois sur 2 le user voulait autre chose.
- **Phase 4 sequentielle** au lieu de parallele → N× plus lent pour rien.
- **Subagents Phase 4 qui touchent `tasks/todo.md` ou `CLAUDE.md`** → race condition, write-after-write loss.
- **Subagents Phase 4 qui se lisent entre eux** → ils ne peuvent pas, les autres jours n'existent pas. Ils lisent `PLAN.md`.
- **Sources generiques** ("the official docs") → en Phase 1 on veut des URLs precises avec `pourquoi`.
- **Une seule passe de verif** → Phase 5 (toi) attrape les hallus structurelles, Phase 6 (subagents) attrape les bugs d'execution. Les deux sont necessaires.
- **Cherrypicker les sources qui confirment** → le subagent Phase 2 doit pouvoir descendre frontalement le plan.
- **Hardcoder commandes Linux** (`tree -L 3`, `python3`, `/dev/null`) → casse sur Windows. Utiliser `Glob`, `python -m py_compile`, et le tool natif PowerShell pour les bouts OS-specific.

## Resume "checklist" (utilisable comme TaskCreate)

```
[ ] Phase 0 — Interview validee (mode + N fixes)
[ ] Phase 1 — REFERENCES.md cree, valide par utilisateur
[ ] Phase 2 — Plan N jours ecrit, challenge applique, valide par utilisateur
[ ] Phase 3 — Squelette + PLAN.md fige
[ ] Phase 4 — N jours generes en parallele, py_compile vert, checkpoint commit
[ ] Phase 5 — REVIEW-pass1.md, 0 high-severity
[ ] Phase 6 — REVIEW-pass2.md, code-runner vert (ou compile-vert pour deps externes)
[ ] Phase 7 — Capstone OK, CLAUDE.md/todo.md a jour, commit final
```

## References internes

- `references/repo-structure.md` — convention exacte VERIFIEE des dossiers/fichiers du repo.
- `references/subagent-prompts.md` — templates de prompts pour Phase 1, 2, 4, 6.
