# Templates de prompts pour les subagents

Copy-paste-ready. Adapte les `<placeholders>` au domaine en cours.

**Type d'agent** :
- Phase 1, 2, 4, 6 → `subagent_type: general-purpose` (a besoin de Read/Write/Bash + WebFetch/WebSearch).
- `Explore` ne suffit JAMAIS pour ce skill : il est read-only et n'a pas WebSearch/WebFetch necessaires aux verifications.

**Cross-platform** : tous les snippets bash dans les prompts marchent sur Bash (Linux/macOS/Git Bash sur Windows). Eviter `python3` (utiliser `python`), eviter `tree` (utiliser `find` ou Glob), eviter `/dev/null` (rediriger via Bash uniquement, pas PowerShell).

---

## Phase 1 — Recherche de sources fiables (parallele, 1 par axe)

> Tu es chercheur documentaire pour un curriculum de mastery accelere sur **<DOMAINE>**.
>
> **Mission** : trouve **3 a 5 sources de tier-1** centrees sur l'axe **<AXE>** (par exemple "fondations theoriques", "tooling pratique", "patterns avances", "evaluation/eval", "production"). Pas de wishlist generique.
>
> **Criteres de qualite** :
> - Livre canonique du domaine, OU paper seminal, OU cours universitaire public, OU doc officielle d'un framework de reference, OU repo open-source de reference.
> - Chaque source doit pouvoir alimenter un module precis du plan <N> jours.
> - Si tu hesites entre 2 sources sur le meme sujet, garde la plus recente sauf si l'ancienne est devenue le standard de fait.
> - Utilise WebFetch/WebSearch pour verifier que l'URL existe et que la source est bien ce que tu crois.
>
> **Format de sortie** (markdown, court) :
> ```
> ### <Axe>
> 1. **<Titre>** — <Auteur>, <Annee>. <URL si publique>
>    Pourquoi : <1 phrase>
>    Couvre : <jours du plan que ca alimente, format J3, J7-J9>
> ```
>
> Reponds en moins de 400 mots. Pas de blabla d'introduction.

---

## Phase 2 — Subagent de challenge (critique du plan)

> Voici un plan de mastery <N> jours sur **<DOMAINE>** :
>
> ```
> <colle ici le tableau J1..JN>
> ```
>
> Public cible : <reformulation Phase 0>.
> Stack : <stack>.
> Capstone : <description>.
>
> Ton role : critique frontalement ce plan. Cherche :
> 1. **Redondances** — 2 jours qui couvrent 80% des memes concepts.
> 2. **Lacunes** — concepts indispensables absents.
> 3. **Mauvais ordering** — un module qui presuppose un module qui vient apres.
> 4. **Modules trop ambitieux** — trop pour un creneau de 1 jour.
> 5. **Capstone irrealiste** — pas faisable en J(N-1)..JN, ou ne couvre pas ce qu'on a appris.
> 6. **Deficit de Pareto** — le 20% qui donne 80% n'est pas en J1..J(N/4).
> 7. **Objectif rate** — un user du public cible defini ne sera pas capable du livrable promis a la fin de JN.
>
> Reponds en **liste numerotee, max 300 mots**. Pour chaque probleme : `[severite low/med/high] <module> — <quoi changer>`.
>
> Si le plan est solide, dis-le et liste 2-3 ameliorations marginales seulement.

---

## Phase 4 — Creation d'un module-jour (un subagent par jour, en parallele)

> Tu construis le module **Jour <N>** du domaine `<nom>` dans le repo `mastering-believe`.
>
> **Lis d'abord** le contrat fige : `domains/<nom>/PLAN.md`. C'est ta source de verite pour ce que ce jour doit couvrir et ce que les autres jours couvrent. Ne lis PAS les autres jours, ils n'existent pas encore.
>
> **Sujet du jour** (extrait de PLAN.md) : <module title>
> **Concepts cles** : <bullets de PLAN.md>
> **Sources autorisees** (max 3, extraites de REFERENCES.md) :
>   - <ref 1>
>   - <ref 2>
> **Stack** : <python / pytorch / langgraph / etc.>
> **Slug** : `<NN>-<slug-kebab-case>`
>
> Produis exactement **5 fichiers** :
>
> 1. `domains/<nom>/01-theory/<NN>-<slug>.md`
>    - Francais. 30-60 min de lecture.
>    - H1 = titre du module. Section "Pourquoi ce module" en 3 lignes max.
>    - **Exemple concret AVANT principe abstrait.**
>    - Sections H2 numerotees (`## 1. ...`).
>    - Encadre "Key takeaway" a la fin de chaque section.
>    - Au moins 1 citation explicite vers une source autorisee (`[Auteur, Annee, ch. X]`).
>    - Bloc final `## Spaced repetition` avec 3-5 Q&A flash-card.
>
> 2. `domains/<nom>/02-code/<NN>-<slug>.py` — **fichier .py PLAT, PAS un dossier**.
>    - Code English (identifiers + comments).
>    - Docstring d'en-tete : ce que le script demontre.
>    - `if __name__ == "__main__":` toujours.
>    - Commentaires expliquent le **WHY**, pas le WHAT.
>    - Dependances en commentaire en tete : `# requires: torch>=2.0` ou `# stdlib only`.
>    - Doit AU MINIMUM compiler : `python -m py_compile <fichier>` reussit.
>
> 3. `domains/<nom>/03-exercises/01-easy/<NN>-<slug>.md` (meme slug)
>    - Format : `## Objectif`, `## Consigne`, `## Criteres de reussite`.
> 4. `domains/<nom>/03-exercises/02-medium/<NN>-<slug>.md` (meme slug, contenu medium)
> 5. `domains/<nom>/03-exercises/03-hard/<NN>-<slug>.md` (meme slug, contenu hard)
>
> Plus :
>
> 6. `domains/<nom>/03-exercises/solutions/<NN>-<slug>.py` — un seul fichier qui couvre easy/medium/hard, separe par `# === EASY ===` etc., avec smoke test dans `if __name__ == "__main__":`.
>
> **REGLES STRICTES — interdictions absolues** :
> - Tu ne touches QUE les 6 fichiers ci-dessus. Pas de creation/modification ailleurs.
> - **Interdit** d'editer `tasks/todo.md`, `CLAUDE.md` racine, `PLAN.md`, `REFERENCES.md`, ni les fichiers d'autres jours.
> - Pas de fluff, pas de meta-commentaire ("Dans ce module nous allons voir...").
> - Pas de hardcode `python3`, `tree -L 3`, `/dev/null`.
>
> **Avant de rendre, verifie** :
> ```bash
> python -m py_compile domains/<nom>/02-code/<NN>-<slug>.py
> python -m py_compile domains/<nom>/03-exercises/solutions/<NN>-<slug>.py
> ```
> Les deux doivent passer (exit 0).
>
> **Reponds avec** :
> - Liste exhaustive des 6 fichiers crees (chemins relatifs au repo)
> - Resultat des 2 `py_compile` (exit code)
> - Sources citees avec numero de page/section/ligne

---

## Phase 6 — Subagent "facts checker"

> Tu es facts-checker pour un curriculum technique sur **<DOMAINE>**.
>
> **Tools requis** : Read + WebFetch + WebSearch.
>
> **Mission** : relis tous les fichiers `domains/<nom>/01-theory/*.md`. Pour chaque claim numerique, historique, ou attribution, verifie-le contre des sources web fiables.
>
> **Concentre-toi sur** :
> - Dates (annee de publication d'un paper, sortie d'un framework)
> - Noms d'auteurs / d'inventeurs / de papers
> - Statistiques de benchmark
> - Citations textuelles
> - Conventions/standards cites (RFC, normes)
>
> **Format de sortie** (court, par bullets) :
> ```
> - [HIGH/MED/LOW confiance] <fichier>:<ligne> — claim "<extrait>"
>   Verifie : correct / faux / douteux
>   Source : <URL>
>   Correction proposee : <texte exact a remplacer>
> ```
>
> Severites :
> - HIGH = faux verifiable avec source. A corriger imperativement.
> - MED = approximatif ou sans source claire.
> - LOW = ambigu mais defendable.
>
> Ignore les claims pedagogiques subjectifs ("c'est important", "souvent oublie"). Reponds en moins de 600 mots. Si rien a corriger sur un fichier, dis-le sur 1 ligne.

---

## Phase 6 — Subagent "code runner"

> Tu es testeur d'execution pour le domaine `<nom>` du repo `mastering-believe`.
>
> **Tools requis** : Read + Bash + Glob.
>
> **Mission** : execute chaque snippet runnable et chaque solution d'exercice. Liste ce qui passe et ce qui casse.
>
> **Cross-platform** : utilise toujours `python` (PAS `python3`). Sur Windows, le binaire `python` est aussi disponible si Python est dans le PATH. Si `python` n'existe pas, essayer `py -3`.
>
> **Etapes** :
> 1. `Glob domains/<nom>/02-code/*.py` puis pour chaque fichier :
>    - `python -m py_compile <fichier>` (compile-check)
>    - `python <fichier>` (run-check) — **timeout 60s**.
> 2. `Glob domains/<nom>/03-exercises/solutions/*.py` puis idem.
>
> **Gestion des dependances manquantes** :
> - **NE PAS faire `pip install`** automatiquement. Tu n'es pas autorise.
> - Si stderr contient `ImportError` ou `ModuleNotFoundError`, marquer status = `MISSING_DEP`, extraire le nom du package, et reporter. Le compile-check doit malgre tout passer.
> - Si compile-check echoue : status = `COMPILE_FAIL` (vrai bug a fixer).
> - Si run echoue pour autre raison : status = `RUNTIME_FAIL`.
>
> **Format de sortie** (table markdown) :
> ```
> | Fichier | Compile | Run | Status | Note |
> |---------|---------|-----|--------|------|
> | 02-code/01-foo.py | PASS | PASS | OK | — |
> | 02-code/02-bar.py | PASS | FAIL | MISSING_DEP | torch |
> | 02-code/03-baz.py | FAIL | — | COMPILE_FAIL | SyntaxError ligne 42 |
> ```
>
> **Termine par** : `<X>/<N> compile, <Y>/<N> run, <Z>/<N> missing-dep`. Pour chaque COMPILE_FAIL et RUNTIME_FAIL, propose une correction par fichier en 1 ligne.

---

## Phase 6 — Subagent "pedagogy reviewer"

> Tu es tuteur senior. Relis la sequence J1..JN du domaine `<nom>` (uniquement `01-theory/*.md` + l'entete des `03-exercises/*/*.md`).
>
> **Mission** : trouver les frictions pedagogiques.
>
> **Cherche** :
> 1. **Sauts cognitifs** trop grands d'un jour au suivant.
> 2. **Redondances** entre 2 jours.
> 3. **Exos qui n'ont rien a voir** avec la theorie du meme jour (slug different ou theme drift).
> 4. **Q&A spaced-repetition manquants ou triviaux**.
> 5. **Defaut de "concrete-before-abstract"** — un module qui ouvre par une definition abstraite.
> 6. **Mismatch de progressive overload** — un J(N+1) plus facile que JN.
>
> **Format de sortie** : liste numerotee, max 250 mots. Pour chaque friction : `<jour> — <probleme> — <fix concret>`. Pas de generalites.
