# Exercices Medium — Coding agents : ACI & boucle edit/run (J21)

---

## Exercice 1 : Outil `edit` robuste avec contrainte d'unicite (search/replace)

### Objectif
Aller au-dela du `edit` "premiere occurrence" du module : construire un outil de search/replace **structure** qui applique un edit block (`find` → `replace`) sur le contenu d'un fichier, mais qui **rejette** un edit dont le bloc `find` est introuvable OU ambigu (plusieurs occurrences). C'est exactement la garantie qu'Aider et SWE-agent exigent : un edit doit cibler un endroit **unique**, sinon il est dangereux (modification silencieuse au mauvais endroit).

### Consigne
Tu travailles sur des **chaines en memoire** (pas de vrais fichiers) pour rester deterministe.

1. Definis un `@dataclass EditBlock` avec au moins : `find: str`, `replace: str`.
2. Definis un `@dataclass EditResult` avec : `ok: bool`, `new_source: str | None`, `reason: str` (vide si succes, explicite sinon ex : `"not found"`, `"ambiguous: 3 matches"`).
3. Implemente une fonction `apply_edit(source: str, block: EditBlock) -> EditResult` qui :
   - compte le nombre d'occurrences de `block.find` dans `source` ;
   - **0 occurrence** → `ok=False`, `reason` commence par `"not found"`, `new_source=None` ;
   - **2+ occurrences** → `ok=False`, `reason` commence par `"ambiguous"` et contient le nombre de matches, `new_source=None` (on **refuse** plutot que de deviner) ;
   - **exactement 1 occurrence** → `ok=True`, `new_source` = source avec le bloc remplace, `reason=""`.
4. Implemente `apply_edits(source: str, blocks: list[EditBlock]) -> EditResult` qui applique une **sequence** de blocs : chaque bloc est applique sur le resultat du precedent ; si un bloc echoue, on s'arrete et on renvoie l'echec (les blocs deja appliques sont conserves dans `new_source` partiel ? **Non** : renvoie le `new_source` au moment de l'echec pour inspection, mais `ok=False`).
5. Teste avec assertions :
   - un edit valide (1 match) modifie bien la source ;
   - un edit sur un `find` absent est rejete (`not found`) ;
   - un edit sur un `find` present 2 fois est rejete (`ambiguous: 2 matches`) ;
   - une sequence de 2 edits valides s'enchaine correctement ;
   - une sequence qui contient un edit invalide en 2e position s'arrete au bon endroit.

### Criteres de reussite
- [ ] `EditBlock` et `EditResult` sont des dataclasses
- [ ] `apply_edit` refuse un `find` absent (`reason` commence par `"not found"`)
- [ ] `apply_edit` refuse un `find` ambigu et indique le nombre de matches dans `reason`
- [ ] Un edit valide (1 seul match) renvoie `ok=True` et la nouvelle source attendue
- [ ] `apply_edits` enchaine les blocs et s'arrete au premier echec
- [ ] Tout tourne offline, stdlib pur, deterministe

---

## Exercice 2 : Applicateur de patch unified-diff avec detection de conflit

### Objectif
Reproduire le coeur d'un outil de patch (Claude Code / git apply) : parser un **unified diff simplifie** (en-tete de hunk `@@ -l,c +l,c @@` + lignes de contexte ` `, suppression `-`, ajout `+`) et l'appliquer sur un fichier en memoire. Tu dois prouver deux choses : (a) un patch propre s'applique correctement, et (b) un patch dont le **contexte ne correspond pas** au fichier (le fichier a derive) est detecte comme **conflit** plutot qu'applique au mauvais endroit.

### Consigne
1. Ecris `parse_hunks(diff: str) -> list[Hunk]` ou `Hunk` est un `@dataclass` avec : `old_start: int`, et `lines: list[tuple[str, str]]` (tag parmi `" "`, `"-"`, `"+"` et le texte de la ligne).
   - Parse uniquement les lignes commençant par `@@`, ` `, `-`, `+`. Ignore les en-tetes `---`/`+++`.
   - Tu peux te contenter du `old_start` de l'en-tete `@@ -old_start,old_count +new_start,new_count @@`.
2. Ecris `apply_patch(source: str, diff: str) -> tuple[bool, str, str]` retournant `(ok, new_source, reason)` :
   - pour chaque hunk, verifie que les lignes de **contexte** (`" "`) et de **suppression** (`"-"`) correspondent **exactement** aux lignes de `source` a partir de `old_start - 1` (1-indexe → 0-indexe) ;
   - si une ligne attendue ne matche pas la ligne reelle → `ok=False`, `reason="conflict at line N: ..."`, ne modifie rien ;
   - sinon, construit la nouvelle version : garde les lignes de contexte, supprime les `"-"`, insere les `"+"`.
3. Teste avec assertions :
   - un patch valide (un hunk qui corrige `return a - b` → `return a + b`) produit la source corrigee ;
   - le meme patch applique sur une source **derivee** (la ligne de contexte attendue n'existe plus / a change) renvoie `ok=False` avec un `reason` qui contient `"conflict"` ;
   - apres un conflit, la source d'origine est **inchangee** (le patch est atomique au niveau du hunk).

### Criteres de reussite
- [ ] `parse_hunks` parse l'en-tete `@@` et recupere `old_start` + les lignes taguees
- [ ] Un patch propre s'applique et produit la source corrigee attendue
- [ ] Un patch dont le contexte ne matche pas est detecte (`reason` contient `"conflict"`)
- [ ] En cas de conflit, la source n'est pas modifiee (atomicite verifiee par assertion)
- [ ] Les ajouts (`+`) et suppressions (`-`) sont correctement appliques
- [ ] Execution offline, stdlib pur, deterministe

---

## Exercice 3 : Boucle plan-edit-test sur un mock repo

### Objectif
Implementer la boucle fondamentale du module — `search → edit → run_tests → iterer` — sur un **mock repo** (dict `nom_fichier -> source`). L'agent edite une fonction buggee et lance des tests **en process** (pas de subprocess), iterant jusqu'au vert ou epuisement du budget. C'est la brique de base d'un coding agent, isolee de toute la machinerie de fichiers reels.

### Consigne
1. Modelise un mock repo : `repo: dict[str, str]` (ex : `{"calc.py": "def add(a, b):\n    return a - b\n"}`).
2. Ecris un **runner de tests en process** `run_tests(repo: dict, tests: list[tuple]) -> dict` ou chaque test est `(fn_name, args, expected)` :
   - compile la source du repo dans un namespace isole (`exec` dans un dict frais — c'est un mock pedagogique, pas du code non-fiable) ;
   - pour chaque test, appelle `fn_name(*args)` et compare a `expected` ;
   - retourne `{"passed": int, "failed": int, "failures": list, "error": str | None}` ; capture `ZeroDivisionError`, `AssertionError`, etc. comme un echec de test, et une `SyntaxError`/`NameError` a la compilation comme `error` (l'edit a casse le module).
3. Ecris une suite d'**edit blocks candidats** (liste ordonnee de `(filename, find, replace)`) que l'agent essaiera dans l'ordre — c'est le "plan".
4. Ecris `plan_edit_test_loop(repo, tests, candidate_edits, max_iters=10) -> dict` qui :
   - lance les tests ; si deja vert, s'arrete ;
   - sinon applique le **prochain** edit candidat (refuse proprement si le `find` est absent et passe au suivant), relance les tests ;
   - **garde l'edit seulement s'il fait progresser** (strictement plus de tests passes qu'avant) ; sinon **revient en arriere** (rollback de cet edit) ;
   - s'arrete quand tout est vert ou que les candidats / `max_iters` sont epuises ;
   - retourne `{"green": bool, "iters": int, "final_repo": dict, "history": list}`.
5. Teste avec assertions :
   - un repo avec 1 bug (`add` qui soustrait) est repare par le bon edit candidat → `green=True` ;
   - un edit candidat **inutile** (qui ne fait pas progresser) est bien **rollbacke** (le repo final ne le contient pas) ;
   - un repo dont **aucun** candidat ne corrige le bug se termine avec `green=False` sans planter.

### Criteres de reussite
- [ ] Le mock repo est un dict `nom_fichier -> source`
- [ ] `run_tests` execute en process et distingue `failed` (assertion) de `error` (module casse)
- [ ] La boucle applique les edits candidats dans l'ordre et relance les tests apres chaque edit
- [ ] Un edit qui ne fait pas progresser est rollbacke (verifie : il n'apparait pas dans le repo final)
- [ ] La boucle s'arrete au vert ET sur un bug non corrigeable (`green=False`, pas de boucle infinie)
- [ ] Execution offline, stdlib pur, deterministe
