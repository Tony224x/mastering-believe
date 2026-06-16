# Exercices Hard — Coding agents : mini-agent SWE & transactions d'edition (J21)

---

## Exercice 1 : Mini coding-agent qui localise, corrige et itere sur un bug seede

### Objectif
Assembler toute la boucle du module en un **mini coding-agent SWE-bench-like** : a partir d'un test qui echoue sur un mock repo, l'agent doit **localiser** le fichier/la fonction fautive (via des signaux simples : nom du test, traceback-like, recherche du symbole), **proposer** un edit via l'outil de search/replace, l'**appliquer**, **relancer** les tests, et **iterer** jusqu'au vert ou epuisement du budget. Tu dois prouver les deux comportements critiques : il **repare** un bug seede, et il **s'arrete proprement** (sans boucle, sans casser le repo) sur un bug qu'il ne sait pas corriger.

### Consigne
Tu construis tout en memoire (mock repo = dict `nom_fichier -> source`), offline et deterministe.

1. **Mock repo + tests** : `repo: dict[str, str]` et une suite `tests: list[tuple[fn_name, args, expected]]`. Sème au moins **deux** bugs independants (ex : `add` qui fait `a - b`, `mul` qui fait `a + b`).
2. **Runner** : `run_tests(repo, tests) -> dict` (en process, dict frais via `exec`) qui renvoie `{"passed", "failed", "failures", "error"}`. Chaque `failure` doit inclure le `fn_name` du test echoue — c'est le **signal de localisation**.
3. **Localizer** : `localize(repo, failing_fn) -> tuple[str, int] | None` qui, a partir du nom de fonction d'un test echoue (ex : `add`), cherche dans le repo le fichier + la ligne ou `def <fn>` est definie (search du symbole). Retourne `(filename, lineno)` ou `None` si introuvable.
4. **Banque de correctifs** : une fonction `propose_edit(repo, filename, fn_name) -> (find, replace) | None` qui propose un edit block plausible pour la fonction localisee (tu peux mapper des patterns connus : `return a - b` → `return a + b` pour `add`, etc.). Si elle ne sait pas corriger → `None`.
5. **Agent** : `mini_swe_agent(repo, tests, max_iters=12) -> dict` qui boucle :
   - lance les tests ; si vert → stop, succes ;
   - prend le **premier** test echoue, en extrait le `fn_name`, appelle `localize` puis `propose_edit` ;
   - applique l'edit via un `apply_edit` qui **refuse** un `find` absent/ambigu (reutilise l'esprit du medium) ;
   - relance les tests ; **garde l'edit s'il fait progresser**, sinon rollback ;
   - si aucun correctif n'est proposable pour ce bug (ou s'il ne fait pas progresser) → **note l'echec de localisation/fix** et passe ou s'arrete pour eviter la boucle ;
   - respecte `max_iters` (budget) ;
   - retourne `{"green": bool, "iters": int, "fixed": list[str], "final_repo": dict, "trajectory": list}`.
6. **Prouve par assertions** :
   - sur le repo a 2 bugs seedes, l'agent atteint `green=True`, `fixed` contient les 2 fonctions, et `iters <= max_iters` ;
   - sur un repo avec un bug **non corrigeable** (aucun `propose_edit` ne matche), l'agent termine `green=False`, **sans modifier** la partie du repo qu'il ne sait pas reparer, et **sans depasser** `max_iters` (pas de boucle infinie) ;
   - chaque edit conserve dans le repo final a strictement augmente le nombre de tests passes (verifie via la trajectory).

### Criteres de reussite
- [ ] `localize` retrouve le fichier + ligne d'une fonction a partir du nom du test echoue
- [ ] `propose_edit` renvoie un edit block pour un bug connu, `None` pour un inconnu
- [ ] L'agent applique l'edit, relance les tests, et **rollback** tout edit qui ne fait pas progresser
- [ ] Sur 2 bugs seedes : `green=True`, les 2 fonctions sont dans `fixed`, `iters <= max_iters`
- [ ] Sur un bug non corrigeable : `green=False`, repo non casse, budget respecte (pas de boucle)
- [ ] La trajectory trace chaque (localisation, edit, resultat de test) ; execution offline & deterministe

---

## Exercice 2 : Systeme d'edition transactionnel atomique (stage → lint+test gate → commit/rollback)

### Objectif
Construire un systeme d'**edition transactionnelle** sur un mock repo multi-fichiers : on **stage** plusieurs edits a travers plusieurs fichiers, on passe un **gate** (lint + tests) sur l'etat candidat, et on ne **commit atomiquement** que si tout passe — sinon on **rollback la totalite** (aucun edit appliqué). C'est la garantie "tout ou rien" qu'un coding agent doit offrir pour ne jamais laisser un repo dans un etat partiellement casse. La propriete a prouver est l'**atomicite** : un echec partiel laisse le repo strictement inchange.

### Consigne
1. **Repo & staging** : `repo: dict[str, str]`. Construis une classe `EditTransaction` qui prend une **copie** du repo a l'ouverture (snapshot), accumule des edits stages sans toucher au repo reel.
2. **Stage** : `stage(filename, find, replace)` applique l'edit sur la **copie de travail** (working copy) avec la regle d'unicite (refuse `find` absent/ambigu en levant une exception ou en marquant la transaction comme invalide). Le repo reel n'est pas modifie.
3. **Gate** : `gate(tests)` lance, sur la working copy :
   - un **lint** minimal : la source de chaque fichier doit etre compilable (`compile(...)` sans `SyntaxError`) ; sinon le gate echoue ;
   - puis les **tests** en process (comme au medium/hard 1) ; si un test echoue → le gate echoue.
   - retourne `(ok, report)`.
4. **Commit / rollback** :
   - `commit(tests)` : appelle le gate ; **si OK**, recopie la working copy dans le repo reel et renvoie `True` ; **sinon**, ne touche a rien (le repo reste au snapshot) et renvoie `False`.
   - `rollback()` : jette la working copy, le repo reel reste au snapshot.
5. **Scenario a tester** :
   - **(A) Commit reussi** : stage 2 edits sur 2 fichiers differents qui, ensemble, rendent les tests verts → `commit` renvoie `True` et le repo reel contient bien les 2 modifications.
   - **(B) Echec atomique par test** : sur un repo neuf, stage 2 edits dont l'un casse un test (regression) → `commit` renvoie `False` et le repo reel est **strictement identique** au snapshot d'origine (aucun des 2 edits appliqué).
   - **(C) Echec atomique par lint** : stage un edit qui introduit une `SyntaxError` → le gate echoue au lint, `commit` renvoie `False`, repo inchange.
6. **Prouve par assertions** :
   - apres un commit reussi, **toutes** les modifications stagees sont presentes dans le repo reel ;
   - apres un commit refuse (test OU lint), le repo reel `==` snapshot d'origine (egalite de dict prouvee) — **atomicite** ;
   - un edit stage avec un `find` ambigu/absent est rejete et n'entre pas dans la working copy.

### Criteres de reussite
- [ ] `EditTransaction` prend un snapshot et stage les edits sur une working copy (repo reel intact avant commit)
- [ ] `stage` applique la regle d'unicite (refus si `find` absent ou ambigu)
- [ ] Le `gate` enchaine lint (compilable) puis tests, et echoue si l'un des deux echoue
- [ ] `commit` n'ecrit dans le repo reel **que** si le gate passe entierement (tout-ou-rien)
- [ ] Echec par test : repo reel strictement egal au snapshot (atomicite verifiee par assertion)
- [ ] Echec par lint : repo reel strictement egal au snapshot (atomicite verifiee par assertion)
- [ ] Execution offline, stdlib pur, deterministe
