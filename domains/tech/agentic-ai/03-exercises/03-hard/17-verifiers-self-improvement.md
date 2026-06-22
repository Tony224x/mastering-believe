# Exercices Hard — Verifiers & self-improvement (J17)

---

## Exercice 1 : Reward hacking — un verifier naif se fait berner, un verifier durci resiste

### Objectif
Demontrer concretement le **reward hacking** (section 6.4 du cours, terme KL) : un generateur "tricheur" exploite les failles d'un verifier naif pour obtenir un score parfait **sans resoudre la tache**. Tu dois (a) construire un tricheur qui maximise le score du verifier naif, (b) durcir le verifier avec des controles de coherence et adversariaux, et (c) **prouver par assertions** que le naif est berne mais que le durci, lui, attrape la triche.

### Consigne
La tache : produire une chaine de raisonnement qui calcule un `target` arithmetique. Le verifier doit recompenser un **vrai** raisonnement, pas une chaine qui contient juste le bon nombre.

1. **Verifier naif** `NaiveVerifier.score(candidate) -> float` : reprend la logique simple du module — il cherche `answer: X` et donne `1.0` si `X == target`. (C'est sa faille : il ne verifie ni les etapes, ni la coherence interne.)
2. **Generateur tricheur** `gaming_generator(target) -> str` : produit une chaine qui **ment** — par exemple des etapes incoherentes (`step1: 2 * 2 = 5`) suivies du bon `answer: target`, ou bien la cible recopiee sans aucun calcul (`answer: 24` sans rien deriver). Le but : score naif = `1.0`.
3. **Generateur honnete** `honest_generator(target) -> str` : produit une chaine ou les etapes **se recalculent reellement** jusqu'a la cible.
4. **Verifier durci** `HardenedVerifier.score(candidate) -> float` qui ajoute au moins **trois** controles anti-gaming :
   - **coherence interne** : chaque etape `a op b = c` doit reellement evaluer a `c` (sinon penalite forte),
   - **trace de derivation** : la reponse finale doit decouler d'une etape verifiable, pas apparaitre ex nihilo,
   - **perturbation adversariale** : reverifie que les sous-resultats annonces sont arithmetiquement exacts (re-execution independante), et invalide la candidate si une seule etape est fausse.
5. **Prouve par assertions** :
   - `NaiveVerifier` donne `1.0` (ou presque) **a la fois** au tricheur et a l'honnete (il ne les distingue pas),
   - `HardenedVerifier` donne un **score eleve a l'honnete** et un **score bas/nul au tricheur** (il les separe nettement),
   - donc : la triche **fonctionne** contre le naif et **echoue** contre le durci (assertion explicite sur l'ecart de scores).
6. Termine par un best-of-N qui, **avec le verifier durci**, selectionne une candidate honnete meme quand le lot contient des tricheurs — alors que le **meme** best-of-N avec le verifier naif peut elire un tricheur.

### Criteres de reussite
- [ ] `gaming_generator` obtient `1.0` (ou tres proche) du `NaiveVerifier` sans resoudre la tache
- [ ] `HardenedVerifier` integre >= 3 controles distincts (coherence interne, trace de derivation, re-execution adversariale)
- [ ] Assertion : le naif **ne distingue pas** tricheur et honnete ; le durci **les separe** (ecart de score significatif)
- [ ] Best-of-N avec verifier durci selectionne une candidate honnete face a un lot mixte (verifie par assertion)
- [ ] Best-of-N avec verifier naif peut elire un tricheur (contraste demontre)
- [ ] Execution offline, deterministe, sans cle API ni dependance

---

## Exercice 2 : Pipeline generate -> verify -> refine avec verifier par execution de tests

### Objectif
Construire un pipeline complet d'**auto-amelioration pilotee par un verifier execution-based** : a chaque tour, un generateur propose une implementation de fonction, un runner **execute de vrais tests unitaires en memoire**, et un refiner corrige le code selon les tests qui echouent — jusqu'a ce que tous les tests passent ou que le budget soit epuise. Tu dois prouver la **convergence** sur un cas resoluble ET le **rejet correct** d'un probleme insoluble (budget epuise sans crash).

### Consigne
Construis un `ExecutionRefineLoop` qui gouverne le cycle pour une specification de fonction donnee :

1. **Banc de tests** : pour chaque probleme, une liste de cas `(args, expected)`. Ecris `run_tests(func, cases) -> dict` qui execute la fonction en capturant les exceptions et retourne `{passed, total, failures}` (liste des cas echoues avec la valeur obtenue vs attendue). **Aucun test ne doit faire crasher le loop.**
2. **Verifier execution-based** : le score = `passed / total`. Une candidate est `accepted` ssi `passed == total`.
3. **Generateur stub deterministe** `staged_generator(problem, attempt) -> str` : renvoie le **code source d'une fonction** (chaine) qui s'ameliore avec `attempt` — la premiere version est buggee (rate un cas limite), les versions suivantes integrent le feedback. Le code est charge via `exec` dans un namespace isole pour recuperer le callable.
4. **Refiner pilote par les echecs** : `refine(problem, failures, attempt) -> str` choisit la prochaine version du code en fonction des cas qui ont echoue (ex : "echec sur liste vide" → ajoute la garde `if not xs`).
5. **Boucle** `run(problem, max_attempts) -> dict` :
   - genere/raffine, charge le callable, lance `run_tests`,
   - s'arrete des que `passed == total` (convergence) ou a `max_attempts` (budget),
   - retourne `{converged, attempts_used, score_history, final_code}`.
6. **Prouve par assertions** :
   - sur un probleme **resoluble** (ex : `sum_positive(xs)` avec un cas limite liste vide), le loop **converge** (`converged=True`) en `<= max_attempts`, et le `score_history` est **non decroissant** jusqu'a `1.0`,
   - sur un probleme **insoluble par le stub** (la spec demande un comportement que le generateur ne produira jamais), le loop **n'invente pas** un succes : `converged=False`, budget epuise, **aucune exception remontee** (le rejet est propre),
   - une solution finale qui passe tous les tests est bien celle retournee, et une candidate intermediaire incorrecte n'est jamais marquee acceptee.

### Criteres de reussite
- [ ] `run_tests` execute la fonction candidate en isolant et capturant toute exception (zero crash du loop)
- [ ] Le score du verifier est `passed/total` et `accepted` ssi tous les tests passent
- [ ] Sur le probleme resoluble : `converged=True`, `score_history` non decroissant, jusqu'a `1.0`
- [ ] Sur le probleme insoluble : `converged=False`, budget epuise, rejet propre sans exception
- [ ] Une candidate intermediaire incorrecte n'est jamais retournee comme acceptee
- [ ] `run` retourne `converged`, `attempts_used`, `score_history`, `final_code`
- [ ] Execution offline, deterministe, sans dependance externe
