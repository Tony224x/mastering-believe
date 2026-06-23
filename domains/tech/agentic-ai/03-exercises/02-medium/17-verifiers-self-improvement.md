# Exercices Medium — Verifiers & self-improvement (J17)

---

## Exercice 1 : Best-of-N pilote par un verifier a rubrique

### Objectif
Aller au-dela du best-of-N du module (qui ne note que la valeur finale) : construire un **verifier a rubrique** qui evalue chaque candidate sur plusieurs criteres ponderes, puis prouver que le best-of-N selectionne **bien la candidate qui maximise le score agrege** — pas juste la premiere correcte par hasard.

### Consigne
En t'inspirant de `best_of_n` et `OutcomeVerifier` du module 17 (tu peux les reembarquer dans ta solution) :

1. Definis un `@dataclass RubricCriterion` avec `name: str`, `weight: float`, et `check: Callable[[str], float]` (retourne un score `[0, 1]` pour cette dimension de la candidate).
2. Ecris une classe `RubricVerifier` avec une methode `score(candidate: str) -> float` qui calcule la **moyenne ponderee** des criteres (somme des `weight * check`, normalisee par la somme des poids). Inclus au moins 3 criteres : correction du resultat final, presence d'etapes de verification, et absence d'etape morte (division par zero, valeur negative).
3. Ecris une methode `breakdown(candidate: str) -> dict` qui retourne le detail par critere (utile pour le debug).
4. Implemente `best_of_n_rubric(generator, verifier, target, n) -> tuple[str, float, list[float]]` : genere `n` candidates, score chacune, retourne la meilleure **plus la liste de tous les scores**.
5. **Prouve par assertions** que le score retourne est bien `max(tous_les_scores)`, et qu'aucune autre candidate n'a un score strictement superieur a la candidate selectionnee.
6. Construis un cas-test deterministe ou une candidate "correcte mais sans verification" perd contre une candidate "correcte et qui verifie ses etapes" — montre que la rubrique departage les deux.

### Criteres de reussite
- [ ] `RubricCriterion` est un dataclass avec `name`, `weight`, `check`
- [ ] `RubricVerifier.score` est une moyenne **ponderee** normalisee dans `[0, 1]`
- [ ] La rubrique combine au moins 3 criteres distincts (correction + verification + absence d'etape morte)
- [ ] `best_of_n_rubric` retourne la candidate dont le score == max de tous les scores (prouve par assertion)
- [ ] Un cas-test montre que deux candidates **toutes deux correctes** sont departagees par les criteres de processus
- [ ] Execution offline, deterministe, sans cle API ni dependance

---

## Exercice 2 : Boucle self-refine avec amelioration monotone garantie

### Objectif
Implementer la boucle **generator -> critic -> refiner** (section 4 du cours) et **prouver** qu'elle ameliore un score de qualite a chaque iteration jusqu'a atteindre un seuil ou un budget d'iterations — sans jamais regresser. C'est le piege du self-refine naif : sans garde-fou, un refiner peut **degrader** une bonne reponse.

### Consigne
1. Choisis une tache mesurable : produire une chaine arithmetique qui atteint un `target`. Le score de qualite combine la justesse du resultat ET le nombre d'etapes de verification presentes.
2. Ecris un `critic(output, target) -> str` qui identifie **le defaut le plus grave** restant (resultat faux, ou pas d'etape de verification).
3. Ecris un `refiner(output, critique, target) -> str` qui applique **une** correction ciblee par la critique.
4. Implemente `self_refine_monotone(initial, target, scorer, max_iters, threshold) -> dict` qui :
   - garde une trace `history` de `(iteration, output, score)`,
   - a chaque iteration, calcule le score du candidat raffine et **ne l'accepte que s'il est >= au meilleur score vu jusqu'ici** (garde-fou anti-regression : sinon on conserve l'ancien),
   - s'arrete des que `score >= threshold` (succes) **ou** que `max_iters` est atteint.
5. **Prouve par assertions** :
   - la suite des **meilleurs scores** retenus est **non decroissante** (monotone),
   - le score final est `>= score initial`,
   - sur un cas favorable, le seuil est atteint avant `max_iters` ; sur un cas difficile, on s'arrete proprement au budget sans crasher,
   - le garde-fou anti-regression empeche bien un refiner volontairement nuisible de degrader le score (test dedie avec un refiner qui casse la reponse).

### Criteres de reussite
- [ ] La boucle suit le schema generator/critic/refiner du cours
- [ ] La suite des meilleurs scores retenus est non decroissante (verifie par assertion)
- [ ] L'arret se declenche soit au `threshold`, soit au `max_iters`, sans exception
- [ ] Le garde-fou rejette une revision qui ferait baisser le score (test avec refiner nuisible)
- [ ] `history` trace `(iteration, output, score)` pour chaque tour
- [ ] Execution offline, deterministe

---

## Exercice 3 : Ensemble de verifiers (format + rubrique + tests unitaires)

### Objectif
Combiner **plusieurs verifiers heterogenes** en un verdict agrege, comme en production ou un format-checker, une rubrique de contenu et un runner de tests unitaires votent ensemble. Tu dois gerer le cas ou un verifier est **bloquant** (un echec de tests doit dominer un bon score de format).

### Consigne
1. Definis une interface commune : chaque verifier est un callable `verifier(candidate: str) -> tuple[float, bool]` retournant `(score [0,1], is_blocking_failure)`.
2. Implemente 3 verifiers sur des candidates de **code Python** (sous forme de chaine) cense definir une fonction :
   - `format_checker` : `1.0` si la candidate contient bien `def <nom>(` et `return`, score partiel sinon, jamais bloquant.
   - `rubric_checker` : note la presence d'un docstring et d'une gestion de cas limite (`if`), jamais bloquant.
   - `unittest_runner` : execute la fonction **en memoire** (via `exec` dans un namespace isole) contre une petite batterie de cas `(input, expected)` ; retourne le ratio de tests passes et `is_blocking_failure=True` si **au moins un test echoue ou si le code leve une exception**.
3. Implemente `ensemble_verdict(candidate, verifiers, weights) -> dict` qui retourne :
   - `aggregate`: moyenne ponderee des scores,
   - `accepted`: `True` seulement si `aggregate >= seuil` **ET** aucun verifier bloquant n'a echoue,
   - `details`: le `(score, blocking)` de chaque verifier.
4. **Prouve par assertions** sur 3 candidates :
   - une candidate **correcte** (tests passent, bon format) → `accepted = True`,
   - une candidate **bien formatee mais fausse** (un test echoue) → `accepted = False` malgre un `aggregate` eleve (le bloquant domine),
   - une candidate **qui crashe** (exception a l'execution) → `accepted = False` et le runner remonte le blocage sans faire planter l'ensemble.

### Criteres de reussite
- [ ] Chaque verifier respecte l'interface `(score, is_blocking_failure)`
- [ ] `unittest_runner` execute le code candidat en namespace isole et capture les exceptions sans crasher l'ensemble
- [ ] `ensemble_verdict` combine les scores en moyenne ponderee ET applique la regle bloquante
- [ ] Une candidate bien formatee mais fausse est **rejetee** malgre un bon score de format (verifie par assertion)
- [ ] Une candidate qui leve une exception est rejetee proprement (pas de crash de l'ensemble)
- [ ] Execution offline, deterministe, sans dependance externe
