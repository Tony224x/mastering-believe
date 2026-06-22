# Exercices — Capstone build & eval (J28)

---

## Exercice 1 : Ajouter un cas d'eval golden

### Objectif

Etendre le harness d'evaluation avec un nouveau cas critique et verifier qu'il participe au gate de regression.

### Consigne

En partant de `02-code/28-capstone-build-eval.py` :

1. Cree un nouvel `EvalCase` `id="no-forbidden"` qui exige la trajectoire `["plan","code","run_tests","verify"]`, `max_steps=12`, tag `golden`, `expected="SUCCESS"`.
2. Construis une liste `CASES = DEMO_CASES + [le nouveau cas]`.
3. Lance `run_suite` pour un agent fiable (`error_rate=0.05`) et un agent casse (`error_rate=0.9`) sur `CASES`.
4. Passe les deux rapports a `regression_report` et verifie que le verdict candidate (casse) vs baseline (fiable) est `BLOCKED` (golden regression).

### Criteres de reussite

- [ ] Le nouveau cas golden est ajoute et evalue
- [ ] L'agent casse a un pass^k nettement plus bas sur les cas golden
- [ ] `regression_report` retourne un verdict `BLOCKED`
- [ ] Le flag `GOLDEN REGRESSION` est affiche
- [ ] Le code tourne sans erreur

---

## Exercice 2 : Un sous-agent supplementaire (linter)

### Objectif

Ajouter un sous-agent a contexte isole dans le pipeline du DeepOpsAgent.

### Consigne

1. Cree `LinterSubAgent(SubAgent)` dont `run(task)` lit `calc.py` dans le repo jouet et renvoie `"lint-ok"` si la ligne `return a + b` est presente, sinon `"lint-fail"`.
2. Sous-classe `DeepOpsAgent` en `DeepOpsAgentWithLint` dont `solve` insere une etape `lint` **apres** `code` et **avant** `verify`, en ajoutant `"lint"` a la trajectoire.
3. Le `LinterSubAgent` doit lire le meme `VirtualFS` que le coder (partage du repo) — attention a reutiliser le bon `fs`.
4. Teste un run reussi : la trajectoire contient `lint` et l'output reste `SUCCESS`.

### Criteres de reussite

- [ ] `LinterSubAgent` a son propre contexte isole (herite de `SubAgent`)
- [ ] L'etape `lint` apparait dans la trajectoire entre `code` et `verify`
- [ ] Le lint passe quand le fix est applique
- [ ] Le run complet reste `SUCCESS`
- [ ] Le code tourne sans erreur

---

## Exercice 3 : Critere de scoring sequence-aware

### Objectif

Renforcer le `score` pour qu'il verifie l'**ordre** des etapes, pas seulement leur presence.

### Consigne

1. Ecris `score_ordered(result, case, ordered_steps)` qui retourne True seulement si `ordered_steps` apparait comme **sous-sequence ordonnee** de `result.trajectory` (les etapes dans le bon ordre, pas forcement contigues), en plus des criteres final + budget existants.
2. Teste sur une vraie trajectoire du `DeepOpsAgent` :
   - `["plan","research","code","verify"]` → True
   - `["verify","plan"]` → False (ordre viole)
3. Compare avec le `score` original (presence seule) sur la trajectoire `["verify","plan","code","research"]` melangee : `score` (presence) doit etre True, `score_ordered` doit etre False.

### Criteres de reussite

- [ ] `score_ordered` verifie la sous-sequence ordonnee
- [ ] Les criteres final-answer et budget restent appliques
- [ ] Une trajectoire dans le desordre echoue a `score_ordered` mais passe `score`
- [ ] Au moins 2 cas de test (ordre ok / ordre viole) sont affiches
- [ ] Le code tourne sans erreur
