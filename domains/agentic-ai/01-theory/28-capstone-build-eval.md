# J28 — Capstone (build & eval) : assembler et evaluer le deep ops agent

> **Temps estime** : 5h | **Prerequis** : J1-J27
> **Objectif** : assembler les briques du J27 en un agent complet runnable, lui faire reparer un bug de bout en bout, prouver la reprise apres crash, et l'evaluer avec un harness pass^k + rapport de regression.

---

## 1. Vue d'ensemble du build

Le code complet vit dans `02-code/28-capstone-build-eval.py`. Il **importe les briques du J27** (`import_module("27-capstone-architecture")`) et ajoute la couche agent :

```
DeepOpsAgent
 ├── _plan(fs, task)          -> ecrit todo.md (deep agent / J15)
 ├── ResearchSubAgent.run()   -> research.md          (contexte isole)
 ├── CoderSubAgent.run()      -> edit/search/run loop  (coding tool / J21+J23)
 ├── VerifierSubAgent.run()   -> verdict               (verifier / J17)
 ├── solve(task) -> AgentResult(output, trajectory, steps)   # 1 run complet
 └── run_durable(run_id, cp)  -> DurableEngine (reprise apres crash / J20)
```

Deux chemins d'execution coexistent :
- **`solve()`** : un run complet en memoire, renvoie une trajectoire (utilisee par l'eval) ;
- **`run_durable()`** : les memes etapes sous le `DurableEngine`, pour la demo de durabilite.

> **Analogie** : `solve()` est le "vol normal" ; `run_durable()` est le meme vol avec une **boite noire** (SQLite) qui permet de redecoller exactement la ou on s'est arrete apres une panne.

---

## 2. L'outil de coding : edit / search / run

Le `CoderSubAgent` cree un mini-repo jouet (un `calc.py` avec un bug planté `return a - b`, un `test_calc.py`), puis :

1. `run_tests()` (rouge) → 2. `search` la ligne fautive → 3. `edit` (`a - b` → `a + b`) → 4. `run_tests()` (vert).

Les tests tournent dans un **sous-process** (`subprocess.run([... , "-B", "test_calc.py"])`), ce qui isole l'execution du code genere (principe J23).

> **Piege classique a connaitre** : si l'edition et le run precedent tombent dans la **meme seconde**, Python peut reutiliser un `.pyc` perime et masquer le fix. Le capstone nettoie `__pycache__` et passe `-B` avant chaque run. C'est un vrai bug d'agent de code qu'on rencontre en pratique — le mentionner en revue est un signe de maturite.

---

## 3. La reprise apres crash (durabilite)

```python
cp = SQLiteCheckpointer(db)            # fichier durable
try:
    agent.run_durable(run_id, cp, crash_before="verify")  # crash simule
except CrashSignal: ...
cp.close()                             # le process "meurt"

cp2 = SQLiteCheckpointer(db)           # nouveau process, MEME fichier
ctx = agent.run_durable(run_id, cp2)   # reprise
# skipped == ["plan","research","code"] ; executed == ["verify"]
```

Le test du capstone **asserte** ce comportement : les etapes deja faites ne sont pas refaites. C'est la preuve concrete de la durabilite, pas une simple affirmation.

---

## 4. Le harness d'evaluation : pass^k + regression

On evalue l'agent **sur lui-meme** (J26, pas le catalogue J11) :

- `EvalCase(id, task, expected, required_steps, max_steps, tags)` ;
- `score(result, case)` = (final contient `expected`) ET (toutes les `required_steps` presentes) ET (budget respecte) ;
- on lance chaque cas **k fois** → `successes/k = p_hat` → **pass^k = p_hat^k** (fiabilite : les k essais reussissent tous) ;
- `regression_report(baseline, candidate)` compare deux versions de l'agent et **bloque** sur toute regression d'un cas `golden`.

La variance necessaire au pass^k vient d'un `error_rate` injecte : avec une proba `error_rate`, le coder "abandonne" (pas de fix) → le cas echoue. Un agent v1.0 (error_rate=0.40) vs v1.1 (0.05) montre une amelioration nette du pass^k et un verdict `APPROVED`.

> **Pourquoi pass^k et pas accuracy** : un agent qui reussit 8 fois sur 10 a une accuracy de 80% mais un pass^5 de seulement 0.8^5 ≈ 0.33. En production, l'utilisateur veut que ca marche **a chaque fois** : pass^k capture cette exigence de fiabilite la ou l'accuracy la cache.

---

## 5. Lire le rapport

```
case              base p^k  cand p^k    delta  flags
budget               0.078     0.328   +0.250
fix-add              0.328     1.000   +0.672
fix-add-2            0.078     1.000   +0.922
mean pass^k : 0.161 -> 0.776
VERDICT     : APPROVED -- candidate is better
```

- Une colonne `flags` `GOLDEN REGRESSION -- BLOCK` apparait si un cas `golden` chute de ≥ 0.10 → deploy bloque.
- Le verdict global resume : `APPROVED` / `NEUTRAL` / `BLOCKED`.

---

## 6. Pistes d'extension (portfolio)

- Brancher un **vrai LLM** (clé optionnelle) derriere l'interface mock du router ;
- remplacer le repo jouet par un **vrai mini-projet** avec plusieurs bugs (rapproche de SWE-bench Lite, J21/J26) ;
- ajouter un **backend Postgres/Redis** au checkpointer (J25) pour le scaling horizontal ;
- ajouter un **A2A endpoint** (J19) pour exposer l'agent a d'autres agents ;
- ajouter des cas d'eval `golden` issus d'incidents reels (regression suite vivante).

---

## Flash-cards

**Q1 :** Quels patterns du parcours le capstone assemble-t-il concretement ?
> **R :** Deep agent + scratchpad (J15), sous-agents a contexte isole (J15/J9), verifier (J17), durabilite/reprise (J20), coding tool edit/search/run en sous-process (J21/J23), routing de cout (J24), harness pass^k + regression (J11/J26).

**Q2 :** Comment le capstone *prouve*-t-il la durabilite plutot que de l'affirmer ?
> **R :** Il simule un crash avant l'etape `verify`, ferme la connexion (process mort), rouvre le meme fichier SQLite, relance, et **asserte** que `skipped == ["plan","research","code"]` et `executed == ["verify"]` — les etapes finies ne sont pas refaites.

**Q3 :** Pourquoi `-B` et le nettoyage de `__pycache__` dans l'outil de coding ?
> **R :** Pour eviter qu'un `.pyc` perime (edition dans la meme seconde que le run precedent) soit reutilise et masque le fix. C'est un bug reel des agents de code.

**Q4 :** D'ou vient la variance qui rend le pass^k informatif ?
> **R :** D'un `error_rate` injecte dans l'agent : avec une certaine proba, le coder abandonne le fix, le cas echoue. Sans variance, pass^k vaudrait 0 ou 1 et ne distinguerait pas deux versions.

**Q5 :** Que declenche une regression sur un cas `golden` ?
> **R :** Le rapport flague `GOLDEN REGRESSION -- BLOCK` et le verdict global devient `BLOCKED`, independamment des autres ameliorations : un cas critique bloque le deploy a lui seul.

---

## Points cles a retenir

- Le capstone est **runnable sans cle API** : `python 28-capstone-build-eval.py` → run complet + reprise sur crash + rapport d'eval
- Il **importe** les briques durables du J27 et ajoute la couche deep-agent (planner + sous-agents isoles)
- L'outil de coding tourne en **sous-process** ; attention au piege du `.pyc` perime
- L'evaluation se fait **sur l'agent lui-meme** avec **pass^k** (fiabilite) et un rapport de regression a gate `golden`
- Extensions portfolio : vrai LLM, vrai repo multi-bugs, backend Postgres/Redis, endpoint A2A

---

## Pour aller plus loin

- `02-code/28-capstone-build-eval.py` (livrable) et `02-code/27-capstone-architecture.py` (briques)
- Yao et al. (Sierra), "tau-bench" (pass^k) — https://arxiv.org/abs/2406.12045
- Jimenez et al., "SWE-bench" — https://arxiv.org/abs/2310.06770
- Synthese des sources : [`REFERENCES.md`](../REFERENCES.md)
