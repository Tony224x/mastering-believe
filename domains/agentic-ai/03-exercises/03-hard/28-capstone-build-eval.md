# Exercices Hard — Capstone build & eval (J28)

> Ces exercices DURCISSENT le capstone J28 au niveau production : gate de regression `golden` complet et durabilite sous eval.
> Ne reimplemente pas tout le systeme : etends le harness pass^k + `regression_report` et le chemin `run_durable`.
> Les solutions embarquent un mini-`DeepOpsAgent` + harness + `DurableEngine`/`SQLiteCheckpointer` autonomes (offline, deterministe, fix du coder simule sans sous-process).

---

## Exercice 1 : Gate de regression golden — matrice candidat x verdict + ablation du gate

### Objectif

Le cours (section 4-5) decrit un `regression_report` qui **bloque** sur toute chute >= 0.10 d'un cas `golden`. Prouve que ce gate est **necessaire** : construis une baseline et plusieurs candidats, dont un qui **ameliore la moyenne** mais **regresse un cas golden**, et montre que le gate le BLOQUE tandis qu'il APPROUVE un candidat strictement meilleur — puis montre qu'**en ablant le gate**, le mauvais candidat passerait en prod.

### Consigne

Construis un `RegressionGate` autour du harness J28 :

1. **Suite de cas** : au moins 3 cas, dont >= 2 tagues `golden` (ex: les `fix-*`) et 1 non-golden (ex: `budget`).
2. **Baseline** : un `DeepOpsAgent` de reference (ex: `error_rate=0.30`) evalue avec `run_suite` (k fixe, seed fixee).
3. **Candidats** (au moins 3 variantes evaluees avec la **meme** seed/k) :
   - `cand_better` : strictement meilleur partout (ex: `error_rate=0.05`) ;
   - `cand_golden_regress` : meilleur **en moyenne** sur l'ensemble MAIS qui fait **chuter un cas golden de >= 0.10** (astuce : un agent dont le comportement depend du cas, p.ex. fiable sur les cas non-golden mais degrade sur un golden precis) ;
   - `cand_neutral` : globalement equivalent a la baseline.
4. **Gate** : `gate_verdict(baseline, candidate) -> "APPROVED" | "BLOCKED" | "NEUTRAL"` reutilisant la logique de `regression_report` (chute >= 0.10 d'un `golden` → `BLOCKED`, sinon mean superieure → `APPROVED`, sinon `NEUTRAL`).
5. **Matrice** : produis et affiche un tableau `candidat -> verdict`. Asserte : `cand_better → APPROVED`, `cand_golden_regress → BLOCKED`, `cand_neutral → NEUTRAL`.
6. **Ablation du gate** : recalcule le verdict de `cand_golden_regress` **sans** la regle golden (seulement mean candidate > mean baseline). Asserte qu'**il passerait `APPROVED`** — ce qui prouve que le gate golden est ce qui empeche un deploy regressif de partir en prod.

### Criteres de reussite

- [ ] >= 3 cas dont >= 2 `golden`, evalues a seed/k fixes
- [ ] Les 3 candidats sont evalues contre la baseline
- [ ] `cand_better → APPROVED`, `cand_golden_regress → BLOCKED`, `cand_neutral → NEUTRAL` (matrice affichee)
- [ ] Le cas golden regresse bien de >= 0.10 (mesure affichee)
- [ ] L'ablation du gate fait passer `cand_golden_regress` a `APPROVED` (preuve de necessite)
- [ ] Le code tourne sans erreur et est deterministe

---

## Exercice 2 : Durabilite sous eval — crash-resume vs verdict, et preuve d'idempotence

### Objectif

Le cours (section 3) prouve la reprise apres crash en assertant `skipped == ["plan","research","code"]` / `executed == ["verify"]`. Pousse plus loin : combine `run_durable` avec l'eval pour prouver que **la durabilite ne change pas la correction** — sur de nombreux runs seedes, le verdict apres crash+reprise est **identique** au verdict sans crash — et qu'**aucune etape finie n'est re-executee** (preuve d'idempotence via un compteur d'effets de bord).

### Consigne

Construis un test de durabilite sous eval autour de `run_durable` :

1. **Compteur d'effets de bord** : instrumente chaque etape (`plan`, `research`, `code`, `verify`) pour incrementer un compteur global `EXEC_COUNTS[step]` a **chaque execution reelle** de `fn` (pas a chaque skip).
2. **Crash a divers points** : pour `crash_before in ["research", "code", "verify"]`, lance un run durable qui crashe a ce point (process "meurt" : `cp.close()`), puis rouvre **le meme fichier SQLite** et reprends.
3. **Plusieurs seeds** : repete sur >= 5 seeds. Pour chaque (seed, crash_point) :
   - asserte que les etapes **avant** `crash_before` sont dans `skipped` et **ne sont pas re-executees** (le compteur de chacune reste a 1 sur l'ensemble crash+reprise) ;
   - asserte que les etapes a partir de `crash_before` sont dans `executed` apres reprise.
4. **Equivalence de verdict** : pour chaque seed, calcule un `verdict_no_crash` (run durable complet sans crash, nouveau `run_id`/db) et un `verdict_after_resume` (run crashe puis repris). Asserte `verdict_no_crash == verdict_after_resume` pour **toutes** les seeds — la durabilite ne change pas la correction.
5. **Preuve d'idempotence globale** : sur l'ensemble crash+reprise d'un run, asserte que chaque etape a ete executee **exactement une fois** (somme des compteurs == nombre d'etapes), jamais deux fois.

### Criteres de reussite

- [ ] Un compteur d'effets de bord incremente a chaque execution reelle d'etape
- [ ] Crash teste a >= 3 points (`research`, `code`, `verify`) sur >= 5 seeds
- [ ] Les etapes avant le crash sont `skipped` et jamais re-executees (compteur == 1)
- [ ] `verdict_after_resume == verdict_no_crash` pour toutes les seeds (durabilite n'altere pas la correction)
- [ ] Idempotence prouvee : chaque etape executee exactement une fois sur crash+reprise
- [ ] Le code tourne sans erreur (fichiers SQLite temporaires nettoyes)
