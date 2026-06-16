# Exercices Hard — LangGraph avance (J6)

---

## Exercice 1 : Pipeline map-reduce-rerank avec error budget et tolerance aux pannes

### Objectif
Construire un pipeline de recherche complet **fan-out -> reduce -> rerank** qui parallelise N workers via le Send API, tolere des pannes partielles dans une limite (**error budget**), respecte une **borne de concurrence**, puis re-classe les resultats survivants avant de synthetiser.

### Consigne
En partant du stub `MiniLangGraph` (Send + Checkpointer) :

1. **Fan-out (map)** — un node `dispatch(state)` emet un `Send("worker", ...)` par source (au moins 8 sources). Chaque branche recoit son `source` et la `query`.
2. **Workers avec pannes** — `worker(state)` :
   - Retourne un `hit` `{"source", "doc", "raw_score": float, "ok": True}` pour les sources saines
   - Pour les sources marquees "flaky", **simule une panne** (capturee en interne) et retourne `{"source", "ok": False, "error": "..."}`
   - Chaque hit porte aussi le **cout** (`cost`) de l'appel (proxy de tokens/latence)
3. **Error budget + concurrence** — passe dans le state :
   - `max_concurrency` : on ne lance pas plus de `max_concurrency` branches en meme temps (simule le batching : decoupe les Send en vagues, chaque vague <= max_concurrency)
   - `error_budget` : nombre maximal d'echecs tolere. Si le nombre d'echecs **depasse** `error_budget`, le pipeline passe en `status="degraded"` mais **continue** avec ce qu'il a (il ne crashe pas)
4. **Reduce** — `reduce(state)` agrege les hits OK via un reducer custom qui **deduplique par `source`** (garde le `raw_score` max) et calcule des stats : `n_ok`, `n_fail`, `total_cost`.
5. **Rerank** — `rerank(state)` re-classe les hits survivants par un **score composite** : `final_score = raw_score * recency_weight - cost_penalty * cost`, puis garde le **top_k**.
6. **Synthesize** — produit une reponse qui liste les top_k sources dans l'ordre re-classe et expose le `status` (`ok` / `degraded`).

Teste **deux scenarios** sur le meme graph :
- (A) peu de pannes (sous le budget) -> `status="ok"`, top_k correct, dedup applique
- (B) beaucoup de pannes (au-dessus du budget) -> `status="degraded"` mais le pipeline **termine quand meme** avec les survivants

### Criteres de reussite
- [ ] Le fan-out lance N branches via Send ; la borne `max_concurrency` est respectee (vagues <= max_concurrency)
- [ ] Les workers flaky echouent **gracieusement** (aucune exception ne remonte a `invoke`)
- [ ] Le reduce deduplique par `source` (meilleur `raw_score`) et compte `n_ok` / `n_fail`
- [ ] Le rerank applique le score composite et tronque au `top_k`
- [ ] Scenario A : `status="ok"`, l'ordre du top_k est strictement decroissant en `final_score`
- [ ] Scenario B : `status="degraded"`, le pipeline **termine** malgre le depassement du budget
- [ ] Un assert verifie : nombre de vagues, dedup, ordre du rerank, et la bascule de `status` selon le budget

---

## Exercice 2 : Time-travel debugging — fork depuis un checkpoint passe et comparaison de branches

### Objectif
Implementer un veritable workflow de **time-travel debugging** : checkpointer chaque step, lister l'historique, **forker** depuis un checkpoint arbitraire du passe en appliquant un **input alternatif**, puis **comparer** les deux branches (originale vs alternative) step par step.

### Consigne
En partant du `Checkpointer` du stub (etends-le si besoin avec `history` / `load_at` / `update_state`) :

1. **Graph deterministe** — au moins 4 nodes (`route -> compute -> validate -> finalize`) ou une **decision de branchement** depend d'un champ du state (ex : `mode` ou un seuil numerique). Le node `route` choisit la suite selon ce champ — donc changer ce champ dans le passe change la trajectoire.
2. **Run original** — execute sur `thread_id="main"` avec un checkpoint **a chaque step**. Recupere l'historique complet (`get_state_history` / `history`).
3. **Inspection** — affiche, pour chaque snapshot : le step, le node a venir (ou execute), et les valeurs cles du state. Repere un checkpoint **intermediaire** (ex : juste apres `route`, avant `compute`).
4. **Fork** — a partir de ce checkpoint intermediaire :
   - Charge le state passe
   - Applique un **override** (ex : change `mode` / le seuil) — c'est l'`update_state` de LangGraph
   - Re-execute sur un **nouveau** `thread_id="fork"` a partir de ce point modifie
5. **Comparaison** — ecris une fonction `compare_branches(ckpt, "main", "fork")` qui :
   - Aligne les deux historiques step par step
   - Montre **a partir de quel step** les deux branches divergent (premier champ qui differe)
   - Resume la difference de resultat final (`finalize`) entre les deux branches
6. **Invariant** — le thread `main` doit rester **strictement intact** apres le fork (le fork n'ecrit que dans son propre thread).

### Criteres de reussite
- [ ] Chaque step du run original est checkpointe (l'historique a une entree par step)
- [ ] Le fork part d'un checkpoint **intermediaire** (pas du START, pas de la fin)
- [ ] L'override modifie un champ qui **change la trajectoire** (les deux branches divergent reellement)
- [ ] `compare_branches` identifie le **premier step de divergence** correctement
- [ ] Les resultats finaux des deux branches **different** (preuve que le fork a eu un effet)
- [ ] Le thread `main` est **inchange** apres le fork (assert sur l'historique original)
- [ ] Un assert prouve : nb de checkpoints, point de divergence, et resultats finaux distincts

---

## Exercice 3 : Supervisor subgraph qui dispatche vers des worker subgraphs

### Objectif
Composer une **hierarchie de graphs** : un supervisor (graph parent) route une requete vers le bon **worker subgraph** specialise, chaque worker etant un graph compile a part entiere avec son **propre schema de state** mappe vers/depuis le parent.

### Consigne
En partant du stub (composition de subgraphs + conditional edges) :

1. **Trois worker subgraphs** independants, chacun compile seul, chacun avec **son propre state schema** (transformed state, pas shared) :
   - `math_subgraph` : 2 nodes (`parse -> compute`) — resout une mini-operation
   - `text_subgraph` : 2 nodes (`tokenize -> summarize`) — resume un texte
   - `lookup_subgraph` : 2 nodes (`search -> format`) — simule une recherche factuelle
2. **Supervisor (parent graph)** :
   - Un node `classify(state)` determine la **categorie** de la requete (`math` / `text` / `lookup`) — heuristique simple sur le contenu
   - Des **conditional edges** routent vers un **wrapper** par worker. Chaque wrapper :
     - Extrait le sous-ensemble du parent state attendu par le worker (input mapping)
     - Invoque le worker subgraph compile
     - **Re-mappe** la sortie du worker vers le parent state (output mapping) — y compris le champ `answer` et un champ `handled_by`
   - Un node `finalize(state)` formate la reponse finale en exposant **quel worker** a traite la requete
3. **Routing par defaut** — si aucune categorie ne matche, route vers un wrapper `fallback` qui renvoie une reponse polie "je ne sais pas traiter ca".
4. **Tests** — envoie **au moins une requete par categorie** (+ une requete fallback) et verifie pour chacune :
   - Qu'elle a ete routee vers le bon worker (`handled_by`)
   - Que la `answer` est coherente avec ce worker
   - Que chaque worker subgraph reste **testable en isolation** (invocation directe sans le supervisor)

### Criteres de reussite
- [ ] Les 3 worker subgraphs ont chacun **leur propre schema** et sont compilables/testables seuls
- [ ] Le supervisor route via des conditional edges vers le bon wrapper selon `classify`
- [ ] Chaque wrapper fait un **input mapping** ET un **output mapping** explicites (pas de state partage)
- [ ] `handled_by` indique correctement le worker pour chaque requete
- [ ] Le cas `fallback` est gere (aucune categorie reconnue -> reponse par defaut)
- [ ] Chaque worker invoque en isolation produit le bon resultat (assert dedie)
- [ ] Un assert verifie le routing correct pour les 4 cas (math / text / lookup / fallback)
