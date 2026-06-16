# Exercices Hard — Benchmarking pratique (J26)

> Defi. Tout reste **simule, deterministe et offline** : agents seeded, scores reproductibles, aucune API. On construit ici un harness d'eval complet et un audit de stabilite de leaderboard — les deux pieges majeurs du benchmarking d'agents (Goodhart, instabilite de classement).

---

## Exercice 1 : Harness d'eval reproductible multi-metrique + detection de derive Goodhart

### Objectif

Construire un harness d'evaluation **reproductible et seede** qui mesure trois axes (qualite, cout, latence simules), estime la **variance** de la qualite entre runs, puis prouve la **loi de Goodhart** : un agent **sur-optimise sur le leaderboard** (set d'eval public) s'effondre sur un **holdout** (set cache), tandis qu'un agent honnete tient sur les deux (sections 4, 5 et 6 du cours).

### Consigne

En t'inspirant de `EvalCase`, `MockAgent`, `Scorer` et `run_suite` de `02-code/26-benchmarking-pratique.py` (recopie le minimum necessaire, **n'importe pas** le module) :

1. Definis deux datasets disjoints de cas : `leaderboard_set` (cas "publics", potentiellement memorisables) et `holdout_set` (cas "caches", meme distribution de difficulte mais inedits).
2. Implemente un `SeededAgent` deterministe parametre par :
   - `quality_pub` : proba de succes sur les cas du leaderboard,
   - `quality_holdout` : proba de succes sur le holdout,
   - un `cost_per_run` et `latency_per_run` simules (bruit borne via `random.Random(seed)`).
   Un agent **honnete** a `quality_pub == quality_holdout` ; un agent **Goodhart** a `quality_pub` tres eleve mais `quality_holdout` bien plus bas (il a overfit le leaderboard).
3. Implemente `run_eval(agent, cases, k, seed) -> dict` qui, pour chaque cas, fait `k` runs seedes et retourne par cas `p_hat`, `pass_k`, et agrege au niveau suite : `mean_pass_k`, `mean_cost`, `mean_latency`, et la **variance** de `p_hat` entre cas (`statistics.pvariance`).
4. **Reproductibilite** : lance `run_eval` DEUX fois avec le **meme** seed et verifie par assertion que les sorties sont **identiques** (memes `mean_pass_k`, meme variance). Relance avec un seed different et montre que ca peut changer.
5. **Goodhart drift** : pour l'agent Goodhart, calcule `drift = mean_pass_k(leaderboard) - mean_pass_k(holdout)` et verifie qu'il est **grand et positif** ; pour l'agent honnete, verifie que `drift` est **proche de 0**. Conclus : le score leaderboard seul est trompeur.
6. Produis un mini-dashboard imprime : par agent, une ligne `qualite_pub | qualite_holdout | drift | cout | latence` et un verdict `OVERFIT` / `ROBUST`.

### Criteres de reussite

- [ ] `SeededAgent` est entierement deterministe (meme seed -> memes runs)
- [ ] `run_eval` agrege qualite (pass^k), cout, latence ET variance de p_hat entre cas
- [ ] La reproductibilite est prouvee par assertion (deux runs identiques au meme seed)
- [ ] L'agent Goodhart montre un `drift` leaderboard->holdout grand et positif ; l'honnete ~0 (assertions)
- [ ] Le dashboard imprime distingue clairement `OVERFIT` de `ROBUST`

---

## Exercice 2 : Audit de stabilite de leaderboard sous bootstrap (le classement change-t-il ?)

### Objectif

Prouver qu'un classement de leaderboard base sur un **echantillon fini** de cas est **instable** : sous reechantillonnage bootstrap des cas, l'ordre des agents change. Un ecart de score qui semble decisif peut etre du bruit d'echantillonnage — piege classique des leaderboards d'agents (sections 1 et 4 du cours).

### Consigne

1. Construis `N >= 4` agents mock avec des `mean_pass_k` **vrais mais proches** (ex. 0.40, 0.42, 0.44, 0.46), simules de facon deterministe : chaque agent a, sur chaque cas du dataset, un `pass_k` par cas tire d'un `random.Random(seed)` autour de sa moyenne vraie (bruit borne, clamp dans `[0, 1]`).
2. Construis le **classement nominal** : moyenne du `pass_k` par agent sur **tous** les cas, trie decroissant. Affiche-le.
3. Implemente `bootstrap_rankings(agent_scores, n_boot, seed) -> dict` ou `agent_scores[agent]` est la liste des `pass_k` par cas. Pour chaque iteration bootstrap : tire un echantillon de cas **avec remise** (memes indices pour tous les agents), recalcule la moyenne par agent, et enregistre le **rang** de chaque agent (1 = meilleur).
4. Calcule pour chaque agent : sa **distribution de rangs** (combien de fois rang 1, rang 2, ...) et la **probabilite qu'il finisse #1**. Verifie par assertion qu'**aucun** agent n'a P(rang 1) == 1.0 (le classement n'est PAS stable) et que la **somme** des P(rang 1) sur tous les agents vaut ~1.0.
5. Calcule la **frequence d'inversion top-1/top-2** : la proportion d'echantillons bootstrap ou le #1 nominal n'est PAS le #1 bootstrap. Verifie qu'elle est `> 0` (le classement bascule parfois).
6. **Contre-exemple stable** : ajoute un agent **dominant** (mean_pass_k = 0.90, tres au-dessus des autres) et montre que LUI finit #1 dans ~100 % des echantillons bootstrap (P(rang 1) >= 0.99) — un ecart franc est, lui, robuste.
7. Imprime un rapport d'audit : tableau `agent | rang_nominal | P(#1) | rang median bootstrap` + la conclusion sur la stabilite.

### Criteres de reussite

- [ ] >= 4 agents aux scores **proches** + 1 agent **dominant** comme contre-exemple
- [ ] `bootstrap_rankings` reechantillonne les cas **avec remise** (memes indices pour tous les agents) et recalcule les rangs
- [ ] La distribution de rangs et `P(#1)` sont calculees ; somme des `P(#1)` ~ 1.0 (assertion)
- [ ] Le classement serre est **instable** : frequence d'inversion top-1 > 0 ET aucun agent serre n'a `P(#1) == 1.0` (assertions)
- [ ] L'agent dominant est **stable** : `P(#1) >= 0.99` (assertion) — un ecart franc resiste au bootstrap
- [ ] Le rapport d'audit est imprime lisiblement et tout est deterministe
