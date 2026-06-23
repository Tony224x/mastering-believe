# Exercices — Benchmarking pratique (J26)

---

## Exercice 1 : Enrichir un EvalCase avec des metatdonnees de criticite

### Objectif

Comprendre que tous les cas d'evaluation ne sont pas egaux : un cas metier critique (commande annulee a tort, donnee effacee) doit bloquer le deploy a lui seul, meme si 10 autres cas s'ameliorent.

### Consigne

En partant de `02-code/26-benchmarking-pratique.py` :

1. Ajoute un champ `priority: str` a `EvalCase` avec les valeurs possibles `"low"`, `"medium"`, `"critical"`.
2. Modifie `regression_report()` pour qu'elle accepte un parametre `block_on_priority: str = "critical"`.
3. Si un cas avec `priority == "critical"` (ou la valeur de `block_on_priority`) regresse, le verdict doit etre `"BLOCKED -- critical regression"`, independamment des autres resultats.
4. Cree un dataset de 4 cas :
   - `CMD-cancel` : priority=critical (doit appeler `search_order` puis `cancel_order`)
   - `CMD-status` : priority=medium
   - `CMD-edge` : priority=low
   - `CMD-email` : priority=low
5. Cree deux agents mock :
   - `agent_safe` : error_rate=0.1 (toujours meilleur sauf sur CMD-cancel ou il regresse de 0.8 -> 0.2)
   - Simule la regression en surchargeant l'agent pour le cas `CMD-cancel` uniquement
6. Appelle `regression_report(baseline, candidate, ...)` et verifie que le verdict est `BLOCKED` meme si les 3 autres cas s'ameliorent.

### Criteres de reussite

- [ ] `EvalCase` a un champ `priority` avec une valeur par defaut `"medium"`
- [ ] `regression_report` detecte la regression sur le cas critical
- [ ] Le verdict est `"BLOCKED"` meme si le score global s'ameliore
- [ ] Le rapport affiche `[CRITICAL -- BLOCKING]` a cote du cas bloquant
- [ ] Le code tourne sans erreur et le verdict BLOCKED est visible dans la sortie

---

## Exercice 2 : Calculer l'intervalle de confiance sur pass^k

### Objectif

Comprendre que pass^k est une estimation probabiliste avec une variance qui depend de k. Tirer des conclusions sur la fiabilite d'un agent necessite de savoir si la difference entre baseline et candidate est statistiquement significative.

### Consigne

1. La fonction `p_hat_confidence_interval(successes, k, z=1.96)` de `02-code/26-benchmarking-pratique.py` donne l'IC sur `p_hat`. En utilisant la **propagation d'erreur** (delta method), derive l'IC sur `pass^k = p_hat^k` :

   ```
   Var(pass^k) ≈ (k * p_hat^(k-1))^2 * Var(p_hat)
   Var(p_hat) = p_hat * (1 - p_hat) / n    (n = nombre de runs)
   IC sur pass^k = pass^k +/- z * sqrt(Var(pass^k))
   ```

2. Implemente une fonction `pass_k_confidence_interval(successes: int, n_runs: int, k: int, z: float = 1.96) -> tuple[float, float]`.

3. Cree une fonction `is_improvement_significant(base_successes, cand_successes, n_runs, k, z=1.96) -> bool` qui retourne `True` si les intervalles de confiance ne se chevauchent pas (amelioration statistiquement significative).

4. Sur le dataset de la demo (`DEMO_CASES`), affiche pour chaque cas :
   - `p_hat` baseline et candidate
   - `pass^k` avec son IC a 95% pour les deux agents
   - Si la difference est significative (bool)

5. Utilise `k=5` et `n_runs=5` (coherent avec la demo).

### Criteres de reussite

- [ ] `pass_k_confidence_interval` implemente la formule delta method
- [ ] Les bornes sont clampees dans `[0, 1]`
- [ ] `is_improvement_significant` detecte le chevauchement des IC
- [ ] L'affichage montre clairement les IC pour chaque cas
- [ ] Au moins 1 cas montre une amelioration significative et 1 cas montre une difference non significative

---

## Exercice 3 : Harness multi-agent avec classement

### Objectif

Comparer N agents sur le meme dataset et produire un classement par pass^k, comme un leaderboard interne avant de choisir quel agent deployer.

### Consigne

1. Cree une liste de 4 agents mock avec des profils differents :
   - `agent_conservative` : error_rate=0.05 (tres fiable, mais lent -> max_steps souvent depasse)
   - `agent_balanced`     : error_rate=0.20 (profil standard)
   - `agent_creative`     : error_rate=0.35 (moins fiable mais toujours dans le budget)
   - `agent_broken`       : error_rate=0.60 (version cassee)

2. Pour simuler que `agent_conservative` depasse souvent le budget, surchage son `_successful_run` pour doubler le nombre de steps (simulate `steps = steps * 2`).

3. Lance `run_suite(agent, DEMO_CASES, k=5)` pour chaque agent (reutilise `DEMO_CASES` depuis le module).

4. Produis un tableau de classement tri par `mean_pass_k` decroissant :

   ```
   LEADERBOARD (pass^5, k=5 runs par cas)
   =======================================
   Rang | Agent              | mean_p_hat | mean_pass^5 | Golden pass^5
   -----|--------------------|-----------:|------------:|-------------:
    1   | agent_balanced     |      0.80  |        0.33 |          0.45
    2   | agent_conservative |      0.75  |        0.24 |          0.30
    ...
   ```

5. Ajoute une colonne `Golden pass^5` : la moyenne de pass^5 uniquement sur les cas tagges `golden`.

6. Le classement doit etre affiche en ordre decroissant de `mean_pass_k`.

### Criteres de reussite

- [ ] Les 4 agents sont evalues sur le meme dataset avec le meme k
- [ ] `agent_conservative` depasse le budget sur les cas nominaux (steps > max_steps)
- [ ] Le tableau est trie par `mean_pass_k` decroissant
- [ ] La colonne `Golden pass^5` est correctement calculee (cas avec tag "golden" uniquement)
- [ ] Le code tourne sans erreur et le leaderboard est lisible
