# Exercices Medium — Capstone build & eval (J28)

> Ces exercices ETENDENT le capstone J28 (`02-code/28-capstone-build-eval.py` : `DeepOpsAgent` + harness pass^k + `regression_report`).
> Ne reimplemente pas tout le systeme : ajoute/durcis une metrique d'eval a la fois.
> Les solutions fournies embarquent un mini-`DeepOpsAgent` + harness autonome (mock LLM deterministe, `error_rate` injectable, `random.Random` seede, fix du coder simule sans sous-process) pour tourner offline.

---

## Exercice 1 : pass^k vs pass@k — fiabilite contre couverture

### Objectif

Le cours (section 4) insiste sur **pass^k** (les k essais reussissent **tous**, mesure de fiabilite) plutot que sur l'accuracy. Implemente aussi son cousin **pass@k** (au moins **un** des k essais reussit, mesure de couverture / best-of-k) et montre comment les deux divergent quand l'`error_rate` du `DeepOpsAgent` augmente.

### Consigne

En partant du harness J28 :

1. Sur un `CaseReport` (qui expose deja `p_hat` et `pass_k = p_hat ** k`), ajoute une fonction ou propriete `pass_at_k = 1 - (1 - p_hat) ** k`.
2. Ecris une fonction `compare_pass_metrics(agent, case, k)` qui lance le cas `k` fois, calcule `p_hat`, `pass_k` (pass^k) et `pass_at_k` (pass@k), et retourne les trois.
3. Fais varier l'`error_rate` du `DeepOpsAgent` sur au moins 4 valeurs (ex: `0.0, 0.2, 0.5, 0.8`) avec le **meme** cas et la **meme** seed, et affiche un petit tableau `error_rate | p_hat | pass^k | pass@k`.
4. **Asserte l'ordre fondamental** : pour chaque ligne, `pass_k <= p_hat <= pass_at_k` (avec tolerance numerique). Montre aussi que l'ecart `pass@k - pass^k` se **creuse** quand `error_rate` croit (les deux metriques s'eloignent quand l'agent devient peu fiable).

### Criteres de reussite

- [ ] `pass_at_k = 1 - (1 - p_hat) ** k` est implemente a cote de `pass_k`
- [ ] `compare_pass_metrics` renvoie `p_hat`, `pass^k` et `pass@k` pour un cas
- [ ] L'ordre `pass^k <= p_hat <= pass@k` est asserte sur >= 4 valeurs d'`error_rate`
- [ ] L'ecart `pass@k - pass^k` croit avec l'`error_rate`
- [ ] Le code tourne sans erreur et est deterministe (seed fixee)

---

## Exercice 2 : Intervalle de confiance sur p_hat — quand declarer une vraie amelioration

### Objectif

`p_hat = successes / k` est une **estimation bruitee**. Le cours compare v1.0 vs v1.1 sur pass^k, mais une difference de moyenne peut etre du bruit d'echantillonnage. Ajoute un **intervalle de confiance (IC)** sur `p_hat` et prouve qu'on ne declare une version meilleure que si les IC se **separent**.

### Consigne

1. Implemente `wilson_interval(successes, k, z=1.96) -> (low, high)` (intervalle de Wilson a 95%, en stdlib `math` — pas de SciPy).
2. Etends `CaseReport` (ou enrobe-le) pour exposer `ci = wilson_interval(successes, k)` par cas.
3. Montre que **l'IC se resserre quand k grandit** : pour un agent a `error_rate` fixe, calcule la largeur de l'IC pour `k in {5, 20, 80}` et asserte qu'elle decroit strictement.
4. **Decision robuste** : ecris `is_real_improvement(report_a, report_b)` qui renvoie `True` seulement si l'IC de B est **entierement au-dessus** de celui de A (`B.ci_low > A.ci_high`). Teste deux cas :
   - A et B tres proches (ex: `error_rate` 0.40 vs 0.35) a petit `k` → IC qui se chevauchent → `False` (verdict prudent, on n'avance rien) ;
   - A et B nettement separes (ex: `error_rate` 0.60 vs 0.05) → IC disjoints → `True`.

### Criteres de reussite

- [ ] `wilson_interval` est implemente en stdlib (math seulement)
- [ ] Chaque cas expose un IC sur `p_hat`
- [ ] La largeur de l'IC decroit strictement quand `k` passe de 5 a 20 a 80
- [ ] `is_real_improvement` n'affirme une amelioration que si les IC sont disjoints
- [ ] Les deux scenarios (proche → False, separe → True) sont testes et affiches

---

## Exercice 3 : Axe cout/qualite — quality-per-dollar

### Objectif

pass^k mesure la fiabilite mais ignore le **cout**. Le `ModelRouter` (J24, importe par le capstone) suit `total_cost`. Ajoute un axe **cout** au harness et calcule une metrique **quality-per-dollar**, puis montre qu'un agent moins cher peut **gagner** sur ce ratio meme avec un pass^k legerement inferieur.

### Consigne

1. Fais en sorte que `agent.solve(task)` (ou son `AgentResult`) remonte le `total_cost` du `ModelRouter` du run (somme des appels weak/strong des sous-agents).
2. Etends `run_suite` en `run_suite_cost` qui agrege, en plus du `mean_pass_k`, le **cout moyen par cas** (`mean_cost`) sur les k essais.
3. Definis `quality_per_dollar = mean_pass_k / mean_cost`.
4. Compare deux variantes :
   - `agent_strong` : fiable (`error_rate=0.05`) mais qui route systematiquement vers le modele cher (cout par run eleve) ;
   - `agent_cheap` : un peu moins fiable (`error_rate=0.15`) mais qui route vers le modele weak (cout par run plus bas).
5. **Asserte** : `agent_strong` a un `mean_pass_k` >= celui de `agent_cheap`, MAIS `agent_cheap` a un `quality_per_dollar` strictement superieur. Affiche le tableau `variant | mean_pass^k | mean_cost | quality/$`.

### Criteres de reussite

- [ ] Le cout du run (router) est remonte dans le resultat de `solve`
- [ ] `run_suite_cost` agrege `mean_pass_k` ET `mean_cost`
- [ ] `quality_per_dollar = mean_pass_k / mean_cost` est calcule par variante
- [ ] La variante cheap gagne sur quality/$ malgre un pass^k plus bas
- [ ] Le code tourne sans erreur et le tableau est affiche
