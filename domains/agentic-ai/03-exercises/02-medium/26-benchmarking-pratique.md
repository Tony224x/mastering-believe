# Exercices Medium — Benchmarking pratique (J26)

> Tout est **simule et offline** : agents mock parametres par un `error_rate`, scores deterministes via un `random.Random(seed)`. On ne contacte jamais d'API. L'objectif est de raisonner sur la **fiabilite** (pass^k) et la **significativite** des differences, pas de battre un benchmark public.

---

## Exercice 1 : Estimateur pass@k + intervalle de confiance bootstrap

### Objectif

Distinguer le **potentiel** (pass@k) de la **fiabilite** (pass^k) sur des resultats de runs simules, et quantifier l'incertitude de l'estimation avec un **bootstrap** — en montrant que l'intervalle de confiance se resserre quand le nombre de runs augmente (section 4 du cours).

### Consigne

En t'appuyant sur les fonctions `pass_k` / `pass_at_k` de `02-code/26-benchmarking-pratique.py` :

1. Genere des resultats de runs binaires (succes=1, echec=0) pour un cas dont la vraie proba de succes est `p_true`, via un `random.Random(seed)` : `runs = [1 if rng.random() < p_true else 0 for _ in range(n)]`.
2. Implemente `estimate(runs, k) -> dict` qui retourne `p_hat`, `pass_k` (= `p_hat ** k`) et `pass_at_k` (= `1 - (1 - p_hat) ** k`).
3. Implemente `bootstrap_ci(runs, k, n_resamples, seed, z_level=0.95) -> tuple[float, float]` : tire `n_resamples` echantillons **avec remise** depuis `runs`, recalcule `pass^k` sur chacun, et retourne les percentiles (2.5 %, 97.5 %) de la distribution bootstrap.
4. Compare la **largeur** de l'IC bootstrap pour `n = 10` vs `n = 200` (meme `p_true`, meme seed de tirage des runs) et verifie par assertion que l'IC se resserre (`largeur_200 < largeur_10`).
5. Affiche un petit tableau : pour `p_true in [0.5, 0.7, 0.9]`, montre `p_hat`, `pass^3`, `pass@3` et l'IC bootstrap de `pass^3`.

### Criteres de reussite

- [ ] `estimate` calcule `p_hat`, `pass^k` et `pass@k` de facon coherente avec le cours
- [ ] `bootstrap_ci` reechantillonne **avec remise** et renvoie des percentiles (bornes dans `[0, 1]`)
- [ ] L'IC pour `n = 200` est strictement plus etroit que pour `n = 10` (assertion)
- [ ] Le tableau montre `pass@k > pass^k` pour tout `p_true < 1.0`
- [ ] Le code est deterministe (memes seeds -> memes sorties) et tourne sans erreur

---

## Exercice 2 : Detecteur de contamination de benchmark

### Objectif

Detecter qu'un dataset de test est **contamine** par des items deja vus a l'entrainement (fuite). Un benchmark contamine surestime la performance : c'est un piege majeur des leaderboards (section 1 du cours, limites des benchmarks publics).

### Consigne

1. Definis un `train_set` (liste de chaines, ex. des enonces de taches) et un `test_set`. Certains items de test sont des copies exactes ou quasi-exactes d'items du train.
2. Implemente `normalize(text) -> str` : minuscule, espaces multiples reduits, ponctuation de bord retiree (normalisation simple, stdlib uniquement).
3. Implemente `jaccard(a, b) -> float` sur les ensembles de mots (tokens) de deux chaines normalisees.
4. Implemente `detect_contamination(train_set, test_set, threshold=0.8) -> dict` qui, pour chaque item de test, calcule le Jaccard max contre tout le train et **flague** l'item si ce max `>= threshold`. Retourne `{"flagged": [...ids ou textes...], "contamination_rate": float, "details": {...}}`.
5. Calcule deux scores d'agent simules : le score "brut" sur tout `test_set` et le score "clean" sur `test_set` prive des items flagues. Verifie par assertion que `score_clean <= score_brut` (la contamination gonflait le score) et que `contamination_rate > 0`.
6. Affiche les items flagues et l'ecart de score brut vs clean.

### Criteres de reussite

- [ ] `normalize` + `jaccard` sont purs et stdlib (aucune dependance externe)
- [ ] `detect_contamination` flague les items dont le Jaccard max >= seuil
- [ ] `contamination_rate` = nb_flagues / nb_test, dans `[0, 1]`
- [ ] Le score "clean" (hors items fuites) est <= au score "brut" (assertion)
- [ ] Un item de test totalement different n'est PAS flague (pas de faux positif)

---

## Exercice 3 : Harness A/B avec test de significativite

### Objectif

Comparer deux agents sur le meme dataset et decider si la difference observee est **reelle** ou du **bruit**, via un test de significativite (z-test sur deux proportions). C'est le coeur d'une decision de deploiement (section 6 du cours, rapport de regression).

### Consigne

En reutilisant l'esprit de `MockAgent` et `run_suite` de `02-code/26-benchmarking-pratique.py` :

1. Construis deux agents mock parametres par `error_rate` (A = baseline, B = candidate), evalues sur `k` runs par cas avec un `random.Random(seed)` par agent (deterministe).
2. Agrege sur l'ensemble du dataset : `successes_A / total_A` et `successes_B / total_B` (total = `nb_cas * k`).
3. Implemente `two_proportion_ztest(s_a, n_a, s_b, n_b) -> tuple[float, float]` qui retourne `(z, p_value)` avec la proportion poolee `p = (s_a + s_b) / (n_a + n_b)` et `se = sqrt(p*(1-p)*(1/n_a + 1/n_b))`. Approxime la p-value bilaterale via une `erf` (fournie en stdlib `math.erf`).
4. Implemente `ab_decision(...) -> str` : `"B significativement meilleur"` si `p_value < 0.05` et `prop_B > prop_A`, `"A significativement meilleur"` si `p_value < 0.05` et `prop_A > prop_B`, sinon `"difference non significative (bruit)"`.
5. Cree DEUX scenarios verifiables : (a) un grand ecart d'`error_rate` avec un `n` suffisant -> difference **significative** ; (b) un ecart minuscule -> difference **non significative**. Verifie chaque verdict par assertion.

### Criteres de reussite

- [ ] L'agregation des succes A/B est correcte (total = nb_cas * k)
- [ ] `two_proportion_ztest` utilise la proportion poolee et `math.erf` pour la p-value
- [ ] `ab_decision` renvoie les 3 verdicts attendus selon le seuil 0.05
- [ ] Scenario (a) -> verdict significatif ; scenario (b) -> verdict non significatif (assertions)
- [ ] Tout est deterministe et tourne offline sans erreur
