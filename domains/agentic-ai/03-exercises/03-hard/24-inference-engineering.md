# Exercices Hard — Inference engineering (J24)

> **Note** : simulation **100 % deterministe**, en *tokens / steps* (jamais d'horloge murale, jamais de `time.sleep`), aucun reseau, aucun vrai modele. Toute alea est seedee. Chaque exercice se termine par des assertions qui doivent toutes passer.

---

## Exercice 1 : Simulateur de speculative decoding (draft + verify) avec ablation

### Objectif

Modeliser le **speculative decoding** : un petit modele *draft* (rapide) propose un bloc de `k` tokens, le gros modele *target* (lent mais qui fait foi) les **verifie en un seul passage**. Tous les tokens du prefixe accepte sont valides d'un coup ; au premier rejet on garde le prefixe accepte + 1 token corrige par le target, puis on recommence. Le gain de vitesse depend du **taux d'acceptation**. On mesure le speedup vs un decode baseline (target seul, 1 token/step) et on fait une **ablation** : quand l'acceptation s'effondre, le speedup disparait (voire devient negatif a cause du surcout du draft).

### Consigne

En restant dans le style deterministe du module :

1. Modelise les couts en *unites de calcul* (pas de temps reel) :
   - Baseline (target seul) : produire `N` tokens coute `N` steps target (1 token/step).
   - Speculative : a chaque ronde, le draft propose `k` tokens (cout draft = `k * draft_cost` ou `draft_cost < 1`), puis le target fait **1 step de verification** qui valide d'un coup le prefixe accepte et emet 1 token corrige.
2. Cree une fonction `accepted_count(k, accept_prob, rng) -> int` : tire token par token, s'arrete au premier rejet ; retourne le nombre de tokens *draft* acceptes (entre 0 et k).
3. Cree une classe `SpeculativeDecoder(k: int, draft_cost: float, accept_prob: float, seed: int)` avec `generate(n_tokens: int) -> dict` qui boucle des rondes jusqu'a produire au moins `n_tokens`, et retourne :
   - `tokens_produced`, `target_steps` (1 par ronde), `draft_compute` (somme des `k * draft_cost`),
   - `total_cost = target_steps + draft_compute`,
   - `acceptance_rate = total tokens acceptes du draft / total tokens proposes`.
4. Calcule `speedup = baseline_cost / total_cost` ou `baseline_cost = n_tokens` (1 step/token pour le target seul). **Assert** que pour un `accept_prob` eleve (ex. 0.9) on a `speedup > 1.3`.
5. **Ablation** : refais la generation avec un `accept_prob` faible (ex. 0.1) en gardant tout le reste constant ; **assert** que le `speedup` chute nettement (ex. `speedup_low < speedup_high` et `speedup_low` proche de 1 ou inferieur). Affiche un tableau : `accept_prob`, `acceptance_rate`, `total_cost`, `speedup`.

### Criteres de reussite

- [ ] `accepted_count` s'arrete au premier rejet et retourne un entier dans `[0, k]`
- [ ] Une ronde coute `1` step target + `k * draft_cost` de calcul draft
- [ ] Pour un `accept_prob` eleve, `speedup > 1.3` (la speculation paie)
- [ ] L'ablation montre que `speedup_low < speedup_high` quand l'acceptation s'effondre
- [ ] Tout est deterministe (meme `seed` → memes resultats) et l'`acceptance_rate` reportee est coherente avec l'`accept_prob`

---

## Exercice 2 : Optimiseur de config d'inference — frontiere de Pareto sous SLO

### Objectif

Construire un mini-optimiseur qui balaie un espace de configurations de serving — **batch size**, **niveau de quantization**, **prompt caching on/off** — modelise leur effet sur la **latence** et le **cout par requete**, filtre celles qui tiennent un **SLO de latence** ET un **plancher de qualite**, calcule la **frontiere de Pareto** (cout vs latence) parmi les configs valides, et retourne la config **valide la moins chere**. C'est l'aboutissement des trois leviers du cours (routing/quantization implicite, caching, batching) combines sous contrainte.

### Consigne

1. Cree une dataclass `InferenceConfig(batch_size: int, quant: str, cache: bool)` avec `quant in {"fp16", "int8", "int4"}`.
2. Cree un modele de cout/latence deterministe (commente, mais simple) :
   - **Quantization** : un dict `QUANT = {"fp16": (1.00, 1.00, 1.00), "int8": (0.55, 0.75, 0.97), "int4": (0.35, 0.55, 0.90)}` donnant `(cost_factor, latency_factor, quality)`. Plus on quantize, moins c'est cher et plus c'est rapide, mais la qualite baisse.
   - **Batch size** : amortit le cout par requete (`per_req_cost ∝ 1/batch_size` borne) mais **augmente** la latence par requete (file d'attente : `latency ∝ batch_size`). Modelise un compromis explicite.
   - **Cache** : si `True`, applique `cache_cost_factor = 0.4` et `cache_latency_factor = 0.5` (le prefixe partage est servi depuis le KV cache).
3. Cree une classe `InferenceOptimizer` avec :
   - `evaluate(cfg) -> dict` retournant `{"config": cfg, "cost": float, "latency": float, "quality": float}` (deterministe).
   - `sweep() -> list[dict]` qui evalue **toutes** les combinaisons (batch_size dans une grille p.ex. `[1, 4, 16, 64]`, les 3 quant, cache on/off → 24 configs).
   - `pareto_front(evaluated) -> list[dict]` : garde les configs **non dominees** (aucune autre n'est a la fois ≤ en cout ET ≤ en latence avec au moins une inegalite stricte).
   - `optimize(slo_latency: float, min_quality: float) -> dict` : filtre les configs respectant `latency <= slo_latency` ET `quality >= min_quality`, puis retourne la **moins chere** parmi elles (ou leve/`None` documente si aucune).
4. Lance `optimize` avec un SLO realiste et **assert** :
   - La config choisie respecte bien `latency <= slo_latency` et `quality >= min_quality`.
   - Elle appartient a la frontiere de Pareto **des configs valides** (un plancher de qualite peut exclure des configs moins cheres mais moins bonnes qui la domineraient dans le sweep complet).
   - Aucune autre config **valide** n'a un cout strictement inferieur.
   - Resserrer le SLO (latence plus stricte) change la config retenue vers quelque chose de plus rapide (souvent plus quantize ou batch plus petit) — **assert** que la nouvelle config a une latence inferieure ou egale a l'ancienne.
5. Affiche la frontiere de Pareto et la config optimale retenue pour chaque SLO teste.

### Criteres de reussite

- [ ] `sweep()` evalue les 24 configurations (4 batch x 3 quant x 2 cache)
- [ ] `pareto_front` ne garde que des configs non dominees (verifie par assertion : aucune paire dominee n'y subsiste)
- [ ] `optimize` retourne la config **valide la moins chere** respectant SLO latence + plancher qualite
- [ ] La config optimale respecte effectivement le SLO et le plancher de qualite, et est non dominee parmi les configs valides (assertions)
- [ ] Resserrer le SLO selectionne une config a latence inferieure ou egale (compromis cout/latence/qualite explicite)
