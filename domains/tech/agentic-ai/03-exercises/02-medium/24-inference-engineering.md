# Exercices Medium — Inference engineering (J24)

> **Prerequis** : avoir lu `01-theory/24-inference-engineering.md` et execute `02-code/24-inference-engineering.py`.
> **Note** : tout est **simule de maniere deterministe** — on raisonne en *tokens / steps*, jamais en `time.sleep` ni en horloge murale. Aucun appel reseau, aucun vrai modele. Le focus est la machinerie d'inference (prefill/decode, batching, SLO), pas un LLM.

---

## Exercice 1 : Simulateur de continuous batching vs static batching

### Objectif

Le cours montre le routing et le caching. Ici on descend d'un cran cote serving : modeliser un moteur d'inference qui execute des requetes en **prefill** (cout proportionnel a la taille du prompt) puis en **decode** (1 token par step), et comparer le **static batching** (on attend que tout le batch soit fini avant d'en lancer un nouveau) au **continuous batching** (des qu'une requete termine, on injecte la suivante dans le slot libre). Objectif : prouver que le continuous batching augmente le **throughput** (tokens/step) et reduit la **latence moyenne** sur des requetes de tailles heterogenes.

### Consigne

En t'inspirant du style deterministe de `02-code/24-inference-engineering.py` :

1. Cree une dataclass `Request(rid: int, prompt_tokens: int, output_tokens: int)`.
2. Cree une classe `StaticBatchEngine(batch_size: int)` avec une methode `run(requests: list[Request]) -> dict` qui simule, **par step entier** :
   - Decoupe les requetes en vagues (batchs) de `batch_size`.
   - Pour chaque vague : 1 step de prefill (toutes les requetes du batch font leur prefill ensemble), puis autant de steps de decode que le **max** des `output_tokens` du batch (les requetes courtes restent dans le slot a ne rien faire — c'est le gaspillage du static batching).
   - Comptabilise `total_steps`, et pour chaque requete son `finish_step` (le step ou son dernier token de sortie est produit).
3. Cree une classe `ContinuousBatchEngine(batch_size: int)` avec la meme signature `run`, mais :
   - On garde en permanence jusqu'a `batch_size` requetes actives.
   - A chaque step, chaque requete active produit 1 token ; une requete qui a fini libere son slot **immediatement** et la prochaine requete en attente y entre (et fait son prefill).
   - Comptabilise `total_steps` et les `finish_step` de la meme maniere.
4. Construis un jeu de 12 requetes de tailles tres heterogenes (melange de courtes et de longues, deterministe via `random.Random(seed)`), passe-le aux deux moteurs, et calcule pour chacun : `total_steps`, `throughput = total_output_tokens / total_steps`, `avg_latency = mean(finish_step)`.
5. Affiche un petit tableau comparatif et **assert** que le continuous batching a un `throughput` strictement superieur et une `avg_latency` strictement inferieure au static batching.

### Criteres de reussite

- [ ] `StaticBatchEngine` paie bien le max des `output_tokens` par vague (les slots courts sont gaspilles)
- [ ] `ContinuousBatchEngine` reinjecte une requete en attente des qu'un slot se libere
- [ ] Le throughput (tokens/step) du continuous batching est strictement superieur
- [ ] La latence moyenne (`avg_latency`) du continuous batching est strictement inferieure
- [ ] Les deux moteurs produisent le meme nombre total de tokens de sortie (rien n'est perdu)

---

## Exercice 2 : KV / prefix cache et economie de prefill (TTFT)

### Objectif

Approfondir le `PromptCache` du cours (section 3.2) en modelisant explicitement la separation **prefill / decode** et le **time-to-first-token (TTFT)**. Le prefill recalcule le KV cache de tout le prefixe ; un *prefix cache hit* permet de **sauter** le recalcul du prefixe partage et donc d'effondrer le TTFT. On prouve l'economie en *steps de calcul* et en tokens factures.

### Consigne

1. Cree une classe `PrefillCacheEngine(read_discount: float = 0.10)` qui modelise un cache de prefixe (hash du prefixe statique, comme dans `PromptCache`).
2. Methode `serve(prefix: str, prefix_tokens: int, suffix_tokens: int, output_tokens: int) -> dict` qui retourne :
   - `prefill_steps` : sur un **miss**, le prefill coute `prefix_tokens + suffix_tokens` (tout est recalcule) ; sur un **hit**, il ne coute que `suffix_tokens` (le KV du prefixe est reutilise tel quel).
   - `ttft` : on modelise le TTFT comme `prefill_steps + 1` (premier token de decode juste apres le prefill).
   - `billed_input_tokens` : sur un hit, les `prefix_tokens` sont factures a `read_discount` (ex. 10 %) + `suffix_tokens` plein tarif ; sur un miss, tout plein tarif.
   - `hit` : booleen.
   - Met a jour des compteurs internes (`self.hits`, `self.misses`).
3. Simule un agent avec un long prefixe systeme fixe de 8000 tokens : 1 appel a froid (miss) suivi de 49 appels a chaud (meme prefixe). Chaque appel : `suffix_tokens=150`, `output_tokens=60`.
4. Calcule et compare a un baseline **sans cache** (chaque appel recalcule tout le prefixe) : le `ttft` moyen et le total des `billed_input_tokens`.
5. **Assert** que : (a) le TTFT d'un hit est strictement inferieur au TTFT du miss froid, (b) le total des tokens d'input factures avec cache est au moins 5x inferieur au baseline sans cache, (c) `self.hits == 49`.

### Criteres de reussite

- [ ] Un hit saute le recalcul du prefixe : `prefill_steps` d'un hit = `suffix_tokens` seulement
- [ ] Le TTFT d'un hit est nettement inferieur a celui du miss froid (prefill complet)
- [ ] Les `prefix_tokens` d'un hit sont factures a `read_discount`
- [ ] L'economie totale de tokens d'input vs baseline sans cache est >= 5x
- [ ] Les compteurs `hits` / `misses` sont corrects (1 miss, 49 hits)

---

## Exercice 3 : Routeur sous contrainte de SLO de latence (TTFT/TPOT)

### Objectif

Combiner routing et serving : chaque profil de modele a une **vitesse** (TPOT = time-per-output-token, en steps/token) et une **qualite**. Etant donne un budget de latence (SLO) exprime en steps, le routeur doit choisir le modele le **moins cher** qui **tient le SLO** pour une requete donnee, et tomber sur une strategie de repli (la plus rapide possible) si aucun ne tient — exactement le genre d'arbitrage cout/latence/qualite de la table du cours (section 2.4).

### Consigne

En reutilisant l'idee de `ModelSpec` du cours (mais enrichie) :

1. Cree une dataclass `ServingModel(name, tier, cost_in, cost_out, ttft_steps: int, tpot_steps: float, quality: float)`.
2. Cree une classe `SLORouter(models: list[ServingModel])`.
3. Methode `estimate_latency(model, output_tokens) -> float` = `model.ttft_steps + model.tpot_steps * output_tokens`.
4. Methode `route(output_tokens: int, slo_steps: float, min_quality: float = 0.0) -> dict` :
   - Considere les modeles dont `quality >= min_quality` ET dont `estimate_latency(...) <= slo_steps`.
   - Parmi les eligibles, choisis celui de **cout** le plus faible (cout = `cost_in + cost_out`, ordre de grandeur suffisant ici).
   - Si **aucun** modele n'est eligible, choisis le modele de **latence estimee minimale** (best-effort) et marque `slo_met = False`.
   - Retourne `{"model": <name>, "latency": <float>, "slo_met": <bool>, "cost": <float>, "reason": <str>}`.
5. Teste 3 scenarios :
   - SLO genereux + petit `output_tokens` → un modele cheap/lent tient le SLO → on prend le moins cher (`slo_met == True`).
   - SLO serre + gros `output_tokens` → seul un modele rapide (mais cher) tient → on l'a choisi et `slo_met == True`.
   - SLO impossible (tres serre) → aucun ne tient → repli sur le plus rapide avec `slo_met == False`.

### Criteres de reussite

- [ ] `estimate_latency` = `ttft_steps + tpot_steps * output_tokens`
- [ ] Quand plusieurs modeles tiennent le SLO, le **moins cher** est choisi
- [ ] Un SLO serre force le choix d'un modele plus rapide (et plus cher)
- [ ] Un SLO impossible bascule en best-effort (`slo_met == False`) sur le modele le plus rapide
- [ ] Le filtre `min_quality` exclut bien les modeles trop faibles
