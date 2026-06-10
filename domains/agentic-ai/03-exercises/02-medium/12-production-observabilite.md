# Exercices Medium — Production & Observabilite (J12)

---

## Exercice 1 : Percentiles de latence et alerting SLO

### Objectif
Passer des moyennes (trompeuses) aux percentiles, et implementer une alerte SLO avec fenetre glissante et cooldown — le quotidien du monitoring d'agents.

### Consigne
En partant du `Tracer` de `02-code/12-production-observabilite.py` :

1. Ecris `percentile(values: list[float], p: float) -> float` a la main (tri + interpolation lineaire entre rangs, pas de numpy) et verifie sur des cas connus : `percentile([1..100], 50) == 50.5`, `percentile([10], 99) == 10`
2. Cree un `LatencyMonitor` :
   - `record(operation: str, duration_ms: float, ts: float)` — alimente par les spans du tracer
   - `stats(operation) -> dict` : count, p50, p95, p99, max
   - Demontre sur une distribution simulee deterministe : 95 requetes entre 200-400ms + 5 requetes a 3000-9000ms -> la moyenne semble OK (~600ms) mais le p99 explose (affiche le contraste)
3. Implemente le SLO : `SLOChecker(monitor, slo={"p95_ms": 1000, "error_rate": 0.02}, window_size=50)` :
   - `check(operation) -> list[Alert]` evalue le SLO sur les `window_size` DERNIERES mesures (fenetre glissante, pas l'historique complet)
   - Une `Alert` contient : metric, valeur observee, seuil, fenetre
4. Ajoute un **cooldown** : une fois une alerte emise pour (operation, metric), pas de re-alerte pendant 60 "secondes" simulees (les timestamps sont passes a la main, pas de vraie horloge) — mais une alerte sur une AUTRE metric reste possible
5. Scenario de demo : flux de 120 mesures ou la latence se degrade progressivement -> la premiere alerte part au bon moment, les suivantes sont supprimees par le cooldown, puis une nouvelle alerte part apres expiration du cooldown

### Criteres de reussite
- [ ] `percentile` est correcte sur les cas de verification (asserts)
- [ ] Le contraste moyenne vs p99 est demontre chiffres a l'appui
- [ ] Le SLO est evalue sur la fenetre glissante uniquement
- [ ] Le cooldown supprime les alertes repetees et expire correctement
- [ ] Le scenario complet affiche les alertes aux bons timestamps simules

---

## Exercice 2 : Cache semantique d'appels LLM

### Objectif
Reduire les couts en evitant les appels LLM redondants : cache exact-match d'abord, puis matching approximatif par similarite de tokens, avec TTL et comptabilite des economies.

### Consigne
1. Cree un `SemanticCache` :
   - `normalize(prompt) -> str` : minuscules, espaces compactes, ponctuation finale retiree
   - `get(prompt) -> CacheHit | None` et `put(prompt, response, cost_usd)`
   - **Niveau 1 (exact)** : lookup par hash du prompt normalise
   - **Niveau 2 (approximatif)** : similarite de Jaccard sur les ensembles de tokens (`|A∩B| / |A∪B|`) ; hit si >= 0.8 avec une entree existante — retourne aussi le score
2. Chaque entree a un `ttl_seconds` (horloge simulee injectable `now_fn`) : une entree expiree est ignoree ET purgee au passage
3. `CacheHit` contient : `response`, `level` ("exact"/"semantic"), `similarity`, `saved_cost_usd`
4. Cree `cached_llm_call(cache, llm_fn, prompt) -> tuple[str, dict]` qui :
   - Verifie le cache ; en cas de miss, appelle `llm_fn` (mock qui retourne une reponse + un cout), stocke et retourne
   - Maintient les stats globales : hits exact, hits semantic, misses, `total_saved_usd`, hit_rate
5. Demo deterministe avec 8 appels :
   - "What is the revenue of Acme?" (miss) ; la meme question exacte (hit exact) ; "what is the revenue of acme ?" (hit exact apres normalisation) ; "What is the annual revenue of Acme?" (hit semantic ~0.83) ; une question differente (miss) ; la premiere question apres expiration du TTL (miss + purge) ; etc.
6. Affiche le rapport final : stats, economies, contenu du cache
7. Discussion en commentaire : pourquoi le seuil 0.8 est dangereux pour des questions qui different d'un seul mot critique ("revenue 2023" vs "revenue 2024") — ajoute un garde : jamais de hit semantic si les ensembles de **nombres** different

### Criteres de reussite
- [ ] Les 2 niveaux de cache fonctionnent et sont distingues dans les stats
- [ ] La normalisation rattrape les variations triviales
- [ ] Le TTL expire les entrees avec l'horloge simulee (pas de sleep)
- [ ] Le garde sur les nombres bloque le faux-hit "2023 vs 2024" (teste)
- [ ] Les economies cumulees et le hit rate sont exacts (asserts)

---

## Exercice 3 : Degradation gracieuse — machine a etats de modes de service

### Objectif
Implementer la strategie "degrade, don't die" : un agent qui change de mode de fonctionnement selon la sante de ses dependances et son budget, au lieu de crasher.

### Consigne
1. Definis 4 modes (enum) : `FULL` (modele premium + outils), `ECO` (modele cheap + outils), `DEGRADED` (sans outils, reponses templatees avec avertissement), `OFFLINE` (message d'indisponibilite + mise en queue)
2. Cree un `HealthRegistry` : etat de chaque dependance (`llm_premium`, `llm_cheap`, `search_tool`) avec `healthy: bool` — modifiable par la demo pour simuler des pannes
3. Cree un `ModeController(health, budget_tracker)` avec `current_mode() -> Mode` selon des regles explicites :
   - budget restant < 10% -> au mieux ECO
   - `llm_premium` down -> au mieux ECO ; `llm_cheap` AUSSI down -> DEGRADED
   - `search_tool` down -> au mieux DEGRADED si la requete necessite des outils, sinon ECO/FULL
   - tout down -> OFFLINE
4. Cree `ResilientAgent.run(query) -> dict` qui :
   - Determine le mode AVANT l'appel, puis route vers le bon handler (4 handlers mocks)
   - Ajoute systematiquement dans la reponse : `mode`, et en mode DEGRADED/OFFLINE un bandeau d'avertissement honnete ("Reduced service: live data unavailable...")
   - En OFFLINE : empile la requete dans `queue` pour rejeu, et la rejoue automatiquement quand un mode >= DEGRADED redevient disponible
5. Scenario de demo (sequence deterministe) : 2 requetes en FULL -> panne du premium (ECO) -> budget epuise a 95% (reste ECO) -> panne du cheap (DEGRADED) -> panne du search (OFFLINE, mise en queue) -> retablissement du cheap (rejeu de la queue en DEGRADED)
6. Journalise chaque transition de mode avec sa cause

### Criteres de reussite
- [ ] Les regles de transition sont implementees exactement et testees une a une
- [ ] Chaque reponse declare son mode et le bandeau apparait en mode degrade
- [ ] La queue OFFLINE se rejoue automatiquement au retablissement
- [ ] Le journal des transitions liste mode_avant -> mode_apres + cause
- [ ] Aucune exception non geree sur tout le scenario de pannes
