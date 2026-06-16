# Exercices Medium — Production & Observabilite (J12)

---

## Exercice 1 : Arbre de spans + percentiles de latence (p50/p95/p99)

### Objectif
Passer d'une liste plate de spans a un **arbre** (parent/enfant) et calculer les metriques de latence qui comptent en prod : pas la moyenne, mais p50/p95/p99 (section 4.1 du cours).

### Consigne
En partant de `02-code/12-production-observabilite.py` (les spans ont deja `parent_span_id`) :

1. Ecris `build_span_tree(spans, trace_id) -> dict` qui reconstruit l'arbre d'un trace : chaque noeud = `{span, children: [...]}`, racines = spans sans parent
2. Ecris `print_tree(tree, indent=0)` qui affiche l'arbre avec indentation et la duree de chaque span (visualisation type waterfall)
3. **Self-time vs total-time** : pour un span parent, distingue `total_ms` (sa duree mesuree) et `self_ms` (total moins la somme des durees de ses enfants). Un parent lent dont les enfants sont rapides = le temps est passe DANS le parent (pas en attente d'un sous-appel)
4. Ecris `latency_percentiles(durations: list[float]) -> dict` qui retourne p50, p95, p99 (interpolation lineaire, pas de numpy)
5. Genere ~30 spans `llm_step` avec des durees variees (dont quelques outliers lents) et affiche les percentiles ; verifie que p99 >> p50 quand il y a des outliers

### Criteres de reussite
- [ ] `build_span_tree` reconstruit correctement la hierarchie parent/enfant
- [ ] `print_tree` affiche un waterfall indente lisible
- [ ] `self_ms` est calcule correctement (total - somme des enfants)
- [ ] `latency_percentiles` retourne p50/p95/p99 corrects (verifie sur un cas connu)
- [ ] Sur des donnees avec outliers, p99 est nettement superieur a p50

---

## Exercice 2 : Modele de cout avec prompt caching (3 niveaux)

### Objectif
Modeliser l'economie du **prompt caching** (section 3.4 du cours) : ecriture du cache plus chere, lecture beaucoup moins chere, et calculer a partir de combien de hits le cache devient rentable.

### Consigne
1. Cree une classe `CachedCostModel` avec, par modele, 4 prix au 1K tokens : `input`, `output`, `cache_write` (~1.25x input), `cache_read` (~0.1x input)
2. Methode `cost_uncached(model, tokens_in, tokens_out)` : cout sans cache
3. Methode `cost_cached(model, cached_tokens, fresh_in, tokens_out, is_first_call)` :
   - 1er appel : `cached_tokens` factures au prix `cache_write` + `fresh_in` au prix input + output
   - appels suivants : `cached_tokens` au prix `cache_read` (90% moins cher) + `fresh_in` au prix input + output
4. Methode `break_even_calls(model, cached_tokens) -> int` : a partir de combien d'appels (en reutilisant le meme prefixe cache) le total cached devient < total uncached
5. Simule un agent avec un gros system prompt (3000 tokens caches) appele 10 fois : compare le cout cumule avec/sans cache, et affiche le point de rentabilite
6. **Edge case** : un prefixe trop petit (ex: 200 tokens) ne sera peut-etre jamais rentable a cause du surcout d'ecriture — montre-le

### Criteres de reussite
- [ ] Les 4 niveaux de prix sont modelises (input, output, cache_write, cache_read)
- [ ] Le 1er appel est plus cher avec cache (cache_write), les suivants beaucoup moins
- [ ] `break_even_calls` retourne le nombre d'appels correct pour la rentabilite
- [ ] La simulation sur 10 appels montre une economie cumulee avec cache
- [ ] Le cas du petit prefixe non-rentable est identifie

---

## Exercice 3 : Fallback multi-niveaux avec graceful degradation et tracing par tentative

### Objectif
Etendre la `FallbackChain` du code en une chaine a N niveaux qui degrade gracieusement la qualite (gros modele → petit modele → cache → reponse statique) en tracant CHAQUE tentative comme un span enfant, et en exposant le niveau de degradation au caller (section 5.2 et 5.3).

### Consigne
1. Cree une `DegradingChain` configuree par une liste ordonnee de `Tier(name, callable, quality_level)` ou `quality_level` decroit (ex: 1.0 → 0.7 → 0.4 → 0.1)
2. A l'appel : tente chaque tier dans l'ordre ; si un tier leve `TransientError`, retry 2x (backoff court) puis passe au tier suivant
3. Chaque tentative produit un span enfant (reutilise `@traced` ou un enregistrement manuel) avec le nom du tier et son resultat (succes/echec)
4. Le retour expose `{answer, served_by, quality_level, attempts}` — le caller sait s'il a recu une reponse premium ou degradee (utile pour afficher un bandeau "service degrade")
5. **Cache layer** : un des tiers est un cache en memoire `{prompt: answer}` ; s'il a deja vu le prompt, il repond instantanement
6. Teste 3 scenarios : tout marche (tier 1), tier 1+2 down (sert tier 3 cache ou statique), tout down sauf le statique final

### Criteres de reussite
- [ ] La chaine tente les tiers dans l'ordre de qualite decroissante
- [ ] Chaque tier echoue avec retry avant de passer au suivant
- [ ] Chaque tentative est tracee comme un span enfant
- [ ] Le retour indique `served_by` et `quality_level` (le caller sait s'il y a degradation)
- [ ] Le tier cache repond instantanement sur un prompt deja vu
- [ ] Les 3 scenarios produisent le bon niveau de degradation
