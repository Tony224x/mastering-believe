# Jour 18 — Context engineering & prompt caching : l'art de l'economie 2026

> **Temps estime** : 4h | **Prerequis** : J11 (inference, KV cache), J16 (agents), J17 (RAG)

---

## 1. "Prompt engineering" est mort — vive "context engineering"

En 2023, le metier etait "prompt engineer" : savoir formuler une instruction. En 2026, c'est **context engineer** : savoir construire le **contexte complet** d'un appel LLM, en optimisant 4 variables simultanement :

1. **Qualite** : le modele a-t-il les bons elements pour decider ?
2. **Cout** : tokens input × $prix (et prix cache vs non-cache)
3. **Latence** : TTFT (time to first token) et inter-token latency
4. **Securite** : scope minimal, pas de leak entre utilisateurs

### La structure canonique d'un contexte 2026

```
┌─────────────────────────────────────────┐
│ 1. System prompt (stable, long)         │  ← CACHEABLE
│    - Role, ton, contraintes             │
│    - Regles metier                      │
│    - Format de sortie                   │
├─────────────────────────────────────────┤
│ 2. Tools / schemas                      │  ← CACHEABLE
│    - Definitions JSON schema            │
├─────────────────────────────────────────┤
│ 3. Contexte long partage                │  ← CACHEABLE
│    - Knowledge base, docs, examples     │
├─────────────────────────────────────────┤
│ 4. Contexte session / user              │  ← Cacheable par session
│    - Preferences, historique            │
├─────────────────────────────────────────┤
│ 5. Retrieved context (RAG)              │  ← Pas cacheable
│    - Chunks du turn courant             │
├─────────────────────────────────────────┤
│ 6. Conversation (messages)              │  ← Cacheable jusqu'au dernier
├─────────────────────────────────────────┤
│ 7. Derniere query utilisateur           │  ← Pas cacheable (change)
└─────────────────────────────────────────┘
```

**Principe cle** : ce qui est **stable et prefixe** est cache. Ce qui change a chaque tour doit etre **en fin** de contexte.

---

## 2. Prompt caching : l'optimisation qui change tout

### Comment ca marche

Quand un LLM traite un prompt, il construit une representation interne (les K et V du KV cache) pour chaque token. Cette computation scale lineairement avec la longueur. Si deux appels partagent un **prefixe identique**, on peut **reutiliser les K/V** du prefixe → pas de re-computation.

### Prix et disponibilite en 2026

| Provider | Latence cache hit | Prix input cache read | Prix input frais | TTL |
|---|---|---|---|---|
| Anthropic (Claude) | -85% TTFT | 10% du prix normal | 100% | 5 min defaut / 1h ext |
| OpenAI | -50% TTFT | 50% du prix normal | 100% | 5-10 min auto |
| Google (Gemini) | -75% TTFT | 25% du prix normal | 100% | configurable |
| DeepSeek / Together | -50% TTFT | ~25% | 100% | variable |

**L'impact pratique** : pour un RAG avec 50k tokens de contexte reutilise, passer de pas-de-cache a cache hit divise la facture par 3-5 et la latence par 3-7.

### Les trois strategies de cache

1. **Automatique** (OpenAI, Gemini) : le provider detecte les prefixes repetes. Simple, marche mieux a volume.
2. **Explicite** (Anthropic) : tu marques `cache_control: {type: "ephemeral"}` sur le dernier bloc que tu veux cacher. Controle fin, TTL plus long (1h).
3. **Prefix deliberate** (best practice 2026) : tu structures tes prompts pour que les prefixes soient identiques autant que possible.

### Les 5 regles pour maximiser le cache hit rate

1. **Ordre stable** : system → tools → docs → user-specific → query. Jamais de variation dans les 3 premiers blocs.
2. **Normalisation** : whitespace, JSON formatting, ordre des cles : tout doit etre deterministe.
3. **Modifications en append only** : on ajoute a la fin d'un turn, on ne reecrit pas les precedents.
4. **Eviter les timestamps dans le prefixe** : si tu mets `Current time: {now}` en header, tu casses le cache a chaque seconde. Mets-le dans la section query.
5. **Segmenter par utilisateur** : deux users dans une meme session partagent le prefixe system+docs → cache explicite reutilisable.

---

## 3. Les pathologies du long contexte

### Lost in the middle (Liu et al., 2023, toujours d'actualite)

```
Position de l'info pertinente dans le contexte :
  debut   : 85% de retrieval reussi
  25%     : 65%
  milieu  : 55%  ← le creux
  75%     : 70%
  fin     : 90%
```

Meme avec 1M tokens de context window, l'info au milieu est moins utilisee. **Implication** : placer les elements cles soit au debut (system prompt) soit a la fin (pres de la query).

### Context rot (Claude Code/Anthropic, 2025)

Au-dela de ~50 tool calls (dans un agent) ou de contextes > 200k tokens, les modeles :
- Repetent des actions deja faites
- Oublient l'objectif initial
- "Se deconcentrent" sur des details

Solutions :
- **Compaction** : resumer l'historique tous les N tours (API native chez Anthropic, manual ailleurs)
- **Sub-agents** : deleguer a un agent fresh avec un contexte minimal
- **Scratchpad externe** : memoire persistante hors-prompt

### Needle-in-a-haystack vs effective context

Tous les modeles 2025-2026 passent "needle-in-a-haystack" a 1M tokens (retrouver une phrase cachee). Mais **les benchmarks realistes** (ex: RULER, NoLiMa) montrent que l'**effective context** est beaucoup plus court : souvent 20-50% du context window marketing. Gemini 2.5 Pro 1M window ≈ 200k tokens effectifs. A prendre en compte pour le design.

---

## 4. Context engineering patterns pour agents

### Pattern 1 — Contexte hierarchique

```
Permanent : (en system prompt, cache long)
  - Role, outils, regles generales

Session : (cache court, par user)
  - Profil, permissions, prefs

Turn : (non cache)
  - Derniere query, retrieval frais
```

### Pattern 2 — Compaction prompt

Quand la conversation devient longue, appeler un LLM dedie pour la compresser :

```
Input  : [message_1, ..., message_N]
Prompt : "Resume cette conversation en preservant :
          - Les decisions prises
          - Les resultats des tool calls
          - Les contraintes etablies par l'utilisateur
          - Le plan actuel"
Output : resume condense qui remplace messages 1..N-5
```

Anthropic expose ca en API direct (parametre `compaction: {enabled: true}` dans certaines integrations).

### Pattern 3 — Sliding window + memoire long terme

```
Conversation :
  [derniers 30 tours en clair]
  +
  [resume des tours 1..N-30]
  +
  [faits epingles : preferences user, decisions, contraintes]
```

### Pattern 4 — Structured state externe

Au lieu de passer tout l'historique au modele, maintenir un **state JSON** :
```json
{
  "goal": "refactor the auth module",
  "plan": ["read tests", "identify patterns", "propose changes"],
  "completed": ["read tests"],
  "findings": {"uses_jwt": true, "token_ttl": 3600}
}
```
Inject ce JSON (petit, structure) au lieu du transcript complet.

---

## 5. Economie des tokens — calcul concret

### Exemple : chatbot support client

- 1000 utilisateurs/h, 5 tours moyens par conversation
- Contexte docs FAQ : 20k tokens
- Historique moyen : 2k tokens
- Query moyenne : 100 tokens
- Output moyen : 500 tokens

**Sans cache** :
- Input = 22k tokens × 5 tours = 110k tokens/conversation
- Output = 500 × 5 = 2.5k tokens/conversation
- Total par heure = 1000 × (110k × $3/M + 2.5k × $15/M) = 1000 × ($0.33 + $0.037)
- **~$370/h input + $37/h output = $407/h**

**Avec cache (FAQ cacheable, 5x cache hits)** :
- Premier turn : 22k input normal
- Turns 2-5 : 20k input cache (10% prix) + 2k nouveau
- Input facture = 22k + 4×(20k×0.1 + 2k) = 22k + 16k = 38k/conv
- Total par heure = 1000 × (38k × $3/M + 2.5k × $15/M) = $114/h + $37/h
- **~$151/h, economie 63%**

Ce calcul se fait en 5 minutes. Le faire avant de choisir architecture et provider economise 6 chiffres par an sur des volumes moyens.

---

## 6. Context caching et RAG : la combinaison gagnante

Le RAG injecte des chunks differents a chaque query → non cacheable en theorie. Mais :

### Strategie hybride

```
Contexte = [
  system_prompt,          # cache permanent
  toolkit_defs,           # cache permanent
  top_100_hot_docs,       # cache long (docs les plus frequents)
  user_session_context,   # cache session
  ---- cache breakpoint ----
  retrieved_chunks,       # varie par query, non cache
  user_query
]
```

Au lieu de retrouver 10 chunks pour chaque query, on peut :
- Injecter les 100 chunks les plus accedes en cache permanent (~50k tokens)
- Ne faire retrieval que pour les 5 chunks specifiques a la query (5k tokens)
- Total : meilleure latence, meilleur hit rate, meme qualite

### Prompt caching + long context = tuer RAG dans certains cas

Si ta KB fait 200k tokens (pas 200GB), avec prompt caching tu peux :
- Injecter TOUTE la KB dans le system prompt
- Payer 90% de reduction sur les cache hits
- Zero retrieval failure (tout est la)
- Latence = celle du dense ret + LLM, mais sans RAG pipeline

C'est economiquement viable si : KB < 500k tokens, volume > 1000 queries/h (pour amortir les first-miss).

---

## 7. Outils et methodologie

### Mesurer ton cache hit rate

Toujours logger par appel :
- `cache_creation_input_tokens` (premier hit, plein prix)
- `cache_read_input_tokens` (hit cache, 10% prix)
- `input_tokens` (non cache)

Tableau de bord minimum : cache_read / (cache_read + input_tokens + cache_creation) par endpoint. Cible : > 70% pour les produits batch, > 50% pour chat.

### A/B tester sur le prix reel

Deux architectures egales en qualite peuvent differer de 5x en facture. A/B test ton pipeline avec un traffic echantillon sur :
- Differents providers
- Differents placements de cache_control
- Differents TTLs (5 min vs 1h)

### Anti-patterns communs

- Mettre `current_date` dans le system prompt (casse le cache)
- Ordre randomise des tools (casse le cache)
- User ID numerique au debut (segmentation excessive du cache, faibles hit rates)
- Multiple system prompts par type de query sans cache_control (jamais hit)

---

## Key takeaways (flashcards)

**Q1** — Pourquoi le terme "context engineering" a-t-il remplace "prompt engineering" ?
> Parce que la qualite finale depend de la construction globale du contexte (system, tools, docs, historique, retrieval, query), pas juste de la formulation d'une instruction. Les optimisations se jouent sur 4 axes : qualite/cout/latence/securite.

**Q2** — Quelle est la regle de placement pour maximiser le cache hit rate ?
> Stable et long en premier (cacheable), variable et court a la fin (non cacheable). Jamais de timestamp ou random dans le prefixe.

**Q3** — Quelle est la difference typique de prix entre cache read et input frais chez Anthropic ?
> Cache read = 10% du prix input normal. Cache write (premier hit, plus court TTL) = ~125% pour les TTL longs. Avec hit rate 70%+, economie 50-70% sur la facture input.

**Q4** — Qu'est-ce que "effective context" ?
> Le context window reellement utilisable sans degradation de qualite. Generalement 20-50% du context window marketing. Benchmarks : RULER, NoLiMa, LongBench.

**Q5** — Quand prompt caching + long context peut-il remplacer RAG ?
> Quand la KB tient dans le context window (< 500k tokens) et le volume est assez eleve pour amortir le first-miss. Zero retrieval failure, latence reduite, simplicite maintenance.

**Q6** — Quelle metrique minimum dois-tu monitorer en prod ?
> cache_read_tokens / total_input_tokens par endpoint. Cible 50-70%. En dessous, chercher les causes (timestamps, reordering, multiple prompts).
