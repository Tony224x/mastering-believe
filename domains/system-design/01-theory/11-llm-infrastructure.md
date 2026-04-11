# Jour 11 — LLM Infrastructure

## Pourquoi une couche d'infra dediee aux LLMs

**Exemple d'abord** : Tu as branche l'API OpenAI dans ton produit. Ca marche. Trois mois plus tard : la facture OpenAI explose ($15K/mois au lieu des $3K prevus), un user signale que le modele a hallucine un faux numero de telephone dans une reponse client, une maintenance OpenAI fait tomber 30% de ton trafic pendant 40 minutes, et ton CTO demande comment tu vas gerer si Anthropic sort un modele 2x moins cher demain.

Une couche d'infrastructure LLM repond a ces 4 problemes : **cout**, **qualite/securite**, **fiabilite**, **portabilite**. En 2025-2026, brancher directement une app sur un provider LLM sans cette couche n'est plus une option en production.

**Key takeaway** : L'API LLM n'est pas HTTP. C'est un backend cher, lent, non deterministe, probabiliste, multi-provider. Il merite son propre middleware.

---

## Architecture type d'une couche LLM infra

```
  client ──> App ──> LLM Gateway ──> Router ──> Provider 1 (OpenAI)
                       │                    ├─> Provider 2 (Anthropic)
                       │                    └─> Provider 3 (self-hosted)
                       │
                       ├─> Semantic cache
                       ├─> Prompt compression
                       ├─> Guardrails (input + output)
                       ├─> Observability / tracing
                       ├─> Rate limiter / quota
                       └─> Fallback chain
```

Chaque brique resout une dimension de probleme. On les ajoute au fur et a mesure que le produit grandit.

---

## Prompt routing : le bon modele pour la bonne tache

L'erreur du debutant : utiliser GPT-4o pour tout. La realite : 80% des taches peuvent etre servies par un modele 10x moins cher.

### Tiers de modeles

| Tier | Modeles (2026) | Cout relatif | Use case |
|---|---|---|---|
| **Nano** | gpt-5.4-nano, Haiku 4.5, Llama-3.1-8B | 1x | Classification, routing, simple extraction |
| **Mini** | gpt-5.4-mini, Haiku 4.5, Mistral Small | 3-5x | Q&A factuelle, resumes, transformations |
| **Standard** | gpt-5.4, Sonnet 4.6, Llama-3-70B | 15-25x | Reasoning complexe, code, RAG |
| **Frontier** | Opus 4.6, gpt-5.4 avec reasoning etendu | 50-100x | Problemes mathematiques, agents, analyse profonde |

### Strategie de routage

```python
def route(task_type: str, complexity: int) -> str:
    if task_type in {"classify", "extract", "route"}:
        return "haiku-4-5"
    if task_type in {"summarize", "rewrite", "translate"}:
        return "sonnet-4-6" if complexity > 5 else "haiku-4-5"
    if task_type in {"code", "reason", "plan"}:
        return "opus-4-6" if complexity > 7 else "sonnet-4-6"
    return "sonnet-4-6"  # safe default
```

### Techniques de routage

1. **Rule-based** : regex, classifieurs sur le type de tache (trivial, rapide)
2. **Classifier LLM nano** : un petit modele classifie la requete, puis dispatch
3. **Semantic routing** : embedder la query et comparer a des exemples par tier
4. **Learned router** : un modele entraine sur des paires (query, best_model)

**Regle** : commence par rule-based, ajoute du classifier quand ca ne suffit pas. Le learned router est overkill avant d'avoir 100K+ requetes.

---

## Semantic caching : reutiliser les reponses intelligemment

Un cache HTTP classique matche sur une cle exacte. Un **semantic cache** matche sur la **similarite semantique** de la requete.

### Flow

```
  query ──> embed query ──> vector_db.search(threshold=0.95) ──> HIT ──> return cached
                                                             └─> MISS ──> call LLM
                                                                           └─> store(embed, response)
```

### Parametres critiques

- **Threshold de similarite** : typiquement 0.92-0.97 en cosine. Trop bas = collisions (deux questions differentes retournent la meme reponse). Trop haut = cache quasi jamais hit.
- **TTL** : pour les reponses qui dependent du temps (news, prix), TTL court (minutes). Pour des definitions / Q&A stables, TTL long (jours).
- **Scope** : cache global vs par user. Un cache par user a 0% de hit, un cache global peut fuir des donnees privees. Solution : cache global pour les prompts generiques, pas de cache pour les prompts avec du PII.

### Taux de hit realistes

- Chatbot generaliste : 15-30%
- FAQ / support assistant : 40-60%
- Code completion : 5-10% (variations infinies)
- Translation : 70%+ (meme texte traduit plusieurs fois)

**Un cache a 30% de hit = 30% de cout en moins sur le compte OpenAI**.

### Outils

- GPTCache (open source)
- Redis avec modules vector
- Qdrant + custom logic
- Portkey, Helicone, LangSmith caches

---

## Guardrails : ne pas envoyer de la m**de a l'utilisateur

Un LLM est non deterministe. Sans garde-fous, il peut :
- Renvoyer des donnees personnelles (PII) qui lui ont ete fournies
- Inclure du contenu toxique
- Produire du JSON invalide et casser ta chaine de traitement
- Halluciner des faits
- Se faire hijack par prompt injection

### Guardrails d'entree (input)

Avant d'envoyer la requete au LLM :
- **PII detection** : regex + NER (Presidio) pour masquer emails, numeros de carte, SSN
- **Prompt injection** : detection de patterns "ignore all previous instructions"
- **Content moderation** : OpenAI moderation API, Perspective API, Llama Guard
- **Rate limiting par user** : evite l'abus

### Guardrails de sortie (output)

Apres la reponse du LLM :
- **JSON validation** : parse + valider contre un schema Pydantic/JSONSchema. Retry si KO.
- **PII scrubbing** : enlever toute donnee sensible qui aurait leak
- **Toxicity check** : bloquer si score > seuil
- **Groundedness check** (pour RAG) : verifier que la reponse est supportee par les docs
- **Format check** : longueur, langue, presence de citations, etc.

### Patterns d'implementation

**1. Fail-fast** : si un guardrail echoue, retourner une erreur claire
**2. Fallback** : si guardrail echoue, retourner une reponse par defaut ("Je ne peux pas repondre a cette question.")
**3. Retry with constraints** : re-demander au LLM avec un prompt plus strict ("Ta reponse precedente etait du JSON invalide. Reessaie.")
**4. Human handoff** : router vers un humain si uncertainty > seuil

### Outils

- **NeMo Guardrails** (NVIDIA) : framework complet
- **Guardrails.ai** : librairie de validateurs
- **Llama Guard** : modele dedie a la safety
- **Presidio** (Microsoft) : PII detection
- **Instructor / outlines** : JSON structuree via grammaire

---

## Cost optimization : l'autre obsession

### 1. Token counting et budgeting

Avant d'envoyer un prompt, compte les tokens (tiktoken) et applique des limites :
- Budget max par user (quota quotidien)
- Budget max par conversation
- Budget max par requete

#### Budget par session vs par user

**Budget par user** (naif) : quota quotidien par user id. Simple, mais ne capture pas la realite des conversations longues ou les tokens s'empilent. Un user peut consommer 200K tokens en 20 minutes sur une seule session agentique, puis rester calme 24h.

**Budget par session** (pattern 2026) : tracker la conversation **complete** (prompt tokens cumules + completion tokens cumules) et couper a un seuil. Chaque message genere un span de tracing avec un **delta de cost** qui s'additionne au running total. Quand le seuil est atteint :
- Soft cap : avertir l'utilisateur, proposer de resumer la conversation (context summarization) pour en reduire la taille
- Hard cap : forcer un nouveau thread

**Implementation** : chaque appel LLM emet un span OpenTelemetry avec `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.cost.usd`. Un middleware agrege par `session_id` et expose un compteur running. C'est la seule facon de ne pas se reveiller avec une facture de $8000 sur une conversation qui est partie en boucle.

### 2. Prompt compression

Les prompts trop longs coutent cher. Techniques :
- **Retirer les exemples few-shot** quand le modele est deja bon
- **Compresser avec un modele dedie** (LLMLingua reduit 5-10x sans perte majeure)
- **Supprimer les repetitions** (ex: mentionner la meme regle 3 fois)
- **Utiliser des abreviations** stables dans les prompts systeme

### 3. Caching de prompt (prompt caching natif)

Anthropic et OpenAI proposent du **prompt caching** natif : tu marques une portion stable du prompt, et les tokens sont factures 10% de leur prix normal aux requetes suivantes.

**Typical savings** : 50-80% sur les appels repetes avec un system prompt long. Standard pour les agents.

#### Cascade de caching : 3 niveaux a combiner

Le prompt caching n'est pas un simple flag on/off. En 2026, les equipes qui font tourner des agents a l'echelle empilent trois niveaux :

1. **Ephemeral cache** (Anthropic `cache_control: ephemeral`, TTL 5 min). Parfait pour des conversations courtes ou des salves de requetes rapprochees (ex: un user qui enchaine 10 questions en 2 min). Cout d'ecriture du cache 25% au-dessus du normal, cout de lecture 10% du normal. Rentable des la 2eme requete.

2. **Session cache** (attach au system prompt + outils, reutilise N turns). Pour les agents multi-turn ou le system prompt de 5000 tokens reste stable sur toute la session. Chaque turn paye 10% au lieu de 100%. Gain sur un agent 10-turn : ~80% de la facture system prompt.

3. **Cascading cache** (tool results ajoutes incrementalement). Pattern avance : a chaque turn, le resultat du tool precedent est ajoute au contexte, et **chaque increment beneficie du cache precedent**. Le prompt croit en sablier : tu ne paies plein tarif que sur les tokens vraiment nouveaux. C'est le pattern utilise par Claude Code et les agents Anthropic en production.

**Regle** : active le caching des que ton system prompt depasse 1024 tokens ET que tu fais > 2 appels avec ce meme prompt sur 5 min. En dessous, l'overhead d'ecriture annule le gain.

### 4. Fine-tuning pour remplacer les prompts longs

Si tu as un prompt de 3000 tokens qui contient des exemples, un fine-tune sur 500 paires peut remplacer ce prompt. Gain : prompt de 200 tokens au lieu de 3000.

### 5. Streaming

Sert les tokens des qu'ils arrivent -> perception de latence divisee par 3-5. Pas un vrai gain de cout, mais un gain massif de UX.

---

## Fallback chains : la fiabilite par la redondance

Si le provider primaire est KO, bascule automatiquement vers un secondaire.

```
  primary  : GPT-4o (OpenAI)
  fallback : Claude Sonnet 4.6 (Anthropic)
  last resort : Llama-70B self-hosted
```

### Implementation simple

```python
def call_with_fallback(messages, chain):
    for model in chain:
        try:
            return call_model(model, messages, timeout=10)
        except (APIError, TimeoutError) as e:
            log("fallback", model=model, error=e)
    raise LastResortFailed()
```

### Considerations

- **Prompts portables** : un prompt qui marche sur GPT-4 peut mal marcher sur Claude. A tester sur les 2.
- **Cout du fallback** : si le fallback est plus cher, attention a l'addition en cas de degradation du primaire.
- **Fail-fast timeouts** : ne pas attendre 60 s pour decider qu'un provider est KO. Timeout agressif (5-10 s).
- **Circuit breaker** : si un provider echoue N fois de suite, bypass-le pendant M minutes. Eviter de taper un serveur mort.

### Multi-provider par design

LLM Gateways commerciales : **Portkey, Helicone, LiteLLM, OpenRouter**. Ils abstraient les SDKs et offrent routing, fallback, observability en une ligne.

---

## Observability specifique LLM

Les metriques classiques (latence, error rate) ne suffisent pas. Il faut aussi :

- **Tokens in / out** par requete
- **Cost per request** (calcul a partir des tokens)
- **Model used** (quand il y a du routing)
- **Cache hit / miss**
- **Guardrail pass / fail**
- **Fallback triggered** (pour les alerts)
- **Prompt fingerprint** (hash du prompt) pour debugging
- **User feedback** (thumbs up/down)

#### Les plateformes d'observability LLM qui comptent en 2026

| Outil | Force | Quand le choisir |
|---|---|---|
| **Langfuse** | Open source, self-hostable, tracing fin, prompt management, evals integres | Startup, equipe produit, besoin de controler la data |
| **Helicone** | Proxy natif, setup en 2 lignes, cost tracking immediat | Equipes qui veulent zero instrumentation, cost-first |
| **Arize Phoenix** | Devenu standard prod 2025 en banking/healthcare, eval natif, anomaly detection, drift monitoring | Critical prod, compliance, ML + LLM unifie |
| **LangSmith** | Integration profonde LangChain/LangGraph, replay, datasets | Equipes deja sur l'ecosysteme LangChain |
| **Weights & Biases Traces** | Lien avec les runs de training, bon pour les equipes ML research | Equipes qui fine-tunent beaucoup |

**Arize vs Langfuse** : le choix le plus frequent en 2026. Arize est plus mature pour la critical prod (banking, healthcare) avec anomaly detection, drift sur les embeddings, eval comme feature native. Langfuse est plus leger, open source, et parfait pour les startups qui veulent self-host et controler leur stack sans payer un SaaS enterprise. LangSmith reste le defaut pour les equipes deja investies dans LangChain.

Tous ces outils exposent le meme pattern : un SDK qui wrap les appels LLM et emet des **spans** (`gen_ai.*` dans la semantique OpenTelemetry 2025), agregeables par `session_id`, `user_id`, `prompt_version`. Voir J13 pour le detail.

---

## Tradeoffs recapitulatifs

| Dimension | Approche simple | Approche robuste | Quand ca vaut le cout |
|---|---|---|---|
| Routing | 1 seul modele | Tier rule-based + classifier | Volume > 100K req/jour |
| Cache | Aucun | Semantic + TTL | Domaine stable, 20%+ de hit |
| Guardrails | Regex basique | NeMo + Llama Guard | Produit B2C, risque reputationnel |
| Fallback | Aucun | Primary + secondary + circuit breaker | SLA 99.9%+ |
| Observability | Logs simples | Langfuse / LangSmith | Des le jour 1 |

---

## Exemples reels

- **GitHub Copilot** : routing agressif (plusieurs tiers), caching lourd, telemetry detaillee
- **Notion AI** : prompt caching Anthropic, fallback chain, guardrails sur le contenu
- **Perplexity** : multi-provider routing, semantic cache, reranker
- **Portkey / Helicone** : LLM gateways managed qui implementent tout ca out-of-the-box

---

## Flash cards

**Q: Quelle est la difference entre un cache HTTP et un semantic cache ?**
R: HTTP cache matche sur cle exacte, semantic cache matche sur similarite d'embedding (threshold 0.92-0.97).

**Q: Pourquoi ne jamais utiliser GPT-4 pour tout ?**
R: Parce que 80% des taches sont servies par un modele 10x moins cher avec 95% de la qualite. Le routage par tier divise la facture par 5-10.

**Q: A quoi sert un guardrail de sortie JSON ?**
R: A valider que le LLM a bien respecte le format attendu. Sinon : parse error -> crash de la chaine de traitement. Retry with constraints ou fallback.

**Q: Qu'est-ce qu'un circuit breaker dans un fallback chain ?**
R: Un compteur d'echecs qui, apres N erreurs, bypass un provider pour M minutes pour eviter de taper un serveur mort.

**Q: Quel gain apporte le prompt caching d'Anthropic ?**
R: 50-80% sur les tokens repetes (system prompt long). Standard pour les agents avec contexte stable.

---

## Key takeaways

- Un LLM en prod sans middleware = dette technique immediate. Gateway obligatoire.
- Routing par tier = 5-10x de reduction de cout sans perte de qualite sur 80% des cas.
- Semantic cache = 20-30% de hit sur la plupart des produits. Gain direct.
- Guardrails sont un must en production : PII, toxicity, JSON validation, retry with constraints.
- Fallback chain + circuit breaker = pre-requis pour un SLA 99.9%+.
