# Jour 14 — Capstone : 2 designs en 45 minutes chacun

## Objectif de la journee

C'est le jour ou tout ce que tu as appris depuis J1 doit se materialiser en designs complets, defendables en entretien. L'enjeu n'est pas de connaitre des technos, c'est de **raisonner structure** : clarifier -> estimer -> high level -> deep dive -> bottlenecks -> extensions.

**Key takeaway** : En entretien, un candidat qui dessine 3 boxes justifiees battra toujours un candidat qui recite 30 technos. La structure bat le vocabulaire.

---

## Framework de resolution (45 min, reutilise)

1. **Clarifier** (3 min) : fonctional + non-functional requirements
2. **Estimer** (5 min) : users, QPS, storage, bandwidth
3. **High-level design** (10 min) : 4-7 boxes, flow principal
4. **Deep dive** (15 min) : 2-3 composants critiques
5. **Bottlenecks & scale** (7 min) : ou ca casse, comment on scale
6. **Extensions** (5 min) : ce qu'on ajouterait si on avait 3 mois de plus

---

# DESIGN 1 — Dropbox (Distributed File Storage)

## 1. Clarifier (3 min)

**Fonctionnel** :
- Upload / download fichiers
- Sync multi-device
- Versioning
- Sharing (link, permissions)
- Search par nom / contenu

**Non-fonctionnel** :
- 500M users, 50M actifs/jour
- 95% des fichiers < 10 Mo, p99 < 500 Mo
- Latence upload : < 1 s pour 1 Mo
- Fiabilite : durabilite 99.999999999% (11 nines)
- Disponibilite : 99.9% (SLA)

**Questions a poser** :
- Max file size ? (250 Go standard, ok)
- Type d'utilisateurs : individuel ou entreprise ? (mix, on focus indiv)
- Conflits d'ecriture ? (last-write-wins ou CRDT ? -> last-write-wins)
- Encryption at rest ? (oui)

## 2. Estimer (5 min)

```
  Users actifs / jour : 50 M
  Uploads / user / jour : 2        -> 100M uploads / jour ~ 1160 uploads/s
  Downloads / user / jour : 10     -> 500M DL / jour ~ 5800 DL/s
  Taille moyenne fichier : 2 Mo
  Bandwidth upload : 1160 * 2 Mo = 2.3 Go/s ~ 18 Gbps
  Bandwidth download : 5800 * 2 Mo = 11.6 Go/s ~ 93 Gbps
  Storage net / jour : 100M * 2 Mo = 200 To / jour
  Storage net / an : 73 Po
  Avec replication x3 : 219 Po / an
  Metadata : 500 bytes par file * 100M files/jour * 365 = 18 To/an
```

Conclusion : le gros morceau c'est la bandwidth download (CDN obligatoire) et le storage (blob store specialise). Metadata est trivialement manageable.

## 3. High-level design

```
  client ──> LB ──> API gateway ──> Metadata service ──> Metadata DB
                          │                                 (postgres)
                          │
                          ├──> Auth service ──> Users DB
                          │
                          ├──> Upload service ──> Block storage (S3-like)
                          │                         │
                          │                         └──> CDN (downloads)
                          │
                          └──> Notification service ──> Other devices (WebSocket)
```

**Idee cle : chunking**. Chaque fichier est decoupe en blocs de 4 Mo avec deduplication (hash SHA-256). Si le user upload le meme PDF que son collegue, un seul bloc est stocke. Gain storage : 30-40%.

## 4. Deep dive

### 4.a Chunking + Deduplication

- Client calcule le hash SHA-256 de chaque bloc de 4 Mo
- Upload envoie d'abord les hashes : le server repond "ceux-ci existent deja, ceux-la il manque" (pattern "rolling hash check")
- Economie de bandwidth massive (resync rapide)
- Deduplication globale : meme bloc stocke 1 fois dans le block store

Avantages :
- Bandwidth economise (sync partiel)
- Storage economise (dedup)
- Reprise apres interruption (bloc par bloc)

### 4.b Metadata store

Structure (tables principales) :

```
users(id, email, plan, created_at)
files(id, user_id, name, parent_folder_id, size, mime, created_at, latest_version_id)
file_versions(id, file_id, size, hash, created_at, created_by)
file_blocks(file_version_id, block_index, block_hash)   -- ordered blocks
blocks(hash, size, storage_location, ref_count)          -- dedup-able
share_links(id, file_id, permission, expires_at)
```

- Postgres avec partitioning par user_id (hash partitioning)
- Index crucial : `(user_id, parent_folder_id)` pour lister un dossier
- Replicas read-heavy pour la liste

### 4.c Block storage

- Cible : S3-like (Amazon S3, MinIO, Ceph)
- Replication : 3x within region, 1x cross-region
- TTL : jamais (storage forever par defaut)
- Encryption : AES-256 par bloc, cle par tenant
- Tiering : hot (S3 standard) -> warm (Infrequent Access) -> cold (Glacier) apres 90 jours d'inactivite

### 4.d Sync multi-device

- WebSocket persistent du device vers un "notification service"
- Quand un device upload un nouveau bloc, le metadata service emet un event "file_updated"
- Tous les devices abonnes recoivent l'event en temps reel
- Ils declenchent un "pull" des nouveaux blocs manquants

## 5. Bottlenecks & scale

| Composant | Bottleneck | Solution |
|---|---|---|
| Metadata DB | Lecture heavy sur list_folder | Read replicas, cache Redis (user folder tree) |
| Block upload | Bandwidth dans un DC | LB geographique, ingestion edge |
| Block download | Requetes sur fichiers populaires | CDN (CloudFront) avec cache long |
| Metadata writes | Un user avec 1M fichiers -> partition hot | Sub-partitioning par folder, rate limit per user |
| Dedup table | Table blocks tres lourde | Sharding par hash prefix, cache LRU des blocs populaires |
| Notification | Millions de WS connections | Pool de serveurs de WebSocket avec sticky sessions + pub/sub Redis |

## 6. Extensions

- Versioning granulaire (undo)
- OCR / recherche par contenu (indexation asynchrone avec ElasticSearch)
- Collaboration temps reel (type Google Docs) : ajout CRDT
- Mobile low-bandwidth : chunks plus petits + compression
- GDPR : delete complet + proof of deletion

---

# DESIGN 2 — LLM-Powered Customer Support Assistant

## 1. Clarifier (3 min)

**Fonctionnel** :
- Chat avec les clients sur site / mobile / email
- Repond aux questions frequentes a partir de la knowledge base
- Escalate vers humain si confidence faible ou sujet critique
- Memoire conversationnelle (contexte dans la session)
- Multi-langue
- Outils : consulter le CRM client, creer des tickets, rembourser

**Non-fonctionnel** :
- 500K conversations / jour, moyenne 5 tours
- Latence p95 < 3 s (TTFT < 1 s)
- Uptime 99.9%
- Faithfulness > 90% (pas d'hallucination sur les faits produits)
- Cost target : < $0.10 par conversation

## 2. Estimer (5 min)

```
  Conversations / jour : 500K
  Tours / conversation : 5     -> 2.5M LLM calls / jour ~ 29 calls/s
  Peak factor 3x               -> ~90 calls/s
  Tokens par call : ~3000 in + 500 out (context RAG + response)
    -> 10.5B tokens in / jour
    -> 1.25B tokens out / jour
  Cost gpt-4o-mini : (10500 * 0.15 + 1250 * 0.60) / 1M = ~$2300 / jour
    -> $70K / mois. Trop cher, il faudra router / cache / smaller model.
  Storage knowledge base : 100K articles * 2 Ko ~ 200 Mo (rien)
  Vector index : 1M chunks * 1536 dims * 4 bytes = 6 Go
  Conversations log : 500K * 10 Ko ~ 5 Go / jour
  Cible cout par conversation : 5 calls * $0.005 = $0.025 => on a de la
    marge si on optimise.
```

## 3. High-level design

```
  client (web/mobile) ──> API gateway ──> Auth
                                │
                                v
                        ┌────────────────┐
                        │  LLM Gateway   │
                        │  (router,      │
                        │   cache,       │
                        │   guardrails,  │
                        │   fallback)    │
                        └────┬──────┬────┘
                             │      │
                             v      v
                   ┌──────────┐  ┌────────────┐
                   │ Semantic │  │ RAG engine │
                   │ cache    │  │ - chunker  │
                   │ (Redis)  │  │ - retriever│
                   └──────────┘  │ - reranker │
                                 └────┬───────┘
                                      │
                                      v
                              ┌──────────────┐
                              │  Vector DB   │
                              │  + BM25      │
                              └──────────────┘
                             │
            ┌────────────────┼────────────────┐
            v                v                v
      ┌───────────┐   ┌───────────┐   ┌───────────┐
      │  Agent    │   │ Tools     │   │ Memory    │
      │  runtime  │   │ (CRM,     │   │ - short   │
      │(LangGraph)│   │  ticket,  │   │ - long    │
      │           │   │  refund)  │   │           │
      └─────┬─────┘   └───────────┘   └───────────┘
            │
            v
      ┌───────────────┐
      │ LLM providers │
      │ (OpenAI,      │
      │  Anthropic,   │
      │  self-host)   │
      └───────────────┘

  Transverse :
      - Observability: Langfuse (traces + scoring)
      - MLOps: eval gold set + daily PSI on prompt/response
      - Guardrails: PII scrubbing, prompt injection, groundedness check
```

## 4. Deep dive

### 4.a RAG pipeline

- Source : knowledge base (help center, 100K articles) + FAQs + product docs
- Chunking : document-aware (split par section avec header conservé)
- Embedder : text-embedding-3-small (1536 dims), re-indexation hebdo
- Vector DB : Qdrant ou Pinecone
- Retrieval hybrid : dense + BM25 + RRF
- Reranker : cross-encoder (Cohere Rerank ou BGE)
- Top-k : 5 apres rerank

### 4.b LLM routing

- **Classification task** (intent) : Haiku / nano, cache quasi 60% hit
- **FAQ simple** : Haiku + RAG
- **Probleme complexe** (fraud, refund) : Sonnet + RAG + agent tools
- **Cas ambigu** : fallback vers humain (escalation)

Expected savings : 70% sur le cout LLM brut.

### 4.c Agent + tools

- Supervisor LangGraph : (intent classifier) -> (RAG answerer | tool agent)
- Tools exposes : `lookup_order(order_id)`, `create_ticket(description)`, `refund(order_id, amount)`
- Chaque tool verifie l'authz (le user_id doit matcher le user connecte)
- Memoire courte : 10 derniers messages dans le contexte + resume compact des messages plus anciens

### 4.d Guardrails

- **Input** : PII masking (email, tel, carte), detection de prompt injection
- **Output** :
  - JSON validation pour les tool calls
  - Groundedness check : toute affirmation factuelle doit etre supportee par un chunk retrieved
  - Toxicity check
  - Si le bot "ne sait pas" -> escalade humaine (pas d'invention)

### 4.e Observability

- Tous les calls traces dans Langfuse
- Scoring automatique toutes les 100 conversations : LLM-as-a-judge sur faithfulness, helpfulness, tone
- Dashboard dedie : hit rate cache, cost/conv, escalation rate, top failed queries
- Drift : monitoring des topics entrants (PSI sur la distribution des intents)

## 5. Bottlenecks & scale

| Composant | Bottleneck | Solution |
|---|---|---|
| LLM providers | Quota OpenAI / latency spikes | Multi-provider routing + fallback chain |
| Vector DB | Croissance de la knowledge base | HNSW, sharding par topic, quantization int8 |
| Semantic cache | Faux positifs sur queries proches | Threshold + LLM-judge verifier |
| Agent memory | Context overflow sur longues sessions | Summarization intermediaire + long-term store |
| Guardrails | Latence additive | Parallel guardrail calls + short-circuit |

## 6. Extensions

- Voice support (Whisper + TTS)
- Multi-langue avec auto-translation layer
- A/B testing framework pour prompts / modeles
- Fine-tuning leger d'un modele sur les conversations anotees (reduction de taille + cout)
- Human handoff robuste (transfert context a un agent humain via Slack / Zendesk)

---

## Comparaison : Dropbox vs LLM Support

| Dimension | Dropbox | LLM Support |
|---|---|---|
| Challenge principal | Storage + bandwidth | Qualite + cout + latence |
| Composant critique | Block store + CDN | LLM gateway + RAG |
| Bottleneck | Bandwidth download | Cost per query |
| Scaling | Horizontal (stateless services) | Smart routing + cache + smaller models |
| Observability | Classical (latency, errors) | Tracing + drift + faithfulness |
| Tradeoffs | Storage cost vs latency vs durabilite | Qualite vs cout vs latence vs securite |

---

## Flash cards

**Q: Dans un storage distribue, pourquoi faire du chunking + dedup plutot que de stocker les fichiers entiers ?**
R: Permet de sync partiellement (resume), d'economiser storage (dedup 30-40%), et d'economiser bandwidth (upload incremental).

**Q: Pour un support LLM a $70K/mois brut, comment reduire sous $10K ?**
R: Routing (modeles moins chers pour 70% des queries), semantic cache (20-40% de hits), prompt compression, smaller models after fine-tune.

**Q: Pourquoi faut-il un guardrail de groundedness dans un RAG de support client ?**
R: Parce qu'une affirmation inventee sur une politique de remboursement peut couter cher (legal + reputation). Le bot doit refuser de repondre si pas de source.

**Q: Quel est le single biggest cost driver d'un systeme LLM en prod ?**
R: Le cout des tokens en input (RAG context long). D'ou l'importance de chunking fin, top-k limite, cache semantique, et prompt caching natif.

---

## Key takeaways

- Le framework 6 etapes est reutilisable sur TOUT design d'entretien.
- Les designs "classiques" (Dropbox) tournent autour de storage, bandwidth et metadata DB.
- Les designs "IA" tournent autour de qualite, cout et latence non-deterministe.
- Les patterns J1-J13 sont reutilises partout : load balancing, cache, DB, queue, feature store, RAG, agents, observability.
- La difference entre un design passable et un design senior : les chiffres, les tradeoffs expliques, et les failure modes anticipes.
