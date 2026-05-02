# J8 — RAG Agentique : quand la recherche devient un raisonnement

> **Temps estime** : 3h | **Prerequis** : J1-J7 (agent, tools, memory, planning, reflexion, react)
> **Objectif** : comprendre ce qui separe un RAG "bete" d'un RAG agentique, implementer query decomposition, routing, retrieval grading et multi-hop reasoning.

---

## 1. Le probleme du RAG vanilla

**RAG (Retrieval-Augmented Generation)** dans sa forme la plus simple :

```
User query → Embed → Similarity search → Top-k chunks → LLM genere une reponse
```

Un seul passage. Un seul retrieve. Un seul generate. C'est efficace pour des questions simples ("Qu'est-ce que X ?") mais catastrophique sur des questions composees.

### Exemple qui casse un RAG vanilla

```
Question : "Quel est le CA de Acme en 2025 compare a celui de son plus gros concurrent,
            et quelle est la difference en pourcentage ?"
```

Un RAG vanilla va :
1. Embed la question entiere
2. Chercher 5 chunks "proches semantiquement"
3. Probablement trouver un chunk sur le CA de Acme, mais pas sur les concurrents
4. Generer une reponse incomplete ou hallucinee

Les problemes :
- **Question composee** : 3 informations a chercher (CA Acme, CA concurrent, difference)
- **Pas de plan** : le retriever ne sait pas qu'il doit chercher plusieurs choses
- **Pas de verification** : aucune evaluation de la pertinence des chunks trouves
- **Pas de retry** : si la premiere recherche rate, rien ne se passe

> **Analogie humaine** : un RAG vanilla, c'est un stagiaire qui va sur Google, copie les 3 premiers resultats et te rend une fiche. Un RAG agentique, c'est un analyste qui lit la question, la decompose, cherche chaque sous-partie, verifie la source et synthetise.

---

## 2. RAG vanilla vs RAG agentique — la difference fondamentale

| Aspect | RAG vanilla | RAG agentique |
|--------|-------------|---------------|
| **Nombre de retrieves** | 1 | N (autant que necessaire) |
| **Decomposition** | Aucune — query telle quelle | Decompose en sous-questions |
| **Routing** | Une seule source | Choisit parmi plusieurs sources |
| **Verification** | Aucune — on fait confiance au top-k | LLM juge la pertinence des chunks |
| **Retry** | Jamais | Reformule et re-cherche si echec |
| **Multi-hop** | Impossible | Enchaine les retrieves (step 2 depend du step 1) |
| **Cout** | Faible et previsible | Plus eleve, variable |
| **Qualite** | Bonne pour questions simples | Bonne pour questions complexes |

### Quand utiliser quoi ?

| Type de question | Choisir |
|------------------|---------|
| "Qu'est-ce que X ?" | RAG vanilla |
| "Quels sont les 3 points cles de X ?" | RAG vanilla |
| "Compare X et Y sur le critere Z" | Agentique |
| "Quel est l'impact de X sur Y ?" (necessite plusieurs sauts) | Agentique |
| "Donne-moi les dernieres decisions sur X et leur rationale" | Agentique |

**Regle pragmatique** : si la question necessite plus d'un "lookup", va en agentique.

---

## 3. Les 5 briques d'un RAG agentique

Un RAG agentique bien concu combine 5 techniques. Chacune peut etre ajoutee incrementalement — commence par 1, ajoute les autres selon les besoins.

### 3.1 Query Decomposition — decomposer la question

**Principe** : prendre une question composee et la transformer en N sous-questions simples.

```
Question : "Quel est le CA de Acme en 2025 compare a celui de son plus gros concurrent ?"

LLM decompose →
  Sub-question 1 : "Quel est le CA de Acme en 2025 ?"
  Sub-question 2 : "Qui est le plus gros concurrent de Acme ?"
  Sub-question 3 : "Quel est le CA de <concurrent> en 2025 ?"
```

**Pattern** : prompt le LLM avec `"Decompose this question into 2-5 simple sub-questions, each answerable with a single lookup."`.

**Piege courant** : sur-decomposition. Si tu decomposes "Qu'est-ce que Python ?" en 5 sous-questions, tu exploses le cout pour rien. Laisse le LLM decider combien de sous-questions sont necessaires (0 si la question est deja simple).

### 3.2 Routing — choisir la bonne source

**Principe** : tu as plusieurs bases de connaissances. Le LLM decide laquelle interroger selon le type de question.

```
Bases disponibles :
  - docs_produit   : documentation produit interne
  - docs_api       : reference API technique
  - blog           : articles de blog, insights marche
  - jira           : tickets, decisions produit

Question : "Comment je fais un POST sur /users ?"
Routing → docs_api (seul)

Question : "Pourquoi on a decide de supprimer la feature X ?"
Routing → jira + blog
```

**Implementation** : un petit prompt "Given this question, which of these sources should be queried? Return a JSON list." Le LLM retourne une liste de sources.

**Pourquoi c'est crucial** : ca evite de chercher dans du contenu non-pertinent. Chercher une question technique dans des articles de blog = bruit garanti.

### 3.3 Retrieval Grading — juger la pertinence

**Principe** : apres chaque retrieve, le LLM juge si chaque chunk est **relevant** pour la sous-question. Les chunks non-pertinents sont jetes.

```
Sub-question : "Quel est le CA de Acme en 2025 ?"
Retrieved chunks :
  1. "Acme a leve 2M€ en 2024..."         ← grade : IRRELEVANT (ce n'est pas le CA)
  2. "Le chiffre d'affaires 2025 : 4.2M€"    ← grade : RELEVANT
  3. "Acme cible les TPE et PME..."       ← grade : IRRELEVANT
```

**Pattern** : pour chaque chunk, prompter `"Does this chunk help answer the sub-question? Answer RELEVANT or IRRELEVANT."`.

**Variantes avancees** :
- **Grade continu** : score 0-1 plutot que binaire
- **Grade avec justification** : "IRRELEVANT because this is about funding, not revenue"
- **Grade groupe** : juger l'ensemble des chunks d'un coup (moins precis, moins cher)

> **Opinion** : le retrieval grading est le plus gros levier de qualite d'un RAG. Un grading correct divise par 3 le taux d'hallucinations.

### 3.4 Adaptive Retrieval — retry avec reformulation

**Principe** : si apres grading, aucun chunk pertinent n'est trouve, le LLM **reformule la question** et re-cherche.

```
Sub-question originale : "Quel est le CA de Acme en 2025 ?"
→ Retrieve → 0 chunks relevants

LLM reformule → "Chiffre d'affaires revenue 2025 Acme SAS"
→ Retrieve → 2 chunks relevants ✓
```

**Strategies de reformulation** :
- **Synonymes** : "CA" → "chiffre d'affaires", "revenue"
- **Expansion** : ajouter des termes de contexte ("Acme SAS", "2025 annee")
- **Restriction** : enlever des termes trop specifiques qui bloquent la recherche
- **HyDE** (Hypothetical Document Embeddings) : generer une reponse hypothetique et embedder cette reponse plutot que la question

**Budget de retry** : limiter a 2-3 essais. Au-dela, considerer que l'info n'est pas dans la base et remonter le signal "not found" plutot que de boucler.

### 3.5 Multi-hop Reasoning — enchainer les recherches

**Principe** : le resultat d'un retrieve devient l'input du suivant. On enchaine les retrieves pour faire un **raisonnement par etapes**.

```
Question : "Qui est le CEO de la company qui a achete Instagram ?"

Hop 1 : "Qui a achete Instagram ?"
  → Retrieve → "Facebook (now Meta) acquired Instagram in 2012"
  → Extraction : "Meta"

Hop 2 : "Qui est le CEO de Meta ?"
  → Retrieve → "Meta's CEO is Mark Zuckerberg"
  → Extraction : "Mark Zuckerberg"

Reponse finale : "Mark Zuckerberg"
```

**Pattern** : a chaque hop, le LLM extrait l'info cle de la reponse, l'utilise pour formuler la prochaine sous-question.

**Limite** : chaque hop ajoute de la latence et du cout. En pratique, 2-4 hops max.

---

## 4. Architecture complete d'un RAG agentique

```
┌──────────────────────────────────────────────────────────────────┐
│                          USER QUERY                              │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
                    ┌────────────────┐
                    │  DECOMPOSER    │   → [sub_q1, sub_q2, ...]
                    └────────┬───────┘
                             ▼
                    ┌────────────────┐
                    │    ROUTER      │   → source per sub-query
                    └────────┬───────┘
                             ▼
              ┌──────────────┴───────────────┐
              ▼              ▼                ▼
         ┌────────┐    ┌────────┐       ┌────────┐
         │ source1 │    │ source2 │       │ source3 │
         └────┬───┘    └────┬───┘       └────┬───┘
              └──────────────┼────────────────┘
                             ▼
                    ┌────────────────┐
                    │    GRADER      │   → keep only relevant
                    └────────┬───────┘
                             ▼
                        [relevant?]
                           /    \
                         Yes     No
                         │        │
                         │        ▼
                         │   ┌────────────┐
                         │   │ REFORMULATE│ (retry, max 3)
                         │   └────┬───────┘
                         │        │ (back to router)
                         ▼
                    ┌────────────────┐
                    │  NEXT HOP?     │   (multi-hop loop)
                    └────────┬───────┘
                             │ (no)
                             ▼
                    ┌────────────────┐
                    │   SYNTHESIZER  │   → final answer
                    └────────┬───────┘
                             ▼
                        FINAL ANSWER
```

**Points cles** :
- Le **decomposer** et le **router** s'executent **une seule fois** au debut
- Le **grader** et le **reformulator** forment une **boucle** jusqu'a trouver des chunks pertinents
- Le **multi-hop** est une autre boucle a un niveau superieur (apres qu'une sous-question est repondue, on decide si on lance une nouvelle sous-question derivee)
- Le **synthesizer** fait la synthese finale a partir de toutes les reponses intermediaires

---

## 5. Retrieval strategies — au-dela de la similarity brute

La brique retriever (le composant qui cherche dans le corpus) a plusieurs strategies.

### 5.1 Dense retrieval (embeddings)

```
Query → embedding → cosine similarity with doc embeddings → top-k
```

**Avantages** : capture la semantique ("laptop" et "ordinateur portable" sont proches).
**Limites** : rate les termes rares, les noms propres, les codes produits.

### 5.2 Sparse retrieval (BM25 / TF-IDF)

```
Query → keyword matching avec ponderation → top-k
```

**Avantages** : excellent pour les termes rares, les noms exacts, les codes.
**Limites** : ne capture pas la semantique ("car" et "automobile" sont independants).

### 5.3 Hybrid retrieval (dense + sparse)

```
Dense → top-k1
Sparse → top-k2
Merge + rerank → final top-k
```

**Avantages** : le meilleur des deux mondes. Standard en production.

### 5.4 Reranking

```
Top 50 candidats (dense + sparse) → Cross-encoder rerank → top 5
```

Un modele de rerank (Cohere Rerank, BGE-reranker) prend chaque (query, doc) et donne un score precis. C'est plus lent mais **beaucoup plus precis** que la similarity cosine.

**Heuristique** : retrieve top-50, rerank, garde top-5. Le saut de qualite est tres significatif.

### 5.5 Query expansion / HyDE

Au lieu d'embedder la question, on embedd **une reponse hypothetique** a la question :

```
Query : "Comment configurer le webhook Stripe ?"

LLM genere une reponse hypothetique :
  "Pour configurer un webhook Stripe, allez dans Dashboard > Developers > Webhooks
   et ajoutez l'URL de votre endpoint. Selectionnez les events a ecouter..."

Embed cette reponse hypothetique → cherche dans le corpus
```

**Pourquoi ca marche** : la reponse hypothetique ressemble plus a un chunk de doc que la question initiale. L'embedding est plus proche des vrais chunks pertinents.

---

## 6. Advanced RAG Patterns (2024-2025)

Les 5 briques de la section 3 constituent la base d'un RAG agentique. En 2024-2025, plusieurs patterns plus avances sont apparus et se sont imposes en production.

### Self-RAG (Asai et al., 2024)

**Idee** : l'agent genere des **tokens speciaux** avant et pendant sa reponse pour controler explicitement le retrieval.

```
[RETRIEVE]       → Est-ce que j'ai besoin de chercher ? oui/non
[RELEVANCE]      → Pour chaque chunk : pertinent / pas pertinent
[SUPPORTED]      → Cette phrase est-elle supportee par un chunk cite ?
[UTILITY]        → La reponse finale est-elle utile a l'utilisateur ?
```

**Exemple simplifie** :

```
Query : "Quel est le CA de Acme en 2025 ?"

LLM genere :
  [RETRIEVE=yes]
  -> chunk1: "Acme a leve 2M€ en 2024"        [RELEVANCE=no]
  -> chunk2: "CA 2025 : 4.2M€"                  [RELEVANCE=yes]
  Reponse : "Le CA de Acme en 2025 est de 4.2M€."  [SUPPORTED=yes] [UTILITY=5]
```

**Avantages** : auto-correction, transparence (chaque decision est explicite), evaluation facile a posteriori. **Inconvenient** : necessite un modele fine-tune sur ces tokens (ou un prompting tres careful avec un LLM generaliste).

### Corrective RAG (CRAG, Yan et al., 2024)

**Idee** : avant de synthetiser, un **evaluateur leger** (T5 fine-tune) grade le retrieval avec un score de confidence.

```
Score eleve (> 0.8)   → utiliser direct
Score bas  (< 0.4)    → fallback : web search + reformulation de query
Score intermediaire    → merge retrieval local + web search
```

**Difference avec "adaptive RAG"** : adaptive reformule seulement la query, CRAG **decide dynamiquement** d'aller chercher ailleurs (web, API) selon la confidence du retrieval initial. C'est une strategie de **triage** avec plusieurs chemins possibles.

**Utilite** : bonne defense contre les bases de connaissances incompletes. Si ta doc interne n'a pas l'info, l'agent passe automatiquement au web.

### GraphRAG (Microsoft, 2024)

**Idee** : indexer le corpus non pas comme des chunks independants mais comme un **knowledge graph** d'entites et de relations.

```
Phase offline :
  1. Extraire les entites et relations de chaque doc (ex: (Acme) --[produit]-> (Legaly))
  2. Construire un graph global (nodes + edges)
  3. Detecter des "communities" (clusters d'entites liees)
  4. Generer un summary par community (via LLM)

Phase query :
  1. Extraire les entites de la query
  2. Traverser le graph a partir de ces entites
  3. Retourner les communities pertinentes (summaries) + chunks associes
```

**Gain** : beaucoup mieux que dense retrieval sur les queries **multi-hop** qui necessitent de relier plusieurs concepts.

```
Query : "Quelles startups soutenues par Acme travaillent sur l'IA en Afrique de l'Ouest ?"

Dense RAG : fait un matching semantique, trouve 5 chunks qui parlent de Acme ou d'IA, peine a faire le lien
GraphRAG  : traverse (Acme)->[invest]->(X,Y,Z), filtre sur (region=Afrique de l'Ouest), retourne la liste
```

**Cout** : l'indexation initiale est chere (N appels LLM pour extraire entites/relations de chaque doc), mais les queries sont rapides ensuite. Utile sur des corpus stables et structures (docs internes, wikis).

### ColBERT v2 (late interaction)

**Idee** : au lieu d'embedder chaque chunk en **un seul vecteur**, on stocke **un embedding par token** du chunk. Au query time, on calcule :

```
score(query, chunk) = sum over q_tokens of max_similarity(q_token, chunk_tokens)
```

C'est une "late interaction" : on retient l'info token-level au lieu de la pooler en un vecteur unique.

**Gain** : **+10-30 points de recall** sur les queries avec entites rares, nuances fines, ou jargon technique. Les mots peu frequents (noms propres, codes, termes techniques) sont conserves au lieu d'etre noyes dans un pooling.

**Cout** : 10-100x plus cher en storage qu'un dense retriever classique, et plus lent au query time. **Solution pragma** : utiliser ColBERT comme **rerank** (top 50 candidats du dense retriever) plutot que comme retriever principal.

### Hybrid search + RRF + cross-encoder rerank — le pattern production 2026

C'est la pipeline standard de **toutes les apps RAG serieuses** en 2026 :

```
Query
  │
  ├─> Dense retriever (bge-m3, text-embedding-3-large)  → top 50
  │
  ├─> Sparse retriever (BM25)                            → top 50
  │
  ▼
Reciprocal Rank Fusion (RRF) : fusion des 2 listes → top 50 merge
  │
  ▼
Cross-encoder rerank (bge-reranker-v2-m3, Cohere rerank v3) : scoring precis → top 5
  │
  ▼
LLM answer generation
```

**Pourquoi cette combinaison** :
- **Dense** capture la semantique (gere les synonymes)
- **Sparse (BM25)** capture les termes rares et les noms propres (que le dense rate)
- **RRF** combine les deux rankings sans avoir a normaliser les scores (robuste, facile a implementer)
- **Cross-encoder rerank** redonne un score precis query-doc (beaucoup plus fin que cosine)
- Prendre top 5 en entree du LLM = minimiser le bruit dans le contexte

**Resultat empirique** : cette pipeline passe typiquement le recall@5 de **60% (dense seul)** a **85-90%** (hybrid + rerank).

> **Retour d'experience** : si tu construis un nouveau RAG en 2026, ne commence pas par "je fais du dense retrieval avec Chroma et un top 5". Commence directement avec cette pipeline — l'ecart de qualite est enorme pour une complexite marginalement superieure.

---

## 7. Pieges courants et comment les eviter

### 7.1 Hallucination "plausible mais fausse"

**Symptome** : le LLM genere une reponse qui sonne bien mais ne vient pas des documents.

**Causes** :
- Chunks non-pertinents dans le contexte (grading manque)
- Prompt trop permissif ("answer the question" sans "based on the provided context only")
- Manque de citations

**Defense** :
- Grading strict
- Prompt "Answer ONLY using the provided context. If the answer is not in the context, say 'I don't know'."
- Forcer les citations : chaque phrase doit pointer vers un chunk source

### 7.2 Boucle infinie sur reformulation

**Symptome** : l'agent reformule 15 fois sans jamais trouver de chunk pertinent.

**Cause** : l'info n'est pas dans la base, mais l'agent insiste.

**Defense** : budget de retry (max 3), puis escalade en "not found" propre.

### 7.3 Explosion du cout en multi-hop

**Symptome** : une question simple fait 10 hops a 5 sous-questions chacun = 50 appels LLM.

**Cause** : decomposition trop agressive, absence de budget.

**Defense** :
- Compter les appels LLM par query
- Budget max (ex: 20 appels LLM par query)
- Le decomposer doit respecter un max de 5 sous-questions

### 7.4 Chunks trop gros ou trop petits

**Chunk trop gros** (>2000 tokens) : le retriever ramene des chunks qui contiennent l'info mais diluee dans du bruit.
**Chunk trop petit** (<100 tokens) : l'info est coupee, on perd le contexte.

**Sweet spot** : 300-800 tokens par chunk, avec **overlap** de 50-100 tokens entre chunks adjacents.

### 7.5 Pas de metadata

**Symptome** : le retriever cherche dans tout le corpus, trouve des chunks obsoletes ou d'autres utilisateurs.

**Defense** : attacher des metadata (date, type, source, user_id) et **pre-filtrer** avant la recherche vectorielle.

---

## 8. Frameworks et outils

| Outil | Niveau | Quand l'utiliser |
|-------|--------|------------------|
| **LlamaIndex** | Framework RAG dedie | Si tu construis un RAG comme produit principal |
| **LangChain RAG chains** | Briques composables | Si tu veux assembler RAG + agent + tools |
| **LangGraph avec retriever nodes** | Bas niveau, flexible | Si tu as besoin d'un RAG agentique custom |
| **Haystack** | Alternative open-source | Si tu veux une stack enterprise Python |
| **RAGatouille** | Focus sur rerank + ColBERT | Si la precision est critique |

**Vector stores** : Chroma (local dev), Qdrant (prod self-hosted), Pinecone (prod managed), Weaviate (avec hybrid natif).

**Embeddings** :
- OpenAI `text-embedding-3-small` — defaut pragmatique
- Voyage `voyage-3` — meilleur qualite/prix
- `bge-large-en-v1.5` — local, gratuit, excellent

---

## 9. Flash Cards — Test de comprehension

**Q1 : Quelle est la difference fondamentale entre un RAG vanilla et un RAG agentique ?**
> R : Le RAG vanilla fait **un seul** retrieve + generate sans verification. Le RAG agentique decompose la question, route vers la bonne source, juge la pertinence des chunks, retry si necessaire, et peut enchainer plusieurs retrieves (multi-hop). L'agentique est un raisonnement avec la recherche, pas juste une recherche.

**Q2 : A quoi sert le "retrieval grading" et pourquoi est-il considere comme le plus gros levier de qualite ?**
> R : Le grading demande au LLM de juger si chaque chunk recupere est reellement pertinent pour la question, et de jeter les chunks non-pertinents avant la generation. C'est le plus gros levier parce qu'il divise drastiquement le taux d'hallucinations : on ne genere que sur du contenu verifie comme utile.

**Q3 : Dans quels cas utiliser un RAG vanilla vs un RAG agentique ?**
> R : RAG vanilla pour les questions simples qui necessitent **un seul lookup** ("Qu'est-ce que X ?", "Definition de Y"). RAG agentique des que la question est composee, necessite une comparaison, un raisonnement multi-etapes, ou plusieurs sources differentes. Regle pragmatique : plus d'un lookup = agentique.

**Q4 : Qu'est-ce que HyDE et pourquoi ca marche ?**
> R : HyDE (Hypothetical Document Embeddings) consiste a demander au LLM de **generer une reponse hypothetique** a la question, puis a embedder cette reponse plutot que la question. Ca marche parce que la reponse hypothetique ressemble plus a un chunk de doc que la question initiale — les embeddings sont donc plus proches des vrais chunks pertinents dans le vector store.

**Q5 : Quels sont les 3 pieges les plus courants d'un RAG agentique et leur defense ?**
> R : (1) **Hallucination plausible** → grading strict + prompt "only use context" + citations obligatoires. (2) **Boucle infinie de reformulation** → budget max de retry (3) et escalade "not found". (3) **Explosion du cout en multi-hop** → compter les appels LLM par query et imposer un budget max (ex: 20).

---

## Points cles a retenir

- RAG vanilla = 1 retrieve + 1 generate. RAG agentique = boucle de recherche, grading, retry, multi-hop
- **Query decomposition** : transformer une question composee en sous-questions simples
- **Routing** : choisir la source de donnees appropriee selon la question
- **Retrieval grading** : LLM juge la pertinence de chaque chunk, jette le bruit (plus gros levier qualite)
- **Adaptive retrieval** : reformuler et retry si la premiere recherche rate (avec budget max)
- **Multi-hop** : enchainer les retrieves pour repondre a des questions necessitant plusieurs sauts
- Retrieval hybrid (dense + sparse + rerank) > similarity cosine seule
- Les chunks doivent faire 300-800 tokens avec overlap de 50-100 tokens
- Metadata filtering avant similarity search = performance + pertinence
- Budget strict sur les appels LLM pour eviter l'explosion du cout


---

## Pour aller plus loin

Lectures couvrant ce sujet (playlists dans [`shared/external-courses.md`](../../../shared/external-courses.md)) :

- **CMU 11-711 (Neubig, Fa24) — Lec. 10 (Retrieval and RAG)** — fondations academiques du RAG, dense vs sparse, evaluation.
- **Berkeley CS294-196 (Fa24) — Lec. 8 (Compound AI & DSPy, Khattab)** — pipelines de retrieval optimisables et auto-tunes.
