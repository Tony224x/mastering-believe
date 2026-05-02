# Jour 10 — RAG Architecture

## Pourquoi le RAG est devenu le pattern dominant

**Exemple d'abord** : Ton boss te demande un chatbot qui repond aux questions sur la documentation interne (10K pages). Option 1 : fine-tuner un LLM sur tout ce corpus. Cout : des milliers de dollars, plusieurs jours, et tu dois recommencer a chaque mise a jour de la doc. Option 2 : RAG. Tu indexes une fois (10 minutes), tu branches un petit moteur de retrieval, et un LLM generaliste repond en citant les sources. Cout : quelques dollars, 1 journee de dev, mise a jour en temps reel.

C'est pour ca que le RAG (Retrieval-Augmented Generation) est devenu le pattern par defaut pour brancher un LLM a une base de connaissances. Le fine-tuning est reserve aux cas ou on veut changer le **style** ou le **comportement** du modele, pas quand on veut lui apprendre des **faits**.

**Key takeaway** : Un LLM est un moteur de raisonnement. Un RAG est un moteur de raisonnement + une bibliotheque. Si les faits que tu veux injecter changent souvent, tu veux une bibliotheque, pas un cerveau grave dans le silicium.

---

## L'architecture RAG canonique

```
  1) INDEXATION (offline, periodic)
  ┌───────────────┐    ┌──────────┐    ┌───────────────┐    ┌──────────────┐
  │  Documents    │───>│ Chunker  │───>│ Embedder      │───>│ Vector DB    │
  │  (pdf, md,    │    │          │    │ (model API)   │    │ (Qdrant,     │
  │   html, ...)  │    └──────────┘    └───────────────┘    │  Pinecone)   │
  └───────────────┘                                          └──────────────┘
                                                                    │
                                                                    v
                                                            ┌──────────────┐
                                                            │  BM25 index  │
                                                            │  (optional,  │
                                                            │   hybrid)    │
                                                            └──────────────┘

  2) QUERY (online, par requete)
  ┌─────────┐   ┌─────────┐   ┌────────────────┐   ┌──────────┐   ┌──────┐   ┌────────────┐
  │  User   │──>│ Query   │──>│  Hybrid search │──>│ Reranker │──>│ LLM  │──>│ Response   │
  │ question│   │ rewrite │   │ (dense + BM25) │   │ (cross-  │   │      │   │ + citations│
  └─────────┘   └─────────┘   └────────────────┘   │ encoder) │   └──────┘   └────────────┘
                                                    └──────────┘
```

Chaque flache est une optimisation potentielle. Un RAG naif qui saute des etapes (pas de reranker, pas de hybrid) marche souvent a 60-70%. Le passage a 90% demande de reflechir chaque brique.

---

## Vector databases : le stockage des embeddings

Un **embedding** est un vecteur de dimension 768, 1024 ou 3072 qui represente un bout de texte dans un espace semantique. Deux textes proches en sens ont des embeddings proches (distance cosine).

### Les options

| Solution | Type | Force | Quand |
|---|---|---|---|
| **pgvector** | Extension PostgreSQL | Simple, dans ta DB existante | < 1M vecteurs, proto rapide |
| **Qdrant** | DB vectorielle dedie, open source | Rust, tres rapide, filtres hybrides | Prod moyenne, self-hosted |
| **Pinecone** | SaaS managed | Zero ops, autoscale | Prod enterprise, budget |
| **Weaviate** | Open source + cloud | GraphQL, modules d'enrichissement | Besoins schemas complexes |
| **Milvus / Zilliz** | Open source + cloud | Tres gros volumes | 100M+ vecteurs |
| **ChromaDB** | Local, embed dans l'app | Hyper simple pour prototypes | Demos, pas de prod |
| **FAISS** | Lib C++, pas de serveur | Le plus rapide brut | Recherche pure, pas de CRUD |

### Indexation approximate (ANN)

Avec 10M vecteurs de dimension 1536, une recherche exacte coute O(10M * 1536) = trop lent. Les vector DBs utilisent des index **ANN** (Approximate Nearest Neighbor) :

- **HNSW** (Hierarchical Navigable Small World) : graphe multi-niveaux, le plus utilise, recall 95%+ avec 10x d'acceleration
- **IVF** (Inverted File Index) : clustering par k-means, moins precis mais moins couteux en memoire
- **PQ** (Product Quantization) : compression des vecteurs, trade memoire vs precision

**Regle** : HNSW est le defaut moderne. Sauf si tu as des contraintes memoire extremes (100M+ vecteurs), HNSW est le bon choix.

---

## Chunking : la decision la plus sous-estimee

Tu ne peux pas embed une doc entiere de 50 pages. Tu dois la couper en **chunks**. Mal chunker = les bons passages sont noyes dans du bruit ou coupes en deux. Bien chunker = la moitie du travail du RAG.

### Strategies

**1. Fixed-size chunking**
- Decouper tous les N tokens (ex: 512 tokens avec 50 tokens d'overlap)
- Simple, robuste, fonctionne partout
- Defaut acceptable pour demarrer

**2. Recursive chunking** (par hierarchie de separateurs)
- Essaie de couper sur `\n\n`, puis `\n`, puis `. `, puis ` `
- Respecte la structure logique (paragraphes, phrases)
- Implemente dans LangChain `RecursiveCharacterTextSplitter`
- Bon defaut pour le texte general

**3. Semantic chunking**
- Calcule l'embedding de chaque phrase
- Coupe quand la distance cosine entre 2 phrases consecutives depasse un seuil
- Plus cher, mais respecte les frontieres de sens
- Utile pour des documents a structure heterogene

**4. Document-aware chunking**
- Utiliser la structure du document : un chunk = une section, un chunk = une page, un chunk = un paragraphe avec son titre
- Specifique au format (markdown headers, PDF TOC, HTML tags)
- Meilleure qualite mais plus d'engineering

**5. Sliding window avec overlap**
- Chunks qui se recouvrent de ~10-20%
- Evite qu'une info cle soit coupee en deux
- Contrepartie : redondance dans l'index

### Taille des chunks

| Contexte | Taille recommandee |
|---|---|
| Documentation technique | 500-1000 tokens |
| Articles / blogs | 300-500 tokens |
| Code source | 200-500 tokens, chunking par fonction |
| Legal / contrats | 800-1500 tokens (contexte critique) |
| Transcripts (audio) | 200-300 tokens par tour de parole |

**Principe** : plus petit = plus precis mais plus de chunks a traiter. Plus grand = plus de contexte mais plus de bruit. Un bon defaut est ~500 tokens.

---

## Retrieval : dense vs sparse vs hybrid

### Dense retrieval (semantic)

- Embedder la query et les chunks avec un modele (OpenAI, Cohere, BGE)
- Chercher les top-k plus proches en cosine
- **Force** : comprend les synonymes, paraphrases, concepts
- **Faiblesse** : rate les mots exacts (noms propres, codes produits, chiffres specifiques)

### Sparse retrieval (BM25 / keyword)

- Algorithme classique de recherche textuelle (TF-IDF ameliore)
- **Force** : match exact des mots, tres bon sur les termes rares
- **Faiblesse** : ne comprend pas la semantique

### Hybrid retrieval (dense + sparse)

Combiner les deux et fusionner les scores :

```
    top_k_dense  = vector_db.search(query_embedding, k=20)
    top_k_sparse = bm25_index.search(query, k=20)
    # Reciprocal Rank Fusion (RRF)
    score(doc) = sum(1 / (rank_in_list + k)) for each list the doc is in
```

**RRF** (Reciprocal Rank Fusion) : la methode de fusion la plus populaire, parametre `k=60` par defaut.

**Regle** : pour du RAG serieux, **toujours hybrid**. C'est un gain de 10-20% de recall gratuit.

---

## Reranking : le secret de la qualite

Le retrieval retourne les top-20 ou top-50 candidats. Mais les positions 1-5 ne sont pas forcement les meilleures. Un **reranker** est un modele plus gros et plus precis qui re-score les candidats.

### Cross-encoder vs bi-encoder

- **Bi-encoder** (celui utilise pour l'embedding) : encode query et doc separement, compare les vecteurs. Rapide, scalable. Utilise pour le retrieval.
- **Cross-encoder** : encode (query, doc) ensemble et predit un score de pertinence. Plus lent (quadratique), mais bien plus precis.

### Workflow

```
  query ──> embedder ──> retrieve top 50 (dense + bm25)
                                     │
                                     v
                        cross-encoder(query, doc) for each
                                     │
                                     v
                              top 5 reranked ──> LLM
```

**Modeles courants** : Cohere Rerank, BGE-Reranker, BAAI-bge-reranker-v2-m3, JinaAI reranker.

**Gain typique** : +15-30% sur MRR (Mean Reciprocal Rank) du premier vrai positif. Souvent la difference entre "marche" et "marche pas".

---

## Evaluer un RAG : les metriques qui comptent

Un RAG a deux etapes : retrieval et generation. Il faut evaluer les deux separement.

### Metriques de retrieval

- **Recall@k** : la bonne reponse est-elle dans les top-k ? Cible : recall@10 > 90%
- **MRR** (Mean Reciprocal Rank) : 1/position_du_premier_vrai. Plus c'est haut, mieux c'est.
- **nDCG** : recompense les positions correctes en haut de liste
- **Hit rate** : fraction des queries ou un document pertinent est retourne

### Metriques de generation

- **Faithfulness / groundedness** : la reponse est-elle supportee par les documents retrieves ? (evalue par LLM-as-a-judge ou cross-reference)
- **Answer relevance** : la reponse repond-elle a la question ?
- **Context precision** : fraction des docs retrieves qui sont utilises dans la reponse
- **Context recall** : fraction de l'info cle qui est dans les docs retrieves

**Outils** : Ragas, TruLens, DeepEval, Langsmith.

### Dataset d'evaluation

- Toujours creer un "gold set" : 50-200 paires (question, reponse_attendue, documents_pertinents)
- Re-evaluer a chaque changement de config (modele embedding, taille chunks, reranker)
- Sans dataset, tu optimises a l'aveugle

---

## Citations : la base du trust

Un RAG doit **toujours** citer ses sources. Sans citations :
- L'utilisateur ne peut pas verifier
- Le LLM a plus tendance a halluciner
- Tu perds la traçabilite

### Patterns de citation

**1. Inline numbered** (Perplexity, Bing Chat)
- Chaque phrase est suivie d'un `[1]`, `[2]` qui renvoie au document
- Standard de facto

**2. Footnote style**
- Citations en bas de reponse
- Moins gouvernante mais plus lisible pour de longues reponses

**3. Structured JSON**
- `{"answer": "...", "citations": [{"chunk_id": "...", "quote": "..."}]}`
- Ideal pour APIs et verifications automatisees

**Prompt pattern** :
```
Tu dois repondre a la question en utilisant UNIQUEMENT les passages numerotes ci-dessous.
Chaque affirmation dans ta reponse doit etre suivie d'un [N] renvoyant au passage utilise.
Si l'info n'est pas dans les passages, reponds "Je ne sais pas d'apres les documents fournis."

Passages:
[1] ...
[2] ...

Question: ...
```

---

## RAG avance 2025-2026 : au-dela du pipeline canonique

Le pipeline "chunk -> embed -> hybrid search -> rerank -> generate" reste le defaut. Mais l'etat de l'art 2025-2026 empile des techniques qui traitent les cas ou ce pipeline bute : queries multi-hop, entites rares, nuances fines, budget de recherche dynamique.

### GraphRAG (Microsoft Research, 2024)

Au lieu d'indexer les docs comme un sac de chunks, on construit d'abord un **knowledge graph** : entites (personnes, projets, concepts) extraites par LLM, relations explicites entre elles, communautes detectees par clustering de graphe. A la query, on **traverse les relations** au lieu de faire du retrieval plat.

- **Force** : queries multi-hop du type "qui a travaille avec X sur le projet Y en 2023 ?". Un dense retrieval rate ce genre de question parce que l'info est eclatee entre plusieurs docs. Le graphe la reconstitue.
- **Faiblesse** : cout d'indexation eleve (chaque doc est traite par un LLM pour extraire entites + relations), et le graph a besoin d'etre re-genere quand le corpus bouge.
- **Quand** : bases internes riches en entites (wiki d'entreprise, CRM, documents juridiques), surtout si l'audit et l'explicabilite comptent.

### Agentic RAG

Au lieu d'un pipeline fixe, un **agent** decide dynamiquement quoi retrieve, quand reformuler la query, quand s'arreter. Patterns standards :

- **Self-RAG** : le LLM note chaque passage retrieve (pertinent ? suffisant ?). Si insuffisant, il lance une nouvelle recherche avec une query raffinee. Si contradictoire, il genere une reponse prudente avec marqueurs d'incertitude.
- **Corrective RAG (CRAG)** : un evaluateur leger classe les resultats retrieves en `correct` / `ambiguous` / `incorrect`. Sur `ambiguous`, l'agent declenche une recherche web. Sur `incorrect`, il reformule.
- **Gain** : +10-25% sur des benchmarks complexes comme PopQA ou TriviaQA, au prix de 2-4x plus d'appels LLM par requete.

### ColBERT v2 : late interaction

Le dense retrieval classique encode un chunk en **un seul vecteur** : tu perds toutes les nuances intra-chunk. **ColBERT v2** stocke un embedding **par token**, et matche au niveau token entre query et doc (MaxSim).

- **Force** : gain de 10-30% en recall sur les queries avec entites rares, acronymes, nuances linguistiques. Difference massive sur du legal, medical, scientifique.
- **Cout** : 10-50x plus de stockage que le dense classique. PLAID est l'implementation de reference qui compresse et accelere.
- **Quand** : prod serieuse ou la qualite domine le cout, et ou le hybrid dense+BM25 plafonne.

### Hybrid search + RRF + cross-encoder rerank

Le pattern de prod 2026 generalise le hybrid a un pipeline en 3 etapes :

1. **Retrieval parallele** : dense (top-50) + BM25 (top-50)
2. **Fusion par RRF** : merge des 2 listes via Reciprocal Rank Fusion (top-30-50)
3. **Rerank cross-encoder** : re-scoring des candidats avec un gros modele de rerank, puis top-5 au LLM

Les rerankers qui dominent en 2026 : **BGE-reranker-v2-m3** (open source, multilingue, tres bon), **Cohere Rerank v3** (API, gere 100+ langues), **Jina Reranker v2** (petit, rapide, bon compromis edge). Toujours tester sur ton gold set : l'ecart entre rerankers peut atteindre 15% en MRR.

### Vector DBs 2026 : les nouveaux entrants

Pinecone et Weaviate restent valides mais leur tarification devient dure a justifier face aux alternatives modernes :

- **Turbopuffer** : serverless, stockage sur blob storage (S3 / R2), tres cost-efficient pour des workloads asymetriques (lourd en write, leger en read ou l'inverse). Choix montant pour les startups 2025.
- **LanceDB** : embedded (comme SQLite pour vecteurs), format Lance columnar, zero ops, excellent pour des apps locales, edge, ou embedded dans un worker Python. Gere le multimodal (texte + images) nativement.
- **Qdrant** : open source et cloud, filtres hybrides puissants, payload riche, tres bon DX. Le defaut pragmatique pour une prod self-hostable.

Pinecone reste un choix solide pour l'enterprise avec du budget et une exigence "zero ops totale". Weaviate pour qui a besoin des modules d'enrichissement et d'un schema GraphQL.

**Key takeaway** : en 2026, un RAG competitif empile GraphRAG (quand le domaine est riche en entites) OU agentic RAG (quand les queries sont ouvertes), avec hybrid + RRF + cross-encoder rerank par defaut, sur une vector DB adaptee au profil de cout (Turbopuffer / LanceDB / Qdrant > Pinecone pour la plupart des cas).

---

## Tradeoffs recapitulatifs

| Dimension | Petit | Grand | Tradeoff |
|---|---|---|---|
| Chunk size | 200 tokens | 1000 tokens | Precision vs contexte |
| Top-k retrieval | 5 | 50 | Bruit vs recall |
| Reranker | Aucun | Cross-encoder | Latence vs qualite |
| Vector DB | pgvector | Pinecone | Cout/ops vs scale |
| Embedding model | MiniLM local | OpenAI 3-large | Cout vs qualite |

---

## Exemples reels

- **Perplexity** : hybrid search + reranker + citations inline + post-hoc verification
- **Notion AI** : chunking par block, embeddings pre-computed
- **GitHub Copilot Chat** : retrieval du code voisin + tree-sitter pour le contexte
- **Claude / ChatGPT plugins avec retrieval** : vector DB par tenant, reranking systematique

---

## Flash cards

**Q: Pourquoi utiliser un hybrid search dense + BM25 ?**
R: Le dense capture la semantique, BM25 capture les matches exacts (noms propres, codes). Ensemble : recall 10-20% superieur a chacun seul.

**Q: Qu'est-ce qu'un cross-encoder ?**
R: Un modele qui prend (query, doc) en entree et predit un score. Plus precis qu'un bi-encoder mais plus lent. Utilise pour reranker apres le retrieval initial.

**Q: Taille de chunk typique pour de la doc technique ?**
R: 500-1000 tokens avec 10-20% d'overlap. Assez pour contenir un bloc de code + son explication, pas trop pour eviter le bruit.

**Q: Qu'est-ce que le Reciprocal Rank Fusion (RRF) ?**
R: Methode de fusion de plusieurs listes de candidats : score = sum(1/(rank+k)). Standard pour hybrid retrieval.

**Q: Quelle metrique pour evaluer le retrieval seul ?**
R: Recall@k (la bonne reponse est-elle dans les top-k ?) et MRR (1/position_du_premier_vrai).

---

## Key takeaways

- RAG = indexation (chunks + embeddings + vector DB) + query (retrieve + rerank + generate).
- Hybrid retrieval (dense + BM25 + RRF) est le standard. Pas d'excuse pour ne faire que dense.
- Reranker cross-encoder = 15-30% de gain gratuit. A toujours tester.
- Le chunking est la decision la plus sous-estimee. Investis du temps dessus.
- Toujours citer les sources. Toujours creer un gold set d'evaluation des le debut.


---

## Pour aller plus loin

Lectures couvrant ce sujet (playlists dans [`shared/external-courses.md`](../../../shared/external-courses.md)) :

- **CMU 11-711 (Neubig) Fa24 — Lec. 10 (Retrieval and RAG)** — fondations academiques du RAG, retrieval dense vs sparse, evaluation.
- **Berkeley CS294-196 Fa24 — Lec. 8 (Compound AI & DSPy — Khattab)** — RAG comme systeme compose, optimisation programmatique des pipelines.
