# Jour 17 — RAG 2026 : retrieval augmente au-dela des vector DBs

> **Temps estime** : 5h | **Prerequis** : J3 (embeddings), J15 (reasoning), J16 (agents)

---

## 1. Pourquoi RAG n'est pas mort en 2026

En 2024-2025, certains ont predit la mort de RAG avec l'arrivee des contextes 1M-10M tokens (Gemini 1.5 Pro, GPT-5). **C'est faux**. Raisons :

1. **Cout** : meme avec prompt caching (J18), injecter 1M tokens a chaque query coute 5-50x plus qu'un retrieval + 10k tokens ciblés.
2. **Latence** : TTFT (time-to-first-token) proportionnel au contexte. 1M tokens = 10-60s juste pour commencer a repondre.
3. **Lost-in-the-middle** : les LLMs utilisent moins bien l'information au milieu de tres longs contextes. Passer de 10M → top 10 pertinents = meilleure precision.
4. **Donnees > context window** : la plupart des corpus d'entreprise font 100GB-10TB. Aucun contexte ne les contient.
5. **Frais de securite** : injecter tout le corpus expose tout. RAG avec permissions par doc = acces-controle natif.

**Ce qui a change** : RAG n'est plus "embed + cosine similarity". C'est un **stack hybride** combinant plusieurs retrievers, un reranker, des metadatas, et parfois un agent. Le RAG "chunking + FAISS + top-k" est devenu le baseline minimal, pas le SOTA.

---

## 2. L'architecture RAG de reference 2026

```
Query
  │
  ├──→ [Query rewriting / expansion] (LLM : decompose, synonymes, HyDE)
  │
  ├──→ [Hybrid retrieval]
  │         ├── BM25 (lexical, mots exacts)
  │         ├── Dense embeddings (semantique)
  │         └── ColBERT / late interaction (token-level matching)
  │
  ├──→ [Fusion] (RRF — Reciprocal Rank Fusion)
  │
  ├──→ [Reranker] (cross-encoder : LLM ou modele dedie)
  │
  ├──→ [Context assembly] (enrichissement, citations, permissions)
  │
  └──→ [LLM generation] (avec citations et refus si insuffisant)
```

Chaque etape est optionnelle, mais **au moins hybrid + reranker est standard** en 2026.

---

## 3. Pourquoi "juste des embeddings" ne suffit pas

### Les 3 limites des dense embeddings seuls

**Limite 1 — Matching lexical faible** : "AAPL" et "Apple Inc." ont des embeddings proches, mais une query sur "error code NS-101" ne matchera PAS le doc contenant exactement "NS-101" si celui-ci n'apparait que dans une ligne d'exception. BM25 (tf-idf ameliore) le matche parfaitement.

**Limite 2 — Un seul vecteur par passage** : un chunk de 500 tokens est compresse en 1 vecteur de ~1024 dim. Si ton chunk contient 20 faits independants, le vecteur est une moyenne — la query sur un fait specifique match moyennement.

**Limite 3 — Embeddings out-of-domain** : le modele d'embedding a ete entraine sur du web generaliste. Pour un corpus juridique, medical, ou technique specifique, la performance chute de 20-40%.

### La solution 2026 : hybrid + late interaction

**Hybrid retrieval** = combiner BM25 (lexical) + dense (semantique) avec **Reciprocal Rank Fusion** :

```python
# RRF formula — sans parametre a tuner, c'est sa force
rrf_score(doc) = sum over retrievers of  1 / (k + rank(doc, retriever))
# k = 60 generalement
```

**Late interaction (ColBERT-v2 / ColPali)** : au lieu d'un vecteur par passage, **un vecteur par token**. La similarity est calculee comme "pour chaque token de la query, quelle est la meilleure matche dans les tokens du doc ?" (MaxSim). C'est 10-100x plus stockage, mais drastiquement meilleur sur les queries complexes ou factuelles.

### Contextual Retrieval (Anthropic, sept 2024)

Un truc simple et redoutable : **enrichir chaque chunk avec son contexte doc avant d'embedder**. Un chunk isole "Le CA a augmente de 15%" devient "Dans le rapport annuel 2023 d'ACME, chapitre Europe : le CA a augmente de 15%". Gain typique : -35% de chunks rates. Cout : un appel LLM par chunk au moment de l'indexation (cacheable via prompt caching).

---

## 4. Le reranker : l'etape la plus rentable

Un retriever rapide retourne un top-100 bruite. Un **reranker** cross-encoder re-score chaque paire (query, doc) en passant les deux dans un LLM specialise. 100x plus lent par paire, mais sur 100 paires, c'est un budget raisonnable et l'ordre devient drastiquement meilleur.

### Les trois classes de reranker en 2026

| Type | Exemples | Latence / 100 docs | Qualite |
|---|---|---|---|
| Cross-encoder specialise | Cohere rerank-3.5, Jina rerank-v2, bge-reranker-v3 | 100-500ms | Excellent |
| LLM-as-reranker | Haiku ou small LLM avec prompt de scoring | 1-3s | Tres bon, flexible |
| Reasoning-as-reranker | Reasoning model qui justifie le score | 10-30s | Best-in-class, cher |

Regle : **toujours** commencer par un cross-encoder dedie. Passer a LLM-as-reranker uniquement si tu as besoin de prise en compte de metadata complexe (permissions, date, source).

### Gain typique

Sur un benchmark RAG type BEIR :
- Dense seul : nDCG@10 = 0.42
- Hybrid (+BM25 + RRF) : 0.50
- Hybrid + reranker : **0.58**
- Hybrid + reranker + contextual retrieval : **0.62**

Chaque etape coute 5-20% de latence et 10-30% de cout, mais le ROI qualite est massif.

---

## 5. Agentic RAG — quand le retrieval devient une boucle

Le RAG "one-shot" retrouve, concatene, genere. **Agentic RAG** (standard 2025+) transforme le retrieval en outil :

```
User query
  │
  ▼
[Agent (reasoning model)]
  │
  ├─ tool: search(query) → top-k docs
  ├─ tool: search(reformulated_query)
  ├─ tool: read_full_doc(doc_id)
  ├─ tool: graph_lookup(entity)
  │
  ▼
Reponse avec citations
```

L'agent decide :
- Faut-il vraiment retrouver, ou ai-je deja la reponse ?
- Si le top-k n'est pas satisfaisant, re-query avec d'autres mots
- Si un doc contient la reponse complete, la lire en entier plutot que le chunk
- Agreger des infos de plusieurs sources

**Quand utiliser agentic RAG** :
- Queries ambigues necessitant du rafinement
- Questions compositionnelles ("compare X et Y sur la metrique Z")
- Corpus heterogenes (docs + DB + API + web)

**Cout** : 5-10x un RAG simple. Utiliser seulement si le RAG simple plafonne.

---

## 6. GraphRAG — structurer le corpus

Sorti par Microsoft en 2024, popularise en 2025. Idee : plutot que de chunker naivement, construire un **graphe de knowledge** a partir du corpus (entites, relations, communautes), puis retrouver via le graphe + texte.

```
Indexation :
  1. Extraire entites de chaque chunk (LLM)
  2. Extraire relations entre entites (LLM)
  3. Clusterer les entites en communautes (Leiden algorithm)
  4. Resumer chaque communaute (LLM)

Retrieval :
  - Query → entites mentionnees
  - Remonter les communautes associees → resume
  - + chunks associes
```

**Force** : queries qui demandent une synthese cross-documents ("quelles sont les principales tendances dans ces 10 000 rapports ?") — le resume des communautes est precalcule.

**Faiblesse** : cout d'indexation x10 vs RAG classique (beaucoup d'appels LLM). A reserver aux corpus statiques ou l'indexation se fait rarement.

**Variante simplifiee en prod** : `LightRAG` (2024) — les entites et relations sont des nodes, pas besoin de clustering Leiden. Indexation 5x moins chere.

---

## 7. Citations, groundedness, refus

Un RAG qui hallucine des faits non presents dans les docs **est pire** qu'un LLM sans RAG : l'utilisateur pense que la reponse est sourcee.

### Citations obligatoires (2026 standard)

Chaque assertion factuelle = une reference au chunk source. Implementation :

1. **Chunks avec IDs** dans le contexte injecte
2. **System prompt strict** : "chaque phrase factuelle doit finir par [doc_id:chunk_id]"
3. **Post-processing** : parser les citations, verifier que chaque ID existe, que le chunk supporte l'assertion (avec un LLM juge)
4. **UI** : affichage cliquable des sources

### Groundedness check

Apres generation, un second LLM (Haiku, Gemma) verifie :
> "La reponse A affirme X. Le chunk referencé [doc_id:chunk_id] dit-il X ? Repondre oui/non."

Si non → relancer la generation ou retourner "information insuffisante".

Cost add : 20-50% latence. ROI : -80% hallucinations factuelles. Non-negociable pour les produits serieux (legal, medical, finance).

### Refus controle

Le modele doit repondre "Je ne trouve pas cette information dans les documents fournis" quand c'est le cas. C'est difficile a obtenir en prompting seul. Solutions :
- Fine-tuner un modele sur des paires (contexte insuffisant → refus)
- Ajouter un classifieur prealable "le contexte couvre-t-il la question ?"

---

## 8. Chunking : la guerre n'est pas finie

Le chunking naif (500 tokens overlap 50) reste majoritaire en prod, mais sous-optimal. Alternatives :

- **Semantic chunking** : grouper les phrases par proximite d'embedding. Meilleur sur les textes structures.
- **Late chunking** (Jina, 2024) : embedder le doc entier d'un coup, puis chunker les embeddings. Preserve le contexte long.
- **Hierarchical chunking** : parent-child (doc → section → paragraphe → phrase). Retrieval a plusieurs niveaux.
- **Token-budget chunking** : ajuster la taille des chunks selon le budget du LLM cible (32k context → chunks 1k, 200k context → chunks 8k).

Regle empirique : chunks de 256-512 tokens avec 15% d'overlap = baseline solide. Optimiser seulement si tu mesures un plafond de qualite retrieval.

---

## 9. RAG vs long-context vs agent : la matrice de decision

| Cas d'usage | Strategie 2026 |
|---|---|
| Corpus stable, queries repetees, budget serre | RAG classique + reranker |
| Corpus volumineux dynamique | RAG hybrid + reranker + contextual retrieval |
| Queries analytiques cross-docs | GraphRAG ou agentic RAG |
| Quelques documents massifs (rapports, livres) | Long-context + prompt caching |
| Queries creatives / brainstorm | Long-context, pas de RAG |
| Corpus multimodal (docs scannes, schemas) | ColPali (late interaction sur images) |
| Tres haute precision (medical, legal) | Hybrid + reranker + groundedness check + citations |

---

## 10. Les pieges 2026 specifiques

1. **Chunk-leakage** : un chunk retrouve peut contenir des donnees hors-scope de l'utilisateur. Filtrer par ACL AVANT le retrieval, pas apres.
2. **Embedding drift** : changer de modele d'embedding = reindex total. Prevoir un systeme de dual-index pendant la migration.
3. **Evaluation RAG** : les metriques retrieval (recall@k) sont decorrelees de la qualite finale. Il FAUT evaluer end-to-end avec un LLM juge sur des paires (query, expected_answer).
4. **Cache poisoning** : si tu embeddes les query utilisateurs, un attaquant peut polluer ton index. Toujours embedder les docs, pas les queries, en reference.
5. **Multimodal RAG mis en place trop tot** : avant de faire du ColPali sur les PDFs, verifier qu'OCR + chunk text regle deja 80% des cas. Le multimodal RAG est 10x plus cher.

---

## Key takeaways (flashcards)

**Q1** — Pourquoi hybrid retrieval (BM25 + dense) bat-il dense seul ?
> Dense rate les matches lexicaux exacts (codes erreur, noms propres rares, identifiants). BM25 les trouve. RRF fusionne sans hyperparametre a tuner.

**Q2** — Qu'apporte un reranker cross-encoder par rapport au retrieval seul ?
> Re-score precis de 100 candidats via un modele qui voit query+doc ensemble. Gain typique +10-15 points nDCG@10. Latence 100-500ms, indispensable.

**Q3** — Qu'est-ce que Contextual Retrieval ?
> Avant d'embedder un chunk, l'enrichir avec son contexte doc (via LLM) pour que son embedding soit desambigue. -35% chunks rates (Anthropic, 2024).

**Q4** — Quand preferer GraphRAG a un RAG classique ?
> Queries qui demandent une synthese cross-documents sur de larges corpus. Indexation 10x plus chere mais retrieval de resumes de communautes pre-calcules.

**Q5** — Pourquoi le long-context n'a pas tue RAG ?
> Cout (5-50x), latence TTFT, lost-in-the-middle, corpus > context, permissions par doc. Long-context complete RAG, ne le remplace pas.

**Q6** — Quelle est la difference entre retrieval metrics et RAG end-to-end metrics ?
> recall@k mesure si le bon chunk est retrouve. Une reponse finale peut etre fausse malgre un bon retrieval (mauvaise synthese). Il faut les deux : metriques retrieval + eval LLM-as-judge de la reponse finale contre un golden set.
