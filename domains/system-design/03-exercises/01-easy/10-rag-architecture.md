# Exercices Easy — RAG Architecture

---

## Exercice 1 : Choisir sa strategie de chunking

### Objectif
Matcher un type de contenu a une strategie de chunking adaptee.

### Consigne
Pour chacun des corpus suivants, indique :
- La **strategie de chunking** (fixed, recursive, semantic, document-aware)
- La **taille** approximative (en tokens)
- L'**overlap** (en %)
- Une justification en 2 phrases

Corpus :
1. 10 000 pages de documentation d'un produit SaaS, en Markdown avec headers H1-H3
2. 50 000 transcripts d'appels support client (texte brut, pas de structure)
3. Un repo de code Python de 500 fichiers
4. 2 000 contrats juridiques PDF de 20 a 100 pages chacun
5. Un blog de 800 articles en HTML (articles de 500 a 3000 mots)
6. 30 000 tweets / short-form social posts

### Livrables
Un tableau : corpus / strategie / taille / overlap / justification.

### Criteres de reussite
- [ ] La doc Markdown utilise document-aware (split par header)
- [ ] Le code utilise un chunking par fonction / classe (200-500 tokens)
- [ ] Les contrats utilisent des chunks plus grands (800-1500 tokens, contexte critique)
- [ ] Les tweets : 1 tweet = 1 chunk (pas de split)
- [ ] Au moins une justification cite le tradeoff precision vs contexte

---

## Exercice 2 : Debugger un RAG qui marche mal

### Objectif
Diagnostic structure d'un RAG qui donne de mauvais resultats.

### Consigne
On te donne un RAG deploye en production avec ces symptomes :
- Les utilisateurs se plaignent que les reponses sont incorrectes 40% du temps
- Recall@10 mesure sur un gold set = 72%
- Les reponses citent souvent des chunks qui contiennent les bons mots-cles mais dans un mauvais contexte
- La latence p99 est de 2 s (acceptable)
- Le modele utilise est GPT-4, embedder = text-embedding-3-small, chunks = 512 tokens fixed, pas de reranker, dense retrieval uniquement

**Travail attendu :**
1. Liste les hypotheses de root causes (minimum 5)
2. Pour chaque hypothese, indique l'experimentation que tu ferais pour la verifier
3. Par ou tu commences (priorisation) et pourquoi
4. Quelles metriques tu veux mesurer apres chaque fix

### Livrables
Un doc structure : hypotheses / experiences / priorisation / metriques.

### Criteres de reussite
- [ ] L'absence de reranker est identifiee comme un fix facile et a fort impact
- [ ] L'absence de hybrid search (BM25) est identifiee
- [ ] Le chunking fixed-size est remis en question (recursive ou semantic)
- [ ] Le prompt de generation (faithfulness instructions) est mentionne
- [ ] La priorisation classe les quick wins d'abord (reranker, hybrid) avant les changements plus lourds (re-chunker, changer d'embedder)
- [ ] Recall@10 et faithfulness sont mentionnees comme metriques de validation

---

## Exercice 3 : Estimer une infra RAG

### Objectif
Dimensionner un systeme RAG pour une charge donnee.

### Consigne
Tu deploies un RAG pour un produit B2B de recherche juridique.

**Donnees :**
- 500 000 documents juridiques, moyenne 15 pages chacun
- 1 page = ~500 tokens
- Chunks de 800 tokens avec 15% overlap
- Embedder : text-embedding-3-large, dimension 3072
- 50 users concurrents en peak, chacun fait 5 queries/minute
- Gold set disponible, SLA : p99 latency < 4 s, recall@10 > 85%
- Budget cible : < $2000 / mois

**Questions :**
1. Combien de chunks au total apres indexation ?
2. Taille de l'index vector (memoire) ?
3. Quelle DB vectorielle choisis-tu (Pinecone / Qdrant / pgvector) et pourquoi ?
4. Cout approximatif de l'indexation initiale (embeddings) ?
5. Cout runtime mensuel (queries + LLM generation) ?
6. Quelles optimisations envisages-tu si tu depasses les $2000 / mois ?

Prix approximatifs a utiliser :
- text-embedding-3-large : $0.13 / 1M tokens
- GPT-4o-mini (generation) : $0.15 input + $0.60 output par 1M tokens
- Qdrant Cloud : ~$50/mois par Go de vecteurs
- Pinecone : ~$70/mois pour 1M vecteurs standard

### Livrables
Calculs detailles avec hypotheses explicites.

### Criteres de reussite
- [ ] Nombre de chunks dans l'ordre de ~4-5M (500K docs * 15 pages * ~600 tokens / 800 tokens * 1.15)
- [ ] Taille vector index : ~50-60 Go (chunks * 3072 * 4 bytes)
- [ ] Une DB cible est recommandee avec justification (Qdrant self-hosted ou Pinecone selon budget)
- [ ] Cout d'indexation unique : quelques centaines de dollars
- [ ] Cout runtime : estimation argumentee, incluant LLM generation
- [ ] Au moins 3 optimisations citees (embedding model moins cher, quantization des vecteurs, caching, moins de chunks top-k, smaller LLM)
