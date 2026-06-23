# Exercices Medium — RAG Architecture

---

## Exercice 1 : Dimensionner et budgeter un RAG documentaire

### Objectif
Mener un sizing complet d'un pipeline RAG (vector DB, embeddings, latence, cout) et justifier chaque choix d'index.

### Consigne
Tu construis un assistant qui repond sur la documentation interne d'une entreprise.

**Chiffres :**
- Corpus : 2M de pages, ~600 mots/page
- Chunking : 500 tokens/chunk avec 15% d'overlap (1 token ~ 0.75 mot)
- Embedding model : 1024 dimensions, float32 (4 bytes/dim)
- Trafic : 200K requetes/jour, pic x4 a l'heure de pointe
- Pipeline par requete : query rewrite -> hybrid search (dense + BM25) -> rerank top-50 -> generation
- SLA : latence p95 < 2.5s

**Questions :**
1. Combien de chunks au total apres chunking (avec overlap) ? Combien de vecteurs a indexer ?
2. Quelle taille brute pour l'index dense (vecteurs seuls) ? Et avec l'overhead d'un index HNSW (~1.5x) ?
3. Quel QPS moyen et quel QPS pic faut-il tenir ?
4. Decompose le budget de latence p95 entre les etapes (rewrite, retrieval, rerank, generation). Quelle etape domine ?
5. HNSW ou IVF pour cet index ? Justifie avec le volume et le SLA.
6. Estime le cout mensuel d'embedding pour la re-indexation hebdomadaire complete (prix : $0.02 / 1M tokens d'embedding).

### Criteres de reussite
- [ ] Le nombre de chunks tient compte de l'overlap (corpus utile gonfle de ~15%)
- [ ] La taille de l'index dense est coherente (de l'ordre de la dizaine de Go)
- [ ] Le QPS pic est calcule a partir du facteur de pointe (et non du QPS moyen sur 86400s)
- [ ] Le budget de latence montre que la generation LLM domine (et non le retrieval)
- [ ] HNSW est choisi (volume < 100M, SLA serre) avec une justification du recall vs latence
- [ ] Le cout d'embedding hebdo est calcule sur le total de tokens du corpus

---

## Exercice 2 : Diagnostiquer un RAG qui repond a cote

### Objectif
Lire des metriques de retrieval et de generation, isoler la cause racine, et prioriser les corrections.

### Consigne
Ton RAG est en prod depuis 1 mois. Les utilisateurs se plaignent : "il invente" et "il ne trouve pas ce que je cherche". Voici l'eval sur ton gold set (100 paires question/reponse) :

**Metriques actuelles :**
- Recall@5 (dense seul) : 62%
- Recall@5 (avec hybrid + RRF) : 71%
- Recall@5 (avec hybrid + reranker) : non mesure
- Faithfulness (LLM-as-a-judge) : 74%
- Context precision : 38% (sur 5 chunks retournes, ~2 sont utiles)
- Taille de chunk actuelle : 1500 tokens, 0% d'overlap
- Top-k passe au LLM : 5
- Les questions echouees contiennent souvent des codes produits exacts (ex: "REF-4471-B")

**Questions :**
1. Le RAG a deux etages (retrieval, generation). Lequel est le maillon faible ici ? Justifie avec les chiffres.
2. La faithfulness a 74% : est-ce un probleme de generation ou la consequence d'un retrieval faible ? Explique le lien.
3. Pourquoi les codes produits exacts echouent-ils ? Quelle brique manque ou est mal calibree ?
4. Le chunk de 1500 tokens sans overlap : quel risque cree-t-il sur la context precision ? Propose une nouvelle config.
5. Propose 3 actions ordonnees par rapport impact/cout, avec le gain de recall@5 attendu pour chacune.
6. Quel est le risque de monter le top-k a 20 sans reranker ? Et avec reranker ?

### Criteres de reussite
- [ ] Le retrieval est identifie comme le maillon faible (recall 71%, context precision 38%)
- [ ] Le lien est etabli : un mauvais retrieval plafonne la faithfulness (on ne peut pas etre fidele a un mauvais contexte)
- [ ] Les codes produits echouent car le dense rate les matches exacts -> BM25 sous-pondere ou reranker absent
- [ ] La nouvelle config reduit la taille de chunk (~500 tokens) et ajoute de l'overlap (10-20%)
- [ ] Le reranker cross-encoder est l'action #1 (gros gain, cout modere) ; le re-chunking #2
- [ ] Le risque du top-k=20 sans reranker (bruit -> baisse de faithfulness) est explicite, le reranker le neutralise

---

## Exercice 3 : Choisir entre GraphRAG, Agentic RAG et hybrid classique

### Objectif
Mapper trois profils de besoin a la bonne architecture RAG avancee, en argumentant les tradeoffs de cout et de latence.

### Consigne
Tu conseilles 3 equipes, chacune avec un corpus et un type de question different :

| Equipe | Corpus | Type de question dominant |
|---|---|---|
| **A — Legal** | 80K contrats, clauses denses, jargon, references croisees | "Quelles clauses de non-concurrence apparaissent dans les contrats signes avec des filiales de X ?" (multi-hop, entites) |
| **B — Support** | 100K articles FAQ stables | "Comment reinitialiser mon mot de passe ?" (factuel, direct) |
| **C — Recherche** | Web ouvert + corpus interne | "Quel est l'etat de l'art 2026 sur les batteries solides ?" (ouvert, exploratoire, sources a verifier) |

**Questions :**
1. Pour chaque equipe, recommande une architecture (hybrid classique, GraphRAG, agentic RAG, ColBERT) et justifie en 3 phrases.
2. Pour l'equipe A : pourquoi un dense retrieval plat echoue-t-il sur la question multi-hop ? Qu'apporte le graphe ?
3. Pour l'equipe B : pourquoi serait-ce une erreur de partir sur de l'agentic RAG ? Chiffre le surcout.
4. Pour l'equipe C : quel pattern agentic (Self-RAG, CRAG) et pourquoi ? Comment gerer les sources non fiables ?
5. Pour chaque equipe, donne le principal risque de ton choix et sa mitigation.
6. Une equipe veut "le meilleur RAG possible" sans contrainte de budget. Est-ce une bonne approche ? Que reponds-tu ?

### Criteres de reussite
- [ ] A -> GraphRAG (ou ColBERT en complement), B -> hybrid classique simple, C -> agentic RAG (CRAG/Self-RAG)
- [ ] Le multi-hop est explique : l'info est eclatee sur plusieurs docs, le dense plat ne la reconstitue pas
- [ ] Le surcout de l'agentic RAG est chiffre (2-4x d'appels LLM par requete) -> injustifiable pour une FAQ
- [ ] CRAG est propose pour C avec un evaluateur qui declenche une recherche web sur les cas ambigus
- [ ] Chaque choix a un risque concret (cout d'indexation GraphRAG, latence agentic, stockage ColBERT)
- [ ] La reponse a "le meilleur RAG sans budget" rappelle que la simplicite qui marche bat la sophistication inutile (gold set d'abord)
