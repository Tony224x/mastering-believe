"""
Solutions -- Jour 10 : RAG Architecture
"""


def solution_exercice_1() -> None:
    """
    Exercice 1 -- Strategies de chunking.

    +--------------------------+-------------------+---------+---------+
    | Corpus                   | Strategie         | Taille  | Overlap |
    +--------------------------+-------------------+---------+---------+
    | Doc SaaS Markdown        | document-aware    | 400-800 | 10 %    |
    | (split par H1/H2/H3)     |                   | tokens  |         |
    +--------------------------+-------------------+---------+---------+
    | Transcripts support      | recursive         | 300-500 | 15 %    |
    +--------------------------+-------------------+---------+---------+
    | Code Python              | par fonction /    | 200-500 | 0 %     |
    |                          | classe (tree-sit) |         |         |
    +--------------------------+-------------------+---------+---------+
    | Contrats juridiques      | document-aware    | 1000-   | 20 %    |
    |                          | (par section /    | 1500    |         |
    |                          | clause)           | tokens  |         |
    +--------------------------+-------------------+---------+---------+
    | Blog HTML (500-3000 mots)| recursive + tag-  | 400-600 | 10 %    |
    |                          | aware (h2, p)     |         |         |
    +--------------------------+-------------------+---------+---------+
    | Tweets / short-form      | 1 post = 1 chunk  | ~50     | 0       |
    +--------------------------+-------------------+---------+---------+

    Justifications (extraits) :
    - Markdown avec headers : la structure semantique est deja la. Il
      serait aberrant de la couper en ignorant les H2/H3. On gagne la
      granularite "section".
    - Code : on chunk par fonction car une fonction est un bloc logique
      complet. Un modele comprend mieux une fonction entiere qu'un bout
      au milieu.
    - Contrats : un chunk coupe au mauvais endroit peut omettre une
      clause determinante. On prefere plus de contexte et plus d'overlap.
    - Tweets : la taille naturelle est < 300 chars. Pas de decoupe.

    Tradeoff rappel : plus petit = plus precis mais risque de perdre du
    contexte. Plus grand = plus de contexte mais plus de bruit et queries
    plus couteuses au LLM.
    """


def solution_exercice_2() -> None:
    """
    Exercice 2 -- Debug d'un RAG.

    Symptomes : reponses incorrectes 40%, recall@10=72%, chunks avec bons
    mots-cles mais mauvais contexte. GPT-4, fixed 512 chunks, dense-only,
    pas de reranker.

    Hypotheses de root causes + experiences :

    H1. **Absence de reranker** (le plus probable, quick win)
        - Exp : ajouter un cross-encoder (BGE-reranker ou Cohere Rerank)
          sur le top 20 des candidats dense, ne prendre que le top 5.
        - Attendu : recall@5 apres rerank > 90%, faithfulness monte.
        - Cout : ~50 ms de latence en plus, c'est acceptable.
        - PRIORITE 1 : 1 ligne de code, resultat immediat.

    H2. **Absence de hybrid retrieval (BM25 manquant)**
        - Exp : indexer un BM25 en parallele et faire RRF.
        - Attendu : recall@10 passe de 72% a ~85%, surtout sur les
          queries avec des noms propres ou des identifiants.
        - PRIORITE 2 : effort moyen, gain 10-20%.

    H3. **Chunking fixed-size casse le contexte**
        - Exp : passer a recursive chunking (separateurs paragraphes /
          phrases) ou semantic chunking. Tester avec la meme taille.
        - Attendu : les chunks contiennent des unites de sens completes,
          faithfulness augmente.
        - PRIORITE 3 : demande de reindexer (quelques heures-jours).

    H4. **Prompt de generation faible**
        - Exp : ajouter des instructions explicites "utilise UNIQUEMENT
          les passages fournis, si l'info n'est pas la dis 'je ne sais
          pas'", forcer les citations [N].
        - Attendu : faithfulness monte (moins d'hallucinations).
        - PRIORITE 2 : trivial, tester en priorite.

    H5. **Embedding model trop faible pour le domaine**
        - text-embedding-3-small fait 1536 dims et est general-purpose.
          Sur un domaine specialise (medical, legal, finance), des
          modeles domaine-specifiques (BGE, Voyage, Cohere) performent
          mieux.
        - Exp : remplacer l'embedder sur un sous-ensemble du corpus et
          comparer recall@10.
        - PRIORITE 4 : tres impactant mais implique de reindexer tout.

    H6. **Pas de query rewriting / expansion**
        - Les queries utilisateurs sont souvent courtes et ambigues. Un
          LLM peut les reformuler avant retrieval.
        - Exp : HyDE (generer une reponse hypothetique et embedder
          celle-ci) ou query expansion classique.
        - PRIORITE 3 : gain 5-15%.

    Priorisation : H1 -> H4 -> H2 -> H3 -> H6 -> H5.
    (Quick wins d'abord, changements structuraux ensuite.)

    Metriques a mesurer apres chaque fix :
    - Recall@10 (retrieval pur) sur le gold set
    - MRR
    - Faithfulness (LLM-as-a-judge)
    - Answer relevance
    - p99 latency (verifier qu'on ne degrade pas)
    - Cost per query

    Regle : un seul changement a la fois. Sinon impossible d'attribuer
    les variations de metriques.
    """


def solution_exercice_3() -> None:
    """
    Exercice 3 -- Dimensionnement infra RAG juridique.

    Hypotheses :
      500K docs * 15 pages * 500 tokens/page = 3.75 B tokens
      Chunks de 800 tokens, 15% overlap = taille effective ~680 tokens
      => 3.75B / 680 = ~5.5 M chunks

    Q1 -- Chunks : ~5.5 M

    Q2 -- Memoire index vecteur :
      5.5M * 3072 dims * 4 bytes (float32) = 67.6 Go
      Avec HNSW overhead (~30%) : ~87 Go.
      Avec quantization int8 : 67.6/4 = 17 Go -- option serieuse.
      Avec quantization binaire : ~2 Go -- mais plus bruite.

    Q3 -- Choix de DB :
      - pgvector : trop lent / memoire pour 5.5M * 3072 dims. Non.
      - Pinecone : ~$70 * 5.5 = $385/mois minimum. Simple, zero ops.
      - Qdrant Cloud : facturation memoire, ~$200-400/mois pour 17 Go
        quantise. Bon compromis.
      - Qdrant self-hosted : une VM 32 Go RAM ~$150/mois. Le moins cher.
      Recommandation : **Qdrant self-hosted** si on a une equipe DevOps,
      sinon Qdrant Cloud. Pinecone si on veut zero friction.

    Q4 -- Cout d'indexation initiale :
      Tokens a embedder : 3.75B input tokens.
      text-embedding-3-large : $0.13 / 1M tokens
      => 3750 * 0.13 = **$487.5** (one-shot, pas recurrent).
      Astuce : si on prend text-embedding-3-small ($0.02 / 1M), c'est
      $75. Test d'abord la qualite small avant de payer large.

    Q5 -- Cout runtime mensuel :
      Queries : 50 users * 5 q/min * 60 min * 8 h peak * 22 jours
             = 2 640 000 queries / mois.
      (hypothese aggressive -- en pratique beaucoup moins au peak reel)

      Par query :
        - Embed query : ~50 tokens -> 50 * $0.13/1M = $0.0000065
        - Retrieve : couvert par l'abonnement DB
        - LLM generation : ~3000 tokens input (5 chunks * 600 tokens
          context) + 400 tokens output
          GPT-4o-mini : 3000*0.15/1M + 400*0.60/1M = $0.00045 + $0.00024
                     = ~$0.00069 / query
      Total par query : ~$0.00070
      Mensuel : 2.64M * 0.00070 = **~$1848 / mois**.
      On est presque au budget ! Et ce scenario est pessimiste.

      Cout DB : $200 (Qdrant Cloud) ou $150 (self-hosted).
      => Total : ~$2000 - $2100 / mois. On est TRES serre.

    Q6 -- Optimisations pour rester sous $2000 :
      1. **Passer a text-embedding-3-small** : qualite souvent suffisante,
         cout x6 inferieur a l'indexation, dimension 1536 donc index x2
         plus petit.
      2. **Quantization des vecteurs** (int8 dans Qdrant) : memoire /4,
         cout DB baisse.
      3. **Semantic caching** sur les queries frequentes : 20-40% des
         queries sont des variations d'une meme question -> cache hit
         evite un appel LLM.
      4. **Reduire le top-k** passe au LLM : de 5 a 3 chunks -> input
         tokens baisse de 40%.
      5. **Reranker** pour gagner sur le top-k : 3 bons chunks valent
         mieux que 5 moyens.
      6. **Distillation du LLM** vers un modele plus petit (Haiku,
         Llama-3-8B en self-hosted) pour le use case specifique.
      7. **Batching des requetes** si le SLA le permet.
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)
