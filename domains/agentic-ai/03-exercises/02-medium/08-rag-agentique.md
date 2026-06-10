# Exercices Medium — RAG Agentique (J8)

---

## Exercice 1 : Boucle de reformulation pilotee par le grading

### Objectif
Implementer le coeur du RAG agentique : quand le retrieval echoue (tous les chunks juges irrelevants), reformuler la requete avec une strategie differente a chaque tentative, au lieu d'abandonner.

### Consigne
En partant de `02-code/08-rag-agentique.py` :

1. Ajoute au `MockLLM` deux strategies de reformulation distinctes (tasks `"reformulate_expand"` et `"reformulate_hyde"`) :
   - `expand` : ajoute des synonymes/termes voisins a la requete ("CA" -> "chiffre d'affaires revenue annual sales")
   - `hyde` : genere une **reponse hypothetique** courte (2 phrases plausibles) qui sert de nouvelle requete de recherche (pattern HyDE)
2. Ecris `retrieve_with_retry(query, retriever, llm, max_attempts=3) -> dict` :
   - Tentative 1 : requete brute
   - Apres chaque retrieval, grade les chunks ; s'il y a au moins 1 chunk RELEVANT, retourne `{"chunks": ..., "attempts": [...], "strategy_used": ...}`
   - Tentative 2 : strategie `expand` ; tentative 3 : strategie `hyde`
   - Chaque tentative est tracee : requete utilisee, nb de chunks, nb de relevants
3. Cree dans le corpus un document que seule la reformulation permet de trouver (vocabulaire different de la question, ex: question avec "CA" et document avec "revenue")
4. Demo : une question qui reussit a la tentative 1, une qui reussit a la tentative 2 (expand), une qui ne reussit qu'en hyde
5. Si les 3 tentatives echouent, retourne un resultat vide avec `strategy_used: "none"` — le synthesizer doit alors repondre "insufficient context" sans halluciner

### Criteres de reussite
- [ ] Les 2 strategies de reformulation produisent des requetes differentes et visibles dans la trace
- [ ] Le cas "vocabulaire different" echoue en brut et reussit apres reformulation
- [ ] La boucle s'arrete des qu'un chunk RELEVANT est trouve (pas de tentative inutile)
- [ ] Le cas "echec total" produit une reponse honnete sans hallucination
- [ ] La trace des tentatives est complete (requete, chunks, verdicts par tentative)

---

## Exercice 2 : Retrieval hybride avec Reciprocal Rank Fusion

### Objectif
Combiner deux strategies de retrieval (TF-IDF semantique-like et matching exact par mots-cles) et fusionner leurs classements avec RRF — la technique standard du retrieval hybride.

### Consigne
1. Implemente un second retriever `KeywordRetriever` :
   - Score = nombre de termes exacts de la query presents dans le chunk (apres tokenization), bonus x2 si un **chiffre ou code exact** de la query est dans le chunk (ex: "EUR 50M", "2023")
2. Implemente `rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]` :
   - Pour chaque chunk_id, score RRF = somme sur les classements de `1 / (k + rang)` (rang commence a 1)
   - Retourne les chunks tries par score RRF decroissant
3. Ecris `hybrid_search(query, tfidf_retriever, keyword_retriever, top_k=3)` qui :
   - Recupere le top-5 de chaque retriever
   - Fusionne avec RRF et retourne le top-3 final avec le detail des scores (`rang TF-IDF`, `rang keyword`, `score RRF`)
4. Construis 2 requetes de demonstration :
   - Une requete "semantique" (paraphrase, peu de mots exacts) ou TF-IDF seul est meilleur
   - Une requete "exacte" (code produit ou chiffre precis) ou keyword seul est meilleur
   - Montre que l'hybride RRF place le bon chunk en position 1 dans LES DEUX cas
5. Affiche un tableau comparatif : position du chunk attendu selon TF-IDF seul / keyword seul / hybride

### Criteres de reussite
- [ ] Le KeywordRetriever score correctement le matching exact et le bonus chiffres
- [ ] L'implementation RRF est conforme a la formule (verifiable sur un mini-cas a la main)
- [ ] La requete semantique et la requete exacte donnent chacune le bon chunk en top-1 en hybride
- [ ] Le tableau comparatif montre au moins un cas ou l'hybride bat chaque retriever seul
- [ ] `rrf_fuse` est une fonction pure testee independamment par un assert

---

## Exercice 3 : Synthese avec citations et verification de fidelite

### Objectif
Forcer le synthesizer a citer ses sources, puis verifier automatiquement que chaque affirmation chiffree est soutenue par le chunk cite — la base de la lutte anti-hallucination en RAG.

### Consigne
1. Modifie la synthese (task `"synthesize_cited"` du MockLLM) pour produire des phrases avec marqueurs de citation : `"Acme generated EUR 50M in 2023 [doc_finance_01]. The team grew to 85 people [doc_hr_02]."`
2. Ecris un verificateur local `check_faithfulness(answer, chunks_by_id) -> dict` :
   - Decoupe la reponse en phrases
   - Pour chaque phrase : extrait la citation `[doc_id]` (phrase sans citation -> violation `"uncited"`)
   - Extrait les **nombres** de la phrase (regex, en ignorant les nombres dans le doc_id) et verifie que chacun apparait dans le chunk cite (nombre absent -> violation `"unsupported_number"`)
   - Citation vers un doc_id inexistant -> violation `"bad_citation"`
   - Retourne `{"faithful": bool, "violations": [...], "coverage": phrases_ok / phrases_total}`
3. Teste avec 3 reponses fabriquees :
   - Une reponse correcte (toutes les phrases citees et soutenues)
   - Une reponse avec un chiffre hallucine (le chunk cite contient un autre chiffre)
   - Une reponse avec une phrase sans citation
4. Integre le verificateur dans le pipeline : si `faithful == False`, le pipeline re-synthetise en ajoutant les violations au prompt (1 retry max), sinon il livre avec le score de coverage
5. Affiche le rapport de verification pour chaque cas

### Criteres de reussite
- [ ] L'extraction des citations et des nombres fonctionne sur les 3 cas
- [ ] Le chiffre hallucine est detecte avec la violation `unsupported_number`
- [ ] La phrase non citee est detectee avec `uncited`
- [ ] Le retry de synthese corrige la reponse fautive (le mock retourne une version corrigee)
- [ ] Le rapport final inclut `coverage` et la liste exacte des violations
