# Exercices Medium — RAG agentique (J8)

---

## Exercice 1 : Corrective RAG (CRAG) avec triage et re-retrieve

### Objectif
Implementer le coeur du pattern **Corrective RAG** : apres un retrieve, grader globalement la qualite du retrieval (`CORRECT` / `AMBIGUOUS` / `INCORRECT`), et declencher une reformulation + re-retrieve **uniquement** quand le retrieval est juge `INCORRECT`. C'est plus fin qu'un retry aveugle : on ne paye le cout d'une seconde recherche que si la premiere a echoue.

### Consigne
En t'appuyant sur le retriever et le MockLLM auto-contenus fournis dans la solution :

1. Ecris une fonction `grade_retrieval(query, chunks) -> str` qui retourne un verdict de triage a 3 niveaux sur **l'ensemble** des chunks retournes :
   - `CORRECT` : au moins un chunk a un overlap fort avec la query (>= 3 mots de contenu en commun) **et** repond au type de question (si la query demande un chiffre, au moins un chunk pertinent contient un chiffre).
   - `INCORRECT` : aucun chunk n'a un overlap minimal (>= 2 mots) avec la query.
   - `AMBIGUOUS` : tous les autres cas.
2. Ecris `corrective_rag(query, retriever, llm, max_corrections=2) -> dict` qui :
   - Retrieve un top-k, grade le retrieval.
   - Si `CORRECT` → garde les chunks et synthetise.
   - Si `INCORRECT` → reformule la query (task `reformulate` du MockLLM) et re-retrieve, en bornant le nombre de corrections (`max_corrections`).
   - Si `AMBIGUOUS` → garde les chunks mais marque la reponse comme `confidence="low"`.
   - Retourne un dict avec `answer`, `verdict_history` (la liste des verdicts par tour), `corrections` (nombre de re-retrieves declenches) et `confidence`.
3. Construis un scenario ou la **query initiale echoue** (verdict `INCORRECT`, ex : une query en francais avec des termes que le corpus contient en anglais) puis se **corrige** apres reformulation (verdict `CORRECT`).
4. Prouve par assertions : `verdict_history` commence par `INCORRECT`, finit par `CORRECT`, `corrections >= 1`, et la reponse finale est grounded (contient une info issue d'un chunk garde).

### Criteres de reussite
- [ ] `grade_retrieval` retourne bien les 3 verdicts distincts selon les cas
- [ ] Une query `CORRECT` du premier coup ne declenche **aucune** correction
- [ ] Une query `INCORRECT` declenche une reformulation + re-retrieve
- [ ] Le nombre de corrections est borne par `max_corrections` (pas de boucle infinie)
- [ ] Le scenario de demonstration passe de `INCORRECT` a `CORRECT` apres reformulation
- [ ] La reponse finale d'un cas `AMBIGUOUS` porte `confidence="low"`

---

## Exercice 2 : Reranking deux etages (similarite puis signal deterministe)

### Objectif
Comprendre pourquoi la **similarite seule ne suffit pas** et qu'un second etage de rerank change l'ordre final. Tu retrieves un top-k par similarite (etage 1), puis tu **reranks** ce candidat-set par un second signal deterministe (overlap de mots-cles exacts + recence), et tu prouves que l'ordre final differe de l'ordre par similarite brute.

### Consigne
En t'appuyant sur le retriever a embeddings mock fourni (chaque doc a un champ `year` de metadata) :

1. Recupere un candidat-set large : `top_k=5` documents par similarite cosine (etage 1, `retriever.search`). Conserve leur ordre par score de similarite.
2. Ecris `rerank(query, candidates, alpha=0.5) -> list` qui calcule pour chaque candidat un **score de rerank** combinant deux signaux normalises :
   - `keyword_overlap` : nombre de mots de contenu exacts partages entre la query et le doc (normalise sur le max des candidats).
   - `recency` : score de recence base sur `year` (le plus recent = 1.0, le plus ancien = 0.0, interpolation lineaire).
   - `rerank_score = alpha * keyword_overlap + (1 - alpha) * recency`.
   - Retourne les candidats **tries par `rerank_score` decroissant**.
3. Construis une query et un corpus tels que **l'ordre apres rerank differe de l'ordre par similarite** (ex : un doc tres pertinent en mots-cles mais legerement moins bien classe en similarite, ou un doc recent qui remonte).
4. Prouve par assertions :
   - L'ordre des `doc_id` apres rerank n'est **pas egal** a l'ordre par similarite brute.
   - Le top-1 apres rerank a bien le `rerank_score` maximal.
   - Faire varier `alpha` (0.0 = recence pure, 1.0 = mots-cles purs) change l'ordre du top (le top a `alpha=0.0` differe du top a `alpha=1.0`).

### Criteres de reussite
- [ ] L'etage 1 retourne un candidat-set ordonne par similarite
- [ ] `rerank` combine `keyword_overlap` et `recency` avec un poids `alpha`
- [ ] L'ordre final apres rerank differe de l'ordre par similarite brute (assertion)
- [ ] Le top-1 apres rerank possede le `rerank_score` maximal
- [ ] `alpha=0.0` et `alpha=1.0` produisent un top different (les deux signaux pesent reellement)
- [ ] Tout est deterministe et tourne offline

---

## Exercice 3 : Reponse grounded avec citations verifiees

### Objectif
Forcer une reponse **citation-grounded** : chaque phrase de la reponse pointe explicitement vers un `doc_id` retrouve, et un **checker** verifie que chaque citation est valide (le doc cite existe dans le contexte retrouve) et que la phrase est reellement supportee par ce doc. C'est la defense de fond contre l'hallucination "plausible mais fausse".

### Consigne
En t'appuyant sur le MockLLM et le retriever fournis :

1. Ecris un synthesizer `answer_with_citations(query, retrieved) -> list` qui retourne une liste de phrases, chacune sous la forme `{"sentence": str, "cites": [doc_id, ...]}`. Le MockLLM mock fourni produit une phrase par chunk garde, en y attachant l'`id` du chunk.
2. Ecris un checker `check_groundedness(claims, retrieved) -> dict` qui verifie, pour chaque phrase :
   - **Citation valide** : chaque `doc_id` cite existe bien dans `retrieved` (sinon → hallucination de citation).
   - **Support reel** : la phrase partage au moins 2 mots de contenu avec le texte du doc cite (sinon → claim non supporte).
   - Retourne `{"grounded": bool, "violations": [...]}` ou `violations` liste les phrases problematiques avec leur motif (`"unknown_doc"` ou `"unsupported"`).
3. Construis un cas **nominal** : toutes les phrases citent un doc existant et reellement support → `grounded=True`, `violations=[]`.
4. Construis un cas **adversarial** : injecte manuellement dans la liste de claims (a) une phrase qui cite un `doc_id` inexistant, et (b) une phrase qui cite un doc reel mais dont le contenu ne supporte pas la phrase. Le checker doit signaler **exactement** ces 2 violations avec le bon motif.

### Criteres de reussite
- [ ] `answer_with_citations` produit des phrases avec leur liste de `doc_id` cites
- [ ] Le checker detecte une citation vers un `doc_id` inexistant (`unknown_doc`)
- [ ] Le checker detecte une phrase non supportee par le doc cite (`unsupported`)
- [ ] Le cas nominal renvoie `grounded=True` avec zero violation
- [ ] Le cas adversarial renvoie exactement les 2 violations attendues avec le bon motif
- [ ] Tout est deterministe et tourne offline
