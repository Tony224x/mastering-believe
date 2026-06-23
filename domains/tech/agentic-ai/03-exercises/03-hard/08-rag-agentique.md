# Exercices Hard — RAG agentique (J8)

---

## Exercice 1 : Boucle agentique self-corrective complete

### Objectif
Assembler la boucle complete d'un RAG agentique auto-correctif : `retrieve -> grade -> (reformuler & retry | repondre) -> groundedness check -> (re-retrieve | finir)`, avec une **garde d'iterations** et une **preuve** qu'une query initialement fausse se corrige toute seule jusqu'a une reponse grounded. C'est la synthese des briques 3.3 (grading), 3.4 (adaptive retrieval) et 7.1 (groundedness) de la theorie.

### Consigne
Construis une classe `SelfCorrectiveRAG` (auto-contenue, retriever + MockLLM mock embarques) qui implemente une machine a etats explicite :

1. **State** : un dict/objet portant au minimum `query`, `current_query`, `iterations`, `kept_chunks`, `answer`, `stop_reason`, et un journal `trace` (la liste des etapes franchies, ex : `["retrieve", "grade:INCORRECT", "reformulate", "retrieve", "grade:CORRECT", "answer", "ground:ok"]`).
2. **Noeuds** :
   - `retrieve` : cherche un top-k pour `current_query`.
   - `grade` : verdict de retrieval (`CORRECT` / `INCORRECT`) ; si `INCORRECT`, passer a `reformulate`, sinon a `answer`.
   - `reformulate` : reformule `current_query` (synonymes / FR→EN) et reboucle sur `retrieve`.
   - `answer` : synthetise une reponse a partir des `kept_chunks`.
   - `groundedness` : verifie que la reponse est supportee par au moins un chunk garde. Si **non grounded**, declenche un **re-retrieve** (retour a `reformulate`) ; si grounded, termine.
3. **Garde-fous** : `max_iterations` (ex : 5). Au-dela → `stop_reason="max_iterations"` et terminaison gracieuse (jamais d'exception non geree). Sortie normale → `stop_reason="grounded_answer"`.
4. **Scenario self-correction** : une query initiale qui **echoue** au premier grade (ex : termes francais "chiffre affaires concurrent" alors que le corpus est en anglais "revenue competitor") doit, apres reformulation, **trouver** des chunks pertinents et produire une reponse grounded.
   - Prouve par assertions : le `trace` contient bien un `grade:INCORRECT` **avant** un `grade:CORRECT`, au moins une etape `reformulate`, `stop_reason == "grounded_answer"`, et la reponse cite une info reellement presente dans un chunk garde.
5. **Scenario impossible** : une query dont l'info n'existe **pas** dans le corpus doit s'arreter proprement sur `max_iterations` (ou un `stop_reason="not_found"` explicite), sans boucler a l'infini.

### Criteres de reussite
- [ ] La boucle enchaine retrieve → grade → reformulate → retrieve → answer → groundedness sur le scenario self-correction
- [ ] Le `trace` montre un `grade:INCORRECT` puis un `grade:CORRECT` (preuve de la correction)
- [ ] La reponse finale du scenario self-correction est grounded (`stop_reason="grounded_answer"`)
- [ ] La garde `max_iterations` empeche toute boucle infinie sur le scenario impossible
- [ ] La terminaison est toujours gracieuse (aucune exception non geree)
- [ ] `stop_reason` final est lisible et correct dans les deux scenarios
- [ ] Tout est deterministe et tourne offline (zero dependance)

---

## Exercice 2 : Multi-hop RAG avec chainage de deux retrieves

### Objectif
Implementer un **vrai multi-hop** : une question necessite deux sauts de retrieval ou la reponse du hop 1 alimente la sous-question du hop 2. Tu dois prouver que les deux hops se sont reellement declenches et que la reponse finale combine les deux. C'est la brique 3.5 de la theorie, dans sa version chainee (pas juste une boucle naive).

### Consigne
Construis un `MultiHopRAG` auto-contenu (retriever + MockLLM embarques) :

1. **Decompose en plan multi-hop** : a partir de la question, le MockLLM produit un plan ordonne ou le **second hop reference le resultat du premier** via un placeholder (ex : `hop2 = "who is the CEO of {hop1_result}"`).
2. **Execution chainee** :
   - Hop 1 : retrieve + extraction de l'**entite cle** de la reponse (task `extract_entity` du MockLLM).
   - Hop 2 : substitue l'entite extraite dans la sous-question du hop 2, puis retrieve a nouveau.
   - La reponse finale combine explicitement les deux resultats.
3. **Question test** (necessite 2 hops) : par exemple "Who founded the company that built {product X}, and where is that founder based?"
   - Hop 1 : "who built {product X}" → entite (ex : `Globex`).
   - Hop 2 : "who founded Globex / where is the founder based" → reponse finale.
4. **Observabilite** : expose `hops` (liste des enregistrements par hop : `query`, `kept_chunks`, `entity`), et `final_answer`.
5. Prouve par assertions :
   - **Exactement 2 hops** ont ete executes.
   - L'entite extraite au hop 1 apparait bien dans la sous-question du hop 2 (preuve du chainage, pas deux retrieves independants).
   - La reponse finale **combine** une info du hop 1 ET une info du hop 2.
   - Un **cas de garde** : si le hop 1 n'extrait aucune entite, le hop 2 ne se lance pas et le systeme remonte un signal `"hop1_failed"` propre (pas de crash).

### Criteres de reussite
- [ ] Le plan multi-hop est genere avec un placeholder reference par le hop 2
- [ ] Exactement 2 hops sont executes sur la question test
- [ ] L'entite extraite au hop 1 est bien injectee dans la sous-question du hop 2
- [ ] La reponse finale combine une info du hop 1 et une info du hop 2
- [ ] Le cas de garde (hop 1 sans entite) remonte `"hop1_failed"` sans crash
- [ ] `hops` est inspectable (query + chunks + entite par hop)
- [ ] Tout est deterministe et tourne offline (zero dependance)
