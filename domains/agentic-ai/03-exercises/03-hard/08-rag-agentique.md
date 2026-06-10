# Exercices Hard — RAG Agentique (J8)

---

## Exercice 1 : Corrective RAG (CRAG) de bout en bout

### Objectif
Implementer le pipeline Corrective RAG complet : evaluer la qualite du retrieval, declencher des actions correctives differenciees (raffinement, recherche web de secours), et synthetiser avec attribution de sources et niveau de confiance.

### Consigne
Construis une classe `CorrectiveRAG` avec le flow suivant :

1. **Retrieval evaluator** : pour chaque chunk recupere, score de confiance 0-1 (mock deterministe base sur l'overlap). Trois verdicts globaux :
   - `CORRECT` : au moins 1 chunk > 0.7
   - `AMBIGUOUS` : meilleur chunk entre 0.4 et 0.7
   - `INCORRECT` : tous les chunks < 0.4
2. **Actions correctives** :
   - `CORRECT` -> **knowledge refinement** : decoupe chaque chunk retenu en "knowledge strips" (phrases), re-score chaque strip individuellement, ne garde que les strips pertinents (filtre le bruit intra-chunk)
   - `AMBIGUOUS` -> refinement + **web search de secours** (mock `mock_web_search` avec 3 resultats predefinis) et fusion des deux sources
   - `INCORRECT` -> web search seul, en reformulant la query d'abord
3. **Synthese** avec attribution : chaque element de la reponse indique sa provenance (`[corpus:doc_x]` ou `[web:result_y]`) et la reponse se termine par une ligne `Confidence: high/medium/low` derivee du verdict
4. Demo sur 3 questions calibrees pour declencher chacune un verdict different :
   - Question bien couverte par le corpus -> CORRECT
   - Question partiellement couverte -> AMBIGUOUS (la reponse fusionne corpus + web)
   - Question hors corpus -> INCORRECT (reponse 100% web)
5. Trace complete : verdict, scores par chunk, strips gardes/jetes, sources utilisees
6. Asserts : le cas CORRECT n'appelle pas le web ; le cas INCORRECT n'utilise aucun chunk corpus ; chaque reponse contient au moins une attribution et la ligne Confidence

### Criteres de reussite
- [ ] Les 3 verdicts sont correctement attribues sur les 3 questions de demo
- [ ] Le knowledge refinement filtre reellement des strips non pertinents (visible dans la trace)
- [ ] Le web search n'est declenche que pour AMBIGUOUS et INCORRECT
- [ ] Chaque affirmation de la reponse a une attribution corpus/web correcte
- [ ] Le niveau de confiance est coherent avec le verdict
- [ ] Tous les asserts passent et la trace permet de rejouer le raisonnement

---

## Exercice 2 : RAG multi-index avec routage et boucle d'auto-evaluation

### Objectif
Construire un RAG agentique production : plusieurs corpus specialises, un routeur qui choisit le(s) bon(s) index par sous-question, et une boucle d'auto-evaluation qui detecte les aspects manquants de la reponse et relance des recherches ciblees — le tout sous budget.

### Consigne
1. Construis 3 corpus distincts avec 3-4 docs chacun (contexte Acme) :
   - `products` : specs produits, versions, fonctionnalites
   - `finance` : revenus, marges, effectifs, levees de fonds
   - `hr` : equipes, recrutement, politique remote
   Chaque corpus a son propre `TinyTfIdfRetriever`
2. **Router** (task MockLLM `"route_index"`) : pour chaque sous-question, choisit un ou plusieurs index avec une justification (`{"indexes": ["finance"], "reason": ...}`) — regles deterministes par mots-cles, avec un cas multi-index ("combien d'ingenieurs et quel revenu par ingenieur ?" -> hr + finance)
3. **Pipeline** : decompose la question en sous-questions -> route chaque sous-question -> retrieve sur le(s) bon(s) index -> grade -> reponse partielle par sous-question
4. **Boucle d'auto-evaluation** (task `"self_eval"`) : apres la premiere synthese, le mock evalue la completude en verifiant qu'un ensemble d'aspects attendus est couvert (`{"missing_aspects": [...]}`) :
   - Pour chaque aspect manquant, genere une sous-question ciblee et relance UN cycle retrieve/grade/answer
   - Maximum 2 rounds d'auto-evaluation
5. **Budget** : `llm_budget=15` appels — chaque appel passe par un compteur central ; depassement -> `RuntimeError`. La demo doit terminer SOUS le budget et afficher le decompte final
6. Question de demo : "Give a complete overview of Acme: main product, latest revenue, and team size" — calibre pour que la premiere passe oublie un aspect (team size) et que l'auto-eval le rattrape au round 2
7. Rapport final : sous-questions, index choisis, rounds d'auto-eval, aspects rattrapes, appels LLM consommes

### Criteres de reussite
- [ ] Chaque sous-question est routee vers le bon index (y compris le cas multi-index)
- [ ] Aucune recherche n'est faite sur un index non pertinent (verifiable dans la trace)
- [ ] L'auto-evaluation detecte l'aspect manquant et le round 2 le couvre
- [ ] La reponse finale couvre les 3 aspects demandes
- [ ] Le budget est respecte et le compteur exact (assert)
- [ ] La limite de 2 rounds d'auto-eval est respectee meme si des aspects manquent encore
