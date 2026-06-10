# Exercices Medium — Agent complet (J7, capstone semaine 1)

---

## Exercice 1 : Memoire long-terme avec court-circuit du plan

### Objectif
Rendre l'agent plus econome : si une information demandee a deja ete calculee dans un run precedent, le planner doit court-circuiter les steps de recherche et reutiliser la memoire.

### Consigne
En partant de `02-code/07-agent-complet.py` :

1. Ajoute un store `long_term: list[dict]` partage entre les runs (chaque entree : `{"question_pattern": ..., "answer": ..., "facts": {...}, "hits": 0}`)
2. A la fin d'un run reussi, le synthesizer ecrit dans `long_term` : la reponse finale ET les faits intermediaires du scratchpad (ex: `africa_population`, `africa_area`)
3. Modifie le `planner_node` :
   - Avant de planifier, cherche dans `long_term` une entree qui couvre la question (match par mots-cles normalises)
   - **Hit complet** (la meme question) : plan reduit a un seul step `recall:final_answer`
   - **Hit partiel** (les faits necessaires existent, ex: la population a deja ete cherchee) : plan sans les steps de search deja couverts, avec des steps `recall:<fact>` a la place
4. L'executor gere la nouvelle action `recall:` en lisant la memoire au lieu d'appeler un outil
5. Demo : pose 2 fois "What is the population density of Africa?" — le run 1 fait 4 steps avec searches, le run 2 fait 1 step recall. Puis pose "What is the population of Africa?" — hit partiel, 0 search
6. Affiche pour chaque run : nombre de steps, nombre de tool calls, hits memoire

### Criteres de reussite
- [ ] Le run 2 de la meme question ne fait AUCUN appel aux outils de recherche
- [ ] Le hit partiel reutilise les faits sans refaire les searches correspondantes
- [ ] Les compteurs (steps, tool calls, hits) prouvent l'economie
- [ ] La reponse du run 2 est identique a celle du run 1
- [ ] Une question jamais vue continue de fonctionner normalement (pas de faux hit)

---

## Exercice 2 : Executor robuste — retry, fallback d'outil et rapport d'echec

### Objectif
Doter l'executor d'une vraie strategie de resilience : retenter un outil qui echoue, basculer sur un outil alternatif, et produire un rapport d'echec exploitable si tout echoue.

### Consigne
1. Modifie `mock_web_search` pour simuler des pannes : une closure `make_flaky_search(fail_first_n)` qui leve `RuntimeError("search backend timeout")` les N premiers appels pour une meme query
2. Cree une table de fallback : `FALLBACKS = {"search": ["web_search", "read_doc"]}` — si `web_search` echoue definitivement, l'executor tente `read_doc` avec un nom de doc derive de la query
3. Dans `executor_node`, implemente la politique :
   - Max 2 tentatives par outil (pas de sleep reel, juste un compteur)
   - Apres 2 echecs, passe a l'outil suivant de la chaine de fallback
   - Chaque tentative est journalisee dans `state["incidents"]` : `{"step": ..., "tool": ..., "attempt": ..., "error": ...}`
4. Si TOUTE la chaine echoue, le step est marque `FAILED` et le synthesizer doit produire une reponse honnete : "I could not retrieve X (2 tools failed). Partial answer based on: ..."
5. Teste 3 scenarios : (a) panne transitoire (1 echec puis succes au retry), (b) bascule sur read_doc, (c) echec total avec reponse partielle honnete

### Criteres de reussite
- [ ] Le retry fonctionne : 1 echec transitoire n'impacte pas la reponse finale
- [ ] Le fallback `read_doc` est utilise quand `web_search` echoue 2 fois
- [ ] `incidents` contient l'historique complet des tentatives avec les erreurs
- [ ] L'echec total produit une reponse partielle honnete (pas d'hallucination)
- [ ] Les 3 scenarios sont verifies par asserts (sur la reponse et sur incidents)

---

## Exercice 3 : Node de clarification — savoir quand poser une question

### Objectif
Implementer la brique "when to ask the user" : detecter qu'une question est ambigue AVANT de lancer le plan, demander une clarification, et replanifier avec la reponse.

### Consigne
1. Ecris un detecteur d'ambiguite local `detect_ambiguity(question) -> list[str]` qui retourne les ambiguites trouvees :
   - Entite manquante : "the density" sans nom de lieu -> "which place?"
   - Reference pronominale sans antecedent : "its population" en debut de session -> "what does 'it' refer to?"
   - Comparaison incomplete : "is it denser?" -> "denser than what?"
   - Unite ambigue : "how big" -> "area or population?"
2. Ajoute un node `clarifier_node` dans le graph, AVANT le planner :
   - Si `detect_ambiguity` retourne des items, le node ecrit `{"needs_clarification": True, "questions": [...]}` et le graph route vers END (premier passage)
   - La fonction `run_sample_question` detecte ce cas, simule la reponse de l'utilisateur via un dict `USER_ANSWERS` (mock), reformule la question (`question + " (clarified: " + answer + ")"`) et relance le graph
3. La question reformulee doit passer le detecteur et derouler le plan normalement
4. Teste : "What is the density?" -> clarification "which place?" -> reponse mock "Paris" -> le run 2 repond avec la densite de Paris
5. Teste aussi une question claire ("What is the population of Africa?") : aucun detour par la clarification
6. Garde-fou : maximum 2 rounds de clarification, ensuite l'agent repond avec ses hypotheses explicites ("Assuming you mean...")

### Criteres de reussite
- [ ] Les 4 types d'ambiguite sont detectes par des regles locales testables
- [ ] Une question ambigue declenche exactement un aller-retour de clarification
- [ ] La question clarifiee produit la bonne reponse finale
- [ ] Une question claire ne passe jamais par le clarifier
- [ ] Le garde-fou de 2 rounds fonctionne (teste avec un USER_ANSWERS qui reste ambigu)
