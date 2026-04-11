# Exercices Faciles — RAG Agentique (J8)

---

## Exercice 1 : Retrieval grader a 3 niveaux

### Objectif
Passer d'un grading binaire (RELEVANT / IRRELEVANT) a un grading a 3 niveaux pour ameliorer la precision.

### Consigne
En partant de `MockLLM._grade` dans `02-code/08-rag-agentique.py` :

1. Modifie la methode pour retourner un des 3 verdicts : `HIGHLY_RELEVANT`, `PARTIALLY_RELEVANT`, `IRRELEVANT`
2. Regles :
   - `HIGHLY_RELEVANT` : overlap >= 4 mots ET le chunk contient un chiffre si la question en demande un
   - `PARTIALLY_RELEVANT` : overlap >= 2 mots
   - `IRRELEVANT` : sinon
3. Adapte `AgenticRAG._answer_sub_question` pour garder les chunks `HIGHLY_RELEVANT` en priorite, puis completer avec les `PARTIALLY_RELEVANT` seulement si on a moins de 2 chunks hautement pertinents
4. Ajoute un champ `trace.highly_relevant_count` et `trace.partially_relevant_count`
5. Test : lance le demo et affiche combien de chunks sont gardes par niveau sur la requete composee

### Criteres de reussite
- [ ] 3 verdicts distincts sont retournes par `_grade`
- [ ] La logique de priorisation dans `_answer_sub_question` est correcte
- [ ] Les traces affichent les comptes par niveau
- [ ] Le demo tourne sans erreur et montre une difference de comportement vs le grading binaire

---

## Exercice 2 : Budget d'appels LLM

### Objectif
Proteger le RAG agentique contre une explosion du cout en appels LLM.

### Consigne
1. Ajoute un champ `llm_budget: int = 20` dans `AgenticRAGConfig`
2. Ajoute un compteur `self._llm_calls: int` dans `AgenticRAG` qui s'incremente a chaque appel `self.llm(...)`
3. Cree un helper interne `self._call_llm(task, payload)` que tout le code doit utiliser a la place de `self.llm(...)` directement
4. Ce helper doit :
   - Lever une `RuntimeError("LLM budget exceeded")` si on depasse le budget
   - Afficher un warning a 80% du budget ("[AgenticRAG] warning: 16/20 LLM calls used")
5. Teste avec un budget tres bas (`llm_budget=5`) pour voir le budget exceeded en action
6. Teste avec un budget normal (`llm_budget=20`) pour voir que le pipeline termine

### Criteres de reussite
- [ ] Le compteur s'incremente a chaque appel LLM
- [ ] Le warning a 80% s'affiche correctement
- [ ] Le pipeline leve `RuntimeError` proprement a depassement
- [ ] Le budget normal laisse le pipeline aller au bout
- [ ] Le `explain()` affiche le nombre total d'appels LLM utilises

---

## Exercice 3 : Multi-hop simple

### Objectif
Implementer un mini multi-hop : le resultat d'une sous-question permet d'en deduire une nouvelle sous-question.

### Consigne
1. Etends `CORPUS` avec 2 nouveaux documents :
   - Un qui dit "Artefact was founded by Vincent Luciani and Philippe Rolet in 2015."
   - Un qui dit "Vincent Luciani lives in Paris and speaks at AI conferences in Europe."
2. Ecris une fonction `multi_hop_answer(query, rag, retriever, llm, max_hops=3)` qui :
   - Lance un premier retrieve sur la query
   - Si le LLM detecte une entite dans les chunks trouves (via un nouveau task `"extract_entity"` dans MockLLM), utilise cette entite pour formuler une nouvelle sous-question
   - Boucle jusqu'a `max_hops` ou jusqu'a n'avoir plus de nouvelle entite
3. Le `extract_entity` du MockLLM peut etre naif : il retourne le premier nom propre (mot capitalise de 2+ caracteres) present dans les chunks
4. Teste avec la question "Who founded the main French competitor of Kalira and where does that person live?"
   - Hop 1 : "main French competitor of Kalira" -> Artefact
   - Hop 2 : "who founded Artefact" -> Vincent Luciani
   - Hop 3 : "where does Vincent Luciani live" -> Paris

### Criteres de reussite
- [ ] Les 2 nouveaux documents sont ajoutes au corpus
- [ ] `extract_entity` existe et fonctionne sur des chunks simples
- [ ] `multi_hop_answer` loop correctement avec une limite de hops
- [ ] La question test traverse bien 3 hops
- [ ] Chaque hop est affiche avec l'entite extraite et la nouvelle sous-question
