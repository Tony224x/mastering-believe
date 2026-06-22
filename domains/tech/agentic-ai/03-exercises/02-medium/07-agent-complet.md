# Exercices Medium — Agent complet (J7, capstone semaine 1)

---

## Exercice 1 : Replan dynamique sur echec d'etape

### Objectif
Passer d'un planning statique (le plan est fige une fois pour toutes) a un planning dynamique : quand une etape de l'executor echoue, l'agent revient au planner pour produire un plan alternatif.

### Consigne
En t'inspirant de l'architecture `planner -> executor -> analyzer -> synthesizer` de `02-code/07-agent-complet.py` :

1. Modifie le graphe pour qu'un echec d'executor (`stuck = True`) route vers un node `replanner` au lieu de terminer
2. Le `replanner` :
   - Recoit le plan courant, l'index de l'etape qui a echoue, et un compteur `replan_count`
   - Produit un plan alternatif (ex: remplacer une `search:X` qui a echoue par `read_doc:X_report` puis re-essayer)
   - Incremente `replan_count`
3. Garde-fou : au-dela de `max_replans = 2`, l'agent abandonne avec un verdict `failed` propre (pas de boucle infinie)
4. Teste 3 scenarios :
   - Une question qui passe du premier coup (0 replan)
   - Une question dont la 1re recherche echoue mais qui reussit apres 1 replan
   - Une question dont l'info n'existe nulle part (2 replans puis abandon propre)

### Criteres de reussite
- [ ] Un echec d'etape route vers `replanner`, pas vers `END`
- [ ] Le `replanner` produit un plan different du plan initial
- [ ] `replan_count` est borne par `max_replans` (pas de boucle infinie)
- [ ] Les 3 scenarios produisent respectivement 0, 1 et 2 replans
- [ ] Le scenario d'echec total termine avec un verdict explicite (`failed`), pas un crash

---

## Exercice 2 : Cache long-term partage entre executions

### Objectif
Transformer la long-term memory en un vrai cache persistant entre plusieurs questions, et mesurer le gain (cache hits) sur une session multi-questions.

### Consigne
Le code de J7 cree un agent neuf a chaque question, donc la long-term memory est perdue. Corrige ca :

1. Cree une classe `ResearchSession` qui :
   - Detient une `long_term: list[dict]` persistante (faits accumules)
   - Expose `ask(question) -> dict` qui lance l'agent en **injectant** la long-term courante dans le state initial et en **recuperant** la long-term enrichie a la fin
2. Ajoute un compteur de **tool calls reels** vs **cache hits** par question (un cache hit = un fait deja present dans la long-term, donc pas d'appel a `mock_web_search`)
3. Lance une session avec ces 3 questions, dans l'ordre :
   - "What is the population of Africa?"
   - "What is the area of Africa?"
   - "What is the population density of Africa?"  (doit reutiliser les 2 faits deja en cache -> 2 cache hits)
4. Affiche, par question : nb de tool calls reels, nb de cache hits, et la taille de la long-term apres

### Criteres de reussite
- [ ] `ResearchSession` conserve la long-term entre les appels a `ask`
- [ ] La 3e question fait au moins 2 cache hits (population + area deja connus)
- [ ] Le compteur distingue tool calls reels et cache hits
- [ ] La long-term grandit puis se stabilise (pas de doublons pour le meme fait)
- [ ] La derniere question calcule correctement la densite sans nouvel appel reseau

---

## Exercice 3 : Synthesizer avec citations de sources

### Objectif
Ameliorer le synthesizer pour qu'il produise une reponse **sourcee** : chaque fait du rapport final cite l'outil et le document d'ou il provient.

### Consigne
1. Enrichis le format des findings : au lieu d'une simple `str`, chaque finding devient un dict `{"text": ..., "tool": "mock_web_search" | "read_doc", "source": "<query ou doc_name>"}`
2. L'analyzer doit propager la source quand il extrait un fait dans `short_term` (ex: `short_term["population"] = {"value": 1_460_000_000, "source": "read_doc(africa_report_2024.pdf)"}`)
3. Le synthesizer doit produire une reponse du type :
   ```
   The population density of Africa is approximately 48 hab/km2.
     - area: 30,370,000 km2 (source: mock_web_search 'africa area')
     - population: 1,460,000,000 inhabitants (source: read_doc africa_report_2024.pdf)
   ```
4. Ecris une fonction `verify_citations(answer: str, short_term: dict) -> bool` qui verifie que chaque fait numerique de `short_term` a bien une citation dans la reponse finale
5. Teste sur la question de densite : la reponse doit citer les 2 sources, et `verify_citations` doit retourner `True`

### Criteres de reussite
- [ ] Les findings portent la trace de leur outil ET de leur source
- [ ] L'analyzer propage la source jusqu'a `short_term`
- [ ] Le synthesizer affiche une citation par fait utilise
- [ ] `verify_citations` detecte une citation manquante (teste le cas negatif aussi)
- [ ] La reponse reste lisible (pas juste un dump de dict)
