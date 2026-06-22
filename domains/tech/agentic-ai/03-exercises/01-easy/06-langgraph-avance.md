# Exercices Faciles — LangGraph avance (J6)

---

## Exercice 1 : Composer deux subgraphs dans un parent

### Objectif
Pratiquer la composition : construire deux subgraphs independants et les assembler dans un parent graph.

### Consigne
En partant du code de `02-code/06-langgraph-avance.py` :

1. Cree un premier subgraph `translator_subgraph` qui a 2 nodes :
   - `detect_language(state)` : ecrit `{"lang": "fr"}` dans le state
   - `translate(state)` : ecrit `{"translated": "<text>_translated_from_<lang>"}`
2. Cree un second subgraph `summarizer_subgraph` qui a 2 nodes :
   - `extract_keywords(state)` : ecrit `{"keywords": ["a", "b", "c"]}`
   - `build_summary(state)` : ecrit `{"summary": "summary with keywords: ..."}`
3. Construis un parent graph qui execute : `START -> translator -> summarizer -> END`
4. Invoque avec un input initial et verifie que le state final contient `lang`, `translated`, `keywords`, et `summary`

### Criteres de reussite
- [ ] Les 2 subgraphs sont **independants** et peuvent etre compiles seuls
- [ ] Le parent les appelle via des fonctions wrapper
- [ ] Le state final contient les 4 champs attendus
- [ ] Chaque subgraph est testable en isolation
- [ ] Aucun node ne mute le state directement

---

## Exercice 2 : Fan-out sur 3 outils

### Objectif
Utiliser le Send API (version stub ou reelle) pour lancer 3 outils en parallele et collecter leurs resultats dans une liste.

### Consigne
Construis un graph qui, a partir d'une query utilisateur, execute 3 "outils mocks" en parallele :

1. `web_search_tool(query) -> str` (retourne "WEB: ...")
2. `wikipedia_tool(query) -> str` (retourne "WIKI: ...")
3. `arxiv_tool(query) -> str` (retourne "ARXIV: ...")

Le flow doit etre :
- `START -> dispatcher -> (3 parallel tool nodes) -> aggregator -> END`

Le `dispatcher` retourne 3 `Send` (un par outil). Chaque tool node ecrit son resultat dans `results` (qui utilise le reducer `add`). L'aggregator doit voir 3 elements dans `results`.

### Criteres de reussite
- [ ] Les 3 outils sont appeles en parallele (pas en serie)
- [ ] L'aggregator recoit exactement 3 resultats
- [ ] Chaque resultat est prefixe par le nom de son outil (WEB/WIKI/ARXIV)
- [ ] Un assert verifie que `len(results) == 3`
- [ ] Le code reste lisible et sans duplication (factoriser si possible)

---

## Exercice 3 : Checkpoint + branching

### Objectif
Pratiquer le time-travel : executer un graph, sauvegarder les checkpoints, puis brancher une variante a partir d'un point passe.

### Consigne
En partant du `Checkpointer` stub de `02-code/06-langgraph-avance.py` :

1. Construis un graph simple avec 4 nodes sequentiels : `a -> b -> c -> d`
   - Chaque node appende un message et incremente un compteur
2. Execute le graph sur un `thread_id="original"` et inspecte l'historique
3. Charge l'etat a l'etape 2 avec `ckpt.load_at("original", 2)`
4. Modifie le state charge pour injecter un message `"[BRANCHED] I took a different path"`
5. Relance le graph depuis cet etat modifie avec `thread_id="fork"`
6. Compare les 2 historiques : le thread "original" doit etre intact, le thread "fork" doit avoir le message injecte

### Criteres de reussite
- [ ] L'historique du thread "original" contient bien 4 steps
- [ ] Le chargement a l'etape 2 donne un state coherent
- [ ] Le thread "fork" contient le message `[BRANCHED]`
- [ ] Le thread "original" est inchange apres le branching
- [ ] Un `print` affiche les 2 historiques cote a cote pour visualisation
