# Exercices Medium — LangGraph avance (J6)

---

## Exercice 1 : Streaming multi-modes (values / updates / events)

### Objectif
Implementer les 3 modes de streaming de LangGraph dans le stub pour comprendre ce que chaque mode expose et a quel moment.

### Consigne
En partant de `MiniCompiled.stream` de `02-code/06-langgraph-avance.py` :

1. Ajoute un parametre `mode` a `stream(initial_state, mode="values")` :
   - `"values"` : yield le **state complet** apres chaque superstep
   - `"updates"` : yield uniquement `{node_name: updates_retournees}` apres chaque node
   - `"events"` : yield des events fins : `{"event": "on_node_start", "node": ...}`, `{"event": "on_node_end", "node": ..., "updates": ...}`, et `{"event": "on_graph_end", "state": ...}`
2. Construis un graph de 3 nodes sequentiels (`fetch -> analyze -> report`) ou chaque node ajoute un message et met a jour un champ `progress: float` (0.33, 0.66, 1.0)
3. Lance le meme graph dans les 3 modes et affiche ce que chaque mode produit, clairement etiquete
4. Ecris un consommateur `progress_bar(stream)` qui utilise le mode `"updates"` pour afficher une barre de progression textuelle basee sur `progress`
5. Verifie par asserts : `values` yield 3 states complets, `updates` yield 3 dicts a 1 cle, `events` yield 7 events (3 starts + 3 ends + 1 graph_end)

### Criteres de reussite
- [ ] Les 3 modes produisent des structures differentes et correctes
- [ ] Le mode `values` montre le state cumule, le mode `updates` seulement les deltas
- [ ] Le mode `events` encadre chaque node avec start/end
- [ ] La progress bar fonctionne en consommant le stream (pas le state final)
- [ ] Les asserts sur le nombre d'elements yieldes passent

---

## Exercice 2 : Map-reduce sur documents avec Send

### Objectif
Utiliser le pattern Send pour traiter un nombre **dynamique** de documents en parallele (map), puis agreger les resultats (reduce) — le pattern de base du traitement de corpus.

### Consigne
1. State :
   ```python
   class MapReduceState(TypedDict):
       docs: list[str]
       summaries: Annotated[list, add]
       final_digest: str
   ```
2. Node `dispatcher` : retourne une liste de `Send("summarize_one", {"doc": doc, "index": i})` — un par document (le nombre de docs n'est PAS connu a l'avance par le graph)
3. Node `summarize_one` : recoit UN doc (pas le state complet), produit un resume mock (`premiere phrase + " [" + str(len(doc)) + " chars]"`) et l'ajoute a `summaries` avec son index
4. Node `aggregate` : trie les resumes par index (l'ordre d'arrivee n'est pas garanti), construit `final_digest` qui concatene les resumes numerotes
5. Teste avec 2 corpus de tailles differentes (3 docs puis 5 docs) **sans modifier le graph**
6. Ajoute un doc vide dans le corpus : `summarize_one` doit produire `"[EMPTY DOC]"` au lieu de planter

### Criteres de reussite
- [ ] Le nombre de branches paralleles s'adapte au nombre de docs (3 puis 5)
- [ ] Chaque worker ne recoit que son doc (verifiable par print)
- [ ] L'aggregateur produit un digest ordonne par index meme si les resultats arrivent dans le desordre
- [ ] Le doc vide est gere proprement
- [ ] Asserts : `len(summaries) == len(docs)` et le digest contient tous les index

---

## Exercice 3 : Persistence multi-sessions avec reprise de conversation

### Objectif
Utiliser le checkpointer pour simuler un assistant qui survit a un "redemarrage" : la conversation reprend la ou elle s'etait arretee grace au thread_id.

### Consigne
En partant du `MiniCheckpointer` :

1. Construis un mini-graph conversationnel : `START -> recall -> respond -> END`
   - `recall` : compte combien de tours user existent deja dans `messages` et ecrit `turn_count`
   - `respond` (mock) : repond en se referant au contexte (ex: si un tour precedent mentionne un budget, la reponse le rappelle)
2. Implemente une fonction `chat(app, ckpt, thread_id, user_message) -> str` qui :
   - Charge le dernier state du thread via `ckpt.load_latest(thread_id)` (ou state vierge si nouveau thread)
   - Ajoute le message user, invoque le graph, sauvegarde le state final dans le checkpointer
   - Retourne la reponse de l'assistant
3. Session 1 (thread "alice") : "Je cherche un laptop, budget 500 EUR" puis "Plutot leger si possible"
4. **Simule un redemarrage** : recree `app` (nouveau compile), garde le checkpointer
5. Session 2 (meme thread "alice") : "Quel etait mon budget deja ?" — la reponse doit citer 500 EUR et `turn_count` doit valoir 3
6. Session 3 (thread "bob") : meme question — la reponse doit indiquer qu'aucun budget n'est connu (isolation des threads)

### Criteres de reussite
- [ ] Le state survit a la re-creation du graph (persiste uniquement via le checkpointer)
- [ ] Le thread "alice" retrouve le budget apres redemarrage
- [ ] `turn_count` reflete bien l'historique cumule (3 au 3e tour)
- [ ] Le thread "bob" est totalement isole d'alice
- [ ] Asserts sur la reponse d'alice (contient "500") et celle de bob (ne contient pas "500")
