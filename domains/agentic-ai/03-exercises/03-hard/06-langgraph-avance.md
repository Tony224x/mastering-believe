# Exercices Hard — LangGraph avance (J6)

---

## Exercice 1 : Time-travel debugger complet (inspect / diff / fork / replay)

### Objectif
Construire un outil de debug autour du checkpointer : naviguer dans l'historique d'un run, comparer des etats, forker des scenarios alternatifs et rejouer — la boite a outils du debugging d'agents en production.

### Consigne
1. Instrumente le graph pour qu'un checkpoint soit sauvegarde **a chaque superstep** avec des metadonnees : `step`, `node_execute`, `timestamp`, `updates_appliquees`
2. Implemente une classe `TimeTravelDebugger(ckpt, app)` avec :
   - `timeline(thread_id) -> list[str]` : resume lisible de chaque step (`"step 3 | node=executor | updates: results(+2)"`)
   - `inspect(thread_id, step) -> dict` : state complet a ce step
   - `diff(thread_id, step_a, step_b) -> dict` : champs modifies entre 2 steps avec from/to (gere les listes : affiche `+N elements`)
   - `fork(thread_id, step, mutation: dict, new_thread_id) -> dict` : charge le state au step, applique la mutation, rejoue le graph **depuis ce point** sur un nouveau thread, retourne le state final
   - `replay(thread_id) -> dict` : rejoue tout le run depuis le step 0 et verifie que le state final est identique (determinisme)
3. Scenario de demo : un agent de 5 nodes (plan -> search -> compute -> draft -> finalize) ou `search` ecrit une valeur erronee dans le state
4. Utilise le debugger pour : voir la timeline, identifier le step fautif via diff, forker au step juste avant avec la valeur corrigee, montrer que le fork produit le bon resultat final alors que l'original est faux
5. Verifie l'isolation : le thread original est intact apres le fork (asserts)
6. `replay` doit detecter une non-reproductibilite : injecte volontairement un node non deterministe (`time.time()` dans le state) et montre que replay le signale

### Criteres de reussite
- [ ] Chaque step a un checkpoint avec metadonnees completes
- [ ] `timeline` et `inspect` donnent une vue exploitable du run
- [ ] `diff` identifie precisement le step ou la valeur erronee apparait
- [ ] `fork` produit un resultat final correct sans toucher au thread original
- [ ] `replay` valide le determinisme et detecte le node non deterministe
- [ ] Tous les scenarios sont verifies par asserts

---

## Exercice 2 : Subgraphs a state isole avec mapping et politique d'erreur

### Objectif
Implementer la composition avancee de LangGraph : un subgraph dont le schema de state est DIFFERENT de celui du parent, avec transformation entree/sortie explicite et propagation d'erreur configurable.

### Consigne
1. Parent state : `{"query": str, "report": str, "errors": Annotated[list, add], "attempts": int}`
   Subgraph state (different !) : `{"keywords": list[str], "hits": Annotated[list, add], "ranked": list[str]}`
2. Implemente un wrapper generique :
   ```python
   def as_subgraph_node(sub_app, map_in: Callable, map_out: Callable,
                        on_error: str = "raise", max_retries: int = 0) -> Callable
   ```
   - `map_in(parent_state) -> sub_state` : ex. extrait les keywords de `query`
   - `map_out(sub_final_state) -> parent_updates` : ex. transforme `ranked` en `report`
   - `on_error` : `"raise"` (propage), `"continue"` (ecrit l'erreur dans `errors` et retourne des updates vides), `"retry"` (retente jusqu'a `max_retries`, puis applique `"continue"`)
3. Le subgraph "research" : `extract -> search -> rank`. Le node `search` echoue (`RuntimeError("index unavailable")`) aux 2 premiers appels d'un meme run, reussit au 3e (compteur module-level ou closure)
4. Construis le parent : `START -> research_node (subgraph wrappe) -> summarize -> END`
5. Teste les 3 politiques :
   - `on_error="raise"` : l'invocation leve bien RuntimeError
   - `on_error="continue"` : le run termine, `errors` contient le message, `report` indique "no data"
   - `on_error="retry", max_retries=3` : le run reussit au 3e essai, `attempts == 3`
6. Verifie l'isolation : aucun champ du sub-state (`keywords`, `hits`, `ranked`) ne fuit dans le parent state final

### Criteres de reussite
- [ ] Le subgraph compile et tourne seul avec son propre schema
- [ ] `map_in`/`map_out` font une vraie transformation (pas un passthrough)
- [ ] Les 3 politiques d'erreur ont le comportement attendu (asserts)
- [ ] Le retry reussit au 3e essai et le compte d'attempts est correct
- [ ] Le parent state final ne contient aucune cle du sub-state
- [ ] Le wrapper est generique (fonctionne tel quel pour un 2e subgraph de schema different)
