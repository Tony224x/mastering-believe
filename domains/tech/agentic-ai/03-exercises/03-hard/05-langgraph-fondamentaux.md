# Exercices Hard — LangGraph fondamentaux (J5)

---

## Exercice 1 : Human-in-the-loop avec interrupt, checkpoint et resume

### Objectif
Implementer le pattern human-in-the-loop du module : etendre le stub LangGraph pour supporter `interrupt_before`, un checkpointer par `thread_id`, et la reprise via `invoke(None)` (avec possibilite de modifier le state avant de reprendre).

### Consigne
Construis un mini-LangGraph capable de mettre l'execution en pause avant un node sensible :

1. **Etends le `MiniCompiledGraph`** (copie-le dans ta solution) :
   - `compile(checkpointer=..., interrupt_before=[...])` accepte une liste de nodes avant lesquels faire pause.
   - Un `MemoryCheckpointer` stocke le state par `thread_id` (dict en memoire, deterministe).
   - `invoke(state, config)` ou `config = {"configurable": {"thread_id": "..."}}` :
     - Execute jusqu'a atteindre un node dans `interrupt_before` → sauvegarde le state au checkpoint et **retourne le state partiel** (status `"interrupted"`, avec le nom du node en attente).
     - Si appele avec `state=None`, **reprend** depuis le dernier checkpoint du `thread_id` et continue jusqu'a END (ou prochain interrupt).
   - Avant de reprendre, on doit pouvoir **modifier le state checkpointe** (ex : un humain edite un champ `approved=True`).
2. **Scenario** : un agent qui veut executer une action sensible `send_email`. Le graph : `START -> draft -> [interrupt] -> send -> END`.
   - 1er `invoke` : s'arrete avant `send`, retourne le brouillon pour validation humaine.
   - On simule l'humain : on lit le brouillon, on set `approved=True` (ou `False` → rejet).
   - 2e `invoke(None, config)` : reprend. Si `approved`, `send` s'execute ; sinon le node `send` refuse.
3. **Deux runs** sur deux `thread_id` differents prouvent l'isolation des checkpoints.

### Criteres de reussite
- [ ] `interrupt_before` met bien le graph en pause avant le node cible (status `"interrupted"`)
- [ ] Le state au moment de la pause est checkpointe par `thread_id`
- [ ] `invoke(None, config)` reprend exactement la ou on s'etait arrete
- [ ] Modifier le state checkpointe entre les deux invokes change le comportement de la reprise (approuve vs rejete)
- [ ] Deux `thread_id` differents ont des checkpoints isoles (pas de fuite d'etat)
- [ ] Le graph ne re-execute pas les nodes deja passes lors de la reprise
- [ ] Tout tourne offline avec le stub (aucune dependance)

---

## Exercice 2 : Agent ReAct complet avec budget de cycles et detecteur de blocage

### Objectif
Assembler un agent ReAct production-grade en LangGraph : 2 outils reels, routage conditionnel, boucle bornee, detecteur de "blocage" (l'agent qui n'avance plus), et terminaison gracieuse avec une raison explicite.

### Consigne
Construis un agent capable de repondre a une question qui necessite **plusieurs outils chaines** :

1. **2 outils** : `search(query) -> str` (mock deterministe, renvoie des faits) et `calculator(expr) -> str` (eval securise N op N).
2. **Graph** : `agent -> (conditional) -> tools | end`, `tools -> agent`. Le node `agent` (MockLLM deterministe) decide a chaque tour : appeler un outil ou repondre.
3. **Garde-fous** dans le state et le routage :
   - `cycle_budget` : nombre max de tours `agent` (ex : 8). Au-dela → sortie `"budget_exhausted"`.
   - **Detecteur de blocage** : si le state n'a pas progresse (meme ensemble de faits collectes) pendant 2 tours consecutifs → sortie `"stuck"`.
   - Sortie normale `"final_answer"` quand l'agent produit une reponse sans tool call.
4. **Scenario reussi** : "Quelle est la population de Paris divisee par sa superficie ?" → l'agent doit faire `search(population)`, `search(area)`, `calculator(pop/area)`, puis repondre. Verifie que la reponse contient la densite (~20581).
5. **Scenario degenere** : un MockLLM qui boucle sur le meme `search` → le detecteur de blocage doit declencher `"stuck"` avant le budget.
6. Le state final expose `stop_reason`, `cycles_used`, et la liste des outils appeles dans l'ordre.

### Criteres de reussite
- [ ] L'agent chaine correctement search → search → calculator → reponse sur le scenario reussi
- [ ] La reponse finale contient la densite calculee (~20581 hab/km2)
- [ ] Le `cycle_budget` borne le nombre de tours et produit `stop_reason == "budget_exhausted"` si depasse
- [ ] Le detecteur de blocage declenche `"stuck"` sur le scenario degenere, avant d'epuiser le budget
- [ ] Le graph ne leve jamais d'exception non geree (terminaison toujours gracieuse)
- [ ] `stop_reason`, `cycles_used` et l'ordre des outils sont lisibles dans le state final
- [ ] Tout est deterministe et tourne offline avec le stub
