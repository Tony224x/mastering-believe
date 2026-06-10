# Exercices Hard — LangGraph fondamentaux (J5)

---

## Exercice 1 : Etendre le mini-runtime — supersteps paralleles, reducers et detection de cycle

### Objectif
Comprendre comment LangGraph execute reellement un graph (modele Pregel/BSP) en etendant le stub pour supporter plusieurs nodes actifs par superstep, l'application correcte des reducers, et des protections runtime.

### Consigne
Reecris le moteur d'execution du stub en modele **superstep** :

1. **Plusieurs nodes actifs** : `add_edge` peut etre appele plusieurs fois depuis la meme source (`a -> b` ET `a -> c`). Au superstep suivant, `b` et `c` s'executent tous les deux sur le **meme snapshot** du state
2. **Application des reducers** : les updates de `b` et `c` sont appliques au state via les reducers APRES que les deux ont tourne (ordre d'enregistrement des nodes). Un champ sans reducer ecrase ; si 2 nodes ecrivent le meme champ sans reducer dans le meme superstep, leve `InvalidUpdateError` (comme la vraie lib)
3. **Convergence** : les deux branches pointent vers un node `join` — il ne doit s'executer qu'UNE fois par superstep, meme s'il est cible par 2 edges
4. **Protections** :
   - `recursion_limit` (defaut 25 supersteps) -> `GraphRecursionError`
   - Detection statique des nodes inatteignables depuis START a la compilation -> warning
   - Edge vers un node inexistant -> `ValueError` a la compilation
5. Demo : un graph fan-out/fan-in : `START -> splitter -> (search_web, search_docs) -> join -> END` ou les 2 searches ecrivent dans `results: Annotated[list, add]` et `join` synthetise
6. Demo de chaque protection : graph cyclique sans condition de sortie (recursion limit), 2 nodes ecrivant `status: str` en parallele (InvalidUpdateError), edge vers node inconnu (ValueError)

### Criteres de reussite
- [ ] Les 2 branches paralleles voient le meme snapshot d'entree (verifiable : aucune ne voit les resultats de l'autre)
- [ ] `results` contient les 2 contributions apres le superstep (reducer add)
- [ ] L'ecriture concurrente d'un champ sans reducer leve InvalidUpdateError
- [ ] `join` ne s'execute qu'une fois
- [ ] Les 3 protections levent les bonnes erreurs avec des messages clairs
- [ ] La demo fan-out/fan-in produit une synthese contenant les 2 resultats

---

## Exercice 2 : Compilateur de graph declaratif (config -> StateGraph)

### Objectif
Construire un layer "no-code" au-dessus du runtime : une spec declarative (dict JSON-compatible) est validee puis compilee en graph executable — le pattern des plateformes d'orchestration d'agents.

### Consigne
1. Definis un format de spec :
   ```python
   spec = {
       "state": {
           "messages": {"type": "list", "reducer": "add"},
           "category": {"type": "str"},
           "draft": {"type": "str"},
       },
       "nodes": [
           {"name": "classify", "fn": "classify_fn"},
           {"name": "answer_math", "fn": "math_fn"},
           {"name": "answer_general", "fn": "general_fn"},
       ],
       "edges": [
           {"from": "START", "to": "classify"},
           {"from": "classify", "if": "route_fn",
            "map": {"math": "answer_math", "general": "answer_general"}},
           {"from": "answer_math", "to": "END"},
           {"from": "answer_general", "to": "END"},
       ],
   }
   ```
2. Ecris `compile_spec(spec, registry) -> CompiledGraph` ou `registry` est un dict `{nom_fn: callable}` :
   - Construit dynamiquement la classe de state (TypedDict ou dict de reducers) a partir de `spec["state"]`
   - Mappe `"add"` vers `operator.add` (et supporte `"merge_dict"` custom)
   - Ajoute nodes et edges (simples et conditionnels)
3. Ecris `validate_spec(spec, registry) -> list[str]` qui detecte AVANT compilation :
   - Reference a une fn absente du registry
   - Edge vers un node non declare
   - Node sans aucun edge entrant (inatteignable)
   - Node sans aucun chemin vers END (cul-de-sac)
   - Cle dupliquee dans `nodes`
   - Reducer inconnu
4. La compilation refuse une spec invalide avec la liste complete des erreurs (pas juste la premiere)
5. Demo : compile la spec ci-dessus, execute sur 2 questions (math / general), puis montre 3 specs invalides avec leurs erreurs respectives
6. Bonus : `export_mermaid(spec) -> str` qui genere le diagramme Mermaid du graph

### Criteres de reussite
- [ ] Une spec valide compile et s'execute correctement sur les 2 questions
- [ ] Les reducers declares dans la spec sont appliques (messages accumules)
- [ ] `validate_spec` detecte les 6 categories d'erreurs
- [ ] Toutes les erreurs d'une spec invalide sont remontees d'un coup
- [ ] Le code de validation est teste sur au moins 3 specs cassees differentes
- [ ] (Bonus) le Mermaid genere est syntaxiquement plausible (`graph TD`, fleches)
