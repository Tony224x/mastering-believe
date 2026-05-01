# Exercices Faciles — LangGraph fondamentaux (J5)

---

## Exercice 1 : Ajouter un node "greeter" avant l'agent

### Objectif
Comprendre comment ajouter un node au graph et comment les edges s'enchainent.

### Consigne
En partant du graph defini dans `02-code/05-langgraph-fondamentaux.py` :

1. Ajoute un nouveau node `greeter_node(state)` qui :
   - Regarde le premier message user
   - Genere un message assistant qui dit "Bonjour ! Je vais m'occuper de votre demande: {query}"
   - Retourne `{"messages": [new_message], "step_count": state.get("step_count", 0) + 1}`
2. Modifie le graph pour que le flux soit : `START -> greeter -> agent -> (tools ou end)`
3. Lance `invoke` sur la question "Compute 5 + 7"
4. Verifie que le state final contient **au moins 3 messages** (user, greeter, assistant final)
5. Verifie que `step_count >= 3`

### Criteres de reussite
- [ ] Le node `greeter` est ajoute et appele exactement 1 fois
- [ ] Le state final contient user + greeter + au moins une reponse d'agent
- [ ] `step_count` est >= 3
- [ ] Le graph fonctionne sur la question math ET sur une question non-math

---

## Exercice 2 : Conditional edge avec 3 destinations

### Objectif
Comprendre comment un conditional edge peut router vers plus de 2 nodes.

### Consigne
Cree un mini-graph classifier qui, a partir d'une question user :

1. Route vers un node `math_node` si la question contient des chiffres
2. Route vers un node `weather_node` si la question contient "meteo" ou "weather"
3. Route vers un node `default_node` sinon

Chaque node doit ajouter un message d'assistant specifique :
- `math_node` : "Je vois une question math"
- `weather_node` : "Je vois une question meteo"
- `default_node` : "Je vais faire de mon mieux"

Apres chacun de ces nodes, le graph doit aller a END.

Teste avec 3 questions differentes et verifie que le bon node a ete execute dans chaque cas (via le message).

### Criteres de reussite
- [ ] Le classifier a exactement 4 nodes (classifier, math, weather, default)
- [ ] Le conditional edge a 3 destinations mappees correctement
- [ ] Les 3 questions testees declenchent chacune le bon node
- [ ] Aucune question ne declenche 2 nodes en meme temps
- [ ] Le graph termine en exactement 2 steps (classifier + 1 des 3 nodes)

---

## Exercice 3 : State avec reducer custom

### Objectif
Comprendre comment un reducer custom permet de merger des champs complexes sans les ecraser.

### Consigne
Cree un state avec un champ `findings: dict` qui utilise un reducer custom `merge_findings`.

1. Le reducer `merge_findings(existing, new)` doit merger deux dicts tels que :
   - Si une cle existe dans les 2, concatener les valeurs (listes)
   - Si une cle n'existe que dans `new`, l'ajouter
   - Si une cle n'existe que dans `existing`, la conserver
2. Declare le state :
   ```python
   class ResearchState(TypedDict):
       findings: Annotated[dict, merge_findings]
       step_count: int
   ```
3. Cree 2 nodes qui retournent des `findings` differents :
   - `searcher_node` retourne `{"findings": {"sources": ["src_1", "src_2"]}}`
   - `analyzer_node` retourne `{"findings": {"sources": ["src_3"], "keywords": ["ia"]}}`
4. Apres execution, le state final doit contenir :
   ```python
   {"sources": ["src_1", "src_2", "src_3"], "keywords": ["ia"]}
   ```
5. Verifie avec un `assert`.

### Criteres de reussite
- [ ] Le reducer `merge_findings` est une fonction pure (pas de mutation)
- [ ] Le state final contient les 3 sources dans l'ordre d'insertion
- [ ] Le state final contient la cle `keywords` ajoutee par le second node
- [ ] L'assert passe
- [ ] Le code fonctionne avec la mini-stub (pas besoin de langgraph)
