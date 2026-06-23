# Exercices Medium — LangGraph fondamentaux (J5)

---

## Exercice 1 : Boucle ReAct avec garde d'iterations et detection de boucle

### Objectif
Aller au-dela du graph minimal : ajouter les deux garde-fous indispensables d'un vrai agent ReAct — un plafond d'iterations et un detecteur de boucle (l'agent qui redemande sans cesse le meme tool call).

### Consigne
En partant du pattern `agent -> tools -> agent` du module 05 :

1. Definis un state `ReactState` (TypedDict) avec au minimum :
   - `messages: Annotated[list, add]`
   - `iterations: int` (scalaire, incremente a chaque passage par `agent`)
   - `tool_history: Annotated[list, add]` (la liste des `(name, args)` deja executes)
2. Ecris un `MockLLM` deterministe **defectueux** : pour une question donnee, il redemande **toujours** le meme tool call (`search`, `{"q": "paris"}`), meme apres avoir recu un resultat. C'est la cause classique d'une boucle infinie.
3. Implemente le conditional edge `should_continue(state)` qui route vers `"end"` si **l'une** de ces conditions est vraie :
   - `iterations >= max_iterations` (plafond, ex : 5)
   - le prochain tool call demande est **identique** a un appel deja present dans `tool_history` (loop detection)
   - le dernier message n'a pas de `tool_call` (reponse finale)
   - sinon, route vers `"tools"`.
4. Le `tool_node` doit enregistrer chaque appel dans `tool_history` et ajouter le resultat dans `messages`.
5. Lance `invoke` sur le LLM defectueux : le graph doit **s'arreter proprement** (pas de `RuntimeError`), avec une raison de sortie identifiable dans le state (`stop_reason`).
6. Verifie qu'avec un LLM **correct** (qui finit par repondre apres un tool call), le graph termine via `"reponse finale"` et non via la garde.

### Criteres de reussite
- [ ] Le graph ne leve jamais `RuntimeError` (la garde s'enclenche avant `max_steps`)
- [ ] Le LLM defectueux declenche la sortie par **loop detection** (meme tool call deja vu), pas par le plafond brut
- [ ] `iterations` est correctement incremente et borne par `max_iterations`
- [ ] `tool_history` contient bien les appels executes, sans doublon execute deux fois
- [ ] Le LLM correct termine via `"reponse finale"` avec `stop_reason == "final_answer"`
- [ ] Le `stop_reason` final est lisible dans le state (`"loop_detected"`, `"max_iterations"` ou `"final_answer"`)

---

## Exercice 2 : Implementer les modes de stream `values` et `updates`

### Objectif
Comprendre concretement la difference entre les modes de streaming `updates` (uniquement le delta de chaque node) et `values` (le state complet apres chaque step), en les implementant toi-meme sur le `MiniCompiledGraph`.

### Consigne
Le stub du module ne fournit qu'un `stream` en mode `updates` implicite. Etends-le :

1. Reprends le `MiniCompiledGraph` (ou copie-le) et ajoute un parametre `stream_mode` a `stream` :
   - `stream_mode="updates"` : yield `{node_name: updates}` (le dict retourne par le node uniquement).
   - `stream_mode="values"` : yield le **state complet** apres le merge de chaque step (une copie, pas une reference).
2. Construis un graph simple a 2-3 nodes qui accumulent des messages (`Annotated[list, add]`) et incrementent un `step_count`.
3. Collecte les evenements des deux modes dans deux listes et **prouve par assertions** que :
   - En mode `updates`, chaque evenement ne contient que le delta du node (ex : 1 message ajoute, pas tout l'historique).
   - En mode `values`, chaque evenement contient le state complet **cumule** (le dernier evenement = le state final de `invoke`).
   - La taille de `messages` est **croissante** en mode `values` et **constante (= delta)** en mode `updates`.
4. Verifie qu'un mode inconnu leve une `ValueError`.
5. Verifie que le mode `values` ne partage pas de reference mutable avec le state interne (modifier un evenement yieldé ne casse pas l'execution).

### Criteres de reussite
- [ ] `stream(mode="updates")` yield uniquement les deltas par node
- [ ] `stream(mode="values")` yield le state complet apres chaque step
- [ ] Le dernier evenement de `values` est egal au resultat de `invoke` sur la meme entree
- [ ] La longueur de `messages` est strictement croissante en mode `values`
- [ ] Un `stream_mode` inconnu leve `ValueError`
- [ ] Les evenements de `values` sont des copies (modifier un evenement n'altere pas l'execution)

---

## Exercice 3 : Reducer custom `add_messages` qui deduplique par `id`

### Objectif
Reproduire le comportement intelligent du reducer `add_messages` de LangGraph : concatener les nouveaux messages, mais **deduplicquer et mettre a jour** ceux qui portent un `id` deja present (au lieu d'accumuler des doublons).

### Consigne
Le reducer `operator.add` accumule betement : si un node re-emet un message deja en state, on a un doublon. Construis mieux :

1. Ecris un reducer pur `add_messages(existing, new) -> list` qui :
   - Assigne un `id` a tout message qui n'en a pas (ex : un compteur stable ou un hash deterministe du contenu — pas `uuid` aleatoire, pour rester deterministe et testable).
   - Si un message entrant a un `id` deja present dans `existing`, il **remplace** l'ancien **a sa position d'origine** (mise a jour in-place logique, pas de doublon, l'ordre est preserve).
   - Sinon, il est **ajoute a la fin**.
   - La fonction est pure (ne mute ni `existing` ni `new`).
2. Declare un state `class ChatState(TypedDict): messages: Annotated[list, add_messages]`.
3. Construis un graph ou :
   - un node `draft` ajoute un message assistant avec `id="a1"` et contenu `"brouillon"`.
   - un node `revise` re-emet un message `id="a1"` avec contenu `"version finale"` (mise a jour, pas ajout) + un nouveau message `id="a2"`.
4. Apres execution, verifie que `messages` contient exactement 1 message `a1` (avec le contenu final `"version finale"`), 1 message `a2`, plus le message user initial — **sans doublon**.
5. Teste un cas ou deux messages **sans id** au contenu identique sont bien dedupliques (meme hash → meme id).

### Criteres de reussite
- [ ] `add_messages` est une fonction pure (aucune mutation de `existing` ni `new`)
- [ ] Un message dont l'`id` existe deja **remplace** l'ancien a sa position, sans creer de doublon
- [ ] Un message avec un nouvel `id` est ajoute a la fin
- [ ] Les messages sans `id` recoivent un `id` deterministe (memes inputs → meme id)
- [ ] Apres le graph, `messages` ne contient aucun doublon d'`id`
- [ ] L'ordre d'insertion des `id` distincts est preserve
