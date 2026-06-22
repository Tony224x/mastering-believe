# Exercices Medium — LangGraph avance (J6)

---

## Exercice 1 : Map-reduce avec custom reducer et branche qui echoue

### Objectif
Construire un pipeline map-reduce via le Send API ou le fan-out d'une branche peut **echouer** sans faire tomber tout le graph, et ou le reduce utilise un **reducer custom** (pas un simple `add`) qui deduplique et garde le meilleur score par cle.

### Consigne
En partant du stub `MiniLangGraph` (Send + Checkpointer) de `02-code/06-langgraph-avance.py` :

1. Definis un state avec un champ `results` annote par un **reducer custom** `merge_best(left, right)` :
   - Chaque resultat est un dict `{"source": str, "score": float, "ok": bool}`
   - Le reducer fusionne deux listes en gardant, **pour chaque `source`, l'entree au score le plus eleve** (deduplication par cle)
2. Le node `dispatch` emet un `Send("fetch_one", ...)` par source (au moins 5 sources)
3. Le node worker `fetch_one(state)` :
   - Retourne un resultat normal `{"ok": True, ...}` pour la plupart des sources
   - Pour une source designee comme "flaky" (ex : `src_C`), **simule un echec** : il attrape l'exception en interne et retourne `{"ok": False, "error": "...", ...}` au lieu de propager — la branche echoue **gracieusement**
4. Un node `reduce` final calcule un `summary` : nombre de succes, nombre d'echecs, et la liste des sources OK triees par score decroissant
5. Verifie qu'une source dupliquee (la meme `source` envoyee deux fois avec des scores differents) ne produit **qu'une seule** entree dans `results` — celle au meilleur score

### Criteres de reussite
- [ ] Le reducer custom deduplique par `source` et garde le meilleur score
- [ ] La branche "flaky" echoue **sans** interrompre les autres branches
- [ ] Le node `reduce` distingue correctement succes et echecs (`ok=True` vs `ok=False`)
- [ ] Une source envoyee en double ne laisse qu'une entree (le meilleur score gagne)
- [ ] Un assert verifie le compte exact de succes/echecs et l'unicite par source
- [ ] Aucune exception ne remonte jusqu'a l'appelant de `invoke`

---

## Exercice 2 : Checkpoint + resume apres un crash en milieu de graph

### Objectif
Simuler un crash au milieu d'un graph long, puis **reprendre** l'execution depuis le dernier checkpoint sauvegarde sans rejouer les nodes deja faits.

### Consigne
En partant du `Checkpointer` du stub :

1. Construis un graph sequentiel d'au moins 5 nodes : `step_1 -> step_2 -> ... -> step_5`. Chaque node ajoute un message et incremente un compteur.
2. Ajoute un mecanisme de **crash injecte** : un node configure pour lever une exception la **premiere fois** qu'il s'execute (ex : `step_3` echoue au 1er passage, puis reussit au resume). Le checkpointer doit avoir sauvegarde l'etat **jusqu'au step_2 inclus** avant le crash.
3. Ecris une fonction `run_with_resume(app, ckpt, initial, thread_id)` qui :
   - Lance `invoke` ; si une exception survient, **ne perd pas** les checkpoints deja ecrits
   - Recharge le **dernier** state checkpointe (`load_latest` ou equivalent) et **reprend** l'execution a partir du node suivant — pas depuis le debut
4. Verifie qu'apres le resume :
   - Le state final contient bien les 5 messages (un par step), **dans l'ordre, sans doublon**
   - `step_1` et `step_2` ne se sont **executes qu'une seule fois** (compter les appels effectifs)
   - Le compteur final vaut 5

### Criteres de reussite
- [ ] Le crash injecte interrompt l'execution au milieu du graph
- [ ] Les checkpoints anterieurs au crash sont conserves
- [ ] Le resume repart du node qui a crashe (ou du suivant), **pas** du START
- [ ] `step_1` et `step_2` ne sont **pas** rejoues (instrumente un compteur d'appels)
- [ ] Le state final a exactement 5 messages distincts et `counter == 5`
- [ ] Un assert prouve l'absence de doublon et le bon nombre d'executions de chaque node

---

## Exercice 3 : Dynamic fan-out — nombre de branches fonction du state

### Objectif
Construire un fan-out **dynamique** ou le nombre de `Send` n'est pas fixe a l'avance mais **calcule a partir du state** (chunking d'un document en sous-taches), puis collecter et ordonner les resultats.

### Consigne
Tu recois un "document" (une liste de phrases) a annoter. Le nombre de branches doit dependre de la **taille du document** et d'un `chunk_size` configurable.

1. State : `document: list[str]`, `chunk_size: int`, `annotations: Annotated[list, custom_reducer]`
2. Node `planner(state)` :
   - Decoupe `document` en chunks de taille `chunk_size`
   - Emet **un `Send("annotate_chunk", {...})` par chunk**, en passant a chaque branche son `chunk_id` (index) et la sous-liste de phrases
   - Le nombre de Send varie : 3 phrases + `chunk_size=1` => 3 branches ; 6 phrases + `chunk_size=2` => 3 branches ; etc.
3. Node worker `annotate_chunk(state)` : retourne `{"annotations": [{"chunk_id": id, "n": len(chunk), "text": "..."}]}`
4. Le reducer custom **trie les annotations par `chunk_id`** au moment du merge (l'ordre d'arrivee des branches est non-deterministe, mais le resultat final doit etre ordonne)
5. Node `assemble(state)` : reconstruit le `final` (concatenation ordonnee) et un `coverage` = nombre total de phrases annotees
6. Teste avec **au moins deux configurations** (`chunk_size` differents) et verifie que le nombre de branches et la couverture sont corrects dans chaque cas

### Criteres de reussite
- [ ] Le nombre de `Send` emis est calcule depuis le state (pas hardcode)
- [ ] Deux configurations differentes produisent un nombre de branches different
- [ ] Le reducer custom garantit un ordre final par `chunk_id` malgre l'arrivee non-ordonnee
- [ ] `coverage` egale le nombre total de phrases du document (aucune phrase perdue)
- [ ] Le dernier chunk (potentiellement plus court) est gere correctement
- [ ] Un assert verifie le nombre de branches ET la couverture pour chaque config
